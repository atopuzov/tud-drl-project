"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is," without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import argparse
import os
from pathlib import Path
from typing import Dict

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

import customcnn
import tetrisenv  # noqa: F401  # pylint: disable=unused-import

def sample_dqn_params(trial: optuna.Trial) -> Dict[str, any]:
    """
    Sample hyperparameters for DQN.

    Args:
        trial (optuna.Trial): Optuna trial object

    Returns:
        Dict[str, any]: Sampled hyperparameters
    """
    return {
        # Network architecture
        "net_arch": [trial.suggest_int("net_arch_hidden", 64, 512)],
        "features_dim": trial.suggest_int("features_dim", 64, 512),

        # Training hyperparameters
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 200000, 500000]),
        "learning_starts": trial.suggest_int("learning_starts", 10000, 100000),
        "target_update_interval": trial.suggest_int("target_update_interval", 500, 10000),
        "gamma": trial.suggest_float("gamma", 0.9, 0.99999, log=True),

        # Exploration parameters
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.5),
        "exploration_initial_eps": trial.suggest_float("exploration_initial_eps", 0.1, 1.0),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
    }

def create_env(args: argparse.Namespace):
    """Create vectorized environment."""
    env_kwargs = {
        "grid_size": (20, 10),
        "tetrominoes": ["I", "O", "T", "L", "J"],
        "render_mode": None,
    }

    vec_env_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    env = make_vec_env(
        args.env_name,
        env_kwargs=env_kwargs,
        n_envs=args.num_envs,
        vec_env_cls=vec_env_cls
    )

    if args.frame_stack:
        env = VecFrameStack(env, args.frame_stack_size, channels_order="first")

    return env

def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """
    Optuna objective function.

    Args:
        trial (optuna.Trial): Optuna trial object
        args (argparse.Namespace): Command line arguments

    Returns:
        float: Mean reward achieved
    """
    # Sample hyperparameters
    hyper_params = sample_dqn_params(trial)

    # Create environment
    env = create_env(args)

    # Get feature extractor
    feature_extractor = customcnn.FEATURE_EXTRACTORS.get(
        args.extractor_name,
        customcnn.TetrisFeatureExtractor
    )

    # Set up policy kwargs
    policy_kwargs = {
        "features_extractor_class": feature_extractor,
        "features_extractor_kwargs": {
            "features_dim": hyper_params["features_dim"]
        },
        "net_arch": hyper_params["net_arch"],
        "activation_fn": torch.nn.ReLU,
    }

    # Set device
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    # Create model
    model = DQN(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=hyper_params["learning_rate"],
        batch_size=hyper_params["batch_size"],
        buffer_size=hyper_params["buffer_size"],
        learning_starts=hyper_params["learning_starts"],
        target_update_interval=hyper_params["target_update_interval"],
        gamma=hyper_params["gamma"],
        exploration_fraction=hyper_params["exploration_fraction"],
        exploration_initial_eps=hyper_params["exploration_initial_eps"],
        exploration_final_eps=hyper_params["exploration_final_eps"],
        tensorboard_log=args.tensorboard_log,
        device=device,
        verbose=0,
    )

    # Set up evaluation
    eval_path = Path(args.work_dir) / f"trial_{trial.number}"
    eval_path.mkdir(parents=True, exist_ok=True)
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(eval_path),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=0
    )

    try:
        model.learn(
            total_timesteps=args.timestamps,
            callback=eval_callback,
            progress_bar=True
        )
    except (AssertionError, ValueError) as e:
        # Sometimes the training can fail due to NaN
        print(f"Training failed: {str(e)}")
        return float("-inf")

    # Return best mean reward
    return eval_callback.best_mean_reward

def optimize(args: argparse.Namespace):
    """Run the optimization."""
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=TPESampler(n_startup_trials=args.n_startup_trials),
        pruner=MedianPruner(n_startup_trials=args.n_startup_trials),
        direction="maximize",
        load_if_exists=True,
    )

    try:
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        pass

    # Print results
    print("Number of finished trials:", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Optimize Tetris DQN hyperparameters")

    # Optuna parameters
    parser.add_argument("--study-name", type=str, default="tetris_optimization")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-startup-trials", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)

    # Environment parameters
    parser.add_argument("--env-name", type=str, default="Tetris-v3")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument("--frame-stack", action="store_true")
    parser.add_argument("--frame-stack-size", type=int, default=4)

    # Training parameters
    parser.add_argument("--timestamps", type=int, default=int(1e5))
    parser.add_argument("--work-dir", type=Path, default=Path("optuna_results"))
    parser.add_argument("--tensorboard-log", type=Path)
    parser.add_argument("--extractor-name", type=str, default="TFEAtari")

    args = parser.parse_args()

    # Set up directories
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if not args.tensorboard_log:
        args.tensorboard_log = args.work_dir / "tensorboard"
    args.tensorboard_log.mkdir(parents=True, exist_ok=True)

    optimize(args)

if __name__ == "__main__":
    main()
