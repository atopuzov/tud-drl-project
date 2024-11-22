"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is," without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import argparse
import signal
import sys
from pathlib import Path

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack

import customcnn
import tetrisenv  # noqa: F401  # pylint: disable=unused-import


class EpisodeEndMetricsCallback(BaseCallback):
    """
    Custom callback for logging episode score and lines cleared to TensorBoard only at the end of each game.
    """

    def _on_step(self) -> bool:
        """
        This method is called at each step of the environment.

        Returns:
            bool: Always returns True so that the training process continues.

        The method checks if any environment is done (end of episode). If an episode
        is done, it retrieves the score, lines cleared and pieces placed  from the
        environment's info and logs these metrics to TensorBoard.
        """
        # Check if any environment is done (end of episode)
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                score = info.get("score", 0)
                lines_cleared = info.get("lines_cleared", 0)
                pieces_placed = info.get("pieces_placed", 0)
                bumpiness = info.get("bumpiness", 0)
                holes = info.get("holes", 0)
                # Log to TensorBoard at the end of the episode
                self.logger.record_mean("episode/score", score)
                self.logger.record_mean("episode/lines_cleared", lines_cleared)
                self.logger.record_mean("episode/pieces_placed", pieces_placed)
                self.logger.record_mean("episode/bumpiness", bumpiness)
                self.logger.record_mean("episode/holes", holes)
        return True


def continue_learning(args: argparse.Namespace, env: VecEnv):
    """
    Continue the learning process for a given environment and model.
    Args:
        args (argparse.Namespace): Command-line arguments containing model file path,
                                   replay buffer file path, and other configurations.
        env (VecEnv): The environment in which the model will be trained.
    Raises:
        SystemExit: If the specified model file cannot be found.
    Notes:
        - Loads a pre-trained DQN model from the specified file.
        - If the replay buffer file is provided, it loads the replay buffer into the model.
        - Calls the `do_z_learning` function to continue the learning process.
    """
    try:
        model = DQN.load(
            args.model_file,
            env=env,
            tensorboard_log=args.tensorboard_log,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            buffer_size=args.buffer_size,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            verbose=0 if args.quiet else 1,
        )
    except FileNotFoundError:
        print(f"Unable to find {args.model_file}")
        sys.exit(-1)

    if args.load_replay:
        model.load_replay_buffer(args.replay_buffer_file)

    do_z_learning(args, env, model)


def start_learning(args: argparse.Namespace, env: VecEnv) -> None:
    """
    Initializes and starts the learning process for a given environment using the DQN algorithm.
    Args:
        args (argparse.Namespace): Command-line arguments passed to the script.
        env: (VecEnv): The environment in which the agent will learn.
    Returns:
        None
    """

    feature_extractor = customcnn.FEATURE_EXTRACTORS.get(args.extractor_name, customcnn.TetrisFeatureExtractor)

    policy_kwargs = {
        "features_extractor_class": feature_extractor,
        "features_extractor_kwargs": {"features_dim": args.extractor_features},
        "net_arch": args.net_arch,
        "activation_fn": torch.nn.ReLU,
    }

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        # PyTorch config for better GPU usage
        torch.backends.cudnn.benchmark = True  # Enable auto-tuner
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere (Gf RTX 30, RTX A)
        torch.backends.cudnn.allow_tf32 = True
        device = "cuda"

    model = DQN(
        "CnnPolicy",  # No difference to MlpPolicy since we are using a custom feature extractor
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/policies.py
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        buffer_size=args.buffer_size,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        # exploration_final_eps=0.1, # try with 0.05/0.1
        verbose=0 if args.quiet else 1,
        tensorboard_log=args.tensorboard_log,
        device=device,
    )

    do_z_learning(args, env, model)


def do_z_learning(args: argparse.Namespace, env: VecEnv, model) -> None:
    """
    Perform the learning process for the Tetris model.

    Args:
        args (argparse.Namespace): The command-line arguments.
        env: (VecEnv): The environment for the model to interact with.
        model: The model to be trained.
    """
    eval_callback = EvalCallback(env, eval_freq=10000, best_model_save_path=args.best_model_path)

    record_game_info_callback = EpisodeEndMetricsCallback()

    callbacks = [eval_callback, record_game_info_callback]

    if args.checkpoint:
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_interval,
            save_path=args.checkpoint_path,
            name_prefix=args.checkpoint_prefix,
            save_replay_buffer=args.save_replay,
            verbose=0 if args.quiet else 1,
        )
        callbacks.append(checkpoint_callback)

    try:
        for i in range(100000):
            print(f"Cycle {i}")
            model.learn(
                total_timesteps=args.timestamps,
                callback=callbacks,
                log_interval=100,
                reset_num_timesteps=False,
                tb_log_name=args.log_name,
            )
            model.save(args.model_file)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        signal_type = signal.CTRL_C_EVENT if sys.platform.startswith("win") else signal.SIGINT
        original_handler = signal.signal(signal_type, signal.SIG_IGN)
        model.save(args.model_file)
        signal.signal(signal_type, original_handler)


def learn():
    """Set up the learning environment and start the learning process."""
    parser = argparse.ArgumentParser(description="Game of Tetris")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--continue", action="store_true", help="Continue training")
    group.add_argument("--new", action="store_true", help="Start new training")
    parser.add_argument("--save-replay", action="store_true", help="Save replay buffer")
    parser.add_argument("--load-replay", action="store_true", help="Load replay buffer")
    parser.add_argument(
        "--replay-buffer-file",
        type=Path,
        default=Path("replay_buffer.zip"),
        help="Replay buffer file",
    )
    parser.add_argument("--timestamps", type=int, default=10e4, help="Number of timestamps")
    parser.add_argument("--model-file", type=Path, help="Model file")

    # Model parameters https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py
    parser.add_argument("--gamma", type=float, default=0.99, help="Gamma")
    parser.add_argument("--buffer-size", type=int, default=500000, help="Buffer size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-starts", type=int, default=50000, help="Learning starts")
    parser.add_argument("--target-update-interval", type=int, default=2000, help="Target update interval")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--exploration-fraction", type=float, default=0.3, help="Exploration fraction")
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0, help="Exploration initial eps")
    parser.add_argument("--exploration-final-eps", type=float, default=0.01, help="Exploration final eps")
    parser.add_argument("--train-freq", type=int, default=4, help="Train frequency")
    parser.add_argument("--gradient-steps", type=int, default=1, help="Gradinet steps")

    # Model architecture
    parser.add_argument("--extractor-name", type=str, default="TFEAtari", help="Feature extractor")
    parser.add_argument("--extractor-features", type=int, default=256, help="Number of features to extract")
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256], help="NN architecture")

    # Directories
    parser.add_argument("--work-dir", type=Path, default=Path("."), help="Base directory for logs and checkpoints")
    parser.add_argument("--tensorboard-log", type=Path, help="Tensorboard logs path")
    parser.add_argument("--checkpoint-path", type=Path, help="Checkpoint path")
    parser.add_argument("--best-model-path", type=Path, help="Best model save path")
    parser.add_argument("--log-name", type=str, default="DQN", help="Name for tensorboard")

    # Checkpoints
    parser.add_argument("--checkpoint", action="store_true", help="Checkpoint models")
    parser.add_argument("--checkpoint-interval", type=int, default=10000, help="Checkpoint interval")
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="tetris_model",
        help="Model file name prefix for checkpointing",
    )

    # Environment
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--subproc", action="store_true", help="Use SubprocVecEnv")
    parser.add_argument("--env-name", type=str, default="Tetris-imgh", help="Environment name")
    parser.add_argument("--piece-gen", choices=("7bag", "14bag", "rnd"), default=None, help="Piece generator")
    parser.add_argument("--frame-stack", action="store_true", help="Use frame stacking")
    parser.add_argument("--frame-stack-size", type=int, default=4, help="Frame stack size")
    parser.add_argument(
        "--tetrominoes",
        type=lambda s: s.upper(),
        nargs="+",
        default=["I", "O", "T", "L", "J"],
        choices=["I", "O", "T", "L", "J", "S", "Z"],
        help="Tetrominoes to use",
    )

    # Other
    parser.add_argument("--random-seed", type=int, default=None, help="Use a random number seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument(
        "--device", type=str, choices=("auto", "cpu", "cuda", "mps"), default="auto", help="Device to use"
    )

    args = parser.parse_args()
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    # Set default paths based on work dir
    if not args.tensorboard_log:
        args.tensorboard_log = args.work_dir / "logs"
    if not args.checkpoint_path:
        args.checkpoint_path = args.work_dir / "checkpoints"
    if not args.best_model_path:
        args.best_model_path = args.work_dir / "best_model"
    if not args.model_file:
        args.model_file = args.work_dir / "tetris_model.zip"

    # Ensure directories exist
    args.tensorboard_log.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path.mkdir(parents=True, exist_ok=True)
    args.best_model_path.mkdir(parents=True, exist_ok=True)

    # Create the environment
    env_kwargs = {
        "grid_size": (20, 10),
        "tetrominoes": args.tetrominoes,
        "render_mode": "ansi",
        "piece_gen": args.piece_gen,
    }

    vec_env_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    env = make_vec_env(args.env_name, env_kwargs=env_kwargs, n_envs=args.num_envs, vec_env_cls=vec_env_cls)
    print(f"Created {args.num_envs} {args.env_name} environments. Using {args.extractor_name}.")

    if args.frame_stack:
        print(f"Using {args.frame_stack_size} frame stacking")
        env = VecFrameStack(env, args.frame_stack_size, channels_order="first")

    new_learning = args.new or not args.model_file.exists()

    if new_learning:
        print("Starting new learning ...")
        start_learning(args, env)
    else:
        print("Continuing learning ...")
        continue_learning(args, env)

    print("Training complete ...")


if __name__ == "__main__":
    learn()
