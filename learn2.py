"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is," without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import argparse
import sys
from pathlib import Path

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

import tetrisenv  # noqa: F401  # pylint: disable=unused-import
import customcnn


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
        is done, it retrieves the score and lines cleared from the environment's info
        and logs these metrics to TensorBoard.
        """
        # Check if any environment is done (end of episode)
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                score = info.get("score", 0)
                lines_cleared = info.get("lines_cleared", 0)
                # Log to TensorBoard at the end of the episode
                self.logger.record("episode/score", score)
                self.logger.record("episode/lines_cleared", lines_cleared)
        return True


LOG_PATH = "logs"
TETRIS_MODEL = "tetris_model.zip"
CHECKPOINT_PATH = "models"


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
        model = DQN.load(TETRIS_MODEL, env=env, tensorboard_log=LOG_PATH)
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
    policy_kwargs = {
        "features_extractor_class": customcnn.TetrisFeatureExtractor1,
        "net_arch": [128, 128],  # MLP architecture after feature extraction
        "activation_fn": torch.nn.ReLU,
    }

    # TODO: See if it makes a difference (except on big convolutions)
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    # device = "cpu"

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=128,
        learning_starts=10000,
        buffer_size=100000,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        # exploration_final_eps=0.01,
        exploration_final_eps=0.05,
        # exploration_final_eps=0.1, # try with 0.05/0.1
        verbose=1,
        tensorboard_log=LOG_PATH,
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
    best_model = "best_model"
    eval_callback = EvalCallback(env, eval_freq=10000, best_model_save_path=best_model)

    record_game_info_callback = EpisodeEndMetricsCallback()

    callbacks = [eval_callback, record_game_info_callback]

    if args.checkpoint:
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_interval,
            save_path=CHECKPOINT_PATH,
            name_prefix=args.checkpoint_prefix,
            save_replay_buffer=args.save_replay,
            verbose=1,
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
            )
            model.save(TETRIS_MODEL)
    except KeyboardInterrupt:
        pass

    model.save(TETRIS_MODEL)


def learn():

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
    parser.add_argument("--timestamps", type=int, default=10e4, help="Number of timestumps")
    parser.add_argument("--model-file", type=Path, default="tetris_model.zip", help="Model file")
    # Checkpoints
    parser.add_argument("--checkpoint", action="store_true", help="Checkpoint models")
    parser.add_argument("--checkpoint-interval", type=int, default=10000, help="Checkpoint interval")
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="tetris_model",
        help="Model file name prefix for checkpointing",
    )
    # TODO:
    parser.add_argument("--num-envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--subproc", action="store_true", help="Use SubprocVecEnv")
    parser.add_argument("--env-name", type=str, default="Tetris-v3", help="Environment name")
    args = parser.parse_args()

    # Create the environment
    tetrominoes = ["I", "O", "T", "L", "J"]

    env_kwargs = {
        "grid_size": (20, 10),
        "tetrominoes": tetrominoes,
        "render_mode": "human",
    }

    vec_env_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    env = make_vec_env(args.env_name, env_kwargs=env_kwargs, n_envs=args.num_envs, vec_env_cls=vec_env_cls)

    if args.new:
        start_learning(args, env)
    else:
        continue_learning(args, env)


if __name__ == "__main__":
    learn()
