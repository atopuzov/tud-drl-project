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
        model = DQN.load(args.model_file, env=env, tensorboard_log=LOG_PATH)
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
        device = "cuda"

    model = DQN(
        "CnnPolicy", # No difference to MlpPolicy since we are using a custom feature extractor
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/policies.py
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        gamma=0.99,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        buffer_size=args.buffer_size,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        # exploration_final_eps=0.1, # try with 0.05/0.1
        verbose=1,
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
    best_model = "best_model"
    eval_callback = EvalCallback(env, eval_freq=10000, best_model_save_path=best_model)

    record_game_info_callback = EpisodeEndMetricsCallback()

    callbacks = [eval_callback, record_game_info_callback]

    if args.checkpoint:
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_interval,
            save_path=args.checpoint_path,
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
                tb_log_name=args.log_name,
            )
            model.save(args.model_file)
    except KeyboardInterrupt:
        pass

    model.save(args.model_file)


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

    # Model parametars
    # parser.add_argument("", type=float, default=0.1, help="")
    # parser.add_argument("", type=int, default=1, help="")

    parser.add_argument("--buffer-size", type=int, default=500000, help="Buffer size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-starts", type=int, default=50000, help="Learning starts")
    parser.add_argument("--target-update-interval", type=int, default=2000, help="Target update interval")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--exploration-fraction", type=float, default=0.3, help="Exploration fraction")
    parser.add_argument("--exploration-initial-eps", type=float, default=0.3, help="Exploration initial eps")
    parser.add_argument("--exploration-final-eps", type=float, default=0.01, help="Exploration final eps")

    # tensorboard
    parser.add_argument("--tensorboard-log", type=Path, default=LOG_PATH, help="Tensorboard logs path")
    parser.add_argument("--log-name", type=str, default="DQN", help="Name for tensorboard")

    # Checkpoints
    parser.add_argument("--checkpoint", action="store_true", help="Checkpoint models")
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH, help="Checkpoint path")
    parser.add_argument("--checkpoint-interval", type=int, default=10000, help="Checkpoint interval")
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="tetris_model",
        help="Model file name prefix for checkpointing",
    )
    # TODO:
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--subproc", action="store_true", help="Use SubprocVecEnv")
    parser.add_argument("--env-name", type=str, default="Tetris-imgh", help="Environment name")
    parser.add_argument("--extractor-name", type=str, default="TFEAtari", help="Feature extractor")
    parser.add_argument('--extractor-features', type=int, default=256, help='Number of features to extract')    
    parser.add_argument("--frame-stack", action="store_true", help="Use frame stacking")
    parser.add_argument("--frame-stack-size", type=int, default=4, help="Frame stack size")
    parser.add_argument('--net-arch', type=int, nargs='+', default=[256], help='NN architecture')
    parser.add_argument("--random-seed", type=int, default=None, help="Use a random number seed")
    args = parser.parse_args()

    # Create the environment
    tetrominoes = ["I", "O", "T", "L", "J"]

    env_kwargs = {
        "grid_size": (20, 10),
        "tetrominoes": tetrominoes,
        "render_mode": "ansi",
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


if __name__ == "__main__":
    learn()
