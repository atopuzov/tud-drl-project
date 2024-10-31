"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is," without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import sys
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor

from tetrisenv import StandardReward2TetrisEnv, StandardRewardTetrisEnv


class EpisodeEndMetricsCallback(BaseCallback):
    """
    Custom callback for logging episode score and lines cleared to TensorBoard only at the end of each game.
    """

    def __init__(self, verbose=0):
        super(EpisodeEndMetricsCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
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


log_path = "logs"
tetris_model = "tetris_model.zip"
checkpoint_path = "models"


def continue_learning(args, env):
    try:
        model = DQN.load(tetris_model, env=env, tensorboard_log=log_path)
    except FileNotFoundError:
        print(f"Unable to find {args.model_file}")
        sys.exit(-1)

    if args.load_replay:
        model.load_replay_buffer(args.replay_buffer_file)


def start_learning(args, env):
    policy_kwargs = {"net_arch": [128, 128]}  # MLP architecture after feature extraction

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
        tensorboard_log=log_path,
    )

    best_model = "best_model"
    eval_callback = EvalCallback(env, eval_freq=10000, best_model_save_path=best_model)

    record_game_info_callback = EpisodeEndMetricsCallback()

    callbacks = [eval_callback, record_game_info_callback]

    if args.checkpoint:
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_interval,
            save_path=checkpoint_path,
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
            model.save(tetris_model)
    except KeyboardInterrupt:
        pass

    model.save(tetris_model)


def learn():
    import argparse

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
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--subproc", action="store_true", help="Use SubprocVecEnv")
    args = parser.parse_args()

    # Create the environment
    tetrominoes = ["I", "O", "T", "L", "J"]
    env = StandardReward2TetrisEnv(grid_size=(20, 10), tetrominoes=tetrominoes, render_mode="human")
    if args.subproc:
        env = make_vec_env(lambda: env, n_envs=args.num_envs, vec_env_cls=SubprocVecEnv)
    else:
        env = DummyVecEnv([lambda: Monitor(env)])

    if args.new:
        start_learning(args, env)
    else:
        continue_learning(args, env)


if __name__ == "__main__":
    learn()
