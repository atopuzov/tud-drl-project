"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is," without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from tetrisenv import StandardRewardTetrisEnv

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Game of Tetris")
    parser.add_argument("--render", action="store_true", help="Render")
    game_mode = parser.add_mutually_exclusive_group()
    game_mode.add_argument("--pygame", action="store_true", help="Use pygame interface")
    game_mode.add_argument("--ascii", action="store_true", help="Use ascii interface")
    parser.add_argument(
        "--model-file", type=Path, default="tetris_model.zip", help="Model file"
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    args = parser.parse_args()

    render_mode = "pygame" if args.pygame else "ansi"

    tetrominoes = ["I", "O", "T", "L", "J"]
    env = StandardRewardTetrisEnv(
        grid_size=(20, 10), tetrominoes=tetrominoes, render_mode=render_mode
    )
    env = DummyVecEnv([lambda: Monitor(env)])

    try:
        model = DQN.load(args.model_file, env=env)
    except FileNotFoundError:
        print(f"Unable to find {args.model_file}")
        sys.exit(-1)

    reward_mean, reward_std = evaluate_policy(
        model,
        env,
        n_eval_episodes=20,
        deterministic=True,
        render=args.render,
    )
    print(f"Mean: {reward_mean}, std: {reward_std}")

    env.close()
