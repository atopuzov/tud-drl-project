"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is" without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import tetrisenv  # noqa: F401  # pylint: disable=unused-import
from rndagent import RandomAgent


def main(cmdline: Optional[str]) -> None:
    """Main function for playing Tetris"""
    parser = argparse.ArgumentParser(description="Game of Tetris")
    parser.add_argument("--delay", type=float, default=0.01, help="Delay between frames")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model-file", type=Path, default="tetris_model.zip", help="Model file")
    group.add_argument("--random", action="store_true", help="Use a random agent")
    game_mode = parser.add_mutually_exclusive_group()
    game_mode.add_argument("--pygame", action="store_true", help="Use pygame interface")
    game_mode.add_argument("--ascii", action="store_true", help="Use ascii interface")
    parser.add_argument("--env-name", type=str, default="Tetris-v3", help="Use SubprocVecEnv")
    parser.add_argument("--frame-stack", action="store_true", help="Use frame stacking")
    parser.add_argument("--frame-stack-size", type=int, default=4, help="Frame stack size")
    parser.add_argument("--random-seed", type=int, default=None, help="Use a random number seed")
    parser.add_argument("--pause", action="store_true", help="Pasue after gameplay")
    parser.add_argument(
        "--tetrominoes",
        type=lambda s: s.upper(),
        nargs="+",
        default=["I", "O", "T", "L", "J"],
        choices=["I", "O", "T", "L", "J", "S", "Z"],
        help="Tetrominoes to use",
    )
    if cmdline:
        args = parser.parse_args(cmdline.split())
    else:
        args = parser.parse_args()

    genv = gym.make(
        args.env_name,
        grid_size=(20, 10),
        tetrominoes=args.tetrominoes,
        render_mode="pygame" if args.pygame else "ansi",
    )
    env = DummyVecEnv([lambda: Monitor(genv)])
    if args.frame_stack:
        print(f"Using {args.frame_stack_size} frame stacking")
        env = VecFrameStack(env, args.frame_stack_size, channels_order="first")

    if args.random:
        model = RandomAgent(env, seed=args.random_seed)
    else:
        try:
            model = DQN.load(args.model_file, env=env)
        except FileNotFoundError:
            print(f"Unable to find {args.model_file}")
            sys.exit(-1)

    env.seed(seed=args.random_seed)
    obs = env.reset()
    terminated = False
    try:
        while not terminated:
            env.render()  # Render the game state
            action, _states = model.predict(obs, deterministic=True)  # Predict the next action
            (
                obs,
                reward,
                terminated,
                information,
            ) = env.step(
                action
            )  # Perform the next action
            time.sleep(args.delay)
    finally:
        # env.render()  # Render the game state
        if args.pause:
            _ = input("Press enter to continue")
        env.close()


if __name__ == "__main__":
    main()
