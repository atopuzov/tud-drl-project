"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is," without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import tetrisenv  # noqa: F401  # pylint: disable=unused-import

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game of Tetris")
    parser.add_argument("--delay", type=float, default=0.01, help="Delay between frames")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model-file", type=Path, default="tetris_model.zip", help="Model file")
    group.add_argument("--random", action="store_true", help="Use a random agent")
    game_mode = parser.add_mutually_exclusive_group()
    game_mode.add_argument("--pygame", action="store_true", help="Use pygame interface")
    game_mode.add_argument("--ascii", action="store_true", help="Use ascii interface")
    parser.add_argument("--env-name", type=str, default="Tetris-v3", help="Use SubprocVecEnv")
    args = parser.parse_args()

    render_mode = "pygame" if args.pygame else "ansi"

    tetrominoes = ["I", "O", "T", "L", "J"]
    genv = gym.make(
        args.env_name,
        grid_size=(20, 10),
        tetrominoes=tetrominoes,
        render_mode=render_mode,
    )
    env = DummyVecEnv([lambda: Monitor(genv)])

    if args.random:
        model = DQN(
            "MlpPolicy",
            env,
            exploration_fraction=1.0,  # Maintain maximum exploration
            exploration_initial_eps=1.0,  # Start with 100% random actions
            exploration_final_eps=1.0,  # Keep 100% random actions (never decrease)
        )
    else:
        try:
            model = DQN.load(args.model_file, env=env)
        except FileNotFoundError:
            print(f"Unable to find {args.model_file}")
            sys.exit(-1)

    obs = env.reset()
    terminated = False
    try:
        while not terminated:
            env.render()  # Render the game state
            action, _states = model.predict(obs, deterministic=True)  # Predict the next action
            obs, reward, terminated, information = model.env.step(action)  # Perform the next action
            time.sleep(args.delay)
    finally:
        # env.render()  # Render the game state
        env.close()
