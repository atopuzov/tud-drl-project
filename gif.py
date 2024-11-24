"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is" without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import tetrisenv
from rndagent import RandomAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game of Tetris")
    parser.add_argument("--delay", type=float, default=0.01, help="Delay between frames")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model-file", type=Path, default="tetris_model.zip", help="Model file")
    group.add_argument("--random", action="store_true", help="Use a random agent")
    parser.add_argument("--random-seed", type=int, default=None, help="Use a random number seed")
    parser.add_argument("--env-name", type=str, default="Tetris-v3", help="Environment name")
    parser.add_argument(
        "--tetrominoes",
        type=lambda s: s.upper(),
        nargs="+",
        default=["I", "O", "T", "L", "J"],
        choices=["I", "O", "T", "L", "J", "S", "Z"],
        help="Tetrominoes to use",
    )
    args = parser.parse_args()
    render_mode = "rgb_array"

    env = gym.make(
        args.env_name,
        grid_size=(20, 10),
        tetrominoes=args.tetrominoes,
        render_mode=render_mode,
    )
    env = DummyVecEnv([lambda: Monitor(env)])

    images = Path("images")
    images.mkdir(parents=True, exist_ok=True)

    if args.random:
        model = RandomAgent(env, seed=args.random_seed)
        gif_file = images / Path("random.gif")
    else:
        try:
            model = DQN.load(args.model_file, env=env)
            gif_file = images / Path("tetris.gif")
        except FileNotFoundError:
            print(f"Unable to find {args.model_file}")
            sys.exit(-1)

    images = []
    env.seed(seed=args.random_seed)
    obs = model.env.reset()
    img = model.env.render(mode="rgb_array")
    done = False
    while not done:
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, done, _ = model.env.step(action)
        img = model.env.render(mode="rgb_array")

    imageio.mimsave(gif_file, [np.array(img) for _, img in enumerate(images)], fps=29)
