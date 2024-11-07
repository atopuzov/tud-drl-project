"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is," without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

import tetrisenv

if __name__ == "__main__":
    for env_name in (
        "Tetris-base",
        "Tetris-score",
        "Tetris-simpleheuristic",
        "Tetris-heuristic1",
        "Tetris-heuristic2",
        "Tetris-heuristic3",
        "Tetris-v2",
        "Tetris-imgh",
        "Tetris-v3",
    ):
        print(f"Checking the {env_name} env ...")
        base_env = gym.make(env_name)
