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
    for env_name in ("Tetris-v0", "Tetris-v1", "Tetris-v2", "Tetris-v3"):
        print("Checking the {env_name} env ...")
        base_env = gym.make(env_name)
