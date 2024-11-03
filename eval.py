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
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import tetrisenv  # noqa: F401  # pylint: disable=unused-import


class CustomMetricsCallback:
    """
    CustomMetricsCallback is a callback class used to track and compute statistics for episode scores and lines cleared
    during the training of a reinforcement learning model.
    Attributes:
        episode_scores (list): A list to store the scores of each episode.
        episode_lines (list): A list to store the number of lines cleared in each episode.
    Methods:
        __call__(locals_, globals_):
            Called at the end of each episode to record the score and lines cleared.
            Args:
                locals_ (dict): Local variables from the training environment.
                globals_ (dict): Global variables from the training environment.
        get_results():
            Computes and returns the mean and standard deviation of the episode scores and lines cleared.
            Returns:
                tuple: A tuple containing the mean and standard deviation of the episode scores and lines cleared.
    """

    def __init__(self):
        self.episode_scores = []
        self.episode_lines = []

    def __call__(self, locals_: dict, globals_: dict) -> None:
        if locals_["done"]:
            score = locals_["info"].get("score", 0)
            lines = locals_["info"].get("lines_cleared", 0)
            self.episode_scores.append(score)
            self.episode_lines.append(lines)

    def get_results(self):
        """
        Calculate and return the mean and standard deviation of episode scores and lines.

        Returns:
            tuple: A tuple containing:
                - score_mean (float): The mean of the episode scores.
                - score_std (float): The standard deviation of the episode scores.
                - lines_mean (float): The mean of the episode lines.
                - lines_std (float): The standard deviation of the episode lines.
        """
        return (
            np.mean(self.episode_scores),
            np.std(self.episode_scores),
            np.mean(self.episode_lines),
            np.std(self.episode_lines),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Game of Tetris")
    parser.add_argument("--render", action="store_true", help="Render")
    game_mode = parser.add_mutually_exclusive_group()
    game_mode.add_argument("--pygame", action="store_true", help="Use pygame interface")
    game_mode.add_argument("--ascii", action="store_true", help="Use ascii interface")
    parser.add_argument("--model-file", type=Path, default="tetris_model.zip", help="Model file")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--env-name", type=str, default="Tetris-v3", help="Environment name")
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

    try:
        model = DQN.load(args.model_file, env=env)
    except FileNotFoundError:
        print(f"Unable to find {args.model_file}")
        sys.exit(-1)

    callback = CustomMetricsCallback()

    rewards, lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.episodes,
        deterministic=True,
        render=args.render,
        callback=callback,
        return_episode_rewards=True,
    )
    score_mean, score_std, lines_mean, lines_std = callback.get_results()
    reward_mean, reward_std = np.mean(rewards), np.std(rewards)
    episode_mean, episode_std = np.mean(lengths), np.std(lengths)
    print(
        f"Reward: {reward_mean} +/- {reward_std}, "
        f"ep. length: {episode_mean} +/- {episode_std} "
        f"score: {score_mean} +/- {score_std}, "
        f"lines: {lines_mean} +/- {lines_std}"
    )

    env.close()
