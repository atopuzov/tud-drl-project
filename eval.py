"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is" without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import sys
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import tetrisenv  # noqa: F401  # pylint: disable=unused-import
from rndagent import RandomAgent


class CustomMetricsCallback:
    """
    CustomMetricsCallback is a callback class used to track and compute statistics for episode scores and lines cleared
    during the training of a reinforcement learning model.
    Attributes:
        episode_scores (list): A list to store the scores of each episode.
        episode_lines (list): A list to store the number of lines cleared in each episode.
        episode_pieces (list): A list to store the number of pieces placed in each episode.
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
        self.episode_pieces = []
        self.episode_bumpiness = []
        self.episode_holes = []

        self.last_pieces = 0
        self.current_bumpiness = []
        self.current_holes = []

    def __call__(self, locals_: dict, globals_: dict) -> None:
        pieces = locals_["info"].get("pieces_placed", 0)
        if pieces > self.last_pieces:
            self.last_pieces = pieces
            bumpiness = locals_["info"].get("bumpiness", 0)
            holes = locals_["info"].get("holes", 0)
            self.current_bumpiness.append(bumpiness)
            self.current_holes.append(holes)

        if locals_["done"]:
            score = locals_["info"].get("score", 0)
            lines = locals_["info"].get("lines_cleared", 0)
            self.episode_scores.append(score)
            self.episode_lines.append(lines)
            self.episode_pieces.append(pieces)

            self.episode_bumpiness.append(np.mean(self.current_bumpiness))
            self.episode_holes.append(np.mean(self.current_holes))

            # Reset current episode
            self.current_bumpiness = []
            self.current_holes = []
            self.last_pieces = 0

    def get_results(self):
        """
        Calculate and return the mean and standard deviation of episode scores and lines.

        Returns:
            tuple: A tuple containing:
                - score_mean (float): The mean of the episode scores.
                - score_std (float): The standard deviation of the episode scores.
                - lines_mean (float): The mean of the episode lines.
                - lines_std (float): The standard deviation of the episode lines.
                - pieces_mean (float): The mean of the episode pieces.
                - pieces_std (float): The standard deviation of the episode pieces.
        """
        return (
            np.mean(self.episode_scores),
            np.std(self.episode_scores),
            np.mean(self.episode_lines),
            np.std(self.episode_lines),
            np.mean(self.episode_pieces),
            np.std(self.episode_pieces),
        )


def main(cmdline: Optional[str]) -> None:
    """Main function for evaluating a trained model on the Tetris environment."""
    import argparse

    parser = argparse.ArgumentParser(description="Game of Tetris")
    parser.add_argument("--render", action="store_true", help="Render")
    game_mode = parser.add_mutually_exclusive_group()
    game_mode.add_argument("--pygame", action="store_true", help="Use pygame interface")
    game_mode.add_argument("--ascii", action="store_true", help="Use ascii interface")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model-file", type=Path, default="tetris_model.zip", help="Model file")
    group.add_argument("--random", action="store_true", help="Use a random agent")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--env-name", type=str, default="Tetris-v3", help="Environment name")
    parser.add_argument("--frame-stack", action="store_true", help="Use frame stacking")
    parser.add_argument("--frame-stack-size", type=int, default=4, help="Frame stack size")
    parser.add_argument("--random-seed", type=int, default=None, help="Use a random number seed")
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

    render_mode = "pygame" if args.pygame else "ansi"

    genv = gym.make(
        args.env_name,
        grid_size=(20, 10),
        tetrominoes=args.tetrominoes,
        render_mode=render_mode,
    )
    env = DummyVecEnv([lambda: Monitor(genv)])
    if args.frame_stack:
        print(f"Using {args.frame_stack_size} frame stacking")
        env = VecFrameStack(env, args.frame_stack_size, channels_order="first")

    env.seed(seed=args.random_seed)

    if args.random:
        model = RandomAgent(env, seed=args.random_seed)
    else:
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
    score_mean, score_std, lines_mean, lines_std, pieces_mean, pieces_std = callback.get_results()
    reward_mean, reward_std = np.mean(rewards), np.std(rewards)
    episode_mean, episode_std = np.mean(lengths), np.std(lengths)
    print(
        f"Reward: {reward_mean:.2f} +/- {reward_std:.2f}, "
        f"ep. length: {episode_mean} +/- {episode_std:.2f} "
        f"score: {score_mean:.2f} +/- {score_std:.2f}, "
        f"lines: {lines_mean:.2f} +/- {lines_std:.2f}, "
        f"pieces: {pieces_mean:.2f} +/- {pieces_std:.2f}"
    )

    env.close()


if __name__ == "__main__":
    main()
