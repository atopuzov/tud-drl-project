"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is," without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tetrisgame import Actions, TetrisGame
from tetrisrenderer import TetrisASCIIRenderer, TetrisPyGameRenderer, TetrisRGBArrayRenderer

env_kwargs = {
    "grid_size": (20, 10),
    "tetrominoes": None,
    "render_mode": None,
    "ticks_per_drop": 1,
}
gym.register("Tetris-v0", entry_point="tetrisenv:StandardRewardTetrisEnv", kwargs=env_kwargs)
gym.register("Tetris-v1", entry_point="tetrisenv:StandardReward2TetrisEnv", kwargs=env_kwargs)
gym.register("Tetris-v2", entry_point="tetrisenv:MyTetrisEnv", kwargs=env_kwargs)
gym.register("Tetris-v3", entry_point="tetrisenv:MyTetrisEnv2", kwargs=env_kwargs)


class BaseRewardTetrisEnv(gym.Env):
    """Base Tetris environment with minimal reward structure."""

    metadata = {
        "render_modes": ["pygame", "ansi", "rgb_array", "human"],
        "render_fps": 4,
    }

    def __init__(self, grid_size=(20, 10), tetrominoes=None, render_mode=None, ticks_per_drop=1):
        super().__init__()

        self.render_mode = render_mode
        self.grid_size = grid_size
        self.tetrominoes = tetrominoes
        self.ticks_per_drop = ticks_per_drop
        self.state = None
        self.game = TetrisGame(
            grid_size=self.grid_size,
            tetrominoes=self.tetrominoes,
            ticks_per_drop=self.ticks_per_drop,
            rng=self.np_random,
        )

        self.action_space = spaces.Discrete(len(Actions))
        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=grid_size, dtype=np.int32
        # )
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size[0] * grid_size[1],), dtype=np.int32)

        self.renderer = None
        if self.render_mode in {"pygame", "human"}:
            self.renderer = TetrisPyGameRenderer()
        elif self.render_mode == "ansi":
            self.renderer = TetrisASCIIRenderer()
        elif self.render_mode == "rgb_array":
            self.renderer = TetrisRGBArrayRenderer()

    def reset(self, seed=None, options=None):
        """Reset the environment to start a new game."""
        super().reset(seed=seed)
        if seed is not None:
            self.game = TetrisGame(
                grid_size=self.grid_size,
                tetrominoes=self.tetrominoes,
                ticks_per_drop=self.ticks_per_drop,
                rng=self.np_random,
            )

        self.state = self.game.reset()
        return self._get_observation(), self._get_info()

    def _get_info(self):
        return {
            "score": self.state["score"],
            "lines_cleared": self.state["lines_cleared"],
            "current_ticks": self.state["current_ticks"],
            **self.state["metrics"],
        }

    def _get_observation(self) -> np.ndarray:
        """Return the current state of the game grid."""
        grid = self.state["grid"].copy()
        grid[grid > 0] = 1
        return grid.flatten()

    def step(self, action):
        self.state, game_over, drop_distance, lines_cleared = self.game.step(action)
        reward = self.calculate_reward(game_over, drop_distance, lines_cleared)

        observation = self._get_observation()
        info = self._get_info()

        # observation, reward, terminated, truncated, info
        return observation, reward, game_over, False, info

    def calculate_reward(self, game_over, drop_distance, lines_cleared):
        """Base reward function to be overridden by subclasses."""
        return 1 if not game_over else -100

    def render(self):
        """Render the current state of the game."""
        if self.renderer is not None:
            return self.renderer.render(self.state)


class StandardRewardTetrisEnv(BaseRewardTetrisEnv):
    """Tetris environment with standard scoring-based rewards."""

    GAME_OVER_PENALTY = -100
    HEIGHT_PENALTY = -0.51
    LINE_REWARD = 0.76
    HOLE_PENALTY = -0.36
    BUMPINESS_PENALTY = -0.18

    # -0.51 × Height
    # 0.76 × Lines
    # -0.36 × Holes
    # -0.18 × Bumpiness

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitness = 0
        self.score = 0

    def reset(self, *args, **kwargs):
        """Reset the environment to start a new game."""
        self.fitness = 0
        self.score = 0
        return super().reset(*args, **kwargs)

    def calculate_reward(self, game_over, drop_distance, lines_cleared):
        """Custom reward function."""
        reward = 0
        holes = self.state["metrics"]["holes"]
        bumpiness = self.state["metrics"]["bumpiness"]
        height = self.state["metrics"]["sum_height"]

        new_fitness = -0.51 * height + 0.76 * lines_cleared - 0.36 * holes - 0.18 * bumpiness
        change_in_fitness = new_fitness - self.fitness
        self.fitness = new_fitness
        reward += change_in_fitness

        # score = self.state["score"]
        # change_in_score = score - self.score
        # self.score = score
        # reward += change_in_score

        if game_over:
            reward += self.GAME_OVER_PENALTY

        return reward


class StandardReward2TetrisEnv(StandardRewardTetrisEnv):
    """Tetris environment with standard scoring-based rewards but more for lines"""

    LINE_REWARD = [0, 100, 250, 400, 550]

    def calculate_reward(self, game_over, drop_distance, lines_cleared):
        """Custom reward function."""
        reward = super().calculate_reward(game_over, drop_distance, lines_cleared)
        reward += self.LINE_REWARD[lines_cleared]
        return reward


class MyTetrisEnv(BaseRewardTetrisEnv):
    """Tetris environment with custom rewards."""

    GAME_OVER_REWARD = -100
    PLACED_REWARD = 0.5
    LINE_REWARD = [0, 100, 250, 400, 550]
    HEIGHT_PENALTY = -0.1
    HOLE_PENALTY = -2.0
    BUMPINESS_PENALTY = -0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bumpiness = 0
        self.holes = 0
        self.max_height = 0

    def reset(self, *args, **kwargs):
        """Reset the environment to start a new game."""
        self.bumpiness = 0
        self.holes = 0
        self.max_height = 0
        return super().reset(*args, **kwargs)

    def calculate_reward(self, game_over, drop_distance, lines_cleared):
        """Custom reward function."""
        reward = 0
        reward += drop_distance

        holes = self.state["metrics"]["holes"]
        bumpiness = self.state["metrics"]["bumpiness"]
        max_height = self.state["metrics"]["max_height"]

        delta_bumpiness = bumpiness - self.bumpiness
        delta_holes = holes - self.holes
        delta_max_height = max_height - self.max_height

        # Penalties for creating holes or increasing bumpiness
        reward += delta_holes * self.HOLE_PENALTY
        reward += delta_bumpiness * self.BUMPINESS_PENALTY

        # Penalty for high towers
        reward += delta_max_height * self.HEIGHT_PENALTY

        self.bumpiness = bumpiness
        self.holes = holes
        self.max_height = max_height

        # if piece_placed:
        #     reward += self.PLACED_REWARD

        reward += self.LINE_REWARD[lines_cleared]

        if game_over:
            reward += self.GAME_OVER_REWARD

        return reward


class MyTetrisEnv2(BaseRewardTetrisEnv):
    """Tetris environment with custom rewards and RGB array observations."""

    GAME_OVER_REWARD = -100
    PLACED_REWARD = 0.5
    LINE_REWARD = [0, 100, 250, 400, 550]
    HEIGHT_PENALTY = -0.1
    HOLE_PENALTY = -2.0
    BUMPINESS_PENALTY = -0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The minimal resolution for an image is 36x36 for the default `CnnPolicy`.
        # You might need to use a custom features extractor cf.
        # FIX: np.kron(array, np.ones(2,2)) -> scale 2x
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, self.grid_size[0], self.grid_size[1]),
            dtype=np.uint8,
        )
        self.bumpiness = 0
        self.holes = 0
        self.max_height = 0

    def reset(self, *args, **kwargs):
        """Reset the environment to start a new game."""
        self.bumpiness = 0
        self.holes = 0
        self.max_height = 0
        return super().reset(*args, **kwargs)

    def _get_observation(self) -> np.ndarray:
        """Return the current state of the game grid."""
        grid = self.state["grid"].copy()
        np.putmask(grid, grid > 0, 255)
        # axis=0 channel first
        # axis=1 channel last
        return np.expand_dims(grid, axis=0).astype(np.uint8)

    def calculate_reward(self, game_over, drop_distance, lines_cleared):
        """Custom reward function."""
        reward = 0
        reward += drop_distance

        holes = self.state["metrics"]["holes"]
        bumpiness = self.state["metrics"]["bumpiness"]
        max_height = self.state["metrics"]["max_height"]

        delta_bumpiness = bumpiness - self.bumpiness
        delta_holes = holes - self.holes
        delta_max_height = max_height - self.max_height

        # Penalties for creating holes or increasing bumpiness
        reward += delta_holes * self.HOLE_PENALTY
        reward += delta_bumpiness * self.BUMPINESS_PENALTY

        # Penalty for high towers
        reward += delta_max_height * self.HEIGHT_PENALTY

        self.bumpiness = bumpiness
        self.holes = holes
        self.max_height = max_height

        reward += self.LINE_REWARD[lines_cleared]

        if game_over:
            reward += self.GAME_OVER_REWARD

        return reward


def clear_screen():
    """Clears the console screen (cross-platform compatible)."""
    print("\033c\033[3J", end="")


# Example usage
def play_tetris():
    """Main function to run the Tetris environment."""
    env = StandardRewardTetrisEnv(render_mode="ascii")  # Initialize the Tetris environment
    _observation, _info = env.reset()  # Reset the environment to start a new game

    terminated = False
    try:
        while not terminated:
            clear_screen()  # Clear the console screen
            env.render()  # Render the game state
            action = env.action_space.sample()  # Sample a random action
            _observation, _reward, terminated, _truncated, info = env.step(action)  # Take a step in the environment
            time.sleep(0.05)  # Control game speed
    finally:
        clear_screen()  # Clear the screen before exiting
        env.render()  # Render final game state
        env.close()  # Close the environment


if __name__ == "__main__":
    play_tetris()
