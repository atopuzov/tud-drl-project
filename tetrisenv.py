"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is," without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tetrisgame import Actions, TetrisGame
from tetrisrenderer import TetrisASCIIRenderer, TetrisPyGameRenderer, TetrisRenderer, TetrisRGBArrayRenderer

env_kwargs = {
    "grid_size": (20, 10),
    "tetrominoes": None,
    "render_mode": None,
    # "ticks_per_drop": 1,
}
gym.register("Tetris-base", entry_point="tetrisenv:BaseRewardTetrisEnv", kwargs=env_kwargs)
gym.register("Tetris-score", entry_point="tetrisenv:ScoreRewardTetrisEnv", kwargs=env_kwargs)
gym.register("Tetris-simpleheuristic", entry_point="tetrisenv:SimpleHeuristicTetris", kwargs=env_kwargs)

gym.register("Tetris-heuristic1", entry_point="tetrisenv:HeuristicRewardTetrisEnv", kwargs=env_kwargs)
gym.register("Tetris-heuristic2", entry_point="tetrisenv:HeuristicReward2TetrisEnv", kwargs=env_kwargs)
gym.register("Tetris-heuristic3", entry_point="tetrisenv:HeuristicReward3TetrisEnv", kwargs=env_kwargs)

gym.register("Tetris-v2", entry_point="tetrisenv:MyTetrisEnv", kwargs=env_kwargs)
gym.register("Tetris-imgh", entry_point="tetrisenv:ImgHTetrisEnv", kwargs=env_kwargs)
gym.register("Tetris-v3", entry_point="tetrisenv:MyTetrisEnv2", kwargs=env_kwargs)


class BaseTetrisEnv(gym.Env):
    """Base Tetris Gym Environment"""

    metadata = {
        "render_modes": ["pygame", "ansi", "rgb_array", "human"],
        "render_fps": 4,
    }

    def __init__(self, grid_size=(20, 10), tetrominoes: Optional[List[str]] = None, render_mode=None, ticks_per_drop=1):
        super().__init__()

        self.render_mode = render_mode
        self.grid_size = grid_size
        self.tetrominoes = tetrominoes
        self.ticks_per_drop = ticks_per_drop
        self.state: Optional[Dict] = None
        self.game = TetrisGame(
            grid_size=self.grid_size,
            tetrominoes=self.tetrominoes,
            ticks_per_drop=self.ticks_per_drop,
            rng=self.np_random,
        )

        self.renderer: Optional[TetrisRenderer] = None
        if self.render_mode in {"pygame", "human"}:
            self.renderer = TetrisPyGameRenderer()
        elif self.render_mode == "ansi":
            self.renderer = TetrisASCIIRenderer()
        elif self.render_mode == "rgb_array":
            self.renderer = TetrisRGBArrayRenderer()

        self.action_space = spaces.Discrete(len(Actions))

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray | Dict, Dict]:
        """Reset the environment to start a new game."""
        super().reset(seed=seed)
        self.game.rng = self.np_random
        self.state = self.game.reset()  # todo pass random
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        self.state, game_over, drop_distance, lines_cleared = self.game.step(Actions(action))
        reward = self.calculate_reward(game_over, drop_distance, lines_cleared)

        observation = self._get_observation()
        info = self._get_info()

        # observation, reward, terminated, truncated, info
        return observation, reward, game_over, False, info

    def _get_info(self) -> Dict[str, Any]:
        assert self.state is not None, "self.state should not be None"
        return {
            "score": self.state["score"],
            "lines_cleared": self.state["lines_cleared"],
            "pieces_placed": self.state["pieces_placed"],
            "current_ticks": self.state["current_ticks"],
            **self.state["metrics"],
        }

    def render(self):
        """Render the current state of the game."""
        assert self.state is not None, "self.state should not be None"
        if self.renderer is not None:
            return self.renderer.render(self.state)

    def _get_observation(self):
        pass

    def calculate_reward(self, game_over, drop_distance, lines_cleared):
        pass


class BaseRewardTetrisEnv(BaseTetrisEnv):
    """Base Tetris environment with minimal reward structure."""

    GAME_OVER_PENALTY = -100

    def __init__(self, grid_size=(20, 10), tetrominoes: Optional[List[str]] = None, render_mode=None):
        super().__init__(grid_size=grid_size, tetrominoes=tetrominoes, render_mode=render_mode)
        # self.observation_space = spaces.Box(low=0, high=1, shape=grid_size, dtype=np.int32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size[0] * grid_size[1],), dtype=np.int32)

    def _get_observation(self) -> np.ndarray | Dict:
        """Return the current state of the game grid."""
        assert self.state is not None, "self.state should not be None"
        grid = self.state["grid"].copy()
        grid[grid > 0] = 1
        return grid.flatten()

    def calculate_reward(self, game_over, drop_distance, lines_cleared) -> int:
        """Base reward function to be overridden by subclasses."""
        return 1 if not game_over else self.GAME_OVER_PENALTY


class ScoreRewardTetrisEnv(BaseRewardTetrisEnv):
    GAME_OVER_PENALTY = -100

    def __init__(self, grid_size=(20, 10), tetrominoes: Optional[List[str]] = None, render_mode=None):
        super().__init__(grid_size=grid_size, tetrominoes=tetrominoes, render_mode=render_mode)
        self.score = 0

    def reset(self, seed=None):
        self.score = 0
        return super().reset(seed=seed)

    def calculate_reward(self, game_over, drop_distance, lines_cleared) -> int:
        """Just uses score and game over"""
        assert self.state is not None, "self.state should not be None"
        reward = 0
        score = self.state.get("score", 0)
        reward += score - self.score
        self.score = score
        if game_over:
            reward += self.GAME_OVER_PENALTY

        return reward


class SimpleHeuristicTetris(BaseTetrisEnv):
    """A very simple environment using board features"""

    GAME_OVER_PENALTY = -100
    HEIGHT_PENALTY = -0.51
    LINE_REWARD = 0.76
    HOLE_PENALTY = -0.36
    BUMPINESS_PENALTY = -0.18

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Dict(
            {
                "bumpiness": spaces.Box(0, self.grid_size[0] * self.grid_size[1], dtype=np.uint8),
                "holes": spaces.Box(0, self.grid_size[0] * self.grid_size[1], dtype=np.uint8),
                "min_height": spaces.Box(0, self.grid_size[0], dtype=np.uint8),
                "max_height": spaces.Box(0, self.grid_size[0], dtype=np.uint8),
                # "heights": spaces.Box(0, self.grid.size[0], shape=(self.grid_size[0] * self.grid_size[1], dtype=np.uint8))
            }
        )
        self.fitness = 0
        self.score = 0

    def reset(self, *args, **kwargs):
        """Reset the environment to start a new game."""
        self.fitness = 0
        self.score = 0
        return super().reset(*args, **kwargs)

    def _get_observation(self) -> Dict:
        assert self.state is not None, "self.state should not be None"
        return {
            "bumpiness": self.state["metrics"]["bumpiness"],
            "holes": self.state["metrics"]["holes"],
            "min_height": self.state["metrics"]["min_height"],
            "max_height": self.state["metrics"]["max_height"],
            # "heights": self.heights,
        }

    def calculate_reward(self, game_over, drop_distance, lines_cleared) -> int:
        assert self.state is not None, "self.state should not be None"
        reward = 0
        holes = self.state["metrics"]["holes"]
        bumpiness = self.state["metrics"]["bumpiness"]
        height = self.state["metrics"]["sum_height"]

        new_fitness = (
            self.HEIGHT_PENALTY * height
            + self.LINE_REWARD * lines_cleared
            + self.HOLE_PENALTY * holes
            + self.BUMPINESS_PENALTY * bumpiness
        )
        change_in_fitness = new_fitness - self.fitness
        self.fitness = new_fitness

        reward += change_in_fitness

        if game_over:
            reward += self.GAME_OVER_PENALTY
        return reward


class HeuristicRewardTetrisEnv(BaseRewardTetrisEnv):
    """Tetris environment with standard scoring-based rewards."""

    GAME_OVER_PENALTY = -100
    HEIGHT_PENALTY = -0.51
    LINE_REWARD = 0.76
    HOLE_PENALTY = -0.36
    BUMPINESS_PENALTY = -0.18

    def __init__(
        self,
        use_score=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fitness = 0
        self.score = 0
        self.use_score = use_score

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

        new_fitness = (
            self.HEIGHT_PENALTY * height
            + self.LINE_REWARD * lines_cleared
            + self.HOLE_PENALTY * holes
            + self.BUMPINESS_PENALTY * bumpiness
        )
        change_in_fitness = new_fitness - self.fitness
        self.fitness = new_fitness
        reward += change_in_fitness

        if self.use_score:
            score = self.state["score"]
            delta_score = score - self.score
            self.score = score
            reward += delta_score

        if game_over:
            reward += self.GAME_OVER_PENALTY

        return reward


class HeuristicReward2TetrisEnv(HeuristicRewardTetrisEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(use_score=True, *args, **kwargs)


class HeuristicReward3TetrisEnv(HeuristicRewardTetrisEnv):
    """Tetris environment with standard scoring-based rewards but more for lines"""

    LINES_REWARDS = [0, 100, 250, 400, 550]

    def calculate_reward(self, game_over, drop_distance, lines_cleared):
        """Custom reward function."""
        reward = super().calculate_reward(game_over, drop_distance, lines_cleared)
        reward += self.LINES_REWARD[lines_cleared]
        return reward


class MyTetrisEnv(BaseRewardTetrisEnv):
    """Tetris environment with custom rewards."""

    GAME_OVER_REWARD = -100
    PLACED_REWARD = 0.5
    LINES_REWARD = [0, 100, 250, 400, 550]
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

        reward += self.LINES_REWARD[lines_cleared]

        if game_over:
            reward += self.GAME_OVER_REWARD

        return reward


class ImageTetrisBaseEnv(BaseTetrisEnv):
    """Tetris environment with image observation"""

    def __init__(self, grid_size=(20, 10), tetrominoes: Optional[List[str]] = None, render_mode=None):
        super().__init__(grid_size=grid_size, tetrominoes=tetrominoes, render_mode=render_mode)
        # The minimal resolution for an image is 36x36 for the default `CnnPolicy`.
        # You might need to use a custom features extractor cf.
        # FIX: np.kron(array, np.ones(2,2)) -> scale 2x
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, self.grid_size[0], self.grid_size[1]),
            dtype=np.uint8,
        )

    def _get_observation(self) -> np.ndarray:
        """Return the current state of the game grid."""
        assert self.state is not None, "self.state should not be None"
        grid = self.state["grid"].copy()
        np.putmask(grid, grid > 0, 255)
        # axis=0 channel first
        # axis=1 channel last
        return np.expand_dims(grid, axis=0).astype(np.uint8)


class ImgHTetrisEnv(ImageTetrisBaseEnv):
    """Image observation and heuristic award"""

    GAME_OVER_PENALTY = -100
    HEIGHT_PENALTY = -0.51
    LINE_REWARD = 0.76
    HOLE_PENALTY = -0.36
    BUMPINESS_PENALTY = -0.18

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fitness = 0
        self.score = 0
        self.use_score = False

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

        new_fitness = (
            self.HEIGHT_PENALTY * height
            + self.LINE_REWARD * lines_cleared
            + self.HOLE_PENALTY * holes
            + self.BUMPINESS_PENALTY * bumpiness
        )
        change_in_fitness = new_fitness - self.fitness
        self.fitness = new_fitness
        reward += change_in_fitness

        if self.use_score:
            score = self.state["score"]
            delta_score = score - self.score
            self.score = score
            reward += delta_score

        if game_over:
            reward += self.GAME_OVER_PENALTY

        return reward


class MyTetrisEnv2(ImageTetrisBaseEnv):
    """Tetris environment with custom rewards and RGB array observations."""

    GAME_OVER_REWARD = -100
    PLACED_REWARD = 0.5
    LINE_REWARD = [0, 100, 250, 400, 550]
    HEIGHT_PENALTY = -0.1
    HOLE_PENALTY = -2.0
    BUMPINESS_PENALTY = -0.5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bumpiness: int = 0
        self.holes: int = 0
        self.max_height: int = 0

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
    env = BaseRewardTetrisEnv(render_mode="ansi")  # Initialize the Tetris environment
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
