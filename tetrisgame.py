"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is," without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import time
from collections import defaultdict
from enum import IntEnum, unique

import numpy as np


@unique
class Actions(IntEnum):
    """
    Enum class representing possible actions in the Tetris game.

    Attributes:
        LEFT (int): Action to move the piece to the left.
        RIGHT (int): Action to move the piece to the right.
        ROTATE (int): Action to rotate the piece.
        HARD_DROP (int): Action to perform a hard drop of the piece.
        # DROP (int): Action to perform a drop of the piece.
    """

    LEFT = 0
    RIGHT = 1
    ROTATE = 2
    HARD_DROP = 3
    # DROP = 4


@unique
class Rotation(IntEnum):
    """
    An enumeration representing the four possible rotation states for Tetris pieces.

    Attributes:
        R000 (int): Represents a 0-degree rotation.
        R090 (int): Represents a 90-degree rotation.
        R180 (int): Represents a 180-degree rotation.
        R270 (int): Represents a 270-degree rotation.
    """

    R000 = 0
    R090 = 1
    R180 = 2
    R270 = 3


def next_rotation(rot: Rotation) -> Rotation:
    """Get the next rotation"""
    if rot == Rotation.R000:
        return Rotation.R090
    elif rot == Rotation.R090:
        return Rotation.R180
    elif rot == Rotation.R180:
        return Rotation.R270
    return Rotation.R000


class TetrisGame:
    """Core Tetris game logic independent of any specific interface."""

    LINE_SCORE = [0, 40, 100, 300, 1200]

    grid_dtype = np.int32
    TETROMINOES = {
        "I": np.array([[1, 1, 1, 1]], dtype=grid_dtype),
        "O": np.array([[1, 1], [1, 1]], dtype=grid_dtype),
        "T": np.array([[0, 1, 0], [1, 1, 1]], dtype=grid_dtype),
        "S": np.array([[0, 1, 1], [1, 1, 0]], dtype=grid_dtype),
        "Z": np.array([[1, 1, 0], [0, 1, 1]], dtype=grid_dtype),
        "J": np.array([[1, 0, 0], [1, 1, 1]], dtype=grid_dtype),
        "L": np.array([[0, 0, 1], [1, 1, 1]], dtype=grid_dtype),
    }
    TETROMINOES_COLOR = {
        "I": 1,
        "O": 2,
        "T": 3,
        "S": 4,
        "Z": 5,
        "J": 6,
        "L": 7,
    }

    # Color tetrominos
    for tetromino, shape in TETROMINOES.items():
        shape[shape == 1] = TETROMINOES_COLOR.get(tetromino, 1)

    # Naive rotations, some elements only have 2 rotations
    TETROMINOES_ROT: defaultdict[str, dict[Rotation, np.ndarray]] = defaultdict(dict)
    for tetromino, shape in TETROMINOES.items():
        for rotation in Rotation:
            TETROMINOES_ROT[tetromino][rotation] = np.rot90(shape, rotation)

    def __init__(
        self, grid_size=(20, 10), tetrominoes=None, rng=None, ticks_per_drop=1
    ):
        self.grid_size = grid_size
        self.board_height, self.board_width = grid_size
        self.tetrominoes = tetrominoes or list(self.TETROMINOES.keys())
        self.rng = rng if rng is not None else np.random.default_rng()
        self.ticks_per_drop = ticks_per_drop

        # Game state
        self.grid = None
        self.current_piece = None
        self.current_piece_type = None
        self.next_piece_type = None
        self.tetromino_position = None
        self.rotation = Rotation.R000
        self.score = 0
        self.lines_cleared = 0
        self.current_ticks = 0

        # Board analysis metrics
        self.bumpiness = 0
        self.holes = 0
        self.heights = np.zeros(self.board_width)
        self.min_height = 0
        self.max_height = 0
        self.sum_height = 0
        self.pieces_placed = 0

        self.reset()

    def reset(self):
        """Reset the game state."""
        self.grid = np.zeros(self.grid_size, dtype=self.grid_dtype)
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.current_ticks = 0
        self._reset_metrics()
        self.current_piece = self._get_next_tetromino()
        self.rotation = Rotation.R000
        return self.get_state()

    def _reset_metrics(self):
        """Reset board analysis metrics."""
        self.bumpiness = 0
        self.holes = 0
        self.heights = np.zeros(self.board_width)
        self.min_height = 0
        self.max_height = 0
        self.sum_height = 0

    def get_state(self):
        """Return current game state including grid and metrics."""
        return {
            "grid": self.get_grid(),
            "width": self.board_width,
            "height": self.board_height,
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "pieces_placed": self.pieces_placed,
            "current_ticks": self.current_ticks,
            "metrics": {
                "bumpiness": self.bumpiness,
                "holes": self.holes,
                "heights": self.heights.copy(),
                "min_height": self.min_height,
                "max_height": self.max_height,
                "sum_height": self.sum_height,
            },
        }

    def step(self, action):
        """Execute one step and return if piece was placed."""
        piece_placed = False
        game_over = False
        hard_drop = False
        drop_distance = 0
        lines_cleared = 0

        # Process action
        if action == Actions.LEFT:
            self._move_tetromino(dx=-1)
        elif action == Actions.RIGHT:
            self._move_tetromino(dx=1)
        elif action == Actions.ROTATE:
            self._rotate_tetromino()
        elif action == Actions.HARD_DROP:
            drop_distance = self._hard_drop()
            self.score += drop_distance  # Score for hard drop
            hard_drop = True
        # elif action == Actions.DROP:
        #     self._move_tetromino(dy=1)

        # Handle ticks and piece dropping
        self.current_ticks += 1
        should_drop = self.current_ticks >= self.ticks_per_drop

        if should_drop or hard_drop:
            self.current_ticks = 0
            if not self._move_tetromino(dy=1):
                self._place_tetromino()
                piece_placed = True
                self.pieces_placed += 1

                lines_cleared = self._clear_lines()
                if lines_cleared > 0:
                    self.lines_cleared += lines_cleared
                    self.score += self._calculate_score(lines_cleared)

                self._update_metrics()
                self.current_piece = self._get_next_tetromino()

                if self.current_piece is None:
                    game_over = True

        return self.get_state(), game_over, drop_distance, lines_cleared  # piece_placed

    def _get_next_tetromino(self):
        """Get the next tetromino piece."""
        self.current_piece_type = self.rng.choice(self.tetrominoes)
        self.rotation = Rotation.R000

        tetromino_shape = self.TETROMINOES_ROT[self.current_piece_type][self.rotation]
        spawn_position = (0, self.board_width // 2 - len(tetromino_shape[0]) // 2)

        if self._is_valid_position(spawn_position, tetromino_shape):
            self.tetromino_position = spawn_position
            return tetromino_shape
        return None

    def _move_tetromino(self, dx=0, dy=0):
        """Move the tetromino by the given delta. Returns True if successful."""
        new_position = (
            self.tetromino_position[0] + dy,
            self.tetromino_position[1] + dx,
        )
        if self._is_valid_position(new_position, self.current_piece):
            self.tetromino_position = new_position
            return True
        return False

    def _rotate_tetromino(self):
        """Rotate the current tetromino if possible."""
        new_rotation = next_rotation(self.rotation)
        new_piece = self.TETROMINOES_ROT[self.current_piece_type][new_rotation]
        if self._is_valid_position(self.tetromino_position, new_piece):
            self.rotation = new_rotation
            self.current_piece = new_piece

    def _hard_drop(self):
        """Drop the tetromino to the bottom. Returns the distance dropped."""
        drop_distance = 0
        while self._move_tetromino(dy=1):
            drop_distance += 1
        return drop_distance

    def _is_valid_position(self, position, tetromino):
        """Check if the tetromino position is valid."""
        y, x = position
        height, width = tetromino.shape

        if (
            y < 0
            or y + height > self.board_height
            or x < 0
            or x + width > self.board_width
        ):
            return False

        for i, row in enumerate(tetromino):
            for j, cell in enumerate(row):
                if cell and self.grid[y + i, x + j]:
                    return False
        return True

    def _place_tetromino(self):
        """Place the current tetromino on the grid."""
        y, x = self.tetromino_position
        for i, row in enumerate(self.current_piece):
            for j, cell in enumerate(row):
                if cell:
                    self.grid[y + i, x + j] = cell

    def _clear_lines(self):
        """Clear completed lines and return number cleared."""
        filled_rows = [y for y in range(self.grid_size[0]) if np.all(self.grid[y])]
        lines_cleared = len(filled_rows)

        if lines_cleared > 0:
            self.grid = np.delete(self.grid, filled_rows, axis=0)
            self.grid = np.vstack(
                [
                    np.zeros((lines_cleared, self.grid_size[1]), dtype=self.grid_dtype),
                    self.grid,
                ]
            )

        return lines_cleared

    def get_grid(self):
        """Get the current grid state including active piece."""
        grid = self.grid.copy()

        if self.current_piece is not None:
            y, x = self.tetromino_position
            for i, row in enumerate(self.current_piece):
                for j, cell in enumerate(row):
                    if cell:
                        grid[y + i, x + j] = cell
        return grid

    def _update_metrics(self):
        """Update board analysis metrics."""
        (
            self.bumpiness,
            self.holes,
            self.heights,
            self.min_height,
            self.max_height,
            self.sum_height,
        ) = self._analyze_board()

    def _calculate_score(self, lines_cleared):
        """Calculate score based on lines cleared."""
        # 40 * (level + 1) 	100 * (level + 1) 	300 * (level + 1) 	1200 * (level + 1)
        return self.LINE_SCORE[lines_cleared]

    def _analyze_board(self):
        """Analyze the current board state."""
        column_heights = np.zeros(self.board_width, dtype=np.int32)
        min_height = self.board_height
        max_height = 0
        sum_height = 0
        holes = 0
        bumpiness = 0

        # Go trough the board
        for j in range(self.board_width):
            found_block = False
            col_height = 0
            col_holes = 0

            for i in range(self.board_height):
                if self.grid[i, j]:
                    if not found_block:
                        col_height = self.board_height - i
                        found_block = True
                if found_block and not self.grid[i, j]:
                    col_holes += 1  # Count holes below the first block.

            column_heights[j] = col_height
            holes += col_holes
            sum_height += col_height

            if col_height < min_height:
                min_height = col_height
            if col_height > max_height:
                max_height = col_height

            if j > 0:
                bumpiness += abs(column_heights[j] - column_heights[j - 1])

        return bumpiness, holes, column_heights, min_height, max_height, sum_height


def ascii_render(observation):
    """Render the game state as text in the console."""

    def clear_screen():
        """Clears the console screen."""
        print("\033[2J\033[H", end="")

    TETROMINOES_ASCII_COLORS = {
        1: "\033[36m",  # Cyan
        2: "\033[33m",  # Yellow
        3: "\033[35m",  # Purple
        4: "\033[32m",  # Green
        5: "\033[31m",  # Red
        6: "\033[34m",  # Blue
        7: "\033[38;5;214m",  # Orange
    }

    def cell_str(cell, wide=1, piece="█", space="."):
        color = TETROMINOES_ASCII_COLORS.get(cell, "")
        end = "\033[0m"
        return f"{color}{piece*wide}{end}" if cell else space * wide

    grid = observation["grid"]
    board_width = observation["width"]
    wide = 2

    clear_screen()
    print("\n+" + "-" * board_width * wide + "+")
    for row in grid:
        print("|" + "".join(cell_str(cell, wide=wide) for cell in row) + "|")
    print("+" + "-" * board_width * wide + "+")


def play_tetris():
    """
    Play a game of Tetris.

    This function initializes a Tetris game, continuously renders the game state,
    randomly selects actions, and updates the game state until the game is terminated.
    The game speed is controlled by a sleep interval.

    The game state is rendered in ASCII format.

    Raises:
        Exception: If an error occurs during the game execution.
    """
    game = TetrisGame()
    terminated = False
    try:
        observation = game.reset()
        while not terminated:
            ascii_render(observation)  # Render the game state
            action = game.rng.choice(list(Actions))
            observation, terminated, _, _ = game.step(
                action
            )  # Take a step in the environment
            time.sleep(0.05)  # Control game speed
    finally:
        ascii_render(observation)  # Render final game state


if __name__ == "__main__":
    play_tetris()
