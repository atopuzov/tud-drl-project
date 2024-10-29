import numpy as np
import pytest

from tetrisgame import Actions, Rotation, TetrisGame, next_rotation


def test_next_rotation_R000():
    assert next_rotation(Rotation.R000) == Rotation.R090


def test_next_rotation_R090():
    assert next_rotation(Rotation.R090) == Rotation.R180


def test_next_rotation_R180():
    assert next_rotation(Rotation.R180) == Rotation.R270


def test_next_rotation_R270():
    assert next_rotation(Rotation.R270) == Rotation.R000


def test_tetris_game_initial_state():
    game = TetrisGame()
    state = game.reset()
    assert state["score"] == 0
    assert state["lines_cleared"] == 0
    assert state["pieces_placed"] == 0
    assert state["current_ticks"] == 0
    assert state["metrics"]["bumpiness"] == 0
    assert state["metrics"]["holes"] == 0
    assert state["metrics"]["min_height"] == 0
    assert state["metrics"]["max_height"] == 0
    assert state["metrics"]["sum_height"] == 0


def test_tetris_game_step_left():
    game = TetrisGame()
    game.reset()
    initial_position = game.tetromino_position
    game.step(Actions.LEFT)
    assert game.tetromino_position[1] == initial_position[1] - 1


def test_tetris_game_step_right():
    game = TetrisGame()
    game.reset()
    initial_position = game.tetromino_position
    game.step(Actions.RIGHT)
    assert game.tetromino_position[1] == initial_position[1] + 1


def test_tetris_game_step_rotate():
    game = TetrisGame()
    game.reset()
    initial_rotation = game.rotation
    game.step(Actions.ROTATE)
    assert game.rotation == next_rotation(initial_rotation)


def test_tetris_game_step_hard_drop():
    game = TetrisGame()
    game.reset()
    initial_position = game.tetromino_position
    game.step(Actions.HARD_DROP)
    assert game.tetromino_position[0] > initial_position[0]


def test_tetris_game_step_left():
    game = TetrisGame()
    game.reset()
    initial_position = game.tetromino_position
    game.step(Actions.LEFT)
    assert game.tetromino_position[1] == initial_position[1] - 1


def test_tetris_game_step_right():
    game = TetrisGame()
    game.reset()
    initial_position = game.tetromino_position
    game.step(Actions.RIGHT)
    assert game.tetromino_position[1] == initial_position[1] + 1


def test_tetris_game_step_rotate():
    game = TetrisGame()
    game.reset()
    initial_rotation = game.rotation
    game.step(Actions.ROTATE)
    assert game.rotation == next_rotation(initial_rotation)


def test_tetris_game_step_hard_drop():
    game = TetrisGame()
    game.reset()
    initial_position = game.tetromino_position
    _, game_over, drop_distance, lines_cleared = game.step(Actions.HARD_DROP)
    assert game.tetromino_position[0] > initial_position[0]
    assert drop_distance > 0
    assert not game_over
    assert lines_cleared == 0


def test_tetris_game_step_piece_placed():
    game = TetrisGame()
    game.reset()
    game.ticks_per_drop = 1  # Ensure piece drops every step
    initial_pieces_placed = game.pieces_placed
    _, game_over, _, _ = game.step(Actions.HARD_DROP)
    assert game.pieces_placed == initial_pieces_placed + 1
    assert not game_over


def test_tetris_game_step_game_over():
    game = TetrisGame(grid_size=(4, 4))  # Small grid to force game over quickly
    game.reset()
    game.grid[0:3, :] = 1  # Fill the grid to leave only one row
    _, game_over, _, _ = game.step(Actions.HARD_DROP)
    assert game_over


def test_tetris_game_step_lines_cleared():
    game = TetrisGame(grid_size=(4, 4))  # Small grid for easier testing
    game.reset()
    game.grid[3, :] = 1  # Fill the bottom row
    _, game_over, _, lines_cleared = game.step(Actions.HARD_DROP)
    assert lines_cleared == 1
    assert not game_over


def test_move_tetromino_left_success():
    game = TetrisGame()
    game.reset()
    initial_position = game.tetromino_position
    success = game._move_tetromino(dx=-1)
    assert success
    assert game.tetromino_position[1] == initial_position[1] - 1


def test_move_tetromino_right_success():
    game = TetrisGame()
    game.reset()
    initial_position = game.tetromino_position
    success = game._move_tetromino(dx=1)
    assert success
    assert game.tetromino_position[1] == initial_position[1] + 1


def test_move_tetromino_down_success():
    game = TetrisGame()
    game.reset()
    initial_position = game.tetromino_position
    success = game._move_tetromino(dy=1)
    assert success
    assert game.tetromino_position[0] == initial_position[0] + 1


def test_move_tetromino_left_fail():
    game = TetrisGame()
    game.reset()
    game.tetromino_position = (0, 0)  # Place piece at the left edge
    success = game._move_tetromino(dx=-1)
    assert not success
    assert game.tetromino_position == (0, 0)


def test_move_tetromino_right_fail():
    game = TetrisGame()
    game.reset()
    game.tetromino_position = (
        0,
        game.board_width - len(game.current_piece[0]),
    )  # Place piece at the right edge
    success = game._move_tetromino(dx=1)
    assert not success
    assert game.tetromino_position == (
        0,
        game.board_width - len(game.current_piece[0]),
    )


def test_move_tetromino_down_fail():
    game = TetrisGame()
    game.reset()
    game.tetromino_position = (
        game.board_height - len(game.current_piece),
        0,
    )  # Place piece at the bottom
    success = game._move_tetromino(dy=1)
    assert not success
    assert game.tetromino_position == (
        game.board_height - len(game.current_piece),
        0,
    )


def test_hard_drop_distance():
    game = TetrisGame()
    game.reset()
    initial_position = game.tetromino_position
    drop_distance = game._hard_drop()
    assert drop_distance > 0
    assert game.tetromino_position[0] == initial_position[0] + drop_distance


def test_hard_drop_to_bottom():
    game = TetrisGame()
    game.reset()
    initial_position = game.tetromino_position
    game._hard_drop()
    assert game.tetromino_position[0] == game.board_height - len(game.current_piece)


def test_hard_drop_on_filled_row():
    game = TetrisGame()
    game.reset()
    game.grid[game.board_height - 1, :] = 1  # Fill the bottom row
    initial_position = game.tetromino_position
    drop_distance = game._hard_drop()
    assert drop_distance > 0
    assert game.tetromino_position[0] < game.board_height - len(game.current_piece)
    assert game.tetromino_position[0] == initial_position[0] + drop_distance


def test_hard_drop_no_movement():
    game = TetrisGame()
    game.reset()
    game.tetromino_position = (
        game.board_height - len(game.current_piece),
        0,
    )  # Place piece at the bottom
    drop_distance = game._hard_drop()
    assert drop_distance == 0
    assert game.tetromino_position == (game.board_height - len(game.current_piece), 0)


def test_analyze_board_empty():
    game = TetrisGame()
    game.reset()
    bumpiness, holes, column_heights, min_height, max_height, sum_height = (
        game._analyze_board()
    )
    assert bumpiness == 0
    assert holes == 0
    assert np.all(column_heights == 0)
    assert min_height == game.board_height
    assert max_height == 0
    assert sum_height == 0


def test_analyze_board_single_block():
    game = TetrisGame()
    game.reset()
    game.grid[game.board_height - 1, 0] = 1  # Place a single block at the bottom-left
    bumpiness, holes, column_heights, min_height, max_height, sum_height = (
        game._analyze_board()
    )
    assert bumpiness == 0
    assert holes == 0
    assert column_heights[0] == 1
    assert np.all(column_heights[1:] == 0)
    assert min_height == 1
    assert max_height == 1
    assert sum_height == 1


def test_analyze_board_multiple_blocks():
    game = TetrisGame()
    game.reset()
    game.grid[game.board_height - 1, 0] = 1
    game.grid[game.board_height - 2, 1] = 1
    bumpiness, holes, column_heights, min_height, max_height, sum_height = (
        game._analyze_board()
    )
    assert bumpiness == 1
    assert holes == 0
    assert column_heights[0] == 1
    assert column_heights[1] == 2
    assert np.all(column_heights[2:] == 0)
    assert min_height == 1
    assert max_height == 2
    assert sum_height == 3


def test_analyze_board_with_holes():
    game = TetrisGame()
    game.reset()
    game.grid[game.board_height - 1, 0] = 1
    game.grid[game.board_height - 3, 0] = 1  # Create a hole in the first column
    bumpiness, holes, column_heights, min_height, max_height, sum_height = (
        game._analyze_board()
    )
    assert bumpiness == 0
    assert holes == 1
    assert column_heights[0] == 3
    assert np.all(column_heights[1:] == 0)
    assert min_height == 3
    assert max_height == 3
    assert sum_height == 3


def convert_board_to_array(board_str):
    """
    Converts a string representation of a Tetris board into a 2D NumPy array.

    The input board string should have borders represented by any characters
    (e.g., '|', '-', '+') on the first and last lines, and the first and last
    characters of each line. The playable area should be represented by the
    characters '█', '#', or 'x' for filled cells, and '.' or ' ' for empty cells.

    Args:
        board_str (str): The string representation of the Tetris board.

    Returns:
        np.ndarray: A 2D NumPy array where filled cells are represented by 1
                    and empty cells are represented by 0.
    """
    lines = board_str.strip().split("\n")
    height = len(lines) - 2  # Exclude the first and last lines (borders)
    width = len(lines[1]) - 2  # Exclude the first and last characters (borders)

    array = np.zeros((height, width), dtype=int)
    for i, line in enumerate(lines[1:-1]):  # Skip the first and last lines (borders)
        for j, char in enumerate(
            line[1:-1]
        ):  # Skip the first and last characters (borders)
            if char in ("█", "#", "x"):
                array[i, j] = 1
            elif char in (".", " "):
                array[i, j] = 0
    return array


def convert_board_to_array_2x(board_str):
    """
    Converts a board string representation to an array, downsampling the board by a factor of 2 in both dimensions.

    Args:
        board_str (str): The string representation of the board, where each line represents a row of the board.

    Returns:
        np.ndarray: A 2D NumPy array representing the downsampled board.
    """

    def downsample_line(line):
        # Skip the first and last characters (borders) and take every second character in between
        return line[0] + "".join(line[i] for i in range(1, len(line) - 1, 2)) + line[-1]

    lines = board_str.strip().split("\n")
    downsampled_lines = [downsample_line(line) for line in lines]

    downsampled_board_str = "\n".join(downsampled_lines)

    return convert_board_to_array(downsampled_board_str)


TEST_BOARD_1 = """
+--------------------+
|....................|
|....................|
|....................|
|........████........|
|........████........|
|........██████......|
|........██..██......|
|██████████████....██|
|..████..████████████|
|██████████████....██|
|....██████████....██|
|██████..████..██████|
|██████..████████████|
|██..██..██████..████|
|████████..██████████|
|██████████████....██|
|████████..████..████|
|████████..██..██████|
|██████████████..████|
|████████..████..██..|
+--------------------+
"""

TEST_BOARD_2 = """
+----------+
|          |
| x  x     |
| x  x     |
| xxxx     |
+----------+
"""


def test_analyze_board_bumpiness():
    game = TetrisGame(grid_size=(20, 10))  # Smaller grid for easier testing
    game.reset()
    game.grid = convert_board_to_array_2x(TEST_BOARD_1)
    print(game.grid)
    print(game.grid.shape)

    bumpiness, holes, column_heights, min_height, max_height, sum_height = (
        game._analyze_board()
    )
    assert bumpiness == 10  #
    assert holes == 26
    assert min_height == 1
    assert max_height == 3
    assert sum_height == 6


if __name__ == "__main__":
    pytest.main()
