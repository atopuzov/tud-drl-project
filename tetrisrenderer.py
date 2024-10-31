"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is," without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

from typing import Dict, Optional

import numpy as np

from tetrisgame import Actions, TetrisGame


class TetrisRenderer:
    """Base class for Tetris renderers"""

    COLORS = {
        0: (0, 0, 0),  # Empty - Black
        1: (0, 255, 255),  # I - Cyan
        2: (255, 255, 0),  # O - Yellow
        3: (128, 0, 128),  # T - Purple
        4: (0, 255, 0),  # S - Green
        5: (255, 0, 0),  # Z - Red
        6: (0, 0, 255),  # J - Blue
        7: (255, 165, 0),  # L - Orange
    }

    ASCII_COLORS = {
        0: "",  # Empty
        1: "\033[36m",  # I - Cyan
        2: "\033[33m",  # O - Yellow
        3: "\033[35m",  # T - Purple
        4: "\033[32m",  # S - Green
        5: "\033[31m",  # Z - Red
        6: "\033[34m",  # J - Blue
        7: "\033[38;5;214m",  # L - Orange
    }


class TetrisPyGameRenderer(TetrisRenderer):
    """PyGame-based renderer for Tetris"""

    def __init__(self, cell_size: int = 30, info_width: int = 200):
        self.cell_size = cell_size
        self.info_width = info_width
        self.window = None
        self.font = None
        self.info_font = None
        self.grid_surface = None
        self.board_width = None
        self.board_height = None

    def initialize(self, board_width: int, board_height: int) -> None:
        """Initialize PyGame window and surfaces"""
        import pygame

        pygame.init()
        pygame.display.set_caption("Tetris")

        self.board_width = board_width
        self.board_height = board_height

        window_width = board_width * self.cell_size + self.info_width
        window_height = board_height * self.cell_size

        self.window = pygame.display.set_mode((window_width, window_height), pygame.DOUBLEBUF)
        self.font = pygame.font.Font(None, 36)
        self.info_font = pygame.font.Font(None, 24)
        self.grid_surface = pygame.Surface((board_width * self.cell_size, board_height * self.cell_size))

    def render(self, state: Dict) -> None:
        """Render the current game state"""
        import pygame

        if not self.window:
            self.initialize(state["width"], state["height"])

        pygame.event.pump()  # Stop the window from freezing

        # Clear surfaces
        self.window.fill((128, 128, 128))  # Gray background
        self.grid_surface.fill((0, 0, 0))  # Black grid background

        # Draw grid
        grid = state["grid"]
        for y in range(self.board_height):
            for x in range(self.board_width):
                cell_value = grid[y, x]
                if cell_value > 0:
                    self._draw_cell(x, y, cell_value)

        # Draw grid lines
        for x in range(self.board_width + 1):
            pygame.draw.line(
                self.grid_surface,
                (50, 50, 50),
                (x * self.cell_size, 0),
                (x * self.cell_size, self.board_height * self.cell_size),
            )
        for y in range(self.board_height + 1):
            pygame.draw.line(
                self.grid_surface,
                (50, 50, 50),
                (0, y * self.cell_size),
                (self.board_width * self.cell_size, y * self.cell_size),
            )

        # Draw game information
        self._draw_info(state)

        # Update display
        self.window.blit(self.grid_surface, (0, 0))
        pygame.display.flip()

    def _draw_cell(self, x: int, y: int, cell_value: int) -> None:
        """Draw a single cell"""
        import pygame

        rect = pygame.Rect(
            x * self.cell_size + 1,
            y * self.cell_size + 1,
            self.cell_size - 2,
            self.cell_size - 2,
        )
        pygame.draw.rect(self.grid_surface, self.COLORS[cell_value], rect)

    def _draw_info(self, state: Dict) -> None:
        """Draw game information panel"""
        info_x = self.board_width * self.cell_size + 10
        info_y = 10
        line_height = 30
        lines = 1

        # Draw score
        score_text = self.font.render(f"Score: {state['score']}", True, (255, 255, 255))
        self.window.blit(score_text, (info_x, info_y))

        # Draw lines cleared
        lines_text = self.info_font.render(f"Lines: {state['lines_cleared']}", True, (255, 255, 255))
        self.window.blit(lines_text, (info_x, info_y + lines * line_height))
        lines += 1

        # Draw pieces placed
        pieces_text = self.info_font.render(f"Pieces: {state['pieces_placed']}", True, (255, 255, 255))
        self.window.blit(pieces_text, (info_x, info_y + lines * line_height))
        lines += 1

        # Draw bumpiness
        pieces_text = self.info_font.render(f"Bumpiness: {state['metrics']['bumpiness']}", True, (255, 255, 255))
        self.window.blit(pieces_text, (info_x, info_y + lines * line_height))
        lines += 1

        # Draw holes
        pieces_text = self.info_font.render(f"Holes: {state['metrics']['holes']}", True, (255, 255, 255))
        self.window.blit(pieces_text, (info_x, info_y + lines * line_height))
        lines += 1

        # Draw max height
        pieces_text = self.info_font.render(f"Max height: {state['metrics']['max_height']}", True, (255, 255, 255))
        self.window.blit(pieces_text, (info_x, info_y + lines * line_height))
        lines += 1

        # Draw min height
        pieces_text = self.info_font.render(f"Min height: {state['metrics']['min_height']}", True, (255, 255, 255))
        self.window.blit(pieces_text, (info_x, info_y + lines * line_height))
        lines += 1

    def close(self) -> None:
        """Clean up PyGame resources"""
        import pygame

        pygame.quit()


class TetrisASCIIRenderer(TetrisRenderer):
    """ASCII-based console renderer for Tetris"""

    def __init__(self, cell_width: int = 2):
        self.cell_width = cell_width

    def render(self, state: Dict) -> None:
        """Render the current game state in ASCII"""
        self._clear_screen()
        grid = state["grid"]

        # Print top border
        print("+" + "-" * (state["width"] * self.cell_width) + "+")

        # Print grid with colored blocks
        for row in grid:
            print("|", end="")
            for cell in row:
                print(self._cell_str(cell), end="")
            print("|")

        # Print bottom border
        print("+" + "-" * (state["width"] * self.cell_width) + "+")

        # Print game information
        print(f"\nScore: {state['score']}")
        print(f"Lines Cleared: {state['lines_cleared']}")
        print(f"Pieces Placed: {state['pieces_placed']}")
        print(f"Bumpiness:     {state['metrics']['bumpiness']}")
        print(f"Holes:         {state['metrics']['holes']}")
        print(f"Min height:    {state['metrics']['min_height']}")
        print(f"Max height:    {state['metrics']['max_height']}")

    def _cell_str(self, cell: int, block: str = "█", empty: str = ".") -> str:
        """Convert a cell value to a colored string"""
        if cell == 0:
            return empty * self.cell_width

        color = self.ASCII_COLORS[cell]
        return f"{color}{block * self.cell_width}\033[0m"

    def _clear_screen(self) -> None:
        """Clear the console screen"""
        print("\033[2J\033[H", end="")

    def close(self) -> None:
        """Clean up resources (not needed for ASCII renderer)"""
        pass


import numpy as np


class TetrisRGBArrayRenderer(TetrisRenderer):
    """RGB array renderer for Tetris with grid lines."""

    GRID_COLOR = (50, 50, 50)  # Gray color for grid lines

    def __init__(self, cell_size: int = 30):
        self.cell_size = cell_size
        self.board_width = None
        self.board_height = None

    def initialize(self, board_width: int, board_height: int) -> None:
        """Initialize board dimensions."""
        self.board_width = board_width
        self.board_height = board_height

    def render(self, state: Dict) -> np.ndarray:
        """Render the current game state as an RGB array with grid lines."""
        if self.board_width is None or self.board_height is None:
            self.initialize(state["width"], state["height"])

        # Initialize the RGB array with black background
        image_height = self.board_height * self.cell_size
        image_width = self.board_width * self.cell_size
        rgb_array = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        # Draw cells
        grid = state["grid"]
        for y in range(self.board_height):
            for x in range(self.board_width):
                cell_value = grid[y, x]
                color = self.COLORS.get(cell_value, (0, 0, 0))  # Default to black for empty cells
                self._draw_cell(rgb_array, x, y, color)

        # Draw grid lines
        self._draw_grid_lines(rgb_array)

        return rgb_array

    def _draw_cell(self, rgb_array: np.ndarray, x: int, y: int, color: tuple) -> None:
        """Draw a single cell in the RGB array."""
        y_start = y * self.cell_size
        y_end = y_start + self.cell_size
        x_start = x * self.cell_size
        x_end = x_start + self.cell_size

        rgb_array[y_start:y_end, x_start:x_end] = color

    def _draw_grid_lines(self, rgb_array: np.ndarray) -> None:
        """Draw grid lines on the RGB array."""
        # Draw vertical lines
        for x in range(1, self.board_width):
            x_pos = x * self.cell_size
            rgb_array[:, x_pos : x_pos + 1] = self.GRID_COLOR  # Draw line 1 pixel wide

        # Draw horizontal lines
        for y in range(1, self.board_height):
            y_pos = y * self.cell_size
            rgb_array[y_pos : y_pos + 1, :] = self.GRID_COLOR  # Draw line 1 pixel wide

    def close(self) -> None:
        """Clean up resources (not needed for RGB array renderer)."""
        pass


def play_tetris_pygame(game) -> None:
    """Play Tetris with PyGame interface"""
    import pygame

    pygame.init()
    renderer = TetrisPyGameRenderer()
    terminated = False
    clock = pygame.time.Clock()

    try:
        observation = game.reset()
        while not terminated:
            # Handle PyGame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
                ):
                    terminated = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = Actions.LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = Actions.RIGHT
                    elif event.key == pygame.K_UP:
                        action = Actions.ROTATE
                    elif event.key == pygame.K_SPACE:
                        action = Actions.HARD_DROP
                    else:
                        continue
                    if action is not None:
                        observation, terminated, _, _ = game.step(action)

            # TODO: Natural drop based on ticks
            # observation, terminated, _, _ = game.step(None)

            # Render and control game speed
            renderer.render(observation)
            clock.tick(60)

    finally:
        renderer.close()


def play_tetris_ascii(game) -> None:
    """Play Tetris with ASCII interface"""
    renderer = TetrisASCIIRenderer()
    terminated = False

    try:
        observation = game.reset()
        while not terminated:
            renderer.render(observation)
            try:
                kbd = str(input("Action: "))
            except KeyboardInterrupt:
                break
            if kbd.lower() == "q":
                terminated = True
                break
            elif kbd == "1":
                action = Actions.LEFT
            elif kbd == "2":
                action = Actions.HARD_DROP
            elif kbd == "3":
                action = Actions.RIGHT
            elif kbd == "5":
                action = Actions.ROTATE
            else:
                continue

            observation, terminated, _, _ = game.step(action)
    finally:
        renderer.render(observation)
        renderer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Game of Tetris")
    parser.add_argument("--ticks", type=int, default=1, help="Ticks per drop")
    game_mode = parser.add_mutually_exclusive_group()
    game_mode.add_argument("--pygame", action="store_true", help="Use pygame interface")
    game_mode.add_argument("--ascii", action="store_true", help="Use ascii interface")
    args = parser.parse_args()

    game = TetrisGame(ticks_per_drop=args.ticks)
    if args.pygame:
        play_tetris_pygame(game)  # Use PyGame renderer
    elif args.ascii:
        play_tetris_ascii(game)  # Or use ASCII renderer:
