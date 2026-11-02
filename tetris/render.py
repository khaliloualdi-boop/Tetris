"""Pygame window renderer for TetrisEnv's "human" render mode.

The renderer only *draws*: it never reads or changes game state by itself.
The env passes it everything to show on each frame.

pygame is imported by this module only, and the env imports this module
lazily (on the first human-mode frame), so training never needs pygame.
"""

from collections.abc import Iterable

import numpy as np
import pygame

from tetris.pieces import PIECES

# --- layout (pixels) --------------------------------------------------------------
CELL = 32          # size of one board cell
GAP = 2            # space between cells
RADIUS = 6         # corner radius of a block
MARGIN = 28        # outer margin around everything
PANEL_W = 190      # width of the side panel
PREVIEW_CELL = 20  # cell size in the piece previews

# --- colours ------------------------------------------------------------------------
BG = (17, 19, 27)
WELL = (26, 29, 40)
GRID = (34, 38, 52)
BLOCK = (96, 108, 140)
BLOCK_TOP = (126, 138, 172)
LAST = (255, 196, 84)
LAST_TOP = (255, 222, 150)
GHOST = (220, 225, 240)
TEXT = (232, 235, 245)
MUTED = (128, 135, 158)
CARD = (26, 29, 40)

PIECE_COLORS = {
    "I": (64, 208, 224),
    "O": (244, 204, 64),
    "T": (176, 104, 232),
    "S": (96, 208, 112),
    "Z": (236, 88, 96),
    "J": (80, 128, 240),
    "L": (244, 148, 64),
}


def _lighter(color: tuple[int, int, int], amount: int = 40) -> tuple[int, int, int]:
    r, g, b = color
    return min(r + amount, 255), min(g + amount, 255), min(b + amount, 255)


class PygameRenderer:
    """Owns the window, the clock and the fonts. Create it once, call draw() per frame."""

    def __init__(self, height: int, width: int, fps: int) -> None:
        self.height = height
        self.width = width
        self.fps = fps

        pygame.init()
        pygame.display.set_caption("Tetris AI")
        self.well_rect = pygame.Rect(MARGIN, MARGIN, width * CELL, height * CELL)
        size = (MARGIN * 3 + width * CELL + PANEL_W, MARGIN * 2 + height * CELL)
        self.screen = pygame.display.set_mode(size)
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.Font(None, 44)
        self.font_label = pygame.font.Font(None, 22)
        self.font_value = pygame.font.Font(None, 40)

    # --- public ---------------------------------------------------------------------

    def draw(
        self,
        board: np.ndarray,
        last_cells: Iterable[tuple[int, int]] = (),
        ghost_cells: Iterable[tuple[int, int]] = (),
        current_piece: str | None = None,
        next_piece: str | None = None,
        pieces_placed: int = 0,
        total_lines: int = 0,
    ) -> None:
        """Draw one frame, show it, and wait to keep the frame rate."""
        # Keep the window responsive without consuming events: a script
        # (e.g. play.py) may still want to read key presses itself.
        pygame.event.pump()

        last = set(last_cells)
        self.screen.fill(BG)
        self._draw_well(board, last)
        self._draw_ghost(ghost_cells)
        self._draw_panel(current_piece, next_piece, pieces_placed, total_lines)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        pygame.display.quit()
        pygame.quit()

    # --- board ------------------------------------------------------------------------

    def _cell_rect(self, r: int, c: int) -> pygame.Rect:
        return pygame.Rect(
            self.well_rect.x + c * CELL + GAP // 2,
            self.well_rect.y + r * CELL + GAP // 2,
            CELL - GAP,
            CELL - GAP,
        )

    def _draw_block(self, rect: pygame.Rect, color, top_color, radius: int = RADIUS) -> None:
        pygame.draw.rect(self.screen, color, rect, border_radius=radius)
        # thin lighter strip on top: a subtle bevel
        top = pygame.Rect(rect.x + 3, rect.y + 2, rect.w - 6, max(2, rect.h // 8))
        pygame.draw.rect(self.screen, top_color, top, border_radius=2)

    def _draw_well(self, board: np.ndarray, last: set[tuple[int, int]]) -> None:
        frame = self.well_rect.inflate(12, 12)
        pygame.draw.rect(self.screen, WELL, frame, border_radius=12)

        for r in range(self.height):
            for c in range(self.width):
                rect = self._cell_rect(r, c)
                if (r, c) in last:
                    self._draw_block(rect, LAST, LAST_TOP)
                elif board[r, c]:
                    self._draw_block(rect, BLOCK, BLOCK_TOP)
                else:
                    pygame.draw.rect(self.screen, GRID, rect, width=1, border_radius=RADIUS)

    def _draw_ghost(self, ghost_cells: Iterable[tuple[int, int]]) -> None:
        for r, c in ghost_cells:
            rect = self._cell_rect(r, c).inflate(-4, -4)
            pygame.draw.rect(self.screen, GHOST, rect, width=2, border_radius=RADIUS)

    # --- side panel --------------------------------------------------------------------

    def _draw_panel(self, current, next_, pieces_placed, total_lines) -> None:
        x = self.well_rect.right + MARGIN + 6
        y = self.well_rect.y

        title = self.font_title.render("TETRIS", True, TEXT)
        self.screen.blit(title, (x, y))
        y += 58

        y = self._stat(x, y, "LINES", str(total_lines))
        y = self._stat(x, y, "PIECES", str(pieces_placed))
        y += 10
        y = self._preview(x, y, "CURRENT", current)
        self._preview(x, y, "NEXT", next_)

    def _stat(self, x: int, y: int, label: str, value: str) -> int:
        self.screen.blit(self.font_label.render(label, True, MUTED), (x, y))
        self.screen.blit(self.font_value.render(value, True, TEXT), (x, y + 18))
        return y + 64

    def _preview(self, x: int, y: int, label: str, name: str | None) -> int:
        self.screen.blit(self.font_label.render(label, True, MUTED), (x, y))
        card = pygame.Rect(x, y + 22, PANEL_W - 10, 4 * PREVIEW_CELL + 24)
        pygame.draw.rect(self.screen, CARD, card, border_radius=10)

        if name is not None:
            shape = PIECES[name].rotations[0]
            h = max(r for r, _ in shape) + 1
            w = max(c for _, c in shape) + 1
            ox = card.centerx - w * PREVIEW_CELL // 2
            oy = card.centery - h * PREVIEW_CELL // 2
            color = PIECE_COLORS.get(name, BLOCK)
            for r, c in shape:
                rect = pygame.Rect(
                    ox + c * PREVIEW_CELL + 1, oy + r * PREVIEW_CELL + 1,
                    PREVIEW_CELL - 2, PREVIEW_CELL - 2,
                )
                self._draw_block(rect, color, _lighter(color), radius=3)
        return card.bottom + 18
