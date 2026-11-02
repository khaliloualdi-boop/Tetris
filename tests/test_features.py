"""Tests for Milestone 7, task 1: board features."""

import numpy as np
import pytest

from tetris.board import empty_board
from tetris.features import aggregate_height, bumpiness, column_heights, features, holes

H, W = 20, 10


def small_board(rows: list[str]) -> np.ndarray:
    """Build a board from strings: '#' filled, '.' empty, row 0 first."""
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in rows], dtype=np.int8)


# The 4 x 3 example board used in the explanations:
#   row 0:  . . .
#   row 1:  # # .
#   row 2:  . # .
#   row 3:  . # #
EXAMPLE = small_board(["...", "##.", ".#.", ".##"])


# --- column_heights --------------------------------------------------------------

def test_heights_empty_board():
    assert list(column_heights(empty_board())) == [0] * W


def test_heights_single_bottom_cell():
    board = empty_board()
    board[19, 3] = 1
    heights = column_heights(board)
    assert heights[3] == 1
    assert heights.sum() == 1


@pytest.mark.parametrize("c", [0, 4, 9])
def test_heights_top_cell_gives_full_height(c):
    # argmax returns 0 for this column, like for an empty one: must give H, not 0.
    board = empty_board()
    board[0, c] = 1
    assert column_heights(board)[c] == H


def test_heights_use_the_top_cell_not_the_count():
    board = empty_board()
    board[10, 0] = 1                        # overhang, empty cells below
    board[19, 0] = 1
    assert column_heights(board)[0] == 10


def test_heights_example_board():
    assert list(column_heights(EXAMPLE)) == [3, 3, 1]


def test_heights_custom_size():
    board = empty_board(6, 4)
    board[3, 2] = 1
    assert list(column_heights(board)) == [0, 0, 3, 0]


# --- aggregate_height ------------------------------------------------------------

def test_aggregate_height_empty():
    assert aggregate_height(empty_board()) == 0


def test_aggregate_height_example():
    assert aggregate_height(EXAMPLE) == 3 + 3 + 1


def test_aggregate_height_full_bottom_row():
    board = empty_board()
    board[19, :] = 1
    assert aggregate_height(board) == W


# --- holes -------------------------------------------------------------------------

def test_holes_empty():
    assert holes(empty_board()) == 0


def test_holes_single():
    board = empty_board()
    board[18, 0] = 1                        # (19, 0) below it stays empty
    assert holes(board) == 1


def test_holes_full_column():
    board = empty_board()
    board[:, 0] = 1
    assert holes(board) == 0


def test_holes_example_board():
    # Column 0: top at row 1, rows 2 and 3 empty below it -> 2 holes.
    assert holes(EXAMPLE) == 2


def test_holes_several_columns():
    board = empty_board()
    board[15, 0] = 1                        # 4 empty cells below (rows 16..19)
    board[17, 5] = 1
    board[19, 5] = 1                        # 1 empty cell in between (row 18)
    board[19, 9] = 1                        # no hole
    assert holes(board) == 4 + 1


def test_holes_empty_cells_above_the_stack_dont_count():
    board = empty_board()
    board[19, :] = 1                        # flat, no overhang
    assert holes(board) == 0


# --- bumpiness ------------------------------------------------------------------

def test_bumpiness_empty():
    assert bumpiness(empty_board()) == 0


def test_bumpiness_flat():
    board = empty_board()
    board[17:, :] = 1
    assert bumpiness(board) == 0


def test_bumpiness_middle_column():
    board = empty_board()
    board[16:, 5] = 1                       # height 4: |0 - 4| + |4 - 0|
    assert bumpiness(board) == 8


def test_bumpiness_edge_column():
    board = empty_board()
    board[16:, 0] = 1                       # only one neighbour: |4 - 0|
    assert bumpiness(board) == 4


def test_bumpiness_example_board():
    # heights (3, 3, 1): |3 - 3| + |3 - 1|
    assert bumpiness(EXAMPLE) == 2


# --- return types ----------------------------------------------------------------

@pytest.mark.parametrize("f", [aggregate_height, holes, bumpiness])
def test_features_return_python_int(f):
    assert isinstance(f(EXAMPLE), int)


# --- features vector -------------------------------------------------------------

def test_features_vector():
    phi = features(EXAMPLE, 1)
    assert phi.shape == (4,)
    assert phi.dtype == float
    # order: aggregate height, holes, bumpiness, lines
    assert list(phi) == [7.0, 2.0, 2.0, 1.0]


def test_features_lines_passthrough():
    assert features(empty_board(), 3)[3] == 3


def test_features_empty_board():
    assert list(features(empty_board(), 0)) == [0.0, 0.0, 0.0, 0.0]


# --- purity -----------------------------------------------------------------------

def test_features_do_not_modify_board():
    board = EXAMPLE.copy()
    column_heights(board)
    aggregate_height(board)
    holes(board)
    bumpiness(board)
    features(board, 0)
    assert np.array_equal(board, EXAMPLE)
