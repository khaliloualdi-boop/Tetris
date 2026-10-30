"""Tests for Milestone 2: board and collision."""

import numpy as np
import pytest

from tetris.board import drop_row, empty_board, fits

# Shapes written out directly, so these tests don't depend on pieces.py naming.
O = ((0, 0), (0, 1), (1, 0), (1, 1))
T = ((0, 0), (0, 1), (0, 2), (1, 1))
I_H = ((0, 0), (0, 1), (0, 2), (0, 3))  # horizontal I
I_V = ((0, 0), (1, 0), (2, 0), (3, 0))  # vertical I


# --- empty_board -----------------------------------------------------------

def test_empty_board_default():
    board = empty_board()
    assert board.shape == (20, 10)
    assert board.dtype == np.int8
    assert not board.any()


def test_empty_board_custom_size():
    assert empty_board(6, 4).shape == (6, 4)


def test_empty_board_returns_new_array():
    a, b = empty_board(), empty_board()
    a[0, 0] = 1
    assert b[0, 0] == 0


# --- fits: bounds ----------------------------------------------------------

def test_fits_top_left_empty_board():
    assert fits(empty_board(), T, 0, 0)


def test_fits_right_edge():
    board = empty_board()
    assert fits(board, I_H, 0, 6)       # columns 6..9
    assert not fits(board, I_H, 0, 7)   # column 10 is outside


def test_fits_left_edge():
    assert not fits(empty_board(), O, 0, -1)


def test_fits_bottom_edge():
    board = empty_board()
    assert fits(board, O, 18, 0)        # rows 18..19
    assert not fits(board, O, 19, 0)    # row 20 is outside


def test_fits_negative_row_trap():
    # board[-1, c] would silently read row 19; it must be rejected as out of bounds.
    assert not fits(empty_board(), O, -1, 0)


# --- fits: collisions ------------------------------------------------------

def test_fits_collision_with_filled_cell():
    board = empty_board()
    board[5, 5] = 1
    assert not fits(board, O, 4, 4)     # covers (5, 5)
    assert fits(board, O, 4, 6)         # one column to the right: free


# --- drop_row ----------------------------------------------------------------

@pytest.mark.parametrize("shape, expected", [(O, 18), (I_V, 16), (I_H, 19), (T, 18)])
def test_drop_empty_board(shape, expected):
    assert drop_row(empty_board(), shape, 0) == expected


def test_drop_onto_block():
    board = empty_board()
    board[19, 0] = 1
    assert drop_row(board, O, 0) == 17


def test_drop_stops_above_overhang():
    # A filled cell at (10, 0) with empty cells below: the piece must not
    # end up in the cavity, even though it would fit there.
    board = empty_board()
    board[10, 0] = 1
    assert drop_row(board, I_V, 0) == 6


def test_drop_lands_at_row_zero():
    board = empty_board()
    board[2:, :] = 1                    # only rows 0 and 1 empty
    assert drop_row(board, O, 0) == 0


def test_drop_blocked_spawn_returns_none():
    board = empty_board()
    board[0, 0] = 1
    assert drop_row(board, O, 0) is None


def test_drop_out_of_bounds_column_returns_none():
    assert drop_row(empty_board(), I_H, 7) is None


def test_drop_custom_board_size():
    board = empty_board(6, 4)
    assert drop_row(board, O, 0) == 4
    assert drop_row(board, I_H, 0) == 5
    assert drop_row(board, I_H, 1) is None


# --- purity ------------------------------------------------------------------

def test_fits_and_drop_do_not_modify_board():
    board = empty_board()
    board[10, 0] = 1
    board[19, 3:7] = 1
    before = board.copy()
    fits(board, T, 5, 2)
    drop_row(board, I_V, 0)
    drop_row(board, O, 4)
    assert np.array_equal(board, before)
