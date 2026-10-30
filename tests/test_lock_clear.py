"""Tests for Milestone 3: lock, clear_lines, place."""

import numpy as np

from tetris.board import clear_lines, empty_board, lock, place

O = ((0, 0), (0, 1), (1, 0), (1, 1))
T = ((0, 0), (0, 1), (0, 2), (1, 1))
I_H = ((0, 0), (0, 1), (0, 2), (0, 3))


# --- lock --------------------------------------------------------------------

def test_lock_writes_exactly_the_four_cells():
    new = lock(empty_board(), T, 5, 2)
    assert new.sum() == 4
    for r, c in [(5, 2), (5, 3), (5, 4), (6, 3)]:
        assert new[r, c] == 1


def test_lock_does_not_modify_input():
    board = empty_board()
    lock(board, T, 5, 2)
    assert not board.any()


def test_lock_preserves_shape_and_dtype():
    new = lock(empty_board(), O, 0, 0)
    assert new.shape == (20, 10)
    assert new.dtype == np.int8


# --- clear_lines ---------------------------------------------------------------

def test_clear_no_full_rows():
    board = empty_board()
    board[19, 0:9] = 1                      # almost full, not full
    new, k = clear_lines(board)
    assert k == 0
    assert np.array_equal(new, board)


def test_clear_returns_python_int():
    _, k = clear_lines(empty_board())
    assert type(k) is int


def test_clear_one_row_marker_falls():
    board = empty_board()
    board[19, :] = 1
    board[18, 3] = 1
    new, k = clear_lines(board)
    assert k == 1
    assert new[19, 3] == 1
    assert new.sum() == 1


def test_clear_two_non_adjacent_rows():
    board = empty_board()
    board[17, :] = 1
    board[19, :] = 1
    board[16, 0] = 1
    board[18, 5] = 1
    new, k = clear_lines(board)
    assert k == 2
    assert new[18, 0] == 1
    assert new[19, 5] == 1
    assert new.sum() == 2


def test_clear_four_rows_tetris():
    board = empty_board()
    board[16:20, :] = 1
    board[15, 2] = 1
    new, k = clear_lines(board)
    assert k == 4
    assert new[19, 2] == 1
    assert new.sum() == 1


def test_clear_new_top_rows_are_empty():
    board = empty_board()
    board[18:20, :] = 1
    board[0, :9] = 1                        # top row almost full
    new, k = clear_lines(board)
    assert k == 2
    assert not new[0:2].any()               # the two added rows
    assert new[2, :9].all()                 # old top row moved down by 2


def test_clear_all_rows_full():
    board = np.ones((6, 4), dtype=np.int8)
    new, k = clear_lines(board)
    assert k == 6
    assert new.shape == (6, 4)
    assert not new.any()


def test_clear_preserves_shape_and_dtype():
    board = empty_board()
    board[19, :] = 1
    new, _ = clear_lines(board)
    assert new.shape == (20, 10)
    assert new.dtype == np.int8


def test_clear_does_not_modify_input():
    board = empty_board()
    board[19, :] = 1
    before = board.copy()
    clear_lines(board)
    assert np.array_equal(board, before)


# --- place -----------------------------------------------------------------------

def test_place_completes_a_line():
    board = empty_board()
    board[19, 4:] = 1                       # bottom row full except columns 0..3
    result = place(board, I_H, 0)
    assert result is not None
    new, k = result
    assert k == 1
    assert not new.any()


def test_place_without_clearing():
    result = place(empty_board(), O, 0)
    assert result is not None
    new, k = result
    assert k == 0
    assert new.sum() == 4
    assert new[18:20, 0:2].all()


def test_place_illegal_column_returns_none():
    assert place(empty_board(), I_H, 7) is None


def test_place_blocked_spawn_returns_none():
    board = empty_board()
    board[0, 0] = 1
    assert place(board, O, 0) is None


def test_place_does_not_modify_input():
    board = empty_board()
    board[19, 4:] = 1
    before = board.copy()
    place(board, I_H, 0)
    assert np.array_equal(board, before)
