"""Tests for Milestone 4: placements, action indices, mask, game over."""

import numpy as np
import pytest

from tetris.board import empty_board, place
from tetris.env import (
    action_mask,
    afterstates,
    decode,
    encode,
    is_game_over,
    legal_actions,
)
from tetris.pieces import PIECES

EMPTY_COUNTS = {"I": 17, "O": 9, "S": 17, "Z": 17, "T": 34, "L": 34, "J": 34}
W = 10


# --- legal_actions ---------------------------------------------------------------

@pytest.mark.parametrize("name", EMPTY_COUNTS)
def test_legal_count_empty_board(name):
    assert len(legal_actions(empty_board(), PIECES[name])) == EMPTY_COUNTS[name]


@pytest.mark.parametrize("name", EMPTY_COUNTS)
def test_legal_actions_in_range(name):
    piece = PIECES[name]
    for k, x in legal_actions(empty_board(), piece):
        assert 0 <= k < len(piece.rotations)
        assert 0 <= x < W


@pytest.mark.parametrize("name", EMPTY_COUNTS)
def test_legal_actions_sorted(name):
    actions = legal_actions(empty_board(), PIECES[name])
    assert actions == sorted(actions)


def test_legal_actions_respect_board():
    # Top-left 2x5 block filled: an O can only go in columns 5..8.
    board = empty_board()
    board[0:2, 0:5] = 1
    assert legal_actions(board, PIECES["O"]) == [(0, 5), (0, 6), (0, 7), (0, 8)]


def test_legal_count_small_board():
    # Width 4: O (width 2) has 4 - 2 + 1 = 3 columns.
    assert len(legal_actions(empty_board(6, 4), PIECES["O"])) == 3


# --- afterstates -----------------------------------------------------------------

@pytest.mark.parametrize("name", EMPTY_COUNTS)
def test_afterstates_keys_match_legal_actions(name):
    board = empty_board()
    board[19, 2:7] = 1
    piece = PIECES[name]
    assert list(afterstates(board, piece)) == legal_actions(board, piece)


@pytest.mark.parametrize("name", EMPTY_COUNTS)
def test_afterstates_match_place(name):
    board = empty_board()
    board[19, 1:] = 1                       # bottom row full except column 0
    piece = PIECES[name]
    for (k, x), (new, lines) in afterstates(board, piece).items():
        expected = place(board, piece.rotations[k], x)
        assert expected is not None
        assert np.array_equal(new, expected[0])
        assert lines == expected[1]


def test_afterstates_do_not_modify_board():
    board = empty_board()
    board[19, 1:] = 1
    before = board.copy()
    afterstates(board, PIECES["T"])
    assert np.array_equal(board, before)


# --- encode / decode -------------------------------------------------------------

def test_decode_encode_identity():
    for k in range(4):
        for x in range(W):
            assert decode(encode(k, x, W), W) == (k, x)


def test_encode_decode_identity():
    for a in range(4 * W):
        assert encode(*decode(a, W), W) == a


def test_encode_range():
    assert encode(0, 0, W) == 0
    assert encode(3, W - 1, W) == 4 * W - 1
    assert encode(2, 3, W) == 23


# --- action_mask -----------------------------------------------------------------

@pytest.mark.parametrize("name", EMPTY_COUNTS)
def test_mask_shape_and_dtype(name):
    mask = action_mask(empty_board(), PIECES[name])
    assert mask.shape == (4 * W,)
    assert mask.dtype == bool


@pytest.mark.parametrize("name", EMPTY_COUNTS)
def test_mask_matches_legal_actions(name):
    board = empty_board()
    board[0:2, 0:5] = 1
    piece = PIECES[name]
    mask = action_mask(board, piece)
    legal = {encode(k, x, W) for k, x in legal_actions(board, piece)}
    assert mask.sum() == len(legal)
    for a in range(4 * W):
        assert mask[a] == (a in legal)


@pytest.mark.parametrize("name", EMPTY_COUNTS)
def test_mask_missing_rotations_are_false(name):
    piece = PIECES[name]
    mask = action_mask(empty_board(), piece)
    n = len(piece.rotations)
    assert not mask[n * W:].any()           # indices with k >= n


# --- game over -------------------------------------------------------------------

@pytest.mark.parametrize("name", EMPTY_COUNTS)
def test_not_game_over_on_empty_board(name):
    assert not is_game_over(empty_board(), PIECES[name])


@pytest.mark.parametrize("name", EMPTY_COUNTS)
def test_game_over_when_top_rows_full(name):
    board = empty_board()
    board[0:2, :] = 1
    piece = PIECES[name]
    assert legal_actions(board, piece) == []
    assert is_game_over(board, piece)
    assert not action_mask(board, piece).any()
