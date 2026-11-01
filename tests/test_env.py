"""Tests for Milestone 4: placements, action indices, mask, game over."""

import numpy as np
import pytest
from gymnasium import spaces

from tetris.board import empty_board, place
from tetris.env import (
    action_mask,
    afterstates,
    decode,
    encode,
    is_game_over,
    legal_actions,
    TetrisEnv,
    PIECE_NAMES,
    reward
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


def piece_sequence(env: TetrisEnv, seed: int, n: int) -> list[str]:
    """First n pieces of a game: current, next, then further draws."""
    env.reset(seed=seed)
    assert env.current_piece is not None and env.next_piece is not None
    seq = [env.current_piece, env.next_piece]
    seq += [env._draw_piece() for _ in range(n - 2)]
    return seq


def random_legal_action(info: dict, rng: np.random.Generator) -> int:
    return int(rng.choice(np.flatnonzero(info["action_mask"])))


# --- constructor ---------------------------------------------------------------

def test_invalid_randomizer_raises():
    with pytest.raises(ValueError):
        TetrisEnv(randomizer="bagg")


def test_spaces():
    env = TetrisEnv()
    assert isinstance(env.action_space, spaces.Discrete)
    assert isinstance(env.observation_space, spaces.Dict)
    board_space = env.observation_space["board"]
    piece_space = env.observation_space["piece"]
    assert isinstance(board_space, spaces.Box)
    assert isinstance(piece_space, spaces.Discrete)
    assert env.action_space.n == 40
    assert board_space.shape == (20, 10)
    assert piece_space.n == 7


# --- reset -----------------------------------------------------------------------

def test_reset_observation():
    env = TetrisEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert not obs["board"].any()
    assert 0 <= obs["piece"] < 7
    assert PIECE_NAMES[obs["piece"]] == env.current_piece


def test_reset_info():
    env = TetrisEnv()
    _, info = env.reset(seed=0)
    assert env.current_piece is not None
    expected = action_mask(env.board, PIECES[env.current_piece])
    assert np.array_equal(info["action_mask"], expected)
    assert info["lines"] == 0
    assert info["total_lines"] == 0
    assert info["pieces_placed"] == 0
    assert info["next_piece"] == env.next_piece


def test_reset_clears_previous_game():
    env = TetrisEnv()
    _, info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        _, _, _, _, info = env.step(random_legal_action(info, rng))
    obs, info = env.reset(seed=1)
    assert not obs["board"].any()
    assert info["pieces_placed"] == 0


# --- randomizer --------------------------------------------------------------------

def test_same_seed_same_pieces():
    assert piece_sequence(TetrisEnv(), 42, 50) == piece_sequence(TetrisEnv(), 42, 50)


def test_different_seeds_differ():
    assert piece_sequence(TetrisEnv(), 1, 50) != piece_sequence(TetrisEnv(), 2, 50)


@pytest.mark.parametrize("seed", range(5))
def test_bag_property(seed):
    seq = piece_sequence(TetrisEnv(), seed, 21)
    for start in (0, 7, 14):
        assert sorted(seq[start:start + 7]) == sorted(PIECE_NAMES)


def test_uniform_randomizer_valid_names():
    seq = piece_sequence(TetrisEnv(randomizer="uniform"), 0, 700)
    assert set(seq) == set(PIECE_NAMES)
    assert all(type(p) is str for p in seq)


def test_bag_returns_plain_str():
    seq = piece_sequence(TetrisEnv(), 0, 14)
    assert all(type(p) is str for p in seq)


# --- step: legal actions -------------------------------------------------------------

def test_legal_step_outputs():
    env = TetrisEnv()
    _, info = env.reset(seed=0)
    old_next = env.next_piece
    action = random_legal_action(info, np.random.default_rng(0))
    result = env.step(action)
    assert len(result) == 5
    obs, r, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert obs["board"].sum() == 4              # one piece, no line on an empty board
    assert info["pieces_placed"] == 1
    assert env.current_piece == old_next        # the queue advanced
    assert info["next_piece"] == env.next_piece
    assert not terminated and not truncated
    assert r == reward(0, False)


def test_step_clears_line():
    env = TetrisEnv()
    env.reset(seed=0)
    env.board = empty_board()
    env.board[19, 4:] = 1                       # bottom row full except columns 0..3
    env.current_piece = "I"                     # rotation 0 = horizontal I
    _, r, _, _, info = env.step(encode(0, 0, 10))
    assert info["lines"] == 1
    assert info["total_lines"] == 1
    assert not env.board.any()
    assert r == reward(1, False)


# --- step: illegal actions --------------------------------------------------------------

def test_missing_rotation_raises():
    env = TetrisEnv()
    env.reset(seed=0)
    env.current_piece = "O"                     # only rotation 0 exists
    with pytest.raises(ValueError):
        env.step(encode(1, 0, 10))


def test_out_of_bounds_column_raises():
    env = TetrisEnv()
    env.reset(seed=0)
    env.current_piece = "I"
    with pytest.raises(ValueError):
        env.step(encode(0, 7, 10))              # horizontal I would reach column 10


def test_illegal_step_leaves_state_unchanged():
    env = TetrisEnv()
    env.reset(seed=0)
    env.current_piece = "O"
    board_before = env.board.copy()
    with pytest.raises(ValueError):
        env.step(encode(3, 0, 10))
    assert np.array_equal(env.board, board_before)
    assert env.pieces_placed == 0
    assert env.current_piece == "O"


# --- full games -----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(3))
def test_random_game_terminates(seed):
    env = TetrisEnv()
    _, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        _, _, terminated, truncated, info = env.step(random_legal_action(info, rng))
        if terminated or truncated:
            break
    assert terminated
    assert not truncated
    assert not info["action_mask"].any()        # no legal action left


def test_truncation():
    env = TetrisEnv(max_pieces=5)
    _, info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for i in range(5):
        _, _, terminated, truncated, info = env.step(random_legal_action(info, rng))
        assert not terminated
        assert truncated == (i == 4)


def test_same_seed_same_game():
    def play(seed):
        env = TetrisEnv()
        obs, info = env.reset(seed=seed)
        boards = [obs["board"]]
        for _ in range(10):
            action = int(np.flatnonzero(info["action_mask"])[0])
            obs, _, terminated, _, info = env.step(action)
            boards.append(obs["board"])
            if terminated:
                break
        return boards

    a, b = play(7), play(7)
    assert len(a) == len(b)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


# --- observation is a copy -------------------------------------------------------------------

def test_observation_is_a_copy():
    env = TetrisEnv()
    obs, _ = env.reset(seed=0)
    obs["board"][19, 0] = 1
    assert env.board[19, 0] == 0


# --- reward -------------------------------------------------------------------------------------

@pytest.mark.parametrize("lines, expected", [(0, 1), (1, 11), (2, 41), (3, 91), (4, 161)])
def test_reward_values(lines, expected):
    assert reward(lines, False) == expected


def test_reward_game_over_penalty():
    assert reward(0, True) == reward(0, False) - 5
