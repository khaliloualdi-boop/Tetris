"""Tests for Milestone 6: rendering (text mode only; windows aren't tested automatically)."""

import numpy as np
import pytest

from tetris.board import empty_board
from tetris.env import TetrisEnv, encode

H, W = 20, 10


def board_lines(env: TetrisEnv) -> list[str]:
    """The last H lines of the ansi render: the board, below the header."""
    text = env.render()
    assert isinstance(text, str)
    return text.splitlines()[-env.height:]


def count(lines: list[str], char: str) -> int:
    return sum(line.count(char) for line in lines)


def first_legal_action(info: dict) -> int:
    return int(np.flatnonzero(info["action_mask"])[0])


# --- render modes ------------------------------------------------------------------

def test_invalid_render_mode_raises():
    with pytest.raises(ValueError):
        TetrisEnv(render_mode="rgb")


def test_render_mode_is_stored():
    assert TetrisEnv(render_mode="ansi").render_mode == "ansi"
    assert TetrisEnv().render_mode is None


def test_render_none_returns_none():
    env = TetrisEnv()
    env.reset(seed=0)
    assert env.render() is None


# --- ansi output ---------------------------------------------------------------------

def test_ansi_after_reset():
    env = TetrisEnv(render_mode="ansi")
    env.reset(seed=0)
    lines = board_lines(env)
    assert len(lines) == H
    assert all(len(line) == W for line in lines)
    assert all(set(line) <= {".", "#", "@"} for line in lines)
    assert count(lines, "#") == 0
    assert count(lines, "@") == 0


def test_ansi_after_one_step():
    env = TetrisEnv(render_mode="ansi")
    _, info = env.reset(seed=0)
    env.step(first_legal_action(info))      # no line can be cleared on an empty board
    lines = board_lines(env)
    assert count(lines, "@") == 4
    assert count(lines, "#") == 0


def test_ansi_after_two_steps():
    env = TetrisEnv(render_mode="ansi")
    _, info = env.reset(seed=0)
    _, _, _, _, info = env.step(first_legal_action(info))
    env.step(first_legal_action(info))      # 8 cells can't fill a 10-wide row
    lines = board_lines(env)
    assert count(lines, "@") == 4
    assert count(lines, "#") == 4


@pytest.mark.parametrize("seed", range(3))
def test_symbols_match_board(seed):
    env = TetrisEnv(render_mode="ansi")
    _, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(30):
        action = int(rng.choice(np.flatnonzero(info["action_mask"])))
        _, _, terminated, truncated, info = env.step(action)
        lines = board_lines(env)
        assert count(lines, "#") + count(lines, "@") == env.board.sum()
        if terminated or truncated:
            break


def test_symbols_at_the_right_cells():
    env = TetrisEnv(render_mode="ansi")
    _, info = env.reset(seed=0)
    env.step(first_legal_action(info))
    lines = board_lines(env)
    for r in range(H):
        for c in range(W):
            if (r, c) in env.last_cells:
                assert lines[r][c] == "@"
            elif env.board[r, c]:
                assert lines[r][c] == "#"
            else:
                assert lines[r][c] == "."


# --- last placed piece -----------------------------------------------------------------

def test_last_cells_are_filled():
    env = TetrisEnv()
    _, info = env.reset(seed=0)
    env.step(first_legal_action(info))
    assert len(env.last_cells) == 4
    for r, c in env.last_cells:
        assert env.board[r, c] == 1


def test_no_highlight_after_line_clear():
    env = TetrisEnv(render_mode="ansi")
    env.reset(seed=0)
    env.board = empty_board()
    env.board[19, 4:] = 1                   # bottom row full except columns 0..3
    env.current_piece = "I"                 # rotation 0 = horizontal I
    env.step(encode(0, 0, W))
    assert len(env.last_cells) == 0
    assert count(board_lines(env), "@") == 0


def test_reset_clears_last_cells():
    env = TetrisEnv(render_mode="ansi")
    _, info = env.reset(seed=0)
    env.step(first_legal_action(info))
    env.reset(seed=1)
    assert len(env.last_cells) == 0
    assert count(board_lines(env), "@") == 0


# --- rendering never changes the game --------------------------------------------------

def test_render_does_not_modify_state():
    env = TetrisEnv(render_mode="ansi")
    _, info = env.reset(seed=0)
    env.step(first_legal_action(info))
    board, piece, placed = env.board.copy(), env.current_piece, env.pieces_placed
    env.render()
    env.render()
    assert np.array_equal(env.board, board)
    assert env.current_piece == piece
    assert env.pieces_placed == placed


# --- close -------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [None, "ansi"])
def test_close_twice_is_safe(mode):
    env = TetrisEnv(render_mode=mode)
    env.reset(seed=0)
    env.close()
    env.close()
