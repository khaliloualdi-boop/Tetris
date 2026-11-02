"""Tests for Milestone 7, task 2: HeuristicAgent and RandomAgent."""

import numpy as np
import pytest

from tetris.agents import HeuristicAgent, RandomAgent
from tetris.board import empty_board
from tetris.env import PIECE_NAMES, TetrisEnv, action_mask, encode, Observation
from tetris.pieces import PIECES

W = 10


def make_obs(board: np.ndarray, name: str) -> Observation:
    return {"board": board, "piece": PIECE_NAMES.index(name)}


def make_info(board: np.ndarray, name: str) -> dict:
    return {"action_mask": action_mask(board, PIECES[name])}


def play_game(agent, seed: int, max_pieces: int) -> dict:
    """Play one game; check every action is legal; return the last info."""
    env = TetrisEnv(max_pieces=max_pieces)
    obs, info = env.reset(seed=seed)
    while True:
        a = agent.act(obs, info)
        assert isinstance(a, int)
        assert info["action_mask"][a], f"illegal action {a}"
        obs, _, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            return info


# --- HeuristicAgent: weights ----------------------------------------------------

def test_default_weight_signs():
    # order: aggregate height, holes, bumpiness, lines
    w = HeuristicAgent().weights
    assert w.shape == (4,)
    assert w[0] < 0 and w[1] < 0 and w[2] < 0 and w[3] > 0


def test_weights_accept_a_list_and_become_floats():
    agent = HeuristicAgent([-1, -1, -1, 1])
    assert isinstance(agent.weights, np.ndarray)
    assert agent.weights.dtype == float


def test_weights_are_copied():
    w = np.array([-0.5, -0.4, -0.2, 0.8])
    agent = HeuristicAgent(w)
    w[0] = 100.0                            # modifying the caller's array...
    assert agent.weights[0] == -0.5         # ...doesn't change the agent


def test_default_weights_not_shared():
    a, b = HeuristicAgent(), HeuristicAgent()
    a.weights[0] = 100.0
    assert b.weights[0] != 100.0


# --- HeuristicAgent: decisions -----------------------------------------------------

def test_heuristic_clears_a_line_when_it_can():
    board = empty_board()
    board[19, 4:] = 1                       # bottom row full except columns 0..3
    agent = HeuristicAgent()
    a = agent.act(make_obs(board, "I"), make_info(board, "I"))
    assert a == encode(0, 0, W)             # horizontal I at x = 0


def test_heuristic_avoids_creating_a_hole():
    # Columns 0..8 have height 1, column 9 is empty. An O dropped at x = 8
    # would hang over column 9 and create a hole; x = 0 keeps the stack clean.
    board = empty_board()
    board[19, :9] = 1
    agent = HeuristicAgent()
    a = agent.act(make_obs(board, "O"), make_info(board, "O"))
    assert a != encode(0, 8, W)


def test_heuristic_works_without_info():
    board = empty_board()
    a = HeuristicAgent().act(make_obs(board, "T"))
    assert make_info(board, "T")["action_mask"][a]


def test_heuristic_does_not_modify_the_observation():
    board = empty_board()
    board[19, 3:] = 1
    obs = make_obs(board, "L")
    before = board.copy()
    HeuristicAgent().act(obs, make_info(board, "L"))
    assert np.array_equal(obs["board"], before)


@pytest.mark.parametrize("name", PIECE_NAMES)
def test_scale_invariance(name):
    # Multiplying all weights by c > 0 never changes the decision (design question 2).
    board = empty_board()
    board[17:, :] = 1
    board[17:, 4] = 0
    board[16, 0:3] = 1
    w = np.array([-0.51, -0.36, -0.18, 0.76])
    obs, info = make_obs(board, name), make_info(board, name)
    assert HeuristicAgent(w).act(obs, info) == HeuristicAgent(3.7 * w).act(obs, info)


# --- HeuristicAgent: full games ------------------------------------------------------

@pytest.mark.parametrize("seed", range(2))
def test_heuristic_always_legal_over_a_game(seed):
    play_game(HeuristicAgent(), seed, max_pieces=200)


def test_heuristic_is_deterministic():
    a = play_game(HeuristicAgent(), 3, max_pieces=200)
    b = play_game(HeuristicAgent(), 3, max_pieces=200)
    assert a["total_lines"] == b["total_lines"]
    assert a["pieces_placed"] == b["pieces_placed"]


# --- RandomAgent ----------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(3))
def test_random_always_legal_over_a_game(seed):
    play_game(RandomAgent(seed=seed), seed, max_pieces=1000)


def test_random_only_picks_masked_actions():
    board = empty_board()
    board[0:2, 0:5] = 1                     # only columns 5..8 legal for O
    info = make_info(board, "O")
    agent = RandomAgent(seed=0)
    picks = {agent.act(make_obs(board, "O"), info) for _ in range(200)}
    assert picks == {encode(0, x, W) for x in range(5, 9)}


def test_random_same_seed_same_choices():
    board = empty_board()
    obs, info = make_obs(board, "T"), make_info(board, "T")
    a, b = RandomAgent(seed=42), RandomAgent(seed=42)
    assert [a.act(obs, info) for _ in range(30)] == [b.act(obs, info) for _ in range(30)]


def test_random_different_seeds_differ():
    board = empty_board()
    obs, info = make_obs(board, "T"), make_info(board, "T")
    a, b = RandomAgent(seed=1), RandomAgent(seed=2)
    assert [a.act(obs, info) for _ in range(30)] != [b.act(obs, info) for _ in range(30)]


# --- the gap ----------------------------------------------------------------------------

def test_heuristic_beats_random_by_far():
    n, cap = 5, 300                         # 300 pieces allow at most 120 lines
    heur = [play_game(HeuristicAgent(), s, cap)["total_lines"] for s in range(n)]
    rand = [play_game(RandomAgent(seed=s), s, cap)["total_lines"] for s in range(n)]
    assert np.mean(heur) >= 50
    assert np.mean(rand) <= 5
