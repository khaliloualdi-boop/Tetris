from numpy.typing import ArrayLike
from typing import Any
from tetris.features import features
from tetris.pieces import PIECES
from tetris.env import Observation, PIECE_NAMES, afterstates, encode
from collections.abc import Sequence

import numpy as np


DEFAULT_WEIGHTS = (-0.51, -0.36, -0.18, 0.76)

class HeuristicAgent:
    def __init__(self, weights: ArrayLike = DEFAULT_WEIGHTS) -> None:
        self.weights = np.array(weights, dtype = float)

    def act(self, obs: Observation, info:dict[str, Any] | None = None) -> int:
        board = obs["board"]
        W = board.shape[1]
        piece = PIECES[PIECE_NAMES[obs["piece"]]]
        states = afterstates(board, piece)
        result: dict[int, float] = {}
        for (k, x), (pot_board, pot_lines) in states.items():
            a = encode(k,x,W)
            result[a] = sum(self.weights*features(pot_board, pot_lines))
        return max(result, key=lambda a: result[a])

class RandomAgent:
    def __init__(self, seed: int | None = None):
        self.generator = np.random.default_rng(seed)

    def act(self, Obs: Observation, info: dict[str, Any]) -> int:
        mask = info["action_mask"]
        valid = np.flatnonzero(mask)
        return int(self.generator.choice(valid))
