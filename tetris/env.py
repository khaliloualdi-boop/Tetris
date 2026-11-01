from typing import TypedDict, Any

import numpy as np
import gymnasium as gym

from gymnasium import spaces
from gymnasium.utils import env_checker

from tetris.board import drop_row, place, empty_board
from tetris.pieces import Piece, PIECES

PIECE_NAMES = ("I", "O", "T", "S", "J", "L", "Z")
PIECE_DICT = {name: i for i, name in enumerate(PIECE_NAMES)}

def id_to_name(piece_id: int) -> str:
    return PIECE_NAMES[piece_id]

def name_to_id(name: str) -> int:
    return PIECE_DICT[name]

def afterstates(board: np.ndarray, piece: Piece) -> dict[tuple[int, int], tuple[np.ndarray, int]]:
    W = board.shape[1]
    n = len(piece.rotations)
    result = {}
    for k in range(n):
        for x in range(W):
            output = place(board, piece.rotations[k], x)
            if output is not None:
                result[(k,x)] = output
    return result

def legal_actions(board: np.ndarray, piece: Piece) -> list[tuple[int, int]]:
    W = board.shape[1]
    n = len(piece.rotations)
    result = []
    for k in range(n):
        for x in range(W):
            if drop_row(board, piece.rotations[k], x) is not None:
                result.append((k, x))
    return result

def encode(k: int, x: int, width: int) -> int:
    return k*width + x

def decode(a: int , width: int) -> tuple[int, int]:
    x = a % width
    k = a//width
    return k, x

def action_mask(board: np.ndarray, piece: Piece) -> np.ndarray:
    W = board.shape[1]
    result = np.zeros(4*W, dtype=bool)
    for k, x in legal_actions(board, piece):
        a = encode(k, x, W)
        result[a] = True
    return result

def is_game_over(board: np.ndarray, piece: Piece) -> bool:
    return not legal_actions(board, piece)

def reward(lines: int, terminated: bool) -> int:
    if terminated :
        return 1 + 10*lines**2 - 5
    return 1+ 10*lines**2

class Observation(TypedDict):
    board: np.ndarray
    piece: int


class TetrisEnv(gym.Env):

    def __init__(self, height: int = 20, width: int = 10, max_pieces: int = 10000, randomizer: str = "bag"):
        super().__init__()
        self.height: int = height
        self.width: int = width
        self.max_pieces: int = max_pieces
        if randomizer not in {"bag", "uniform"}:
            raise ValueError(f"randomizer must be 'bag' or 'uniform', got {randomizer!r}")
        self.randomizer: str = randomizer
        self.board: np.ndarray = empty_board(height = self.height, width = self.width)
        self.current_piece: str | None = None
        self.next_piece: str | None = None
        self.bag: list[str] = []
        self.pieces_placed: int = 0
        self.total_lines: int = 0
        self.last_lines: int = 0
        self.action_space = spaces.Discrete(4*width)
        self.observation_space = spaces.Dict({"board": spaces.Box(low=0, high=1, shape=(height,width), dtype=np.int8), "piece":spaces.Discrete(7)})

    def _draw_piece(self) -> str:
        match self.randomizer:
            case "uniform":
                return str(self.np_random.choice(PIECE_NAMES))
            case "bag":
                if not self.bag:
                    new_bag = list(PIECE_NAMES)
                    self.np_random.shuffle(new_bag)
                    self.bag = new_bag
                return self.bag.pop()
            case _:
                raise ValueError(f"Unknown randomizer, {self.randomizer!r}")

    def reset(self, *, seed : int | None = None, options: dict | None = None) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self.board = empty_board(self.height, self.width)
        self.bag = []
        self.current_piece = self._draw_piece()
        self.next_piece = self._draw_piece()
        self.pieces_placed = 0
        self.total_lines = 0
        self.last_lines = 0
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[Observation, int, bool, bool, dict[str, Any]]:
        k, x = decode(action, self.width)
        assert self.current_piece is not None
        piece = PIECES[self.current_piece]
        if (k,x) not in legal_actions(self.board, piece):
            raise ValueError(f"illegal action {action} = (k={k}, x={x}) for piece {self.current_piece}")

        output = place(self.board, piece.rotations[k], x)
        assert output is not None
        self.board, self.last_lines = output
        self.total_lines += self.last_lines
        self.pieces_placed += 1
        self.current_piece = self.next_piece
        self.next_piece = self._draw_piece()
        assert self.current_piece is not None
        terminated = is_game_over(self.board, PIECES[self.current_piece])
        truncated = self.pieces_placed >= self.max_pieces and not terminated
        Reward = reward(self.last_lines, terminated)
        obs = self._observation()
        info = self._info()
        return (obs, Reward, terminated, truncated, info)

    def _observation(self) -> Observation:
        assert self.current_piece is not None
        return {"board": self.board.copy(), "piece": PIECE_DICT[self.current_piece]}

    def _info(self) -> dict[str, Any]:
        assert self.current_piece is not None
        mask = action_mask(self.board, PIECES[self.current_piece])
        return {"action_mask": mask, "lines": self.last_lines, "total_lines": self.total_lines, "pieces_placed": self.pieces_placed, "next_piece": self.next_piece}
