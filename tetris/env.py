from typing import TypedDict, Any

import numpy as np
import gymnasium as gym

from gymnasium import spaces
from gymnasium.utils import env_checker

from tetris.board import drop_row, place, empty_board, board_to_str
from tetris.pieces import Cell, Piece, PIECES

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
    metadata = {"render_modes":["ansi", "human"], "render_fps": 4}

    def __init__(self, height: int = 20, width: int = 10, max_pieces: int = 10000, randomizer: str = "bag", render_mode: str | None = None):
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
        if render_mode is not None and not render_mode in TetrisEnv.metadata["render_modes"]:
            raise ValueError(f"unexcpected render Mode, had {self.render_mode} expected 'ansi', 'human' or None")
        self.render_mode = render_mode
        self.last_cells : list[Cell] = []
        self._renderer = None

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
        self.last_cells = []
        if self.render_mode == "human":
            self._render_frame()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[Observation, int, bool, bool, dict[str, Any]]:
        k, x = decode(action, self.width)
        assert self.current_piece is not None
        piece = PIECES[self.current_piece]
        if (k,x) not in legal_actions(self.board, piece):
            raise ValueError(f"illegal action {action} = (k={k}, x={x}) for piece {self.current_piece}")
        y = drop_row(self.board, piece.rotations[k], x)
        assert y is not None
        self.last_cells = [(r + y, c + x) for r, c in piece.rotations[k]]
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
        if self.last_lines > 0:
            self.last_cells = []
        if self.render_mode == "human":
            self._render_frame()
        return (obs, Reward, terminated, truncated, info)

    def render(self):
        match self.render_mode:
            case "ansi":
                return self._render_text()
            case "human":
                self._render_frame()
            case None:
                return None
            case _:
                raise ValueError(f"unexpected value for the render mode : {self.render_mode!r}")

    def close(self) -> None:
       if self._renderer is not None:
           self._renderer.close()
           self._renderer = None

    def _render_text(self) -> str:
        header = f"pieces placed: {self.pieces_placed}, lines: {self.total_lines}, current: {self.current_piece}, next: {self.next_piece}"
        return f"{header}\n{board_to_str(self.board, self.last_cells)}"

    def _render_frame(self) -> None:
        if self._renderer is None:
            from tetris.render import PygameRenderer
            self._renderer = PygameRenderer(self.height, self.width, self.metadata["render_fps"])
        self._renderer.draw(self.board, self.last_cells,current_piece=self.current_piece, next_piece=self.next_piece,pieces_placed=self.pieces_placed, total_lines=self.total_lines)

    def _observation(self) -> Observation:
        assert self.current_piece is not None
        return {"board": self.board.copy(), "piece": PIECE_DICT[self.current_piece]}

    def _info(self) -> dict[str, Any]:
        assert self.current_piece is not None
        mask = action_mask(self.board, PIECES[self.current_piece])
        return {"action_mask": mask, "lines": self.last_lines, "total_lines": self.total_lines, "pieces_placed": self.pieces_placed, "next_piece": self.next_piece}
