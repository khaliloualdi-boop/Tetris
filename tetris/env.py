import numpy as np

from tetris.pieces import Piece
from tetris.board import place




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
    return list(afterstates(board, piece))

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
    return not bool(legal_actions(board, piece))
