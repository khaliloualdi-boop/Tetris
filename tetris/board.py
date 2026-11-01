import numpy as np

from tetris.pieces import Shape


def empty_board(height: int = 20, width: int = 10) -> np.ndarray:
    return np.zeros((height, width), dtype=np.int8)

def fits(board: np.ndarray, shape: Shape, y: int, x: int) -> bool:
    H, W = board.shape
    for i in shape:
        if i[0] + y >= H or i[0] + y < 0 or i[1] + x < 0 or i[1] + x >= W:
            return False
        if board[i[0]+y, i[1]+x] == 1:
            return False
    return True

def drop_row(board: np.ndarray, shape: Shape, x: int) -> int | None:
    H = board.shape[0]
    if not fits(board, shape, 0, x):
        return None
    result = next(y for y in range(H) if not fits(board, shape, y+1, x))
    return result

def board_to_str(board: np.ndarray) -> str:
    H, W = board.shape
    result = []
    for r in range(H):
        intermediate = ""
        for c in range(W):
            if board[r, c] == 0:
                intermediate += "."
            else:
                intermediate += "#"
        result.append(intermediate)
    return "\n".join(result)

def lock(board: np.ndarray, shape: Shape, y: int, x: int) -> np.ndarray:
    new_board = board.copy()
    for i in shape:
        new_board[i[0] + y,i[1] + x] = 1
    return new_board

def clear_lines(board: np.ndarray) -> tuple[np.ndarray, int]:
    W = board.shape[1]
    full = board.all(axis=1)
    k = int(full.sum())
    new_board = np.vstack([np.zeros((k, W), dtype= board.dtype), board[~full]])

    return new_board, k

def place(board: np.ndarray, shape: Shape, x: int) -> tuple[np.ndarray, int] | None:
    y = drop_row(board, shape, x)
    if y is None:
        return None
    new_board = lock(board, shape, y, x)
    return clear_lines(new_board)
