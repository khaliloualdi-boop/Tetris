import numpy as np


def column_heights(board: np.ndarray) -> np.ndarray:
    H = board.shape[0]
    result = board.argmax(axis=0)
    check = board.any(axis=0)
    for i, r in enumerate(result):
        if r == 0:
            if check[i] == 0:
                result[i] = 0
            else:
                result[i] = H
        else:
            result[i] = H - r
    return result

def aggregate_height(board: np.ndarray) -> int:
    heights = column_heights(board)
    return int(heights.sum())

def bumpiness(board: np.ndarray) -> int:
    diff = np.diff(column_heights(board))
    result = [abs(x) for x in diff]
    return int(sum(result))

def holes(board: np.ndarray) -> int:
    height = aggregate_height(board)
    count = board.sum()
    return int(height - count)

def features(board: np.ndarray, lines: int) -> np.ndarray:
    return np.array([aggregate_height(board), holes(board), bumpiness(board), lines], dtype=float)
