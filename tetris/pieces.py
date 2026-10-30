from collections.abc import Iterable
from dataclasses import dataclass

type Cell = tuple[int, int]
type Shape = tuple[Cell,...]


I = tuple(sorted([(0,0),(0,1),(0,2),(0,3)]))
O = tuple(sorted([(0,0),(0,1),(1,0),(1,1)]))
T = tuple(sorted([(1,1),(0,0),(0,1),(0,2)]))
S = tuple(sorted([(0,1),(0,2),(1,1),(1,0)]))
J = tuple(sorted([(0,0),(1,0),(1,1),(1,2)]))
L = tuple(sorted([(0,2),(1,0),(1,1),(1,2)]))
Z = tuple(sorted([(0,0),(0,1),(1,1),(1,2)]))


def normalize(cells: Iterable[Cell]) -> Shape:
    cells = list(cells)
    r_min = min([cell[0] for cell in cells])
    c_min = min([cell[1] for cell in cells])
    result = []

    for i in cells:
        new_piece = (i[0] - r_min, i[1] - c_min)
        result.append(new_piece)

    return tuple(sorted(result))

def rotate(cells: Iterable[Cell]) -> Shape:
    result = []

    for cell in cells:
        new_cell = (cell[1], -cell[0])
        result.append(new_cell)

    return normalize(result)

def distinct_rotations(shape: Shape) -> tuple[Shape,...]:
    valid_shape = normalize(shape)
    result = [valid_shape]
    new_shape = rotate(shape)
    while new_shape != valid_shape:
        result.append(new_shape)
        new_shape = rotate(new_shape)

    return tuple(result)

def to_grid(shape: Shape) -> str:
    result = []
    h = max(shape, key=lambda y: y[0])[0] + 1
    l = max(shape, key=lambda y: y[1])[1] + 1

    for r in range(h):
        line = ""
        for c in range(l):
            if (r,c) in shape:
                line += "#"
            else:
                line += "."
        result.append(line)
    return "\n".join(result)

Base_shapes = {"I": I,"O": O,"T": T,"S": S,"J": J,"L": L,"Z": Z}

@dataclass(frozen=True)
class Piece :
    name: str
    rotations: tuple[Shape,...]

PIECES = {name: Piece(name, distinct_rotations(p)) for name, p in Base_shapes.items()}
