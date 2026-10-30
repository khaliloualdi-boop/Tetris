import pytest

from tetris.pieces import PIECES, I, J, L, O, S, T, Z, distinct_rotations, normalize, rotate

"""Tests for Milestone 1: pieces and rotations."""

BASE_SHAPES = {"I": I, "O": O, "T": T, "S": S, "Z": Z, "L": L, "J": J}
EXPECTED_COUNTS = {"O": 1, "I": 2, "S": 2, "Z": 2, "T": 4, "L": 4, "J": 4}


# --- normalize -------------------------------------------------------------

def test_normalize_known_value():
    assert normalize([(2, 5), (3, 5), (3, 6), (3, 7)]) == ((0, 0), (1, 0), (1, 1), (1, 2))


def test_normalize_order_independent():
    assert normalize([(1, 1), (0, 0), (1, 0), (0, 1)]) == normalize([(0, 0), (0, 1), (1, 0), (1, 1)])


@pytest.mark.parametrize("name", BASE_SHAPES)
def test_normalize_translation_invariant(name):
    shape = BASE_SHAPES[name]
    shifted = [(r + 5, c - 3) for r, c in shape]
    assert normalize(shifted) == normalize(shape)


def test_normalize_accepts_generator():
    gen = ((r, c) for r, c in [(1, 1), (1, 2)])
    assert normalize(gen) == ((0, 0), (0, 1))


def test_normalize_returns_sorted_tuple():
    result = normalize([(1, 0), (0, 0)])
    assert isinstance(result, tuple)
    assert list(result) == sorted(result)


# --- rotate ----------------------------------------------------------------

def test_rotate_known_value_J():
    # #..        ##
    # ###   ->   #.
    #            #.
    assert rotate(J) == ((0, 0), (0, 1), (1, 0), (2, 0))


@pytest.mark.parametrize("name", BASE_SHAPES)
def test_rotate_period_4(name):
    shape = normalize(BASE_SHAPES[name])
    assert rotate(rotate(rotate(rotate(shape)))) == shape


# --- distinct_rotations ----------------------------------------------------

@pytest.mark.parametrize("name", BASE_SHAPES)
def test_rotation_counts(name):
    assert len(distinct_rotations(BASE_SHAPES[name])) == EXPECTED_COUNTS[name]


@pytest.mark.parametrize("name", BASE_SHAPES)
def test_rotations_have_4_cells(name):
    for shape in distinct_rotations(BASE_SHAPES[name]):
        assert len(shape) == 4
        assert len(set(shape)) == 4  # no duplicate cells


@pytest.mark.parametrize("name", BASE_SHAPES)
def test_rotations_are_normalized(name):
    for shape in distinct_rotations(BASE_SHAPES[name]):
        assert min(r for r, _ in shape) == 0
        assert min(c for _, c in shape) == 0


@pytest.mark.parametrize("name", BASE_SHAPES)
def test_rotations_are_distinct(name):
    rotations = distinct_rotations(BASE_SHAPES[name])
    assert len(set(rotations)) == len(rotations)


def test_distinct_rotations_unnormalized_input_terminates():
    shifted_O = ((1, 1), (1, 2), (2, 1), (2, 2))
    assert distinct_rotations(shifted_O) == (((0, 0), (0, 1), (1, 0), (1, 1)),)


"""Tests for Milestone 1, section 1.7: Piece class, PIECES, to_grid."""

import dataclasses

import pytest

from tetris.pieces import PIECES, Piece, to_grid

EXPECTED_COUNTS = {"O": 1, "I": 2, "S": 2, "Z": 2, "T": 4, "L": 4, "J": 4}


# --- PIECES ----------------------------------------------------------------

def test_pieces_has_the_seven_keys():
    assert set(PIECES) == {"I", "O", "T", "S", "Z", "L", "J"}


@pytest.mark.parametrize("name", EXPECTED_COUNTS)
def test_piece_name_matches_key(name):
    assert PIECES[name].name == name


@pytest.mark.parametrize("name", EXPECTED_COUNTS)
def test_piece_rotation_counts(name):
    assert len(PIECES[name].rotations) == EXPECTED_COUNTS[name]


@pytest.mark.parametrize("name", EXPECTED_COUNTS)
def test_piece_rotations_is_tuple(name):
    assert isinstance(PIECES[name].rotations, tuple)


@pytest.mark.parametrize("name", EXPECTED_COUNTS)
def test_pieces_values_are_piece_instances(name):
    assert isinstance(PIECES[name], Piece)


# --- Piece is immutable and hashable ---------------------------------------

def test_piece_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PIECES["T"].name = "X" #type:ignore

def test_piece_is_hashable():
    assert len({PIECES[k] for k in PIECES}) == 7


# --- to_grid ---------------------------------------------------------------

def test_to_grid_I_horizontal():
    assert to_grid(((0, 0), (0, 1), (0, 2), (0, 3))) == "####"


def test_to_grid_I_vertical():
    assert to_grid(((0, 0), (1, 0), (2, 0), (3, 0))) == "#\n#\n#\n#"


def test_to_grid_O():
    assert to_grid(((0, 0), (0, 1), (1, 0), (1, 1))) == "##\n##"


def test_to_grid_T():
    assert to_grid(((0, 0), (0, 1), (0, 2), (1, 1))) == "###\n.#."


@pytest.mark.parametrize("name", EXPECTED_COUNTS)
def test_to_grid_has_four_hashes(name):
    for shape in PIECES[name].rotations:
        assert to_grid(shape).count("#") == 4
