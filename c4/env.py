"""Bitboard Connect Four engine.

Board is 7 columns x 6 rows. Each column uses 7 bits (6 rows + 1 sentinel row)
in a 49-bit layout: bit index = col * 7 + row, row 0 = bottom.

State is (current_mask, both_mask): bits of the player to move, and bits of all pieces.
"""

import numpy as np

WIDTH = 7
HEIGHT = 6
COL_BITS = 7  # 6 rows + sentinel

BOTTOM_MASK = sum(1 << (c * COL_BITS) for c in range(WIDTH))
BOARD_MASK = sum(((1 << HEIGHT) - 1) << (c * COL_BITS) for c in range(WIDTH))
TOP_MASK = sum(1 << (c * COL_BITS + HEIGHT - 1) for c in range(WIDTH))


def empty_state():
    return (0, 0)


def legal_moves(both):
    """List of playable columns."""
    return [c for c in range(WIDTH) if not (both & (1 << (c * COL_BITS + HEIGHT - 1)))]


def play(state, col):
    """Play col for the side to move. Returns new state (from the new mover's view)."""
    current, both = state
    new_both = both | (both + (1 << (col * COL_BITS)))
    # opponent's pieces (current ^ old both) become the new mover's perspective;
    # the added stone lands in the previous mover's set (new_both ^ new current)
    return (current ^ both, new_both)


def is_win(pos):
    """True if bitmask pos contains 4-in-a-row."""
    # vertical
    m = pos & (pos >> 1)
    if m & (m >> 2):
        return True
    # horizontal
    m = pos & (pos >> COL_BITS)
    if m & (m >> (2 * COL_BITS)):
        return True
    # diagonal /
    m = pos & (pos >> (COL_BITS + 1))
    if m & (m >> (2 * (COL_BITS + 1))):
        return True
    # diagonal \
    m = pos & (pos >> (COL_BITS - 1))
    if m & (m >> (2 * (COL_BITS - 1))):
        return True
    return False


def last_mover_won(state):
    """After play(), the previous mover's pieces are (current ^ both)... check them."""
    current, both = state
    return is_win(current ^ both)


def is_full(both):
    return both == BOARD_MASK


def winning_cols(state):
    """Columns where the side to move wins immediately."""
    _, both = state
    out = []
    for c in legal_moves(both):
        if last_mover_won(play(state, c)):
            out.append(c)
    return out


def move_count(both):
    return bin(both).count("1")


def encode(state):
    """3x6x7 float32 planes: current player's pieces, opponent's, plies-played plane."""
    current, both = state
    opp = current ^ both
    arr = np.zeros((3, HEIGHT, WIDTH), dtype=np.float32)
    for c in range(WIDTH):
        for r in range(HEIGHT):
            bit = 1 << (c * COL_BITS + r)
            if current & bit:
                arr[0, HEIGHT - 1 - r, c] = 1.0
            elif opp & bit:
                arr[1, HEIGHT - 1 - r, c] = 1.0
    arr[2, :, :] = move_count(both) / 42.0
    return arr


def to_grid(state, current_is_p1):
    """6x7 grid of 0/1/2 (top row first) for the UI. current_is_p1: side to move is player 1."""
    current, both = state
    opp = current ^ both
    grid = [[0] * WIDTH for _ in range(HEIGHT)]
    for c in range(WIDTH):
        for r in range(HEIGHT):
            bit = 1 << (c * COL_BITS + r)
            if both & bit:
                mine = bool(current & bit)
                p = (1 if mine else 2) if current_is_p1 else (2 if mine else 1)
                grid[HEIGHT - 1 - r][c] = p
    return grid
