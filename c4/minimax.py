"""Alpha-beta minimax — evaluation benchmark only, never a teacher."""

import random

from . import env

ORDER = [3, 2, 4, 1, 5, 0, 6]

_EVAL_CACHE = {}


def _score_position(state):
    """Cheap heuristic: count open 3s/2s for each side."""
    current, both = state
    opp = current ^ both
    return _count_windows(current) - _count_windows(opp)


def _count_windows(pos):
    score = 0
    for shift in (1, env.COL_BITS, env.COL_BITS + 1, env.COL_BITS - 1):
        m = pos & (pos >> shift)
        three = m & (m >> shift)
        score += 5 * bin(three & env.BOARD_MASK).count("1")
        score += bin(m & env.BOARD_MASK).count("1")
    return score


def _negamax(state, depth, alpha, beta):
    key = (state, depth)
    if key in _EVAL_CACHE:
        return _EVAL_CACHE[key]
    if env.last_mover_won(state):
        return -10000 - depth  # side to move already lost
    legal = env.legal_moves(state[1])
    if not legal:
        return 0
    if depth == 0:
        return _score_position(state)
    best = -10**9
    for c in ORDER:
        if c not in legal:
            continue
        v = -_negamax(env.play(state, c), depth - 1, -beta, -alpha)
        if v > best:
            best = v
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    if len(_EVAL_CACHE) < 2_000_000:
        _EVAL_CACHE[key] = best
    return best


def best_move(state, depth, randomize=True):
    legal = env.legal_moves(state[1])
    scored = []
    for c in ORDER:
        if c not in legal:
            continue
        v = -_negamax(env.play(state, c), depth - 1, -10**9, 10**9)
        scored.append((v, c))
    top = max(scored)[0]
    if randomize:
        choices = [c for v, c in scored if v >= top - 1]
        return random.choice(choices)
    return max(scored)[1]
