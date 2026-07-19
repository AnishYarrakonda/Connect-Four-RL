"""Parallel self-play game generation with randomized openings."""

import random

import numpy as np

from . import env, mcts


def _safe_random_opening(max_plies):
    """Random legal opening avoiding immediate wins/losses, so games start mid-flight."""
    state = env.empty_state()
    plies = random.randint(0, max_plies)
    for _ in range(plies):
        legal = env.legal_moves(state[1])
        # skip moves that win on the spot or hand the opponent a win next move
        ok = []
        for c in legal:
            nxt = env.play(state, c)
            if env.last_mover_won(nxt):
                continue
            if env.winning_cols(nxt):
                continue
            ok.append(c)
        if not ok:
            break
        state = env.play(state, random.choice(ok))
    return state


class Game:
    def __init__(self, max_opening_plies):
        self.state = _safe_random_opening(max_opening_plies)
        self.history = []  # (state, pi) from the mover's perspective
        self.result = None  # z for the player who moved FIRST in history

    def ply(self):
        return env.move_count(self.state[1])


def play_games(net, device, n_games, n_sims, max_opening_plies=8, temp_plies=10):
    """Play n_games self-play games in parallel. Returns list of (encoded, pi, z) samples."""
    games = [Game(max_opening_plies) for _ in range(n_games)]
    active = list(games)

    while active:
        roots = [mcts.Node(g.state) for g in active]
        # expand + noise
        mcts.search_batch(net, device, roots, 0)
        for r in roots:
            if r.expanded:
                mcts.add_dirichlet_noise(r)
        mcts.search_batch(net, device, roots, n_sims)

        still = []
        for g, r in zip(active, roots):
            temp = 1.0 if g.ply() < temp_plies else 0.05
            pi = mcts.policy_from_visits(r, temp)
            a = int(np.random.choice(7, p=pi / pi.sum()))
            # store the *visit* distribution at temp=1 as the training target
            target = mcts.policy_from_visits(r, 1.0)
            g.history.append((g.state, target))
            g.state = env.play(g.state, a)
            if env.last_mover_won(g.state):
                # the player who just moved won
                g.result = ("last_mover",)
            elif env.is_full(g.state[1]):
                g.result = ("draw",)
            else:
                still.append(g)
        active = still

    samples = []
    for g in games:
        n = len(g.history)
        for i, (state, pi) in enumerate(g.history):
            if g.result[0] == "draw":
                z = 0.0
            else:
                # winner is the mover of the LAST history entry
                z = 1.0 if (n - 1 - i) % 2 == 0 else -1.0
            x = env.encode(state)
            samples.append((x, pi, z))
            # horizontal flip symmetry
            samples.append((x[:, :, ::-1].copy(), pi[::-1].copy(), z))
    return samples
