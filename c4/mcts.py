"""PUCT MCTS with batched leaf evaluation across many parallel searches."""

import math

import numpy as np
import torch

from . import env

C_PUCT = 1.5


class Node:
    __slots__ = ("state", "children", "P", "N", "W", "legal", "terminal_value", "expanded", "_leaf_value")

    def __init__(self, state):
        self.state = state
        self.children = {}
        self.P = None
        self.N = np.zeros(7, dtype=np.float32)
        self.W = np.zeros(7, dtype=np.float32)
        self.legal = env.legal_moves(state[1])
        # terminal from the perspective of the side to move at this node
        if env.last_mover_won(state):
            self.terminal_value = -1.0  # previous mover won; side to move has lost
        elif not self.legal:
            self.terminal_value = 0.0
        else:
            self.terminal_value = None
        self.expanded = False

    def select_child(self):
        sqrt_total = math.sqrt(self.N.sum() + 1)
        best, best_score = -1, -1e9
        for a in self.legal:
            q = self.W[a] / self.N[a] if self.N[a] > 0 else 0.0
            u = C_PUCT * self.P[a] * sqrt_total / (1 + self.N[a])
            s = q + u
            if s > best_score:
                best_score, best = s, a
        return best


def _evaluate_batch(net, device, nodes):
    x = torch.from_numpy(np.stack([env.encode(n.state) for n in nodes])).to(device)
    with torch.no_grad():
        logits, values = net(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    values = values.cpu().numpy()
    for n, p, v in zip(nodes, probs, values):
        mask = np.zeros(7, dtype=np.float32)
        mask[n.legal] = 1.0
        p = p * mask
        s = p.sum()
        n.P = p / s if s > 1e-8 else mask / mask.sum()
        n.expanded = True
        n._leaf_value = float(v)


def add_dirichlet_noise(node, alpha=1.0, frac=0.25):
    noise = np.random.dirichlet([alpha] * len(node.legal))
    for i, a in enumerate(node.legal):
        node.P[a] = (1 - frac) * node.P[a] + frac * noise[i]


def search_batch(net, device, roots, n_sims):
    """Run n_sims MCTS simulations on each root simultaneously (batched NN calls)."""
    # expand any unexpanded roots first
    fresh = [r for r in roots if not r.expanded and r.terminal_value is None]
    if fresh:
        _evaluate_batch(net, device, fresh)

    for _ in range(n_sims):
        paths = []  # (root_idx, [(node, action), ...], leaf)
        leaves_to_eval = []
        for r in roots:
            if r.terminal_value is not None:
                paths.append(None)
                continue
            node, path = r, []
            while True:
                if node.terminal_value is not None:
                    paths.append((path, node, node.terminal_value))
                    break
                if not node.expanded:
                    leaves_to_eval.append(node)
                    paths.append((path, node, None))
                    break
                a = node.select_child()
                if a not in node.children:
                    node.children[a] = Node(env.play(node.state, a))
                path.append((node, a))
                node = node.children[a]

        if leaves_to_eval:
            _evaluate_batch(net, device, leaves_to_eval)

        for item in paths:
            if item is None:
                continue
            path, leaf, tv = item
            v = tv if tv is not None else leaf._leaf_value
            # v is from the perspective of the side to move at the leaf
            for node, a in reversed(path):
                v = -v
                node.N[a] += 1
                node.W[a] += v


def policy_from_visits(root, temperature):
    visits = root.N.copy()
    if visits.sum() == 0:
        pi = np.zeros(7, dtype=np.float32)
        pi[root.legal] = 1.0 / len(root.legal)
        return pi
    if temperature < 1e-3:
        pi = np.zeros(7, dtype=np.float32)
        pi[int(visits.argmax())] = 1.0
        return pi
    v = (visits / visits.max()) ** (1.0 / temperature)
    return (v / v.sum()).astype(np.float32)
