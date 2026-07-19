"""Evaluate a checkpoint against minimax at several depths."""

import random

import numpy as np

from . import env, mcts, minimax


def agent_move(net, device, state, n_sims=64):
    root = mcts.Node(state)
    mcts.search_batch(net, device, [root], n_sims)
    pi = mcts.policy_from_visits(root, 0.0)
    return int(np.argmax(pi))


def play_vs_minimax(net, device, depth, agent_first, n_sims=64):
    """Returns +1 agent win, 0 draw, -1 agent loss."""
    state = env.empty_state()
    agent_turn = agent_first
    while True:
        if agent_turn:
            c = agent_move(net, device, state, n_sims)
        else:
            c = minimax.best_move(state, depth)
        state = env.play(state, c)
        if env.last_mover_won(state):
            return 1 if agent_turn else -1
        if env.is_full(state[1]):
            return 0
        agent_turn = not agent_turn


def evaluate(net, device, depths=(1, 3, 5), games_per_side=4, n_sims=64):
    net.eval()
    out = {}
    for d in depths:
        wins = draws = losses = 0
        for i in range(games_per_side * 2):
            r = play_vs_minimax(net, device, d, agent_first=(i % 2 == 0), n_sims=n_sims)
            if r > 0:
                wins += 1
            elif r == 0:
                draws += 1
            else:
                losses += 1
        n = games_per_side * 2
        out[f"d{d}"] = {"win": wins / n, "draw": draws / n, "loss": losses / n}
    return out


if __name__ == "__main__":
    import sys

    import torch

    from .net import C4Net, get_device

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/latest.pt"
    device = get_device()
    net = C4Net().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    print(evaluate(net, device, depths=(1, 3, 5, 7), games_per_side=5))
