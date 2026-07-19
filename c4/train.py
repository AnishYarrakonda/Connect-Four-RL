"""AlphaZero-lite training loop. Run: python -m c4.train

Runs forever until killed. Checkpoints to runs/latest.pt continuously,
numbered snapshots + eval results to runs/log.json periodically.
"""

import json
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from . import selfplay
from .eval import evaluate
from .net import C4Net, get_device

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

GAMES_PER_ITER = 48
N_SIMS = 100
BUFFER_SIZE = 300_000
BATCH_SIZE = 512
TRAIN_STEPS_PER_ITER = 60
LR = 1e-3
EVAL_EVERY = 10
SNAPSHOT_EVERY = 25


def save_ckpt(net, opt, iteration, path):
    tmp = path + ".tmp"
    torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "iter": iteration}, tmp)
    os.replace(tmp, path)


def main():
    os.makedirs(RUNS, exist_ok=True)
    device = get_device()
    print(f"device: {device}")
    net = C4Net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)

    start_iter = 0
    latest = os.path.join(RUNS, "latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=device)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck["iter"] + 1
        print(f"resumed from iteration {start_iter}")

    log_path = os.path.join(RUNS, "log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else []

    buffer = deque(maxlen=BUFFER_SIZE)
    t0 = time.time()
    it = start_iter
    while True:
        # ---- self-play ----
        net.eval()
        t = time.time()
        samples = selfplay.play_games(net, device, GAMES_PER_ITER, N_SIMS)
        buffer.extend(samples)
        sp_time = time.time() - t

        # ---- train ----
        net.train()
        t = time.time()
        losses = []
        if len(buffer) >= BATCH_SIZE:
            for _ in range(TRAIN_STEPS_PER_ITER):
                batch = random.sample(range(len(buffer)), BATCH_SIZE)
                xs = torch.from_numpy(np.stack([buffer[i][0] for i in batch])).to(device)
                pis = torch.from_numpy(np.stack([buffer[i][1] for i in batch])).to(device)
                zs = torch.tensor([buffer[i][2] for i in batch], dtype=torch.float32).to(device)
                logits, values = net(xs)
                policy_loss = -(pis * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                value_loss = F.mse_loss(values, zs)
                loss = policy_loss + value_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append((policy_loss.item(), value_loss.item()))
        tr_time = time.time() - t

        pl = float(np.mean([l[0] for l in losses])) if losses else 0.0
        vl = float(np.mean([l[1] for l in losses])) if losses else 0.0
        entry = {
            "iter": it,
            "elapsed_min": round((time.time() - t0) / 60, 1),
            "buffer": len(buffer),
            "policy_loss": round(pl, 4),
            "value_loss": round(vl, 4),
            "selfplay_s": round(sp_time, 1),
            "train_s": round(tr_time, 1),
        }

        save_ckpt(net, opt, it, latest)
        if it % SNAPSHOT_EVERY == 0:
            save_ckpt(net, opt, it, os.path.join(RUNS, f"ckpt_{it:05d}.pt"))

        # ---- eval ----
        if it % EVAL_EVERY == 0:
            net.eval()
            entry["eval"] = evaluate(net, device, depths=(1, 3, 5), games_per_side=3, n_sims=64)

        log.append(entry)
        with open(log_path, "w") as f:
            json.dump(log, f)
        print(entry, flush=True)
        it += 1


if __name__ == "__main__":
    main()
