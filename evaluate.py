"""Greedy evaluation + comparison of the three trained policies.

    python evaluate.py                 # all models with a best.pt
    python evaluate.py --episodes 500  # more eval episodes
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import MODELS, Config
from env import DodgeEnv
from train import build_policy, run_episode
from utils import ObsBuilder, SparsityTracker, load_checkpoint

console = Console()

LABELS = {
    "mlp1": "MLP · single frame",
    "mlp3": "MLP · 3-frame stack",
    "snn": "SNN · single frame, persistent membrane",
}


def evaluate_model(model: str, episodes: int, max_steps: int, seed: int = 123):
    path = os.path.join("checkpoints", model, "best.pt")
    if not os.path.exists(path):
        return None
    ck = load_checkpoint(path)
    cfg = Config(**ck["config"])
    policy = build_policy(cfg)
    policy.load_state_dict(ck["model_state"])
    policy.eval()

    env = DodgeEnv(spawn_prob=cfg.spawn_prob, max_steps=max_steps, seed=seed)
    obs_builder = ObsBuilder(cfg)
    tracker = SparsityTracker(policy, cfg.obs_dim, cfg.hidden) if cfg.model == "snn" else None

    survivals = []
    with torch.no_grad():
        for _ in range(episodes):
            *_, steps = run_episode(env, policy, obs_builder, cfg, tracker, greedy=True)
            survivals.append(steps)
    s = np.array(survivals)
    return {
        "model": model,
        "trained_episodes": ck["episode"],
        "mean": s.mean(), "median": np.median(s), "max": s.max(),
        "p_survive_50": (s >= 50).mean(), "p_survive_cap": (s >= max_steps).mean(),
        "tracker": tracker, "cfg": cfg, "survivals": s,
    }


def plot_all(results, max_steps):
    os.makedirs("plots", exist_ok=True)

    # training curves from CSV logs
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in MODELS:
        path = os.path.join("checkpoints", model, "training_log.csv")
        if not os.path.exists(path):
            continue
        eps, avgs = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                eps.append(int(row["episode"]))
                avgs.append(float(row["avg100"]))
        ax.plot(eps, avgs, label=LABELS[model])
    ax.set_xlabel("episode"); ax.set_ylabel("avg(100) survival steps")
    ax.set_title("Training curves"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("plots/training_curves.png", dpi=150)

    # eval comparison bars
    fig, ax = plt.subplots(figsize=(7, 5))
    names = [LABELS[r["model"]] for r in results]
    means = [r["mean"] for r in results]
    ax.bar(range(len(results)), means, color=["#888", "#4a90d9", "#e07b39"][: len(results)])
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel(f"mean survival steps (greedy, cap {max_steps})")
    ax.set_title("Evaluation comparison"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig("plots/eval_comparison.png", dpi=150)
    console.print("[cyan]plots saved to plots/training_curves.png and plots/eval_comparison.png[/]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=1000)
    args = ap.parse_args()

    results = []
    for model in MODELS:
        console.print(f"[cyan]evaluating {model}...[/]")
        r = evaluate_model(model, args.episodes, args.max_steps)
        if r is None:
            console.print(f"[yellow]  no checkpoint for {model} — skipped[/]")
        else:
            results.append(r)
    if not results:
        console.print("[red]no checkpoints found — train first[/]")
        return

    t = Table(title=f"Greedy evaluation · {args.episodes} episodes · step cap {args.max_steps}",
              header_style="bold cyan")
    t.add_column("Policy")
    t.add_column("Trained eps", justify="right")
    t.add_column("Mean survival", justify="right")
    t.add_column("Median", justify="right")
    t.add_column("Max", justify="right")
    t.add_column("≥50 steps", justify="right")
    t.add_column("Hit cap", justify="right")
    for r in results:
        t.add_row(LABELS[r["model"]], str(r["trained_episodes"]),
                  f"[bold]{r['mean']:.1f}[/]", f"{r['median']:.0f}", f"{r['max']}",
                  f"{r['p_survive_50']:.0%}", f"{r['p_survive_cap']:.0%}")
    console.print(t)

    for r in results:
        tr = r["tracker"]
        if tr is None:
            continue
        console.print(Panel(
            f"spike sparsity: [bold]{tr.sparsity:.2%}[/] of hidden neurons fire per step\n"
            f"estimated SOPs/step (event-driven): [bold]{tr.est_sops_per_step():,.0f}[/]  vs  "
            f"dense MACs/step (same-shape MLP): [bold]{tr.dense_macs_per_step():,}[/]  "
            f"(~[bold]{tr.dense_macs_per_step() / max(tr.est_sops_per_step(), 1):.0f}x[/] theoretical reduction)\n\n"
            "[dim]Theoretical efficiency only: these savings require neuromorphic hardware "
            "(Loihi, Akida, ...). On CPU/GPU, PyTorch does dense matmuls regardless of "
            "sparsity — no wall-clock or energy savings are observed here.[/]",
            title="[bold]SNN efficiency (theoretical)[/]", border_style="magenta"))

    plot_all(results, args.max_steps)


if __name__ == "__main__":
    main()
