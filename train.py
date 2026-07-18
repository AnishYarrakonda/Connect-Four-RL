"""REINFORCE training loop — identical rollout/loss logic for all three models.

    python train.py --model mlp1   # single-frame MLP (expected to struggle)
    python train.py --model mlp3   # frame-stacked MLP (3 frames)
    python train.py --model snn    # single-frame SNN, persistent membrane

Resume:  python train.py --model snn --resume
"""

import argparse
import math
import os
import time
from collections import deque

import torch
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from config import MODELS, make_config
from env import DodgeEnv
from policy_mlp import MLPPolicy
from policy_snn import SNNPolicy
from utils import (
    CSVLogger, EMABaseline, ObsBuilder, SparsityTracker, ckpt_dir,
    load_checkpoint, returns_to_go, save_checkpoint,
)

console = Console()


def build_policy(cfg):
    if cfg.model == "snn":
        return SNNPolicy(cfg.obs_dim, cfg.hidden, cfg.beta,
                         learn_beta=getattr(cfg, "learn_beta", False),
                         recurrent=getattr(cfg, "recurrent", False))
    return MLPPolicy(cfg.obs_dim, cfg.hidden)


def run_episode(env, policy, obs_builder, cfg, tracker=None, greedy=False):
    """Play one episode. Returns (log_probs, entropies, rewards, steps_survived)."""
    occ, _ = env.reset()
    policy.reset_state()
    obs_builder.reset(occ)
    log_probs, entropies, rewards, obs_seq = [], [], [], []

    done = False
    while not done:
        obs = obs_builder.build(env.agent_onehot())
        obs_seq.append(obs)
        logits = policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax() if greedy else dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        if tracker is not None:
            tracker.record(policy)

        occ, _, reward, done = env.step(int(action))
        obs_builder.push(occ)
        rewards.append(reward)

    return log_probs, entropies, rewards, obs_seq, env.steps


def make_stats_table(cfg, ep, stats):
    t = Table(expand=True, header_style="bold cyan", border_style="dim")
    t.add_column("Episode", justify="right")
    t.add_column("Steps", justify="right")
    t.add_column("Return", justify="right")
    t.add_column(f"Avg({cfg.avg_window})", justify="right")
    t.add_column("Best Avg", justify="right")
    t.add_column("Loss", justify="right")
    t.add_column("Entropy", justify="right")
    if cfg.model == "snn":
        t.add_column("Sparsity", justify="right")
    t.add_column("Ep/s", justify="right")

    for row in stats:
        steps_style = "red" if row["died"] else "green"
        avg_style = "green" if row["avg_up"] else "yellow"
        cells = [
            f"{row['episode']}",
            Text(f"{row['steps']}", style=steps_style),
            f"{row['ret']:.1f}",
            Text(f"{row['avg']:.1f}", style=avg_style),
            Text(f"{row['best']:.1f}", style="bold magenta"),
            f"{row['loss']:.3f}",
            f"{row['entropy']:.3f}",
        ]
        if cfg.model == "snn":
            cells.append(f"{row['sparsity']*100:.1f}%")
        cells.append(f"{row['eps']:.1f}")
        t.add_row(*cells)
    return t


def train(cfg, resume=False, tag=""):
    torch.manual_seed(cfg.seed)
    env = DodgeEnv(spawn_prob=cfg.spawn_prob, max_steps=cfg.max_steps_train)
    policy = build_policy(cfg)
    params = list(policy.parameters())
    critic = None
    if cfg.critic:
        # training-only value baseline; the deployed policy stays purely spiking
        critic = torch.nn.Sequential(
            torch.nn.Linear(cfg.obs_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 1))
        params += list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    baseline = EMABaseline(cfg.baseline_alpha)
    obs_builder = ObsBuilder(cfg)
    tracker = SparsityTracker(policy, cfg.obs_dim, cfg.hidden) if cfg.model == "snn" else None

    d = ckpt_dir(cfg.model + tag)
    start_ep, best_avg = 0, float("-inf")
    if resume:
        path = os.path.join(d, "latest.pt")
        if os.path.exists(path):
            ck = load_checkpoint(path)
            policy.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["optimizer_state"])
            baseline.value = ck["baseline"]
            start_ep = ck["episode"]
            best_avg = ck["best_avg"]
            console.print(f"[yellow]resumed from episode {start_ep} (best avg {best_avg:.1f})[/]")
        else:
            console.print("[red]no checkpoint to resume from — starting fresh[/]")

    logger = CSVLogger(os.path.join(d, "training_log.csv"), resume=resume)
    survival_window = deque(maxlen=cfg.avg_window)
    recent_rows = deque(maxlen=12)
    prev_avg = 0.0

    header = Panel(
        f"[bold]{cfg.model}[/]  ·  obs {cfg.obs_dim}d  ·  hidden {cfg.hidden}  ·  "
        f"γ={cfg.gamma}  lr={cfg.lr}  spawn={cfg.spawn_prob}"
        + (f"  β={cfg.beta}" if cfg.model == "snn" else "")
        + f"  ·  {cfg.episodes} episodes",
        title="[bold cyan]REINFORCE training[/]", border_style="cyan",
    )
    progress = Progress(
        TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), TimeRemainingColumn(),
    )
    task = progress.add_task(f"[cyan]{cfg.model}", total=cfg.episodes, completed=start_ep)

    t0 = time.time()
    done_eps = 0
    with Live(console=console, refresh_per_second=8) as live:
        for ep in range(start_ep + 1, cfg.episodes + 1):
            frac = ep / cfg.episodes
            if cfg.lr_final > 0:
                lr_now = cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (1 + math.cos(math.pi * frac))
                for g in optimizer.param_groups:
                    g["lr"] = lr_now
            ent_coef = (cfg.entropy_coef + (cfg.entropy_final - cfg.entropy_coef) * frac
                        if cfg.entropy_final >= 0 else cfg.entropy_coef)
            # spawn-rate curriculum: easy early episodes, full difficulty after ramp
            at_full_difficulty = True
            if cfg.spawn_ramp > 0:
                ramp = min(1.0, frac / cfg.spawn_ramp)
                env.set_spawn_prob(cfg.spawn_start + (cfg.spawn_prob - cfg.spawn_start) * ramp)
                at_full_difficulty = ramp >= 1.0
            if tracker:
                tracker.reset()
            log_probs, entropies, rewards, obs_seq, steps = run_episode(
                env, policy, obs_builder, cfg, tracker
            )

            G = returns_to_go(rewards, cfg.gamma)
            ep_return = float(G[0])
            b = baseline.update(ep_return)
            critic_loss = 0.0
            if critic is not None:
                V = critic(torch.stack(obs_seq)).squeeze(-1)
                adv = G - V.detach()
                critic_loss = torch.nn.functional.mse_loss(V, G)
            else:
                adv = G - b
            if len(adv) > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            log_probs = torch.stack(log_probs)
            entropy = torch.stack(entropies).mean()
            k = max(1, cfg.batch_episodes)
            loss = (-(log_probs * adv).sum() - ent_coef * entropy
                    + 0.5 * critic_loss) / k

            # accumulate gradients over k episodes per optimizer step
            loss.backward()
            if ep % k == 0 or ep == cfg.episodes:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            survival_window.append(steps)
            avg = sum(survival_window) / len(survival_window)
            died = rewards[-1] < 0
            done_eps += 1
            sparsity = tracker.sparsity if tracker else 0.0

            min_window = min(cfg.avg_window, cfg.episodes)
            new_best = (len(survival_window) >= min_window and avg > best_avg
                        and at_full_difficulty)
            if new_best:
                best_avg = avg
                save_checkpoint(os.path.join(d, "best.pt"), policy, optimizer,
                                baseline, ep, best_avg, cfg)
            if ep % cfg.ckpt_every == 0 or ep == cfg.episodes:
                save_checkpoint(os.path.join(d, "latest.pt"), policy, optimizer,
                                baseline, ep, best_avg, cfg)

            recent_rows.append({
                "episode": ep, "steps": steps, "ret": ep_return, "avg": avg,
                "best": best_avg if best_avg > float("-inf") else 0.0,
                "loss": float(loss), "entropy": float(entropy),
                "sparsity": sparsity, "died": died,
                "avg_up": avg >= prev_avg, "eps": done_eps / (time.time() - t0),
            })
            prev_avg = avg
            logger.log(episode=ep, steps=steps, **{"return": f"{ep_return:.2f}"},
                       avg100=f"{avg:.2f}", loss=f"{float(loss):.4f}",
                       entropy=f"{float(entropy):.4f}",
                       sparsity=f"{sparsity:.4f}" if tracker else "")

            progress.update(task, completed=ep)
            renderables = [header, make_stats_table(cfg, ep, recent_rows), progress]
            if new_best:
                renderables.append(Text(f"  ✓ new best avg {best_avg:.1f} → best.pt", style="bold yellow"))
            live.update(Group(*renderables))

    console.print(f"\n[bold green]done[/] — best avg({cfg.avg_window}) survival: "
                  f"[bold magenta]{best_avg:.1f}[/] steps · checkpoints in [cyan]{d}/[/]")


def main():
    ap = argparse.ArgumentParser(description="Train a dodge policy with REINFORCE")
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--spawn-prob", dest="spawn_prob", type=float, default=None)
    ap.add_argument("--entropy-coef", dest="entropy_coef", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for the checkpoint dir (e.g. seed sweeps)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("model", "resume", "tag")}
    cfg = make_config(args.model, **overrides)
    train(cfg, resume=args.resume, tag=args.tag)


if __name__ == "__main__":
    main()
