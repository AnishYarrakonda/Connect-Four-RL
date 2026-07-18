"""Shared helpers: observations, returns, baseline, sparsity/SOPs, checkpoints, plots."""

import csv
import os
from collections import deque

import numpy as np
import torch

from config import Config


# ----------------------------------------------------------------- observation

class ObsBuilder:
    """Builds the flat observation vector; handles frame stacking (zero-padded
    at episode start, newest frame last)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.frames = deque(maxlen=cfg.frame_stack)

    def reset(self, occupancy: np.ndarray):
        self.frames.clear()
        for _ in range(self.cfg.frame_stack):
            self.frames.append(np.zeros_like(occupancy))
        self.frames.append(occupancy)

    def push(self, occupancy: np.ndarray):
        self.frames.append(occupancy)

    def build(self, agent_onehot: np.ndarray) -> torch.Tensor:
        parts = list(self.frames) + [agent_onehot]
        return torch.from_numpy(np.concatenate([p.ravel() for p in parts]))


# --------------------------------------------------------------------- returns

def returns_to_go(rewards: list[float], gamma: float) -> torch.Tensor:
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return torch.from_numpy(out)


class EMABaseline:
    """Exponential moving average of episode returns."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = None

    def update(self, episode_return: float) -> float:
        if self.value is None:
            self.value = episode_return
        else:
            self.value += self.alpha * (episode_return - self.value)
        return self.value


# ------------------------------------------------------------- sparsity / SOPs

class SparsityTracker:
    """Tracks spike sparsity and estimated synaptic operations (SOPs).

    Theoretical efficiency only: on GPU/CPU the dense matmuls run regardless of
    sparsity — real savings require neuromorphic hardware (Loihi, Akida, ...).
    """

    def __init__(self, policy, obs_dim: int, hidden: int):
        self.n_neurons = getattr(policy, "n_hidden_neurons", 0)
        self.hidden = hidden
        self.obs_dim = obs_dim
        self.reset()

    def reset(self):
        self.total_spikes = 0.0
        self.steps = 0

    def record(self, policy):
        self.total_spikes += policy.last_spike_count
        self.steps += 1

    @property
    def sparsity(self) -> float:
        """Fraction of hidden neurons firing per step (0 = silent, 1 = all fire)."""
        if self.steps == 0 or self.n_neurons == 0:
            return 0.0
        return self.total_spikes / (self.steps * self.n_neurons)

    def dense_macs_per_step(self) -> int:
        """MACs a dense MLP of the same shape performs every step."""
        h, d = self.hidden, self.obs_dim
        return d * h + h * h + h * 5

    def est_sops_per_step(self) -> float:
        """Estimated synaptic ops/step on event-driven hardware: input synapses
        are driven by the (binary, typically sparse) observation; hidden-layer
        synapses fire only on spikes. Approximation: active fraction x fan-out."""
        h = self.hidden
        avg_spikes_per_step = self.total_spikes / max(self.steps, 1)
        # spikes are split across two layers; layer1 spikes drive fc2 (h fan-out),
        # layer2 spikes drive the 5-unit head
        per_layer = avg_spikes_per_step / 2
        return per_layer * h + per_layer * 5


# ----------------------------------------------------------------- checkpoints

def ckpt_dir(model: str) -> str:
    d = os.path.join("checkpoints", model)
    os.makedirs(d, exist_ok=True)
    return d


def save_checkpoint(path, policy, optimizer, baseline, episode, best_avg, cfg: Config):
    torch.save(
        {
            "model_state": policy.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "baseline": baseline.value if baseline else None,
            "episode": episode,
            "best_avg": best_avg,
            "config": cfg.as_dict(),
        },
        path,
    )


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class CSVLogger:
    FIELDS = ["episode", "steps", "return", "avg100", "loss", "entropy", "sparsity"]

    def __init__(self, path: str, resume: bool = False):
        self.path = path
        new = not (resume and os.path.exists(path))
        self.f = open(path, "w" if new else "a", newline="")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(self.FIELDS)

    def log(self, **kw):
        self.w.writerow([kw.get(k, "") for k in self.FIELDS])
        self.f.flush()
