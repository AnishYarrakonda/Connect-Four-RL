"""Standard feedforward MLP policy (single-frame or frame-stacked input)."""

import torch
import torch.nn as nn

from config import N_ACTIONS


class MLPPolicy(nn.Module):
    is_snn = False

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),  # plain linear decoder, no activation
        )

    def reset_state(self):
        pass  # stateless

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
