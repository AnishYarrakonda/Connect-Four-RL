"""SNN policy: LIF hidden layers with membrane potential persisting across
real game timesteps. One forward pass per game tick — no internal num_steps
loop. Membrane state carries the autograd graph, so the once-per-episode
backward pass is BPTT through the whole episode.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from config import N_ACTIONS


class SNNPolicy(nn.Module):
    is_snn = True

    def __init__(self, obs_dim: int, hidden: int = 128, beta: float = 0.9):
        super().__init__()
        grad = surrogate.atan()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=False)
        self.fc2 = nn.Linear(hidden, hidden)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=False)
        self.head = nn.Linear(hidden, N_ACTIONS)  # plain linear decoder
        # The observation is sparse binary (a handful of active pixels), so
        # default init leaves membranes far below the spike threshold and the
        # network nearly silent. Scale up the synaptic weights so hidden layers
        # fire at a healthy sparse rate from the start.
        with torch.no_grad():
            self.fc1.weight.mul_(5.0)
            self.fc2.weight.mul_(5.0)
        self.hidden = hidden
        self.mem1 = None
        self.mem2 = None
        # spike stats for the most recent forward pass
        self.last_spike_count = 0.0

    def reset_state(self):
        """Call once per episode start — never between steps."""
        self.mem1 = self.lif1.reset_mem()
        self.mem2 = self.lif2.reset_mem()

    def detach_state(self):
        """Cut the autograd graph at episode boundaries / for inference reuse."""
        if self.mem1 is not None:
            self.mem1 = self.mem1.detach()
        if self.mem2 is not None:
            self.mem2 = self.mem2.detach()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.mem1 is None:
            self.reset_state()
        spk1, self.mem1 = self.lif1(self.fc1(obs), self.mem1)
        spk2, self.mem2 = self.lif2(self.fc2(spk1), self.mem2)
        self.last_spike_count = float(spk1.detach().sum() + spk2.detach().sum())
        return self.head(spk2)

    @property
    def n_hidden_neurons(self) -> int:
        return 2 * self.hidden
