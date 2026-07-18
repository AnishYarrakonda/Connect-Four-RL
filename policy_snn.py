"""SNN policy: LIF hidden layers with membrane potential persisting across
real game timesteps. One forward pass per game tick — no internal num_steps
loop. Membrane state carries the autograd graph, so the once-per-episode
backward pass is BPTT through the whole episode.

Two variants (cfg.recurrent):
  - Leaky:  feed-forward LIF, memory = passive membrane decay only
  - RLeaky: recurrent LIF — learned all-to-all recurrent synapses give each
    hidden layer explicit spiking working memory on top of the membrane trace
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from config import N_ACTIONS


class SNNPolicy(nn.Module):
    is_snn = True

    def __init__(self, obs_dim: int, hidden: int = 128, beta: float = 0.9,
                 learn_beta: bool = False, recurrent: bool = False):
        super().__init__()
        grad = surrogate.atan()
        self.recurrent = recurrent
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        if recurrent:
            self.lif1 = snn.RLeaky(beta=beta, linear_features=hidden, all_to_all=True,
                                   spike_grad=grad, learn_beta=learn_beta)
            self.lif2 = snn.RLeaky(beta=beta, linear_features=hidden, all_to_all=True,
                                   spike_grad=grad, learn_beta=learn_beta)
        else:
            self.lif1 = snn.Leaky(beta=beta * torch.ones(hidden) if learn_beta else beta,
                                  spike_grad=grad, init_hidden=False, learn_beta=learn_beta)
            self.lif2 = snn.Leaky(beta=beta * torch.ones(hidden) if learn_beta else beta,
                                  spike_grad=grad, init_hidden=False, learn_beta=learn_beta)
        # Non-spiking readout: logits are the membrane potential of an output
        # Leaky layer that never resets — the canonical snnTorch decoder. The
        # hidden layers still communicate purely by spikes.
        self.head = nn.Linear(hidden, N_ACTIONS)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=False,
                                 learn_beta=learn_beta,
                                 reset_mechanism="none", output=True)
        # The observation is sparse binary (a handful of active pixels), so
        # default init leaves membranes far below the spike threshold and the
        # network nearly silent. Scale up the synaptic weights so hidden layers
        # fire at a healthy sparse rate from the start.
        with torch.no_grad():
            self.fc1.weight.mul_(5.0)
            self.fc2.weight.mul_(5.0)
        self.hidden = hidden
        self.mem1 = self.mem2 = self.mem_out = None
        self.spk1 = self.spk2 = None
        # spike stats for the most recent forward pass
        self.last_spike_count = 0.0

    def reset_state(self):
        """Call once per episode start — never between steps."""
        if self.recurrent:
            self.spk1, self.mem1 = self.lif1.init_rleaky()
            self.spk2, self.mem2 = self.lif2.init_rleaky()
        else:
            self.mem1 = self.lif1.reset_mem()
            self.mem2 = self.lif2.reset_mem()
        self.mem_out = self.lif_out.reset_mem()

    def detach_state(self):
        """Cut the autograd graph at episode boundaries / for inference reuse."""
        for name in ("mem1", "mem2", "mem_out", "spk1", "spk2"):
            v = getattr(self, name)
            if isinstance(v, torch.Tensor):
                setattr(self, name, v.detach())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.mem1 is None:
            self.reset_state()
        if self.recurrent:
            self.spk1, self.mem1 = self.lif1(self.fc1(obs), self.spk1, self.mem1)
            self.spk2, self.mem2 = self.lif2(self.fc2(self.spk1), self.spk2, self.mem2)
        else:
            self.spk1, self.mem1 = self.lif1(self.fc1(obs), self.mem1)
            self.spk2, self.mem2 = self.lif2(self.fc2(self.spk1), self.mem2)
        self.last_spike_count = float(self.spk1.detach().sum() + self.spk2.detach().sum())
        _, self.mem_out = self.lif_out(self.head(self.spk2), self.mem_out)
        return self.mem_out

    @property
    def n_hidden_neurons(self) -> int:
        return 2 * self.hidden
