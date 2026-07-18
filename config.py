"""Single source of truth for hyperparameters and per-model presets."""

from dataclasses import dataclass, field, asdict

GRID = 10
N_ACTIONS = 5  # N, S, E, W, Stay
ACTION_NAMES = ["North", "South", "East", "West", "Stay"]
# (dx, dy) with y increasing downward (row index)
ACTION_DELTAS = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0)]

MODELS = ("mlp1", "mlp3", "snn")


@dataclass
class Config:
    model: str = "mlp1"
    # environment
    grid: int = GRID
    spawn_prob: float = 0.15
    spawn_start: float = 0.15  # curriculum: initial spawn prob...
    spawn_ramp: float = 0.0  # ...ramped to spawn_prob over this fraction of episodes
    max_steps_train: int = 200
    reward_step: float = 1.0
    reward_death: float = -20.0
    # observation
    frame_stack: int = 1  # occupancy frames fed to the network (mlp3 -> 3)
    # network
    hidden: int = 128
    beta: float = 0.9  # SNN membrane decay
    learn_beta: bool = False  # per-neuron learnable membrane decay
    recurrent: bool = False  # RLeaky hidden layers (learned recurrent synapses)
    # REINFORCE
    lr: float = 1e-3
    lr_final: float = 0.0  # >0 enables cosine decay from lr to lr_final
    gamma: float = 0.97
    entropy_coef: float = 0.01
    entropy_final: float = -1.0  # >=0 enables linear anneal from entropy_coef
    critic: bool = False  # learned value baseline (training-only, not deployed)
    baseline_alpha: float = 0.05  # EMA rate for the return baseline
    grad_clip: float = 1.0
    batch_episodes: int = 1  # episodes accumulated per optimizer step
    episodes: int = 5000
    seed: int = 0
    # checkpointing
    ckpt_every: int = 250
    avg_window: int = 100

    @property
    def obs_dim(self) -> int:
        # frame_stack occupancy grids + one agent one-hot grid
        return (self.frame_stack + 1) * self.grid * self.grid

    def as_dict(self) -> dict:
        return asdict(self)


def make_config(model: str, **overrides) -> Config:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}, expected one of {MODELS}")
    cfg = Config(model=model)
    if model == "mlp3":
        cfg.frame_stack = 3
    if model == "snn":
        cfg.episodes = 10000
        cfg.beta = 0.95
        cfg.learn_beta = True
        cfg.hidden = 256
        cfg.batch_episodes = 8
        cfg.lr_final = 1e-4
        cfg.entropy_final = 0.002
        cfg.seed = 2  # best of a 4-seed sweep (REINFORCE is seed-sensitive)
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg
