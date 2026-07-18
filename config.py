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
    max_steps_train: int = 200
    reward_step: float = 1.0
    reward_death: float = -20.0
    # observation
    frame_stack: int = 1  # occupancy frames fed to the network (mlp3 -> 3)
    # network
    hidden: int = 128
    beta: float = 0.9  # SNN membrane decay
    # REINFORCE
    lr: float = 1e-3
    gamma: float = 0.97
    entropy_coef: float = 0.01
    baseline_alpha: float = 0.05  # EMA rate for the return baseline
    grad_clip: float = 1.0
    episodes: int = 5000
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
        cfg.episodes = 8000
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg
