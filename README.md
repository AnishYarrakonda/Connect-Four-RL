# Obstacle Dodging — SNN vs MLP

Does a spiking network's built-in membrane-potential memory let it dodge moving
obstacles from **single-frame input**, matching an MLP that needs frame-stacking
to infer motion?

Three REINFORCE-trained policies on the same 10x10 grid environment:

| Model | Input | Temporal memory |
|---|---|---|
| `mlp1` | 1 occupancy frame | none (expected to struggle) |
| `mlp3` | 3 stacked frames | engineered (frame stack) |
| `snn`  | 1 occupancy frame | built-in (LIF membrane, never reset between steps) |

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --model mlp1        # ~5000 episodes
python train.py --model mlp3
python train.py --model snn         # ~10000 episodes (slower per episode)
```

Live colored progress table in the terminal. Checkpoints land in
`checkpoints/<model>/` — `best.pt` (highest avg-100 survival) and `latest.pt`
(every 250 episodes). Resume an interrupted run with `--resume`. Useful
overrides: `--episodes --lr --gamma --beta --hidden --spawn-prob
--entropy-coef --seed --tag` (`--tag` suffixes the checkpoint dir, handy for
seed sweeps — REINFORCE is seed-sensitive, the shipped SNN is the best of 4).

## Evaluate

```bash
python evaluate.py                  # 200 greedy episodes per model
```

Prints a comparison table (mean/median survival, success rates) plus the SNN's
spike sparsity and estimated synaptic operations, and saves
`plots/training_curves.png` + `plots/eval_comparison.png`.

**Efficiency caveat:** sparsity/SOPs are *theoretical* hardware savings
(Loihi/Akida-class chips). On CPU/GPU, PyTorch runs dense matmuls regardless —
no wall-clock or energy savings are observed here, by design of the report.

## Watch (interactive browser viewer)

```bash
python server.py                    # then open http://127.0.0.1:8000
```

- Pick a policy (loads its `best.pt`), watch greedy play with smooth
  interpolated animation (sim stays fully discrete server-side).
- Live controls: pause/reset, tick speed, spawn rate, auto-spawn toggle.
- Click any **edge tile** to hand-drop an obstacle heading inward — line
  several up to test the "multiple incoming obstacles" scenario.
- SNN mode shows live spike sparsity.

## Environment rules

10x10 bounded grid; actions N/S/E/W/Stay (off-grid move = stay). Obstacles
spawn at random edges (configurable probability, no cap), move 1 tile/step in a
straight line, pass through each other, despawn off-grid. Agent moves first,
then obstacles move with **swept-path** collision against the agent's new
position (tunnel-proof if speeds > 1 are ever reintroduced; stepping onto an
obstacle also kills). Reward: **+1**/step survived, **−20** on death. Training
episodes cap at 200 steps; watch mode is uncapped.

## Key design points

- **SNN:** `snn.Leaky` hidden layers (surrogate gradient, `atan`, per-neuron
  *learnable* β initialized at 0.95), one forward pass per real game tick — no
  internal `num_steps` loop. Membrane potential persists across steps and
  carries the autograd graph, so the backward pass is BPTT through the whole
  episode. Readout is the canonical snnTorch decoder: a non-spiking output
  `Leaky` (`reset_mechanism="none"`) whose membrane potential is the logits —
  hidden layers still communicate purely by spikes.
- **REINFORCE:** returns-to-go, EMA baseline, per-episode advantage
  normalization, entropy bonus (0.01, annealed for the SNN), grad-clip 1.0,
  Adam 1e-3 (cosine-decayed to 1e-4 for the SNN), γ=0.97. SNN updates
  accumulate gradients over 8 episodes per optimizer step to tame variance.
- Swapping policies changes only which module `train.py` builds — rollout,
  loss, and logging are identical across all three.
