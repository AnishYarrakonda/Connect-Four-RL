# Connect Four RL — Plan

## Goal
A pure-RL Connect Four agent (AlphaZero-lite) trained by self-play on this Mac, playable in the browser.
Strength targets: decent after ~1h, better than ~95% of humans after ~3h, near-perfect overnight.

## Approach
Pure reinforcement learning via self-play — no imitation of minimax anywhere in training.

- **Engine**: bitboard Connect Four (two 64-bit masks), O(1) win detection via bit shifts.
- **Network**: small ResNet in PyTorch — input 3×6×7 planes (my pieces, opponent pieces, turn),
  ~6 residual blocks × 64 channels, policy head (7 logits) + value head (tanh scalar). Runs on MPS.
- **MCTS**: PUCT search guided by the net. Self-play uses batched leaf evaluation across many
  parallel games so the GPU stays fed.
- **Self-play**: each game starts with 0–8 random opening plies (both colors randomized) so the
  agent sees diverse, human-like positions — the "random starting positions" idea, done with
  reachable positions instead of unreachable random boards. Dirichlet noise at the root +
  temperature schedule for exploration.
- **Training loop**: generate games → push (state, MCTS visit distribution, outcome) into a replay
  buffer → gradient steps (policy cross-entropy + value MSE) → checkpoint → repeat forever until killed.
- **Minimax role**: evaluation benchmark ONLY (never a teacher). Win rate vs minimax at depths
  1/3/5/7 is logged over training to `runs/log.json`.

## Files
- `c4/env.py` — bitboard engine
- `c4/net.py` — policy/value ResNet
- `c4/mcts.py` — batched MCTS
- `c4/minimax.py` — alpha-beta benchmark opponent
- `c4/selfplay.py` — parallel self-play game generation
- `c4/train.py` — main training loop (checkpoints + eval + logging), run: `python -m c4.train`
- `c4/eval.py` — standalone eval of a checkpoint vs minimax
- `server.py` — FastAPI server serving moves from the latest checkpoint, run: `uvicorn server:app`
- `static/index.html` — browser UI: playable board, AI move probabilities + value display,
  training-progress chart

## Usage
1. `pip install -r requirements.txt`
2. `python -m c4.train` — leave running (1h / 3h / overnight; checkpoints save continuously to `runs/`)
3. In another terminal: `uvicorn server:app --port 8400`, open http://localhost:8400
   (port 8000 is already occupied by another app on this machine)
   (works while training is still running — server picks up the newest checkpoint)

## Acceptance
- After overnight training: agent (with search at play time) is effectively unbeatable for humans;
  win rate vs depth-7 minimax high.
- Browser: play vs AI, see its per-column probabilities and value estimate, view training curves.
