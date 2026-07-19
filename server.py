"""FastAPI server: serves the browser UI and moves from the latest checkpoint.

Run: uvicorn server:app --port 8000
Reloads runs/latest.pt automatically whenever training saves a newer one.
"""

import json
import os
import threading

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from c4 import env, mcts
from c4.net import C4Net, get_device

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "runs", "latest.pt")
LOG = os.path.join(ROOT, "runs", "log.json")

app = FastAPI()
device = get_device()
net = C4Net().to(device)
net.eval()
_lock = threading.Lock()
_ckpt_mtime = 0.0
_ckpt_iter = -1


def _maybe_reload():
    global _ckpt_mtime, _ckpt_iter
    if not os.path.exists(CKPT):
        return
    m = os.path.getmtime(CKPT)
    if m > _ckpt_mtime:
        try:
            ck = torch.load(CKPT, map_location=device)
            net.load_state_dict(ck["model"])
            net.eval()
            _ckpt_mtime = m
            _ckpt_iter = ck.get("iter", -1)
        except Exception:
            pass  # mid-write; picked up next request


class MoveRequest(BaseModel):
    moves: list[int]  # column history from the empty board
    sims: int = 200


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))


@app.get("/api/progress")
def progress():
    if os.path.exists(LOG):
        with open(LOG) as f:
            return JSONResponse(json.load(f))
    return JSONResponse([])


@app.post("/api/move")
def ai_move(req: MoveRequest):
    with _lock:
        _maybe_reload()
        state = env.empty_state()
        for c in req.moves:
            state = env.play(state, c)
        if env.last_mover_won(state) or env.is_full(state[1]):
            return {"error": "game over"}

        root = mcts.Node(state)
        mcts.search_batch(net, device, [root], max(1, min(req.sims, 2000)))
        visits = root.N
        pi = (visits / visits.sum()).tolist() if visits.sum() > 0 else [0.0] * 7
        col = int(np.argmax(visits))
        # value estimate for the AI (side to move) from the search
        q = root.W[col] / root.N[col] if root.N[col] > 0 else 0.0

        new_state = env.play(state, col)
        won = env.last_mover_won(new_state)
        draw = (not won) and env.is_full(new_state[1])
        return {
            "column": col,
            "probs": pi,
            "value": round(float(q), 3),
            "ai_won": bool(won),
            "draw": bool(draw),
            "ckpt_iter": _ckpt_iter,
        }
