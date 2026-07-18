"""Interactive watch server: runs greedy inference server-side, streams ticks
over a WebSocket to the canvas viewer, accepts live controls.

    uvicorn server:app --port 8000     (or: python server.py)
"""

import asyncio
import json
import os

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from config import MODELS, Config
from env import DodgeEnv
from train import build_policy
from utils import ObsBuilder, load_checkpoint

app = FastAPI()

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/models")
async def models():
    out = []
    for m in MODELS:
        for kind in ("best", "latest"):
            path = os.path.join("checkpoints", m, f"{kind}.pt")
            if os.path.exists(path):
                out.append({"model": m, "checkpoint": kind, "path": path})
                break
    return out


class Session:
    """One viewer connection: env + policy + control state."""

    def __init__(self):
        self.env = DodgeEnv(spawn_prob=0.15, max_steps=None)
        self.policy = None
        self.cfg = None
        self.obs_builder = None
        self.model_name = None
        self.paused = True
        self.tick_ms = 250
        self.last_survival = None
        self.sparsity = 0.0
        self._begin_episode()

    def load_model(self, model: str) -> str | None:
        for kind in ("best", "latest"):
            path = os.path.join("checkpoints", model, f"{kind}.pt")
            if os.path.exists(path):
                ck = load_checkpoint(path)
                self.cfg = Config(**ck["config"])
                self.policy = build_policy(self.cfg)
                self.policy.load_state_dict(ck["model_state"])
                self.policy.eval()
                self.obs_builder = ObsBuilder(self.cfg)
                self.model_name = model
                self._begin_episode()
                return kind
        return None

    def _begin_episode(self):
        occ, _ = self.env.reset()
        if self.policy is not None:
            self.policy.reset_state()
            self.obs_builder.reset(occ)

    def reset(self):
        self._begin_episode()

    def tick(self):
        """Advance the sim one step. Returns True if the agent died this tick."""
        if self.policy is None:
            return False
        with torch.no_grad():
            obs = self.obs_builder.build(self.env.agent_onehot())
            logits = self.policy(obs)
            action = int(logits.argmax())
        if self.policy.is_snn:
            n = self.policy.n_hidden_neurons
            self.sparsity = self.policy.last_spike_count / n if n else 0.0
        occ, _, reward, done = self.env.step(action)
        self.obs_builder.push(occ)
        if done:
            self.last_survival = self.env.steps
            return True
        return False

    def state(self):
        s = self.env.state_dict()
        s.update({
            "model": self.model_name,
            "paused": self.paused,
            "tick_ms": self.tick_ms,
            "last_survival": self.last_survival,
            "sparsity": self.sparsity if self.model_name == "snn" else None,
        })
        return s


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    sess = Session()

    async def sim_loop():
        while True:
            if not sess.paused and sess.policy is not None:
                died = sess.tick()
                await sock.send_text(json.dumps(sess.state()))
                if died:
                    await asyncio.sleep(1.2)  # let the death flash play
                    sess.reset()
                    await sock.send_text(json.dumps(sess.state()))
            await asyncio.sleep(sess.tick_ms / 1000)

    loop_task = asyncio.create_task(sim_loop())
    try:
        await sock.send_text(json.dumps(sess.state()))
        while True:
            msg = json.loads(await sock.receive_text())
            cmd = msg.get("cmd")
            if cmd == "model":
                kind = sess.load_model(msg["model"])
                if kind is None:
                    await sock.send_text(json.dumps(
                        {"error": f"no checkpoint for {msg['model']} — train it first"}))
                    continue
                sess.paused = False
            elif cmd == "pause":
                sess.paused = True
            elif cmd == "resume":
                sess.paused = False
            elif cmd == "reset":
                sess.reset()
            elif cmd == "speed":
                sess.tick_ms = max(40, min(2000, int(msg["ms"])))
            elif cmd == "spawn_prob":
                sess.env.set_spawn_prob(float(msg["p"]))
            elif cmd == "spawn_enabled":
                sess.env.spawn_enabled = bool(msg["on"])
            elif cmd == "drop":
                sess.env.drop_obstacle_at(
                    int(msg["x"]), int(msg["y"]), int(msg["dx"]), int(msg["dy"]))
            await sock.send_text(json.dumps(sess.state()))
    except WebSocketDisconnect:
        pass
    finally:
        loop_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
