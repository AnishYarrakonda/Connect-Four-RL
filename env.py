"""10x10 obstacle-dodging grid environment with swept-path collision detection.

Move order per timestep: agent moves first (based on the state it observed),
then obstacles move and are swept-checked against the agent's NEW position.
"""

import random
from dataclasses import dataclass

import numpy as np

from config import ACTION_DELTAS, GRID


@dataclass
class Obstacle:
    x: int
    y: int
    dx: int
    dy: int
    id: int
    speed: int = 1

    def path_cells(self):
        """Cells swept this step, from the current cell up to and including the
        new position. Including the current cell makes the agent stepping onto
        an obstacle (the swap case) a collision; the full sweep is correct for
        any speed (avoids tunneling)."""
        return [
            (self.x + self.dx * k, self.y + self.dy * k)
            for k in range(0, self.speed + 1)
        ]


class DodgeEnv:
    def __init__(self, grid: int = GRID, spawn_prob: float = 0.15, max_steps=None, seed=None):
        self.grid = grid
        self.spawn_prob = spawn_prob
        self.max_steps = max_steps  # None = uncapped (watch mode)
        self.spawn_enabled = True
        self.rng = random.Random(seed)
        self._next_id = 0
        self.reset()

    # ------------------------------------------------------------------ core

    def reset(self):
        g = self.grid
        self.agent_x = self.rng.randrange(g)
        self.agent_y = self.rng.randrange(g)
        self.obstacles: list[Obstacle] = []
        self.steps = 0
        self.alive = True
        return self.occupancy(), (self.agent_x, self.agent_y)

    def step(self, action: int):
        """Returns (occupancy, agent_pos, reward, done)."""
        assert self.alive, "step() called on a dead episode; call reset()"
        g = self.grid

        # 1. agent moves (off-grid attempt = stay in place)
        dx, dy = ACTION_DELTAS[action]
        nx, ny = self.agent_x + dx, self.agent_y + dy
        if 0 <= nx < g and 0 <= ny < g:
            self.agent_x, self.agent_y = nx, ny

        # 2. obstacles move; swept-path check against the agent's new position
        agent = (self.agent_x, self.agent_y)
        hit = False
        survivors = []
        for ob in self.obstacles:
            if agent in ob.path_cells():
                hit = True
            ob.x += ob.dx * ob.speed
            ob.y += ob.dy * ob.speed
            if 0 <= ob.x < g and 0 <= ob.y < g:
                survivors.append(ob)
        self.obstacles = survivors

        self.steps += 1

        if hit:
            self.alive = False
            return self.occupancy(), agent, -20.0, True

        # 3. spawn new obstacles (they enter the grid next step conceptually,
        #    but appear on an edge cell now; agent standing on that edge cell
        #    is only hit when the obstacle sweeps over it next step)
        if self.spawn_enabled and self.rng.random() < self.spawn_prob:
            self._spawn_random()

        done = self.max_steps is not None and self.steps >= self.max_steps
        return self.occupancy(), agent, 1.0, done

    # ------------------------------------------------------------ obstacles

    def _spawn_random(self):
        edge = self.rng.randrange(4)  # 0=top 1=bottom 2=left 3=right
        idx = self.rng.randrange(self.grid)
        self.drop_obstacle(edge, idx)

    def drop_obstacle(self, edge: int, index: int):
        """Spawn on an edge cell moving straight across the grid."""
        g = self.grid
        index = max(0, min(g - 1, index))
        if edge == 0:
            x, y, dx, dy = index, 0, 0, 1
        elif edge == 1:
            x, y, dx, dy = index, g - 1, 0, -1
        elif edge == 2:
            x, y, dx, dy = 0, index, 1, 0
        else:
            x, y, dx, dy = g - 1, index, -1, 0
        self.drop_obstacle_at(x, y, dx, dy)

    def drop_obstacle_at(self, x: int, y: int, dx: int, dy: int):
        self.obstacles.append(Obstacle(x, y, dx, dy, id=self._next_id))
        self._next_id += 1

    def set_spawn_prob(self, p: float):
        self.spawn_prob = max(0.0, min(1.0, p))

    # ---------------------------------------------------------- observation

    def occupancy(self) -> np.ndarray:
        """Binary grid[y][x] of obstacle presence."""
        occ = np.zeros((self.grid, self.grid), dtype=np.float32)
        for ob in self.obstacles:
            occ[ob.y, ob.x] = 1.0
        return occ

    def agent_onehot(self) -> np.ndarray:
        oh = np.zeros((self.grid, self.grid), dtype=np.float32)
        oh[self.agent_y, self.agent_x] = 1.0
        return oh

    def state_dict(self) -> dict:
        """Serializable snapshot for the watch UI."""
        return {
            "agent": [self.agent_x, self.agent_y],
            "obstacles": [
                {"id": ob.id, "x": ob.x, "y": ob.y, "dx": ob.dx, "dy": ob.dy}
                for ob in self.obstacles
            ],
            "steps": self.steps,
            "alive": self.alive,
            "spawn_prob": self.spawn_prob,
            "spawn_enabled": self.spawn_enabled,
        }


# ---------------------------------------------------------------- sanity tests

def _sanity():
    # 1. obstacle sweeps into stationary agent -> death
    env = DodgeEnv(seed=0)
    env.spawn_enabled = False
    env.reset()
    env.obstacles = []
    env.agent_x, env.agent_y = 5, 5
    env.drop_obstacle_at(5, 4, 0, 1)  # directly above, moving down
    _, _, r, done = env.step(4)  # Stay
    assert done and r == -20.0, "obstacle should hit stationary agent"

    # 2. agent steps aside -> survives
    env.reset()
    env.obstacles = []
    env.alive = True
    env.agent_x, env.agent_y = 5, 5
    env.drop_obstacle_at(5, 4, 0, 1)
    _, _, r, done = env.step(2)  # East
    assert not done and r == 1.0, "agent stepping aside should survive"

    # 3. swap case: agent moves into obstacle's old cell while obstacle moves
    #    into agent's old cell -> swept path covers agent's new position -> death
    env.reset()
    env.obstacles = []
    env.alive = True
    env.agent_x, env.agent_y = 5, 5
    env.drop_obstacle_at(5, 4, 0, 1)
    _, _, r, done = env.step(0)  # North, into (5,4)
    assert done and r == -20.0, "swap case should kill"

    # 4. off-grid move = stay
    env.reset()
    env.obstacles = []
    env.alive = True
    env.agent_x, env.agent_y = 0, 0
    env.step(3)  # West off-grid
    assert (env.agent_x, env.agent_y) == (0, 0), "off-grid move should be a no-op"

    # 5. obstacles despawn at the far edge
    env.reset()
    env.obstacles = []
    env.alive = True
    env.agent_x, env.agent_y = 0, 0
    env.drop_obstacle_at(9, 9, 0, 1)
    env.step(4)
    assert len(env.obstacles) == 0, "obstacle should despawn off-grid"

    # 6. tunneling caught at speed > 1
    env.reset()
    env.obstacles = []
    env.alive = True
    env.agent_x, env.agent_y = 5, 5
    env.drop_obstacle_at(5, 3, 0, 1)
    env.obstacles[-1].speed = 3  # jumps 3 -> 6, sweeping through 5
    _, _, r, done = env.step(4)
    assert done and r == -20.0, "swept-path check should catch tunneling"

    print("env sanity: all 6 checks passed")


if __name__ == "__main__":
    _sanity()
