"""
drone_env.py — Gymnasium-compatible Drone Search & Rescue environment.

ROOT CAUSE OF PREVIOUS FAILURES: The observation was a flat grid of cell
codes (0,1,2,3). This gave the neural network no useful spatial structure.
The network had to learn from scratch that "value 2 at position 7 means
a victim is 2 cells away" — an extremely hard inductive leap.

THE FIX — Rich, explicit observation vector:
  Instead of a flat grid, we give the agent exactly what it needs to know:
  1. Relative direction + distance to EACH victim (dx/W, dy/H, dist/diag)
  2. Drone position (x/H, y/W)
  3. Battery remaining (%)
  4. Victims remaining (%)
  5. Wall proximity in all 4 directions (1 step look-ahead)

  This is the same principle used in all successful grid-world RL papers:
  "Don't make the network learn the coordinate system — give it to them."

  With 2 victims: obs_dim = 3×2 + 2 + 1 + 1 + 4 = 14 dimensions.
  Much smaller AND much more informative than 25 (5×5 grid) random numbers.

Reward design (unchanged and working):
  R_STEP=-0.1, R_WALL=-0.5, R_RESCUE=+10, R_COMPLETION=+25
  Perfect episode: 2×10 + 25 - ~40×0.1 = +41  (clearly positive)
  Random walk:     0 + 0 - 100×0.1 = -10        (clearly negative)
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env.grid import Grid
from src.env.entities import Drone, Victim


ACTIONS: dict[int, tuple[int, int]] = {
    0: (-1,  0),   # up
    1: ( 1,  0),   # down
    2: ( 0, -1),   # left
    3: ( 0,  1),   # right
}

R_STEP          = -0.1
R_WALL          = -0.5
R_RESCUE        =  10.0
R_COMPLETION    =  25.0
R_BATTERY_DEAD  = -5.0
SHAPING_WEIGHT  =  0.3    # slightly reduced — rich obs makes shaping less critical


class DroneEnv(gym.Env):
    """
    Grid-based Drone Search & Rescue with a rich, explicit observation space.

    Observation per step (all values normalized to [0, 1]):
      For each victim (active or rescued):
        - rel_x:  (victim.x - drone.x + H) / (2H)     direction row
        - rel_y:  (victim.y - drone.y + W) / (2W)     direction col
        - dist:   manhattan_dist / (H + W)             how far
        - active: 1.0 if not rescued, 0.0 if rescued   mask
      Drone state:
        - drone_x:   drone.x / (H-1)
        - drone_y:   drone.y / (W-1)
        - battery:   battery / battery_life
        - victims_left: remaining / num_victims
      Wall proximity (1-step look-ahead):
        - blocked_up, blocked_down, blocked_left, blocked_right  (0 or 1)

    Total: num_victims×4 + 4 + 4 = num_victims×4 + 8 dimensions
    For 2 victims: 16 dims (vs 29 flat grid — smaller AND more useful)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        width:         int  = 5,
        height:        int  = 5,
        num_victims:   int  = 2,
        num_obstacles: int  = 3,
        max_steps:     int  = 100,
        battery_life:  int  = 100,
        use_shaping:   bool = True,
        seed:          int | None = None,
    ):
        super().__init__()

        self.width         = width
        self.height        = height
        self.num_victims   = num_victims
        self.num_obstacles = num_obstacles
        self.max_steps     = max_steps
        self.battery_life  = battery_life
        self.use_shaping   = use_shaping
        self._diag         = float(height + width)   # manhattan diag for normalisation

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # obs_dim: 4 features per victim slot + 4 drone state + 4 wall sensors
        obs_dim = num_victims * 4 + 8
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.grid:    Grid | None  = None
        self.drone:   Drone | None = None
        self.victims: list[Victim] = []
        self.steps  = 0
        self._prev_shaping: float  = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.grid    = Grid(self.width, self.height)
        self.steps   = 0
        self.victims = []

        # Place obstacles
        for _ in range(self.num_obstacles):
            pos = self.grid.random_empty()
            if pos:
                self.grid.place(*pos, Grid.OBSTACLE)

        # Place drone
        pos = self.grid.random_empty()
        if pos is None:
            raise RuntimeError("Grid too full — reduce num_obstacles.")
        self.drone = Drone(x=pos[0], y=pos[1], battery=self.battery_life)
        self.grid.force_place(pos[0], pos[1], Grid.DRONE)

        # Place victims
        for _ in range(self.num_victims):
            pos = self.grid.random_empty()
            if pos:
                v = Victim(x=pos[0], y=pos[1])
                self.victims.append(v)
                self.grid.place(pos[0], pos[1], Grid.VICTIM)

        self._prev_shaping = self._potential()
        return self._obs(), {}

    def step(self, action: int):
        assert self.grid is not None, "Call reset() before step()."

        dx, dy = ACTIONS[action]
        cx, cy = self.drone.x, self.drone.y
        nx, ny = cx + dx, cy + dy

        terminated = False
        truncated  = False
        reward     = R_STEP

        # ── Wall / obstacle ────────────────────────────────────────────
        cell = self.grid.get(nx, ny)
        if cell is None or cell == Grid.OBSTACLE:
            reward = R_WALL
            self.steps += 1
            self.drone.consume()
            if self.drone.battery <= 0:
                reward    += R_BATTERY_DEAD
                terminated = True
            truncated = (not terminated) and (self.steps >= self.max_steps)
            return self._obs(), reward, terminated, truncated, self._info()

        # ── Move ───────────────────────────────────────────────────────
        self.grid.remove(cx, cy)

        if cell == Grid.VICTIM:
            for v in self.victims:
                if not v.is_rescued and v.x == nx and v.y == ny:
                    v.rescue()
                    self.drone.rescues += 1
                    reward += R_RESCUE
                    break

        self.drone.x, self.drone.y = nx, ny
        self.grid.force_place(nx, ny, Grid.DRONE)
        self.drone.consume()
        self.steps += 1

        # ── Distance shaping ───────────────────────────────────────────
        if self.use_shaping:
            curr           = self._potential()
            reward        += SHAPING_WEIGHT * (0.99 * curr - self._prev_shaping)
            self._prev_shaping = curr

        # ── Termination ────────────────────────────────────────────────
        if all(v.is_rescued for v in self.victims):
            reward    += R_COMPLETION
            terminated = True
        elif self.drone.battery <= 0:
            reward    += R_BATTERY_DEAD
            terminated = True

        truncated = (not terminated) and (self.steps >= self.max_steps)
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self) -> str:
        if self.grid is None:
            return "Not initialized."
        return (f"{self.grid.render()}\n"
                f"Step {self.steps}/{self.max_steps} | "
                f"Battery {self.drone.battery} | "
                f"Rescued {self.drone.rescues}/{self.num_victims}")

    def close(self): pass

    # ------------------------------------------------------------------
    # Rich observation — the key fix
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        H, W = self.height, self.width
        dx_n = self.drone.x
        dy_n = self.drone.y
        parts = []

        # ── Per-victim features ────────────────────────────────────────
        for v in self.victims:
            if not v.is_rescued:
                rel_x  = (v.x - dx_n + H) / (2 * H)     # [0,1] direction row
                rel_y  = (v.y - dy_n + W) / (2 * W)     # [0,1] direction col
                dist   = (abs(v.x - dx_n) + abs(v.y - dy_n)) / self._diag
                active = 1.0
            else:
                rel_x = rel_y = dist = 0.0
                active = 0.0
            parts.extend([rel_x, rel_y, dist, active])

        # ── Drone state ────────────────────────────────────────────────
        parts.append(dx_n / max(H - 1, 1))
        parts.append(dy_n / max(W - 1, 1))
        parts.append(self.drone.battery / self.battery_life)
        parts.append(self._victims_remaining / max(self.num_victims, 1))

        # ── Wall sensors (1-step look-ahead) ───────────────────────────
        for ddx, ddy in ACTIONS.values():
            nx, ny = dx_n + ddx, dy_n + ddy
            cell = self.grid.get(nx, ny)
            blocked = 1.0 if (cell is None or cell == Grid.OBSTACLE) else 0.0
            parts.append(blocked)

        return np.array(parts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _victims_remaining(self) -> int:
        return sum(1 for v in self.victims if not v.is_rescued)

    def _potential(self) -> float:
        active = [v for v in self.victims if not v.is_rescued]
        if not active:
            return 0.0
        min_d = min(abs(self.drone.x - v.x) + abs(self.drone.y - v.y) for v in active)
        return -(min_d / self._diag)

    def _info(self) -> dict[str, Any]:
        return {
            "step":              self.steps,
            "victims_remaining": self._victims_remaining,
            "battery":           self.drone.battery,
            "rescues":           self.drone.rescues,
        }
