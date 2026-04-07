"""
q_agent.py — Tabular Q-learning agent (baseline / comparison).

Updated to use the new Gymnasium-compatible DroneEnv API:
  - env.reset() returns (obs, info)
  - env.step() returns (obs, reward, terminated, truncated, info)
"""

from __future__ import annotations

import os
import pickle
import numpy as np

from src.agents.base_agent import BaseAgent
from src.env.drone_env import DroneEnv


class QAgent(BaseAgent):
    """
    Tabular Q-learning agent.

    Expects a discrete, integer-encoded state space.
    Works as a lightweight baseline to compare against PPO.
    """

    def __init__(
        self,
        env:          DroneEnv,
        state_size:   int   = None,
        action_size:  int   = 4,
        lr:           float = 0.1,
        gamma:        float = 0.99,
        epsilon:      float = 1.0,
        epsilon_min:  float = 0.05,
        epsilon_decay:float = 0.995,
    ):
        super().__init__(env)

        self.state_size   = state_size or (env.width * env.height)
        self.action_size  = action_size
        self.lr           = lr
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay= epsilon_decay

        # Q-table initialised optimistically (small positives encourage exploration)
        self.q_table = np.zeros((self.state_size, self.action_size), dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _state_idx(self, obs: np.ndarray) -> int:
        """
        Map observation vector → scalar index for Q-table lookup.
        Uses the first element (encoded drone position) from DroneEnv._obs().
        """
        # DroneEnv encodes drone position as first H*W elements; argmax finds drone cell
        grid_part = obs[:self.env.width * self.env.height]
        # drone cell is encoded as DRONE/3 ≈ 0.333
        # find nearest: position where grid_part is closest to 1/3
        return int(np.argmax(grid_part))

    def choose_action(self, obs: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))
        idx = self._state_idx(obs)
        return int(np.argmax(self.q_table[idx]))

    def update(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        s  = self._state_idx(obs)
        s_ = self._state_idx(next_obs)

        target = reward + (0.0 if done else self.gamma * np.max(self.q_table[s_]))
        self.q_table[s, action] += self.lr * (target - self.q_table[s, action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # BaseAgent API
    # ------------------------------------------------------------------

    def train(self, timesteps: int = 50_000) -> None:
        """Run Q-learning for approximately `timesteps` env steps."""
        steps_done = 0
        episode    = 0

        while steps_done < timesteps:
            obs, _ = self.env.reset()
            total_r = 0.0
            done = False

            while not done:
                action = self.choose_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.update(obs, action, reward, next_obs, done)
                obs = next_obs
                total_r += reward
                steps_done += 1

            self.decay_epsilon()
            episode += 1

            if episode % 50 == 0:
                print(f"  [Q] episode={episode:>5} | reward={total_r:+.1f} | "
                      f"ε={self.epsilon:.3f} | steps={steps_done:,}")

    def predict(self, obs: np.ndarray) -> int:
        """Greedy action (no exploration)."""
        idx = self._state_idx(obs)
        return int(np.argmax(self.q_table[idx]))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)
        print(f"[Q] Q-table saved → {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data.get("epsilon", self.epsilon_min)
        print(f"[Q] Q-table loaded ← {path}")
