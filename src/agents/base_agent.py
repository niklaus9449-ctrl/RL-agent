"""
base_agent.py — Abstract base for all RL agents in the Drone S&R system.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseAgent(ABC):
    """
    Interface every agent must satisfy.
    Concrete agents: PPOAgent, QAgent, etc.
    """

    def __init__(self, env: Any):
        self.env = env

    @abstractmethod
    def train(self, timesteps: int) -> None:
        """Run the training loop for `timesteps` environment steps."""
        ...

    @abstractmethod
    def predict(self, obs: np.ndarray) -> int:
        """Return the best action for a given observation (deterministic)."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model weights / Q-table to `path`."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore model weights / Q-table from `path`."""
        ...

    def evaluate(self, n_episodes: int = 20) -> dict[str, float]:
        """
        Run greedy rollouts and return aggregate statistics.
        Works for any agent that implements predict().
        """
        rewards, rescues, lengths = [], [], []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            total_r = 0.0
            done = False

            while not done:
                action = self.predict(obs)
                obs, r, terminated, truncated, info = self.env.step(action)
                total_r += r
                done = terminated or truncated

            rewards.append(total_r)
            rescues.append(info.get("rescues", 0))
            lengths.append(info.get("step", 0))

        return {
            "mean_reward":  float(np.mean(rewards)),
            "std_reward":   float(np.std(rewards)),
            "mean_rescues": float(np.mean(rescues)),
            "mean_length":  float(np.mean(lengths)),
        }
