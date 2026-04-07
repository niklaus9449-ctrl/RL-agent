"""
trainer.py — Unified training & evaluation orchestrator.

KEY FIX: Curriculum learning.
  The previous config (8×8 grid, 4 victims, 6 obstacles) is too hard
  for cold-start PPO. The agent needs to occasionally succeed by luck
  early on to get a positive training signal. On an 8×8 grid with 4
  victims spread randomly, random walk finds a victim only ~1% of the
  time — not enough signal to learn from.

  Start with 5×5 grid, 2 victims, 3 obstacles.
  This gives ~10-15% chance of random-walk success early on.
  Once trained, you can increase difficulty by changing ENV_CONFIG.

Usage
-----
  python3 -m src.training.trainer --agent ppo --timesteps 300000
  python3 -m src.training.trainer --agent ppo --timesteps 300000 --hard   # 8x8 grid
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.env.drone_env import DroneEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.q_agent import QAgent


# ---------------------------------------------------------------------------
# Environment configs
#
# EASY (default — use for initial training):
#   5×5 grid, 2 victims, 3 obstacles
#   Random walk finds a victim ~15% of the time → enough signal to bootstrap
#
# HARD (use --hard flag, or after easy training converges):
#   8×8 grid, 4 victims, 6 obstacles
#   Harder exploration problem — needs a pre-trained policy to start from
# ---------------------------------------------------------------------------

ENV_CONFIG_EASY = dict(
    width         = 5,
    height        = 5,
    num_victims   = 2,
    num_obstacles = 3,
    max_steps     = 100,
    battery_life  = 100,
    use_shaping   = True,
)

ENV_CONFIG_HARD = dict(
    width         = 8,
    height        = 8,
    num_victims   = 4,
    num_obstacles = 6,
    max_steps     = 150,
    battery_life  = 150,
    use_shaping   = True,
)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        agent_type:    str  = "ppo",
        timesteps:     int  = 300_000,
        save_dir:      str  = "checkpoints",
        eval_episodes: int  = 50,
        n_envs:        int  = 4,
        hard_mode:     bool = False,
    ):
        self.agent_type    = agent_type.lower()
        self.timesteps     = timesteps
        self.save_dir      = Path(save_dir)
        self.eval_episodes = eval_episodes
        self.n_envs        = n_envs

        self.env_config = ENV_CONFIG_HARD if hard_mode else ENV_CONFIG_EASY
        self.env = DroneEnv(**self.env_config)
        self.agent = self._build_agent()

    def _build_agent(self):
        if self.agent_type == "ppo":
            return PPOAgent(env=self.env, n_envs=self.n_envs)
        elif self.agent_type == "q":
            return QAgent(env=self.env)
        else:
            raise ValueError(f"Unknown agent: '{self.agent_type}'")

    def run(self) -> dict:
        self._print_header()
        t0 = time.time()

        if self.agent_type == "ppo":
            self.agent.train(
                timesteps        = self.timesteps,
                reward_threshold = 20.0,   # achievable: rescue 2 victims = ~18 reward
                save_best_to     = str(self.save_dir / "best"),
            )
        else:
            self.agent.train(timesteps=self.timesteps)

        elapsed = time.time() - t0

        self.save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = str(self.save_dir / f"{self.agent_type}_final")
        self.agent.save(ckpt)

        print(f"\n{'─'*45}")
        print(f"  Evaluating over {self.eval_episodes} episodes …")
        metrics = self.agent.evaluate(n_episodes=self.eval_episodes)
        self._print_results(metrics, elapsed)
        return metrics

    def _print_header(self) -> None:
        env_str = ", ".join(f"{k}={v}" for k, v in self.env_config.items())
        print(f"\n{'═'*55}")
        print(f"  Drone Search & Rescue — RL Trainer")
        print(f"  Agent      : {self.agent_type.upper()}")
        print(f"  Timesteps  : {self.timesteps:,}")
        print(f"  Env config : {env_str}")
        print(f"{'═'*55}\n")

    @staticmethod
    def _print_results(metrics: dict, elapsed: float) -> None:
        print(f"\n{'═'*55}")
        print(f"  EVALUATION RESULTS")
        print(f"{'─'*55}")
        print(f"  Mean reward  : {metrics['mean_reward']:+.2f}  ± {metrics['std_reward']:.2f}")
        print(f"  Mean rescues : {metrics['mean_rescues']:.2f}  victims/episode")
        print(f"  Mean length  : {metrics['mean_length']:.1f}  steps/episode")
        print(f"  Training time: {elapsed:.1f}s")
        print(f"{'═'*55}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Drone S&R RL Trainer")
    p.add_argument("--agent",      default="ppo",     choices=["ppo", "q"])
    p.add_argument("--timesteps",  default=300_000,   type=int)
    p.add_argument("--n_envs",     default=4,         type=int)
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--eval_eps",   default=50,        type=int)
    p.add_argument("--hard",       action="store_true",
                   help="Use 8x8 hard config instead of 5x5 easy config")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trainer = Trainer(
        agent_type    = args.agent,
        timesteps     = args.timesteps,
        save_dir      = args.save_dir,
        eval_episodes = args.eval_eps,
        n_envs        = args.n_envs,
        hard_mode     = args.hard,
    )
    trainer.run()
