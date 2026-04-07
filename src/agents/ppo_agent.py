"""
ppo_agent.py — PPO agent tuned for sparse-reward grid-world S&R.

Hyperparameter rationale
────────────────────────
n_steps = 2048
  With n_envs=4, each PPO update sees 4×2048 = 8192 steps = ~80 complete
  episodes on a 5×5/100-step env. This gives a low-variance gradient estimate.
  Original 512 → only ~20 episodes per update → too noisy.

batch_size = 256
  Larger minibatch → more stable gradient, better GPU utilisation.
  Must divide n_steps × n_envs = 8192 evenly. 256 does (32 minibatches).

n_epochs = 10
  Standard. More passes over each rollout batch.

ent_coef = 0.05
  Entropy bonus encourages exploration. Grid worlds need high entropy early
  so the agent doesn't collapse to "always go right" and miss victims.
  0.01 (previous) is too low for sparse rewards. 0.05 is the SB3 recommended
  starting point for environments with sparse rewards.

learning_rate = 3e-4 with linear decay
  Start fast, finish fine. Linear decay from 3e-4 → 1e-5 over training.

gamma = 0.995
  Slightly higher than default 0.99. Our episodes are up to 100 steps and
  the completion bonus at the end must propagate back to early decisions.
  Higher gamma means future rewards (like R_COMPLETION) stay meaningful.

net_arch = [256, 256]
  Larger than [128, 128] because the observation space is 29-dimensional
  (5×5 grid + 4 extras). Needs capacity to learn spatial reasoning.
"""

from __future__ import annotations

import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor

from src.agents.base_agent import BaseAgent
from src.env.drone_env import DroneEnv


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class TrainingLogger(BaseCallback):
    """Prints progress every `log_interval` episodes. No tqdm/rich needed."""

    def __init__(self, total_timesteps: int, log_interval: int = 20, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_interval    = log_interval
        self._ep_count       = 0
        self._rewards: list[float] = []
        self._rescues: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._rewards.append(ep["r"])
                self._rescues.append(info.get("rescues", 0))
                self._ep_count += 1

                if self._ep_count % self.log_interval == 0 and self.verbose:
                    n   = self.log_interval
                    r   = np.mean(self._rewards[-n:])
                    rsc = np.mean(self._rescues[-n:])
                    pct = 100.0 * self.num_timesteps / self.total_timesteps
                    print(
                        f"  [PPO] ep={self._ep_count:>5} | "
                        f"reward={r:+7.2f} | "
                        f"rescues={rsc:.2f} | "
                        f"step={self.num_timesteps:>8,} ({pct:5.1f}%)"
                    )
        return True


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent(BaseAgent):
    """
    PPO agent for Drone S&R. Tuned for sparse-reward grid worlds.
    """

    def __init__(
        self,
        env:           DroneEnv,
        n_envs:        int   = 4,
        learning_rate: float = 3e-4,
        n_steps:       int   = 2048,    # was 512 — too few episodes per update
        batch_size:    int   = 256,     # was 64 — must divide n_steps×n_envs
        n_epochs:      int   = 10,
        gamma:         float = 0.995,   # was 0.99 — need to value future rewards more
        gae_lambda:    float = 0.95,
        clip_range:    float = 0.2,
        ent_coef:      float = 0.05,    # was 0.01 — need more exploration
        verbose:       int   = 0,
    ):
        super().__init__(env)

        # ── Build vectorised training envs ──────────────────────────────
        # IMPORTANT: pass use_shaping=True explicitly — don't rely on default
        env_kwargs = dict(
            width         = env.width,
            height        = env.height,
            num_victims   = env.num_victims,
            num_obstacles = env.num_obstacles,
            max_steps     = env.max_steps,
            battery_life  = env.battery_life,
            use_shaping   = True,          # was silently False before (bug fix)
        )
        self._env_fn = lambda: Monitor(DroneEnv(**env_kwargs))
        self._n_envs = n_envs

        self._vec_env: VecEnv = make_vec_env(self._env_fn, n_envs=n_envs)

        # ── Linear LR schedule: 3e-4 → 1e-5 ────────────────────────────
        def lr_schedule(progress: float) -> float:
            """progress goes 1.0 → 0.0 during training."""
            return 1e-5 + progress * (learning_rate - 1e-5)

        self.model = PPO(
            policy        = "MlpPolicy",
            env           = self._vec_env,
            learning_rate = lr_schedule,
            n_steps       = n_steps,
            batch_size    = batch_size,
            n_epochs      = n_epochs,
            gamma         = gamma,
            gae_lambda    = gae_lambda,
            clip_range    = clip_range,
            ent_coef      = ent_coef,
            verbose       = verbose,
            policy_kwargs = dict(net_arch=[256, 256]),  # larger net for 29-dim obs
        )

        self._eval_env = Monitor(DroneEnv(**env_kwargs))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        timesteps:        int   = 300_000,
        reward_threshold: float = 15.0,
        log_interval:     int   = 20,
        save_best_to:     str | None = None,
    ) -> None:

        callbacks = [TrainingLogger(
            total_timesteps = timesteps,
            log_interval    = log_interval,
            verbose         = 1,
        )]

        if save_best_to:
            stop_cb = StopTrainingOnRewardThreshold(
                reward_threshold = reward_threshold,
                verbose          = 1,
            )
            eval_cb = EvalCallback(
                eval_env             = self._eval_env,
                callback_on_new_best = stop_cb,
                best_model_save_path = save_best_to,
                log_path             = save_best_to,
                eval_freq            = max(5_000 // self._n_envs, 1),
                n_eval_episodes      = 30,
                deterministic        = True,
                verbose              = 1,
            )
            callbacks.append(eval_cb)

        print(f"\n{'='*55}")
        print(f"  Training PPO | timesteps={timesteps:,} | n_envs={self._n_envs}")
        print(f"  n_steps={self.model.n_steps} | batch={self.model.batch_size} | "
              f"ent={self.model.ent_coef} | γ={self.model.gamma}")
        print(f"{'='*55}\n")

        self.model.learn(
            total_timesteps = timesteps,
            callback        = callbacks,
        )

        print("\n[PPO] Training complete.\n")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)
        print(f"[PPO] Model saved → {path}")

    def load(self, path: str) -> None:
        self.model = PPO.load(path, env=self._vec_env)
        print(f"[PPO] Model loaded ← {path}")
