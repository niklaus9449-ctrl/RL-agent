# evaluator.py - Evaluates an agent on a set of metrics over multiple episodes.
import numpy as np
class Evaluator:
    def __init__(self, env_fn, metrics, n_episodes=5):
        """
        env_fn: function that returns a NEW environment
        metrics: dict of {name: function(env)}
        """
        self.env_fn = env_fn
        self.metrics = metrics
        self.n_episodes = n_episodes

    def evaluate(self, agent):
        results = {name: [] for name in self.metrics}

        for _ in range(self.n_episodes):
            env = self.env_fn()
            obs = env.reset()
            done = False

            while not done:
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)

            # collect metrics at end of episode
            for name, fn in self.metrics.items():
                results[name].append(fn(env))

        return self._aggregate(results)

    def _aggregate(self, results):
        summary = {}

        for key, values in results.items():
            values = np.array(values)
            summary[key] = {
                "mean": float(values.mean()),
                "std": float(values.std())
            }

        return summary
