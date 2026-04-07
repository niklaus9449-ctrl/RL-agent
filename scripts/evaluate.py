from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import (
    victims_found,
    steps_taken,
    exploration_efficiency,
)

from src.env.drone_env import DroneEnv
from src.agents.q_agent import DQNAgent
# from src.agents.ppo_agent import PPOAgent


def make_env():
    # adjust config as needed
    return DroneEnv()


def main():
    metrics = {
        "victims": victims_found,
        "steps": steps_taken,
        "efficiency": exploration_efficiency,
    }

    evaluator = Evaluator(make_env, metrics, n_episodes=10)

    agent = DQNAgent()  # or load pretrained

    results = evaluator.evaluate(agent)

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: mean={v['mean']:.3f}, std={v['std']:.3f}")


if __name__ == "__main__":
    main()
