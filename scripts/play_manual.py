from src.env.drone_env import DroneEnv
import random

env = DroneEnv()

state = env.reset()

for _ in range(20):

    action = random.randint(0,3)
    state, reward, done, _ = env.step(action)
    env.render()
    if done:
        print("Episode finished")
        break
