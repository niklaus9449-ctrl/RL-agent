from stable_baselines3 import PPO
from .base_agent import BaseAgent

class PPOAgent(BaseAgent):
   def __init__(self,env):
      super().__init__(env)
      self.model =PPO("MLpolicy",env,verbose=0)

   def train(self,Time_steps):
      self.model.learn(total_timesteps=Time_steps)

   def predict(self,obs):
      action,_=self.model.predict(obs)
      return action

   def save (self,path):
        self.model.save(path)

   def load(self,path):
        self.model.load(path)


