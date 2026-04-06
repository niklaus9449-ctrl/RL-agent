from abc import ABC, abstractmethod

class BaseAgent(ABC):

    def __init__(self, env):
        # store the environment
        self.env = env

    @abstractmethod
    def observe(self, obs):
        # receives what drone currently sees
        # child class fills real logic
        pass

    @abstractmethod
    def act(self, obs):
        # decides action based on observation
        # returns → up, down, left, right
        pass

    @abstractmethod
    def train(self, step_count):
        # child class will implement learning
        pass

    @abstractmethod
    def save(self, path):
        # saves model/policy to hard disk
        pass

    @abstractmethod
    def load(self, path):
        # loads model back into memory
        pass
