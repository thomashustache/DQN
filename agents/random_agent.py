import sys
sys.path.append("..")

from core.iagent import IAgent
import numpy as np

class RandomAgent(IAgent):
    def __init__(self, epsilon: float, n_action: int):
        super(RandomAgent, self).__init__(epsilon, n_action)
        

    def act(self, s: np.ndarray) -> int:
        return np.random.randint(self.n_action)
    
    def save(self):
        pass
    
    def load(self):
        pass

