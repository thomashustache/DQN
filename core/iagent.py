from abc import ABC, abstractmethod
from core.api import Action

class IAgent(ABC):
    def __init__(self, epsilon: float, n_action: int):
        self.epsilon = epsilon
        self.n_action = n_action

    @abstractmethod
    def act(self) -> Action:
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
    