import random
from collections import deque
from typing import NamedTuple, Tuple
import numpy as np
from numpy.core.defchararray import index
from config_parameters import HISTORIC_SIZE

class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    
class Memory(object):
    """Memory Class
    Args:
                max_memory (int): Max size of this memory
    """
    
    def __init__(self, max_memory=100):
        self.max_memory = max_memory
        self.memory = deque(maxlen=max_memory)

    def remember(self, m):
        '''Store transition'''
        for t in m:
            self.memory.append(t)

    def random_access(self, size=1):
        '''Sample a batch from the memory'''
        return random.sample(self.memory, size)
 
    def __len__(self):
        """overload length operator"""
        return(len(self.memory))