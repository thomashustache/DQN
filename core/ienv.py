from typing import Tuple, Any, Union
import numpy as np
from abc import ABC, abstractmethod

class IEnvironment(ABC):
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[Union[Tuple[np.ndarray], np.ndarray], float, bool, Any]:
        pass
    


    