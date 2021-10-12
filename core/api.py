from typing import NamedTuple
import numpy as np


class ActionType(NamedTuple):
    DOWN:int = 0
    UP:int = 1
    RIGHT:int = 2
    LEFT:int = 3
    
class Action(NamedTuple):
    action_type: ActionType
    
a = Action(action_type=ActionType.DOWN)