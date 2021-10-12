from env.RatEnv import RatEnv
from core.ienv import IEnvironment
from typing import Tuple
import numpy as np
import copy

class HistoricWrapper(IEnvironment):
    def __init__(self, env: RatEnv, historic_size: int, visibility: int):
        self.env = env
        self.historic = historic_size
        self.visibility = visibility
        # self.grid_size = self.env.game.grid_size
        self.historic_states = []
    
    def reset(self) -> Tuple[np.ndarray]:
        obs = self.env.reset()
        obs = np.expand_dims(obs, 3)
        if self.visibility != 0:
            b = np.zeros((self.visibility, self.visibility, 2, self.historic - 1))
        else:
            b = np.zeros((self.grid_size, self.grid_size, 2, self.historic - 1))
        zero_obs = np.concatenate((b, obs), axis=3)
        self.historic_states = copy.deepcopy(zero_obs)
        return zero_obs   
    
    def get_historic_state(self, obs: np.ndarray):
        obs = np.expand_dims(obs, 3)    
        self.historic_states = np.delete(self.historic_states, 0, axis=3)
        self.historic_states = np.concatenate((self.historic_states, obs), axis = 3)
        return self.historic_states
        
    
    def step(self, action: int):
        next_obs, r, d, info = self.env.step(action)
        return self.get_historic_state(next_obs), r, d, info 
    
    def draw(self, dir_path:str , first_xp: bool, xp_name: str =''):
        self.env.draw(dir_path=dir_path, xp_name=xp_name, first_xp=first_xp)
        