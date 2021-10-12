from env.game import Game
from core.ienv import IEnvironment
import numpy as np

class RatEnv(IEnvironment):
    def __init__(self,
                 grid_size: int,
                 visibility: int,
                 max_life: float,
                 temperature: float,
                 side_limit: int
                 ):

        self.game = Game(grid_size=grid_size,
                         visibility=visibility,
                         max_time=max_life,
                         temperature=temperature,
                         side_limit=side_limit
                         )
        
        self.nb_step = 0
        self.max_life = max_life
        
    
    def reset(self):
        """Reset Game
        """
        
        self.nb_step = 0
        self.state = self.game.reset()
        return self.state
                
    def step(self, action: int):
        """step method
        """
        self.game.update(action)
        self.state, r = self.game.get_state()
        done = self.end_episode()
        self.nb_step += 1
        info = {}
        return self.state, r, done, info

    def draw(self, dir_path: str, xp_name: str, first_xp: bool):
        """for rendering

        """
        self.game.draw(dir_path, xp_name, first_xp)
        
    def pos(self):
        return [self.game.x, self.game.y]
    
    def end_episode(self) -> bool:
        """Check if episode is over
        """
        if self.game.t >= self.max_life:
            return True
        else:
            return False
