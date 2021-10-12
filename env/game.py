import numpy as np
import skvideo.io
import cv2
import os
from pathlib import Path


class Game(object):
    def __init__(self,
                 grid_size: int = 10,
                 max_time: int = 200,
                 temp_bonus: float = 0.05,
                 temp_malus: float = 0.5,
                 temperature: float = 0.1,
                 n_state: int = 2,
                 visibility: int = 5,
                 side_limit: int = 2):
        
        ## size of the map (including limits)
        grid_size = grid_size + 4
        self.grid_size = grid_size
        self.visibility = visibility
        ## max number of actions
        self.max_time = max_time
        ## probability of having a cheesy cell or a poisonous cell
        self.temperature = temperature
        self.temp_bonus = temp_bonus
        self.temp_malus = temp_malus
        self.n_state = n_state

        #board on which one plays  
        ## store items
        self.board = np.zeros((grid_size,grid_size))
        ## store positions (rat pos. and game limits)
        self.position = np.zeros((grid_size,grid_size))

        # coordinate of the rat
        self.x = 0
        self.y = 1
        
        self.bonus = 0.5
        self.malus = - 1
        self.repass = 0

        # self time
        self.t = 0

        ## Parameter for the display
        self.scale = 16

        ## store every step of the game for video displaying
        self.to_draw = np.zeros((max_time + 2, grid_size * self.scale, grid_size * self.scale, 3))

    def draw(self, dir_path: str, suffix: str):
        """saves the game as a mp4 video"""
        if not os.path.isdir(dir_path):
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        skvideo.io.vwrite(str(dir_path) + suffix + '.mp4', self.to_draw)
        
    def get_frame(self, t: int):
        '''Called before every taken action to store the game as an array'''
        
        left_v = int(self.visibility / 2)
        right_v = int(self.visibility / 2) + 1
        
        b = np.zeros((self.grid_size, self.grid_size, 3)) + 128
        b[self.x - left_v: self.x + right_v, self.y - left_v: self.y + right_v, :] += 25
        b[self.board > 0, 0] = 256        ## channel0=256 -> cheese = RED
        b[self.board < 0, 2] = 256     ## channel2=256 -> poisonous = BLUE
        b[self.x, self.y, :] = 256         ## channel1/2/3=256 -> rat position = WHITE
        b[-2:, :, :] = 0                   ## 1/2/3channel = 0 -> limit of the map = BLACK
        b[:, -2:, :] = 0
        b[:2, :, :] = 0
        b[:, :2, :] = 0
        b = cv2.resize(b, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)

        self.to_draw[t, :, :, :]= b ## save frame t to the array


    def act(self, action: int):
        """This function returns the new state, reward and decides if the
        game ends.
        Actions:
                        N.1
                    O.3     E.2
                        S.0
        """

        ## Update video array
        self.get_frame(int(self.t)) 

        self.position = np.zeros((self.grid_size, self.grid_size))

        ## Set the limits  of the game (-1)
        self.position[0: 2, :] = -1
        self.position[:, 0: 2] = -1
        self.position[-2:, :] = -1
        self.position[:, -2:] = -1

        if action == 0: ## Go S except if you touch the limit then go N (repulsive move)
            if self.x == self.grid_size - 3:
                self.x = self.x-1
            else:
                self.x = self.x + 1
        elif action == 1: ## Go N except if you touch the limit then go S (repulsive move)
            if self.x == 2:
                self.x = self.x + 1
            else:
                self.x = self.x-1
        elif action == 2: ## Go E except if you touch the limit then go O (repulsive move)
            if self.y == self.grid_size - 3:
                self.y = self.y - 1
            else:
                self.y = self.y + 1
        elif action == 3: ## Go O except if you touch the limit then go E (repulsive move)
            if self.y == 2:
                self.y = self.y + 1
            else:
                self.y = self.y - 1
        else:
            RuntimeError('Error: action not recognized')

        ## Rat position is 1
        self.position[self.x, self.y] = 1
        self.t = self.t + 1 
        if self.n_state == 3:              
            reward = self.board[self.x, self.y] - self.malus_repass[self.x, self.y]
            self.malus_repass[self.x, self.y] += 0.01
        else:
            reward = self.board[self.x, self.y]
            
        self.board[self.x, self.y] = 0      ## remove the item from the map 
        game_over = self.t > self.max_time  
        
        if self.n_state == 3:
            state = np.concatenate((self.board.reshape(self.grid_size, self.grid_size, 1),
                                self.position.reshape(self.grid_size, self.grid_size, 1),
                                self.malus_repass.reshape(self.grid_size, self.grid_size, 1)),
                                axis=2)
        else:
            
            state = np.concatenate((self.board.reshape(self.grid_size, self.grid_size, 1),
                                self.position.reshape(self.grid_size, self.grid_size, 1)),
                                axis=2)
        
        state = state[self.x - 2: self.x + 3, self.y - 2: self.y + 3, :] # visibility zone of the rat is 5*5

        return state, reward, game_over

    def reset(self):
        """This function resets the game and returns the initial state"""
        
        # init rat pos
        self.x = np.random.randint(3, self.grid_size - 3, size=1)[0]
        self.y = np.random.randint(3, self.grid_size - 3, size=1)[0]
        
        # init bonus and malus
        bonus = self.bonus * np.random.binomial(1,self.temp_bonus, size=self.grid_size ** 2)
        bonus = bonus.reshape(self.grid_size,self.grid_size)
        malus = self.malus * np.random.binomial(1,self.temp_malus, size=self.grid_size ** 2)
        malus = malus.reshape(self.grid_size, self.grid_size)
        malus[bonus > 0] = 0
        self.malus_repass = np.zeros((self.grid_size, self.grid_size))
        self.board = bonus + malus
        
        self.to_draw = np.zeros((self.max_time+2, self.grid_size*self.scale, self.grid_size*self.scale, 3))
        
        self.position = np.zeros((self.grid_size, self.grid_size))
        self.position[0 : 2, :] = -1 
        self.position[:, 0 : 2] = -1
        self.position[-2:, :] = -1
        self.position[:, -2:] = -1 
        
        self.board[self.x, self.y] = 0 
        self.position[self.x,self.y] = 1 
        
        self.malus_repass = np.zeros((self.grid_size, self.grid_size))
        self.t = 0

        if self.n_state == 3:
            # new state
            state = np.concatenate((self.board.reshape(self.grid_size, self.grid_size,1),
                                    self.position.reshape(self.grid_size, self.grid_size,1),
                                    self.malus_repass.reshape(self.grid_size, self.grid_size,1)),axis=2)
        
        else:
            # new state
            state = np.concatenate((self.board.reshape(self.grid_size, self.grid_size, 1),
                                self.position.reshape(self.grid_size, self.grid_size, 1)),
                                axis=2)
        
        state = state[self.x - 2 : self.x + 3, self.y - 2 : self.y + 3, :]
                
        return state
    
        

            
            
        
        
        
        
        
        
        
    
        

