import sys

from config_parameters import N_ACTIONS
from env.HistoricWrapper import HistoricWrapper
from env.RatEnv import RatEnv
from agents.dqn_agent import DQNAgent
from env.game import Game

from utils.memory import Memory, Transition
from utils.plots import plot_score
from typing import Tuple, List
import numpy as np
import tensorflow.python.keras.backend as K


class Trainer(object):
    def __init__(self,
                 visibility: int,
                 n_state: int,
                 grid_size: int = 10,
                 side_limit: int = 2,
                 max_life: float = 100,
                 historic: int = 1,
                 temperature: float = 0.3,
                 temp_bonus: float = 0.05,
                 temp_malus: float = 0.5,
                 lr: float = 0.0005,
                 epsilon: float = 0.6,
                 eps_decay: float = 0.97,
                 min_epsilon: float= 0.1,
                 gamma: float = 0.99,
                 max_optim_steps: int = 100,
                 batch_size: int = 32,
                 train_frequency: int = 1,
                 draw_frequency: int = 50,
                 epochs: int = 5,
                 max_len_memory: int = 2000,
                 model_type: str = 'mlp',
                 draw_path: str = 'experiences/',
                 save_path: str = 'save_models/'
                 ):
        
        K.clear_session()
        self.env = Game(grid_size=grid_size,
                        n_state=n_state,
                             visibility=visibility,
                             max_time=max_life,
                             temperature=temperature,
                             temp_bonus=temp_bonus,
                             temp_malus=temp_malus,
                             side_limit=side_limit
                             )
        
        self.env_wrapper = HistoricWrapper(env=self.env,
                                           historic_size=historic,
                                           visibility=visibility)
        
        self.agent = DQNAgent(epsilon=epsilon,
                              model_type=model_type,
                              n_action=N_ACTIONS,
                              n_state=n_state,
                              lr=lr,
                              historic_size=historic,
                              min_epsilon=min_epsilon,
                              decay=eps_decay,
                              grid_size=grid_size,
                              visibility=visibility,
                              side_limit=side_limit
                              )
        
        self.buffer = Memory(max_memory=max_len_memory)
        
        
        self.n_state = n_state
        if self.n_state == 3:
            self.suffix = 'newr'
            
        else:
            self.suffix = ''
        self.grid_size = grid_size + side_limit
        self.side_limit = side_limit
        self.historic = historic
        self.visibility = visibility
        self.decay = eps_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.model_type = model_type
        self.max_optim_steps = max_optim_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_frequency = train_frequency
        self.losses = []
        self.dictount_rewards = [0] * self.max_optim_steps
        self.rewards = [0] * self.max_optim_steps
        self.current_step = 0
        self.draw_path = draw_path
        self.draw_frequency = draw_frequency
        self.save_path = save_path
        self.first_xp = True
        
    def update_target_graph(self):
        updated_weights = np.array(self.agent.main_qnetwork.model.get_weights())
        self.agent.qtargetnetwork.model.set_weights(updated_weights)
    
    def get_position(self):
        return [self.env_wrapper.env.game.x, self.env_wrapper.env.game.y]
    
    def get_possible_reward(self, action: int) -> float:
        
        [x, y] = self.get_position()
        if action == 0: ## Go S except if you touch the limit then go N (repulsive move)
            if x == self.grid_size - (self.side_limit + 1):
                return self.env_wrapper.env.game.board[x - 1, y]
            else:
                return self.env_wrapper.env.game.board[x + 1, y]
        elif action == 1: ## Go N except if you touch the limit then go S (repulsive move)
            if x == self.side_limit:
                return self.env_wrapper.env.game.board[x + 1, y]
            else:
                return self.env_wrapper.env.game.board[x - 1, y]
        elif action == 2: ## Go E except if you touch the limit then go O (repulsive move)
            if y == self.grid_size - (self.side_limit + 1):
                return self.env_wrapper.env.game.board[x, y - 1]
            else:
                return self.env_wrapper.env.game.board[x, y + 1]
        elif action == 3: ## Go O except if you touch the limit then go E (repulsive move)
            if y == self.side_limit:
                return self.env_wrapper.env.game.board[x, y + 1]
            else:
                return self.env_wrapper.env.game.board[x, y - 1]
            
    def run_one_episode(self):
        
        collected_xp = []
        d = False
        obs = self.env.reset()
        k = 0
        while not d:
            action = self.agent.act(obs)
            # expected_reward = self.get_possible_reward(action)
            next_obs, r, d = self.env.act(action)
            self.dictount_rewards[self.current_step] += self.gamma ** k * r
            self.rewards[self.current_step] += r
            t = [obs, action, r, next_obs, d]
            collected_xp.append(Transition(*t))
            obs = next_obs
            k += 1
        return collected_xp

    def add_experience(self, exp: List[Transition]):
        self.buffer.remember(exp)
        
    def split_batch(self, transitions: Tuple[Transition]) -> Tuple[np.ndarray]:
        
        train_states = np.array([t.state for t in transitions])
        train_actions = np.array([t.action for t in transitions])
        train_next_states = np.array([t.next_state for t in transitions])
        train_rewards = np.array([t.reward for t in transitions])
        train_done = np.array([t.done for t in transitions])
        
        return train_states, train_actions, train_rewards, train_next_states, train_done
            
    def generate_target_q(self,
                          train_states: np.ndarray,
                          train_actions: np.ndarray,
                          train_rewards: np.ndarray,
                          train_next_states: np.ndarray,
                          train_done: np.ndarray) -> np.ndarray:

        q_target = self.agent.main_qnetwork.model.predict(train_states)
        q_values_next_state = self.agent.qtargetnetwork.model.predict(train_next_states)
            
        train_gameover = train_done == 0        
        maxq_values_next_state = np.max(q_values_next_state[range(self.batch_size)], axis=1)
        actual_reward = train_rewards + self.gamma * (maxq_values_next_state * train_gameover)
        q_target[range(self.batch_size), train_actions] = actual_reward
        
        # to avoid exploding gradients
        q_target = np.clip(q_target, -3, 3)
        return q_target
    
    def train_one_step(self) -> float:
        
        transitions = self.buffer.random_access(self.batch_size)
        train_states, train_actions, train_rewards, train_next_states, train_done = self.split_batch(transitions)
        
        # Generate target Q
        target_q = self.generate_target_q(
            train_states=train_states,
            train_actions=train_actions,
            train_rewards=train_rewards,
            train_next_states=train_next_states,
            train_done=train_done
        )
        
        loss = self.agent.main_qnetwork.model.train_on_batch(train_states, target_q)
        return loss 
    
    def verbose(self, e: int):
        print("Epoch {:03d}/{:03d} | discount reward : {:2.2f} | proba random action : {:1.2f}"
              .format(e, self.max_optim_steps, self.dictount_rewards[self.current_step], self.agent.epsilon))
              
        self.agent.save(path=self.save_path + self.model_type)     
    
    def train(self):
        
        # make both Qnetwork equal
        self.update_target_graph()
        self.current_step = 0
        self.first_xp = True
        while self.current_step < self.max_optim_steps:
            xp = self.run_one_episode()
            self.add_experience(xp)
            if len(self.buffer) >= self.batch_size:
                
                
                if self.current_step % self.train_frequency == 0:
                    current_loss = 0
                    for _ in range(self.epochs):
                        current_loss += self.train_one_step()
                    
                    self.losses += [current_loss / self.epochs]
                    self.update_target_graph()
                    self.agent.update_epsilon()

            if self.current_step % 10 == 0:
                self.verbose(e=self.current_step)
            if self.current_step % self.draw_frequency == 0:
                self.env.draw(dir_path=self.draw_path + self.model_type +'/',
                                      suffix=self.model_type + str(self.current_step) + self.suffix)
                self.first_xp = False
                
            self.current_step += 1
            
        self.env.draw(dir_path=self.draw_path + self.model_type +'/',
                      suffix=self.model_type + str(self.current_step) + self.suffix)
            
        plot_score(score=self.rewards, alpha=0.5, memory=50)    
        plot_score(score=self.dictount_rewards, alpha=0.5, memory=50)
        plot_score(score=self.losses, alpha=0.5, memory=50)
        
        
           

    
            