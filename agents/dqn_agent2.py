
from core.agent import Agent
from utils.memory import Memory
import numpy as np

import tensorflow.python.layers as kl
import tensorflow.python.keras.models as km
import tensorflow.python.keras.layers as kl
import tensorflow.python.keras.optimizers as ko
from tensorflow.python.keras.models import model_from_json
import json

class DQN(Agent):
    def __init__(self, grid_size: int,  epsilon: float = 0.1, memory_size: int = 100, batch_size: int = 16, n_state: int = 2):
        super(DQN, self).__init__(epsilon = epsilon)

        # Discount for Q learning
        self.discount = 0.99
        self.grid_size = grid_size
        self.n_state = n_state
        self.memory = Memory(memory_size)
        self.batch_size = batch_size
        

    def learned_act(self, s):
        
        return np.argmax(self.model.predict(np.expand_dims(s, 0)))
    
    def reinforce(self, s_, n_s_, a_, r_, game_over_):
        
        # first memorize the states
        self.memory.remember([s_, n_s_, a_, r_, game_over_])
        
        # second learn from the pool
        input_states = np.zeros((self.batch_size, 5, 5, self.n_state))
        target_q = np.zeros((self.batch_size, self.n_action))
        
        if len(self.memory) < self.batch_size:
            return
        
        for i in range(self.batch_size):
            s_, n_s, a_, r_, game_over_ = self.memory.random_access()[0]
            input_states[i] = s_
            
            target_q[i] = self.model.predict(np.expand_dims(s_, 0))
            
            if game_over_:
                target_q[i, a_] = r_
            else:
                target_q[i, a_] = r_ + self.discount * np.max(self.model.predict(np.expand_dims(n_s, 0)))
                       
        # HINT: Clip the target to avoid exploiding gradients
        target_q = np.clip(target_q, -3, 3)
        l = self.model.train_on_batch(input_states, target_q)

        return l
    
    def save(self,name_weights='model.h5',name_model='model.json'):
        self.model.save_weights(name_weights, overwrite=True)
        with open(name_model, "w") as outfile:
            json.dump(self.model.to_json(), outfile)
            
    def load(self,name_weights='model.h5',name_model='model.json'):
        with open(name_model, "r") as jfile:
            model = model_from_json(json.load(jfile))
        model.load_weights(name_weights)
        model.compile("sgd", "mse")
        self.model = model

 
class DQN_FC(DQN):
    def __init__(self, *args, lr=0.1, visibility: int=5, **kwargs):
        super(DQN_FC, self).__init__( *args,**kwargs)
        
        if visibility != 0:
            
            self.inputs = kl.Input(shape=(5, 5 , 2), name="main_input")
        # else:
        #     self.inputs = kl.Input(shape=(grid_size + side_limit, grid_size + side_limit, 2, historic_size), name="main_input")
        
        self.model = kl.Flatten()(self.inputs)
        self.model = kl.Dense(64, activation="relu")(self.model)
        self.model = kl.Dense(16, activation="relu")(self.model)
        self.model = kl.Dense(self.n_action, activation="linear")(self.model)
        self.model = km.Model(self.inputs, self.model)
        
        self.model.compile(ko.sgd(lr=lr, decay=1e-4, momentum=0.0), "mse")
