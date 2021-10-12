from matplotlib.pyplot import grid
from core.iagent import IAgent
from models.dueling import Dueling
from models.mlp import MLPnetwork
from models.cnn import CNN
import numpy as np
import os
from tensorflow.python.keras.models import save_model
from pathlib import Path
import json

# import tensorflow as tf
# graph = tf.get_default_graph()

class DQNAgent(IAgent):
    def __init__(self,
                 epsilon: float,
                 n_state: int,
                 min_epsilon: float,
                 decay: float,
                 historic_size: int,
                 n_action: int,
                 model_type: str = 'mlp',
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 grid_size: int = 10,
                 visibility: int = 5,
                 side_limit: int = 2
                 
                 ):
        super(DQNAgent, self).__init__(epsilon, n_action)
        
        self.epsilon = epsilon
        self.gamma = gamma # Discount factor
        self.min_epsilon = min_epsilon  # Ending chance of random action
        self.decay_rate = decay
        
        if model_type == 'mlp':
            self.main_qnetwork = MLPnetwork(lr=lr,
                                            historic_size=historic_size,
                                            out_size=n_action,
                                            n_state=n_state,
                                            grid_size=grid_size,
                                            visibility=visibility,
                                            side_limit=side_limit)
            self.qtargetnetwork = MLPnetwork(lr=lr,
                                             historic_size=historic_size,
                                             out_size=n_action,
                                             n_state=n_state,
                                             side_limit=side_limit,
                                             grid_size=grid_size,
                                             visibility=visibility)
        if model_type == 'cnn':
            self.main_qnetwork = CNN(lr=lr,
                                            historic_size=historic_size,
                                            out_size=n_action,
                                            n_state=n_state,
                                            grid_size=grid_size,
                                            visibility=visibility,
                                            side_limit=side_limit)
            self.qtargetnetwork = CNN(lr=lr,
                                             historic_size=historic_size,
                                             out_size=n_action,
                                             n_state=n_state,
                                             side_limit=side_limit,
                                             grid_size=grid_size,
                                             visibility=visibility)
        if model_type =='dueling':
            self.main_qnetwork = Dueling(lr=lr,
                                            historic_size=historic_size,
                                            out_size=n_action,
                                            n_state=n_state,
                                            grid_size=grid_size,
                                            visibility=visibility,
                                            side_limit=side_limit)

            self.qtargetnetwork = Dueling(lr=lr,
                                             historic_size=historic_size,
                                             out_size=n_action,
                                             n_state=n_state,
                                             side_limit=side_limit,
                                             grid_size=grid_size,
                                             visibility=visibility)
            
            # self.main_qnetwork.model._make_predict_function()
            # self.qtargetnetwork.model._make_predict_function()
        
        else:
            print('Not a valid model type')
    
    def update_epsilon(self):
        self.epsilon = np.maximum(self.epsilon * self.decay_rate, self.min_epsilon)
            
    def act(self, s: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.n_action)
        else:
            
            s = np.expand_dims(s, axis=0)
            a = np.argmax(self.main_qnetwork.model.predict(s))
            return a
    
    def save(self, path: str, name_model: str = ''):
        """tf.keras.models.save_model(
            model, filepath, overwrite=True, include_optimizer=True, save_format=None,
            signatures=None, options=None, save_traces=True)

        """
        # save_model(self.main_qnetwork, path, save_format='h5')
        if not os.path.isdir('save_models/'):
            Path("save_models/").mkdir(parents=True, exist_ok=True)
        save_model(self.main_qnetwork.model, filepath=path + '.h5')
        # with open(name_model, "w") as outfile:
        #     json.dump(self.model.to_json(), outfile)
        
    
    def load(self):
        pass