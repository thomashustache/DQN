import tensorflow.python.layers as kl
import tensorflow.python.keras.models as km
import tensorflow.python.keras.layers as kl
import tensorflow.python.keras.optimizers as ko

class MLPnetwork:
    def __init__(self,
                 lr: float,
                 n_state: int,
                 historic_size: int,
                 out_size: int,
                 visibility: int,
                 side_limit: int,
                 grid_size: int
                 ):
        
        if visibility != 0:
            
            self.inputs = kl.Input(shape=(visibility, visibility, n_state), name="main_input")
        else:
            self.inputs = kl.Input(shape=(grid_size + side_limit, grid_size + side_limit, n_state), name="main_input")
        self.model = kl.Flatten()(self.inputs)
        self.model = kl.Dense(64, activation="relu")(self.model)
        self.model = kl.Dense(16, activation="relu")(self.model)
        self.model = kl.Dense(out_size, activation="linear")(self.model)
        self.model = km.Model(self.inputs, self.model)
        
        self.model.compile(loss='mse', optimizer=ko.Adam(lr=lr))
        # self.model.compile(ko.sgd(lr=lr, decay=1e-4, momentum=0.0), "mse")
        # self.model.compile(ko.rmsprop(lr=lr), "mse")
        
        