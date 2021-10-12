import tensorflow.python.layers as kl
import tensorflow.python.keras.models as km
import tensorflow.python.keras.layers as kl
import tensorflow.python.keras.optimizers as ko
import tensorflow.python.keras.backend as K

class Dueling():
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
        
        # self.model = kl.Conv2D(32, 3, activation='relu')(self.inputs)
        self.model = kl.Conv2D(64, 3, activation='relu')(self.inputs)
        self.model = kl.Conv2D(64, 3, activation='relu')(self.model)
        # self.model = kl.Conv2D(512, 3, activation='relu')(self.model)
       
       # We then separate the final convolution layer into an advantage and value
        # stream. The value function is how well off you are in a given state.
        # The advantage is the how much better off you are after making a particular
        # move. Q is the value function of a state after a given action.
        # Advantage(state, action) = Q(state, action) - Value(state)
        
        self.stream_AC = kl.Lambda(lambda layer: layer[:, :, :, :32], name="advantage")(self.model)
        self.stream_VC = kl.Lambda(lambda layer: layer[:, :, :, 32:], name="value")(self.model)

        # We then flatten the advantage and value functions
        self.stream_AC = kl.Flatten(name="advantage_flatten")(self.stream_AC)
        self.stream_VC = kl.Flatten(name="value_flatten")(self.stream_VC)

        # We define weights for our advantage and value layers. We will train these
        # layers so the matmul will match the expected value and advantage from play
        self.Advantage = kl.Dense(out_size, name="advantage_final")(self.stream_AC)
        self.Value = kl.Dense(1, name="value_final")(self.stream_VC)

        # To get the Q output, we need to add the value to the advantage.
        # The advantage to be evaluated will bebased on how good the action
        # is based on the average advantage of that state
        self.model = kl.Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - K.mean(val_adv[1], axis=1, keepdims=True)),
                            name="final_out")([self.Value, self.Advantage])
        
        self.model = km.Model(self.inputs, self.model)
        # self.model.compile(ko.sgd(lr=lr, decay=1e-4, momentum=0.0), "mse")
        self.model.compile(ko.adam(lr=lr), "mse")