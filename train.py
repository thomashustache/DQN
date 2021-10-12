import sys
sys.path.append("/Users/hustachethomas/Desktop/MasterIA/Projets/DQN/")


from utils.plots import plot_score
import pickle
from env.trainer import Trainer

# Important Parameters
BATCH_SIZE: int = 32
BUFFER_LEN = 20000
MAX_STEPS: int = 1000
MAX_LIFE = 150
LR: float = 0.001
DRAW_FREQ: int = 250
HISTORIC_SIZE: int = 1
EPSILON: float = 0.6
MIN_EPSILON: float = 0.05
DECAY = 0.97
TRAIN_FREQ: int = 1
EPOCHS = 25
DISCOUNT = 0.99
GRID_SIZE = 10
VISIBILITY = 5
N_STATE = 3
MODEL_TYPE = 'cnn'
TEMPERATURE_BONUS = 0.15
TEMPERATURE_MALUS = 0.4

my_trainer = Trainer(batch_size=BATCH_SIZE,
                        max_len_memory=BUFFER_LEN,
                        lr=LR,
                        temp_bonus=TEMPERATURE_BONUS,
                        temp_malus=TEMPERATURE_MALUS,
                        max_optim_steps=MAX_STEPS,
                        max_life=MAX_LIFE,
                        draw_frequency=DRAW_FREQ,
                        historic=HISTORIC_SIZE,
                        epsilon=EPSILON,
                        min_epsilon=MIN_EPSILON,
                        eps_decay=DECAY,
                        train_frequency=TRAIN_FREQ,
                        model_type=MODEL_TYPE,
                        epochs=EPOCHS,
                        gamma=DISCOUNT,
                        visibility=VISIBILITY,
                        n_state=N_STATE,
                        grid_size=GRID_SIZE)

if __name__ == "__main__" :
    
    my_trainer.train()

    
    
    
