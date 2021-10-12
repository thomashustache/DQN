import sys
sys.path.append("/Users/hustachethomas/Desktop/MasterIA/Projets/DRL_rat/")

from env.trainer import Trainer
import numpy as np
# Important Parameters
BATCH_SIZE: int = 2
MAX_STEPS: int = 3
LR: float = 0.1
DRAW_FREQ: int = 100
HISTORIC_SIZE: int = 1
EPSILON: float = 0.5
DECAY = 0.95
TRAIN_FREQ: int = 5
DISCOUNT = 0.95
GRID_SIZE = 10
SIDE_LIMIT = 2
VISIBILITY = 5

if __name__ == "__main__" :
    my_trainer = Trainer(batch_size=BATCH_SIZE,
                        lr=LR,
                        max_optim_steps=MAX_STEPS,
                        draw_frequency=DRAW_FREQ,
                        historic=HISTORIC_SIZE,
                        epsilon=EPSILON,
                        eps_decay=DECAY,
                        train_frequency=TRAIN_FREQ,
                        gamma=DISCOUNT,
                        visibility=VISIBILITY,
                        grid_size=GRID_SIZE,
                        side_limit=SIDE_LIMIT)
    
    xp = my_trainer.run_one_episode()


state = np.expand_dims(my_trainer.env_wrapper.reset(), axis=0)
target_q = my_trainer.generate_target_q(
    train_states = np.vstack([state, state]),
    train_actions = np.array([0, 0]),
    train_rewards = np.array([1.0, 2.0]),
    train_next_states = np.vstack([state, state]),
    train_done = np.array([1, 1])
)

assert target_q.shape == (2,4)