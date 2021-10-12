import sys
sys.path.append("/Users/hustachethomas/Desktop/MasterIA/Projets/DRL_rat/")

from env.trainer import Trainer

# Important Parameters
BATCH_SIZE: int = 128
MAX_STEPS: int = 250
LR: float = 0.1
DRAW_FREQ: int = 100
HISTORIC_SIZE: int = 1
EPSILON: float = 0.5
DECAY = 0.95
TRAIN_FREQ: int = 5
DISCOUNT = 0.95
GRID_SIZE = 5
SIDE_LIMIT = 1
VISIBILITY = 0

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
    
    my_trainer.run_one_episode()
    
