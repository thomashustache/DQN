import sys
sys.path.append("/Users/hustachethomas/Desktop/MasterIA/Projets/DRL_rat/")

from env.trainer2 import Trainer

# Important Parameters
BATCH_SIZE: int = 1024
MAX_STEPS: int = 200
LR: float = 0.0001
DRAW_FREQ: int = 100
HISTORIC_SIZE: int = 1
EPSILON: float = 0.6
DECAY = 0.95
TRAIN_FREQ: int = 20
DISCOUNT = 0.9
GRID_SIZE = 10
SIDE_LIMIT = 2
VISIBILITY = 0

if __name__ == "__main__" :
    my_trainer = Trainer(
                        lr=LR,
                        historic=HISTORIC_SIZE,
                        visibility=VISIBILITY,
                        grid_size=GRID_SIZE,
                        side_limit=SIDE_LIMIT,
                        gamma=DISCOUNT,
                        temperature=0.3,
                        max_memory=2000,
                        max_life=MAX_STEPS)
    
    my_trainer.train()
    
