import sys
sys.path.append("/Users/hustachethomas/Desktop/MasterIA/Projets/DRL_rat/")

from models.mlp import HISTORIC_SIZE
from env.trainer import Trainer

trainer = Trainer(historic=HISTORIC_SIZE)
xp = trainer.run_one_episode()
trainer.add_experience(xp)
loss = trainer.train_one_step()
