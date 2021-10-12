import sys
sys.path.append("/Users/hustachethomas/Desktop/MasterIA/Projets/DRL_rat/")

from env.RatEnv import RatEnv
from env.HistoricWrapper import HistoricWrapper

ratenv = RatEnv(grid_size=10, visibility=5, max_life=500, temperature=0.3, side_limit=2)
ratenv_wrapper = HistoricWrapper(env=ratenv, historic_size=2, visibility=5)

o = ratenv_wrapper.reset()

next_o, r, d, _ = ratenv_wrapper.step(0)
assert(next_o.shape == (5, 5, 2))