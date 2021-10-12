import sys
sys.path.append("/Users/hustachethomas/Desktop/MasterIA/Projets/DRL_rat/")

import matplotlib.pyplot as plt
from typing import Tuple, Any

def test(agent, env, epochs, prefix= '', plot = False, save = True) -> Tuple[list, Any]:
    
    # Number of won games
    score = 0
    
    score_per_epoch = [] 
        
    for e in range(1, epochs + 1):
        state = env.reset()
        done = False

        while not done:
            # The agent performs an action
            action = agent.act(state)
            state, reward, done, _ = env.step(action)

        # Save as a mp4
        if save:
            env.draw(e=prefix + str(e))

        # Update stats
        score += reward
        score_per_epoch.append(reward)

        print("{}, Average score ({})".format(e, score / e))
    final_score = score / epochs
    print('Final score: {}'.format(final_score))

    if plot:
        plt.figure()
        plt.plot(score_per_epoch)
        plt.title("Score per Epoch")
        plt.xlabel("epoch")
        plt.ylabel("score")
        
    return score_per_epoch, final_score


from env.RatEnv import RatEnv
from agents.random_agent import RandomAgent

ratenv = RatEnv()
agent = RandomAgent(epsilon=1, n_action=4)

test(agent=agent,
     env=ratenv,
     epochs=1)


    
    