from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

def exp_smoothing(l,alpha=0.1):
    '''Exponential smoothing'''
    new_l = [l[0]]
    for i in range(1,len(l)):
        new_l.append(alpha*l[i] + (1-alpha)*new_l[-1])
    return new_l

def moving_average(l, memory):
    '''Moving average smoothing'''
    #return np.convolve(l, np.ones((memory,))/memory, mode='valid')
    cumsum = np.cumsum(np.insert(l, 0, 0)) 
    return (cumsum[memory:] - cumsum[:-memory]) / float(memory)

def plot_score(score, memory=1, alpha=1):
    '''Helper function to plot scores'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,7))
    plt.grid(True)
    ax1.plot(exp_smoothing(score, alpha))
    ax1.set_title("Score per Epoch with exponential smoothing (alpha = {})".format(alpha))
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("score")
    

    ax2.plot(moving_average(score, memory))
    ax2.set_title("Moving average of the score (memory={})".format(memory))
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score")
    
    plt.show()