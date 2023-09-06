import matplotlib.pyplot as plt
import numpy as np
from fba import forward_recursion, backward_recursion, fba
from grid_env import GridEnvironment

if __name__ == '__main__':

    '''
    The code below demonstrates how to create a GridEnvironment object that represents the robot localization problem.
    '''

    width = 16 # specify width in cells
    height = 4 # specify height in cells

    # Define which cell coordinates correspond to empty squares in which the robot can be
    empty_cell_coords = [[0,0],[0,1],[0,2],[0,3],[0,5],[0,6],[0,7],[0,8],[0,9],[0,11],[0,12],[0,13],[0,15],
                        [1,2],[1,3],[1,5],[1,8],[1,10],[1,12],
                        [2,1],[2,2],[2,3],[2,5],[2,8],[2,9],[2,10],[2,11],[2,12],[2,15],
                        [3,0],[3,1],[3,3],[3,4],[3,5],[3,7],[3,8],[3,9],[3,10],[3,12],[3,13],[3,14],[3,15]]
    n_empty_cells = len(empty_cell_coords)
    
    init_cell = 31 # Set the robot's initial location
    epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # Set the robot's sensor error rate
    n_epsilons = len(epsilons)
    actions = [1,1,0,0,0,0,3,1,2,1] # actions
    n_actions = len(actions)
    probs_init = [1/n_empty_cells] * n_empty_cells # initial probability follows uniform distribution
    avg_max_prob = [0] * (n_actions + 1)
    for j in range(n_epsilons):
        max_prob = [0] * n_actions
        for i in range(10):            
            # set observations and environmentGrid
            observ = [0] * (n_actions + 1)
            env = GridEnvironment(width, height, empty_cell_coords, init_cell, epsilons[j], i) # use i as seed
            observ[0] = env.reset()
            for k in range(n_actions):
                observ[k + 1] = env.step(actions[k])
            
            # run fba and get maximum probability
            fba1 = fba(env, actions, observ, probs_init)
            max_prob[i] = max(fba1[-1])
        
        avg_max_prob[j] = sum(max_prob)/len(max_prob)

    # plot maximum probability in the robot's belief distribution,
    # averaged across each trial against epsilon
    plt.plot(epsilons, avg_max_prob)
    plt.xlabel('Epsilon')
    plt.ylabel('Average Maximum Probability at t = 10')
    plt.show()
   
    ####################################
    epsilon = 0.2
    gammas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # Set the robot's sensor error rate
    n_gammas = len(gammas)
    avg_max_prob = [0] * (n_actions + 1)
    for j in range(n_gammas):
        max_prob = [0] * n_actions
        for i in range(10):            
            # set observations and environmentGrid
            observ = [0] * (n_actions + 1)
            env = GridEnvironment(width, height, empty_cell_coords, init_cell, epsilon, i, gammas[j]) # use i as seed
            observ[0] = env.reset()
            for k in range(n_actions):
                observ[k + 1] = env.step(actions[k])
            
            # run fba and get maximum probability
            fba2 = fba(env, actions, observ, probs_init)
            max_prob[i] = max(fba2[4])
        
        avg_max_prob[j] = sum(max_prob)/len(max_prob)

    # plot maximum probability in the robot's belief distribution,
    # averaged across each trial against epsilon
    plt.plot(gammas, avg_max_prob)
    plt.xlabel('Gamma')
    plt.ylabel('Average Maximum Probability at t = 4')
    plt.show()