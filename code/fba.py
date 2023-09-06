import numpy as np
from typing import List, Dict
from grid_env import GridEnvironment

def forward_recursion(env: GridEnvironment, actions: List[int], observ: List[int], \
    probs_init: List[float]) -> np.ndarray:
    '''
    Perform forward recursion to calculate the filtering probabilities.

    Calculate and return a 2D numpy array containing the filtering probabilities 
    f_{0:0} to f_{0,t-1} where t = len(observ) is the number of time steps (t >= 1).

    :param env:         The environment.
        In the Marmoset tests, env.observe_matrix is None, and exactly one of 
        env.action_effects and env.transition_matrices is None.
    :param actions:     The actions for time steps 0 to t - 2.
    :param observ:      The observations for time steps 0 to t - 1.
    :param probs_init:  The prior probabilities of all the states at time step 0.

    :return: A 2D numpy array with shape (t, N) (N is the number of states)
        the k'th row represents the values of f_{0:k} (0 <= k <= t - 1).
    '''
    
    assert len(actions) == len(observ) - 1, \
            "There should be n - 1 actions for n time steps."
            
    f_values = np.zeros((len(observ), env.n_states))
    # base case
    f00_denom = 0 # track denom (i.e. sum of f_0:0 for all states) to normalize over
    # update probabilities
    for cell_index in range(env.n_states):
        f_values[0, cell_index] = env.obs_probs[cell_index,observ[0]] * probs_init[cell_index]
        f00_denom += f_values[0, cell_index]
    # normalize probabilities
    f_values[0] /= f00_denom

    # recursive case
    for t in range(1, len(observ)): # update f_0:1, f_0:2, ... , f_0:(t-1)
        denom = 0
        for cell_index in range(env.n_states): # loop over all S_k
            inner_sum = 0
            for prev_state in range(env.n_states): # inner loop over s_{k-1}
                inner_sum += (env.trans_probs[actions[t-1],prev_state, cell_index] * f_values[t - 1, prev_state])

            # update probabilities
            f_values[t, cell_index] = env.obs_probs[cell_index,observ[t]] * inner_sum
            denom += f_values[t, cell_index]
        # normalize probabilities
        f_values[t] /= denom

    return f_values


def backward_recursion(env: GridEnvironment, actions: List[int], observ: List[int] \
    ) -> np.ndarray:
    '''
    Perform backward recursion.

    Calculate and return a 2D numpy array containing the backward recursion messages
    b_{1:t-1} to b_{t:t-1} where t = len(observ).

    :param env:         The environment.
        In the Marmoset tests, env.observe_matrix is None, and exactly one of 
        env.action_effects and env.transition_matrices is None.    
    :param actions:     The actions for time steps 0 to t - 2.
    :param observ:      The observations for time steps 0 to t - 1.

    :return: A 2D numpy array with shape (t, N) (N is the number of states)
        the k'th row represents the values of b_{k+1:t-1} (0 <= k <= t - 1).
    '''
    
    assert len(actions) == len(observ) - 1, \
            "There should be n - 1 actions for n time steps."
         
    b_values = np.zeros((len(observ), env.n_states))
    t = len(observ)
    # base case
    b_values[t-1] = 1
    # recursive case
    for k in range(t-2, -1, -1): # update b_t-1:t-1, b_t-2:t-1,..., b_1:t-1
        for state_current in range(env.n_states): # loop over all S_k-1
            inner_sum = 0
            for state_after in range(env.n_states): # inner loop over s_k
                inner_sum += (env.obs_probs[state_after, observ[k+1]] * b_values[k+1, state_after] * env.trans_probs[actions[k], state_current, state_after])

            b_values[k, state_current] = inner_sum

    return b_values


def fba(env: GridEnvironment, actions: List[int], observ: List[int], \
    probs_init: List[float]) -> np.ndarray:
    '''
    Execute the forward-backward algorithm. 

    Calculate and return a 2D numpy array containing the smoothed probabilities.
    The k'th row represents the smoothed probability distribution over all the states at time step k.

    :param env:         The environment.
        In the Marmoset tests, env.observe_matrix is None, and exactly one of 
        env.action_effects and env.transition_matrices is None.
    :param actions:     The actions for time steps 0 to t - 2.
    :param observ:      The observations for time steps 0 to t - 1.
    :param probs_init:  The prior probabilities of all the states at time step 0.

    :return: A 2D numpy array with shape (t, N) (N is the number of states)
        the k'th row represents the smoothed probabilities for time step k (0 <= k <= t - 1).
    '''
    fba = np.zeros((len(observ), env.n_states))
    f_values = forward_recursion(env, actions, observ, probs_init)
    b_values = backward_recursion(env, actions, observ)

    t = len(observ)
    for k in range(t):
        denom = 0
        for cell_index in range(env.n_states):
            fba[k, cell_index] = f_values[k, cell_index] * b_values[k, cell_index]
            denom += fba[k, cell_index]
        fba[k] /= denom
    
    return fba

