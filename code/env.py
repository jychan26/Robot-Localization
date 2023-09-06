# version 1.0

import numpy as np
from abc import ABC, abstractmethod, abstractproperty

class Environment(ABC):
    '''
    An abstract class that represents the dynamics of an environment. An environment consists of a set of states, legal
    actions, possible observations, state transition probabilities, and observation probabilities. States, observations
    and actions are identified by integers in {0, 1, ..., n_states - 1}, {0, 1, ..., n_observations - 1} and
    {0, 1, ..., n_actions - 1} respectively.

    :attr n_states (int): Number of possible states
    :attr n_observations (int): Number of possible observations
    :attr n_actions (int): Number of distinct possible actions
    :attr t (int): Time step
    :attr s0 (int): Initial state
    :attr trans_probs (np.ndarray) [n_actions, n_states, n_states]: State transition matrices for each action.
                      trans_probs[a] is the transition matrix for action a. A vector trans_probs[a, s1] is a probability
                      distribution over states such that trans_probs[a, s1, s2] is the probability of
                      transitioning to state s2 from s1 when action a is taken.
    :attr obs_probs: (np.array) [n_states, n_observations]: A matrix of observational distributions for each state. Each
                      row is a probability distribution over observations such that obs_probs[s, o] is the probability
                      of observing observation o when in state s.
    :attr cur_state (int): Current state
    '''

    def __init__(self, n_states: int, n_observations: int, n_actions: int, s0: int, seed: int):
        '''
        Initialize a representation of the environment.
        Set attributes of the environment and initialize the state at time step 0.
        :param n_states: Number of possible states
        :param n_observations: Number of possible observations
        :param n_actions: Number of distinct possible actions
        :param s0: Initial state
        :param seed: Random seed to ensure reproducible results
        '''

        self.n_states = n_states
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.s0 = s0
        self.t = 0
        self.cur_state = s0
        self.seed = seed
        np.random.seed(self.seed)

    @property
    @abstractmethod
    def trans_probs(self):
        return np.stack(self.n_actions * [np.eye(self.n_states)])

    @property
    @abstractmethod
    def obs_probs(self):
        return np.full((self.n_states, self.n_observations), 1. / self.n_observations)

    def reset(self):
        '''
        Resets the time step and sets the current state to the initial state.
        :return (int): Observation collected in the initial state
        '''
        self.t = 0
        self.cur_state = self.s0
        np.random.seed(self.seed)
        observation = np.random.choice(self.n_observations, p=self.obs_probs[self.cur_state])
        return observation

    def step(self, action: int):
        '''
        Given an action, samples the next state, then samples and returns an observation for the next state.
        :param action: Action to be taken
        :return (int): Observation collected in the next state
        '''
        self.t += 1
        next_state = np.random.choice(self.n_states, p=self.trans_probs[action, self.cur_state])
        observation = np.random.choice(self.n_observations, p=self.obs_probs[next_state])
        self.cur_state = next_state
        return observation

    @abstractmethod
    def visualize_state(self):
        '''
        Provides a visualization of the current state.
        '''
        pass
