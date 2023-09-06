# version 1.0

from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors

from env import Environment


class GridEnvironment(Environment):
    '''
    A cell's identifier is its index in cell_coords.
    :attr width: Width of grid, in cells
    :attr height: Height of grid, in cells
    :attr empty_cell_coords: List of the coordinates of empty cells. The i'th element is a 2-element list giving the
                            (y, x) coordinate of the i'th empty cell. y=0 is the top and x=0 is the left.
    :attr full_cell_coords: List of the coordinates of full cells. The i'th element is a 2-element list giving the
                            (y, x) coordinate of the i'th empty cell. y=0 is the top and x=0 is the left.
    :attr epsilon: The error rate of the robot's sensors, in [0, 1]
    :attr cell_neighbours: Adjacency list for cells in the grid. The i'th element is a list of the other empty
                                cells that neighbour cell i
    '''

    def __init__(self, width: int, height: int, empty_cell_coords: List[List[int]], init_cell: int, epsilon: float, seed: int = 0, gamma: float = 0.85):
        '''
        Initialize a representation of the grid environment from Russell and Norvig (section 14.3.2 in 4th edition).
        There are len(empty_cell_coords) empty squares on a height X width grid of cells. The state is the cell in which
        the robot currently is.
        :param empty_cell_coords: List of the coordinates of empty cells. The i'th element is a 2-element list giving the
                            (y, x) coordinate of the i'th empty cell. y=0 is the top and x=0 is the left.
        :param width: Width of grid, in cells
        :param height: Height of grid, in cells
        :param init_cell: Initial cell that the robot is in
        :param epsilon: The error rate of the robot's sensors, in [0, 1]
        :param seed: Random seed to ensure reproducible results
        :param gamma: The probability the robot sucessfully makes then intended action
        '''
        self.validate_inputs(width, height, empty_cell_coords, init_cell, epsilon)
        n_states = len(empty_cell_coords)
        n_actions = 4               # The possible actions are up, right, down, and left
        n_observations = 2 ** 4     # The robot can observe a wall in the up, right, down or left direction
        super(GridEnvironment, self).__init__(n_states, n_observations, n_actions, init_cell, seed)

        self.width = width
        self.height = height
        self.empty_cell_coords = empty_cell_coords
        self.full_cell_coords = self.determine_full_cells()
        self.epsilon = epsilon
        self.gamma = gamma
        self.actions = [[-1,0], [0,1], [1,0], [0,-1]] # Up, Right, Down, Left
        self.cell_neighbours = self.determine_cell_neighbours()
        self._trans_probs = self.create_trans_probs()
        self._obs_probs = self.create_obs_probs()

    @property
    def trans_probs(self):
        return self._trans_probs

    @property
    def obs_probs(self):
        return self._obs_probs

    ######## DO NOT CHANGE ANY CODE ABOVE THIS LINE #################################################################

    def create_trans_probs(self):
        '''
        Calculate the transition probabilities for each state
        :return (np.array) [4, self.n_states, self.n_states]: Transition probabilities
        '''

        # Update transition probability:
        # For each empty cell, update the transition probability matrix relative to the intended action
        # and for each neighbour (and itself so that probability sums to 1)
        # i.e. for each cell, update trans_probs[a,i,j] where a is a number corresponding to the intended
        #      action, i is the current cell index, and j is the neighbouring cell we moved to (including itself)
        # Note: due to Python float division rounding error, we multiply all probabilities by 100 then divide by 100 again
        trans_probs = np.zeros((4, self.n_states, self.n_states)) # track transition probabilities
        prob_intended_direction = self.gamma*100 # probability of moving in the intended direction
        prob_other_direction = (100-self.gamma*100)/3 # probability of moving in other directions
        for cell_index in range(self.n_states):
            cell_neighbours =  self.cell_neighbours[cell_index]
            prob_stationary = [100] * 4 # keep track of the probability of remaining stationary for each action
            # update probability relative to each neighbour and intended action
            for nb_index in cell_neighbours:
                nb = self.empty_cell_coords[nb_index]
                vertical_move = nb[0] - self.empty_cell_coords[cell_index][0] # -1 if up, 1 if down
                horizontal_move = nb[1] - self.empty_cell_coords[cell_index][1] # -1 if left, 1 if right
                if vertical_move == -1: # up
                    trans_probs[0,cell_index,nb_index] = prob_intended_direction
                    trans_probs[1,cell_index,nb_index] = prob_other_direction
                    trans_probs[2,cell_index,nb_index] = prob_other_direction
                    trans_probs[3,cell_index,nb_index] = prob_other_direction
                elif vertical_move == 1: # down
                    trans_probs[0,cell_index,nb_index] = prob_other_direction
                    trans_probs[1,cell_index,nb_index] = prob_other_direction
                    trans_probs[2,cell_index,nb_index] = prob_intended_direction
                    trans_probs[3,cell_index,nb_index] = prob_other_direction
                elif horizontal_move == -1: # left
                    trans_probs[0,cell_index,nb_index] = prob_other_direction
                    trans_probs[1,cell_index,nb_index] = prob_other_direction
                    trans_probs[2,cell_index,nb_index] = prob_other_direction
                    trans_probs[3,cell_index,nb_index] = prob_intended_direction
                elif horizontal_move == 1: # right
                    trans_probs[0,cell_index,nb_index] = prob_other_direction
                    trans_probs[1,cell_index,nb_index] = prob_intended_direction
                    trans_probs[2,cell_index,nb_index] = prob_other_direction
                    trans_probs[3,cell_index,nb_index] = prob_other_direction
                # update probability of remaining stationary for each action
                for i in range(4):
                    prob_stationary[i] -= trans_probs[i,cell_index,nb_index]
            # update the probability of remaining stationary    
            for i in range(4):
                trans_probs[i,cell_index,cell_index] = prob_stationary[i]
        
        trans_probs /= 100
        
        # make sure negative values are 0
        trans_probs[trans_probs < 0] = 0

        return trans_probs
    
    def create_obs_probs(self):
        '''
        Calculate the observational probability distributions for each state. In the textbook, each observation is
        a string of 4 bits, where each bit is set if the robot's sensors indicate that there is a wall in the up,
        right, down and left directions. The robot has 4 sensors - one for each direction. The sensors are each
        inaccurate with probability epsilon. Here we represent the observation as the integer equivalent for the
        4-bit string. For example, if the robot's sensors report that there are walls above and to the left of the
        robot, the 4-bit observation would be 1001 and the corresponding integer observation would be 9.
        :return (np.array) [self.n_states, self.n_observations]: Observation probabilities
        '''

        # Update observational probabilities:
        # For eaach cell, get the correct sensor readings by going over the neighbours.
        # Then, get the discrepancy d between the correct and actual sensor readings.
        # Then, obs_probs[i,j] = (1-epsilon)**(4-d) * epsilon**d
        obs_probs = np.zeros((self.n_states, self.n_observations)) # track observational probabilities
        for cell_index in range(self.n_states): # get the correct sensor reading
            cell_neighbours =  self.cell_neighbours[cell_index]
            has_walls = [1] * 4 # track walls in order up, right, down, left
            for nb_index in cell_neighbours:
                nb = self.empty_cell_coords[nb_index]
                vertical_move = nb[0] - self.empty_cell_coords[cell_index][0] # -1 if up, 1 if down
                horizontal_move = nb[1] - self.empty_cell_coords[cell_index][1] # -1 if left, 1 if right
                if vertical_move == -1: # up
                    has_walls[0] = 0
                elif vertical_move == 1: # down
                    has_walls[2] = 0
                elif horizontal_move == -1: # left
                    has_walls[3] = 0   
                elif horizontal_move == 1: # right
                    has_walls[1] = 0
            correct_obs = (has_walls[0] * 2**3) + (has_walls[1] * 2**2) + (has_walls[2] * 2**1) + (has_walls[3] * 2**0)
            for actual_obs in range(self.n_observations):
                d = self.obs_discrepancy(correct_obs, actual_obs)
                obs_probs[cell_index,actual_obs] = (1-self.epsilon)**(4-d) * self.epsilon**d

        return obs_probs

    ######## DO NOT CHANGE ANY CODE AFTER THIS LINE #################################################################

    def validate_inputs(self, width: int, height: int, empty_cell_coords: List[List[int]], init_cell: int, epsilon: float):
        '''
        Raises error if the arguments provided to the constructor are inadmissible. The parameters are identical to
        the arguments of __init__().
        '''
        assert width > 0, "Grid width must be greater than 0."
        assert height > 0, "Grid height must be greater than 0."
        assert len(empty_cell_coords) > 0, "There must be at least 1 empty cell."
        assert len(empty_cell_coords) <= width * height, "There cannot be more empty cells than cells on the grid."
        assert all([0 <= c[0] < height and 0 <= c[1] < width for c in empty_cell_coords]), "Illegal cell coordinates detected."
        assert init_cell in range(len(empty_cell_coords)), "Initial cell does not exist in cell_coords"
        assert 0. <= epsilon <= 1., "Epsilon must be in [0, 1]"

    def determine_cell_neighbours(self):
        '''
        Determine the neighbouring empty cells for all empty cells on the grid.
        :return List[List[int]]: Adjacency list for cells in the grid. The i'th element is a list of the other empty
                                 cells that neighbour cell i
        '''
        cell_neighbours = [[] for _ in range(self.n_states)]
        for cell in range(self.n_states):
            for other_cell in range(self.n_states):
                if abs(self.empty_cell_coords[cell][0] - self.empty_cell_coords[other_cell][0]) \
                        + abs(self.empty_cell_coords[cell][1] - self.empty_cell_coords[other_cell][1]) == 1:
                    (cell_neighbours[cell]).append(other_cell)
        return cell_neighbours

    def determine_full_cells(self):
        '''
        Assemble a list of the cells that are inaccessible to the robot.
        :return full_cell_coords (List[List[int]]): List of the coordinates of full cells. The i'th element is a
                                  2-element list giving the (y, x) coordinate of the i'th empty cell. y=0 is the top
                                  and x=0 is the left.
        '''
        full_cell_coords = []
        for i in range(self.height):
            for j in range(self.width):
                if [i, j] not in self.empty_cell_coords:
                    full_cell_coords.append([i, j])
        return full_cell_coords

    def obs_discrepancy(self, obs1: int, obs2: int):
        '''
        Calculate the discrepancy between two observations.
        :param obs1: Observation #1
        :param obs2: Observation #2
        '''
        x = obs1 ^ obs2
        d = 0
        while (x > 0):
            d += x & 1
            x >>= 1
        return d

    def obs_to_binary_string(self, obs: int):
        '''
        Converts an observation from integer to 4-bit representation. The first, second, third and fourth bits indicate
        whether the sensor sensed an obstacle in the north, east, south and west directions respectively.
        :return (str): 4-bit sequence representation of the observation
        '''
        return "{0:04b}".format(obs)

    def draw_grid(self, loc_arr: np.ndarray, low_colour: str, high_colour: str):
        '''
        Draw a grid diagram, given values for each cell in the grid.
        :param loc_arr (np.array) [self.height, self.width]: Values to draw for each cell in the grid
        :param low_colour (str): Colour representing 0
        :param high_colour (str): Colour representing 1
        '''

        # Set up axes
        fig, ax = plt.subplots()
        ax.set_xlim(-0.75, self.width - 0.25)
        ax.set_ylim(-0.75, self.height - 0.25)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(self.width + 1) - 0.5)
        ax.set_yticks(np.arange(self.height + 1) - 0.5)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Draw grid cells
        ax.imshow(loc_arr, cmap=colors.LinearSegmentedColormap.from_list('custom', [low_colour, high_colour],
                                                                            N=256))

        # Add cell/state numbers
        for i in range(self.n_states):
            y, x = self.empty_cell_coords[i]
            ax.text(x, y, '{}'.format(i), ha='center', va='center')

        # Fill in inaccessible cells
        for coord in self.full_cell_coords:
            ax.add_patch(patches.Rectangle((coord[1]-0.5, coord[0]-0.5), 1, 1, facecolor='#71716c',
                                           fill=True, hatch='\\\\\\', edgecolor='black', linewidth=1.))

        # Draw a border
        ax.add_patch(patches.Rectangle((-0.75, -0.75), 0.25, self.height + 0.5, facecolor='#71716c',
                                       fill=True, hatch='///'))
        ax.add_patch(patches.Rectangle((-0.5, -0.75), self.width, 0.25, facecolor='#71716c',
                                       fill=True, hatch='///'))
        ax.add_patch(patches.Rectangle((self.width - 0.5, -0.75), 0.25, self.height + 0.5, facecolor='#71716c',
                                       fill=True, hatch='///'))
        ax.add_patch(patches.Rectangle((-0.5, self.height - 0.5), self.width, 0.25, facecolor='#71716c',
                                       fill=True, hatch='///'))

        # Add grid lines to the cells
        for i in range(self.width):
            for j in range(self.height):
                ax.add_patch(patches.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=1.))

        # Display the plot
        plt.show()


    def visualize_state(self):
        '''
        Draw a plot similar to Figure 14.7 in Russell & Norvig 4e, depicting the robot's current location.
        '''
        loc_arr = np.zeros((self.height, self.width))
        cur_coords = self.empty_cell_coords[self.cur_state]
        loc_arr[cur_coords[0], cur_coords[1]] = 1.
        self.draw_grid(loc_arr, 'lightgray', 'red')  # Draw the grid

    def visualize_belief(self, belief_probs: List[float]):
        '''
        Draw a plot similar to Figure 14.7 in Russell & Norvig 4e, depicting the probability distribution that
        reflects the robot's belief of its current state.
        belief_probs: A list of self.n_states probabilities. Should be a valid probability distribution.
        '''

        assert all([p >= 0 for p in belief_probs]) and sum(belief_probs) - 1 < 1e-4, \
            "belief_probs is not a valid probability distribution"

        # Create a 2D array representing probabilities for each empty grid cell
        loc_arr = np.zeros((self.height, self.width))
        for i in range(self.n_states):
            x, y = self.empty_cell_coords[i]
            loc_arr[x, y] = belief_probs[i]

        self.draw_grid(loc_arr, '#ffffff', '#006540')   # Draw beliefs on grid
