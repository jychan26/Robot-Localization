# Robot-Localization
Implemented the forward backward algorithm to infer the robot’s location at some timestep, given a series of actions and observations. To do so, the robot’s interactions with the environment were modeled as a hidden Markov model (HMM). 

This task is inspired by the example in  “Artificial Intelligence: a Modern Approach” by Russell and
Norvig (section 14.3.2, page 477 in the 4th edition).

## Environment
A robot is in a grid environment and it has a correct map of the environment.

## Actions
The robot has four available actions: Up, Right, Down, and Left. Each action is intended
to move the robot by 1 cell in the corresponding direction. When it executes an action, it will move in the intended
direction with probability γ. The robot moves in one of the other directions with probability
(1 − γ)/3. If the robot is in a cell adjacent to a wall and it moves in the direction
of that wall, it remains in the same location.

## Sensors
The robot has four noisy sensors that report whether there is an inner or outer wall in each of the four
directions (up, right, left, down). At each time step, the robot receives a sensor reading, with each
bit giving the presence (bit 1) or absence (bit 0) of a wall in the up, right, down, and left
directions respectively.

Each sensor is noisy and flips the bit with a probability of 0 ≤ ϵ ≤ 1. The errors occur
independently for the four sensor directions.

The robot starts at an unknown location. At time step 0, we will assume a uniform distribution
over all the empty cells.
