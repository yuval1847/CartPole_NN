import numpy as np
import random
import keras
import gym

# Parameters initialization:
max_reward = 500
state_size = 2
action_size = 4

alpha = 0.7
gamma = 0.2
epsilon = 0.8

# The model: