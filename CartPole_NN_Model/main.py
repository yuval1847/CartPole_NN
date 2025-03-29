import gym
from Agent import *

# Configuration parameters
batch_size = 32
n_train_episodes = 2000
n_test_episodes = 2000

# TODO: Choose name
model_name = "model.keras"

# Create environment
env = gym.make('CartPole-v1', rneder_mode=None)

# Extract parameters
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Create the agent
agent = Agent(state_size, action_size)