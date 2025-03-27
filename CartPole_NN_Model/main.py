from collections import deque 

import numpy as np
import random
import keras
import gym

class CartPoleModel:
    """
    A class which represent the cart pole model of gym
    """
    # Parameters initialization:
    def __init__(self):
        self.max_reward = 500
        self.state_size = 2
        self.action_size = 4

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0

        self.memory:deque = deque()
        self.model = None
        
        # Creating the enviroment
        self.env = gym.make('CartPole-v1', render_mode=None)

    # The model:
    def _build_model(self):
        # Input: Nothing.
        # Output: The function creates a NN model.
        self.model = keras.Sequential()

        # Input & Hidden layers
        self.model.add(keras.layers.Dense(units=32, input_dim=self.state_size, activation="relu"))
        self.model.add(keras.layers.Dense(units=32, activation='relu'))
        
        # Output layer
        self.model.add(keras.layers.Dense(units=self.action_size, activation="linear"))
        
        # Compilation of the model
        self.model.compile(loss='mse', optimizer=keras.Adam(learning_rate=self.alpha))

    
    def _save_model(self):
        # Input: Nothing.
        # Output: The function saves the model to a .keras file
        file_name = "model.keras"
        self.model.save("fashion_mnist_model.h5")

    