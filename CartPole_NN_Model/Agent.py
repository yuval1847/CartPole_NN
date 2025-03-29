from collections import deque
import numpy as np
import random
import keras


class Agent:
    """
    A class which represent the cart pole model of gym
    """

    def __init__(self, state_size, action_size):
        #  TODO: create the agent's fields
        self.max_reward = 500
        self.state_size = state_size
        self.action_size = action_size

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0

        self.memory:deque = deque()
        self.model = None

    def _build_model(self):
        # TODO: create the agent's model
        self.model = keras.Sequential()

        # Input & Hidden layers
        self.model.add(keras.layers.Dense(units=32, input_dim=self.state_size, activation="relu"))
        self.model.add(keras.layers.Dense(units=32, activation='relu'))
        
        # Output layer
        self.model.add(keras.layers.Dense(units=self.action_size, activation="linear"))
        
        # Compilation of the model
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.alpha))

    def _save_model(self, model_name):
        # TODO: create save function
        self.model.save(model_name)

    def _load_model(self, model_name):
        # TODO: create load function
        return keras.models.load_model(model_name)

    def _update_memory(self, state, action, reward, next_state, done):
        # TODO: create update memory function
        self.memory.append(tuple(state, action, reward, next_state, done))

    def _act(self, state):
        # TODO: create act function
        exp_exp_random = random.uniform(0, 1)

        if exp_exp_random > self.epsilon:
            # Exploite
            state = self._reshape_state(state)
            values = self.model.predict(state, verbose=0)
            return np.argmax(values[0])
        else:
            # Explore
            return random.randint(0, self.action_size - 1)

    def _learn(self, batch_size):
        """
        This function implements the learning and updates the agent's weights and
        parameters according to past experience.
        """
        # If the memory is not big enough
        if len(self.memory) < batch_size:
            return
        # Sample from the memory
        sample_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in sample_batch:
            est_reward = reward
            if not done:
                # learning role:
                next_state = self._reshape_state(next_state)
                prediction = self.model.predict(next_state)
                predicted_reward = np.amax(prediction[0])
                est_reward = reward + self.gamma * predicted_reward
            state = self._reshape_state(state)
            curr_value = self.model.predict(state)
            curr_value[0][action] = est_reward
            # Update the model
            self.model.fit(state, curr_value, epochs=1, verbose=0)
            self._update_epsilon()

    def _update_epsilon(self):
        """
        Minimize epsilon as the learning progress
        """
        epsilon_min = 0.01
        epsilon_decay = 0.995

        if self.epsilon > epsilon_min:
            self.epsilon = self.epsilon * epsilon_decay
    
    @staticmethod
    def _reshape_state(state):
        """
        Reshapes the state so we can feed it to the model
        """
        return state.reshape(1, -1)

    def train(self, env, n_train_episodes, batch_size, model_name):
        # TODO: create train function
        self._build_model()
        for i in range(n_train_episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self._act(state)
                print(action)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                self._update_memory(state, action, reward, next_state, done)
                self._learn(batch_size)
                state = next_state

            if total_reward >= self.max_reward:
                break

        self._save_model(model_name)

    def test(self, env, n_test_episodes, model_name):
        # TODO: create test function
        self.model = self._load_model(model_name)
        for i in range(n_test_episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                state = self._reshape_state(state)
                values = self.model.predict(state, verbose=0)
                action = np.argmax(values[0])
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state