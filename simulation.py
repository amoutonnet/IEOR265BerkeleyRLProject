import gym
import time
import sys
import tensorflow as tf
import numpy as np
import itertools as it
from datetime import datetime
from tqdm import tqdm
from collections import deque
from collections import namedtuple
import random


tf.random.set_seed(1)


def check_env(f):
    # A decorator to check if the environment is defined in the Simulation class
    def wrapper(*args, **kwargs):
        if args[0].env is not None:
            return f(*args, **kwargs)
        else:
            print('You need to make an environment!')
    return wrapper


class Simulation():

    def __init__(self, main_method, variation, parameters, gamma=0.99, alpha=1e-2, test_every=50, hidden_conv_layers=[], hidden_dense_layers=[32], nb_stacked_frame=1):
        assert(nb_stacked_frame >= 1)
        # Main attributes of the simluation
        self.gamma = gamma   # discounting factor
        self.env = None   # environement
        self.main_method = main_method   # RL main_method used
        self.variation = variation  # The variation of the method e.g. DDQ for Double DQN
        self.lr = alpha  # The learning rate for the optimizer
        self.hidden_dense_layers = hidden_dense_layers   # A list of number of neurons for hidden dense layers
        self.hidden_conv_layers = hidden_conv_layers   # A list of number of neurons for hidden convolutionnal layers
        self.optimizer = tf.keras.optimizers.Adam(self.lr)   # Optimizer used for neural network training
        self.test_every = test_every   # We will track the evolution of the training every test_every step
        self.nb_stacked_frame = nb_stacked_frame   # The number of frame we want to stack
        self.set_parameters(parameters)   # We set all the attributes proper to the method

    def set_parameters(self, parameters):
        # We set all the attributes proper to the method
        if self.main_method == 'PGN':
            self.memory = deque()  # The memory used to track rewards, states and actions during an episode
        else:
            self.epsilons = np.linspace(parameters['eps_start'], parameters['eps_end'], parameters['eps_decay_steps'])   # epsilon greedy strategy (start, stop, decay)
            self.replay_memory_size = parameters['replay_memory_size']   # experience in memory max size for experience replay
            self.update_target_estimator_every = parameters['update_target_estimator_every']   # update the TD targets q-values every 50 optimization steps
            self.batch_size = parameters['batch_size']   # Batch size for learning
            self.memory = deque(maxlen=self.replay_memory_size)    # Dequeue to serve as replay memory
            self.opti_step = 0  # The number of optimization step done

    def make_env(self, name_of_environment, verbose=False):
        # Here we set up the Open AI environment
        self.env = gym.make(name_of_environment)   # We make the env
        self.action_space_size = self.env.action_space.n   # We get the action space size
        self.state_space_shape = self.reset_env().shape   # We get the state space shape
        if self.main_method == 'PGN':
            self.create_policy_gradient_networks(verbose)   # We initialize the neural network for policy gradient
        else:
            self.create_Q_learning_networks(verbose)  # We initialize the neural network for q-learning

    @check_env
    def reset_env(self):
        return np.squeeze(np.stack([self.env.reset()] * self.nb_stacked_frame, axis=1).astype(np.float32))

    @check_env
    def test_random(self, verbose=False):
        # Here we simulate a random evolution of the environment, with random actions taken by the agent
        done = False
        while not done:
            self.env.render()
            obs, rew, done, info = self.env.step(self.env.action_space.sample())
            if verbose:
                print(obs, rew, info)
            if done:
                self.env.render()
                break
        time.sleep(2)
        self.env.close()

    @check_env
    def test_intelligent(self, num_tests):
        # Here we simulate an intelligent evolution of the environment, with intelligent actions taken by the agent
        score = []   # Over the num_tests games, we track the score of the agent at each game
        for _ in range(num_tests):
            done = False
            state = self.reset_env()  # We get x0
            rew_sum = 0   # We initialize the total reward of that game to 0
            # We play a game
            while not done:
                self.env.render()   # We render the environment to vizualize it
                action = self.get_action(state)
                obs, rew, done, _ = self.env.step(action)   # We take a step in the environment by taking the sampled action
                rew_sum += rew   # We add the reward earned to the total reward
                state = self.get_next_state(state, obs)   # We get the next state of the evironment
            score += [rew_sum]   # Once the game is played, we store the total reward
        self.env.close()   # Once every game is played, we close the rendered envrionment
        return np.mean(score)   # We return the average score  over all games

    @check_env
    def create_policy_gradient_networks(self, verbose=False):
        # Here we create the Policy Gradient neural network
        # There are two inputs
        states = tf.keras.Input(shape=self.state_space_shape)  # The current state
        advantages = tf.keras.Input(shape=(1,))  # The future reward associated
        x = states
        if self.nb_stacked_frame > 1:
            # Hidden Conv layers, relu activated
            for c in self.hidden_conv_layers:
                x = tf.keras.layers.Conv1D(filters=c[0], kernel_size=c[1], activation='relu',
                                           kernel_initializer=tf.keras.initializers.he_normal())(x)
            # We flatten before dense layers
            x = tf.keras.layers.Flatten()(x)
        # Hidden Dense layers, relu activated
        for h in self.hidden_dense_layers:
            x = tf.keras.layers.Dense(h, activation='relu',
                                      kernel_initializer=tf.keras.initializers.he_normal())(x)
        # One dense output layer, softmax activated (to get probabilities)
        outputs = tf.keras.layers.Dense(self.action_space_size, activation='softmax',
                                        kernel_initializer=tf.keras.initializers.he_normal())(x)
        # Two different neural networks with these weights
        self.policy = tf.keras.Model(inputs=[states, advantages], outputs=outputs)  # One for training
        self.predict = tf.keras.Model(inputs=states, outputs=outputs)  # One for predicting the probabilities

        def custom_loss(y_true, y_pred):
            # Here we define a custom loss
            out = tf.keras.backend.clip(y_pred, 1e-8, 1 - 1e-8)  # We need to clip y_pred as it may be equal to zero (otherwise problem with log afterwards)
            log_lik = y_true * tf.keras.backend.log(out)  # We get the log likelyhood associated to the predictions
            return tf.keras.backend.mean(-log_lik * advantages, keepdims=True)  # We multiply it by the advantage (future reward here), and return the mean

        # We compile the neural network used for training with the given optimizer, and the loss function
        self.policy.compile(loss=custom_loss, optimizer=self.optimizer, experimental_run_tf_function=False)

    @check_env
    def create_Q_learning_networks(self, verbose=False):
        # Input last frame of the game
        states = tf.keras.Input(shape=self.state_space_shape)
        x = states
        if self.nb_stacked_frame > 1:
            # Hidden Conv layers, relu activated
            for c in self.hidden_conv_layers:
                x = tf.keras.layers.Conv1D(filters=c[0], kernel_size=c[1], activation='relu',
                                           kernel_initializer=tf.keras.initializers.he_normal())(x)
            # We flatten before dense layers
            x = tf.keras.layers.Flatten()(x)
        # Hidden Dense layers, relu activated
        for h in self.hidden_dense_layers:
            x = tf.keras.layers.Dense(h, activation='relu',
                                      kernel_initializer=tf.keras.initializers.he_normal())(x)
        outputs = tf.keras.layers.Dense(self.action_space_size, activation='linear',
                                        kernel_initializer=tf.keras.initializers.he_normal())(x)

        # Network to be updated
        self.q_network = tf.keras.Model(inputs=states, outputs=outputs)
        if self.variation == 'DDQN':
            # Copy of the network for target calculation, updated every self.update_target_estimator_every
            self.target_model = tf.keras.models.clone_model(self.q_network)

    @check_env
    def update_policy_gradient_network(self):
        states, actions, rewards, _, _ = map(np.array, zip(*self.memory))
        # Here we process the future discounted sum of reward for each step of the simulation
        discounted_rewards = np.array(list(it.accumulate(rewards[::-1], lambda x, y: x * self.gamma + y))[::-1], dtype=np.float64)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        # We one hot encode the taken actions
        one_hot_encoded_actions = np.zeros((len(actions), self.action_space_size))
        one_hot_encoded_actions[np.arange(len(actions)), actions.astype(int)] = 1
        # And we train the neural network on this episode (which has given us a batch of actions, states and rewards)
        loss = self.policy.train_on_batch([np.stack(states, axis=0), discounted_rewards], one_hot_encoded_actions)
        # We return the associated loss
        return loss

    @check_env
    def update_q_network(self):
        """Input current experience replay memory with length > self.batch_size
        Sample minibatch
        Compute max expected Q-values and best actions
        Compute TD-targets
        Train on minibatch and output resulting loss
        """
        # minibatch selection
        # returns a list of samples (state, action, reward, next_state, done)
        mini_batch = random.sample(self.memory, self.batch_size)
        # unzip minibatch an returns a list of each elements in np.array types
        states_batch, action_batch, rewards_batch, next_states_batch, done_batch = map(np.array, zip(*mini_batch))

        if self.variation == 'DDQN':
            q_target_next_values = self.target_model(next_states_batch).numpy()
            value_next = np.max(q_target_next_values, axis=1)
        else:
            q_next_values = self.q_network(next_states_batch).numpy()
            value_next = np.amax(q_next_values, axis=1)

        actual_values = np.where(done_batch, rewards_batch, rewards_batch + self.gamma * value_next)
        # gradient descent update
        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.q_network(states_batch) * tf.one_hot(action_batch, self.action_space_size), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.q_network.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    @check_env
    def get_next_state(self, current_state, new_obs):
        # We process the next state based on the current one and the new observation, and return it
        if self.nb_stacked_frame == 1:
            return new_obs.astype(np.float32)
        else:
            return np.append(current_state[:, 1:], np.expand_dims(new_obs, 1), axis=1).astype(np.float32)

    @check_env
    def get_action(self, state, train=False):
        # We get the action to do based on the state
        state = np.expand_dims(state, axis=0)
        if main_method == 'PGN':
            action_probs = self.predict(state)  # We sample actions probabilities with the neural network
            action = np.random.choice(self.action_space_size, p=action_probs[0].numpy())  # We sample the action based on these probabilities
        else:
            if train and np.random.rand() < self.epsilons[min(self.opti_step, len(self.epsilons) - 1)]:  # proba epsilon of making a random action choice
                action = self.env.action_space.sample()  # Random action
            else:
                action = np.argmax(self.q_network(state)[0])  # otherwise, choose the action with the best predicted reward starting next step
        return action

    @check_env
    def train(self, total_episodes=1000):
        # Here we train our neural network with the given method
        total_rewards = []  # We initialize the total reward list
        for ep in range(total_episodes):
            state = self.reset_env()  # We get x0
            episode_reward = 0
            done = False
            # While the game is not finished
            while not done:
                action = self.get_action(state, train=True)  # we sample the action
                obs, reward, done, _ = self.env.step(action)  # We take a step forward int the environment by taking the sampled action
                episode_reward += reward
                next_state = self.get_next_state(state, obs)
                self.memory.append((state, action, reward, next_state, done))

                if self.main_method == 'PGN':
                    pass
                else:
                    if self.variation == 'DDQN':
                        # Update target network if we reached the update step
                        if (self.opti_step + 1) % self.update_target_estimator_every == 0:
                            self.target_model.set_weights(self.q_network.get_weights())
                    # Train on minibatch and update q network
                    if len(self.memory) > self.batch_size:
                        loss = self.update_q_network()
                        self.opti_step += 1

                state = next_state

            if self.main_method == 'PGN':
                # We update the neural network with the data gathered during this episode
                loss = self.update_policy_gradient_network()
                self.memory.clear()

            total_rewards += [episode_reward]
            if (ep + 1) % self.test_every == 0:
                # Every test_every episodes, we track the progress
                mean_reward = np.mean(total_rewards)   # Mean reward over the last test_every episodes
                total_rewards = []   # We reset the list of total rewards
                score = self.test_intelligent(10)   # We simulate a few test games to track the evolution of the abilities of our agent
                print("Episode: %d,  Mean Training Reward: %.2f, Mean Test Score: %.2f, Loss: %.5f" % (ep + 1, mean_reward, score, loss))


if __name__ == "__main__":
    main_method = 'DQN'
    # main_method = 'PGN'
    variation = 'DDQN'  # Set equal to None to have original version of the method
    # variation = None
    parameters_dqn = {
        'eps_start': 1.0,
        'eps_end': 0.1,
        'eps_decay_steps': 100,
        'replay_memory_size': 2000,
        'update_target_estimator_every': 50,
        'batch_size': 32,
    }
    parameters_pgn = {

    }
    parameters = parameters_pgn if main_method == 'PGN' else parameters_dqn
    # We create a Simulation object
    sim = Simulation(main_method=main_method, variation=variation, parameters=parameters, hidden_dense_layers=[32], test_every=20)
    # We make the environment
    sim.make_env("CartPole-v0")
    # We train the neural network
    sim.train(total_episodes=200)
