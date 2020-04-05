import gym
import time
import sys
import tensorflow as tf
import numpy as np
import itertools as it
from datetime import datetime
from tqdm import tqdm


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

    def __init__(self, method):
        # Main attributes of the simluation
        self.gamma = 0.99   # The discounting factor
        self.env = None   # The environement
        self.method = method   # The RL method used
        self.optimizer = tf.keras.optimizers.Adam(1e-2)   # The optimizer used for neural network training
        self.track_step = 50   # We will track the evolution of the training every track_step step

    def make_env(self, name_of_environment, verbose=False):
        # Here we set up the Open AI environment
        self.env = gym.make(name_of_environment)   # We make the env
        self.action_space_size = self.env.action_space.n   # We get the action space size
        self.state_space_shape = self.env.reset().shape   # We get the state space shape
        if self.method == 'PG':
            self.create_policy_gradient_networks(verbose)   # We initialize the neural network for policy gradient
        else:
            self.create_q_learning_networks(verbose)  # We initialize the neural network for q-learning

    @check_env
    def reset_env(self):
        # Here we reset the environment, and we return x0
        return self.env.reset().astype(np.float32)

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
                action_probs = self.predict(np.expand_dims(state, axis=0))   # We get the action probabilities outputed by the neural network
                action = np.random.choice(self.action_space_size, p=action_probs[0].numpy())   # We sample an action based on these probabilities
                obs, rew, done, _ = self.env.step(action)   # We take a step in the environment by taking the sampled actio
                rew_sum += rew   # We add the reward earned to the total reward
                state = obs.astype(np.float32)   # We get the next state of the evironment
            score += [rew_sum]   # Once the game is played, we store the total reward
        self.env.close()   # Once every game is played, we close the rendered envrionment
        return np.mean(score)   # We return the average score  over all games

    @check_env
    def create_policy_gradient_networks(self, verbose=False):
        # Here we create the Policy Gradient neural network
        # There are two inputs
        inputs = tf.keras.Input(shape=self.state_space_shape)  # The current state
        advantages = tf.keras.Input(shape=(1,))  # The future reward associated
        # One dense hidden layer, relu activated
        x = tf.keras.layers.Dense(16, activation='relu',
                                  use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal())(inputs)
        # One dense output layer, softmax activated (to get probabilities)
        outputs = tf.keras.layers.Dense(self.action_space_size, activation='softmax',
                                        use_bias=False,
                                        kernel_initializer=tf.keras.initializers.he_normal())(x)
        # Two different neural networks with these weights
        self.policy = tf.keras.Model(inputs=[inputs, advantages], outputs=outputs)  # One for training
        self.predict = tf.keras.Model(inputs=inputs, outputs=outputs)  # One for predicting the probabilities

        def custom_loss(y_true, y_pred):
            # Here we define a custom loss
            out = tf.keras.backend.clip(y_pred, 1e-8, 1 - 1e-8)  # We need to clip y_pred as it may be equal to zero (otherwise problem with log afterwards)
            log_lik = y_true * tf.keras.backend.log(out)  # We get the log likelyhood associated to the predictions
            return tf.keras.backend.mean(-log_lik * advantages, keepdims=True)  # We multiply it by the advantage (future reward here), and return the mean

        # We compile the neural network used for training with the given optimizer, and the loss function
        self.policy.compile(loss=custom_loss, optimizer=self.optimizer, experimental_run_tf_function=False)

    @check_env
    def update_policy_gradient_network(self, rewards, states, actions):
        # Here we process the future discounted sum of reward for each step of the simulation
        discounted_rewards = np.array(list(it.accumulate(rewards[::-1], lambda x, y: x * self.gamma + y))[::-1], dtype=np.float64)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        # We one hot encode the taken actions
        one_hot_encoded_actions = np.zeros((len(actions), self.action_space_size))
        one_hot_encoded_actions[np.arange(len(actions)), actions] = 1
        # And we train the neural network on this episode (which has given us a batch of actions, states and rewards)
        loss = self.policy.train_on_batch([np.stack(states, axis=0), discounted_rewards], one_hot_encoded_actions)
        # We return the associated loss
        return loss

    @check_env
    def train_policy_gradient(self, total_episodes=1000):
        # Here we train our policy gradient neural network
        total_rewards = []  # We initialize the total reward list
        for ep in range(total_episodes):
            state = self.reset_env()  # We get x0
            states = []  # We initialize the list of states
            rewards = []  # We initialize the list of rewards
            actions = []  # We initialize the list of actions
            done = False
            while not done:
                # While the game is not finished
                action_probs = self.predict(np.expand_dims(state, axis=0))  # We sample actions probabilities with the neural network
                action = np.random.choice(self.action_space_size, p=action_probs[0].numpy())  # We sample the action based on these probabilities
                next_state, rew, done, _ = self.env.step(action)  # We take a step forward int the environment by taking the sampled action
                # We store the action, state, and reward
                states += [state]
                rewards += [rew]
                actions += [action]
                # We update the state of the environment
                state = next_state.astype(np.float32)
            # We update the neural network with the data gathered during this episode
            loss = self.update_policy_gradient_network(rewards, states, actions)
            # We add the total reward of this episode to the list of total rewards
            total_rewards += [sum(rewards)]
            if (ep + 1) % self.track_step == 0:
                # Every track_step episodes, we track the progress
                mean_reward = np.mean(total_reward)   # Mean reward over the last track_step episodes
                total_reward = []   # We reset the list of total rewards
                score = self.test_intelligent(10)   # We simulate a few test games to track the evolution of the abilities of our agent
                print("Episode: %d,  Mean Training Reward: %.2f, Mean Test Score: %.2f, Loss: %.5f" % (ep + 1, mean_reward, score, loss))

    @check_env
    def create_q_learning_networks(self, verbose=False):
        # TO IMPLEMENT
        self.network = None
        if verbose:
            self.network.summary()
        pass


if __name__ == "__main__":
    # We create a Simulation object
    sim = Simulation(method='PG')
    # We make the environment
    sim.make_env("CartPole-v0")
    # We train the neural network
    sim.train_policy_gradient()
