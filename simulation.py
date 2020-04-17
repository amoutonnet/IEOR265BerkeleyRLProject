import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import time
import sys
import random
import numpy as np
from tqdm import tqdm
from collections import deque
from utils import agent_dql
from utils import agent_pg
import matplotlib.pyplot as plt
import tensorflow.keras.initializers as initializers
from tensorflow import random as tf_random
from time import time
import pandas as pd

random.seed(1)
np.random.seed(1)
tf_random.set_seed()


class Simulation():

    def __init__(self, name_of_environment='CartPole-v0', nb_stacked_frame=1):
        assert(nb_stacked_frame >= 1)
        # Main attributes of the Simulation
        self.env = gym.make(name_of_environment)   # We make the env
        self.nb_stacked_frame = nb_stacked_frame   # The number of frame observed we want to stack for the available state
        self.state_space_shape = self.reset_env().shape
        self.action_space_size = self.env.action_space.n
        self.agent = None

    def set_agent(self, agent):
        self.agent = agent

    def reset_env(self):
        """
        This function is used to get x0
        """
        return np.squeeze(np.stack([self.env.reset()] * self.nb_stacked_frame, axis=1).astype(np.float32))

    def get_next_state(self, current_state, new_obs):
        """
        This function is used to process x_k+1 based on the current observation of the environment we
        have just after taking the action. Useful mainly when we stack frames.
        """
        if self.nb_stacked_frame == 1:
            return new_obs.astype(np.float32)
        else:
            return np.append(current_state[:, 1:], np.expand_dims(new_obs, 1), axis=1).astype(np.float32)

    def test_random(self, verbose=False):
        """
        This function is used to simulate a random evolution of the environment, with random actions taken by the agent
        """
        self.env.reset()
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

    def train(self, target_score, max_episodes=1000, process_average_over=100, test_every=50, test_on=0, save_training_data=False):
        if self.agent is None:
            raise Exception('You need to set an actor before training it !')
        print('\n%s\n' % ('Training'.center(100, '-')))
        # Here we train our neural network with the given method
        training_score = np.empty((max_episodes,))
        training_rolling_average = np.empty((max_episodes,))
        total_rewards = deque(maxlen=process_average_over)  # We initialize the total reward list
        rolling_mean_score = 0
        timestamps = np.empty((max_episodes))
        start_date = time()
        ep = 0
        while rolling_mean_score < target_score and ep < max_episodes:
            state = self.reset_env()  # We get x0
            episode_reward = 0
            done = False
            visualize = ep > test_on and (ep + 1) % test_every < test_on
            # While the game is not finished
            while not done:
                action = self.agent.predict_action(state)  # we sample the action
                obs, reward, done, _ = self.env.step(action)  # We take a step forward in the environment by taking the sampled action
                if visualize:
                    # If its test time, we vizualize the environment
                    self.env.render()
                episode_reward += reward
                next_state = self.get_next_state(state, obs)
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.learn_off_policy()
                state = next_state
            self.agent.learn_on_policy()
            total_rewards.append(episode_reward)
            rolling_mean_score = np.mean(total_rewards)
            training_score[ep] = episode_reward
            training_rolling_average[ep] = rolling_mean_score
            timestamps[ep] = time() - start_date
            self.agent.print_verbose(ep + 1, max_episodes, episode_reward, rolling_mean_score)
            self.env.close()
            ep += 1
        print('\n\n%s\n' % ('Training Done'.center(100, '-')))
        training_score = training_score[:ep]
        training_rolling_average = training_rolling_average[:ep]
        timestamps = timestamps[:ep]
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(training_score, 'b', linewidth=1, label='Score')
        ax1.plot(training_rolling_average, 'orange', linewidth=1, label='Rolling Average')
        ax1.plot([target_score] * ep, 'r', linewidth=1, label='Target Score')
        ax1.set(xlabel='Episodes', ylabel='Score')
        ax2.plot(timestamps, training_score, 'b', linewidth=1)
        ax2.plot(timestamps, training_rolling_average, 'orange', linewidth=1)
        ax2.plot(timestamps, [target_score] * ep, 'r', linewidth=1)
        ax2.set(xlabel='Time (s)', ylabel='Score')
        fig.suptitle('Evolution of the score during the Training {}'.format(target_score))
        fig.legend()
        plt.show()
        if save_training_data:
            self.save_training_data(target_score, training_score, training_rolling_average, timestamps)

    def save_training_data(self, target_score, training_score, training_rolling_average, timestamps):
        """
        Saving training data to csv file
        """
        agent_type = self.agent.__class__.__name__
        if agent_type == 'AgentPG':
            archive_name = '{} target {} entropy {} ppo {} lambd {} .csv'.format(
                agent_type,
                target_score,
                self.agent.temperature,
                self.agent.epsilon,
                self.agent.lambd
            )
        else:
            archive_name = '{} target {} double {} dueling {} per {} epssteps {} memorysize {} .csv'.format(
                agent_type,
                target_score,
                self.agent.update_target_every,
                self.agent.usedueling,
                self.agent.useper,
                len(self.agent.epsilons),
                self.agent.max_memroy_size
            )
        data = np.stack((timestamps, training_score, training_rolling_average)).reshape(-1, 3)
        columns = ['timestamps', 'training_score', 'training_rolling_average']
        pd.DataFrame(data=data, columns=columns).to_csv(archive_name, sep=';')


if __name__ == "__main__":
    # We create a Simulation object
    sim = Simulation(name_of_environment="CartPole-v0", nb_stacked_frame=1)
    # We create an Agent to evolve in the simulation
    method = 'PG'
    if method == 'PG':
        agent = agent_pg.AgentPG(
            sim.state_space_shape,              # The size of the spate space
            sim.action_space_size,              # The size of the action space
            gamma=0.99,                         # The discounting factor
            hidden_conv_layers=[],              # A list of parameters of for each hidden convolutionnal layer
            hidden_dense_layers=[32],           # A list of parameters of for each hidden dense layer
            initializer='random_normal',        # Initializer to use for weights
            verbose=True,                       # A live status of the training
            lr_actor=1e-2,                      # Learning rate
            lr_critic=1e-2,                     # Learning rate for A2C critic part
            lambd=0.5,                          # General Advantage Estimate term, 1 for full discounted reward, 0 for TD residuals
            entropy_dict={
                'used': False,                   # Whether or not Entropy Regulaarization is used
                'temperature': 1e-3             # Temperature parameter for entropy reg
            },
            ppo_dict={
                'used': False,                  # Whether or not Proximal policy optimization is used
                'epsilon': 0.2                  # Epsilon for PPO
            }
        )
    else:
        agent = agent_dql.AgentDQL(
            sim.state_space_shape,              # The shape of the state space
            sim.action_space_size,              # The size of the action space
            gamma=0.99,                         # The discounting factor
            hidden_conv_layers=[],              # A list of parameters of for each hidden convolutionnal layer
            hidden_dense_layers=[32],           # A list of parameters of for each hidden dense layer
            initializer='random_normal',        # Initializer to use for weights
            verbose=True,                       # A live status of the training
            lr=1e-2,                            # The learning rate
            max_memory_size=40000,              # The maximum size of the replay memory
            epsilon_behavior=(1, 0.1, 5000),    # The decay followed by epsilon
            batch_size=64,                      # The batch size used during the training
            double_dict={
                'used': True,                   # Whether we use double q learning or not
                'update_target_every': 200      # Update the TD targets q-values every update_target_every optimization steps
            },
            dueling_dict={
                'used': False,                  # Whether we use dueling q learning or not
            },
            per_dict={
                'used': True,                   # Whether we use prioritized experience replay or not
                'alpha': 0.6,                   # Prioritization intensity
                'beta': 0.4,                    # Initial parameter for Importance Sampling
                'beta_increment': 0.0001,        # Increment per sampling for Importance Sampling
                'epsilon': 0.01                # Value assigned to have non-zero probailities
            }
        )
    # We build the neural network
    agent.build_network()
    # We set this agent in the simulation
    sim.set_agent(agent)
    # We train the agent
    sim.train(target_score=150, max_episodes=100, process_average_over=100, test_every=50, test_on=0, save_training_data=True)
