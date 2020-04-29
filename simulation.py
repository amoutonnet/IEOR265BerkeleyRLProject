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
import pandas as pd

random.seed(100)
np.random.seed(100)
tf_random.set_seed(100)

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

    def test_intelligent(self, verbose=False):
        """
        This function is used to test the performances of a trained agent in the environment
        """
        state = self.reset_env()  # We get x0
        episode_reward = 0
        done = False
        while not done:
            action = self.agent.predict_action(state, current_eps=0)  # we sample the action
            obs, reward, done, _ = self.env.step(action)  # We take a step forward in the environment by taking the sampled action
            episode_reward += reward
            state = self.get_next_state(state, obs)
        return episode_reward

    def train(self, nb_computations=1, max_episodes=1000, process_average_over=100, save_training_data=False, plot_evolution=False):
        if self.agent is None:
            raise Exception('You need to set an actor before training it !')
        for computation in range(nb_computations):
            print('\n%s\n' % (('Training Computation no. %d' % (computation + 1)).center(100, '-')))
            # Initialize neural network(s)
            self.agent.init_agent_for_training()
            # Here we train our neural network with the given method
            training_score = np.empty((max_episodes,))
            testing_score = np.empty((max_episodes,))
            if process_average_over > 0:
                rolling_ave_training = np.empty((max_episodes,))
                rolling_ave_testing = np.empty((max_episodes,))
            timestamps = np.empty((max_episodes))
            ep = 0
            while ep < max_episodes:
                start_time = time.time()
                state = self.reset_env()  # We get x0
                episode_reward = 0
                done = False
                # While the game is not finished
                while not done:
                    action = self.agent.predict_action(state)  # we sample the action
                    obs, reward, done, _ = self.env.step(action)  # We take a step forward in the environment by taking the sampled action
                    episode_reward += reward
                    next_state = self.get_next_state(state, obs)
                    self.agent.remember(state, action, reward, next_state, done)
                    self.agent.learn_off_policy()
                    state = next_state
                self.agent.learn_on_policy()
                timestamps[ep] = time.time() - start_time
                test_episode_reward = self.test_intelligent()
                training_score[ep] = episode_reward
                testing_score[ep] = test_episode_reward
                if process_average_over > 0:
                    rolling_ave_training[ep] = np.mean(training_score[max(0, ep - process_average_over):ep + 1])
                    rolling_ave_testing[ep] = np.mean(testing_score[max(0, ep - process_average_over):ep + 1])
                    self.agent.print_verbose(ep + 1, max_episodes, episode_reward, test_episode_reward, rolling_ave_training[ep], rolling_ave_testing[ep])
                else:
                    self.agent.print_verbose(ep + 1, max_episodes, episode_reward, test_episode_reward)
                ep += 1
            timestamps = np.cumsum(timestamps)
            if save_training_data:
                self.save_training_data(computation, max_episodes, training_score, testing_score, timestamps)
                print('\n%s\n' % (('Training Computation no. %d Saved' % (computation + 1)).center(100, '-')))
            if plot_evolution:
                fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
                ax1.plot(training_score, 'tab:blue', linewidth=1, label='Train Score')
                ax1.plot(testing_score, 'tab:olive', linewidth=1, label='Test Score')
                ax1.set(xlabel='Episodes', ylabel='Score')
                ax2.plot(timestamps, training_score, 'tab:blue', linewidth=1)
                ax2.plot(timestamps, testing_score, 'tab:olive', linewidth=1)
                ax2.set(xlabel='Time (s)')
                if process_average_over > 0:
                    rolling_ave_training = rolling_ave_training[:ep]
                    rolling_ave_testing = rolling_ave_testing[:ep]
                    ax1.plot(rolling_ave_training, 'orange', linewidth=1, label='Train Rolling Average')
                    ax1.plot(rolling_ave_testing, 'red', linewidth=1, label='Test Rolling Average')
                    ax2.plot(timestamps, rolling_ave_training, 'orange', linewidth=1)
                    ax2.plot(timestamps, rolling_ave_testing, 'red', linewidth=1)
                fig.suptitle('Evolution of the score during the Training')
                fig.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 0.955), prop={'size': 9})
                plt.show()

    def save_training_data(self, computation, max_episodes, training_score, testing_score, timestamps):
        """
        Saving training data to csv file
        """
        agent_type = self.agent.__class__.__name__
        if agent_type == 'AgentPG':
            archive_name = '{}_comp{}_maxep{}_entropy{}_ppo{}_lambd{}'.format(
                agent_type,
                computation + 1,
                max_episodes,
                self.agent.temperature,
                self.agent.epsilon,
                self.agent.lambd
            )
        else:
            archive_name = '{}_comp{}_maxep{}_update{}_double{}_dueling{}_per{}_epssteps{}_rms{}'.format(
                agent_type,
                computation + 1,
                max_episodes,
                self.agent.update_target_every,
                self.agent.use_double,
                self.agent.use_dueling,
                self.agent.use_per,
                len(self.agent.epsilons),
                self.agent.max_memory_size
            )
            # self.agent.q_network.save_weights('Weights/final_weights' + archive_name + ".h5")
        columns = ['timestamps', 'training_score', 'testing_score']
        data = pd.DataFrame(list(zip(timestamps, training_score, testing_score)), columns=columns)
        data.to_csv('Results/' + archive_name + ".csv", sep=';', index=False)


if __name__ == "__main__":
    # We create a Simulation object
    sim = Simulation(name_of_environment="CartPole-v0", nb_stacked_frame=1)
    # We create an Agent to evolve in the simulation
    method = 'DQN'
    if method == 'PG':
        agent = agent_pg.AgentPG(
            sim.state_space_shape,              # The size of the spate space
            sim.action_space_size,              # The size of the action space
            gamma=0.99,                         # The discounting factor
            hidden_conv_layers=[                # A list of parameters of for each hidden convolutionnal layer
            ],
            hidden_dense_layers=[               # A list of parameters of for each hidden dense layer
                32
            ],
            initializer='random_normal',        # Initializer to use for weights
            verbose=True,                       # A live status of the training
            lr_actor=1e-2,                      # Learning rate
            lr_critic=1e-2,                     # Learning rate for A2C critic part
            lambd=1,                            # General Advantage Estimate term, 1 for full discounted reward, 0 for TD residuals
            epochs=1,                           # The number of time we train actor and critic on the batch of data obtained during an episode
            entropy_dict={
                'used': True,                   # Whether or not Entropy Regulaarization is used
                'temperature': 1e-3             # Temperature parameter for entropy reg
            },
            ppo_dict={
                'used': True,                   # Whether or not Proximal policy optimization is used
                'epsilon': 0.2                  # Epsilon for PPO
            }
        )
    else:
        agent = agent_dql.AgentDQL(
            sim.state_space_shape,              # The shape of the state space
            sim.action_space_size,              # The size of the action space
            gamma=0.99,                         # The discounting factor
            hidden_conv_layers=[],              # A list of parameters of for each hidden convolutionnal layer [filters, kernel_size, strides]
            hidden_dense_layers=[32],           # A list of parameters of for each hidden dense layer
            initializer='he_normal',            # Initializer to use for weights
            verbose=True,                       # A live status of the training
            lr=1e-3,                            # The learning rate
            max_memory_size=20000,              # The maximum size of the replay memory
            epsilon_behavior=(1, 0.1, 1000),    # The decay followed by epsilon
            batch_size=32,                      # The batch size used during the training
            update_target_every=200,            # Update the TD targets q-values every update_target_every optimization steps
            double_dict={
                'used': True                    # Whether we use double q learning or not
            },
            dueling_dict={
                'used': False,                  # Whether we use dueling q learning or not
            },
            per_dict={
                'used': False,                   # Whether we use prioritized experience replay or not
                'alpha': 0.6,                   # Prioritization intensity
                'beta': 0.4,                    # Initial parameter for Importance Sampling
                'beta_increment': 0.002,        # Increment per sampling for Importance Sampling
                'epsilon': 0.01                 # Value assigned to have non-zero probabilities
            }
        )
    # We build the neural network
    # agent.build_network()
    # We set this agent in the simulation
    sim.set_agent(agent)
    # We train the agent for a given number of computations and episodes
    sim.train(nb_computations=10, max_episodes=200, process_average_over=100, save_training_data=True, plot_evolution=False)
