import gym
import time
import sys
import numpy as np
from tqdm import tqdm
from collections import deque
import random
import learning
import matplotlib.pyplot as plt


class Simulation():

    def __init__(self, name_of_environment='CartPole-v0', nb_stacked_frame=1, agent_params={}):
        assert(nb_stacked_frame >= 1)
        # Main attributes of the Simulation
        self.env = gym.make(name_of_environment)   # We make the env
        self.nb_stacked_frame = nb_stacked_frame   # The number of frame observed we want to stack for the available state
        self.agent = learning.Agent(self.reset_env().shape, self.env.action_space.n, **agent_params)  # We create an intelligent agent

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

    def train(self, target_score, max_episodes=1000, process_average_over=100, test_every=50, test_on=0):
        print('\n%s\n' % ('Training'.center(100, '-')))
        # Here we train our neural network with the given method
        training_score = np.empty((max_episodes,))
        training_rolling_average = np.empty((max_episodes,))
        total_rewards = deque(maxlen=process_average_over)  # We initialize the total reward list
        rolling_mean_score = 0
        ep = 0
        while rolling_mean_score < target_score and ep < max_episodes:
            state = self.reset_env()  # We get x0
            episode_reward = 0
            done = False
            visualize = (ep + 1) % test_every < test_on
            # While the game is not finished
            while not done:
                action = self.agent.take_action(state, train=True)  # we sample the action
                obs, reward, done, _ = self.env.step(action)  # We take a step forward in the environment by taking the sampled action
                if visualize:
                    # If vizualise is greater than 0, we vizualize the environment
                    self.env.render()
                episode_reward += reward
                next_state = self.get_next_state(state, obs)
                self.agent.learn_during_ep(state, action, reward, next_state, done)
                state = next_state
            self.agent.learn_end_ep()
            total_rewards.append(episode_reward)
            rolling_mean_score = np.mean(total_rewards)
            training_score[ep] = episode_reward
            training_rolling_average[ep] = rolling_mean_score
            self.agent.print_verbose(ep, max_episodes, episode_reward, rolling_mean_score)
            self.env.close()
            ep += 1
        print('\n%s\n' % ('Training Done'.center(100, '-')))
        training_score = training_score[:ep]
        training_rolling_average = training_rolling_average[:ep]
        plt.figure()
        plt.plot(training_score, 'b', linewidth=1, label='Score')
        plt.plot(training_rolling_average, 'orange', linewidth=1, label='Rolling Average')
        plt.plot([target_score] * ep, 'r', linewidth=1, label='Target Score')
        plt.title('Evolution of the score during the Training {}'.format(target_score))
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    method = 'DQN'                # 'PGN' for policy gradient, 'DQN' Deep Q-Learning
    variation = ['DoubleDQN', 'PER']            # Set to None for original method, otherwise 'A2C'/'PPO' for 'PGN, 'DoubleDQN'/'DuelingDDQN' for 'DQN'
    parameters_dqn = {
        'eps_start': 1.0,
        'eps_end': 0.1,
        'eps_decay_steps': 5000,
        'replay_memory_size': 50000,
        'update_target_estimator_every': 100,
        'batch_size': 64,
        'alpha_PER': 0.6,  # from paper, for Double DQN
        'beta_PER': 0.4,  # from paper, for Double DQN
        'beta_increment_PER': 0.0001,
        'epsilon_PER': 0.001
    }
    parameters_pgn = {
        'temperature': 0.001,
        'epsilon_ppo': 0.2,
    }
    method_parameters = parameters_pgn if method == 'PGN' else parameters_dqn
    agent_params = {
        'method': method,
        'variation': variation,
        'method_specific_parameters': method_parameters,  # A fictionnary of parameters proper to the method
        'gamma': 0.99,                                    # The discounting factor
        'lr1': 1e-3,                                      # A first learning rate
        'lr2': 1e-3,                                      # A second learning rate (equal to the first one if None)
        'hidden_conv_layers': [],                   # A list of parameters ((nb of filters, size of filter, strides)) for each hidden convolutionnal layer
        'hidden_dense_layers': [64, 32],                      # A list of parameters (nb of neurons) for each hidden dense layer
        'verbose': True
    }
    # We create a Simulation object
    sim = Simulation(name_of_environment="CartPole-v0", nb_stacked_frame=1, agent_params=agent_params)
    # We train the neural network
    sim.train(target_score=195, max_episodes=1000, process_average_over=100, test_every=50, test_on=5)
