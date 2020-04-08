import gym
import time
import sys
import numpy as np
from tqdm import tqdm
from collections import deque
import random
import learning


class Simulation():

    def __init__(self, name_of_environment='CartPole-v0', test_every=50, nb_stacked_frame=1, agent_params={}):
        assert(nb_stacked_frame >= 1)
        # Main attributes of the Simulation
        self.env = gym.make(name_of_environment)   # We make the env
        self.test_every = test_every   # We will track the evolution of the training every test_every step
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
                action = self.agent.take_action(state)
                obs, rew, done, _ = self.env.step(action)   # We take a step in the environment by taking the sampled action
                rew_sum += rew   # We add the reward earned to the total reward
                state = self.get_next_state(state, obs)   # We get the next state of the evironment
            score += [rew_sum]   # Once the game is played, we store the total reward
        self.env.close()   # Once every game is played, we close the rendered envrionment
        return np.mean(score)   # We return the average score  over all games

    def train(self, total_episodes=1000):
        # Here we train our neural network with the given method
        total_rewards = []  # We initialize the total reward list
        for ep in range(total_episodes):
            state = self.reset_env()  # We get x0
            episode_reward = 0
            done = False
            # While the game is not finished
            while not done:
                action = self.agent.take_action(state, train=True)  # we sample the action
                obs, reward, done, _ = self.env.step(action)  # We take a step forward in the environment by taking the sampled action
                episode_reward += reward
                next_state = self.get_next_state(state, obs)
                self.agent.learn_during_ep(state, action, reward, next_state, done)
                state = next_state
            self.agent.learn_end_ep()

            total_rewards += [episode_reward]
            if (ep + 1) % self.test_every == 0:
                # Every test_every episodes, we track the progress
                mean_reward = np.mean(total_rewards)   # Mean reward over the last test_every episodes
                total_rewards = []   # We reset the list of total rewards
                score = self.test_intelligent(10)   # We simulate a few test games to track the evolution of the abilities of our agent
                print("Episode: %d,  Mean Training Reward: %.2f, Mean Test Score: %.2f, Loss: %.5f" % (ep + 1, mean_reward, score, self.agent.current_loss))


if __name__ == "__main__":
    method = 'DQN'                # 'PGN' for policy gradient, 'DQN' Deep Q-Learning
    variation = None            # Set to None for original method, otherwise 'AC' for 'PGN, 'DDQN' for 'DQN'
    parameters_dqn = {
        'eps_start': 1.0,
        'eps_end': 0.1,
        'eps_decay_steps': 100,
        'replay_memory_size': 2000,
        'update_target_estimator_every': 50,
        'batch_size': 32,
    }
    parameters_pgn = {}
    method_parameters = parameters_pgn if method == 'PGN' else parameters_dqn
    agent_params = {
        'method': method,
        'variation': variation,
        'method_specific_parameters': method_parameters,  # A fictionnary of parameters proper to the method
        'gamma': 0.99,                                    # The discounting factor
        'lr1': 1e-2,                                      # A first learning rate
        'lr2': 1e-5,                                      # A second learning rate (equal to the first one if None)
        'hidden_conv_layers': [],                         # A list of parameters ((nb of filters, size of filter)) for each hidden convolutionnal layer
        'hidden_dense_layers': [32],                 # A list of parameters (nb of neurons) for each hidden dense layer
    }
    # We create a Simulation object
    sim = Simulation(name_of_environment="CartPole-v0", test_every=50, nb_stacked_frame=1, agent_params=agent_params)
    # We train the neural network
    sim.train(total_episodes=1000)
