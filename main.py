import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import time
import sys
import random
import numpy as np
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import tensorflow.keras.initializers as initializers
from tensorflow import random as tf_random
import pandas as pd

from utils import agent_dql
from utils import agent_pg
from utils import graph
from simulation import Simulation

random.seed(100)
np.random.seed(100)
tf_random.set_seed(100)

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
                'used': True,                   # Whether we use prioritized experience replay or not
                'alpha': 0.6,                   # Prioritization intensity
                'beta': 0.4,                    # Initial parameter for Importance Sampling
                'beta_increment': 0.002,        # Increment per sampling for Importance Sampling
                'epsilon': 0.01                 # Value assigned to have non-zero probabilities
            }
        )
    # We set this agent in the simulation
    sim.set_agent(agent)
    # We train the agent for a given number of computations and episodes
    sim.train(nb_computations=10, max_episodes=200, process_average_over=100, save_training_data=True, plot_evolution=True)
    
    # Bootstrap and grap computation from generated files
    nb_computations = 10            # Number of computations for bootstrapping
    alpha = 0.95                    # Confidence interval
    graph.plot_computations(sim.folder_name, nb_computations, alpha)

    