from . import agent
from collections import deque
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import random
from . import per_utils



class AgentDQL(agent.Agent):
    def __init__(self,
                 state_space_shape,                 # The shape of the state space
                 action_space_size,                 # The size of the action space
                 gamma=0.99,                        # The discounting factor
                 hidden_conv_layers=[],             # A list of parameters of for each hidden convolutionnal layer
                 hidden_dense_layers=[32],          # A list of parameters of for each hidden dense layer
                 initializer='random_normal',
                 verbose=False,                     # A live status of the training
                 lr=1e-2,                           # The learning rate
                 max_memory_size=2000,              # The maximum size of the replay memory
                 epsilon_behavior=(1, 0.1, 100),    # The decay followed by epsilon
                 batch_size=32,                     # The batch size used during the training
                 update_target_every=100,           # Update the TD targets q-values every update_target_every optimization steps
                 double_dict={
                     'used': False                  # Whether we use double q learning or not
                 },
                 dueling_dict={
                     'used': False,                 # Whether we use dueling q learning or not
                 },
                 per_dict={
                     'used': False,                 # Whether we use prioritized experience replay or not
                     'alpha': 0.6,                  # Prioritization intensity
                     'beta': 0.4,                   # Initial parameter for Importance Sampling
                     'beta_increment': 0.001,       # Increment per sampling for Importance Sampling
                     'epsilon': 0.001               # Value assigned to have non-zero probailities
                 }):
        super().__init__(
            state_space_shape=state_space_shape,
            action_space_size=action_space_size,
            gamma=gamma,
            hidden_conv_layers=hidden_conv_layers,
            hidden_dense_layers=hidden_dense_layers,
            initializer=initializer,
            verbose=verbose
        )
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.max_memory_size = max_memory_size
        self.memory = None
        self.loss = float('inf')
        self.epsilons = np.linspace(*epsilon_behavior)
        self.batch_size = batch_size
        self.update_target_every=update_target_every
        self.opti_step = 0
        self.main_name = 'dql'
        self.use_double = double_dict.pop('used')
        self.use_dueling = dueling_dict.pop('used')
        self.use_per = per_dict.pop('used')
        if self.use_per:
            self.memory = per_utils.PrioritizedExperienceMemory(
                self.max_memory_size,
                per_dict['alpha'],
                per_dict['beta'],
                per_dict['beta_increment'],
                per_dict['epsilon'],
            )
        else:
            self.memory = deque(maxlen=self.max_memory_size)

    def build_network(self):
        # States as input
        states = tf.keras.Input(shape=self.state_space_shape, name='%s_states' % self.main_name)
        # Conv layers for network
        x = self.create_conv_layers(states, 'network')
        if not self.use_dueling:
            # Dense layers for network
            x = self.create_dense_layers(x, 'network')
            # Qvalues as output
            q_values = tf.keras.layers.Dense(units=self.action_space_size,
                                             activation='linear',
                                             kernel_initializer=self.initializer,
                                             name='%s_qvalues' % self.main_name)(x)
        else:
            # Separated dense layers for state values and actions advantages
            y = self.create_dense_layers(x, 'val_net')
            x = self.create_dense_layers(x, 'adv_net')
            # states values
            state_values = tf.keras.layers.Dense(units=1,
                                                 activation='linear',
                                                 kernel_initializer=self.initializer,
                                                 name='%s_values' % self.main_name
                                                 )(y)

            # actions advantages
            actions_advantages = tf.keras.layers.Dense(units=self.action_space_size,
                                                       activation='linear',
                                                       kernel_initializer=self.initializer,
                                                       name='%s_advantages' % self.main_name
                                                       )(x)

            # Mean of actions advantages
            actions_advantages_average = tf.keras.layers.Lambda(
                lambda x: tf.math.reduce_mean(x, axis=1, keepdims=True),
                name='%s_adv_averaged' % self.main_name
            )(actions_advantages)

            # Centered actions advantages
            centered_actions_advantages = tf.keras.layers.Subtract(
                name='%s_adv_centered' % self.main_name
            )([actions_advantages, actions_advantages_average])

            # The two outputs are merged to obtain the Q-Values
            q_values = tf.keras.layers.Add(
                name='%s_qvalues' % self.main_name
            )([state_values, centered_actions_advantages])

        # The QNetwork
        self.q_network = tf.keras.Model(inputs=states, outputs=q_values, name='Q-Network')

        # Copy of the network for target calculation, updated every self.update_target_estimator_every
        self.target_model = tf.keras.models.clone_model(self.q_network)

        if self.verbose:
            print('\n%s\n' % ('Neural Networks'.center(100, '-')))
            self.q_network.summary(line_length=120)

    def predict_action(self, state, current_eps=None):
        state = state[np.newaxis, :]   # We augment the dimension of the state
        if current_eps is None:
            current_eps = self.epsilons[min(self.opti_step, len(self.epsilons) - 1)]
        if np.random.rand() < current_eps:  # proba epsilon of making a random action choice
            action = np.random.choice(self.action_space_size)  # Random action
        else:
            action = np.argmax(self.q_network(state)[0])  # otherwise, choose the action with the best predicted reward starting next step
        return action

    def get_next_q_values(self, next_states_batch):
        if self.use_double:
            # Find the best action as defined by q_network
            best_next_actions = np.argmax(self.q_network(next_states_batch).numpy(), axis = 1)
            # Compute the Q-value associatde with best actions thanks to the target network
            q_values_next = tf.math.reduce_sum(
                self.target_model(next_states_batch) * tf.one_hot(best_next_actions, self.action_space_size), axis=1)
        else:
            # If Double Q-learning is not used, then just choose the action maximizing the target network output
            q_values_next = np.amax(self.target_model(next_states_batch).numpy(), axis=1)
        return q_values_next

    def learn_on_policy(self):
        pass

    def learn_off_policy(self):
        if (self.opti_step + 1) % self.update_target_every == 0:
            self.target_model.set_weights(self.q_network.get_weights())
        if len(self.memory) >= self.batch_size:
            if self.use_per:
                tree_idx, mini_batch, ISweights = self.memory.sample(self.batch_size)
            else:
                mini_batch = random.sample(self.memory, self.batch_size)
            states_batch, action_batch, rewards_batch, next_states_batch, done_batch = map(np.array, zip(*mini_batch))
            next_values = self.get_next_q_values(next_states_batch)
            actual_values = np.where(done_batch, rewards_batch, rewards_batch + self.gamma * next_values)
            # Gradient Descent Update
            with tf.GradientTape() as tape:
                selected_action_values = tf.math.reduce_sum(
                    self.q_network(states_batch) * tf.one_hot(action_batch, self.action_space_size), axis=1)
                td_errors = actual_values - selected_action_values
                if self.use_per:
                    td_errors = ISweights * td_errors / np.max(ISweights)
                self.loss = tf.math.reduce_mean(tf.square(td_errors))
            variables = self.q_network.trainable_variables
            gradients = tape.gradient(self.loss, variables)
            if self.use_per:
                self.memory.batch_update(tree_idx, tf.math.abs(td_errors)[0])
            self.optimizer.apply_gradients(zip(gradients, variables))
            self.opti_step += 1

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))   # We remember the current flow of state/action/reward/next_state/done

    def print_verbose(self, ep, total_episodes, episode_reward, rolling_score):
        if self.verbose:
            current_eps = self.epsilons[min(self.opti_step, len(self.epsilons) - 1)]
            print('Episode {:3d}/{:5d} | Current Score ({:3.2f}) Rolling Average ({:3.2f}) | Epsilon ({:.2f}) ReplayMemorySize ({:5d}) OptiStep ({:5d}) Loss ({:.10f})'.format(ep,
                                                                                                    total_episodes,
                                                                                                    episode_reward,
                                                                                                    rolling_score,
                                                                                                    current_eps,
                                                                                                    len(self.memory),
                                                                                                    self.opti_step,
                                                                                                    self.loss,
                                                                                                    ),
                  end="\r")