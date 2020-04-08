import tensorflow as tf
import numpy as np
from collections import deque
import random
import itertools


tf.random.set_seed(1)


class Agent():
    def __init__(self,
                 state_space_shape,              # The shape of the state space
                 action_space_size,              # The size of the action space
                 method='DQN',                   # The main method to use
                 variation=None,                 # Which variation of the method to use (None stands for original method)
                 method_specific_parameters={},  # A fictionnary of parameters proper to the method
                 gamma=0.99,                     # The discounting factor
                 lr1=1e-2,                       # A first learning rate
                 lr2=None,                       # A second learning rate
                 hidden_conv_layers=[],          # A list of parameters of for each hidden convolutionnal layer
                 hidden_dense_layers=[32],       # A list of parameters of for each hidden dense layer
                 ):
        assert(method in ['DQN', 'PGN'])
        if method == 'DQN':
            assert(variation in [None, 'DDQN'])
        else:
            assert(variation in [None, 'AC'])
        self.state_space_shape = state_space_shape
        self.action_space_size = action_space_size
        self.method = method
        self.variation = variation
        self.gamma = gamma
        self.lr1 = lr1
        self.lr2 = lr1 if lr2 is None else lr2
        self.optimizer1 = tf.keras.optimizers.Adam(self.lr1)
        self.optimizer2 = tf.keras.optimizers.Adam(self.lr2)
        self.hidden_dense_layers = hidden_dense_layers
        self.hidden_conv_layers = hidden_conv_layers
        self.current_loss = -1
        # We set the specific parameters of the method
        self.set_parameters(method_specific_parameters)
        # We build neural networks
        self.build_network()

    def set_parameters(self, parameters):
        """
        This function is used to set all the attributes specific to the method
        """
        if self.method == 'PGN':
            self.memory = deque()  # The memory used to track rewards, states and actions during an episode
        else:
            self.memory = deque(maxlen=parameters['replay_memory_size'])    # The memory used to track rewards, states and actions during an episode
            self.epsilons = np.linspace(parameters['eps_start'], parameters['eps_end'], parameters['eps_decay_steps'])   # epsilon greedy strategy (start, stop, decay)
            self.update_target_estimator_every = parameters['update_target_estimator_every']   # update the TD targets q-values every 50 optimization steps
            self.batch_size = parameters['batch_size']   # Batch size for learning
            self.opti_step = 0  # The number of optimization step done

    def build_network(self):
        """
        This function is used to build the neural networks needed for the given method
        """

        # We define the inputs of the neural network
        states = tf.keras.Input(shape=self.state_space_shape)  # The current state

        if self.method == 'PGN':
            advantages = tf.keras.Input(shape=(1,))  # The future reward associated
        else:
            pass
        x = states
        if len(self.state_space_shape) > 1:
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
        if self.method == 'PGN':
            # One dense output layer, softmax activated (to get probabilities)
            actions_probs = tf.keras.layers.Dense(self.action_space_size, activation='softmax',
                                                  kernel_initializer=tf.keras.initializers.he_normal())(x)

            self.policy = tf.keras.Model(inputs=states, outputs=actions_probs)  # One for predicting the probabilities

            self.actor = tf.keras.Model(inputs=[states, advantages], outputs=actions_probs)  # Actor for training

            def actor_loss(y_true, y_pred):
                # Here we define a custom loss
                out = tf.keras.backend.clip(y_pred, 1e-8, 1 - 1e-8)  # We need to clip y_pred as it may be equal to zero (otherwise problem with log afterwards)
                log_lik = y_true * tf.keras.backend.log(out)  # We get the log likelyhood associated to the predictions
                return tf.keras.backend.mean(-log_lik * advantages)  # We multiply it by the advantage (future reward here), and return the mean

            self.actor.compile(loss=actor_loss, optimizer=self.optimizer1, experimental_run_tf_function=False)  # Compiling Actor for training with custom loss

            if self.variation == 'AC':
                # One dense output layer, linear activated (to get value of state)
                value = tf.keras.layers.Dense(1, activation='linear',
                                              kernel_initializer=tf.keras.initializers.he_normal())(x)

                self.critic = tf.keras.Model(inputs=states, outputs=value)  # Critic for training

                def critic_loss(y_true, y_pred):
                    return tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred))

                self.critic.compile(loss=critic_loss, optimizer=self.optimizer2, experimental_run_tf_function=False)  # Compiling Critic for training

        else:
            # One dense output layer, linear activated (to get Q value)
            q_values = tf.keras.layers.Dense(self.action_space_size, activation='linear',
                                             kernel_initializer=tf.keras.initializers.he_normal())(x)

            self.q_network = tf.keras.Model(inputs=states, outputs=q_values)   # One network to prdict Q values

            if self.variation == 'DDQN':
                # Copy of the network for target calculation, updated every self.update_target_estimator_every
                self.target_model = tf.keras.models.clone_model(self.q_network)

    def take_action(self, state, train=False):
        """
        This function is used by the agent to take an action depending on the current state
        """
        state = state[np.newaxis, :]
        if self.method == 'PGN':
            action_probs = self.policy(state)  # We sample actions probabilities with the neural network
            action = np.random.choice(self.action_space_size, p=action_probs[0].numpy())  # We sample the action based on these probabilities
        else:
            if train and np.random.rand() < self.epsilons[min(self.opti_step, len(self.epsilons) - 1)]:  # proba epsilon of making a random action choice
                action = np.random.choice(self.action_space_size)  # Random action
            else:
                action = np.argmax(self.q_network(state)[0])  # otherwise, choose the action with the best predicted reward starting next step
        return action

    def learn_during_ep(self, state, action, reward, next_state, done):
        if self.method == 'PGN':
            if self.variation is None:
                self.memory.append((state, action, reward))
            elif self.variation == 'AC':
                state = state[np.newaxis, :]
                next_state = next_state[np.newaxis, :]

                critic_value_next = self.critic(next_state).numpy()[0]
                critic_value = self.critic(state).numpy()[0]

                target = reward + self.gamma * critic_value_next * (1 - int(done))
                advantage = target - critic_value

                # We one hot encode the taken actions
                one_hot_encoded_actions = np.zeros((1, self.action_space_size))
                one_hot_encoded_actions[np.arange(1), action] = 1

                self.current_loss = self.actor.train_on_batch([state, advantage], one_hot_encoded_actions)
                self.critic.train_on_batch(state, target)
        else:
            self.memory.append((state, action, reward, next_state, done))
            if self.variation == 'DDQN':
                # Update target network if we reached the update step
                if (self.opti_step + 1) % self.update_target_estimator_every == 0:
                    self.target_model.set_weights(self.q_network.get_weights())
            # Train on minibatch and update q network
            if len(self.memory) > self.batch_size:
                # We retrieve a batch of experiences from the memory
                mini_batch = random.sample(self.memory, self.batch_size)
                states_batch, action_batch, rewards_batch, next_states_batch, done_batch = map(np.array, zip(*mini_batch))
                if self.variation == 'DDQN':
                    q_target_next_values = self.target_model(next_states_batch).numpy()
                    value_next = np.max(q_target_next_values, axis=1)
                else:
                    q_next_values = self.q_network(next_states_batch).numpy()
                    value_next = np.amax(q_next_values, axis=1)

                actual_values = np.where(done_batch, rewards_batch, rewards_batch + self.gamma * value_next)
                # Gradient Descent Update
                with tf.GradientTape() as tape:
                    selected_action_values = tf.math.reduce_sum(
                        self.q_network(states_batch) * tf.one_hot(action_batch, self.action_space_size), axis=1)
                    loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
                variables = self.q_network.trainable_variables
                gradients = tape.gradient(loss, variables)
                self.optimizer1.apply_gradients(zip(gradients, variables))
                self.opti_step += 1

    def learn_end_ep(self):
        if self.method == 'PGN':
            if self.variation is None:
                # We retrieve all states, actions and reward the agent got during the episode from the memory
                states, actions, rewards = map(np.array, zip(*self.memory))
                # Here we process the future discounted sum of reward for each step of the simulation
                discounted_rewards = np.array(list(itertools.accumulate(rewards[::-1], lambda x, y: x * self.gamma + y))[::-1], dtype=np.float64)
                # We reduce and center it
                discounted_rewards -= np.mean(discounted_rewards)
                discounted_rewards /= np.std(discounted_rewards)
                # We one hot encode the taken actions
                one_hot_encoded_actions = np.zeros((len(actions), self.action_space_size))
                one_hot_encoded_actions[np.arange(len(actions)), actions.astype(int)] = 1
                # And we train the neural network on this episode (which has given us a batch of actions, states and rewards)
                self.current_loss = self.actor.train_on_batch([states, discounted_rewards], one_hot_encoded_actions)
                self.memory.clear()
        else:
            pass
