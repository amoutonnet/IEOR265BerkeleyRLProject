from . import agent
from collections import deque
import tensorflow as tf
import tensorflow.keras.backend as K
import itertools
import functools
import numpy as np
import sys

DELTA = agent.DELTA


class AgentPGBase(agent.Agent):
    def __init__(self,
                 state_space_shape,              # The shape of the state space
                 action_space_size,              # The size of the action space
                 gamma=0.99,                     # The discounting factor
                 hidden_conv_layers=[],          # A list of parameters of for each hidden convolutionnal layer
                 hidden_dense_layers=[32],       # A list of parameters of for each hidden dense layer
                 verbose=False,                  # A live status of the training
                 initializer='random_normal',
                 lr_actor=1e-2,                  # A first learning rate
                 ):
        super().__init__(
            state_space_shape=state_space_shape,
            action_space_size=action_space_size,
            gamma=gamma,
            hidden_conv_layers=hidden_conv_layers,
            hidden_dense_layers=hidden_dense_layers,
            initializer=initializer,
            verbose=verbose
        )
        self.optimizer_actor = tf.keras.optimizers.Adam(lr_actor)
        self.loss_actor = -float('inf')
        self.main_name = 'pg'
        self.first_computation = True  # If True print model summary, if not, do not print

    def init_agent_for_training(self):
        self.memory = list()
        self.build_network()

    def build_network(self):
        raise NotImplementedError

    def predict_action(self, state, current_eps=None):
        state = state[np.newaxis, :]   # We augment the dimension of the state
        action_probs = self.policy(state)  # Sample actions probabilities with the neural network
        action = np.random.choice(self.action_space_size, p=action_probs[0].numpy())  # We sample the action based on these probabilities
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward))   # We remember the current flow of state/action/reward/next_state/done

    def learn_off_policy(self):
        pass


class AgentPG(AgentPGBase):
    def __init__(self,
                 state_space_shape,              # The shape of the state space
                 action_space_size,              # The size of the action space
                 gamma=0.99,                     # The discounting factor
                 hidden_conv_layers=[],          # A list of parameters of for each hidden convolutionnal layer
                 hidden_dense_layers=[32],       # A list of parameters of for each hidden dense layer
                 verbose=False,                  # A live status of the training
                 initializer='random_normal',    # Initializer for neural networks weights
                 lr_actor=1e-2,                  # A first learning rate
                 epochs=1,                       # Number of epochs to realise at each episode
                 a2c_dict={
                     'used': False,              # Whether or not to use A2C
                     'lr_critic': 1e-2           # The learning rate of the critic
                 },
                 gae_dict={
                     'used': True,               # Whether or not to use GAE
                     'lambd': 0.5,               # lambda for GAE
                 },
                 entropy_dict={
                     'used': False,              # Whether or not Entropy Regulaarization is used
                     'temperature': 1e-3         # Temperature parameter for entropy reg
                 },
                 ppo_dict={
                     'used': False,              # Whether we use PPO or not
                     'epsilon': 0.2              # Epsilon for PPO
                 }):
        super().__init__(
            state_space_shape=state_space_shape,
            action_space_size=action_space_size,
            gamma=gamma,
            hidden_conv_layers=hidden_conv_layers,
            hidden_dense_layers=hidden_dense_layers,
            initializer=initializer,
            verbose=verbose,
            lr_actor=lr_actor
        )

        self.use_a2c = a2c_dict.pop('used')
        if self.use_a2c:
            self.optimizer_critic = tf.keras.optimizers.Adam(a2c_dict['lr_critic'])
            self.loss_critic = float('inf')
        else:
            self.optimizer_critic = None
            self.loss_critic = None

        self.use_gae = gae_dict.pop('used')
        if self.use_gae:
            self.lambd = gae_dict.pop('lambd')
            assert self.lambd <= 1 and self.lambd >= 0, "Lambda for GAE need to be between 0 and 1"
        else:
            self.lambd = 1
        self.use_ppo = ppo_dict.pop('used')
        if self.use_ppo:
            self.epsilon = ppo_dict.pop('epsilon')
            assert self.epsilon <= 1 and self.epsilon >= 0, "Epsilon for PPO need to be between 0 and 1"
        else:
            self.epsilon = None

        self.use_entropy_reg = entropy_dict.pop('used')
        if self.use_entropy_reg:
            self.temperature = entropy_dict.pop('temperature')
        else:
            self.temperature = 0

        self.epochs = epochs

    def build_network(self):
        # States as input
        states = tf.keras.Input(shape=self.state_space_shape, name='%s_states' % self.main_name)
        # Advatnages as input
        advantages = tf.keras.Input(shape=(1,), name='%s_advantages' % self.main_name)
        # Conv layers for actor and critic, those are common
        output_conv = self.create_conv_layers(states, 'common')
        # Dense layers for actor
        x = self.create_dense_layers(output_conv, 'actor')
        # One dense output layer, softmax activated (to get probabilities)
        actions_probs = tf.keras.layers.Dense(units=self.action_space_size,
                                              activation='softmax',
                                              kernel_initializer=self.initializer,
                                              name='%s_probs' % self.main_name
                                              )(x)
        # Policy model to predict only
        self.policy = tf.keras.Model(inputs=states, outputs=actions_probs, name='Policy')
        # Actor model to learn
        self.actor = tf.keras.Model(inputs=[states, advantages], outputs=actions_probs, name='Actor')

        if not self.use_ppo:
            # Loss for A2C
            def actor_loss(y_true, y_pred):
                # Here we define a custom loss for A2C policy gradient
                out = K.clip(y_pred, DELTA, 1 - DELTA)  # We need to clip y_pred as it may be equal to zero (otherwise problem with log afterwards)
                log_lik = y_true * K.log(out)  # We get the log likelyhood associated to the predictions
                return K.sum(-log_lik * advantages)  # We multiply it by the advantage (future reward here)
        else:
            # Loss for PPO
            def actor_loss(y_true, y_pred):
                # Here we define a custom for proximal policy optimization
                out = K.clip(y_pred, DELTA, 1 - DELTA)
                log_lik = K.sum(y_true * K.log(out), axis=-1)
                old_log_lik = K.stop_gradient(K.sum(y_true * K.log(out), axis=-1))
                ratio = K.exp(log_lik - old_log_lik)
                clipped_ratio = K.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
                return -K.mean(K.minimum(ratio * advantages, clipped_ratio * advantages))

        self.actor.compile(loss=actor_loss, optimizer=self.optimizer_actor, experimental_run_tf_function=False)

        if self.use_a2c:
            x = self.create_dense_layers(output_conv, 'critic')
            # One dense output layer, linear activated (to get value of state)
            values = tf.keras.layers.Dense(units=1,
                                           activation='linear',
                                           kernel_initializer=self.initializer,
                                           name='%s_values' % self.main_name
                                           )(x)
            # Critic model to learn
            self.critic = tf.keras.Model(inputs=states, outputs=values, name='Critic')
            # Compiling critic model
            self.critic.compile(loss='mse', optimizer=self.optimizer_critic, experimental_run_tf_function=False)

        if self.verbose and self.first_computation:
            self.first_computation = False
            print('\n%s\n' % ('Neural Networks'.center(100, '-')))
            self.actor.summary(line_length=120)
            if self.use_a2c:
                print()
                self.critic.summary(line_length=120)

    def get_advantages(self, critic_values, rewards):
        values = np.empty((len(rewards),))
        for i in range(len(rewards) - 1):
            values[i] = rewards[i] + self.gamma * critic_values[i + 1] - critic_values[i]
        values[-1] = rewards[-1] - critic_values[-1]
        return np.array(list(itertools.accumulate(values[::-1], lambda x, y: x * (self.gamma * self.lambd) + y))[::-1], dtype=np.float32)

    def learn_on_policy(self):
        # We retrieve all states, actions and reward the agent got during the episode from the memory
        states, actions, rewards = map(np.array, zip(*self.memory))
        # We one hot encode the taken actions
        one_hot_encoded_actions = np.zeros((len(actions), self.action_space_size))
        one_hot_encoded_actions[np.arange(len(actions)), actions.astype(int)] = 1
        # We get the discounted sum of rewards
        discounted_rewards = np.array(list(itertools.accumulate(rewards[::-1], lambda x, y: x * self.gamma + y))[::-1], dtype=np.float32)
        if self.use_a2c:
            # We process the states values with the critic network
            critic_values = np.squeeze(self.critic(states).numpy())
            if self.use_gae:
                # We get the target reward
                advantages = self.get_advantages(critic_values, rewards)
            else:
                advantages = discounted_rewards - critic_values
            # We train the critic network
            for _ in range(self.epochs):
                self.loss_critic = self.critic.train_on_batch(states, discounted_rewards)
        else:
            advantages = discounted_rewards
        # We normalize advantages
        # advantages = self.normalize(advantages)
        # We train the actor network
        for _ in range(self.epochs):
            self.loss_actor = self.actor.train_on_batch([states, advantages], one_hot_encoded_actions)
        # We clear the memory
        self.memory.clear()

    def print_verbose(self, ep, total_episodes, train_episode_reward, test_episode_reward, train_rolling_score=None, test_rolling_score=None):
        if self.verbose:
            if self.use_a2c:
                if train_rolling_score is None:
                    print('Ep {:5d}/{:5d} | TrainCS ({:.2f}) | TestCS ({:.2f}) | ActLoss ({:.10f}) CritLoss ({:.10f})'.format(ep,
                                                                                                                              total_episodes,
                                                                                                                              train_episode_reward,
                                                                                                                              test_episode_reward,
                                                                                                                              self.loss_actor,
                                                                                                                              self.loss_critic
                                                                                                                              ), end="                 \r")
                else:
                    print('Ep {:5d}/{:5d} | TrainCS ({:.2f}) TrainRA ({:.2f}) | TestCS ({:.2f}) TestRA ({:.2f}) | ActLoss ({:.10f}) CritLoss ({:.10f})'.format(ep,
                                                                                                                                                               total_episodes,
                                                                                                                                                               train_episode_reward,
                                                                                                                                                               train_rolling_score,
                                                                                                                                                               test_episode_reward,
                                                                                                                                                               test_rolling_score,
                                                                                                                                                               self.loss_actor,
                                                                                                                                                               self.loss_critic
                                                                                                                                                               ), end="               \r")
            else:
                if train_rolling_score is None:
                    print('Ep {:5d}/{:5d} | TrainCS ({:.2f}) | TestCS ({:.2f}) | ActLoss ({:.10f})'.format(ep,
                                                                                                           total_episodes,
                                                                                                           train_episode_reward,
                                                                                                           test_episode_reward,
                                                                                                           self.loss_actor
                                                                                                           ), end="                 \r")
                else:
                    print('Ep {:5d}/{:5d} | TrainCS ({:.2f}) TrainRA ({:.2f}) | TestCS ({:.2f}) TestRA ({:.2f}) | ActLoss ({:.10f})'.format(ep,
                                                                                                                                            total_episodes,
                                                                                                                                            train_episode_reward,
                                                                                                                                            train_rolling_score,
                                                                                                                                            test_episode_reward,
                                                                                                                                            test_rolling_score,
                                                                                                                                            self.loss_actor
                                                                                                                                            ), end="                  \r")
