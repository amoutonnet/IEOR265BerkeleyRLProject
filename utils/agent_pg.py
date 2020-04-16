from . import agent
from collections import deque
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

DELTA = agent.DELTA


class AgentPGBase(agent.Agent):
    def __init__(self,
                 state_space_shape,              # The shape of the state space
                 action_space_size,              # The size of the action space
                 gamma=0.99,                     # The discounting factor
                 hidden_conv_layers=[],          # A list of parameters of for each hidden convolutionnal layer
                 hidden_dense_layers=[32],       # A list of parameters of for each hidden dense layer
                 verbose=False,                  # A live status of the training
                 initializer=tf.keras.initializers.RandomNormal(),
                 lr_actor=1e-2,                  # A first learning rate
                 lr_critic=1e-2,                 # A second learning rate
                 temperature=1e-3                # The temperature parameter for entropy
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
        self.optimizer_critic = tf.keras.optimizers.Adam(lr_critic)
        self.loss_actor = -float('inf')
        self.loss_critic = float('inf')
        self.temperature = temperature
        self.memory = list()
        self.main_name = 'pg'

    def build_network(self):
        # States as input
        states = tf.keras.Input(shape=self.state_space_shape, name='%s_states' % self.main_name)
        # Advatnages as input
        advantages = tf.keras.Input(shape=(1,), name='%s_advantages' % self.main_name)
        # Conv layers for actor
        x = self.create_conv_layers(states, 'actor')
        # Dense layers for actor
        x = self.create_dense_layers(x, 'actor')
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

        # Conv layers for critic
        x = self.create_conv_layers(states, 'critic')
        # Dense layers for critic
        x = self.create_dense_layers(x, 'critic')
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

        if self.verbose:
            print('\n%s\n' % ('Neural Networks'.center(100, '-')))
            self.actor.summary(line_length=120)
            print()
            self.critic.summary(line_length=120)

        return advantages

    def predict_action(self, state):
        state = state[np.newaxis, :]   # We augment the dimension of the state
        action_probs = self.policy(state)  # Sample actions probabilities with the neural network
        action = np.random.choice(self.action_space_size, p=action_probs[0].numpy())  # We sample the action based on these probabilities
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))   # We remember the current flow of state/action/reward/next_state/done

    def print_verbose(self, ep, total_episodes, episode_reward, rolling_score):
        if self.verbose:
            print('Episode {:3d}/{:5d}\
                 | Current Score ({:3.2f}) Rolling Average ({:3.2f}) \
                 | Actor Loss ({:.4f}) Critic Loss ({:.4f})'.format(ep,
                                                                    total_episodes,
                                                                    episode_reward,
                                                                    rolling_score,
                                                                    self.loss_actor,
                                                                    self.loss_critic
                                                                    ),
                  end="\r")

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
                 initializer=tf.keras.initializers.RandomNormal(),
                 lr_actor=1e-2,                  # A first learning rate
                 lr_critic=1e-2,                 # A second learning rate
                 temperature=1e-3,               # The temperature parameter for entropy
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
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            temperature=temperature
        )
        self.use_ppo = ppo_dict.pop('used')
        if self.use_ppo:
            self.epsilon = ppo_dict['epsilon']

    def build_network(self):
        advantages = super().build_network()

        if not self.use_ppo:
            # Loss for A2C
            def actor_loss(y_true, y_pred):
                # Here we define a custom loss for A2C policy gradient
                out = K.clip(y_pred, DELTA, 1)  # We need to clip y_pred as it may be equal to zero (otherwise problem with log afterwards)
                log_lik = K.sum(y_true * K.log(out), axis=1)  # We get the log likelyhood associated to the predictions
                advantages_with_entropy = advantages - self.temperature * K.stop_gradient(log_lik)
                return -K.mean(log_lik * advantages_with_entropy, keepdims=True)  # We multiply it by the advantage (future reward here)
        else:
            # Loss for PPO
            def actor_loss(y_true, y_pred):
                # Here we define a custom for proximal policy optimization
                out = K.clip(y_pred, DELTA, 1)
                log_lik = y_true * K.log(out)
                old_log_lik = K.stop_gradient(log_lik)
                advantages_with_entropy = advantages - K.sum(self.temperature * old_log_lik, axis=-1)
                ratio = K.sum(K.exp(log_lik - old_log_lik), axis=1)
                clipped_ratio = K.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
                return -K.mean(K.minimum(ratio * advantages_with_entropy, clipped_ratio * advantages_with_entropy), keepdims=True)

        self.actor.compile(loss=actor_loss, optimizer=self.optimizer_actor, experimental_run_tf_function=False)

    def learn_on_policy(self):
        # We retrieve all states, actions and reward the agent got during the episode from the memory
        states, actions, rewards, next_states, dones = map(np.array, zip(*self.memory))
        # We one hot encode the taken actions
        one_hot_encoded_actions = np.zeros((len(actions), self.action_space_size))
        one_hot_encoded_actions[np.arange(len(actions)), actions.astype(int)] = 1
        # We process the states values with the critic network
        critic_values = self.critic(states).numpy()
        critic_next_values = self.critic(next_states).numpy()
        # We get the target reward
        targets = rewards + self.gamma * np.squeeze(critic_next_values) * np.invert(dones)
        # We get the advantage (difference between the discounted reward and the baseline)
        advantages = targets - np.squeeze(critic_values)
        # We normalize advantages
        advantages = self.normalize(advantages)
        # We train the two networks
        self.loss_actor = self.actor.train_on_batch([states, advantages], one_hot_encoded_actions)
        self.loss_critic = self.critic.train_on_batch(states, targets)
        self.memory.clear()
