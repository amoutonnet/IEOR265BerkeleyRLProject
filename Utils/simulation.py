import gym
import time
import sys
import tensorflow as tf
import numpy as np
import itertools as it
from datetime import datetime
from tqdm import tqdm

tf.keras.backend.set_floatx('float64')


def check_env(f):
    def wrapper(*args, **kwargs):
        if args[0].env is not None:
            return f(*args, **kwargs)
        else:
            print('You need to make an environment!')
    return wrapper


class Simulation():

    def __init__(self, method='PG'):
        self.gamma = 0.99
        self.n_frames = 4
        self.env = None
        self.method = method
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

    def make_env(self, name_of_environment, verbose=False):
        self.env = gym.make(name_of_environment)
        self.action_space_size = self.env.action_space.n
        self.state_space_shape = self.env.reset().shape
        if self.method == 'PG':
            self.create_policy_gradient_networks(verbose)
        else:
            self.create_q_learning_networks(verbose)

    @check_env
    def reset_env(self):
        return self.env.reset()

    @check_env
    def test_random(self, verbose=False):
        done = False
        while not done:
            self.env.render()
            obs, rew, done, info = self.env.step(self.env.action_space.sample())
            if verbose:
                print(obs.shape, rew, info)
            if done:
                self.env.render()
                break
        time.sleep(2)
        self.env.close()

    @check_env
    def test_intelligent(self, num_tests):
        score = []
        for _ in range(num_tests):
            done = False
            state = self.reset_env()
            rew_sum = 0
            while not done:
                self.env.render()
                action_probs = self.predict(np.expand_dims(state, axis=0))
                action = np.random.choice(self.action_space_size, p=action_probs[0].numpy())
                obs, rew, done, _ = self.env.step(action)
                rew_sum += rew
                state = obs
            score += [rew_sum]
        self.env.close()
        return np.mean(score)

    @check_env
    def create_policy_gradient_networks(self, verbose=False):
        inputs = tf.keras.Input(shape=self.state_space_shape)
        advantages = tf.keras.Input(shape=(1,))
        x = tf.keras.layers.Dense(32, activation='relu',
                                  use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal())(inputs)
        x = tf.keras.layers.Dense(32, activation='relu',
                                  use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal())(x)
        outputs = tf.keras.layers.Dense(self.action_space_size, activation='softmax',
                                        use_bias=False,
                                        kernel_initializer=tf.keras.initializers.he_normal())(x)
        self.policy = tf.keras.Model(inputs=[inputs, advantages], outputs=outputs)
        self.predict = tf.keras.Model(inputs=inputs, outputs=outputs)

        def custom_loss(y_true, y_pred):
            out = tf.keras.backend.clip(y_pred, 1e-8, 1 - 1e-8)
            log_lik = y_true * tf.keras.backend.log(out)
            return tf.keras.backend.mean(-log_lik * advantages, keepdims=True)

        self.policy.compile(loss=custom_loss, optimizer=self.optimizer, experimental_run_tf_function=False)
        if verbose:
            self.network.summary()

    @check_env
    def create_q_learning_networks(self, verbose=False):
        # TO IMPLEMENT
        self.network = None
        if verbose:
            self.network.summary()
        pass

    @check_env
    def update_policy_gradient_network(self, rewards, states, actions):
        discounted_rewards = np.array(list(it.accumulate(rewards[::-1], lambda x, y: x * self.gamma + y))[::-1], dtype=np.float64)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        one_hot_encoded_actions = np.zeros((len(actions), self.action_space_size))
        one_hot_encoded_actions[np.arange(len(actions)), actions] = 1
        loss = self.policy.train_on_batch([np.stack(states, axis=0), discounted_rewards], one_hot_encoded_actions)
        return loss

    @check_env
    def train_policy_gradient(self, total_episodes=1000):
        train_writer = tf.summary.create_file_writer(f"Logs/%sBreakout_%s" % (self.method, datetime.now().strftime('%d%m%Y%H%M')))
        total_reward = []
        for ep in range(total_episodes):
            state = self.reset_env()
            states = []
            rewards = []
            actions = []
            done = False
            while not done:
                action_probs = self.predict(np.expand_dims(state, axis=0))
                action = np.random.choice(self.action_space_size, p=action_probs[0].numpy())
                obs, rew, done, _ = self.env.step(action)
                # next_state = np.append(state[:, 1:], np.expand_dims(obs, -1), axis=-1)
                next_state = obs
                states += [state]
                rewards += [rew]
                actions += [action]
                state = next_state
            loss = self.update_policy_gradient_network(rewards, states, actions)
            total_reward += [sum(rewards)]
            if (ep + 1) % 50 == 0:
                mean_reward = np.mean(total_reward)
                total_reward = []
                score = self.test_intelligent(10)
                print("Episode: %d,  Mean Training Reward: %.2f, Test Score: %.2f, Loss: %.5f" % (ep + 1, mean_reward, score, loss))
                with train_writer.as_default():
                    tf.summary.scalar('reward', mean_reward, step=ep + 1)
                    tf.summary.scalar('loss', loss, step=ep + 1)


if __name__ == "__main__":
    sim = Simulation()
    sim.make_env("CartPole-v0")
    sim.train_policy_gradient()
