import gym
import time


def test(name_of_environment):
    env = gym.make(name_of_environment)
    env.reset()
    done = False
    while not done:
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        print(observation, reward, info)
        if done:
            break
    time.sleep(2)
    env.close()
