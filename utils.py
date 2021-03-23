import numpy as np
from collections import namedtuple
import random

def evaulate(env, model, num_episodes=5, show_render=True, is_noisy=False):
    episode_rewards=[]
    obs = env.reset()
    for i in range(num_episodes):
        obs = env.reset()
        if is_noisy:
            noise = np.random.normal(loc=0.0, scale=0.05, size=2)
            obs[0] += noise[0]
            obs[1] += noise[1]
        episode_rewards.append(0.0)
        done = False
        # steps = 1
        while not done:
            action = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if is_noisy:
                noise = np.random.normal(loc=0.0, scale=0.05, size=2)
                obs[0] += noise[0]
                obs[1] += noise[1]
                
            if show_render: env.render()
            episode_rewards[-1] += reward

    mean = np.round(np.mean(episode_rewards),1)
    return mean


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)