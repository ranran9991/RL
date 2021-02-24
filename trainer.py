import copy
from agent import *
from utils import *
from abc import abstractmethod
from matplotlib import pyplot as plt
# from time import time
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn


class Trainer:
    def __init__(self, lr, epsilon, discount):
        self.lr = lr
        self.epsilon = epsilon
        self.discount = discount

    @abstractmethod
    def train_episode(self, env, agent):
        pass

    def train(self, env, agent:Agent, episodes, eval_freq):
        t0=datetime.now()
        print(f"Run started at {t0}")
        rewards_list, steps_list = [], []
        for e in range(episodes):
            agent.mode = 'train'
            # print(f"Starting episode {e}: steps performed: ",end='')
            print(f"Episode {e}: steps performed: ",end='')
            steps = self.train_episode(env, agent)
            steps_list.append(steps)

            if e % eval_freq==0:
                agent.mode='test'
                print(f"\n\tEvaluating: ",end='')
                avg_reward = evaulate(env, agent)
                rewards_list.append(avg_reward)
                print(f"avg reward = {avg_reward}",end='') # insert weights_and_biases !
                # print(f"Finished Evaluation")

            print(f"\nFinished episode {e}")
            # print("done.")

        print(f"Steps per episode: {steps_list}, avg = {np.round(np.mean(steps_list),2)}")
        print(f"Average rewards: {rewards_list}")
        print(f"Run ended in {datetime.now()}; total runtime = {datetime.now()-t0}")

        plt.figure()
        plt.plot(rewards_list)
        plt.show()


class TDTrainer(Trainer):
    def __init__(self, lr, epsilon, discount, lamda):
        super().__init__(lr, epsilon, discount)
        self.lamda = lamda

    def train_episode(self, env, agent:DiscreteAgent):
        obs = env.reset()
        done = False
        steps = 1
        while not done:
            # print(f"Step {steps}: started...",end=' ')
            print(f"{steps}",end=' ')
            action = np.array(agent.predict(obs, epsilon=self.epsilon))
            observation, reward, done, _ = env.step(action)

            current_obs_indices = observation_to_bucket(np.concatenate((obs,action)), agent.buckets)

            next_action = np.array(agent.predict(observation,epsilon=0))
            next_obs_indices = observation_to_bucket(np.concatenate((observation, next_action)), agent.buckets)
            # print(5,end=' ')
            # t0=time()
            # agent.table[current_obs_indices] = agent.table[current_obs_indices] + self.lr *(reward + self.discount * agent.table[next_obs_indices] - agent.table[current_obs_indices])
            mat = agent.table[current_obs_indices]
            agent.table[current_obs_indices] = (1 - self.lr) * mat + self.lr * (reward + self.discount * agent.table[next_obs_indices])  # mat + self.lr * (reward + self.discount * agent.table[next_obs_indices] - mat)
            # print(f"{np.round(time()-t0,3)}",end=' ')
            # print(6,end=' ')

            obs = observation
            # print(f"ended.")
            steps += 1

        return steps

class QLearningTrainer(Trainer):
    def __init__(self, lr, epsilon, discount, update_freq):
        super().__init__(lr, epsilon, discount)
        self.update_freq = update_freq
        
        self.optimizer = None
        self.prediction_net = None
        
        self.episode_counter = 0

    def train_episode(self, env, agent : ContinuousQLearningAgent):
        self.episode_counter += 1

        if not self.prediction_net:
            self.prediction_net = copy.deepcopy(agent)
            self.optimizer = optim.Adam(self.prediction_net.q_net.parameters(), lr = self.lr)
        
        target_net = agent
        if self.episode_counter % self.update_freq:
            # update target net
            target_net.q_net = copy.deepcopy(self.prediction_net.q_net)
        
        mse = nn.MSELoss()

        obs = env.reset()
        done = False
        steps = 1

        while not done:
            (action, curr_q_val) = self.prediction_net.predict(obs, epsilon=self.epsilon)
            curr_q_val.retain_grad()
            action_index = torch.argmax(curr_q_val)
            observation, reward, done, _ = env.step(np.array(action))
            (_, next_q_val) = target_net.predict(observation, epsilon=0.0)
            next_q_val = next_q_val.detach()

            target_q_vals = reward
            if not done:
                target_q_vals += next_q_val[action_index.item()]*self.discount
            
            target_vec = curr_q_val.clone()
            target_vec[action_index.item()] = target_q_vals
            target_vec = target_vec.detach()
            self.optimizer.zero_grad()
            loss = mse(curr_q_val, target_vec)
            loss.backward()
            self.optimizer.step()
            steps += 1
            obs = observation
        
        return steps