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
                show_render = e%(5*eval_freq)==0
                avg_reward = evaulate(env, agent,show_render=show_render)
                rewards_list.append(avg_reward)
                print(f"avg reward = {avg_reward}",end='') # insert weights_and_biases !
                # print(f"Finished Evaluation")

                # Epsilon & lr decay:
                self.epsilon = self.epsilon * 0.94**(e // eval_freq)
                self.lr = self.lr * 0.98**(e//eval_freq)

            print(f"\nFinished episode {e}")
            # print("done.")

        print(f"Steps per episode: {steps_list}, avg = {np.round(np.mean(steps_list),2)}")
        print(f"Average rewards: {rewards_list}")
        print(f"Run ended in {datetime.now()}; total runtime = {datetime.now()-t0}")

        plt.figure()
        plt.plot(rewards_list)
        plt.show()


class TD0_Trainer(Trainer):
    def __init__(self, lr, epsilon, discount, lamda):
        super().__init__(lr, epsilon, discount)
        self.lamda = lamda

    def train_episode(self, env, agent:DiscreteAgent):
        obs_curr = env.reset()
        done = False
        steps = 1
        # MAX_STEPS = 150
        while not done:
            # print(f"Step {steps}: started...",end=' ')
            print_output = f" {steps} " if (steps%10==0 or steps==1) else "."
            print(print_output,end='')
            action_curr = np.array(agent.predict(obs_curr, epsilon=self.epsilon))

            obs_curr_indices = observation_to_bucket(obs_curr, agent.buckets[:-2])
            action_curr_indices = observation_to_bucket(action_curr, agent.buckets[-2:])
            s_curr_index = buckets2index(agent.buckets[:-2], agent.n_obs_buckets, obs_curr_indices, obs_curr)
            a_curr_index = buckets2index(agent.buckets[-2:], agent.n_action_buckets, action_curr_indices, action_curr)


            obs_next, reward, done, _ = env.step(action_curr)
            action_next = np.array(agent.predict(obs_next,epsilon=self.epsilon))

            obs_next_indices = observation_to_bucket(obs_next, agent.buckets[:-2])
            action_next_indices = observation_to_bucket(action_next, agent.buckets[-2:])
            s_next_index = buckets2index(agent.buckets[:-2], agent.n_obs_buckets, obs_next_indices, obs_next)
            a_next_index = buckets2index(agent.buckets[-2:], agent.n_action_buckets, action_next_indices, action_next)


            mat = agent.table[s_curr_index][a_curr_index]
            agent.table[s_curr_index][a_curr_index] = (1 - self.lr) * mat + self.lr * (reward + self.discount * agent.table[s_next_index][a_next_index]) # Bellman Equation

            obs_curr = obs_next
            # print(f"ended.")
            steps += 1
        print(f" {steps}",end='')
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