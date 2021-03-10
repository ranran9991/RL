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
import torch.nn.functional as F

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

        # mse = nn.MSELoss() # consider checking L1
        mse = lambda x,y : F.smooth_l1_loss(x,y)


        obs = env.reset()
        done = False
        steps = 1

        while not done:
            print_output = f" {steps} " if (steps % 10 == 0 or steps == 1) else "."
            print(print_output, end='')
            (action, curr_q_val) = self.prediction_net.predict(obs, epsilon=self.epsilon)
            curr_q_val.retain_grad()
            action_index = torch.argmax(curr_q_val)
            observation, reward, done, _ = env.step(np.array(action))
            (action_new, next_q_val) = target_net.predict(observation, epsilon=0.0) # Q-Learning!
            next_q_val = next_q_val.detach() # remove gradients history
            action_index_new = torch.argmax(next_q_val)

            target_q_vals = reward
            if not done:
                target_q_vals += next_q_val[action_index_new.item()]*self.discount

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



class BatchedTrainer:
    def __init__(self, lr, batch_size, buffer_capacity, epsilon, discount, update_freq):
        self.lr = lr
        self.epsilon = epsilon
        self.discount = discount
        self.batch_size = batch_size
        self.replay_buffer = ReplayMemory(buffer_capacity)
        self.update_freq = update_freq
        self.policy_net = None
        self.optimizer = None

    def train(self, env, agent:ContinuousQLearningAgent, episodes, eval_freq):
        t0=datetime.now()
        print(f"Run started at {t0}")
        rewards_list, steps_list = [], []

        self.policy_net = copy.deepcopy(agent)
        self.optimizer = optim.Adam(self.policy_net.q_net.parameters(), lr = self.lr)

        for e in range(episodes):
            agent.mode = 'train'
            # print(f"Starting episode {e}: steps performed: ",end='')
            print(f"Episode {e}: steps performed: ",end='')

            if e % self.update_freq == 0:
                agent.q_net.load_state_dict(self.policy_net.q_net.state_dict())

            obs = env.reset()
            done = False
            steps = 1

            while not done:
                print_output = f" {steps} " if (steps % 10 == 0 or steps == 1) else "."
                print(print_output, end='')
                (action, curr_q_val) = self.policy_net.predict(obs, epsilon=self.epsilon)
                observation, reward, done, _ = env.step(np.array(action))
                action = agent.action_to_index[action]
                f = lambda t: torch.tensor(t,dtype=torch.float).to(device).unsqueeze(dim=0)
                self.replay_buffer.push(f(obs), torch.tensor(action, dtype=torch.long).to(device).unsqueeze(dim=0), f(observation), f(reward))
                self.train_batch(agent)
                steps += 1
                obs = observation

            steps_list.append(steps)

            if e % eval_freq==0 and e!=0:
                agent.mode='test'
                print(f"\n\tEvaluating: ",end='')
                show_render = e%(5*eval_freq)==0
                avg_reward = evaulate(env, agent,show_render=False)
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


    def train_batch(self, agent:ContinuousQLearningAgent):
        if len(self.replay_buffer)<self.batch_size:
            return

        sampled_transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*sampled_transitions))

        state_batch = torch.cat(batch.state, dim=0)
        action_batch = torch.cat(batch.action, dim=0)
        reward_batch = torch.cat(batch.reward, dim=0)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # state_action_values = agent.predict(state_batch).gather(1, action_batch)
        # gather by picked action
        prediction = agent.q_net(state_batch)
        state_action_values = torch.gather(prediction, 1, action_batch.reshape((action_batch.size(0), 1)))

        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.policy_net.q_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.discount) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
