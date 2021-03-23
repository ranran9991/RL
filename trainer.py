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
from collections import deque

class Trainer:
    def __init__(self, lr, epsilon, discount, eps_decay):
        self.lr = lr
        self.epsilon = epsilon
        self.discount = discount
        self.eps_decay = eps_decay

    @abstractmethod
    def train_episode(self, env, agent):
        pass

    def train(self, env, agent:Agent, episodes, eval_freq):
        t0=datetime.now()
        print(f"Run started at {t0}")
        steps_list, rewards_list, rewards_list_test = [], [], []
        average_reward = []
        scores_window = deque(maxlen=100)

        for e in range(episodes):
            agent.mode = 'train'
            # print(f"Starting episode {e}: steps performed: ",end='')
            print(f"Episode {e}: steps performed: ",end='')

            steps, score = self.train_episode(env, agent)

            print(f'Episode score: {score:.2f}, next epsilon: {self.epsilon:.4f}')
            scores_window.append(score)
            rewards_list.append(np.round(score, decimals=2))
            steps_list.append(steps)
            if len(scores_window) == 100:
                average_reward.append(scores_window)

            if e % eval_freq==0:
                agent.mode='test'
                print(f"\n\tEvaluating: ",end='')
                show_render = e%eval_freq==0
                avg_reward = evaulate(env, agent,show_render=False)
                rewards_list_test.append(avg_reward)
                print(f"avg reward = {avg_reward}") # insert weights_and_biases !
                # print(f"Finished Evaluation")

                # Epsilon & lr decay:
                self.epsilon = max(self.epsilon * self.eps_decay, 0.02)
                self.lr = self.lr * 0.98**(e//eval_freq)

            if np.mean(scores_window) >= 200.0 and len(scores_window) == 100:
                print(f'Environment solved in {e - 100.:.2f} episodes with reward {np.mean(scores_window):.2f}')
                plt.figure()
                plt.plot(rewards_list)
                plt.xlabel('episodes')
                plt.ylabel('Comulative Reward')
                plt.title('Comulative reward per episode')
                plt.savefig('comulative rewards.png', format='png')

                plt.figure()
                plt.plot(average_reward, list(range(100, len(average_reward + 100))))
                plt.xlabel('episodes')
                plt.ylabel('Average Comulative Reward')
                plt.title('Average Comulative reward over 100 episodes')
                plt.savefig('Average rewards.png', format='png')

                return

            print(f"\nFinished episode {e}")


        print(f"Steps per episode: {steps_list}, avg = {np.round(np.mean(steps_list),2)}")
        print(f"Average rewards: {rewards_list}")
        print(f"Run ended in {datetime.now()}; total runtime = {datetime.now()-t0}")

        plt.figure()
        plt.plot(rewards_list)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Average reward train mode')
        # plt.show()
        plt.savefig('rewards train.png', format='png')

        plt.figure()
        plt.plot(rewards_list_test)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Average reward test mode')
        # plt.show()
        plt.savefig('rewards test.png', format='png')

class TD0_Trainer(Trainer):
    def __init__(self, lr, epsilon, discount, lamda, eps_decay=0.9):
        super().__init__(lr, epsilon, discount, eps_decay)
        self.lamda = lamda

    def train_episode(self, env, agent:DiscreteAgent):
        obs_curr = env.reset()
        done = False
        steps = 1
        score = 0. # score is the accumilated reward of the current episode
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
            agent.table[s_curr_index][a_curr_index] = (1 - self.lr) * mat + self.lr * \
                                      (reward + self.discount * agent.table[s_next_index][a_next_index]) # Bellman Equation

            obs_curr = obs_next
            steps += 1
            score += reward

        print(f" {steps}",end='')
        return steps, score


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
    def __init__(self, lr, batch_size, buffer_capacity, epsilon, discount, update_freq, eps_decay, is_noisy=False):
        self.lr = lr
        self.epsilon = epsilon
        self.discount = discount
        self.batch_size = batch_size
        self.replay_buffer = ReplayMemory(buffer_capacity)
        self.update_freq = update_freq
        self.eps_decay = eps_decay
        self.is_noisy = is_noisy
        self.policy_net = None
        self.optimizer = None

    def train(self, env, agent:ContinuousQLearningAgent, episodes, eval_freq):
        t0=datetime.now()
        print(f"Run started at {t0}")
        steps_list, rewards_list = [], []
        average_reward = []
        scores_window = deque(maxlen=100)
        self.policy_net = copy.deepcopy(agent)
        self.policy_net.q_net.to(device)
        self.optimizer = optim.Adam(self.policy_net.q_net.parameters(), lr = self.lr)

        for e in range(episodes):
            agent.mode = 'train'
            # print(f"Starting episode {e}: steps performed: ",end='')
            print(f"Episode {e}: steps performed: ",end='')

            if e % self.update_freq == 0:
                agent.q_net.load_state_dict(self.policy_net.q_net.state_dict())

            obs = env.reset()
            if self.is_noisy:
                noise = np.random.normal(loc=0.0, scale=0.05, size=2)
                obs[0] += noise[0]
                obs[1] += noise[1]

            done = False
            steps = 1
            # score is the accumilated reward of the current episode
            score = 0.
            while not done:
                print_output = f" {steps} " if (steps % 10 == 0 or steps == 1) else "."
                print(print_output, end='')
                (action, curr_q_val) = self.policy_net.predict(obs, epsilon=self.epsilon)
                observation, reward, done, _ = env.step(np.array(action))
                if self.is_noisy:
                    noise = np.random.normal(loc=0.0, scale=0.05, size=2)
                    observation[0] += noise[0]
                    observation[1] += noise[1]

                action = agent.action_to_index[action]
                f = lambda t: torch.tensor(t,dtype=torch.float).to(device).unsqueeze(dim=0)
                self.replay_buffer.push(f(obs), torch.tensor(action, dtype=torch.long).to(device).unsqueeze(dim=0), f(observation), f(reward))
                self.train_batch(agent)

 
                steps += 1
                score += reward
                obs = observation
            
            self.epsilon = max(self.epsilon * self.eps_decay, 0.01)

            print(f'Episode score: {score:.2f}, next epsilon: {self.epsilon:.4f}')
            scores_window.append(score)
            rewards_list.append(score)
            steps_list.append(steps)
            if len(scores_window) == 100:
                average_reward.append(np.mean(scores_window))
            
            if e % eval_freq==0:
                agent.mode='test'
                avg_reward = evaulate(env, agent,show_render=False)
                if avg_reward >= 200.0:
                    print(f'Environment solved in {e:.2f} episodes with reward {avg_reward:.2f}')
                    plt.figure()
                    plt.plot(rewards_list)
                    plt.xlabel('episodes')
                    plt.ylabel('Comulative Reward')
                    plt.title('Comulative reward per episode')
                    #plt.show()
                    plt.savefig('comulative rewards.png', format='png')

                    plt.figure()
                    plt.plot(list(range(100, len(average_reward)+100)), average_reward)
                    plt.xlabel('episodes')
                    plt.ylabel('Average Comulative Reward')
                    plt.title('Average Comulative reward over 100 episodes')
                    #plt.show()
                    plt.savefig('Average rewards.png', format='png')

                    return
                

            if np.mean(scores_window) >= 200.0 and len(scores_window) == 100:
                print(f'Environment solved in {e-100.:.2f} episodes with reward {np.mean(scores_window):.2f}')
                plt.figure()
                plt.plot(rewards_list)
                plt.xlabel('episodes')
                plt.ylabel('Comulative Reward')
                plt.title('Comulative reward per episode')
                #plt.show()
                plt.savefig('comulative rewards.png', format='png')

                plt.figure()
                plt.plot(list(range(100, len(average_reward)+100)), average_reward)
                plt.xlabel('episodes')
                plt.ylabel('Average Comulative Reward')
                plt.title('Average Comulative reward over 100 episodes')
                #plt.show()
                plt.savefig('Average rewards.png', format='png')

                return
            
            print(f"\nFinished episode {e}")
            # print("done.")

        print(f"Steps per episode: {steps_list}, avg = {np.round(np.mean(steps_list),2)}")
        print(f"Average rewards: {rewards_list}")
        print(f"Run ended in {datetime.now()}; total runtime = {datetime.now()-t0}")

        plt.figure()
        plt.plot(rewards_list)
        #plt.show()
        plt.savefig('rewards.png', format='png')

    def train_batch(self, agent:ContinuousQLearningAgent):
        if len(self.replay_buffer)<self.batch_size:
            return

        self.optimizer.zero_grad()

        sampled_transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*sampled_transitions))

        state_batch = torch.cat(batch.state, dim=0).to(device)
        action_batch = torch.cat(batch.action, dim=0).to(device)
        reward_batch = torch.cat(batch.reward, dim=0).to(device)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # state_action_values = agent.predict(state_batch).gather(1, action_batch)
        # gather by picked action
        prediction = self.policy_net.q_net(state_batch)
        state_action_values = torch.gather(prediction, 1, action_batch.reshape((action_batch.size(0), 1)))

        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = agent.q_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.discount) + reward_batch

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        loss.backward()
        self.optimizer.step()


class RainbowTrainer(BatchedTrainer):
    def __init__(self, lr, batch_size, buffer_capacity, epsilon, discount, update_freq, eps_decay, is_noisy=False):
        super().__init__(lr, batch_size, buffer_capacity, epsilon, discount, update_freq, eps_decay, is_noisy)

        self.v_min = None
        self.v_max = None
        self.support = None
        self.atom_size = None

    def train(self, env, agent : RainbowAgent, episodes, eval_freq):
        self.v_min = agent.v_min
        self.v_max = agent.v_max
        self.support = agent.support
        self.atom_size = agent.atom_size


        super().train(env, agent, episodes, eval_freq)


    def train_batch(self, agent : RainbowAgent):
        if len(self.replay_buffer)<self.batch_size:
            return

        self.optimizer.zero_grad()

        sampled_transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*sampled_transitions))

        state_batch = torch.cat(batch.state, dim=0).to(device)
        action_batch = torch.cat(batch.action, dim=0).to(device)
        reward_batch = torch.cat(batch.reward, dim=0).reshape(-1, 1).to(device)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = torch.zeros(self.batch_size, device=device, dtype=torch.long)
            next_action[non_final_mask] = agent.q_net(non_final_next_states).argmax(1)

            next_dist = torch.zeros((self.batch_size, agent.num_actions, self.atom_size), device=device)
            next_dist[non_final_mask] = agent.q_net.calc_dist(non_final_next_states)
            next_dist = next_dist[range(self.batch_size), next_action]
            
            is_done_tensor = non_final_mask.float().reshape(-1, 1)
            t_z = reward_batch +  is_done_tensor * self.discount * self.support

            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = torch.linspace(
                0, (self.batch_size - 1) * self.atom_size, self.batch_size
            ).long().unsqueeze(1).expand(self.batch_size, self.atom_size).to(device)

            proj_dist = torch.zeros(next_dist.size(), device=device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.policy_net.q_net.calc_dist(state_batch)
        log_p = torch.log(dist[range(self.batch_size), action_batch])

        loss = -(proj_dist * log_p).sum(1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        agent.q_net.reset_noise()
        self.policy_net.q_net.reset_noise()