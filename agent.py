# from utils import *
from discretize import *
import random
import itertools
import torch
import torch.nn as nn
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

class Agent:
    def __init__(self):
        self.mode = 'train'

    def predict(self, observation, epsilon):
        # returns action
        if self.mode=='train':
            pass # epsilon-greedy
        else:
            pass # greedy
        pass


class DiscreteAgent(Agent):
    def __init__(self, n_obs_buckets, obs_ranges, is_q, n_action_buckets, action_ranges):
        super().__init__()
        self.n_obs_buckets = n_obs_buckets
        self.obs_ranges = obs_ranges
        self.is_q = is_q
        self.n_action_buckets = n_action_buckets
        self.action_ranges = action_ranges

        if is_q:
            table, buckets = make_table_and_buckets(n_obs_buckets+n_action_buckets, obs_ranges+action_ranges)
        else:
            table, buckets = make_table_and_buckets(n_obs_buckets, obs_ranges)

        self.table = torch.tensor(table,device=device) # maybe need to be changed, represents either V(s) or Q(s,a)
        self.buckets = buckets


    def predict(self, observation, epsilon=0):
        if self.mode == 'train':
            # epsilon-greedy
            if random.random() >= epsilon:
                if self.is_q:
                    indices = tuple(observation_to_bucket(observation, self.buckets))
                    obs_slice = self.table[indices]
                    action_indices = np.unravel_index(np.argmax(obs_slice.to(device=torch.device('cpu')), axis=None), obs_slice.shape)
                    dim_actions = len(self.n_action_buckets)  # = 2
                    predicted_action = np.zeros(dim_actions)
                    last_buckets = self.buckets[-dim_actions:]
                    for d in range(dim_actions):
                        predicted_action[d] = last_buckets[d][action_indices[d]]
                else:
                    predicted_action = 0 # implement!
            else:
                dim_actions = len(self.n_action_buckets) # = 2
                last_buckets = self.buckets[-dim_actions:]
                random_actions = np.zeros(dim_actions)
                for d in range(dim_actions):
                    random_actions[d] = np.random.choice(last_buckets[d])
                predicted_action = random_actions
        elif self.mode=='test':
            # greedy
            if self.is_q:
                indices = tuple(observation_to_bucket(observation, self.buckets))
                obs_slice = self.table[tuple(indices)]
                action_indices = np.unravel_index(np.argmax(obs_slice.to(device=torch.device('cpu')), axis=None), obs_slice.shape)
                dim_actions = len(self.n_action_buckets)  # = 2
                predicted_action = np.zeros(dim_actions)
                last_buckets = self.buckets[-dim_actions:]
                for d in range(dim_actions):
                    predicted_action[d] = last_buckets[d][action_indices[d]]
            else:
                predicted_action = 0 # implement!
        else: raise Exception("Invalid mode!")

        return predicted_action


class ContinuousQLearningAgent(Agent):
    def __init__(self, obs_size, n_action_buckets, action_ranges):
        super().__init__()


        self.obs_size = obs_size

        _, buckets = make_table_and_buckets(n_action_buckets, action_ranges)
        buckets = [list(arr) for arr in buckets]
        self.actions = list(itertools.product(*buckets))
        # get reverse mapping from list
        self.action_to_index = {
            action:i for i, action in enumerate(self.actions)
        }
        self.num_actions = len(self.actions)


        self.q_net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        ).to(device)


    def predict(self, observation, epsilon=0):
        observation = torch.tensor(observation).to(device)
        if len(observation.size()) == 1:
            observation = observation[None, :]
            
        if self.mode == 'train':
            if random.random() >= epsilon:
                q_values = self.q_net(observation)
                (_, predicted_index)= torch.max(q_values, dim=1)
                predicted_action = self.actions[predicted_index]
            else:
                # generate random
                predicted_index = random.randint(0, self.num_actions-1)
                predicted_action = self.actions[predicted_index]
                q_values = self.q_net(observation)
            return predicted_action, q_values

        elif self.mode == 'test':
            q_values = self.q_net(observation)
            (_, predicted_index)= torch.max(q_values, dim=1)
            predicted_action = self.actions[predicted_index]
            return predicted_action

        else: raise Exception("Invalid mode!")


class DuelingNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.body = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.v_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        self.a_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):

        x = self.body(state)
        val = self.v_head(x)
        adv = self.a_head(x)
        # Q = V + A(a) - 1/|A| * sum A(a')
        output = val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.action_size)
        return output

class DuelingContinuousQLearningAgent(ContinuousQLearningAgent):
    def __init__(self, obs_size, n_action_buckets, action_ranges):
        super().__init__(obs_size, n_action_buckets, action_ranges)


        self.obs_size = obs_size

        _, buckets = make_table_and_buckets(n_action_buckets, action_ranges)
        buckets = [list(arr) for arr in buckets]
        self.actions = list(itertools.product(*buckets))
        # get reverse mapping from list
        self.action_to_index = {
            action:i for i, action in enumerate(self.actions)
        }
        self.num_actions = len(self.actions)


        self.q_net = DuelingNet(obs_size, self.num_actions).to(device)



class DiscreteAgent_compact(Agent):
    def __init__(self, n_obs_buckets, obs_ranges, is_q, n_action_buckets, action_ranges):
        super().__init__()
        self.n_obs_buckets = n_obs_buckets
        self.obs_ranges = obs_ranges
        # self.is_q = is_q
        self.n_action_buckets = n_action_buckets
        self.action_ranges = action_ranges

        table, buckets = compact_Q_table(n_obs_buckets, obs_ranges, n_action_buckets,action_ranges, 'concat')
        self.table = torch.tensor(table,device=device) # represents Q(s,a)
        self.buckets = buckets


    def predict(self, observation, epsilon=0):
        if self.mode == 'train':
            # epsilon-greedy
            if random.random() >= epsilon: # exploitation
                indices = tuple(observation_to_bucket(observation, self.buckets))
                s_index = buckets2index(self.buckets[:-2], self.n_obs_buckets, indices, observation)
                a_index = torch.argmax(self.table[s_index])

                # assert len(self.n_action_buckets) == 2
                action1 = a_index // (self.n_action_buckets[1]+2)
                action2 = a_index % (self.n_action_buckets[1]+2)
                predicted_action = np.array([self.buckets[-2][action1-1], self.buckets[-1][action2-1]])

            else: # exploration
                random_action1 = np.random.choice(self.buckets[-2])
                random_action2 = np.random.choice(self.buckets[-1])

                predicted_action = np.array([random_action1, random_action2])
        elif self.mode=='test':
            # always greedy
            indices = tuple(observation_to_bucket(observation, self.buckets))
            s_index = buckets2index(self.buckets[:-2], self.n_obs_buckets, indices, observation)
            a_index = torch.argmax(self.table[s_index])

            # assert len(self.n_action_buckets) == 2
            action1 = a_index // (self.n_action_buckets[1] + 2)
            action2 = a_index % (self.n_action_buckets[1] + 2)
            predicted_action = np.array([self.buckets[-2][action1 - 1], self.buckets[-1][action2 - 1]])
        else: raise Exception("Invalid mode!")

        return predicted_action
