# from utils import *
from discretize import *
import random
import torch
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

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
