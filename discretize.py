import numpy as np
from itertools import product
# from scipy.stats import binned_statistic_2d


def make_table_and_buckets(num_buckets, ranges):
    """
    :param num_buckets: number of buckets in each dimension
    :param ranges: list of tuples [(start,end)]
    :return: table - a matrix of (num_buckets+2) dimensions; buckets - a list of linspaces.
    """
    assert len(num_buckets)==len(ranges)
    # table = np.random.randn(np.array(num_buckets)+2)
    table = np.zeros(np.array(num_buckets)+2)
    buckets = [np.linspace(r[0],r[1],num_buckets[i]+1) for i,r in enumerate(ranges)]
    return table, buckets


def observation_to_bucket(obs, buckets):
    """
    match indices/buckets to given observations
    :param obs: an array of obseravtion values
    :param buckets: a list of arrays, each represents the axis's bin.
    :return: an array of the corresponding indices
    """
    # assert len(obs)==len(buckets)
    output = np.zeros(len(obs),dtype=int)
    for i in range(len(obs)):
        output[i] = int(np.digitize([obs[i]], buckets[i])[0])
    return output


def compact_Q_table(num_buckets_obs, ranges_obs, num_buckets_actions, ranges_actions, mode:str):
    """
    mode must be in {'obs', 'act', 'concat'}
    """
    assert len(num_buckets_obs) == len(ranges_obs)
    assert len(num_buckets_actions) == len(ranges_actions)
    p_obs = np.prod(np.array(num_buckets_obs)+2)
    p_actions = np.prod(np.array(num_buckets_actions)+2)
    scale=0.05
    table = scale*(2*np.random.rand(p_obs,p_actions)-1)

    t1 = [np.round(np.linspace(r[0],r[1],num_buckets_obs[i]+1),3).tolist() for i,r in enumerate(ranges_obs)]
    t2 = [np.round(np.linspace(r[0],r[1],num_buckets_actions[i]+1),3).tolist() for i,r in enumerate(ranges_actions)]
    buckets_obs = np.array(t1)
    buckets_actions = np.array(t2)
    buckets_concat = np.array(t1+t2)

    if mode=='obs': result = buckets_obs
    elif mode=='act': result = buckets_actions
    elif mode=='concat': result = buckets_concat
    else: raise Exception("Invalid mode!!!")

    return table, result # choose to return buckets_obs, buckets_actions or buckets_concat


def buckets2index(buckets, buckets_lengths, indices, observation):
    buckets_lengths = np.array(buckets_lengths) + 1
    index = 0  # indices[-1]-1
    for j,idx in enumerate(indices): # instead of enumerate(indices[:-1])
        offset = np.prod(buckets_lengths[j+1:])
        index += offset * (idx-1)
    # grid = list(product(*buckets.tolist())) # observation corresponds to the bin grid[index]
    return index
    # ret = binned_statistic_2d() # unnecessary




# ranges = [(-1, 1), (0, 5)]
# ranges2 = [(-0.1,1.1)]
# num_buckets = [10, 10]
# num_buckets2 = [2]
# table, buckets = make_table_and_buckets(num_buckets2,ranges2)
# print(buckets)
# print(observation_to_bucket([-1.1], buckets))