import numpy as np


def make_table_and_buckets(num_buckets, ranges):
    """
    :param ranges: list of tuples [(start,end)]
    :param num_buckets: number of buckets in each dimension
    :return: table, buckets
    """
    table = np.zeros(np.array(num_buckets)+2)
    buckets = [np.linspace(r[0],r[1],num_buckets[i]+1) for i,r in enumerate(ranges)]
    return table, buckets

def observation_to_bucket(obs, buckets):
    assert len(obs)==len(buckets)
    output = np.zeros(len(obs),dtype=int)
    for i in range(len(obs)):
        output[i] = int(np.digitize([obs[i]], buckets[i])[0])
    return output


# ranges = [(-1, 1), (0, 5)]
# ranges2 = [(-0.1,1.1)]
# num_buckets = [10, 10]
# num_buckets2 = [2]
# table, buckets = make_table_and_buckets(num_buckets2,ranges2)
# print(buckets)
# print(observation_to_bucket([-1.1], buckets))