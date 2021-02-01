import gym
import matplotlib.pyplot as plt
import numpy as np
from sys import getsizeof
from discretize import *

Y_LIMIT = (0.0, 5.0)
ANGLE_LIMIT = (-10.0, 10.0)
ANGULAR_VELOCITY = (-5.0, 5.0)
LIMITS = [(-1, 1), Y_LIMIT, (-5.0, 5.0), (-5.0, 5.0), ANGLE_LIMIT, ANGULAR_VELOCITY, (-0.1,1.1), (-0.1,1.1)]

NUM_BUCKETS = [20, 20, 10, 10, 20, 20, 2, 2]
table, buckets = make_table_and_buckets(NUM_BUCKETS, LIMITS)
table += 2.0
print(getsizeof(table))
print(np.prod(NUM_BUCKETS))

fig2 = plt.figure()

ax1 = fig2.add_subplot(211)
ax2 = fig2.add_subplot(212)

env = gym.make('LunarLanderContinuous-v2')
env.reset()
cords = []
for t in range(1000):
    action = env.action_space.sample()
    action[1] = 1.0
    print('Action: ', np.round(action,4))
    observation, reward, done, info = env.step(action)

    # if observation[1] >= Y_LIMIT:
    #     done = True
    #     reward = -100

    cords.append((observation[4], observation[5], t))
    print('Observation: ', np.round(observation,decimals=4), ' Reward: ', np.round(reward,4))
    if done:
        print(f'Episode finished after {t+1} timesteps')
        break
env.close()



xs = [cord[0] for cord in cords]
ys = [cord[1] for cord in cords]
zs = [cord[2] for cord in cords]

ax1.scatter(zs, xs)
# ax1.yaxis('x')
ax2.scatter(zs, ys)
# ax2.yaxis('y')


plt.show()
