import gym
# import matplotlib.pyplot as plt
# import numpy as np
# from sys import getsizeof
from discretize import *
from trainer import *
from agent import DiscreteAgent, DiscreteAgent_compact, ContinuousQLearningAgent
# from time import time

if __name__=='__main__':
    X_LIMIT = (-1.2, 1.2)
    Y_LIMIT = (0.0, 1.5)
    X_VELOCITY = (-3.0, 3.0)
    Y_VELOCITY = (-3.0, 3.0)
    ANGLE_LIMIT = (-2.0, 2.0)
    ANGULAR_VELOCITY = (-2.0, 2.0)
    IS_LEFT_TOUCH_GROUND = (-0.1,1.1)
    IS_RIGHT_TOUCH_GROUND = (-0.1,1.1)
    LIMITS = [X_LIMIT,
              Y_LIMIT,
              X_VELOCITY,
              Y_VELOCITY,
              ANGLE_LIMIT,
              ANGULAR_VELOCITY,
              IS_LEFT_TOUCH_GROUND,
              IS_RIGHT_TOUCH_GROUND]

    NUM_BUCKETS = [4, 4, 3, 3, 4, 4, 2, 2] # [5, 5, 5, 5, 2]

    MAIN_ENGINE_LIMIT = (-1, 1)
    LEFT_RIGHT_ENGINE_LIMIT = (-1, 1)
    ACTION_LIMITS = [MAIN_ENGINE_LIMIT, LEFT_RIGHT_ENGINE_LIMIT]
    ACTION_NUM_BUCKETS = [4,4]

    # table, buckets = make_table_and_buckets(NUM_BUCKETS, LIMITS)
    # table2, buckets2 = compact_Q_table(NUM_BUCKETS, LIMITS, ACTION_NUM_BUCKETS, ACTION_LIMITS, 'concat')

    env = gym.make('LunarLanderContinuous-v2')
    obs = env.reset()
    env.render()

    # action = env.action_space.sample()
    # observation, reward, done, info = env.step(action)

    # table += 2.0

    # print(getsizeof(table))
    # print(np.prod(NUM_BUCKETS))
    #
    # fig2 = plt.figure()
    #
    # ax1 = fig2.add_subplot(211)
    # ax2 = fig2.add_subplot(212)
    #
    # cords = []
    # for t in range(1000):
    #     action = env.action_space.sample()
    #     action[1] = 1.0
    #     print('Action: ', np.round(action,4))
    #     observation, reward, done, info = env.step(action)
    #
    #     # if observation[1] >= Y_LIMIT:
    #     #     done = True
    #     #     reward = -100
    #
    #     cords.append((observation[4], observation[5], t))
    #     print('Observation: ', np.round(observation,decimals=4), ' Reward: ', np.round(reward,4))
    #     if done:
    #         print(f'Episode finished after {t+1} timesteps')
    #         break
    # env.close()
    #
    #
    #
    # xs = [cord[0] for cord in cords]
    # ys = [cord[1] for cord in cords]
    # zs = [cord[2] for cord in cords]
    #
    # ax1.scatter(zs, xs)
    # # ax1.yaxis('x')
    # ax2.scatter(zs, ys)
    # # ax2.yaxis('y')
    #
    # plt.show()

    pass
    # agent1 = DiscreteAgent(NUM_BUCKETS, LIMITS, True, ACTION_NUM_BUCKETS, ACTION_LIMITS)
    # agent1 = DiscreteAgent_compact(NUM_BUCKETS, LIMITS, True, ACTION_NUM_BUCKETS, ACTION_LIMITS)
    # trainer = TD0_Trainer(0.3, epsilon=0.3, discount=0.99, lamda=1)
    # trainer.train(env,agent1,2000,25)

    agent2 = ContinuousQLearningAgent(8, ACTION_NUM_BUCKETS, ACTION_LIMITS)
    # trainer = QLearningTrainer(0.01, epsilon=0.1, discount=0.9, update_freq=25)
    trainer = BatchedTrainer(0.01, 128, 10000, epsilon=0.1, discount=0.9, update_freq=10)
    trainer.train(env, agent2, 1000, 50)

