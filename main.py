import gym
# import matplotlib.pyplot as plt
# import numpy as np
# from sys import getsizeof
from discretize import *
from trainer import *
from agent import DiscreteAgent, DiscreteAgent_compact, ContinuousQLearningAgent, RainbowAgent
import argparse
# from time import time

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", help='The type of agent to use', type=str)
    parser.add_argument("--num_iterations", help='The number of episodes our agent will train on', type=int)
    parser.add_argument("--eval_every", help='Run an evaluation after <> episodes', type=int)
    parser.add_argument("--noisy", help='Whether the environment is noisy or not', action='store_true')
    args = parser.parse_args()

    if args.agent not in ['td', 'ddqn', 'dueling', 'rainbow']:
        print('Argument for flag <agent> incorrect')
        exit()
    if args.num_iterations <= 0 or args.eval_every <= 0 or args.num_iterations < args.eval_every:
        print('Arguemtns <num_iterations> and <eval_every> incorrect')
        exit()
    

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
    ACTION_NUM_BUCKETS = [8, 8]

    env = gym.make('LunarLanderContinuous-v2')
    obs = env.reset()

    if args.agent == 'td':
        agent1 = DiscreteAgent_compact(NUM_BUCKETS, LIMITS, True, ACTION_NUM_BUCKETS, ACTION_LIMITS)
        trainer = TD0_Trainer(0.3, epsilon=0.8, discount=0.99, lamda=1, eps_decay=0.93)
        trainer.train(env,agent1, args.num_iterations, args.eval_every)
    elif args.agent == 'ddqn':
        agent2 = DuelingContinuousQLearningAgent(8, ACTION_NUM_BUCKETS, ACTION_LIMITS)
        #trainer = QLearningTrainer(0.01, epsilon=0.1, discount=0.9, update_freq=25)
        trainer = BatchedTrainer(0.0002, 64, 10000, epsilon=1., discount=0.99, update_freq=5, eps_decay=0.996, is_noisy=args.noisy)
        trainer.train(env, agent2, args.num_iterations, args.eval_every)

    elif args.agent == 'dueling':
        agent2 = ContinuousQLearningAgent(8, ACTION_NUM_BUCKETS, ACTION_LIMITS)
        #trainer = QLearningTrainer(0.01, epsilon=0.1, discount=0.9, update_freq=25)
        trainer = BatchedTrainer(0.0001, 64, 10000, epsilon=1., discount=0.99, update_freq=5, eps_decay=0.996, is_noisy=args.noisy)
        trainer.train(env, agent2, args.num_iterations, args.eval_every)
    elif args.agent == 'rainbow':
        v_min = -100.
        v_max = 100.
        atom_size = 51
        agent3 = RainbowAgent(8, ACTION_NUM_BUCKETS, ACTION_LIMITS, v_min, v_max, atom_size)
        # trainer = QLearningTrainer(0.01, epsilon=0.1, discount=0.9, update_freq=25)
        trainer = RainbowTrainer(0.0001, 64, 10000, epsilon=1.,   discount=0.99, update_freq=5, eps_decay=0.996, is_noisy=True)
        trainer.train(env, agent3, args.num_iterations, args.eval_every)

    