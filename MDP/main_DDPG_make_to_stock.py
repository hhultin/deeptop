import argparse
import random
import time
from copy import deepcopy

import numpy as np
import torch

from deeptop.MDP import parameters_make_to_stock as params
from .MDP_Env import MakeToStockEnv
from .ddpg import DDPG


def sig(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DDPG for the make-to-stock environment')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=1000, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=60000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=20, type=int,
                        help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int,
                        help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=458472, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    env = MakeToStockEnv(seed=params.env_seed, customer_classes=params.customer_classes, m=params.m, B=params.B,
                         mu=params.mu)
    # initialize agent
    hidden = [128, 128]
    agent = DDPG(params.state_dim, params.action_dim, hidden, args)

    state = env.reset()
    agent.reset(state)

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)

    iteration = 0
    num_step = 0
    cumulative_reward = 0

    for t in range(params.num_iter * params.num_runs + 1):
        if t % params.num_iter == 0:
            iteration = iteration + 1
            num_step = 0
            cumulative_reward = 0
            print(f'iteration {iteration}')
            agent = DDPG(params.state_dim, params.action_dim, hidden, args)

            state = env.reset()
            agent.reset(state)

        agent.is_training = True
        num_step = num_step + 1

        # agent pick action ...
        if num_step <= args.warmup:
            action = agent.random_action()
        elif random.uniform(0, 1.0) < 0.05:
            action = agent.random_action()
        else:
            action = agent.select_action(state)

        if random.uniform(0., 1.) < sig(action[0]):
            action = [1]
        else:
            action = [0]

        # env response with next_state, reward, terminate_info
        next_state, reward, done, info = env.step(action[0])
        next_state = deepcopy(next_state)

        # agent observes and update policy
        agent.observe(reward, next_state, done)
        cumulative_reward = cumulative_reward + reward
        if num_step > args.warmup:
            agent.update_policy()
        if num_step % params.log_interval == 0:
            print(f'{cumulative_reward / params.log_interval}')
            cumulative_reward = 0
        state = deepcopy(next_state)
