
import os
import torch
import random
import argparse
from copy import deepcopy
import itertools
import numpy as np
from model import Actor
import operator
import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
import copy
from math import ceil
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0,'./venv/')
from lineEnv import lineEnv
from neurwin_test import NeurWIN_Wrapper


def initializeEnv():
    global envs, state_dims, action_dims, state_sizes, action_sizes, nb_arms, p, q, args
    for i in range(nb_arms):
        envs.append(lineEnv(seed=(i*args.seed)+2357, N=100, OptX=99, p=p[i], q=q[i]))
        state_dims.append(1)
        action_dims.append(1)
        state_sizes.append([100])
        action_sizes.append([2])



def resetEnvs():
    global states, envs
    states.clear()
    for i in range(len(envs)):
        states.append(envs[i].reset())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NeurWIN implementation in the RMAB setting')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=1000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=60000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=87452, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--nb_arms', default=0, type=int, help='Number of arms')
    parser.add_argument('--budget', default=0, type=int, help='Budget')


    args = parser.parse_args()

    directory = (f'neurwin_training_results/arms_{args.nb_arms}_activate_{args.budget}/')

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    p = np.linspace(start=0.2, stop=0.8, num=args.nb_arms)
    q = p
    
    total_episodes = 40
    BATCH_SIZE = 5
    episodes_to_load = np.arange(0, total_episodes+BATCH_SIZE, BATCH_SIZE)

    nb_arms = args.nb_arms
    budget = args.budget

    envs = []
    states = []
    state_dims = []
    action_dims = []
    state_sizes = []
    action_sizes = []
    initializeEnv()
    #initialize agent
    hidden = [128, 128]

    EPISODE_LENGTH = 300
    agent = NeurWIN_Wrapper(nb_arms, budget, state_dims, action_dims, state_sizes, action_sizes, hidden, args)
    
    resetEnvs()

    cumulative_reward = 0

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    
    iteration = 0
    num_step = 0

    for t in range(260001):
        if t % 13000 == 0:
            iteration = iteration + 1
            num_step = 0
            print(f'iteration {iteration}')
            agent = NeurWIN_Wrapper(nb_arms, budget, state_dims, action_dims, state_sizes, action_sizes, hidden, args)
            resetEnvs()
            agent.get_agents(directory=directory, episode=0)

        agent.is_training = True
        num_step = num_step + 1
        
        # agent pick action ...
        if num_step <= args.warmup:
            action = agent.random_action()
        elif random.uniform(0, 1.0) < 0.05:
            action = agent.random_action()
        else:
            action = agent.select_action(states)

        # env response with next_state, reward, terminate_info
        next_state = []
        reward = []
        done = []
        info = []
        for i in range(len(envs)):
            next_state_i, reward_i, done_i, info_i = envs[i].step(action[i])
            next_state.append(next_state_i)
            reward.append(reward_i)
            done.append(done_i)
            info.append(info_i)
        next_state = deepcopy(next_state)

        if num_step > args.warmup:
            # agent observe and update policy

            cumulative_reward = cumulative_reward + sum(reward)
            
            if( (num_step-args.warmup)%100 == 0 ):
                print(f'{cumulative_reward/100}')
                cumulative_reward = 0

            if (int((num_step-args.warmup)%(EPISODE_LENGTH*BATCH_SIZE)) == 0):

                episode = int((num_step-args.warmup)/EPISODE_LENGTH)
                agent.get_agents(directory=directory, episode=episode)

        states = deepcopy(next_state)


