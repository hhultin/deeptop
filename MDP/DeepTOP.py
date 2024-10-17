

import random

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from .model import (Actor, Critic)
from .memory import SequentialMemory
from .random_process import OrnsteinUhlenbeckProcess
from .util import *


criterion = nn.MSELoss()


class DeepTOP_MDP(object):
    # state_dim: the dimension of the vector state. IMPORTANT: not including the scalar state
    # action_dim: the dimension of actions. This should be 1
    # hidden: a list of number of neurons in each hidden layer
    def __init__(self, state_dim, action_dim, hidden, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # Create Actor and Critic Networks
        self.actor = Actor(self.state_dim, 1, hidden)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)
        self.critic = Critic(self.state_dim + 1, 1, hidden)  # Input is both the scalar state and the vector state
        self.critic_target = Critic(self.state_dim + 1, 1, hidden)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)

        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        
        self.epsilon = 1.0
        self.is_training = True

        
        if self.device == torch.device('cuda'): 
            self.cuda()

    def update_policy(self):
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        vector_batch = []  # The batch of vector state
        scalar_batch = []  # The batch of scalar state
        next_vector_batch = []
        next_scalar_batch = []
        
        for i in range(self.batch_size):
            vector_batch.append(list(state_batch[i]))
            next_vector_batch.append(list(next_state_batch[i]))
            scalar_batch.append([vector_batch[i].pop(0)])
            next_scalar_batch.append([next_vector_batch[i].pop(0)])

        # Convert all batches to arrays
        vector_batch = torch.FloatTensor(np.array(vector_batch)).to(self.device)
        scalar_batch = torch.FloatTensor(np.array(scalar_batch)).to(self.device)
        next_vector_batch = torch.FloatTensor(np.array(next_vector_batch)).to(self.device)
        next_scalar_batch = torch.FloatTensor(np.array(next_scalar_batch)).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)

        with torch.no_grad():
            critic_plus = self.critic_target([next_vector_batch,
                                              next_scalar_batch,
                                              to_tensor(np.ones((self.batch_size, 1), dtype=int)).to(self.device)]).cpu()
            critic_minus = self.critic_target([next_vector_batch,
                                               next_scalar_batch,
                                               to_tensor(np.zeros((self.batch_size, 1), dtype=int)).to(self.device)]).cpu()
            next_action_batch = torch.FloatTensor(torch.clamp(torch.sign(critic_plus - critic_minus), min=0.0)).to(self.device)
            
            # Prepare for the target Q batch
            next_q_values = self.critic_target([next_vector_batch,
                                                next_scalar_batch,
                                                next_action_batch])
            
            target_q_batch = reward_batch + self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()
        q_batch = self.critic([vector_batch, scalar_batch, 
                               action_batch])
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        q_diff_batch = self.critic([vector_batch, self.actor(vector_batch),
                                    to_tensor(np.ones((self.batch_size, 1), dtype=int))]) - \
                       self.critic([vector_batch, self.actor(vector_batch),
                                    to_tensor(np.zeros((self.batch_size, 1), dtype=int))])
        
        q_diff_batch = q_diff_batch.detach().cpu().numpy()

        policy_loss = -to_tensor(q_diff_batch).to(self.device) * self.actor(vector_batch)
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        torch.cuda.set_device(0) # specify which gpu to train on 
        self.actor.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = [random.randint(0, 1)]
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        vector = s_t.copy()
        scalar = vector.pop(0)
        threshold = self.actor.forward(torch.FloatTensor(vector).to(self.device)).cpu().item()

        if threshold > scalar:
            action = [1]
        else:
            action = [0]
        self.a_t = action
        return action
        

    def reset(self, obs):
        self.s_t = obs

