#!/usr/bin/python3

import copy
import os
import random
import sys
import traceback
import time

from collections import deque
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

np.random.seed(50321)
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class ActorCriticNetwork(nn.Module):

    def __init__(self, state_dim):
        EPS=3e-3
        super(ActorCriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_lim = 1

        self.fcs1 = nn.Linear(state_dim,256)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(256,128)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

        self.fca1 = nn.Linear(1,128)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fc_critic2 = nn.Linear(256,128)
        self.fc_critic2.weight.data = fanin_init(self.fc_critic2.weight.data.size())

        self.fc_critic3 = nn.Linear(128,1)
        self.fc_critic3.weight.data.uniform_(-EPS,EPS)

        self.fc_actor1 = nn.Linear(state_dim,256)
        self.fc_actor1.weight.data = fanin_init(self.fc_actor1.weight.data.size())

        self.fc_actor2 = nn.Linear(256,128)
        self.fc_actor2.weight.data = fanin_init(self.fc_actor2.weight.data.size())

        self.fc_actor3 = nn.Linear(128,64)
        self.fc_actor3.weight.data = fanin_init(self.fc_actor3.weight.data.size())

        self.fc_actor4 = nn.Linear(64,1)
        self.fc_actor4.weight.data.uniform_(-EPS,EPS)

    def critic(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,1] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        state = state.float()
        action = action.float()
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s2,a1),dim=1)
        x = F.relu(self.fc_critic2(x))
        x = self.fc_critic3(x)
        return x


    def actor(self, state):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,1] )
        """
        state = state.float()
        x = F.relu(self.fc_actor1(state))
        x = F.relu(self.fc_actor2(x))
        x = F.relu(self.fc_actor3(x))
        action = torch.sigmoid(self.fc_actor4(x))
        action = action * self.action_lim
        return action

    def forward(self, state):
        a = self.actor(state)
        v = self.critic(state, a)
        return a, v


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} FOLDER_NAME")
        exit(1)
    else:
        out_dir = sys.argv[1]
        os.makedirs(out_dir, exist_ok = True)
    discount = 0.95
    n_episodes = 100
    n_agents = 2
    ai = np.array([2.0, 2.0])
    a0 = 0
    mu = 0.25
    c = 1
    batch_size = 128
    initial_lr = 3e-4
    tau = 0.001
    lr_decay = 0.995
    nash_price = 1.47
    beta = 3e-3
    seeds = [12345]#54321, 50321, 9827391, 8534101, 4305430, 12654329, 3483055, 348203, 2356, 250917, 200822, 151515, 50505, 301524]
    max_t = int(1e3)
    ir_periods = 20
    #torch.autograd.set_detect_anomaly(True)
    for session in range(len(seeds)):
        # Random seeds
        np.random.seed(seeds[session])
        torch.manual_seed(seeds[session])
        # Initialize replay buffers and NNs
        buffer = []
        net = []
        for i in range(n_agents):
            net.append(ActorCriticNetwork(n_agents))
            buffer.append(deque(maxlen=int(max_t / 10)))
        # Do a deepcopy of the NN to ensure target networks are initialized the same way
        target_net = copy.deepcopy(net)
        # One optimizer per network
        optimizer = [torch.optim.AdamW(n.parameters(), lr=initial_lr) for n in net]
        # One scheduler per network
        scheduler = [torch.optim.lr_scheduler.ExponentialLR(o, gamma = lr_decay) for o in optimizer]
        # Initial state is random, but ensure prices are above marginal cost
        state = torch.sigmoid(torch.randn(n_agents)) + c
        price = np.zeros([n_agents])
        total_reward = np.zeros([n_agents, max_t])
        price_history = np.zeros([n_agents, max_t])
        collusive_t = 0
        for t in tqdm(range(max_t)):
            for i in range(n_agents):
                p = net[i].actor(state)
                price[i] = (p + c + D.Normal(0, np.exp(- beta * t)).sample()).detach().numpy()
                price_history[i, t] = price[i]
            quant = np.exp((ai - price)/mu) / (np.sum(np.exp((ai - price)/mu)) + np.exp(a0/mu))
            profits = (price - c) * quant
            for i in range(n_agents):
                transition = state.detach().numpy(), price[i], profits[i], price
                buffer[i].append(transition)
                total_reward[i, t] = profits[i]
            state = torch.tensor(price)
            if t > batch_size:
                for i in range(n_agents):
                    batch = []
                    batch = random.sample(buffer[i], min(len(buffer[i]), batch_size))
                    s = torch.tensor(np.float32([arr[0] for arr in batch]))
                    a = torch.tensor(np.float32([arr[1] for arr in batch]))
                    r = torch.tensor(np.float32([arr[2] for arr in batch]))
                    s1 = torch.tensor(np.float32([arr[3] for arr in batch]))
                    a1 = target_net[i].actor(s1).detach()
                    next_val = target_net[i].critic(s1, a1).detach().squeeze()
                    y_exp = r + discount * next_val
                    y_pred = net[i].critic(s, a.unsqueeze(0).mT).squeeze()
                    l_critic = F.mse_loss(y_pred, y_exp)
                    _, pred_v = net[i](s)
                    l_actor = -torch.sum(pred_v)
                    l = l_critic + l_actor
                    optimizer[i].zero_grad()
                    l.backward()
                    optimizer[i].step()
                    scheduler[i].step()
                    for target_param, param in zip(target_net[i].parameters(), net[i].parameters()):
                        target_param.data.copy_(
                            target_param.data * (1.0 - tau) + param.data * tau
                        )
        np.save(f"{out_dir}/session_reward_{seeds[session]}.npy", total_reward)
        ir_profits = np.zeros([n_agents, ir_periods])
        ir_prices = np.zeros([n_agents, ir_periods])
        # Impulse response
        for t in range(ir_periods):
            for i in range(n_agents):
                p = net[i].actor(state)
                price[i] = (p + c).detach().numpy()
                ir_prices[i, t] = price[i]
            if t == (ir_periods / 2):
                price[0] = nash_price
                ir_prices[0, t] = nash_price
            quant = np.exp((ai - price)/mu) / (np.sum(np.exp((ai - price)/mu)) + np.exp(a0/mu))
            ir_profits[:, t] = (price - c) * quant
            state = torch.tensor(price)
        np.save(f"{out_dir}/ir_profits_{seeds[session]}.npy", ir_profits)
        np.save(f"{out_dir}/ir_prices_{seeds[session]}.npy", ir_prices)









if __name__ == "__main__":
    main()
