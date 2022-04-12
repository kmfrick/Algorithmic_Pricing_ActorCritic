#!/usr/bin/python3

import copy
import os
import random
import sys

from collections import deque
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

np.random.seed(50321)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


"""
Actor-Critic NN
Adapted from https://github.com/vy007vikas/PyTorch-ActorCriticRL
"""


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim):
        EPS = 3e-3
        super(ActorCriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_lim = 1
        # Critic layers
        self.fcs1 = nn.Linear(state_dim, 256)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(256, 128)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())
        self.fca1 = nn.Linear(1, 128)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        self.fc_critic2 = nn.Linear(256, 128)
        self.fc_critic2.weight.data = fanin_init(self.fc_critic2.weight.data.size())
        self.fc_critic3 = nn.Linear(128, 1)
        self.fc_critic3.weight.data.uniform_(-EPS, EPS)
        # Actor layers
        self.fc_actor1 = nn.Linear(state_dim, 256)
        self.fc_actor1.weight.data = fanin_init(self.fc_actor1.weight.data.size())
        self.fc_actor2 = nn.Linear(256, 128)
        self.fc_actor2.weight.data = fanin_init(self.fc_actor2.weight.data.size())
        self.fc_actor3 = nn.Linear(128, 64)
        self.fc_actor3.weight.data = fanin_init(self.fc_actor3.weight.data.size())
        self.fc_actor4 = nn.Linear(64, 1)
        self.fc_actor4.weight.data.uniform_(-EPS, EPS)

    def critic(self, state, action):
        state = state.float()
        action = action.float()
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s2, a1), dim=1)
        x = F.relu(self.fc_critic2(x))
        x = self.fc_critic3(x)
        return x

    def actor(self, state):
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


def compute_profit(ai, a0, mu, c, p):
    q = np.exp((ai - p) / mu) / (np.sum(np.exp((ai - p) / mu)) + np.exp(a0 / mu))
    pi = (p - c) * q
    return pi


def compute_profit_derivative(ai, a0, mu, c, p, h):
    def f(price):
        return compute_profit(ai, a0, mu, c, price)

    return (f(p + h) - f(p - h)) / (2 * h)


def optimize(batch_size, buffer, net, optimizer, scheduler, target_net):
    DISCOUNT = 0.95
    TARGET_LR = 0.001
    batch = random.sample(buffer, min(len(buffer), batch_size))
    s = torch.tensor(np.float32([arr[0] for arr in batch]))
    a = torch.tensor(np.float32([arr[1] for arr in batch]))
    r = torch.tensor(np.float32([arr[2] for arr in batch]))
    s1 = torch.tensor(np.float32([arr[3] for arr in batch]))
    a1 = target_net.actor(s1).detach()
    next_val = target_net.critic(s1, a1).detach().squeeze()
    y_exp = r + DISCOUNT * next_val
    y_pred = net.critic(s, a.unsqueeze(0).mT).squeeze()
    l_critic = F.smooth_l1_loss(y_pred, y_exp)
    _, pred_v = net(s)
    l_actor = -torch.sum(pred_v)
    loss = l_critic + l_actor
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(target_net.parameters(), 1)
    optimizer.step()
    scheduler.step()
    total_norm = 0
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - TARGET_LR) + param.data * TARGET_LR)
        param_norm = param.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return scheduler.get_last_lr()[0], total_norm


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} FOLDER_NAME")
        exit(1)
    else:
        out_dir = sys.argv[1]
        os.makedirs(out_dir, exist_ok=True)
    n_agents = 2
    ai = np.array([2.0, 2.0])
    a0 = 0
    mu = 0.25
    c = 1
    BATCH_SIZE = 128
    INITIAL_LR = 3e-4
    LR_DECAY = 0.99
    EXPLORATION_DECAY = 3e-3
    MAX_T = int(1e3)
    nash_price = 1.47293
    seeds = [12345, 54321, 50321, 9827391, 8534101, 4305430, 12654329, 3483055, 348203, 2356, 250917, 200822, 151515,
             50505, 301524]

    ir_periods = 20
    # torch.autograd.set_detect_anomaly(True)
    for session in range(len(seeds)):
        # Random seeds
        np.random.seed(seeds[session])
        torch.manual_seed(seeds[session])
        # Initialize replay buffers and NNs
        buffer = []
        net = []
        for i in range(n_agents):
            net.append(ActorCriticNetwork(n_agents))
            buffer.append(deque(maxlen=BATCH_SIZE * 2))
        # Do a deepcopy of the NN to ensure target networks are initialized the same way
        target_net = copy.deepcopy(net)
        # One optimizer per network
        optimizer = [torch.optim.AdamW(n.parameters(), lr=INITIAL_LR) for n in net]
        # One scheduler per network
        scheduler = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=LR_DECAY) for o in optimizer]
        # Initial state is random, but ensure prices are above marginal cost
        state = torch.sigmoid(torch.randn(n_agents)) + c
        price = np.zeros([n_agents])
        total_reward = np.zeros([n_agents, MAX_T])
        price_history = np.zeros([n_agents, MAX_T])
        grad_norm = np.zeros([n_agents, MAX_T])
        net_lr = np.zeros([n_agents, MAX_T])
        for t in tqdm(range(MAX_T)):
            for i in range(n_agents):
                p = net[i].actor(state)
                price[i] = (p + c + D.Normal(0, np.exp(- EXPLORATION_DECAY * t)).sample()).detach().numpy()
                price_history[i, t] = price[i]
            profits = compute_profit(ai, a0, mu, c, price)
            for i in range(n_agents):
                transition = state.detach().numpy(), price[i], profits[i], price
                buffer[i].append(transition)
                total_reward[i, t] = profits[i]
            state = torch.tensor(price)
            if t > 0:  # At least one experience in the buffer
                for i in range(n_agents):
                    nlr, gn = optimize(BATCH_SIZE, buffer[i], net[i], optimizer[i], scheduler[i], target_net[i])
                    net_lr[i, t] = nlr
                    grad_norm[i, t] = gn
        np.save(f"{out_dir}/session_reward_{seeds[session]}.npy", total_reward)
        np.save(f"{out_dir}/grad_norm_{seeds[session]}.npy", grad_norm)
        np.save(f"{out_dir}/net_lr_{seeds[session]}.npy", net_lr)
        # Impulse response
        ir_profits = np.zeros([n_agents, ir_periods])
        ir_prices = np.zeros([n_agents, ir_periods])
        for t in range(ir_periods):
            for i in range(n_agents):
                p = net[i].actor(state)
                price[i] = (p + c).detach().numpy()
                ir_prices[i, t] = price[i]
            if (ir_periods / 2) <= t <= (ir_periods / 2 + 3):
                price[0] = nash_price
                ir_prices[0, t] = price[0]
            ir_profits[:, t] = compute_profit(ai, a0, mu, c, price)
            state = torch.tensor(price)
        np.save(f"{out_dir}/ir_profits_{seeds[session]}.npy", ir_profits)
        np.save(f"{out_dir}/ir_prices_{seeds[session]}.npy", ir_prices)
        # state-action mappings
        w = np.linspace(c, c + 2, 100)
        x = np.linspace(c, c + 2, 100)
        W = np.transpose([np.repeat(w, len(x)), np.tile(x, len(w))])
        for i in range(n_agents):
            # FIXME: Check the layout of what is saved, it's probably rotated wrong
            a, q = net[i](torch.tensor(W))
            a = a.detach().numpy()
            q = q.detach().numpy()
            np.save(f"{out_dir}/actions_{seeds[session]}_{i}.npy", a)
            np.save(f"{out_dir}/values_{seeds[session]}_{i}.npy", a)


if __name__ == "__main__":
    main()
