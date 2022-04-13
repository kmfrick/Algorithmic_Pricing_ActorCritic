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
        # Common first layer
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        # Critic layers
        self.fc_critic_a = nn.Linear(1, 128)
        self.fc_critic_a.weight.data = fanin_init(self.fc_critic_a.weight.data.size())
        self.fc_critic2 = nn.Linear(256, 128)
        self.fc_critic2.weight.data = fanin_init(self.fc_critic2.weight.data.size())
        self.fc_critic3 = nn.Linear(128, 1)
        self.fc_critic3.weight.data.uniform_(-EPS, EPS)
        # Actor layers
        self.fc_actor3 = nn.Linear(128, 64)
        self.fc_actor3.weight.data = fanin_init(self.fc_actor3.weight.data.size())
        self.fc_actor4 = nn.Linear(64, 1)
        self.fc_actor4.weight.data.uniform_(-EPS, EPS)

    def critic(self, state, action):
        state = state.float()
        action = action.float()
        s1 = F.relu(self.fc1(state))
        s2 = F.relu(self.fc2(s1))
        a1 = F.relu(self.fc_critic_a(action))
        x = torch.cat((s2, a1), dim=1)
        x = F.relu(self.fc_critic2(x))
        x = self.fc_critic3(x)
        return x

    def actor(self, state):
        state = state.float()
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
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
    #print(f"exp = {torch.mean(y_exp)}")
    y_pred = net.critic(s, a.unsqueeze(0).mT).squeeze()
    #print(f"pred = {torch.mean(y_pred)}")
    l_critic = F.smooth_l1_loss(y_pred, y_exp)
    optimizer.zero_grad()
    l_critic.backward()
    #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    #torch.nn.utils.clip_grad_norm_(target_net.parameters(), 1)
    optimizer.step()
    pred_a = net.actor(s)
    pred_v = net.critic(s, pred_a)
    l_actor = -torch.sum(pred_v)
    optimizer.zero_grad()
    l_actor.backward()
    #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    #torch.nn.utils.clip_grad_norm_(target_net.parameters(), 1)
    optimizer.step()
    # Decay LR
    scheduler.step()
    total_norm = 0
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - TARGET_LR) + param.data * TARGET_LR)
        param_norm = param.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return l_critic.item(), scheduler.get_last_lr()[0], total_norm


def save_actions(net, n_agents, c, fpostfix, out_dir):
    grid_size = 100
    w = np.linspace(c, c + 2, grid_size)
    for i in range(n_agents):
        A = np.zeros([grid_size, grid_size])
        Q = np.zeros([grid_size, grid_size])
        for ai, p1 in enumerate(w):
            for aj, p2 in enumerate(w):
                state = torch.tensor(np.array([p1, p2]))
                a = net[i].actor(state)
                A[ai, aj] = a.detach().numpy()
        np.save(f"{out_dir}/actions_{fpostfix}_{i}.npy", A)
        #np.save(f"{out_dir}/values_{fpostfix}_{i}.npy", Q)

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
    LR_DECAY = 0.9999
    EXPLORATION_DECAY = 3e-3
    MAX_T = int(1e3)
    nash_price = 1.47293
    seeds = [12345, 54321, 50321, 9827391, 8534101]#, 4305430, 12654329, 3483055, 348203, 2356, 250917, 200822, 151515, 50505, 301524]

    ir_periods = 20
    # torch.autograd.set_detect_anomaly(True)
    for session in range(len(seeds)):
        fpostfix = seeds[session]
        # Random seeds
        np.random.seed(seeds[session])
        torch.manual_seed(seeds[session])
        # Initialize replay buffers and NNs
        buffer = []
        net = []
        target_net = []
        for i in range(n_agents):
            net.append(ActorCriticNetwork(n_agents))
            target_net.append(ActorCriticNetwork(n_agents))
            for target_param, param in zip(target_net[i].parameters(), net[i].parameters()):
                target_param.data.copy_(param.data)
            buffer.append(deque(maxlen=BATCH_SIZE * 2))
        # Initial state is random, but ensure prices are above marginal cost
        state = torch.sigmoid(torch.randn(n_agents)) + c
        print(state)
        # initial state-action mappings
        save_actions(net, n_agents, c, f"initial_{fpostfix}", out_dir)
        # One optimizer per network
        optimizer = [torch.optim.AdamW(n.parameters(), lr=INITIAL_LR) for n in net]
        # One scheduler per network
        scheduler = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=LR_DECAY) for o in optimizer]

        price = np.zeros([n_agents])
        total_reward = np.zeros([n_agents, MAX_T])
        price_history = np.zeros([n_agents, MAX_T])
        grad_norm = np.zeros([n_agents, MAX_T])
        net_lr = np.zeros([n_agents, MAX_T])
        net_lr[:, 0] = INITIAL_LR
        critic_loss = np.zeros([n_agents, MAX_T])
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
                    cl, nlr, gn = optimize(BATCH_SIZE, buffer[i], net[i], optimizer[i], scheduler[i], target_net[i])
                    net_lr[i, t] = nlr
                    grad_norm[i, t] = gn
                    critic_loss[i, t] = cl
        np.save(f"{out_dir}/session_reward_{fpostfix}.npy", total_reward)
        np.save(f"{out_dir}/grad_norm_{fpostfix}.npy", grad_norm)
        np.save(f"{out_dir}/net_lr_{fpostfix}.npy", net_lr)
        np.save(f"{out_dir}/critic_loss_{fpostfix}.npy", critic_loss)
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
        np.save(f"{out_dir}/ir_profits_{fpostfix}.npy", ir_profits)
        np.save(f"{out_dir}/ir_prices_{fpostfix}.npy", ir_prices)
        # final state-action mappings
        save_actions(net, n_agents, c, f"final_{fpostfix}", out_dir)




if __name__ == "__main__":
    main()
