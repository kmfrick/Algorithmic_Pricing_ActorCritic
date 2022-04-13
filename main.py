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


class ActorNetwork(nn.Module):
    def __init__(self, state_dim):
        EPS = 3e-3
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_lim = 1
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc_actor3 = nn.Linear(128, 64)
        self.fc_actor3.weight.data = fanin_init(self.fc_actor3.weight.data.size())
        self.fc_actor4 = nn.Linear(64, 1)
        self.fc_actor4.weight.data.uniform_(-EPS, EPS)

    def forward(self, state):
        state = state.float()
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc_actor3(x))
        x = torch.sigmoid(self.fc_actor4(x)) * self.action_lim
        return x


class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        EPS = 3e-3
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_lim = 1
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc_critic_a = nn.Linear(1, 128)
        self.fc_critic_a.weight.data = fanin_init(self.fc_critic_a.weight.data.size())
        self.fc_critic2 = nn.Linear(256, 128)
        self.fc_critic2.weight.data = fanin_init(self.fc_critic2.weight.data.size())
        self.fc_critic3 = nn.Linear(128, 1)
        self.fc_critic3.weight.data.uniform_(-EPS, EPS)

    def forward(self, state, action):
        state = state.float()
        action = action.float()
        s1 = F.relu(self.fc1(state))
        s2 = F.relu(self.fc2(s1))
        a1 = F.relu(self.fc_critic_a(action))
        x = torch.cat((s2, a1), dim=1)
        x = F.relu(self.fc_critic2(x))
        x = self.fc_critic3(x)
        return x



def compute_profit(ai, a0, mu, c, p):
    q = np.exp((ai - p) / mu) / (np.sum(np.exp((ai - p) / mu)) + np.exp(a0 / mu))
    pi = (p - c) * q
    return pi


def compute_profit_derivative(ai, a0, mu, c, p, h):
    def f(price):
        return compute_profit(ai, a0, mu, c, price)

    return (f(p + h) - f(p - h)) / (2 * h)


def optimize(batch_size, buffer, net_actor, net_critic, target_net_actor, target_net_critic, optimizer_actor, optimizer_critic, scheduler_actor, scheduler_critic):
    DISCOUNT = 0.95
    TARGET_LR = 0.001
    batch = random.sample(buffer, min(len(buffer), batch_size))
    s = torch.tensor(np.float32([arr[0] for arr in batch]))
    a = torch.tensor(np.float32([arr[1] for arr in batch]))
    r = torch.tensor(np.float32([arr[2] for arr in batch]))
    s1 = torch.tensor(np.float32([arr[3] for arr in batch]))
    a1 = target_net_actor(s1).detach()
    next_val = target_net_critic(s1, a1).detach().squeeze()
    y_exp = r + DISCOUNT * next_val
    #print(f"exp = {torch.mean(y_exp)}")
    y_pred = net_critic(s, a.unsqueeze(0).mT).squeeze()
    #print(f"pred = {torch.mean(y_pred)}")
    l_critic = F.smooth_l1_loss(y_pred.squeeze(), y_exp.squeeze())
    optimizer_critic.zero_grad()
    l_critic.backward()
    torch.nn.utils.clip_grad_norm_(net_critic.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(target_net_critic.parameters(), 1)
    optimizer_critic.step()
    pred_a = net_actor(s)
    pred_v = net_critic(s, pred_a)
    l_actor = -torch.sum(pred_v)
    optimizer_actor.zero_grad()
    l_actor.backward()
    torch.nn.utils.clip_grad_norm_(net_critic.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(target_net_critic.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(net_actor.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(target_net_actor.parameters(), 1)
    optimizer_actor.step()
    # Decay LR
    scheduler_actor.step()
    scheduler_critic.step()
    actor_norm = 0
    critic_norm = 0
    for target_param, param in zip(target_net_actor.parameters(), net_actor.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - TARGET_LR) + param.data * TARGET_LR)
        if param.grad is not None:
            param_norm = param.grad.detach().data.norm(2)
            actor_norm += param_norm.item() ** 2
    for target_param, param in zip(target_net_critic.parameters(), net_critic.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - TARGET_LR) + param.data * TARGET_LR)
        if param.grad is not None:
            param_norm = param.grad.detach().data.norm(2)
            critic_norm += param_norm.item() ** 2
    return l_actor.item(), l_critic.item(), actor_norm, critic_norm, scheduler_critic.get_last_lr()[0]


def save_actions(net_actor, net_critic, n_agents, c, fpostfix, out_dir):
    grid_size = 100
    w = np.linspace(c, c + 2, grid_size)
    for i in range(n_agents):
        A = np.zeros([grid_size, grid_size])
        Q = np.zeros([grid_size, grid_size])
        for ai, p1 in enumerate(w):
            for aj, p2 in enumerate(w):
                state = torch.tensor(np.array([p1, p2]))
                a = net_actor[i](state)
                #q = net_critic[i](state, a)
                A[ai, aj] = a.detach().numpy()
                #Q[ai, aj] = q.detach().numpy()
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
        net_actor = []
        net_critic = []
        for i in range(n_agents):
            net_actor.append(ActorNetwork(n_agents))
            net_critic.append(CriticNetwork(n_agents))
            buffer.append(deque(maxlen=BATCH_SIZE * 2))
        target_net_actor = copy.deepcopy(net_actor)
        target_net_critic = copy.deepcopy(net_critic)
        # Initial state is random, but ensure prices are above marginal cost
        state = torch.sigmoid(torch.randn(n_agents)) + c
        # initial state-action mappings
        save_actions(net_actor, net_critic, n_agents, c, f"initial_{fpostfix}", out_dir)
        # One optimizer per network
        optimizer_actor = [torch.optim.AdamW(n.parameters(), lr=INITIAL_LR) for n in net_actor]
        optimizer_critic = [torch.optim.AdamW(n.parameters(), lr=INITIAL_LR) for n in net_critic]
        # One scheduler per network
        scheduler_actor = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=LR_DECAY) for o in optimizer_actor]
        scheduler_critic = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=LR_DECAY) for o in optimizer_critic]

        price = np.zeros([n_agents])
        total_reward = np.zeros([n_agents, MAX_T])
        price_history = np.zeros([n_agents, MAX_T])
        actor_norm = np.zeros([n_agents, MAX_T])
        critic_norm = np.zeros([n_agents, MAX_T])
        net_lr = np.zeros([n_agents, MAX_T])
        net_lr[:, 0] = INITIAL_LR
        actor_loss = np.zeros([n_agents, MAX_T])
        critic_loss = np.zeros([n_agents, MAX_T])
        for t in tqdm(range(MAX_T)):
            for i in range(n_agents):
                p = net_actor[i](state)
                price[i] = (p + c + D.Normal(0, np.exp(- EXPLORATION_DECAY * t)).sample()).detach().numpy()
                price_history[i, t] = price[i]
            profits = compute_profit(ai, a0, mu, c, price)
            for i in range(n_agents):
                transition = state.detach().numpy(), price[i], profits[i], price
                buffer[i].append(transition)
                total_reward[i, t] = profits[i]
            state = torch.tensor(price)
            for i in range(n_agents):
                al, cl, an, cn, nlr = optimize(BATCH_SIZE, buffer[i], net_actor[i], net_critic[i], target_net_actor[i], target_net_critic[i], optimizer_actor[i], optimizer_critic[i], scheduler_actor[i], scheduler_critic[i])
                net_lr[i, t] = nlr
                actor_norm[i, t] = an
                critic_norm[i, t] = cn
                actor_loss[i, t] = al
                critic_loss[i, t] = cl
        np.save(f"{out_dir}/session_reward_{fpostfix}.npy", total_reward)
        np.save(f"{out_dir}/net_lr_{fpostfix}.npy", net_lr)
        np.save(f"{out_dir}/critic_norm_{fpostfix}.npy", critic_norm)
        np.save(f"{out_dir}/critic_loss_{fpostfix}.npy", critic_loss)
        np.save(f"{out_dir}/actor_norm_{fpostfix}.npy", actor_norm)
        np.save(f"{out_dir}/actor_loss_{fpostfix}.npy", actor_loss)
        # Impulse response
        ir_profits = np.zeros([n_agents, ir_periods])
        ir_prices = np.zeros([n_agents, ir_periods])
        for t in range(ir_periods):
            for i in range(n_agents):
                p = net_actor[i](state)
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
        save_actions(net_actor, net_critic, n_agents, c, f"final_{fpostfix}", out_dir)




if __name__ == "__main__":
    main()
