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

# SAC implementation from https://github.com/pranz24/pytorch-soft-actor-critic/blob/SAC_V/model.py
# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        state = state.float()
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.linear3(x)
        return mean


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        state = state.float()
        action = action.float()
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        state = state.float()
        LOG_SIG_MAX = 2
        LOG_SIG_MIN = -20
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.sigmoid(log_std) * np.abs(LOG_SIG_MAX - LOG_SIG_MIN) + LOG_SIG_MIN
        return mean, log_std

def optimize(batch_size, buffer, net_actor, net_critic, net_value, target_net_actor, target_net_critic, target_net_value, optimizer_actor, optimizer_critic, optimizer_value, scheduler_actor, scheduler_critic, scheduler_value):
    DISCOUNT = 0.95
    TARGET_LR = 0.001
    TEMPERATURE = 0.2 # Temperature parameter α determines the relative importance of the entropy term against the reward
    batch = random.sample(buffer, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch = map(np.stack, zip(*batch))

    state_batch = torch.FloatTensor(state_batch)
    next_state_batch = torch.FloatTensor(next_state_batch)
    action_batch = torch.FloatTensor(action_batch)
    reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)

    with torch.no_grad():
        vf_next_target = target_net_value(next_state_batch)
        next_q_value = reward_batch + DISCOUNT * (vf_next_target)
    action_batch = action_batch.unsqueeze(1)
    qf1, qf2 = net_critic(state_batch,
                           action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
    qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    qf_loss = qf1_loss + qf2_loss

    optimizer_critic.zero_grad()
    qf_loss.backward()
    optimizer_critic.step()
    scheduler_critic.step()

    mean, log_std = net_actor.forward(state_batch)
    std = log_std.exp()
    normal = D.Normal(mean, std)
    x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    pi = torch.sigmoid(x_t)
    log_pi = normal.log_prob(x_t)
    # Enforcing Action Bound
    #log_pi -= torch.log(1 - pi.pow(2) + 1e-6)
    log_pi = log_pi.sum(1, keepdim=True)

    qf1_pi, qf2_pi = net_critic(state_batch, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)

    policy_loss = ((TEMPERATURE * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
    # Regularization Loss
    reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
    policy_loss += reg_loss

    optimizer_actor.zero_grad()
    policy_loss.backward()
    optimizer_actor.step()
    scheduler_actor.step()

    vf = net_value(state_batch)

    with torch.no_grad():
        vf_target = min_qf_pi - (TEMPERATURE * log_pi)

    vf_loss = F.mse_loss(vf, vf_target)  # JV = 𝔼(st)~D[0.5(V(st) - (𝔼at~π[Q(st,at) - α * logπ(at|st)]))^2]

    optimizer_value.zero_grad()
    vf_loss.backward()
    optimizer_value.step()
    scheduler_value.step()
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
    for target_param, param in zip(target_net_value.parameters(), net_value.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - TARGET_LR) + param.data * TARGET_LR)
        if param.grad is not None:
            param_norm = param.grad.detach().data.norm(2)
            critic_norm += param_norm.item() ** 2

    return policy_loss, qf_loss, actor_norm, critic_norm

def compute_profit(ai, a0, mu, c, p):
    q = np.exp((ai - p) / mu) / (np.sum(np.exp((ai - p) / mu)) + np.exp(a0 / mu))
    pi = (p - c) * q
    return pi

def save_actions(net_actor, net_critic, n_agents, c, fpostfix, out_dir):
    grid_size = 100
    w = np.linspace(c, c + 2, grid_size)
    for i in range(n_agents):
        A = np.zeros([grid_size, grid_size])
        Q = np.zeros([grid_size, grid_size])
        for ai, p1 in enumerate(w):
            for aj, p2 in enumerate(w):
                state = torch.tensor(np.array([p1, p2]))
                a, _ = net_actor[i](state)
                #q = net_critic[i](state, a)
                A[ai, aj] = torch.sigmoid(a).detach().numpy()
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
    HIDDEN_SIZE = 256
    BUF_SIZE = 300
    INITIAL_LR_ACTOR = 3e-4
    INITIAL_LR_CRITIC = 3e-5
    INITIAL_LR_VALUE = 3e-5
    MAX_T = int(5e3)
    LR_DECAY_ACTOR = 0.999
    LR_DECAY_CRITIC = 1
    LR_DECAY_VALUE = 1
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
        net_value = []
        for i in range(n_agents):
            net_actor.append(GaussianPolicy(n_agents, 1,  HIDDEN_SIZE))
            net_value.append(ValueNetwork(n_agents, HIDDEN_SIZE))
            net_critic.append(QNetwork(n_agents, 1, HIDDEN_SIZE))
            buffer.append(deque(maxlen=BUF_SIZE))
        target_net_actor = copy.deepcopy(net_actor)
        target_net_critic = copy.deepcopy(net_critic)
        target_net_value = copy.deepcopy(net_value)
        # Initial state is random, but ensure prices are above marginal cost
        state = torch.sigmoid(torch.randn(n_agents)) + c
        # initial state-action mappings
        save_actions(net_actor, net_critic, n_agents, c, f"initial_{fpostfix}", out_dir)
        # One optimizer per network
        optimizer_actor = [torch.optim.AdamW(n.parameters(), lr=INITIAL_LR_ACTOR) for n in net_actor]
        optimizer_critic = [torch.optim.Adam(n.parameters(), lr=INITIAL_LR_CRITIC) for n in net_critic]
        optimizer_value = [torch.optim.Adam(n.parameters(), lr=INITIAL_LR_VALUE) for n in net_value]
        # One scheduler per network
        scheduler_actor = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=LR_DECAY_ACTOR) for o in optimizer_actor]
        scheduler_critic = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=LR_DECAY_CRITIC) for o in optimizer_critic]
        scheduler_value = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=LR_DECAY_VALUE) for o in optimizer_value]

        price = np.zeros([n_agents])
        total_reward = np.zeros([n_agents, MAX_T])
        price_history = np.zeros([n_agents, MAX_T])
        actor_norm = np.zeros([n_agents, MAX_T])
        critic_norm = np.zeros([n_agents, MAX_T])
        net_lr = np.zeros([n_agents, MAX_T])
        net_lr[:BATCH_SIZE, 0] = INITIAL_LR_ACTOR
        actor_loss = np.zeros([n_agents, MAX_T])
        critic_loss = np.zeros([n_agents, MAX_T])
        for t in tqdm(range(MAX_T)):
            for i in range(n_agents):
                mean, log_std = net_actor[i].forward(state)
                std = log_std.exp()
                normal = D.Normal(mean, std)
                p = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
                p = torch.sigmoid(p)
                price[i] = p + c
                price_history[i, t] = price[i]
            profits = compute_profit(ai, a0, mu, c, price)
            for i in range(n_agents):
                transition = state.detach().numpy(), price[i] - c, profits[i], price
                buffer[i].append(transition)
                total_reward[i, t] = profits[i]
            state = torch.tensor(price)
            if t >= BATCH_SIZE:
                for i in range(n_agents):
                    al, cl, an, cn = optimize(BATCH_SIZE, buffer[i], net_actor[i], net_critic[i], net_value[i], target_net_actor[i], target_net_critic[i], target_net_value[i], optimizer_actor[i], optimizer_critic[i], optimizer_value[i], scheduler_actor[i], scheduler_critic[i], scheduler_value[i])
                    net_lr[i, t] = scheduler_actor[i].get_last_lr()[0]
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
                p, _ = net_actor[i](state)
                p = torch.sigmoid(p)
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
