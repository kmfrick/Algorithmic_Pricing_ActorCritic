#!/usr/bin/python3

import numpy as np
import sys
import os
from tqdm import tqdm
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2_critic = nn.Linear(hidden_size, 1)
        self.fc2_actor = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = F.relu(x)
        action = self.fc2_actor(x)
        mu, sigma = torch.sigmoid(action)
        value = self.fc2_critic(x)
        return mu, sigma, value


def optimize_net(rewards, saved_actions, discount, optimizer):
    # Compute losses
    returns = []
    actor_losses = []
    critic_losses = []
    R = 0
    # Go through the array in reverse to calculate discounted reward
    for r in rewards[::-1]:
        R = r + discount * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)
    for (l, v), R in zip(saved_actions, returns):
        adv = R - v.item()
        actor_losses.append(-l * adv)
        critic_losses.append(F.mse_loss(v, R.unsqueeze(0)))
    optimizer.zero_grad()
    loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum()
    loss.backward()
    optimizer.step()

def save_actions(net_actor, net_critic, n_agents, c, fpostfix, out_dir):
    grid_size = 100
    w = np.linspace(c, c + 2, grid_size)

    for i in range(n_agents):
        net_actor[i].eval()
        A = np.zeros([grid_size, grid_size])
        Q = np.zeros([grid_size, grid_size])
        for ai, p1 in enumerate(w):
            for aj, p2 in enumerate(w):
                state = torch.tensor(np.array([p1, p2]))
                a, _ , _= net_actor[i](state)
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

    discount = 0.95
    n_episodes = 1000
    n_agents = 2
    a = np.array([2.0, 2.0])
    a0 = 0
    mu = 0.25
    c = 1
    batch_size = 64
    hidden_size = 32

    seeds = [12654329, 4305430, 3483055, 348203, 2356, 250917, 200822, 151515, 50505, 301524, 12345, 54321, 50321, 9827391,
             8534101]
    for session in range(len(seeds)):
        net = []
        net.append(ActorCriticNetwork(n_agents, hidden_size))
        net.append(ActorCriticNetwork(n_agents, hidden_size))
        optimizer = [torch.optim.AdamW(n.parameters(), lr=3e-2) for n in net]
        scheduler = [torch.optim.lr_scheduler.StepLR(o, batch_size, gamma=0.999) for o in optimizer]
        cum_rewards = np.zeros([n_agents, 0])
        nash_price = 1.47293
        state = torch.tensor([nash_price] * n_agents)
        price = np.zeros([n_agents])
        fpostfix = seeds[session]
        # Random seeds
        np.random.seed(seeds[session])
        torch.manual_seed(seeds[session])
        ir_periods = 20

        for i_episode in tqdm(range(n_episodes)):
            saved_actions = [None] * n_agents
            rewards = [None] * n_agents
            for i in range(n_agents):
                rewards[i] = []
                saved_actions[i] = []
            for t in range(batch_size):
                for i in range(n_agents):
                    mean, std, value = net[i](state)
                    m = Normal(mean, std)
                    p = m.sample()
                    price[i] = p + c
                    saved_actions[i].append((m.log_prob(p), value))
                quant = np.exp((a - price) / mu) / (np.sum(np.exp((a - price) / mu)) + np.exp(a0 / mu))
                profits = (price - c) * quant
                for i in range(n_agents):
                    rewards[i].append(profits[i].item())
                state = torch.tensor(price)
            for i in range(n_agents):
                optimize_net(rewards[i], saved_actions[i], discount, optimizer[i])
                scheduler[i].step()
            cum_rewards = np.concatenate((cum_rewards, np.array(rewards)), axis=1)
        # Impulse response
        ir_profits = np.zeros([n_agents, ir_periods])
        ir_prices = np.zeros([n_agents, ir_periods])
        for t in range(ir_periods):
            for i in range(n_agents):
                net[i].eval()
                p, _, _ = net[i](state)
                price[i] = (p + c).detach().numpy()
                ir_prices[i, t] = price[i]
            if (ir_periods / 2) <= t <= (ir_periods / 2 + 3):
                price[0] = nash_price
                ir_prices[0, t] = price[0]
            quant = np.exp((a - price) / mu) / (np.sum(np.exp((a - price) / mu)) + np.exp(a0 / mu))
            profits = (price - c) * quant
            ir_profits[:, t] = profits
            state = torch.tensor(price)
        np.save(f"{out_dir}/ir_profits_{fpostfix}.npy", ir_profits)
        np.save(f"{out_dir}/ir_prices_{fpostfix}.npy", ir_prices)

        reward = np.array(cum_rewards)
        np.save(f"{out_dir}/rewards_{fpostfix}.npy", reward)
        save_actions(net, None, n_agents, c, f"final_{fpostfix}", out_dir)


if __name__ == "__main__":
    main()
