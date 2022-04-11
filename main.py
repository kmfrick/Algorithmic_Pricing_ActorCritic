#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

np.random.seed(50321)

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

def main():
    discount = 0.95
    n_episodes = 1000
    n_agents = 2
    results = np.zeros(n_episodes)
    hidden_size = 32
    a = np.array([2.0, 2.0])
    a0 = 0
    mu = 0.25
    c = 1
    batch_size = 64
    net = []
    net.append(ActorCriticNetwork(n_agents, hidden_size))
    net.append(ActorCriticNetwork(n_agents, hidden_size))
    optimizer = [torch.optim.AdamW(n.parameters(), lr=3e-2) for n in net]
    scheduler = [torch.optim.lr_scheduler.StepLR(o, batch_size, gamma = 0.999) for o in optimizer]
    cum_rewards = np.zeros([n_agents, 0])
    state = torch.tensor([1.47] * n_agents)
    price = np.zeros([n_agents])
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
            quant = np.exp((a - price)/mu) / (np.sum(np.exp((a - price)/mu)) + np.exp(a0/mu))
            profits = (price - c) * quant
            for i in range(n_agents):
                rewards[i].append(profits[i].item())
            state = torch.tensor(price)
        for i in range(n_agents):
            optimize_net(rewards[i], saved_actions[i], discount, optimizer[i])
            scheduler[i].step()
        cum_rewards = np.concatenate((cum_rewards, np.array(rewards)), axis = 1)

    reward = np.array(cum_rewards)
    np.save("rewards.npy", reward)



if __name__ == "__main__":
    main()
