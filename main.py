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
from torch.profiler import profile, record_function, ProfilerActivity


# SAC implementation inspired from https://github.com/pranz24/pytorch-soft-actor-critic/tree/SAC_V
# Added batch norm and used leaky ReLUs
# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        #state = state.float()
        x = F.leaky_relu(self.linear1(state))
        x = self.bn1(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.bn2(x)
        mean = self.linear3(x)
        return mean


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.leaky_relu(self.linear1(xu))
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(self.linear2(x1))
        x2 = self.bn2(x1)
        x1 = self.linear3(x1)

        x2 = F.leaky_relu(self.linear4(xu))
        x2 = self.bn3(x2)
        x2 = F.leaky_relu(self.linear5(x2))
        x2 = self.bn4(x2)
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        #state = state.float()
        LOG_SIG_MAX = 1
        LOG_SIG_MIN = -20
        x = F.leaky_relu(self.linear1(state))
        x = F.leaky_relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = D.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def optimize(device, batch_size, buffer, policy, critic, value, value_target, policy_optim, critic_optim, value_optim, scheduler_actor, scheduler_critic, scheduler_value):
    DISCOUNT = 0.95
    TARGET_LR = 0.005
    TEMPERATURE = 0.1 # Temperature parameter α determines the relative importance of the entropy term against the reward
    batch = random.sample(buffer, min(len(buffer), batch_size))
    state_batch_, action_batch_, reward_batch_, next_state_batch = map(torch.stack, zip(*batch))
    reward_batch = reward_batch_.unsqueeze(1)
    state_batch = state_batch_.squeeze()
    action_batch = action_batch_.unsqueeze(1)
    policy.train()
    critic.train()
    value.train()
    with torch.no_grad():
        vf_next_target = value_target(next_state_batch)
        next_q_value = reward_batch + DISCOUNT * (vf_next_target)

    qf1, qf2 = critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
    qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    qf_loss = qf1_loss + qf2_loss

    critic_optim.zero_grad()
    qf_loss.backward()
    critic_optim.step()

    pi, log_pi, mean, log_std = policy.sample(state_batch)

    qf1_pi, qf2_pi = critic(state_batch, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)

    policy_loss = ((TEMPERATURE * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
    # Regularization Loss
    reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
    policy_loss += reg_loss

    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    vf = value(state_batch)

    with torch.no_grad():
        vf_target = min_qf_pi - (TEMPERATURE * log_pi)

    vf_loss = F.mse_loss(vf, vf_target)  # JV = 𝔼(st)~D[0.5(V(st) - (𝔼at~π[Q(st,at) - α * logπ(at|st)]))^2]

    value_optim.zero_grad()
    vf_loss.backward()
    value_optim.step()

    soft_update(value_target, value, TARGET_LR)
    actor_norm = 0
    #for p in policy.parameters():
    #    if p.grad is not None:
    #        param_norm = p.grad.detach().data.norm(2)
    #        actor_norm += param_norm.item() ** 2
    critic_norm = 0
    #for p in critic.parameters():
    #    if p.grad is not None:
    #        param_norm = p.grad.detach().data.norm(2)
    #        critic_norm += param_norm.item() ** 2

    return policy_loss, qf_loss, actor_norm, critic_norm

def compute_profit(ai, a0, mu, c, p):
    q = torch.exp((ai - p) / mu) / (torch.sum(torch.exp((ai - p) / mu)) + torch.exp(a0 / mu))
    pi = (p - c) * q
    return pi

def save_actions(net_actor, net_critic, n_agents, c, fpostfix, out_dir):
    grid_size = 100
    w = np.linspace(c, c + 2, grid_size)
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    for i in range(n_agents):
        net_actor[i].eval()
        A = np.zeros([grid_size, grid_size])
        Q = np.zeros([grid_size, grid_size])
        for ai, p1 in enumerate(w):
            for aj, p2 in enumerate(w):
                state = torch.tensor([[p1, p2]]).to(device)
                a = select_action(device, net_actor[i], state, c, eval=True)
                #q = net_critic[i](state, a)
                A[ai, aj] = a.cpu().detach().numpy()
                #Q[ai, aj] = q.detach().numpy()
        np.save(f"{out_dir}/actions_{fpostfix}_{i}.npy", A)
        #np.save(f"{out_dir}/values_{fpostfix}_{i}.npy", Q)

def select_action(device, net, state, c, eval=False):
    state = state.unsqueeze(0)
    with torch.no_grad():
      if eval == False:
          action, _, _, _ = net.sample(state)
      else:
          _, _, action, _ = net.sample(state)
          action = torch.tanh(action)
    return rescale_action(action, action_range = [c, c + 1])

def rescale_action(action, action_range):
    return action * (action_range[1] - action_range[0]) / 2.0 + \
           (action_range[1] + action_range[0]) / 2.0


def main():
    if len(sys.argv) < 2:
        out_dir = "exp_colab"
    else:
        out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    #device = xm.xla_device()
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)
    n_agents = 2
    ai = torch.tensor([2.0, 2.0]).to(device)
    a0 = torch.tensor([0]).to(device)
    mu = torch.tensor([0.25]).to(device)
    c = 1
    BATCH_SIZE = 1024
    HIDDEN_SIZE = 512
    BUF_SIZE = int(1e4)
    INITIAL_LR_ACTOR = 3e-4
    INITIAL_LR_CRITIC = 3e-4
    INITIAL_LR_VALUE = 3e-4
    MAX_T = int(1e5)
    LR_DECAY_ACTOR = 1
    LR_DECAY_CRITIC = 1
    LR_DECAY_VALUE = 1
    nash_price = 1.47293
    coop_price = 1.9249689958811602
    param_str = f"BS_{BATCH_SIZE}_HS_{HIDDEN_SIZE}_BUFS_{BUF_SIZE}_LRA_{INITIAL_LR_ACTOR}_LRC_{INITIAL_LR_CRITIC}_LRV_{INITIAL_LR_VALUE}"
    seeds = [12345, 54321, 50321, 9827391, 8534101]#, 4305430, 12654329, 3483055, 348203, 2356, 250917, 200822, 151515, 50505, 301524]

    ir_periods = 20
    #torch.autograd.set_detect_anomaly(True)
    for session in range(len(seeds)):
        fpostfix = seeds[session]
        # Random seeds
        np.random.seed(seeds[session])
        torch.manual_seed(seeds[session])
        # Initialize replay buffers and NNs
        buffer = []
        net_actor = [GaussianPolicy(n_agents, 1,  HIDDEN_SIZE).to(device) for i in n_agents]
        net_critic = [ValueNetwork(n_agents, HIDDEN_SIZE).to(device) for i in n_agents]
        net_value = [QNetwork(n_agents, 1, HIDDEN_SIZE).to(device) for i in n_agents]
        for i in range(n_agents):
            buffer.append(deque(maxlen=BUF_SIZE))
        target_net_value = copy.deepcopy(net_value)
        for i in range(n_agents):
          target_net_value[i] = target_net_value[i].to(device)
        # Initial state is random, but ensure prices are above marginal cost
        state = torch.sigmoid(torch.randn(n_agents)).to(device) + c
        state = state.unsqueeze(0)
        # initial state-action mappings
        #save_actions(net_actor, net_critic, n_agents, c, f"initial_{fpostfix}", out_dir)
        # One optimizer per network
        optimizer_actor = [torch.optim.Adam(n.parameters(), lr=INITIAL_LR_ACTOR) for n in net_actor]
        optimizer_critic = [torch.optim.Adam(n.parameters(), lr=INITIAL_LR_CRITIC) for n in net_critic]
        optimizer_value = [torch.optim.Adam(n.parameters(), lr=INITIAL_LR_VALUE) for n in net_value]
        # One scheduler per network
        scheduler_actor = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=LR_DECAY_ACTOR) for o in optimizer_actor]
        scheduler_critic = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=LR_DECAY_CRITIC) for o in optimizer_critic]
        scheduler_value = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=LR_DECAY_VALUE) for o in optimizer_value]

        price = torch.zeros([n_agents]).to(device)
        total_reward = torch.zeros([n_agents, MAX_T])
        price_history = torch.zeros([n_agents, MAX_T])
        actor_norm = torch.zeros([n_agents, MAX_T])
        critic_norm = torch.zeros([n_agents, MAX_T])
        net_lr = torch.zeros([n_agents, MAX_T])
        net_lr[:BATCH_SIZE, 0] = INITIAL_LR_ACTOR
        actor_loss = torch.zeros([n_agents, MAX_T])
        critic_loss = torch.zeros([n_agents, MAX_T])
        with tqdm(range(MAX_T)) as t_tq:
            for t in t_tq:
                with record_function("model_inference"):
                    for i in range(n_agents):
                        net_actor[i].eval()
                        price[i] = select_action(device, net_actor[i], state, c).squeeze()
                        price_history[i, t] = price[i]
                    profits = compute_profit(ai, a0, mu, c, price)
                    for i in range(n_agents):
                        transition = state, price[i], profits[i], price
                        buffer[i].append(transition)
                        total_reward[i, t] = profits[i]
                    state = price.unsqueeze(0)
                    if t > 0:
                        for i in range(n_agents):
                            al, cl, an, cn = optimize(device, BATCH_SIZE, buffer[i], net_actor[i], net_critic[i], net_value[i], target_net_value[i], optimizer_actor[i], optimizer_critic[i], optimizer_value[i], scheduler_actor[i], scheduler_critic[i], scheduler_value[i])
                            net_lr[i, t] = scheduler_actor[i].get_last_lr()[0]
                            actor_norm[i, t] = an
                            critic_norm[i, t] = cn
                            actor_loss[i, t] = al
                            critic_loss[i, t] = cl
                        t_tq.set_postfix_str(torch.mean(total_reward[:, :t]).cpu().detach().numpy())
        np.save(f"{out_dir}/session_reward_{fpostfix}.npy", total_reward.detach())
        np.save(f"{out_dir}/net_lr_{fpostfix}.npy", net_lr.detach())
        np.save(f"{out_dir}/critic_norm_{fpostfix}.npy", critic_norm.detach())
        np.save(f"{out_dir}/critic_loss_{fpostfix}.npy", critic_loss.detach())
        np.save(f"{out_dir}/actor_norm_{fpostfix}.npy", actor_norm.detach())
        np.save(f"{out_dir}/actor_loss_{fpostfix}.npy", actor_loss.detach())
        # Impulse response
        ir_profits = torch.zeros([n_agents, ir_periods])
        ir_prices = torch.zeros([n_agents, ir_periods])
        for t in range(ir_periods):
            for i in range(n_agents):
                net_actor[i].eval()
                price[i] = select_action(device, net_actor[i], state, c, eval=True).squeeze()
                ir_prices[i, t] = price[i]
            if (ir_periods / 2) <= t <= (ir_periods / 2 + 3):
                price[0] = nash_price
                ir_prices[0, t] = price[0]
            ir_profits[:, t] = compute_profit(ai, a0, mu, c, price)
            state = price
        np.save(f"{out_dir}/ir_profits_{fpostfix}.npy", ir_profits.detach())
        np.save(f"{out_dir}/ir_prices_{fpostfix}.npy", ir_prices.detach())
        # final state-action mappings
        save_actions(net_actor, net_critic, n_agents, c, f"final_{fpostfix}", out_dir)


if __name__ == "__main__":
    main()
