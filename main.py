import copy
import os
import sys
import itertools

from tqdm.auto import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.profiler import profile, record_function, ProfilerActivity

from cpprb import ReplayBuffer

# Soft Actor-Critic from OpenAI https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = D.Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for sigmoid squashing.
            # This formula is computed with the same procedure as the original SAC paper (arXiv 1801.01290)
            # in appendix C, but using the derivative of the sigmoid instead of tanh
            # It is more numerically stable than using the sigmoid derivative expressed as s(x) * (1 - s(x))
            # This can be tested by computing it for torch.linspace(-20, 20, 1000) for example
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi += (2 * F.softplus(-pi_action) + pi_action).sum(dim=1)
        else:
            logp_pi = None

        pi_action = torch.sigmoid(pi_action)

        return pi_action, logp_pi


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPValueFunction(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        v = self.v(obs)
        return v.squeeze()


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(512, 512), activation=nn.LeakyReLU):
        super().__init__()
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation).to(
            self.device
        )
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(self.device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(self.device)
        self.v = MLPValueFunction(obs_dim, hidden_sizes, activation).to(self.device)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a

    def checkpoint(self, fpostfix, out_dir, t, i):
        torch.save(
            self.pi.state_dict(), f"{out_dir}/actor_weights_{fpostfix}_t{t}_agent{i}.pth",
        )
        torch.save(self.q1.state_dict(), f"{out_dir}/q1_weights_{fpostfix}_t{t}_agent{i}.pth")
        torch.save(self.q2.state_dict(), f"{out_dir}/q2_weights_{fpostfix}_t{t}_agent{i}.pth")


def save_state_action_map(actor, n_agents, c, fpostfix, out_dir):
    grid_size = 100
    w = torch.linspace(c, c + 1, grid_size, requires_grad=False)
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for i in range(n_agents):
            A = torch.zeros([grid_size, grid_size], requires_grad=False)
            for ai, p1 in enumerate(w):
                for aj, p2 in enumerate(w):
                    state = torch.tensor([[p1, p2]]).to(device)
                    a = scale_price(actor[i].act(state, deterministic=True), c)
                    A[ai, aj] = a
            np.save(f"{out_dir}/actions_{fpostfix}_{i}.npy", A.cpu().detach().numpy())


def compute_profit(ai, a0, mu, c, p):
    q = torch.exp((ai - p) / mu) / (torch.sum(torch.exp((ai - p) / mu)) + np.exp(a0 / mu))
    pi = (p - c) * q
    return pi


def get_action(ac, o, deterministic=False):
    return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)


def scale_price(price, c):
    return price + c


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} OUT_DIR")
        exit(1)
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    n_agents = 2
    ai = 2
    a0 = 0
    mu = 0.25
    c = 1
    BATCH_SIZE = 512
    HIDDEN_SIZE = 2048
    INITIAL_LR_ACTOR = 3e-3
    INITIAL_LR_CRITIC = 3e-4
    INITIAL_LR_TEMP = 3e-4
    AVG_REW_LR = 0.01
    MAX_T = int(5e5)
    BUF_SIZE = MAX_T // 10
    CKPT_T = MAX_T // 10
    TARG_UPDATE_RATE = 0.999
    TARGET_ENTROPY = -2
    print(f"Will checkpoint every {CKPT_T} episodes")

    SEEDS = [250917]

    # torch.autograd.set_detect_anomaly(True)
    for session in range(len(SEEDS)):
        fpostfix = SEEDS[session]
        # Random seeds
        np.random.seed(SEEDS[session])
        torch.manual_seed(SEEDS[session])
        # Initial state is random, but ensure prices are above marginal cost
        state = torch.rand(n_agents).to(device) + c
        state = state.unsqueeze(0)
        ac = []
        for i in range(n_agents):
            ac.append(MLPActorCritic(n_agents, 1, hidden_sizes=(HIDDEN_SIZE,) * n_agents))
            print(ac[i])
        ac_targ = copy.deepcopy(ac)
        for i in range(n_agents):
            for p in ac_targ[i].parameters():
                p.requires_grad = False
        q_params = [
            itertools.chain(ac[i].q1.parameters(), ac[i].q2.parameters()) for i in range(n_agents)
        ]
        replay_buffer = []
        pi_optimizer = []
        q_optimizer = []
        v_optimizer = []
        temp_optimizer = []
        log_temp = []
        for i in range(n_agents):
            replay_buffer.append(
                ReplayBuffer(
                    BUF_SIZE,
                    env_dict={
                        "obs": {"shape": n_agents},
                        "act": {"shape": 1},
                        "rew": {},
                        "obs2": {"shape": n_agents},
                    },
                )
            )
            pi_optimizer.append(torch.optim.Adam(ac[i].pi.parameters(), lr=INITIAL_LR_ACTOR))
            q_optimizer.append(torch.optim.Adam(q_params[i], lr=INITIAL_LR_CRITIC))
            v_optimizer.append(torch.optim.Adam(ac[i].v.parameters(), lr=INITIAL_LR_CRITIC))
            log_temp.append(torch.zeros(1, device=device, requires_grad=True))
            temp_optimizer.append(torch.optim.Adam([log_temp[i]], lr=INITIAL_LR_TEMP))
        avg_rew = torch.zeros([n_agents]).to(device)
        action = torch.zeros([n_agents]).to(device)
        price = torch.zeros([n_agents]).to(device)
        # Arrays used to save metrics
        total_reward = torch.zeros([n_agents, MAX_T])
        price_history = torch.zeros([n_agents, MAX_T])
        q_loss = np.zeros([n_agents])
        pi_loss = np.zeros([n_agents])
        v_loss = np.zeros([n_agents])
        with tqdm(range(MAX_T)) as t_tq:
            for t in t_tq:
                with torch.no_grad():
                    for i in range(n_agents):
                        # Randomly explore at the beginning
                        if t < BATCH_SIZE * 10:
                            action[i] = torch.rand(1)
                        else:
                            action[i] = ac[i].act(state).squeeze()
                        price[i] = scale_price(action[i], c)
                        price_history[i, t] = price[i]
                    profits = compute_profit(ai, a0, mu, c, price)
                    for i in range(n_agents):
                        replay_buffer[i].add(
                            obs=state.cpu(),
                            act=action[i].cpu(),
                            rew=profits[i].cpu(),
                            obs2=price.cpu(),
                        )
                        total_reward[i, t] = profits[i]
                        with torch.no_grad():
                            avg_rew[i] += AVG_REW_LR * (
                                profits[i]
                                - avg_rew[i]
                                + ac_targ[i].v(price.squeeze())
                                - ac_targ[i].v(state.squeeze())
                            )
                    state = price.unsqueeze(0)
                    if t > 0 and t % CKPT_T == 0:
                        for i in range(n_agents):
                            ac[i].checkpoint(fpostfix, out_dir, t, i)
                            ac_targ[i].checkpoint(f"target{fpostfix}", out_dir, t, i)
                if t >= BATCH_SIZE:
                    for i in range(n_agents):
                        batch = replay_buffer[i].sample(BATCH_SIZE)

                        o, a, r, o2 = (
                            torch.tensor(batch["obs"], device=device).squeeze(),
                            torch.tensor(batch["act"], device=device),
                            torch.tensor(batch["rew"], device=device).squeeze(),
                            torch.tensor(batch["obs2"], device=device).squeeze(),
                        )
                        # Freeze Q-networks so you don't waste computational effort
                        # computing gradients for them during the policy learning step.
                        for p in q_params[i]:
                            p.requires_grad = False

                        # Next run one gradient descent step for pi.
                        pi, logp_pi = ac[i].pi(o)
                        # Entropy loss
                        temp_loss = -(log_temp[i] * (logp_pi + TARGET_ENTROPY).detach()).mean()
                        temp_optimizer[i].zero_grad(set_to_none=True)
                        temp_loss.backward()
                        temp_optimizer[i].step()
                        temp = log_temp[i].exp()
                        q1_pi = ac[i].q1(o, pi)
                        q2_pi = ac[i].q2(o, pi)
                        q_pi = torch.min(q1_pi, q2_pi)
                        # Entropy-regularized policy loss
                        loss_pi = (temp * logp_pi - q_pi).mean()
                        pi_optimizer[i].zero_grad(set_to_none=True)
                        loss_pi.backward()
                        pi_optimizer[i].step()

                        # Unfreeze Q-networks so you can optimize it at next DDPG step.
                        for p in q_params[i]:
                            p.requires_grad = True

                        q1 = ac[i].q1(o, a)
                        q2 = ac[i].q2(o, a)

                        # Bellman backup for Q functions
                        with torch.no_grad():

                            # Target actions come from *current* policy
                            a2, logp_a2 = ac[i].pi(o2)
                            # Target Q-values
                            q1_pi_targ = ac_targ[i].q1(o2, a2)
                            q2_pi_targ = ac_targ[i].q2(o2, a2)
                            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                            backup = r - avg_rew[i] + (q_pi_targ - temp * logp_a2)

                        # MSE loss against Bellman backup
                        loss_q1 = F.smooth_l1_loss(q1, backup)
                        loss_q2 = F.smooth_l1_loss(q2, backup)
                        loss_q = loss_q1 + loss_q2

                        q_optimizer[i].zero_grad(set_to_none=True)
                        loss_q.backward()
                        q_optimizer[i].step()

                        # Compute value function loss
                        v_optimizer[i].zero_grad(set_to_none=True)
                        vf = ac[i].v(o)
                        with torch.no_grad():
                            vf_target = q_pi - temp * logp_pi
                        loss_v = F.smooth_l1_loss(vf, vf_target)
                        loss_v.backward()
                        v_optimizer[i].step()

                        # Finally, update target networks by polyak averaging.
                        with torch.no_grad():
                            for p, p_targ in zip(ac[i].parameters(), ac_targ[i].parameters()):
                                # NB: We use an in-place operations "mul_", "add_" to update target
                                # params, as opposed to "mul" and "add", which would make new tensors.
                                p_targ.data.mul_(TARG_UPDATE_RATE)
                        p_targ.data.add_((1 - TARG_UPDATE_RATE) * p.data)
                        q_loss[i], pi_loss[i], v_loss[i] = (
                            loss_q.item(),
                            loss_pi.item(),
                            loss_v.item(),
                        )
                    q_loss = np.round(q_loss, 3)
                    pi_loss = np.round(pi_loss, 3)
                    v_loss = np.round(v_loss, 3)
                    with torch.no_grad():
                        start_t = t - BATCH_SIZE
                        avg_price = np.round(
                            torch.mean(price_history[:, start_t:t], dim=1).cpu().numpy(), 3,
                        )
                        avg_profit = np.round(avg_rew.cpu().detach().numpy(), 3,)
                        temp = np.round(np.array([a.exp().item() for a in log_temp]), 3,)
                    t_tq.set_postfix_str(
                        f"p = {avg_price}, P = {avg_profit}, QL = {q_loss}, PL = {pi_loss}, VL = {v_loss}, temp = {temp}"
                    )
        np.save(f"{out_dir}/session_reward_{fpostfix}.npy", total_reward.detach())
        for i in range(n_agents):
            ac[i].checkpoint(fpostfix, out_dir, MAX_T, i)
            ac_targ[i].checkpoint(f"target{fpostfix}", out_dir, MAX_T, i)


if __name__ == "__main__":
    main()
