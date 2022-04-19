import copy
import os
import itertools

from tqdm.auto import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.tensorboard import SummaryWriter


# Soft Actor-Critic from OpenAI https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
# Adapted for CUDA
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.obs_buf = torch.zeros(combined_shape(size, obs_dim), dtype=torch.float32).to(device)
        self.obs2_buf = torch.zeros(combined_shape(size, obs_dim), dtype=torch.float32).to(device)
        self.act_buf = torch.zeros(combined_shape(size, act_dim), dtype=torch.float32).to(device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32).to(device)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        indices = torch.randint(low=0, high=self.size, size=[batch_size])
        batch = dict(
            obs=self.obs_buf[indices],
            obs2=self.obs2_buf[indices],
            act=self.act_buf[indices],
            rew=self.rew_buf[indices],
        )
        return batch


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


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


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(512, 512), activation=nn.LeakyReLU):
        super().__init__()
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a

    def checkpoint(self, fpostfix, out_dir, t, i):
        torch.save(
            self.pi.state_dict(),
            f"{out_dir}/actor_weights_{fpostfix}_t{t}_agent{i}.pth",
        )
        torch.save(self.q1.state_dict(), f"{out_dir}/q1_weights_{fpostfix}_t{t}_agent{i}.pth")
        torch.save(self.q2.state_dict(), f"{out_dir}/q2_weights_{fpostfix}_t{t}_agent{i}.pth")


def compute_profit(ai, a0, mu, c, p):
    q = torch.exp((ai - p) / mu) / (torch.sum(torch.exp((ai - p) / mu)) + np.exp(a0 / mu))
    pi = (p - c) * q
    return pi


def update(ac, ac_targ, q_optimizer, pi_optimizer, temperature_optimizer, target_entropy, log_temperature, q_params, data):
    TARG_UPDATE_RATE = 0.999
    # First run one gradient descent step for Q1 and Q2
    q_optimizer.zero_grad(set_to_none=True)
    DISCOUNT = 0.99
    o, a, r, o2 = data["obs"], data["act"], data["rew"], data["obs2"]

    # Freeze Q-networks so you don't waste computational effort
    # computing gradients for them during the policy and temperature learning step.
    for p in q_params:
        p.requires_grad = False

    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad(set_to_none=True)
    pi, logp_pi = ac.pi(o)

    loss_temp = -(log_temperature * (logp_pi + target_entropy).detach()).mean()
    temperature_optimizer.zero_grad()
    loss_temp.backward()
    temperature_optimizer.step()
    temp = log_temperature.exp()

    q1_pi = ac.q1(o, pi)
    q2_pi = ac.q2(o, pi)
    q_pi = torch.min(q1_pi, q2_pi)
    # Entropy-regularized policy loss
    loss_pi = (temp * logp_pi - q_pi).mean()
    loss_pi.backward()
    pi_optimizer.step()

    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in q_params:
        p.requires_grad = True

    q1 = ac.q1(o, a)
    q2 = ac.q2(o, a)
    # Bellman backup for Q functions
    with torch.no_grad():
        # Target actions come from *current* policy
        a2, logp_a2 = ac.pi(o2)
        # Target Q-values
        q1_pi_targ = ac_targ.q1(o2, a2)
        q2_pi_targ = ac_targ.q2(o2, a2)
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
        backup = r + DISCOUNT * (q_pi_targ - temp * logp_a2)
    # MSE loss against Bellman backup
    loss_q1 = F.smooth_l1_loss(q1, backup)
    loss_q2 = F.smooth_l1_loss(q2, backup)
    loss_q = loss_q1 + loss_q2
    loss_q.backward()
    q_optimizer.step()


    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(TARG_UPDATE_RATE)
            p_targ.data.add_((1 - TARG_UPDATE_RATE) * p.data)

    return loss_q.item(), loss_pi.item(), loss_temp.item()


def get_action(ac, o, deterministic=False):
    return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)


def scale_price(price, c):
    return price + c


def main():
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    ai = 2
    a0 = 0
    mu = 0.25
    c = 1
    N_AGENTS = 2
    BATCH_SIZE = 256
    HIDDEN_SIZE = 256
    INITIAL_LR_ACTOR = 3e-3
    INITIAL_LR_CRITIC = 3e-3
    INITIAL_LR_TEMP = 3e-4
    MAX_T = int(1e5)
    BUF_SIZE = MAX_T // 10
    CKPT_T = MAX_T // 10
    SEEDS = [12345, 54321, 464738, 250917]
    #ER_DECAY1 = 2e-5
    #ER_DECAY2 = 2e-4
    #T_ER_THRESH = 6e4
    #x = np.arange(MAX_T)
    #EXP_RATE = np.piecewise(x, [x < T_ER_THRESH], [lambda t : np.exp(-t * ER_DECAY1), lambda t : np.exp(-(t-T_ER_THRESH) * ER_DECAY2) / np.exp(T_ER_THRESH * ER_DECAY1)])

    print(f"Will checkpoint every {CKPT_T} episodes")
    # torch.autograd.set_detect_anomaly(True)
    for seed in SEEDS:
        out_dir = f"weights_{seed}"
        os.makedirs(out_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=f"run_{seed}")
        # Random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Initial state is random, but ensure prices are above marginal cost
        state = torch.rand(N_AGENTS).to(device) + c
        state = state.unsqueeze(0)
        ac = []
        for i in range(N_AGENTS):
            ac.append(MLPActorCritic(N_AGENTS, 1, hidden_sizes=(HIDDEN_SIZE,) * N_AGENTS))
        ac_targ = copy.deepcopy(ac)
        for i in range(N_AGENTS):
            for p in ac_targ[i].parameters():
                p.requires_grad = False
        q_params = [
            itertools.chain(ac[i].q1.parameters(), ac[i].q2.parameters()) for i in range(N_AGENTS)
        ]
        replay_buffer = []
        pi_optimizer = []
        q_optimizer = []
        temperature_optimizer = []
        target_entropy = [-1] * N_AGENTS # SAC paper recommends an entropy of minus the dimension of the action space
        log_temperature = []
        for i in range(N_AGENTS):
            replay_buffer.append(ReplayBuffer(obs_dim=N_AGENTS, act_dim=1, size=BUF_SIZE))
            pi_optimizer.append(torch.optim.Adam(ac[i].pi.parameters(), lr=INITIAL_LR_ACTOR))
            q_optimizer.append(torch.optim.Adam(q_params[i], lr=INITIAL_LR_CRITIC))
            log_temperature.append(torch.zeros(1, requires_grad=True, device=device))
            temperature_optimizer.append(torch.optim.Adam([log_temperature[i]], lr=INITIAL_LR_TEMP))
        action = torch.zeros([N_AGENTS]).to(device)
        price = torch.zeros([N_AGENTS]).to(device)
        for t in tqdm(range(MAX_T)):
            with torch.no_grad():
                for i in range(N_AGENTS):
                    # Randomly explore at the beginning
                    if t < BATCH_SIZE * 10:
                        action[i] = torch.rand(1)
                    else:
                        # Choose the deterministic policy more often
                        #if np.random.rand() < EXP_RATE[t]:
                        action[i] = ac[i].act(state, deterministic=False).squeeze()
                        #else:
                        #    action[i] = ac[i].act(state, deterministic=True).squeeze()
                    price[i] = scale_price(action[i], c)
                profit = compute_profit(ai, a0, mu, c, price)
                for i in range(N_AGENTS):
                    replay_buffer[i].store(state, action[i], profit[i], price)
                state = price.unsqueeze(0)
            if t >= BATCH_SIZE:
                if t % CKPT_T == 0:
                    for i in range(N_AGENTS):
                        ac[i].checkpoint(seed, out_dir, t, i)
                        ac_targ[i].checkpoint(f"target{seed}", out_dir, t, i)
                q_loss, pi_loss, temp_loss = zip(
                    *[
                        update(
                            ac[i],
                            ac_targ[i],
                            q_optimizer[i],
                            pi_optimizer[i],
                            temperature_optimizer[i],
                            target_entropy[i],
                            log_temperature[i],
                            q_params[i],
                            data=replay_buffer[i].sample_batch(BATCH_SIZE),
                        )
                        for i in range(N_AGENTS)
                    ]
                )
                writer.add_scalars("pi_loss", {f"{i}": pi_loss[i] for i in range(N_AGENTS)}, t)
                writer.add_scalars("q_loss", {f"{i}": q_loss[i] for i in range(N_AGENTS)}, t)
                writer.add_scalars("temp_loss", {f"{i}": temp_loss[i] for i in range(N_AGENTS)}, t)
            writer.add_scalars("price", {f"{i}": price[i] for i in range(N_AGENTS)}, t)
            writer.add_scalars("profit", {f"{i}": profit[i] for i in range(N_AGENTS)}, t)
        for i in range(N_AGENTS):
            ac[i].checkpoint(seed, out_dir, t, i)
            ac_targ[i].checkpoint(f"target{seed}", out_dir, t, i)
        writer.close()


if __name__ == "__main__":
    main()
