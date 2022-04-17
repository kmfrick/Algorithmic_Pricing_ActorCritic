import copy
import os
import itertools

from tqdm.auto import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.profiler import profile, record_function, ProfilerActivity

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
        self.device = (
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )
        self.obs_buf = torch.zeros(
            combined_shape(size, obs_dim), dtype=torch.float32
        ).to(self.device)
        self.obs2_buf = torch.zeros(
            combined_shape(size, obs_dim), dtype=torch.float32
        ).to(self.device)
        self.act_buf = torch.zeros(
            combined_shape(size, act_dim), dtype=torch.float32
        ).to(self.device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32).to(self.device)
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
        self.device = (
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(
            obs_dim, act_dim, hidden_sizes, activation
        ).to(self.device)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(
            self.device
        )
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(
            self.device
        )

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a

    def checkpoint(self, fpostfix, out_dir, t, i):
        torch.save(
            self.pi.state_dict(),
            f"{out_dir}/actor_weights_{fpostfix}_t{t}_agent{i}.pth",
        )
        torch.save(
            self.q1.state_dict(), f"{out_dir}/q1_weights_{fpostfix}_t{t}_agent{i}.pth"
        )
        torch.save(
            self.q2.state_dict(), f"{out_dir}/q2_weights_{fpostfix}_t{t}_agent{i}.pth"
        )


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
    q = torch.exp((ai - p) / mu) / (
        torch.sum(torch.exp((ai - p) / mu)) + torch.exp(a0 / mu)
    )
    pi = (p - c) * q
    return pi


def impulse_response(
    ir_periods, a0, ai, c, fpostfix, mu, n_agents, nash_price, actor, out_dir, state, t
):
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    price = torch.zeros([n_agents]).to(device)
    with torch.no_grad():
        ir_profits = torch.zeros([n_agents, ir_periods])
        ir_prices = torch.zeros([n_agents, ir_periods])
        for t in range(ir_periods):
            for i in range(n_agents):
                price[i] = scale_price(actor[i].act(state, deterministic=True), c)
                ir_prices[i, t] = price[i]
            if (ir_periods / 2) <= t <= (ir_periods / 2 + 3):
                price[0] = nash_price
                ir_prices[0, t] = price[0]
            ir_profits[:, t] = compute_profit(ai, a0, mu, c, price)
            state = price
        np.save(f"{out_dir}/ir_profits_{fpostfix}_t{t}.npy", ir_profits.detach())
        np.save(f"{out_dir}/ir_prices_{fpostfix}_t{t}.npy", ir_prices.detach())


def compute_loss_q(ac, ac_targ, data):
    DISCOUNT = 0.99
    TEMP = 0.1
    o, a, r, o2 = data["obs"], data["act"], data["rew"], data["obs2"]

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
        backup = r + DISCOUNT * (q_pi_targ - TEMP * logp_a2)

    # MSE loss against Bellman backup
    loss_q1 = F.smooth_l1_loss(q1, backup)
    loss_q2 = F.smooth_l1_loss(q2, backup)
    loss_q = loss_q1 + loss_q2

    return loss_q


def compute_loss_pi(ac, data, temp):
    o = data["obs"]
    pi, logp_pi = ac.pi(o)
    q1_pi = ac.q1(o, pi)
    q2_pi = ac.q2(o, pi)
    q_pi = torch.min(q1_pi, q2_pi)
    # Entropy-regularized policy loss
    loss_pi = (temp * logp_pi - q_pi).mean()
    return loss_pi


def update(ac, ac_targ, q_optimizer, pi_optimizer, q_params, data, temp):
    TARG_UPDATE_RATE = 0.999

    # First run one gradient descent step for Q1 and Q2
    q_optimizer.zero_grad(set_to_none=True)
    loss_q = compute_loss_q(ac, ac_targ, data)
    loss_q.backward()
    q_optimizer.step()

    # Freeze Q-networks so you don't waste computational effort
    # computing gradients for them during the policy learning step.
    for p in q_params:
        p.requires_grad = False

    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad(set_to_none=True)
    loss_pi = compute_loss_pi(ac, data, temp)
    loss_pi.backward()
    pi_optimizer.step()

    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in q_params:
        p.requires_grad = True

    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(TARG_UPDATE_RATE)
            p_targ.data.add_((1 - TARG_UPDATE_RATE) * p.data)

    return loss_q.item(), loss_pi.item()


def get_action(ac, o, deterministic=False):
    return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)


def checkpoint(
    ir_periods,
    nash_price,
    a0,
    ai,
    c,
    mu,
    ac,
    ac_targ,
    out_dir,
    fpostfix,
    n_agents,
    state,
    t,
):
    for i in range(n_agents):
        ac[i].checkpoint(fpostfix, out_dir, t, i)
        ac_targ[i].checkpoint(f"target{fpostfix}", out_dir, t, i)
    impulse_response(
        ir_periods, a0, ai, c, fpostfix, mu, n_agents, nash_price, ac, out_dir, state, t
    )
    save_state_action_map(ac, n_agents, c, f"final_{fpostfix}", out_dir)


def scale_price(price, c):
    return price + c

NASH_PRICE = 1.47293

def main():
    out_dir = "exp_kaggle_new"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    n_agents = 2
    ai = torch.tensor([2.0] * n_agents).to(device)
    a0 = torch.tensor([0]).to(device)
    mu = torch.tensor([0.25]).to(device)
    c = 1
    BATCH_SIZE = 128
    IR_PERIODS = 20
    HIDDEN_SIZE = 256
    INITIAL_LR_ACTOR = 3e-3
    INITIAL_LR_CRITIC = 3e-3
    MAX_T = int(1e5)
    BUF_SIZE = MAX_T // 10
    CKPT_T = MAX_T // 10
    TEMP_DECAY = -1e-4
    print(f"Will checkpoint every {CKPT_T} episodes")

    NASH_PRICE = 1.47293
    COOP_PRICE = 1.92497
    SEEDS = [54321]

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
        ac_targ = copy.deepcopy(ac)
        for i in range(n_agents):
            for p in ac_targ[i].parameters():
                p.requires_grad = False
        q_params = [
            itertools.chain(ac[i].q1.parameters(), ac[i].q2.parameters())
            for i in range(n_agents)
        ]
        replay_buffer = []
        pi_optimizer = []
        q_optimizer = []
        for i in range(n_agents):
            replay_buffer.append(ReplayBuffer(obs_dim=n_agents, act_dim=1, size=BUF_SIZE))
            pi_optimizer.append(torch.optim.Adam(ac[i].pi.parameters(), lr=INITIAL_LR_ACTOR))
            q_optimizer.append(torch.optim.Adam(q_params[i], lr=INITIAL_LR_CRITIC))
        action = torch.zeros([n_agents]).to(device)
        price = torch.zeros([n_agents]).to(device)
        # Arrays used to save metrics
        total_reward = torch.zeros([n_agents, MAX_T])
        price_history = torch.zeros([n_agents, MAX_T])
        q_loss = np.zeros([n_agents])
        pi_loss = np.zeros([n_agents])
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
                        replay_buffer[i].store(state, action[i], profits[i], price)
                        total_reward[i, t] = profits[i]
                    state = price.unsqueeze(0)
                    if t > 0 and t % CKPT_T == 0:
                        checkpoint(
                            IR_PERIODS,
                            NASH_PRICE,
                            a0,
                            ai,
                            c,
                            mu,
                            ac,
                            ac_targ,
                            out_dir,
                            fpostfix,
                            n_agents,
                            state.clone(),
                            t,
                        )
                if t >= BATCH_SIZE:
                    for i in range(n_agents):
                        batch = replay_buffer[i].sample_batch(BATCH_SIZE)
                        q_loss[i], pi_loss[i] = update(
                            ac[i],
                            ac_targ[i],
                            q_optimizer[i],
                            pi_optimizer[i],
                            q_params[i],
                            data=batch,
                            temp=np.exp(TEMP_DECAY * t)
                        )
                    q_loss = np.round(q_loss, 3)
                    pi_loss = np.round(pi_loss, 3)
                    with torch.no_grad():
                        start_t = t - BATCH_SIZE
                        avg_prof = np.round(
                            torch.mean(total_reward[:, start_t:t], dim=1).cpu().numpy(),
                            3,
                        )
                        avg_price = np.round(
                            torch.mean(price_history[:, start_t:t], dim=1)
                            .cpu()
                            .numpy(),
                            3,
                        )
                    t_tq.set_postfix_str(
                        f"p = {avg_price}, P = {avg_prof}, QL = {q_loss}, PL = {pi_loss}"
                    )
        np.save(f"{out_dir}/session_reward_{fpostfix}.npy", total_reward.detach())
        checkpoint(
            IR_PERIODS,
            NASH_PRICE,
            a0,
            ai,
            c,
            mu,
            ac,
            ac_targ,
            out_dir,
            fpostfix,
            n_agents,
            state.clone(),
            t,
        )


if __name__ == "__main__":
    main()
