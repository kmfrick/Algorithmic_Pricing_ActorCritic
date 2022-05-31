import copy
import os
import sys
import itertools
import math
import random

from tqdm.auto import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from cpprb import ReplayBuffer

from utils import grad_desc, df

import logging
import optuna
from optuna.trial import TrialState


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


# Soft Actor-Critic from OpenAI https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    model = nn.Sequential(*layers)
    #model.apply(init_weights)
    return model


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.lz = math.log(math.sqrt(2 * math.pi))

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = random.gauss(0, 1) * std + mu

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for sigmoid squashing.
            # This formula is computed with the same procedure as the original SAC paper (arXiv 1801.01290)
            # in appendix C, but using the derivative of the sigmoid instead of tanh
            # It is more numerically stable than using the sigmoid derivative expressed as s(x) * (1 - s(x))
            # This can be tested by computing it for torch.linspace(-20, 20, 1000) for example
            var = std ** 2
            logp_pi = -(((pi_action - mu) ** 2) / (2 * var) - log_std - self.lz).sum(axis=-1)
            logp_pi += (2 * F.softplus(-pi_action) + pi_action).sum(dim=1)
        else:
            logp_pi = None

        pi_action = torch.sigmoid(pi_action)

        return pi_action, logp_pi


def softabs(x):
    return torch.where(torch.abs(x) < 1, 0.5 * x ** 2, -0.5 + torch.abs(x))


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
        if obs.dim() != 2:
            obs = obs.unsqueeze(0)
        v = self.v(obs)
        return v.squeeze()


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, device, activation=nn.LeakyReLU):
        super().__init__()

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            self.eval()
            a, _ = self.pi(obs, deterministic, False)
            self.train()
            return a

    def checkpoint(self, fpostfix, out_dir, t, i):
        torch.save(self.pi, f"{out_dir}/actor_weights_{fpostfix}_t{t}_agent{i}.pth")


def compute_profit(ai, a0, mu, c, p):
    q = torch.exp((ai - p) / mu) / (torch.sum(torch.exp((ai - p) / mu)) + np.exp(a0 / mu))
    pi = (p - c) * q
    return pi


def scale_price(price, c, d=None):
    if d is None:
        return price * c + c
    else:
        if c > d:
            c, d = d, c
        return price * (d - c) + c


def objective(trial):
    torch.cuda.set_device(int(sys.argv[2]))
    n_agents = 2
    ai = 2
    a0 = 0
    mu = 0.25
    c = 1
    BATCH_SIZE = 32
    HIDDEN_SIZE = 2048
    MIN_LR = 3e-5
    MAX_LR = 3e-3
    INITIAL_LR_ACTOR = trial.suggest_loguniform("actor_lr", MIN_LR, MAX_LR)
    INITIAL_LR_CRITIC = trial.suggest_loguniform("critic_lr", MIN_LR, MAX_LR)
    INITIAL_LR_TEMP = trial.suggest_loguniform("temp_lr", MIN_LR, MAX_LR)
    AVG_REW_LR = 0.03
    TARGET_ENTROPY = -1
    BUF_SIZE = 7000
    print(trial.params)
    MAX_T = int(1e5)
    CKPT_T = int(1e4)
    TARG_UPDATE_RATE = 0.999
    out_dir = f"{sys.argv[1]}_lr{INITIAL_LR_ACTOR}-{INITIAL_LR_CRITIC}-{INITIAL_LR_TEMP}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Will checkpoint every {CKPT_T} episodes")
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

    nash_price = 1.4729273733327568
    coop_price = 1.9249689958811602
    SEEDS = [250917, 50321, 200722]
    min_price = nash_price - 0.1
    max_price = coop_price + 0.08
    avg_dev_gain = 0
    for session in range(len(SEEDS)):
        fpostfix = SEEDS[session]
        # Random seeds
        np.random.seed(SEEDS[session])
        torch.manual_seed(SEEDS[session])
        # Initial state is random, but ensure prices are above marginal cost
        state = torch.rand(n_agents).to(device) + c
        state = state.unsqueeze(0)
        ac = []
        q_params = []
        replay_buffer = []
        pi_optimizer = []
        q_optimizer = []
        temp_optimizer = []
        log_temp = []
        for i in range(n_agents):
            ac.append(
                MLPActorCritic(n_agents, 1, device=device, hidden_sizes=(HIDDEN_SIZE,) * n_agents)
            )
            q_params.append(itertools.chain(ac[i].q1.parameters(), ac[i].q2.parameters()))
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
            log_temp.append(torch.zeros(1, requires_grad=True, device=device))
            pi_optimizer.append(
                torch.optim.AdamW(ac[i].pi.parameters(), lr=INITIAL_LR_ACTOR, weight_decay=1e-3)
            )
            q_optimizer.append(
                torch.optim.AdamW(q_params[i], lr=INITIAL_LR_CRITIC, weight_decay=1e-3)
            )
            temp_optimizer.append(
                torch.optim.AdamW([log_temp[i]], lr=INITIAL_LR_TEMP, weight_decay=1e-3)
            )
        # Create targt network
        ac_targ = copy.deepcopy(ac)
        for i in range(n_agents):
            for p in ac_targ[i].parameters():
                p.requires_grad = False
        profit_mean = torch.zeros([n_agents]).to(device)
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
                    with torch.no_grad():
                        total_reward[:, t] = profits
                    for i in range(n_agents):
                        replay_buffer[i].add(
                            obs=state.cpu(),
                            act=action[i].cpu(),
                            rew=profits[i].cpu(),
                            obs2=price.cpu(),
                        )
                    if t > MAX_T / 2 and t % CKPT_T == 0:
                        for i in range(n_agents):
                            ac[i].checkpoint(fpostfix, out_dir, t, i)
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
                        with torch.no_grad():
                            temp_obj = logp_pi + TARGET_ENTROPY
                        temp_loss = -(log_temp[i] * temp_obj).mean()
                        temp_optimizer[i].zero_grad(set_to_none=True)
                        temp_loss.backward()
                        temp_optimizer[i].step()
                        temp = torch.exp(log_temp[i])

                        # Entropy-regularized policy loss
                        q1_pi = ac[i].q1(o, pi)
                        q2_pi = ac[i].q2(o, pi)
                        q_pi = torch.min(q1_pi, q2_pi)
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
                            # Target actions from current policy
                            a2, logp_a2 = ac[i].pi(o2)
                            # Target Q-values
                            q1_pi_targ = ac_targ[i].q1(o2, a2)
                            q2_pi_targ = ac_targ[i].q2(o2, a2)
                            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                            backup = (r - profit_mean[i]) + q_pi_targ - temp * logp_a2

                        # MSE loss against Bellman backup
                        loss_q1 = F.mse_loss(q1, backup)
                        loss_q2 = F.mse_loss(q2, backup)
                        loss_q = loss_q1 + loss_q2

                        q_optimizer[i].zero_grad(set_to_none=True)
                        loss_q.backward()
                        q_optimizer[i].step()

                        # Update average reward
                        with torch.no_grad():
                            q1_cur_targ = ac_targ[i].q1(state.squeeze(), action[i].unsqueeze(0))
                            q2_cur_targ = ac_targ[i].q2(state.squeeze(), action[i].unsqueeze(0))
                            q_cur_targ = torch.min(q1_cur_targ, q2_cur_targ)
                            # CRUCIAL and easy to overlook step: state = price
                            state = price.unsqueeze(0)
                            action_next, _ = ac[i].pi(state)
                            q1_next_targ = ac_targ[i].q1(state, action_next)
                            q2_next_targ = ac_targ[i].q2(state, action_next)
                            q_next_targ = torch.min(q1_next_targ, q2_next_targ)
                            profit_mean[i] += AVG_REW_LR * (profits[i] - profit_mean[i] + q_next_targ - q_cur_targ).squeeze()

                        # Finally, update target networks by polyak averaging.
                        with torch.no_grad():
                            for p, p_targ in zip(ac[i].parameters(), ac_targ[i].parameters()):
                                # NB: We use an in-place operations "mul_", "add_" to update target
                                # params, as opposed to "mul" and "add", which would make new tensors.
                                p_targ.data.mul_(TARG_UPDATE_RATE)
                                p_targ.data.add_((1 - TARG_UPDATE_RATE) * p.data)
                        q_loss[i], pi_loss[i] = (
                            loss_q.item(),
                            loss_pi.item(),
                        )
                    q_loss = np.round(q_loss, 3)
                    pi_loss = np.round(pi_loss, 3)
                    with torch.no_grad():
                        start_t = t - BATCH_SIZE
                        avg_price = np.round(
                            torch.mean(price_history[:, start_t:t], dim=1).cpu().numpy(), 3,
                        )
                        avg_profit = np.round(profit_mean.cpu().detach().numpy(), 3,)
                        temp = np.round(np.array([torch.exp(a).item() for a in log_temp]), 3,)
                    bkp = backup.mean().item()
                    t_tq.set_postfix_str(
                        f"p = {avg_price}, P = {avg_profit}, QL = {q_loss.mean().item():.3f} PL = {pi_loss.mean().item():.3f}, te = {temp.mean().item():.3f}, bkp = {bkp:.3f}"
                    )
        np.save(f"{out_dir}/session_reward_{fpostfix}.npy", total_reward.detach())
        ir_periods = 20

        def Pi(p):
            q = np.exp((ai - p) / mu) / (np.sum(np.exp((ai - p) / mu)) + np.exp(a0 / mu))
            pi = (p - c) * q
            return pi

        with torch.no_grad():
            # Impulse response
            state = price.squeeze().clone().detach()
            print(f"Initial state = {state}")
            price = state.clone()
            initial_state = state.clone()
            ir_profit_periods = 1000
            for j in range(n_agents):
                # Impulse response
                price = state.clone()
                # First compute non-deviation profits
                DISCOUNT = 0.99
                nondev_profit = 0
                for t in range(ir_profit_periods):
                    for i in range(n_agents):
                        price[i] = scale_price(ac[i].act(state.unsqueeze(0))[0], c)
                    if t >= (ir_periods / 2):
                        nondev_profit += Pi(price.cpu().numpy())[j] * DISCOUNT ** (
                            t - ir_periods / 2
                        )
                    state = price
                # Now compute deviation profits
                dev_profit = 0
                state = initial_state.clone()
                for t in range(ir_profit_periods):
                    for i in range(n_agents):
                        price[i] = scale_price(ac[i].act(state.unsqueeze(0))[0], c)
                    if t == (ir_periods / 2):
                        br = grad_desc(Pi, price.cpu().numpy(), j)
                        price[j] = torch.tensor(br)
                    if t >= (ir_periods / 2):
                        dev_profit += Pi(price.cpu().numpy())[j] * DISCOUNT ** (t - ir_periods / 2)
                    state = price
                dev_gain = (dev_profit / nondev_profit - 1) * 100
                avg_dev_gain += dev_gain
                print(
                    f"Agent {i}: Non-deviation profits = {nondev_profit:.3f}; Deviation profits = {dev_profit:.3f}; Deviation gain = {dev_gain:.3f}%"
                )
        for i in range(n_agents):
            ac[i].checkpoint(fpostfix, out_dir, MAX_T, i)
            ac_targ[i].checkpoint(f"target{fpostfix}", out_dir, MAX_T, i)
    return avg_dev_gain / len(SEEDS)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} OUT_DIR NUM_DEVICE")
        exit(1)
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = sys.argv[1]  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True
    )
    study.optimize(objective, n_trials=10)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
