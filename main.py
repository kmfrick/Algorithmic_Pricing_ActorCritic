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

from utils import impulse_response, scale_price

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


def compute_profit(ai, a0, mu, c, p):
    q = torch.exp((ai - p) / mu) / (torch.sum(torch.exp((ai - p) / mu)) + np.exp(a0 / mu))
    pi = (p - c) * q
    return pi

class Agent():
    def __init__(self, n_agents, hidden_sizes, buf_size, lr_actor, lr_critic, lr_rew, ur_targ, batch_size, target_entropy):
        self.device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.ac = MLPActorCritic(n_agents, 1, device = self.device, hidden_sizes=hidden_sizes)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.replay_buffer = ReplayBuffer(
                    buf_size,
                    env_dict={
                        "obs": {"shape": n_agents},
                        "act": {"shape": 1},
                        "rew": {},
                        "obs2": {"shape": n_agents},
                    },
                )
        self.log_temp = torch.zeros(1, requires_grad=True, device=self.device)
        self.pi_optimizer = torch.optim.AdamW(self.ac.pi.parameters(), lr=lr_actor, weight_decay=1e-3)
        self.q_optimizer = torch.optim.AdamW(self.q_params, lr=lr_critic, weight_decay=1e-3)
        self.temp_optimizer = torch.optim.Adam([self.log_temp], lr=lr_actor, weight_decay = 0) # Doesn't make sense to use weight decay on the temperature
        self.ac_targ = copy.deepcopy(self.ac)
        # Freeze target network weights
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.batch_size = batch_size
        self.target_entropy = target_entropy
        self.lr_rew = lr_rew
        self.ur_targ = ur_targ
        self.profit_mean = 0

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            self.ac.eval()
            a, _ = self.ac.pi(obs, deterministic, False)
            self.ac.train()
            return a

    def learn(self, state, action, profit, next_state):
        batch = self.replay_buffer.sample(self.batch_size)

        o, a, r, o2 = (
            torch.tensor(batch["obs"], device=self.device).squeeze(),
            torch.tensor(batch["act"], device=self.device),
            torch.tensor(batch["rew"], device=self.device).squeeze(),
            torch.tensor(batch["obs2"], device=self.device).squeeze(),
        )
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi, logp_pi = self.ac.pi(o)
        # Entropy loss
        with torch.no_grad():
            temp_obj = logp_pi + self.target_entropy
        temp_loss = -(self.log_temp * temp_obj).mean()
        self.temp_optimizer.zero_grad(set_to_none=True)
        temp_loss.backward()
        self.temp_optimizer.step()
        temp = torch.exp(self.log_temp)

        # Entropy-regularized policy loss
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (temp * logp_pi - q_pi).mean()
        self.pi_optimizer.zero_grad(set_to_none=True)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions from current policy
            a2, logp_a2 = self.ac.pi(o2)
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = (r - self.profit_mean) + q_pi_targ - temp * logp_a2

        # MSE loss against Bellman backup
        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)
        loss_q = loss_q1 + loss_q2

        self.q_optimizer.zero_grad(set_to_none=True)
        loss_q.backward()
        self.q_optimizer.step()

        # Update average reward
        with torch.no_grad():
            q1_cur_targ = self.ac_targ.q1(state.squeeze(), action.unsqueeze(0))
            q2_cur_targ = self.ac_targ.q2(state.squeeze(), action.unsqueeze(0))
            q_cur_targ = torch.min(q1_cur_targ, q2_cur_targ)
            action_next, _ = self.ac.pi(next_state)
            q1_next_targ = self.ac_targ.q1(next_state, action_next)
            q2_next_targ = self.ac_targ.q2(next_state, action_next)
            q_next_targ = torch.min(q1_next_targ, q2_next_targ)
            self.profit_mean += self.lr_rew * (profit - self.profit_mean + q_next_targ - q_cur_targ).squeeze()

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.ur_targ)
                p_targ.data.add_((1 - self.ur_targ) * p.data)
        return loss_q.item(), loss_pi.item(), temp.item(), backup.mean().item()

    def checkpoint(self, fpostfix, out_dir, t, i):
        torch.save(self.ac.pi, f"{out_dir}/actor_weights_{fpostfix}_t{t}_agent{i}.pth")

def objective(trial):
    torch.cuda.set_device(int(sys.argv[2]))
    n_agents = 2
    ai = 2
    a0 = 0
    mu = 0.25
    c = 1
    MAX_T = int(1e5)
    CKPT_T = int(1e4)
    TARG_UPDATE_RATE = 0.999
    HIDDEN_SIZE = 2048
    MIN_LR = 3e-5
    MAX_LR = 3e-3
    INITIAL_LR_ACTOR = trial.suggest_loguniform("actor_lr", MIN_LR, MAX_LR)
    INITIAL_LR_CRITIC = trial.suggest_loguniform("critic_lr", MIN_LR, MAX_LR)
    AVG_REW_LR = 0.03
    TARGET_ENTROPY = -1
    MIN_BUF_SIZE = 5
    MAX_BUF_SIZE = MAX_T / 1000
    BUF_SIZE = trial.suggest_int("buf_size", MIN_BUF_SIZE, MAX_BUF_SIZE, log=True) * 1000
    BATCH_SIZE = 512
    IR_PERIODS = 20

    def Pi(p):
        q = np.exp((ai - p) / mu) / (np.sum(np.exp((ai - p) / mu)) + np.exp(a0 / mu))
        pi = (p - c) * q
        return pi
    print(trial.params)
    out_dir = f"{sys.argv[1]}_lr{INITIAL_LR_ACTOR}-{INITIAL_LR_CRITIC}_buf{BUF_SIZE}"
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
        action = torch.zeros([n_agents]).to(device)
        price = torch.zeros([n_agents]).to(device)
        # Arrays used to save metrics
        profit_history = torch.zeros([n_agents, MAX_T])
        price_history = torch.zeros([n_agents, MAX_T])
        q_loss = np.zeros([n_agents])
        pi_loss = np.zeros([n_agents])
        temp = np.zeros([n_agents])
        backup = np.zeros([n_agents])
        agents = []
        for i in range(n_agents):
            agents.append(
                Agent(
                    n_agents,
                    (HIDDEN_SIZE, HIDDEN_SIZE),
                    BUF_SIZE,
                    INITIAL_LR_ACTOR,
                    INITIAL_LR_CRITIC,
                    AVG_REW_LR,
                    TARG_UPDATE_RATE,
                    BATCH_SIZE,
                    TARGET_ENTROPY
                )
            )
        with tqdm(range(MAX_T)) as t_tq:
            for t in t_tq:
                with torch.no_grad():
                    for i in range(n_agents):
                        # Randomly explore at the beginning
                        if t < BATCH_SIZE * 10:
                            action[i] = torch.rand(1)
                        else:
                            action[i] = agents[i].act(state).squeeze()
                        price[i] = scale_price(action[i], c)
                        price_history[i, t] = price[i]
                    profits = compute_profit(ai, a0, mu, c, price)
                    profit_history[:, t] = profits
                    for i in range(n_agents):
                        agents[i].replay_buffer.add(
                            obs=state.cpu(),
                            act=action[i].cpu(),
                            rew=profits[i].cpu(),
                            obs2=price.cpu(),
                        )
                        if t > MAX_T / 3 and t % CKPT_T == 0:
                            agents[i].checkpoint(fpostfix, out_dir, t, i)
                    if t > 0 and  t % CKPT_T == 0:
                        impulse_response(n_agents, agents, price, IR_PERIODS, c, Pi)
                if t >= BATCH_SIZE:
                    for i in range(n_agents):
                        q_loss[i], pi_loss[i], temp[i], backup[i] = agents[i].learn(state, action[i], profits[i], price.unsqueeze(0))
                    with torch.no_grad():
                        start_t = t - BATCH_SIZE
                        avg_price = np.round(
                            torch.mean(price_history[:, start_t:t], dim=1).cpu().numpy(), 3,
                        )
                        avg_profit = np.round(
                            torch.mean(profit_history[:, start_t:t], dim=1).cpu().numpy(), 3,
                        )
                        ql = np.round(q_loss, 3)
                        pl = np.round(pi_loss, 3)
                        te = np.round(temp, 3)
                        bkp = np.round(backup, 3)

                    t_tq.set_postfix_str(
                        f"p = {avg_price}, P = {avg_profit}, QL = {ql}, PL = {pl}, temp = {te}, backup = {bkp}"
                    )
                # CRUCIAL and easy to overlook: state = price
                state = price.unsqueeze(0)
        np.save(f"{out_dir}/session_reward_{fpostfix}.npy", profit_history.detach())

        avg_dev_gain += impulse_response(n_agents, agents, price, IR_PERIODS, c, Pi)
        for i in range(n_agents):
            agents[i].checkpoint(fpostfix, out_dir, MAX_T, i)
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

