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

from utils import impulse_response, scale_price, TanhNormal

import logging
import optuna
from optuna.trial import TrialState


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


LOG_STD_MAX = 2
LOG_STD_MIN = -20


# Soft Actor-Critic from OpenAI https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, hidden_size, activation):
        super().__init__()
        fc1  = nn.Linear(obs_dim, hidden_size)
        fc2 = nn.Linear(hidden_size, hidden_size)
        self.net = nn.Sequential(fc1, activation(), fc2, activation())
        self.mu = nn.Linear(hidden_size, 1)
        self.log_std = nn.Linear(hidden_size, 1)
        self.net.apply(init_weights)
        self.mu.apply(init_weights)
        self.log_std.apply(init_weights)

    def forward(self, obs, deterministic=False, with_logprob=True):
        x = self.net(obs)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = TanhNormal(mu, std)

        # Pre-squash distribution and sample
        if deterministic:
            # Only used for evaluating policy at test time.
            u, a = None, mu
        else:
            u, a = dist.rsample_with_pre_tanh_value()

        if with_logprob:
            logp_pi = dist.log_prob(value=a, pre_tanh_value=u)
        else:
            logp_pi = None

        return a, logp_pi


def softabs(x):
    return torch.where(torch.abs(x) < 1, 0.5 * x ** 2, -0.5 + torch.abs(x))


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, hidden_size, activation):
        super().__init__()
        fc1  = nn.Linear(obs_dim + 1, hidden_size)
        fc2 = nn.Linear(hidden_size, hidden_size)
        out = nn.Linear(hidden_size, 1)
        self.net = nn.Sequential(fc1, activation(), fc2, activation(), out)
        self.net.apply(init_weights)

    def forward(self, obs, act):
        q = self.net(torch.cat([obs, act], dim=-1))
        #q = softabs(q)
        return torch.squeeze(q, -1)  # Critical to ensure q has the right shape

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden_size, device, activation=nn.ReLU):
        super().__init__()

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, hidden_size, activation).to(device)
        self.q1 = MLPQFunction(obs_dim, hidden_size, activation).to(device)
        self.q2 = MLPQFunction(obs_dim, hidden_size, activation).to(device)


def compute_profit(ai, a0, mu, c, p):
    q = torch.exp((ai - p) / mu) / (torch.sum(torch.exp((ai - p) / mu)) + np.exp(a0 / mu))
    pi = (p - c) * q
    return pi

class Agent():
    def __init__(self, n_agents, hidden_size, buf_size, lr_actor, lr_critic, lr_rew, ur_targ, batch_size, target_entropy, clip_norm=0.05):
        self.device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.ac = MLPActorCritic(n_agents, device = self.device, hidden_size=hidden_size)
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
        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=lr_actor)
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=lr_critic)
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
        self.clip_norm = clip_norm

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            self.ac.eval()
            a, _ = self.ac.pi(obs, deterministic, False)
            self.ac.train()
            return a

    def update_avg_reward(self, state, action, profit, next_state):
        with torch.no_grad():
            q1_cur_targ = self.ac_targ.q1(state.squeeze(), action.unsqueeze(0))
            q2_cur_targ = self.ac_targ.q2(state.squeeze(), action.unsqueeze(0))
            q_cur_targ = torch.min(q1_cur_targ, q2_cur_targ)
            action_next, _ = self.ac.pi(next_state)
            q1_next_targ = self.ac_targ.q1(next_state, action_next)
            q2_next_targ = self.ac_targ.q2(next_state, action_next)
            q_next_targ = torch.min(q1_next_targ, q2_next_targ)
            self.profit_mean += self.lr_rew * (profit - self.profit_mean + q_next_targ - q_cur_targ).squeeze()

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
        temp_gn = torch.nn.utils.clip_grad_norm_(self.log_temp, self.clip_norm)
        self.temp_optimizer.step()
        self.log_temp.data.clamp_(max=0)
        temp = torch.exp(self.log_temp)

        # Entropy-regularized policy loss
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (temp * logp_pi - q_pi).mean()
        self.pi_optimizer.zero_grad(set_to_none=True)
        loss_pi.backward()
        pi_gn = torch.nn.utils.clip_grad_norm_(self.ac.pi.parameters(), self.clip_norm)
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
            backup = (r - self.profit_mean) + q_pi_targ #- temp * logp_a2 # Remove entropy in evaluation for SACLite

        # MSE loss against Bellman backup
        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)
        loss_q = loss_q1 + loss_q2

        self.q_optimizer.zero_grad(set_to_none=True)
        loss_q.backward()
        q_gn = torch.nn.utils.clip_grad_norm_(self.q_params, self.clip_norm)
        self.q_optimizer.step()


        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.ur_targ)
                p_targ.data.add_((1 - self.ur_targ) * p.data)
        return loss_q.item(), loss_pi.item(), temp.item(), backup.mean().item(), np.mean([q_gn.item(), pi_gn.item(), temp_gn.item()])

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
    MAX_BUF_SIZE = 100
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
    max_price = coop_price + 0.1
    avg_dev_gain = 0
    for session in range(len(SEEDS)):
        fpostfix = SEEDS[session]
        # Random seeds
        np.random.seed(SEEDS[session])
        torch.manual_seed(SEEDS[session])
        # Initial state is random
        state = scale_price(torch.rand(n_agents) * 2 - 1, min_price, max_price).to(device)
        state = state.unsqueeze(0)
        action = torch.zeros([n_agents]).to(device)
        price = torch.zeros([n_agents]).to(device)
        # Arrays used to save metrics
        profit_history = torch.zeros([n_agents, MAX_T + 1])
        price_history = torch.zeros([n_agents, MAX_T + 1])
        q_loss = np.zeros([n_agents])
        pi_loss = np.zeros([n_agents])
        temp = np.zeros([n_agents])
        backup = np.zeros([n_agents])
        grad_norm = np.zeros([n_agents])
        agents = []
        for i in range(n_agents):
            agents.append(
                Agent(
                    n_agents,
                    HIDDEN_SIZE,
                    BUF_SIZE,
                    INITIAL_LR_ACTOR,
                    INITIAL_LR_CRITIC,
                    AVG_REW_LR,
                    TARG_UPDATE_RATE,
                    BATCH_SIZE,
                    TARGET_ENTROPY
                )
            )
        with tqdm(range(MAX_T + 1)) as t_tq:
            for t in t_tq:
                with torch.no_grad():
                    for i in range(n_agents):
                        if t < BATCH_SIZE:
                            action[i] = torch.rand(1) * 2 - 1 # Randomly explore at the beginning
                        else:
                            action[i] = agents[i].act(state).squeeze()
                        price[i] = scale_price(action[i], min_price, max_price)
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
                if t >= BATCH_SIZE:
                    for i in range(n_agents):
                        q_loss[i], pi_loss[i], temp[i], backup[i], grad_norm[i] = agents[i].learn(state, action[i], profits[i], price.unsqueeze(0))
                    with torch.no_grad():
                        start_t = t - BATCH_SIZE
                        avg_price = np.round(
                            torch.mean(price_history[:, start_t:t], dim=1).cpu().numpy(), 3,
                        )
                        std_price = np.round(
                            torch.std(price_history[:, start_t:t], dim=1).cpu().numpy(), 3,
                        )
                        avg_profit = np.round(
                            torch.mean(profit_history[:, start_t:t], dim=1).cpu().numpy(), 3,
                        )
                        ql = np.round(q_loss, 3)
                        pl = np.round(pi_loss, 3)
                        te = np.round(temp, 3)
                        bkp = np.round(backup, 3)
                        gn = np.round(grad_norm, 3)
                    t_tq.set_postfix_str(
                        f"p = {avg_price}, std = {std_price}, P = {avg_profit}, QL = {ql}, PL = {pl}, temp = {te}, backup = {bkp}, GN = {gn}"
                    )
                for i in range(n_agents):
                    agents[i].update_avg_reward(state, action[i], profits[i], price.unsqueeze(0))
                # CRUCIAL and easy to overlook: state = price
                state = price.unsqueeze(0)
        start_t = t - BATCH_SIZE
        profit_gain = (torch.mean(price_history[:, start_t:t], dim=1).cpu().numpy() - nash_price) / (coop_price - nash_price)
        print("PG = {profit_gain}")
        avg_dev_gain += impulse_response(n_agents, agents, ir_price, IR_PERIODS, c, Pi, max_price = max_price)
        np.save(f"{out_dir}/session_prices_{fpostfix}.npy", price_history.detach())
    return avg_dev_gain / len(SEEDS)


def main():
    parser = argparse.ArgumentParser(description="Run a hyperparameter study")
    parser.add_argument("--study_name", type=str, help="Directory")
    parser.add_argument("--device", type=int, help="CUDA device")
    args = parser.parse_args()
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = args.study_name # Unique identifier of the study.
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

