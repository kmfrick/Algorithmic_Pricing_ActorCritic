#!/usr/bin/env python3
# Multi-agent soft actor-critic in a competitive market
# Copyright (C) 2022 Kevin Michael Frick <kmfrick98@gmail.com>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import copy
import os

import itertools

from tqdm.auto import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from cpprb import ReplayBuffer

from utils import scale_price, profit_torch, profit_numpy

from model import *


class Agent:
    def __init__(
        self,
        n_agents,
        hidden_size,
        buf_size,
        lr_actor,
        lr_critic,
        lr_rew,
        ur_targ,
        batch_size,
        target_entropy,
        clip_norm=0.05,
    ):
        self.device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.ac = MLPActorCritic(
            n_agents,
            device=self.device,
            pi_hidden_size=(hidden_size // 8),
            q_hidden_size=hidden_size,
            activation=nn.Tanh,
        )
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
        self.temp_optimizer = torch.optim.Adam(
            [self.log_temp], lr=lr_actor, weight_decay=0
        )  # Doesn't make sense to use weight decay on the temperature
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

    def learn(self):
        batch = self.replay_buffer.sample(min(self.batch_size, self.replay_buffer.get_stored_size()))
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
            backup = (r - self.profit_mean) + q_pi_targ  # - temp * logp_a2 # Remove entropy in evaluation for SACLite

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
        return (
            loss_q.item(),
            loss_pi.item(),
            temp.item(),
            backup.mean().item(),
            np.mean([q_gn.item(), pi_gn.item(), temp_gn.item()]),
        )

    def checkpoint(self, fpostfix, out_dir, t, i):
        torch.save(self.ac.pi.state_dict(), f"{out_dir}/actor_weights_{fpostfix}_t{t}_agent{i}.pth")


def main():
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument("--out_dir", type=str, help="Directory")
    parser.add_argument("--device", type=int, help="CUDA device")
    parser.add_argument("--n_agents", type=int, help="Number of agents")
    parser.add_argument("--ai_last", type=float, help="Last agent's demand parameter")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    n_agents = args.n_agents
    if args.ai_last is not None:
        ai = [2.0] * (n_agents - 1)
        ai += [args.ai_last]
    else:
        ai = [2.0] * n_agents
    ai = np.array(ai)
    a0 = 0
    mu = 0.25
    c = 1
    MAX_T = int(8e4)
    CKPT_T = int(1e4)
    TARG_UPDATE_RATE = 0.999
    HIDDEN_SIZE = 2048
    INITIAL_LR_ACTOR = 1e-3
    INITIAL_LR_CRITIC = 5e-5
    AVG_REW_LR = 0.03
    TARGET_ENTROPY = -1
    BUF_SIZE = 20000
    BATCH_SIZE = 512
    IR_PERIODS = 20

    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    print(f"Will checkpoint every {CKPT_T} episodes")
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

    SEEDS = [250917, 50321, 200722, 190399, 40598, 220720, 71010, 130858, 150462, 1337, 9149,5283,9173,9933,4517,9257,9767,9564,5209,6531,6649,2963,10267,10830,7224,7789,6885,6627,7888,5849,5495,1148,8562,6579,6609,3951,9786,3099,2387,8413,7332,9575,6780,9001,9825,1725,7184,1251,6998,9921,4541,1281,3331,5882,9956,5504,1802,3491,9928,4002,8499,3903,] # 1st run
    #SEEDS=[6299,9397,7986,9865,10500,4875,10706,7213,4124,2250,6300,7129,5699,3450,4059,8667,5174,6889,3071,3286,6194,1665,4538,2217,9482,5592,2642,10421,4395,9911,4780,7462,6402,10471,4376,9788,2727,6906,3633,5876,10703,10954,4912,1822,5997,5153,3795,2275,4497,7908,8828,] # 2nd run
    # Equilibrium price computation by Massimiliano Furlan
    # https://github.com/massimilianofurlangit/algorithmic_pricing/blob/main/functions.jl
    # nash price is the price at which all firms are best-responding to each other
    # coop price maximizes the firms' joint profits
    print("Computing equilibrium prices...")
    nash_price = np.copy(ai)
    coop_price = np.copy(ai)
    def Ix(i, x):
        return np.array([x if i == j else 0 for j in range(n_agents)])
    def grad_profit(i, ai, a0, mu, c, p, h=1e-8):
        return (profit_numpy(ai, a0, mu, c, p + Ix(i, h))[i] - profit_numpy(ai, a0, mu, c, p - Ix(i, h))[i]) / (2 * h)
    def joint_profit(ai, a0, mu, c, p):
        return np.sum(profit_numpy(ai, a0, mu, c, p))
    def grad_joint_profit(ai, a0, mu, c, p, h = 1e-8):
        return (joint_profit(ai, a0, mu, c, p + h) - joint_profit(ai, a0, mu, c, p - h)) / (2 * h)
    while True:
        nash_price_ = np.copy(nash_price)
        for i in range(n_agents):
            df = grad_profit(i, ai, a0, mu, c, nash_price)
            while np.abs(df) > 1e-8:
                nash_price[i] += 1e-3 * df
                df = grad_profit(i, ai, a0, mu, c, nash_price)
        if np.any(nash_price_ - nash_price) < 1e-8:
            break
    df = grad_joint_profit(ai, a0, mu, c, coop_price)
    while np.abs(df) > 1e-7:
        lr = 0.01
        coop_price += lr * df
        df = grad_joint_profit(ai, a0, mu, c, coop_price)
    print(f"No. of agents = {n_agents}. Nash price = {nash_price}. Cooperation price = {coop_price}")
    xi = 0.1
    min_price = torch.tensor(nash_price - xi, device=device)
    max_price = torch.tensor(coop_price + xi, device=device)
    ai = torch.tensor(ai, device=device)
    for session in range(len(SEEDS)):
        fpostfix = SEEDS[session]
        # Random seeds
        np.random.seed(SEEDS[session])
        torch.manual_seed(SEEDS[session])
        # Initial state is random
        state = scale_price(torch.tanh(torch.randn([n_agents], device=device)), min_price, max_price).to(device)
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
                    TARGET_ENTROPY,
                )
            )
        t_tq = tqdm(range(MAX_T + 1))
        for t in t_tq:
            with torch.no_grad():
                for i in range(n_agents):
                    if t < BATCH_SIZE:
                        action[i] = torch.tanh(torch.randn([1], dtype=torch.float64))  # Randomly explore at the beginning
                    else:
                        action[i] = agents[i].act(state).squeeze()
                price = scale_price(action, min_price, max_price)
                price_history[:, t] = price
                profits = profit_torch(ai, a0, mu, c, price)
                profit_history[:, t] = profits
                for i in range(n_agents):
                    agents[i].replay_buffer.add(
                        obs=state.cpu(),
                        act=action[i].cpu(),
                        rew=profits[i].cpu(),
                        obs2=price.cpu(),
                    )
                    if t % CKPT_T == 0:
                        agents[i].checkpoint(fpostfix, out_dir, t, i)
            if t > 1:
                for i in range(n_agents):
                    q_loss[i], pi_loss[i], temp[i], backup[i], grad_norm[i] = agents[i].learn()
                with torch.no_grad():
                    start_t = max(t - BATCH_SIZE, 0)
                    avg_price = np.round(torch.mean(price_history[:, start_t:t], dim=1).cpu().numpy(), 3)
                    std_price = np.round(torch.std(price_history[:, start_t:t], dim=1).cpu().numpy(), 3)
                    avg_profit = np.round(torch.mean(profit_history[:, start_t:t], dim=1).cpu().numpy(), 3)
                ql = np.round(q_loss, 3)
                pl = np.round(pi_loss, 3)
                te = np.round(temp, 3)
                bkp = np.round(backup, 3)
                gn = np.round(grad_norm, 3)
                t_tq.set_postfix_str(
                    f"p = {avg_price}, std = {std_price}, P = {avg_profit}, QL = {ql}, PL = {pl}, temp = {te}, backup = {bkp}, GN = {gn}"
                )
            for i in range(n_agents):
                agents[i].update_avg_reward(state, action[i].double(), profits[i], price.unsqueeze(0))
            # CRUCIAL and easy to overlook: state = price
            state = price.unsqueeze(0)
        np.save(f"{out_dir}/session_prices_{fpostfix}.npy", price_history.detach())
        np.save(f"{out_dir}/session_profits_{fpostfix}.npy", profit_history.detach())


if __name__ == "__main__":
    main()
