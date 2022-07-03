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

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import TanhNormal

# Soft Actor-Critic from OpenAI https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, hidden_size, activation=nn.Tanh):
        super().__init__()
        fc1 = nn.Linear(obs_dim, hidden_size)
        fc2 = nn.Linear(hidden_size, hidden_size)
        self.net = nn.Sequential(fc1, activation(), fc2, activation())
        self.mu = nn.Linear(hidden_size, 1)
        self.std = nn.Linear(hidden_size, 1)
        self.t = 0

    def forward(self, obs, deterministic=False, with_logprob=True):
        x = self.net(obs.float())
        mu = self.mu(x)

        # Pre-squash distribution and sample
        if deterministic:
            # Only used for evaluating policy at test time.
            u, a = None, torch.tanh(mu)
        else:
            std = self.std(x)
            std = F.softplus(std)
            # torch.clamp(std, max = -(20/8e4 ** 3) * (self.t ** 3) + 20) # Decaying max variance. THIS BREAKS THE GRADIENT AT THE BORDER!
            # self.t += 1
            dist = TanhNormal(mu, std)
            u, a = dist.rsample_with_pre_tanh_value()

        if with_logprob:
            logp_pi = dist.log_prob(value=a, pre_tanh_value=u).double()
        else:
            logp_pi = None

        return a.double(), logp_pi


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, hidden_size, activation):
        super().__init__()
        fc1 = nn.Linear(obs_dim + 1, hidden_size)
        fc2 = nn.Linear(hidden_size, hidden_size)
        out = nn.Linear(hidden_size, 1)
        self.net = nn.Sequential(fc1, activation(), fc2, activation(), out)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q = self.net(x.float())
        q = F.softplus(q)
        return torch.squeeze(q, -1).double()  # Critical to ensure q has the right shape


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, pi_hidden_size, q_hidden_size, device, activation):
        super().__init__()

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, pi_hidden_size, activation).to(device)
        self.q1 = MLPQFunction(obs_dim, q_hidden_size, activation).to(device)
        self.q2 = MLPQFunction(obs_dim, q_hidden_size, activation).to(device)
