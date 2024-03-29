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

import os

import torch
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

import numpy as np
import pandas as pd

from utils import profit_numpy, scale_price, grad_desc
from model import SquashedGaussianMLPActor

# It's also possible to use the reduced notation by directly setting font.family:
plt.rcParams.update({
  "text.usetex": True,
})

plt.rcParams['figure.figsize'] = [6.4, 2.4]


def plot_heatmap(arr,  min_price, max_price, w=100):
    plt.tight_layout()
    for i, a in enumerate(arr):
        ax = plt.subplot(1, len(arr), i + 1)
        im = plt.imshow(
            a.reshape([w, w]),
            cmap="Reds",
            extent=(min_price[0], max_price[0], min_price[1], max_price[1]),
            aspect="auto",
            origin="lower",
        )
        ax.set_xlabel("$p_1$")
        ax.set_ylabel("$p_2$")
        plt.colorbar(orientation="horizontal")


def plot_profits(n_agents, profits, pg, movavg_span):
    profits = np.apply_along_axis(pg, 0, profits)
    for i in range(n_agents):
        r_series = pd.Series(profits[i, :]).rolling(window=movavg_span)
        plt.plot(r_series.mean())
        plt.fill_between(
            range(len(r_series.mean())),
            r_series.mean() - r_series.std(),
            r_series.mean() + r_series.std(),
            alpha=0.2,
        )
    plt.axhline(0)
    plt.axhline(1)


def main():
    sns.set_style("ticks")
    sns.set_context("paper")
    parser = argparse.ArgumentParser(description="Plot and write data")
    parser.add_argument("--actor_hidden_size", type=int)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--filename", type=str, default="experiments.csv")
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--out_dir", type=str, help="Directory")
    parser.add_argument("--plot_intermediate", action="store_const", const=True, default=False)
    parser.add_argument("--seeds", type=int, help="Random seeds", nargs="+")
    parser.add_argument("--movavg_span", type=int, help="Moving average span", default=1000)
    parser.add_argument("--parse_csv", action="store_const", const=True, default=False)
    parser.add_argument("--n_agents", type=int, help="Number of agents")
    parser.add_argument("--ai_last", type=float, help="Last agent's demand parameter")
    parser.add_argument("--demand_std", type=float, help="Standard deviation of a0 (for stochastic demand). Will be ignored if 0 or negative.", default=0)
    args = parser.parse_args()
    out_dir = args.out_dir
    n_agents = args.n_agents
    if args.ai_last is not None:
        ai = [2.0] * (n_agents - 1)
        ai += [args.ai_last]
    else:
        ai = [2.0] * n_agents
    ai = np.array(ai)
    a0 = 0
    a0_std = args.demand_std
    mu = 0.25
    c = 1

    # Create output directory
    os.makedirs(f"{out_dir}_plots", exist_ok=True)
    deviation_types = ["nash", "br", "coop", "cost"]
    if args.parse_csv:
        df = pd.read_csv(args.filename)
        df = df.drop(df.columns[[0, 1]], axis=1)
        df["dev_profit_percent_coop"] = None
        df["dev_profit_percent_cost"] = None
        df["dev_profit_percent_nash"] = None
        df["dev_profit_percent_br"] = None
        df["dev_profit_diff_coop"] = None
        df["dev_profit_diff_cost"] = None
        df["dev_profit_diff_nash"] = None
        df["dev_profit_diff_br"] = None
        # Separate columns for each deviation type
        for dtype in deviation_types:
            for seed in df.seed.unique():
                for t in df.t.unique():
                    df.loc[(df["seed"] == seed) & (df["t"] == t), f"dev_profit_percent_{dtype}"] = df.loc[(df["seed"] == seed) & (df["t"] == t) & (df["deviation_type"] == dtype),"deviation_profit_percent"].item()
                    df.loc[(df["seed"] == seed) & (df["t"] == t), f"dev_profit_diff_{dtype}"] = df.loc[(df["seed"] == seed) & (df["t"] == t) & (df["deviation_type"] == dtype),"differential_deviation_profit"].item()
        df = df.drop(["deviation_type", "deviation_profit_percent", "differential_deviation_profit"], axis = 1)
        df["deviation_profit_percent"] = df[[f"dev_profit_percent_{dtype}" for dtype in deviation_types]].mean(axis=1)
        df["differential_deviation_profit"] = df[[f"dev_profit_diff_{dtype}" for dtype in deviation_types]].mean(axis=1)
        df = df.drop_duplicates()

        for dtype in deviation_types:
            df[f"unprofitable_dev_diff_{dtype}"]  = (df[f"dev_profit_diff_{dtype}"] <  0).astype(int)
            df[f"unprofitable_dev_percent_{dtype}"]  = (df[f"dev_profit_percent_{dtype}"] <  0).astype(int)
        # Interesting dtype: best-response
        dtype = "br"
        for t in df.t.unique():
          df_s = df.loc[df.t == t,:]
          plt.hist(df_s["profit_gain"], align="left")
          plt.xlabel(f"Profit gains (t = {t})")
          sns.despine()
          plt.savefig(f"{out_dir}_plots/pg_hist_t{t}.svg")
          plt.clf()
          plt.hist(df_s[f"dev_profit_diff_{dtype}"], align="left")
          plt.xlabel(f"Differential deviation profit (t = {t})")
          sns.despine()
          plt.savefig(f"{out_dir}_plots/diff_dev_prof_hist_t{t}.svg")
          plt.clf()
          plt.hist(df_s[f"dev_profit_percent_{dtype}"], align="left")
          plt.xlabel(f"Discounted deviatiion profit (t = {t}, $\gamma$ = {args.discount})")
          sns.despine()
          plt.savefig(f"{out_dir}_plots/disc_dev_prof_hist_t{t}.svg")
          plt.clf()
        exit()



    # Equilibrium price computation by Massimiliano Furlan
    # https://github.com/massimilianofurlangit/algorithmic_pricing/blob/main/functions.jl
    # Nash price is the price at which all firms are best-responding to each other
    # Cooperation price maximizes the firms' joint profits
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
    min_price = nash_price - xi
    max_price = nash_price + xi
    nash = profit_numpy(ai, a0, mu, c, np.ones(n_agents) * nash_price)
    coop = profit_numpy(ai, a0, mu, c, np.ones(n_agents) * coop_price)
    ir_periods = 30
    dev_t = ir_periods // 2
    df = pd.DataFrame()
    def pg(x):
        return (x - nash) / (coop - nash)

    for t_da in range(1, 8):
        t_max = t_da * 10000
        r = []
        last_prof_gains = {}
        prof_gains_start_meas_t = t_max - args.movavg_span
        state_action_maps = []
        for seed in args.seeds:
            avg_dev_gain = {d: 0 for d in deviation_types}
            avg_dev_diff_profit = {d: 0 for d in deviation_types}
            ir_arrays_compliant = {d: [] for d in deviation_types}
            ir_arrays_defector = {d: [] for d in deviation_types}
            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            ir_profit_periods = 1000
            actor = []
            # Load network parameters
            for i in range(n_agents):
                a = SquashedGaussianMLPActor(n_agents, args.actor_hidden_size)
                a.load_state_dict(
                    torch.load(
                        f"{out_dir}/actor_weights_{seed}_t{t_max}_agent{i}.pth",
                        map_location=torch.device(device),
                    )
                )
                actor.append(a)
            with torch.inference_mode():
                # Compute session profit gains
                raw_out = np.load(f"{out_dir}/session_prices_{seed}.npy")
                raw_out = raw_out[:, :t_max]
                if np.min(raw_out) > c:  # It's prices
                    profits_cur = np.apply_along_axis(lambda x : profit_numpy(ai, a0, mu, c, x), 0, raw_out)
                if args.plot_intermediate:
                    plot_profits(n_agents, profits_cur, pg, args.movavg_span)
                    sns.despine()
                    plt.savefig(f"{out_dir}_plots/{seed}_profit_t{t_max}.svg")
                    plt.clf()
                r.append(profits_cur)
                session_prof_gains = np.mean(np.apply_along_axis(pg, 0, profits_cur[:, prof_gains_start_meas_t:]))
                if np.max(raw_out) < c: # We don't have prices
                    last_prices = torch.zeros(n_agents)
                    for i in range(0, n_agents):
                        last_prices[i] = (coop_price - nash_price) * session_prof_gains + nash_price
                else:
                    last_prices = torch.tensor(raw_out[:, prof_gains_start_meas_t:]).mean(1)
                print(f"Profits at convergence for seed {seed} = {session_prof_gains}")
                print(f"Prices at convergence for seed {seed} = {last_prices}")
                last_prof_gains[seed] = session_prof_gains

                if n_agents == 2:
                    # Create and plot state-action map
                    print("Computing state-action map...")
                    A = [np.zeros([args.grid_size, args.grid_size]) for i in range(n_agents)]
                    for i in range(n_agents):
                        w1 = torch.linspace(min_price[0], max_price[0], args.grid_size, requires_grad=False)
                        w2 = torch.linspace(min_price[1], max_price[1], args.grid_size, requires_grad=False)
                        for a_i, p1 in enumerate(w1):
                            for a_j, p2 in enumerate(w2):
                                state = torch.tensor([[p1, p2]])
                                action, _ = actor[i](state, deterministic=True, with_logprob=False)
                                a = scale_price(action, min_price[i], max_price[i]).detach()
                                # print(f"{state} -> {a}")
                                A[i][a_i, a_j] = a.numpy()
                    state_action_maps.append(A)
                    if args.plot_intermediate:
                        plot_heatmap(A, min_price, max_price, w=args.grid_size)
                        plt.savefig(f"{out_dir}_plots/{seed}_state_action_map_t{t_max}.svg")
                        plt.clf()
                # Plot impulse responses
                for deviation_type in avg_dev_gain.keys():
                    for j in range(n_agents):
                        # Impulse response
                        price_history = np.zeros([n_agents, ir_profit_periods])
                        state = last_prices.clone()
                        price = state.clone()
                        initial_state = state.clone()
                        # First compute non-deviation profits
                        nondev_profit = 0
                        leg = ["Non-deviating agent"] * n_agents
                        leg[j] = "Deviating agent"
                        avg_rew = np.zeros(n_agents)
                        for t in range(ir_profit_periods):
                            action = np.array([actor[i](state.unsqueeze(0), deterministic=True, with_logprob=False)[0].item() for i in range(n_agents)])
                            price = scale_price(action, min_price, max_price)
                            if t >= dev_t:
                                nondev_profit += profit_numpy(ai, a0, mu, c, price)[j] * args.discount ** (t - dev_t)
                            price_history[:, t] = price
                            state = torch.tensor(price)
                            conv_price = price.copy()
                            avg_rew += profit_numpy(ai, a0, mu, c, conv_price)
                        avg_rew /= ir_profit_periods
                        # Now compute deviation profits
                        dev_profit = 0
                        state = initial_state.clone()
                        for t in range(ir_profit_periods):
                            action = np.array([actor[i](state.unsqueeze(0), deterministic=True, with_logprob=False)[0].item() for i in range(n_agents)])
                            price = scale_price(action, min_price, max_price)
                            if t == dev_t:
                                if deviation_type == "nash":
                                    br = nash_price[j]
                                elif deviation_type == "coop":
                                    br = coop_price[j]
                                elif deviation_type == "cost":
                                    br = c
                                else:
                                    br = grad_desc(lambda x : profit_numpy(ai, a0, mu, c, x), price.copy(), j)
                                price[j] = br
                            if t >= dev_t:
                                dev_profit += profit_numpy(ai, a0, mu, c, price)[j] * args.discount ** (t - dev_t)
                            price_history[:, t] = price
                            state = torch.tensor(price)
                        dev_gain = (dev_profit / nondev_profit - 1) * 100
                        avg_dev_gain[deviation_type] += dev_gain
                        dev_diff_profit = (np.apply_along_axis(lambda x : profit_numpy(ai, a0, mu, c, x), 0, price_history).T - avg_rew).T.sum(axis=1)[j]
                        avg_dev_diff_profit[deviation_type] += dev_diff_profit
                        print(f"Deviation differential profits = {dev_diff_profit} (non-dev is 0)")
                        print(
                            f"Non-deviation discounted profits = {nondev_profit:.3f}; Deviation profits = {dev_profit:.3f}; Deviation gain = {dev_gain:.3f}%"
                        )
                        if args.plot_intermediate:
                            for i in range(n_agents):
                                price_series = price_history[i, (dev_t - 1):ir_periods]
                                plt.scatter(list(range(len(price_series))), price_series, s = 16)
                            plt.legend(leg)
                            for i in range(n_agents):
                                plt.plot(price_history[i, (dev_t - 1):ir_periods], linestyle="dashed")
                            plt.ylim(np.min(min_price), np.max(max_price))
                            sns.despine()
                            plt.savefig(f"{out_dir}_plots/{seed}_ir_{deviation_type}_t{t_max}_agent{j}.svg")
                            plt.clf()
                        ir_arrays_defector[deviation_type].append(price_history[j, :ir_periods])
                        for i in range(n_agents):
                            if i != j:
                                ir_arrays_compliant[deviation_type].append(price_history[i, :ir_periods])
                    new_row = {
                        "id": len(df),
                        "experiment_name": out_dir,
                        "seed": int(seed),
                        "profit_gain": session_prof_gains,
                        "t": t_max,
                        "actor_hidden_size": args.actor_hidden_size,
                        "discount": args.discount,
                        "deviation_profit_percent": avg_dev_gain[deviation_type] / n_agents,
                        "deviation_type": deviation_type,
                        "differential_deviation_profit": avg_dev_diff_profit[deviation_type] / n_agents
                    }
                    if len(df) == 0:
                        df = pd.DataFrame(columns=list(new_row.keys()))
                    df.loc[len(df)] = new_row
        # Plot average profits
        profits = np.stack(r, axis=0).mean(axis=0)
        plot_profits(n_agents, profits, pg, args.movavg_span)
        sns.despine()
        plt.savefig(f"{out_dir}_plots/avg_profit_gain_t{t_max}.svg")
        plt.clf()
        # Plot average profits across agents
        print(profits.shape)
        profit_gains = np.apply_along_axis(pg, 0, profits)
        pg_series = pd.Series(profit_gains.mean(axis=0)).rolling(window=args.movavg_span)
        plt.plot(pg_series.mean(), "b-")
        plt.fill_between(
            range(len(pg_series.mean())),
            pg_series.mean() - pg_series.std(),
            pg_series.mean() + pg_series.std(),
            alpha=0.2,
        )
        plt.axhline(0)
        plt.axhline(1)
        sns.despine()
        plt.savefig(f"{out_dir}_plots/avg_mean_profit_gain_t{t_max}.svg")
        plt.clf()

        # Plot average state-action heatmap
        if n_agents == 2:
            average_state_action_map = np.stack(state_action_maps, axis=0).mean(axis=0)
            plot_heatmap(
                average_state_action_map,
                min_price,
                max_price,
                w=args.grid_size,
            )
            plt.savefig(f"{out_dir}_plots/avg_state_action_heatmap_t{t_max}.svg")
            plt.clf()

        # Plot average IR
        for deviation_type in avg_dev_gain.keys():
            ir_stack_compliant = np.stack(ir_arrays_compliant[deviation_type], axis=0)
            ir_stack_defector = np.stack(ir_arrays_defector[deviation_type], axis=0)
            ir_box_compliant = [
                ir_stack_compliant[:, t] - ir_stack_compliant[:, dev_t - 1] for t in range(dev_t - 1, ir_periods)
            ]
            ir_box_defector = [ir_stack_defector[:, t] - ir_stack_defector[:, dev_t - 1] for t in range(dev_t - 1, ir_periods)]
            plt.boxplot(ir_box_compliant)
            sns.despine()
            plt.savefig(f"{out_dir}_plots/avg_ir_{deviation_type}_box_compliant_t{t_max}.svg")
            plt.clf()
            plt.boxplot(ir_box_defector)
            sns.despine()
            plt.savefig(f"{out_dir}_plots/avg_ir_{deviation_type}_box_defector_t{t_max}.svg")
            plt.clf()
            ir_mean_compliant = ir_stack_compliant.mean(axis=0)
            ir_mean_defector = ir_stack_defector.mean(axis=0)

            ir_mean_compliant = ir_mean_compliant[(dev_t - 1) :]
            ir_mean_defector = ir_mean_defector[(dev_t - 1) :]

            leg = ["Non-deviating agent", "Deviating agent"]
            if n_agents > 2:
                leg[0] = "Non-deviating agents (mean)"
            plt.scatter(list(range(len(ir_mean_compliant))), ir_mean_compliant, s=16)
            plt.scatter(list(range(len(ir_mean_defector ))), ir_mean_defector, s=16)
            plt.legend(leg)
            plt.plot(ir_mean_compliant, linestyle="dashed")
            plt.plot(ir_mean_defector, linestyle="dashed")
            for i in range(n_agents):
                plt.axhline(nash_price[i], c=f"C{i}")
                plt.axhline(coop_price[i], c=f"C{i}")
            plt.ylim(np.min(min_price), np.max(max_price))
            sns.despine()
            plt.savefig(f"{out_dir}_plots/avg_ir_{deviation_type}_t{t_max}.svg")
            plt.clf()

    df.to_csv(args.filename)


if __name__ == "__main__":
    main()
