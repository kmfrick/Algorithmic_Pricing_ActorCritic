#!/usr/bin/env python3
import os

import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

import numpy as np
import pandas as pd

from cycler import cycler

from utils import profit_numpy, scale_price, grad_desc
from model import SquashedGaussianMLPActor


def plot_heatmap(arr, title, min_price, max_price, w=100):
    plt.tight_layout()
    for i, a in enumerate(arr):
        ax = plt.subplot(1, len(arr), i + 1)
        im = plt.imshow(
            a.reshape([w, w]),
            cmap="Reds",
            extent=(min_price, max_price, min_price, max_price),
            aspect="auto",
            origin="lower",
        )
        ax.set_xlabel("p0")
        ax.set_ylabel("p1")
        ax.set_title(f"{title}; Agent {i}")
        plt.colorbar(orientation="horizontal")


def plot_profits(n_agents, profits, pg, movavg_span):
    profits = pg(profits)
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
    parser = argparse.ArgumentParser(description="Plot and write data")
    parser.add_argument("--actor_hidden_size", type=int)
    parser.add_argument("--defect_to_c", action="store_const", const=True, default=False)
    parser.add_argument("--defect_to_coop", action="store_const", const=True, default=False)
    parser.add_argument("--defect_to_nash", action="store_const", const=True, default=False)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--filename", type=str, default="experiments.csv")
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--out_dir", type=str, help="Directory")
    parser.add_argument("--plot_intermediate", action="store_const", const=True, default=False)
    parser.add_argument("--seeds", type=int, help="Random seeds", nargs="+")
    parser.add_argument("--t_max", type=int, help="Time steps elapsed", default=80000)
    parser.add_argument("--movavg_span", type=int, help="Moving average span", default=1000)
    parser.add_argument("--undershoot", action="store_const", const=True, default=False)
    args = parser.parse_args()
    out_dir = args.out_dir
    t_max = args.t_max
    # Create output directory
    os.makedirs(f"{out_dir}_plots")
    nash_price = 1.4729273733327568
    coop_price = 1.9249689958811602
    xi = 0.1
    min_price = nash_price - xi
    max_price = coop_price + xi
    nash = 0.22292696
    coop = 0.33749046
    n_agents = 2
    c = 1
    mpl.rcParams["axes.prop_cycle"] = cycler(color=["b", "r", "g", "y"])

    def pg(x):
        return (x - nash) / (coop - nash)

    r = []
    last_prof_gains = {}
    prof_gains_start_meas_t = args.t_max - args.movavg_span
    state_action_maps = []
    ir_periods = 30
    dev_t = ir_periods // 2
    avg_dev_gain = 0
    avg_dev_diff_profit = 0
    df = pd.DataFrame(
        columns=[
            "id",
            "experiment_name",
            "seed",
            "buf_size",
            "actor_lr",
            "critic_lr",
            "profit_gain",
            "t",
            "deviation_profit_percent",
        ]
    )
    for seed in args.seeds:
        ir_arrays_compliant = []
        ir_arrays_defector = []
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
            profits_cur = np.load(f"{out_dir}/session_prices_{seed}.npy")
            if np.min(profits_cur) > 1:  # It's prices
                profits_cur = np.apply_along_axis(profit_numpy, 0, profits_cur)
            if args.plot_intermediate:
                plot_profits(n_agents, profits_cur, pg, args.movavg_span)
                plt.savefig(f"{out_dir}_plots/{seed}_profit.png")
            r.append(profits_cur)
            session_prof_gains = np.mean(pg(profits_cur[:, prof_gains_start_meas_t:]))
            print(f"Profits at convergence for seed {seed} = {session_prof_gains}")
            last_prof_gains[seed] = session_prof_gains

            # Create and plot state-action map
            w = torch.linspace(min_price, max_price, args.grid_size, requires_grad=False)
            A = [np.zeros([args.grid_size, args.grid_size]) for i in range(n_agents)]
            print("Computing state-action map...")
            for i in range(n_agents):
                for a_i, p1 in enumerate(w):
                    for a_j, p2 in enumerate(w):
                        state = torch.tensor([[p1, p2]])
                        action, _ = actor[i](state, deterministic=True, with_logprob=False)
                        a = scale_price(action, min_price, max_price).detach()
                        # print(f"{state} -> {a}")
                        A[i][a_i, a_j] = a.numpy()
            state_action_maps.append(A)
            if args.plot_intermediate:
                plot_heatmap(A, f"Actions for seed {seed}", min_price, max_price, w=args.grid_size)
                plt.savefig(f"{out_dir}_plots/{seed}_state_action_map.png")
            # Plot impulse responses
            for j in range(n_agents):
                # Impulse response
                price_history = np.zeros([n_agents, ir_profit_periods])
                state = torch.zeros(n_agents)
                for i in range(0, n_agents):
                    state[i] = (coop_price - nash_price) * session_prof_gains + nash_price
                price = state.clone()
                initial_state = state.clone()
                # First compute non-deviation profits
                nondev_profit = 0
                leg = ["Non-deviating agent"] * n_agents
                leg[j] = "Deviating agent"
                for t in range(ir_profit_periods):
                    for i in range(n_agents):
                        action, _ = actor[i](state.unsqueeze(0), deterministic=True, with_logprob=False)
                        price[i] = scale_price(action, min_price, max_price)
                    if t >= (ir_periods / 2):
                        nondev_profit += profit_numpy(price.numpy())[j] * args.discount ** (t - ir_periods / 2)
                    price_history[:, t] = price.detach()
                    state = price
                    conv_price = price.clone().numpy()
                    avg_rew = profit_numpy(conv_price)

                # Now compute deviation profits
                dev_profit = 0
                state = initial_state.clone()
                for t in range(ir_profit_periods):
                    for i in range(n_agents):
                        action, _ = actor[i](state.unsqueeze(0), deterministic=True, with_logprob=False)
                        price[i] = scale_price(action, min_price, max_price)
                        orig_price = price[j].item()
                    if t == (ir_periods / 2):
                        if args.defect_to_nash:
                            br = nash_price
                        elif args.defect_to_coop:
                            br = coop_price
                        elif args.defect_to_c:
                            br = c
                        else:
                            br = grad_desc(profit_numpy, price.clone().numpy(), j)
                        if args.undershoot:
                            br = orig_price + (br - orig_price) / 2
                        price[j] = torch.tensor(br)
                    if t >= (ir_periods / 2):
                        dev_profit += profit_numpy(price.numpy())[j] * args.discount ** (t - ir_periods / 2)
                    price_history[:, t] = price
                    state = price
                dev_gain = (dev_profit / nondev_profit - 1) * 100
                avg_dev_gain += dev_gain
                dev_diff_profit = (np.apply_along_axis(profit_numpy, 0, price_history).T - avg_rew).T.sum(axis=1)[j]
                avg_dev_diff_profit += dev_diff_profit
                print(f"Deviation differential profits = {dev_diff_profit} (non-dev is 0)")
                print(
                    f"Non-deviation discounted profits = {nondev_profit:.3f}; Deviation profits = {dev_profit:.3f}; Deviation gain = {dev_gain:.3f}%"
                )
                if args.plot_intermediate:
                    for i in range(n_agents):
                        plt.scatter(list(range(ir_periods)), price_history[i, :ir_periods])
                    plt.legend(leg)
                    for i in range(n_agents):
                        plt.plot(price_history[i, :ir_periods])
                    plt.ylim(min_price, max_price)
                    plt.savefig(f"{out_dir}_plots/{seed}_ir.png")
                ir_arrays_defector.append(price_history[j, :ir_periods])
                for i in range(n_agents):
                    if i != j:
                        ir_arrays_compliant.append(price_history[i, :ir_periods])
        df.loc[len(df)] = {
            "id": len(df),
            "experiment_name": out_dir,
            "seed": int(seed),
            "profit_gain": session_prof_gains,
            "t": t_max,
            "deviation_profit_percent": dev_gain / n_agents,
        }
    # Plot average profits
    profits = np.stack(r, axis=0).mean(axis=0)
    plot_profits(n_agents, profits, pg, args.movavg_span)
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
    plt.savefig(f"{out_dir}_plots/avg_profit.png")
    print(f"Average profit gain at convergence: {np.mean(list(last_prof_gains.values()))}")

    # Plot average state-action heatmap
    average_state_action_map = np.stack(state_action_maps, axis=0).mean(axis=0)
    plot_heatmap(
        average_state_action_map,
        f"Mean action across {len(args.seeds)} seeds",
        min_price,
        max_price,
        w=args.grid_size,
    )
    plt.savefig("f{out_dir}_plots/avg_state_action_heatmap.png")

    # Plot average IR
    avg_dev_gain /= n_agents * len(args.seeds)
    avg_dev_diff_profit /= n_agents * len(args.seeds)
    print(f"Average deviation gain: {avg_dev_gain:.3f}%")
    print(f"Average deviation differential profit: {avg_dev_diff_profit:.3f} (non-deviation is 0)")
    ir_stack_compliant = np.stack(ir_arrays_compliant, axis=0)
    ir_stack_defector = np.stack(ir_arrays_defector, axis=0)
    ir_box_compliant = [
        ir_stack_compliant[:, t] - ir_stack_compliant[:, dev_t - 1] for t in range(dev_t - 1, ir_periods)
    ]
    ir_box_defector = [ir_stack_defector[:, t] - ir_stack_defector[:, dev_t - 1] for t in range(dev_t - 1, ir_periods)]
    plt.boxplot(ir_box_compliant)
    plt.savefig(f"{out_dir}_plots/avg_ir_box_compliant.png")
    plt.boxplot(ir_box_defector)
    plt.savefig(f"{out_dir}_plots/avg_ir_box_defector.png")
    ir_mean_compliant = ir_stack_compliant.mean(axis=0)
    ir_mean_defector = ir_stack_defector.mean(axis=0)

    ir_mean_compliant = ir_mean_compliant[(dev_t - 1) :]
    ir_mean_defector = ir_mean_defector[(dev_t - 1) :]

    leg = ["Non-deviating agent", "Deviating agent"]
    plt.plot(ir_mean_compliant, "bo-")
    plt.plot(ir_mean_defector, "ro-")
    plt.legend(leg)
    plt.ylim(min_price, max_price)
    plt.axhline(nash_price)
    plt.axhline(coop_price)
    plt.savefig(f"{out_dir}_plots/avg_ir.png")

    df.to_csv(args.filename)


if __name__ == "__main__":
    main()
