# Algorithmic Pricing using Soft Actor-Critic

Code for my work on algorithmic pricing in continuous environments.

## License

This code is released under the terms of the GNU Affero General Public License.
If you use any of this in your own research, or take inspiration from it to develop a derivative work, please cite my working paper:

```
Frick, Kevin Michael, Convergence Rates and Collusive Outcomes of Pricing Algorithms (February 12, 2023).
Available at SSRN: https://ssrn.com/abstract=4527452 or http://dx.doi.org/10.2139/ssrn.4527452
```

## Main code

To run an experiment:

```
usage: main.py [-h] [--out_dir OUT_DIR] [--device DEVICE] [--ai_last AI_LAST] [--demand_std DEMAND_STD]

Run an experiment

optional arguments:
  -h, --help               show this help message and exit
  --out_dir OUT_DIR        Directory
  --device DEVICE          CUDA device
  --ai_last AI_LAST        Last agent's demand parameter
  --demand_std DEMAND_STD  Standard deviation of a0 (for stochastic demand). Will be ignored if 0 or negative.
```

## Plotting

To plot profits, state-action heatmaps and impulse responses:

```
usage: plot.py [-h] [--actor_hidden_size ACTOR_HIDDEN_SIZE] [--discount DISCOUNT] [--filename FILENAME] [--grid_size GRID_SIZE] [--out_dir OUT_DIR] [--plot_intermediate] [--seeds SEEDS [SEEDS ...]]
               [--movavg_span MOVAVG_SPAN] [--parse_csv] [--n_agents N_AGENTS]

Plot and write data

optional arguments:
  -h, --help            show this help message and exit
  --actor_hidden_size ACTOR_HIDDEN_SIZE
  --discount DISCOUNT
  --filename FILENAME
  --grid_size GRID_SIZE
  --out_dir OUT_DIR     Directory
  --plot_intermediate
  --seeds SEEDS [SEEDS ...]
                        Random seeds
  --movavg_span MOVAVG_SPAN
                        Moving average span
  --parse_csv
  --n_agents N_AGENTS   Number of agents
```
