# Algorithmic Pricing using Soft Actor-Critic

Code for my master's thesis work on algorithmic pricing in continuous environments.

To run an experiment:

```
usage: main.py [-h] [--out_dir OUT_DIR] [--device DEVICE]

Run an experiment

optional arguments:
  -h, --help         show this help message and exit
  --out_dir OUT_DIR  Directory
  --device DEVICE    CUDA device
```

To plot profits, state-action heatmaps and impulse responses:

```
usage: plot.py [-h] [--actor_hidden_size ACTOR_HIDDEN_SIZE] [--defect_to_c] [--defect_to_coop] [--defect_to_nash] [--discount DISCOUNT] [--filename FILENAME] [--grid_size GRID_SIZE] [--out_dir OUT_DIR]
               [--plot_intermediate] [--seeds SEEDS [SEEDS ...]] [--t_max T_MAX] [--movavg_span MOVAVG_SPAN] [--undershoot]

Plot and write data

optional arguments:
  -h, --help            show this help message and exit
  --actor_hidden_size ACTOR_HIDDEN_SIZE
  --defect_to_c
  --defect_to_coop
  --defect_to_nash
  --discount DISCOUNT
  --filename FILENAME
  --grid_size GRID_SIZE
  --out_dir OUT_DIR     Directory
  --plot_intermediate
  --seeds SEEDS [SEEDS ...]
                        Random seeds
  --t_max T_MAX         Time steps elapsed
  --movavg_span MOVAVG_SPAN
                        Moving average span
  --undershoot
```
