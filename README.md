# **MAPOLight**
### This is the official implement of EAAI [*"Cooperative traffic signal control for a partially observed vehicular network using multi-agent reinforcement learning"*](https://doi.org/10.1016/j.engappai.2025.111813).

## Setup Environment

*It is recommended to use a Linux system to avoid the path resolution bug present in Ray 2.6.3 on Windows (Ray may treat a local path as a remote path for read/write operations, which can cause errors).*

The following steps describe how to install and set up a virtual environment for the repo using **uv**:

```bash
cd MAPOLight
uv venv --python=3.10.12
uv pip install ray[all]==2.6.3
uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt
```

In addition, SUMO must be installed. This repo uses SUMO 1.19.0, which can be installed with the following command:

```bash
uv pip install eclipse-sumo
```

After that, the environment variable `SUMO_HOME` must be configured. If SUMO 1.19.0 is installed using the method above, `SUMO_HOME` should be set to the following path:

```bash
export SUMO_HOME='./.venv/lib/python3.10/site-packages/sumo'
```

## Training

```bash
uv run train.py
```

Optional command-line arguments are available to perform different training configurations. For example:

```bash
uv run train.py --arch A3C --net 5x5  # use A3C train model in 5x5 road network
```

Training will continue indefinitely until the program is manually stopped by pressing Ctrl+C. The two best-performing checkpoints will be saved under:

```
~/ray_results/<method>/<method>_net_xxxx/checkpoint_xxxx
```

### Viewing Training Progress with TensorBoard

```bash
uv run tensorboard --logdir ~/ray_results/PPO/PPO_net_34ccb_00000_0_2026-01-23_18-06-38/
```

Replace the `logdir` path above with your corresponding local path.

It will open a tensorboard server listening on 6006 port by default. You can view it by a browser.

## Evaluation

```bash
uv run test2.py -a ppo -c ~/ray_results/PPO/PPO_net_34ccb_00000_0_2026-01-23_18-06-38/checkpoint_000036 -t 3600
```

Replace the `-c` path above with your corresponding local path.

Use the following command to view all available command-line options for test2.py:

```bash
uv run test2.py -h
```

The above command will generate an outputs directory in the current folder, which contains several CSV files recording simulation data.
