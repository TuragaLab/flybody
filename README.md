# `flybody`
MuJoCo fruit fly body model and reinforcement learning tasks

## Installation

### The `flybody` package can be installed in three modes:

1. Core installation: light-weight installation for experimenting with the
   fly model in MuJoCo or with dm_control task environments. ML components
   such as Tensorflow and Acme are not installed and policy rollouts and
   training are not supported.

2. Tensorflow ML extension: same as (1), plus Tensorflow, Acme to allow bringing
   policy networks into play (e.g. for inference), but without training them with Ray.

3. Distributed Ray training extension: Same as (1) and (2), plus Ray to also allow
   distributed policy training in the dm_control task environments.

### Follow these steps to install the package:

1. Clone this repo and create a new conda environment `flybody`:
```bash
git clone https://github.com/TuragaLab/flybody.git
cd flybody
conda env create -f flybody.yml
conda activate flybody
```
2. Depending on the desired installation mode (see above), proceed with one of the three options:

Core installation:
```bash
pip install -e .
```

Tensorflow ML extension:
```bash
pip install -e .[tf]
```
Distributed Ray training extension:
```bash
pip install -e .[ray]
```
3. Set environment varibles, e.g.:
   ```bash
   export MUJOCO_GL=egl
   export LD_LIBRARY_PATH=/YOUR/PATH/TO/miniconda3/envs/flybody/lib
   ```