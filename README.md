# `flybody` <img src="fly-white.png" width="20%">

![example workflow](https://github.com/TuragaLab/flybody/actions/workflows/pytest.yml/badge.svg)

MuJoCo fruit fly body model and reinforcement learning tasks

## Getting started
Coming soon

## Installation

Follow these steps to install `flybody`:

1. Clone this repo and create a new conda environment:
   ```bash
   git clone https://github.com/TuragaLab/flybody.git
   cd flybody
   conda env create -f flybody.yml
   conda activate flybody
   ```
   `flybody` can be installed in one of the three modes described next. For installation in editable (developer) mode, use the commands as shown. For installation in regular, not editable, mode, drop the `-e` flag.
   
3. **Core installation**: minimal installation for experimenting with the
   fly model in MuJoCo or prototyping task environments. ML dependencies such as [Tensorflow](https://github.com/tensorflow/tensorflow) and [Acme](https://github.com/google-deepmind/acme) are not included and policy rollouts and training are not automatically supported.
   ```bash
   pip install -e .
   ```
   
4. **ML extension (optional)**: same as core installation, plus ML dependencies (Tensorflow, Acme) to allow running
   policy networks, e.g. for inference or for training using third-party agents not included in this library.
   ```bash
   pip install -e .[tf]
   ```

5. **Ray training extension (optional)**: same as core installation and ML extension, plus [Ray](https://github.com/ray-project/ray) to also enable
   distributed policy training in the fly task environments.
   ```bash
   pip install -e .[ray]
   ```

6. You may need to set [MuJoCo rendering](https://github.com/google-deepmind/dm_control/tree/main?tab=readme-ov-file#rendering) environment varibles, e.g.:
   ```bash
   export MUJOCO_GL=egl
   export MUJOCO_EGL_DEVICE_ID=0
   ```
   Also, for the ML and Ray extensions, `LD_LIBRARY_PATH` may require an update, e.g.:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/YOUR/PATH/TO/miniconda3/envs/flybody/lib
   ```

7. You may want to run `pytest` to test the main aspects of the `flybody` installation.

## Citing flybody
Coming soon
