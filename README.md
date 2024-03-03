# `flybody`
MuJoCo fruit fly body model and reinforcement learning tasks

## Installation

#### Follow these steps to install `flybody`:

1. Clone this repo and create a new conda environment:
   ```bash
   git clone https://github.com/TuragaLab/flybody.git
   cd flybody
   conda env create -f flybody.yml
   conda activate flybody
   ```

2. The `flybody` library can be installed in one of three modes. Core installation: minimal installation for experimenting with the
   fly model in MuJoCo or with dm_control task environments. ML dependencies such as Tensorflow and Acme are not included and policy rollouts and
   training are not supported.
   ```bash
   pip install -e .
   ```
   
3. ML extension (optional): same as core installation, plus ML dependencies (Tensorflow, Acme) to allow running
   policy networks, e.g. for inference or for training using custom agents not included in this library.
   ```bash
   pip install -e .[tf]
   ```

4. Ray training extension (optional): Same as core installation and ML extension, plus [Ray](https://github.com/ray-project/ray) to also enable
   distributed policy training in the dm_control task environments.
   ```bash
   pip install -e .[ray]
   ```

5. You may need to set [MuJoCo rendering](https://github.com/google-deepmind/dm_control/tree/main?tab=readme-ov-file#rendering) environment varibles, e.g.:
   ```bash
   export MUJOCO_GL=egl
   export MUJOCO_EGL_DEVICE_ID=0
   ```
   As well as, for the ML and Ray extensions, possibly:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/YOUR/PATH/TO/miniconda3/envs/flybody/lib
   ```