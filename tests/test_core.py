"""Test core installation by creating an RL environment and stepping it."""

import os
import numpy as np
from flybody.fly_envs import walk_on_ball


obs_names = ['accelerometer',
             'actuator_activation',
             'appendages_pos',
             'force',
             'gyro',
             'joints_pos',
             'joints_vel',
             'touch',
             'velocimeter',
             'world_zaxis',
             'ball_qvel']

obs_names = ['walker/' + s for s in obs_names]


def test_can_create_and_run_environment():

    env = walk_on_ball()
    
    obs_spec = env.observation_spec()
    assert list(obs_spec) == obs_names
    
    n_act = env.action_spec().shape
    assert n_act == (59,)

    timestep = env.reset()
    for name in obs_names:
        observation = timestep.observation[name]
        assert isinstance(observation, (float, np.ndarray))

    for _ in range(100):
        action = np.random.uniform(-1, 1, n_act)
        timestep = env.step(action)
        assert timestep.reward is not None

    # For local testing only.
    if 'MUJOCO_GL' in os.environ and os.environ['MUJOCO_GL'] == 'egl':
        _ = env.physics.render()
