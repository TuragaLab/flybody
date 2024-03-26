"""Test walking imitation environment."""

import os
import numpy as np
from flybody.tasks.constants import (
    _WALK_CONTROL_TIMESTEP,
    _WALK_PHYSICS_TIMESTEP)
from flybody.fly_envs import walk_imitation


expect_obs_names = ['accelerometer',
                    'actuator_activation',
                    'appendages_pos',
                    'force',
                    'gyro',
                    'joints_pos',
                    'joints_vel',
                    'touch',
                    'velocimeter',
                    'world_zaxis',
                    'ref_displacement',
                    'ref_root_quat']
expect_obs_names = ['walker/' + s for s in expect_obs_names]
expect_num_act = 59

# Create artificial trajectory: walking straight at 1 cm/s.
n_steps = 200
ctrl_timestep = 0.002
qpos = np.zeros((n_steps, 7))
qpos[:, 0] = np.arange(0, n_steps*ctrl_timestep, ctrl_timestep)
qpos[:, [2, 3]] = [0.14355, 1.]
qvel = np.zeros((n_steps, 6))
qvel[:, 0] = 1.  # Speed: 1 cm/s.
snippet = {'qpos': qpos, 'qvel': qvel}


def test_can_create_env_inference_mode():

    # ref_path not provided, task will run in inference mode.
    env = walk_imitation(terminal_com_dist=float('inf'))
    
    obs_spec = env.observation_spec()
    assert list(obs_spec) == expect_obs_names
    
    n_act = env.action_spec().shape
    assert n_act == (expect_num_act,)

    # Load trajectory to task.
    env.task._traj_generator.set_next_trajectory(
        snippet['qpos'], snippet['qvel'])
    timestep = env.reset()
    for name in expect_obs_names:
        observation = timestep.observation[name]
        assert isinstance(observation, (float, np.ndarray))

    assert np.isclose(env.control_timestep(), _WALK_CONTROL_TIMESTEP)
    assert np.isclose(env.physics.timestep(), _WALK_PHYSICS_TIMESTEP)


def test_can_step_env_inference_mode():

    # ref_path not provided, task will run in inference mode.
    env = walk_imitation(terminal_com_dist=float('inf'))
    # Load trajectory to task.
    env.task._traj_generator.set_next_trajectory(
        snippet['qpos'], snippet['qvel'])
    timestep = env.reset()

    for _ in range(100):
        action = np.random.uniform(-0.5, 0.5, expect_num_act)
        timestep = env.step(action)
        assert timestep.reward == 1.  # At test-time, reward is 1.

    # For local testing only.
    if 'MUJOCO_GL' in os.environ and os.environ['MUJOCO_GL'] == 'egl':
        _ = env.physics.render()
