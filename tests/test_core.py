"""Test core installation by creating an RL environment and stepping it."""

import os
import numpy as np
from dm_control import mujoco
from flybody.fly_envs import template_task


obs_names = ['accelerometer',
             'actuator_activation',
             'appendages_pos',
             'force',
             'gyro',
             'joints_pos',
             'joints_vel',
             'touch',
             'velocimeter',
             'world_zaxis']

obs_names = ['walker/' + s for s in obs_names]


def test_can_create_and_run_environment():

    env = template_task()
    
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
        assert timestep.reward == 1

    # For local testing only.
    if 'MUJOCO_GL' in os.environ and os.environ['MUJOCO_GL'] == 'egl':
        _ = env.physics.render()


def test_action_corruptor():

    # Test clean prev_action.
    env = template_task()
    n_act = env.action_spec().shape
    _ = env.reset()
    for _ in range(100):
        action = np.random.uniform(-1., 1, n_act)
        _ = env.step(action)
        assert all(np.isclose(action, env.task._walker.prev_action))
    
    # Test action corruptor.
    noise = np.random.normal(scale=0.1, size=n_act)
    def corruptor(action, random_state):
        del random_state  # Unused.
        return action + noise
    env = template_task(action_corruptor=corruptor)
    _ = env.reset()
    for _ in range(100):
        clean_action = np.random.uniform(-1., 1, n_act)
        _ = env.step(clean_action)
        assert all(np.isclose(clean_action+noise, env.task._walker.prev_action))


def test_ctrl_callback():

    dof_ids = [*range(6, 9), *range(42, 53), *range(75, 90)]
    dof_ids_complementary = [i for i in range(108) if i not in dof_ids]

    class FakeCallback():
        def __call__(self, model, data):
            # Add noise to a subset of dofs.
            qfrc_actuator = data.qfrc_actuator[dof_ids]
            noise = np.sin(np.arange(len(dof_ids)))
            data.qfrc_applied[dof_ids] = qfrc_actuator * noise
        def reset(self):
            pass

    ctrl_callback = FakeCallback()
    
    env = template_task(mjcb_control=ctrl_callback)
    n_act = env.action_spec().shape
    _ = env.reset()
    for _ in range(100):
        action = np.random.uniform(-1., 1, n_act)
        _ = env.step(action)
        data = env.physics.data
        # Test noise added to select dofs.
        assert all(np.isclose(
            data.qfrc_applied[dof_ids],
            data.qfrc_actuator[dof_ids] * np.sin(np.arange(len(dof_ids)))))
        # Test noise not added to rert of dofs.
        assert all(data.qfrc_applied[dof_ids_complementary] == 0)

    # Reset callback, otherwise subsequent tests will fail.
    mujoco.set_mjcb_control(None)
