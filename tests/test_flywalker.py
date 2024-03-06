"""Test python fruitfly walker FruitFly."""

import os
import numpy as np
from dm_control import mjcf

from flybody.fruitfly.fruitfly import FruitFly


obs_names = ['thorax_height', 'abdomen_height', 'world_zaxis_hover',
             'world_zaxis', 'world_zaxis_abdomen', 'world_zaxis_head',
             'force', 'touch', 'accelerometer', 'gyro', 'velocimeter',
             'actuator_activation', 'appendages_pos']

# For local testing only.
if os.environ['MUJOCO_GL'] == 'egl':
    obs_names = obs_names + ['right_eye', 'left_eye']


def test_can_compile_and_step_simulation():
    fly = FruitFly()
    physics = mjcf.Physics.from_mjcf_model(fly.mjcf_model)
    for _ in range(100):
        physics.step()

def test_evaluate_observables():
    fly = FruitFly()
    physics = mjcf.Physics.from_mjcf_model(fly.mjcf_model)
    for name in obs_names:
        observable = getattr(fly.observables, name)
        observation = observable(physics)
        assert isinstance(observation, (float, np.ndarray))

