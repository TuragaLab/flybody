"""Test python fruitfly wrapper class FruitFly."""

import os
import numpy as np

from dm_control import mjcf
from dm_control.composer.observation.observable import base as observable_base

from flybody.fruitfly.fruitfly import FruitFly
from .common import is_force_actuator


TEST_ACTION = 0.3561
JOINT_FILTER = 0.0123
ADHESION_FILTER = 0.0234
OBS_NAMES = ['thorax_height', 'abdomen_height', 'world_zaxis_hover',
             'world_zaxis', 'world_zaxis_abdomen', 'world_zaxis_head',
             'force', 'touch', 'accelerometer', 'gyro', 'velocimeter',
             'actuator_activation', 'appendages_pos']
# For local testing only.
if 'MUJOCO_GL' in os.environ and os.environ['MUJOCO_GL'] == 'egl':
    OBS_NAMES = OBS_NAMES + ['right_eye', 'left_eye']

# Prepare all possible configurations to test fly walker with.
uses = [(i, j, k, l) for i in range(2) for j in range(2)
                     for k in range(2) for l in range(2)]  # noqa: E741
filters = [(0, 0), (JOINT_FILTER, 0), (0, ADHESION_FILTER), 
           (JOINT_FILTER, ADHESION_FILTER)]
user_actions = [0, 1, 2]
configs = [{'use': use, 'filter': filter, 'user_action': user_action}
           for use in uses
           for filter in filters
           for user_action in user_actions]


def test_fly_bulletproof():
    """Test fly walker in all possible configurations."""

    for config in configs:
        use = config['use']
        filter = config['filter']
        user_action = config['user_action']

        fly = FruitFly(use_legs=use[0],
                       use_wings=use[1],
                       use_mouth=use[2],
                       use_antennae=use[3],
                       joint_filter=filter[0],
                       adhesion_filter=filter[1],
                       num_user_actions=user_action)

        # Test can compile and step simulation.
        physics = mjcf.Physics.from_mjcf_model(fly.mjcf_model)
        n_actions = fly.action_spec.shape[0]
        for i in range(100):
            # Emulate control_timestep.
            if i % 10 == 0:
                physics.data.ctrl[:] = np.random.uniform(-.2, .2, n_actions)
            physics.step()

        # Test action_spec consistency.
        spec = fly.action_spec
        assert (spec.shape[0] == len(spec.name.split()) 
                == len(spec.minimum) == len(spec.maximum))

        # Test that all action values are passed correctly to their
        # corresponding ctrl elements in MuJoCo.
        n_actions = fly.action_spec.shape[0] + user_action
        physics.reset()
        for key, action_indices in fly._action_indices.items():
            if key == 'user':
                continue
            for i, action_idx in enumerate(action_indices):
                # Set all ctrl to zero.
                action = np.zeros(n_actions)
                fly.apply_action(physics, action, None)
                physics.step()
                action[action_idx] = TEST_ACTION
                # Send environment action to mujoco ctrl.
                fly.apply_action(physics, action, None)
                ctrl_idx = fly._ctrl_indices[key][i]
                assert physics.data.ctrl[ctrl_idx] == TEST_ACTION

        # Test that actuator_dynprm and actuator_dyntype are set
        # correctly in adhesion and joint actuators.
        m = physics.model
        for i in range(m.nu):
            # Regular joint actuators.
            if m.actuator_trntype[i] == 0:        
                # print(m.actuator_dynprm[i])
                if filter[0] == 0:
                    # Case of joint_filter == 0.
                    assert m.actuator_dynprm[i, 0] == 1
                    assert m.actuator_dyntype[i] == 0
                else:
                    # Case of joint_filter > 0.
                    assert m.actuator_dynprm[i, 0] == JOINT_FILTER
                    assert m.actuator_dyntype[i] == 2
            # Adhesion actuators.
            if m.actuator_trntype[i] == 5:
                if filter[1] == 0:
                    # Case of adhesion_filter == 0.
                    assert m.actuator_dynprm[i, 0] == 1
                    assert m.actuator_dyntype[i] == 0
                else:
                    # Case of adhesion_filter > 0.
                    assert m.actuator_dynprm[i, 0] == ADHESION_FILTER
                    assert m.actuator_dyntype[i] == 2

        # Test that actuator names in action_spec match their ranges.
        physics.reset()
        action_spec = fly.get_action_spec(physics)
        for i, name in enumerate(action_spec.name.split()):
            for idx in range(physics.model.nu):
                if name == physics.model.id2name(idx, 'actuator'):
                    minimum, maximum = physics.model.actuator_ctrlrange[idx]
                    assert action_spec.minimum[i] == minimum
                    assert action_spec.maximum[i] == maximum
                elif 'user' in name:
                    assert action_spec.minimum[i] == -1
                    assert action_spec.maximum[i] == 1


def test_force_actuators():
    """Test switching to force actuators."""
    fly = FruitFly(use_legs=True,
                   use_wings=True,
                   use_mouth=True,
                   use_antennae=True,
                   joint_filter=0.01,
                   adhesion_filter=0.02,
                   force_actuators=True)
    physics = mjcf.Physics.from_mjcf_model(fly.mjcf_model)
    assert is_force_actuator(physics)


def test_filterexact():
    """Test `filterexact` actuator activation dynamics."""

    for dyntype_filterexact, dyntype_expect in zip([False, True], [2, 3]):

        fly = FruitFly(use_legs=True,
                       use_wings=True,
                       use_mouth=True,
                       use_antennae=True,
                       joint_filter=0.01,
                       adhesion_filter=0.02,
                       dyntype_filterexact=dyntype_filterexact)

        # Test can compile and step simulation.
        physics = mjcf.Physics.from_mjcf_model(fly.mjcf_model)
        n_actions = fly.action_spec.shape[0]
        for i in range(100):
            # Emulate control_timestep.
            if i % 10 == 0:
                physics.data.ctrl[:] = np.random.uniform(-.2, .2, n_actions)
            physics.step()

        # Test that actuator_dyntype is set correctly in adhesion and joint
        # actuators.
        m = physics.model
        for i in range(m.nu):
            # Regular joint actuators.
            if m.actuator_trntype[i] == 0:
                assert m.actuator_dyntype[i] == dyntype_expect
            # Adhesion actuators.
            if m.actuator_trntype[i] == 5:
                assert m.actuator_dyntype[i] == dyntype_expect


def test_prev_action():
    for num_user_actions in user_actions:
        fly = FruitFly(num_user_actions=num_user_actions)
        assert all(fly.prev_action == 0)
        action_size = fly.action_spec.shape[0] + num_user_actions
        physics = mjcf.Physics.from_mjcf_model(fly.mjcf_model)
        for _ in range(10):
            action = np.random.uniform(-1., 1, action_size)
            fly.apply_action(physics, action, random_state=None)
            assert all(np.isclose(action, fly.prev_action))


def test_evaluate_observables():
    fly = FruitFly()
    physics = mjcf.Physics.from_mjcf_model(fly.mjcf_model)
    for name in OBS_NAMES:
        observable = getattr(fly.observables, name)
        observation = observable(physics)
        assert isinstance(observation, (float, np.ndarray))


def test_proprioception():
    fly = FruitFly()
    for item in fly.observables.proprioception:
        assert isinstance(item, observable_base.Observable)


def test_vestibular():
    fly = FruitFly()
    for item in fly.observables.vestibular:
        assert isinstance(item, observable_base.Observable)


def test_orientation():
    fly = FruitFly()
    for item in fly.observables.orientation:
        assert isinstance(item, observable_base.Observable)


def test_set_name():
    name = 'fruity'
    fly = FruitFly(name=name)
    assert fly.name == name
