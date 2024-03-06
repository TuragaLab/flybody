"""Test fly body parameters and pure physics (outside of RL environment)."""

import os
import numpy as np
from dm_control import mjcf


xml_path = os.path.join(os.path.dirname(__file__),
                        '../flybody/fruitfly/assets/fruitfly.xml')

expect = {
    'nq': 109,  # Generalized coordinates.
    'nv': 108,  # DoFs.
    'nu': 78,  # Actuators.
    'nbody': 68,  # Bodies.
    'njnt': 103,  # Joints.
    'ngeom': 159,  # Geoms.
    'nsensor': 15,  # Sensors.
    'nsensordata': 33,  # Sensor readings.
    'nsite': 15,  # Sites.
    'nmesh': 85,  # Meshes.
    'ntendon': 8,  # Tendons.
    'neq': 0,  # Equality constraints.
}

expect_close = {
    'fly_mass': 0.0009846214672177625,  # Total fly mass.
}


def test_model_parameters():

    mjcf_model = mjcf.from_path(xml_path)
    physics = mjcf.Physics.from_mjcf_model(mjcf_model)

    for k, v in expect.items():
        assert getattr(physics.model, k) == v

    assert np.isclose(
        physics.named.model.body_subtreemass['thorax'],
        expect_close['fly_mass'])


def test_can_compile_and_step_simulation():

    mjcf_model = mjcf.from_path(xml_path)
    physics = mjcf.Physics.from_mjcf_model(mjcf_model)

    physics.reset()
    for _ in range(100):
        physics.data.ctrl[:] = np.random.uniform(-.2, .2, physics.model.nu)
        physics.step()
        assert isinstance(physics.data.sensordata, np.ndarray)

    # For local testing only.
    if os.environ['MUJOCO_GL'] == 'egl':
        _ = physics.render()

