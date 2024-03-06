"""Test fly body parameters and pure physics (outside of RL environment)."""

import os

import numpy as np
from dm_control import mjcf


xml_path = os.path.join(os.path.dirname(__file__),
                        '../flybody/fruitfly/assets/fruitfly.xml')

def test_flymodel():

    mjcf_model = mjcf.from_path(xml_path)
    physics = mjcf.Physics.from_mjcf_model(mjcf_model)

    assert physics.model.nq == 109  # Generalized coordinates.
    assert physics.model.nv == 108  # DoFs.
    assert physics.model.nu == 78  # Actuators.
    assert physics.model.nbody == 68  # Bodies.
    assert physics.model.njnt == 103  # Joints.
    assert physics.model.ngeom == 159  # Geoms.
    assert physics.model.nsensordata == 33  # Sensor readings.
    assert physics.model.nsite == 15  # Sites.
    assert physics.model.nmesh == 85  # Meshes.
    assert np.isclose(
        physics.named.model.body_subtreemass['thorax'],
        0.0009846214672177625)  # Total fly mass.

def test_can_compile_and_step_simulation():

    mjcf_model = mjcf.from_path(xml_path)
    physics = mjcf.Physics.from_mjcf_model(mjcf_model)

    physics.reset()
    for _ in range(100):
        physics.data.ctrl[:] = np.random.uniform(-.2, .2, physics.model.nu)
        physics.step()

    _ = physics.render()

