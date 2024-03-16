"""Test stand-alone fly model outside of RL environment."""

import os
import numpy as np

from dm_control import mjcf
import flybody

flybody_path = os.path.dirname(flybody.__file__)
xml_path = os.path.join(flybody_path, 'fruitfly/assets/fruitfly.xml')

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

expect_close_masses = {
    'fly_mass': 0.0009846214672177625,  # Total fly mass.
    'head': 0.0001499089219064366,
    'thorax': 0.00034,
    'abdomen': 0.0003802050947221454,
    'leg_T1': 1.618451244357944e-05,
    'leg_T2': 1.3834187453723818e-05,
    'leg_T3': 1.841834251998194e-05,
    'wing': 8e-6,
}


def test_fly_parameters():
    physics = mjcf.Physics.from_xml_path(xml_path)
    for k, v in expect.items():
        assert getattr(physics.model, k) == v
    

def test_fly_masses():
    physics = mjcf.Physics.from_xml_path(xml_path)
    
    assert np.isclose(
        physics.named.model.body_subtreemass['thorax'],
        expect_close_masses['fly_mass'])
    assert np.isclose(
        physics.named.model.body_subtreemass['head'],
        expect_close_masses['head'])
    assert np.isclose(
        physics.named.model.body_mass['thorax'],
        expect_close_masses['thorax'])
    assert np.isclose(
        physics.named.model.body_subtreemass['abdomen'],
        expect_close_masses['abdomen'])
    
    for side in ['left', 'right']:
        assert np.isclose(
            physics.named.model.body_subtreemass[f'coxa_T1_{side}'],
            expect_close_masses['leg_T1'])
        assert np.isclose(
            physics.named.model.body_subtreemass[f'coxa_T2_{side}'],
            expect_close_masses['leg_T2'])
        assert np.isclose(
            physics.named.model.body_subtreemass[f'coxa_T3_{side}'],
            expect_close_masses['leg_T3'])
        assert np.isclose(
            physics.named.model.body_mass[f'wing_{side}'],
            expect_close_masses['wing'])


def test_control_ranges_match_joint_ranges():
    physics = mjcf.Physics.from_xml_path(xml_path)
    m = physics.model
    for i in range(physics.model.nu):
        # Consider only joint actuators that are also position actuators.
        if m.actuator_trntype[i] == 0 and m.actuator_biastype[i] == 1:
            actuator_name = m.id2name(i, 'actuator')
            joint_name = m.id2name(m.actuator_trnid[i][0], 'joint')
            joint_id = m.actuator_trnid[i][0]
            ctrl_range = m.actuator_ctrlrange[i]
            joint_range = m.jnt_range[joint_id]
            assert actuator_name == joint_name
            assert all(ctrl_range == joint_range)


def test_can_compile_and_step_simulation():
    physics = mjcf.Physics.from_xml_path(xml_path)
    physics.reset()
    for _ in range(100):
        physics.data.ctrl[:] = np.random.uniform(-.2, .2, physics.model.nu)
        physics.step()
        assert isinstance(physics.data.sensordata, np.ndarray)
    # For local testing only.
    if 'MUJOCO_GL' in os.environ and os.environ['MUJOCO_GL'] == 'egl':
        _ = physics.render()
