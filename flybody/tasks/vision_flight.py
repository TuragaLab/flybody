"""Vision-guided flight task."""
# ruff: noqa: F821

from typing import Optional
import numpy as np

from dm_control.utils import rewards
from dm_control.composer.observation import observable
from dm_control import composer
from flybody.tasks.arenas.hills import SineTrench

from flybody.tasks.pattern_generators import (WingBeatPatternGenerator)
from flybody.tasks.task_utils import neg_quat
from flybody.tasks.base import Flying


class VisionFlightImitationWBPG(Flying):
    """Vision-based flight with controllable Wing Beat Pattern Generator."""

    def __init__(self,
                 wbpg: WingBeatPatternGenerator,
                 floor_contacts_fatal: bool = True,
                 eye_camera_fovy: float = 150.,
                 eye_camera_size: int = 32,
                 target_height_range: tuple = (0.5, 0.8),
                 target_speed_range: tuple = (20, 40),
                 init_pos_x_range: Optional[tuple] = (-5, -5),
                 init_pos_y_range: Optional[tuple] = (0, 0),
                 **kwargs):
        """Task of learning a policy for flying and maneuvering while using a
            wing beat pattern generator with controllable wing beat frequency.

        Args:
            wpg: Wing beat generator.
            floor_contacts_fatal: Whether to terminate the episode when the fly
                contacts the floor.
            eye_camera_fovy: Field of view of the eye camera.
            eye_camera_size: Size of the eye camera.
            target_height_range: Range of target height.
            target_speed_range: Range of target speed.
            init_pos_x_range: Range of initial x position.
            init_pos_y_range: Range of initial y position.
            **kwargs: Arguments passed to the superclass constructor.
        """

        super().__init__(add_ghost=False,
                         num_user_actions=1,
                         eye_camera_fovy=eye_camera_fovy,
                         eye_camera_size=eye_camera_size,
                         **kwargs)
        self._wbpg = wbpg
        self._floor_contacts_fatal = floor_contacts_fatal
        self._eye_camera_size = eye_camera_size
        self._target_height_range = target_height_range
        self._target_speed_range = target_speed_range
        self._init_pos_x_range = init_pos_x_range
        self._init_pos_y_range = init_pos_y_range

        # Remove all light.
        for light in self._walker.mjcf_model.find_all('light'):
            light.remove()

        # Get wing joint indices into agent's action vector.
        self._wing_inds_action = self._walker._action_indices['wings']
        # Get 'user' index into agent's action vector (only one user action).
        self._user_idx_action = self._walker._action_indices['user'][0]

        # Dummy initialization.
        self._target_height = 0.
        self._target_speed = 0.

        self._target_zaxis = None
        self._ncol = None
        self._grid_axis = None

        # === Explicitly add/enable/disable vision task observables.
        # Fly observables.
        self._walker.observables.right_eye.enabled = True
        self._walker.observables.left_eye.enabled = True
        self._walker.observables.thorax_height.enabled = False
        # Task observables.
        self._walker.observables.add_observable('task_input', self.task_input)

    def get_hfield_height(self, x, y, physics):
        """Return hfield height at a hfield grid point closest to (x, y)."""

        hfield_half_size = physics.model.hfield_size[0, 0]
        self._ncol = physics.model.hfield_ncol[0]
        self._grid_axis = np.linspace(-hfield_half_size, hfield_half_size,
                                      self._ncol)

        # Get nearest indices.
        x_idx = np.argmin(np.abs(self._grid_axis - x))
        y_idx = np.argmin(np.abs(self._grid_axis - y))
        # physics.model.hfield_data is in range [0, 1], needs to be rescaled.
        elevation_z = physics.model.hfield_size[0, 2]  # z_top scaling.
        return elevation_z * physics.model.hfield_data[y_idx * self._ncol +
                                                       x_idx]

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        super().initialize_episode_mjcf(random_state)

        self._target_height = random_state.uniform(*self._target_height_range)
        self._target_speed = random_state.uniform(*self._target_speed_range)

        theta = np.deg2rad(self._body_pitch_angle)
        self._target_zaxis = np.array([np.sin(theta), 0, np.cos(theta)])

    def initialize_episode(self, physics: 'mjcf.Physics',
                           random_state: np.random.RandomState):
        """Randomly selects a starting point and set the walker.

        Environment call sequence:
            check_termination, get_reward_factors, get_discount
        """
        super().initialize_episode(physics, random_state)

        init_x = random_state.uniform(*self._init_pos_x_range)
        init_y = random_state.uniform(*self._init_pos_y_range)

        # Reset wing pattern generator and get initial wing angles.
        initial_phase = random_state.uniform()
        init_wing_qpos = self._wbpg.reset(initial_phase=initial_phase)

        self._arena.initialize_episode(physics, random_state)

        # Initialize root position and orientation.
        hfield_height = self.get_hfield_height(init_x, init_y, physics)
        init_z = hfield_height + self._target_height
        self._walker.set_pose(physics, np.array([init_x, init_y, init_z]),
                              neg_quat(self._up_dir))

        # Initialize wing qpos.
        physics.bind(self._wing_joints).qpos = init_wing_qpos

        # If enabled, initialize leg joint angles in retracted position.
        if self._leg_joints:
            physics.bind(self._leg_joints).qpos = self._leg_springrefs

        if self._initialize_qvel:
            # Only initialize linear CoM velocity, not rotational velocity.
            init_vel, _ = self._walker.get_velocity(physics)
            self._walker.set_velocity(
                physics, [self._target_speed, init_vel[1], init_vel[2]])

    def before_step(self, physics: 'mjcf.Physics', action,
                    random_state: np.random.RandomState):
        # Get target wing joint angles at beat frequency requested by the agent.
        base_freq, rel_range = self._wbpg.base_beat_freq, self._wbpg.rel_freq_range
        act = action[self._user_idx_action]  # Action in [-1, 1].
        ctrl_freq = base_freq * (1 + rel_range * act)
        ctrl = self._wbpg.step(
            ctrl_freq=ctrl_freq)  # Returns position control.

        length = physics.bind(self._wing_joints).qpos
        # Convert position control to force control.
        action[self._wing_inds_action] += (ctrl - length)

        super().before_step(physics, action, random_state)

    def get_reward_factors(self, physics):
        """Returns the factorized reward terms."""

        # Height.
        xpos, _ = self._walker.get_pose(physics)
        current_height = (xpos[2] - self.get_hfield_height(*xpos[:2], physics))
        height = rewards.tolerance(current_height,
                                   bounds=(self._target_height,
                                           self._target_height),
                                   sigmoid='linear',
                                   margin=0.15,
                                   value_at_margin=0)

        velocity, _ = self._walker.get_velocity(physics)

        # Center-of-trench reward factor.
        center_of_trench = 1.
        if isinstance(self._arena, SineTrench):
            trench_specs = self._arena.trench_specs
            # If we are within the trench bounds.
            if trench_specs['x_coords'][0] <= xpos[0] <= trench_specs[
                    'x_coords'][-1]:
                idx = (np.abs(trench_specs['x_coords'] - xpos[0])).argmin()
                trench_center = trench_specs['y_coords'][idx]
                center_of_trench = rewards.tolerance(
                    xpos[1],
                    bounds=(trench_center, trench_center),
                    sigmoid='linear',
                    margin=0.15,
                    value_at_margin=0.0)

        # Preferred absolute flight direction.
        x_speed = rewards.tolerance(velocity[0],
                                    bounds=(self._target_speed, float('inf')),
                                    sigmoid='linear',
                                    margin=1.1 * self._target_speed,
                                    value_at_margin=0.0)

        # Maintain certain speed.
        speed = rewards.tolerance(np.linalg.norm(velocity),
                                  bounds=(self._target_speed,
                                          self._target_speed),
                                  sigmoid='linear',
                                  margin=1.1 * self._target_speed,
                                  value_at_margin=0.0)

        # Keep zero egocentric side speed.
        vel = self.observables['walker/velocimeter'](physics)
        side_speed = rewards.tolerance(vel[1],
                                       bounds=(0, 0),
                                       sigmoid='linear',
                                       margin=10,
                                       value_at_margin=0.0)

        # World z-axis, to replace root quaternion reward above.
        current_zaxis = self.observables['walker/world_zaxis'](physics)
        angle = np.arccos(np.dot(self._target_zaxis, current_zaxis))
        world_zaxis = rewards.tolerance(angle,
                                        bounds=(0, 0),
                                        sigmoid='linear',
                                        margin=np.pi,
                                        value_at_margin=0.0)

        # Reward for leg retraction during flight.
        qpos_diff = physics.bind(self._leg_joints).qpos - self._leg_springrefs
        retract_legs = rewards.tolerance(qpos_diff,
                                         bounds=(0, 0),
                                         sigmoid='linear',
                                         margin=4.,
                                         value_at_margin=0.0)

        return np.hstack((height, x_speed, speed, side_speed, world_zaxis,
                          center_of_trench, retract_legs))

    def check_floor_contact(self, physics):
        """Check if fly collides with floor geom."""
        world_id = 0
        for contact in physics.data.contact:
            # If the contact is not active, continue.
            if contact.efc_address < 0:
                continue
            # Check floor contact.
            body1 = physics.model.geom_bodyid[contact.geom1]
            body2 = physics.model.geom_bodyid[contact.geom2]
            if body1 == world_id or body2 == world_id:
                return True
        return False

    def check_termination(self, physics: 'mjcf.Physics') -> bool:
        if self._floor_contacts_fatal:
            return (self.check_floor_contact(physics)
                    or super().check_termination(physics))
        else:
            return super().check_termination(physics)

    @property
    def target_height(self):
        return self._target_height

    @property
    def target_speed(self):
        return self._target_speed

    @composer.observable
    def task_input(self):
        """Task-specific input, framed as an observable."""
        def get_task_input(physics: 'mjcf.Physics'):
            del physics
            return np.hstack([self._target_height, self._target_speed])
        return observable.Generic(get_task_input)
