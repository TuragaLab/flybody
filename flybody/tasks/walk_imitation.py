"""Walking imitation task for fruit fly."""
# ruff: noqa: F821

from typing import Sequence
import numpy as np

from dm_control.utils import rewards

from flybody.tasks.base import Walking
from flybody.tasks.constants import (_TERMINAL_ANGVEL, _TERMINAL_LINVEL)
from flybody.tasks.rewards import (get_reference_features, get_walker_features,
                                   reward_factors_deep_mimic)
from flybody.tasks.trajectory_loaders import HDF5WalkingTrajectoryLoader
from flybody.tasks.task_utils import (add_trajectory_sites,
                                      update_trajectory_sites)
from flybody.quaternions import rotate_vec_with_quat


class WalkImitation(Walking):
    """Class for task of fly walking and tracking reference."""

    def __init__(self,
                 traj_generator: HDF5WalkingTrajectoryLoader,
                 mocap_joint_names: Sequence[str],
                 mocap_site_names: Sequence[str],
                 terminal_com_dist: float = 0.33,
                 claw_friction: float | None = 1.0,
                 trajectory_sites: bool = True,
                 inference_mode: bool = False,
                 **kwargs):
        """This task is a combination of imitation walking and ghost tracking.

        Args:
            traj_generator: Trajectory generator for generating walking
                trajectories.
            mocap_joint_names: Names of mocap joints.
            mocap_site_names: Names of mocap sites.
            terminal_com_dist: Episode will be terminated when CoM distance
                from model to ghost exceeds terminal_com_dist.
            claw_friction: Friction of claw.
            trajectory_sites: Whether to render trajectory sites.
            inference_mode: Whether to run in test mode and skip full-body
                reward calculation.
            **kwargs: Arguments passed to the superclass constructor.
        """

        super().__init__(add_ghost=True, ghost_visible_legs=False, **kwargs)

        self._traj_generator = traj_generator
        self._terminal_com_dist = terminal_com_dist
        self._trajectory_sites = trajectory_sites
        self._inference_mode = inference_mode
        self._max_episode_steps = round(
            self._time_limit / self.control_timestep) + 1
        self._next_traj_idx = None

        # Get mocap joints.
        self._mocap_joints = [self._root_joint]
        for mocap_joint_name in mocap_joint_names:
            self._mocap_joints.append(
                self._walker.mjcf_model.find('joint', mocap_joint_name))

        # Get mocap sites.
        self._mocap_sites = []
        for mocap_site_name in mocap_site_names:
            self._mocap_sites.append(
                self._walker.mjcf_model.find('site', mocap_site_name))

        # Maybe change default claw friction.
        if claw_friction is not None:
            self._walker.mjcf_model.find(
                'default',
                'adhesion-collision').geom.friction = (claw_friction, )

        # Maybe add trajectory sites, one every 10 steps.
        if self._trajectory_sites:
            self._n_traj_sites = (
                round(self._time_limit / self.control_timestep) + 1) // 10
            add_trajectory_sites(self.root_entity, self._n_traj_sites, group=1)

        # Additional task observables for tracking reference fly.
        self._walker.observables.add_observable('ref_displacement',
                                                self.ref_displacement)
        self._walker.observables.add_observable('ref_root_quat',
                                                self.ref_root_quat)

    def set_next_trajectory_index(self, idx):
        """In the next episode (only), this requested trajectory will be used.
        Could be used for testing, debugging."""
        self._next_traj_idx = idx

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        super().initialize_episode_mjcf(random_state)

        # Pick walking snippet (get snippet dict).
        self._snippet = self._traj_generator.get_trajectory(
            traj_idx=self._next_traj_idx)
        self._next_traj_idx = None  # Reset if wasn't None.

        # Update reference trajectory for tracking observables.
        self._ref_qpos = self._snippet['qpos']
        self._ref_qvel = self._snippet['qvel']

        self._snippet_steps = self._ref_qpos.shape[0] - self._future_steps - 1
        self._episode_steps = min(self._max_episode_steps, self._snippet_steps)

        # Update positions of trajectory sites.
        if self._trajectory_sites:
            update_trajectory_sites(self.root_entity, self._ref_qpos,
                                    self._n_traj_sites, self._episode_steps)

    def initialize_episode(self, physics: 'mjcf.Physics',
                           random_state: np.random.RandomState):
        """Randomly selects a starting point and set the walker."""
        super().initialize_episode(physics, random_state)

        # Set full initial qpos
        physics.bind(self._mocap_joints).qpos = self._ref_qpos[0, :]

        # Maybe set initial qvel.
        if self._initialize_qvel:
            physics.bind(self._mocap_joints).qvel = self._ref_qvel[0, :]

        # If enabled, initialize wing joint angles in retracted position.
        physics.bind(self._wing_joints).qpos = self._wing_springrefs

        # Rotate ghost offset, depending on initial reference orientation.
        rotated_offset = rotate_vec_with_quat(self._ghost_offset,
                                              self._ref_qpos[0, 3:7])

        rotated_offset[2] = self._ghost_offset[2]  # Restore original z-offset.
        self._ghost_offset_with_quat = np.hstack((rotated_offset, 4 * [0]))

        # Set initial ghost position.
        ghost_qpos = self._ref_qpos[0, :7] + self._ghost_offset_with_quat
        self._ghost.set_pose(physics, ghost_qpos[:3], ghost_qpos[3:])

    def before_step(self, physics: 'mjcf.Physics', action,
                    random_state: np.random.RandomState):
        # Set ghost joint position and velocity.
        step = int(np.round(physics.data.time / self.control_timestep))
        ghost_qpos = self._ref_qpos[step, :7] + self._ghost_offset_with_quat
        ghost_qvel = self._ref_qvel[step, :6]
        self._ghost.set_pose(physics, ghost_qpos[:3], ghost_qpos[3:])
        self._ghost.set_velocity(physics, ghost_qvel[:3], ghost_qvel[3:])

        # Protect from rare NaN actions.
        action[np.isnan(action)] = 0.

        super().before_step(physics, action, random_state)

    def get_reward_factors(self, physics):
        """Returns factorized reward terms."""

        if self._inference_mode:
            return (1,)

        # Walking imitation rewards.
        step = round(physics.time() / self.control_timestep)
        walker_ft = get_walker_features(physics, self._mocap_joints,
                                        self._mocap_sites)
        reference_ft = get_reference_features(self._snippet, step)
        reward_factors = reward_factors_deep_mimic(
            walker_features=walker_ft,
            reference_features=reference_ft,
            weights=(20, 1, 1, 1))

        # Reward for wing retraction.
        qpos_diff = physics.bind(self._wing_joints).qpos - self._wing_springrefs
        retract_wings = rewards.tolerance(qpos_diff,
                                          bounds=(0, 0),
                                          sigmoid='linear',
                                          margin=3.,
                                          value_at_margin=0.0)
        reward_factors = np.hstack((reward_factors, retract_wings))

        return reward_factors

    def check_termination(self, physics: 'mjcf.Physics') -> bool:
        """Check various termination conditions."""
        linvel = np.linalg.norm(self._walker.observables.velocimeter(physics))
        angvel = np.linalg.norm(self._walker.observables.gyro(physics))

        step = round(physics.time() / self.control_timestep)
        com_dist = np.linalg.norm(
            self.observables['walker/ref_displacement'](physics)[0])
        self._reached_traj_end = (step == self._episode_steps)

        return (linvel > _TERMINAL_LINVEL or angvel > _TERMINAL_ANGVEL
                or step == self._episode_steps
                or com_dist > self._terminal_com_dist
                or super().check_termination(physics))

    def get_discount(self, physics: 'mjcf.Physics'):
        """Override base class method to incorporate 'good' termination."""
        del physics  # Not used.
        if self._should_terminate and not self._reached_traj_end:
            # Return 0 only in case of fatal termination, before reaching end of
            # trajectory.
            return 0.0
        # Return 1 during episode and in case of 'good' end-of-trajectory or
        # end-of-episode termination.
        return 1.0
