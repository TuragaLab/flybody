"""Flight imitation task."""
# ruff: noqa: F821

import numpy as np
from dm_control.utils import rewards

from flybody.tasks.pattern_generators import WingBeatPatternGenerator
from flybody.tasks.trajectory_loaders import HDF5FlightTrajectoryLoader
from flybody.quaternions import quat_dist_short_arc
from flybody.tasks.task_utils import (com2root, root2com, add_trajectory_sites,
                                      update_trajectory_sites)
from flybody.tasks.constants import _TERMINAL_HEIGHT
from flybody.tasks.base import Flying


class FlightImitationWBPG(Flying):
    """WBPG-based flight tracking task."""

    def __init__(self,
                 wbpg: WingBeatPatternGenerator,
                 traj_generator: HDF5FlightTrajectoryLoader,
                 terminal_com_dist: float = 2.0,
                 trajectory_sites: bool = True,
                 **kwargs):
        """Task of learning a policy for flying and maneuvering while using a
        wing beat pattern generator with controllable wing beat frequency.

        Args:
            wbpg: Wing beat pattern generator for generating wing beat patterns.
            traj_generator: Trajectory generator for generating flight trajectories.
            terminal_com_dist: Episode will be terminated when CoM distance from
                model to ghost exceeds terminal_com_dist.
            trajectory_sites: Whether to render trajectory sites.
            **kwargs: Arguments passed to the superclass constructor.
        """

        super().__init__(add_ghost=True, num_user_actions=1, **kwargs)

        self._wbpg = wbpg
        self._traj_generator = traj_generator
        self._terminal_com_dist = terminal_com_dist
        self._trajectory_sites = trajectory_sites
        self._next_traj_idx = None

        # Add axis crosshair.
        self._crosshair_sites = []
        world = self.root_entity.mjcf_model.worldbody
        for i, fromto in enumerate([[0, 0, 0, 0, 0, 1],
                                    [-0.5, 0, 0.5, 0.5, 0, 0.5],
                                    [0, -0.5, 0.5, 0, 0.5, 0.5]]):
            self._crosshair_sites.append(
                world.add('site',
                          name=f'crosshair_{i}',
                          type='capsule',
                          size=(0.002, ),
                          fromto=fromto,
                          rgba=(0, 0.8, 0.5, 0.5)))

        # Get wing joint indices into agent's action vector.
        self._wing_inds_action = self._walker._action_indices['wings']
        # Get 'user' index into agent's action vector (only one user action).
        self._user_idx_action = self._walker._action_indices['user'][0]

        # Maybe add trajectory sites, one every 10 steps.
        if self._trajectory_sites:
            self._n_traj_sites = (
                round(self._time_limit / self.control_timestep) + 1) // 10
            add_trajectory_sites(self.root_entity, self._n_traj_sites, group=1)

        # Explicitly add tracking task observables.
        self._walker.observables.add_observable('ref_displacement',
                                                self.ref_displacement)
        self._walker.observables.add_observable('ref_root_quat',
                                                self.ref_root_quat)

    def set_next_trajectory_index(self, idx):
        """In the next episode (only), this requested trajectory will be used.
        Could be used for testing, debugging.
        """
        self._next_traj_idx = idx

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        super().initialize_episode_mjcf(random_state)

        # Get next trajectory.
        self._ref_qpos, self._ref_qvel = self._traj_generator.get_trajectory(
            traj_idx=self._next_traj_idx)
        self._next_traj_idx = None  # Reset if wasn't None.

        # Transform _ghost_qpos trajectory (which is really CoM trajectory from
        # data) to the corresponding actual qpos of ghost root joint.
        ghost_root_pos = com2root(self._ref_qpos[:, :3], self._ref_qpos[:, 3:])
        ghost_root_quat = self._ref_qpos[:, 3:7]
        self._ref_qpos = np.concatenate((ghost_root_pos, ghost_root_quat),
                                        axis=1)

        # Set trajectory time limits for early 'good' termination.
        self._traj_timesteps = min(
            self._ref_qpos.shape[0],
            round(self._time_limit / self.control_timestep))
        self._traj_timesteps -= self._future_steps + 1
        self._reached_traj_end = False

        # Update positions of trajectory sites.
        if self._trajectory_sites:
            update_trajectory_sites(self.root_entity, self._ref_qpos,
                                    self._n_traj_sites, self._traj_timesteps)
        # Update axis crosshair position.
        z = self._ref_qpos[0, 2]
        self._crosshair_sites[0].fromto[[2, 5]] = [0, z + 0.5]
        self._crosshair_sites[1].fromto[[2, 5]] = z
        self._crosshair_sites[2].fromto[[2, 5]] = z

    def initialize_episode(self, physics: 'mjcf.Physics',
                           random_state: np.random.RandomState):
        """Randomly select a starting point and set the walker.

        Environment call sequence:
            check_termination, get_reward_factors, get_discount
        """
        super().initialize_episode(physics, random_state)

        ghost_qpos = self._ref_qpos[0, :] + np.hstack(
            (self._ghost_offset, 4 * [0]))
        self._ghost.set_pose(physics, ghost_qpos[:3], ghost_qpos[3:])

        # Reset wing pattern generator and get initial wing qpos.
        init_wing_qpos, init_wing_qvel = self._wbpg.reset(
            initial_phase=random_state.uniform(), return_qvel=True)
        # Initialize root position and orientation.
        self._walker.set_pose(physics, self._ref_qpos[0, :3],
                              self._ref_qpos[0, 3:])
        # Initialize wing qpos.
        physics.bind(self._wing_joints).qpos = init_wing_qpos
        # Initialize wing qvel.
        physics.bind(self._wing_joints).qvel = init_wing_qvel

        if self._initialize_qvel:
            # Only initialize linear CoM velocity, not rotational velocity.
            self._walker.set_velocity(physics, self._ref_qvel[0, :3])

        # If enabled, initialize leg joint angles in retracted position.
        if self._leg_joints:
            physics.bind(self._leg_joints).qpos = self._leg_springrefs

    def before_step(self, physics: 'mjcf.Physics', action,
                    random_state: np.random.RandomState):
        """Combine action with WPG base pattern. Update ghost pos and vel."""
        # Get target wing joint angles at beat frequency requested by the agent.
        base_freq, rel_range = self._wbpg.base_beat_freq, self._wbpg.rel_freq_range
        act = action[self._user_idx_action]  # action in [-1, 1].
        ctrl_freq = base_freq * (1 + rel_range * act)
        ctrl = self._wbpg.step(
            ctrl_freq=ctrl_freq)  # Returns position control.

        length = physics.bind(self._wing_joints).qpos
        # Convert position control to force control.
        action[self._wing_inds_action] += (ctrl - length)

        # Update ghost joint pos and vel.
        step = int(np.round(physics.data.time / self.control_timestep))
        ghost_qpos = self._ref_qpos[step, :] + np.hstack(
            (self._ghost_offset, 4 * [0]))
        self._ghost.set_pose(physics, ghost_qpos[:3], ghost_qpos[3:])
        self._ghost.set_velocity(physics, self._ref_qvel[step, :3],
                                 self._ref_qvel[step, 3:])

        super().before_step(physics, action, random_state)

    def get_reward_factors(self, physics):
        """Returns factorized reward terms."""

        # Reference CoM displacement reward.
        ghost_xpos, ghost_quat = self._ghost.get_pose(physics)
        ghost_com = root2com(np.concatenate((ghost_xpos, ghost_quat)))
        model_com = physics.named.data.subtree_com['walker/']
        displacement = np.linalg.norm(ghost_com - model_com)
        displacement = rewards.tolerance(displacement,
                                         bounds=(0, 0),
                                         sigmoid='linear',
                                         margin=0.4,
                                         value_at_margin=0.0)

        # Reference root quaternion displacement reward.
        quat = self.observables['walker/ref_root_quat'](physics)[0]
        quat_dist = quat_dist_short_arc(np.array([1., 0, 0, 0]), quat)
        quat_dist = rewards.tolerance(quat_dist,
                                      bounds=(0, 0),
                                      sigmoid='linear',
                                      margin=np.pi,
                                      value_at_margin=0.0)

        # Reward for leg retraction. If legs are disabled, this reward term is 1.
        qpos_diff = physics.bind(self._leg_joints).qpos - self._leg_springrefs
        retract_legs = rewards.tolerance(qpos_diff,
                                         bounds=(0, 0),
                                         sigmoid='linear',
                                         margin=4.,
                                         value_at_margin=0.0)

        return np.hstack((displacement, quat_dist, retract_legs))

    def check_termination(self, physics: 'mjcf.Physics') -> bool:
        """Check various termination conditions."""
        height = self._walker.observables.thorax_height(physics)
        com_dist = np.linalg.norm(
            self.observables['walker/ref_displacement'](physics)[0])
        current_step = np.round(physics.time() / self.control_timestep)
        self._reached_traj_end = current_step == self._traj_timesteps
        return (height < _TERMINAL_HEIGHT or com_dist > self._terminal_com_dist
                or self._reached_traj_end
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
