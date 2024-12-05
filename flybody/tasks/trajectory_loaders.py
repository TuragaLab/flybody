"""Reference trajectory loaders for fruit fly imitation tasks."""

from typing import Sequence
from abc import ABC, abstractmethod

import h5py
import numpy as np

from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from flybody.tasks.constants import _FLY_CONTROL_TIMESTEP, _WALK_CONTROL_TIMESTEP


class HDF5TrajectoryLoader(ABC):
    """Base class for loading and serving trajectories from hdf5 datasets."""

    def __init__(self,
                 path: str,
                 traj_indices: Sequence[int] | None = None,
                 random_state: np.random.RandomState | None = None):
        """Initializes the base trajectory loader.

        Args:
            path: Path to hdf5 dataset file with reference rajectories.
            traj_indices: List of trajectory indices to use, e.g. for train/test
                splitting etc. If None, use all available trajectories.
            random_state: Random state for reproducibility.
        """

        if random_state is None:
            self._random_state = np.random.RandomState(None)
        else:
            self._random_state = random_state

        with h5py.File(path, 'r') as f:
            self._n_traj = len(f['trajectories'])
            self._timestep = f['timestep_seconds'][()]

        if traj_indices is None:
            self._traj_indices = np.arange(self._n_traj)
        else:
            self._traj_indices = traj_indices

    @property
    def timestep(self):
        """Dataset timestep duration, in seconds."""
        return self._timestep

    @property
    def num_trajectories(self):
        """Number of trajectories in dataset."""
        return self._n_traj

    @property
    def traj_indices(self):
        """Indices of trajectories to use for training/testing."""
        return self._traj_indices

    @abstractmethod
    def get_trajectory(self,
                       traj_idx: int | None = None,
                       start_step: int | None = None,
                       end_step: int | None = None):
        """Returns a trajectory."""
        raise NotImplementedError("Subclasses should implement this.")


class HDF5FlightTrajectoryLoader(HDF5TrajectoryLoader):
    """Loads and serves trajectories from hdf5 flight imitation dataset."""

    def __init__(
        self,
        path: str,
        traj_indices: Sequence[int] | None = None,
        randomize_start_step: bool = True,
        random_state: np.random.RandomState | None = None,
    ):
        """Initializes the flight trajectory loader.

        Args:
            path: Path to hdf5 dataset file with reference rajectories.
            traj_indices: List of trajectory indices to use, e.g. for train/test
                splitting etc. If None, use all available trajectories.
            randomize_start_step: Whether to select random start point in each
                get_trajectory call.
            random_state: Random state for reproducibility.
        """
        super().__init__(path, traj_indices, random_state=random_state)

        self._randomize_start_step = randomize_start_step

        self._com_qpos = []
        self._com_qvel = []

        with h5py.File(path, 'r') as f:
            n_zeros = len(str(self._n_traj))
            for idx in range(self._n_traj):
                key = str(idx).zfill(n_zeros)
                self._com_qpos.append(f['trajectories'][key]['com_qpos'][()])
                self._com_qvel.append(f['trajectories'][key]['com_qvel'][()])
                assert self._com_qpos[-1].shape[0] == self._com_qvel[-1].shape[0]

    def trajectory_len(self, traj_idx: int) -> int:
        """Returns length of trajectory with index traj_idx."""
        return len(self._com_qpos[traj_idx])

    def get_trajectory(
            self,
            traj_idx: int | None = None,
            start_step: int | None = None,
            end_step: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns a flight trajectory from the dataset.

        Args:
            traj_idx: Index of the desired trajectory. If None, a random
                trajectory out of traj_indices is selected.
            start_step: Start index for the trajectory slice. If None, defaults
                to the beginning. If attribute randomize_start_step is True,
                start_step is ignored.
            end_step: End index for the trajectory slice. If None, defaults to
                the end. If attribute randomize_start_step is True,
                end_step is ignored.

        Returns:
            tuple: Two numpy arrays for com_qpos and com_qvel respectively.
        """
        if traj_idx is None:
            traj_idx = self._random_state.choice(self._traj_indices)

        traj_len = len(self._com_qpos[traj_idx])
        if self._randomize_start_step:
            start_step = self._random_state.randint(traj_len - 50)
            end_step = traj_len
        else:
            start_step = 0 if start_step is None else start_step
            end_step = traj_len if end_step is None else end_step

        com_qpos = self._com_qpos[traj_idx][start_step:end_step].copy()
        com_qvel = self._com_qvel[traj_idx][start_step:end_step]
        com_qpos[:, :2] -= com_qpos[0, :2]

        return com_qpos, com_qvel


class InferenceFlightTrajectoryLoader():
    """Simple drop-in inference-time replacement for flight trajectory loader.
    
    This trajectory loader can be used for bypassing loading actual flight
    datasets and loading custom trajectories instead, e.g. at inference time.

    A simple synthetic flight trajectory is automatically set upon this class
    initialization.

    To use this class with other custom trajectories, create qpos and qvel for
    your custom trajectory and then set this trajectory for loading in the
    flight task by calling:
    env.task._traj_generator.set_next_trajectory(qpos, qvel)
    """

    def __init__(self):
        # Initially, set a simple synthetic trajectory, e.g. for quick testing.
        qpos, qvel = constant_speed_trajectory(
            n_steps=200, speed=20, init_pos=(0, 0, 1),
            body_rot_angle_y=-47.5, control_timestep=_FLY_CONTROL_TIMESTEP)
        self.set_next_trajectory(qpos, qvel)

    def set_next_trajectory(self, com_qpos: np.ndarray, com_qvel: np.ndarray):
        """Set new trajectory to be returned by get_trajectory.
        
        Args:
            com_qpos: Center-of-mass trajectory, (time, 7).
            com_qvel: Velocity of CoM trajectory, (time, 6).
        """
        self._com_qpos = com_qpos.copy()
        self._com_qpos[:, :2] -= self._com_qpos[0, :2]
        self._com_qvel = com_qvel
        
    def get_trajectory(self, traj_idx: int):
        del traj_idx  # Unused.
        if not hasattr(self, '_com_qpos'):
            raise AttributeError(
                'Trajectory not set yet. Call set_next_trajectory first.')
        return self._com_qpos, self._com_qvel


class HDF5WalkingTrajectoryLoader(HDF5TrajectoryLoader):
    """Loads and serves trajectories from hdf5 walking imitation dataset."""

    def __init__(
        self,
        path: str,
        traj_indices: Sequence[int] | None = None,
        random_state: np.random.RandomState | None = None,
    ):
        """Initializes the walking trajectory loader.

        Args:
            path: Path to hdf5 dataset file with reference rajectories.
            traj_indices: List of trajectory indices to use, e.g. for train/test
                splitting etc. If None, use all available trajectories.
            random_state: Random state for reproducibility.
        """

        super().__init__(path, traj_indices, random_state=random_state)

        self._h5 = h5py.File(path, 'r')
        self._traj_lens = self._h5['trajectory_lengths']
        self._n_zeros = len(str(self._n_traj))

    def trajectory_len(self, traj_idx: int) -> int:
        """Returns length of trajectory with index traj_idx."""
        return self._traj_lens[traj_idx]

    def get_trajectory(
            self,
            traj_idx: int | None = None,
            start_step: int | None = None,
            end_step: int | None = None) -> dict[str, np.ndarray]:
        """Returns a walking trajectory from the dataset.

        Args:
            traj_idx: Index of the desired trajectory. If None, a random
                trajectory is selected.
            start_step: Start index for the trajectory slice. If None, defaults
                to the beginning.
            end_step: End index for the trajectory slice. If None, defaults to
                the end.

        Returns:
            dict: Dictionary containing qpos, qvel, root2site, and joint_quat
                of the trajectory.
        """
        if traj_idx is None:
            traj_idx = self._random_state.choice(self._traj_indices)

        start_step = 0 if start_step is None else start_step
        end_step = self._traj_lens[traj_idx] if end_step is None else end_step

        key = str(traj_idx).zfill(self._n_zeros)
        snippet = self._h5['trajectories'][key]

        qpos = np.concatenate((snippet['root_qpos'][start_step:end_step],
                               snippet['qpos'][start_step:end_step]),
                              axis=1)
        qvel = np.concatenate((snippet['root_qvel'][start_step:end_step],
                               snippet['qvel'][start_step:end_step]),
                              axis=1)
        qpos[:, :2] -= qpos[0, :2]

        trajectory = {
            'qpos': qpos,
            'qvel': qvel,
            'root2site': snippet['root2site'][start_step:end_step],
            'joint_quat': snippet['joint_quat'][start_step:end_step]
        }

        return trajectory

    def get_site_names(self):
        """Returns snippet site names."""
        return [s.decode('utf-8') for s in self._h5['id2name']['sites']]

    def get_joint_names(self):
        """Returns snippet joint names."""
        return [s.decode('utf-8') for s in self._h5['id2name']['joints']]


class InferenceWalkingTrajectoryLoader():
    """Simple drop-in inference-time replacement for walking trajectory loader.
    
    This trajectory loader can be used for bypassing loading actual walking
    datasets and loading custom trajectories instead, e.g. at inference time.

    A simple synthetic walking trajectory is automatically set upon this class
    initialization.

    To use this class with other custom trajectories, create qpos and qvel for
    your custom trajectory and then set this trajectory for loading in the
    walking task by calling:
        env.task._traj_generator.set_next_trajectory(qpos, qvel)
    """

    def __init__(self):
        # Initially, set a simple synthetic trajectory, e.g. for quick testing.
        qpos, qvel = constant_speed_trajectory(
            n_steps=300, speed=2, init_pos=(0, 0, 0.1278),
            control_timestep=_WALK_CONTROL_TIMESTEP)
        self.set_next_trajectory(qpos, qvel)

    def set_next_trajectory(self, qpos: np.ndarray, qvel: np.ndarray):
        """Set new trajectory to be returned by get_trajectory.
        
        Args:
            qpos: Center-of-mass trajectory, (time, 7).
            qvel: Velocity of CoM trajectory, (time, 6).
        """
        self._snippet = {'qpos': qpos, 'qvel': qvel}
        
    def get_trajectory(self, traj_idx: int):
        del traj_idx  # Unused.
        if not hasattr(self, '_snippet'):
            raise AttributeError(
                'Trajectory not set yet. Call set_next_trajectory first.')
        return self._snippet
    
    def get_joint_names(self):
        return []
    
    def get_site_names(self):
        return []
