"""Create examples of flight and walking task environments for fruitfly."""

import numpy as np

from dm_control import composer
from dm_control.locomotion.arenas import floors

from flybody.fruitfly import fruitfly

from flybody.tasks.flight_imitation import FlightImitationWBPG
from flybody.tasks.walk_imitation import WalkImitation
from flybody.tasks.walk_on_ball import WalkOnBall
from flybody.tasks.arenas.ball import BallFloor

from flybody.tasks.pattern_generators import WingBeatPatternGenerator
from flybody.tasks.trajectory_loaders import (
    HDF5FlightTrajectoryLoader,
    HDF5WalkingTrajectoryLoader)


def flight_imitation(wpg_pattern_path: str,
                     ref_path: str,
                     random_state:np.random.RandomState = None):
  """Requires a fruitfly to track a flying reference.
  
    Args:
      wpg_pattern_path: Path to baseline wing beat pattern for WPG.
      ref_path: Path to reference trajectory dataset.
      random_state: Random state for reproducibility.

    Returns:
      Environment for flight tracking task.
  """
  # Build a fruitfly walker and arena.
  walker = fruitfly.FruitFly
  arena = floors.Floor()
  # Initialize wing pattern generator and flight trajectory loader.
  wbpg = WingBeatPatternGenerator(base_pattern_path=wpg_pattern_path)
  traj_generator = HDF5FlightTrajectoryLoader(path=ref_path,
                                              random_state=random_state)
  # Build the task.
  time_limit = 0.6
  task = FlightImitationWBPG(walker=walker,
                             arena=arena,
                             wbpg=wbpg,
                             traj_generator=traj_generator,
                             initialize_qvel=True,
                             time_limit=time_limit,
                             joint_filter=0.,
                             future_steps=5)

  return composer.Environment(time_limit=time_limit,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True)


def walk_imitation(ref_path: str,
                   random_state:np.random.RandomState = None):
  """Requires a fruitfly to track a reference walking fly.

    Args:
      ref_path: Path to reference trajectory dataset.
      random_state: Random state for reproducibility.

    Returns:
      Environment for walking tracking task.
  """
  # Build a fruitfly walker and arena.
  walker = fruitfly.FruitFly
  arena = floors.Floor()
  # Initialize a walking trajectory loader.
  traj_generator = HDF5WalkingTrajectoryLoader(path=ref_path,
                                               random_state=random_state)
  # Build a task that rewards the agent for tracking a walking ghost.
  time_limit = 10.0
  task = WalkImitation(walker=walker,
                       arena=arena,
                       traj_generator=traj_generator,
                       mocap_joint_names=traj_generator.get_joint_names(),
                       mocap_site_names=traj_generator.get_site_names(),
                       joint_filter=0.01,
                       future_steps=64,
                       time_limit=time_limit)
  
  return composer.Environment(time_limit=time_limit,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True)


def walk_on_ball(random_state:np.random.RandomState = None):
  """Requires a tethered fruitfly to walk on a floating ball.

    Args:
      random_state: Random state for reproducibility.

    Returns:
      Environment for fly walking on ball.
  """
  # Build a fruitfly walker and arena.
  walker = fruitfly.FruitFly
  arena = BallFloor(ball_pos=(0, 0, -1), skybox=True)
  # Build a task that rewards the agent for tracking a walking ghost.
  time_limit = 5.
  task = WalkOnBall(walker=walker,
                       arena=arena,
                       joint_filter=0.01,
                       adhesion_filter=0.007,
                       time_limit=time_limit)
  
  return composer.Environment(time_limit=time_limit,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True)