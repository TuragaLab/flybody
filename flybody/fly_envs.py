"""Create examples of flight and walking task environments for fruitfly."""

from typing import Callable, Sequence

import numpy as np

from dm_control import mujoco
from dm_control import composer
from dm_control.locomotion.arenas import floors

from flybody.fruitfly import fruitfly

from flybody.tasks.flight_imitation import FlightImitationWBPG
from flybody.tasks.walk_imitation import WalkImitation
from flybody.tasks.walk_on_ball import WalkOnBall
from flybody.tasks.vision_flight import VisionFlightImitationWBPG
from flybody.tasks.template_task import TemplateTask

from flybody.tasks.arenas.ball import BallFloor
from flybody.tasks.arenas.hills import SineBumps, SineTrench
from flybody.tasks.pattern_generators import WingBeatPatternGenerator
from flybody.tasks.trajectory_loaders import (
    HDF5FlightTrajectoryLoader,
    HDF5WalkingTrajectoryLoader,
    InferenceWalkingTrajectoryLoader,
    InferenceFlightTrajectoryLoader,
)


def flight_imitation(ref_path: str | None = None,
                     wpg_pattern_path: str | None = None,
                     force_actuators: bool = False,
                     disable_legs: bool = True,
                     traj_indices: Sequence[int] | None = None,
                     randomize_start_step: bool = True,
                     joint_filter: float = 0.,
                     future_steps: int = 5,
                     random_state: np.random.RandomState | None = None,
                     terminal_com_dist: float = 2.0):
    """Requires a fruitfly to track a flying reference.
  
    Args:
        ref_path: Path to reference trajectory dataset. If None, task will
            run with InferenceFlightTrajectoryLoader, without loading actual
            flight dataset.
        wpg_pattern_path: Path to baseline wing beat pattern for WPG. If None,
            a simple approximate wing pattern will be generated in the
            WingBeatPatternGenerator class (could be used for testing).
        force_actuators: Whether to use force actuators or position actuators
            for everything other than wings. Wings always use force actuators.
        disable_legs: Whether to retract and disable legs. This includes
            removing leg DoFs, actuators, and sensors.
        traj_indices: List of trajectory indices to use, e.g. for train/test
            splitting etc. If None, use all available trajectories.
        randomize_start_step: Whether to select random start point in each
            get_trajectory call in trajectory loader.
        joint_filter: Time constant of joint actuator filter. 0: disabled.
        future_steps: Number of future steps of reference trajectory to provide
            as observables. Zero means only the current step is used.
        random_state: Random state for reproducibility.
        terminal_com_dist: Episode will be terminated when distance from model
            CoM to ghost CoM exceeds terminal_com_dist. Can be float('inf').

    Returns:
        Environment for flight tracking task.
    """
    # Build a fruitfly walker and arena.
    walker = fruitfly.FruitFly
    arena = floors.Floor()
    # Initialize wing pattern generator and flight trajectory loader.
    wbpg = WingBeatPatternGenerator(base_pattern_path=wpg_pattern_path)
    if ref_path is not None:
        traj_generator = HDF5FlightTrajectoryLoader(
            path=ref_path,
            traj_indices=traj_indices,
            randomize_start_step=randomize_start_step,
            random_state=random_state)
    else:
        traj_generator = InferenceFlightTrajectoryLoader()
    # Build the task.
    time_limit = 0.6
    task = FlightImitationWBPG(walker=walker,
                               arena=arena,
                               wbpg=wbpg,
                               traj_generator=traj_generator,
                               terminal_com_dist=terminal_com_dist,
                               initialize_qvel=True,
                               force_actuators=force_actuators,
                               disable_legs=disable_legs,
                               time_limit=time_limit,
                               joint_filter=joint_filter,
                               future_steps=future_steps)

    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def walk_imitation(ref_path: str | None = None,
                   force_actuators: bool = False,
                   disable_wings: bool = True,
                   traj_indices: Sequence[int] | None = None,
                   random_state: np.random.RandomState | None = None,
                   terminal_com_dist: float = 0.3,
                   joint_filter: float = 0.01):
    """Requires a fruitfly to track a reference walking fly.

    Args:
        ref_path: Path to reference trajectory dataset. If not provided, task
            will run in inference mode with InferenceWalkingTrajectoryLoader,
            without loading actual walking dataset.
        force_actuators: Whether to use force or position actuators.
        disable_wings: Whether to retract and disable wings. This includes
            removing wing DoFs, actuators, and sensors.
        traj_indices: List of trajectory indices to use, e.g. for train/test
            splitting etc. If None, use all available trajectories.
        random_state: Random state for reproducibility.
        terminal_com_dist: Episode will be terminated when distance from model
            CoM to ghost CoM exceeds terminal_com_dist. Can be float('inf').
        joint_filter: Timescale of filter for joint actuators. 0: disabled.

    Returns:
        Environment for walking tracking task.
    """
    # Build a fruitfly walker and arena.
    walker = fruitfly.FruitFly
    arena = floors.Floor()
    # Initialize a walking trajectory loader.
    if ref_path is not None:
        inference_mode = False
        traj_generator = HDF5WalkingTrajectoryLoader(
            path=ref_path, random_state=random_state, traj_indices=traj_indices)
    else:
        inference_mode = True
        traj_generator = InferenceWalkingTrajectoryLoader()
    # Build a task that rewards the agent for tracking a walking ghost.
    time_limit = 10.0
    task = WalkImitation(walker=walker,
                         arena=arena,
                         traj_generator=traj_generator,
                         terminal_com_dist=terminal_com_dist,
                         mocap_joint_names=traj_generator.get_joint_names(),
                         mocap_site_names=traj_generator.get_site_names(),
                         inference_mode=inference_mode,
                         force_actuators=force_actuators,
                         disable_wings=disable_wings,
                         joint_filter=joint_filter,
                         future_steps=64,
                         time_limit=time_limit)

    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def walk_on_ball(force_actuators: bool = False,
                 disable_wings: bool = True,
                 random_state: np.random.RandomState | None = None):
    """Requires a tethered fruitfly to walk on a floating ball.

    Args:
        force_actuators: Whether to use force or position actuators.
        disable_wings: Whether to retract and disable wings. This includes
            removing wing DoFs, actuators, and sensors.
        random_state: Random state for reproducibility.

    Returns:
        Environment for fly walking on ball.
    """
    # Build a fruitfly walker and arena.
    walker = fruitfly.FruitFly
    arena = BallFloor(ball_pos=(-0.05, 0, -0.419),
                      ball_radius=0.454,
                      ball_density=0.0025,
                      skybox=False)
    # Build a task that rewards the agent for tracking a walking ghost.
    time_limit = 2.
    task = WalkOnBall(walker=walker,
                      arena=arena,
                      force_actuators=force_actuators,
                      disable_wings=disable_wings,
                      joint_filter=0.01,
                      adhesion_filter=0.007,
                      time_limit=time_limit)

    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def vision_guided_flight(wpg_pattern_path: str | None = None,
                         bumps_or_trench: str = 'bumps',
                         force_actuators: bool = False,
                         disable_legs: bool = True,
                         random_state: np.random.RandomState | None = None,
                         joint_filter: float = 0.,
                         **kwargs_arena):
    """Vision-guided flight tasks: 'bumps' and 'trench'.

    Args:
        wpg_pattern_path: Path to baseline wing beat pattern for WPG. If None,
            a simple approximate wing pattern will be generated in the
            WingBeatPatternGenerator class (could be used for testing).
        bumps_or_trench: Whether to create 'bumps' or 'trench' vision task.
        force_actuators: Whether to use force actuators or position actuators
            for everything other than wings. Wings always use force actuators.
        disable_legs: Whether to retract and disable legs. This includes
            removing leg DoFs, actuators, and sensors.
        random_state: Random state for reproducibility.
        joint_filter: Timescale of filter for joint actuators. 0: disabled.
        kwargs_arena: kwargs to be passed on to arena.

    Returns:
        Environment for vision-guided flight task.
    """

    if bumps_or_trench == 'bumps':
        arena = SineBumps
    elif bumps_or_trench == 'trench':
        arena = SineTrench
    else:
        raise ValueError("Only 'bumps' and 'trench' terrains are supported.")
    # Build fruitfly walker and arena.
    walker = fruitfly.FruitFly
    arena = arena(**kwargs_arena)
    # Initialize a wing beat pattern generator.
    wbpg = WingBeatPatternGenerator(base_pattern_path=wpg_pattern_path)
    # Build task.
    time_limit = 0.4
    task = VisionFlightImitationWBPG(walker=walker,
                                     arena=arena,
                                     wbpg=wbpg,
                                     time_limit=time_limit,
                                     force_actuators=force_actuators,
                                     disable_legs=disable_legs,
                                     joint_filter=joint_filter,
                                     floor_contacts=True,
                                     floor_contacts_fatal=True)

    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def template_task(random_state: np.random.RandomState | None = None,
                  force_actuators: bool = False,
                  disable_wings: bool = True,
                  joint_filter: float = 0.01,
                  adhesion_filter: float = 0.007,
                  time_limit: float = 1.,
                  mjcb_control: Callable | None = None,
                  observables_options: dict | None = None,
                  action_corruptor: Callable | None = None):
    """An empty no-op walking task for testing.

    Args:
        random_state: Random state for reproducibility.
        force_actuators: Whether to use force or position actuators.
        disable_wings: Whether to retract and disable wings. This includes
            removing wing DoFs, actuators, and sensors.
        joint_filter: Timescale of filter for joint actuators. 0: disabled.
        adhesion_filter: Timescale of filter for adhesion actuators. 0: disabled.
        time_limit: Episode time limit.
        mjcb_control: Optional MuJoCo control callback, a callable with
            arguments (model, data). For more information, see
            https://mujoco.readthedocs.io/en/stable/APIreference/APIglobals.html#mjcb-control
        observables_options (optional): A dict of dicts of configuration options
            keyed on observable names, or a dict of configuration options, which
            will propagate those options to all observables.
        action_corruptor (optional): A callable which takes an action as an
            argument, modifies it, and returns it. An example use case for
            this is to add random noise to the action.

    Returns:
        Template walking environment.
    """
    # Build a fruitfly walker and arena.
    walker = fruitfly.FruitFly
    arena = floors.Floor()
    # Build a no-op task.
    task = TemplateTask(walker=walker,
                        arena=arena,
                        force_actuators=force_actuators,
                        disable_wings=disable_wings,
                        joint_filter=joint_filter,
                        adhesion_filter=adhesion_filter,
                        observables_options=observables_options,
                        mjcb_control=mjcb_control,
                        action_corruptor=action_corruptor,
                        time_limit=time_limit)
    # Reset control callback, if any.
    mujoco.set_mjcb_control(None)
    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)
