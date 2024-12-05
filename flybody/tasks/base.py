"""Base classes for fruitfly tasks."""
# ruff: noqa: F821

from typing import Callable, Union, Sequence
from abc import ABC, abstractmethod
import numpy as np

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable

from flybody.quaternions import get_dquat_local
from flybody.tasks.task_utils import make_ghost_fly
from flybody.utils import any_substr_in_str
from flybody.tasks.constants import (_FLY_PHYSICS_TIMESTEP,
                                     _FLY_CONTROL_TIMESTEP, _BODY_PITCH_ANGLE,
                                     _WALK_PHYSICS_TIMESTEP,
                                     _WALK_CONTROL_TIMESTEP, _TERMINAL_QACC,
                                     _WING_PARAMS)


class FruitFlyTask(composer.Task, ABC):
    """Base class for all tasks with fruitfly walkers."""

    def __init__(
        self,
        walker: Union['base.Walker', Callable],
        arena: composer.Arena,
        time_limit: float,
        use_legs: bool,
        use_wings: bool,
        use_mouth: bool,
        use_antennae: bool,
        physics_timestep: float,
        control_timestep: float,
        joint_filter: float,
        adhesion_filter: float = 0.007,
        force_actuators: bool = False,
        body_pitch_angle: float = _BODY_PITCH_ANGLE,
        stroke_plane_angle: float = 0,
        add_ghost: bool = False,
        ghost_visible_legs: bool = True,
        ghost_offset: Sequence = np.array([0, 0, 0]),
        num_user_actions: int = 0,
        eye_camera_fovy: float = 150.,
        eye_camera_size: int = 32,
        future_steps: int = 0,
        initialize_qvel: bool = False,
        observables_options: dict | None = None,
    ):
        """Construct a fruitfly task.

        Args:
            walker: Walker constructor to be used.
            arena: Arena to be used.
            time_limit: Time limit beyond which episode is forced to terminate.
            use_legs: Whether the legs are active.
            use_wings: Whether the wings are active.
            use_mouth: Whether the mouth is active.
            use_antennae: Whether the antennae are active.
            physics_timestep: Physics timestep to use for simulation.
            control_timestep: Control timestep.
            joint_filter: Timescale of filter for joint actuators. 0: disabled.
            adhesion_filter: Timescale of filter for adhesion actuators. 0: disabled.
            force_actuators: Whether to use force or position actuators.
            body_pitch_angle: Body pitch angle for initial flight pose, relative to
                ground, degrees. 0: horizontal body position. Default value from
                https://doi.org/10.1126/science.1248955
            stroke_plane_angle: Angle of wing stroke plane for initial flight pose,
                relative to ground, degrees. 0: horizontal stroke plane.
            add_ghost: Whether to add ghost fly to arena.
            ghost_visible_legs: Whether to show or hide ghost legs.
            ghost_offset: Shift ghost by this vector for better visualizations.
                In observables, the ghost is kept at its original position.
            num_user_actions: Optional, number of additional actions for custom usage,
                e.g. in before_step callback. The action range is [-1, 1]. 0: Not used.
            eye_camera_fovy: Vertical field of view of the eye cameras, degrees.
            eye_camera_size: Size in pixels (height and width) of the eye cameras.
                Height and width are assumed equal.
            future_steps: Number of future steps of reference trajectory to provide
                as observables. Zero means only the current step is used.
            initialize_qvel: whether to init qvel of root or not (wings are always vel
                inited)
            observables_options (optional): A dict of dicts of configuration options
                keyed on observable names, or a dict of configuration options, which
                will propagate those options to all observables.
        """
        self._time_limit = time_limit
        self._initialize_qvel = initialize_qvel

        # Initialise time offset for phase observation.
        self._time_offset = 0.0
        # Initial value of `_should_terminate`.
        self._should_terminate = False
        # Initialize timestep counter.
        self._step_counter = 0

        # Create the arena.
        self._arena = arena

        self._body_pitch_angle = body_pitch_angle
        self._stroke_plane_angle = stroke_plane_angle
        self._ghost_offset = ghost_offset
        self._num_user_actions = num_user_actions
        self._future_steps = future_steps

        # Instantiate a fruitfly walker.
        self._walker = walker(name='walker',
                              use_legs=use_legs,
                              use_wings=use_wings,
                              use_mouth=use_mouth,
                              use_antennae=use_antennae,
                              force_actuators=force_actuators,
                              joint_filter=joint_filter,
                              adhesion_filter=adhesion_filter,
                              body_pitch_angle=body_pitch_angle,
                              stroke_plane_angle=stroke_plane_angle,
                              physics_timestep=physics_timestep,
                              control_timestep=control_timestep,
                              num_user_actions=num_user_actions,
                              eye_camera_fovy=eye_camera_fovy,
                              eye_camera_size=eye_camera_size)
        # Set options to fly observables, if provided.
        self._walker.observables.set_options(observables_options)

        # Add it to the arena.
        spawn_pos = self._walker.upright_pose.xpos
        spawn_site = self._arena.mjcf_model.worldbody.add('site',
                                                          pos=spawn_pos)
        self._walker.create_root_joints(arena.attach(self._walker, spawn_site))
        spawn_site.remove()

        # Get joints.
        self._root_joint = mjcf.get_frame_freejoint(self._walker.mjcf_model)
        self._non_root_joints = self._walker.mjcf_model.find_all('joint')
        self._joints = [self._root_joint] + self._non_root_joints

        # Maybe add a (possibly invisible) ghost walker.
        if add_ghost:
            self._ghost = walker(name='ghost', use_wings=False, use_legs=False)
            make_ghost_fly(self._ghost,
                           visible=True,
                           visible_legs=ghost_visible_legs)
            spawn_pos = self._walker.upright_pose.xpos
            spawn_site = arena.mjcf_model.worldbody.add('site', pos=spawn_pos)
            self._ghost_frame = arena.attach(self._ghost, spawn_site)
            spawn_site.remove()

            self._ghost_joint = self._ghost_frame.add('joint',
                                                      type='free',
                                                      armature=1)
        else:
            self._ghost = None

        # Set timesteps.
        self.set_timesteps(physics_timestep=physics_timestep,
                           control_timestep=control_timestep)

        # Dummy initialization for base class observables.
        self._ref_qpos = np.zeros((self._future_steps + 1, 7))

        # Change mass and inertia bounds to get correct fly mass.
        self._walker.mjcf_model.compiler.boundmass = 0.
        self._walker.mjcf_model.compiler.boundinertia = 0.

        # === Explicitly enable observables.
        # Basic sensors.
        # vestibular: gyro, accelerometer, velocimeter, world_zaxis.
        # proprioception: joints_pos, joints_vel, actuator_activation.
        for sensor in (self._walker.observables.vestibular +
                       self._walker.observables.proprioception):
            sensor.enabled = True

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        if hasattr(self._arena, 'regenerate'):
            self._arena.regenerate(random_state)
        # Better visual defaults for CGS units.
        # Important: these particular values of znear, zfar, extent are
        # critical for the visually-guided flight task.
        self.root_entity.mjcf_model.visual.map.znear = 0.001
        self.root_entity.mjcf_model.visual.map.zfar = 50.0
        self.root_entity.mjcf_model.visual.map.force = 0.00001
        self.root_entity.mjcf_model.visual.scale.framewidth = 0.06
        self.root_entity.mjcf_model.visual.scale.forcewidth = 0.06
        self.root_entity.mjcf_model.visual.scale.contactwidth = 0.3
        self.root_entity.mjcf_model.visual.scale.contactheight = 0.1
        self.root_entity.mjcf_model.visual.scale.jointwidth = 0.06
        self.root_entity.mjcf_model.statistic.extent = 4.01

    def initialize_episode(self, physics, random_state):
        # Reset control timestep counter.
        self._step_counter = 0

    def before_step(self, physics: 'mjcf.Physics', action,
                    random_state: np.random.RandomState):
        """Apply actions."""
        self._step_counter += 1
        self._walker.apply_action(physics, action, random_state)

    def should_terminate_episode(self, physics: 'mjcf.Physics'):
        return self._should_terminate

    def get_discount(self, physics: 'mjcf.Physics'):
        del physics  # Unused by get_discount.
        if self._should_terminate:
            return 0.0
        return 1.0

    def get_reward(self, physics: 'mjcf.Physics') -> float:
        # Check termination.
        self._should_terminate = self.check_termination(physics)
        return np.prod(self.get_reward_factors(physics))

    @abstractmethod
    def get_reward_factors(self, physics: 'mjcf.Physics') -> Sequence[float]:
        """Reward factors for the walker, overriden in subclasses."""
        raise NotImplementedError("Subclasses should implement this.")

    def check_termination(self, physics: 'mjcf.Physics') -> bool:
        """Check termination conditions."""
        qacc = np.linalg.norm(physics.data.qacc)
        return qacc > _TERMINAL_QACC

    def action_spec(self, physics: 'mjcf.Physics'):
        """Action spec of the walker, see therein."""
        return self._walker.get_action_spec(physics)

    def name(self):
        """"Get task name."""
        return 'FruitFlyTask'

    @property
    def root_entity(self):
        return self._arena

    @property
    def walker(self):
        return self._walker

    # Define observables potentially used in child classes.

    @composer.observable
    def ref_displacement(self):
        """Reference displacement vectors in fly's egocentric reference frame,
        possibly with preview of future timesteps.
        """
        def get_ref_displacement(physics: 'mjcf.Physics'):
            fly_pos, _ = self._walker.get_pose(physics)
            ref_pos = self._ref_qpos[self._step_counter:self._step_counter +
                                     self._future_steps + 1, :3]
            return self._walker.transform_vec_to_egocentric_frame(
                physics, ref_pos - fly_pos)
        return observable.Generic(get_ref_displacement)

    @composer.observable
    def ref_root_quat(self):
        """Reference root quaternions in fly's egocentric reference frame,
        possibly with preview of future timesteps.
        """
        def get_root_quat(physics: 'mjcf.Physics'):
            ref_quat = self._ref_qpos[self._step_counter:self._step_counter +
                                      self._future_steps + 1, 3:7]
            _, fly_quat = self._walker.get_pose(physics)
            return get_dquat_local(fly_quat, ref_quat)
        return observable.Generic(get_root_quat)


class Flying(FruitFlyTask):
    """Base class for all flying tasks."""

    def __init__(
        self,
        wing_gainprm=_WING_PARAMS['gainprm'],
        wing_stiffness=_WING_PARAMS['stiffness'],
        wing_damping=_WING_PARAMS['damping'],
        fluidcoef=_WING_PARAMS['fluidcoef'],
        floor_contacts: bool = False,
        disable_legs: bool = True,
        **kwargs,
    ):
        """Base class for setting fly model configuration for flight tasks.

        Args:
            wing_gainprm: Gain parameter for wing actuators, [yaw, roll, pitch].
            wing_stiffness: Stiffness of wing joints.
            wing_damping: Damping of wing joints.
            fluidcoef: Parameters for new MuJoCo fluid model.
            floor_contacts: Whether to use collision detection with floor.
            disable_legs: Whether to retract and disable legs. This includes
                removing leg DoFs, actuators, and sensors.
            **kwargs: Arguments passed to the superclass constructor.
        """
        super().__init__(use_legs=not disable_legs,
                         use_wings=True,
                         use_mouth=False,
                         use_antennae=False,
                         physics_timestep=_FLY_PHYSICS_TIMESTEP,
                         control_timestep=_FLY_CONTROL_TIMESTEP,
                         **kwargs)

        self._up_dir = self._walker.mjcf_model.find('site',
                                                    'hover_up_dir').quat

        # Maybe disable floor contacts.
        if not floor_contacts:
            for geom in self._arena.ground_geoms:
                geom.contype = 0
                geom.conaffinity = 0

        # Set wing actuator gain.
        for i, dclass in enumerate(['yaw', 'roll', 'pitch']):
            general = self._walker.mjcf_model.find('default', dclass).general
            general.gainprm[0] = wing_gainprm[i]

        # Activate new fluid model for wings and set fluid parameters.
        for geom in self._walker.mjcf_model.find_all('geom'):
            if 'fluid' in geom.name:
                geom.fluidshape = 'ellipsoid'
                geom.fluidcoef = fluidcoef

        # Get wing joints.
        self._wing_joints = []
        for side in ['left', 'right']:
            for axis in ['yaw', 'roll', 'pitch']:
                joint = f'wing_{axis}_{side}'
                self._wing_joints.append(
                    self._walker.mjcf_model.find('joint', joint))

        # Set wing joint stiffness and damping.
        wing_default_joint = self._walker.mjcf_model.find('default',
                                                          'wing').joint
        wing_default_joint.stiffness = wing_stiffness
        wing_default_joint.damping = wing_damping

        # Exclude wing-leg collisions.
        contact = self._walker.mjcf_model.contact
        for body in self._walker.mjcf_model.find_all('body'):
            if any_substr_in_str(['coxa', 'femur', 'tibia', 'tarsus', 'claw'], 
                                 body.name):
                for wing in ['wing_left', 'wing_right']:
                    contact.add('exclude', 
                                name=f'{body.name}_{wing}', 
                                body1=body.name, body2=wing)

        # Get springref angles for retracted leg position.
        self._leg_joints = []
        self._leg_springrefs = []
        for joint in self._walker.mjcf_model.find_all('joint'):
            if any_substr_in_str(['coxa', 'femur', 'tibia', 'tarsus'], joint.name):
                springref = joint.springref or joint.dclass.joint.springref or 0.
                self._leg_joints.append(joint)
                self._leg_springrefs.append(springref)
        if not disable_legs:
            assert len(self._leg_joints) == 66  # 11 joints per leg.

        # Explicitly add/enable/disable additional flying task observables.
        self._walker.observables.thorax_height.enabled = False
        if not disable_legs:
            self._walker.observables.appendages_pos.enabled = True
            self._walker.observables.force.enabled = True
            self._walker.observables.touch.enabled = True


class Walking(FruitFlyTask):
    """Base class for all walking tasks."""

    def __init__(
        self,
        disable_wings: bool = True,
        adhesion_gain: float | None = None,
        **kwargs,
    ):
        """Base class for setting fly model configuration for walking tasks.

        Args:
            disable_wings: Whether to retract and disable wings. This includes
                removing wing DoFs, actuators, and sensors.
            adhesion_gain: Optionally, change the default adhesion actuator gain.
            **kwargs: Arguments passed to the superclass constructor.
        """

        super().__init__(use_legs=True,
                         use_wings=not disable_wings,
                         use_mouth=False,
                         use_antennae=False,
                         physics_timestep=_WALK_PHYSICS_TIMESTEP,
                         control_timestep=_WALK_CONTROL_TIMESTEP,
                         **kwargs)

        if adhesion_gain is not None:
            self._walker.mjcf_model.find(
                'default', 'adhesion_claw').adhesion.gain = adhesion_gain

        # Set floor contact params.
        for geom in self._arena.ground_geoms:
            geom.friction = (0.5, )
            geom.solref = (0.001, 1)
            geom.solimp = (0.95, 0.99, 0.01)

        # Exclude wing-leg collisions.
        contact = self._walker.mjcf_model.contact
        for body in self._walker.mjcf_model.find_all('body'):
            if any_substr_in_str(['coxa', 'femur', 'tibia', 'tarsus', 'claw'],
                                 body.name):
                for wing in ['wing_left', 'wing_right']:
                    contact.add('exclude',
                                name=f'{body.name}_{wing}',
                                body1=body.name, body2=wing)

        # Get springref angles for retracted wing position.
        self._wing_joints = []
        self._wing_springrefs = []
        for joint in self._walker.mjcf_model.find_all('joint'):
            if any_substr_in_str(['yaw', 'roll', 'pitch'], joint.name):
                springref = joint.springref or joint.dclass.joint.springref or 0.
                self._wing_joints.append(joint)
                self._wing_springrefs.append(springref)
        if not disable_wings:
            assert len(self._wing_joints) == 6

        # Explicitly add/enable/disable walking task observables.
        self._walker.observables.appendages_pos.enabled = True
        self._walker.observables.force.enabled = True
        self._walker.observables.touch.enabled = True
        self._walker.observables.self_contact.enabled = False
