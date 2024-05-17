"""Template class for walking fly tasks."""
# ruff: noqa: F821

from typing import Callable

from dm_control import mujoco
import numpy as np

from flybody.tasks.base import Walking


class TemplateTask(Walking):
    """Template class for walking fly tasks."""

    def __init__(self,
                 claw_friction: float = 1.0,
                 mjcb_control: Callable | None = None,
                 action_corruptor: Callable | None = None,
                 **kwargs):
        """Template class for walking fly tasks.

        Args:
            claw_friction: Friction of claw geoms with floor.
            mjcb_control: Optional MuJoCo control callback, a callable with
                arguments (model, data). For more information, see
                https://mujoco.readthedocs.io/en/stable/APIreference/APIglobals.html#mjcb-control
            action_corruptor (optional): A callable which takes an action as an
                argument, modifies it, and returns it. An example use case for
                this is to add random noise to the action.
            **kwargs: Arguments passed to the superclass constructor.
        """

        self._mjcb_control = mjcb_control
        self._action_corruptor = action_corruptor
        super().__init__(add_ghost=False, ghost_visible_legs=False, **kwargs)

        # Maybe do something here.

        # Maybe change default claw friction.
        if claw_friction is not None:
            self._walker.mjcf_model.find(
                'default',
                'adhesion-collision').geom.friction = (claw_friction, )

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        """Modifies the MJCF model of this task before the next episode begins."""
        # Reset control callback, if any, before model compilation.
        mujoco.set_mjcb_control(None)
        super().initialize_episode_mjcf(random_state)
        # Maybe do something here.

    def after_compile(self, physics, random_state):
        """A callback which is executed after the Mujoco Physics is recompiled."""
        del random_state  # Unused.
        assert physics.legacy_step
        # Restore control callback, if any.
        mujoco.set_mjcb_control(self._mjcb_control)

    def initialize_episode(self, physics: 'mjcf.Physics',
                           random_state: np.random.RandomState):
        """Modifies the physics state before the next episode begins."""
        super().initialize_episode(physics, random_state)
        if self._mjcb_control is not None:
            self._mjcb_control.reset()
        # Maybe do something here.

    def before_step(self, physics: 'mjcf.Physics', action,
                    random_state: np.random.RandomState):
        """A callback which is executed before an agent control step."""
        if self._action_corruptor is not None:
            action = self._action_corruptor(action, random_state)
        # Maybe do something here.
        super().before_step(physics, action, random_state)

    def get_reward_factors(self, physics):
        """Returns factorized reward terms."""
        # Calculate reward factors here.
        return (1,)

    def check_termination(self, physics: 'mjcf.Physics') -> bool:
        """Check various termination conditions."""
        # Maybe add some termination conditions.
        should_terminate = False
        return should_terminate or super().check_termination(physics)
