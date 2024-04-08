"""Template class for walking fly tasks."""
# ruff: noqa: F821

import numpy as np

from flybody.tasks.base import Walking


class TemplateTask(Walking):
    """Template class for walking fly tasks."""

    def __init__(self, claw_friction: float = 1.0, **kwargs):
        """Template class for walking fly tasks.

        Args:
            claw_friction: Friction of claw geoms with floor.
            **kwargs: Arguments passed to the superclass constructor.
        """

        super().__init__(add_ghost=False, ghost_visible_legs=False, **kwargs)

        # Maybe do something here.

        # Maybe change default claw friction.
        if claw_friction is not None:
            self._walker.mjcf_model.find(
                'default',
                'adhesion-collision').geom.friction = (claw_friction, )

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        super().initialize_episode_mjcf(random_state)
        # Maybe do something here.

    def initialize_episode(self, physics: 'mjcf.Physics',
                           random_state: np.random.RandomState):
        """Randomly selects a starting point and set the walker."""
        super().initialize_episode(physics, random_state)
        # Maybe do something here.

    def before_step(self, physics: 'mjcf.Physics', action,
                    random_state: np.random.RandomState):
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
