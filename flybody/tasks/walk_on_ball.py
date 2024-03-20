"""Task of tethered fly walking on floating ball."""
# ruff: noqa: F821

from typing import Optional
import numpy as np

from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.utils import rewards

from flybody.tasks.base import Walking
from flybody.tasks.constants import (_TERMINAL_ANGVEL, _TERMINAL_LINVEL)


class WalkOnBall(Walking):
    """Tethered fly walking on floating ball."""

    def __init__(self, claw_friction: Optional[float] = 1.0, **kwargs):
        """Task of tethered fly walking on floating ball.

        Args:
            claw_friction: Friction of claw geoms with floor.
            **kwargs: Arguments passed to the superclass constructor.
        """

        super().__init__(add_ghost=False, ghost_visible_legs=False, **kwargs)

        # Fuse fly's thorax with world.
        self.root_entity.mjcf_model.find('attachment_frame',
                                         'walker').freejoint.remove()

        # Exclude "surprising" thorax-children collisions.
        for child in self._walker.mjcf_model.find('body',
                                                  'thorax').all_children():
            if child.tag == 'body':
                self._walker.mjcf_model.contact.add(
                    'exclude',
                    name=f'thorax_{child.name}',
                    body1='thorax',
                    body2=child.name)

        # Maybe change default claw friction.
        if claw_friction is not None:
            self._walker.mjcf_model.find(
                'default',
                'adhesion-collision').geom.friction = (claw_friction, )

        # Enable task-specific observables.
        self._walker.observables.add_observable('ball_qvel', self.ball_qvel)

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        super().initialize_episode_mjcf(random_state)
        # Maybe do something here.

    def initialize_episode(self, physics: 'mjcf.Physics',
                           random_state: np.random.RandomState):
        """Randomly selects a starting point and set the walker."""
        super().initialize_episode(physics, random_state)

    def before_step(self, physics: 'mjcf.Physics', action,
                    random_state: np.random.RandomState):
        # Maybe do something here.
        super().before_step(physics, action, random_state)

    def get_reward_factors(self, physics):
        """Returns factorized reward terms."""

        ball_qvel = physics.named.data.qvel['ball']
        target_ball_qvel = [0., -5, 0]
        qvel = rewards.tolerance(ball_qvel - target_ball_qvel,
                                 bounds=(0, 0),
                                 sigmoid='linear',
                                 margin=6,
                                 value_at_margin=0.0)
        return np.hstack(qvel)

    def check_termination(self, physics: 'mjcf.Physics') -> bool:
        """Check various termination conditions."""
        linvel = np.linalg.norm(self._walker.observables.velocimeter(physics))
        angvel = np.linalg.norm(self._walker.observables.gyro(physics))
        return (linvel > _TERMINAL_LINVEL or angvel > _TERMINAL_ANGVEL
                or super().check_termination(physics))

    @composer.observable
    def ball_qvel(self):
        """Simple observable of ball rotational velocity."""
        def get_ball_qvel(physics: 'mjcf.Physics'):
            return physics.named.data.qvel['ball']
        return observable.Generic(get_ball_qvel)
