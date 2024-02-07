"""Template class for walking fly tasks."""

from typing import Optional
import numpy as np

from flybody.tasks.base import Walking
from flybody.tasks.constants import (
  _TERMINAL_ANGVEL, 
  _TERMINAL_LINVEL)


class TemplateTask(Walking):
  """Template class for walking fly tasks."""

  def __init__(self,
               claw_friction: Optional[float] = 1.0,
               **kwargs):
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
        'default', 'adhesion-collision').geom.friction = (claw_friction,)

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
    linvel = np.linalg.norm(
        self._walker.observables.velocimeter(physics))
    angvel = np.linalg.norm(self._walker.observables.gyro(physics))
    return (linvel > _TERMINAL_LINVEL
            or angvel > _TERMINAL_ANGVEL
            or super().check_termination(physics))
