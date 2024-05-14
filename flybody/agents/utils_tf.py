"""Utilities for tensorflow networks and nested data structures."""

import numpy as np
from acme import types
from acme.tf import utils as tf2_utils


class TestPolicyWrapper():
    """At test time, wraps policy to work with non-batched observations.
    Works with distributional policies, e.g. trained with the DMPO agent."""

    def __init__(self, policy, sample=False):
        """
        Args:
            policy: Test policy, e.g. trained policy loaded as 
                policy = tf.saved_model.load('path/to/snapshot').
            sample: Whether to return sample or mean of the distribution.
        """
        self._policy = policy
        self._sample = sample

    def __call__(self, observation: types.NestedArray) -> np.ndarray:
        # Add a dummy batch dimension and as a side effect convert numpy to TF,
        # batched_observation: types.NestedTensor.
        batched_observation = tf2_utils.add_batch_dim(observation)
        distribution = self._policy(batched_observation)
        if self._sample:
            action = distribution.sample()
        else:
            action = distribution.mean()
        action = action[0, :].numpy()  # Remove batch dimension.
        return action
