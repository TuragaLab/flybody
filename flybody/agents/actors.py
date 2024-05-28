"""Acme agent implementations."""

from typing import Callable

import numpy as np

from acme import adders
from acme import core
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class DelayedFeedForwardActor(core.Actor):
    """A feed-forward actor with optional delay between observation and action.

    An actor based on a feed-forward policy which takes non-batched observations
    and outputs non-batched actions. It also allows adding experiences to replay
    and updating the weights from the policy on the learner.
    """

    def __init__(
        self,
        policy_network: snt.Module,
        adder: adders.Adder | None = None,
        variable_client: tf2_variable_utils.VariableClient | None = None,
        action_delay: int | None = None,
        observation_callback: Callable | None = None,
    ):
        """Initializes the actor.

        Args:
            policy_network: the policy to run.
            adder: the adder object to which allows to add experiences to a
                dataset/replay buffer.
            variable_client: object which allows to copy weights from the learner copy
                of the policy to the actor copy (in case they are separate).
            action_delay: number of timesteps to delay the action for.
            observation_callback: Optional callable to process observations before
                passing them to policy.
        """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._policy_network = policy_network
        self._action_delay = action_delay
        if action_delay is not None:
            self._action_queue = []
        self._observation_callback = observation_callback

    @tf.function
    def _policy(self, observation: types.NestedArray) -> types.NestedTensor:
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # Compute the policy, conditioned on the observation.
        policy = self._policy_network(batched_observation)

        # Sample from the policy if it is stochastic.
        action = policy.sample() if isinstance(
            policy, tfd.Distribution) else policy

        return action

    def select_action(self,
                      observation: types.NestedArray) -> np.ndarray:
        """Samples from the policy and returns an action."""
        if self._observation_callback is not None:
            observation = self._observation_callback(observation)
        # Pass the observation through the policy network.
        action = self._policy(observation)

        # Maybe delay action.
        if self._action_delay is not None:
            if len(self._action_queue) < self._action_delay:
                self._action_queue.append(action)
                action = 0 * action  # Return 0 while filling the initial queue.
            else:
                self._action_queue.append(action)
                action = self._action_queue.pop(0)

        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(action)

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add_first(timestep)

    def observe(self, action: types.NestedArray,
                next_timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        if self._variable_client:
            self._variable_client.update(wait)
