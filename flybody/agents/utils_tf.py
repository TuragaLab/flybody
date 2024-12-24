"""Utilities for tensorflow networks and nested data structures."""

from typing import Callable

import numpy as np
import tensorflow as tf

import acme
from acme import types
from acme.tf import utils as tf2_utils

from flybody.agents.agent_dmpo import DMPONetworks


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

    def __call__(self,
                 observation: types.NestedArray,
                 test_mode=False) -> np.ndarray:
        """Policy forward pass.

        Args:
            observation: timestep.observation
            test_mode: In test mode, instead of action, both mean and std are
                returned.

        Returns:
            Either action or mean and std.
        """
        # Add a dummy batch dimension and as a side effect convert numpy to TF,
        # batched_observation: types.NestedTensor.
        batched_observation = tf2_utils.add_batch_dim(observation)
        distribution = self._policy(batched_observation)
        if test_mode:
            return (distribution.mean()[0, :].numpy(),
                    distribution.stddev()[0, :].numpy())
        if self._sample:
            action = distribution.sample()
        else:
            action = distribution.mean()
        action = action[0, :].numpy()  # Remove batch dimension.
        return action


def restore_dmpo_networks_from_checkpoint(
    ckpt_path: str,
    network_factory: Callable,
    environment_spec: acme.specs.EnvironmentSpec) -> DMPONetworks:
    """Create DMPO networks and load their weights from checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint, e.g. path/to/checkpoints/dmpo_learner/ckpt-47
        network_factory: A callable to create DMPO networks. The architecture
            must match the network weights to be loaded from checkpoint.
        environment_spec: Env specs generated as
            environment_spec = acme.specs.make_environment_spec(env)

    Returns:
        A DMPONetworks dataclass containing the networks with loaded weights.
    """

    # This sequence replicates how networks are built in our DMPO agent.
    
    def wrapped_network_factory(action_spec):
        networks_dict = network_factory(action_spec)
        networks = DMPONetworks(
            policy_network=networks_dict.get('policy'),
            critic_network=networks_dict.get('critic'),
            observation_network=networks_dict.get('observation', tf.identity))
        return networks
    
    networks = wrapped_network_factory(environment_spec.actions)
    networks.init(environment_spec)
    
    policy_network = networks.policy_network
    critic_network = networks.critic_network
    # Make sure observation networks are snt.Module's so they have variables.
    observation_network = tf2_utils.to_sonnet_module(
        networks.observation_network)

    _checkpoint = tf.train.Checkpoint(
        target_policy=policy_network,
        target_critic=critic_network,
        target_observation=observation_network,
    )
    # For inference or reuse, we only need partial.
    status = _checkpoint.restore(ckpt_path).expect_partial()  # noqa: F841
    # This assertion will not pass because of the counter added by
    # checkpointer.save() when the checkpoint is saved.
    # status.assert_consumed()

    return networks
