"""PPO agent configuration and networks."""

from typing import Optional

from acme import specs
from acme import types
from acme.tf import utils
from acme.tf import networks as network_utils

import dataclasses
import sonnet as snt


@dataclasses.dataclass
class PPOConfig:
    """Configuration options for the PPO agent."""
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    ppo_epochs: int = 4
    num_minibatches: int = 4
    max_grad_norm: float = 0.5
    trajectory_length: int = 2048
    optimizer: Optional[snt.Optimizer] = None
    learning_rate: float = 3e-4
    clipping: bool = True
    normalize_advantage: bool = True
    clip_value_loss: bool = True


@dataclasses.dataclass
class PPONetworks:
    """Structure containing the networks for PPO."""

    policy_network: snt.Module
    value_network: snt.Module
    observation_network: snt.Module

    def __init__(
        self,
        policy_network: snt.Module,
        value_network: snt.Module,
        observation_network: types.TensorTransformation,
    ):
        self.policy_network = policy_network
        self.value_network = value_network
        self.observation_network = utils.to_sonnet_module(observation_network)

    def init(self, environment_spec: specs.EnvironmentSpec):
        """Initialize the networks given an environment spec."""
        obs_spec = environment_spec.observations

        # Create variables for the observation net and get embedding spec.
        emb_spec = utils.create_variables(self.observation_network, [obs_spec])

        # Create variables for the policy and value nets.
        _ = utils.create_variables(self.policy_network, [emb_spec])
        _ = utils.create_variables(self.value_network, [emb_spec])

    def make_policy(self, stochastic: bool = False) -> snt.Module:
        """Create a single network which evaluates the policy."""
        stack = [
            self.observation_network,
            self.policy_network,
        ]
        if stochastic:
            stack += [network_utils.StochasticSamplingHead()]
        return snt.Sequential(stack)
