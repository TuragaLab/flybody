"""Distributed DMPO agent implementation."""

import copy
from typing import Iterator, List, Optional, Tuple

from acme import adders
from acme import core
from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as reverb_adders
from acme.agents import agent
from acme.tf import networks as network_utils
from acme.tf import utils
from acme.tf import variable_utils
from acme.utils import counting
from acme.utils import loggers

import dataclasses
import reverb
import sonnet as snt
import tensorflow as tf

from flybody.agents import learning_dmpo
from flybody.agents.actors import DelayedFeedForwardActor


@dataclasses.dataclass
class DMPOConfig:
    """Configuration options for the DMPO agent."""
    discount: float = 0.99
    batch_size: int = 256
    prefetch_size: int = 4
    target_policy_update_period: int = 100
    target_critic_update_period: int = 100
    min_replay_size: int = 1000
    max_replay_size: int = 1000000
    samples_per_insert: Optional[float] = 32.0
    policy_loss_module: Optional[snt.Module] = None
    policy_optimizer: Optional[snt.Optimizer] = None
    critic_optimizer: Optional[snt.Optimizer] = None
    dual_optimizer: Optional[snt.Optimizer] = None
    n_step: int = 5
    num_samples: int = 20
    clipping: bool = True
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE


@dataclasses.dataclass
class DMPONetworks:
    """Structure containing the networks for DMPO."""

    policy_network: snt.Module
    critic_network: snt.Module
    observation_network: snt.Module

    def __init__(
        self,
        policy_network: snt.Module,
        critic_network: snt.Module,
        observation_network: types.TensorTransformation,
    ):
        # This method is implemented (rather than added by the dataclass decorator)
        # in order to allow observation network to be passed as an arbitrary tensor
        # transformation rather than as a snt Module.
        # TODO(mwhoffman): use Protocol rather than Module/TensorTransformation.
        self.policy_network = policy_network
        self.critic_network = critic_network
        self.observation_network = utils.to_sonnet_module(observation_network)

    def init(self, environment_spec: specs.EnvironmentSpec):
        """Initialize the networks given an environment spec."""
        # Get observation and action specs.
        act_spec = environment_spec.actions
        obs_spec = environment_spec.observations

        # Create variables for the observation net and, as a side-effect, get a
        # spec describing the embedding space.
        emb_spec = utils.create_variables(self.observation_network, [obs_spec])

        # Create variables for the policy and critic nets.
        _ = utils.create_variables(self.policy_network, [emb_spec])
        _ = utils.create_variables(self.critic_network, [emb_spec, act_spec])

    def make_policy(self, stochastic: bool = False) -> snt.Module:
        """Create a single network which evaluates the policy."""
        # Stack the observation and policy networks.
        stack = [
            self.observation_network,
            self.policy_network,
        ]

        # If a stochastic/non-greedy policy is requested, add Gaussian noise on
        # top to enable a simple form of exploration.
        # TODO(mwhoffman): Refactor this to remove it from the class.
        if stochastic:
            stack += [network_utils.StochasticSamplingHead()]

        # Return a network which sequentially evaluates everything in the stack.
        return snt.Sequential(stack)


class DMPOBuilder:
    """Builder for DMPO which constructs individual components of the agent."""

    def __init__(self, config: DMPOConfig):
        self._config = config

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""
        if self._config.samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            limiter = reverb.rate_limiters.MinSize(
                self._config.min_replay_size)

        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._config.samples_per_insert
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=error_buffer)

        replay_table = reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=reverb_adders.NStepTransitionAdder.signature(
                environment_spec))

        return [replay_table]

    def make_dataset_iterator(
        self,
        reverb_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        # The dataset provides an interface to sample from replay.
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=reverb_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size)

        return iter(dataset)  # pytype: disable=wrong-arg-types

    def make_adder(
        self,
        replay_client: reverb.Client,
    ) -> adders.Adder:
        """Create an adder which records data generated by the actor/environment."""
        return reverb_adders.NStepTransitionAdder(
            priority_fns={self._config.replay_table_name: lambda x: 1.},
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount)

    def make_actor(
        self,
        policy_network: snt.Module,
        adder: Optional[adders.Adder] = None,
        variable_source: Optional[core.VariableSource] = None,
    ):
        """Create an actor instance."""
        if variable_source:
            # Create the variable client responsible for keeping the actor up-to-date.
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={'policy': policy_network.variables},
                update_period=1000,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        else:
            variable_client = None

        # This is a modified version of actors.FeedForwardActor in Acme.
        return DelayedFeedForwardActor(
            policy_network=policy_network,
            adder=adder,
            variable_client=variable_client,
            action_delay=None,
        )

    def make_learner(
        self,
        networks: Tuple[DMPONetworks, DMPONetworks],
        dataset: Iterator[reverb.ReplaySample],
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint_enable: bool = False,
        checkpoint_max_to_keep: Optional[int] = 1,
    ):
        """Creates an instance of the learner."""
        online_networks, target_networks = networks

        # The learner updates the parameters (and initializes them).
        return learning_dmpo.DistributionalMPOLearner(
            policy_network=online_networks.policy_network,
            critic_network=online_networks.critic_network,
            observation_network=online_networks.observation_network,
            target_policy_network=target_networks.policy_network,
            target_critic_network=target_networks.critic_network,
            target_observation_network=target_networks.observation_network,
            policy_loss_module=self._config.policy_loss_module,
            policy_optimizer=self._config.policy_optimizer,
            critic_optimizer=self._config.critic_optimizer,
            dual_optimizer=self._config.dual_optimizer,
            clipping=self._config.clipping,
            discount=self._config.discount,
            num_samples=self._config.num_samples,
            target_policy_update_period=self._config.
            target_policy_update_period,
            target_critic_update_period=self._config.
            target_critic_update_period,
            dataset=dataset,
            logger=logger,
            counter=counter,
            checkpoint_enable=checkpoint_enable,
            checkpoint_max_to_keep=checkpoint_max_to_keep)


class DMPO(agent.Agent):
    """DMPO Agent.
    This implements a single-process DMPO agent. This is an actor-critic algorithm
    that generates data via a behavior policy, inserts N-step transitions into
    a replay buffer, and periodically updates the policy (and as a result the
    behavior) by sampling uniformly from this buffer.
    """

    def __init__(
            self,
            environment_spec: specs.EnvironmentSpec,
            policy_network: snt.Module,
            critic_network: snt.Module,
            observation_network: types.TensorTransformation = tf.identity,
            discount: float = 0.99,
            batch_size: int = 256,
            prefetch_size: int = 4,
            target_policy_update_period: int = 100,
            target_critic_update_period: int = 100,
            min_replay_size: int = 1000,
            max_replay_size: int = 1000000,
            samples_per_insert: float = 32.0,
            policy_loss_module: Optional[snt.Module] = None,
            policy_optimizer: Optional[snt.Optimizer] = None,
            critic_optimizer: Optional[snt.Optimizer] = None,
            n_step: int = 5,
            num_samples: int = 20,
            clipping: bool = True,
            logger: Optional[loggers.Logger] = None,
            counter: Optional[counting.Counter] = None,
            checkpoint_enable: bool = True,
            replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE):
        """Initialize the agent.
        Args:
            environment_spec: description of the actions, observations, etc.
            policy_network: the online (optimized) policy.
            critic_network: the online critic.
            observation_network: optional network to transform the observations
                before they are fed into any network.
            discount: discount to use for TD updates.
            batch_size: batch size for updates.
            prefetch_size: size to prefetch from replay.
            target_policy_update_period: number of updates to perform before
                updating the target policy network.
            target_critic_update_period: number of updates to perform before
                updating the target critic network.
            min_replay_size: minimum replay size before updating.
            max_replay_size: maximum replay size.
            samples_per_insert: number of samples to take from replay for every
                insert that is made.
            policy_loss_module: configured MPO loss function for the policy
                optimization; defaults to sensible values on the control suite.
                See `acme/tf/losses/mpo.py` for more details.
            policy_optimizer: optimizer to be used on the policy.
            critic_optimizer: optimizer to be used on the critic.
            n_step: number of steps to squash into a single transition.
            num_samples: number of actions to sample when doing a Monte Carlo
                integration with respect to the policy.
            clipping: whether to clip gradients by global norm.
            logger: logging object used to write to logs.
            counter: counter object used to keep track of steps.
            checkpoint_enable: boolean indicating whether to checkpoint the learner.
            replay_table_name: string indicating what name to give the replay table.
        """
        # Create the Builder object which will internally create agent components.
        builder = DMPOBuilder(
            # TODO(mwhoffman): pass the config dataclass in directly.
            # TODO(mwhoffman): use the limiter rather than the workaround below.
            # Right now this modifies min_replay_size and samples_per_insert so that
            # they are not controlled by a limiter and are instead handled by the
            # Agent base class (the above TODO directly references this behavior).
            DMPOConfig(
                discount=discount,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_policy_update_period=target_policy_update_period,
                target_critic_update_period=target_critic_update_period,
                min_replay_size=min_replay_size,
                max_replay_size=max_replay_size,
                samples_per_insert=samples_per_insert,
                policy_loss_module=policy_loss_module,
                policy_optimizer=policy_optimizer,
                critic_optimizer=critic_optimizer,
                n_step=n_step,
                num_samples=num_samples,
                clipping=True,
                replay_table_name=reverb_adders.DEFAULT_PRIORITY_TABLE,
            ))

        # Create networks
        # TODO(mwhoffman): pass the network dataclass in directly.
        online_networks = DMPONetworks(policy_network=policy_network,
                                       critic_network=critic_network,
                                       observation_network=observation_network)

        # Target networks are just a copy of the online networks.
        target_networks = copy.deepcopy(online_networks)

        # Initialize the networks.
        online_networks.init(environment_spec)
        target_networks.init(environment_spec)

        # TODO(mwhoffman): either make this Dataclass or pass only one struct.
        # The network struct passed to make_learner is just a tuple for the
        # time-being (for backwards compatibility).
        networks = (online_networks, target_networks)

        # Create the behavior policy.
        behavior_network = snt.Sequential([
            online_networks.observation_network,
            online_networks.policy_network,
            network_utils.StochasticSamplingHead(),
        ])

        # Create the replay server and grab its address.
        replay_tables = builder.make_replay_tables(environment_spec)
        replay_server = reverb.Server(replay_tables, port=None)
        replay_client = reverb.Client(f'localhost:{replay_server.port}')

        # Create actor, dataset, and learner for generating, storing, and consuming
        # data respectively.
        adder = builder.make_adder(replay_client)
        actor = builder.make_actor(behavior_network, adder)
        dataset = builder.make_dataset_iterator(replay_client)
        learner = builder.make_learner(networks, dataset, counter, logger,
                                       checkpoint_enable)

        super().__init__(actor=actor,
                         learner=learner,
                         min_observations=max(batch_size, min_replay_size),
                         observations_per_step=float(batch_size) /
                         samples_per_insert)

        # Save the replay so we don't garbage collect it.
        self._replay_server = replay_server
