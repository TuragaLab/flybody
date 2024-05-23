"""Classes for DMPO agent distributed with Ray."""

from typing import Iterator, Callable
import socket
import dataclasses
import copy
import logging
import os

import ray
import numpy as np

import tensorflow as tf
import reverb
import sonnet as snt

import acme
from acme import core
from acme import specs
from acme import datasets
from acme import adders
from acme.utils import counting
from acme.utils import loggers
from acme.tf import variable_utils
from acme.tf import networks as network_utils
from acme.adders import reverb as reverb_adders

from flybody.agents.learning_dmpo import DistributionalMPOLearner
from flybody.agents import agent_dmpo
from flybody.agents.actors import DelayedFeedForwardActor


@dataclasses.dataclass
class DMPOConfig:
    num_actors: int = 32
    batch_size: int = 256
    prefetch_size: int = 4
    min_replay_size: int = 10_000
    max_replay_size: int = 4_000_000
    samples_per_insert: float = 32.  # None: limiter = reverb.rate_limiters.MinSize()
    n_step: int = 5
    num_samples: int = 20
    num_learner_steps: int = 100
    clipping: bool = True
    discount: float = 0.99
    policy_loss_module: snt.Module | None = None
    policy_optimizer: snt.Optimizer | None = None
    critic_optimizer: snt.Optimizer | None = None
    dual_optimizer: snt.Optimizer | None = None
    target_policy_update_period: int = 101
    target_critic_update_period: int = 107
    actor_update_period: int = 1000
    logger: loggers.base.Logger | None = None
    log_every: float = 60.  # Seconds.
    logger_save_csv_data: bool = False
    checkpoint_to_load: str | None = None  # Path to checkpoint.
    checkpoint_max_to_keep: int | None = 1  # None: keep all checkpoints.
    checkpoint_directory: str | None = '~/ray-ckpts/'  # None: no checkpointing.
    time_delta_minutes: float = 30
    terminal: str = 'current_terminal'
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE
    print_fn: Callable = logging.info
    userdata: dict | None = None
    actor_observation_callback: Callable | None = None


class ReplayServer():
    """Reverb replay server, can be used with DMPO agent."""

    def __init__(self, config: DMPOConfig,
                 environment_spec: specs.EnvironmentSpec):
        """Spawn a Reverb server with experience replay tables."""

        self._config = config
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

        replay_buffer = reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=reverb_adders.NStepTransitionAdder.signature(
                environment_spec))

        self._replay_server = reverb.Server(tables=[replay_buffer], port=None)
        # Get hostname and port of the server.
        hostname = socket.gethostname()
        port = self._replay_server.port
        self._replay_server_address = f'{hostname}:{port}'

    def get_server_address(self):
        return self._replay_server_address


class Learner(DistributionalMPOLearner):
    """The Learning part of the DMPO agent."""

    def __init__(
        self,
        replay_server_address: str,
        counter: counting.Counter,
        environment_spec: specs.EnvironmentSpec,
        dmpo_config,
        network_factory,
        label='learner',
    ):

        self._config = dmpo_config
        self._reverb_client = reverb.Client(replay_server_address)
        self._label = label

        def wrapped_network_factory(action_spec):
            networks_dict = network_factory(action_spec)
            networks = agent_dmpo.DMPONetworks(
                policy_network=networks_dict.get('policy'),
                critic_network=networks_dict.get('critic'),
                observation_network=networks_dict.get('observation',
                                                      tf.identity))
            return networks

        # Create the networks to optimize (online) and target networks.
        online_networks = wrapped_network_factory(environment_spec.actions)
        target_networks = copy.deepcopy(online_networks)
        # Initialize the networks.
        online_networks.init(environment_spec)
        target_networks.init(environment_spec)

        dataset = self._make_dataset_iterator(self._reverb_client)
        counter = counting.Counter(parent=counter, prefix=label)
        if self._config.logger is None:
            logger = loggers.make_default_logger(
                label=label,
                time_delta=self._config.log_every,
                steps_key=f'{label}_steps',
                print_fn=self._config.print_fn,
                save_data=self._config.logger_save_csv_data)
        else:
            if 'logger_kwargs' in self._config.userdata:
                logger_kwargs = self._config.userdata['logger_kwargs']
            else:
                logger_kwargs = {}
            logger = self._config.logger(
                label=label,
                time_delta=self._config.log_every,
                **logger_kwargs)

        # Maybe checkpoint and snapshot the learner (saved in ~/acme/).
        checkpoint_enable = self._config.checkpoint_directory is not None

        # Have to call superclass constructor in this way.
        # Solved with Ray issue:  https://github.com/ray-project/ray/issues/449
        DistributionalMPOLearner.__init__(
            self,
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
            checkpoint_max_to_keep=self._config.checkpoint_max_to_keep,
            directory=self._config.checkpoint_directory,
            checkpoint_to_load=self._config.checkpoint_to_load,
            time_delta_minutes=self._config.time_delta_minutes)

    def _step(self):
        # Workaround to access _step in DistributionalMPOLearner:
        # @tf.function
        # def _step(self)
        #    ...
        return DistributionalMPOLearner._step(self)

    def run(self, num_steps=None):
        del num_steps  # Not used.
        # Run fixed number of learning steps and return control to have a chance
        # to process calls to `get_variables`.
        for _ in range(self._config.num_learner_steps):
            self.step()

    def isready(self):
        """Dummy method to check if learner is ready."""
        pass

    def get_checkpoint_dir(self):
        """Return Checkpointer and Snapshotter directories, if any."""
        if self._checkpointer is not None:
            return self._checkpointer._checkpoint_dir, self._snapshotter.directory
        return None, None

    def _make_dataset_iterator(
        self,
        reverb_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        # The dataset provides an interface to sample from replay.
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=reverb_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size,
        )
        return iter(dataset)


class EnvironmentLoop(acme.EnvironmentLoop):
    """Actor and Evaluator class."""

    def __init__(
            self,
            replay_server_address: str,
            variable_source: acme.VariableSource,
            counter: counting.Counter,
            network_factory,
            environment_factory,
            dmpo_config,
            actor_or_evaluator='actor',
            label=None,
            ray_head_node_ip: str | None = None,
            egl_device_id_head_node: list | None = None,
            egl_device_id_worker_node: list | None = None,
    ):
        """The actor process."""

        # Maybe adjust EGL_DEVICE_ID environment variable internally in actor.
        if ray_head_node_ip is not None:
            current_node_id = ray.get_runtime_context().node_id.hex()
            running_on_head_node = False
            for node in ray.nodes():
                if (node['NodeID'] == current_node_id
                        and node['NodeManagerAddress'] == ray_head_node_ip):
                    running_on_head_node = True
                    break
            if running_on_head_node:
                egl_device_id = np.random.choice(egl_device_id_head_node)
            else:
                egl_device_id = np.random.choice(egl_device_id_worker_node)
            os.environ['MUJOCO_EGL_DEVICE_ID'] = str(egl_device_id)

        assert actor_or_evaluator in ['actor', 'evaluator']
        if actor_or_evaluator == 'evaluator':
            del replay_server_address
        else:
            self._reverb_client = reverb.Client(replay_server_address)

        self._config = dmpo_config
        label = label or actor_or_evaluator

        # Create the environment.
        environment = environment_factory(actor_or_evaluator == 'evaluator')
        environment_spec = specs.make_environment_spec(environment)

        def wrapped_network_factory(action_spec):
            networks_dict = network_factory(action_spec)
            networks = agent_dmpo.DMPONetworks(
                policy_network=networks_dict.get('policy'),
                critic_network=networks_dict.get('critic'),
                observation_network=networks_dict.get('observation',
                                                      tf.identity))
            return networks

        # Create the policy network, adder, ...
        networks = wrapped_network_factory(environment_spec.actions)
        networks.init(environment_spec)

        if actor_or_evaluator == 'actor':
            # Actor: sample from policy_network distribution.
            policy_network = snt.Sequential([
                networks.observation_network,
                networks.policy_network,
                network_utils.StochasticSamplingHead(),
            ])
            adder = self._make_adder(self._reverb_client)
            save_data = False

        elif actor_or_evaluator == 'evaluator':
            # Evaluator: get mean from policy_network distribution.
            policy_network = snt.Sequential([
                networks.observation_network,
                networks.policy_network,
                network_utils.StochasticMeanHead(),
            ])
            adder = None
            save_data = self._config.logger_save_csv_data

        # Create the agent.
        actor = self._make_actor(
            policy_network=policy_network,
            adder=adder,
            variable_source=variable_source,
            observation_callback=self._config.actor_observation_callback)

        # Create logger and counter; actors will not spam bigtable.
        counter = counting.Counter(parent=counter, prefix=actor_or_evaluator)
        if self._config.logger is None:
            logger = loggers.make_default_logger(
                label=label,
                save_data=save_data,
                time_delta=self._config.log_every,
                steps_key=actor_or_evaluator + '_steps',
                print_fn=self._config.print_fn,
            )
        else:
            if 'logger_kwargs' in self._config.userdata:
                logger_kwargs = self._config.userdata['logger_kwargs']
            else:
                logger_kwargs = {}
            logger = self._config.logger(
                label=label,
                time_delta=self._config.log_every,
                **logger_kwargs)

        super().__init__(environment, actor, counter, logger)

    def isready(self):
        """Dummy method to check if actor is ready."""
        pass

    def _make_actor(
        self,
        policy_network: snt.Module,
        adder: adders.Adder | None = None,
        variable_source: core.VariableSource | None = None,
        observation_callback: Callable | None = None,
    ):
        """Create an actor instance."""
        if variable_source:
            # Create the variable client responsible for keeping the actor up-to-date.
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={'policy': policy_network.variables},
                update_period=self._config.
                actor_update_period,  # was: hard-coded 1000,
            )
            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()
        else:
            variable_client = None

        # This is a modified version of actors.FeedForwardActor in Acme.
        return DelayedFeedForwardActor(policy_network=policy_network,
                                       adder=adder,
                                       variable_client=variable_client,
                                       action_delay=None,
                                       observation_callback=observation_callback)

    def _make_adder(self, replay_client: reverb.Client) -> adders.Adder:
        """Create an adder which records data generated by the actor/environment."""
        return reverb_adders.NStepTransitionAdder(
            priority_fns={self._config.replay_table_name: lambda x: 1.},
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount)
