"""Classes for PPO agent distributed with Ray."""

from typing import Callable, Dict, List
import dataclasses
import logging
import os

import ray
import numpy as np

import tensorflow as tf
import sonnet as snt

import acme
from acme import specs
from acme.utils import counting
from acme.utils import loggers
from acme.tf import variable_utils
from acme.tf import networks as network_utils
from acme.tf import utils as tf2_utils

from flybody.agents.learning_ppo import PPOLearner
from flybody.agents import agent_ppo

import tensorflow_probability as tfp
tfd = tfp.distributions


@dataclasses.dataclass
class PPOConfig:
    """Configuration for distributed PPO with Ray."""
    num_actors: int = 32
    trajectory_length: int = 2048
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    ppo_epochs: int = 4
    num_minibatches: int = 4
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    clip_value_loss: bool = True
    learning_rate: float = 3e-4
    optimizer: snt.Optimizer | None = None
    actor_update_period: int = 1  # Update weights every collection round.
    logger: loggers.base.Logger | None = None
    log_every: float = 60.  # Seconds.
    logger_save_csv_data: bool = False
    checkpoint_to_load: str | None = None
    checkpoint_max_to_keep: int | None = 1
    checkpoint_directory: str | None = '~/ray-ckpts/'
    time_delta_minutes: float = 30
    print_fn: Callable = logging.info
    userdata: dict | None = None


class Learner(PPOLearner):
    """The Learning part of the PPO agent, Ray-compatible."""

    def __init__(
        self,
        counter: counting.Counter,
        environment_spec: specs.EnvironmentSpec,
        ppo_config: PPOConfig,
        network_factory,
        label='learner',
    ):
        self._config = ppo_config
        self._label = label

        def wrapped_network_factory(action_spec):
            networks_dict = network_factory(action_spec)
            networks = agent_ppo.PPONetworks(
                policy_network=networks_dict.get('policy'),
                value_network=networks_dict.get('value'),
                observation_network=networks_dict.get('observation',
                                                      tf.identity))
            return networks

        # Create networks.
        networks = wrapped_network_factory(environment_spec.actions)
        networks.init(environment_spec)

        counter = counting.Counter(parent=counter, prefix=label)
        if self._config.logger is None:
            logger = loggers.make_default_logger(
                label=label,
                time_delta=self._config.log_every,
                steps_key=f'{label}_steps',
                print_fn=self._config.print_fn,
                save_data=self._config.logger_save_csv_data)
        else:
            if self._config.userdata and 'logger_kwargs' in self._config.userdata:
                logger_kwargs = self._config.userdata['logger_kwargs']
            else:
                logger_kwargs = {}
            logger = self._config.logger(
                label=label,
                time_delta=self._config.log_every,
                **logger_kwargs)

        checkpoint_enable = self._config.checkpoint_directory is not None

        PPOLearner.__init__(
            self,
            policy_network=networks.policy_network,
            value_network=networks.value_network,
            observation_network=networks.observation_network,
            discount=self._config.discount,
            gae_lambda=self._config.gae_lambda,
            clip_ratio=self._config.clip_ratio,
            value_loss_coeff=self._config.value_loss_coeff,
            entropy_coeff=self._config.entropy_coeff,
            ppo_epochs=self._config.ppo_epochs,
            num_minibatches=self._config.num_minibatches,
            max_grad_norm=self._config.max_grad_norm,
            normalize_advantage=self._config.normalize_advantage,
            clip_value_loss=self._config.clip_value_loss,
            optimizer=self._config.optimizer,
            counter=counter,
            logger=logger,
            checkpoint_enable=checkpoint_enable,
            checkpoint_max_to_keep=self._config.checkpoint_max_to_keep,
            directory=self._config.checkpoint_directory,
            checkpoint_to_load=self._config.checkpoint_to_load,
            time_delta_minutes=self._config.time_delta_minutes,
        )

    def run(self, trajectories: List[Dict[str, np.ndarray]]):
        """Run PPO update on collected trajectories, then log."""
        self.update(trajectories)
        self.step()

    def isready(self):
        """Dummy method to check if learner is ready."""
        pass

    def get_checkpoint_dir(self):
        """Return Checkpointer and Snapshotter directories, if any."""
        if self._checkpointer is not None:
            return self._checkpointer._checkpoint_dir, self._snapshotter.directory
        return None, None


class Actor:
    """PPO Actor: collects trajectories by running the environment."""

    def __init__(
        self,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
        network_factory,
        environment_factory,
        ppo_config: PPOConfig,
        label='actor',
        ray_head_node_ip: str | None = None,
        egl_device_id_head_node: list | None = None,
        egl_device_id_worker_node: list | None = None,
    ):
        # Maybe adjust EGL_DEVICE_ID environment variable.
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

        self._config = ppo_config

        # Create the environment.
        self._environment = environment_factory(False)
        environment_spec = specs.make_environment_spec(self._environment)

        # Create networks.
        def wrapped_network_factory(action_spec):
            networks_dict = network_factory(action_spec)
            networks = agent_ppo.PPONetworks(
                policy_network=networks_dict.get('policy'),
                value_network=networks_dict.get('value'),
                observation_network=networks_dict.get('observation',
                                                      tf.identity))
            return networks

        self._networks = wrapped_network_factory(environment_spec.actions)
        self._networks.init(environment_spec)

        # Build full policy and value forward pass networks.
        # observation_network (batch_concat) handles dict->flat conversion.
        self._observation_network = self._networks.observation_network
        self._policy_network = snt.Sequential([
            self._networks.observation_network,
            self._networks.policy_network,
        ])
        self._value_network = snt.Sequential([
            self._networks.observation_network,
            self._networks.value_network,
        ])

        # Variable client to pull updated weights from learner.
        self._variable_client = variable_utils.VariableClient(
            client=variable_source,
            variables={
                'policy': self._policy_network.variables,
                'value': self._value_network.variables,
            },
            update_period=self._config.actor_update_period,
        )
        self._variable_client.update_and_wait()

        # Counter and logger.
        self._counter = counting.Counter(parent=counter, prefix='actor')
        if self._config.logger is None:
            self._logger = loggers.make_default_logger(
                label=label,
                save_data=False,
                time_delta=self._config.log_every,
                steps_key='actor_steps',
                print_fn=self._config.print_fn,
            )
        else:
            if self._config.userdata and 'logger_kwargs' in self._config.userdata:
                logger_kwargs = self._config.userdata['logger_kwargs']
            else:
                logger_kwargs = {}
            self._logger = self._config.logger(
                label=label,
                time_delta=self._config.log_every,
                **logger_kwargs)

        # Reset environment.
        self._timestep = self._environment.reset()

    def isready(self):
        """Dummy method to check if actor is ready."""
        pass

    @tf.function
    def _forward(self, observation):
        """Run policy and value networks on a single observation."""
        batched_obs = tf2_utils.add_batch_dim(observation)
        distribution = self._policy_network(batched_obs)
        value = self._value_network(batched_obs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob, value

    def _flatten_obs(self, observation) -> np.ndarray:
        """Flatten dict observation to a single numpy array."""
        batched_obs = tf2_utils.add_batch_dim(observation)
        flat = self._observation_network(batched_obs)
        return tf2_utils.to_numpy_squeeze(flat)

    def collect(self, num_steps: int) -> Dict[str, np.ndarray]:
        """Collect a trajectory of fixed length.

        Args:
            num_steps: Number of environment steps to collect.

        Returns:
            Dictionary with trajectory data:
                'observations': [T, obs_dim]
                'actions': [T, act_dim]
                'rewards': [T]
                'dones': [T]
                'log_probs': [T]
                'values': [T]
                'last_value': scalar
        """
        # Pull latest weights from learner.
        self._variable_client.update_and_wait()

        observations = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        episode_returns = []
        current_episode_return = 0.0

        for _ in range(num_steps):
            obs = self._timestep.observation
            action, log_prob, value = self._forward(obs)

            # Remove batch dimensions.
            action_np = tf2_utils.to_numpy_squeeze(action)
            log_prob_np = float(tf.squeeze(log_prob))
            value_np = float(tf.squeeze(value))

            # Flatten dict observation to a single array for the learner.
            observations.append(self._flatten_obs(obs))
            actions.append(action_np)
            log_probs.append(log_prob_np)
            values.append(value_np)

            # Step environment.
            self._timestep = self._environment.step(action_np)

            rewards.append(float(self._timestep.reward))
            done = self._timestep.last()
            dones.append(float(done))

            current_episode_return += float(self._timestep.reward)
            if done:
                episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                self._timestep = self._environment.reset()

        # Bootstrap value for the last state.
        last_obs = self._timestep.observation
        _, _, last_value = self._forward(last_obs)
        last_value_np = float(tf.squeeze(last_value))

        # Always increment step count; use generic names so the 'actor'
        # prefix on self._counter produces the expected 'actor_steps' etc.
        counts = dict(steps=num_steps)
        if episode_returns:
            counts['episodes'] = len(episode_returns)
        self._counter.increment(**counts)

        # Log episode returns if any episodes completed.
        if episode_returns:
            self._logger.write({
                'episode_return_mean': np.mean(episode_returns),
                'episode_return_min': np.min(episode_returns),
                'episode_return_max': np.max(episode_returns),
                'episodes_completed': len(episode_returns),
            })

        # Observations are already flattened numpy arrays.
        return {
            'observations': np.stack(observations, axis=0).astype(np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32),
            'dones': np.array(dones, dtype=np.float32),
            'log_probs': np.array(log_probs, dtype=np.float32),
            'values': np.array(values, dtype=np.float32),
            'last_value': last_value_np,
        }


class Evaluator:
    """PPO Evaluator: runs evaluation episodes with mean policy."""

    def __init__(
        self,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
        network_factory,
        environment_factory,
        ppo_config: PPOConfig,
        label='evaluator',
    ):
        self._config = ppo_config

        # Create the environment.
        self._environment = environment_factory(True)
        environment_spec = specs.make_environment_spec(self._environment)

        # Create networks.
        def wrapped_network_factory(action_spec):
            networks_dict = network_factory(action_spec)
            networks = agent_ppo.PPONetworks(
                policy_network=networks_dict.get('policy'),
                value_network=networks_dict.get('value'),
                observation_network=networks_dict.get('observation',
                                                      tf.identity))
            return networks

        self._networks = wrapped_network_factory(environment_spec.actions)
        self._networks.init(environment_spec)

        # Build policy network (use mean, not sample).
        self._policy_network = snt.Sequential([
            self._networks.observation_network,
            self._networks.policy_network,
            network_utils.StochasticMeanHead(),
        ])

        # Variable client.
        self._variable_client = variable_utils.VariableClient(
            client=variable_source,
            variables={'policy': self._policy_network.variables},
            update_period=1,
        )
        self._variable_client.update_and_wait()

        # Counter and logger.
        self._counter = counting.Counter(parent=counter, prefix='evaluator')
        if self._config.logger is None:
            self._logger = loggers.make_default_logger(
                label=label,
                save_data=self._config.logger_save_csv_data,
                time_delta=self._config.log_every,
                steps_key='evaluator_steps',
                print_fn=self._config.print_fn,
            )
        else:
            if self._config.userdata and 'logger_kwargs' in self._config.userdata:
                logger_kwargs = self._config.userdata['logger_kwargs']
            else:
                logger_kwargs = {}
            self._logger = self._config.logger(
                label=label,
                time_delta=self._config.log_every,
                **logger_kwargs)

    def isready(self):
        """Dummy method to check if evaluator is ready."""
        pass

    @tf.function
    def _policy(self, observation):
        batched_obs = tf2_utils.add_batch_dim(observation)
        action = self._policy_network(batched_obs)
        return action

    def run(self):
        """Run evaluation episodes indefinitely."""
        while True:
            self._variable_client.update_and_wait()
            timestep = self._environment.reset()
            episode_return = 0.0
            episode_steps = 0

            while not timestep.last():
                action = self._policy(timestep.observation)
                action = tf2_utils.to_numpy_squeeze(action)
                timestep = self._environment.step(action)
                episode_return += float(timestep.reward)
                episode_steps += 1

            counts = self._counter.increment(
                evaluator_steps=episode_steps, evaluator_episodes=1)
            self._logger.write({
                'episode_return': episode_return,
                'episode_length': episode_steps,
                **counts,
            })
