"""PPO learner implementation."""

import time
from typing import Dict, List, Optional

import acme
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: Rewards, shape [T].
        values: Value estimates, shape [T].
        dones: Done flags, shape [T].
        last_value: Value estimate for the state after the last step.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.

    Returns:
        advantages: GAE advantages, shape [T].
        returns: Discounted returns (advantages + values), shape [T].
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae = (
            delta + gamma * gae_lambda * next_non_terminal * last_gae
        )

    returns = advantages + values
    return advantages, returns


class PPOLearner(acme.Learner):
    """PPO learner with clipped surrogate objective."""

    def __init__(
        self,
        policy_network: snt.Module,
        value_network: snt.Module,
        observation_network: snt.Module,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        ppo_epochs: int = 4,
        num_minibatches: int = 4,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
        clip_value_loss: bool = True,
        optimizer: Optional[snt.Optimizer] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint_enable: bool = True,
        checkpoint_max_to_keep: Optional[int] = 1,
        directory: str | None = '~/acme/',
        checkpoint_to_load: Optional[str] = None,
        time_delta_minutes: float = 30.,
    ):
        self._policy_network = policy_network
        self._value_network = value_network
        self._observation_network = tf2_utils.to_sonnet_module(
            observation_network)

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger('learner')

        # PPO hyperparameters.
        self._discount = discount
        self._gae_lambda = gae_lambda
        self._clip_ratio = clip_ratio
        self._value_loss_coeff = value_loss_coeff
        self._entropy_coeff = entropy_coeff
        self._ppo_epochs = ppo_epochs
        self._num_minibatches = num_minibatches
        self._max_grad_norm = max_grad_norm
        self._normalize_advantage = normalize_advantage
        self._clip_value_loss = clip_value_loss

        # Optimizer.
        self._optimizer = optimizer or snt.optimizers.Adam(3e-4)

        # All trainable variables for PPO.
        self._all_variables = (
            self._observation_network.trainable_variables +
            self._policy_network.trainable_variables +
            self._value_network.trainable_variables
        )

        # Expose the variables for actors to pull.
        policy_network_to_expose = snt.Sequential(
            [self._observation_network, self._policy_network])
        value_network_to_expose = snt.Sequential(
            [self._observation_network, self._value_network])
        self._variables = {
            'policy': policy_network_to_expose.variables,
            'value': value_network_to_expose.variables,
        }

        # Checkpointing.
        self._checkpointer = None
        self._snapshotter = None

        if checkpoint_enable:
            self._checkpointer = tf2_savers.Checkpointer(
                subdirectory='ppo_learner',
                objects_to_save={
                    'counter': self._counter,
                    'policy': self._policy_network,
                    'value': self._value_network,
                    'observation': self._observation_network,
                    'optimizer': self._optimizer,
                },
                directory=directory,
                time_delta_minutes=time_delta_minutes,
                max_to_keep=checkpoint_max_to_keep,
            )
            self._snapshotter = tf2_savers.Snapshotter(
                objects_to_save={
                    'policy':
                    snt.Sequential([
                        self._observation_network,
                        self._policy_network,
                    ])
                },
                directory=directory,
                time_delta_minutes=time_delta_minutes,
            )

        # Maybe load checkpoint.
        if checkpoint_to_load is not None:
            _checkpoint = tf.train.Checkpoint(
                counter=tf2_savers.SaveableAdapter(self._counter),
                policy=self._policy_network,
                value=self._value_network,
                observation=self._observation_network,
                optimizer=self._optimizer,
            )
            _checkpoint.restore(checkpoint_to_load)

        self._timestamp = None

    @tf.function
    def _ppo_step(
        self,
        obs_batch: tf.Tensor,
        actions_batch: tf.Tensor,
        old_log_probs_batch: tf.Tensor,
        advantages_batch: tf.Tensor,
        returns_batch: tf.Tensor,
        old_values_batch: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Single PPO gradient step on a minibatch."""

        with tf.GradientTape() as tape:
            # Forward pass.
            embeddings = self._observation_network(obs_batch)
            distribution = self._policy_network(embeddings)
            values = tf.squeeze(self._value_network(embeddings), axis=-1)

            # Policy loss (clipped surrogate).
            log_probs = distribution.log_prob(actions_batch)
            ratio = tf.exp(log_probs - old_log_probs_batch)
            clipped_ratio = tf.clip_by_value(
                ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio)
            surrogate1 = ratio * advantages_batch
            surrogate2 = clipped_ratio * advantages_batch
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

            # Value loss.
            if self._clip_value_loss:
                value_clipped = old_values_batch + tf.clip_by_value(
                    values - old_values_batch,
                    -self._clip_ratio, self._clip_ratio)
                value_loss1 = tf.square(values - returns_batch)
                value_loss2 = tf.square(value_clipped - returns_batch)
                value_loss = 0.5 * tf.reduce_mean(
                    tf.maximum(value_loss1, value_loss2))
            else:
                value_loss = 0.5 * tf.reduce_mean(
                    tf.square(values - returns_batch))

            # Entropy bonus.
            entropy = tf.reduce_mean(distribution.entropy())

            # Combined loss.
            loss = (policy_loss
                    + self._value_loss_coeff * value_loss
                    - self._entropy_coeff * entropy)

        # Compute and apply gradients.
        all_variables = (
            self._observation_network.trainable_variables +
            self._policy_network.trainable_variables +
            self._value_network.trainable_variables
        )
        gradients = tape.gradient(loss, all_variables)
        gradients, grad_norm = tf.clip_by_global_norm(
            gradients, self._max_grad_norm)
        self._optimizer.apply(gradients, all_variables)

        # Compute approximate KL for diagnostics.
        approx_kl = tf.reduce_mean(old_log_probs_batch - log_probs)
        clip_fraction = tf.reduce_mean(
            tf.cast(tf.abs(ratio - 1.0) > self._clip_ratio, tf.float32))

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'total_loss': loss,
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction,
            'grad_norm': grad_norm,
        }

    def update(self, trajectories: List[Dict[str, np.ndarray]]):
        """Run PPO update on collected trajectories.

        Args:
            trajectories: List of trajectory dicts from actors, each containing:
                'observations': [T, obs_dim]
                'actions': [T, act_dim]
                'rewards': [T]
                'dones': [T]
                'log_probs': [T]
                'values': [T]
                'last_value': scalar
        """
        # Concatenate all trajectories.
        all_obs = np.concatenate(
            [t['observations'] for t in trajectories], axis=0)
        all_actions = np.concatenate(
            [t['actions'] for t in trajectories], axis=0)
        all_log_probs = np.concatenate(
            [t['log_probs'] for t in trajectories], axis=0)
        all_values = np.concatenate(
            [t['values'] for t in trajectories], axis=0)

        # Compute GAE for each trajectory separately (respects episode boundaries).
        all_advantages = []
        all_returns = []
        offset = 0
        for traj in trajectories:
            T = len(traj['rewards'])
            advantages, returns = compute_gae(
                rewards=traj['rewards'],
                values=traj['values'],
                dones=traj['dones'],
                last_value=traj['last_value'],
                gamma=self._discount,
                gae_lambda=self._gae_lambda,
            )
            all_advantages.append(advantages)
            all_returns.append(returns)
            offset += T

        all_advantages = np.concatenate(all_advantages, axis=0)
        all_returns = np.concatenate(all_returns, axis=0)

        # Normalize advantages.
        if self._normalize_advantage:
            all_advantages = (
                (all_advantages - all_advantages.mean())
                / (all_advantages.std() + 1e-8)
            )

        # Convert to tensors.
        obs_tensor = tf.constant(all_obs, dtype=tf.float32)
        actions_tensor = tf.constant(all_actions, dtype=tf.float32)
        old_log_probs_tensor = tf.constant(all_log_probs, dtype=tf.float32)
        advantages_tensor = tf.constant(all_advantages, dtype=tf.float32)
        returns_tensor = tf.constant(all_returns, dtype=tf.float32)
        old_values_tensor = tf.constant(all_values, dtype=tf.float32)

        total_size = len(all_obs)
        batch_size = total_size // self._num_minibatches

        # PPO epochs: multiple passes over the data.
        epoch_stats = {}
        for epoch in range(self._ppo_epochs):
            # Shuffle indices.
            indices = np.random.permutation(total_size)

            for start in range(0, total_size - batch_size + 1, batch_size):
                mb_indices = indices[start:start + batch_size]
                mb_indices = tf.constant(mb_indices, dtype=tf.int32)

                stats = self._ppo_step(
                    obs_batch=tf.gather(obs_tensor, mb_indices),
                    actions_batch=tf.gather(actions_tensor, mb_indices),
                    old_log_probs_batch=tf.gather(
                        old_log_probs_tensor, mb_indices),
                    advantages_batch=tf.gather(
                        advantages_tensor, mb_indices),
                    returns_batch=tf.gather(returns_tensor, mb_indices),
                    old_values_batch=tf.gather(old_values_tensor, mb_indices),
                )
                # Track last minibatch stats.
                epoch_stats = {k: float(v) for k, v in stats.items()}

        self._last_stats = epoch_stats

    def step(self):
        """Step wrapper for counting/logging (called by Ray Learner.run)."""
        # The actual PPO update is done in self.update(), called externally.
        # This method handles logging of the most recent update stats.
        if not hasattr(self, '_last_stats') or self._last_stats is None:
            return

        fetches = dict(self._last_stats)
        self._last_stats = None

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update counts and record.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        # Checkpoint.
        if self._checkpointer is not None:
            self._checkpointer.save()
        if self._snapshotter is not None:
            self._snapshotter.save()

        self._logger.write(fetches)

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        self._counter.increment(get_variables_calls=1)
        return [tf2_utils.to_numpy(self._variables[name]) for name in names]
