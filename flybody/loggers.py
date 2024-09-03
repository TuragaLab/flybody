"""Loggers for RL training with Ray."""

import time
import numpy as np
from acme.utils.loggers import base
import mlflow


class MLflowLogger(base.Logger):
    """Logs training stats to local MLflow tracking server."""

    def __init__(self,
                 uri: str,
                 run_id: str,
                 label: str = '',
                 time_delta: float = 0.,
                ):
        """Initializes the logger.

        Args:
            uri: Tracking server URI, could be e.g. 'http://127.0.0.1:8080'.
            run_id: UUID of existing run to log metrics and parameters for.
                Each process (learner, actors, etc.) will have its own logger
                instance, but they all will log under the same run.
            label: Label string to use when logging, e.g. 'learner', 'actor',
                'evaluator'.
            time_delta: How often (in seconds) to write values.
                If zero, everything is written.
        """
        # Start logging under an existing run.
        mlflow.set_tracking_uri(uri=uri)
        mlflow.start_run(run_id=run_id)

        self._label = label
        self._time = time.time()
        self._time_delta = time_delta
        self._keys2track = [
            'learner_get_variables_calls', 'learner_steps', 'learner_walltime',
            'episode_length', 'episode_return', 'actor_episodes', 'actor_steps',
        ]
        # Log only a subset of tracked and calculated keys.
        self._keys2log = [
            'walltime_hr', 'acting-to-learning', 'episode_length',
            'steps_per_second_actor', 'steps_per_second_learner',
            'evaluator_episode_return', 'actor_episode_return',
        ]

        # Client will be used by Learner to calculate and log average return
        # over all actors.
        self._client = mlflow.tracking.MlflowClient()
        self._run_id = run_id
    
    def write(self, values: base.LoggingData):
        """Write data to destination.
        
        Args:
            values: Mapping[str, Any].
        """

        # Always log saved_snapshot_at_actor_steps when it occurs.
        if 'saved_snapshot_at_actor_steps' in values:
            step = values['saved_snapshot_at_actor_steps']
            mlflow.log_metric(
                key='saved_snapshot_at_actor_steps',
                value=step,
                step=step)

        now = time.time()
        if (now - self._time) < self._time_delta:
            return
        
        # Format metrics.
        metrics = {}
        for k, v in values.items():
            if k in self._keys2track:
                formatted = base.to_numpy(v)
                if isinstance(formatted, np.ndarray):
                    formatted = formatted.item()
                metrics[k] = formatted

        # Calculate additional metrics.
        if 'learner_walltime' in metrics and metrics['learner_walltime'] > 0:
            metrics['walltime_hr'] = metrics['learner_walltime'] / 3600.
            if 'learner_steps' in metrics and metrics['learner_steps'] > 0:
                sps = metrics['learner_steps'] / metrics['learner_walltime']
                metrics['steps_per_second_learner'] = sps
            if 'actor_steps' in metrics and metrics['actor_steps'] > 0:
                sps = metrics['actor_steps'] / metrics['learner_walltime']
                metrics['steps_per_second_actor'] = sps
        if ('steps_per_second_actor' in metrics and
            'steps_per_second_learner' in metrics):
            sps_act = metrics['steps_per_second_actor']
            sps_lrn = metrics['steps_per_second_learner']
            metrics['acting-to-learning'] = sps_act / sps_lrn
        if self._label == 'evaluator' and 'episode_return' in metrics:
            metrics['evaluator_episode_return'] = metrics['episode_return']
        if 'episode_return' in metrics:
            metrics['actor_episode_return'] = metrics['episode_return']

        # Log the subset of metrics.
        step = metrics['actor_steps'] if 'actor_steps' in metrics else 0
        metrics = {k: v for k, v in metrics.items() if k in self._keys2log}
        mlflow.log_metrics(metrics, step=step)
        self._time = now
    
        # If this logger instance is in learner, also calculate and log average
        # return over all actors.
        if self._label == 'learner':
            history = self._client.get_metric_history(
                run_id=self._run_id, key='actor_episode_return')
            if history:
                x = [entry.step for entry in history]
                y = [entry.value for entry in history]
                x_conv, y_conv = self._convolve(x, y)
                logged_so_far = self._client.get_metric_history(
                    run_id=self._run_id, key='average_episode_return')
                idx_from = len(logged_so_far)
                for value, step in zip(y_conv[idx_from:], x_conv[idx_from:]):
                    mlflow.log_metric('average_episode_return', value, step=step)

    def _convolve(self, x, y, kernel_size=50):
        y_conv = np.convolve(y, np.ones(kernel_size)/kernel_size)
        y_conv = y_conv[kernel_size:-kernel_size]
        x_conv = x[kernel_size//2:-kernel_size//2-1]
        return x_conv, y_conv

    def close(self):
        """Closes the logger, not expecting any further write."""
        mlflow.end_run()
