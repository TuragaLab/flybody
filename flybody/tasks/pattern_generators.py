"""Wing beat pattern generators for fly tasks."""

import numpy as np

from flybody.tasks.constants import (_WING_PARAMS, _FLY_CONTROL_TIMESTEP)


class WingBeatPatternGenerator():
    """Class for controllable frequency-modulation of cyclic wing beat data.

    This class generates wing beat patterns at beat frequencies requested by
    the agent and supports seamless switching from one frequency to another
    while maintaining the wing beat phase. As input, this class requires only
    one cycle of wing beat data.
    """

    def __init__(
        self,
        base_pattern_path: str | None = None,
        base_beat_freq=_WING_PARAMS['base_freq'],
        rel_freq_range=_WING_PARAMS['rel_freq_range'],
        num_freqs=_WING_PARAMS['num_freqs'],
        min_repeats: int = 10,
        max_repeats: int = 20,
        dt_ctrl: float = _FLY_CONTROL_TIMESTEP,
        ctrl_filter: float = 0.5 / _WING_PARAMS['base_freq'],
    ):
        """Initialize and construct wing sequencies at different flapping freqs.

        Args:
          base_pattern_path: Path to one cycle of 'base' wing kinematic data,
              a np.array of shape (timesteps, 3). The order of wing joints is
              yaw, roll, pitch. Sampling frequency does not have to match dt_ctrl.
              If not provided, a simple approximation will be generated (see
              equations below). This approximation can be used for prototyping
              and testing, but is not a substitute for a realistic base wing pattern.
          base_beat_freq: Mean beat frequency for the requested range of
              frequencies, Hz.
          rel_freq_range: Relative frequency range. For example, 0.1 means
              frequencies in the range base_beat_freq +/- 10%.
          num_freqs: How many discrete frequencies to generate in the range.
          min_repeats: Maximum number of allowed base_pattern repeats at
              each individual frequency. Larger min_repeats allows
              maintaining perfect connection between repeated wing beat
              cycles longer.
          max_repeats: Maximum number of allowed base_pattern repeats at
              each individual frequency. Larger max_repeats allows finding
              smoother connection of previous beat cycles to the next ones.
          dt_ctrl: Wing control timestep, seconds.
          ctrl_filter: Time constant of control signal filter, seconds.
              0: not used.
        """
        if base_pattern_path is None:
            # Generate a simple artificial base wing pattern approximation.
            x = np.linspace(0, 2*np.pi, 500)
            yaw = 1.1 * np.sin(x-np.pi/2) + 0.3
            roll = 0.25 * np.sin(1.5*x) - 0.1
            pitch = 1.35 * np.sin(x) + 0.8
            base_pattern = np.vstack((yaw, roll, pitch)).T  # Shape (500, 3).
        else:
            # Load base pattern for WBPG, shape (timesteps, 3).
            with open(base_pattern_path, 'rb') as f:
                base_pattern = np.load(f)
        base_pattern = np.tile(base_pattern, (1, 2))  # Duplicate for two wings.

        self.base_beat_freq = base_beat_freq
        self.rel_freq_range = rel_freq_range
        self.ctrl_filter = ctrl_filter
        self._dt_ctrl = dt_ctrl

        if ctrl_filter != 0.:
            self._rate = np.exp(-dt_ctrl / ctrl_filter)

        # Beat frequencies in the requested range.
        self.beat_freqs = np.linspace((1 - rel_freq_range) * base_beat_freq,
                                      (1 + rel_freq_range) * base_beat_freq,
                                      num_freqs)

        # For each beat frequency, construct a wing beat sequence while finding
        # such number of repeats that make sure smoothest possible connection
        # of one sequence cycle to the next.
        self.traj_ctrl = []
        self._rel_errors = []
        self._n_repeats = []
        for beat_freq in self.beat_freqs:

            # Duration of one wing cycle in data at current beat frequency.
            beat_time = 1 / beat_freq
            # Errors (relative to dt_ctrl) when connecting repeated sequences.
            reps = np.arange(min_repeats, max_repeats + 1)
            rel_error = ((reps * beat_time) % dt_ctrl) / dt_ctrl
            # Get number of repeats with smallest relative error.
            argmin1 = np.argmin(rel_error)
            argmin2 = np.argmin(np.abs(1 - rel_error))
            if rel_error[argmin1] < np.abs(1 - rel_error[argmin2]):
                argmin = argmin1  # Overshoot error.
                shift = dt_ctrl
            else:
                argmin = argmin2  # Undershoot error.
                shift = 0.
            n_reps = argmin + 1
            self._rel_errors.append(rel_error[argmin])
            self._n_repeats.append(n_reps)

            # Repeat wing kinematics n_reps times.
            repeated_traj = np.tile(base_pattern, reps=(n_reps, 1))
            # Phase within current repeated beat sequence.
            phase = np.linspace(0,
                                n_reps,
                                n_reps * base_pattern.shape[0],
                                endpoint=False)
            # Time axes for interpolation.
            dt_data = beat_time / base_pattern.shape[0]  # Data timestep.
            traj_duration = repeated_traj.shape[0] * dt_data
            t_axis_data = np.linspace(0, traj_duration, repeated_traj.shape[0])
            t_axis_ctrl = np.arange(0, traj_duration - shift, dt_ctrl)
            # Interpolate wing trajectories and phases to control timesteps.
            n_angles = base_pattern.shape[1]
            repeated_traj_ctrl = np.zeros((t_axis_ctrl.shape[0], n_angles))
            for i in range(n_angles):
                repeated_traj_ctrl[:, i] = np.interp(t_axis_ctrl, t_axis_data,
                                                     repeated_traj[:, i])
            phase_ctrl = np.interp(t_axis_ctrl, t_axis_data, phase)

            self.traj_ctrl.append({
                'traj': repeated_traj_ctrl,
                't_axis': t_axis_ctrl,
                'phase': phase_ctrl,
            })

    def reset(self,
              ctrl_freq: float | None = None,
              initial_phase: float = 0.,
              return_qvel: bool = False) -> np.ndarray:
        """Reset wing sequence to step 0 and set initial phase.

        Args:
          ctrl_freq: Optional, starting beat frequency, Hz. If not provided,
            base_beat_freq is used instead.
          initial_phase: Optional, initial phase within the beat cycle,
            in range [0, 1].
          return_qvel: Whether to return initial wing joint qvel.

        Returns:
          Initial set of wing kinematic angles, shape (n_wing_angles,).
        """
        if ctrl_freq is None:
            self._ctrl_freq = self.base_beat_freq
        else:
            self._ctrl_freq = ctrl_freq
        # Frequency index closest to ctrl_freq.
        self._freq_idx = np.argmin(np.abs(self.beat_freqs - self._ctrl_freq))
        # Initialize wing beat sequence.
        self._traj = self.traj_ctrl[self._freq_idx]['traj']
        self._cycle_len = self._traj.shape[0]
        # Position inside current wing beat sequence.
        self._step = np.argmin(
            np.abs(initial_phase - self.traj_ctrl[self._freq_idx]['phase']))

        if return_qvel:
            return (
                self._traj[self._step, :],
                (self._traj[self._step + 1, :] - self._traj[self._step, :]) /
                self._dt_ctrl)

        return self._traj[self._step, :]

    def step(self, ctrl_freq: float) -> np.ndarray:
        """Step and return the next set of wing angles. Maybe change beat freq.

        Args:
          ctrl_freq: New beat frequency to switch to, or keep current one.

        Returns:
          Next set of wing kinematic angles, shape (n_wing_angles,).
        """
        self._step = (self._step + 1) % self._cycle_len

        # Maybe apply control filter.
        if self.ctrl_filter == 0.:
            self._ctrl_freq = ctrl_freq
        else:
            self._ctrl_freq = (self._ctrl_freq * self._rate + ctrl_freq *
                               (1 - self._rate))

        # Maybe switch to another wing beat frequency sequence, while making
        # sure that the phase in the new sequence matches the current phase as
        # closely as possible.
        idx_new = np.argmin(np.abs(self.beat_freqs - self._ctrl_freq))
        if idx_new != self._freq_idx:
            current_phase = self.traj_ctrl[self._freq_idx]['phase'][self._step]
            # Get new step inside new beat sequence, while preserving the phase
            # within beat cycle.
            self._step = np.argmin(
                np.abs(current_phase % 1 -
                       self.traj_ctrl[idx_new]['phase'] % 1))

            # Pick new wing beat sequence.
            self._traj = self.traj_ctrl[idx_new]['traj']
            self._cycle_len = self._traj.shape[0]
            self._freq_idx = idx_new

        return self._traj[self._step, :]

    def get_last_angles(self):
        """Re-return the last wing angles, could be used for debugging."""
        return self._traj[self._step, :]
