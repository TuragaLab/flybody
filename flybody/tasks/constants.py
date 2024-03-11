"""Module defining constants for fruit fly RL tasks."""

__all__ = [
    '_WALK_CONTROL_TIMESTEP', '_WALK_PHYSICS_TIMESTEP', '_TERMINAL_LINVEL',
    '_TERMINAL_ANGVEL', '_FLY_CONTROL_TIMESTEP', '_FLY_PHYSICS_TIMESTEP',
    '_TERMINAL_HEIGHT', '_BODY_PITCH_ANGLE', '_WING_PARAMS'
]

# Walking constants.
_WALK_CONTROL_TIMESTEP = 2e-3  # s
_WALK_PHYSICS_TIMESTEP = 2e-4
_TERMINAL_LINVEL = 50  # cm/s
_TERMINAL_ANGVEL = 200  # rad/s

# Flight constants.
_FLY_CONTROL_TIMESTEP = 2e-4
_FLY_PHYSICS_TIMESTEP = 5e-5
_BODY_PITCH_ANGLE = 47.5  # deg
_TERMINAL_HEIGHT = 0.2  # cm

_TERMINAL_QACC = 1e14  # mixed units

_WING_PARAMS = {
    'base_freq': 218.,
    'gainprm': [18, 18, 18],
    'damping': 0.007769230,
    'stiffness': 0.01,
    'fluidcoef': [1.0, 0.5, 1.5, 1.7, 1.0],
    'rel_freq_range': 0.05,
    'num_freqs': 201,
}
