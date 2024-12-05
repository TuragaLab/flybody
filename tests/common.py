"""Common functions and fixtures for flybody tests."""

import numpy as np


def is_force_actuator(physics, actuator_id=None):
    """Check if actuator with given id is a force actuator with ctrlrange == (-1, 1).
    If actuator_id is None, check all actuators.
    """
    physics.reset()
    if actuator_id is None:
        inds = [*range(physics.model.nu)]
    else:
        assert isinstance(actuator_id, int)
        inds = [actuator_id]
    for i in inds:
        assert physics.model.actuator_gainprm[i][0] != 0.
        assert np.all(physics.model.actuator_gainprm[i][1:] == 0.)
        assert np.all(physics.model.actuator_biasprm[i][:] == 0.)
        assert physics.model.actuator_gaintype[i] == 0.
        assert physics.model.actuator_biastype[i] == 0.
        if physics.model.actuator_trntype[i] != 5:
            # Force actuator (on either joint or tendon).
            assert np.all(physics.model.actuator_ctrlrange[i] == (-1, 1))
        else:
            # Adhesion actuator.
            assert np.all(physics.model.actuator_ctrlrange[i] == (0, 1))

    return True
