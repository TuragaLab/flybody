"""Multi-site inverse kinematics fitting for MuJoCo models."""
# ruff: noqa: F821

from typing import Sequence, NamedTuple, Union, List
from collections import namedtuple
import logging

import numpy as np
from dm_control.mujoco.wrapper import mjbindings

mjlib = mjbindings.mjlib


def qpos_from_site_xpos(physics: 'mjcf.Physics',
                        site_names: Sequence[str],
                        target_xpos: np.ndarray,
                        joint_names: Sequence[str],
                        reg_strength: float = 0.0,
                        lr: float = 0.01,
                        beta: float = 0.99,
                        progress_threshold: float = 0.01,
                        max_steps: int = 20_000,
                        include_inds: Union[slice, List[int]] = slice(None),
                        inplace: bool = False) -> NamedTuple:
    """Finds joint angles qpos such that the given model sites xpos match
    the target site positions.

    Minimizes the inverse kinematics objective:

    objective = ||s(q)-s*||^2 + a||q||^2

    Where s(q) are site positions as functions of joint angles q
    and s* is the target site positions (data). Only translational
    error is computed for now (rotational error is not computed.)

    TODO: Add support for indices in addition to string names in site_names.
    TODO: Add support for indices in addition to string names in joint_names.
    TODO: Add dtype enforcing to other funtions in this module,
        like dtype = physics.data.qpos.dtype.
    TODO: Maybe add gradient clipping and joint range clipping.
    TODO: Maybe add tol precision goal.

    Args:
        physics: mjcf.Physics instance.
        site_names: List of names of model sites to be matched to data.
        target_xpos: Numpy array of target site positions, (n_sites, 3).
        joint_names: List of joint names to modify by inverse kinematics.
        reg_strength: Coefficient `a` of the quadratic penalty on joint
        deviations from reference (defaults) pose.
        lr: Learning rate for gradient descent.
        beta: Momentum beta, zero means no momentum (reduces to plain
            gradient descent in this case.) With momentum, the weight
            of the previous grad updates is on the order ~beta/(1-beta),
            and the weight of the current grad is ~1.
        progress_threshold: Stop optimization when the ratio
            gradient_update / err_norm becomes smaller than progress threshold.
            Setting progress_threshold to zero means no threshold and
            the update can get arbitrarily small.
        max_steps: Maximum number of iterations to perform.
        include_inds: Which Cartesian components of the site coordinates
            to include in the objective calculation. All included by default.
            The indices are w.r.t. target_xpos.flattened().
            For example, to include only the x, y components of all sites
            (in other words, to exclude the z-component), use
            include_inds = [0, 1, 3, 4, 6, 7, ...], and so on.
        inplace: If True, physics.data will be modified in place.
            Defaults to False, i.e. a copy of physics.data will be made.

    Returns:
        A namedtuple containing the joint angles qpos, translational
            residuals, and number of steps. It also contains the objective's
            first term error for assessing regularization strength and pure
            site position error.
    """

    dtype = physics.data.qpos.dtype

    nv_update = np.zeros(physics.model.nv, dtype=dtype)
    # Select indices of dofs requested to be modified.
    row_indexer = physics.named.model.dof_jntid.axes.row
    dof_indices = row_indexer.convert_key_item(joint_names)

    # Make sure that the Cartesian position of the site is up to date.
    mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

    # Before entering the main loop, prepare indices and joint names
    # for gradient and objective.
    site_indices = name2id(physics, names=site_names, object_type='site')
    row_indexer = physics.named.model.dof_jntid.axes.row
    dof_indices = row_indexer.convert_key_item(joint_names)
    hinge_joint_names = [
        name for name in joint_names if physics.named.model.jnt_type[name] == 3
    ]
    hinge_dof_indices = row_indexer.convert_key_item(hinge_joint_names)

    if not inplace:
        physics = physics.copy(share_model=True)

    success = False
    update = 0.
    for step in range(max_steps):

        site_xpos = physics.named.data.site_xpos[site_indices]
        grad = gradient(physics, target_xpos, site_xpos, site_indices,
                        dof_indices, hinge_joint_names, hinge_dof_indices,
                        reg_strength, include_inds)  # (partial nv,)

        # Prepare a gradient descent step (with momentum).
        update = beta * update + grad
        nv_update[dof_indices] = -lr * update

        # Update physics.qpos, taking quaternions into account.
        mjlib.mj_integratePos(physics.model.ptr, physics.data.qpos, nv_update,
                              1)

        # Compute the new Cartesian position of the sites.
        mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

        # Check error every 100 iterations.
        if step % 100 == 0:
            site_xpos = physics.named.data.site_xpos[site_indices]
            err = objective(physics, target_xpos, site_xpos, hinge_joint_names,
                            reg_strength, include_inds)
            progress_criterion = lr * np.linalg.norm(update) / err
            if progress_criterion < progress_threshold:
                success = True
                logging.debug(
                    f'Progress threshold reached after {step} steps: err = {err}'
                )
                break

    if step == max_steps - 1:
        logging.warning(
            f'Failed to converge after {max_steps} steps: err = {err}')

    if not inplace:
        # Our temporary copy of physics.data is about to go out of scope,
        # and when it does the underlying mjData pointer will be freed and
        # physics.data.qpos will be a view onto a block of deallocated memory.
        # We therefore need to make a copy of physics.data.qpos while
        # physics.data is still alive.
        qpos = physics.data.qpos.copy()
    else:
        # If we're modifying physics.data in place then it's fine to return
        # a view.
        qpos = physics.data.qpos

    # Calculate the residual of the first term only.
    err_first_term = objective(physics,
                               target_xpos,
                               site_xpos,
                               hinge_joint_names,
                               reg_strength=0,
                               include_inds=include_inds)

    IKResult = namedtuple(
        'IKResult',
        ['qpos', 'err_norm', 'err_norm_first_term', 'steps', 'success'])

    return IKResult(qpos=qpos,
                    err_norm=err,
                    err_norm_first_term=err_first_term,
                    steps=step,
                    success=success)


def mj_jac_pos(physics: 'mjcf.Physics', jac: np.ndarray,
               site_indices: Sequence[int]) -> None:
    """Wrapper to generalize mj_jacSite to compute the full translational
    Jacobian analytically for any number of sites.

    For now, computes only the positional part of the Jacobian.

    TODO: Add the rotational part, too.

    Args:
        physics: mjcf.Physics instance.
        jac: Allocated numpy array, to be modified inplace, (3*n_sites, nv).
        site_indices: Indices of sites to differentiate.
    """
    for i, site_index in enumerate(site_indices):
        mjlib.mj_jacSite(physics.model.ptr, physics.data.ptr,
                         jac[3 * i:3 * i + 3, :], None, site_index)


def objective(
        physics: 'mjcf.Physics',
        target_xpos: np.ndarray,
        site_xpos: np.ndarray,
        hinge_joint_names: Sequence[str],
        reg_strength: float,
        include_inds: Union[slice, List[int]] = slice(None),
) -> float:
    """Computes scalar value of the regularized objective function:

    objective = ||s(q)-s*||^2 + a||q||^2

    Where s(q) are site positions as functions of joint angles q
    and s* is the target site positions (data). Only translational
    error is computed for now (rotational error is not computed.)

    Args:
        physics: Instance of mjcf.Physics.
        target_xpos: Numpy array of target site positions, (n_sites, 3).
        site_xpos: Numpy array of current site positions, (n_sites, 3).
        hinge_joint_names: Names of hinge joints out of all joint names
            requested to be modified.
        reg_strength: Coefficient `a` of the quadratic penalty
            on joint deviations from reference (defaults) pose.
        include_inds: Which Cartesian components of the site coordinates
            to include in the objective calculation. All included by default.
            The indices are w.r.t. site_xpos.flattened().
            For example, to include only the x, y components of all sites
            (in other words, to exclude the z-component), use
            include_inds = [0, 1, 3, 4, 6, 7, ...], and so on.

    Returns:
        The objective function's value, scalar.
    """
    hinge_qpos = physics.named.data.qpos[hinge_joint_names]
    diff = (np.array(site_xpos) -
            np.array(target_xpos)).flatten()[include_inds]
    err_pos = np.linalg.norm(diff)**2
    err_pos += reg_strength * np.linalg.norm(hinge_qpos)**2
    return err_pos


def gradient(
        physics: 'mjcf.Physics',
        target_xpos: np.ndarray,
        site_xpos: np.ndarray,
        site_indices: Sequence[int],
        dof_indices: Sequence[int],
        hinge_joint_names: Sequence[str],
        hinge_dof_indices: Sequence[int],
        reg_strength: float,
        include_inds: Union[slice, List[int]] = slice(None),
) -> np.ndarray:
    """Computes the gradient d(objective)/dq, where:

    objective = ||s(q)-s*||^2 + a||q||^2

    Where s(q) are site positions as functions of joint angles q
    and s* is the target site positions (data). This objective computes only
    translational error for now (rotational error is not computed.)

    Args:
        physics: Instance of mjcf.Physics.
        target_xpos: Numpy array of target site positions, (n_sites, 3).
        site_xpos: Numpy array of current site positions, (n_sites, 3).
        site_indices: List of site indices, length n_sites.
        dof_indices: DOF indices in an (nv,) array corresponding to joints
            requested to be modified.
        hinge_joint_names: Names of hinge joints out of all joint names
            requested to be modified.
        hinge_dof_indices: DOF indices in an (nv,) array corresponding to
            subset of hinge joints out of all joints requested to be modified.
        reg_strength: Coefficient `a` of the quadratic penalty
            on joint deviations from reference (defaults) pose.
        include_inds: Which Cartesian components of the site coordinates
            to include in the objective calculation. All included by default.
            The indices are w.r.t. site_xpos.flattened().
            For example, to include only the x, y components of all sites
            (in other words, to exclude the z-component), use
            include_inds = [0, 1, 3, 4, 6, 7, ...], and so on.

    Returns:
        grad: Numpy array of shape (partial nv,), where `partial nv` means
            the number of DOFs, out of all DOFs nv, corresponding to the joints
            requested to be modified.
    """
    # Allocate memory for the full Jacobian of shape (3*n_sites, nv).
    jac_full = np.empty((3 * target_xpos.shape[0], physics.model.nv))

    # This will compute the full translational Jacobian, for all `nv` DOFs.
    # We will have to select only DOFs that correspond to joints we are
    # interested in modifying.
    mj_jac_pos(physics, jac_full, site_indices)  # jac_full: (3*n_sites, nv)

    # Select DOFs chosen to be modified.
    jac_partial = jac_full[:, dof_indices]  # (3*n_sites, partial nv)

    # Now, first create the full hinge_qpos of shape (nv,)
    # Then fill hinge DOFs locations with their corresponding qpos values.
    # Finally, select only the DOFs that are requested to be modified.
    hinge_qpos = np.zeros(physics.model.nv)  # (nv,)
    hinge_qpos[hinge_dof_indices] = physics.named.data.qpos[hinge_joint_names]
    hinge_qpos = hinge_qpos[dof_indices]  # (partial nv,)

    # Compute the gradient itself, shape (partial nv,).
    grad = 2 * np.matmul((site_xpos - target_xpos).flatten()[include_inds],
                         jac_partial[include_inds, :])
    grad += 2 * reg_strength * hinge_qpos  # Regularization term.

    return grad  # (partial nv,)


def name2id(physics: 'mjcf.Physics', names: Union[str, Sequence[str]],
            object_type: str) -> List[int]:
    """Returns list of MuJoCo object ids for specified names and object type."""
    if isinstance(names, str):
        names = [names]
    ids = [physics.model.name2id(name, object_type) for name in names]
    return ids
