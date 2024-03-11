"""Vectorized operations with quaternions with batch dimension support."""

import numpy as np


def get_dquat(quat1, quat2):
    """Returns 'delta' dquat quaternion that transforms quat1 to quat2.
    Namely, multiplying dquat and quat1 as mult_quat(dquat, quat1) gives quat2.
    """
    return mult_quat(quat2, reciprocal_quat(quat1))


def get_dquat_local(quat1, quat2):
    """Returns 'delta' dquat in the local reference frame of quat1.
    This is the orientation quaternion quat2 as seen from local frame of quat1.
    """
    return mult_quat(reciprocal_quat(quat1), quat2)


def get_quat(theta=0, rot_axis=[0., 0, 1]):
    """Unit quaternion for given angle and rotation axis.
    
    Args:
        theta: Angle in radians.
        rot_axis: Rotation axis, does not have to be normalized, shape (3,).

    Returns:
        Rotation unit quaternion, (4,).
    """
    axis = rot_axis / np.linalg.norm(rot_axis)
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array((c, axis[0] * s, axis[1] * s, axis[2] * s))


def random_quat():
    """Returns normalized random quaternion."""
    theta = 2 * np.pi * np.random.rand()
    axis = 2 * np.random.rand(3) - 1
    axis /= np.linalg.norm(axis)
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array((c, axis[0] * s, axis[1] * s, axis[2] * s))


def mult_quat(quat1: np.ndarray, quat2: np.ndarray) -> np.ndarray:
    """Computes the Hamilton product of two quaternions `quat1` * `quat2`.
    This is a general multiplication, the input quaternions do not have to be
    unit quaternions.

    Any number of leading batch dimensions is supported.

    Broadcast rules:
        One of the input quats can be (4,) while the other is (B, 4).

    Args:
        quat1, quat2: Arrays of shape (B, 4) or (4,).

    Returns:
        Product of quat1*quat2, array of shape (B, 4) or (4,).
    """
    a1, b1, c1, d1 = quat1[..., 0], quat1[..., 1], quat1[..., 2], quat1[..., 3]
    a2, b2, c2, d2 = quat2[..., 0], quat2[..., 1], quat2[..., 2], quat2[..., 3]
    prod = np.empty_like(quat1) if quat1.ndim > quat2.ndim else np.empty_like(
        quat2)
    prod[..., 0] = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    prod[..., 1] = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    prod[..., 2] = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    prod[..., 3] = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
    return prod


def conj_quat(quat: np.ndarray) -> np.ndarray:
    """Returns the conjugate quaternion of `quat`.

    Any number of leading batch dimensions is supported.

    Args:
        quat: Array of shape (B, 4).

    Returns:
        Conjugate quaternion(s), array of shape (B, 4).
    """
    quat = quat.copy()
    quat[..., 1:] *= -1
    return quat


def reciprocal_quat(quat: np.ndarray) -> np.ndarray:
    """Returns the reciprocal quaternion of `quat` such that the product
    of `quat` and its reciprocal gives unit quaternion:

    mult_quat(quat, reciprocal_quat(quat)) == [1., 0, 0, 0]

    Any number of leading batch dimensions is supported.

    Args:
        quat: Array of shape (B, 4).

    Returns:
        Reciprocal quaternion(s), array of shape (B, 4).
    """
    return conj_quat(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)**2


def rotate_vec_with_quat(vec, quat):
    """Uses unit quaternion `quat` to rotate vector `vec` according to:

        vec' = quat vec quat^-1.

    Any number of leading batch dimensions is supported.

    Technically, `quat` should be a unit quaternion, but in this particular
    multiplication (quat vec quat^-1) it doesn't matter because an arbitrary
    constant cancels out in the product.

    Broadcasting works in both directions. That is, for example:
    (i) vec and quat can be [1, 1, 3] and [2, 7, 4], respectively.
    (ii) vec and quat can be [2, 7, 3] and [1, 1, 4], respectively.

    Args:
        vec: Cartesian position vector to rotate, shape (B, 3). Does not have
            to be a unit vector.
        quat: Rotation unit quaternion, (B, 4).

    Returns:
        Rotated vec, (B, 3,).
    """
    if vec[..., :-1].size > quat[..., :-1].size:
        # Broadcast quat to vec.
        quat = np.tile(quat, vec.shape[:-1] + (1, ))
    vec_aug = np.zeros_like(quat)
    vec_aug[..., 1:] = vec
    vec = mult_quat(quat, mult_quat(vec_aug, reciprocal_quat(quat)))
    return vec[..., 1:]


def get_egocentric_vec(root_xpos, site_xpos, root_quat):
    """Returns the difference vector (site_xpos - root_xpos) represented
    in the local root's frame of reference.

    Any number of leading batch dimensions is supported.

    Broadcasting works in both directions. That is, for example:
    (i) root_xpos, root_quat, site_xpos can be [1, 1, 3], [1, 1, 4], [7, 9, 3]
    (ii) root_xpos, root_quat, site_xpos can be [4, 7, 3], [4, 7, 4], [1, 1, 3]

    Args:
        root_xpos: Cartesian root position in global coordinates, (B, 3).
        site_xpos: Cartesian position of the site (or anything else)
            in global coordinates, (B, 3).
        root_quat: Orientation unit quaternion of the root w.r.t. world, (B, 4).

    Returns:
        Egocentric representation of the vector (site_xpos - root_xpos), (B, 3).
    """
    # End-effector vector in world reference frame.
    root_to_site = site_xpos - root_xpos
    # Return end-effector vector in local root reference frame.
    return rotate_vec_with_quat(root_to_site, conj_quat(root_quat))


def vec_world_to_local(world_vec, root_quat, hover_up_dir_quat=None):
    """Local reference frame representation of vectors in world coordinates.
    
    Any number of leading batch dimensions is supported.
    
    Args:
        world_vec: Vector in world coordinates, (B, 3).
        root_quat: Root quaternion of the local reference frame, (B, 4).
        hover_up_dir_quat: Optional, fly's hover_up_dir quaternion, (4,).
        
    Returns:
        world_vec in local reference frame, (B, 3).
    """
    root_quat = conj_quat(root_quat)

    if hover_up_dir_quat is not None:
        # For flexible broadcasting, instead of np.tile
        hover_up_dir_quat = np.zeros_like(root_quat) + hover_up_dir_quat
        root_quat = mult_quat(conj_quat(hover_up_dir_quat), root_quat)

    return rotate_vec_with_quat(world_vec, root_quat)


def log_quat(quat: np.ndarray) -> np.ndarray:
    """Computes log of quaternion `quat`. The result is also a quaternion.
    This is a general operation, `quat` does not have to be a unit quaternion.

    Any number of leading batch dimensions is supported.

    Args:
        quat: Array of shape (B, 4).

    Returns:
        Array of shape (B, 4).
    """
    norm_quat = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm_v = np.linalg.norm(quat[..., 1:], axis=-1, keepdims=True)
    log_quat = np.empty_like(quat)
    log_quat[..., 0:1] = np.log(norm_quat)
    log_quat[..., 1:] = quat[..., 1:] / norm_v * np.arccos(
        quat[..., 0:1] / norm_quat)
    return log_quat


def quat_z2vec(vec: np.ndarray) -> np.ndarray:
    """Returns unit quaternion performing rotation from z-axis
    to given `vec`.

    Any number of leading batch dimensions is supported.
    Edge cases such as vec = [0, 0, 0], [0, 0, 1], [0, 0, -1]
    are taken care of.

    Args:
        vec: Vector(s) to rotate to from z-axis, shape (B, 3). Does not have
            to be a unit vector.

    Returns:
        Array of unit quaternions of shape (B, 4).
    """
    # Find indices of edge cases, if present.
    edge_inds = np.argwhere((vec[..., :2] == 0.).all(axis=-1, keepdims=False))
    if edge_inds.size:
        vec = vec.copy()
        # Temporarily put placeholders into `vec` to avoid nans.
        for edge_ind in edge_inds:
            ind = tuple(edge_ind) + (slice(0, 1), )
            vec[ind] = 1.  # Placeholder.

    # Get axis-and-angle representation first.
    vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
    # Cross product with [0, 0, 1].
    axis = np.stack([-vec[..., 1], vec[..., 0],
                     np.zeros_like(vec[..., 0])],
                    axis=-1)
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = np.arccos(vec[..., 2:3])
    # Compose quaternion.
    quat = np.zeros(vec.shape[:-1] + (4, ))
    quat[..., 0:1] = np.cos(angle / 2)
    quat[..., 1:] = np.sin(angle / 2) * axis

    # Clean edge case placeholders, if there are any.
    for edge_ind in edge_inds:
        ind_vec = tuple(edge_ind) + (slice(2, 3), )
        ind_quat = tuple(edge_ind) + (slice(None), )
        if vec[ind_vec] < 0:
            quat[ind_quat] = [0., 1., 0., 0.]
        else:
            quat[ind_quat] = [1., 0., 0., 0.]

    return quat


def axis_angle_to_quat(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Converts axis-angle representation of rotation to the corresponding
    rotation unit quaternion.

    Any number of leading batch dimensions is supported.

    Args:
        axis: Cartesian directions of rotation axes, shape (B, 3). Do not have
            to be unit vectors.
        angle: Angle of rotation around `axis`, radians, shape (B,).

    Returns:
        Rotation (unit) quaternions, shape (B, 4).
    """
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    quat = np.zeros(axis.shape[:-1] + (4, ))
    quat[..., 0] = np.cos(angle / 2)
    quat[..., 1:] = np.sin(angle / 2)[..., None] * axis
    return quat


def quat_dist_short_arc(quat1: np.ndarray, quat2: np.ndarray) -> np.ndarray:
    """Returns the shortest geodesic distance between two unit quaternions.

    angle = arccos(2(p.q)^2 - 1)

    The distance is the angle subtended by `quat1` and `quat2` along
    a great arc of the 4D sphere. The arc IS the shortest one out of the two
    possible arcs, and the returned angle is in range: 0 <= angle < pi.

    Any number of leading batch dimensions is supported.

    Args:
        quat1, quat2: Arrays of shape (B, 4), any number of batch
            dimensions is supported.

    Returns:
        An array of quaternion distances, shape (B,).
    """
    quat1 = quat1 / np.linalg.norm(quat1, axis=-1, keepdims=True)
    quat2 = quat2 / np.linalg.norm(quat2, axis=-1, keepdims=True)
    x = 2 * np.sum(quat1 * quat2, axis=-1)**2 - 1
    x = np.minimum(1., x)
    return np.arccos(x)


def joint_orientation_quat(xaxis: np.ndarray, qpos: float) -> np.ndarray:
    """Computes joint orientation quaternion from joint's Cartesian axis
    direction in world coordinates and joint angle `qpos`.

    Any number of leading batch dimensions is supported.

    Args:
        xaxis: Cartesian direction of joint axis in world coordinates, (B, 3).
            Do not have to be unit vectors.
        qpos: Corresponding joint angles, shape (B,).

    Returns:
        Unit quaternions representing joint orientations in world's frame
            of reference, (B, 4).
    """
    # Quaternion that rotates from Z to `xaxis`.
    quat1 = quat_z2vec(xaxis)

    # Quaternion that rotates around `xaxis` by `qpos`.
    quat2 = axis_angle_to_quat(xaxis, qpos)

    # Final quaternion (combination of quat1 & quat2) for the joint orientation.
    joint_quat = mult_quat(quat2, quat1)
    return joint_quat


def quat_seq_to_angvel(quats, dt=1., local_ref_frame=False):
    """Covert sequence of orientation quaternions to angular velocities.
    
    Args:
        quats: Sequence of quaternions. List of quaternions or array (time, 4).
        dt: Timestep.
        local_ref_frame: Whether to return angular velocity in global or local
            reference frame.
            Global reference frame: the frame the quats are defined in.
            Local reference frame: the frame attached to the body with
                orientation defined by quats.
            
    Returns:
        Sequence of angular velovicies in either global or local reference frame.
    """
    dquats = get_dquat(quats[:-1], quats[1:])
    ang_vel = quat_to_angvel(dquats, dt=dt)
    if local_ref_frame:
        ang_vel = vec_global_to_local(ang_vel, quats[:-1])
    return ang_vel


def quat_to_angvel(quat, dt=1.):
    """Convert quaternion (corresponding to orientation difference) to angular velocity.
    Input and output are in the same (global) reference frame.
    
    Any number of leading batch dimensions is supported.
    
    This is a python implementation of MuJoCo's mju_quat2Vel function.
    
    Args:
        quat: Orientation difference quaternion, (B, 4).
        dt: Timestep.
    
    Returns:
        Angular velocity vector, (B, 3), in the same (global) reference frame
            as the input quat.
    """
    sin_a_2 = np.linalg.norm(quat[..., 1:], axis=-1, keepdims=True)
    axis = quat[..., 1:] / sin_a_2  # Normalize.
    speed = 2 * np.arctan2(sin_a_2, quat[..., 0:1])
    # When axis-angle is larger than pi, rotation is in opposite direction.
    if speed.shape:
        speed[speed > np.pi] -= 2 * np.pi  # speed is vector.
    elif speed > np.pi:
        speed -= 2 * np.pi  # speed is scalar.
    return speed * axis / dt


def vec_global_to_local(vec, body_quat):
    """Convert vector in global coordinates to body's local reference frame."""
    return rotate_vec_with_quat(vec, reciprocal_quat(body_quat))
