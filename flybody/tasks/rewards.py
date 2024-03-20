"""Define reward function for imitation tasks."""

from typing import Dict
import numpy as np

from flybody import quaternions


def compute_diffs(walker_features: Dict[str, np.ndarray],
                  reference_features: Dict[str, np.ndarray],
                  n: int = 2) -> Dict[str, float]:
    """Computes sums of absolute values of differences between components of
    model and reference features.

    Args:
        model_features, reference_features: Dictionaries of features to compute
            differences of.
        n: Exponent for differences. E.g., for squared differences use n = 2.

    Returns:
        Dictionary of differences, one value for each entry of input dictionary.
    """
    diffs = {}
    for k in walker_features:
        if 'quat' not in k:
            # Regular vector differences.
            diffs[k] = np.sum(
                np.abs(walker_features[k] - reference_features[k])**n)
        else:
            # Quaternion differences (always positive, no need to use np.abs).
            diffs[k] = np.sum(
                quaternions.quat_dist_short_arc(walker_features[k],
                                                reference_features[k])**n)
    return diffs


def get_walker_features(physics, mocap_joints, mocap_sites):
    """Returns model pose features."""

    qpos = physics.bind(mocap_joints).qpos
    qvel = physics.bind(mocap_joints).qvel
    sites = physics.bind(mocap_sites).xpos
    root2site = quaternions.get_egocentric_vec(qpos[:3], sites, qpos[3:7])

    # Joint quaternions in local egocentric reference frame,
    # (except root quaternion, which is in world reference frame).
    root_quat = qpos[3:7]
    xaxis1 = physics.bind(mocap_joints).xaxis[1:, :]
    xaxis1 = quaternions.rotate_vec_with_quat(
        xaxis1, quaternions.reciprocal_quat(root_quat))
    qpos7 = qpos[7:]
    joint_quat = quaternions.joint_orientation_quat(xaxis1, qpos7)
    joint_quat = np.vstack((root_quat, joint_quat))

    model_features = {
        'com': qpos[:3],
        'qvel': qvel,
        'root2site': root2site,
        'joint_quat': joint_quat,
    }

    return model_features


def get_reference_features(reference_data, step):
    """Returns reference pose features."""

    qpos_ref = reference_data['qpos'][step, :]
    qvel_ref = reference_data['qvel'][step, :]
    root2site_ref = reference_data['root2site'][step, :, :]
    joint_quat_ref = reference_data['joint_quat'][step, :, :]
    joint_quat_ref = np.vstack((qpos_ref[3:7], joint_quat_ref))

    reference_features = {
        'com': reference_data['qpos'][step, :3],
        'qvel': qvel_ref,
        'root2site': root2site_ref,
        'joint_quat': joint_quat_ref,
    }

    return reference_features


def reward_factors_deep_mimic(walker_features,
                              reference_features,
                              std=None,
                              weights=(1, 1, 1, 1)):
    """Returns four reward factors, each of which is a product of individual
    (unnormalized) Gaussian distributions evaluated for the four model
    and reference data features:
        1. Cartesian center-of-mass position, qpos[:3].
        2. qvel for all joints, including the root joint.
        3. Egocentric end-effector vectors.
        4. All joint orientation quaternions (in egocentric local reference
          frame), and the root quaternion.

    The reward factors are equivalent to the ones in the DeepMimic:
    https://arxiv.org/abs/1804.02717
    """
    if std is None:
        # Default values for fruitfly walking imitation task.
        std = {
            'com': 0.078487,
            'qvel': 53.7801,
            'root2site': 0.0735,
            'joint_quat': 1.2247
        }

    diffs = compute_diffs(walker_features, reference_features, n=2)
    reward_factors = []
    for k in walker_features.keys():
        reward_factors.append(np.exp(-0.5 / std[k]**2 * diffs[k]))
    reward_factors = np.array(reward_factors)
    reward_factors *= np.asarray(weights)

    return reward_factors
