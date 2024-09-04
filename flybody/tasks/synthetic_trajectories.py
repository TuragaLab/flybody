"""Artificial flight and walking trajectories for testing and inference."""

import numpy as np

from dm_control import mujoco

from flybody.quaternions import mult_quat


def constant_speed_trajectory(n_steps: int,
                              speed: float,
                              yaw_speed: float = 0,
                              init_pos: tuple[float] = (0, 0, 0.1278),
                              init_heading: float = 0,
                              body_rot_angle_y: float = 0.,
                              body_rot_angle_x: float = 0.,
                              control_timestep: float = 0.002):
    """Generate a simple constant-speed trajectory, either walking/flying
    straight or turning.
    
    Args:
        n_steps: Number of timesteps.
        speed: Trajectory speed, cm/s.
        yaw_speed: Turning speed, can be zero, rad/s.
            Positive: counter-clockwise turning about z-axis.
        init_pos: Initial xyz position.
        init_heading: Angle of initial fly heading, rad.
        body_rot_angle_y: Body rotation angle around y-axis, degrees.
            Zero: horizontal body. Positive: nose down.
        body_rot_angle_x: Body rotation angle around x-axis, degrees.
            Zero horizontal wing plane. Positive: right bank.
        control_timestep: Environment control timestep, seconds.
        
    Returns:
        qpos: Full trajectory qpos, including quaternion, (n_steps, 7).
        qvel: Full qvel, including quaternion velocity, (n_steps, 6).
    """
    
    qpos = np.zeros((n_steps, 7))
    qvel = np.zeros((n_steps, 6))
    qpos[0, :3] = init_pos
    qpos[:, 2] = init_pos[2]  # Fixed height.
    y_angle = np.deg2rad(body_rot_angle_y)  # Rotation angle around y-axis.
    x_angle = np.deg2rad(body_rot_angle_x)  # Rotation angle around x-axis.
    # Possible initial y-rotation.
    qpos[0, 3:] = [np.cos(y_angle/2), 0., np.sin(y_angle/2), 0.]
    # Possible initial x-rotation.
    qpos[0, 3:] = mult_quat(
        np.array([np.cos(x_angle/2), np.sin(x_angle/2), 0., 0]), qpos[0, 3:])
    # Possible initial z-rotation (heading).
    dquat = np.array(
        [np.cos(init_heading/2), 0, 0, np.sin(init_heading/2)])
    qpos[0, 3:] = mult_quat(dquat, qpos[0, 3:])
    qvel[0, :2] = speed * np.array(
        [np.cos(init_heading), np.sin(init_heading)])
    # Quat velocity.
    dtheta = yaw_speed * control_timestep
    dquat = np.array([np.cos(dtheta/2), 0, 0, np.sin(dtheta/2)])
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, dquat, 1)
    qvel[:, 3:] = vel

    M = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                  [np.sin(dtheta), np.cos(dtheta)]])
    for i in range(1, n_steps):
        qvel[i, :2] = M @ qvel[i-1, :2]
        qpos[i, :2] = qpos[i-1, :2] + qvel[i, :2] * control_timestep
        qpos[i, 3:] = mult_quat(dquat, qpos[i-1, 3:])

    return qpos, qvel
