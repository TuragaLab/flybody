"""MuJoCo's ellipsoid fluid force model, rewritten in python.

The original C code for passive forces, including fluid forces, is here:
https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_passive.c
"""

import numpy as np

from dm_control import mujoco
from dm_control import mjcf

mjNFLUID = 12
mjMINVAL = 1e-15


def ellipsoid_fluid_forces(
    physics: mjcf.Physics,
    ) -> tuple[dict[str, dict[int, dict[str, np.ndarray]]], np.ndarray]:
    """For current `physics` state, calculate individual force and torque
    components of the ellipsoid fluid model. The inertia-based fluid model is ignored.

    See the MuJoCo docs for more detail:
    https://mujoco.readthedocs.io/en/stable/computation/fluid.html#ellipsoid-model
    And the original C code:
    https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_passive.c

    For reference, forces and torques and fluidcoefs controlling them (if any):
    
    Forces:
        fA: added mass, no fluidcoef.
        fD: viscous drag, fluidcoef[0], fluidcoef[1]
        fM: Magnus force, fluidcoef[4].
        fK: Kutta lift, fluidcoef[3].
        fV: viscous resistance, no fluidcoef.

    Torques:
        gD: viscous drag, fluidcoef[1], fluidcoef[2].
        gV: viscous resistance, no fluidcoef.

    Args:
        physics: A Physics instance. No in-place changes will be done to
            this physics instance.

    Returns:
        fluid_forces: Ellipsoid fluid force components for all bodies for which
            the ellipsoid model applies. All forces are in global coordinates.
            Format:
                dict[body_name: dict[geom_id: dict['fA': 3-vec, 'fD': 3-vec, ...]]]
        qfrc_fluid: Calculated mjData.qfrc_fluid. Only includes contributions
            from the ellipsoid fluid model, the inertia fluid model is ignored.
    """

    physics = physics.copy(share_model=True)
    m = physics.model
    d = physics.data
    d.qfrc_fluid[:] = 0.  # Clean up before calculation.

    # Results for all bodies will be stored in this dict.
    fluid_forces = {}

    for i in range(m.nbody):

        use_ellipsoid_model = False
        
        for j in range(m.body_geomnum[i]):
            geomid = m.body_geomadr[i] + j
            if m.geom_fluid[geomid][0]:
                use_ellipsoid_model = True
                
        if use_ellipsoid_model:
            # The returned forces are vectors fA, fD, fM, fK, fV, gA, gD, gV
            # in global coordinates.
            _fluid_forces_for_body_geoms = mj_ellipsoidFluidModel(m, d, i)
            # Store results.
            body_name = m.id2name(i, 'body')
            fluid_forces[body_name] = _fluid_forces_for_body_geoms

    return fluid_forces, physics.data.qfrc_fluid


def mji_ellipsoid_max_moment(size, dir):
    d0 = size[dir]
    d1 = size[(dir+1) % 3]
    d2 = size[(dir+2) % 3]
    return 8.0 / 15.0 * np.pi * d0 * (np.max((d1, d2)))**4


def mj_addedMassForces(local_vels, local_accels, fluid_density,
                       virtual_mass, virtual_inertia, local_force):
    lin_vel = local_vels[3:]
    ang_vel = local_vels[:3]
    virtual_lin_mom = fluid_density * virtual_mass * lin_vel
    virtual_ang_mom = fluid_density * virtual_inertia * ang_vel

    # Here, a block of the original C code skips a calculation that depends on
    # acceleration `qacc` because passive forces don't involve acceleration.
    # // disabled due to dependency on qacc but included for completeness

    added_mass_force = np.cross(virtual_lin_mom, ang_vel)
    added_mass_torque1 = np.cross(virtual_lin_mom, lin_vel)
    added_mass_torque2 = np.cross(virtual_ang_mom, ang_vel)

    local_force[:3] += added_mass_torque1  # This is first term of g_A.
    local_force[:3] += added_mass_torque2  # This is second term of g_A.
    local_force[3:] += added_mass_force  # This is f_A.

    # Return results (python addition to C code).
    fA = added_mass_force
    gA = added_mass_torque1 + added_mass_torque2
    return fA, gA


def mj_viscousForces(local_vels, fluid_density, fluid_viscosity, size,
                     magnus_lift_coef, kutta_lift_coef, blunt_drag_coef,
                     slender_drag_coef, ang_drag_coef, local_force):

    lin_vel = local_vels[3:]
    ang_vel = local_vels[:3]
    volume = 4.0 / 3.0 * np.pi * size[0] * size[1] * size[2]
    d_max = np.max(size)
    d_min = np.min(size)
    d_mid = size[0] + size[1] + size[2] - d_max - d_min
    A_max = np.pi * d_max * d_mid

    # This is f_M.
    magnus_force = np.cross(ang_vel, lin_vel)
    magnus_force *= magnus_lift_coef * fluid_density * volume

    # The dot product between velocity and the normal to the cross-section that
    # defines the body's projection along velocity is proj_num/sqrt(proj_denom)
    proj_denom = (
        ((size[1] * size[2])**4) * (lin_vel[0]**2) +
        ((size[2] * size[0])**4) * (lin_vel[1]**2) +
        ((size[0] * size[1])**4) * (lin_vel[2]**2)
    )
    proj_num = (
        (size[1] * size[2] * lin_vel[0])**2 +
        (size[2] * size[0] * lin_vel[1])**2 +
        (size[0] * size[1] * lin_vel[2])**2
    )
    # Projected surface in the direction of the velocity.
    A_proj = np.pi * np.sqrt(proj_denom / np.max((mjMINVAL, proj_num)))

    # Not-unit normal to ellipsoid's projected area in the direction of velocity.
    norm = np.array([
        (size[1] * size[2])**2 * lin_vel[0],
        (size[2] * size[0])**2 * lin_vel[1],
        (size[0] * size[1])**2 * lin_vel[2],
    ])

    # Cosine between velocity and normal to the surface divided by proj_denom
    # instead of sqrt(proj_denom) to account for skipped normalization in norm.
    cos_alpha = proj_num / np.max((mjMINVAL, np.linalg.norm(lin_vel) * proj_denom))
    kutta_circ = np.cross(norm, lin_vel)
    kutta_circ *= kutta_lift_coef * fluid_density * cos_alpha * A_proj
    kutta_force = np.cross(kutta_circ, lin_vel)  # This is f_K.

    # Viscous force and torque in Stokes flow, analytical for spherical bodies.
    eq_sphere_D = 2.0 / 3.0 * (size[0] + size[1] + size[2])
    lin_visc_force_coef = 3.0 * np.pi * eq_sphere_D
    lin_visc_torq_coef = np.pi * eq_sphere_D * eq_sphere_D * eq_sphere_D

    # Moments of inertia used to compute angular quadratic drag.
    I_max = 8.0 / 15.0 * np.pi * d_mid * d_max**4
    II = np.array([
        mji_ellipsoid_max_moment(size, 0),
        mji_ellipsoid_max_moment(size, 1),
        mji_ellipsoid_max_moment(size, 2),
    ])
    mom_visc = np.array([
        ang_vel[0] * (ang_drag_coef*II[0] + slender_drag_coef*(I_max - II[0])),
        ang_vel[1] * (ang_drag_coef*II[1] + slender_drag_coef*(I_max - II[1])),
        ang_vel[2] * (ang_drag_coef*II[2] + slender_drag_coef*(I_max - II[2])),
    ])

    # Linear plus quadratic.
    drag_lin_coef = (
        fluid_viscosity * lin_visc_force_coef + 
        fluid_density * np.linalg.norm(lin_vel) * (
            A_proj * blunt_drag_coef + slender_drag_coef * (A_max - A_proj)
        )
    )
    # Linear plus quadratic.
    drag_ang_coef = (
        fluid_viscosity * lin_visc_torq_coef + fluid_density * np.linalg.norm(mom_visc)
    )

    # local_force[:3] is (g_D + g_V).
    # local_force[3:] is ... + ... - (f_D + f_V).
    local_force[0] -= drag_ang_coef * ang_vel[0]
    local_force[1] -= drag_ang_coef * ang_vel[1]
    local_force[2] -= drag_ang_coef * ang_vel[2]
    local_force[3] += magnus_force[0] + kutta_force[0] - drag_lin_coef*lin_vel[0]
    local_force[4] += magnus_force[1] + kutta_force[1] - drag_lin_coef*lin_vel[1]
    local_force[5] += magnus_force[2] + kutta_force[2] - drag_lin_coef*lin_vel[2]

    # Return results.
    fM = magnus_force
    fK = kutta_force
    fD = - fluid_density * np.linalg.norm(lin_vel) * (
        A_proj*blunt_drag_coef + slender_drag_coef*(A_max - A_proj)) * lin_vel
    fV = - fluid_viscosity * lin_visc_force_coef * lin_vel
    assert np.all(np.isclose(fD + fV, - drag_lin_coef * lin_vel))
    
    gD = - fluid_density * np.linalg.norm(mom_visc) * ang_vel
    gV = - fluid_viscosity * lin_visc_torq_coef * ang_vel
    assert np.all(np.isclose(gD + gV, - drag_ang_coef * ang_vel))
    
    return fM, fK, fD, fV, gD, gV
    

def mj_ellipsoidFluidModel(m, d, bodyid):

    lvel = np.zeros(6)
    lwind = np.zeros(6)
    bfrc = np.zeros(6)

    _fluid_forces_for_body_geoms = {}
    
    for j in range(m.body_geomnum[bodyid]):
        geomid = m.body_geomadr[bodyid] + j

        # Results for this geom will be stored here.
        _fluid_forces_for_current_geom = {}

        semiaxes = m.geom_size[geomid]

        # void readFluidGeomInteraction
        geom_fluid_coefs = m.geom_fluid[geomid]
        geom_interaction_coef = geom_fluid_coefs[0]
        blunt_drag_coef = geom_fluid_coefs[1]
        slender_drag_coef = geom_fluid_coefs[2]
        ang_drag_coef = geom_fluid_coefs[3]
        kutta_lift_coef = geom_fluid_coefs[4]
        magnus_lift_coef = geom_fluid_coefs[5]
        virtual_mass = geom_fluid_coefs[6:9]
        virtual_inertia = geom_fluid_coefs[9:12]

        if geom_interaction_coef == 0.0:
            continue

        # Map from CoM-centered to local body-centered 6D velocity.
        mujoco.mj_objectVelocity(m.ptr, d.ptr, 5, geomid, lvel, 1)

        # Compute wind in local coordinates.
        wind = np.zeros(6)
        wind[3:] = m.opt.wind
        mujoco.mju_transformSpatial(
            lwind, wind, 0,
            d.geom_xpos[geomid],  # Frame of ref's origin.
            d.subtree_com[m.body_rootid[bodyid]],
            d.geom_xmat[geomid])  # Frame of ref's orientation.
        # Subtract translational component from grom velocity.
        mujoco.mju_subFrom3(lvel[3:], lwind[3:])
        
        # Initialize viscous force and torque
        lfrc = np.zeros(6)
        
        # Added-mass forces and torques.
        fA, gA = mj_addedMassForces(
            lvel, None, m.opt.density, virtual_mass, virtual_inertia, lfrc)
        _fluid_forces_for_current_geom['fA'] = fA
        _fluid_forces_for_current_geom['gA'] = gA
        
        # Lift force orthogonal to lvel from Kutta-Joukowski theorem.
        fM, fK, fD, fV, gD, gV = mj_viscousForces(
            lvel, m.opt.density, m.opt.viscosity, semiaxes, magnus_lift_coef,
            kutta_lift_coef, blunt_drag_coef, slender_drag_coef, ang_drag_coef, lfrc)
        _fluid_forces_for_current_geom['fM'] = fM
        _fluid_forces_for_current_geom['fK'] = fK
        _fluid_forces_for_current_geom['fD'] = fD
        _fluid_forces_for_current_geom['fV'] = fV
        _fluid_forces_for_current_geom['gD'] = gD
        _fluid_forces_for_current_geom['gV'] = gV
        # Transform all forces and torques to global coordinates and scale
        # by geom_interaction_coef.
        for k in _fluid_forces_for_current_geom.keys():
            vec = _fluid_forces_for_current_geom[k] * geom_interaction_coef
            mujoco.mju_mulMatVec3(
                _fluid_forces_for_current_geom[k], d.geom_xmat[geomid], vec)
        
        # Scale by geom_interaction_coef (1.0 by default)
        # mju_scl(lfrc, lfrc, geom_interaction_coef, 6);
        lfrc = lfrc * geom_interaction_coef  # This is the same as mju_scl above.

        # Rotate to global orientation: lfrc -> bfrc
        mujoco.mju_mulMatVec3(bfrc[:3], d.geom_xmat[geomid], lfrc[:3])
        mujoco.mju_mulMatVec3(bfrc[3:], d.geom_xmat[geomid], lfrc[3:])

        # Sanity check.
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        for k, v in _fluid_forces_for_current_geom.items():
            if 'f' in k:
                total_force += v
            else:
                total_torque += v
        assert np.all(np.isclose(total_torque, bfrc[:3]))
        assert np.all(np.isclose(total_force, bfrc[3:]))

        # Apply force and torque to body com.
        mujoco.mj_applyFT(
            m.ptr, d.ptr, bfrc[3:], bfrc[:3],  # model, data, force, torque
            d.geom_xpos[geomid],  # Point where FT is generated
            bodyid, d.qfrc_fluid)

        _fluid_forces_for_body_geoms[geomid] = _fluid_forces_for_current_geom
        
    # Return forces and torques for current body.
    return _fluid_forces_for_body_geoms
