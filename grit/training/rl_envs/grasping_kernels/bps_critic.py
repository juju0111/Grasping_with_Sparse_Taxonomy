"""BPS shape feature tail + privileged critic block — warp kernels of the grasping task."""
import mujoco as _mj  # type: ignore
import numpy as np  # type: ignore
import warp as wp  # type: ignore


@wp.kernel
def _obs_bps_feature_kernel(
    xpos:               wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore (NWORLD, nbody)
    xmat:               wp.array(dtype=wp.mat33, ndim=2),  # type: ignore (NWORLD, nbody)
    transformed_pcd:    wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore (NWORLD, n_pts) world-frame obj PCD
    bps_local:          wp.array(dtype=wp.vec3,  ndim=1),  # type: ignore (M,) hand-center-local basis points
    wrist_body_id:      int,
    hand_center_offset: wp.vec3,                            # rh_hand_center[:3, 3]
    hand_center_R:      wp.mat33,                           # rh_hand_center[:3, :3]
    n_pts:              int,
    gamma:              float,
    mask_below_z:       float,                              # support-surface z: basis point below → feature 0 (masked)
    offset:             int,
    out_obs:            wp.array(dtype=float, ndim=2),     # type: ignore (NWORLD, obs_dim)
):
    """Per-(world, basis-point) BPS feature = exp(-gamma · min-dist to obj PCD).

    2D launch over ``(NWORLD, M)`` — thread ``(w, b)`` owns ONE basis point's
    nearest-PCD scan, so only the ``n_pts`` inner loop is serial (same
    occupancy design as ``_object_grasping_nearest_pcd_kernel``). The basis
    point is composed into world frame via the wrist pose + the constant
    ``rh_hand_center`` offset/rotation (``hc = R_wrist · rh_hand_center``), then
    its distance is measured against the precomputed world-frame object PCD
    ``transformed_pcd`` (no re-transform). ``n_pts == 0`` → feature 0 (no
    object → "far").

    **Below-support mask** (``mask_below_z``): a basis point whose WORLD z falls
    below the support surface (table top; floor when no table) writes feature 0
    — same "nothing there → far" convention as ``n_pts == 0``. The BPS grid is
    rigidly attached to the hand-center and penetrates the table freely, so an
    unmasked below-table point still reports live proximity to the object's
    underside — rewarding hand poses whose graspable-shape evidence lies in an
    UNREACHABLE (colliding) region and pulling the hand into the table. Masking
    makes the observation itself say "below the surface there is nothing to
    grasp". Pass ``-1e9`` (or any z below the scene) to disable."""
    w, b = wp.tid()                                       # type: ignore
    R_wrist = xmat[w, wrist_body_id]
    wrist_p = xpos[w, wrist_body_id]
    hc_p = R_wrist * hand_center_offset + wrist_p          # hand-center origin (world)
    hc_R = R_wrist * hand_center_R                          # hand-center rotation (world)
    bp_world = hc_R * bps_local[b] + hc_p                   # basis point in world

    # Below-support mask: point sits under the table top → feature 0, skip the
    # scan entirely (the region is unreachable — see docstring).
    if bp_world[2] < mask_below_z:
        out_obs[w, offset + b] = 0.0
        return

    best_d2 = float(1.0e18)
    for k in range(n_pts):
        d = transformed_pcd[w, k] - bp_world
        d2 = wp.dot(d, d)
        if d2 < best_d2:
            best_d2 = d2

    feat = float(0.0)
    if n_pts > 0:
        feat = wp.exp(-gamma * wp.sqrt(best_d2))
    out_obs[w, offset + b] = feat


@wp.kernel
def _obs_privileged_gt_kernel(
    xpos:               wp.array(dtype=wp.vec3,            ndim=2),  # type: ignore (NWORLD, nbody)
    xmat:               wp.array(dtype=wp.mat33,           ndim=2),  # type: ignore (NWORLD, nbody)
    cvel:               wp.array(dtype=wp.spatial_vector,  ndim=2),  # type: ignore (NWORLD, nbody) world [ang, lin]
    obj_p_init:         wp.array(dtype=float,              ndim=2),  # type: ignore (NWORLD, 3)
    target_pnt:         wp.array(dtype=float,              ndim=2),  # type: ignore (NWORLD, 3)
    obj_body_id:        int,
    wrist_body_id:      int,
    hand_center_offset: wp.vec3,                                      # rh_hand_center[:3, 3]
    lift_target_m:      float,
    offset:             int,
    out_obs:            wp.array(dtype=float, ndim=2),     # type: ignore (NWORLD, critic_obs_dim)
):
    """Write the privileged GT block into the critic buffer at ``offset`` (20 floats)."""
    w = wp.tid()
    p_obj   = xpos[w, obj_body_id]
    R_obj   = xmat[w, obj_body_id]
    R_wrist = xmat[w, wrist_body_id]
    wrist_p = xpos[w, wrist_body_id]
    RwT     = wp.transpose(R_wrist)

    # obj rotation 6D — world x / y axis columns (6)
    out_obs[w, offset + 0] = R_obj[0, 0]
    out_obs[w, offset + 1] = R_obj[1, 0]
    out_obs[w, offset + 2] = R_obj[2, 0]
    out_obs[w, offset + 3] = R_obj[0, 1]
    out_obs[w, offset + 4] = R_obj[1, 1]
    out_obs[w, offset + 5] = R_obj[2, 1]
    # obj velocity RELATIVE to the wrist rigid frame, in wrist coords (3 + 3).
    # cvel convention as elsewhere (world [ang | lin]); object velocity as seen
    # from a frame fixed to the wrist — translation is corrected for the wrist's
    # own translation plus the rotational (ω×r) component:
    #   v_rel = R_wᵀ·(v_obj − v_w − ω_w×(p_obj − p_w)),  ω_rel = R_wᵀ·(ω_obj − ω_w)
    v_rel = RwT * (wp.spatial_bottom(cvel[w, obj_body_id])
                   - wp.spatial_bottom(cvel[w, wrist_body_id])
                   - wp.cross(wp.spatial_top(cvel[w, wrist_body_id]), p_obj - wrist_p))
    w_rel = RwT * (wp.spatial_top(cvel[w, obj_body_id])
                   - wp.spatial_top(cvel[w, wrist_body_id]))
    out_obs[w, offset + 6]  = v_rel[0]
    out_obs[w, offset + 7]  = v_rel[1]
    out_obs[w, offset + 8]  = v_rel[2]
    out_obs[w, offset + 9]  = w_rel[0]
    out_obs[w, offset + 10] = w_rel[1]
    out_obs[w, offset + 11] = w_rel[2]
    # hand-center → target_pnt vector (world) (3)
    hc_p = R_wrist * hand_center_offset + wrist_p
    out_obs[w, offset + 12] = target_pnt[w, 0] - hc_p[0]
    out_obs[w, offset + 13] = target_pnt[w, 1] - hc_p[1]
    out_obs[w, offset + 14] = target_pnt[w, 2] - hc_p[2]
    # gravity direction in wrist coords (3) — R_wᵀ·(0,0,−1), unit vector
    g_local = RwT * wp.vec3(0.0, 0.0, -1.0)
    out_obs[w, offset + 15] = g_local[0]
    out_obs[w, offset + 16] = g_local[1]
    out_obs[w, offset + 17] = g_local[2]
    # obj lift height + gap to LIFT_TARGET_M (1 + 1)
    lift = p_obj[2] - obj_p_init[w, 2]
    out_obs[w, offset + 18] = lift
    out_obs[w, offset + 19] = lift_target_m - lift
