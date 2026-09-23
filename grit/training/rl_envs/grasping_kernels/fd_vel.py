"""Finite-difference velocity obs — warp kernels of the grasping task."""
import mujoco as _mj  # type: ignore
import numpy as np  # type: ignore
import warp as wp  # type: ignore
from .common import _grit_r2rpy, _grit_rot_to_rotvec


# ══════════════════════════════════════════════════════════════════════════
# Finite-difference velocity obs (JAX v2 channel layout + FD computation)
# ══════════════════════════════════════════════════════════════════════════
# Restores the 12-channel velocity obs of JAX ``grit_with_obj_coll_v2`` (wrist
# lin/ang + obj lin/ang RELATIVE, wrist frame) into the actor obs, but computes it
# as a **finite difference against the pose snapshot of the previous control step**
# instead of ``d.cvel``/``d.qvel`` (a signal reproducible from pose history alone
# at deployment, without simulator velocities). All components are expressed in
# "the previous frame's wrist-local coordinates": translation ``R_w_prevᵀ·Δp/dt``,
# rotation ``log(R_prevᵀ·R_cur)/dt`` (prev body frame axis-angle). The object is a
# relative velocity with the wrist component subtracted, per the JAX convention.


@wp.kernel
def _fd_vel_update_kernel(
    xpos:          wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore (NWORLD, nbody)
    xmat:          wp.array(dtype=wp.mat33, ndim=2),  # type: ignore (NWORLD, nbody)
    wrist_body_id: int,
    obj_body_id:   int,
    inv_dt:        float,                              # 1 / (sim_nstep · timestep)
    clip:          float,                              # per-component |v| cap (m/s, rad/s)
    fd_valid:      wp.array(dtype=int,     ndim=1),   # type: ignore IN/OUT 0 → seed only
    prev_wrist_p:  wp.array(dtype=wp.vec3,  ndim=1),  # type: ignore IN/OUT
    prev_wrist_R:  wp.array(dtype=wp.mat33, ndim=1),  # type: ignore IN/OUT
    prev_obj_p:    wp.array(dtype=wp.vec3,  ndim=1),  # type: ignore IN/OUT
    prev_obj_R:    wp.array(dtype=wp.mat33, ndim=1),  # type: ignore IN/OUT
    out_fd_vel:    wp.array(dtype=float,   ndim=2),   # type: ignore (NWORLD, 12)
    out_fd_obj_vz: wp.array(dtype=float,   ndim=1),   # type: ignore (NWORLD,) obj WORLD z velocity (for the obj_fell done)
):
    """Pose-FD velocity — runs exactly once per control step (reward collect).

    When ``fd_valid[w]==0`` (right after a reset) it writes velocity 0 and only seeds
    prev with the current pose — structurally preventing the teleport from leaking in
    as a spike. Channel order matches the JAX v2 hand_state::

        [0:3]  wrist lin   R_w_prevᵀ·(p_w − p_w_prev)/dt
        [3:6]  wrist ang   log(R_w_prevᵀ·R_w)/dt              (prev wrist frame)
        [6:9]  obj lin     R_w_prevᵀ·(p_o − p_o_prev)/dt − wrist_lin
        [9:12] obj ang     R_w_prevᵀ·(R_o_prev·log(R_o_prevᵀ·R_o))/dt − wrist_ang
    """
    w = wp.tid()
    pw = xpos[w, wrist_body_id]
    Rw = xmat[w, wrist_body_id]
    po = xpos[w, obj_body_id]
    Ro = xmat[w, obj_body_id]

    if fd_valid[w] == 0:
        for i in range(12):
            out_fd_vel[w, i] = 0.0
        out_fd_obj_vz[w] = 0.0
        fd_valid[w] = 1
    else:
        Rw0T = wp.transpose(prev_wrist_R[w])
        wrist_lin = Rw0T * (pw - prev_wrist_p[w]) * inv_dt
        wrist_ang = _grit_rot_to_rotvec(Rw0T * Rw) * inv_dt
        obj_lin = Rw0T * (po - prev_obj_p[w]) * inv_dt - wrist_lin
        Ro0 = prev_obj_R[w]
        # obj rotational displacement: obj body-frame rotvec → world → prev wrist frame.
        obj_ang_world = Ro0 * _grit_rot_to_rotvec(wp.transpose(Ro0) * Ro)
        obj_ang = Rw0T * obj_ang_world * inv_dt - wrist_ang
        for i in range(3):
            out_fd_vel[w, 0 + i] = wp.clamp(wrist_lin[i], -clip, clip)
            out_fd_vel[w, 3 + i] = wp.clamp(wrist_ang[i], -clip, clip)
            out_fd_vel[w, 6 + i] = wp.clamp(obj_lin[i],   -clip, clip)
            out_fd_vel[w, 9 + i] = wp.clamp(obj_ang[i],   -clip, clip)
        # obj WORLD z velocity (consumed by the obj_fell done) — cvel picks up
        # contact-solver jitter as instantaneous velocity and overestimates even at
        # rest; FD averages over the control step so the jitter cancels out.
        out_fd_obj_vz[w] = wp.clamp(
            (po[2] - prev_obj_p[w][2]) * inv_dt, -clip, clip)

    prev_wrist_p[w] = pw
    prev_wrist_R[w] = Rw
    prev_obj_p[w]   = po
    prev_obj_R[w]   = Ro


@wp.kernel
def _fd_vel_invalidate_masked_kernel(
    mask:      wp.array(dtype=int,   ndim=1),  # type: ignore (NWORLD,) done mask
    obs_off:   int,                             # obs column offset of the fd_vel block (-1 → skip the obs write)
    fd_valid:  wp.array(dtype=int,   ndim=1),  # type: ignore IN/OUT
    fd_vel:    wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, 12)
    fd_obj_vz: wp.array(dtype=float, ndim=1),  # type: ignore (NWORLD,)
    obs:       wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, obs_dim)
):
    """Invalidate the FD state of per-world-reset worlds and also rewrite the fd_vel
    slice of the already collected obs to 0 (this kernel runs **after** the obs
    re-collection of a per-world reset, so the obs must be cleared too, otherwise the
    pre-teleport velocity would remain in the new episode's first obs). The next step's
    update kernel seeds from the fresh pose."""
    w = wp.tid()
    if mask[w] == 0:
        return
    fd_valid[w] = 0
    fd_obj_vz[w] = 0.0
    for i in range(12):
        fd_vel[w, i] = 0.0
        if obs_off >= 0:
            obs[w, obs_off + i] = 0.0


@wp.kernel
def _obs_fd_vel_kernel(
    fd_vel:  wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, 12)
    offset:  int,
    out_obs: wp.array(dtype=float, ndim=2),  # type: ignore
):
    """Copy the FD velocity buffer into the obs slice (idempotent — safe under obs re-collection)."""
    w = wp.tid()
    for i in range(12):
        out_obs[w, offset + i] = fd_vel[w, i]


@wp.kernel
def _object_grasping_dynamic_target_pnt_kernel(
    best_pt_hc: wp.array(dtype=wp.vec3, ndim=1),  # type: ignore (NWORLD,) PCD vertex nearest to the hand-center (world)
    noise_amp:  float,
    seed:       int,
    out_pnt:    wp.array(dtype=float, ndim=2),    # type: ignore (NWORLD, 3) cond target_pnt gpu
):
    """JAX v2-style dynamic target_pnt — every control step ``target_pnt[w] =
    (obj-PCD vertex nearest to the hand-center) + U(-noise, noise)³``.

    Replaces the reset-snapshot (frozen) approach: it reuses the ``best_pt_hc``
    already computed by the reward pipeline, so there is no extra scan cost. Since
    the cond field is overwritten, all consumers (the async critic GT's
    hand-center→target_pnt vector, eval markers) automatically see the dynamic target."""
    w = wp.tid()
    base = w * 4
    nx = (wp.randf(wp.rand_init(seed, base + 0)) * 2.0 - 1.0) * noise_amp
    ny = (wp.randf(wp.rand_init(seed, base + 1)) * 2.0 - 1.0) * noise_amp
    nz = (wp.randf(wp.rand_init(seed, base + 2)) * 2.0 - 1.0) * noise_amp
    p = best_pt_hc[w]
    out_pnt[w, 0] = p[0] + nx
    out_pnt[w, 1] = p[1] + ny
    out_pnt[w, 2] = p[2] + nz


@wp.kernel
def _object_grasping_finger_tax_pull_kernel(
    mask:        wp.array(dtype=int,   ndim=1),  # type: ignore (NWORLD,) worlds reset this step
    target_qpos: wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, n_ctrl) freshly drawn taxonomy template
    finger_qpa:  wp.array(dtype=int,   ndim=1),  # type: ignore (n_ctrl,) ctrl→qpos adr (-1 = tendon)
    n_ctrl:      int,
    prob:        float,                           # per-world pull probability
    ratio_min:   float,
    ratio_max:   float,
    seed:        int,
    qpos:        wp.array(dtype=float, ndim=2),  # type: ignore IN/OUT
    ctrl:        wp.array(dtype=float, ndim=2),  # type: ignore IN/OUT (servo setpoint sync)
):
    """Finger init taxonomy-pull (port of the JAX v2 ``_finger_init_fn`` template block).

    For each reset world, with probability ``prob``::

        ratio      ~ U[ratio_min, ratio_max]
        qpos[qa]  += ratio · (template − qpos[qa])
        ctrl[i]    = qpos[qa]                    # sync so the servo does not pull it back

    Partially interpolates from the uniform initial pose toward that episode's
    taxonomy template — exposing near-template states as a curriculum so the combined
    mimic/contact reward is discovered early. target_qpos must be the value **after**
    the taxonomy overlay, so this is called as the last stage of
    ``_post_cond_update_hook`` (uniform-fallback worlds are harmlessly pulled toward
    the uniformly resampled target — still a valid pose)."""
    w = wp.tid()
    if mask[w] == 0:
        return
    base = w * 2
    if wp.randf(wp.rand_init(seed, base + 0)) >= prob:
        return
    ratio = ratio_min + wp.randf(wp.rand_init(seed, base + 1)) * (ratio_max - ratio_min)
    for i in range(n_ctrl):
        qa = finger_qpa[i]
        if qa < 0:
            continue
        q_new = qpos[w, qa] + ratio * (target_qpos[w, i] - qpos[w, qa])
        qpos[w, qa] = q_new
        ctrl[w, i]  = q_new


@wp.kernel
def _obs_site_pcd_err_kernel(
    xmat:            wp.array(dtype=wp.mat33, ndim=2),  # type: ignore
    site_xpos:       wp.array(dtype=wp.vec3, ndim=2),   # type: ignore
    best_pt:         wp.array(dtype=wp.vec3, ndim=2),   # type: ignore
    finger_site_ids: wp.array(dtype=int,    ndim=1),    # type: ignore
    wrist_body_id:   int,
    n_sensors:       int,
    arm_sensor_idx:  int,
    offset:          int,
    out_obs:         wp.array(dtype=float, ndim=2),     # type: ignore
):
    """Per-site -> nearest obj-PCD vertex error in wrist frame (fore-arm excl.).
    Output index closes the gap left by the skipped arm sensor."""
    w = wp.tid()
    Rw_T = wp.transpose(xmat[w, wrist_body_id])
    for s in range(n_sensors):
        if s != arm_sensor_idx:
            oj = s
            if s > arm_sensor_idx:
                oj = s - 1
            err = Rw_T * (best_pt[w, s] - site_xpos[w, finger_site_ids[s]])
            out_obs[w, offset + oj * 3 + 0] = err[0]
            out_obs[w, offset + oj * 3 + 1] = err[1]
            out_obs[w, offset + oj * 3 + 2] = err[2]


@wp.kernel
def _obs_hand_center_pcd_err_kernel(
    xpos:               wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore
    xmat:               wp.array(dtype=wp.mat33, ndim=2),  # type: ignore
    best_pt_hc:         wp.array(dtype=wp.vec3,  ndim=1),  # type: ignore (NWORLD,)
    wrist_body_id:      int,
    hand_center_offset: wp.vec3,                            # rh_hand_center[:3, 3]
    offset:             int,
    out_obs:            wp.array(dtype=float, ndim=2),     # type: ignore
):
    """hand-center -> nearest obj-PCD vertex error in the wrist frame (3 floats).

    Same wrist-frame convention as :func:`_obs_site_pcd_err_kernel`, but the
    origin is the hand-center (``hc_p = R_wrist · offset + wrist_p``) instead of a
    finger site, and the target is the hand-center's nearest vertex from
    :func:`_object_grasping_nearest_pcd_hand_center_kernel`."""
    w = wp.tid()
    R_wrist = xmat[w, wrist_body_id]
    wrist_p = xpos[w, wrist_body_id]
    hc_p = R_wrist * hand_center_offset + wrist_p
    err = wp.transpose(R_wrist) * (best_pt_hc[w] - hc_p)
    out_obs[w, offset + 0] = err[0]
    out_obs[w, offset + 1] = err[1]
    out_obs[w, offset + 2] = err[2]


@wp.kernel
def _obs_rot6d_kernel(
    xmat:          wp.array(dtype=wp.mat33, ndim=2),  # type: ignore
    target_R:      wp.array(dtype=float,    ndim=3),  # type: ignore
    hand_center_R: wp.mat33,
    wrist_body_id: int,
    offset:        int,
    out_obs:       wp.array(dtype=float, ndim=2),     # type: ignore
):
    """6D rotation (x,y axes) of hand-center(world) + frozen target_R(world)."""
    w = wp.tid()
    R_hc = xmat[w, wrist_body_id] * hand_center_R
    hc_x = R_hc * wp.vec3(1.0, 0.0, 0.0)
    hc_y = R_hc * wp.vec3(0.0, 1.0, 0.0)
    out_obs[w, offset + 0] = hc_x[0] - target_R[w, 0, 0]
    out_obs[w, offset + 1] = hc_x[1] - target_R[w, 0, 1]
    out_obs[w, offset + 2] = hc_x[2] - target_R[w, 0, 2]
    out_obs[w, offset + 3] = hc_y[0] - target_R[w, 1, 0]
    out_obs[w, offset + 4] = hc_y[1] - target_R[w, 1, 1]
    out_obs[w, offset + 5] = hc_y[2] - target_R[w, 1, 2]
    # out_obs[w, offset + 6]  = target_R[w, 0, 0]
    # out_obs[w, offset + 7]  = target_R[w, 1, 0]
    # out_obs[w, offset + 8]  = target_R[w, 2, 0]
    # out_obs[w, offset + 9]  = target_R[w, 0, 1]
    # out_obs[w, offset + 10] = target_R[w, 1, 1]
    # out_obs[w, offset + 11] = target_R[w, 2, 1]


@wp.kernel
def _obs_site_z_above_table_kernel(
    site_xpos:       wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
    finger_site_ids: wp.array(dtype=int,    ndim=1),   # type: ignore
    n_sensors:       int,
    table_height:    float,
    offset:          int,
    out_obs:         wp.array(dtype=float, ndim=2),    # type: ignore
):
    """Per-site z height above the table (global): site_xpos.z - table_height."""
    w = wp.tid()
    for s in range(n_sensors):
        sp = site_xpos[w, finger_site_ids[s]]
        out_obs[w, offset + s] = sp[2] - table_height


@wp.kernel
def _obs_obj_pose_diff_kernel(
    xpos:        wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore
    xmat:        wp.array(dtype=wp.mat33, ndim=2),  # type: ignore
    obj_p_init:  wp.array(dtype=float,    ndim=2),  # type: ignore
    obj_R_init:  wp.array(dtype=float,    ndim=3),  # type: ignore
    obj_body_id: int,
    offset:      int,
    out_obs:     wp.array(dtype=float, ndim=2),     # type: ignore
):
    """Object pose vs spawn: p_diff(world xyz) + rpy_diff(R_init^T . R_obj)."""
    w = wp.tid()
    obj_p = xpos[w, obj_body_id]
    out_obs[w, offset + 0] = obj_p_init[w, 0] - obj_p[0] 
    out_obs[w, offset + 1] = obj_p_init[w, 1] - obj_p[1]
    out_obs[w, offset + 2] = obj_p_init[w, 2] - obj_p[2]
    R_init = wp.mat33(
        obj_R_init[w, 0, 0], obj_R_init[w, 0, 1], obj_R_init[w, 0, 2],
        obj_R_init[w, 1, 0], obj_R_init[w, 1, 1], obj_R_init[w, 1, 2],
        obj_R_init[w, 2, 0], obj_R_init[w, 2, 1], obj_R_init[w, 2, 2],
    )
    rpy = _grit_r2rpy(wp.transpose(R_init) * xmat[w, obj_body_id])  # type: ignore 
    out_obs[w, offset + 3] = rpy[0]
    out_obs[w, offset + 4] = rpy[1]
    out_obs[w, offset + 5] = rpy[2]


@wp.kernel
def _obs_ep_stage_kernel(
    ep_step:      wp.array(dtype=int, ndim=1),    # type: ignore
    lift_step:    int,
    hold_step:    int,
    max_ep_steps: int,
    offset:       int,
    out_obs:      wp.array(dtype=float, ndim=2),  # type: ignore
    out_stage:    wp.array(dtype=int,   ndim=1),  # type: ignore (NWORLD,) curriculum stage 0/1/2
):
    """ep_step_norm(1) + stage **one-hot**(3) curriculum signals (block width 4).

    The stage is one-hot encoded instead of a scalar 0/1/2 — this does not force an
    ordinal (0<1<2) assumption onto the MLP (discrete channels, hence obs_norm
    passthrough). Also writes the per-world stage (0/1/2) to a dedicated
    ``out_stage`` buffer so consumers outside the observation (e.g. the
    rule-based wrist-lift in the action applier) can read it directly.
    """
    w = wp.tid()
    denom = float(max_ep_steps)
    if denom < 1.0:
        denom = 1.0
    out_obs[w, offset] = float(ep_step[w]) / denom
    s = int(0)
    if ep_step[w] >= hold_step:
        s = 2
    elif ep_step[w] >= lift_step:
        s = 1
    for i in range(3):
        v = float(0.0)
        if i == s:
            v = 1.0
        out_obs[w, offset + 1 + i] = v
    out_stage[w] = s


@wp.kernel
def _stage_only_kernel(
    ep_step:   wp.array(dtype=int, ndim=1),    # type: ignore
    lift_step: int,
    hold_step: int,
    out_stage: wp.array(dtype=int, ndim=1),    # type: ignore (NWORLD,) curriculum stage 0/1/2
):
    """Compute the per-world curriculum stage (0/1/2) into ``out_stage`` ONLY —
    no observation write. Used when ``INCLUDE_EP_STAGE_IN_OBS`` is False (the
    actor is time-blind) but ``stage_wp`` is still required by the rule-based
    wrist-lift action pre-processor. Mirrors the stage half of
    :func:`_obs_ep_stage_kernel`."""
    w = wp.tid()
    s = int(0)
    if ep_step[w] >= hold_step:
        s = 2
    elif ep_step[w] >= lift_step:
        s = 1
    out_stage[w] = s


@wp.kernel
def _wrist_translation_slow_kernel(
    action:        wp.array(dtype=float,   ndim=2),  # type: ignore (NWORLD, action_dim) — modified IN PLACE
    ep_step:       wp.array(dtype=int,     ndim=1),  # type: ignore (NWORLD,)
    xpos:          wp.array(dtype=wp.vec3, ndim=2),  # type: ignore (NWORLD, nbody) — d.xpos
    xmat:          wp.array(dtype=wp.mat33, ndim=2), # type: ignore (NWORLD, nbody) — for composing the hand-center
    hand_center_offset: wp.vec3,                     # rh_hand_center[:3, 3] (wrist frame)
    obj_body_id:   int,
    wrist_body_id: int,
    xyz_off:       int,
    lift_step:     int,                              # slowdown ONLY while ep_step < lift_step (approach/grasp)
    near_m:        float,                            # d <= near → full slowdown (SLOW_FACTOR)
    far_m:         float,                            # d >= far → no slowdown (1.0); linear ramp between
    slow_factor:   float,                            # min wrist-xyz scale near the object (∈(0,1])
    xyz_scale:     float,                            # applier.xyz_scale (to measure the deadzone in metric units)
    use_deadzone:  int,                              # 1 → apply the deadzone to the command 'before' slowdown
    xyz_deadzone_m: float,                           # pre-slow deadzone threshold (m)
):
    """Manipulation-precision wrist-translation slowdown (action pre-processor).

    **Just before / at the moment of grasping** (small hand-object distance) the
    magnitude of the wrist translation action (Δxyz) is reduced down to
    ``slow_factor``, making the approach/grasp precise and slow. In free space (far
    away) it is 1.0 (full speed). **From the lift stage on (``ep_step >= lift_step``)
    always full speed** — lifting moves at the original speed.

    Linear ramp on the distance ``d = |hand_center − obj|``: ``d>=far→1.0``,
    ``d<=near→slow_factor``. hand_center = R_wrist·rh_hand_center_offset + wrist_p —
    same convention as the approach/done kernels. (An earlier version used the wrist
    body; for some hands the wrist↔obj center distance never drops below ~0.16 m even
    while grasping, so with FAR≤0.2 the ramp rarely or never fired. The hand-center↔obj
    center distance is ≈0.05~0.09 m while grasping, so the NEAR/FAR scale is meaningful.)
    action[xyz] is scaled by s in place (the apply kernel applies Δxyz·xyz_scale, so
    this is equivalent to a dynamic per-world xyz_scale). Runs before
    ``_stage_wrist_lift_kernel`` and the stage gates make them mutually exclusive, so
    there is no conflict with the lift override."""
    w = wp.tid()
    # ── Pre-slow deadzone (when use_deadzone): the apply kernel's deadzone measures
    # the magnitude 'after' slowdown, so in the near zone (e.g. factor 0.3 × 12.5mm =
    # 3.75mm) the maximum displacement falls entirely below the deadzone (10mm) and
    # the wrist freezes. Here the decision is made first on the **pre-slowdown
    # command** (the correct criterion for intent to stop): sub-deadzone commands
    # become 0, surviving commands are scaled by s. The double cut on the apply
    # kernel side is prevented by ``_apply_action_curriculum`` lowering
    # applier.xyz_deadzone_m to dz·slow_factor (guaranteeing a post-slow magnitude
    # ≥ dz·s for surviving commands).
    ax = action[w, xyz_off + 0]
    ay = action[w, xyz_off + 1]
    az = action[w, xyz_off + 2]
    if use_deadzone == 1:
        nrm = wp.sqrt(ax * ax + ay * ay + az * az) * xyz_scale
        if nrm < xyz_deadzone_m:
            action[w, xyz_off + 0] = 0.0
            action[w, xyz_off + 1] = 0.0
            action[w, xyz_off + 2] = 0.0
            return
    if ep_step[w] >= lift_step:      # lift/hold stage → keep full speed (the full deadzone was applied above)
        return
    hc_p = xmat[w, wrist_body_id] * hand_center_offset + xpos[w, wrist_body_id]
    d = wp.length(hc_p - xpos[w, obj_body_id])
    s = float(1.0)
    if d <= near_m:
        s = slow_factor
    elif d < far_m:
        t = (d - near_m) / (far_m - near_m)          # 0 at near → 1 at far
        s = slow_factor + t * (1.0 - slow_factor)
    # else d >= far_m → s = 1.0 (full speed)
    action[w, xyz_off + 0] = ax * s
    action[w, xyz_off + 1] = ay * s
    action[w, xyz_off + 2] = az * s


@wp.kernel
def _stage_action_jax_gate_kernel(
    action:        wp.array(dtype=float,    ndim=2),  # type: ignore (NWORLD, action_dim) — modified IN PLACE
    ep_step:       wp.array(dtype=int,      ndim=1),  # type: ignore (NWORLD,)
    xmat:          wp.array(dtype=wp.mat33, ndim=2),  # type: ignore (NWORLD, nbody) — R_wrist (world z → wrist-local)
    xpos:          wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore (NWORLD, nbody) — obj z (height gate in integrate mode)
    obj_p_init:    wp.array(dtype=float,    ndim=2),  # type: ignore (NWORLD, 3) — spawn obj pos
    obj_body_id:   int,
    lift_target_m: float,                              # lift cap in integrate mode (LIFT_TARGET_M)
    wrist_body_id: int,
    lift_step:     int,
    hold_step:     int,
    xyz_off:       int,
    rot6d_off:     int,
    finger_off:    int,
    n_ctrl:        int,
    xyz_scale:     float,                              # applier.xyz_scale (m → action units)
    lift_dist_m:   float,                              # LIFT_ACTION_DIST_M (JAX 0.01)
    ramp_steps:    float,                              # LIFT_RAMP_STEPS (JAX 5)
    ramp_rate:     float,                              # LIFT_RAMP_RATE (JAX 0.25)
    min_dist_m:    float,                              # floor guarding against the deadzone (LIFT_MIN_DIST_M; 0 = pure ramp)
    finger_slow:   float,                              # finger Δ multiplier in lift/hold (JAX ÷5 = 0.2)
    hold_lock:     int,                                # 1 = keep the mocap target in hold/rotation (STAGE_TARGET_HOLD)
    lift_integrate: int,                               # 1 = manual lift as target integration (LIFT_TARGET_INTEGRATE)
    wrist_rise_cap_m: float,                           # >0: stop the rule lift once the WRIST rose this much since lift_step (then hold)
    wrist_z_lift0: wp.array(dtype=float, ndim=1),      # type: ignore (NWORLD,) IN/OUT — wrist z at lift_step (written here)
    pos_hold:      wp.array(dtype=int, ndim=1),        # type: ignore (NWORLD,) OUT — apply kernel wrist_pos_hold
    rot_hold:      wp.array(dtype=int, ndim=1),        # type: ignore (NWORLD,) OUT — apply kernel wrist_rot_hold
):
    """Original JAX v2 stage action masking (STAGE_ACTION_JAX_MODE=true only).

    Reproduces the stage rules of the taxonomy_guided_RL v2 ``step()`` as is —
    unlike the default Grit mode (`_stage_wrist_lift_kernel`: manual lift 'added' to
    the policy Δxyz, with curriculum decay and latch):

      * **lift stage** (lift_step ≤ ep < hold_step):
          - wrist Δxyz is **taken from the policy** → replaced by a world +z manual
            lift (``exp(-rate·max(0, ramp−(ep−lift)))·dist``; JAX 0.25/5/0.01 —
            **no** object-height gate, latch or training decay; rule-based throughout)
          - wrist rotation **frozen** (rot6d residual = 0 → identity)
          - finger Δ ×``finger_slow`` (JAX ÷5)
      * **hold stage** (ep ≥ hold_step): wrist Δxyz = 0 (frozen; the JAX 0.5mm xy
        noise is omitted — pointless as a solver-jitter device), rotation freeze and
        finger slow kept.
      * approach (ep < lift_step): full policy control (no-op).

    The manual lift must pass the apply kernel's xyz deadzone, so it is guarded by the
    ``min_dist_m`` floor (yaml LIFT_MIN_DIST_M=0.01 ≥ deadzone). Exclusive with the
    Grit mode — ``_apply_action_curriculum`` branches on the flag."""
    w = wp.tid()
    if ep_step[w] < lift_step:
        return
    if ep_step[w] == lift_step:
        wrist_z_lift0[w] = xpos[w, wrist_body_id][2]      # reference for the wrist-rise cap
    # rotation freeze + finger slow (shared by lift and hold, JAX convention)
    for i in range(6):
        action[w, rot6d_off + i] = 0.0
    for i in range(n_ctrl):
        action[w, finger_off + i] = action[w, finger_off + i] * finger_slow
    # keep the rotation target: residual=0 would re-anchor to the measured rotation
    # (ratchet drift), so the write itself is skipped and the weld holds the previous target.
    if hold_lock == 1:
        rot_hold[w] = 1
    # wrist Δxyz: taken from the policy
    if ep_step[w] >= hold_step:
        action[w, xyz_off + 0] = 0.0
        action[w, xyz_off + 1] = 0.0
        action[w, xyz_off + 2] = 0.0
        # hold: keep the previous target (the last lift target) — the weld supports
        # the final lift height against gravity (prevents re-anchoring sag).
        if hold_lock == 1:
            pos_hold[w] = 1
        return
    # lift: manual lift as target integration — without the servo lag of a
    # measured-anchor (~2.4mm realized per 10mm commanded), unrealized commands
    # accumulate in the target and the weld keeps pulling (the z-lead cap is guarded
    # by the apply kernel). Integrate mode requires a height gate: the original JAX
    # time-based lift (40 steps) was naturally braked by the lag, but integration
    # realizes ~6.6mm/step and would overshoot the success band
    # (LIFT_TARGET·(1±TOL)) if left alone. Once reached, switch to holding the
    # target — "lift to 10cm and support that height".
    if lift_integrate == 1:
        lift_z = xpos[w, obj_body_id][2] - obj_p_init[w, 2]
        done_lift = lift_z >= lift_target_m
        # wrist-rise cap (opt-in): the object did not follow (slip / no grasp) — do not
        # keep driving the wrist up; freeze it where it is instead of climbing to the
        # end of the lift window.
        if wrist_rise_cap_m > 0.0:
            if xpos[w, wrist_body_id][2] - wrist_z_lift0[w] >= wrist_rise_cap_m:
                done_lift = True
        if done_lift:
            action[w, xyz_off + 0] = 0.0
            action[w, xyz_off + 1] = 0.0
            action[w, xyz_off + 2] = 0.0
            if hold_lock == 1:
                pos_hold[w] = 1
            return
        pos_hold[w] = 2
    # lift-stage manual lift (world +z, soft ramp; no decay or height gate)
    t = float(ep_step[w] - lift_step)
    gap = ramp_steps - t
    if gap < 0.0:
        gap = 0.0
    lift = lift_dist_m * wp.exp(-ramp_rate * gap)
    if lift < min_dist_m:
        lift = min_dist_m
    inv_scale = 1.0 / xyz_scale
    dz_local = wp.transpose(xmat[w, wrist_body_id]) * wp.vec3(0.0, 0.0, lift)
    action[w, xyz_off + 0] = dz_local[0] * inv_scale
    action[w, xyz_off + 1] = dz_local[1] * inv_scale
    action[w, xyz_off + 2] = dz_local[2] * inv_scale


@wp.kernel
def _stage_wrist_lift_kernel(
    action:        wp.array(dtype=float,    ndim=2),  # type: ignore (NWORLD, action_dim) — modified IN PLACE
    ep_step:       wp.array(dtype=int,      ndim=1),  # type: ignore (NWORLD,) episode step
    xpos:          wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore (NWORLD, nbody) — current obj z
    xmat:          wp.array(dtype=wp.mat33, ndim=2),  # type: ignore (NWORLD, nbody) — R_wrist
    obj_p_init:    wp.array(dtype=float,    ndim=2),  # type: ignore (NWORLD, 3) — spawn obj pos
    lift_reached:  wp.array(dtype=int,      ndim=1),  # type: ignore (NWORLD,) IN/OUT — per-world latch (1 = target hit once this episode)
    obj_body_id:   int,
    wrist_body_id: int,
    lift_step:     int,
    hold_step:     int,
    xyz_off:       int,
    rot6d_off:     int,                                # 6D-rot action offset — forced to identity for es>=lift_step
    xyz_scale:     float,                              # action→m scale (apply: R_wrist·Δxyz·xyz_scale)
    lift_target_m: float,                              # stop lifting once obj lift_z >= this
    lift_dist_m:   float,                              # base per-step WORLD-z lift (m), decay-scaled
    ramp_steps:    float,                              # soft-start ramp duration (steps)
    ramp_rate:     float,                              # exp ramp rate
    lift_min_m:    float,                              # floor: each ACTIVE lift step moves ≥ this (m); 0 disables
    lift_max_m:    float,                              # over-band cap: obj lift_z above this → strip world +z from the wrist action (≤0 disables)
    hold_lock:     int,                                # 1 = keep the mocap target in latch/hold (STAGE_TARGET_HOLD)
    lift_integrate: int,                               # 1 = manual lift as target integration (LIFT_TARGET_INTEGRATE)
    pos_hold:      wp.array(dtype=int, ndim=1),        # type: ignore (NWORLD,) OUT — apply kernel wrist_pos_hold
    rot_hold:      wp.array(dtype=int, ndim=1),        # type: ignore (NWORLD,) OUT — apply kernel wrist_rot_hold
):
    """Rule-based wrist-z lift (action pre-processor), mirroring the legacy
    JAX manual-lift strategy.

    Fires ONLY in the **lift stage** (``lift_step <= ep_step < hold_step``) and
    ONLY until the object FIRST reaches the target lift
    (``obj_z - obj_z_init >= lift_target_m``). That first hit **latches**
    ``lift_reached[w] = 1`` for the rest of the episode, so the rule never
    lifts again even if gravity later pulls the object back below the target
    — this avoids the wrist juddering up-and-down as the rule re-triggers on
    every small dip below the threshold (the legacy step-only JAX rule had no
    such re-trigger because it ignored the object position entirely). Once
    latched the wrist Δxyz is **frozen at 0** for the rest of the lift window
    (from the latch step itself onward; the hold stage below continues the
    same freeze) — control is NEVER handed back to the policy's wrist
    translation mid-air, only finger control remains. Handing control back
    was the old behaviour and let a policy whose wrist-z channel never
    received real lift-stage credit (its Δxyz is overridden here, yet the
    rollout buffer stores the raw sample) drive the freshly-lifted object
    straight back into the floor. In the approach stage the rule is a no-op
    and the policy keeps full control. The latch resets to 0 on every
    (per-world) episode reset.

    The per-step lift distance ramps in softly:
    ``lift_dist_m · exp(-ramp_rate · max(0, ramp_steps - (ep_step-lift_step)))``
    (legacy ``exp(-0.25·max(0, 5-Δ))·0.01``), then floored at ``lift_min_m`` so
    each ACTIVE lift step moves at least that much (``lift_min_m = 0`` disables
    the floor → pure soft-start ramp). The floor overrides the soft-start ramp
    while it is below ``lift_min_m`` (the first few steps); ``lift_dist_m`` is
    already curriculum-decay-scaled, and ``_apply_action_curriculum`` skips the
    whole launch once it reaches 0, so the floor only applies while the rule is
    active. It is applied as a **world +z** displacement: the raw action's Δxyz
    is OVERRIDDEN with ``R_wrist^T · [0,0, lift_m/xyz_scale]`` so the apply
    kernel's ``R_wrist · (Δxyz·xyz_scale)`` reproduces a true world-frame upward
    motion (lateral Δxyz forced to 0). Overriding — not adding — is what the
    legacy JAX did (``delta_transl_world = lift_value``); adding let the policy's
    own Δxyz cancel the lift so the wrist often failed to rise.

    **Wrist rotation freeze**: for the ENTIRE ``es >= lift_step`` span (both lift
    and hold stages, regardless of the lift latch) the 6 rotation-action
    components at ``rot6d_off`` are zeroed. The apply kernel adds the
    ``[1,0,0,0,1,0]`` identity bias on top of the 6D action, so action-zero
    decodes to an identity ``R_delta`` — the wrist orientation is held fixed and
    the policy can no longer tilt/rotate the wrist once lifting begins. Only the
    +z lift (and the hold-stage Δxyz=0) remains rule-driven; finger ctrl is
    untouched. Modifies ``action`` in place.

    **Over-band lift cap** (``lift_max_m``; ≤ 0 disables): for the whole
    ``es >= lift_step`` span, once the object's lift height
    (``obj_z − obj_z_init``) exceeds ``lift_max_m`` — the caller passes the
    success band's UPPER edge ``LIFT_TARGET_M · (1 + SUCCESS_LIFT_TOL_FRAC)`` —
    the wrist Δxyz currently in the action buffer (whether policy-written or the
    manual override above) has its **world +z component stripped**, so nothing
    can lift the object further past the band. Lateral and DOWNWARD motion stay
    untouched (the policy may need to lower the object back into the band), and
    the check is re-evaluated every step, so control returns in full as soon as
    the object settles back below the cap."""
    w  = wp.tid()
    es = ep_step[w]
    if es >= lift_step:
        # 6D wrist-rotation → identity for the whole lift+hold span. The apply
        # kernel adds the ``[1,0,0,0,1,0]`` identity bias on top of the 6D
        # action, so zeroing the 6 rot components makes the decoded R_delta
        # exactly identity → the wrist orientation is FROZEN (no policy tilt)
        # from lift_step onward, while only the +z lift / hold is rule-driven.
        # TODO: scale the current action according to the curriculum weight.
        action[w, rot6d_off + 0] = 0.0
        action[w, rot6d_off + 1] = 0.0
        action[w, rot6d_off + 2] = 0.0
        action[w, rot6d_off + 3] = 0.0
        action[w, rot6d_off + 4] = 0.0
        action[w, rot6d_off + 5] = 0.0
        # keep the rotation target (blocks measured re-anchoring ratchet drift) — whole lift+hold span.
        if hold_lock == 1:
            rot_hold[w] = 1
        if es < hold_step:                                   # lift stage only (NOT hold)
            if lift_reached[w] == 0:                          # not yet latched this episode
                lift_z = xpos[w, obj_body_id][2] - obj_p_init[w, 2]
                if lift_z >= lift_target_m:                  # FIRST time at target → latch + FREEZE
                    lift_reached[w] = 1
                    # Freeze starts on the latch step itself — the policy's own
                    # Δxyz must never execute mid-air (a policy whose wrist-z
                    # channel never got real lift-stage credit can output a
                    # saturated downward command and pile-drive the held object
                    # back into the floor).
                    action[w, xyz_off + 0] = 0.0
                    action[w, xyz_off + 1] = 0.0
                    action[w, xyz_off + 2] = 0.0
                    # keep the target from the latch moment on — the weld supports the final lift height.
                    if hold_lock == 1:
                        pos_hold[w] = 1
                else:                                        # target not reached → keep lifting
                    t = ramp_steps - float(es - lift_step)
                    if t < 0.0:
                        t = 0.0
                    lift_m = lift_dist_m * wp.exp(-ramp_rate * t)
                    if lift_m < lift_min_m:                   # floor: ≥ lift_min_m per active step
                        lift_m = lift_min_m
                    # world +z → wrist-frame action delta (apply re-rotates by R_wrist).
                    # OVERRIDE (not add) the policy's Δxyz so the wrist is FORCED to
                    # move purely +z in world frame — matching the legacy JAX
                    # ``delta_transl_world = lift_value``. Adding (+=) let the policy's
                    # own Δxyz cancel the lift (e.g. a downward/hold command), so the
                    # wrist often didn't rise; assignment makes the manual lift an
                    # authoritative control-target override during the lift stage.
                    d_act = wp.transpose(xmat[w, wrist_body_id]) * wp.vec3(0.0, 0.0, lift_m / xyz_scale)
                    action[w, xyz_off + 0] = d_act[0]
                    action[w, xyz_off + 1] = d_act[1]
                    action[w, xyz_off + 2] = d_act[2]
                    # active lift uses target-integration mode — commands that were
                    # lost to servo lag accumulate in the target and the weld keeps pulling.
                    if lift_integrate == 1:
                        pos_hold[w] = 2
            else:
                # Latched → wrist Δxyz stays frozen at 0 for the REST of the
                # lift window (the hold stage below continues the freeze), so
                # once the object has reached the target the wrist is held in
                # place and the policy keeps only finger control.
                action[w, xyz_off + 0] = 0.0
                action[w, xyz_off + 1] = 0.0
                action[w, xyz_off + 2] = 0.0
                # keep the target during the latched span too (prevents re-anchoring sag).
                if hold_lock == 1:
                    pos_hold[w] = 1
        # Over-band lift cap: once the object is lifted ABOVE ``lift_max_m``
        # (= the success band's upper edge, LIFT_TARGET_M·(1+SUCCESS_LIFT_TOL_FRAC)),
        # strip the wrist command's world +z component so nothing lifts it
        # further out of the band. GENERAL: acts on whatever Δxyz is now in the
        # action buffer — the policy's own command OR the manual override above
        # (mutually exclusive in practice: the manual lift only fires while
        # lift_z < lift_target_m < lift_max_m). Lateral / downward motion stays
        # policy-controlled; re-checked every step so full control returns as
        # soon as the object settles back below the cap. ``lift_max_m <= 0``
        # disables (no behaviour change).
        if lift_max_m > 0.0:
            cap_lift_z = xpos[w, obj_body_id][2] - obj_p_init[w, 2]
            if cap_lift_z > lift_max_m:
                R_w   = xmat[w, wrist_body_id]
                d_loc = wp.vec3(action[w, xyz_off + 0],
                                action[w, xyz_off + 1],
                                action[w, xyz_off + 2])
                d_wld = R_w * d_loc                    # wrist-local → world (xyz_scale > 0 → direction unchanged)
                if d_wld[2] > 0.0:                     # upward component only
                    d_wld = wp.vec3(d_wld[0], d_wld[1], 0.0)
                    d_loc = wp.transpose(R_w) * d_wld  # back to wrist-local action units
                    action[w, xyz_off + 0] = d_loc[0]
                    action[w, xyz_off + 1] = d_loc[1]
                    action[w, xyz_off + 2] = d_loc[2]
    if es >= hold_step:
        # TODO: scale the current action according to the curriculum weight.
        action[w, xyz_off + 0] = 0.0
        action[w, xyz_off + 1] = 0.0
        action[w, xyz_off + 2] = 0.0
        # hold: keep the previous target — the weld supports the final lift height
        # (the target at latch time) against gravity (blocks the sag observed over the hold window).
        if hold_lock == 1:
            pos_hold[w] = 1


@wp.kernel
def _zero_masked_lift_latch_kernel(
    done_mask:    wp.array(dtype=int, ndim=1),  # type: ignore (NWORLD,)
    lift_reached: wp.array(dtype=int, ndim=1),  # type: ignore (NWORLD,) OUT
):
    """Clear the manual-lift latch for done worlds after per-world reset, so
    the next episode starts with the rule-based wrist lift armed again."""
    w = wp.tid()
    if done_mask[w] == 1:
        lift_reached[w] = 0
