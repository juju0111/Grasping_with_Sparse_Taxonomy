"""Student observation: frozen partial view, masking, stage machine, history — warp kernels of the grasping task."""
import mujoco as _mj  # type: ignore
import numpy as np  # type: ignore
import warp as wp  # type: ignore
from .common import _grit_r2rpy


@wp.kernel
def _obs_cache_zero_cols_kernel(
    obs:   wp.array(dtype=float, ndim=2),   # type: ignore (NWORLD, obs_dim) in/out
    c0:    int,
    width: int,
    cache: wp.array(dtype=float, ndim=2),   # type: ignore (NWORLD, width) out
):
    """Back up the contiguous columns ``[c0, c0+width)`` into ``cache`` and zero them in obs.

    The student obs is zero-masked so it cannot see the contact-related terms
    (per_slot_contact + obj_touch_force). The original values remain in ``cache`` and
    ``collect_teacher_obs`` restores them into the privileged teacher obs."""
    w, j = wp.tid()
    cache[w, j] = obs[w, c0 + j]
    obs[w, c0 + j] = 0.0


@wp.kernel
def _obs_partial_bps_feature_kernel(
    xpos:               wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore (NWORLD, nbody)
    xmat:               wp.array(dtype=wp.mat33, ndim=2),  # type: ignore (NWORLD, nbody)
    transformed_pcd:    wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore (NWORLD, n_pts) world-frame obj PCD
    visible:            wp.array(dtype=int,      ndim=2),  # type: ignore (NWORLD, n_pts) 1 = visible from the camera
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
    """Visible-masked variant of :func:`_obs_bps_feature_kernel`.

    Points with ``visible[w, k] == 0`` are excluded from the nearest scan — back-side
    geometry not seen by the camera does not leak into the feature. With no visible
    point at all the feature is 0 (same convention as the full version's
    ``n_pts == 0``: "not visible → far").

    **Below-support mask** (``mask_below_z``): as in the full kernel — a basis point
    whose world z is below the support surface (table top; floor if no table) gets
    feature 0. Prevents object-proximity signals from below the table (unreachable
    region) from leaking into obs and pulling the hand into the table. Disable with
    ``-1e9``."""
    w, b = wp.tid()                                       # type: ignore
    R_wrist = xmat[w, wrist_body_id]
    wrist_p = xpos[w, wrist_body_id]
    hc_p = R_wrist * hand_center_offset + wrist_p          # hand-center origin (world)
    hc_R = R_wrist * hand_center_R                          # hand-center rotation (world)
    bp_world = hc_R * bps_local[b] + hc_p                   # basis point in world

    # Below-support mask: point below the table top → feature 0, skip the scan.
    if bp_world[2] < mask_below_z:
        out_obs[w, offset + b] = 0.0
        return

    best_d2 = float(1.0e18)
    found   = int(0)
    for k in range(n_pts):
        if visible[w, k] == 0:
            continue
        d = transformed_pcd[w, k] - bp_world
        d2 = wp.dot(d, d)
        if d2 < best_d2:
            best_d2 = d2
            found   = 1

    feat = float(0.0)
    if found == 1:
        feat = wp.exp(-gamma * wp.sqrt(best_d2))
    out_obs[w, offset + b] = feat


@wp.kernel
def _frozen_site_pcd_err_kernel(
    xmat:            wp.array(dtype=wp.mat33, ndim=2),  # type: ignore
    site_xpos:       wp.array(dtype=wp.vec3, ndim=2),   # type: ignore
    frozen_pcd:      wp.array(dtype=wp.vec3, ndim=2),   # type: ignore (NWORLD, n_pts)
    visible:         wp.array(dtype=int, ndim=2),       # type: ignore (NWORLD, n_pts)
    finger_site_ids: wp.array(dtype=int, ndim=1),       # type: ignore
    wrist_body_id:   int,
    n_sensors:       int,
    n_pts:           int,
    arm_sensor_idx:  int,
    offset:          int,
    out_obs:         wp.array(dtype=float, ndim=2),     # type: ignore
):
    """Tracking-free substitute: site→nearest **frozen partial PCD** vector (student only).

    Same layout as ``_obs_site_pcd_err_kernel`` (arm excluded, index gap closed), but
    scans the frozen one-shot snapshot from the episode start (visible points only)
    instead of the live full PCD. Before the grasp (approach) the object is at rest →
    frozen ≈ live. Outputs 0 when there are no visible points."""
    w, s = wp.tid()
    if s == arm_sensor_idx:
        return
    o = s
    if s > arm_sensor_idx:
        o = s - 1
    p = site_xpos[w, finger_site_ids[s]]
    best_d2 = float(1.0e12)
    best = wp.vec3(0.0, 0.0, 0.0)
    found = int(0)
    for k in range(n_pts):
        if visible[w, k] == 1:
            d = frozen_pcd[w, k] - p
            d2 = wp.dot(d, d)
            if d2 < best_d2:
                best_d2 = d2
                best = frozen_pcd[w, k]
                found = 1
    Rw_T = wp.transpose(xmat[w, wrist_body_id])
    e = wp.vec3(0.0, 0.0, 0.0)
    if found == 1:
        e = Rw_T * (best - p)
    out_obs[w, offset + 3 * o + 0] = e[0]
    out_obs[w, offset + 3 * o + 1] = e[1]
    out_obs[w, offset + 3 * o + 2] = e[2]


@wp.kernel
def _frozen_hand_center_pcd_err_kernel(
    xpos:          wp.array(dtype=wp.vec3, ndim=2),     # type: ignore
    xmat:          wp.array(dtype=wp.mat33, ndim=2),    # type: ignore
    frozen_pcd:    wp.array(dtype=wp.vec3, ndim=2),     # type: ignore
    visible:       wp.array(dtype=int, ndim=2),         # type: ignore
    wrist_body_id: int,
    hc_offset:     wp.vec3,
    n_pts:         int,
    offset:        int,
    out_obs:       wp.array(dtype=float, ndim=2),       # type: ignore
    out_hc_dist:   wp.array(dtype=float, ndim=1),       # type: ignore frozen distance for the stage state machine
):
    """hand-center→nearest frozen PCD vector + distance (student only, tracking-free)."""
    w = wp.tid()
    R_wrist = xmat[w, wrist_body_id]
    hc_p = R_wrist * hc_offset + xpos[w, wrist_body_id]
    best_d2 = float(1.0e12)
    best = wp.vec3(0.0, 0.0, 0.0)
    found = int(0)
    for k in range(n_pts):
        if visible[w, k] == 1:
            d = frozen_pcd[w, k] - hc_p
            d2 = wp.dot(d, d)
            if d2 < best_d2:
                best_d2 = d2
                best = frozen_pcd[w, k]
                found = 1
    Rw_T = wp.transpose(R_wrist)
    e = wp.vec3(0.0, 0.0, 0.0)
    dist = float(1.0e6)
    if found == 1:
        e = Rw_T * (best - hc_p)
        dist = wp.sqrt(best_d2)
    out_obs[w, offset + 0] = e[0]
    out_obs[w, offset + 1] = e[1]
    out_obs[w, offset + 2] = e[2]
    out_hc_dist[w] = dist


@wp.kernel
def _phase_gate_freeze_kernel(
    stage:   wp.array(dtype=int, ndim=1),     # type: ignore student stage (0=approach)
    c0:      int,
    width:   int,
    obs:     wp.array(dtype=float, ndim=2),   # type: ignore IN/OUT student obs
    cache:   wp.array(dtype=float, ndim=2),   # type: ignore last-known values (NWORLD, width)
):
    """Approach-only tracking: in stage 0 the live values are copied into the cache;
    in stage ≥ 1 (post-grasp — tracking assumed lost) the obs is **frozen at the
    last-known values from the moment of grasp** and written back. Wrist-frame relative
    vectors are approximately invariant under a rigid grasp, so the frozen values are
    also a physically valid approximation."""
    w, i = wp.tid()
    if stage[w] == 0:
        cache[w, i] = obs[w, c0 + i]
    else:
        obs[w, c0 + i] = cache[w, i]


@wp.kernel
def _phase_gate_rigid_attach_kernel(
    stage:           wp.array(dtype=int, ndim=1),      # type: ignore student stage (0=approach)
    xpos:            wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
    xmat:            wp.array(dtype=wp.mat33, ndim=2), # type: ignore
    site_xpos:       wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
    finger_site_ids: wp.array(dtype=int, ndim=1),      # type: ignore
    obj_p_init:      wp.array(dtype=float, ndim=2),    # type: ignore
    obj_R_init:      wp.array(dtype=float, ndim=3),    # type: ignore
    wrist_body_id:   int,
    obj_body_id:     int,
    n_sensors:       int,
    arm_sensor_idx:  int,
    pcd_c0:          int,   # first col of site_pcd_err (hc_err follows contiguously at pcd_c0 + 3·(ns−1))
    objpd_c0:        int,
    obs:             wp.array(dtype=float, ndim=2),    # type: ignore IN/OUT student obs
    bp_cache:        wp.array(dtype=float, ndim=2),    # type: ignore (NWORLD, 3·(ns−1)+3)
    rel_p:           wp.array(dtype=wp.vec3, ndim=1),  # type: ignore R_w^T(obj_p−wrist_p) at grasp time
    rel_R:           wp.array(dtype=wp.mat33, ndim=1), # type: ignore R_w^T·R_obj at grasp time
    wg_p:            wp.array(dtype=wp.vec3, ndim=1),  # type: ignore wrist_p at grasp time (for rattach-BPS)
    wg_R:            wp.array(dtype=wp.mat33, ndim=1), # type: ignore R_w at grasp time (for rattach-BPS)
):
    """Rigid-attach variant of approach-only tracking: after the grasp (stage≥1) the
    object is assumed rigidly attached to the wrist and the tracking obs is
    **propagated via wrist FK**.

    stage 0 (tracking valid): only capture the relative state each step, obs stays live —
      rel_p/rel_R = obj pose seen from the wrist frame, bp_cache = wrist-frame position
      of the PCD point matched to each site (err + site_p_wrist) + the 3 raw hc_err values.
    stage ≥ 1: estimate T_obj_est = T_wrist(t)·T_rel, then
      obj_pose_diff ← diff w.r.t. spawn (lift progress stays alive in obs),
      site_pcd_err ← wrist-frame fixed PCD point − live site (reflects finger flexion),
      hand_center_pcd_err ← cached (hand-center is rigid with the wrist → exactly invariant)."""
    w = wp.tid()
    R_w     = xmat[w, wrist_body_id]
    Rw_T    = wp.transpose(R_w)
    wrist_p = xpos[w, wrist_body_id]
    n_out   = n_sensors - 1
    hc_c0   = pcd_c0 + 3 * n_out
    if stage[w] == 0:
        rel_p[w] = Rw_T * (xpos[w, obj_body_id] - wrist_p)
        rel_R[w] = Rw_T * xmat[w, obj_body_id]
        wg_p[w]  = wrist_p
        wg_R[w]  = R_w
        for s in range(n_sensors):
            if s != arm_sensor_idx:
                o = s
                if s > arm_sensor_idx:
                    o = s - 1
                sp_w = Rw_T * (site_xpos[w, finger_site_ids[s]] - wrist_p)
                bp_cache[w, 3 * o + 0] = obs[w, pcd_c0 + 3 * o + 0] + sp_w[0]
                bp_cache[w, 3 * o + 1] = obs[w, pcd_c0 + 3 * o + 1] + sp_w[1]
                bp_cache[w, 3 * o + 2] = obs[w, pcd_c0 + 3 * o + 2] + sp_w[2]
        bp_cache[w, 3 * n_out + 0] = obs[w, hc_c0 + 0]
        bp_cache[w, 3 * n_out + 1] = obs[w, hc_c0 + 1]
        bp_cache[w, 3 * n_out + 2] = obs[w, hc_c0 + 2]
    else:
        obj_p_est = wrist_p + R_w * rel_p[w]
        R_obj_est = R_w * rel_R[w]
        obs[w, objpd_c0 + 0] = obj_p_init[w, 0] - obj_p_est[0]
        obs[w, objpd_c0 + 1] = obj_p_init[w, 1] - obj_p_est[1]
        obs[w, objpd_c0 + 2] = obj_p_init[w, 2] - obj_p_est[2]
        R_init = wp.mat33(
            obj_R_init[w, 0, 0], obj_R_init[w, 0, 1], obj_R_init[w, 0, 2],
            obj_R_init[w, 1, 0], obj_R_init[w, 1, 1], obj_R_init[w, 1, 2],
            obj_R_init[w, 2, 0], obj_R_init[w, 2, 1], obj_R_init[w, 2, 2],
        )
        rpy = _grit_r2rpy(wp.transpose(R_init) * R_obj_est)  # type: ignore
        obs[w, objpd_c0 + 3] = rpy[0]
        obs[w, objpd_c0 + 4] = rpy[1]
        obs[w, objpd_c0 + 5] = rpy[2]
        for s in range(n_sensors):
            if s != arm_sensor_idx:
                o = s
                if s > arm_sensor_idx:
                    o = s - 1
                sp_w = Rw_T * (site_xpos[w, finger_site_ids[s]] - wrist_p)
                obs[w, pcd_c0 + 3 * o + 0] = bp_cache[w, 3 * o + 0] - sp_w[0]
                obs[w, pcd_c0 + 3 * o + 1] = bp_cache[w, 3 * o + 1] - sp_w[1]
                obs[w, pcd_c0 + 3 * o + 2] = bp_cache[w, 3 * o + 2] - sp_w[2]
        obs[w, hc_c0 + 0] = bp_cache[w, 3 * n_out + 0]
        obs[w, hc_c0 + 1] = bp_cache[w, 3 * n_out + 1]
        obs[w, hc_c0 + 2] = bp_cache[w, 3 * n_out + 2]


@wp.kernel
def _student_hist_kernel(
    src_cols:  wp.array(dtype=int, ndim=1),   # type: ignore source obs column indices (n_obs_src,)
    n_obs_src: int,
    act:       wp.array(dtype=float, ndim=2), # type: ignore previous commanded action (NWORLD, n_act)
    n_act:     int,                           # 0 → action not included
    n_src:     int,                           # = n_obs_src + n_act (frame width)
    hist_c0:   int,
    K:         int,
    skip_prob: float,   # probability of skipping this tick's update (per-world) — control-rate / latency jitter DR
    noise_std: float,   # Gaussian noise on recorded history values (encoder / estimation noise DR)
    seed:      int,
    obs:       wp.array(dtype=float, ndim=2), # type: ignore IN/OUT student obs
):
    """Update the history block of the student obs (frame 0 = newest, 1..K−1 = past).

    Temporal context that lets a student without contact sensors infer contact
    indirectly from temporal patterns (joint stall / torque_proxy onset). Each (w, i)
    thread touches only its own column offset, so the shift has no race. The teacher
    obs uses a layout without this block — the original observation is unchanged.

    Sim2real DR: to mimic the real robot's control-period mismatch, latency and
    jitter, the whole update is skipped with probability ``skip_prob`` each tick
    (one decision per world — prevents desync across columns). When skipped, the
    effective spacing between frames varies from 1 to several ticks, so the learned
    policy does not overfit to a fixed-interval assumption. ``noise_std`` Gaussian
    noise is added to the recorded value (in frame 0, then propagated to later frames)."""
    w, i = wp.tid()
    if skip_prob > 0.0:
        st_w = wp.rand_init(wp.int32(seed), wp.int32(w))   # shared per-world stream
        if wp.randf(st_w) < skip_prob:
            return
    k = K - 1
    while k > 0:
        obs[w, hist_c0 + k * n_src + i] = obs[w, hist_c0 + (k - 1) * n_src + i]
        k -= 1
    v = float(0.0)
    if i < n_obs_src:
        v = obs[w, src_cols[i]]
    else:
        v = act[w, i - n_obs_src]   # commanded action (known to the policy itself on the real robot)
    if noise_std > 0.0:
        st = wp.rand_init(wp.int32(seed), wp.int32(1000003 + w * n_src + i))
        v = v + noise_std * wp.randn(st)
    obs[w, hist_c0 + i] = v


@wp.kernel
def _student_hist_gate_kernel(
    skip_prob: float,
    seed:      int,
    upd:       wp.array(dtype=int, ndim=1),  # type: ignore per-world update counter
    skip:      wp.array(dtype=int, ndim=1),  # type: ignore skip flag for this tick
):
    """Per-world skip decision + update counter for the multi-scale history (once per tick)."""
    w = wp.tid()
    s = int(0)
    if skip_prob > 0.0:
        st = wp.rand_init(wp.int32(seed), wp.int32(w))
        if wp.randf(st) < skip_prob:
            s = 1
    skip[w] = s
    if s == 0:
        upd[w] = upd[w] + 1


@wp.kernel
def _student_hist_ms_kernel(
    src_cols:  wp.array(dtype=int, ndim=1),    # type: ignore
    n_obs_src: int,
    act:       wp.array(dtype=float, ndim=2),  # type: ignore
    n_act:     int,
    n_src:     int,
    lags:      wp.array(dtype=int, ndim=1),    # type: ignore exponentially spaced lags (in update units)
    K:         int,
    L:         int,                            # ring length = max(lags)+1
    upd:       wp.array(dtype=int, ndim=1),    # type: ignore (already incremented by the gate)
    skip:      wp.array(dtype=int, ndim=1),    # type: ignore
    hist_c0:   int,
    noise_std: float,
    seed:      int,
    ring:      wp.array(dtype=float, ndim=3),  # type: ignore (NWORLD, L, n_src)
    obs:       wp.array(dtype=float, ndim=2),  # type: ignore IN/OUT
):
    """Multi-scale history: extends the temporal receptive field to max(lags) ticks with
    the same number of frames (dense recent past, sparse distant past — dilated-conv
    idea). Frames are written to a ring buffer and only the lag positions {1,2,4,8,...}
    are exposed in obs. On a skip (DR) lags are counted in 'update units', so the
    effective time spacing naturally stretches (consistent with the timing DR)."""
    w, i = wp.tid()
    if skip[w] == 1:
        return                     # obs hist columns keep their previous values
    u = upd[w] - 1                 # update index of the current frame
    v = float(0.0)
    if i < n_obs_src:
        v = obs[w, src_cols[i]]
    else:
        v = act[w, i - n_obs_src]
    if noise_std > 0.0:
        st = wp.rand_init(wp.int32(seed), wp.int32(1000003 + w * n_src + i))
        v = v + noise_std * wp.randn(st)
    ring[w, u % L, i] = v
    for j in range(K):
        idx = u - lags[j]
        val = float(0.0)
        if idx >= 0:
            val = ring[w, idx % L, i]
        obs[w, hist_c0 + j * n_src + i] = val


@wp.kernel
def _rattach_bps_pcd_kernel(
    stage:         wp.array(dtype=int, ndim=1),       # type: ignore
    xpos:          wp.array(dtype=wp.vec3, ndim=2),   # type: ignore
    xmat:          wp.array(dtype=wp.mat33, ndim=2),  # type: ignore
    wrist_body_id: int,
    frozen_pcd:    wp.array(dtype=wp.vec3, ndim=2),   # type: ignore (NWORLD, n_pts)
    wg_p:          wp.array(dtype=wp.vec3, ndim=1),   # type: ignore wrist_p at grasp time
    wg_R:          wp.array(dtype=wp.mat33, ndim=1),  # type: ignore R_w at grasp time
    out_pcd:       wp.array(dtype=wp.vec3, ndim=2),   # type: ignore
):
    """rattach-BPS: for stage≥1, move the frozen partial PCD by the wrist's rigid transform.

    The frozen PCD holds world points at the spawn pose — after the grasp the object
    moves with the hand, so it must be propagated as p' = T_wrist(t)·T_wrist(g)^{-1}·p
    for the BPS tail to stay consistent with the in-hand object (freezing would yield
    the contradictory observation 'the object is still on the table')."""
    w, k = wp.tid()
    if stage[w] == 0:
        out_pcd[w, k] = frozen_pcd[w, k]
    else:
        local = wp.transpose(wg_R[w]) * (frozen_pcd[w, k] - wg_p[w])
        out_pcd[w, k] = xpos[w, wrist_body_id] + xmat[w, wrist_body_id] * local


@wp.kernel
def _student_stage_kernel(
    best_dist_hc: wp.array(dtype=float, ndim=1),      # type: ignore hand-center→nearest PCD dist (m)
    xpos:         wp.array(dtype=wp.vec3, ndim=2),    # type: ignore
    obj_p_init:   wp.array(dtype=float, ndim=2),      # type: ignore (NWORLD, 3)
    obj_body_id:  int,
    dist_thresh:  float,                              # 0→1: hand-center↔obj distance threshold (m)
    near_ticks:   int,                                # 0→1: ticks to stay below the threshold
    # contact gate (proprioceptive): a distance-only rule mistakes 'approach done' for
    # 'grasp done' too early (the transition can lead actual contact by ~11 ticks).
    # near is accepted only when the mean |torque_proxy| (ctrl−qpos, joint stall on
    # contact) exceeds the threshold — reproducible on the real robot with encoders alone.
    use_touch:    int,
    tq_c0:        int,
    n_tq:         int,
    touch_thresh: float,
    lift_thresh_m: float,                             # 1→2: z-rise threshold (m)
    # tracking-free: use the wrist z rise as a proxy for 1→2 instead of obj z (the
    # wrist z at the 0→1 transition is recorded as the anchor — if the grasp holds,
    # the object rises with the hand).
    wrist_body_id: int,
    use_wrist_proxy: int,
    wrist_anchor: wp.array(dtype=float, ndim=1),      # type: ignore IN/OUT
    stage:        wp.array(dtype=int, ndim=1),        # type: ignore IN/OUT — monotone 0→1→2 within an episode
    near_cnt:     wp.array(dtype=int, ndim=1),        # type: ignore IN/OUT
    stage_col:    int,
    out_obs:      wp.array(dtype=float, ndim=2),      # type: ignore student obs — the stage column is overwritten
):
    """Data-driven stage state machine for sim2real (student obs only).

    Defines the stage from observable physical quantities instead of the env tick
    (LIFT_STEP/HOLD_STEP):
      0→1 (approach→lift): closest hand-center↔obj distance < ``dist_thresh``
                            sustained for ``near_ticks`` consecutive ticks
      1→2 (lift→hold)   : obj z − initial z > ``lift_thresh_m``
    Computable identically on the real robot (object pose estimate + internal counter
    only). Advances monotonically within an episode; the reset hook sets it back to 0.
    The teacher obs keeps the tick-based stage (reproducibility principle) — this
    kernel writes only the student ``out_obs``."""
    w = wp.tid()
    s = stage[w]
    if s == 0:
        near = best_dist_hc[w] < dist_thresh
        if near and use_touch == 1:
            acc = float(0.0)
            for j in range(n_tq):
                acc += wp.abs(out_obs[w, tq_c0 + j])
            near = (acc / float(n_tq)) > touch_thresh
        if near:
            c = near_cnt[w] + 1
            near_cnt[w] = c
            if c >= near_ticks:
                s = 1
                wrist_anchor[w] = xpos[w, wrist_body_id][2]   # proxy anchor (wrist z at grasp time)
        else:
            near_cnt[w] = 0
    if s == 1:
        lift = float(0.0)
        if use_wrist_proxy == 1:
            lift = xpos[w, wrist_body_id][2] - wrist_anchor[w]
        else:
            lift = xpos[w, obj_body_id][2] - obj_p_init[w, 2]
        if lift > lift_thresh_m:
            s = 2
    stage[w] = s
    out_obs[w, stage_col] = float(s)


# obs terms blanked once the student is "blind" (proximity latch / stage ≥ 1)
