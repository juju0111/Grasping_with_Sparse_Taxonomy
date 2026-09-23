"""Self/table contact flags + joint-limit penalty — warp kernels of the grasping task."""
import mujoco as _mj  # type: ignore
import numpy as np  # type: ignore
import warp as wp  # type: ignore


# ══════════════════════════════════════════════════════════════════════════
# Shared contact / joint-limit kernels (formerly in hand_pose_tracking)
# ══════════════════════════════════════════════════════════════════════════
@wp.kernel
def _hand_contact_active_kernel(
    sensordata:        wp.array(dtype=float, ndim=2),     # type: ignore  (NWORLD, n_sensordata)
    n_self:            int,
    self_found_idx:    wp.array(dtype=int,   ndim=1),     # type: ignore  (n_self,)
    self_touch_idx:    wp.array(dtype=int,   ndim=1),     # type: ignore  (n_self,)  -1 = no gate
    self_slot_weight:  wp.array(dtype=float, ndim=1),     # type: ignore  (n_self,)  finger_contact_weights per slot
    self_slot_is_tip:  wp.array(dtype=int,   ndim=1),     # type: ignore  (n_self,)  1 = distal (tip) segment slot
    n_tbl:             int,
    tbl_found_idx:     wp.array(dtype=int,   ndim=1),     # type: ignore  (n_tbl,)
    tbl_touch_idx:     wp.array(dtype=int,   ndim=1),     # type: ignore  (n_tbl,)
    touch_eps:         float,
    gate_by_touch:     int,                                # 0 = trust contact alone, 1 = AND with touch
    force_scale:       float,                              # Training.force_scale_for_reward — scales the self impulse
    tip_deadband:      float,                              # [A] contact-force deadband for tip slots (N; no penalty below)
    out_self_active:   wp.array(dtype=int,   ndim=1),     # type: ignore  per-world flag (1 if any self-coll slot fires)
    out_self_impulse_w: wp.array(dtype=float, ndim=1),    # type: ignore  (Σ force·w)/n_self — finger-weighted impulse mean
    out_tbl_active:    wp.array(dtype=int,   ndim=1),     # type: ignore  per-world flag (1 if any hand↔table slot fires)
):
    """Compute per-world self-collision / table-contact **flags** plus the
    finger-weighted self-collision **impulse mean** (no ``W_*`` applied —
    the mixer weights them downstream).

    For each contact-sensor slot ``k`` (already flattened across every
    sensor's ``num`` packets), the slot fires iff the ``found`` bit
    ``sensordata[w, found_idx[k]] > 0.5``. When ``gate_by_touch == 1`` and
    ``touch_idx[k] >= 0``, additionally require
    ``sensordata[w, touch_idx[k]] > touch_eps`` — the touch sensor acts as
    ground truth against phantom ``found`` bits in the contact pipeline.

    Flags are OR-reduced across slots within each family.
    ``out_self_impulse_w`` mirrors object_grasping's
    ``_self_coll_weighted_penalty_kernel`` impulse: per accepted self slot,
    the contact packet's own force magnitude ([found, force(+1:4), …]) ×
    ``force_scale`` × the slot's ``finger_contact_weights`` value, mean'd
    over the FIXED slot count ``n_self`` (legacy ``mean`` semantics). The
    mixer applies the penalty sign + ``W_SELF_IMPULSE``.
    """
    w = wp.tid()

    # ── self-collision family ───────────────────────────────────────────
    self_active = int(0)
    impulse_sum = float(0.0)
    for k in range(n_self):
        f_idx = self_found_idx[k]
        if sensordata[w, f_idx] > 0.5:
            accept = int(1)
            if gate_by_touch == 1:
                t_idx = self_touch_idx[k]
                if t_idx >= 0:
                    if sensordata[w, t_idx] <= touch_eps:
                        accept = 0
            if accept == 1:
                # The tip-touch exemption applies **only to the flag (W_SELF_COLL)**:
                # distal (tip) slots do not raise self_active (intentional pinch
                # contact is not penalized). The impulse (W_SELF_IMPULSE) **always**
                # accumulates tip slots too — contact itself is allowed, but
                # pressing hard is penalized proportionally.
                if self_slot_is_tip[k] == 0:
                    self_active = 1
                # Contact packet's own force magnitude — same read as
                # object_grasping's self-coll penalty kernel.
                f_raw = wp.length(wp.vec3(        # type: ignore 
                    sensordata[w, f_idx + 1],     # type: ignore (NWORLD, n_sensordata)
                    sensordata[w, f_idx + 2],     # type: ignore (NWORLD, n_sensordata)
                    sensordata[w, f_idx + 3]))    # type: ignore (NWORLD, n_sensordata)
                # [A] tip slots: light touches below the deadband are free (only the excess counts).
                if self_slot_is_tip[k] == 1:
                    f_raw = f_raw - tip_deadband
                    if f_raw < 0.0:
                        f_raw = 0.0
                impulse_sum += f_raw * force_scale * self_slot_weight[k]
    out_self_active[w] = self_active
    denom = float(n_self)
    if denom < 1.0:
        denom = 1.0
    out_self_impulse_w[w] = impulse_sum / denom

    # ── hand ↔ table/floor family ────────────────────────────────────────
    tbl_active = int(0)
    for k in range(n_tbl):
        f_idx = tbl_found_idx[k]
        if sensordata[w, f_idx] > 0.5:
            accept = int(1)
            if gate_by_touch == 1:
                t_idx = tbl_touch_idx[k]
                if t_idx >= 0:
                    if sensordata[w, t_idx] <= touch_eps:
                        accept = 0
            if accept == 1:
                tbl_active = 1
    out_tbl_active[w] = tbl_active


@wp.kernel
def _joint_limit_penalty_kernel(
    raw_ctrl_target: wp.array(dtype=float, ndim=2),   # type: ignore (NWORLD, n_ctrl) PRE-clamp command
    ctrl_min:        wp.array(dtype=float, ndim=1),    # type: ignore (n_ctrl,)
    ctrl_max:        wp.array(dtype=float, ndim=1),    # type: ignore (n_ctrl,)
    ctrl_qpa:        wp.array(dtype=int,   ndim=1),    # type: ignore (n_ctrl,) -1 = non-joint actuator → skip
    n_ctrl:          int,
    margin:          float,                            # danger-band width near each bound (ctrl units)
    jl_scale:        float,                            # 1 / n_ctrl (per-actuator mean)
    w_joint_limit:   float,
    dt:              float,
    min_reward:      float,
    max_reward:      float,
    out_r_joint_limit: wp.array(dtype=float, ndim=1),  # type: ignore metric (signed ≤ 0)
    reward:            wp.array(dtype=float, ndim=1),   # type: ignore IN/OUT (post-mixer add)
):
    """Penalize the policy for **commanding** a finger target near / past a
    joint limit — the ``WarpActionApplier`` clamps ``d.ctrl`` into
    ``[ctrl_min, ctrl_max]``, so a policy that keeps pushing into a rail keeps
    saturating (a common source of chattering / vibration in the rolled-out
    deterministic mean). We read the **un-clamped** commanded target
    (``raw_ctrl_target``) and add a quadratic-hinge penalty per direct-joint
    actuator:

        d_hi = (t - (hi - margin)) / margin      # >0 within ``margin`` of hi or beyond
        d_lo = ((lo + margin) - t) / margin
        pen += relu(d_hi)² + relu(d_lo)²

    ``pen`` is 0 while the command stays ``margin`` inside both bounds, 1 at a
    rail, and grows quadratically past it. Non-joint actuators (``qpa == -1``)
    and unbounded ones (``±1e9`` bounds → band never reached) contribute 0.
    Mean over the ``n_ctrl`` actuators, weighted by ``w_joint_limit``,
    subtracted from ``reward`` (dt-scaled + re-clipped)."""
    w = wp.tid()
    inv_m = float(0.0)
    if margin > 1.0e-9:
        inv_m = 1.0 / margin
    pen = float(0.0)
    for i in range(n_ctrl):
        if ctrl_qpa[i] >= 0:                       # direct joint actuators only
            t  = raw_ctrl_target[w, i]
            d_hi = (t - (ctrl_max[i] - margin)) * inv_m
            d_lo = ((ctrl_min[i] + margin) - t) * inv_m
            if d_hi > 0.0:
                pen += d_hi * d_hi
            if d_lo > 0.0:
                pen += d_lo * d_lo
    r_jl = -w_joint_limit * pen * jl_scale
    out_r_joint_limit[w] = r_jl
    r = reward[w] + r_jl * dt
    if r < min_reward:
        r = min_reward
    if r > max_reward:
        r = max_reward
    reward[w] = r
