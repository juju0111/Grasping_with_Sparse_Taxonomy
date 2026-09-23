"""Done detection + bonus gate — warp kernels of the grasping task."""
import mujoco as _mj  # type: ignore
import numpy as np  # type: ignore
import warp as wp  # type: ignore


# ══════════════════════════════════════════════════════════════════════════
# Done detection + bonus gate (separate, weight-free — matches tracking)
# ══════════════════════════════════════════════════════════════════════════


@wp.kernel
def _wrist_height_penalty_kernel(
    xpos:          wp.array(dtype=wp.vec3, ndim=2),   # type: ignore (NWORLD, nbody)
    wrist_body_id: int,
    floor_z:       float,
    margin_m:      float,
    w_pen:         float,
    dt:            float,
    min_reward:    float,
    max_reward:    float,
    out_pen:       wp.array(dtype=float, ndim=1),     # type: ignore  ≤0 (logged, r_wrist_height)
    out_reward:    wp.array(dtype=float, ndim=1),     # type: ignore  IN/OUT (post-mixer add)
):
    """Floor-grazing prevention shaping — quadratic hinge on wrist z (post-mixer).

    ``pen = -w · ((floor_z + margin − wrist_z)/margin)²`` (by how far the wrist has
    descended into the margin band), 0 above the band. Unlike the contact-based
    penalties (W_TABLE_*), this creates a gradient along the approach path before
    any collision happens. Wired like the joint-limit penalty: dt-scaled, added to
    ``out_reward`` and re-clipped."""
    w = wp.tid()
    z   = xpos[w, wrist_body_id][2]
    gap = (floor_z + margin_m) - z
    pen = float(0.0)
    if gap > 0.0:
        r = gap / margin_m
        pen = -w_pen * r * r
    out_pen[w] = pen
    r_new = out_reward[w] + pen * dt
    if r_new < min_reward:
        r_new = min_reward
    if r_new > max_reward:
        r_new = max_reward
    out_reward[w] = r_new


@wp.kernel
def _finger_height_penalty_kernel(
    site_xpos:       wp.array(dtype=wp.vec3, ndim=2),   # type: ignore (NWORLD, nsite)
    finger_site_ids: wp.array(dtype=int,    ndim=1),    # type: ignore (n_sensors,)
    xpos:            wp.array(dtype=wp.vec3, ndim=2),   # type: ignore (NWORLD, nbody)
    obj_body_id:     int,
    n_sensors:       int,
    arm_sensor_idx:  int,
    floor_z:         float,
    margin_m:        float,
    obj_clear_m:     float,
    w_pen:           float,
    dt:              float,
    min_reward:      float,
    max_reward:      float,
    out_pen:         wp.array(dtype=float, ndim=1),     # type: ignore ≤0 (logged, r_finger_height)
    out_reward:      wp.array(dtype=float, ndim=1),     # type: ignore IN/OUT (post-mixer add)
):
    """Dense penalty for finger sites approaching the floor (post-mixer).

    Contact-based penalties (W_TABLE_*) only signal **after** a collision, so they
    cannot shape the approach path. Here each finger site gets a quadratic hinge
    by how far it has descended into the ``margin`` band above the floor, creating
    a gradient before the collision::

        floor_eff = floor_z + min(obj_bottom_clearance, obj_clear_m)
        gap       = (floor_eff + margin) − site_z
        pen      -= w · (gap/margin)²            (0 outside the band)

    **Flat objects**: using the floor plane directly would penalize lowering the
    fingers to grasp a low object at all, blocking the grasp. The reference plane is
    therefore raised by at most ``obj_clear_m`` relative to the object bottom (the
    height needed to wrap the object is allowed), and only going lower and scraping
    the floor is penalized.
    The forearm slot is excluded; all finger sites are included regardless of
    active/rest, then averaged by count → scale-invariant across hands."""
    w = wp.tid()
    # allowed clearance relative to the object center z (flatter objects allow lower).
    obj_z = xpos[w, obj_body_id][2]
    clear = obj_z - floor_z
    if clear > obj_clear_m:
        clear = obj_clear_m
    if clear < 0.0:
        clear = 0.0
    floor_eff = floor_z + clear
    acc = float(0.0)
    cnt = float(0.0)
    for s in range(n_sensors):
        if s == arm_sensor_idx:
            continue
        z   = site_xpos[w, finger_site_ids[s]][2]
        gap = (floor_eff + margin_m) - z
        cnt += 1.0
        if gap > 0.0:
            r = gap / margin_m
            acc += r * r
    if cnt < 1.0:
        cnt = 1.0
    pen = -w_pen * acc / cnt
    out_pen[w] = pen
    r_new = out_reward[w] + pen * dt
    if r_new < min_reward:
        r_new = min_reward
    if r_new > max_reward:
        r_new = max_reward
    out_reward[w] = r_new


@wp.kernel
def _table_impulse_deriv_penalty_kernel(
    tbl_impulse_w:   wp.array(dtype=float, ndim=1),     # type: ignore  raw_table_impulse_w (this step)
    prev_impulse:    wp.array(dtype=float, ndim=1),     # type: ignore  IN/OUT persistent prev-step value
    ep_step:         wp.array(dtype=int,   ndim=1),     # type: ignore  gate the first step (post-increment ==1)
    w_pen:           float,
    deriv_clip:      float,                             # cap per-step Δ⁺ (0 = no cap)
    dt:              float,
    min_reward:      float,
    max_reward:      float,
    out_pen:         wp.array(dtype=float, ndim=1),     # type: ignore ≤0 (logged, r_table_impulse_deriv)
    out_reward:      wp.array(dtype=float, ndim=1),     # type: ignore IN/OUT (post-mixer add)
):
    """Dense penalty on the **rise (time derivative Δ⁺)** of the table impulse (post-mixer).

    W_TABLE_IMPULSE penalizes the impulse **magnitude**, so it cannot distinguish a
    normal grasp contact that "grazes lightly for a long time" from a "slamming"
    impact. Here only the per-step **increase** of the finger-weighted table impulse
    (``raw_table_impulse_w``) is penalized::

        Δ  = impulse_now − impulse_prev
        pen = -w · clip(max(Δ, 0), 0, deriv_clip)

    Decreases (contact release) are not penalized; the gradual impulse increase of a
    normal wrap is penalized lightly and an abrupt slam (large Δ) heavily → only
    impacts are suppressed. A curriculum ramp from base (small) → final (large) is
    recommended (refine impacts after the grasp has been learned). Only table/forearm
    contact impulses are targeted, not object contact, so grip force is unaffected.

    ``ep_step`` has already been incremented by the done kernel, so the first step
    reads 1 → gated with ``> 1`` to block spurious spikes from the previous episode's
    leftover prev value (physically there is no table contact right after a reset,
    so Δ≤0 and it is usually harmless, but it is blocked defensively)."""
    w = wp.tid()
    cur  = tbl_impulse_w[w]
    d = cur - prev_impulse[w]           # >0 = impulse rising (impact building up)
    if d < 0.0:
        d = 0.0                         # penalize the rise only (release is free)
    if deriv_clip > 0.0 and d > deriv_clip:
        d = deriv_clip
    pen = float(0.0)
    if ep_step[w] > 1:
        pen = -w_pen * d
    out_pen[w] = pen
    r_new = out_reward[w] + pen * dt
    if r_new < min_reward:
        r_new = min_reward
    if r_new > max_reward:
        r_new = max_reward
    out_reward[w] = r_new
    # store the current value for the next step's derivative (persistent state).
    prev_impulse[w] = cur


@wp.kernel
def _torque_balance_kernel(
    actuator_force: wp.array(dtype=float, ndim=2),   # type: ignore (NWORLD, nu) per-motor output force (torque)
    ctrl_group_id:  wp.array(dtype=int,   ndim=1),   # type: ignore (n_ctrl,) finger group id, -1 = not a finger
    n_ctrl:         int,
    n_groups:       int,
    torque_eps:     float,                           # fingers with torque at or below this are ignored (resting fingers)
    out_coeff:      wp.array(dtype=float, ndim=1),   # type: ignore (NWORLD,) ∈(0,1], 1 = uniform
):
    """**Intra-finger** motor torque balance coefficient (independent per finger).

    For each finger f with motor torques {τ_i}, the participation-ratio balance::

        coeff_f = (Σ|τ_i|)² / (k_f · Σ τ_i²)   ∈ [1/k_f, 1]

    =1 means the motors within the finger share the torque **perfectly evenly**,
    =1/k means **everything is concentrated on one motor**. Scale-invariant
    (independent of absolute torque), so it measures "is it concentrated?" rather
    than "how hard is the grip?". When one joint is overloaded while bracing the
    object, coeff↓ → the multiplied ineq_coeff↓ → that grasp is rewarded less →
    encourages spreading torque across motors.

    Each finger is handled **independently** (index/middle/ring/thumb). A finger
    with a single motor (no notion of balance) or torque at or below torque_eps
    (resting) is excluded from the mean — only multi-joint fingers that actually
    exert force are regulated, so the term is not diluted. If all are excluded → 1.0 (off)."""
    w = wp.tid()
    acc = float(0.0)   # sum of per-finger balance coefficients
    ng  = float(0.0)   # number of fingers included in the coefficient
    for g in range(n_groups):
        s1 = float(0.0)   # Σ|τ|
        s2 = float(0.0)   # Σ τ²
        k  = float(0.0)   # motor count of this finger
        for i in range(n_ctrl):
            if ctrl_group_id[i] == g:
                t = wp.abs(actuator_force[w, i])
                s1 += t
                s2 += t * t
                k  += 1.0
        if k >= 2.0 and s1 > torque_eps and s2 > 1.0e-12:
            coeff = (s1 * s1) / (k * s2)
            acc += coeff
            ng  += 1.0
    if ng < 1.0:
        out_coeff[w] = 1.0
    else:
        out_coeff[w] = acc / ng


@wp.kernel
def _object_grasping_done_kernel(
    xpos:              wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore
    xmat:              wp.array(dtype=wp.mat33, ndim=2),  # type: ignore
    fd_obj_vz:         wp.array(dtype=float,    ndim=1),  # type: ignore (NWORLD,) obj world-z velocity (pose-FD; avoids cvel jitter)
    obj_p_init:        wp.array(dtype=float,    ndim=2),  # type: ignore
    obj_contact_impulse: wp.array(dtype=float,  ndim=1),  # type: ignore  max hand↔obj contact force
    tbl_impulse_w:     wp.array(dtype=float,    ndim=1),  # type: ignore  weighted table(floor)+forearm impulse (per step)
    rest_obj_impulse:  wp.array(dtype=float,    ndim=1),  # type: ignore  max rest (non-taxonomy) body↔obj contact force (aux kernel)
    obj_ground_force:  wp.array(dtype=float,    ndim=1),  # type: ignore  estimate of the force the floor exerts on the object (aux kernel)
    best_dist_hc:      wp.array(dtype=float,    ndim=1),  # type: ignore  closest hand-center↔object-surface distance (PCD scan, same step)
    obj_body_id:       int,
    wrist_body_id:     int,
    hand_center_offset: wp.vec3,                          # rh_hand_center[:3,3] — wrist→hand-center offset
    done_obj_z:        float,
    done_obj_xy_drift: float,
    done_wrist_x_z:    float,
    done_obj_touch:    float,
    done_obj_touch_rest: float,                           # rest body↔obj contact force above which → abort (reason 11; 1e9 = off)
    done_table_impulse: float,                            # weighted floor/forearm impulse above which → abort (reason 10)
    done_obj_ground:   float,                             # obj-ground (floor) force above which → abort (reason 12; 1e9 = off)
    done_obj_vel_z:    float,
    done_hand_obj_dist: float,                            # hand-center↔obj dist (m) above which → abort (reason 9)
    done_hold_obj_dist: float,                            # hold stage: closest hand-center↔surface distance above which → reason 13 (1e9 = off)
    lift_step:         int,                               # obj_fell (velocity) condition applies from the lift stage only (JAX v2 convention)
    hold_step:         int,                               # step at which the hold-far check starts (HOLD_STEP)
    max_ep_steps:      int,
    timeout_only:      int,                               # 1 → eval: only timeout sets done (full-window play-out)
    ep_step:           wp.array(dtype=int,   ndim=1),     # type: ignore IN/OUT (++ here)
    out_done:          wp.array(dtype=float, ndim=1),     # type: ignore
    out_mask:          wp.array(dtype=int,   ndim=1),     # type: ignore
    out_reason:        wp.array(dtype=int,   ndim=1),     # type: ignore
):
    """Done detection + ep_step increment.

    Reason priority (matches the codebase ``REASON_NAME`` map
    ``{1:success, 2:timeout, 3:obj_z, 4:wrist_up, 5:obj_xy,
    6:hand_object_touch, 7:obj_fell}``; the success kernel stamps ``1`` for
    success promotion (overriding a coincident timeout), so the failure reasons
    below never collide with success)::

        3 obj_z    : obj falls below ``done_obj_z`` (drop)
        5 obj_xy   : obj drifts > ``done_obj_xy_drift`` from init xy
        6 hand_object_touch : max hand↔obj contact force > ``done_obj_touch``
                              (object crushed too hard → abort)
        11 rest_crush : max rest (non-taxonomy) body↔obj force > ``done_obj_touch_rest``
                        (pressing with the wrong fingers — JAX v2 masked_impulse done)
        10 table_crush : weighted floor/forearm impulse > ``done_table_impulse``
                         (hard floor slam — soft grazes stay penalty-only;
                         disable with a huge threshold)
        12 obj_ground : obj-ground force estimate > ``done_obj_ground`` (object pressed
                        into the floor — approximates the JAX v2 obj_table_impulse done)
        7 obj_fell : obj_vel[2] < ``done_obj_vel_z`` AND ep_step >= lift_step
                     (object fell down → abort; lift stage only, JAX v2 convention)
        9 hand_obj_far : ||hand_center - obj|| > ``done_hand_obj_dist`` (failed
                         lift — object left behind while the wrist moved away)
        13 hold_far : in the hold stage (ep_step >= hold_step) the closest
                      hand-center↔object-surface distance > ``done_hold_obj_dist`` —
                      reclaims the remaining timeout steps of a failed-grasp episode early
        2 timeout  : ``ep_step >= max_ep_steps`` (checked AFTER ++)
        4 wrist_up : wrist x-axis world.z > ``done_wrist_x_z`` (viz guard)
    """
    w = wp.tid()
    obj_p   = xpos[w, obj_body_id]
    R_wrist = xmat[w, wrist_body_id]
    wrist_p = xpos[w, wrist_body_id]

    obj_p0 = wp.vec3(obj_p_init[w, 0], obj_p_init[w, 1], obj_p_init[w, 2])
    dx = obj_p[0] - obj_p0[0]
    dy = obj_p[1] - obj_p0[1]
    xy_drift = wp.sqrt(dx * dx + dy * dy)

    # hand-center world pos = R_wrist · offset + wrist_p (same as the approach
    # kernel); abort if it drifts too far from the object (failed lift: object
    # left on the floor while the wrist moved away).
    hc_p   = R_wrist * hand_center_offset + wrist_p
    hc_obj = hc_p - obj_p
    hand_obj_dist = wp.length(hc_obj)

    ep_step[w] = ep_step[w] + 1

    obj_height_done = float(0.0)
    if obj_p[2] < done_obj_z:
        obj_height_done = 1.0

    xy_done = float(0.0)
    if xy_drift > done_obj_xy_drift:
        xy_done = 1.0

    # Excessive grasp force: the dominant hand↔obj contact impulse exceeds the
    # crush threshold → abort the episode.
    touch_done = float(0.0)
    if obj_contact_impulse[w] > done_obj_touch:
        touch_done = 1.0

    # Rest-finger crush (JAX v2 ``masked_impulse`` done): abort immediately when a
    # body the taxonomy says not to use (rest) presses the object beyond the
    # threshold — kept separate from the legitimate holding force of the active
    # fingers (the done_obj_touch total guard above), targeting only "grasping with
    # the wrong fingers".
    rest_done = float(0.0)
    if rest_obj_impulse[w] > done_obj_touch_rest:
        rest_done = 1.0

    # Hard floor(table)/forearm slam: light grazes are handled by the W_TABLE_*
    # penalties; only strong collisions above the threshold end the episode
    # (two-tier structure).
    tbl_done = float(0.0)
    if tbl_impulse_w[w] > done_table_impulse:
        tbl_done = 1.0

    # Object pressing the floor/table beyond the threshold (approximates the JAX v2
    # ``obj_table_impulse`` done) — aborts grasps that push the object into the floor.
    obj_ground_done = float(0.0)
    if obj_ground_force[w] > done_obj_ground:
        obj_ground_done = 1.0

    # Object falling: GLOBAL downward (-z) velocity exceeds the threshold
    # (``obj_vel_z < done_obj_vel_z``, a negative value) → dropped/plummeting
    # → abort. The velocity source is **pose-FD** (``fd_obj_vz``) — cvel picks up
    # contact-solver jitter as instantaneous velocity and can false-trigger at the
    # moment of impact. Following the JAX v2 convention it is checked **from the lift
    # stage only**, so a momentary downward velocity from lightly nudging the object
    # during approach does not end the episode (lift_step < 0 → always).
    fell_done = float(0.0)
    if fd_obj_vz[w] < done_obj_vel_z:
        if lift_step < 0 or ep_step[w] >= lift_step:
            fell_done = 1.0

    hand_far_done = float(0.0)
    if hand_obj_dist > done_hand_obj_dist:
        hand_far_done = 1.0

    # Hold-far (reason 13): in the hold stage (ep_step >= hold_step) the object must
    # be held, so if the hand-center is farther than the threshold from the object's
    # **closest surface point** the episode has already failed — reset immediately
    # instead of dragging on to timeout (up to 40 steps remaining), reclaiming wasted
    # samples. Using the closest surface distance (best_dist_hc) keeps the
    # "holding ≈ 0" scale even for large objects (complements the center-distance
    # based reason 9; calibrate the threshold with the approach_err metric).
    hold_far_done = float(0.0)
    if ep_step[w] >= hold_step:
        if best_dist_hc[w] > done_hold_obj_dist:
            hold_far_done = 1.0

    timeout = float(0.0)
    if ep_step[w] >= max_ep_steps:
        timeout = 1.0

    x_axis_world = R_wrist * wp.vec3(1.0, 0.0, 0.0)
    wrist_up = float(0.0)
    if x_axis_world[2] > done_wrist_x_z:
        wrist_up = 1.0

    done_f = float(0.0)
    reason = int(0)
    if obj_height_done > 0.5:
        done_f = 1.0
        reason = 3                       # obj_z (REASON_NAME[3])
    elif xy_done > 0.5:
        done_f = 1.0
        reason = 5
    elif touch_done > 0.5:
        done_f = 1.0
        reason = 6
    elif rest_done > 0.5:
        done_f = 1.0
        reason = 11                      # rest_crush (REASON_NAME[11]) — non-taxonomy body pressing
    elif tbl_done > 0.5:
        done_f = 1.0
        reason = 10                      # table_crush (REASON_NAME[10])
    elif obj_ground_done > 0.5:
        done_f = 1.0
        reason = 12                      # obj_ground (REASON_NAME[12]) — object pressed into the floor
    elif fell_done > 0.5:
        done_f = 1.0
        reason = 7                       # obj_fell (REASON_NAME[7]); NOT 1 (= success)
    elif hand_far_done > 0.5:
        done_f = 1.0
        reason = 9                       # hand_obj_far (REASON_NAME[9])
    elif hold_far_done > 0.5:
        done_f = 1.0
        reason = 13                      # hold_far (REASON_NAME[13]) — early reclaim of a failed grasp in the hold stage
    elif timeout > 0.5:
        done_f = 1.0
        reason = 2
    elif wrist_up > 0.5:
        done_f = 1.0
        reason = 4                       # wrist_up (REASON_NAME[4])

    # ``out_reason`` always carries the real (priority-resolved) terminal
    # reason — used by the success kernel (to promote a coincident timeout)
    # and by evaluate.py's diagnostics / success marker.
    out_reason[w] = reason

    # ``timeout_only`` (eval): only ``timeout`` (ep_step >= max) sets
    # done/mask, so each world plays out the FULL episode window and
    # ``ep_step`` is never zeroed by a state-based failure (the "zombie" bug —
    # see SubEnvHandler.timeout_only_done). Non-timeout reasons stay in
    # ``out_reason`` for diagnostics but don't terminate the episode.
    if timeout_only == 1:
        out_done[w] = timeout
        out_mask[w] = int(timeout)
    else:
        out_done[w] = done_f
        out_mask[w] = int(done_f)


@wp.kernel
def _object_grasping_success_kernel(
    bonus_active:        wp.array(dtype=int,   ndim=1),    # type: ignore (NWORLD,) bonus condition this step
    ep_step:             wp.array(dtype=int,   ndim=1),    # type: ignore (NWORLD,) post-increment counter
    bonus_streak:        wp.array(dtype=int,   ndim=1),    # type: ignore IN/OUT streak (mixer reads it)
    xpos:                wp.array(dtype=wp.vec3, ndim=2),  # type: ignore (NWORLD, nbody) — current obj z
    obj_p_init:          wp.array(dtype=float, ndim=2),    # type: ignore (NWORLD, 3) — spawn obj pos
    obj_body_id:         int,
    max_ep_steps:        int,
    lift_target_m:       float,                            # target lift (m)
    success_tol_frac:    float,                            # ±frac·lift_target_m band
    success_reason_code: int,
    disable_done_promotion: int,                           # 1 → eval: only update streak
    out_done:            wp.array(dtype=float, ndim=1),    # type: ignore IN/OUT
    out_mask:            wp.array(dtype=int,   ndim=1),    # type: ignore IN/OUT
    out_reason:          wp.array(dtype=int,   ndim=1),    # type: ignore IN/OUT
):
    """Object-grasping success: at the LAST episode step
    (``ep_step >= max_ep_steps``), the object's lift
    (``obj_z - obj_z_init``) lies within ``±success_tol_frac · lift_target_m``
    of the target lift — i.e. the object was held into the target-z band. The
    coincident timeout (reason=2) is overwritten with ``success_reason_code``
    (terminal success). Still maintains ``bonus_streak`` (the reward mixer reads
    it) and resets it on any done; eval mode only updates the streak."""
    w = wp.tid()
    # ① streak bookkeeping (mixer reads it)
    if bonus_active[w] >= 1:
        bonus_streak[w] = bonus_streak[w] + 1
    else:
        bonus_streak[w] = 0
    if disable_done_promotion == 1:
        return
    # ② success = last step AND object lifted into the target-z band
    success = int(0)
    if ep_step[w] >= max_ep_steps:
        lift_z = xpos[w, obj_body_id][2] - obj_p_init[w, 2]
        diff   = lift_z - lift_target_m
        if diff < 0.0:
            diff = -diff
        if diff <= success_tol_frac * lift_target_m:
            success = 1
    # ③ promote done → success (overrides the coincident timeout)
    if success == 1:
        out_done[w]   = 1.0
        out_mask[w]   = 1
        out_reason[w] = success_reason_code
    # ④ streak reset on any done
    if out_done[w] > 0.5:
        bonus_streak[w] = 0


@wp.kernel
def _object_grasping_bonus_gate_kernel(
    raw_lift:         wp.array(dtype=float, ndim=1),      # type: ignore  ∈ [0, 1]
    obj_active:       wp.array(dtype=int,   ndim=1),      # type: ignore  0/1
    out_bonus_active: wp.array(dtype=int, ndim=1),        # type: ignore
):
    """Bonus gate flag — "lifted AND still grasping".

    Fires when BOTH hold this step:
      * ``raw_lift >= 1`` (object lifted to ``LIFT_TARGET_M``), AND
      * ``obj_active == 1`` (≥ 1 touch-gated hand↔obj contact slot fires).

    (No wrist-proximity check — an earlier design also gated on
    ``approach_err``, but hand↔obj contact already implies proximity.)

    Consumers: the reward mixer's streak-scaled bonus term
    (``bonus_streak``, capped at ``SUCCESS_STREAK_MIN``) and ``run_eval``'s
    streak-based ``success_rate``. NOT the training success promotion —
    that is decided by ``_object_grasping_success_kernel`` (object lift
    within the ±``SUCCESS_LIFT_TOL_FRAC`` band at the last episode step).
    """
    w = wp.tid()
    active = int(0)
    if raw_lift[w] >= 1.0:
        if obj_active[w] == 1:
            active = 1
    out_bonus_active[w] = active
