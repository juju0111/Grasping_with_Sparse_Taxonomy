"""Re-grasp retry / early success / single-attempt clock — warp kernels of the grasping task."""
import mujoco as _mj  # type: ignore
import numpy as np  # type: ignore
import warp as wp  # type: ignore


@wp.kernel
def _retry_early_success_kernel(
    xpos:            wp.array(dtype=wp.vec3, ndim=2),  # type: ignore (NWORLD, nbody)
    obj_p_init:      wp.array(dtype=float,   ndim=2),  # type: ignore (NWORLD, 3)
    hand_obj_active: wp.array(dtype=int,     ndim=1),  # type: ignore (NWORLD,) ≥1 touch-gated obj contact slot
    bonus_streak:    wp.array(dtype=int,     ndim=1),  # type: ignore IN/OUT
    obj_body_id:     int,
    lift_step:       int,
    hold_step:       int,
    max_ep_steps:    int,
    # retry knobs
    retry_enabled:   int,
    retry_max:       int,
    retry_approach:  int,
    retry_min_lift:  int,
    retry_min_hold:  int,
    retry_drop_k:    int,
    retry_thresh_m:  float,
    retry_require_contact: int,                        # 1 → retry only worlds that have touched the object
    # early-success knobs
    early_enabled:   int,                              # 0 → eval / promotion off
    train_mode:      int,                              # 1 → training rollout: always reset bookkeeping counters at the window end (timeout)
    streak_min:      int,
    lift_thresh_m:   float,
    lift_hold_steps: int,
    lift_target_m:   float,
    success_tol_frac: float,
    success_code:    int,
    # state (IN/OUT)
    ep_step:         wp.array(dtype=int,   ndim=1),    # type: ignore post-increment (after the done kernel's ++)
    ep_elapsed:      wp.array(dtype=int,   ndim=1),    # type: ignore true elapsed steps (++ here)
    retry_count:     wp.array(dtype=float, ndim=1),    # type: ignore
    drop_streak:     wp.array(dtype=int,   ndim=1),    # type: ignore consecutive steps with no contact and no lift
    grasp_progress:  wp.array(dtype=int,   ndim=1),    # type: ignore touched the object at least once in this attempt (0/1)
    lift_hold_streak: wp.array(dtype=float, ndim=1),   # type: ignore consecutive steps holding the lift during the hold stage
    lift_reached:    wp.array(dtype=int,   ndim=1),    # type: ignore legacy lift latch (0 on rewind)
    out_retry_fired: wp.array(dtype=float, ndim=1),    # type: ignore diagnostic 0/1
    out_early:       wp.array(dtype=float, ndim=1),    # type: ignore diagnostic 0/1(lift)/2(strict)
    out_done:        wp.array(dtype=float, ndim=1),    # type: ignore IN/OUT
    out_mask:        wp.array(dtype=int,   ndim=1),    # type: ignore IN/OUT
    out_reason:      wp.array(dtype=int,   ndim=1),    # type: ignore IN/OUT
):
    w = wp.tid()
    out_retry_fired[w] = 0.0
    out_early[w] = 0.0
    ep_elapsed[w] = ep_elapsed[w] + 1
    elapsed = ep_elapsed[w]
    lift_z = xpos[w, obj_body_id][2] - obj_p_init[w, 2]
    step = ep_step[w]

    # ── contact-experience latch ──────────────────────────────────────
    # Retry exists to rescue worlds that "grasped and then dropped". Granting a
    # retry to a world that **never even touched** the object merely repeats the
    # failed approach and burns episode budget (stealing the time hold_far would
    # have used to terminate early). A world that touched at least once gets its
    # episode extended by a retry for another chance. Reset to 0 at episode start
    # and on rewind, so only contact "in this attempt" counts.
    if hand_obj_active[w] >= 1:
        grasp_progress[w] = 1

    # ── lift-hold streak during the hold stage (for S2) ───────────────
    if step >= hold_step and lift_z >= lift_thresh_m:
        lift_hold_streak[w] = lift_hold_streak[w] + 1.0
    else:
        lift_hold_streak[w] = 0.0

    # ── failure terminals are left alone (except hold_far(13) — a drop-retry candidate) ──
    if out_done[w] > 0.5 and out_reason[w] != 2 and out_reason[w] != 13:
        ep_elapsed[w] = 0
        retry_count[w] = 0.0
        drop_streak[w] = 0
        grasp_progress[w] = 0
        lift_hold_streak[w] = 0.0
        return

    # ── (S1)(S2) early success (training only) ───────────────────────
    if early_enabled == 1:
        early = int(0)
        if bonus_streak[w] >= streak_min:
            early = 2
        elif lift_hold_steps > 0 and lift_hold_streak[w] >= float(lift_hold_steps):
            early = 1
        if early > 0:
            out_early[w] = float(early)
            out_done[w] = 1.0
            out_mask[w] = 1
            out_reason[w] = success_code
            bonus_streak[w] = 0
            ep_elapsed[w] = 0
            retry_count[w] = 0.0
            drop_streak[w] = 0
            grasp_progress[w] = 0
            lift_hold_streak[w] = 0.0
            return

    # ── window end: elapsed-based timeout / band success (covers rewound worlds) ──
    if elapsed >= max_ep_steps:
        if early_enabled == 1:
            diff = lift_z - lift_target_m
            if diff < 0.0:
                diff = -diff
            if diff <= success_tol_frac * lift_target_m:
                out_reason[w] = success_code
                out_early[w] = 3.0
            elif out_reason[w] == 0 or out_reason[w] == 13:
                out_reason[w] = 2
        elif out_reason[w] == 0 or out_reason[w] == 13:
            out_reason[w] = 2
        out_done[w] = 1.0
        out_mask[w] = 1
        # Note: no bookkeeping reset in eval — run_eval reads ``bonus_streak``
        # **after** the loop ends to compute the streak-based success_rate, so
        # zeroing it on the last step would zero that metric entirely. **In training
        # rollouts the reset is mandatory regardless of the early-success setting** —
        # guarding on early_enabled alone left ep_elapsed >= MAX after a timeout when
        # SUCCESS_EARLY_ENABLED=false, trapping that world in a timeout every step
        # (1-step episodes); the better the policy, the more worlds got trapped.
        if early_enabled == 1 or train_mode == 1:
            bonus_streak[w] = 0
            ep_elapsed[w] = 0
            retry_count[w] = 0.0
            drop_streak[w] = 0
            grasp_progress[w] = 0
            lift_hold_streak[w] = 0.0
        return

    # ── retry decision (lift/hold stages only) ────────────────────────
    trigger = int(0)
    if retry_enabled == 1 and step >= lift_step:
        # drop / no-contact streak counter (from min_lift steps after entering lift)
        if step >= lift_step + retry_min_lift and lift_z < retry_thresh_m and hand_obj_active[w] == 0:
            drop_streak[w] = drop_streak[w] + 1
        else:
            drop_streak[w] = 0
        if drop_streak[w] >= retry_drop_k:                 # (R1) drop
            trigger = 1
        if step == hold_step and lift_z < retry_thresh_m:  # (R2) not lifted when entering hold
            trigger = 1
        if out_reason[w] == 13:                            # (R3) hold_far = object moved away from the hand
            trigger = 1
        if retry_count[w] >= float(retry_max):
            trigger = 0
        # never retry a world that has not touched the object (see the latch comment above)
        if retry_require_contact == 1 and grasp_progress[w] == 0:
            trigger = 0
        # only if at least one cycle (approach + lift + min hold) of time remains
        need = retry_approach + (hold_step - lift_step) + retry_min_hold
        if max_ep_steps - elapsed < need:
            trigger = 0
    else:
        drop_streak[w] = 0

    if trigger == 1:
        # ── rewind ────────────────────────────────────────────────────
        new_step = lift_step - retry_approach
        if new_step < 0:
            new_step = 0
        ep_step[w] = new_step
        retry_count[w] = retry_count[w] + 1.0
        drop_streak[w] = 0
        grasp_progress[w] = 0
        bonus_streak[w] = 0
        lift_hold_streak[w] = 0.0
        lift_reached[w] = 0
        out_retry_fired[w] = 1.0
        # a hold_far(13) raised on the same step yields to the retry → cancel done
        if out_done[w] > 0.5 and out_reason[w] == 13:
            out_done[w] = 0.0
            out_mask[w] = 0
            out_reason[w] = 0
        return

    # no retry possible + done via hold_far → terminate as is (clean up counters)
    if out_done[w] > 0.5:
        ep_elapsed[w] = 0
        retry_count[w] = 0.0
        drop_streak[w] = 0
        lift_hold_streak[w] = 0.0


@wp.kernel
def _obs_ep_norm_horizon_kernel(
    ep_step:  wp.array(dtype=int,   ndim=1),   # type: ignore
    horizon:  int,                              # single-attempt horizon T1 (steps)
    offset:   int,                              # ep_step_norm slot (first slot of the ep_stage block)
    out_obs:  wp.array(dtype=float, ndim=2),   # type: ignore actor obs (overwritten in place)
):
    """ep_step_norm := clamp(ep_step / T1, 0, 1) — overwrites the ep_step/MAX written by the base kernel."""
    w = wp.tid()
    denom = float(horizon)
    if denom < 1.0:
        denom = 1.0
    v = float(ep_step[w]) / denom
    if v > 1.0:
        v = 1.0
    out_obs[w, offset] = v
