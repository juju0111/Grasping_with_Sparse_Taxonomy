"""Warp kernel + helper class for applying delta-actions to a parallel
warp env (wrist mocap pose + finger ctrl).

Both translation and rotation deltas are interpreted in the **wrist-body
local frame** and composed onto the wrist's current world pose. This
matches the JAX-era policy convention: the policy commands "move/rotate
along my own axes", not "along world axes". A pure +x translation always
means "forward along the wrist's forward axis" regardless of how the
wrist is oriented in the world.

Action layout per world (size = 3 + 6 + n_ctrl):

    [0:3]        wrist_xyz_delta   raw translation delta in **wrist-local
                                   frame**. Effective world-frame delta =
                                   ``R_wrist @ action * xyz_scale`` (m), so
                                   each component picks the wrist's own
                                   x/y/z axis.
    [3:9]        wrist_R_6d        6D rotation representation (Zhou et al.
                                   2019). Two 3-vectors orthonormalized via
                                   Gram-Schmidt → R_delta. Composed onto
                                   the wrist's current world rotation in
                                   **body frame** (= wrist-local rotation):
                                       R_new = R_wrist @ R_delta
                                       new_quat = cur_quat ⊗ quat(R_delta)
                                   The apply kernel adds an identity bias
                                   ``[1,0,0,0,1,0]`` internally, so the
                                   policy's natural zero output → R_delta =
                                   identity. The 6D output is treated as a
                                   residual Δr around identity: post-tanh
                                   action ≈ 0 → no rotation; small noise
                                   around 0 → small rotations near identity.
                                   No actor-side identity-bias init needed.
    [9:9+n_ctrl] finger_delta      ctrl delta, scaled by ``finger_scale``,
                                   then clamped to per-actuator
                                   ``[ctrl_min, ctrl_max]``.

The kernel writes directly to ``d.mocap_pos / d.mocap_quat / d.ctrl`` so
the captured CUDA graph (``mjwarp.step``) consumes the freshly applied
action when launched next.

Designed for plug-in replacement of the action source: random test loop
or policy output (numpy or wp.array). See ``WarpActionApplier`` and
``WarpActionApplier.sample_random``.
"""
from typing import Optional

import numpy as np
import torch
import warp as wp
import mujoco

from grit.training.orchestrator.base import _grit_r2quat_wxyz, _grit_set_seed_kernel


# ──────────────────────────────────────────────────────────────────────────
# Warp helper functions
# ──────────────────────────────────────────────────────────────────────────

@wp.func
def _quat_mul_wxyz(a: wp.quat, b: wp.quat) -> wp.quat:                  # type: ignore
    """Hamilton product, wxyz convention."""
    return wp.quat(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


@wp.func
def _gram_schmidt_6d(a1: wp.vec3, a2: wp.vec3) -> wp.mat33:              # type: ignore
    """Decode a 6D rotation representation (two 3-vectors) into R via
    Gram-Schmidt. Falls back to identity if a1 or projected a2 are degenerate.
    """
    n1 = wp.length(a1)
    if n1 < 1.0e-9:
        e1 = wp.vec3(1.0, 0.0, 0.0)
    else:
        e1 = a1 / n1

    a2_perp = a2 - wp.dot(e1, a2) * e1
    n2 = wp.length(a2_perp)
    if n2 < 1.0e-9:
        # Pick any vector perpendicular to e1
        if wp.abs(e1[0]) < 0.9:
            tmp = wp.vec3(1.0, 0.0, 0.0)
        else:
            tmp = wp.vec3(0.0, 1.0, 0.0)
        e2 = wp.normalize(tmp - wp.dot(e1, tmp) * e1)
    else:
        e2 = a2_perp / n2

    e3 = wp.cross(e1, e2)
    return wp.mat33(
        e1[0], e2[0], e3[0],
        e1[1], e2[1], e3[1],
        e1[2], e2[2], e3[2],
    )


# ──────────────────────────────────────────────────────────────────────────
# Random-action sampling kernel (GPU-resident; no host roundtrip)
# ──────────────────────────────────────────────────────────────────────────

@wp.kernel
def _grit_sample_random_action_kernel(
    seed_arr:     wp.array(dtype=int),                # type: ignore (1,)
    action:       wp.array(dtype=float, ndim=2),      # type: ignore (NWORLD, action_dim)
    xyz_off:      int,
    rot6d_off:    int,
    finger_off:   int,
    n_ctrl:       int,
    xyz_sigma:    float,
    rot_noise:    float,
    finger_sigma: float,
):
    """Sample one (NWORLD, action_dim) random action in-place.

    - xyz       : N(0, xyz_sigma)
    - rot 6D    : N(0, rot_noise)  (residual around identity — the apply
                                    kernel adds the +[1,0,0,0,1,0] bias on
                                    top, so action zero == identity rotation)
    - finger    : N(0, finger_sigma)

    All scaling-to-physical-units happens later in the apply kernel via
    ``xyz_scale`` / ``finger_scale``.
    """
    w = wp.tid()
    state = wp.rand_init(seed_arr[0] + 4093, w)   # offset so action stream != obj/wrist init

    # xyz delta
    action[w, xyz_off + 0] = wp.randn(state) * xyz_sigma
    action[w, xyz_off + 1] = wp.randn(state) * xyz_sigma
    action[w, xyz_off + 2] = wp.randn(state) * xyz_sigma

    # 6D rotation residual centred at 0 — apply kernel adds identity bias.
    action[w, rot6d_off + 0] = wp.randn(state) * rot_noise
    action[w, rot6d_off + 1] = wp.randn(state) * rot_noise
    action[w, rot6d_off + 2] = wp.randn(state) * rot_noise
    action[w, rot6d_off + 3] = wp.randn(state) * rot_noise
    action[w, rot6d_off + 4] = wp.randn(state) * rot_noise
    action[w, rot6d_off + 5] = wp.randn(state) * rot_noise

    # finger ctrl deltas
    for i in range(n_ctrl):
        action[w, finger_off + i] = wp.randn(state) * finger_sigma


# ──────────────────────────────────────────────────────────────────────────
# Action kernel
# ──────────────────────────────────────────────────────────────────────────

@wp.kernel
def _apply_action_kernel(
    action:        wp.array(dtype=float, ndim=2),         # type: ignore (NWORLD, action_dim)
    mocap_id:      int,
    wrist_body_id: int,
    n_ctrl:        int,
    # ── Actuator scoping (multi-hand scenes) ─────────────────────────────
    # ``act_ids[i]`` = scene actuator column driven by local slot ``i``.
    # Single-hand scenes pass ``arange(nu)`` (identity) — bit-identical to
    # the pre-``act_ids`` behaviour. In a two-hand (leader-follower) scene
    # each hand's applier gets its own actuator subset so the two appliers
    # write disjoint ``ctrl`` columns of the SAME shared buffer.
    act_ids:       wp.array(dtype=int),                   # type: ignore  (n_ctrl,)
    xyz_off:       int,
    rot6d_off:     int,
    finger_off:    int,
    xyz_scale:     float,
    finger_scale:  float,
    rot_scale:     float,
    # ── finger action representation ─────────────────────────────────────
    # 0 = delta (default): ``ctrl = qpos[qpa] + a·finger_scale`` — re-anchored to
    #     the realised qpos every step. During contact the steady-state servo
    #     error is pinned to a·finger_scale, so **action noise becomes force
    #     noise**, and the setpoint moves every step so the policy cannot
    #     express a "rest point".
    # 1 = absolute-normalized: ``ctrl = cmin + (a+1)/2·(cmax−cmin)`` — the policy
    #     output a∈[−1,1] is de-normalized to the actuator ctrl range as an
    #     **absolute setpoint**. Equal a gives equal ctrl, so the policy predicts
    #     the target pose directly and force = kp·(ctrl − qpos) is continuous in qpos.
    #     finger_scale is **ignored** in this mode (deadzone is meaningless too → skipped).
    # 2 = integrated-setpoint: ``ctrl = clamp(ctrl_prev + a·finger_scale,
    #     qpos ± finger_lead_max)`` — the delta is integrated onto the **previous
    #     setpoint, not the realised qpos**. With a=0 the setpoint (= force) is
    #     held (rest point); a is the setpoint rate, so fine resolution (in units
    #     of finger_scale) is preserved. Force = kp·(ctrl − qpos) becomes the
    #     **integral** of the action, so action noise only enters the force rate
    #     (delta mode maps noise to force 1:1). Integrator wind-up (setpoint
    #     running away from the realised qpos → force blow-up) is prevented by
    #     the lead clamp: max force = kp·finger_lead_max, an explicit knob playing
    #     the role of delta mode's implicit cap (kp·finger_scale).
    #     Note: the setpoint becomes hidden state, so put ctrl−qpos (servo error) in the obs.
    finger_mode:   int,
    finger_lead_max: float,
    # integ-only leak λ ∈ [0,1]: ``ctrl = qpos + λ·(ctrl_prev − qpos) + a·s``.
    # λ=1 = pure integration (exploration noise random-walks onto the lead clamp
    # rails: σ 0.06 rad/step × √256 steps ≈ 0.96 rad ≫ lead 0.2); λ=0 = identical
    # to delta mode. Intermediate values give a leaky integrator = first-order
    # low-pass of the action: the stationary noise std is bounded by
    # σ·s/√(1−λ²) (λ=0.8 → 0.10 rad), and with a=0 the setpoint relaxes toward
    # qpos with time constant 1/(1−λ) steps (not a true rest point, but a
    # continuous knob between delta's "instant release" and integ's "hold forever").
    finger_leak:   float,
    # 3 = **history-window (sliding sum)**: ``ctrl = clamp(qpos + s·Σ_{i<K} a_{t-i}/norm,
    #     qpos ± finger_lead_max)`` — like delta it **re-anchors to the realised
    #     qpos every step** (adapts to a moving scene, no drift), but the command
    #     is the sum/mean of the last K actions, i.e. an **FIR low-pass**
    #     (K=1 is exactly delta).
    #     Difference from integ (mode 2): integ accumulates the setpoint without
    #     bound, has no anchor and easily sticks to the lead clamp rails, whereas
    #     here the anchor is always the current qpos so the command follows a
    #     moving object. In exchange a=0 means "zero force after K steps" rather
    #     than "hold", so integ's rest-point property is weaker (larger K → closer to hold).
    finger_hist:      wp.array(dtype=float, ndim=3),  # type: ignore (NWORLD, K, n_ctrl) ring buffer
    finger_hist_len:  int,                             # K (0 → mode disabled)
    finger_hist_idx:  int,                             # slot written this step
    finger_hist_norm: float,                           # sum divisor (K = mean, 1 = plain sum)
    # control_wrist=False → 0: this kernel does not touch the wrist mocap.
    # The mocap DEFAULT pose fixed at reset is kept, so the wrist is "held".
    # (1 → anchor the delta to the realised wrist pose every step, as before.)
    update_wrist:  int,
    ctrl_min:      wp.array(dtype=float),                 # type: ignore  (n_ctrl,)
    ctrl_max:      wp.array(dtype=float),                 # type: ignore  (n_ctrl,)
    ctrl_qpa:      wp.array(dtype=int),                   # type: ignore  (n_ctrl,) qpos addr per ctrl, -1 = use ctrl[i]
    qpos:          wp.array(dtype=float, ndim=2),         # type: ignore  (NWORLD, nq)
    xpos:          wp.array(dtype=wp.vec3, ndim=2),       # type: ignore  (NWORLD, nbody) — post-forward body world pos
    xmat:          wp.array(dtype=wp.mat33, ndim=2),      # type: ignore  (NWORLD, nbody) — post-forward body world rot
    mocap_pos:     wp.array(dtype=wp.vec3, ndim=2),       # type: ignore  (NWORLD, nmocap)
    mocap_quat:    wp.array(dtype=wp.quat, ndim=2),       # type: ignore  (NWORLD, nmocap)
    ctrl:          wp.array(dtype=float, ndim=2),         # type: ignore  (NWORLD, nu)
    # ── Post-scale deadzone (no-op when ``use_deadzone == 0``) ───────────
    # Each component's scaled delta is clamped to zero when its magnitude
    # falls below the per-component threshold. Prevents the wrist / finger
    # from drifting past a target once the policy's residual output decays
    # toward zero — a "stay put" command (||action·scale|| < ε) now
    # actually holds the wrist / joints in place instead of feeding tiny
    # setpoint nudges that integrate over many control steps.
    #
    #   ``xyz_deadzone_m``      — threshold on ``||action_xyz · xyz_scale||₂``
    #                             in meters (frame-invariant since R_wrist is
    #                             orthonormal).
    #   ``rot_deadzone_rad``    — threshold on the final SLERP-scaled
    #                             rotation angle (radians). Compared against
    #                             ``2 · atan2(||v||, |w|)`` of the scaled
    #                             delta quaternion.
    #   ``finger_deadzone_rad`` — per-actuator threshold on
    #                             ``|action_i · finger_scale|`` (radians for
    #                             revolute joints; arbitrary ctrl-units for
    #                             tendon / non-joint actuators).
    use_deadzone:        int,
    xyz_deadzone_m:      float,
    rot_deadzone_rad:    float,
    finger_deadzone_rad: float,
    # ── Per-world wrist target-hold flags (written by the stage kernel; default 0) ─
    # With the realised-anchor convention (mocap = xpos[wrist] + Δ), Δ=0 does not
    # mean "hold pose" but "re-base the target on the sagged realised pose", so
    # the weld's steady-state error accumulates every step (ratchet sinking:
    # about −24 mm of wrist drop over a 40-step hold).
    #   pos_hold = 0: original convention (realised + Δ)
    #   pos_hold = 1: skip the mocap_pos write → keep the previous target (the
    #                 weld holds that pose against gravity; for lift-latch / hold stages)
    #   pos_hold = 2: integrate the target (mocap_pos += Δ_world) — for manual lift.
    #                 Under realised-anchoring only ~25% of a command is realised
    #                 because of servo lag; with integration the unrealised part
    #                 stays in the target and the weld keeps pulling. pos_lead_max_m
    #                 stops the target from leading the realised wrist in z by more
    #                 than that (prevents weld force blow-up when blocked).
    #   rot_hold = 1: skip the mocap_quat write → keep the previous rotation target.
    #   rot_hold = 2: integrate the rotation target (mocap_quat ← prev_target ⊗ Δ) —
    #                 rotational counterpart of pos_hold=2. Realised re-anchoring (0)
    #                 realises only ~25% of a single-step command due to weld lag and
    #                 discards the rest at the next re-anchor, so **the effect of a
    #                 single-step perturbation is erased** (RL cannot assign credit
    #                 to the rotation channel, as observed in in-hand reorientation).
    #                 With integration the unrealised command stays in the target
    #                 and keeps pulling. rot_lead_max_rad bounds the target↔realised
    #                 angle (prevents blow-up when blocked).
    wrist_pos_hold:      wp.array(dtype=int),              # type: ignore  (NWORLD,)
    wrist_rot_hold:      wp.array(dtype=int),              # type: ignore  (NWORLD,)
    pos_lead_max_m:      float,
    rot_lead_max_rad:    float,
    rot_leak:            float,   # rot_hold=2: fraction by which the target relaxes toward the realised pose each step (0 = pure integration; 0.1 → time constant ~10 steps)
    # ── Raw (pre-clamp) commanded finger target export ───────────────────
    # The commanded actuator target BEFORE the ``[ctrl_min, ctrl_max]`` clamp
    # is written here per world/actuator. ``ctrl[w, i]`` itself is clamped, so
    # it can never reveal how far past a joint limit the policy pushed — this
    # buffer keeps the un-clamped value so a joint-limit penalty term can see
    # the overshoot (see ``_joint_limit_penalty_kernel``). No behavioural
    # change: it is a pure side-output written next to the existing clamp.
    out_raw_ctrl_target: wp.array(dtype=float, ndim=2),   # type: ignore  (NWORLD, n_ctrl)
):
    """Apply per-world delta-action to the live state.

    Semantics: every delta is rooted in the **current physical state** so
    the policy command is "where to move *from where I actually am*",
    not "where to drift the previous setpoint to". This stops the desired
    setpoint from wandering away from the realised state across many
    control steps under contact / damping.

    Both translation and rotation deltas are expressed in the wrist
    body's local frame and composed onto its current world pose:

      * ``mocap_pos[w] = xpos[w, wrist] + R_wrist @ (action_xyz · xyz_scale)``
        — wrist body's post-forward world position + world-frame
        translation obtained by rotating the wrist-local delta. A pure
        +x action moves along the wrist's own forward axis regardless
        of its world orientation.
      * ``mocap_quat[w] = quat_from(R_wrist) ⊗ delta_quat`` — wrist
        body's post-forward world rotation is the rotation anchor; the
        delta is composed in body-frame so it rotates the wrist around
        its own current axes.
      * ``ctrl[w, i] = qpos[w, ctrl_qpa[i]] + delta * finger_scale``  —
        the actuator's driven joint qpos is the finger-delta anchor.
        Tendon / non-joint actuators (``ctrl_qpa[i] == -1``) fall back to
        ``ctrl[w, i]`` since they have no scalar joint-qpos counterpart.
    """
    w = wp.tid()

    # Wrist body's post-forward world pose (anchors translation + rotation).
    wrist_p = xpos[w, wrist_body_id]
    R_wrist = xmat[w, wrist_body_id]

    # ── Finger-only (update_wrist == 0): skip ALL mocap writes ───────────
    # Re-anchoring to the realised pose every step lets the mocap follow
    # whatever the finger reaction forces push the wrist by, producing a
    # slow ratchet drift. If the mocap is never written, the weld keeps
    # pulling toward the DEFAULT pose fixed at reset and the wrist stays put.
    if update_wrist == 1:
        # ── delta xyz: wrist-local translation, rotated to world by R_wrist ──
        dxyz_local = wp.vec3(
            action[w, xyz_off + 0],
            action[w, xyz_off + 1],
            action[w, xyz_off + 2],
        ) * xyz_scale
        # Post-scale deadzone: if the scaled translation magnitude is below the
        # threshold, treat the command as "hold position" (zero delta). Norm is
        # frame-invariant under the orthonormal ``R_wrist``, so we measure it
        # in wrist-local before rotating to world.
        if use_deadzone == 1:
            if wp.length(dxyz_local) < xyz_deadzone_m:
                dxyz_local = wp.vec3(0.0, 0.0, 0.0)
        dxyz_world = R_wrist * dxyz_local
        ph = wrist_pos_hold[w]
        if ph == 0:
            mocap_pos[w, mocap_id] = wrist_p + dxyz_world
        elif ph == 2:
            # Target integration (manual lift): accumulate the world-frame delta
            # onto the previous target. z-lead cap: clamp when the target leads
            # the realised wrist upward by more than pos_lead_max_m (guards the
            # case where a grasped object snags and the wrist cannot follow).
            tgt = mocap_pos[w, mocap_id] + dxyz_world
            if pos_lead_max_m > 0.0 and (tgt[2] - wrist_p[2]) > pos_lead_max_m:
                tgt = wp.vec3(tgt[0], tgt[1], wrist_p[2] + pos_lead_max_m)
            mocap_pos[w, mocap_id] = tgt
        # ph == 1: skip the write; keep the previous target (blocks ratchet sinking)

        # ── delta rotation: 6D → R_delta (Gram-Schmidt) → SLERP scale → compose ──
        # Heuristic identity bias: add [1,0,0] / [0,1,0] inside the kernel so the
        # policy's natural zero output → R_delta = identity. Network output is
        # interpreted as a residual Δr around identity, which gives a much better
        # initialization for residual / damped rotation control:
        #   * post-tanh action ≈ 0 (typical at init) → R_delta ≈ I (no rotation)
        #   * exploration noise around 0 → small rotations centred on identity
        #   * no need for ``init_rot6d_identity_bias`` on the actor head
        a1 = wp.vec3(action[w, rot6d_off + 0] + 1.0,
                     action[w, rot6d_off + 1],
                     action[w, rot6d_off + 2])
        a2 = wp.vec3(action[w, rot6d_off + 3],
                     action[w, rot6d_off + 4] + 1.0,
                     action[w, rot6d_off + 5])
        R_delta    = _gram_schmidt_6d(a1, a2)
        delta_quat = _grit_r2quat_wxyz(R_delta)
        # SLERP(identity, delta_quat, rot_scale): scale the rotation angle by rot_scale.
        # q^t = (cos(t·θ/2),  sin(t·θ/2)/sin(θ/2) · (x,y,z))  where cos(θ/2) = delta_quat.w
        dqx      = delta_quat[1]
        dqy      = delta_quat[2]
        dqz      = delta_quat[3]
        sin_half = wp.sqrt(dqx * dqx + dqy * dqy + dqz * dqz)
        if sin_half > 1.0e-9:
            half_angle = wp.atan2(sin_half, delta_quat[0])
            new_half   = rot_scale * half_angle
            s          = wp.sin(new_half) / sin_half
            delta_quat = wp.quat(wp.cos(new_half), dqx * s, dqy * s, dqz * s)
        # Post-scale rotation deadzone: measure the angle of the FINAL
        # (post-SLERP) delta_quat. ``|qw|`` collapses the sign ambiguity (q
        # and -q encode the same rotation), and ``2·atan2(||v||, |w|)`` is
        # the numerically stable form of ``2·acos(|qw|)`` across the full
        # range. Sub-threshold deltas collapse to identity.
        if use_deadzone == 1:
            fqx     = delta_quat[1]
            fqy     = delta_quat[2]
            fqz     = delta_quat[3]
            sin_h_f = wp.sqrt(fqx * fqx + fqy * fqy + fqz * fqz)
            cos_h_f = wp.abs(delta_quat[0])
            angle_rad = 2.0 * wp.atan2(sin_h_f, cos_h_f)
            if angle_rad < rot_deadzone_rad:
                delta_quat = wp.quat(1.0, 0.0, 0.0, 0.0)
        # Body-frame compose: cur ⊗ delta_quat == R_wrist @ R_delta. The delta
        # is interpreted in the wrist's local frame so it rotates the wrist
        # around its own current axes (matches the wrist-local translation
        # path above — both deltas live in the same frame).
        cur        = _grit_r2quat_wxyz(R_wrist)
        new_quat   = _quat_mul_wxyz(cur, delta_quat)
        # Renormalize (compounding small float errors over many calls drifts ||q||).
        qn = wp.sqrt(new_quat[0] * new_quat[0] + new_quat[1] * new_quat[1]
                     + new_quat[2] * new_quat[2] + new_quat[3] * new_quat[3])
        # rot_hold == 1: skip the write; keep the previous rotation target (blocks
        # the rotational ratchet drift caused by realised re-basing; measured
        # ~6.8°/100 steps in free space).
        if wrist_rot_hold[w] == 0:
            if qn < 1.0e-9:
                mocap_quat[w, mocap_id] = wp.quat(1.0, 0.0, 0.0, 0.0)
            else:
                inv = 1.0 / qn
                mocap_quat[w, mocap_id] = wp.quat(
                    new_quat[0] * inv, new_quat[1] * inv,
                    new_quat[2] * inv, new_quat[3] * inv,
                )
        elif wrist_rot_hold[w] == 2:
            # Rotation target integration: previous target ⊗ Δ (composed about wrist-local axes).
            prev_t = mocap_quat[w, mocap_id]
            # Leak: relax the target toward the realised pose by a fraction rot_leak.
            # With pure integration, exploration noise becomes a random walk of the
            # target and shakes the grasp. A leaky integrator keeps a single-step
            # command alive for ~1/leak steps, preserving credit assignment while
            # bounding the random-walk variance (same idea as finger_leak).
            if rot_leak > 0.0:
                cur_conj0 = wp.quat(cur[0], -cur[1], -cur[2], -cur[3])
                rel0 = _quat_mul_wxyz(cur_conj0, prev_t)        # cur ⊗ rel0 = prev_t
                r0w = rel0[0]
                r0x = rel0[1]
                r0y = rel0[2]
                r0z = rel0[3]
                if r0w < 0.0:
                    r0w = -r0w
                    r0x = -r0x
                    r0y = -r0y
                    r0z = -r0z
                sh0 = wp.sqrt(r0x * r0x + r0y * r0y + r0z * r0z)
                if sh0 > 1.0e-9:
                    ang0 = 2.0 * wp.atan2(sh0, r0w) * (1.0 - rot_leak)
                    nh0 = 0.5 * ang0
                    sc0 = wp.sin(nh0) / sh0
                    rel0 = wp.quat(wp.cos(nh0), r0x * sc0, r0y * sc0, r0z * sc0)
                    prev_t = _quat_mul_wxyz(cur, rel0)
            tq = _quat_mul_wxyz(prev_t, delta_quat)
            tn = wp.sqrt(tq[0] * tq[0] + tq[1] * tq[1] + tq[2] * tq[2] + tq[3] * tq[3])
            if tn < 1.0e-9:
                tq = cur
            else:
                tinv = 1.0 / tn
                tq = wp.quat(tq[0] * tinv, tq[1] * tinv, tq[2] * tinv, tq[3] * tinv)
            # lead cap: if the target leads the realised wrist orientation by more
            # than rot_lead_max_rad, keep only that angle along the realised→target
            # direction (prevents weld force blow-up).
            if rot_lead_max_rad > 0.0:
                cur_conj = wp.quat(cur[0], -cur[1], -cur[2], -cur[3])
                rel = _quat_mul_wxyz(cur_conj, tq)          # cur ⊗ rel = tq
                rw = rel[0]
                rx = rel[1]
                ry = rel[2]
                rz = rel[3]
                if rw < 0.0:
                    rw = -rw
                    rx = -rx
                    ry = -ry
                    rz = -rz
                sh = wp.sqrt(rx * rx + ry * ry + rz * rz)
                ang = 2.0 * wp.atan2(sh, rw)
                if ang > rot_lead_max_rad and sh > 1.0e-9:
                    nh = 0.5 * rot_lead_max_rad
                    sc = wp.sin(nh) / sh
                    rel = wp.quat(wp.cos(nh), rx * sc, ry * sc, rz * sc)
                    tq = _quat_mul_wxyz(cur, rel)
            mocap_quat[w, mocap_id] = tq

    # ── finger ctrl: delta (qpos re-anchoring) or absolute-normalized ─────
    for i in range(n_ctrl):
        qpa = ctrl_qpa[i]
        cmin = ctrl_min[i]
        cmax = ctrl_max[i]
        new_c = float(0.0)
        if finger_mode == 2:
            # Integrate onto the previous setpoint + lead clamp about the realised qpos (anti-windup).
            prev_c = ctrl[w, act_ids[i]]
            if qpa >= 0:
                q = qpos[w, qpa]
                # leak: shrink the previous lead (prev_c − q) by λ, then integrate
                new_c = q + finger_leak * (prev_c - q) + action[w, finger_off + i] * finger_scale
                if new_c > q + finger_lead_max:
                    new_c = q + finger_lead_max
                if new_c < q - finger_lead_max:
                    new_c = q - finger_lead_max
            else:
                new_c = prev_c + action[w, finger_off + i] * finger_scale
        elif finger_mode == 3:
            # history-window: add the last K actions onto the qpos anchor (FIR low-pass).
            # Write the current action into the ring buffer first, then sum all K slots.
            finger_hist[w, finger_hist_idx, i] = action[w, finger_off + i]
            acc = float(0.0)
            for kk in range(finger_hist_len):
                acc += finger_hist[w, kk, i]
            if qpa >= 0:
                q = qpos[w, qpa]
            else:
                q = ctrl[w, act_ids[i]]
            new_c = q + (acc / finger_hist_norm) * finger_scale
            if new_c > q + finger_lead_max:
                new_c = q + finger_lead_max
            if new_c < q - finger_lead_max:
                new_c = q - finger_lead_max
        elif finger_mode == 1:
            # Linear de-normalization a ∈ [−1, 1] → [cmin, cmax]. The policy's
            # (tanh) output space is the normalized ctrl space, so the policy sees
            # a uniform [−1,1] even when actuator ranges differ. Out-of-range
            # outputs are handled by the clamp.
            a = action[w, finger_off + i]
            new_c = cmin + (a + 1.0) * 0.5 * (cmax - cmin)
        else:
            # Direct joint actuator → use realised joint qpos as the anchor.
            # Tendon / site / body actuators (qpa == -1) have no 1:1 qpos
            # counterpart — fall back to the existing ctrl setpoint.
            if qpa >= 0:
                base = qpos[w, qpa]
            else:
                base = ctrl[w, act_ids[i]]
            raw_delta = action[w, finger_off + i] * finger_scale
            # Per-actuator post-scale deadzone — sub-threshold setpoint nudges
            # are squashed to zero so the joint actually holds at ``base``.
            if use_deadzone == 1:
                if wp.abs(raw_delta) < finger_deadzone_rad:
                    raw_delta = 0.0
            new_c = base + raw_delta
        # Export the un-clamped commanded target FIRST (so a downstream
        # joint-limit penalty can measure how far past the limit the policy
        # commanded), then clamp the value actually driven into the sim.
        out_raw_ctrl_target[w, i] = new_c
        if new_c < cmin:
            new_c = cmin
        if new_c > cmax:
            new_c = cmax
        ctrl[w, act_ids[i]] = new_c


@wp.kernel
def _sync_ctrl_to_qpos_kernel(
    mask:     wp.array(dtype=int),                 # type: ignore (NWORLD,) 1 = target world (all-ones stands in for None)
    act_ids:  wp.array(dtype=int),                 # type: ignore (n_ctrl,)
    ctrl_qpa: wp.array(dtype=int),                 # type: ignore (n_ctrl,)
    n_ctrl:   int,
    qpos:     wp.array(dtype=float, ndim=2),       # type: ignore
    ctrl:     wp.array(dtype=float, ndim=2),       # type: ignore
):
    """integ-mode reset hook: align the setpoint of the target worlds with the realised qpos.

    After the reset path (``mjwarp.reset_data`` + pose-init graph) ``d.ctrl``
    is left at 0 (measured |ctrl−qpos| of 0.2–1.4 rad across actuators).
    delta/abs modes overwrite ctrl entirely every step so this is irrelevant
    for them, but integ mode **integrates onto the previous ctrl**, so the
    first step would start from a stale 0, jump to the end of the lead clamp
    (±0.2 rad) and produce an initial force spike on every reset. Setting
    ctrl := qpos here starts from lead 0 (= zero force)."""
    w = wp.tid()
    if mask[w] == 1:
        for i in range(n_ctrl):
            qa = ctrl_qpa[i]
            if qa >= 0:
                ctrl[w, act_ids[i]] = qpos[w, qa]


# ──────────────────────────────────────────────────────────────────────────
# Action applier
# ──────────────────────────────────────────────────────────────────────────

class WarpActionApplier:
    """Apply per-world delta-actions to a parallel warp env.

    Both wrist deltas (translation + rotation) live in the wrist body's
    local frame and are composed onto its current world pose; the policy
    commands "move/rotate along my own axes" rather than "along world
    axes", matching the JAX-era convention.

    Layout of one action (size = ``self.action_dim`` = 3 + 6 + n_ctrl):
        [0:3]        wrist_xyz_delta  (wrist-local frame; rotated to world
                                       via ``R_wrist`` and scaled by
                                       ``xyz_scale``)
        [3:9]        wrist_R_6D       (residual around identity — kernel
                                       adds ``[1,0,0,0,1,0]`` bias so action
                                       zero decodes to identity. Gram-Schmidt
                                       → R_delta, SLERP-scaled by
                                       ``rot_scale``, body-frame composed
                                       onto the wrist's current world rotation.)
        [9:9+n_ctrl] finger_delta     (scaled by ``finger_scale``, clamped to ctrl range)

    Optional task hooks (general — bound by the RL env, no-op when unused):
        * ``bind_stage_buffer(stage_wp)``  — bridge a per-world curriculum stage
          buffer (``(NWORLD,) int 0/1/2``) from the handler. Stored as
          ``self._stage_wp`` (ALWAYS valid — defaults to all-zeros, never None),
          so a stage-conditioned consumer can read it unconditionally.
        * ``bind_action_preprocessor(fn)`` — register ``fn(applier)`` called at
          the START of every :meth:`apply` to modify ``self._gpu_action`` in
          place (e.g. object_grasping's rule-based wrist-z lift curriculum). It
          runs only when bound AND ``self._action_preprocessor_enabled`` is True
          (evaluate.py flips that flag with ``[L]`` to compare raw vs assisted).
          Tasks without a curriculum (tracking, …) never bind one → skipped.

    Usage:
        applier = WarpActionApplier(env, hand_util, xyz_scale=0.005,
                                    finger_scale=0.02, rot_scale=0.1)
        # per control step:
        action = sample_or_policy_output()       # (NWORLD, action_dim) numpy/wp.array
        applier.write_action(action)
        applier.apply()                          # writes d.mocap_*, d.ctrl
        wp.capture_launch(env.capture_step.graph)  # mjwarp.step consumes new state

    Args:
        env:           ``SingleHandSubEnv`` instance with ``.m / .d / .mjm / .NWORLD``.
        hand_util:     HandUtils — used for ``rh_mocap_name``.
        xyz_scale:     m per unit raw action (default 0.005 = 5 mm).
        finger_scale:  ctrl unit per unit raw action (default 0.02 ≈ 1.1°).
        rot_scale:     SLERP coefficient applied to the decoded rotation delta.
                       1.0 = full rotation, 0.1 = 10% of decoded angle per step.
                       Policy outputs (tanh-bounded) can produce up to ~90° rotations,
                       so values in [0.05, 0.2] give smooth 4–18° per step.

        use_deadzone:        Master flag. ``False`` (default) keeps the
                             original behaviour — every scaled delta is
                             written through to mocap / ctrl regardless of
                             magnitude. ``True`` activates the three
                             component-wise deadzones below; sub-threshold
                             commands are zeroed before reaching the kernel
                             outputs so the wrist / fingers actually
                             "hold" once the policy's residual decays
                             (no slow drift past the target).
        xyz_deadzone_m:      Threshold on ``||action_xyz · xyz_scale||₂`` in
                             meters. Suggested order of magnitude: ~0.1 ×
                             ``xyz_scale``.
        rot_deadzone_deg:    Threshold on the final SLERP-scaled rotation
                             angle, **in degrees** (converted to radians
                             internally for the kernel). Suggested ~0.5–2°
                             for a smooth hold.
        finger_deadzone_deg: Per-actuator threshold on the scaled finger
                             delta, **in degrees** (radian-based comparison
                             inside the kernel). Applied to every ctrl
                             slot uniformly — for tendon / non-revolute
                             actuators the unit interpretation is the
                             user's responsibility.
    """

    def __init__(
        self,
        env,
        hand_util,
        xyz_scale:    float = 0.005,
        finger_scale: float = 0.02,
        rot_scale:    float = 1.0,
        # ── Wrist-control flag ────────────────────────────────────────────
        # ``True`` (default) → policy commands the full ``[Δxyz | Δrot6d |
        # Δfinger]`` action (wrist + fingers). ``False`` → **finger-only**:
        # the exposed ``action_dim`` shrinks to ``n_ctrl`` (policy outputs only
        # the finger deltas) and the wrist slots stay zero, so the wrist is
        # held at whatever pose it was reset to. Used to build a finger-pose-
        # only tracking env (see ``control_wrist`` YAML flag).
        control_wrist:       bool  = True,
        # ── Deadzone knobs (off by default → backward-compatible) ─────────
        use_deadzone:        bool  = False,
        xyz_deadzone_m:      float = 0.0,
        rot_deadzone_deg:    float = 0.0,
        finger_deadzone_deg: float = 0.0,
        # ── Actuator scoping (multi-hand scenes) ─────────────────────────
        # ``None`` (default) → drive ALL scene actuators (``arange(mjm.nu)``)
        # — bit-identical to the historical single-hand behaviour. In a
        # two-hand scene (``LeaderFollowerSubEnv``) pass each hand's actuator
        # id subset (e.g. ``ctrl_idxs_under_wrist(mjm, wid)``) so this
        # applier's ``n_ctrl``/``action_dim``/ctrl tables cover only that
        # hand and ``apply()`` writes only that hand's ``ctrl`` columns.
        act_ids: "Optional[np.ndarray]" = None,
        # ── finger action representation ─────────────────────────────────
        # False (default) = delta: ``ctrl = qpos + a·finger_scale`` (re-anchoring).
        # True = absolute-normalized: ``ctrl = denorm(a ∈ [−1,1] → ctrl range)``
        # — the policy predicts the absolute setpoint directly. See the kernel comment (finger_abs).
        finger_abs_action:   bool  = False,
        # "delta" | "abs" | "integ" — see the kernel finger_mode comment. finger_abs_action
        # is a backward-compatible alias for "abs" (the mode string wins if both are given).
        finger_action_mode:  str   = "delta",
        # Lead clamp for integ mode [rad] = max force / kp. Default equals delta
        # mode's implicit cap (finger_scale).
        finger_lead_max_rad: float = 0.2,
        # integ leak λ (see the kernel finger_leak comment). 1.0 = pure integration.
        finger_leak:         float = 1.0,
        # Window length K for "hist" mode (see the kernel finger_mode==3 comment). K=1 equals delta.
        finger_hist_len:     int   = 4,
        # Sum normalization: True → /K (mean; same command scale as delta), False → plain sum
        # (K× the lead for the same a → saturates the lead clamp immediately; not recommended).
        finger_hist_mean:    bool  = True,
    ):
        self.env          = env
        self.hand_util    = hand_util
        self.xyz_scale    = float(xyz_scale)
        self.finger_scale = float(finger_scale)
        self.rot_scale    = float(rot_scale)
        mode = str(finger_action_mode).lower()
        if mode == "delta" and bool(finger_abs_action):
            mode = "abs"
        if mode not in ("delta", "abs", "integ", "hist"):
            raise ValueError(
                f"finger_action_mode must be delta|abs|integ|hist, got {finger_action_mode!r}")
        self.finger_action_mode = mode
        self.finger_abs_action  = (mode == "abs")
        self.finger_lead_max_rad = float(finger_lead_max_rad)
        self.finger_leak         = float(finger_leak)
        self._finger_mode_int = {"delta": 0, "abs": 1, "integ": 2, "hist": 3}[mode]
        self.finger_hist_len  = max(1, int(finger_hist_len))
        self.finger_hist_mean = bool(finger_hist_mean)
        self._finger_hist_idx = 0

        # ── Deadzone state ────────────────────────────────────────────────
        # Stored as both the user-facing degree values (for inspection /
        # logging) and the radian counterparts the kernel actually consumes
        # — converting once per attribute change beats converting per world
        # per step. ``set_deadzone`` keeps the two representations in sync.
        self.use_deadzone        = bool(use_deadzone)
        self.xyz_deadzone_m      = float(xyz_deadzone_m)
        self.rot_deadzone_deg    = float(rot_deadzone_deg)
        self.finger_deadzone_deg = float(finger_deadzone_deg)
        self._rot_deadzone_rad    = float(rot_deadzone_deg)    * (np.pi / 180.0)
        self._finger_deadzone_rad = float(finger_deadzone_deg) * (np.pi / 180.0)

        mjm = env.mjm

        mocap_body_id = mujoco.mj_name2id(                                     # type: ignore
            mjm, mujoco.mjtObj.mjOBJ_BODY, hand_util.rh_mocap_name             # type: ignore
        )
        if mocap_body_id == -1:
            raise ValueError(f"mocap body '{hand_util.rh_mocap_name}' not found")
        self.mocap_id = int(mjm.body_mocapid[mocap_body_id])
        if self.mocap_id == -1:
            raise ValueError(f"body '{hand_util.rh_mocap_name}' is not a mocap body")

        # Wrist base body — the freejoint-driven body the mocap welds to.
        # Used as the translation anchor for ``mocap_pos = xpos[wrist] + dxyz``.
        wrist_body_id = mujoco.mj_name2id(                                     # type: ignore
            mjm, mujoco.mjtObj.mjOBJ_BODY, hand_util.rh_wrist_base_name        # type: ignore
        )
        if wrist_body_id == -1:
            raise ValueError(
                f"wrist base body '{hand_util.rh_wrist_base_name}' not found"
            )
        self.wrist_body_id = int(wrist_body_id)

        # Actuator scope: identity (all scene actuators) unless a subset is
        # given. All per-actuator tables below (ctrl range / qpa / raw-target
        # export) are indexed in LOCAL slot order ``i``; the kernel maps a
        # local slot to its scene ``ctrl`` column via ``act_ids[i]``.
        if act_ids is None:
            act_ids_np = np.arange(int(mjm.nu), dtype=np.int32)
        else:
            act_ids_np = np.asarray(act_ids, dtype=np.int32).ravel()
            assert act_ids_np.size > 0, "act_ids must be non-empty"
            assert (0 <= act_ids_np).all() and (act_ids_np < int(mjm.nu)).all(), (
                f"act_ids out of range [0, {int(mjm.nu)}): {act_ids_np}")
            assert np.unique(act_ids_np).size == act_ids_np.size, (
                f"act_ids has duplicates: {act_ids_np}")
        self._act_ids_np = act_ids_np
        self.n_ctrl = int(act_ids_np.size)
        self.NWORLD = int(env.NWORLD)

        # action layout offsets. The INTERNAL action buffer is ALWAYS the full
        # ``9 + n_ctrl`` layout ([Δxyz(3) | Δrot6d(6) | Δfinger(n_ctrl)]) so the
        # apply kernel is unchanged. ``control_wrist=False`` only shrinks the
        # POLICY-facing ``action_dim`` to ``n_ctrl`` and leaves the wrist slots
        # ([0:9]) permanently zero → the wrist is held.
        self.xyz_off          = 0
        self.rot6d_off        = 3
        self.finger_off       = 9
        self._full_action_dim = 9 + self.n_ctrl
        self.control_wrist    = bool(control_wrist)
        self.action_dim       = self._full_action_dim if self.control_wrist else int(self.n_ctrl)

        device       = env.d.qpos.device
        self._device = device

        # ctrl bounds (per-actuator, gathered over the actuator scope)
        ctrl_range  = np.asarray(mjm.actuator_ctrlrange, dtype=np.float32)
        ctrl_min_np = ctrl_range[act_ids_np, 0].copy()
        ctrl_max_np = ctrl_range[act_ids_np, 1].copy()
        # Some actuators may be unbounded (range=0,0). Use ±inf to skip clipping.
        unbounded = (ctrl_min_np == 0) & (ctrl_max_np == 0)
        ctrl_min_np[unbounded] = -1e9
        ctrl_max_np[unbounded] = +1e9
        self._ctrl_min = wp.array(ctrl_min_np, dtype=float, device=device)
        self._ctrl_max = wp.array(ctrl_max_np, dtype=float, device=device)

        # ctrl → qpos-addr lookup (per LOCAL actuator slot).
        # ``ctrl_qpa[i] = qposadr(joint of act_ids[i])`` for direct joint
        # actuators (``mjTRN_JOINT``), ``-1`` otherwise — tendon / site / body
        # actuators have no scalar qpos counterpart, so the kernel falls
        # back to using ``ctrl[w, act_ids[i]]`` as the base for those slots.
        ctrl_qpa_np = np.full(self.n_ctrl, -1, dtype=np.int32)
        for i, a in enumerate(act_ids_np):
            tt = int(mjm.actuator_trntype[a])
            if tt == int(mujoco.mjtTrn.mjTRN_JOINT):                            # type: ignore
                jid = int(mjm.actuator_trnid[a, 0])
                if 0 <= jid < int(mjm.njnt):
                    ctrl_qpa_np[i] = int(mjm.jnt_qposadr[jid])
        self._ctrl_qpa = wp.array(ctrl_qpa_np, dtype=int, device=device)
        self._act_ids  = wp.array(act_ids_np, dtype=int, device=device)
        self._n_direct_ctrl = int((ctrl_qpa_np >= 0).sum())

        # action buffer (GPU-resident, reused every call) — ALWAYS full-width.
        zeros = np.zeros((self.NWORLD, self._full_action_dim), dtype=np.float32)
        self._gpu_action = wp.array(zeros, dtype=float, device=device)
        # Policy-facing writable view: the full buffer when control_wrist, else
        # only the finger columns ([9:]) so writes never touch the (held) wrist
        # slots. Shape is always ``(NWORLD, self.action_dim)``.
        _full_view = wp.to_torch(self._gpu_action)
        self.action_torch = _full_view if self.control_wrist else _full_view[:, self.finger_off:]

        # CPU-side staging buffer (reused by write_action(numpy) — avoids
        # re-allocating a fresh wp.array on every call).
        self._cpu_staging = wp.array(zeros.copy(), dtype=float, device='cpu')

        # Raw (pre-clamp) commanded finger target, filled by ``apply()`` every
        # step. ``d.ctrl`` is clamped to ``[ctrl_min, ctrl_max]`` inside the
        # kernel, so it hides any command that overshoots a joint limit; this
        # buffer preserves the un-clamped value for a joint-limit penalty term
        # (``HandPoseTrackingHandler._joint_limit_penalty_kernel``). Exposed as
        # a zero-copy torch view for convenience.
        # Ring buffer for "hist" mode (NWORLD, K, n_ctrl). The kernel takes the
        # argument in every mode, so always allocate it but minimize to K=1
        # elsewhere (negligible memory).
        _K = self.finger_hist_len if self._finger_mode_int == 3 else 1
        self._finger_hist_wp = wp.zeros((self.NWORLD, _K, self.n_ctrl),
                                        dtype=float, device=device)
        self._finger_hist_K = _K
        self._raw_ctrl_target = wp.zeros((self.NWORLD, self.n_ctrl),
                                         dtype=float, device=device)
        self.raw_ctrl_target_torch = wp.to_torch(self._raw_ctrl_target)

        # GPU seed buffer for sample_random_gpu() (mirrors orchestrator pattern).
        self._gpu_action_seed   = wp.array(np.asarray([0], dtype=np.int32),
                                           dtype=int, device=device)
        self._action_seed_value = 0

        # Per-world curriculum stage buffer (wp.array (NWORLD,) int 0/1/2),
        # consumed by stage-conditioned action pre-processing in ``apply()``
        # (e.g. a rule-based wrist-z lift curriculum). It is ALWAYS a valid
        # array — defaults to all-zeros (stage 0 → no-op), so tasks WITHOUT a
        # curriculum (e.g. tracking) work bug-free without ever calling
        # ``bind_stage_buffer``. A curriculum task overrides it with its own
        # live buffer via ``bind_stage_buffer``; it is never ``None``, so any
        # consuming kernel can read it unconditionally.
        self._default_stage_wp = wp.zeros(self.NWORLD, dtype=int, device=device)
        self._stage_wp = self._default_stage_wp

        # Per-world wrist target-hold flags (see the apply kernel; default
        # all-zeros = the original realised-anchor convention). Tasks with a
        # stage curriculum bind their own live buffers via
        # ``bind_wrist_hold_buffers`` to drive target hold (1) in lift-latch /
        # hold and manual-lift target integration (2).
        self._default_wrist_hold_wp = wp.zeros(self.NWORLD, dtype=int, device=device)
        self._wrist_pos_hold_wp = self._default_wrist_hold_wp
        self._wrist_rot_hold_wp = self._default_wrist_hold_wp
        # Max lead (m) by which the target may lead the realised wrist in z under
        # pos_hold=2 (target integration). ≤0 = no cap.
        self.pos_lead_max_m = 0.03
        # Upper bound (rad) on the target↔realised angle under rot_hold=2 (rotation target integration); 0 = unbounded.
        self.rot_lead_max_rad = 0.0
        self.rot_leak = 0.0

        # Optional task-specific action pre-processor, bridged in from the RL
        # env via ``bind_action_preprocessor``. Called at the START of every
        # ``apply()`` as ``fn(self)`` and may modify ``self._gpu_action`` in
        # place (e.g. a stage-conditioned wrist-z lift curriculum). ``None`` =
        # no pre-processing (the general/default case — tracking etc.).
        self._action_preprocessor = None
        # Runtime on/off switch for the pre-processor (default ON for training).
        # evaluate.py flips this to compare the raw policy vs the rule-based
        # assist without re-binding. apply() runs the pre-processor only when
        # both a preprocessor is bound AND this flag is True.
        self._action_preprocessor_enabled = True

    # ──────────────────────────────────────────────────────────────────
    # API: write + apply
    # ──────────────────────────────────────────────────────────────────

    def write_action(self, action) -> None:
        """Upload ``(NWORLD, action_dim)`` action to the GPU buffer.

        Accepts ``np.ndarray`` (float32) or pre-existing ``wp.array``.
        For the numpy path the data is written into a re-used CPU staging
        wp.array, then copied to GPU — avoids per-call wp.array allocation.
        Use this just before ``apply()`` each control step.
        """
        exp = (self.NWORLD, self.action_dim)
        if isinstance(action, np.ndarray):
            assert action.shape == exp, f"action shape {action.shape} != {exp}"
            buf = self._cpu_staging.numpy()
            if self.control_wrist:
                # In-place fill of the pre-allocated CPU staging buffer
                buf[:] = action.astype(np.float32, copy=False)
            else:
                # finger-only: wrist slots [0:9] stay zero (wrist held)
                buf[:] = 0.0
                buf[:, self.finger_off:] = action.astype(np.float32, copy=False)
            wp.copy(self._gpu_action, self._cpu_staging)
        else:
            assert action.shape == exp
            if self.control_wrist:
                wp.copy(self._gpu_action, action)
            else:
                # zero-copy write into the finger columns of the full buffer
                self.action_torch[...] = wp.to_torch(action)

    def bind_stage_buffer(self, stage_wp) -> None:
        """Bridge a per-world curriculum **stage** buffer from the handler.

        ``stage_wp`` is a ``wp.array`` of shape ``(NWORLD,)`` (int 0/1/2)
        produced by the task handler (e.g. ObjectGrasping's
        ``_obs_ep_stage_kernel``). It is stored on the applier so
        :meth:`apply` can run stage-conditioned pre-processing of the action
        (e.g. a rule-based wrist-z lift that fires from stage ≥ 1).

        Pass ``None`` to revert to the default all-zeros (stage 0) buffer.
        ``self._stage_wp`` is NEVER ``None`` afterwards, so tasks without a
        curriculum (which never call this) and unbinds both stay safe — a
        consuming kernel always has a valid array reading stage 0 (no-op).
        """
        if stage_wp is None:
            self._stage_wp = self._default_stage_wp
            return
        if int(stage_wp.shape[0]) != self.NWORLD:
            raise ValueError(
                f"stage buffer NWORLD {int(stage_wp.shape[0])} != applier NWORLD {self.NWORLD}"
            )
        self._stage_wp = stage_wp

    def bind_wrist_hold_buffers(self, pos_hold_wp, rot_hold_wp) -> None:
        """Bridge per-world wrist target-hold flag buffers from the handler.

        Both are ``wp.array(dtype=int)`` of shape ``(NWORLD,)``. The apply kernel
        reads them to condition the mocap writes — pos: 0=realised+Δ (original)
        /1=skip write (keep previous target)/2=integrate target (manual lift),
        rot: 0=original/1=skip write. Bind live buffers that the handler's
        action pre-processor rewrites every step. Passing ``None`` reverts to
        the default all-zeros (original behaviour)."""
        if pos_hold_wp is None or rot_hold_wp is None:
            self._wrist_pos_hold_wp = self._default_wrist_hold_wp
            self._wrist_rot_hold_wp = self._default_wrist_hold_wp
            return
        for buf in (pos_hold_wp, rot_hold_wp):
            if int(buf.shape[0]) != self.NWORLD:
                raise ValueError(
                    f"hold buffer NWORLD {int(buf.shape[0])} != applier NWORLD {self.NWORLD}"
                )
        self._wrist_pos_hold_wp = pos_hold_wp
        self._wrist_rot_hold_wp = rot_hold_wp

    def bind_action_preprocessor(self, fn) -> None:
        """Register a task-specific **action pre-processor** (general hook).

        ``fn`` is a callable ``fn(applier) -> None`` invoked at the start of
        every :meth:`apply` (before the apply kernel). It may modify
        ``applier._gpu_action`` in place — typically a stage-conditioned rule
        (e.g. ObjectGrasping's wrist-z lift curriculum), reading
        ``applier._stage_wp``. Each RL env writes its own curriculum function
        and binds it here; tasks without one (tracking, …) simply never bind,
        so ``apply()`` skips the hook. Pass ``None`` to clear.
        """
        self._action_preprocessor = fn

    @property
    def has_action_preprocessor(self) -> bool:
        """True when a task bound a pre-processor via ``bind_action_preprocessor``.
        Public probe for drivers (e.g. ``scripts/evaluate.py``'s [L] toggle) —
        avoids reaching into the private ``_action_preprocessor`` attr."""
        return self._action_preprocessor is not None

    def set_action_preprocessor_enabled(self, enabled: bool) -> None:
        """Enable/disable the bound pre-processor without unbinding it.

        Public toggle for drivers: eval typically disables it (no training-step
        decay → a curriculum assist would run at full strength forever) and can
        re-enable it interactively. No-op when nothing is bound."""
        self._action_preprocessor_enabled = bool(enabled)

    def reset_finger_hist(self, mask_wp=None) -> None:
        """"hist"-mode reset hook: zero the action history of the target worlds.

        Without clearing, the last K−1 actions of the previous episode would
        leak into the first-step command of the new episode (force spike at reset).

        **The ring index is deliberately not touched here.** ``per_world_reset_if_done``
        is called **unconditionally every step** even when no world is done
        (sync-free convention), so resetting idx to 0 here would make apply
        overwrite only slot 0 every step and collapse the window: the remaining
        K−1 slots would stay 0 and the effective command would be **weakened
        K-fold** to ``finger_scale/K`` (with an effective finger_scale of
        0.2→0.05, contact never forms at all).
        idx is a counter shared across worlds and the kernel **sums the K slots
        order-independently**, so as long as it keeps advancing it works
        correctly regardless of per-world resets."""
        if self._finger_mode_int != 3:
            return
        h = wp.to_torch(self._finger_hist_wp)
        if mask_wp is None:
            h.zero_()
            self._finger_hist_idx = 0      # reset the index only on a full reset
        else:
            m = wp.to_torch(mask_wp).to(torch.bool)
            h[m] = 0.0

    def sync_setpoint_to_qpos(self, mask_wp=None) -> None:
        """integ-mode reset hook: ``ctrl[w] := qpos[w]`` (all worlds if mask is
        None). In "hist" mode this delegates to clearing the action history;
        delta/abs overwrite every step, so it is a no-op for them."""
        if self.finger_action_mode == "hist":
            self.reset_finger_hist(mask_wp)
            return
        if self.finger_action_mode != "integ":
            return
        if mask_wp is None:
            if not hasattr(self, "_all_ones_mask_wp"):
                self._all_ones_mask_wp = wp.array(
                    np.ones(self.NWORLD, dtype=np.int32), dtype=int, device=self._gpu_action.device)
            mask_wp = self._all_ones_mask_wp
        wp.launch(_sync_ctrl_to_qpos_kernel, dim=self.NWORLD,
                  inputs=[mask_wp, self._act_ids, self._ctrl_qpa, int(self.n_ctrl),
                          self.env.d.qpos, self.env.d.ctrl])

    def apply(self) -> None:
        """Launch kernel — writes to ``env.d.mocap_pos / mocap_quat / ctrl``.

        Call AFTER ``write_action`` and BEFORE the captured step launch.

        Deadzone parameters are read at launch time, so updates via
        ``set_deadzone(...)`` (or direct attribute assignment) take effect
        on the next ``apply()`` without re-compiling the kernel.
        """
        # Task-specific action pre-processing hook (general). The RL env
        # binds a curriculum fn via ``bind_action_preprocessor``; it modifies
        # ``self._gpu_action`` in place using the bridged per-world stage
        # (``self._stage_wp``, always a valid (NWORLD,) int array — default
        # all-zeros = stage 0 = no-op). No bind → skipped (tracking etc. safe).
        if self._action_preprocessor is not None and self._action_preprocessor_enabled:
            self._action_preprocessor(self)

        wp.launch(
            _apply_action_kernel, dim=self.NWORLD,
            inputs=[
                self._gpu_action,
                int(self.mocap_id),
                int(self.wrist_body_id),
                int(self.n_ctrl),
                self._act_ids,
                int(self.xyz_off),
                int(self.rot6d_off),
                int(self.finger_off),
                float(self.xyz_scale),
                float(self.finger_scale),
                float(self.rot_scale),
                int(self._finger_mode_int),
                float(self.finger_lead_max_rad),
                float(self.finger_leak),
                self._finger_hist_wp,
                int(self._finger_hist_K),
                int(self._finger_hist_idx),
                float(self._finger_hist_K if self.finger_hist_mean else 1),
                # In finger-only mode the mocap is not written, so the wrist pose
                # fixed at reset is kept (blocks re-anchor drift).
                int(1) if self.control_wrist else int(0),
                self._ctrl_min,
                self._ctrl_max,
                self._ctrl_qpa,
                self.env.d.qpos,
                self.env.d.xpos,
                self.env.d.xmat,
                self.env.d.mocap_pos,
                self.env.d.mocap_quat,
                self.env.d.ctrl,
                # Deadzone (host → kernel; rad form pre-converted).
                int(1 if self.use_deadzone else 0),
                float(self.xyz_deadzone_m),
                float(self._rot_deadzone_rad),
                float(self._finger_deadzone_rad),
                # Wrist target-hold flags (default all-zeros = original convention).
                self._wrist_pos_hold_wp,
                self._wrist_rot_hold_wp,
                float(self.pos_lead_max_m),
                float(self.rot_lead_max_rad),
                float(self.rot_leak),
                # Raw (pre-clamp) commanded target side-output.
                self._raw_ctrl_target,
            ],
        )
        if self._finger_mode_int == 3:      # advance to the next ring-buffer slot
            self._finger_hist_idx = (self._finger_hist_idx + 1) % self._finger_hist_K

    # ──────────────────────────────────────────────────────────────────
    # Deadzone configuration
    # ──────────────────────────────────────────────────────────────────

    def set_deadzone(
        self,
        use_deadzone:        Optional[bool]  = None,
        xyz_deadzone_m:      Optional[float] = None,
        rot_deadzone_deg:    Optional[float] = None,
        finger_deadzone_deg: Optional[float] = None,
    ) -> None:
        """Update deadzone settings in-place.

        Any argument left as ``None`` keeps its current value. The next
        ``apply()`` picks up the change — there is no kernel recompile or
        graph re-capture (kernel reads the scalars at launch time).

        Notes:
          * The two angular thresholds are stored in degrees on ``self``
            (for inspection) and pre-converted to radians on the private
            ``_*_rad`` slots that the kernel reads.
          * ``use_deadzone=False`` keeps the thresholds but skips the
            deadzone branches entirely — cheap toggling.
        """
        if use_deadzone is not None:
            self.use_deadzone = bool(use_deadzone)
        if xyz_deadzone_m is not None:
            self.xyz_deadzone_m = float(xyz_deadzone_m)
        if rot_deadzone_deg is not None:
            self.rot_deadzone_deg = float(rot_deadzone_deg)
            self._rot_deadzone_rad = float(rot_deadzone_deg) * (np.pi / 180.0)
        if finger_deadzone_deg is not None:
            self.finger_deadzone_deg = float(finger_deadzone_deg)
            self._finger_deadzone_rad = float(finger_deadzone_deg) * (np.pi / 180.0)

    # ──────────────────────────────────────────────────────────────────
    # Convenience: random action samplers
    # ──────────────────────────────────────────────────────────────────

    def sample_random_gpu(
        self,
        xyz_sigma:    float = 1.0,
        rot_noise:    float = 0.05,
        finger_sigma: float = 1.0,
    ) -> None:
        """Sample a random action on the GPU IN-PLACE into ``self._gpu_action``.

        No host roundtrip — call ``apply()`` afterwards (or use
        ``apply_random_action()`` which fuses both). Faster than the numpy
        ``sample_random`` + ``write_action`` path: skips numpy RNG, host
        memory traffic, and CPU→GPU copy.

        Each call advances an internal seed buffer so successive launches
        produce independent samples (graph-capture-safe — the seed is read
        from a wp.array, so we update it via a tiny set-seed kernel).
        """
        # Bump seed (matches Env_Orchestrator pattern)
        self._action_seed_value = (self._action_seed_value + 1) & 0x7FFFFFFF
        wp.launch(
            _grit_set_seed_kernel, dim=1,
            inputs=[self._gpu_action_seed, int(self._action_seed_value)],
        )
        # Sample (fills the full [xyz|rot6d|finger] buffer)
        wp.launch(
            _grit_sample_random_action_kernel, dim=self.NWORLD,
            inputs=[
                self._gpu_action_seed,
                self._gpu_action,
                int(self.xyz_off),
                int(self.rot6d_off),
                int(self.finger_off),
                int(self.n_ctrl),
                float(xyz_sigma),
                float(rot_noise),
                float(finger_sigma),
            ],
        )
        # finger-only: re-zero the wrist slots so the sampled wrist noise is
        # discarded and the wrist stays held.
        if not self.control_wrist:
            wp.to_torch(self._gpu_action)[:, :self.finger_off] = 0.0

    def apply_random_action(
        self,
        xyz_sigma:    float = 1.0,
        rot_noise:    float = 0.05,
        finger_sigma: float = 1.0,
    ) -> None:
        """Fused convenience: sample (GPU) + apply. Equivalent to
        ``self.sample_random_gpu(...); self.apply()`` but groups the two
        launches for callsite brevity.
        """
        self.sample_random_gpu(xyz_sigma, rot_noise, finger_sigma)
        self.apply()

    def sample_random(
        self,
        xyz_sigma:    float = 1.0,
        rot_noise:    float = 0.05,
        finger_sigma: float = 1.0,
        rng=None,
    ) -> np.ndarray:
        """Sample a random ``(NWORLD, action_dim)`` action on CPU as numpy.

        Use ``sample_random_gpu()`` (or ``apply_random_action()``) instead in
        tight loops — the GPU sampler skips numpy RNG + CPU→GPU upload.
        This numpy version is kept for debugging / inspection / unit tests.

        - xyz: ``N(0, xyz_sigma)`` — multiplied by ``self.xyz_scale`` in the
          kernel (so default σ=1 gives 5 mm jitter at xyz_scale=0.005).
        - 6D rotation: ``N(0, rot_noise)`` — residual around identity. The
          apply kernel adds ``[1,0,0,0,1,0]`` internally so action zero
          decodes to identity (small rotation delta around identity).
        - finger: ``N(0, finger_sigma)`` — multiplied by ``self.finger_scale``.
        """
        rng = rng if rng is not None else np.random.default_rng()
        # finger-only: the policy action is just the n_ctrl finger deltas.
        if not self.control_wrist:
            return rng.normal(0.0, finger_sigma,
                              (self.NWORLD, self.n_ctrl)).astype(np.float32)
        a = np.zeros((self.NWORLD, self.action_dim), dtype=np.float32)
        a[:, self.xyz_off:self.xyz_off + 3] = rng.normal(
            0.0, xyz_sigma, (self.NWORLD, 3)
        )
        a[:, self.rot6d_off:self.rot6d_off + 6] = rng.normal(
            0.0, rot_noise, (self.NWORLD, 6)
        ).astype(np.float32)
        a[:, self.finger_off:self.finger_off + self.n_ctrl] = rng.normal(
            0.0, finger_sigma, (self.NWORLD, self.n_ctrl)
        )
        return a
