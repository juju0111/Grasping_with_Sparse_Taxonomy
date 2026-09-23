"""grit/training/rl_envs/grasping_core.py — the grasp-and-lift task (shared by teacher and student).

:class:`GraspingCore` owns everything both policies share: the observation
blocks, the reward terms + mixer, the curriculum, the stage clock, termination,
the re-grasp retry / early-success episode logic and the BPS shape-feature tail.
:mod:`grasping_teacher` adds the privileged critic and the Lagrangian penalty
constraints; :mod:`grasping_student` replaces the observation tail with the
deployable partial-view / masked / history observation. Each is ONE level of
inheritance from this class and overrides a handful of named hooks:

    _setup_obs_buffers · _setup_reward_done_buffers · _reset_task_buffers
    _before_core_obs · _collect_obs_tail · _apply_ep_norm_horizon · _collect_obs_kernel
    _collect_success_kernel · _post_cond_update_hook · step · on_train_progress

────────────────────────────────────────────────────────────────────────────
Original module notes (core task):
ObjectGrasping RL task — handler + env + kernels.

Lift a single target object off the table while:

  * approaching it with the wrist toward a per-world **target_pnt** sampled
    from the object PCD (legacy ``hand_approach_process`` +
    ``target_pnt = pcd[argmin(dist)] + noise``),
  * closing the active-finger bodies around the object (legacy
    ``tip_link_process``),
  * mimicking a per-world **taxonomy template** ``target_qpos`` so the
    policy converges on a *chosen* grasp class (legacy ``mimic_process``
    with taxonomy-injected uniform sampler),
  * keeping the wrist x-axis aligned to the **first column** of a frozen
    per-world ``target_R`` (3, 3) snapshotted at reset — i.e. the legacy
    ``target_dir`` is now ``target_R[:, 0]`` (legacy ``wrist_dir_coeff`` /
    ``heading_direction``),
  * lifting the object once the curriculum gate opens — ``r_lift`` only
    fires when ``ep_step >= LIFT_STEP`` (legacy ``obj_z_process`` gated by
    ``lift_step``), and
  * avoiding self-collisions / table pushes (legacy
    ``self_collision_contact_penalty`` + ``obj_table_push_penalty``).

Curriculum stages (matching the legacy ``lift_step`` / ``hold_step``):

  * **stage 0** (``ep_step <  LIFT_STEP``)   approach + finger close + mimic.
  * **stage 1** (``LIFT_STEP <= ep_step < HOLD_STEP``)  lift reward
                                              ``r_lift`` starts firing.
  * **stage 2** (``ep_step >= HOLD_STEP``)   hold — bonus streak counts
                                              toward success.

Observation — block order MATCHES :meth:`GraspingHandler.obs_term_layout`
/ :meth:`_collect_obs_kernel` (the single source of truth; ``obs_dim`` asserts
the total). size = ``3·(n_sensors-1) + 3 + 6 + n_sensors + 6 + 3·n_ctrl
+ n_sensors + 4·n_obj_contact + 1 + 12 + 2``::

    [ site_pcd_err(3·(n_sensors-1))  R_wrist^T · (best_pt[s] − site_xpos[s]),
                                     per hand site, fore-arm excluded (REUSED reward best_pt)
    | hand_center_pcd_err(3)   R_wrist^T · (best_pt_hc − hand_center_p)  (hand-center → nearest PCD)
    | rot6d(6)                 hand-center 6D (x,y axes) MINUS frozen target_R (x,y axes), world
                               (target_R-only half is commented out in the kernel → 6, not 12)
    | site_z_above_table(n_sensors)  site_xpos.z − table_height, per hand site (global)
    | obj_pose_diff(6)         obj_p_init − obj_p (world xyz, 3) + r2rpy(R_init^T · R_obj) (3)
    | joint_qpos(n_ctrl)       qpos[ctrl_adr]
    | qpos_tax_err(n_ctrl)     target_qpos − qpos[ctrl_adr]  (per-joint taxonomy mimic err;
                               |err| < OBS_TAX_ERR_DEADZONE → 0, JAX v2 hand_pose_error convention)
    | finger_link_mask(n_sensors)  specific_finger_link_mask: taxonomy active(1)/rest(0),
                                   sensor space (palm + arm + fingers; arm slot always 0)
    | per_slot_contact(4·n)    obj_contact_bool(n) + obj_contact_impulse(n)
                               + obstacle_bool(n) + obstacle_impulse(n)   (n = n_obj_contact;
                               REUSES the reward per-slot impulse buffers, force-scaled)
    | obj_touch_force(1)       Σ object touch sensors · force_scale (total force on obj)
    | obj_ground_force(1)      approximate object-ground force (optional pre-lift gating)
      ── the contact-force channels above (per_slot impulse included) are compressed
         by OBS_FORCE_LOG1P to log1p(clip(f,0,CAP))/log1p(CAP) ∈ [0,1] (bool channels unchanged)
    | torque_proxy(n_ctrl)     ctrl_target − qpos[ctrl_adr]  (position err ≈ torque;
                               OBS_TORQUE_PROXY_LOG1P → sign·log1p compression)
    | fd_vel(12)               pose-FD velocity, prev-wrist local frame (d.cvel not used):
                               [R_w_prevᵀ·Δp_w/dt | log(R_w_prevᵀ·R_w)/dt
                               | R_w_prevᵀ·Δp_obj/dt − wrist_lin | obj rotvec(wrist) − wrist_ang]
                               (JAX v2 channel layout; INCLUDE_FD_VEL_IN_OBS, _fd_vel_update_kernel)
    | proj_gravity(3)          R_wristᵀ·(0,0,−1) — absolute orientation cue
    | lift_target_vec(3)       R_wristᵀ·(obj_spawn+LIFT_TARGET_M − p_obj) — residual to the lift target, always on
    | ep_step_norm(1)          ep_step / MAX_EPISODE_STEPS in [0, 1]
    | stage_onehot(3)          curriculum stage one-hot (0/1/2)
    ]

REUSE NOTE: the per-slot contact block is NOT recomputed — it reads the reward
pipeline's existing per-slot impulse buffers (``hand_obj_contact_impulse_per_slot``
/ ``self_impulse_per_slot`` / ``tbl_impulse_per_slot``, all force-scaled, parallel
slot order). The bools are ``impulse > 0``. These ``n_obj_contact`` slots are the
hand contact bodies (palm + fingers; NO arm — distinct from the ``n_sensors`` mask
space above, which carries an always-zero arm slot at index 1).

Action layout is inherited from ``WarpActionApplier`` (``9 + n_ctrl``)::

    [Δxyz(3) wrist-local | Δrot6D(6) residual-around-identity | Δfinger_ctrl(n_ctrl)]

Reward (NOT a flat sum — a **multiplicative inequality-coefficient** mixer;
see ``_object_grasping_reward_mix_kernel``). Each weighted term::

    r_approach   =  W_APPROACH    · exp(-K_APPROACH · ||hand_center - nearest_obj_pcd||)
    r_finger     =  W_FINGER      · exp(-K_FINGER   · mean_i ||p_af_i - nearest_obj_pcd||²)
    r_face_dir   =  W_FACE_DIR    · mean_active max(0, <face_dir, look_at_obj>)
    r_mimic_qerr = -W_MIMIC       · mimic_qpos_err  (split-L1 hinge, active/rest dead-zones)
    r_mimic      =  W_MIMIC       · exp(-K_MIMIC · …)        (RBF form, used as a coefficient)
    r_wrist_dir  =  W_WRIST_DIR   · max(0, <hand_center_x, target_R[:, 0]>)   ∈ [0, W]
    r_wrist_vel  =  W_VEL         · exp(-K_VEL · …)          (wrist stationarity)   ∈ [0, W]
    r_obj_vel    =  W_OBJ_VEL     · exp(-K_OBJ_VEL · …)      (object stationarity)  ∈ [0, W]
    r_obj_xy_c   =  W_OBJ_XY_COEFF· exp(coeff·|Δxy|)         (object near spawn xy) ∈ [0, W]
    r_obj_R_c    =  W_OBJ_R_COEFF · exp(coeff·(1-upright))   (object upright)       ∈ [0, W]
    r_lift       =  W_LIFT        · clip(obj_lift_z / LIFT_TARGET_M, 0, 1)   (only ep_step ≥ LIFT_STEP)
    r_ctrl       =  exp(-CTRL_COST       · ||action||²)      ∈ (0, 1]
    r_smooth     =  exp(-K_ACTION_SMOOTH · ||action_t - action_{t-1}||²)     ∈ (0, 1]
    r_contact_dir/pos, r_affordance_pos      taxonomy-aware contact bonuses (+)
    r_contact_neg, r_affordance_neg,         taxonomy-aware contact penalties (positive magnitudes)
    r_self(+impulse), r_table(+impulse),     finger-weighted self / obstacle (table+forearm) penalties
    r_obj_touch  =  W_OBJ_TOUCH   · Σ obj touch sensors      (force on object → gentle-grasp penalty)

assembled as (see the mixer kernel for the exact expression)::

    hand_ineq_coeff = r_wrist_dir · r_wrist_vel · r_ctrl · r_smooth
    r_hand_process  = (r_approach + r_finger + r_face_dir + r_mimic_qerr) · hand_ineq_coeff
    obj_ineq_coeff  = r_obj_vel · r_obj_xy_c · r_obj_R_c
    ineq_coeff      = r_mimic · obj_ineq_coeff
    r_obj_process   = r_lift · ineq_coeff
    hand_obj_bonus  = (r_contact_dir + r_contact_pos + r_affordance_pos) · ineq_coeff
    hand_obj_penalty= r_contact_neg + r_affordance_neg + r_self(+impulse) + r_table(+impulse) + r_obj_touch
    bonus           = W_BONUS · GRASP_BONUS_VALUE · min(streak, SUCCESS_STREAK_MIN)   (when bonus_active)
    r_violation_pen = W_VIOLATION_PEN   on a violation-terminal step — done reason
                      ∉ {success, timeout} (obj_z / wrist_up / obj_xy / crush / obj_fell)
    total = (r_hand_process + r_obj_process + hand_obj_bonus − hand_obj_penalty + bonus − r_violation_pen) · REWARD_DT
    reward = clip(total, MIN_REWARD, MAX_REWARD)

Each raw signal is computed by its own **weight-free** warp kernel — the mixer
is the sole place weights, the multiplicative coefficient assembly, dt-scaling
and clipping touch the signal. Hot-swapping any ``W_*`` / ``K_*`` knob via
``apply_handler_knobs`` costs only the mixer relaunch; the upstream
RBF/contact compute is reused verbatim. (Per-term contributions and the
intermediate ``*_ineq_coeff`` / ``r_*_process`` expressions are all exposed as
``r_*`` buffers for inspection / plotting.)

Regularization terms (`CTRL_COST`, `K_ACTION_SMOOTH`) reuse the shared
warp kernels defined in :mod:`grit.training.rl_env_base`
(``_action_sqnorm_kernel`` / ``_action_smooth_sqnorm_kernel`` /
``_zero_masked_action_kernel``) — same wiring as the tracking task so a
new RL algorithm or task module gets jitter suppression "for free".

Bonus gate fires when ``obj_lift_z >= LIFT_TARGET_M`` AND wrist within
``APPROACH_BONUS`` of the target point AND at least one hand↔obj contact slot
is active; the bonus streak feeds the streak-scaled reward (capped at
``SUCCESS_STREAK_MIN``). **Success** is decided at the LAST episode step: the
object lift ``obj_z - obj_z_init`` lies within ``±SUCCESS_LIFT_TOL_FRAC ·
LIFT_TARGET_M`` of the target lift → the success kernel overrides the
coincident timeout with ``reason = SUCCESS_REASON_CODE``.

Done sources (priority obj_z > obj_xy > hand_object_touch > obj_fell > timeout
> wrist_up; see ``_object_grasping_done_kernel``)::

    3 obj_z     : obj falls below ``DONE_OBJ_Z_MIN``                       (drop)
    5 obj_xy    : obj drifts > ``DONE_OBJ_XY_DRIFT`` from init xy           (xy fail)
    6 hand_object_touch : max hand↔obj contact force > ``DONE_OBJ_TOUCH``   (crush guard)
    7 obj_fell  : obj world-z velocity < ``DONE_OBJ_VEL_Z`` (negative)      (dropped/plummeting)
    2 timeout   : ``ep_step >= MAX_EPISODE_STEPS``                          (last step)
    4 wrist_up  : wrist x-axis world-z > ``DONE_WRIST_X_Z``                 (viz guard)
    1 success   : at the last step, obj lift within ±SUCCESS_LIFT_TOL_FRAC  (success kernel,
                  · LIFT_TARGET_M of target  → overrides timeout            overrides reason 2)

Rule-based wrist-z lift curriculum (action pre-processor)
---------------------------------------------------------
Optional training aid: nudge the wrist UP during the lift stage so the policy
sees lift trajectories early, then fade out so the final policy lifts on its
own. Wired through the GENERAL applier hook
(``WarpActionApplier.bind_action_preprocessor``); bound in
``_setup_obs_buffers`` to :meth:`_apply_action_curriculum`
(kernel ``_stage_wrist_lift_kernel``):

  * fires ONLY in the lift stage (``LIFT_STEP <= ep_step < HOLD_STEP``) AND only
    while the object has not reached the target lift
    (``obj_z - obj_z_init < LIFT_TARGET_M``). Once the target is reached the
    wrist Δxyz is FROZEN at 0 for the rest of the episode (latch; the hold
    stage continues the freeze) — the policy keeps finger control only.
    Approach stage is a no-op (full policy control).
  * adds a WORLD +z displacement ``LIFT_ACTION_DIST_M · lift_curriculum_scale``
    per step, soft-ramped over ``LIFT_RAMP_STEPS`` (legacy
    ``exp(-LIFT_RAMP_RATE · max(0, RAMP_STEPS - (ep_step-LIFT_STEP)))``),
    applied as ``R_wrist^T · [0,0,lift]`` to the raw action's Δxyz so the apply
    kernel reproduces a true world-frame lift.
  * ``lift_curriculum_scale`` (1 → 0) is decayed by :meth:`on_train_progress`
    (called once per training iteration by ``scripts/train.py`` with the global
    control-step count — train.py stays task-agnostic): full until
    ``LIFT_DECAY_START_STEP``, linear 1 → 0 by ``LIFT_DECAY_END_STEP``
    (<= 0 → ``total_new_steps``). ``evaluate.py`` keeps it OFF by default
    (``applier._action_preprocessor_enabled``; toggle with ``[L]``) so the raw
    trained policy is shown.

Stage / observation / scan internals
------------------------------------
  * Per-world curriculum **stage** (0/1/2) is exposed as ``handler.stage_torch``
    (written by ``_obs_ep_stage_kernel`` alongside the obs slot) and bridged to
    the applier (``applier._stage_wp``, default all-zeros) for stage-conditioned
    pre-processing — tasks without a stage stay safe.
  * The observation is assembled block-by-block: ONE small warp kernel per
    component (``_obs_*_kernel``), each writing its slice at a host-accumulated
    ``offset`` in :meth:`_collect_obs_kernel` (the single source of truth for the
    slot layout, asserted to total ``obs_dim``). Blocks are independently
    reusable / testable; ``obs_dim`` flows to the policy / rollout buffer.
  * The PCD nearest-vertex scan ``_object_grasping_nearest_pcd_kernel`` is a 2D
    launch over ``(NWORLD, n_sensors)`` (each (world, sensor) its own thread,
    only ``n_pts`` serial) — far better GPU occupancy than ``dim=NWORLD`` with an
    inner sensor loop. It writes ``best_d2`` / ``best_pt`` reused by the finger /
    face-dir reward AND the ``site_pcd_err`` obs block.

Eval ghost: :meth:`eval_ghost_targets` returns the taxonomy ``target_qpos`` FK'd
at a fixed display pose beside each world's object (consumed by ``evaluate.py``;
tracking returns its wrist target pose instead).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import mujoco as _mj   # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
import warp as wp  # type: ignore

from grit.training.orchestrator.base import _grit_r2quat_wxyz

from grit.training.rl_env_base import (
    SubEnvHandler,
    # Shared action regularization kernels — reused verbatim so any new
    # task gets the same ctrl_cost / smoothness wiring as tracking.
    _action_sqnorm_kernel,
    _action_smooth_sqnorm_kernel,
    _zero_masked_action_kernel,
    _episode_success_kernel,            # shared streak + success promotion (task-agnostic)
    _dynamic_target_switch_mask_kernel, # shared per-world dynamic-target trigger
)

import os
import warnings
from typing import Any
from grit.training.orchestrator.base import home_dir as _HOME_DIR
from grit.util.utils import print_red

# Pure-numpy per-world target rotation sampler — shared with the
# parallel-warp viewer notebooks (see notebook/hand/03_warp_parallel/
# 08_warp_contact_sensor.ipynb).
#
# NOTE: the CPU ``sample_target_R_init_per_world`` is no longer imported — the
# per-world target_R sampling now runs on the GPU
# (``_object_grasping_sample_target_R_kernel``) inside ``_post_cond_update_hook``.


from grit.training.rl_envs.grasping_kernels import *  # noqa: F401,F403 — warp kernels of the task


class GraspingCore(SubEnvHandler):
    """Per-sub-env handler for the object-grasping task.

    Cond fields:

      * ``target_qpos``  (n_ctrl,)  — taxonomy-injected uniform sampler
                                       (same overlay as hand_pose_tracking).
      * ``target_pnt``   (3,)        — per-world hand-center target point.
                                       ``TARGET_PNT_DYNAMIC`` (default on): refreshed
                                       every control step to "obj-PCD vertex nearest
                                       the hand-center + jitter" (JAX v2 scheme;
                                       ``_object_grasping_dynamic_target_pnt_kernel``).
                                       When off, frozen at the reset snapshot (legacy).
      * ``target_R``     (3, 3)      — per-world frozen target rotation
                                       (sampled by ``sample_target_R_init_per_world``
                                       mirroring ``wrist_pose_init``).
                                       Legacy ``target_dir`` = ``target_R[:, 0]``.
      * ``obj_p_init``   (3,)        — per-world obj xpos at reset, used as
                                       anchor for lift / xy_drift.

    Plus per-world taxonomy-overlay cond fields written by the GPU gather
    (``_object_grasping_taxonomy_gather_kernel``) each reset: ``obj_R_init``
    (3, 3, obj spawn orientation), ``specific_finger_link_mask`` (n_sensors),
    ``specific_ctrl_mask`` (n_ctrl), ``face_dir_idx_in_mat`` /
    ``face_dir_sign`` (n_sensors). Of all cond fields only ``target_qpos`` has a
    standalone GPU uniform sampler; the rest are filled by
    :meth:`_post_cond_update_hook` from physics / orchestrator state once per
    reset.

    Per control step the handler runs (see the collect methods for the full,
    authoritative launch order):

      * :meth:`_collect_obs_kernel` — self/table/obj per-slot contact OR-reduce
        (weight-free flags, shared with the bonus gate), then one small
        ``_obs_*`` kernel per observation block.
      * :meth:`_collect_reward_done_kernel` — PCD transform + nearest scans →
        the weight-free per-term reward kernels (approach / finger / face-dir /
        contact / mimic / wrist-dir / wrist+obj velocity / obj-pose coeffs /
        lift / action reg) → done detection (ep_step ++) → bonus gate → reward
        mixer (weights + multiplicative coeff assembly + curriculum + dt + clip).
      * :meth:`_collect_success_kernel` — ``_episode_success_kernel`` (streak +
        success promotion, shared; stamps reason 1 over a coincident timeout).
    """

    # ── Run-config introspection: every knob the YAML / nb may tune ─────
    CONFIG_KEYS_GROUPS: Dict[str, Tuple[str, ...]] = {
        "reward_tracking": (
            "W_APPROACH", "K_APPROACH",
            "W_FINGER",   "K_FINGER",
            "W_MIMIC",    "K_MIMIC", "W_MIMIC_NAIVE", "MIMIC_ON_CTRL",
            "MIMIC_ACTIVE_DEADZONE", "MIMIC_REST_DEADZONE",
            "W_WRIST_DIR", "K_WRIST_DIR",
            "W_LIFT",
            "W_VEL", "K_VEL", "VEL_ANG_WEIGHT",
            "W_OBJ_VEL", "K_OBJ_VEL", "OBJ_VEL_ANG_WEIGHT",
            "W_OBJ_XY_COEFF", "W_OBJ_R_COEFF", "OBJ_DIST_PENALTY_COEFF",
            "W_FACE_DIR",
        ),
        "reward_penalty": (
            "W_SELF_COLL", "W_SELF_IMPULSE",
            "W_TABLE_CONTACT", "W_TABLE_IMPULSE",
            "W_TABLE_IMPULSE_DERIV", "TABLE_IMPULSE_DERIV_CLIP",
            "TABLE_PEN_OBJ_CONTACT_SCALE", "TABLE_PEN_OBJ_CONTACT_MIN_RATIO",
            "W_OBJ_TOUCH", "TOUCH_PEN_LIFT_SCALE",
            "OBJ_TOUCH_FREE_WEIGHT_K", "OBJ_TOUCH_FREE_ALPHA",
            "W_OBJ_TABLE_FORCE", "OBJ_TABLE_FREE_K", "OBJ_TABLE_FREE_ALPHA",
            "W_VIOLATION_PEN",
            "CTRL_COST", "K_ACTION_SMOOTH",
            "W_JOINT_LIMIT", "JOINT_LIMIT_MARGIN",
            "W_WRIST_HEIGHT", "WRIST_HEIGHT_MARGIN_M",
            "W_FINGER_HEIGHT", "FINGER_HEIGHT_MARGIN_M", "FINGER_HEIGHT_OBJ_CLEAR_M",
        ),
        "reward_contact": (
            "W_CONTACT_DIR", "W_CONTACT_RATIO", "W_CONTACT_POS", "W_CONTACT_NEG",
            "W_AFFORDANCE_POS", "W_AFFORDANCE_NEG",
            "W_FORCE_CLOSURE", "K_FORCE_CLOSURE",
            "W_SEM_RECALL", "W_SEM_PRECISION", "W_SEM_F1", "W_SEM_IOU",
            "CONTACT_POS_SUM", "CONTACT_POS_BINARY_COS",
            "CONTACT_RATIO_SUBGROUPS", "CONTACT_RATIO_LIFT_BOOST",
            "TORQUE_BALANCE_COUPLING", "TORQUE_BALANCE_EPS",
        ),
        "reward_post": (
            "REWARD_DT", "MIN_REWARD", "MAX_REWARD",
        ),
        "contact_gating": (
            "GATE_CONTACT_BY_TOUCH", "TOUCH_EPS",
        ),
        "target_sampler": (
            "TAXONOMY_PROB", "TARGET_PNT_NOISE_M",
            "TARGET_PNT_DYNAMIC", "TARGET_PNT_DYNAMIC_NOISE_M",
            "FINGER_INIT_TAX_PULL_PROB",
            "FINGER_INIT_TAX_PULL_MIN", "FINGER_INIT_TAX_PULL_MAX",
            "EVAL_RESET_SEED",
            "using_taxonomy_names",
            "TAXONOMY_SIZE_CONDITIONED", "TAXONOMY_SMALL_OBJ_DIAM_M",
        ),
        "obs": (
            "OBS_TAX_ERR_DEADZONE",
            "OBS_OBJ_GROUND_PRELIFT_GATED",
            "OBS_FORCE_LOG1P", "OBS_FORCE_LOG1P_CAP",
            "OBS_TORQUE_PROXY_LOG1P",
        ),
        "termination": (
            "DONE_OBJ_Z_MIN", "DONE_OBJ_XY_DRIFT",
            "DONE_WRIST_X_Z", "DONE_OBJ_TOUCH", "DONE_OBJ_TOUCH_REST",
            "DONE_OBJ_VEL_Z",
            "DONE_HAND_OBJ_DIST", "DONE_HOLD_OBJ_DIST",
            "DONE_TABLE_IMPULSE", "DONE_OBJ_GROUND_IMPULSE",
            "MAX_EPISODE_STEPS",
        ),
        "curriculum": (
            "LIFT_STEP", "HOLD_STEP", "LIFT_TARGET_M",
            "LIFT_ACTION_DIST_M", "LIFT_RAMP_STEPS", "LIFT_RAMP_RATE",
            "LIFT_MIN_DIST_M",
            "LIFT_DECAY_START_STEP", "LIFT_DECAY_END_STEP",
            "ANTIGRAV_ALPHA",
            "ANTIGRAV_DECAY_START_STEP", "ANTIGRAV_DECAY_END_STEP",
            "XYZ_SCALE_TARGET_FRAC",
            "XYZ_SCALE_DECAY_START_STEP", "XYZ_SCALE_DECAY_END_STEP",
        ),
        "action_shaping": (
            "WRIST_SLOW_NEAR_M", "WRIST_SLOW_FAR_M", "WRIST_SLOW_FACTOR",
            "STAGE_ACTION_JAX_MODE", "LIFT_WRIST_RISE_CAP_M", "STAGE_FINGER_SLOW",
            "STAGE_TARGET_HOLD", "LIFT_TARGET_INTEGRATE",
            "WRIST_TARGET_LEAD_MAX_M",
        ),
        "success": (
            "W_BONUS", "APPROACH_BONUS",
            "GRASP_BONUS_VALUE", "SUCCESS_STREAK_MIN", "SUCCESS_REASON_CODE",
            "SUCCESS_LIFT_TOL_FRAC", "LIFT_SUCCESS_THRESH", "W_LIFT_SUCCESS_BONUS",
            "COVERAGE_BONUS_COUPLING", "COVERAGE_BONUS_EXP",
            "SUCCESS_DONE_PROMOTION",
        ),
        "dynamic_target": (
            "DYNAMIC_TARGET_ENABLED",
            "DYNAMIC_TARGET_MIN_FRAC", "DYNAMIC_TARGET_MAX_FRAC",
        ),
    }

    # ── Reward weights / decay rates ────────────────────────────────────
    # class attribute
    W_APPROACH: float = 2.0
    K_APPROACH: float = 1.5     # err in m (≈ 0.5 at err = 0.14)

    W_FINGER:   float = 1.0
    K_FINGER:   float = 15.0    # err is mean-of-squared distance (m²)

    W_MIMIC:    float = 1.0
    K_MIMIC:    float = 4.0     # err is L1 hinge sum (active + rest branches)
    # What the mimic term measures: true → the **command (ctrl setpoint)** (JAX v2
    # hand_action semantics). While grasping, the object blocks the fingers so the
    # measured qpos can never physically reach the template → the error saturates
    # at an irreducible residual, the gradient vanishes and the multiplicative gate
    # (raw_mimic) becomes a constant tax. On the command the error can still reach
    # 0 mid-grasp, so the mimic learning signal stays alive. false = measured qpos (legacy).
    MIMIC_ON_CTRL: bool = True
    # Naive L1 mimic pressure without a dead-zone (additive simplification of the
    # JAX v2 controllable_process −0.1·Σ|diff| term): a weak, always-on gradient
    # towards the template even inside the hinge band. Applied to **rest ctrl
    # only** — active ctrl legitimately departs far from the template to grasp,
    # and an always-on L1 there would suppress wrapping (active is handled solely
    # by the hinge term with MIMIC_ACTIVE_DEADZONE). Since it is a Σ over **rest
    # ctrl** only, mind the scale: it is smaller than the old all-joint sum
    # (6 rest joints × 0.3 rad ≈ 1.8), so W must grow accordingly for equal pressure. 0 = off.
    W_MIMIC_NAIVE: float = 0.0
    # Per-branch dead-zone (rad). Inside the dead-zone the per-ctrl
    # contribution is 0; outside it grows linearly as ``|diff| - deadzone``.
    # Matches the legacy JAX hinge thresholds: active fingers (taxonomy's
    # "specific" ctrls) get the larger 0.6 rad tolerance because they
    # actively move for the grasp; rest fingers should stay closer to the
    # template (tighter 0.2 rad tolerance).
    MIMIC_ACTIVE_DEADZONE: float = 0.6
    MIMIC_REST_DEADZONE:   float = 0.2

    W_WRIST_DIR:   float = 1.0     # cos similarity ∈ [0, 1] coefficient
    K_WRIST_DIR:   float = 0.1    # err is cosine similarity ∈ [0, 1]
    W_LIFT:        float = 20.0     # max contribution per step once obj_lift_z >= LIFT_TARGET_M

    # ── Wrist stationarity coefficient ──────────────────────────────────
    # ``r_wrist_vel = W_VEL · exp(-K_VEL · (||vlin||² + VEL_ANG_WEIGHT · ||vang||²))``
    # ∈ [0, W_VEL]; maximised when the wrist is still. Defaults match the
    # legacy JAX coefficients (ang_weight = 0.1, decay = 0.1).
    W_VEL:           float = 1.0
    K_VEL:           float = 0.1
    VEL_ANG_WEIGHT:  float = 0.1

    # ── Object stationarity coefficient ─────────────────────────────────
    # ``r_obj_vel = W_OBJ_VEL · exp(-K_OBJ_VEL · (Σ clip(v_in_wrist)²
    #               + OBJ_VEL_ANG_WEIGHT · Σ clip(w_in_wrist)²))``
    # ∈ [0, W_OBJ_VEL]; maximised when the OBJECT is still (rewards a
    # stable, non-jostled object during approach / grasp). Per-component
    # clip to [-1, 1] in the wrist frame matches the legacy
    # ``obj_vel_coeff`` (ang_weight = 0.01, decay = 0.1).
    W_OBJ_VEL:          float = 1.0
    K_OBJ_VEL:          float = 0.1
    OBJ_VEL_ANG_WEIGHT: float = 0.01

    # ── Object pose-vs-spawn coefficients (xy drift + uprightness) ──────
    # ``r_obj_xy_coeff = W_OBJ_XY_COEFF · exp(OBJ_DIST_PENALTY_COEFF · |Δxy|)``
    # ``r_obj_R_coeff  = W_OBJ_R_COEFF  · exp(OBJ_DIST_PENALTY_COEFF/2.5 · (1 - clip(R[2,2], 0.1, 1)))``
    # Both ∈ [0, W]; = W when the object sits exactly as spawned, decaying as
    # it slides (xy) or tips (R[2,2] = uprightness). ``OBJ_DIST_PENALTY_COEFF``
    # is the legacy ``obj_dist_penalty_coeff`` — it MUST be ≤ 0 (negative) for
    # the coefficient to decay; the rotation term reuses it scaled by 1/2.5.
    # This kernel is the SINGLE owner of the object xy-drift reward (the old
    # linear ``r_obj_xy_pen`` / ``W_OBJ_XY`` term has been removed; the lift
    # kernel no longer computes xy drift).
    W_OBJ_XY_COEFF:         float = 1.0
    W_OBJ_R_COEFF:          float = 1.0
    OBJ_DIST_PENALTY_COEFF: float = -5.0
    # Clip floor on obj_R[2, 2] before the uprightness penalty (legacy 0.1).
    # Not a YAML knob (kept off CONFIG_KEYS_GROUPS, like CONTACT_NEG_SCALE); override
    # via subclass / direct attribute if a hand needs a different floor.
    OBJ_R_CLIP_MIN:         float = 0.1

    # ── Face-direction alignment coefficient ────────────────────────────
    # ``r_face_dir = W_FACE_DIR · mean_active_f max(<face_dir_f, look_dir_f>, 0)``
    # ∈ [0, W_FACE_DIR]; rewards active palm/finger faces pointing toward
    # the object (legacy ``hand_to_obj_cos_sim``). The per-sensor face
    # direction is the taxonomy-selected ``face_dir_idx_in_mat`` column of
    # each sensor body's world rotation (× ``face_dir_sign``); the active
    # set + weights reuse ``specific_finger_link_mask`` / ``finger_weights``.
    W_FACE_DIR:      float = 0.3

    # Per-sensor face↔contact-normal cosine (legacy ``frames_r_sim``) is exposed
    # as the ``contact_dir_cos_torch`` (NWORLD, n_sensors) array, then folded by
    # ``_object_grasping_contact_reward_kernel`` into 5 taxonomy-aware contact
    # components (dir_reward / contact_pos / contact_neg / affordance_impulse
    # pos / neg), each wired into the mixer below with its own weight.
    # ``CONTACT_NEG_SCALE`` is the legacy ×3 boost on the shouldn't-touch
    # (rest) contact penalty, baked into ``contact_neg`` inside the kernel; not
    # a YAML knob (like OBJ_R_CLIP_MIN).
    CONTACT_NEG_SCALE: float = 5.0
    # Mixer weights for the five contact-affordance components. All default
    # 0.0 (opt-in) so wiring them into the reward leaves existing training
    # behaviour unchanged until tuned. dir / pos are bonuses (+); the two
    # ``*_NEG`` are penalties — the kernel returns positive magnitudes and the
    # mixer applies the minus sign.
    W_CONTACT_DIR:    float = 5.0   # frames_r_sim signed-alignment reward (dir_reward)
    W_CONTACT_RATIO:  float = 0.25
    # Finer contact_ratio units: count per finger in two sub-groups,
    # {MCP+PIP (non-distal links)} / {DIP (distal link)}. Fixes the finger-unit
    # tip-only exploit (one distal contact counted as full finger coverage, which
    # removed any incentive to wrap in power taxonomies) — full ratio now requires
    # both the distal AND a non-distal link of each finger to touch. Tip-only
    # taxonomies (do_finger_tip) have only tip slots active, so they are unaffected.
    CONTACT_RATIO_SUBGROUPS: bool = False
    # Scale the contact-ratio term of ineq_coeff by this factor during the lift
    # phase (ep_step>=LIFT_STEP). Approach / initial grasp stay gentle; only while
    # lifting is "more fingers in contact → more grasp reward" pressure applied. 1.0 = off.
    CONTACT_RATIO_LIFT_BOOST: float = 1.0
    W_CONTACT_POS:    float = 5.0   # active-finger in-contact·[cos>0] bonus
    W_CONTACT_NEG:    float = 10.0   # rest-finger in-contact penalty (× CONTACT_NEG_SCALE in-kernel)
    W_AFFORDANCE_POS: float = 0.1   # active-finger [cos>0]·impulse bonus
    W_AFFORDANCE_NEG: float = 0.5   # rest-finger impulse penalty
    # JAX v2 contact_pos/aff_pos mode switches (both False = weighted-mean behaviour):
    #  * CONTACT_POS_SUM        — raw SUM instead of weighted-mean(∈[0,1]). The mean
    #    dilutes the marginal gain of one extra finger by 1/Σw, which is the
    #    structural reason the costly-to-reach ring/little fingers get neglected.
    #    SUM credits the full +w per slot.
    #  * CONTACT_POS_BINARY_COS — [cos>0] binary instead of the continuous cos⁺ gate.
    #    Fingers that are hard to align get full credit once contact + direction
    #    are right (JAX frames_positive).
    # Enabling either grows the contact_pos scale from [0,1] to [0, Σw_active], so
    # W_CONTACT_POS must be re-scaled (JAX parity: W_CONTACT_POS 5.0 / W_AFFORDANCE_POS 0.5).
    CONTACT_POS_SUM:        bool = False
    CONTACT_POS_BINARY_COS: bool = False
    # Force-closure bonus (port of JAX v2 force_closure_reward): the smaller the
    # net wrench residual ‖G·f‖ of the contact forces (forces wrapping the object
    # and cancelling out), the larger the bonus. Rewards "enveloping" rather than
    # "pushing" grasps — directly encourages opposing fingers (ring/little included)
    # from a force-balance viewpoint. 0 = off. The residual is in
    # force_scale_for_reward units, so K=1.0 is equivalent to JAX's exp(-0.1·raw).
    # metrics: force_closure_res / raw_force_closure / r_force_closure.
    W_FORCE_CLOSURE: float = 0.0
    K_FORCE_CLOSURE: float = 1.0
    # ── sem_contact_* as direct rewards (eval metric = learning signal) ─────────
    # These four metrics used to be computed only at eval time. The kernel already
    # knows slot-level active/contact, so the same formulas are produced per step
    # and added to the mixer bonus group (× r_hand_process). Differences from the
    # existing contact terms:
    #   * ``W_CONTACT_POS`` — slot **weighted sum/mean** + cos alignment gate. Scale
    #     grows with the number of links in the taxonomy (SUM mode); 0 when misaligned.
    #   * ``W_CONTACT_RATIO`` — **finger-level** coverage (one contact per finger is
    #     full credit). Also, the current mixer does not use this term directly (only via cov).
    #   * ``W_SEM_*`` — **slot (link) level set metrics**, no cos gate, all ∈[0,1]
    #     and thus taxonomy-size invariant. RECALL asks "are all required links in
    #     contact" (neglected ring/little fingers are penalised right here);
    #     PRECISION asks "no contact through links that should not be used" (rest
    #     contact penalised as a **ratio**, not a count — complementary to
    #     W_CONTACT_NEG which penalises the absolute amount); F1/IOU summarise both.
    # Enabling all three strongly triple-counts the same signal; F1 (+ auxiliary
    # RECALL) is recommended. raw ∈[0,1], so W is the upper bound of the term —
    # use W_CONTACT_DIR (3.0) as the yardstick.
    W_SEM_RECALL:    float = 0.0
    W_SEM_PRECISION: float = 0.0
    W_SEM_F1:        float = 0.0
    W_SEM_IOU:       float = 0.0

    # ── Penalties / bonus ───────────────────────────────────────────────
    # Self-collision penalties — both finger_contact_weights-weighted (legacy
    # ``self_collision_contact_penalty`` / ``self_collision_impulse_penalty``).
    # Per self-coll slot the contact flag / touch force is weighted by that
    # link's ``finger_contact_weights`` (tip-boosted, palm/fore-arm ×4):
    #   r_self_pen         = -W_SELF_COLL    · Σ_k accept_k·w_k          (weighted contact sum)
    #   r_self_impulse_pen = -W_SELF_IMPULSE · (Σ_k force_k·w_k)/n_slots (weighted impulse mean)
    # NOTE: the contact driver is the WEIGHTED sum (``raw_self_contact_w``), not
    # the old binary ``self_active`` flag — its magnitude differs, so retune
    # ``W_SELF_COLL`` accordingly. The impulse weight scales an unnormalised
    # force — tune to the hand's force scale.
    W_SELF_COLL:     float = 1.0
    W_SELF_IMPULSE:  float = 1.0
    # Obstacle-collision penalties — finger_contact_weights-weighted hand↔table
    # contact PLUS fore-arm touch, summed (``_obstacle_coll_weighted_penalty_kernel``):
    #   r_table_pen         = -W_TABLE_CONTACT · Σ_k accept_k·w_k          (weighted contact sum)
    #   r_table_impulse_pen = -W_TABLE_IMPULSE · (Σ_k force_k·w_k)/n_total (weighted impulse mean)
    # over table found-slots + fore-arm touch-slots. Like W_SELF_COLL, the
    # contact driver is the WEIGHTED sum (``raw_table_contact_w``), not the old
    # binary ``tbl_active`` — retune accordingly. Impulse weight scales an
    # unnormalised force (default 0.0 opt-in).
    W_TABLE_CONTACT: float = 5.0
    W_TABLE_IMPULSE: float = 5.0
    # Penalty on the **positive increment (time derivative Δ⁺)** of the table
    # impulse (post-mixer, 0 = off). W_TABLE_IMPULSE penalises impulse magnitude and
    # cannot separate a normal "light, prolonged brush" from a "slam". This term
    # penalises only the step-to-step increase of the finger-weighted table impulse,
    # suppressing abrupt slams (decreases and gentle increases are free).
    # A ramp from a small base to a larger curriculum final is recommended:
    #   handler.curriculum.reward.W_TABLE_IMPULSE_DERIV=[<final>, <start>, <end>]
    # TABLE_IMPULSE_DERIV_CLIP: per-step cap on Δ⁺ (0 = unlimited), so that rare
    # extreme spikes early in training do not dominate the gradient.
    W_TABLE_IMPULSE_DERIV:    float = 0.0
    TABLE_IMPULSE_DERIV_CLIP: float = 0.0
    # State-conditional table penalty gate (_table_pen_grasp_gate_kernel):
    # on steps with contact_ratio >= MIN_RATIO, raw_table_* is discounted by SCALE —
    # relaxes the "necessary contact" of flat objects in penalty/done/constraint. 1.0 = off (default).
    TABLE_PEN_OBJ_CONTACT_SCALE:     float = 1.0
    TABLE_PEN_OBJ_CONTACT_MIN_RATIO: float = 0.5
    # Intra-finger motor torque balance (torque_regulate) — per finger, encourage
    # torque to spread across joint motors instead of piling onto one. The mixer
    # multiplies ``ineq_coeff`` by a participation-balance coefficient (∈(0,1],
    # 1 = uniform). COUPLING sets the strength:
    #   torque_regulate_term = (1−c) + c·balance_coeff   (c=0 → 1.0, disabled/default)
    # With c>0 a grasp whose torque concentrates on one motor loses that much
    # process/bonus reward, favouring grasps spread across several motors. Fingers
    # with torque below EPS (resting fingers) are excluded. Default 0 (opt-in).
    TORQUE_BALANCE_COUPLING:  float = 0.0
    TORQUE_BALANCE_EPS:       float = 0.02   # fingers whose Σ|actuator_force| is below this are ignored
    # Force-on-object penalty: ``r_obj_touch_pen = -W_OBJ_TOUCH · obj_touch_force``
    # where ``obj_touch_force`` = Σ of the object's touch-sensor readings (total
    # normal force on the object). Minimising it teaches gentle grasps. Default
    # 0.0 (opt-in): the touch value is an unnormalised force, so tune the weight
    # to the object's force scale.
    W_OBJ_TOUCH:     float = 0.5
    # Touch-penalty multiplier during the lift phase (ep_step>=LIFT_STEP). Approach /
    # initial grasp keep the full gentle penalty (W_OBJ_TOUCH); only while lifting
    # is the penalty lowered to allow the holding force needed against gravity
    # (prevents strong gentle pressure from blocking lift success). 1.0 = off.
    TOUCH_PEN_LIFT_SCALE: float = 1.0
    # Weight-based free allowance: the obj touch penalty only charges the excess
    # ``max(0, ΣF − (K·mg_eff + ALPHA))``. mg_eff = per-world (1−antigrav α)·m·g
    # (force-scaled; metric obj_weight_force). Removes the unfairness of penalising
    # the "legitimate grasp force" of heavy objects in a mixed-mass pool. K=0 → legacy (raw ΣF).
    # K scale: with opposing squeeze + friction (μ~1.2) the required normal-force sum ≈ a few × mg.
    OBJ_TOUCH_FREE_WEIGHT_K: float = 0.0
    OBJ_TOUCH_FREE_ALPHA:    float = 0.0
    # ── Object↔table collision penalty (an axis **separate** from hand↔table W_TABLE_*) ──
    # Signal = ``obj_ground_force`` from the aux kernel (object touch-sensor total −
    # Σ hand slot forces) = force the table exerts on the object. W_TABLE_CONTACT /
    # W_TABLE_IMPULSE only penalise the **hand** touching the table, so pressing or
    # slamming the object into the table was never reflected in the reward
    # (consumed only by obs + done 12). The penalty is additive and ungated in the
    # mixer, so the contact-bonus gate cannot suppress it.
    #
    # FREE_K: the static load of an object resting on the table (≈mg) is free —
    # before the grasp the object sitting on the table is normal, and penalising it
    # would be an always-on penalty that makes the policy push the object away.
    # Only the excess (pressing / slamming) is penalised. K=0 → penalise the full raw value.
    W_OBJ_TABLE_FORCE:  float = 0.0     # 0 = off (opt-in)
    OBJ_TABLE_FREE_K:     float = 1.0   # free allowance = K·mg_eff + ALPHA
    OBJ_TABLE_FREE_ALPHA: float = 0.0
    # Violation terminal penalty: ``r_violation_pen = W_VIOLATION_PEN`` is
    # subtracted on the step a ``done`` fires for a VIOLATION — i.e. the done
    # reason is NOT success (SUCCESS_REASON_CODE) and NOT timeout
    # (TIMEOUT_REASON_CODE): obj_z / wrist_up / obj_xy / crush / obj_fell
    # (reasons 3/4/5/6/7). dt-scaled with the rest of the reward (so the
    # effective magnitude is ``W_VIOLATION_PEN · REWARD_DT`` on that single
    # terminal step); scale it up here / via YAML for a stronger signal. Set 0
    # to disable.
    W_VIOLATION_PEN: float = 5.0
    CTRL_COST:       float = 0.1
    # Action-smoothness penalty: -K_ACTION_SMOOTH · ||action_t - action_{t-1}||²
    # Mirrors the tracking task; 0 disables. The shared
    # ``_action_smooth_sqnorm_kernel`` is launched unconditionally so the
    # mixer always has a fresh signal — flipping the weight on costs zero.
    K_ACTION_SMOOTH: float = 1.0
    # ── Joint-limit penalty (additive, post-mixer) ───────────────────────
    # Penalize the policy for COMMANDING a finger target near / past a joint
    # limit. ``WarpActionApplier`` clamps ``d.ctrl`` into ``[ctrl_min,
    # ctrl_max]``, so a policy that keeps pushing into a rail keeps saturating
    # — a source of finger chattering in the rolled-out deterministic mean.
    # Reads the un-clamped commanded target (``applier._raw_ctrl_target``) and
    # applies a quadratic hinge over a ``JOINT_LIMIT_MARGIN``-wide band at each
    # bound (shared ``_joint_limit_penalty_kernel``). 0 = off (byte-identical to
    # before). metric ``r_joint_limit`` (≤0, should converge to 0 as it learns).
    W_JOINT_LIMIT:      float = 0.0
    # Floor-scrape prevention shaping: quadratic hinge penalty when the wrist
    # (forearm base) z drops below floor_z + margin (post-mixer, wired like the
    # joint-limit term). Contact penalties only yield a gradient "after touching";
    # this term shapes the approach path before contact. 0 = off.
    W_WRIST_HEIGHT:        float = 0.0
    # Dense penalty for finger sites approaching the floor (0 = off). Contact
    # penalties only signal after a collision and cannot shape the approach path —
    # when table contact is frequent on flat objects this term provides a
    # pre-collision gradient.
    #   FINGER_HEIGHT_MARGIN_M — hinge band width (penalised by how far below this height)
    #   FINGER_HEIGHT_OBJ_CLEAR_M — raises the reference plane by the object height
    #     (capped at this value) so descending to wrap a low object is not penalised.
    W_FINGER_HEIGHT:            float = 0.0
    FINGER_HEIGHT_MARGIN_M:     float = 0.03
    FINGER_HEIGHT_OBJ_CLEAR_M:  float = 0.03
    WRIST_HEIGHT_MARGIN_M: float = 0.05
    JOINT_LIMIT_MARGIN: float = 0.1   # rad (≈5.7°) — danger-band width per bound

    # ── Contact gating (false-positive suppression via touch sensor) ────
    GATE_CONTACT_BY_TOUCH: bool  = True
    TOUCH_EPS:             float = 1e-4

    # Fore-arm touch sensor — touch-only family (no found/pos/normal contact
    # sensor). Touch sensor name suffix used to locate it in
    # ``sampled_env.touch_sensor_index`` (general, no hard-coded name);
    # override via subclass / attribute if a hand uses a different suffix.
    FOREARM_TOUCH_SUFFIX: str = "_arm_part_touch"

    # Dynamic-object touch sensor — located by substring matching the compiled
    # model's sensor names. The substring is built at setup from the yaml
    # ``Training.obj_name`` (the object's renamed base, e.g.
    # ``top_watertight_tiny``) + this suffix, so it tracks the configured
    # object instead of a hard-coded name. The matched touch sensor's summed
    # value = force applied to the object (penalised via W_OBJ_TOUCH).
    OBJ_TOUCH_SENSOR_SUFFIX: str = "_touch"

    # ── Target sampler — taxonomy overlay on target_qpos ────────────────
    TAXONOMY_PROB: float = 1.0      # 0 disables taxonomy overlay entirely
    # (a) Object-size-conditioned taxonomy sampling: worlds whose object diameter
    # is < SMALL_OBJ_DIAM only draw small-compatible taxonomies (do_finger_tip or ≤3 active fingers).
    TAXONOMY_SIZE_CONDITIONED:  bool  = False
    TAXONOMY_SMALL_OBJ_DIAM_M:  float = 0.06   # measured on the "grip width" (middle bbox edge)
    # Taxonomy NAME filter — the taxonomy counterpart of ``obj_idxs``. ``None`` →
    # use every taxonomy (original behaviour). ``list[str]`` → only the rows of
    # ``hand_util.taxonomy_name_list`` with those names are used for per-world
    # sampling (every taxonomy_*_array is built in that list order, so the
    # name → row index mapping is unique). An unknown name raises ValueError
    # (fail-fast, so a typo cannot silently derail an experiment).
    # yaml: ``handler.target_sampler.using_taxonomy_names: null | [name, ...]``.
    # The filter applies only in the SAMPLER (GPU gather tables stay full) — row
    # index semantics are unchanged, so ghost / eval / existing code work as is.
    using_taxonomy_names: Optional[List[str]] = None
    # Per-world target_pnt noise (m) — small box around the sampled PCD
    # vertex / obj centroid at reset. Mirrors the legacy
    # ``target_pnt = pcd[argmin] + noise(3)`` jitter.
    TARGET_PNT_NOISE_M: float = 0.02
    # JAX v2 style dynamic target_pnt: every control step the ``target_pnt`` cond
    # is overwritten with "obj-PCD vertex nearest the hand-center + U(±NOISE)"
    # (reuses the ``best_pt_hc`` the reward already computed — no extra scan).
    # false → keep the reset-frozen snapshot. The static sample at reset
    # (_sample_target_pnt) still runs but is overwritten on the first step.
    TARGET_PNT_DYNAMIC:         bool  = True
    TARGET_PNT_DYNAMIC_NOISE_M: float = 0.001   # JAX v2 _get_noise scale (m)
    # Finger init taxonomy-pull (port of the last block of JAX v2 ``_finger_init_fn``):
    # at episode reset, with probability PROB, interpolate finger qpos/ctrl from
    # the uniform initial value towards the episode's taxonomy template by a
    # U[MIN,MAX] fraction — curriculum-style exposure that lets the policy
    # experience directly that grasping near the template pose pays well.
    # 0 = off (legacy uniform-only).
    FINGER_INIT_TAX_PULL_PROB: float = 0.0
    FINGER_INIT_TAX_PULL_MIN:  float = 0.05
    FINGER_INIT_TAX_PULL_MAX:  float = 0.5
    # Eval reset reproducibility protocol (JAX use_eval_protocol equivalent): if
    # >= 0, run_eval pins every reset RNG (pose init, finger init, cond GPU/host,
    # target_R/pnt, tax pull) to this seed right before the reset — every eval sees
    # the same spawn/taxonomy/target draws, removing reset-draw variance from the
    # eval metrics. The training RNG streams are restored after the rollout.
    # -1 = off (a fresh random draw at every eval).
    EVAL_RESET_SEED: int = -1

    # ── Termination ──────────────────────────────────────────────────────
    # NOTE: ``DONE_OBJ_Z_MIN`` is OVERWRITTEN at setup with the scene's table
    # surface height (``sampled_env.table_height``) in ``_setup_condition`` —
    # this -0.10 is only the pre-table fallback default.
    DONE_OBJ_Z_MIN:    float = -0.10
    DONE_OBJ_XY_DRIFT: float = 0.1
    DONE_WRIST_X_Z:    float = 0.9
    # Crush guard: episode aborts (reason=6) when the max hand↔obj contact force
    # (``obj_contact_impulse``) exceeds this. Default huge → effectively off; set
    # to a real force threshold (watch the ``obj_contact_impulse`` metric for the
    # scale) to enable. Mirrors the ``DONE_WRIST_X_Z`` "set high to never fire" guard.
    DONE_OBJ_TOUCH:    float = 15.0 # force on the object, in force-scaled units.
    # Rest-crush guard (reason=11; port of the JAX v2 masked_impulse done): abort
    # immediately when the per-slot max contact force that a rest hand body (one
    # the taxonomy says must not be used) applies to the object exceeds this
    # value — separated from the legitimate holding force of the active fingers
    # (the DONE_OBJ_TOUCH total guard), it targets only "grasping with the wrong
    # fingers". force_scale_for_state units (watch the rest_obj_impulse metric
    # for the scale). 1e9 = off.
    DONE_OBJ_TOUCH_REST: float = 1.0e+9
    # obj-ground crush guard (reason=12; approximation of the JAX v2
    # obj_table_impulse done): abort when the residual "object touch total −
    # Σ hand↔obj forces" (≈ force with which the floor supports/hits the object)
    # exceeds this — blocks grasps that push the object down into the floor.
    # Watch the obj_ground_force metric for the scale. 1e9 = off.
    DONE_OBJ_GROUND_IMPULSE: float = 1.0e+9
    # Fell-down guard: episode aborts (reason=7) when the object's GLOBAL
    # downward linear velocity exceeds this — i.e. ``obj_vel_z < DONE_OBJ_VEL_Z``
    # (a NEGATIVE value; the object is dropping fast). Read from ``d.cvel``
    # (world-frame linear, same source as the obj-vel reward). Set very
    # negative (e.g. -1e9) to disable. Following the JAX v2 convention it is only
    # evaluated from the lift phase (ep_step >= LIFT_STEP) on (lift_step gate in the done kernel).
    DONE_OBJ_VEL_Z:    float = -1.25
    # hand-center ↔ object distance (m) above which the episode aborts → reason 9
    # (hand_obj_far). Catches "object left on the floor while the wrist lifted
    # away" (failed lift). Set very large (e.g. 1e9) to disable.
    DONE_HAND_OBJ_DIST: float = 0.4
    # Hold-far early termination (reason=13): during the hold phase (ep_step >=
    # HOLD_STEP), abort when the hand-center ↔ object **nearest surface point**
    # distance (best_dist_hc) exceeds this — reclaims the remaining steps of
    # episodes that failed the grasp and would otherwise drag on to timeout.
    # Worlds holding the object have a surface distance of ~0.0x m; failed worlds
    # open up by lift height + approach error. Calibrate on the approach_err
    # metric distribution. 1e9 = off.
    DONE_HOLD_OBJ_DIST: float = 1.0e+9
    # Hard floor/forearm slam termination: reason=10 (table_crush) when the
    # per-step weighted table impulse (raw_table_impulse_w, force-scaled) exceeds
    # this. Default 1e9 = off. Calibration reference: late-training per-step mean
    # ~0.007 with spike tails of 0.2~1+, so 2.0 catches only "real slams".
    DONE_TABLE_IMPULSE: float = 1.0e9
    MAX_EPISODE_STEPS: int   = 150

    # Whether the 2-col episode-stage block (ep_step_norm + stage) is part of the
    # ACTOR observation. Default True = original behaviour. Subclasses set this
    # False to make the actor time-blind (e.g. when the time signal is moved to a
    # privileged critic obs, or for a time-invariant task). ``stage_wp`` is
    # ALWAYS computed regardless (the rule-based wrist-lift reads it).
    INCLUDE_EP_STAGE_IN_OBS: bool = True

    # Finite-difference velocity obs (12ch, JAX v2 channel layout) — appends
    # [wrist lin|wrist ang|obj lin rel|obj ang rel] (prev-wrist local frame) to
    # the actor obs. ``d.cvel``/``d.qvel`` are not used: computed by FD against the
    # previous control step's pose snapshot (``_fd_vel_update_kernel``). A subclass
    # setting this False removes the 12 obs slots (class-level only — it changes
    # obs_dim, so it is not a yaml knob).
    INCLUDE_FD_VEL_IN_OBS: bool = True
    # Per-component |v| cap of the FD velocity (m/s, rad/s) — safety clip so that
    # spikes from contact blow-ups / recovery steps do not pollute the obs (reset
    # teleports are masked via fd_valid).
    FD_VEL_CLIP: float = 10.0
    # Dead-zone (rad) of the taxonomy-error obs — |target_qpos−qpos| < this → 0
    # (JAX v2 ``hand_pose_error`` convention). 0 = raw error (original behaviour).
    OBS_TAX_ERR_DEADZONE: float = 0.1
    # Pre-lift masking of the obj_ground_force obs (1ch, approximate force the
    # object receives from the floor) — when true it is 0 while ep_step < LIFT_STEP
    # (JAX v2 obj_table_impulse obs gating convention: hides the ever-present
    # support force during approach; after lift it signals "hit the floor").
    # false = always raw.
    OBS_OBJ_GROUND_PRELIFT_GATED: bool = True
    # log1p [0,1] compression of the contact-force obs channels: applies
    # ``log1p(clip(f,0,CAP))/log1p(CAP)`` to per_slot impulse(2n) + obj_touch_force
    # + obj_ground_force. Structurally prevents heavy-tail spikes (force-scaled
    # 26+ = thousands of N) from hitting the policy/critic input while preserving
    # resolution near 0 (light contact). false → raw (old behaviour). bool channels unchanged.
    OBS_FORCE_LOG1P:     bool  = True
    OBS_FORCE_LOG1P_CAP: float = 50.0   # cap in force-scaled units (covers DONE_OBJ_TOUCH=40)
    # Signed log1p compression of torque_proxy(ctrl−qpos): sign(e)·log1p(|e|) —
    # compresses the tail under contact saturation while preserving sign and the
    # slope near 0. obs_norm passthrough channel.
    OBS_TORQUE_PROXY_LOG1P: bool = True
    # Terms subject to obs running mean/std standardisation (by obs_term_layout
    # name). Terms not listed here (BPS, contact, masks, one-hot, rot6d, gravity,
    # log1p-compressed channels, ...) pass through — sparse/discrete/bounded
    # channels suffer from running-σ collapse → ±clip saturation, so they are
    # excluded from standardisation.
    OBS_NORM_STD_TERMS: Tuple[str, ...] = (
        "site_pcd_err", "hand_center_pcd_err", "site_z_above_table",
        "obj_pose_diff", "joint_qpos", "qpos_tax_err", "fd_vel",
    )

    # ── Curriculum (legacy lift_step / hold_step) ───────────────────────
    LIFT_STEP:     int   = 75
    HOLD_STEP:     int   = 120
    LIFT_TARGET_M: float = 0.10        # 10 cm lift → r_lift saturates at 1

    # ── Rule-based wrist-z lift curriculum (action pre-processor) ───────
    # In the LIFT stage only (LIFT_STEP <= ep_step < HOLD_STEP), and only while
    # the object has not yet reached the target lift (obj_z - obj_z_init <
    # LIFT_TARGET_M), the applier's bound pre-processor nudges the wrist up by a
    # WORLD-frame +z of ``LIFT_ACTION_DIST_M · lift_curriculum_scale`` per step,
    # soft-ramped over LIFT_RAMP_STEPS (legacy exp(-0.25·max(0,5-Δ))·0.01). Once
    # the object is up — or in the hold/approach stage — it is a no-op (policy
    # keeps control). ``LIFT_ACTION_DIST_M = 0`` disables it. ``lift_curriculum_scale``
    # is a RUNTIME knob (1 → 0) the training loop decays over episodes so the
    # rule fades out (curriculum) — NOT a static CONFIG key.
    LIFT_ACTION_DIST_M: float = 0.01    # per-step world-z lift distance (m)
    LIFT_RAMP_STEPS:    float = 5.0     # soft-start ramp duration (steps)
    LIFT_RAMP_RATE:     float = 0.5    # exp ramp rate
    # Floor on the per-step manual lift (m): each ACTIVE lift step moves at least
    # this much, overriding the soft-start ramp's small early values. 0.0 = no
    # floor (pure ramp). Note: the floor is NOT curriculum-scaled, so while the
    # rule is active each step moves >= LIFT_MIN_DIST_M; the curriculum still
    # turns the rule OFF entirely once lift_curriculum_scale reaches 0 (the
    # launch early-returns when LIFT_ACTION_DIST_M·scale == 0), so a nonzero
    # floor makes the fade a hard on→off rather than a gradual magnitude decay.
    LIFT_MIN_DIST_M:    float = 0.01     # set 0.01 for ">= 1cm per lift step"
    # Curriculum decay (driven by ``on_train_progress`` from the training loop):
    # ``lift_curriculum_scale`` stays 1.0 (full rule-based lift) until
    # LIFT_DECAY_START_STEP global control steps, then decays linearly 1 → 0,
    # reaching 0 at LIFT_DECAY_END_STEP (<= 0 → use the run's total_new_steps).
    LIFT_DECAY_START_STEP: int = 50_000_000
    LIFT_DECAY_END_STEP:   int = 75_000_000      # <= 0 → total_new_steps
    lift_curriculum_scale: float = 1.0   # runtime; set by on_train_progress (1 → 0)

    # ── Object anti-gravity assist (dynamics curriculum; ANTIGRAV_ALPHA=0 → off) ──
    # Applies an upward force α·m·g to the object only, via d.xfrc_applied
    # (α=1 → weightless). Hand dynamics are untouched, so finger behaviour is
    # real-like from the start, while early sloppy grasps still lift successfully
    # and the success signal (lift bonus / success-gate) appears early. α decays as
    # ANTIGRAV_ALPHA·(1−progress) with progress = max(step-window progress,
    # success-gate progress (_gate_progress)) — converging to real gravity as
    # competence is verified. xfrc_applied is not cleared every step, so it is
    # written only when α changes (per-world mass: variant-expanded body_mass).
    # ⚠ With DECAY_END == total there is no real-gravity (α=0) phase — set
    # END ≈ 0.7·total so the last 20~30% consolidates under real gravity.
    ANTIGRAV_ALPHA:            float = 0.0   # fraction of gravity cancelled at progress=0 (0.9 recommended; 0 = off)
    ANTIGRAV_DECAY_START_STEP: int   = 0
    ANTIGRAV_DECAY_END_STEP:   int   = -1    # <= 0 → total_new_steps

    # ── xyz translation scale curriculum (approach speed decay) ──────────────
    # Early on, approach fast with the base xyz_scale (yaml rl_env.xyz_scale,
    # default 0.02) so grasping is easy to learn; as training progresses the
    # per-step xyz displacement cap decays linearly to base·XYZ_SCALE_TARGET_FRAC,
    # making the approach slow and smooth. applier.xyz_scale is read via float()
    # on every apply(), so it can be adjusted at runtime without touching the
    # kernel. TARGET_FRAC=1.0 → off (no decay).
    # progress is the [START, END] step-window progress (0→1); END<=0 → total_new_steps.
    # e.g. TARGET_FRAC=0.5 → halve the movement.
    XYZ_SCALE_TARGET_FRAC:     float = 1.0   # final xyz_scale = base·frac (1.0 = off)
    XYZ_SCALE_DECAY_START_STEP: int  = 0
    XYZ_SCALE_DECAY_END_STEP:   int  = -1    # <= 0 → total_new_steps
    _xyz_scale_base: float = -1.0            # runtime; base value captured lazily

    # ── Manipulation-precision wrist translation slow-down (proximity gated) ──────
    # Shrinks the wrist Δxyz right before / while grasping (hand-object distance
    # close) so approach and grasp are precise and slow. Full speed when far, and
    # **always full speed from the lift phase on**. Distance ramp: d>=FAR→1.0,
    # d<=NEAR→FACTOR. FACTOR>=1.0 → off. Launched only in stage 0 from the action
    # pre-processor (_apply_action_curriculum), applied on every apply without a kernel change.
    WRIST_SLOW_NEAR_M:  float = 0.12   # d<=this → full slow-down (FACTOR)
    WRIST_SLOW_FAR_M:   float = 0.20   # d>=this → full speed (1.0); linear in between
    WRIST_SLOW_FACTOR:  float = 1.0    # wrist xyz multiplier when close (1.0 = off; e.g. 0.4)
    # JAX v2 stage action-masking mode (flag; false = original Grit mode):
    # in the lift/hold phases the wrist Δxyz is removed (→ manual lift, no decay
    # or latch) + rotation frozen + finger Δ ×STAGE_FINGER_SLOW. An exclusive
    # branch from the original mode (_stage_wrist_lift_kernel: lift is 'added' to
    # the policy and faded out via LIFT_DECAY_*, with an over-band latch) — for
    # A/B reproduction of taxonomy_guided_RL.
    STAGE_ACTION_JAX_MODE: bool  = False
    # Opt-in cap on the rule lift (JAX mode): once the WRIST has risen this far since
    # LIFT_STEP the rule stops integrating +z and holds — the wrist no longer climbs
    # to the end of the lift window when the object slipped. 0 = off (training default).
    LIFT_WRIST_RISE_CAP_M: float = 0.0
    STAGE_FINGER_SLOW:     float = 0.2   # JAX ÷5
    # Wrist target-hold (both modes; guards against the measured-anchor ratchet of
    # the apply kernel):
    #  * STAGE_TARGET_HOLD: in the lift-latch/hold phases keep the mocap target at
    #    its previous value (skip the write) + hold the rotation target from lift
    #    entry — Δ=0 then means genuine pose holding rather than "re-anchoring to
    #    the sagged measured pose" (blocks the observed −24mm wrist sag and
    #    rotation drift over a 40-step hold).
    #  * LIFT_TARGET_INTEGRATE: perform the active manual lift by integrating the
    #    target (mocap += Δ) instead of anchoring to the measured pose — resolves
    #    servo lag (~2.4mm realised per 10mm commanded) by letting the weld keep
    #    pulling the unrealised remainder (aggressive lift).
    #  * WRIST_TARGET_LEAD_MAX_M: max z-lead (m) of the target over the measured pose when integrating.
    STAGE_TARGET_HOLD:        bool  = True
    LIFT_TARGET_INTEGRATE:    bool  = True
    WRIST_TARGET_LEAD_MAX_M:  float = 0.03

    # ── Success ──────────────────────────────────────────────────────────
    W_BONUS:             float = 0            
    APPROACH_BONUS:      float = 0.10
    GRASP_BONUS_VALUE:   float = 0.01
    SUCCESS_STREAK_MIN:  int   = 50    # bonus-streak cap for the reward mixer (no longer the success criterion)
    SUCCESS_REASON_CODE: int   = 1     # REASON_NAME[1]="success"
    # Switch for success done-promotion during training (relabel the final step's
    # timeout(2) as success(1)). false → no promotion, as in JAX v2: episodes
    # always end in timeout and success is judged solely by the eval
    # success_rate_lift/strict (reward, GAE and reset timing are unchanged —
    # truncation is based on ep_step; only the done_reason statistics and the
    # train n_success count change). Eval already disables promotion
    # (eval_context), so this knob does not affect it.
    SUCCESS_DONE_PROMOTION: bool = True
    TIMEOUT_REASON_CODE: int   = 2     # REASON_NAME[2]="timeout" — excluded from the violation penalty
    # Success criterion: at the LAST episode step the object lift
    # (obj_z - obj_z_init) is within ±SUCCESS_LIFT_TOL_FRAC · LIFT_TARGET_M of
    # the target lift (object held into the target-z band).
    SUCCESS_LIFT_TOL_FRAC: float = 0.10
    # JAX-convention success threshold (eval-only, reporting). One-sided lift
    # height: at the final eval step, ``obj_z - obj_z_init > LIFT_SUCCESS_THRESH``
    # counts as a "lift" success. Mirrors the JAX grasping env's
    # ``obj_lift_z > lift_success_threshold`` (=0.05). Used ONLY by
    # ``eval_success_masks`` for the comparison metric ``success_rate_lift`` —
    # does NOT affect training done-promotion (that still uses the ±tol band).
    LIFT_SUCCESS_THRESH: float = 0.05
    # Per-step reward bonus (pre-dt) added once the object is lifted past
    # LIFT_SUCCESS_THRESH (i.e. raw_lift > LIFT_SUCCESS_THRESH/LIFT_TARGET_M),
    # gated by the lift curriculum like r_lift. Rewards reaching the JAX-style
    # lift-success condition. 0 → off (default; opt in via config).
    W_LIFT_SUCCESS_BONUS: float = 0.0
    # Coverage coupling c ∈ [0,1] of the hold/lift bonuses: bonus ×[(1−c)+c·contact_ratio].
    # 0 = original (no coupling). Directly pressures coverage in the hold phase,
    # addressing the late-episode reward being coverage-agnostic so that the ratio
    # plateaus around ~0.46 (a gate ramp 0→c is recommended).
    COVERAGE_BONUS_COUPLING: float = 0.0
    # Coverage exponent p: cov = (1−c) + c·ratio^p (1.0 = linear/original). p>1 →
    # the bonus jumps once "every required finger is in contact" (devalues partial coverage).
    COVERAGE_BONUS_EXP: float = 1.0

    # ── Per-world buffer declarations (allocated dynamically) ───────────
    # Type hints for the buffers used below.
    # Mixer outputs (weighted contributions to the per-step reward).
    r_approach_wp:    wp.array;         r_approach_torch:    torch.Tensor
    r_finger_wp:      wp.array;         r_finger_torch:      torch.Tensor
    r_mimic_wp:       wp.array;         r_mimic_torch:       torch.Tensor
    r_wrist_dir_wp:   wp.array;         r_wrist_dir_torch:   torch.Tensor
    r_lift_wp:        wp.array;         r_lift_torch:        torch.Tensor
    r_self_pen_wp:    wp.array;         r_self_pen_torch:    torch.Tensor
    r_self_impulse_pen_wp: wp.array;    r_self_impulse_pen_torch: torch.Tensor
    r_table_impulse_pen_wp: wp.array;   r_table_impulse_pen_torch: torch.Tensor
    r_table_impulse_deriv_wp: wp.array; r_table_impulse_deriv_torch: torch.Tensor
    r_obj_touch_pen_wp: wp.array;       r_obj_touch_pen_torch: torch.Tensor
    r_obj_table_pen_wp: wp.array;       r_obj_table_pen_torch: torch.Tensor
    r_violation_pen_wp: wp.array;       r_violation_pen_torch: torch.Tensor
    obj_touch_force_wp: wp.array;       obj_touch_force_torch: torch.Tensor
    obj_ground_force_wp: wp.array;     obj_ground_force_wp: torch.Tensor
    r_table_pen_wp:   wp.array;         r_table_pen_torch:   torch.Tensor
    r_bonus_wp:       wp.array;         r_bonus_torch:       torch.Tensor
    r_lift_success_bonus_wp: wp.array;  r_lift_success_bonus_torch: torch.Tensor
    r_ctrl_cost_wp:   wp.array;         r_ctrl_cost_torch:   torch.Tensor
    r_action_smooth_wp:      wp.array;  r_action_smooth_torch:      torch.Tensor
    r_wrist_vel_wp:         wp.array;   r_wrist_vel_torch:         torch.Tensor
    r_obj_vel_wp:     wp.array;         r_obj_vel_torch:     torch.Tensor
    r_obj_xy_coeff_wp: wp.array;        r_obj_xy_coeff_torch: torch.Tensor
    r_obj_R_coeff_wp:  wp.array;        r_obj_R_coeff_torch:  torch.Tensor
    r_face_dir_wp:    wp.array;         r_face_dir_torch:    torch.Tensor
    r_contact_dir_wp: wp.array;         r_contact_dir_torch: torch.Tensor
    r_contact_pos_wp: wp.array;         r_contact_pos_torch: torch.Tensor
    r_contact_neg_pen_wp: wp.array;     r_contact_neg_pen_torch: torch.Tensor
    r_affordance_pos_wp: wp.array;      r_affordance_pos_torch: torch.Tensor
    r_affordance_neg_pen_wp: wp.array;  r_affordance_neg_pen_torch: torch.Tensor
    r_force_closure_wp: wp.arary;       r_force_closure_torch: torch.Tensor
    # Composite / coefficient terms (mixer-internal expressions, exposed).
    r_mimic_qpos_err_wp:  wp.array;     r_mimic_qpos_err_torch:  torch.Tensor
    hand_ineq_coeff_wp:   wp.array;     hand_ineq_coeff_torch:   torch.Tensor
    r_hand_process_wp:    wp.array;     r_hand_process_torch:    torch.Tensor
    obj_ineq_coeff_wp:    wp.array;     obj_ineq_coeff_torch:    torch.Tensor
    ineq_coeff_wp:        wp.array;     ineq_coeff_torch:        torch.Tensor
    r_obj_process_wp:     wp.array;     r_obj_process_torch:     torch.Tensor
    r_hand_obj_bonus_wp:  wp.array;     r_hand_obj_bonus_torch:  torch.Tensor
    r_hand_obj_penalty_wp: wp.array;    r_hand_obj_penalty_torch: torch.Tensor
    r_joint_limit_wp: wp.array;         r_joint_limit_torch: torch.Tensor
    r_wrist_height_wp: wp.array;        r_wrist_height_torch: torch.Tensor
    r_finger_height_wp: wp.array;       r_finger_height_torch: torch.Tensor
    # Raw drivers from the per-term compute kernels (pre-mixer).
    raw_approach_wp:  wp.array;         raw_approach_torch:  torch.Tensor
    raw_finger_wp:    wp.array;         raw_finger_torch:    torch.Tensor
    raw_mimic_wp:     wp.array;         raw_mimic_torch:     torch.Tensor
    raw_wrist_dir_wp: wp.array;         raw_wrist_dir_torch: torch.Tensor
    raw_lift_wp:      wp.array;         raw_lift_torch:      torch.Tensor
    raw_vel_wp:       wp.array;         raw_vel_torch:       torch.Tensor
    raw_obj_vel_wp:   wp.array;         raw_obj_vel_torch:   torch.Tensor
    raw_self_contact_w_wp: wp.array;    raw_self_contact_w_torch: torch.Tensor
    raw_self_impulse_w_wp: wp.array;    raw_self_impulse_w_torch: torch.Tensor
    raw_table_contact_w_wp: wp.array;   raw_table_contact_w_torch: torch.Tensor
    raw_table_impulse_w_wp: wp.array;   raw_table_impulse_w_torch: torch.Tensor
    prev_table_impulse_w_wp: wp.array;  prev_table_impulse_w_wp_torch: torch.Tensor
    raw_obj_xy_coeff_wp: wp.array;      raw_obj_xy_coeff_torch: torch.Tensor
    raw_obj_R_coeff_wp:  wp.array;      raw_obj_R_coeff_torch:  torch.Tensor
    raw_face_dir_wp:  wp.array;         raw_face_dir_torch:  torch.Tensor
    # Taxonomy-aware contact-affordance reward components (per-world scalars).
    raw_contact_dir_reward_wp: wp.array; raw_contact_dir_reward_torch: torch.Tensor
    raw_contact_pos_wp:    wp.array;     raw_contact_pos_torch:    torch.Tensor
    raw_contact_neg_wp:    wp.array;     raw_contact_neg_torch:    torch.Tensor
    raw_affordance_pos_wp: wp.array;     raw_affordance_pos_torch: torch.Tensor
    raw_affordance_neg_wp: wp.array;     raw_affordance_neg_torch: torch.Tensor
    raw_force_closure_wp:wp.array;       raw_force_closure_torch: torch.Tensor
    # sem_contact_* (slot-level taxonomy compliance metrics, per-world ∈ [0,1]).
    raw_sem_recall_wp:    wp.array;      raw_sem_recall_torch:    torch.Tensor
    raw_sem_precision_wp: wp.array;      raw_sem_precision_torch: torch.Tensor
    raw_sem_f1_wp:        wp.array;      raw_sem_f1_torch:        torch.Tensor
    raw_sem_iou_wp:       wp.array;      raw_sem_iou_torch:       torch.Tensor
    raw_contact_ratio_wp: wp.array;      raw_contact_ratio_torch: torch.Tensor 
    action_sqnorm_wp: wp.array;          action_sqnorm_torch: torch.Tensor
    # Per-sensor face↔contact-normal cosine (NWORLD, n_sensors) — array signal
    # for downstream contact reward terms (not a scalar mixer term).
    contact_dir_cos_wp: wp.array;        contact_dir_cos_torch: torch.Tensor
    action_smooth_sqnorm_wp: wp.array;   action_smooth_sqnorm_torch: torch.Tensor
    # Diagnostics (raw errors — log to W&B to see which signal is the blocker).
    approach_err_wp:  wp.array;          approach_err_torch:  torch.Tensor
    finger_err_wp:    wp.array;          finger_err_torch:    torch.Tensor
    mimic_qpos_err_wp:      wp.array;    mimic_qpos_err_torch:      torch.Tensor
    lift_z_wp:        wp.array;          lift_z_torch:        torch.Tensor
    total_vel_wp:     wp.array;          total_vel_torch:     torch.Tensor
    total_obj_vel_wp: wp.array;          total_obj_vel_torch: torch.Tensor
    # Contact / bonus / success state.
    self_active_wp:   wp.array;          self_active_torch:   torch.Tensor
    self_impulse_wp:  wp.array;          self_impulse_torch:  torch.Tensor
    tbl_active_wp:    wp.array;          tbl_active_torch:    torch.Tensor
    tbl_impulse_wp:   wp.array;          tbl_impulse_torch:   torch.Tensor
    torque_balance_coeff_wp: wp.array;   torque_balance_coeff_torch: torch.Tensor 
    # Per-slot (per hand contact sensor) touch force — (NWORLD, n_slots).
    self_impulse_per_slot_wp: wp.array;  self_impulse_per_slot_torch: torch.Tensor
    tbl_impulse_per_slot_wp:  wp.array;  tbl_impulse_per_slot_torch:  torch.Tensor
    forearm_touch_wp: wp.array;          forearm_touch_torch: torch.Tensor
    hand_obj_active_wp:    wp.array;     hand_obj_active_torch:    torch.Tensor
    rest_obj_impulse_wp: wp.array;      rest_obj_impulse_torch: torch.Tensor
    force_closure_res_wp: wp.array;      force_closure_res_torch: torch.Tensor
    # Dominant hand↔obj contact: scalar force (NWORLD,) + world pos / unit
    # normal (NWORLD, vec3).
    hand_obj_contact_impulse_wp: wp.array; hand_obj_contact_impulse_torch: torch.Tensor
    hand_obj_contact_pos_wp:   wp.array;   hand_obj_contact_pos_torch:   torch.Tensor
    hand_obj_contact_dir_wp:   wp.array;   hand_obj_contact_dir_torch:   torch.Tensor
    # Per-slot hand↔obj contact force — (NWORLD, n_obj_contact).
    hand_obj_contact_impulse_per_slot_wp: wp.array; hand_obj_contact_impulse_per_slot_torch: torch.Tensor
    bonus_active_wp:  wp.array;                     bonus_active_torch:  torch.Tensor
    bonus_streak_wp:  wp.array;                     bonus_streak_torch:  torch.Tensor
    # Dynamic mid-episode target switch buffers (NWORLD,) int.
    target_switch_step_wp: wp.array;                target_switch_step_torch: torch.Tensor # type: ignore
    switch_mask_wp:        wp.array;                switch_mask_torch:        torch.Tensor # type: ignore
    # Action-smoothness state. ``last_action`` shape is (NWORLD, action_dim)
    # — allocated in _setup_reward_done_buffers once action_dim is known.
    last_action_wp:           wp.array;             last_action_torch:           torch.Tensor

    def _core_obs_dim(self) -> int:
        # Sum grouped by term type (NOT obs-column order — that is
        # ``obs_term_layout`` / ``_collect_obs_kernel``). Per-term widths:
        #   joint_qpos(n_ctrl) + qpos_tax_err(n_ctrl)        → 2·n_ctrl
        #   ep_step_norm(1) + stage(1)                       → 2
        #   finger_link_mask(n_sensors)                      → n_sensors
        #   per_slot_contact [REUSED reward buffers, n = n_obj_contact]:
        #     obj_bool(n) + obj_impulse(n) + obstacle_bool(n) + obstacle_impulse(n) → 4·n
        #   obj_touch_force(1)  [Σ obj touch sensors, force-scaled]   → 1
        #   wrist_obj_vel(12)  wrist_vel + wrist_qvel + obj_vel_rel + obj_qvel_rel → 12
        #   site_pcd_err  per-site → nearest PCD (wrist frame, fore-arm excl.) → 3·(n_sensors-1)
        #   hand_center_pcd_err  hand-center → nearest PCD (wrist frame)        → 3
        #   rot6d  hand-center 6D MINUS target_R 6D (target_R-only half is       → 6
        #          commented out in the kernel, so 6 — NOT 12)
        #   site_z_above_table(n_sensors)                    → n_sensors
        #   obj_pose_diff  p_diff(3) + rpy_diff(3)           → 6
        #   torque_proxy(n_ctrl)                             → n_ctrl
        return (2 * self.n_ctrl
                + (4 if self.INCLUDE_EP_STAGE_IN_OBS else 0)  # ep_step_norm(1) + stage one-hot(3)
                + self._n_sensors
                + 4 * self._n_obj_contact
                + 1
                + 1   # obj_ground_force (approximate object-ground force; optional pre-lift gating)
                + (12 if self.INCLUDE_FD_VEL_IN_OBS else 0)  # fd_vel: pose-FD wrist+obj velocity (JAX v2 channels)
                + 3   # projected gravity (wrist frame; absolute orientation cue)
                + 3   # lift_target_vec (obj→spawn+LIFT_TARGET_M point, wrist frame; always on)
                + 3 * (self._n_sensors - 1)
                + 3
                + 6
                + self._n_sensors
                + 6
                + self.n_ctrl)

    @staticmethod
    def _count_contact_slots(contact_index: dict) -> int:
        """Slot count for a contact family (matches ``_build_contact_idx_wp``:
        sum of ``num`` over entries with a valid ``dim`` multiple of 7)."""
        n = 0
        for _name, (_adr, dim, num) in (contact_index or {}).items():
            if num > 0 and dim > 0 and (dim % num) == 0 and (dim // num) in (7, 10):
                n += int(num)
        return n

    def _setup_obs_buffers(self) -> None:
        self._load_bps_data()          # obs_dim needs _n_bps before the obs buffer is allocated
        # ``obs_dim`` needs ``_n_obj_contact`` (per-slot contact reuse), but the
        # base init allocates the obs buffer HERE, *before*
        # ``_setup_reward_done_buffers`` builds the contact-idx arrays. Resolve
        # the obj-contact slot count up-front from the env's contact index so
        # ``obs_dim`` is valid; ``_setup_reward_done_buffers`` re-sets the same
        # value when it builds the full idx arrays.
        self._n_obj_contact = self._count_contact_slots(
            getattr(self.sampled_env, "contact_sensor_for_object_index", {}) or {},
        )
        super()._setup_obs_buffers()
        # Per-world curriculum stage (0/1/2), written by ``_obs_ep_stage_kernel``
        # alongside the obs slot. Exposed as ``handler.stage_torch`` so an
        # external rule (e.g. the action applier's wrist-lift curriculum) can
        # read the stage without re-deriving it from ep_step.
        self.stage_wp, self.stage_torch = self._alloc_world(int)
        # Bridge the stage buffer into the action applier so stage-conditioned
        # pre-processing (rule-based wrist-z lift curriculum) can read it, and
        # register this task's action pre-processor (the lift rule).
        self.applier.bind_stage_buffer(self.stage_wp)
        self.applier.bind_action_preprocessor(self._apply_action_curriculum)
        # Wrist target-hold flag buffers: written by the stage kernel every step;
        # the apply kernel conditions the mocap write on them (0 = measured+Δ /
        # 1 = keep previous target / 2 = integrate target). Reset to 0 every step
        # at the top of _apply_action_curriculum, so no stale values remain in
        # configurations where the stage kernel does not run.
        self.wrist_pos_hold_wp = wp.zeros((self.NWORLD,), dtype=int, device=self.device)
        self.wrist_rot_hold_wp = wp.zeros((self.NWORLD,), dtype=int, device=self.device)
        self.wrist_z_lift0_wp  = wp.zeros((self.NWORLD,), dtype=float, device=self.device)   # wrist z at LIFT_STEP
        self.applier.bind_wrist_hold_buffers(self.wrist_pos_hold_wp,
                                             self.wrist_rot_hold_wp)
        self.applier.pos_lead_max_m = float(self.WRIST_TARGET_LEAD_MAX_M)

    # ── Cond setup ───────────────────────────────────────────────────────
    def _setup_condition(self) -> None:
        """Register the cond fields + upload all static buffers used by the
        per-term compute kernels.

        Cond fields:
          * ``target_qpos`` (n_ctrl,) — GPU-uniform sampler over actuator
            ctrlrange. The CPU **taxonomy overlay** that used to live here
            has moved into :meth:`_post_cond_update_hook` so the same
            per-world taxonomy draw drives BOTH ``target_qpos`` and the new
            ``specific_finger_link_mask`` field (the finger-close kernel
            needs the mask to match the chosen grasp class). Removing the
            CPU sampler also drops the once-per-reset CPU sync that the
            ``WarpPerWorldCondition.reset`` would otherwise pay for the
            sampler dispatch.
          * ``specific_finger_link_mask`` (n_sensors,) — per-world float
            mask used by ``_object_grasping_finger_close_kernel``. Default
            value = ``np.ones(n_sensors)`` with arm slot 1 zeroed (so the
            forearm site doesn't contribute even on the uniform fallback).
            Populated jointly with ``target_qpos`` in the post-cond hook.
          * ``target_pnt`` / ``target_R`` / ``obj_p_init`` — sampler-free
            snapshots populated by :meth:`_post_cond_update_hook`.
            ``target_R[:, 0]`` is the legacy ``target_dir`` heading.

        Static GPU buffers built here (uploaded once):
          * ``self._finger_site_ids_wp`` — site ids for the palm + arm +
            fingers (the **full** ``hand_sensor_site_id_list`` — the arm
            row's mask is 0 in every taxonomy + the default, so the kernel
            naturally excludes it).
          * ``self._finger_weights_wp`` — ``self.sampled_env.finger_weights``
            (built by ``SingleHandSubEnv._build_finger_weights``).
          * ``self._taxonomy_mask_np`` — host-side ``(n_tax, n_sensors)``
            mask table (numpy). Sampled per-world during the post-cond
            hook to fill the ``specific_finger_link_mask`` cond field.
          * ``self._taxonomy_qpos_np`` — host-side ``(n_tax, n_ctrl)``
            qpos table (numpy). Same role for ``target_qpos``.

        Backward-compatible attributes (kept for any code that introspects
        ``n_af`` / ``af_body_ids_wp`` — the active-finger body list is no
        longer read by the finger-close kernel, but still exposed):
          * ``self.n_af`` — number of resolved active-finger bodies.
          * ``self._af_body_ids_wp`` — wp.array(int) of those body ids.
        """
        # ── target_qpos: GPU uniform only (taxonomy overlay moved to the
        # post-cond hook so it stays in sync with the mask field).
        # Actuator scope: in a two-hand scene (LFHandView) the applier holds only
        # this hand's actuator subset — gather the ctrl range in the same scope
        # (single hand: act_ids = arange → behaviour unchanged).
        ctrl_range = np.asarray(self.mjm.actuator_ctrlrange, dtype=np.float32)
        _act_ids = getattr(self.applier, "_act_ids_np", None)
        if _act_ids is not None:
            ctrl_range = ctrl_range[np.asarray(_act_ids, dtype=int)]
        qpos_low   = ctrl_range[:, 0]
        qpos_high  = ctrl_range[:, 1]
        self.cond.register("target_qpos", shape=(self.n_ctrl,))
        self.cond.register_uniform_gpu(
            "target_qpos", low=qpos_low, high=qpos_high,
        )

        # ── Sampler-free snapshots populated by ``_post_cond_update_hook``.
        # Initialised to safe defaults: target_R = identity (so
        # ``target_R[:, 0] = (1, 0, 0)`` makes the cos alignment term
        # well-defined until the first reset fills real values);
        # target_pnt = obj_p_init (so approach_err == 0 before reset).
        self.cond.register("target_pnt", shape=(3,))
        # target_R: per-world (3, 3) world-frame rotation init. The legacy
        # ``target_dir`` (unit X-axis vector) is now ``target_R[w, :, 0]``.
        # Default = identity so the very first step before any reset has
        # ``target_R[:, :, 0] = (1, 0, 0)`` (a sane default heading) and the
        # cos alignment term stays well-defined.
        self.cond.register("target_R", shape=(3, 3), init=np.eye(3, dtype=np.float32))
        self.cond.register("obj_p_init", shape=(3,))
        # Object spawn ORIENTATION (3x3) — set at reset alongside obj_p_init in
        # _post_cond_update_hook; used for the obs obj-pose-vs-spawn rpy error.
        self.cond.register("obj_R_init", shape=(3, 3), init=np.eye(3, dtype=np.float32))

        # ── GPU target_R sampler constants (replaces the CPU
        # ``sample_target_R_init_per_world`` loop — measured as ~79% of
        # ``_post_cond_update_hook``, itself ~65% of the reset that dominates
        # training). The target rotation depends only on the wrist orientation +
        # these hand-center columns + per-world noise, so it runs as a sync-free
        # warp kernel (``_object_grasping_sample_target_R_kernel``).
        hc = np.asarray(self.hand_util.rh_hand_center, dtype=np.float32)
        self._hc_col0_np = np.ascontiguousarray(hc[:3, 0])
        self._hc_col2_np = np.ascontiguousarray(hc[:3, 2])
        # Per-call seed (incremented each reset) → fresh per-world noise. The CPU
        # original used the un-seeded module np.random, so it was non-deterministic
        # anyway; this is at least reproducible given the start seed.
        self._target_R_seed   = int(self.handler_idx) * 1_000_003 + 12_345
        self._target_pnt_seed = int(self.handler_idx) * 1_000_003 + 67_890
        # Tiny (NWORLD,) int mask staged host→GPU once per reset and shared by the
        # GPU cond-update kernels — replaces the big ``d.xpos/xmat.numpy()`` sync.
        self._cond_mask_host = wp.zeros(int(self.NWORLD), dtype=int, device="cpu")
        self._cond_mask_wp   = wp.zeros(int(self.NWORLD), dtype=int, device=self.device)

        # ── specific_finger_link_mask: per-world (n_sensors,) float mask.
        # Default = ones-with-arm-zeroed so the uniform-fallback worlds
        # naturally exclude the forearm site even before a taxonomy is
        # rolled (and so the very first step before any reset() call
        # produces a finite weighted MSE).
        env       = self.sampled_env
        n_sensors = int(len(env.hand_sensor_site_id_list))
        self._n_sensors = n_sensors

        default_mask = np.ones(n_sensors, dtype=np.float32)
        if n_sensors >= 2:
            default_mask[1] = 0.0   # forearm slot — never contributes !!!!!
        self._default_mask_np = default_mask
        self.cond.register(
            "specific_finger_link_mask",
            shape=(n_sensors,),
            init=default_mask,
        )

        # ── specific_ctrl_mask: per-world (n_ctrl,) float mask.
        # Indexed by actuator (`mj_env.ctrl_names`), parallel to
        # ``target_qpos``. Used by ``_object_grasping_qpos_mimic_kernel`` to
        # split the L1 hinge loss into:
        #   * "active" ctrls (mask = 1)  → dead-zone ``MIMIC_ACTIVE_DEADZONE``
        #   * "rest"   ctrls (mask = 0)  → dead-zone ``MIMIC_REST_DEADZONE``
        # Default = ones so the uniform-fallback worlds (where no taxonomy
        # is rolled) use the *active* branch for every ctrl — the policy
        # then sees a single dead-zone (0.6 rad by default) rather than a
        # split loss. Matches the JAX ``specific_ctrl_mask`` /
        # ``rest_ctrl_mask`` semantics from grit_with_obj_coll_v2.
        default_ctrl_mask = np.ones(self.n_ctrl, dtype=np.float32)
        self._default_ctrl_mask_np = default_ctrl_mask
        self.cond.register(
            "specific_ctrl_mask",
            shape=(self.n_ctrl,),
            init=default_ctrl_mask,
        )

        # ── face_dir_idx_in_mat / face_dir_sign cond fields ─────────────
        # Per-world (n_sensors,) float arrays carrying, per sensor (palm,
        # arm, fingers), the taxonomy-selected face direction. A consumer
        # kernel can compute:
        #
        #     face_dir_world(w, i) = R_site(w, sid_i) · (col[face_idx[w, i]] * face_sign[w, i])
        #
        # i.e. pick the ``face_idx``-th column of the sensor site's
        # world rotation matrix and apply the sign — same recipe as the
        # ``05_fk_taxonomy.ipynb`` arrow visualisation, just batched.
        #
        # The values are sampled jointly with target_qpos /
        # specific_finger_link_mask / specific_ctrl_mask in
        # :meth:`_apply_taxonomy_to_qpos_and_mask` so a world's entire
        # taxonomy state (template + masks + face dirs) comes from a
        # single ``tax_idx`` draw.
        #
        # Defaults — used for uniform-fallback worlds AND when the hand
        # has no face-direction annotation:
        #   * ``face_dir_idx_in_mat`` = 0 (column 0)
        #   * ``face_dir_sign`` = 1
        # The arm slot's default is the same; it never contributes
        # because every consumer kernel multiplies by
        # ``specific_finger_link_mask[:, arm] == 0``.
        default_face_dir_idx  = np.zeros(n_sensors, dtype=np.float32)
        default_face_dir_sign = np.ones(n_sensors,  dtype=np.float32)
        self._default_face_dir_idx_np  = default_face_dir_idx
        self._default_face_dir_sign_np = default_face_dir_sign
        # Cond fields store float — kernels that need ints cast at read
        # time (``int(face_dir_idx[w, i])`` is cheap and avoids forking
        # the cond machinery for a bespoke dtype).
        self.cond.register(
            "face_dir_idx_in_mat",
            shape=(n_sensors,),
            init=default_face_dir_idx,
        )
        self.cond.register(
            "face_dir_sign",
            shape=(n_sensors,),
            init=default_face_dir_sign,
        )

        # ── Resolve af bodies (kept for backward-compat introspection). ─
        # The finger-close kernel no longer uses these (it reads palm +
        # finger sites + taxonomy mask), but other tasks / notebooks may
        # still query ``handler.n_af``.
        af_part_names = list(getattr(self.hand_util, "rh_af_parts", []) or [])
        af_body_ids   = []
        for n in af_part_names:
            bid = int(_mj.mj_name2id(self.mjm, _mj.mjtObj.mjOBJ_BODY, n))  # type: ignore
            if bid >= 0:
                af_body_ids.append(bid)
        self.n_af = len(af_body_ids)
        if self.n_af > 0:
            self._af_body_ids_wp = wp.array(
                np.asarray(af_body_ids, dtype=np.int32),
                dtype=int, device=self.device,
            )
        else:
            self._af_body_ids_wp = wp.array(
                np.zeros(1, dtype=np.int32),
                dtype=int, device=self.device,
            )
        self._af_err_scale = (1.0 / float(self.n_af)) if self.n_af > 0 else 0.0

        # ── Upload palm + finger site ids + sensor-space finger_weights ──
        # The kernel iterates the **full** sensor list (palm = 0, arm = 1,
        # fingers = 2…). The arm row is masked out by every taxonomy +
        # default mask, so its contribution is zero without special-casing.
        site_ids_np = np.asarray(env.hand_sensor_site_id_list, dtype=np.int32)
        self._finger_site_ids_wp = wp.array(
            site_ids_np, dtype=int, device=self.device,
        )
        finger_weights_np = np.asarray(
            getattr(env, "finger_weights", np.ones(n_sensors)),
            dtype=np.float32,
        ).reshape(-1)
        if finger_weights_np.size != n_sensors:
            finger_weights_np = np.ones(n_sensors, dtype=np.float32)
        self._finger_weights_wp = wp.array(
            finger_weights_np, dtype=float, device=self.device,
        )

        # Sensor BODY ids (parallel to ``hand_sensor_site_id_list`` — palm,
        # arm, fingers). The face-direction kernel reads each sensor body's
        # world rotation ``xmat[w, bid]`` to extract the taxonomy face
        # column, and its ``xpos[w, bid]`` as the origin of the
        # body→object look direction (matches the legacy
        # ``data.xmat[hand_sensor_body_id_array]`` / ``data.xpos[...]``).
        sensor_body_ids_np = np.asarray(
            getattr(env, "hand_sensor_body_id_list", site_ids_np),
            dtype=np.int32,
        ).reshape(-1)
        if sensor_body_ids_np.size != n_sensors:
            # Defensive fallback — keep the kernel launchable even if the
            # body-id list is malformed (all-zero → obj body row, masked out
            # for the arm slot and harmless for the default mask path).
            sensor_body_ids_np = np.zeros(n_sensors, dtype=np.int32)
        self._hand_sensor_body_ids_wp = wp.array(
            sensor_body_ids_np, dtype=int, device=self.device,
        )

        # ── Host-side taxonomy tables (sampled in the post-cond hook). ──
        # All default to None when the hand has no taxonomy data — then
        # the hook just leaves target_qpos at the GPU-uniform value and
        # the masks at their respective defaults.
        self._taxonomy_qpos_np = self._build_taxonomy_qpos_table()  # (n_tax, n_ctrl) or None
        tax_mask_raw = np.asarray(
            getattr(env, "taxonomy_specific_finger_body_mask_array", None)
            if hasattr(env, "taxonomy_specific_finger_body_mask_array")
            else None,
            dtype=np.float32,
        )
        if (tax_mask_raw is not None
                and tax_mask_raw.ndim == 2
                and tax_mask_raw.shape[1] == n_sensors):
            self._taxonomy_mask_np = tax_mask_raw.astype(np.float32, copy=False)
        else:
            self._taxonomy_mask_np = None

        # Ctrl mask table — built directly from the orchestrator (shape
        # ``(n_tax, n_ctrl)`` matching ``mj_env.ctrl_names``). Used by the
        # post-cond hook to populate ``specific_ctrl_mask``.
        tax_ctrl_mask_raw = np.asarray(
            getattr(env, "taxonomy_specific_ctrl_mask_array", None)
            if hasattr(env, "taxonomy_specific_ctrl_mask_array")
            else None,
            dtype=np.float32,
        )
        if (tax_ctrl_mask_raw is not None
                and tax_ctrl_mask_raw.ndim == 2
                and tax_ctrl_mask_raw.shape[1] == self.n_ctrl):
            self._taxonomy_ctrl_mask_np = tax_ctrl_mask_raw.astype(np.float32, copy=False)
        else:
            self._taxonomy_ctrl_mask_np = None

        # Face direction tables — re-indexed from
        # ``hand_util.taxonomy_hand_face_dir_{idx_in_mat, sign}_array``
        # (palm + fingers, no arm) into sensor space (palm + arm + fingers)
        # via :meth:`_build_taxonomy_face_dir_table`. Each row of these
        # tables specifies, per (taxonomy, sensor), which column of the
        # site rotation matrix carries the outward face direction and
        # which sign to apply. Tracked here so future kernels can compute
        # ``face_dir_world = site_xmat[:, idx] · sign`` per sensor for the
        # chosen taxonomy. Shape is ``(n_tax, n_sensors)``; both ``None``
        # when the hand has no face-direction annotation.
        face_dir_tables = self._build_taxonomy_face_dir_table()
        if face_dir_tables is not None:
            self._taxonomy_face_dir_idx_np, self._taxonomy_face_dir_sign_np = face_dir_tables
        else:
            self._taxonomy_face_dir_idx_np  = None
            self._taxonomy_face_dir_sign_np = None

        # If the loaded taxonomy tables disagree on n_tax (shouldn't happen
        # in practice but be defensive), pick the minimum so the per-world
        # index stays in range for all of them.
        tables = [
            t for t in (
                self._taxonomy_qpos_np,
                self._taxonomy_mask_np,
                self._taxonomy_ctrl_mask_np,
                self._taxonomy_face_dir_idx_np,
                self._taxonomy_face_dir_sign_np,
            ) if t is not None
        ]
        if len(tables) >= 2:
            n_tax = min(t.shape[0] for t in tables)
            if self._taxonomy_qpos_np           is not None:
                self._taxonomy_qpos_np           = self._taxonomy_qpos_np[:n_tax]
            if self._taxonomy_mask_np           is not None:
                self._taxonomy_mask_np           = self._taxonomy_mask_np[:n_tax]
            if self._taxonomy_ctrl_mask_np      is not None:
                self._taxonomy_ctrl_mask_np      = self._taxonomy_ctrl_mask_np[:n_tax]
            if self._taxonomy_face_dir_idx_np   is not None:
                self._taxonomy_face_dir_idx_np   = self._taxonomy_face_dir_idx_np[:n_tax]
            if self._taxonomy_face_dir_sign_np  is not None:
                self._taxonomy_face_dir_sign_np  = self._taxonomy_face_dir_sign_np[:n_tax]

        # ── GPU taxonomy tables (uploaded ONCE) for the masked gather kernel. ─
        # Moving the per-reset overlay onto the GPU (``_object_grasping_taxonomy_
        # gather_kernel``) removes the host round-trip the old hook did EVERY
        # reset: 5 full-buffer D2H mirror + 5 H2D writeback + a Python per-world
        # loop + a forced GPU sync (``.numpy()``). The host now only draws/uploads
        # the small ``(NWORLD,)`` ``tax_row`` and launches one masked kernel.
        nct, nsen = int(self.n_ctrl), int(self._n_sensors)
        has_qf  = (self._taxonomy_qpos_np is not None) and (self._taxonomy_mask_np is not None)
        n_tax_g = int(self._taxonomy_qpos_np.shape[0]) if has_qf else 0 # type: ignore
        self._n_tax        = n_tax_g
        self._has_taxonomy = has_qf and (n_tax_g > 0)
        nrow = max(n_tax_g, 1)                       # keep buffers 2D even with no taxonomy

        # ── Taxonomy NAME filter (``using_taxonomy_names`` knob) ────────────
        # Just as obj_idxs selects the object pool, taxonomy rows are restricted by
        # name. Row order = ``hand_util.taxonomy_name_list`` (hand_utils.py builds
        # every taxonomy_*_array in this order). The filter applies only to the
        # sampler's row draw below — GPU gather tables stay full so row index
        # semantics are preserved. apply_handler_knobs(rebuild_cond=True) re-runs
        # this method after applying the yaml values, so the knob takes effect in
        # both training and inference.
        self._tax_allowed_rows: Optional[np.ndarray] = None
        _sel = self.using_taxonomy_names
        if _sel is not None:
            _sel = [str(n) for n in _sel]
            if not self._has_taxonomy:
                print(f"[object_grasping] ⚠ using_taxonomy_names={_sel} was given but "
                      f"this hand has no taxonomy table — filter ignored (uniform-only sampling).")
            else:
                names_all = [str(n) for n in
                             (getattr(self.hand_util, "taxonomy_name_list", []) or [])]
                unknown = [n for n in _sel if n not in names_all]
                if unknown:
                    raise ValueError(
                        f"using_taxonomy_names: unknown taxonomy name(s) {unknown} for "
                        f"hand {getattr(self.hand_util, 'hand_name', '?')!r}.\n"
                        f"  available ({len(names_all)}): {names_all}")
                rows = sorted({names_all.index(n) for n in _sel})
                rows = [r for r in rows if r < int(self._n_tax)]   # guard against table truncation
                if not rows:
                    raise ValueError(
                        f"using_taxonomy_names: none of {_sel} fall inside the loaded "
                        f"taxonomy tables (n_tax={self._n_tax}).")
                self._tax_allowed_rows = np.asarray(rows, dtype=np.int64)
                print(f"[object_grasping] taxonomy filter: {len(rows)}/{self._n_tax} rows "
                      f"active — {[names_all[r] for r in rows]}")

        def _f2d(a):
            return np.ascontiguousarray(a, dtype=np.float32)

        # Required tables (qpos + finger-mask); dummy (1, dim) when no taxonomy
        # (never indexed — tax_row stays < 0 in that case).
        eff_qpos  = self._taxonomy_qpos_np if self._has_taxonomy else np.zeros((1, nct),  np.float32)
        eff_fmask = self._taxonomy_mask_np if self._has_taxonomy else np.zeros((1, nsen), np.float32)
        # Optional tables fall back to the DEFAULT row broadcast across all rows.
        eff_cmask = (self._taxonomy_ctrl_mask_np
                     if (self._has_taxonomy and self._taxonomy_ctrl_mask_np is not None)
                     else np.broadcast_to(self._default_ctrl_mask_np, (nrow, nct)))
        eff_fidx  = (self._taxonomy_face_dir_idx_np
                     if (self._has_taxonomy and self._taxonomy_face_dir_idx_np is not None)
                     else np.broadcast_to(self._default_face_dir_idx_np, (nrow, nsen)))
        eff_fsgn  = (self._taxonomy_face_dir_sign_np
                     if (self._has_taxonomy and self._taxonomy_face_dir_sign_np is not None)
                     else np.broadcast_to(self._default_face_dir_sign_np, (nrow, nsen)))

        self._tax_qpos_gpu     = wp.array(_f2d(eff_qpos),  dtype=float, device=self.device)
        self._tax_fmask_gpu    = wp.array(_f2d(eff_fmask), dtype=float, device=self.device)
        self._tax_cmask_gpu    = wp.array(_f2d(eff_cmask), dtype=float, device=self.device)
        self._tax_fdir_idx_gpu = wp.array(_f2d(eff_fidx),  dtype=float, device=self.device)
        self._tax_fdir_sgn_gpu = wp.array(_f2d(eff_fsgn),  dtype=float, device=self.device)
        self._def_fmask_gpu    = wp.array(_f2d(self._default_mask_np),          dtype=float, device=self.device)
        self._def_cmask_gpu    = wp.array(_f2d(self._default_ctrl_mask_np),     dtype=float, device=self.device)
        self._def_fdir_idx_gpu = wp.array(_f2d(self._default_face_dir_idx_np),  dtype=float, device=self.device)
        self._def_fdir_sgn_gpu = wp.array(_f2d(self._default_face_dir_sign_np), dtype=float, device=self.device)
        # Per-reset (NWORLD,) tax_row staging — host writes it, copied H2D each
        # reset (small async copy; no sync). cpu staging avoids per-reset alloc.
        # ── per-taxonomy diagnostics + object-size-conditioned sampling state ─────────────
        self._setup_taxonomy_diag_and_size()
        self._tax_row_host = wp.zeros(int(self.NWORLD), dtype=int, device="cpu")
        self._tax_row_gpu  = wp.zeros(int(self.NWORLD), dtype=int, device=self.device)

        # PCD GPU buffer is shared with the visualisation helper on
        # SingleHandSubEnv — let that helper own the buffer lazily.
        self.sampled_env._ensure_obj_pcd_gpu_buffers()
        self._pcd_gpu        = self.sampled_env._obj_pcd_gpu
        self._pcd_assignment = self.sampled_env._obj_pcd_assignment_gpu
        self._pcd_n_pts      = int(self.sampled_env._obj_pcd_n_pts)
        # Resolve graspable obj slot + body id (single-obj grasping target).
        self._grasp_obj_slot, self._grasp_obj_body_id = \
            self.sampled_env._resolve_obj_body_id(None)

        # Resolve the wrist freejoint's qvel address. The wrist body holds
        # a single 6-DOF freejoint (3 lin + 3 ang); its qvel slice is
        # ``d.qvel[w, _wrist_qva : _wrist_qva + 6]``. Independent of
        # ``wrist_qpa`` (qpos addr, 7 values for the freejoint — 3 pos +
        # 4 quat). The wrist_vel kernel reads this slice for the
        # stationarity coefficient.
        wrist_jntadr   = int(self.mjm.body(self.hand_util.rh_wrist_base_name).jntadr[0])
        self._wrist_qva = int(self.mjm.jnt_dofadr[wrist_jntadr])

        # Hand-center SE3 in wrist frame (constant per hand —
        # ``rh_hand_center`` 4×4 from ``hand_info/<hand>/rh_info.py``).
        # Both translation (``[:3, 3]``) and rotation (``[:3, :3]``) are
        # exposed so kernels can compose the FULL world-frame hand-center
        # pose, not just its position:
        #
        #     p_hc(w) = R_wrist(w) · rh_hand_center[:3, 3] + wrist_p(w)
        #     R_hc(w) = R_wrist(w) · rh_hand_center[:3, :3]
        #     hc_x_world(w) = R_hc(w)[:, 0]   # palm-out direction
        #
        # Matches the CPU formula
        # ``rh_hand_center_T_calculated = wrist_T @ hand_util.rh_hand_center``
        # from ``notebook/hand/01_hand_setup/10_viz_hand_center.ipynb``.
        #
        # Three exposures:
        #   * ``_hand_center_offset_np``  — numpy 3-vec. Retained as a
        #                                   convenience attribute for
        #                                   downstream code that may still
        #                                   want a CPU view of the offset.
        #                                   (``target_R`` is now sampled by
        #                                   ``hand_utils.sample_target_R_init_per_world``
        #                                   which takes the full 4x4
        #                                   ``rh_hand_center`` directly.)
        #   * ``_hand_center_offset_vec`` — wp.vec3 scalar, consumed by
        #                                   ``_object_grasping_approach_err_kernel``.
        #   * ``_hand_center_R_mat``      — wp.mat33 scalar, consumed by
        #                                   ``_object_grasping_wrist_dir_kernel``
        #                                   to compose the hand-center
        #                                   rotation each step.
        # Both wp arrays use the same row-major flatten pattern as
        # ``_gpu_p_inv`` / ``_gpu_R_inv`` in
        # ``SingleHandSubEnv.setup_warp_pose_init_kernels``.
        hc_full_np = np.asarray(self.hand_util.rh_hand_center, dtype=np.float32)
        hc_pos_np  = hc_full_np[:3, 3].astype(np.float32, copy=False)
        hc_R_np    = hc_full_np[:3, :3].astype(np.float32, copy=False)
        self._hand_center_offset_np  = hc_pos_np
        self._hand_center_offset_vec = wp.vec3(*hc_pos_np.tolist())
        self._hand_center_R_mat      = wp.mat33(*hc_R_np.flatten().tolist())

        # DONE_OBJ_Z_MIN follows the **table surface height** 
        table_h = getattr(self.sampled_env, "table_height", None)
        if table_h is not None:
            self.DONE_OBJ_Z_MIN = float(table_h) - 0.05

    def _build_taxonomy_qpos_table(self) -> Optional[np.ndarray]:
        """Re-index ``hand_util.taxonomy_qpos_array`` (joint-indexed) into
        actuator order. Same algorithm as ``HandPoseTrackingHandler``."""
        ctrl_idx = np.asarray(
            getattr(self.hand_util, "ctrl_joint_idx_array", []), dtype=int,
        )
        tax_raw  = np.asarray(
            getattr(self.hand_util, "taxonomy_qpos_array", []), dtype=np.float32,
        )
        if (tax_raw.ndim != 2 or tax_raw.size == 0
                or ctrl_idx.size == 0
                or tax_raw.shape[1] < int(ctrl_idx.max()) + 1):
            return None
        tax = tax_raw[:, ctrl_idx].astype(np.float32)
        if tax.shape[1] != self.n_ctrl:
            return None
        return tax

    def _build_taxonomy_face_dir_table(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Re-index ``hand_util.taxonomy_hand_face_dir_{idx_in_mat,sign}_array``
        (palm + fingers, no arm) into sensor space (palm + arm + fingers).

        Mirrors the pattern of :meth:`_build_taxonomy_qpos_table` but for the
        per-sensor "face direction" arrays defined in
        ``hand_info/<hand>/<hand>_taxonomy_annotation.py``::

            'hand_face_dir_idx_in_mat': [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0]
            'hand_face_dir_sign'      : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1]

        These specify, **per (taxonomy, hand sensor)**, which column of the
        sensor site's world rotation matrix (``site_xmat[w, sid, :, k]``)
        carries the "facing-outward" direction the body should push toward
        the object, plus a sign multiplier (``±1``). The visualisation in
        ``notebook/hand/01_hand_setup/05_fk_taxonomy.ipynb`` uses
        exactly these arrays to draw the green / red arrows out of each
        palm + finger sensor.

        **Indexing convention.** The legacy hand_util arrays are laid out
        as ``[palm, finger_link_1, finger_link_2, ...]`` — palm at index 0,
        then per-finger links, with **NO** arm slot. Our sensor space
        (``hand_sensor_site_id_list``) inserts the forearm sensor at index
        1, giving the layout ``[palm, arm, finger_1, ...]`` of length
        ``self._n_sensors``. To align so a kernel can index ``face_dir[w, i]``
        with ``i`` in sensor space, we insert a placeholder column at the
        arm slot:

          * ``face_dir_idx[:, 1]  = 0``  (column 0 — irrelevant, arm is
                                          always masked out via
                                          ``specific_finger_link_mask[:, 1] == 0``)
          * ``face_dir_sign[:, 1] = 1``  (any sign — also masked out)

        Returns ``(idx_table, sign_table)`` of shapes ``(n_tax, n_sensors)``
        ``int32`` and ``(n_tax, n_sensors)`` ``float32``; or ``None`` when
        the hand_util arrays are missing / shaped wrong (e.g. a hand whose
        ``taxonomy_annotation`` doesn't declare face directions).
        """
        raw_idx = np.asarray(
            getattr(self.hand_util, "taxonomy_hand_face_dir_idx_in_mat_array", []),
            dtype=np.int32,
        )
        raw_sgn = np.asarray(
            getattr(self.hand_util, "taxonomy_hand_face_dir_sign_array", []),
            dtype=np.float32,
        )

        # n_sensors = palm + arm + n_fingers, so the hand_util arrays (palm
        # + n_fingers, no arm) should have width n_sensors - 1.
        n_sensors = int(self._n_sensors)
        expected_w = n_sensors - 1
        if (raw_idx.ndim != 2 or raw_sgn.ndim != 2
                or raw_idx.size == 0 or raw_sgn.size == 0
                or raw_idx.shape != raw_sgn.shape
                or raw_idx.shape[1] != expected_w):
            return None

        n_tax = int(raw_idx.shape[0])

        # Build sensor-space tables: insert arm placeholder at index 1.
        idx_table  = np.zeros((n_tax, n_sensors), dtype=np.int32)
        sign_table = np.ones((n_tax, n_sensors),  dtype=np.float32)
        # palm at index 0
        idx_table[:,  0] = raw_idx[:, 0]
        sign_table[:, 0] = raw_sgn[:, 0]
        # arm at index 1 stays at the safe (0, +1) defaults.
        # fingers at indices 2..n_sensors-1 from raw[:, 1:]
        idx_table[:,  2:] = raw_idx[:, 1:]
        sign_table[:, 2:] = raw_sgn[:, 1:]

        return idx_table, sign_table

    # ── Buffer + metrics setup ───────────────────────────────────────────
    def _setup_reward_done_buffers(self) -> None:
        super()._setup_reward_done_buffers()

        n_direct = int(getattr(self.applier, "_n_direct_ctrl", self.n_ctrl))
        self._qpos_err_scale = 1.0 / float(max(n_direct, 1))

        # Joint-limit penalty metric per-actuator mean factor (1 / n_ctrl); the
        # ``r_joint_limit`` buffer itself is allocated via FLOAT_BUFS below and
        # registered in ``metrics`` (post-mixer additive term; ≤ 0).
        self._jl_scale = 1.0 / float(max(int(self.n_ctrl), 1))
        # Reference plane for wrist-height shaping: the table top with_table, else the floor (0).
        self._floor_z = float(getattr(self.sampled_env, "table_height", 0.0) or 0.0)

        # Force scale applied to every raw touch-sensor reading before it is
        # used as a contact "impulse" (self / table / obj). Parsed from the run
        # config (``Training.force_scale_for_state``); defaults to 1.0 when
        # absent. Brings the unnormalised MuJoCo touch force into the scale the
        # state / reward terms expect.
        force_scale, force_scale_for_reward = 1.0, 1.0
        try:
            force_scale = float(self.sampled_env.overall_cfg.Training.force_scale_for_state)
            force_scale_for_reward = float(self.sampled_env.overall_cfg.Training.force_scale_for_reward)
        except Exception:
            force_scale = 0.01
            force_scale_for_reward = 0.05
        self._force_scale_for_state = force_scale
        self._force_scale_for_reward = force_scale_for_reward

        # Build contact-sensor index wp.arrays for the three families.
        # Same algorithm as HandPoseTrackingHandler._build_contact_idx_wp.
        self._n_self_coll, self._self_coll_found_wp, self._self_coll_touch_wp = \
            self._build_contact_idx_wp(
                getattr(self.sampled_env, "contact_sensor_index", {}) or {},
                contact_suffix="_self_coll",
            )
        self._n_table_contact, self._table_contact_found_wp, self._table_contact_touch_wp = \
            self._build_contact_idx_wp(
                getattr(self.sampled_env, "contact_sensor_for_table_index", {}) or {},
                contact_suffix="_table_contact",
            )
        self._n_obj_contact, self._hand_obj_contact_found_wp, self._hand_obj_contact_touch_wp = \
            self._build_contact_idx_wp(
                getattr(self.sampled_env, "contact_sensor_for_object_index", {}) or {},
                contact_suffix="_body_obj_contact",
            )
        # Per-obj-contact-slot sensor-space index — for the contact-direction
        # kernel to fetch each firing body's face column / sign / weight.
        self._hand_obj_contact_sensor_idx_wp = self._build_contact_sensor_idx_wp(
            getattr(self.sampled_env, "contact_sensor_for_object_index", {}) or {},
        )
        # obj / self-coll / table contact families are built in parallel from
        # the same per-hand-body rh_info lists, so their per-slot impulse
        # buffers (``hand_obj_contact_impulse_per_slot`` / ``self_impulse_per_slot``
        # / ``tbl_impulse_per_slot``) share one slot order. The observation
        # REUSES those buffers directly (no extra kernel) — guard the invariant.
        assert self._n_obj_contact == self._n_self_coll == self._n_table_contact, (
            f"contact families must be parallel for obs reuse: "
            f"obj={self._n_obj_contact} self={self._n_self_coll} table={self._n_table_contact}"
        )
        # Fore-arm touch-only family (no contact sensor) — suffix-matched
        # addresses into ``touch_sensor_index``.
        self._n_forearm_touch, self._forearm_touch_adr_wp = \
            self._build_touch_idx_wp(self.FOREARM_TOUCH_SUFFIX)
        # Dynamic-object touch sensor addresses (force applied to the object).
        self._n_obj_touch, self._obj_touch_adr_wp = self._build_obj_touch_adr_wp()
        # Per-slot finger_contact_weights aligned with the self-coll slots
        # above (for the weighted self-collision penalties).
        self._hand_only_contact_weight_wp = self._build_contact_weight_wp(
            getattr(self.sampled_env, "contact_sensor_index", {}) or {},
        )
        # Slot-aligned finger-group ids (palm / each finger = one unit, -1 =
        # unmapped) — finger-level contact_ratio counting. Gathering the
        # sensor-space group table (env.sensor_finger_group_id) in obj-contact
        # slot order lets the kernel read it directly without a slot→sensor indirection.
        _gid  = np.asarray(self.sampled_env.sensor_finger_group_id, dtype=np.int32)
        # NOTE: cfg.handler knobs are applied after the handler is created
        # (apply_handler_knobs), so this mapping, frozen at construction time,
        # reads the cfg directly (same pitfall as the tax-size mask).
        _cfg_h = getattr(self.sampled_env.overall_cfg, "handler", None)
        if _cfg_h is not None:
            try:
                _rc = _cfg_h.get("reward_contact", None)
                if _rc is not None and _rc.get("CONTACT_RATIO_SUBGROUPS", None) is not None:
                    self.CONTACT_RATIO_SUBGROUPS = bool(_rc["CONTACT_RATIO_SUBGROUPS"])
            except Exception:
                pass
        if bool(self.CONTACT_RATIO_SUBGROUPS):
            # Sub-group id = 2·finger + is_distal — two units per finger, {MCP+PIP}/{DIP}.
            # tip_body_idx is a sensor-space index (same convention as the
            # orchestrator's tip_or_palm); the palm has no distal flag so it stays
            # a single 2·g(+0) unit (its odd partner never appears in active_bits, harmless).
            _tipm = np.zeros(len(_gid), dtype=np.int32)
            _tips = np.asarray(getattr(self.hand_util, "tip_body_idx", []) or [], dtype=int)
            _tipm[_tips] = 1
            _gid = np.where(_gid >= 0, _gid * 2 + _tipm, -1).astype(np.int32)
        _sidx = self._hand_obj_contact_sensor_idx_wp.numpy()
        self._slot_finger_id_wp = wp.array(
            np.where(_sidx >= 0, _gid[np.clip(_sidx, 0, None)], -1).astype(np.int32),
            dtype=int, device=self.device,
        )
        self._n_finger_groups = len(self.sampled_env.sensor_finger_group_names) * (
            2 if bool(self.CONTACT_RATIO_SUBGROUPS) else 1)
        if bool(self.CONTACT_RATIO_SUBGROUPS):
            print(f"[contact-ratio] sub-group counting ON — two units per finger (MCP+PIP / DIP), "
                  f"groups {len(self.sampled_env.sensor_finger_group_names)}→{self._n_finger_groups}, "
                  f"slot ids: {self._slot_finger_id_wp.numpy().tolist()}")
        assert self._n_finger_groups <= 30, (
            f"contact_ratio bitmask counting supports ≤30 finger groups, "
            f"got {self._n_finger_groups}"
        )
        # ctrl→finger group mapping for intra-finger motor torque balance (torque_regulate).
        # Based on the base finger groups (no sub-group split) — torque balance is per finger.
        # ctrl index == actuator index, so it aligns directly with d.actuator_force.
        _ctrl_gid = np.asarray(
            getattr(self.sampled_env, "ctrl_finger_group_id", []), dtype=np.int32)
        self._ctrl_finger_group_id_wp = wp.array(_ctrl_gid, dtype=int, device=self.device)
        self._n_ctrl_for_torque = int(len(_ctrl_gid))
        self._n_torque_groups = int(len(self.sampled_env.sensor_finger_group_names))
        if float(getattr(self, "TORQUE_BALANCE_COUPLING", 0.0)) > 0.0:
            print(f"[torque-balance] intra-finger motor torque balance ON — "
                  f"c={self.TORQUE_BALANCE_COUPLING}, groups={self._n_torque_groups}, "
                  f"ctrl→finger id: {_ctrl_gid.tolist()}")
        # Compatibility args for the SHARED ``_hand_contact_active_kernel``,
        # which gained tip-slot handling + a finger-weighted impulse output for
        # the tracking/geom pinch reward (commit 918feb8). object_grasping uses
        # that shared kernel ONLY for the binary self_active / tbl_active flags
        # and computes its own weighted impulse via
        # ``_self_coll_weighted_penalty_kernel`` (see below), so we neutralise
        # the tip logic (is_tip = all-zeros → every slot contributes to
        # self_active exactly like the old signature) and route the kernel's
        # unused impulse output to a scratch buffer so ``raw_self_impulse_w_wp``
        # is not clobbered.
        _n_self_slots = int(self._self_coll_found_wp.shape[0])
        self._self_coll_is_tip_wp = wp.zeros(_n_self_slots, dtype=int, device=self.device)
        self._shared_self_impulse_scratch_wp = wp.zeros(
            int(self.NWORLD), dtype=float, device=self.device)
        self._forearm_touch_weight_wp = self._build_touch_weight_wp(
            self.FOREARM_TOUCH_SUFFIX,
        )

        FLOAT_BUFS = (
            # Mixer outputs (weighted)
            "r_approach", "r_finger", "r_mimic", "r_wrist_dir", "r_lift",
            "r_self_pen", "r_self_impulse_pen", "r_table_pen", "r_table_impulse_pen",
            "r_table_impulse_deriv",
            "r_obj_touch_pen", "r_obj_table_pen",
            "r_violation_pen", "r_joint_limit", "r_wrist_height", "r_finger_height",
            "r_bonus", "r_lift_success_bonus",
            "r_ctrl_cost", "r_action_smooth", "r_wrist_vel", "r_obj_vel",
            "r_obj_xy_coeff", "r_obj_R_coeff", "r_face_dir",
            "r_contact_dir", "r_contact_pos", "r_contact_neg_pen",
            "r_affordance_pos", "r_affordance_neg_pen",
            "r_force_closure",
            "r_sem_contact",        # Σ W_SEM_*·sem_* (single summed slot in the bonus group)
            # Persistent prev-step table impulse (state for the Δ⁺ penalty; NOT a
            # per-step output — updated in-place by the deriv kernel, zeroed on
            # full reset). Not logged.
            "prev_table_impulse_w",
            # Composite / coefficient terms (mixer; the expressions that use
            # the per-term rewards above — exposed for inspection/plotting).
            "r_mimic_qpos_err", "r_mimic_naive", "hand_ineq_coeff", "r_hand_process",
            "obj_ineq_coeff", "ineq_coeff", "r_obj_process",
            "r_hand_obj_bonus", "r_hand_obj_penalty",
            # Compute kernel outputs (raw, pre-mixer)
            "raw_approach", "raw_finger", "raw_mimic",
            "raw_wrist_dir", "raw_lift", "raw_vel", "raw_obj_vel",
            "raw_self_contact_w", "raw_self_impulse_w",
            "raw_table_contact_w", "raw_table_impulse_w",
            "raw_obj_xy_coeff", "raw_obj_R_coeff", "raw_face_dir",
            # Taxonomy-aware contact-affordance reward components.
            "raw_contact_dir_reward", "raw_contact_pos", "raw_contact_neg",
            "raw_affordance_pos", "raw_affordance_neg",
            "raw_force_closure", "force_closure_res",   # exp(-k·‖Gf‖) / wrench residual
            # slot-level taxonomy compliance metrics (always computed; same formulas as eval sem_contact_*)
            "raw_sem_recall", "raw_sem_precision", "raw_sem_f1", "raw_sem_iou",
            "rest_obj_impulse",    # max rest (non-taxonomy) body↔obj contact force (scale for done 11)
            "obj_ground_force",    # approximate object-ground force (scale for done 12 + obs channel)
            "raw_contact_ratio",   # (#active fingers touched)/(#active fingers), finger units
            "torque_balance_coeff",  # intra-finger motor torque balance ∈(0,1] (torque_regulate)
            "action_sqnorm", "action_smooth_sqnorm",
            # Diagnostics
            "approach_err", "finger_err", "mimic_qpos_err", "mimic_naive_err",
            "lift_z", "total_vel", "total_obj_vel",
            # Dominant hand↔obj contact force (scalar; direction is a vec3
            # buffer allocated separately below).
            "hand_obj_contact_impulse",
            # Total self / table contact impulse (scalar magnitude beyond the
            # binary self_active / tbl_active flags).
            "self_impulse", "tbl_impulse",
            # Fore-arm touch sensor reading (touch-only family).
            "forearm_touch",
            # Force applied to the dynamic object (Σ object touch sensors).
            "obj_touch_force",
        )
        INT_BUFS = (
            "self_active", "tbl_active", "hand_obj_active",
            "bonus_active", "bonus_streak",
            # Dynamic mid-episode target switch (shared base feature).
            "target_switch_step", "switch_mask",
        )
        for name in FLOAT_BUFS:
            arr, view = self._alloc_world(float)
            setattr(self, f"{name}_wp", arr)
            setattr(self, f"{name}_torch", view)
        for name in INT_BUFS:
            arr, view = self._alloc_world(int)
            setattr(self, f"{name}_wp", arr)
            setattr(self, f"{name}_torch", view)

        # ``target_switch_step`` starts at -1 (= "no switch scheduled") so nothing
        # fires before the first reset samples valid ticks via _resample_switch_step.
        self.target_switch_step_torch.fill_(-1)

        # last_action: (NWORLD, action_dim) float — stores a_{t-1} for the
        # smoothness kernel. action_dim = 9 + n_ctrl (xyz + rot6D + fingers).
        self._action_dim = 9 + self.n_ctrl
        self.last_action_wp    = wp.zeros(
            (self.NWORLD, self._action_dim), dtype=float, device=self.device,
        )
        self.last_action_torch = wp.to_torch(self.last_action_wp)

        # Per-world manual-lift latch — 1 once the object has reached
        # ``LIFT_TARGET_M`` at least once this episode. Set inside
        # ``_stage_wrist_lift_kernel`` and read there to suppress re-triggering
        # the rule-based wrist lift when gravity later dips the object below the
        # target (prevents the wrist judder). Cleared on every (per-world)
        # episode reset (``_reset_task_buffers`` / ``per_world_reset_if_done``).
        self.lift_reached_wp = wp.zeros(
            (self.NWORLD,), dtype=int, device=self.device,
        )
        self.lift_reached_torch = wp.to_torch(self.lift_reached_wp)   # (NWORLD,) latch view (debug/overlay)

        # Dominant hand↔obj contact pos / direction — (NWORLD,) of vec3
        # (world-frame contact position + unit normal of the max-force
        # contact). Allocated manually (the scalar ``_alloc_world`` helper
        # only makes (NWORLD,) floats); ``wp.to_torch`` exposes each as a
        # (NWORLD, 3) tensor. Kept out of ``self.metrics`` (which expects
        # scalar per-env terms) — read via ``handler.obj_contact_{pos,dir}_torch``.
        self.hand_obj_contact_pos_wp    = wp.zeros(
            (self.NWORLD,), dtype=wp.vec3, device=self.device,
        )
        self.hand_obj_contact_pos_torch = wp.to_torch(self.hand_obj_contact_pos_wp)
        self.hand_obj_contact_dir_wp    = wp.zeros(
            (self.NWORLD,), dtype=wp.vec3, device=self.device,
        )
        self.hand_obj_contact_dir_torch = wp.to_torch(self.hand_obj_contact_dir_wp)

        # Per-slot self / table contact force — (NWORLD, n_slots), one column
        # per hand contact sensor (0 for inactive slots). Exposed via
        # ``handler.{self,tbl}_impulse_per_slot_torch`` for per-sensor signals.
        # max(n, 1) keeps the buffer 2D even when a family has no sensors.
        self.self_impulse_per_slot_wp    = wp.zeros(
            (self.NWORLD, max(int(self._n_self_coll), 1)), dtype=float, device=self.device,
        )
        self.self_impulse_per_slot_torch = wp.to_torch(self.self_impulse_per_slot_wp)
        self.tbl_impulse_per_slot_wp     = wp.zeros(
            (self.NWORLD, max(int(self._n_table_contact), 1)), dtype=float, device=self.device,
        )
        self.tbl_impulse_per_slot_torch  = wp.to_torch(self.tbl_impulse_per_slot_wp)
        # Per-slot hand↔obj contact force — (NWORLD, n_obj_contact), one column
        # per obj-contact sensor. Exposed via ``hand_obj_contact_impulse_per_slot_torch``.
        self.hand_obj_contact_impulse_per_slot_wp    = wp.zeros(
            (self.NWORLD, max(int(self._n_obj_contact), 1)), dtype=float, device=self.device,
        )
        self.hand_obj_contact_impulse_per_slot_torch = wp.to_torch(self.hand_obj_contact_impulse_per_slot_wp)

        # Per-sensor face↔contact-normal cosine — (NWORLD, n_sensors) array
        # (sensor space: palm / arm / fingers). Manually allocated (2D, not a
        # scalar metric); exposed via ``handler.contact_dir_cos_torch`` for
        # downstream contact reward terms. Zeroed each step by the kernel.
        self.contact_dir_cos_wp    = wp.zeros(
            (self.NWORLD, self._n_sensors), dtype=float, device=self.device,
        )
        self.contact_dir_cos_torch = wp.to_torch(self.contact_dir_cos_wp)

        # World-frame object PCD — output of ``_object_grasping_transform_pcd_kernel``
        # (the per-world assigned variant PCD rotated/translated into world
        # frame ONCE per point). Consumed by ``_object_grasping_nearest_pcd_kernel``
        # and exposed for the observation kernel via ``handler.transformed_pcd_torch``
        # (NWORLD, n_pts, 3).
        self.transformed_pcd_wp    = wp.zeros(
            (self.NWORLD, self._pcd_n_pts), dtype=wp.vec3, device=self.device,
        )
        self.transformed_pcd_torch = wp.to_torch(self.transformed_pcd_wp)

        # Per-sensor nearest object-PCD vertex — output of the shared
        # ``_object_grasping_nearest_pcd_kernel`` scan, reused by BOTH the
        # finger-close (best_d2) and face-dir (best_pt) reduction kernels and
        # exposed for the observation kernel. best_d2: (NWORLD, n_sensors) min
        # squared distance; best_pt: (NWORLD, n_sensors) vec3 nearest vertex
        # (world frame) → ``handler.best_{d2,pt}_torch``.
        self.best_d2_wp    = wp.zeros(
            (self.NWORLD, self._n_sensors), dtype=float, device=self.device,
        )
        self.best_d2_torch = wp.to_torch(self.best_d2_wp)
        self.best_pt_wp    = wp.zeros(
            (self.NWORLD, self._n_sensors), dtype=wp.vec3, device=self.device,
        )
        self.best_pt_torch = wp.to_torch(self.best_pt_wp)
        # Hand-center nearest-PCD vertex (1 per world) — output of
        # ``_object_grasping_nearest_pcd_hand_center_kernel``, consumed by the
        # ``hand_center_pcd_err`` obs block. (NWORLD,) vec3 world-frame.
        self.best_pt_hc_wp    = wp.zeros(
            (self.NWORLD,), dtype=wp.vec3, device=self.device,
        )
        self.best_pt_hc_torch = wp.to_torch(self.best_pt_hc_wp)
        # L2 distance hand-center → its nearest obj-PCD vertex (1 per world) —
        # output of the same kernel; consumed by the reward function.
        self.best_dist_hc_wp    = wp.zeros(
            (self.NWORLD,), dtype=float, device=self.device,
        )
        self.best_dist_hc_torch = wp.to_torch(self.best_dist_hc_wp)

        # ── Finite-difference velocity obs state (INCLUDE_FD_VEL_IN_OBS) ────
        # Previous control step's wrist/obj pose snapshot + 12ch FD velocity buffer
        # + per-world seed flag (0 = next update is seed only — blocks the reset
        # teleport spike). Updated once per step by ``_fd_vel_update_kernel``.
        self.prev_wrist_p_wp = wp.zeros((self.NWORLD,), dtype=wp.vec3,  device=self.device)
        self.prev_wrist_R_wp = wp.zeros((self.NWORLD,), dtype=wp.mat33, device=self.device)
        self.prev_obj_p_wp   = wp.zeros((self.NWORLD,), dtype=wp.vec3,  device=self.device)
        self.prev_obj_R_wp   = wp.zeros((self.NWORLD,), dtype=wp.mat33, device=self.device)
        self.fd_valid_wp     = wp.zeros((self.NWORLD,), dtype=int,      device=self.device)
        self.fd_vel_wp       = wp.zeros((self.NWORLD, 12), dtype=float, device=self.device)
        self.fd_vel_torch    = wp.to_torch(self.fd_vel_wp)
        # obj world-z FD velocity — consumed by the obj_fell done (avoids cvel jitter).
        self.fd_obj_vz_wp    = wp.zeros((self.NWORLD,), dtype=float, device=self.device)
        self.fd_obj_vz_torch = wp.to_torch(self.fd_obj_vz_wp)

        # ── Per-world object weight force (force-scaled) — reference for the obj-touch free allowance ──
        # mg·force_scale_for_state (same scale as obj_touch_force). When antigrav is
        # on, set_antigrav_alpha rescales it by (1−α) (effective weight).
        self.obj_weight_force_wp    = wp.zeros((self.NWORLD,), dtype=float, device=self.device)
        self.obj_weight_force_torch = wp.to_torch(self.obj_weight_force_wp)
        _mass = wp.to_torch(self.m.body_mass)
        _obj_mass = (_mass[:, int(self.obj_body_id)] if _mass.dim() == 2
                     else _mass[int(self.obj_body_id)].expand(self.NWORLD))
        _g = float(-self.sampled_env.mjm.opt.gravity[2])
        self._obj_weight_force_base = (
            _obj_mass.to(self.obj_weight_force_torch.dtype)
            * _g * float(self._force_scale_for_state)).clone()
        self.obj_weight_force_torch.copy_(self._obj_weight_force_base)
        # Physical control dt (sim_nstep · timestep) — REWARD_DT may be set to a
        # different value in yaml, so the FD must divide by the actual dt.
        self._fd_inv_dt = 1.0 / (
            float(self.sim_nstep) * float(self.sampled_env.mjm.opt.timestep))

        # Public metric dict — consumed by RewardTermReturnTracker / W&B.
        metric_names = (
            "reward",
            "r_approach", "r_finger", "r_mimic", "r_wrist_dir", "r_lift",
            "r_self_pen", "r_self_impulse_pen", "r_table_pen", "r_table_impulse_pen",
            "r_table_impulse_deriv",
            "r_obj_touch_pen", "r_obj_table_pen",
            "r_violation_pen", "r_joint_limit", "r_wrist_height", "r_finger_height",
            "r_bonus", "r_lift_success_bonus",
            "r_ctrl_cost", "r_action_smooth", "r_wrist_vel", "r_obj_vel",
            "r_obj_xy_coeff", "r_obj_R_coeff", "r_face_dir",
            "r_contact_dir", "r_contact_pos", "r_contact_neg_pen",
            "r_affordance_pos", "r_affordance_neg_pen",
            "r_force_closure", "r_sem_contact",
            # Composite / coefficient terms (mixer-internal, now exposed).
            "r_mimic_qpos_err", "r_mimic_naive", "hand_ineq_coeff", "r_hand_process",
            "obj_ineq_coeff", "ineq_coeff", "r_obj_process",
            "r_hand_obj_bonus", "r_hand_obj_penalty",
            "raw_approach", "raw_finger", "raw_mimic",
            "raw_wrist_dir", "raw_lift", "raw_vel", "raw_obj_vel",
            "raw_self_contact_w", "raw_self_impulse_w",
            "raw_table_contact_w", "raw_table_impulse_w",
            "raw_obj_xy_coeff", "raw_obj_R_coeff", "raw_face_dir",
            "raw_contact_dir_reward", "raw_contact_pos", "raw_contact_neg",
            "raw_affordance_pos", "raw_affordance_neg",
            "raw_force_closure", "force_closure_res",
            "raw_sem_recall", "raw_sem_precision", "raw_sem_f1", "raw_sem_iou",
            "rest_obj_impulse", "obj_ground_force",
            "raw_contact_ratio", "torque_balance_coeff",
            "action_sqnorm", "action_smooth_sqnorm",
            "self_active", "tbl_active", "hand_obj_active",
            "bonus_active", "bonus_streak",
            "approach_err", "finger_err", "mimic_qpos_err", "mimic_naive_err",
            "lift_z", "total_vel", "total_obj_vel",
            "hand_obj_contact_impulse", "self_impulse", "tbl_impulse",
            "forearm_touch", "obj_touch_force",
            "obj_weight_force",   # per-world (1−α)·m·g (scale for the free allowance)
        )
        self.metrics: Dict[str, torch.Tensor] = {
            n: getattr(self, f"{n}_torch") for n in metric_names
        }
        self._setup_v2_buffers()

    def _reset_task_buffers(self) -> None:
        """Clear streak / bonus_active / last_action on every full reset so
        the next episode batch starts from a clean state."""
        if hasattr(self, "bonus_streak_wp"):
            self.bonus_streak_wp.zero_()
        if hasattr(self, "bonus_active_wp"):
            self.bonus_active_wp.zero_()
        if hasattr(self, "last_action_wp"):
            self.last_action_wp.zero_()
        if hasattr(self, "lift_reached_wp"):
            self.lift_reached_wp.zero_()
        # prev state of the Δ⁺ table-impulse penalty — a new episode batch starts clean.
        if hasattr(self, "prev_table_impulse_w_wp"):
            self.prev_table_impulse_w_wp.zero_()
        # Invalidate the pose-FD velocity state — a full reset teleports every
        # world, so the next step's update kernel seeds from the fresh pose (velocity 0).
        if hasattr(self, "fd_valid_wp"):
            self.fd_valid_wp.zero_()
            self.fd_vel_wp.zero_()
            self.fd_obj_vz_wp.zero_()
        for nm in ("ep_elapsed_wp", "drop_streak_wp", "grasp_progress_wp", "retry_count_wp",
                   "lift_hold_streak_wp", "retry_fired_wp", "early_success_wp"):
            if hasattr(self, nm):
                getattr(self, nm).zero_()

    def _build_contact_idx_wp(
        self,
        contact_index: dict,
        contact_suffix: str,
    ) -> Tuple[int, wp.array, wp.array]:
        """Flatten one contact-sensor family into ``(found_idx, touch_idx)``
        wp.arrays. Same algorithm as in HandPoseTrackingHandler."""
        touch_index = (getattr(self.sampled_env, "touch_sensor_index", {}) or {})
        # Case-INSENSITIVE lookup ``lower(name) → adr``: some hands (allex) name
        # the touch sensor link CamelCase ("..._Thumb_Proximal_touch") while the
        # contact sensor is lowercase ("..._thumb_proximal_..._obj_contact").
        touch_index_lc = {str(k).lower(): int(v[0]) for k, v in touch_index.items()}
        found_list: List[int] = []
        touch_list: List[int] = []
        for name, (adr, dim, num) in contact_index.items():
            if num <= 0 or dim <= 0 or (dim % num) != 0 or (dim // num) not in (7, 10):
                continue
            psize = dim // num   # per-record width: 7 ([found,pos,normal]) or 10 (+force)
            # Resolve the paired TOUCH (force) sensor address. Hands name their
            # touch sensors INCONSISTENTLY relative to their contact sensors:
            #   * robotis : contact "<link>_body_obj_contact"  ↔ touch "<link>_touch"
            #               ("_body" lives only in the contact suffix)
            #   * tesollo : contact "<link>_body_obj_contact" / "<link>_table_contact"
            #               / "<link>_self_coll"  ↔ touch "<link>_body_touch"
            #               (touch KEEPS "_body"; table/self_coll contacts have none)
            # So try several candidate touch names and take the first that exists:
            # strip the full family suffix AND each generic contact tail, each with
            # and without an inserted "_body". robotis matches "<link>_touch" first
            # (backward-compatible); tesollo falls through to "<link>_body_touch".
            # (Previously only "<link>_touch" was tried → tesollo missed → adr=-1
            #  → all hand contact FORCE/impulse/affordance terms read 0.)
            touch_adr = -1
            bases: List[str] = []
            if name.endswith(contact_suffix):
                bases.append(name[: -len(contact_suffix)])
            for _tail in ("_body_obj_contact", "_obj_contact", "_table_contact",
                          "_self_coll", "_contact"):
                if name.endswith(_tail):
                    bases.append(name[: -len(_tail)])
            cands: List[str] = []
            for _b in bases:
                cands.append(_b + "_touch")
                if not _b.endswith("_body"):
                    cands.append(_b + "_body_touch")
            for _c in dict.fromkeys(cands):
                _adr = touch_index_lc.get(_c.lower())
                if _adr is not None:
                    touch_adr = _adr
                    break
            for k in range(int(num)):
                found_list.append(int(adr) + psize * k)
                touch_list.append(int(touch_adr))
        n_slots = len(found_list)
        if n_slots == 0:
            found_np = np.zeros(1, dtype=np.int32)
            touch_np = np.zeros(1, dtype=np.int32)
        else:
            found_np = np.asarray(found_list, dtype=np.int32)
            touch_np = np.asarray(touch_list, dtype=np.int32)
        return (
            n_slots,
            wp.array(found_np, dtype=int, device=self.device),
            wp.array(touch_np, dtype=int, device=self.device),
        )

    def _build_touch_idx_wp(self, touch_suffix: str) -> Tuple[int, wp.array]:
        """Collect the ``sensordata`` addresses of every **touch sensor**
        whose name ends with ``touch_suffix`` into a wp.array.

        For sensor families that have ONLY a standalone touch sensor and no
        found/pos/normal contact sensor — e.g. the fore-arm
        (``_arm_part_touch``). Reads ``sampled_env.touch_sensor_index``
        (``name → (adr, dim)``, built by
        ``SingleHandSubEnv._build_touch_sensor_index`` from
        ``hand_util.fore_arm_touch_sensor`` + ``hand_touch_sensors``). A
        MuJoCo touch sensor is dim-1, but we expand ``dim`` slots defensively
        so a multi-component pad still sums correctly. Suffix matching keeps
        this general (no hard-coded sensor name); returns ``(0, dummy)`` when
        no sensor matches.
        """
        touch_index = (getattr(self.sampled_env, "touch_sensor_index", {}) or {})
        adr_list: List[int] = []
        for name, val in touch_index.items():
            if not name.endswith(touch_suffix):
                continue
            adr = int(val[0])
            dim = int(val[1]) if len(val) > 1 else 1
            for k in range(max(dim, 1)):
                adr_list.append(adr + k)
        n_slots = len(adr_list)
        adr_np = (np.asarray(adr_list, dtype=np.int32)
                  if n_slots > 0 else np.zeros(1, dtype=np.int32))
        return n_slots, wp.array(adr_np, dtype=int, device=self.device)

    def _build_contact_weight_wp(self, contact_index: dict) -> wp.array:
        """Per-slot ``finger_contact_weights`` aligned with the found/touch
        slots that :meth:`_build_contact_idx_wp` produces for the SAME
        ``contact_index``.

        Re-iterates ``contact_index.items()`` in identical order (same
        record-width filter (7 or 10), same ``num`` expansion) so slot ``k`` here matches
        slot ``k`` in ``found_idx`` / ``touch_idx``. Each contact sensor is
        mapped to its hand-sensor body (``sampled_env.sensor_body_index``),
        then to the sensor-space index (position in
        ``hand_sensor_body_id_list``), then to
        ``sampled_env.finger_contact_weights[idx]``. Sensors whose body isn't
        a known hand sensor default to weight 1.0.
        """
        env          = self.sampled_env
        body_index   = (getattr(env, "sensor_body_index", {}) or {})
        body_ids     = np.asarray(
            getattr(env, "hand_sensor_body_id_list", []), dtype=np.int64,
        ).reshape(-1)
        fcw          = np.asarray(
            getattr(env, "finger_contact_weights", []), dtype=np.float32,
        ).reshape(-1)
        # body id → first sensor-space index.
        body_to_idx: Dict[int, int] = {}
        for i, b in enumerate(body_ids.tolist()):
            body_to_idx.setdefault(int(b), i)

        weight_list: List[float] = []
        for name, (adr, dim, num) in contact_index.items():
            if num <= 0 or dim <= 0 or (dim % num) != 0 or (dim // num) not in (7, 10):
                continue
            wv = 1.0
            bid = body_index.get(name, None)
            if bid is not None and int(bid) in body_to_idx:
                si = body_to_idx[int(bid)]
                if si < fcw.size:
                    wv = float(fcw[si])
            for _k in range(int(num)):
                weight_list.append(wv)
        weight_np = (np.asarray(weight_list, dtype=np.float32)
                     if weight_list else np.zeros(1, dtype=np.float32))
        return wp.array(weight_np, dtype=float, device=self.device)

    def _build_contact_sensor_idx_wp(self, contact_index: dict) -> wp.array:
        """Per-slot **sensor-space index** aligned with the found/touch slots
        :meth:`_build_contact_idx_wp` produces for the SAME ``contact_index``.

        Same iteration as :meth:`_build_contact_weight_wp`, but stores each
        slot's index into the sensor-space arrays (``hand_sensor_body_id_list``
        / ``finger_weights`` / the ``face_dir_*`` cond fields) instead of its
        weight. Used by ``_object_grasping_contact_dir_kernel`` to fetch, per
        obj-contact slot, the firing body's face column / sign / weight.
        Unmapped slots (sensor body not in the hand-sensor list) get ``-1``
        and are skipped by the kernel.
        """
        env        = self.sampled_env
        body_index = (getattr(env, "sensor_body_index", {}) or {})
        body_ids   = np.asarray(
            getattr(env, "hand_sensor_body_id_list", []), dtype=np.int64,
        ).reshape(-1)
        body_to_idx: Dict[int, int] = {}
        for i, b in enumerate(body_ids.tolist()):
            body_to_idx.setdefault(int(b), i)

        idx_list: List[int] = []
        for name, (adr, dim, num) in contact_index.items():
            if num <= 0 or dim <= 0 or (dim % num) != 0 or (dim // num) not in (7, 10):
                continue
            si  = -1
            bid = body_index.get(name, None)
            if bid is not None and int(bid) in body_to_idx:
                si = int(body_to_idx[int(bid)])
            for _k in range(int(num)):
                idx_list.append(si)
        idx_np = (np.asarray(idx_list, dtype=np.int32)
                  if idx_list else np.full(1, -1, dtype=np.int32))
        return wp.array(idx_np, dtype=int, device=self.device)

    def _build_touch_weight_wp(self, touch_suffix: str) -> wp.array:
        """Per-slot ``finger_contact_weights`` aligned with the touch-only
        slots that :meth:`_build_touch_idx_wp` produces for the same
        ``touch_suffix`` (e.g. the fore-arm ``_arm_part_touch``).

        Mirrors ``_build_touch_idx_wp``'s iteration (same suffix filter, same
        ``dim`` expansion, same order) and maps each touch sensor to its body
        (``sensor_body_index``) → sensor-space index → ``finger_contact_weights``
        (so the fore-arm slot picks up the arm-row ×4-boosted weight).
        Unknown bodies default to 1.0.
        """
        env         = self.sampled_env
        touch_index = (getattr(env, "touch_sensor_index", {}) or {})
        body_index  = (getattr(env, "sensor_body_index", {}) or {})
        body_ids    = np.asarray(
            getattr(env, "hand_sensor_body_id_list", []), dtype=np.int64,
        ).reshape(-1)
        fcw         = np.asarray(
            getattr(env, "finger_contact_weights", []), dtype=np.float32,
        ).reshape(-1)
        body_to_idx: Dict[int, int] = {}
        for i, b in enumerate(body_ids.tolist()):
            body_to_idx.setdefault(int(b), i)

        weight_list: List[float] = []
        for name, val in touch_index.items():
            if not name.endswith(touch_suffix):
                continue
            dim = int(val[1]) if len(val) > 1 else 1
            wv  = 1.0
            bid = body_index.get(name, None)
            if bid is not None and int(bid) in body_to_idx:
                si = body_to_idx[int(bid)]
                if si < fcw.size:
                    wv = float(fcw[si])
            for _k in range(max(dim, 1)):
                weight_list.append(wv)
        weight_np = (np.asarray(weight_list, dtype=np.float32)
                     if weight_list else np.zeros(1, dtype=np.float32))
        return wp.array(weight_np, dtype=float, device=self.device)

    def _build_obj_touch_adr_wp(self) -> Tuple[int, wp.array]:
        """Collect ``sensordata`` addresses of the **dynamic object's** TOUCH
        sensors (force applied to the object).

        The graspable object carries a MuJoCo touch sensor (added in
        ``hand_utils.make_mjcf_from_spec`` on the renamed object body) named
        ``<obj_name>_touch`` — it sums the normal force on the object's touch
        zone. The match substring is derived from the yaml ``Training.obj_name``
        (the object's renamed base, also used as ``rename_body_name`` in the
        orchestrator) + ``OBJ_TOUCH_SENSOR_SUFFIX``, rather than hard-coded, so
        it tracks the configured object. We scan the compiled model for touch
        sensors whose name contains that substring and return their per-world
        sensordata addresses; summing them gives the total force on the object
        (penalised to learn gentle grasps). ``(0, dummy)`` when ``obj_name`` is
        unset or no sensor matches.
        """
        m = self.mjm
        # Object base name from the run config (same value the orchestrator
        # passes as ``rename_body_name`` to ``make_mjcf_from_spec``).
        obj_name = ""
        try:
            obj_name = str(self.sampled_env.overall_cfg.Training.obj_name or "")
        except Exception:
            obj_name = ""
        if not obj_name:
            # No configured object name → can't disambiguate the object touch
            # sensor from hand touch sensors; report none (penalty inert).
            return 0, wp.array(np.zeros(1, dtype=np.int32), dtype=int, device=self.device)
        substr = obj_name + self.OBJ_TOUCH_SENSOR_SUFFIX

        # Slot scoping: the substring alone matches EVERY object slot's touch
        # sensor (``<obj>_touch_0``, ``<obj>_touch_1``, …) — in a multi-slot
        # scene (n_obj ≥ 2, or the leader-follower 2·n_obj layout) that would
        # sum forces on OTHER slots' objects into this handler's
        # ``obj_touch_force``. Keep only sensors whose site sits inside the
        # TARGET object's body subtree (``_grasp_obj_body_id``). n_obj = 1
        # single-hand scenes are unaffected (the only match IS the target).
        def _in_target_subtree(body_id: int) -> bool:
            tgt = int(self._grasp_obj_body_id)
            b   = int(body_id)
            while b > 0:
                if b == tgt:
                    return True
                b = int(m.body_parentid[b])
            return b == tgt

        adr_list: List[int] = []
        for sid in range(int(m.nsensor)):
            if int(m.sensor_type[sid]) != int(_mj.mjtSensor.mjSENS_TOUCH):  # type: ignore
                continue
            name = _mj.mj_id2name(m, _mj.mjtObj.mjOBJ_SENSOR, sid) or ""     # type: ignore
            if substr not in name:
                continue
            site_id = int(m.sensor_objid[sid])
            if not _in_target_subtree(int(m.site_bodyid[site_id])):
                continue
            adr_list.append(int(m.sensor_adr[sid]))
        n = len(adr_list)
        adr_np = (np.asarray(adr_list, dtype=np.int32)
                  if n > 0 else np.zeros(1, dtype=np.int32))
        return n, wp.array(adr_np, dtype=int, device=self.device)

    # ── Kernel launches ──────────────────────────────────────────────────
    def _launch_contact_active_kernels(self) -> None:
        """Self / table / obj contact flags. Skips weight=0 + sensor-empty
        families to keep the no-penalty path cheap; always launches the
        obj kernel when slots exist (the obs and bonus gate read it)."""
        # Self + table family.
        if ((self.W_SELF_COLL == 0.0) and (self.W_TABLE_CONTACT == 0.0)) \
                or (self._n_self_coll == 0 and self._n_table_contact == 0):
            self.self_active_wp.zero_()
            self.tbl_active_wp.zero_()
            self.self_impulse_wp.zero_()
            self.tbl_impulse_wp.zero_()
            self.self_impulse_per_slot_wp.zero_()
            self.tbl_impulse_per_slot_wp.zero_()
        else:
            wp.launch(
                _hand_contact_active_kernel, dim=self.NWORLD,
                inputs=[
                    self.d.sensordata,
                    int(self._n_self_coll),
                    self._self_coll_found_wp, self._self_coll_touch_wp,
                    self._hand_only_contact_weight_wp,   # self_slot_weight
                    self._self_coll_is_tip_wp,           # self_slot_is_tip (all-zeros → no tip exemption)
                    int(self._n_table_contact),
                    self._table_contact_found_wp, self._table_contact_touch_wp,
                    float(self.TOUCH_EPS),
                    int(1) if self.GATE_CONTACT_BY_TOUCH else int(0),
                    float(self._force_scale_for_reward),  # force_scale
                    float(0.0),                           # tip_deadband (unused here)
                    self.self_active_wp,
                    self._shared_self_impulse_scratch_wp,  # out_self_impulse_w (ignored; own kernel computes it)
                    self.tbl_active_wp,
                ],
            )
            # Self-collision + table contact impulse (scalar magnitude) —
            # recorded alongside the shared kernel's binary ``self_active`` /
            # ``tbl_active``. Generic local kernel run once per family so the
            # shared (tracking-shared) kernel signature is untouched. Each
            # self-handles ``n_slots == 0`` (writes 0).
            wp.launch(
                _contact_impulse_kernel, dim=self.NWORLD,
                inputs=[
                    self.d.sensordata,
                    int(self._n_self_coll),
                    self._self_coll_found_wp, self._self_coll_touch_wp,
                    float(self.TOUCH_EPS),
                    int(1) if self.GATE_CONTACT_BY_TOUCH else int(0),
                    float(self._force_scale_for_state),
                    self.self_impulse_wp,
                    self.self_impulse_per_slot_wp,
                ],
            )
            wp.launch(
                _contact_impulse_kernel, dim=self.NWORLD,
                inputs=[
                    self.d.sensordata,
                    int(self._n_table_contact),
                    self._table_contact_found_wp, self._table_contact_touch_wp,
                    float(self.TOUCH_EPS),
                    int(1) if self.GATE_CONTACT_BY_TOUCH else int(0),
                    float(self._force_scale_for_state),
                    self.tbl_impulse_wp,
                    self.tbl_impulse_per_slot_wp,
                ],
            )

        # Hand ↔ object family.
        if self._n_obj_contact == 0:
            self.hand_obj_active_wp.zero_()
            self.hand_obj_contact_impulse_wp.zero_()
            self.hand_obj_contact_pos_wp.zero_()
            self.hand_obj_contact_dir_wp.zero_()
            self.hand_obj_contact_impulse_per_slot_wp.zero_()
        else:
            wp.launch(
                _obj_contact_active_kernel, dim=self.NWORLD,
                inputs=[
                    self.d.sensordata,
                    int(self._n_obj_contact),
                    self._hand_obj_contact_found_wp, self._hand_obj_contact_touch_wp,
                    float(self.TOUCH_EPS),
                    int(1) if self.GATE_CONTACT_BY_TOUCH else int(0),
                    float(self._force_scale_for_state),
                    self.hand_obj_active_wp,
                    self.hand_obj_contact_impulse_wp,
                    self.hand_obj_contact_pos_wp,
                    self.hand_obj_contact_dir_wp,
                    self.hand_obj_contact_impulse_per_slot_wp,
                ],
            )

        # Fore-arm touch-only family (no contact sensor — touch sensor only).
        # Sums the suffix-matched fore-arm touch scalar(s) into a per-world
        # value. Skip the launch when no fore-arm touch sensor is present to
        # keep the no-sensor path cheap (the kernel also self-handles n == 0).
        if self._n_forearm_touch == 0:
            self.forearm_touch_wp.zero_()
        else:
            wp.launch(
                _touch_sum_kernel, dim=self.NWORLD,
                inputs=[
                    self.d.sensordata,
                    int(self._n_forearm_touch),
                    float(self._force_scale_for_state),
                    self._forearm_touch_adr_wp,
                    self.forearm_touch_wp,
                ],
            )

    # ── Eval ghost (taxonomy target_qpos beside the object) ─────────────
    # Display-only pose for the evaluate.py ghost hand: grasping has NO wrist
    # target, so the ghost wrist is parked at a fixed offset beside each
    # world's object and only the fingers carry meaning — they show the
    # taxonomy ``target_qpos``. Mirrors notebook 31's grasping ghost
    # (``GHOST_OBJ_OFF`` / identity display quat).
    GHOST_WRIST_OFFSET: Tuple[float, float, float]        = (0.0, -0.28, 0.12)
    GHOST_WRIST_QUAT:   Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def eval_ghost_targets(self):                                                 # type: ignore
        """Grasping ghost = taxonomy ``target_qpos`` FK'd at a fixed display
        pose beside each world's object (no wrist target in this task)."""
        off  = np.asarray(self.GHOST_WRIST_OFFSET, dtype=np.float32)
        obj_xpos = self.d.xpos.numpy()[:, self.obj_body_id, :].astype(np.float32)  # (N,3) per-world local
        pos  = obj_xpos + off[None, :]                                             # grid offset added by set_ghost_targets
        quat = np.tile(np.asarray(self.GHOST_WRIST_QUAT, dtype=np.float32),
                       (self.NWORLD, 1)).astype(np.float32)                        # (N,4) fixed display orientation
        qpos = self.cond.get_numpy("target_qpos").astype(np.float32)              # (N, n_ctrl) taxonomy fingers
        return pos, quat, qpos

    # Ghost finger colours (RGB; alpha kept from the ghost model's geom_rgba).
    # active = contact-target finger/palm (green); rest = non-contact (red).
    GHOST_ACTIVE_RGB: Tuple[float, float, float] = (0.15, 0.85, 0.20)
    GHOST_REST_RGB:   Tuple[float, float, float] = (0.90, 0.20, 0.18)

    def _init_ghost_colorize(self, ghost_model) -> None:
        """Build the ghost-geom → taxonomy-sensor mapping ONCE (cached).

        Ported from notebook 31's grasping ghost colourise: each ghost geom is
        mapped to its finger token (own body name; palm = ``'base'``), and each
        token to the ``specific_finger_link_mask`` sensor columns of the same
        finger. ``specific_ctrl_mask`` is all-zero for these hands (annotation
        token vs actuator-name mismatch), so the sensor-space link mask is the
        reliable active/rest signal. Geoms with no finger token (arm / world /
        mocap) carry ``None`` → left at the ghost's own (grey) colour.
        """
        finger_tokens = sorted(
            {t.lower() for names in self.hand_util.taxonomy_specific_finger_names_array
             for t in names},
            key=len, reverse=True,
        )

        def tok_of(name):
            n = (name or "").lower()
            for t in finger_tokens:
                if t in n:
                    return t
            return None

        geom_bid = np.asarray(ghost_model.geom_bodyid, dtype=int)
        # ghost geom → finger token (geom's own body name)
        self._ghost_geom_tok = [
            tok_of(_mj.mj_id2name(ghost_model, _mj.mjtObj.mjOBJ_BODY, int(b)))
            for b in geom_bid
        ]
        # finger token → specific_finger_link_mask sensor columns
        self._ghost_tok2sensors = {}
        for si, b in enumerate(self.sampled_env.hand_sensor_body_name_list):
            st = tok_of(b)
            if st is not None:
                self._ghost_tok2sensors.setdefault(st, []).append(si)
        # Base RGBA = the ghost model's own per-geom colour (grey + alpha) — we
        # only override RGB on the finger geoms and keep each geom's alpha.
        self._ghost_base_rgba   = np.asarray(ghost_model.geom_rgba, dtype=np.float32)  # (ngeom, 4)
        self._ghost_active_rgb  = np.asarray(self.GHOST_ACTIVE_RGB, np.float32)
        self._ghost_rest_rgb    = np.asarray(self.GHOST_REST_RGB,   np.float32)

    def eval_ghost_geom_rgba(self):
        """Per-world ghost geom RGBA from the per-world taxonomy mask: a finger
        token is active (green) when ANY of its ``specific_finger_link_mask``
        sensor columns is set this world, else rest (red); tokenless geoms keep
        the ghost's neutral grey. Shape ``(NWORLD, ngeom, 4)``; ``None`` when
        the ghost model isn't built yet (evaluate.py builds it before the loop).
        """
        ghost_model = getattr(self.sampled_env, "ghost_hand_model", None)
        if ghost_model is None or "specific_finger_link_mask" not in self.cond._fields:
            return None
        if getattr(self, "_ghost_geom_tok", None) is None:
            self._init_ghost_colorize(ghost_model)
        flink = self.cond.get_numpy("specific_finger_link_mask").astype(np.float32)   # (N, n_sensors)
        tok_active = {t: (flink[:, cols] > 0.5).any(1)
                      for t, cols in self._ghost_tok2sensors.items()}
        rgba = np.tile(self._ghost_base_rgba[None], (self.NWORLD, 1, 1))              # (N, ngeom, 4)
        for gi, t in enumerate(self._ghost_geom_tok):
            a = tok_active.get(t)
            if a is None:
                continue
            rgba[:, gi, :3] = np.where(a[:, None], self._ghost_active_rgb, self._ghost_rest_rgb)
        return rgba

    # Extra knobs allowed in the curriculum.reward schedule: keys outside the
    # reward_* groups for which a schedule is meaningful (raise the success
    # incentive in proportion to the gate during penalty ramps, so the
    # "success gain / violation loss" ratio does not collapse). _flatten_groups
    # forbids duplicate group membership, so this explicit whitelist is used
    # instead of an alias group. DONE_TABLE_IMPULSE is included so the hard
    # termination threshold can also tighten gradually with the gate (large base
    # → small final, log mode — early on only extreme slams terminate, becoming
    # stricter as performance rises, minimising SR loss).
    CURRICULUM_EXTRA_KNOBS: Tuple[str, ...] = (
        "W_BONUS", "W_LIFT_SUCCESS_BONUS", "DONE_TABLE_IMPULSE",
        "COVERAGE_BONUS_COUPLING")

    def curriculum_pressure(self) -> float:
        """Current "pressure" of the penalty curriculum ∈ [0, 1] — a scalar for
        conditioning the critic obs (``OBS_CURR_PROGRESS``). Default = success-gate
        progress. Subclasses with a different pressure source override this
        (e.g. the Lagrangian handler uses mean λ/λ_max)."""
        return float(getattr(self, "_gate_progress", 0.0))

    _TAX_DIAG_EVERY: int = 250   # [tax-diag] print period (iterations)

    def on_train_progress(self, cur_step: int, total_steps: int) -> None:
        """Decay the rule-based wrist-z lift over training (called once per
        iteration by the training loop). ``lift_curriculum_scale`` = 1.0 until
        ``LIFT_DECAY_START_STEP`` global control steps, then linearly 1 → 0 by
        ``LIFT_DECAY_END_STEP`` (<= 0 → ``total_steps``). Keeps all lift-curriculum
        logic inside the task env (train.py only reports progress)."""
        # Generic reward-knob curriculum (handler.curriculum.reward) — base impl.
        super().on_train_progress(cur_step, total_steps)
        # (c) periodic print of the per-taxonomy diagnostics table
        if getattr(self, "_tax_stats", None) is not None:
            self._tax_diag_iter += 1
            if self._tax_diag_iter % int(self._TAX_DIAG_EVERY) == 0:
                self._print_tax_diag()
        start = int(self.LIFT_DECAY_START_STEP)
        end   = int(self.LIFT_DECAY_END_STEP)
        if end <= 0:
            end = int(total_steps)
        if cur_step < start:
            self.lift_curriculum_scale = 1.0
        elif cur_step >= end:
            self.lift_curriculum_scale = 0.0
        else:
            self.lift_curriculum_scale = 1.0 - float(cur_step - start) / float(max(1, end - start))

        # ── Object anti-gravity assist decay (only when ANTIGRAV_ALPHA > 0) ──────────
        # progress = max(step-window progress, success-gate progress) — the same
        # hybrid pacing as the reward curriculum (early decay once the gate opens,
        # step window as fallback otherwise). α = ANTIGRAV_ALPHA · (1 − progress).
        if float(self.ANTIGRAV_ALPHA) > 0.0:
            a_start = int(float(self.ANTIGRAV_DECAY_START_STEP))
            a_end   = int(float(self.ANTIGRAV_DECAY_END_STEP))
            if a_end <= 0:
                a_end = int(total_steps)
            if cur_step < a_start:
                p = 0.0
            else:
                p = min(float(cur_step - a_start) / float(max(1, a_end - a_start)), 1.0)
            p = max(p, float(getattr(self, "_gate_progress", 0.0)))
            self.set_antigrav_alpha(float(self.ANTIGRAV_ALPHA) * (1.0 - min(p, 1.0)))

        # ── xyz translation scale curriculum (approach speed decay) ──────────────
        # Decay applier.xyz_scale linearly from base to base·XYZ_SCALE_TARGET_FRAC
        # over the [START,END] step window. base is captured on the first call
        # (yaml rl_env value). TARGET_FRAC>=1.0 → off. apply() reads
        # float(self.xyz_scale) every step, so changing the value here takes
        # effect from the next step.
        applier = getattr(self, "applier", None)
        if applier is not None:
            if float(self._xyz_scale_base) < 0.0:
                self._xyz_scale_base = float(applier.xyz_scale)
            frac = float(self.XYZ_SCALE_TARGET_FRAC)
            base = float(self._xyz_scale_base)
            if frac < 1.0 and base > 0.0:
                x_start = int(float(self.XYZ_SCALE_DECAY_START_STEP))
                x_end   = int(float(self.XYZ_SCALE_DECAY_END_STEP))
                if x_end <= 0:
                    x_end = int(total_steps)
                if cur_step < x_start:
                    xp = 0.0
                elif cur_step >= x_end:
                    xp = 1.0
                else:
                    xp = float(cur_step - x_start) / float(max(1, x_end - x_start))
                applier.xyz_scale = base * (1.0 - xp * (1.0 - frac))

    def set_antigrav_alpha(self, alpha: float, force: bool = False) -> None:
        """Apply an upward force ``α·m·g`` to the object (α=1 → weightless, 0 → real gravity).

        ``d.xfrc_applied`` is a (nworld, nbody) spatial_vector — mjwarp packs it
        as [force(3) | torque(3)], so flat index 2 is the world-z force. Being a
        data array it is safe to write under CUDA graph capture, and since it is
        not cleared every step it is written only when α changes (``force=True``
        → rewrite even if the value is unchanged; for a full reset that
        re-initialises d and clears xfrc — :meth:`reset`). Per-world object mass
        comes from the variant-expanded ``m.body_mass`` (NWORLD, nbody). May be
        called directly for manual on/off (``set_antigrav_alpha(0.0)``)."""
        alpha = float(alpha)
        if not force and getattr(self, "_antigrav_alpha", None) == alpha:
            return
        # Fresh view every time, no cached view — no stale reference even when an
        # env rebuild / put_data swaps the d/m arrays (called only on α change + reset).
        mass = wp.to_torch(self.m.body_mass)
        obj_mass = (mass[:, self.obj_body_id] if mass.dim() == 2
                    else mass[self.obj_body_id].expand(self.NWORLD))
        mg = obj_mass * float(-self.mjm.opt.gravity[2])
        wp.to_torch(self.d.xfrc_applied)[:, self.obj_body_id, 2] = alpha * mg
        # Update the effective weight of the obj-touch free allowance — under the
        # anti-gravity assist the force needed to lift the object also shrinks by (1−α).
        if hasattr(self, "obj_weight_force_torch"):
            self.obj_weight_force_torch.copy_(
                self._obj_weight_force_base * (1.0 - alpha))
        prev_print = getattr(self, "_antigrav_last_print", None)
        self._antigrav_alpha = alpha
        # Log only at 0.05 intervals or at on/off boundaries (avoids per-iteration spam from the fine decay).
        if (prev_print is None or abs(alpha - prev_print) >= 0.05
                or (alpha == 0.0 and prev_print != 0.0)):
            self._antigrav_last_print = alpha
            print(f"[antigrav] α={alpha:.3f} ({alpha*100:.0f}% of the object's gravity cancelled)")

    def reset(self) -> torch.Tensor:
        """Full reset — re-apply the anti-gravity assist after the base reset.

        A full reset re-initialises ``d`` wholesale, zeroing ``xfrc_applied``
        too (the per-world async reset kernel does not touch xfrc, so it is
        unaffected). The current α is rewritten so the assist persists across the reset."""
        obs = super().reset()
        a = getattr(self, "_antigrav_alpha", 0.0)
        if a:
            self.set_antigrav_alpha(a, force=True)
        # ── Reset-obs refresh: the base reset's obs collect reads the buffers
        # filled by the reward pipeline (transformed_pcd/best_pt*, obj_touch_force,
        # obj_ground/rest forces) at their **last-step values of the previous
        # rollout** (the documented stale-at-reset convention). The first obs is
        # thus polluted by pre-reset history — breaking determinism under the eval
        # reset reproducibility protocol (EVAL_RESET_SEED) and distorting the first
        # obs of new episodes in training — so the prefix scans are re-run here on
        # the reset state and the obs re-collected. The first obs is now a function
        # of the reset state alone.
        self._launch_pcd_scans()
        self._launch_obj_touch_sum()
        wp.launch(
            _object_grasping_obj_contact_aux_kernel, dim=self.NWORLD,
            inputs=[
                self.d.sensordata,
                int(self._n_obj_contact),
                self._hand_obj_contact_found_wp, self._hand_obj_contact_touch_wp,
                self._hand_obj_contact_sensor_idx_wp,
                self.cond.get("specific_finger_link_mask"),
                self.obj_touch_force_wp,
                float(self.TOUCH_EPS),
                int(1) if self.GATE_CONTACT_BY_TOUCH else int(0),
                float(self._force_scale_for_state),
                self.rest_obj_impulse_wp,
                self.obj_ground_force_wp,
            ],
        )
        self._collect_obs_kernel()
        self._sanitize_obs()
        return self.obs_torch

    def _apply_action_curriculum(self, applier) -> None:
        """Task action pre-processor (bound to the applier via
        ``bind_action_preprocessor``). Rule-based wrist-z lift that fires ONLY
        in the lift stage (``LIFT_STEP <= ep_step < HOLD_STEP``) and ONLY while
        the object has not yet reached ``LIFT_TARGET_M`` — once lifted (or in
        hold/approach), it is a no-op. The per-step world-z lift distance is
        ``LIFT_ACTION_DIST_M · lift_curriculum_scale`` soft-ramped over
        ``LIFT_RAMP_STEPS``; the scale decays over training (curriculum) until
        the rule fades and the policy lifts on its own. Also enforces the
        **over-band lift cap**: while the object sits above the success band's
        upper edge (``LIFT_TARGET_M · (1 + SUCCESS_LIFT_TOL_FRAC)``), the wrist
        action's world +z component is stripped so nothing lifts it further
        (see ``_stage_wrist_lift_kernel``). Edits
        ``applier._gpu_action`` in place; reads the handler's live ep_step /
        object pose directly (so no staleness)."""
        # ── Wrist target-hold flag reset (every step) ──────────────────────
        # The stage kernel rewrites values only for lift/hold worlds. The
        # unconditional pre-reset means no stale hold survives in configurations
        # where the stage kernel is not launched (legacy lift_dist=0 etc.) or
        # right after a per-world reset.
        self.wrist_pos_hold_wp.zero_()
        self.wrist_rot_hold_wp.zero_()
        # ── Manipulation-precision wrist slow-down (proximity gated; stage 0 only) ──
        # Shrinks the wrist Δxyz near the object for a precise, slow approach/grasp.
        # The lift phase (ep_step >= LIFT_STEP) is skipped inside the kernel →
        # full speed when lifting. Runs before the lift kernel (the two are
        # mutually exclusive via the stage gate).
        if float(self.WRIST_SLOW_FACTOR) < 1.0:
            # When combined with the deadzone: the slow kernel handles the
            # pre-slow deadzone, and the apply kernel's post-slow deadzone is
            # lowered to dz·slow_factor to avoid double-cutting surviving commands
            # (base dz is snapshotted once — the original knob value is preserved).
            if bool(getattr(applier, "use_deadzone", False)):
                if not hasattr(self, "_xyz_deadzone_base"):
                    self._xyz_deadzone_base = float(applier.xyz_deadzone_m)
                applier.xyz_deadzone_m = (
                    self._xyz_deadzone_base * float(self.WRIST_SLOW_FACTOR))
            wp.launch(
                _wrist_translation_slow_kernel, dim=self.NWORLD,
                inputs=[
                    applier._gpu_action, self.ep_step_wp, self.d.xpos,
                    self.d.xmat, self._hand_center_offset_vec,
                    int(self.obj_body_id), int(self.wrist_body_id),
                    int(applier.xyz_off), int(self.LIFT_STEP),
                    float(self.WRIST_SLOW_NEAR_M), float(self.WRIST_SLOW_FAR_M),
                    float(self.WRIST_SLOW_FACTOR),
                    float(applier.xyz_scale),
                    int(1) if bool(getattr(applier, "use_deadzone", False)) else int(0),
                    float(getattr(self, "_xyz_deadzone_base",
                                  getattr(applier, "xyz_deadzone_m", 0.0))),
                ],
            )
        # ── JAX v2 stage mode: strip xyz + freeze rot + slow fingers in lift/hold.
        # Mutually exclusive with the original mode's manual-lift kernel (no decay, latch or over-band cap).
        if bool(self.STAGE_ACTION_JAX_MODE):
            wp.launch(
                _stage_action_jax_gate_kernel, dim=self.NWORLD,
                inputs=[
                    applier._gpu_action, self.ep_step_wp,
                    self.d.xmat, self.d.xpos, self.cond.get("obj_p_init"),
                    int(self.obj_body_id), float(self.LIFT_TARGET_M),
                    int(self.wrist_body_id),
                    int(self.LIFT_STEP), int(self.HOLD_STEP),
                    int(applier.xyz_off), int(applier.rot6d_off),
                    int(applier.finger_off), int(self.n_ctrl),
                    float(applier.xyz_scale),
                    float(self.LIFT_ACTION_DIST_M),
                    float(self.LIFT_RAMP_STEPS), float(self.LIFT_RAMP_RATE),
                    float(self.LIFT_MIN_DIST_M),
                    float(self.STAGE_FINGER_SLOW),
                    int(1) if bool(self.STAGE_TARGET_HOLD) else int(0),
                    int(1) if bool(self.LIFT_TARGET_INTEGRATE) else int(0),
                    float(self.LIFT_WRIST_RISE_CAP_M),
                    self.wrist_z_lift0_wp,
                    self.wrist_pos_hold_wp, self.wrist_rot_hold_wp,
                ],
            )
            return
        lift_dist = float(self.LIFT_ACTION_DIST_M) * float(self.lift_curriculum_scale)
        if lift_dist == 0.0:
            return
        wp.launch(
            _stage_wrist_lift_kernel, dim=self.NWORLD,
            inputs=[
                applier._gpu_action, self.ep_step_wp,
                self.d.xpos, self.d.xmat, self.cond.get("obj_p_init"),
                self.lift_reached_wp,
                int(self.obj_body_id), int(self.wrist_body_id),
                int(self.LIFT_STEP), int(self.HOLD_STEP),
                int(applier.xyz_off), int(applier.rot6d_off), float(applier.xyz_scale),
                float(self.LIFT_TARGET_M), lift_dist,
                float(self.LIFT_RAMP_STEPS), float(self.LIFT_RAMP_RATE),
                float(self.LIFT_MIN_DIST_M),
                # Over-band lift cap = success band upper edge. Derived from the
                # EXISTING knobs (LIFT_TARGET_M / SUCCESS_LIFT_TOL_FRAC) so every
                # subclass / config picks it up with no new yaml key; the kernel
                # treats <= 0 as disabled.
                float(self.LIFT_TARGET_M) * (1.0 + float(self.SUCCESS_LIFT_TOL_FRAC)),
                int(1) if bool(self.STAGE_TARGET_HOLD) else int(0),
                int(1) if bool(self.LIFT_TARGET_INTEGRATE) else int(0),
                self.wrist_pos_hold_wp, self.wrist_rot_hold_wp,
            ],
        )

    def _core_obs_terms(self):
        """Per-obs-term ``(name, start, end)`` column ranges — mirrors the launch
        order in :meth:`_collect_obs_kernel`. Consumed by the NaN debugger
        (``_debug_check_finite``) to name WHICH observation term went non-finite.
        Cached; the ``assert`` keeps it from silently drifting off the kernel."""
        cached = getattr(self, "_obs_term_layout_cache", None)
        if cached is not None:
            return cached
        nc  = int(self.n_ctrl)
        ns  = int(self._n_sensors)
        ncs = int(self._n_obj_contact)
        # ORDER MUST MATCH the launch order in ``_collect_obs_kernel`` (the
        # ``off`` running sum), NOT the obs_dim comment order — they differ.
        # (The total length is what obs_dim asserts; the per-term column ranges
        # here are what the NaN debugger uses to name a non-finite term, so the
        # ORDER must mirror the kernel.)
        terms = [
            ("site_pcd_err",        3 * (ns - 1)), # finger site → nearest obj-PCD vector
            ("hand_center_pcd_err", 3),            # hand-center → nearest obj-PCD vector (wrist frame)
            ("rot6d",               6),            # hand-center 6D rotation vs target_R (x,y axes; target_R half commented out in the kernel)
            ("site_z_above_table",  ns),           # site height above table
            ("obj_pose_diff",       6),            # obj xyz + rpy vs spawn
            ("joint_qpos",          nc),           # joint angles
            ("qpos_tax_err",        nc),           # target_qpos − qpos (taxonomy mimic err)
            ("finger_link_mask",    ns),           # taxonomy active/rest sensor mask
            ("per_slot_contact",    4 * ncs),      # obj/self/table contact flags+impulse
            ("obj_touch_force",     1),            # total normal force on object
            ("obj_ground_force",    1),            # approximate object-ground force (optional pre-lift gating)
            ("torque_proxy",        nc),           # ctrl − qpos
        ]
        # pose-FD velocity 12ch (JAX v2 channel layout; d.cvel not used — see
        # _fd_vel_update_kernel). Omitted by INCLUDE_FD_VEL_IN_OBS=False subclasses.
        if self.INCLUDE_FD_VEL_IN_OBS:
            terms.append(("fd_vel", 12))
        terms.append(("proj_gravity", 3))      # wrist-frame downward unit vector
        terms.append(("lift_target_vec", 3))   # obj→lift-target residual displacement (wrist frame, always on)
        # ep_step_norm + stage — only in the actor obs when enabled (time-blind
        # actors set INCLUDE_EP_STAGE_IN_OBS=False; see _collect_obs_kernel).
        if self.INCLUDE_EP_STAGE_IN_OBS:
            terms.append(("ep_stage", 4))          # ep_step_norm + stage one-hot(3)
        out, off = [], 0
        for nm, ln in terms:
            out.append((nm, off, off + ln)); off += ln
        assert off == self._core_obs_dim(), (
            f"obs_term_layout total {off} != core obs_dim {self._core_obs_dim()} "
            f"(drifted from _collect_core_obs_blocks)")
        self._obs_term_layout_cache = out
        return out

    def _collect_core_obs_blocks(self) -> None:
        # Run the contact kernels FIRST — obs reads ``hand_obj_active_wp`` and
        # the reward bonus gate also reads it. Doing it here means we
        # don't have to re-launch in ``_collect_reward_done_kernel``.
        self._launch_contact_active_kernels()   # fills the per-slot impulse buffers (obj/self/table)
        # (c) per-taxonomy diagnostics: accumulate the episode-mean contact_ratio
        # (previous-step value, the 1-step lag is harmless; _harvest_tax_stats
        # collects and clears it at reset).
        if getattr(self, "_ep_ratio_sum_t", None) is not None:
            self._ep_ratio_sum_t += self.raw_contact_ratio_torch
            self._ep_len_t += 1.0
        # Obs is built block-by-block: one kernel per component, each writing
        # its slice at a host-accumulated ``off``. REUSES the reward's per-slot
        # impulse buffers / best_pt (no recompute). The single ``off`` running
        # sum below is the ONLY place the slot layout lives — it must end at
        # exactly ``obs_dim`` (asserted), which structurally prevents the
        # blocks from clobbering one another.
        NW   = self.NWORLD
        obs  = self.obs_wp
        qpa  = self.applier._ctrl_qpa
        nc   = int(self.n_ctrl)
        ns   = int(self._n_sensors)
        ncs  = int(self._n_obj_contact)
        sites = self._finger_site_ids_wp
        th   = float(getattr(self.sampled_env, "table_height", 0.0) or 0.0)
        arm  = int(self.hand_util.fore_arm_sensor_idx)
        off  = 0
        ###### Geometry-related obs
        # BPS observation

        ######
        wp.launch(_obs_site_pcd_err_kernel, dim=NW,
                  inputs=[self.d.xmat, self.d.site_xpos, self.best_pt_wp, sites,
                          self.wrist_body_id, ns, arm, off, obs]);                    off += 3 * (ns - 1)
        wp.launch(_obs_hand_center_pcd_err_kernel, dim=NW,
                  inputs=[self.d.xpos, self.d.xmat, self.best_pt_hc_wp,
                          self.wrist_body_id, self._hand_center_offset_vec, off, obs]); off += 3
        wp.launch(_obs_rot6d_kernel, dim=NW,
                  inputs=[self.d.xmat, self.cond.get("target_R"), self._hand_center_R_mat,
                          self.wrist_body_id, off, obs]);                             off += 6
        wp.launch(_obs_site_z_above_table_kernel, dim=NW,
                  inputs=[self.d.site_xpos, sites, ns, th, off, obs]);                off += ns
        wp.launch(_obs_obj_pose_diff_kernel, dim=NW,
                  inputs=[self.d.xpos, self.d.xmat, self.cond.get("obj_p_init"),
                          self.cond.get("obj_R_init"), self.obj_body_id, off, obs]);  off += 6

        ## Finger-related obs
        wp.launch(_obs_joint_qpos_kernel, dim=NW,
                  inputs=[self.d.qpos, qpa, nc, off, obs]);                           off += nc
        wp.launch(_obs_qpos_tax_err_kernel, dim=NW,
                  inputs=[self.d.qpos, qpa, self.cond.get("target_qpos"), nc,
                          float(self.OBS_TAX_ERR_DEADZONE), off, obs]);              off += nc
        wp.launch(_obs_finger_link_mask_kernel, dim=NW,
                  inputs=[self.cond.get("specific_finger_link_mask"), ns, off, obs]); off += ns

        ## Contact-related obs — force channels are log-compressed to [0,1] via OBS_FORCE_LOG1P.
        _cap = float(self.OBS_FORCE_LOG1P_CAP) if self.OBS_FORCE_LOG1P else 0.0
        _inv = float(1.0 / np.log1p(_cap)) if _cap > 0.0 else 1.0
        wp.launch(_obs_per_slot_contact_kernel, dim=NW,
                  inputs=[self.hand_obj_contact_impulse_per_slot_wp,
                          self.self_impulse_per_slot_wp, self.tbl_impulse_per_slot_wp,
                          ncs, _cap, _inv, off, obs]);                                off += 4 * ncs
        wp.launch(_obs_obj_touch_force_kernel, dim=NW,
                  inputs=[self.obj_touch_force_wp, _cap, _inv, off, obs]);            off += 1
        # Approximate obj-ground (floor) force — optional pre-lift masking as in JAX v2.
        wp.launch(_obs_obj_ground_force_kernel, dim=NW,
                  inputs=[self.obj_ground_force_wp, self.ep_step_wp,
                          int(self.LIFT_STEP) if self.OBS_OBJ_GROUND_PRELIFT_GATED else int(-1),
                          _cap, _inv, off, obs]);                                     off += 1

        # Torque proxy (OBS_TORQUE_PROXY_LOG1P → signed log1p compression)
        wp.launch(_obs_torque_proxy_kernel, dim=NW,
                  inputs=[self.d.qpos, qpa, self.d.ctrl, nc,
                          int(1) if self.OBS_TORQUE_PROXY_LOG1P else int(0),
                          off, obs]);                                                 off += nc
        # Velocity obs — pose-FD 12ch (JAX v2 channels; d.cvel not used). The
        # values are produced by ``_fd_vel_update_kernel`` (reward collect, once
        # per step) and only copied here, so re-collecting the obs (reset path)
        # is idempotent. (The old cvel-based ``_obs_wrist_obj_vel_kernel`` is
        # kept because the partial_bps critic still uses it — the actor obs uses the FD version.)
        if self.INCLUDE_FD_VEL_IN_OBS:
            wp.launch(_obs_fd_vel_kernel, dim=NW,
                      inputs=[self.fd_vel_wp, off, obs]);                             off += 12
        # Projected gravity (wrist-frame downward unit vector; absolute orientation cue)
        wp.launch(_obs_projected_gravity_kernel, dim=NW,
                  inputs=[self.d.xmat, self.wrist_body_id, off, obs]);                off += 3
        # Always-on lift-target channel (fixed per episode, always non-zero)
        wp.launch(_obs_lift_target_kernel, dim=NW,
                  inputs=[self.d.xpos, self.d.xmat, self.cond.get("obj_p_init"),
                          self.obj_body_id, self.wrist_body_id,
                          float(self.LIFT_TARGET_M), off, obs]);                      off += 3

        # Episode-stage obs
        # stage_wp is ALWAYS computed (the rule-based wrist-lift reads it). The
        # 2-col ep_step_norm + stage block is written into the actor obs only when
        # INCLUDE_EP_STAGE_IN_OBS is True; a time-blind actor (False) computes
        # stage_wp via the stage-only kernel and contributes 0 obs columns.
        if self.INCLUDE_EP_STAGE_IN_OBS:
            wp.launch(_obs_ep_stage_kernel, dim=NW,
                      inputs=[self.ep_step_wp, int(self.LIFT_STEP), int(self.HOLD_STEP),
                              int(self.MAX_EPISODE_STEPS), off, obs, self.stage_wp]);  off += 4
        else:
            wp.launch(_stage_only_kernel, dim=NW,
                      inputs=[self.ep_step_wp, int(self.LIFT_STEP), int(self.HOLD_STEP),
                              self.stage_wp])

        assert off == self._core_obs_dim(), f"obs layout {off} != core obs_dim {self._core_obs_dim()}"

    def _launch_pcd_scans(self) -> None:
        """The 3 PCD-scan launches — the obs-shared prefix of the reward pipeline.

        Kernels 1–3 of ``_collect_reward_done_kernel`` moved verbatim (pure
        extraction; behaviour unchanged). The obs blocks (``site_pcd_err`` /
        ``hand_center_pcd_err``) and the BPS tail reuse ``transformed_pcd_wp`` /
        ``best_pt*``, so a consumer that wants to refresh the obs without running
        the reward (e.g. the frozen leader driver of LF object interaction) calls
        this method, then ``_launch_obj_touch_sum``, then ``_collect_obs_kernel``.
        """
        # Transform the per-world assigned object PCD body-local → world
        # ONCE per point (2D launch over NWORLD × n_pts). The world-frame PCD
        # ``transformed_pcd`` feeds the nearest-PCD scan below AND is exposed
        # to the obs kernel — so the rigid transform is no longer re-run per
        # sensor inside the scan.
        wp.launch(
            _object_grasping_transform_pcd_kernel,
            dim=(self.NWORLD, int(self._pcd_n_pts)),
            inputs=[
                self.d.xpos,
                self.d.xmat,
                self._pcd_gpu,
                self._pcd_assignment,
                int(self._grasp_obj_slot),
                int(self._grasp_obj_body_id),
                self.transformed_pcd_wp, # OK
            ],
        )

        # 1b. Hand-center → nearest object-PCD vertex (1 point per world), over
        # the same world-frame ``transformed_pcd``. Feeds the
        # ``hand_center_pcd_err`` obs block (wrist-frame vector from the
        # hand-center to its nearest object surface point). 1D launch (no sensor
        # dim) — ~1/n_sensors the cost of the per-site scan below.
        wp.launch(
            _object_grasping_nearest_pcd_hand_center_kernel,
            dim=self.NWORLD,
            inputs=[
                self.d.xpos,
                self.d.xmat,
                self.transformed_pcd_wp,
                int(self.wrist_body_id),
                int(self._grasp_obj_body_id),
                self._hand_center_offset_vec,
                int(self._pcd_n_pts),
                self.best_dist_hc_wp,
                self.best_pt_hc_wp,
            ],
        )

        # 2. Shared nearest-PCD scan (the ONE expensive brute-force loop):
        # per (world, sensor) min squared distance + nearest vertex from the
        # finger SITE, over the world-frame ``transformed_pcd``. Both the
        # finger-close and face-dir reductions below — and the obs kernel —
        # read ``best_d2`` / ``best_pt`` instead of each re-running the search.
        wp.launch(
            _object_grasping_nearest_pcd_kernel,
            dim=(self.NWORLD, int(self._n_sensors)),   # 2D: (world, sensor) parallel
            inputs=[
                self.d.site_xpos,
                self.d.xpos,
                self.transformed_pcd_wp,
                self._finger_site_ids_wp,
                int(self._grasp_obj_body_id),
                int(self._n_sensors),
                int(self._pcd_n_pts),
                self.best_d2_wp, # OK
                self.best_pt_wp, # OK
            ],
        )

    def _launch_obj_touch_sum(self) -> None:
        """Single launch summing the object touch sensors — updates ``obj_touch_force_wp``.

        Extracted from reward kernel 11 (behaviour unchanged). The
        ``_obs_obj_touch_force_kernel`` obs block reads this buffer, so obs-only
        consumers must call this method too for a complete obs."""
        # Force applied to the dynamic object — Σ of the object's touch
        # sensor readings (reuses the generic touch-sum kernel). Penalised via
        # W_OBJ_TOUCH to learn gentle grasps. Self-handles n == 0 (writes 0).
        wp.launch(
            _touch_sum_kernel, dim=self.NWORLD,
            inputs=[
                self.d.sensordata,
                int(self._n_obj_touch),
                float(self._force_scale_for_state),
                self._obj_touch_adr_wp,
                self.obj_touch_force_wp, # OK
            ],
        )

    def _launch_fd_vel_update(self) -> None:
        """Update the pose-FD velocity buffers — the 12 ``fd_vel`` obs slots and ``fd_obj_vz``.

        Must be called exactly once per step, **right after the physics
        advance** (it compares the previous snapshot with the current
        xpos/xmat). Not called on the reset path.

        Normally :meth:`_collect_reward_done_kernel` is the only caller, but a
        path that drives the teacher **for policy only** (the LF scene's
        `LeaderGraspingDriver`) does not go through reward/done → it must call
        this method every step as well. Otherwise fd_vel stays 0 forever and
        the teacher never sees velocity.
        """
        wp.launch(
            _fd_vel_update_kernel, dim=self.NWORLD,
            inputs=[
                self.d.xpos, self.d.xmat,
                int(self.wrist_body_id), int(self.obj_body_id),
                float(self._fd_inv_dt), float(self.FD_VEL_CLIP),
                self.fd_valid_wp,
                self.prev_wrist_p_wp, self.prev_wrist_R_wp,
                self.prev_obj_p_wp, self.prev_obj_R_wp,
                self.fd_vel_wp,
                self.fd_obj_vz_wp,
            ],
        )

    def _collect_reward_done_kernel(self) -> None:
        """Per-step reward + done dispatcher (compute → mix split, mirroring
        the tracking task). Every per-term kernel is UNWEIGHTED — the mixer
        (last launch) is the sole place weights / curriculum / dt-scale / clip
        are applied. Launch order (all 1D over NWORLD unless noted):

          1.  transform_pcd                 obj PCD body-local → world  (2D: NWORLD × n_pts)
          2.  nearest_pcd_hand_center       hand-center → nearest PCD vertex (best_pt_hc)
          3.  nearest_pcd (per-sensor)      per-site nearest PCD scan  (2D: NWORLD × n_sensors)
          4.  finger_close                  → raw_finger      (RBF over best_d2)
          5.  hand_center_close             → raw_approach    (RBF over best_dist_hc; LIVE)
          6.  face_dir                      → raw_face_dir    (active face ↔ object cos)
          7.  contact_dir                   → contact_dir_cos (per-sensor face ↔ contact-normal)
          8.  contact_reward                → raw_contact_dir/pos/neg + affordance_pos/neg
          9.  self_coll_weighted_penalty    → raw_self_contact_w / raw_self_impulse_w
          10. obstacle_coll_weighted_penalty→ raw_table_contact_w / raw_table_impulse_w
          11. touch_sum                     → obj_touch_force  (Σ object touch sensors)
          12. qpos_mimic                    → raw_mimic + mimic_qpos_err  (split L1 hinge)
          13. wrist_dir                     → raw_wrist_dir    (hand-center x ↔ target_R[:,0])
          14. wrist_vel                     → raw_vel          (wrist stationarity coeff)
          15. obj_vel                       → raw_obj_vel      (object stationarity coeff)
          16. obj_pose_coeff                → raw_obj_xy_coeff / raw_obj_R_coeff (xy-drift + upright)
          17. action_sqnorm (shared)        → ctrl-cost driver
          18. action_smooth_sqnorm (shared) → smoothness driver (updates last_action)
          19. lift                          → raw_lift         (obj_z normalised to [0,1])
          20. done detection                (ep_step ++ here)
          21. bonus gate                    (lift ∧ obj_active → bonus_active)
          22. mixer                         (weights + curriculum + dt-scale + clip → reward)

        ``raw_approach`` is LIVE (kernel 5) — it is no longer disabled. The
        self/table/obj per-slot contact kernels run at the top of
        ``_collect_obs_kernel`` (so obs and the bonus gate share the same
        per-step contact flags — no double-launch); kernels 8–11 here read
        those already-filled found/touch/impulse buffers. Success promotion is
        a SEPARATE launch in ``_collect_success_kernel``.
        """
        # 0. Pose-FD velocity update (for obs; exactly once per step).
        self._launch_fd_vel_update()

        # 1–3. Shared PCD scans (transform → hand-center nearest → per-site
        # nearest). Extracted so an obs-only consumer (leader-follower object
        # interaction's frozen leader driver) can refresh ``transformed_pcd`` /
        # ``best_pt*`` without running the full reward pipeline — single source
        # of truth, no behaviour change here.
        self._launch_pcd_scans()

        # 1b. JAX v2 style dynamic target_pnt — overwrite the cond every step with
        # the freshly scanned ``best_pt_hc`` (obj-PCD vertex nearest the
        # hand-center) + jitter. (The static sample from reset is replaced on the first step.)
        if bool(self.TARGET_PNT_DYNAMIC):
            self._target_pnt_seed = (self._target_pnt_seed + 1) & 0x7FFFFFFF
            wp.launch(
                _object_grasping_dynamic_target_pnt_kernel, dim=self.NWORLD,
                inputs=[
                    self.best_pt_hc_wp,
                    float(self.TARGET_PNT_DYNAMIC_NOISE_M),
                    int(self._target_pnt_seed),
                    self.cond.get("target_pnt"),
                ],
            )

        # 2a. Finger-close err + RBF (UNWEIGHTED) — REDUCTION over best_d2.
        # ``link_dist = Σ best_d2 · w · mask / Σ mask`` — legacy
        # ``tip_link_process`` semantics from grit_with_obj_coll_v2.
        wp.launch(
            _object_grasping_finger_close_kernel, dim=self.NWORLD,
            inputs=[
                self.best_d2_wp,
                self._finger_weights_wp,
                self.cond.get("specific_finger_link_mask"),
                int(self._n_sensors),
                float(self.K_FINGER),
                self.finger_err_wp, # OK
                self.raw_finger_wp, # OK
            ],
        )

        # 2a'. Hand-center close RBF (UNWEIGHTED) — direct exp over the scalar
        # ``best_dist_hc`` (from the hand-center nearest-PCD kernel above).
        wp.launch(
            _object_grasping_hand_center_close_kernel, dim=self.NWORLD,
            inputs=[
                self.best_dist_hc_wp,
                float(self.K_APPROACH),
                self.raw_approach_wp,
            ],
        )

        # 2b. Face-direction alignment (UNWEIGHTED) — REDUCTION over best_pt.
        # Active palm/finger body face direction ↔ direction from the finger
        # site toward its nearest object-PCD vertex (``best_pt``). Reuses
        # ``specific_finger_link_mask`` + ``finger_weights`` (active set /
        # weighting) and the ``face_dir_idx_in_mat`` / ``face_dir_sign`` cond
        # fields (taxonomy face columns). ``raw_face_dir ∈ [0, 1]`` — 1 when
        # every active face points straight at the object. Legacy
        # ``hand_to_obj_cos_sim``.
        wp.launch(
            _object_grasping_face_dir_kernel, dim=self.NWORLD,
            inputs=[
                self.d.site_xpos,
                self.d.xmat,
                self.best_pt_wp,
                self._hand_sensor_body_ids_wp,
                self._finger_site_ids_wp,
                self._finger_weights_wp,
                self.cond.get("specific_finger_link_mask"),
                self.cond.get("face_dir_idx_in_mat"),
                self.cond.get("face_dir_sign"),
                int(self._n_sensors),
                self.raw_face_dir_wp, # OK 
            ],
        )


        # 2c. Per-sensor contact-direction alignment: each contacting finger's
        # face dir ↔ its object-contact normal (legacy ``frames_r_sim``). Writes
        # a (NWORLD, n_sensors) cosine array for downstream contact reward terms.
        wp.launch(
            _object_grasping_contact_dir_kernel, dim=self.NWORLD,
            inputs=[
                self.d.sensordata,
                self.d.xmat,
                int(self._n_obj_contact),
                self._hand_obj_contact_found_wp,
                self._hand_obj_contact_touch_wp,
                self._hand_obj_contact_sensor_idx_wp,
                self._hand_sensor_body_ids_wp,
                self.cond.get("face_dir_idx_in_mat"),
                self.cond.get("face_dir_sign"),
                int(self._n_sensors),
                float(self.TOUCH_EPS),
                int(1) if self.GATE_CONTACT_BY_TOUCH else int(0),
                self.contact_dir_cos_wp, # OK 
            ],
        )

        # 2d. Taxonomy-aware contact-affordance reward components — folds the
        # per-sensor contact cosine + obj-contact found / impulse with the
        # taxonomy active/rest split (specific_finger_link_mask) and per-slot
        # contact weighting (_hand_only_contact_weight_wp) into 5 components
        # (legacy contact block). Exposed as raw buffers (not mixer terms yet).
        wp.launch(
            _object_grasping_contact_reward_kernel, dim=self.NWORLD,
            inputs=[
                self.d.sensordata,
                self.contact_dir_cos_wp,
                self.cond.get("specific_finger_link_mask"),
                int(self._n_obj_contact),
                self._hand_obj_contact_found_wp,
                self._hand_obj_contact_touch_wp,
                self._hand_obj_contact_sensor_idx_wp,
                self._hand_only_contact_weight_wp,
                self._slot_finger_id_wp,
                int(self._n_finger_groups),
                float(self.TOUCH_EPS),
                int(1) if self.GATE_CONTACT_BY_TOUCH else int(0),
                float(self.CONTACT_NEG_SCALE),
                float(self._force_scale_for_reward),
                int(1) if self.CONTACT_POS_SUM else int(0),
                int(1) if self.CONTACT_POS_BINARY_COS else int(0),
                self.raw_contact_dir_reward_wp, # OK
                self.raw_contact_pos_wp, # OK
                self.raw_contact_neg_wp, # OK
                self.raw_affordance_pos_wp, # OK
                self.raw_affordance_neg_wp, # OK
                self.raw_contact_ratio_wp,   # (#active fingers touched)/(#active fingers)
                # sem_contact_* (slot-level taxonomy compliance metrics; with
                # W_SEM_*=0 they carry no reward but are always computed as
                # metrics — for comparison with the eval metrics).
                self.raw_sem_recall_wp,
                self.raw_sem_precision_wp,
                self.raw_sem_f1_wp,
                self.raw_sem_iou_wp,
            ],
        )

        # 4b. Force-closure wrench residual (port of JAX v2 force_closure_reward) —
        # always computed (for metrics; no reward when W_FORCE_CLOSURE=0 in the mixer).
        wp.launch(
            _object_grasping_force_closure_kernel, dim=self.NWORLD,
            inputs=[
                self.d.sensordata, self.d.xpos,
                int(self._n_obj_contact),
                self._hand_obj_contact_found_wp, self._hand_obj_contact_touch_wp,
                int(self.obj_body_id),
                float(self.TOUCH_EPS),
                int(1) if self.GATE_CONTACT_BY_TOUCH else int(0),
                float(self._force_scale_for_reward),
                float(self.K_FORCE_CLOSURE),
                self.force_closure_res_wp,
                self.raw_force_closure_wp,
            ],
        )

        # finger_contact_weights-weighted self-collision penalty drivers
        wp.launch(
            _self_coll_weighted_penalty_kernel, dim=self.NWORLD,
            inputs=[
                self.d.sensordata,
                int(self._n_self_coll),
                self._self_coll_found_wp, self._self_coll_touch_wp,
                self._hand_only_contact_weight_wp,
                float(self.TOUCH_EPS),
                int(1) if self.GATE_CONTACT_BY_TOUCH else int(0),
                float(self._force_scale_for_reward),
                self.raw_self_contact_w_wp, # OK 
                self.raw_self_impulse_w_wp, # OK 
            ],
        )

        # finger_contact_weights-weighted obstacle penalty drivers =
        # hand↔table contact + fore-arm touch, summed into contact / impulse.
        wp.launch(
            _obstacle_coll_weighted_penalty_kernel, dim=self.NWORLD,
            inputs=[
                self.d.sensordata,
                int(self._n_table_contact),
                self._table_contact_found_wp, self._table_contact_touch_wp,
                self._hand_only_contact_weight_wp,
                int(self._n_forearm_touch),
                self._forearm_touch_adr_wp, self._forearm_touch_weight_wp,
                float(self._force_scale_for_reward),
                float(self.TOUCH_EPS),
                int(1) if self.GATE_CONTACT_BY_TOUCH else int(0),
                self.raw_table_contact_w_wp, # OK 
                self.raw_table_impulse_w_wp, # OK 
            ],
        )
        # 10b. State-conditional table gate — discount the table penalty while a
        # grasp is established (by ratio). raw_contact_ratio is filled by the
        # contact kernel above in the same step.
        if float(self.TABLE_PEN_OBJ_CONTACT_SCALE) < 1.0:
            wp.launch(
                _table_pen_grasp_gate_kernel, dim=self.NWORLD,
                inputs=[
                    self.raw_contact_ratio_wp,
                    float(self.TABLE_PEN_OBJ_CONTACT_MIN_RATIO),
                    float(self.TABLE_PEN_OBJ_CONTACT_SCALE),
                    self.raw_table_contact_w_wp,
                    self.raw_table_impulse_w_wp,
                ],
            )

        # 11. Force applied to the dynamic object (extracted helper — shared
        # with the obs-only leader-driver path).
        self._launch_obj_touch_sum()

        # 11b. Rest-crush / obj-ground force signals — consumed by done (reason
        # 11/12) + obs channels. obj_touch_force was just updated, so the residual
        # is consistent with the same step's physics (reads sensordata directly —
        # avoids the 1-step lag of the per-slot buffers).
        wp.launch(
            _object_grasping_obj_contact_aux_kernel, dim=self.NWORLD,
            inputs=[
                self.d.sensordata,
                int(self._n_obj_contact),
                self._hand_obj_contact_found_wp, self._hand_obj_contact_touch_wp,
                self._hand_obj_contact_sensor_idx_wp,
                self.cond.get("specific_finger_link_mask"),
                self.obj_touch_force_wp,
                float(self.TOUCH_EPS),
                int(1) if self.GATE_CONTACT_BY_TOUCH else int(0),
                float(self._force_scale_for_state),
                self.rest_obj_impulse_wp,
                self.obj_ground_force_wp,
            ],
        )

        # 2b. Intra-finger motor torque balance coefficient (torque_regulate) —
        # groups d.actuator_force per finger and computes a participation balance
        # ∈(0,1]. Always computed (cheap, and observed as a metric) — the mixer
        # sets the strength via c=TORQUE_BALANCE_COUPLING (0 = disabled).
        wp.launch(
            _torque_balance_kernel, dim=self.NWORLD,
            inputs=[
                self.d.actuator_force,
                self._ctrl_finger_group_id_wp,
                int(self._n_ctrl_for_torque),
                int(self._n_torque_groups),
                float(self.TORQUE_BALANCE_EPS),
                self.torque_balance_coeff_wp,
            ],
        )

        # 3. Qpos taxonomy mimic (UNWEIGHTED).
        # NEW: split L1 hinge with per-branch dead-zones via
        # ``specific_ctrl_mask`` (set per world by the taxonomy hook).
        # Active fingers get ``MIMIC_ACTIVE_DEADZONE`` tolerance; rest
        # fingers get the tighter ``MIMIC_REST_DEADZONE``.
        wp.launch(
            _object_grasping_qpos_mimic_kernel, dim=self.NWORLD,
            inputs=[
                self.d.qpos,
                self.d.ctrl,                      # commanded setpoint (for MIMIC_ON_CTRL; post-clamp = JAX hand_action)
                self.applier._ctrl_qpa,
                self.cond.get("target_qpos"),
                self.cond.get("specific_ctrl_mask"),
                self.n_ctrl,
                int(1) if self.MIMIC_ON_CTRL else int(0),
                float(self.K_MIMIC),
                float(self.MIMIC_ACTIVE_DEADZONE),
                float(self.MIMIC_REST_DEADZONE),
                self.mimic_qpos_err_wp, # OK
                self.raw_mimic_wp, # OK
                self.mimic_naive_err_wp,
            ],
        )

        # 4. Wrist-dir cosine alignment (UNWEIGHTED, clipped ≥ 0).
        # NEW: aligns the **hand_center** x-axis (composed via R_wrist · R_hc_local)
        # against the frozen ``target_R[:, 0]`` (= legacy ``target_dir``),
        # not the bare wrist body x-axis. Matches the legacy
        # ``hand_R = (wrist_T @ rh_hand_center)[:3, :3]`` frame.
        wp.launch(
            _object_grasping_wrist_dir_kernel, dim=self.NWORLD,
            inputs=[
                self.d.xmat,
                self.cond.get("target_R"),
                int(self.wrist_body_id),
                self._hand_center_R_mat,
                float(self.K_WRIST_DIR),
                self.raw_wrist_dir_wp, # OK  
            ],
        )

        # 5. Wrist velocity stationarity coefficient (UNWEIGHTED).
        # raw_vel ∈ (0, 1] — peaks at 1 when the wrist is still; decays
        # as wrist linear/angular velocity grows. The mixer applies W_VEL
        # to convert this into an additive reward term (legacy was
        # multiplicative ``wrist_vel_coeff`` on hand_process). Reads
        # ``d.qvel[w, _wrist_qva : +6]`` — the wrist freejoint's 6 DOFs.
        wp.launch(
            _object_grasping_wrist_vel_kernel, dim=self.NWORLD,
            inputs=[
                self.fd_vel_wp,                # pose-FD (jitter-free step-averaged velocity)
                float(self.VEL_ANG_WEIGHT),
                float(self.K_VEL),
                self.total_vel_wp, # OK
                self.raw_vel_wp, # OK
            ],
        )

        # 5b. Object velocity stationarity coefficient (UNWEIGHTED).
        # raw_obj_vel ∈ (0, 1] — peaks at 1 when the OBJECT is still; decays
        # as it is pushed / spun. World-frame ``d.cvel[w, obj_body_id]`` is
        # rotated into the wrist frame, clipped per-component to [-1, 1],
        # then ``exp(-K_OBJ_VEL · (Σvlin² + OBJ_VEL_ANG_WEIGHT·Σvang²))``.
        # Legacy ``obj_vel_coeff`` (sibling of the wrist-vel term).
        wp.launch(
            _object_grasping_obj_vel_kernel, dim=self.NWORLD,
            inputs=[
                self.fd_vel_wp,                # pose-FD, wrist-relative (original JAX semantics)
                float(self.OBJ_VEL_ANG_WEIGHT),
                float(self.K_OBJ_VEL),
                self.total_obj_vel_wp, # OK
                self.raw_obj_vel_wp, # OK
            ],
        )

        # Object pose-vs-spawn coefficients (UNWEIGHTED): exp-form xy-drift
        # + uprightness coefficients relative to the reset spawn pose
        # (legacy ``obj_xy_coeff`` / ``obj_R_coeff``). Sole owner of the object
        # xy-drift reward (the lift kernel only does z now). ``raw_obj_R_coeff``
        # rewards the object staying upright (obj_R[2,2]). Both ∈ (0, 1], mixer
        # applies the weights.
        wp.launch(
            _grasping_obj_pose_coeff_kernel, dim=self.NWORLD,
            inputs=[
                self.d.xpos,
                self.d.xmat,
                self.cond.get("obj_p_init"),
                int(self.obj_body_id),
                float(self.OBJ_DIST_PENALTY_COEFF),
                float(self.OBJ_R_CLIP_MIN),
                self.raw_obj_xy_coeff_wp, # OK 
                self.raw_obj_R_coeff_wp, # OK 
            ],
        )

        # Action sqnorm (raw ctrl-cost driver — shared kernel).
        wp.launch(
            _action_sqnorm_kernel, dim=self.NWORLD,
            inputs=[
                self.applier._gpu_action,
                int(self._action_dim),
                self.action_sqnorm_wp, # OK 
            ],
        )

        # Action-smoothness (||a_t - a_{t-1}||², updates last_action).
        wp.launch(
            _action_smooth_sqnorm_kernel, dim=self.NWORLD,
            inputs=[
                self.applier._gpu_action,
                self.last_action_wp,
                int(self._action_dim),
                self.action_smooth_sqnorm_wp, # OK 
            ],
        )

        # Lift z (UNWEIGHTED, normalised to [0, 1]). xy/R coeffs are computed
        # by the obj-pose-coeff kernel — the single owner of object-pose reward.
        wp.launch(
            _object_grasping_lift_kernel, dim=self.NWORLD,
            inputs=[
                self.d.xpos,
                self.cond.get("obj_p_init"),
                self.obj_body_id,
                float(self.LIFT_TARGET_M),
                self.lift_z_wp,
                self.raw_lift_wp,
            ],
        )

        # Done detection + ep_step ++.
        wp.launch(
            _object_grasping_done_kernel, dim=self.NWORLD,
            inputs=[
                self.d.xpos, self.d.xmat,
                self.fd_obj_vz_wp,
                self.cond.get("obj_p_init"),
                self.hand_obj_contact_impulse_wp,
                self.raw_table_impulse_w_wp,
                self.rest_obj_impulse_wp,
                self.obj_ground_force_wp,
                self.best_dist_hc_wp,
                self.obj_body_id, self.wrist_body_id,
                self._hand_center_offset_vec,
                float(self.DONE_OBJ_Z_MIN),
                float(self.DONE_OBJ_XY_DRIFT),
                float(self.DONE_WRIST_X_Z),
                float(self.DONE_OBJ_TOUCH),
                float(self.DONE_OBJ_TOUCH_REST),
                float(self.DONE_TABLE_IMPULSE),
                float(self.DONE_OBJ_GROUND_IMPULSE),
                float(self.DONE_OBJ_VEL_Z),
                float(self.DONE_HAND_OBJ_DIST),
                float(self.DONE_HOLD_OBJ_DIST),
                int(self.LIFT_STEP),
                int(self.HOLD_STEP),
                int(self.MAX_EPISODE_STEPS),
                int(1) if self.timeout_only_done else int(0),
                self.ep_step_wp,
                self.done_wp, self.done_mask_wp, self.done_reason_wp, # OK
            ],
        )

        # ⑧ Bonus gate (lift ∧ obj_active — see kernel docstring).
        wp.launch(
            _object_grasping_bonus_gate_kernel, dim=self.NWORLD,
            inputs=[
                self.raw_lift_wp,
                self.hand_obj_active_wp,
                self.bonus_active_wp,
            ],
        )

        # ⑨b Pre-mix hook (subclass extension point; no-op by default). After all
        #     raw terms are filled and **right before** the mixer reads them, raw
        #     buffers may be overwritten per world (e.g. neutralising the
        #     xy-drift/upright coefficients to 1 during an in-hand reorient phase).
        self._pre_mix_hook()

        # ⑩ Mixer: weights × raw + bonus + ctrl pen + smooth pen + vel coeff + curriculum + dt-scale + clip.
        wp.launch(
            _object_grasping_reward_mix_kernel, dim=self.NWORLD,
            inputs=[
                self.raw_approach_wp, self.raw_finger_wp, self.mimic_qpos_err_wp,
                self.mimic_naive_err_wp, self.raw_mimic_wp,
                self.raw_wrist_dir_wp, self.raw_lift_wp,
                self.raw_vel_wp,
                self.raw_obj_vel_wp,
                self.raw_obj_xy_coeff_wp,
                self.raw_obj_R_coeff_wp,
                self.raw_face_dir_wp,
                self.raw_contact_dir_reward_wp,
                self.raw_contact_ratio_wp,
                self.torque_balance_coeff_wp,
                self.raw_contact_pos_wp,
                self.raw_contact_neg_wp,
                self.raw_affordance_pos_wp,
                self.raw_affordance_neg_wp,
                self.raw_force_closure_wp,
                self.raw_sem_recall_wp, self.raw_sem_precision_wp,
                self.raw_sem_f1_wp, self.raw_sem_iou_wp,
                self.action_sqnorm_wp, self.action_smooth_sqnorm_wp,
                self.raw_self_contact_w_wp, self.raw_self_impulse_w_wp,
                self.raw_table_contact_w_wp, self.raw_table_impulse_w_wp,
                self.obj_touch_force_wp,
                self.obj_weight_force_wp,
                self.obj_ground_force_wp,
                self.bonus_active_wp, self.bonus_streak_wp,
                self.ep_step_wp,
                self.done_reason_wp,
                int(self.LIFT_STEP),
                float(self.W_APPROACH), float(self.W_FINGER), float(self.W_MIMIC),
                float(self.W_MIMIC_NAIVE),
                float(self.W_WRIST_DIR), float(self.W_LIFT),
                float(self.W_SELF_COLL), float(self.W_SELF_IMPULSE),
                float(self.W_TABLE_CONTACT), float(self.W_TABLE_IMPULSE),
                float(self.W_OBJ_TOUCH),
                float(self.TOUCH_PEN_LIFT_SCALE),
                float(self.OBJ_TOUCH_FREE_WEIGHT_K),
                float(self.OBJ_TOUCH_FREE_ALPHA),
                float(self.W_OBJ_TABLE_FORCE),
                float(self.OBJ_TABLE_FREE_K),
                float(self.OBJ_TABLE_FREE_ALPHA),
                float(self.CTRL_COST), float(self.K_ACTION_SMOOTH),
                float(self.W_VEL),
                float(self.W_OBJ_VEL),
                float(self.W_OBJ_XY_COEFF),
                float(self.W_OBJ_R_COEFF),
                float(self.W_FACE_DIR),
                float(self.W_CONTACT_DIR),
                float(self.W_CONTACT_RATIO),
                float(self.CONTACT_RATIO_LIFT_BOOST),
                float(self.TORQUE_BALANCE_COUPLING),
                float(self.W_CONTACT_POS),
                float(self.W_CONTACT_NEG),
                float(self.W_AFFORDANCE_POS),
                float(self.W_AFFORDANCE_NEG),
                float(self.W_FORCE_CLOSURE),
                float(self.W_SEM_RECALL), float(self.W_SEM_PRECISION),
                float(self.W_SEM_F1), float(self.W_SEM_IOU),
                float(self.W_BONUS),
                float(self.GRASP_BONUS_VALUE),
                int(self.SUCCESS_STREAK_MIN),
                float(self.COVERAGE_BONUS_COUPLING),
                float(self.COVERAGE_BONUS_EXP),
                float(self.W_LIFT_SUCCESS_BONUS),
                float(self.LIFT_SUCCESS_THRESH / self.LIFT_TARGET_M) if self.LIFT_TARGET_M > 0.0 else 1.0,
                float(self.W_VIOLATION_PEN), int(self.TIMEOUT_REASON_CODE), int(self.SUCCESS_REASON_CODE),
                float(self.REWARD_DT), float(self.MIN_REWARD), float(self.MAX_REWARD),
                self.r_approach_wp, self.r_finger_wp, self.r_mimic_wp,
                self.r_wrist_dir_wp, self.r_lift_wp,
                self.r_self_pen_wp, self.r_self_impulse_pen_wp,
                self.r_table_pen_wp, self.r_table_impulse_pen_wp,
                self.r_obj_touch_pen_wp,
                self.r_obj_table_pen_wp,
                self.r_violation_pen_wp,
                self.r_bonus_wp, self.r_lift_success_bonus_wp,
                self.r_ctrl_cost_wp, self.r_action_smooth_wp,
                self.r_wrist_vel_wp,
                self.r_obj_vel_wp,
                self.r_obj_xy_coeff_wp,
                self.r_obj_R_coeff_wp,
                self.r_face_dir_wp,
                self.r_contact_dir_wp,
                self.r_contact_pos_wp,
                self.r_contact_neg_pen_wp,
                self.r_affordance_pos_wp,
                self.r_affordance_neg_pen_wp,
                self.r_force_closure_wp,
                self.r_sem_contact_wp,
                # Composite / coefficient term outputs (inspect/plot).
                self.r_mimic_qpos_err_wp,
                self.r_mimic_naive_wp,
                self.hand_ineq_coeff_wp,
                self.r_hand_process_wp,
                self.obj_ineq_coeff_wp,
                self.ineq_coeff_wp,
                self.r_obj_process_wp,
                self.r_hand_obj_bonus_wp,
                self.r_hand_obj_penalty_wp,
                self.reward_wp,
            ],
        )

        # ⑪ Joint-limit penalty (post-mixer add; off when W_JOINT_LIMIT == 0).
        # Reads the applier's PRE-clamp commanded finger target so it can see
        # commands that overshoot a joint limit (``d.ctrl`` is clamped and would
        # hide them). Discourages parking finger commands at the rails — a source
        # of finger chattering in the deterministic rollout. Shared kernel with
        # hand_pose_tracking.
        if float(self.W_JOINT_LIMIT) > 0.0:
            wp.launch(
                _joint_limit_penalty_kernel, dim=self.NWORLD,
                inputs=[
                    self.applier._raw_ctrl_target,
                    self.applier._ctrl_min, self.applier._ctrl_max,
                    self.applier._ctrl_qpa, int(self.n_ctrl),
                    float(self.JOINT_LIMIT_MARGIN), float(self._jl_scale),
                    float(self.W_JOINT_LIMIT),
                    float(self.REWARD_DT), float(self.MIN_REWARD), float(self.MAX_REWARD),
                    self.r_joint_limit_wp, self.reward_wp,
                ],
            )
        else:
            self.r_joint_limit_wp.zero_()

        # ⑫ Wrist-height floor-scrape prevention shaping (post-mixer add; 0 = off).
        # Contact penalties (W_TABLE_*) only yield a gradient after touching, so
        # the wrist is penalised in advance once it enters the floor+margin band.
        if float(self.W_WRIST_HEIGHT) > 0.0:
            wp.launch(
                _wrist_height_penalty_kernel, dim=self.NWORLD,
                inputs=[
                    self.d.xpos, int(self.wrist_body_id),
                    float(self._floor_z), float(self.WRIST_HEIGHT_MARGIN_M),
                    float(self.W_WRIST_HEIGHT),
                    float(self.REWARD_DT), float(self.MIN_REWARD), float(self.MAX_REWARD),
                    self.r_wrist_height_wp, self.reward_wp,
                ],
            )
        else:
            self.r_wrist_height_wp.zero_()

        # Dense penalty for finger sites approaching the floor (pre-collision
        # gradient; for flat objects the reference plane is raised by the object
        # height so the necessary descent is allowed).
        if float(self.W_FINGER_HEIGHT) > 0.0:
            wp.launch(
                _finger_height_penalty_kernel, dim=self.NWORLD,
                inputs=[
                    self.d.site_xpos, self._finger_site_ids_wp,
                    self.d.xpos, int(self.obj_body_id),
                    int(self._n_sensors), int(self.hand_util.fore_arm_sensor_idx),
                    float(self._floor_z), float(self.FINGER_HEIGHT_MARGIN_M),
                    float(self.FINGER_HEIGHT_OBJ_CLEAR_M),
                    float(self.W_FINGER_HEIGHT),
                    float(self.REWARD_DT), float(self.MIN_REWARD), float(self.MAX_REWARD),
                    self.r_finger_height_wp, self.reward_wp,
                ],
            )
        else:
            self.r_finger_height_wp.zero_()

        # Table-impulse increment (Δ⁺) penalty (post-mixer add; 0 = off).
        # raw_table_impulse_w is computed unconditionally every step above, so it
        # is always fresh. prev is updated in place inside the kernel → never zero
        # it per step (only _reset_task_buffers clears it on a full reset).
        if float(self.W_TABLE_IMPULSE_DERIV) > 0.0:
            wp.launch(
                _table_impulse_deriv_penalty_kernel, dim=self.NWORLD,
                inputs=[
                    self.raw_table_impulse_w_wp, self.prev_table_impulse_w_wp,
                    self.ep_step_wp,
                    float(self.W_TABLE_IMPULSE_DERIV),
                    float(self.TABLE_IMPULSE_DERIV_CLIP),
                    float(self.REWARD_DT), float(self.MIN_REWARD), float(self.MAX_REWARD),
                    self.r_table_impulse_deriv_wp, self.reward_wp,
                ],
            )
        else:
            self.r_table_impulse_deriv_wp.zero_()

    def _pre_mix_hook(self) -> None:
        """Extension point called by :meth:`_collect_reward_done_kernel` right
        before the reward mixer launch (all ``raw_*`` buffers are final at that
        point). Default no-op. Subclasses may overwrite raw driver buffers
        per-world here (e.g. neutralise ``raw_obj_xy_coeff`` / ``raw_obj_R_coeff``
        for worlds that are *supposed* to move the object)."""
        return

    def _collect_success_kernel(self) -> None:
        """Success = at the last episode step the object's lift is within
        ``±SUCCESS_LIFT_TOL_FRAC · LIFT_TARGET_M`` of the target lift (object
        held into the target-z band). Overrides the timeout (reason=2) with the
        success reason. Still maintains ``bonus_streak`` for the reward mixer."""
        wp.launch(
            _object_grasping_success_kernel, dim=self.NWORLD,
            inputs=[
                self.bonus_active_wp,
                self.ep_step_wp,
                self.bonus_streak_wp,
                self.d.xpos,
                self.cond.get("obj_p_init"),
                int(self.obj_body_id),
                int(self.MAX_EPISODE_STEPS),
                float(self.LIFT_TARGET_M),
                float(self.SUCCESS_LIFT_TOL_FRAC),
                int(self.SUCCESS_REASON_CODE),
                # Suppress promotion: eval (original convention) + SUCCESS_DONE_PROMOTION=false
                # (JAX v2 convention — success is judged solely by the eval metrics;
                # training dones are always physics/timeout reasons). The kernel
                # still maintains the streak.
                int(1) if (self.eval_mode or not self.SUCCESS_DONE_PROMOTION) else int(0),
                self.done_wp, self.done_mask_wp, self.done_reason_wp,
            ],
        )
        self._launch_retry_early_success()      # re-grasp rewind / early success / elapsed timeout

    @property
    def obj_lift_z_torch(self) -> torch.Tensor:
        """Per-world object lift height above spawn ``(obj_z - obj_z_init)`` as a
        live (NWORLD,) torch view. Used by ``eval_success_masks`` and exposed for
        debug overlays (e.g. evaluate.py's selected-world readout)."""
        obj_z  = wp.to_torch(self.d.xpos)[:, int(self.obj_body_id), 2]
        obj_z0 = wp.to_torch(self.cond.get("obj_p_init"))[:, 2]
        return obj_z - obj_z0

    def eval_success_masks(self) -> Dict[str, torch.Tensor]:
        """Optional eval hook (consumed by ``run_eval``) returning extra
        per-world boolean success masks evaluated at the CURRENT (final eval)
        step. ``run_eval`` reports each as ``success_rate_<name>`` /
        ``n_success_<name>`` alongside its own streak-based ``success_rate``.

        ``lift`` — JAX-convention success: object lifted one-sidedly above
        ``LIFT_SUCCESS_THRESH`` from spawn (``obj_z - obj_z_init >
        LIFT_SUCCESS_THRESH``), mirroring the JAX grasping env's
        ``obj_lift_z > lift_success_threshold``. Lets us compare success_rate
        against the JAX codebase on the SAME definition. Does not touch
        training done-promotion (which keeps the ±tol band)."""
        return {"lift": self.obj_lift_z_torch > float(self.LIFT_SUCCESS_THRESH)}

    # ── Eval reset reproducibility protocol (run_eval hook; EVAL_RESET_SEED >= 0) ─────────
    # All RNG sources on the reset path: ① wrist/obj pose init (sampled_env
    # ``_gpu_init_seed_value`` — the bump writes from the python counter into the
    # GPU buffer, so only the counter needs to be set) ② finger init
    # (``_seed_value``) ③ cond GPU sampler (target_qpos uniform;
    # ``_gpu_seed_value``) ④ cond host RNG (taxonomy row draw; numpy Generator
    # state) ⑤ handler counters (target_R / target_pnt / tax_pull). Physics and
    # the (deterministic eval) policy are not random sources, so pinning these
    # five makes the eval reset draws identical across runs.
    def seed_eval_reset(self):
        """Pin the reset RNGs to ``EVAL_RESET_SEED`` and return the previous state.

        Called by run_eval **right before** ``h.reset()``; after the rollout,
        :meth:`restore_reset_rng` restores the training RNG streams.
        knob off (-1) → None (no-op)."""
        seed = int(self.EVAL_RESET_SEED)
        if seed < 0:
            return None
        fi = self.finger_initializer
        saved = dict(
            pose_init = int(self.sampled_env._gpu_init_seed_value),
            finger    = (int(fi._seed_value) if fi is not None else None),
            cond_gpu  = int(self.cond._gpu_seed_value),
            cond_host = self.cond._rng.bit_generator.state,
            target_R  = int(self._target_R_seed),
            target_pnt = int(self._target_pnt_seed),
            tax_pull  = int(getattr(self, "_tax_pull_seed", 0)),
        )
        base = seed + 1_000_000 * int(self.handler_idx)
        self.sampled_env._gpu_init_seed_value = base & 0x7FFFFFFF
        if fi is not None:
            fi._seed_value = (base + 101) & 0x7FFFFFFF
            fi._seed_wp.assign(np.array([fi._seed_value], dtype=np.int32))
        self.cond._gpu_seed_value = (base + 202) & 0x7FFFFFFF
        self.cond._rng = np.random.default_rng(base + 303)
        self._target_R_seed   = (base + 404) & 0x7FFFFFFF
        self._target_pnt_seed = (base + 505) & 0x7FFFFFFF
        self._tax_pull_seed   = (base + 606) & 0x7FFFFFFF
        return saved

    def restore_reset_rng(self, saved) -> None:
        """Inverse of :meth:`seed_eval_reset` — restore the training RNG streams."""
        if saved is None:
            return
        fi = self.finger_initializer
        self.sampled_env._gpu_init_seed_value = saved["pose_init"]
        if fi is not None and saved["finger"] is not None:
            fi._seed_value = saved["finger"]
            fi._seed_wp.assign(np.array([fi._seed_value], dtype=np.int32))
        self.cond._gpu_seed_value = saved["cond_gpu"]
        self.cond._rng = np.random.default_rng()
        self.cond._rng.bit_generator.state = saved["cond_host"]
        self._target_R_seed   = saved["target_R"]
        self._target_pnt_seed = saved["target_pnt"]
        self._tax_pull_seed   = saved["tax_pull"]

    def obs_norm_passthrough_mask(self) -> np.ndarray:
        """obs running-norm passthrough mask (obs_dim,) bool — True = pass raw.

        Derived from ``obs_term_layout`` (the single source of truth for the obs):
        only terms in :attr:`OBS_NORM_STD_TERMS` are standardised (False); the
        rest (BPS tail included — automatic, since the layout covers the whole
        obs_dim) pass through (True). Injected by the builders at policy
        creation and stored in the ckpt as a normalizer buffer."""
        mask = np.ones(int(self.obs_dim), dtype=bool)            # passthrough by default
        std = set(self.OBS_NORM_STD_TERMS)
        for name, s, e in self.obs_term_layout:
            if name in std:
                mask[s:e] = False
        return mask

    def eval_semantic_metrics_per_world(self) -> Dict[str, torch.Tensor]:
        """**Per-world (NWORLD,) version** of :meth:`eval_semantic_metrics` — returns
        the same tensors **before** the ``.mean()`` reduction.

        Returns: ``contact_iou / contact_precision / contact_recall / contact_f1 /
        face_angle_err_deg`` (float) + ``has_contact`` (bool — for a world with
        no contact slot at all ``face_angle_err_deg`` is meaningless and exactly 0.0).

        Used by the per-taxonomy / per-object breakdown (scripts/eval_taxonomy.py).
        Recomputing in the script would drift from the handler's private
        slot→sensor mapping and eps conventions, so **this is the single source
        of truth** and :meth:`eval_semantic_metrics` reduces it.
        """
        # slot → sensor-space mapping (cached once).
        idx_t = getattr(self, "_sem_slot_sensor_idx_t", None)
        if idx_t is None:
            idx_np = self._hand_obj_contact_sensor_idx_wp.numpy().astype(np.int64)
            idx_t = torch.as_tensor(idx_np, device=self.torch_device)
            self._sem_slot_sensor_idx_t = idx_t
        valid = idx_t >= 0                                        # (n_slots,)
        idx_c = idx_t.clamp(min=0)

        mask_t   = wp.to_torch(self.cond.get("specific_finger_link_mask"))  # (NW, n_sensors)
        active   = (mask_t[:, idx_c] > 0.5) & valid               # (NW, n_slots) taxonomy active slot
        contact  = (self.hand_obj_contact_impulse_per_slot_torch > 0) & valid
        inter    = (active & contact).sum(dim=1).float()
        n_act    = active.sum(dim=1).float()
        n_con    = contact.sum(dim=1).float()
        union    = (active | contact).sum(dim=1).float()
        eps      = 1e-6
        precision = inter / (n_con + eps)
        recall    = inter / (n_act + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        iou       = inter / (union + eps)

        # face↔contact-normal angular error (deg) of the slots in contact.
        cos_slot  = self.contact_dir_cos_torch[:, idx_c]          # (NW, n_slots)
        ang_err   = torch.rad2deg(torch.arccos(cos_slot.clamp(-1.0, 1.0)))
        con_f     = contact.float()
        face_err  = (ang_err * con_f).sum(dim=1) / (con_f.sum(dim=1) + eps)
        any_con   = con_f.sum(dim=1) > 0

        return {
            "contact_iou":        iou,
            "contact_precision":  precision,
            "contact_recall":     recall,
            "contact_f1":         f1,
            "face_angle_err_deg": face_err,
            "has_contact":        any_con,
        }

    def eval_semantic_metrics(self) -> Dict[str, float]:
        """Taxonomy-compliance semantic metrics (port of the JAX v2 debug_mode
        block; run_eval reports them under ``sem_<name>`` keys). Compares the
        per-slot contact state at the current (final eval) step against the
        taxonomy active mask — pure torch ops (once per eval, no impact on the
        training hot path):

          * ``contact_iou / precision / recall / f1`` — slot level:
            precision = |active∩contact| / |contact| (penalises wrong-finger contact),
            recall    = |active∩contact| / |active|  (coverage of the fingers that
            should be used — **neglected ring/little fingers show up as low recall**),
          * ``face_angle_err_deg`` — mean angular error (deg) between the taxonomy
            face direction and the contact normal over the slots in contact.

        The computation lives in :meth:`eval_semantic_metrics_per_world`
        (per-world tensors); this method only reduces it, so the two paths cannot diverge.
        """
        pw       = self.eval_semantic_metrics_per_world()
        face_err = pw["face_angle_err_deg"]
        any_con  = pw["has_contact"]
        return {
            "contact_iou":       float(pw["contact_iou"].mean().item()),
            "contact_precision": float(pw["contact_precision"].mean().item()),
            "contact_recall":    float(pw["contact_recall"].mean().item()),
            "contact_f1":        float(pw["contact_f1"].mean().item()),
            # Worlds without any contact are excluded from the angular-error sample.
            "face_angle_err_deg": float(face_err[any_con].mean().item()) if bool(any_con.any()) else 0.0,
        }

    def per_world_reset_if_done(self) -> None:
        """Extends base per-world reset to zero ``last_action`` for done worlds.

        Calls super (which resets physics + cond + obs for done worlds), then
        zeroes ``last_action[w]`` for those worlds so the first step of the
        new episode computes action smoothness against zero rather than the
        stale last action from the previous episode (matches the tracking
        task's reset semantics — first-step penalty = ``||a_0||²``).
        """
        super().per_world_reset_if_done()
        # Re-arm the manual-lift latch for worlds that just reset — must run in
        # eval too (the rule still fires there), so do it BEFORE the eval early-out.
        if hasattr(self, "lift_reached_wp"):
            wp.launch(
                _zero_masked_lift_latch_kernel, dim=self.NWORLD,
                inputs=[self.done_mask_wp, self.lift_reached_wp],
            )
        if self.eval_mode:
            return
        wp.launch(
            _zero_masked_action_kernel, dim=self.NWORLD,
            inputs=[
                self.done_mask_wp,
                self.last_action_wp,
                int(self._action_dim),
            ],
        )
        # Invalidate the pose-FD velocity (reset worlds only): fd_valid=0 → the
        # next step seeds from the fresh pose (velocity 0). super() has already
        # re-collected the obs, so the fd_vel slice of the obs is rewritten to 0
        # here too (prevents the pre-teleport velocity from leaking into the new
        # episode's first obs). done_mask is super()'s final mask, nonfinite flag included.
        if hasattr(self, "fd_valid_wp"):
            if not hasattr(self, "_fd_obs_off"):
                self._fd_obs_off = next(
                    (s for (n, s, _e) in self.obs_term_layout if n == "fd_vel"), -1)
            wp.launch(
                _fd_vel_invalidate_masked_kernel, dim=self.NWORLD,
                inputs=[self.done_mask_wp, int(self._fd_obs_off),
                        self.fd_valid_wp, self.fd_vel_wp, self.fd_obj_vz_wp,
                        self.obs_wp],
            )

    def restore_train_state(self) -> None:
        """Base restore + invalidate the pose-FD velocity state — after eval has
        stepped the physics, the pose returns to the pre-eval pose, so an FD
        against the prev snapshot (last eval pose) would be a meaningless jump.
        Every world gets one step of zero velocity (negligible), then re-seeds."""
        super().restore_train_state()
        if hasattr(self, "fd_valid_wp"):
            self.fd_valid_wp.zero_()
            self.fd_vel_wp.zero_()
            self.fd_obj_vz_wp.zero_()
        for n, buf in getattr(self, "_v2_snap", {}).items():
            wp.copy(getattr(self, n), buf)

    # ── Post-cond hook: snapshot ``obj_p_init`` / ``target_pnt`` /
    # ``target_R`` from physics for every world that was just reset. ──
    def _has_post_cond_update_hook(self) -> bool:
        return bool(self.has_object)

    def _post_cond_update_hook(self, world_mask_np=None, is_episode_reset=True) -> None:
        """Refresh the snapshot + taxonomy-driven cond fields after a reset.

        ``is_episode_reset=False`` is the mid-episode dynamic-target-switch path:
        it re-samples only the GRASP TARGET (target_pnt / target_R / taxonomy)
        and **preserves** ``obj_p_init`` (the spawn lift/xy-drift anchor must not
        jump mid-episode) and does NOT reschedule ``target_switch_step`` (at most
        one switch per episode). ``True`` (episode reset) re-anchors obj_p_init
        and schedules the next switch tick.


        Two groups of fields:

          1. **Physics snapshots** (cheap, no per-step FK):
             * ``obj_p_init[w]`` ← current ``d.xpos[w, obj_body_id]`` —
               authoritative anchor for the lift / xy_drift signals.
             * ``target_pnt[w]`` ← ``pcd_world[argmin(||wrist - pcd_world||)] +
               noise(TARGET_PNT_NOISE_M)`` when the variant PCD cache is
               available, mirroring the legacy ``target_pnt`` derivation.
               When the PCD cache is missing (e.g. obj_pcd not built), falls
               back to ``obj_p_init[w] + uniform[-noise, noise]`` so the field
               always has a meaningful value.
             * ``target_R[w]`` ← per-world (3, 3) frozen rotation init
               sampled by :func:`hand_utils.sample_target_R_init_per_world`
               (mirrors :func:`wrist_pose_init`'s target draw). Its first
               column ``target_R[w, :, 0]`` is the legacy ``target_dir``
               (unit X-axis heading); the reward kernel reads column 0
               and the obs kernel exposes that column rotated into the
               wrist frame. The hand-center x-axis is aligned with this
               heading at init by ``wrist_pose_init``.

          2. **Per-world taxonomy draw** (links ``target_qpos`` and
             ``specific_finger_link_mask`` to the same choice):
             For each masked world, with probability ``TAXONOMY_PROB`` we
             pick ``tax_idx ~ uniform(n_tax)`` and write BOTH
             ``target_qpos[w] = taxonomy_qpos[tax_idx]`` AND
             ``specific_finger_link_mask[w] = taxonomy_mask[tax_idx]``.
             With probability ``1 - TAXONOMY_PROB`` the qpos field keeps
             the GPU-uniform sample and the mask falls back to
             ``self._default_mask_np`` (ones with arm slot zeroed). Applied by
             :meth:`_apply_taxonomy_to_qpos_and_mask` as a **GPU masked gather**
             (host draws only the per-world ``tax_row``); replaces the legacy
             CPU sampler attached to ``target_qpos`` alone — keeping the fields
             consistent matters for the finger-close kernel, which uses the
             mask to focus the distance reward on the taxonomy's active fingers.
        """
        if not self.has_object:
            return

        # ── FULLY GPU-resident cond refresh (no d.xpos/xmat.numpy() sync) ───
        # This hook used to be the training bottleneck (~85% of a PPO iter went
        # to per_world_reset_if_done, ~65% of THAT was this hook): it copied
        # ``d.xpos``/``d.xmat`` host-side and ran two per-world Python loops
        # (target_R + nearest-PCD target_pnt). All of it is now warp kernels
        # writing the cond GPU buffers directly, gated by a tiny (NWORLD,) int
        # mask. ``cond.reset`` only samples GPU-sampler fields (target_qpos), so
        # these direct GPU writes for the snapshot/target fields are authoritative.
        f = self.cond._fields

        # Stage the (NWORLD,) int mask once (tiny host→GPU copy) — shared by all
        # kernels below. None → all worlds; else the bool world-mask.
        mh = self._cond_mask_host.numpy()
        if world_mask_np is None:
            mh[:] = 1
        else:
            mh[:] = np.asarray(world_mask_np, dtype=bool).astype(np.int32)
        wp.copy(self._cond_mask_wp, self._cond_mask_host)

        # (a) target_R — per-world heading rotation (GPU port of
        #     ``hand_utils.sample_target_R_init_per_world``). ``target_R[:, 0]``
        #     stays the legacy ``target_dir`` read by the wrist-dir reward.
        self._target_R_seed = (self._target_R_seed + 1) & 0x7FFFFFFF
        wp.launch(
            _object_grasping_sample_target_R_kernel, dim=self.NWORLD,
            inputs=[
                self.d.xmat, int(self.wrist_body_id),
                wp.vec3(float(self._hc_col0_np[0]), float(self._hc_col0_np[1]), float(self._hc_col0_np[2])),
                wp.vec3(float(self._hc_col2_np[0]), float(self._hc_col2_np[1]), float(self._hc_col2_np[2])),
                self._cond_mask_wp, int(self._target_R_seed),
                f["target_R"]["gpu"],
            ],
        )

        # (b) target_pnt — nearest object-PCD vertex to the wrist + jitter (GPU
        #     port of ``_sample_target_pnt``; same nearest-PCD scan the reward
        #     path uses, run for the wrist).
        self._target_pnt_seed = (self._target_pnt_seed + 1) & 0x7FFFFFFF
        wp.launch(
            _object_grasping_sample_target_pnt_kernel, dim=self.NWORLD,
            inputs=[
                self.d.xpos, self.d.xmat, self._pcd_gpu, self._pcd_assignment,
                int(self._grasp_obj_slot), int(self._grasp_obj_body_id),
                int(self.wrist_body_id), int(self._pcd_n_pts),
                float(self.TARGET_PNT_NOISE_M), int(self._target_pnt_seed),
                self._cond_mask_wp, f["target_pnt"]["gpu"],
            ],
        )

        # (c) obj_p_init / obj_R_init — spawn-pose anchor for lift / xy-drift /
        #     obj-pose-vs-spawn obs. Only re-anchored on an episode reset, NOT on
        #     a mid-episode target switch (the anchor must not jump mid-episode).
        if is_episode_reset:
            wp.launch(
                _object_grasping_snapshot_obj_pose_kernel, dim=self.NWORLD,
                inputs=[
                    self.d.xpos, self.d.xmat, int(self.obj_body_id),
                    self._cond_mask_wp,
                    f["obj_p_init"]["gpu"], f["obj_R_init"]["gpu"],
                ],
            )

        # (d) Joint taxonomy draw → target_qpos + specific_finger_link_mask
        #     (already a GPU masked gather).
        self._apply_taxonomy_to_qpos_and_mask(world_mask_np=world_mask_np)

        # (d') Finger init taxonomy-pull (port of JAX v2 _finger_init_fn) —
        #     stochastic partial interpolation of the reset worlds' finger
        #     qpos/ctrl towards the freshly drawn target_qpos. Must come **after**
        #     the taxonomy overlay to see the fresh template. Since qpos changed,
        #     forward is re-run → the site/xpos FK of the following obs collect
        #     matches the pulled pose.
        if is_episode_reset and float(self.FINGER_INIT_TAX_PULL_PROB) > 0.0:
            self._tax_pull_seed = (int(getattr(
                self, "_tax_pull_seed",
                int(self.handler_idx) * 7_000_003 + 12_345)) + 1) & 0x7FFFFFFF
            wp.launch(
                _object_grasping_finger_tax_pull_kernel, dim=self.NWORLD,
                inputs=[
                    self._cond_mask_wp,
                    f["target_qpos"]["gpu"],
                    self.applier._ctrl_qpa,
                    int(self.n_ctrl),
                    float(self.FINGER_INIT_TAX_PULL_PROB),
                    float(self.FINGER_INIT_TAX_PULL_MIN),
                    float(self.FINGER_INIT_TAX_PULL_MAX),
                    int(self._tax_pull_seed),
                    self.d.qpos, self.d.ctrl,
                ],
            )
            if hasattr(self, "capture_forward"):
                wp.capture_launch(self.capture_forward.graph)

        # (e) Dynamic-target switch tick — only schedule on episode reset
        #     (the switch path consumed the previous tick; ≤ 1 switch per episode).
        if is_episode_reset and self.DYNAMIC_TARGET_ENABLED:
            self._resample_switch_step(world_mask_np=world_mask_np)

    def _maybe_dynamic_target_switch(self) -> None:
        """Mid-episode grasp-target switch (object_grasping impl of the base
        hook). For worlds whose ``ep_step`` reached their scheduled
        ``target_switch_step``:

          1. trigger kernel → ``switch_mask`` + consume the tick
             (``target_switch_step = -1``),
          2. ``cond.reset(switch_mask)`` — re-samples target_qpos (GPU uniform)
             for fired worlds (sync-free; no host samplers),
          3. ``_post_cond_update_hook(switch_mask_np, is_episode_reset=False)`` —
             re-samples target_pnt / target_R / taxonomy for fired worlds while
             PRESERVING obj_p_init (lift anchor) and NOT rescheduling the switch,
          4. ``_collect_obs_kernel`` — refresh obs so the policy sees the new
             target on the next action.

        Reward at the switch step is still against the OLD target; obs/reward
        align to the new target from the next step. No-op when disabled / no obj.
        """
        if not self.DYNAMIC_TARGET_ENABLED or not self.has_object:
            return
        # 1. per-world trigger mask + consume the scheduled tick.
        wp.launch(
            _dynamic_target_switch_mask_kernel, dim=self.NWORLD,
            inputs=[
                self.ep_step_wp,
                self.target_switch_step_wp,
                self.switch_mask_wp,
            ],
        )
        # 2. re-sample GPU cond (target_qpos uniform) on fired worlds.
        self.cond.reset(mask_wp=self.switch_mask_wp)
        # 3. host-side grasp-target re-sample for fired worlds (one mask sync;
        #    skip everything when nothing fired).
        switch_mask_np = self.switch_mask_torch.cpu().numpy().astype(bool)
        if switch_mask_np.any():
            self._post_cond_update_hook(
                world_mask_np=switch_mask_np, is_episode_reset=False,
            )
        # 4. re-collect obs in place (base ``step`` also collects after this,
        #    but this keeps the new-target obs ready immediately).
        self._collect_obs_kernel()

    # ══════════════════════════════════════════════════════════════════════
    # Per-taxonomy diagnostics + object-size-conditioned taxonomy sampling
    # ══════════════════════════════════════════════════════════════════════
    def _setup_taxonomy_diag_and_size(self) -> None:
        """Prepare (c) the per-taxonomy episode-mean contact_ratio / SR
        aggregation state and (a) the per-world small-object mask based on the
        object diameter.

        Diagnostics: ``_collect_obs_kernel`` accumulates the ratio every step,
        ``_harvest_tax_stats`` collects it per taxonomy at reset, and
        ``on_train_progress`` periodically prints the ``[tax-diag]`` table
        (window reset).

        Size condition: with ``TAXONOMY_SIZE_CONDITIONED=True``, worlds whose
        object diameter is < ``TAXONOMY_SMALL_OBJ_DIAM_M`` draw only from
        small-compatible taxonomies (``do_finger_tip`` or active fingers ≤ 3) —
        never assigning combinations a full-hand taxonomy physically cannot
        reach, which removes the structural low equilibrium of contact_ratio."""
        n_tax = int(getattr(self, "_n_tax", 0) or 0)
        dev = self.d.qpos.device
        self._cur_tax_np     = np.full(int(self.NWORLD), -1, dtype=np.int64)
        # Eval stratification hook (None = the usual random draw). See ``set_forced_taxonomy_rows``.
        # ⚠ This method runs once more via apply_handler_knobs(rebuild_cond=True)
        #   (which is why ``[tax-size]`` prints twice), so set it only **after** the env is built.
        self._tax_force_rows: Optional[np.ndarray] = None
        self._tax_stats      = np.zeros((n_tax + 1, 3), dtype=np.float64)  # [n_eps, ratio_sum, n_success]; last row = uniform(-1)
        self._ep_ratio_sum_t = torch.zeros(int(self.NWORLD), device=str(dev))
        self._ep_len_t       = torch.zeros(int(self.NWORLD), device=str(dev))
        self._tax_diag_iter  = 0

        # small-compatible taxonomy pool (do_finger_tip or ≤ 3 active fingers)
        hu = getattr(self, "hand_util", None)
        do_tip = np.asarray(getattr(hu, "taxonomy_do_finger_tip_array", []), dtype=bool)
        names  = list(getattr(hu, "taxonomy_specific_finger_names_array", []) or [])
        if n_tax > 0 and do_tip.size == n_tax:
            n_fingers = np.array([len([x for x in ns if "palm" not in x]) for ns in names])
            smallok = np.where(do_tip | (n_fingers <= 3))[0].astype(np.int32)
            self._tax_rows_smallok = smallok if smallok.size else np.arange(n_tax, dtype=np.int32)
        else:
            self._tax_rows_smallok = np.arange(max(n_tax, 1), dtype=np.int32)

        # per-world object diameter (from the variant PCD bbox) → small mask
        env = self.sampled_env
        pcd_cache  = getattr(env, "variant_pcd_cache", None) or {}
        assignment = getattr(env, "assignment", None)
        renamed    = list(getattr(env, "renamed_obj_names", []) or [])
        self._world_obj_small_np = None
        if pcd_cache and assignment is not None and renamed:
            body = renamed[0]
            # "grip width" ≈ the middle of the sorted bbox edges (a bottle:
            # 5×5×19cm → width 5cm — the diagonal (20cm) would overestimate it
            # regardless of grasp difficulty).
            diam = {}
            for v, d in pcd_cache.items():
                p = d.get(body)
                diam[v] = (float(np.sort(p.max(0) - p.min(0))[1])
                           if p is not None and len(p) else 1e9)
            dm = np.array([diam.get(int(assignment[w]), 1e9) for w in range(int(self.NWORLD))])
            self._world_obj_diam_np = dm
            # The small test is recomputed at draw time with the current threshold
            # (reflects the yaml knob); this print is informational, based on init-time values.
            thr = float(self.TAXONOMY_SMALL_OBJ_DIAM_M)
            print(f"[tax-size] grasp-width(m): min={dm.min():.3f} max={dm.max():.3f} "
                  f"| small(<{thr:.2f}m) worlds: {int((dm < thr).sum())}/{int(self.NWORLD)} "
                  f"| smallok tax pool: {self._tax_rows_smallok.size}/{n_tax}")

    def _harvest_tax_stats(self, idxs: np.ndarray) -> None:
        """Collect (taxonomy, ep-mean ratio, success) for the worlds whose episode just ended."""
        it = torch.as_tensor(idxs, device=self._ep_ratio_sum_t.device, dtype=torch.long)
        ratio = (self._ep_ratio_sum_t[it] /
                 self._ep_len_t[it].clamp(min=1.0)).cpu().numpy()
        reason = self.done_reason_torch[it].cpu().numpy()
        tax  = self._cur_tax_np[idxs]
        row  = np.where(tax >= 0, tax, self._tax_stats.shape[0] - 1)
        np.add.at(self._tax_stats[:, 0], row, 1.0)
        np.add.at(self._tax_stats[:, 1], row, ratio)
        np.add.at(self._tax_stats[:, 2], row,
                  (reason == int(self.SUCCESS_REASON_CODE)).astype(np.float64))
        self._ep_ratio_sum_t[it] = 0.0
        self._ep_len_t[it] = 0.0

    def _print_tax_diag(self) -> None:
        """Print the [tax-diag] per-taxonomy (n, SR, ratio) window table, then reset it."""
        st = self._tax_stats
        if st[:, 0].sum() < 1:
            return
        names = list(getattr(self.hand_util, "taxonomy_name_list", [])) + ["uniform"]
        lines = []
        for i in range(st.shape[0]):
            n = st[i, 0]
            if n < 1:
                continue
            nm = names[i] if i < len(names) else f"tax{i}"
            lines.append(f"{nm}:n={int(n)},sr={st[i,2]/n:.2f},ratio={st[i,1]/n:.2f}")
        print(f"[tax-diag] {'  '.join(lines)}")
        st[:] = 0.0

    def taxonomy_rows_from_names(self, names) -> np.ndarray:
        """List of taxonomy names → row index array (``hand_util.taxonomy_name_list`` order).

        Unknown names fail immediately with a ``ValueError`` listing the
        available candidates — so a typo cannot silently turn into "that
        taxonomy was not evaluated". The returned rows are **global** indices
        into the GPU gather tables and can be passed directly to
        :meth:`set_forced_taxonomy_rows` (same coordinates as the
        ``using_taxonomy_names`` filter).
        """
        catalog = list(getattr(self.hand_util, "taxonomy_name_list", []) or [])
        if not catalog:
            raise RuntimeError("taxonomy_rows_from_names: this hand has no taxonomy_name_list.")
        rows, unknown = [], []
        for nm in names:
            nm = str(nm).strip()
            if nm in catalog:
                rows.append(catalog.index(nm))
            else:
                unknown.append(nm)
        if unknown:
            raise ValueError(
                f"unknown taxonomy name(s) {unknown}. "
                f"available ({len(catalog)}): {catalog}")
        n_tax = int(getattr(self, "_n_tax", 0) or 0)
        over  = [r for r in rows if r >= n_tax]
        if over:
            raise ValueError(
                f"taxonomy rows {over} exceed the built table (_n_tax={n_tax}) — "
                f"the GPU tables were truncated below taxonomy_name_list length.")
        return np.asarray(rows, dtype=np.int64)

    def set_forced_taxonomy_rows(self, rows) -> None:
        """Pin the per-world taxonomy row (**eval only**; no effect on the training path).

        ``rows`` — a ``(NWORLD,)`` int array, or ``None`` to return to the usual
        random draw. Encoding matches ``tax_row`` / :attr:`_cur_tax_np`:
        ``>=0`` = taxonomy row, ``-1`` = uniform fallback (no taxonomy overlay).

        Re-applied on every reset (full and partial), so the grouping key
        (``_cur_tax_np``) is structurally stable across the rollout — which is
        what makes per-taxonomy metric aggregation valid.

        The forced path **bypasses all of** ``TAXONOMY_PROB``,
        ``_tax_allowed_rows`` (``using_taxonomy_names``) and
        ``TAXONOMY_SIZE_CONDITIONED`` (pool validity is the caller's
        responsibility), and does not consume ``cond._rng``, so the reset stream
        stays a pure function of ``EVAL_RESET_SEED``.

        To evaluate only specific taxonomies, tile just the desired rows over the worlds::

            rows = h.taxonomy_rows_from_names(["palmar_pinch", "tripod", "tip_pinch"])
            h.set_forced_taxonomy_rows(np.tile(rows, h.NWORLD // len(rows)))
        """
        if rows is None:
            self._tax_force_rows = None
            return
        if not self._has_taxonomy:
            raise RuntimeError(
                "set_forced_taxonomy_rows: this hand has no taxonomy tables "
                f"(_has_taxonomy=False, _n_tax={getattr(self, '_n_tax', 0)}).")
        a = np.asarray(rows, dtype=np.int64).reshape(-1)
        if a.shape[0] != int(self.NWORLD):
            raise ValueError(
                f"rows must be (NWORLD={int(self.NWORLD)},), got {tuple(a.shape)}")
        bad = np.unique(a[(a < -1) | (a >= int(self._n_tax))])
        if bad.size:
            raise ValueError(
                f"rows out of range [-1, {int(self._n_tax)}): {bad.tolist()}")
        self._tax_force_rows = a

    def _apply_taxonomy_to_qpos_and_mask(self, world_mask_np=None) -> None:
        """Per-world taxonomy overlay onto the cond fields — **GPU masked gather**.

        Writes the same FIVE consistent cond fields as before (all from the
        SAME random ``tax_idx`` per world)::

          target_qpos · specific_finger_link_mask · specific_ctrl_mask ·
          face_dir_idx_in_mat · face_dir_sign

        but the heavy lifting now runs on the GPU
        (:func:`_object_grasping_taxonomy_gather_kernel`). The host only draws
        the small ``(NWORLD,)`` ``tax_row`` (using the seeded cond RNG) and
        uploads it — NO full-buffer CPU↔GPU round-trip, no per-world Python
        loop, no forced sync (the old host overlay did 10 full-buffer copies +
        a `.numpy()` sync every reset).

        ``tax_row[w]`` encodes the draw: ``-2`` = world not reset (skip),
        ``-1`` = "rolled uniform" → keep GPU-uniform ``target_qpos`` + default
        masks/face-dirs, ``>=0`` = taxonomy row index. With probability
        ``TAXONOMY_PROB`` a reset world takes a random taxonomy row; otherwise
        it falls back to defaults. When taxonomy tables are unavailable every
        reset world rolls ``-1`` (idempotent default reset).

        GPU-direct write is safe: ``sample_all_gpu`` writes ``target_qpos``
        GPU-side first (the mask fields have no GPU sampler → skipped), and
        nothing copies CPU→GPU for these fields afterward, so the kernel's
        ``f['gpu']`` writes are authoritative.
        """
        n_world = int(self.NWORLD)
        tax_row = self._tax_row_host.numpy()
        tax_row[:] = -2                              # -2 = not reset this step

        if world_mask_np is None:
            idxs = np.arange(n_world)
        else:
            mask_np = np.asarray(world_mask_np, dtype=bool)
            if mask_np.size == 0 or not mask_np.any():
                return
            idxs = np.where(mask_np)[0]
        if idxs.size == 0:
            return

        # (c) Diagnostics harvest: these worlds just finished an episode (the
        # initial full reset(None) has no completed episodes, so skip).
        if world_mask_np is not None and getattr(self, "_tax_stats", None) is not None:
            self._harvest_tax_stats(idxs)

        tax_row[idxs] = -1                           # reset worlds default to uniform-fallback
        _forced = getattr(self, "_tax_force_rows", None)
        if _forced is not None:
            # Eval stratification (set_forced_taxonomy_rows): bypasses the draw
            # entirely — no TAXONOMY_PROB roll, no _tax_allowed_rows /
            # size-conditioned pool, no cond host RNG consumption (the reset
            # stream stays a pure function of EVAL_RESET_SEED). Only the ``idxs``
            # slice is written, so on a partial reset the other worlds'
            # _cur_tax_np is preserved and the grouping key stays intact.
            _rows_f = _forced[idxs].astype(np.int32)
            tax_row[idxs]          = _rows_f
            self._cur_tax_np[idxs] = _rows_f          # -1 keeps the uniform fallback
        elif self._has_taxonomy:
            prob = float(np.clip(self.TAXONOMY_PROB, 0.0, 1.0))
            if prob > 0.0:
                use = self.cond._rng.random(idxs.size) < prob
                allowed = getattr(self, "_tax_allowed_rows", None)
                diam = getattr(self, "_world_obj_diam_np", None)
                if bool(self.TAXONOMY_SIZE_CONDITIONED) and diam is not None:
                    # (a) Object-size-conditioned draw: small-object worlds draw
                    # only from the small-compatible pool, the rest from the
                    # usual pool. The mask is computed at draw time with the
                    # current threshold — always reflecting yaml knob values
                    # applied after init (no frozen mask).
                    pool_a = (allowed if allowed is not None
                              else np.arange(self._n_tax, dtype=np.int32))
                    pool_s = self._tax_rows_smallok
                    rows = np.empty(idxs.size, dtype=np.int32)
                    ms = diam[idxs] < float(self.TAXONOMY_SMALL_OBJ_DIAM_M)
                    if ms.any():
                        rows[ms] = pool_s[self.cond._rng.integers(
                            0, pool_s.size, size=int(ms.sum()))]
                    if (~ms).any():
                        rows[~ms] = pool_a[self.cond._rng.integers(
                            0, pool_a.size, size=int((~ms).sum()))]
                elif allowed is not None:
                    # Name-filtered draw (``using_taxonomy_names``): uniform over
                    # the ALLOWED row set only — the tables/kernel keep using the
                    # full row index, so narrowing here alone keeps the whole path consistent.
                    rows = allowed[self.cond._rng.integers(
                        0, allowed.size, size=idxs.size)].astype(np.int32)
                else:
                    rows = self.cond._rng.integers(0, self._n_tax, size=idxs.size).astype(np.int32)
                tax_row[idxs[use]] = rows[use]       # taxonomy-active worlds → row index
                # (c) Record the current taxonomy (uniform fallback = -1)
                self._cur_tax_np[idxs] = -1
                self._cur_tax_np[idxs[use]] = rows[use]

        # Upload the small (NWORLD,) draw (async H2D) and run the masked gather.
        wp.copy(self._tax_row_gpu, self._tax_row_host)
        f = self.cond._fields
        wp.launch(
            _object_grasping_taxonomy_gather_kernel, dim=n_world,
            inputs=[
                self._tax_row_gpu,
                self._tax_qpos_gpu, self._tax_fmask_gpu, self._tax_cmask_gpu,
                self._tax_fdir_idx_gpu, self._tax_fdir_sgn_gpu,
                self._def_fmask_gpu, self._def_cmask_gpu,
                self._def_fdir_idx_gpu, self._def_fdir_sgn_gpu,
                int(self.n_ctrl), int(self._n_sensors),
                f["target_qpos"]["gpu"],
                f["specific_finger_link_mask"]["gpu"],
                f["specific_ctrl_mask"]["gpu"],
                f["face_dir_idx_in_mat"]["gpu"],
                f["face_dir_sign"]["gpu"],
            ],
        )

    def _sample_target_pnt(
        self,
        obj_xpos:    np.ndarray,    # (NWORLD, 3)  obj centroid xpos
        wrist_xpos:  np.ndarray,    # (NWORLD, 3)  wrist xpos at reset
    ) -> np.ndarray:
        """Pick a target contact point on the object PCD (or centroid fallback).

        Mirrors the legacy ``target_pnt = pcd[argmin(||hand - pcd||)] + noise``
        by using ``orchestrator.variant_pcd_cache`` (body-local PCD) +
        per-world ``xmat`` to transform vertices into world frame, then
        picking the nearest-to-wrist vertex and adding a small uniform
        jitter (``TARGET_PNT_NOISE_M``) so consecutive resets don't always
        select identical anchors.

        Falls back to ``obj_xpos + uniform[-noise, noise]`` when the PCD
        cache isn't available (e.g. obj_pcd not built) so the field always
        has a meaningful value.
        """
        rng       = self.cond._rng
        n_world   = int(self.NWORLD)
        noise_amp = float(self.TARGET_PNT_NOISE_M)

        # ── PCD-based path ───────────────────────────────────────────────
        env        = self.sampled_env
        pcd_cache  = getattr(env, "variant_pcd_cache", None) or {}
        assignment = getattr(env, "assignment", None)
        renamed    = list(getattr(env, "renamed_obj_names", []) or [])
        body_name  = renamed[0] if renamed else None

        if pcd_cache and assignment is not None and body_name is not None:
            xmat_np = self.d.xmat.numpy()                     # (NWORLD, nbody, 3, 3) or flat
            if xmat_np.ndim == 3:                              # (NWORLD, nbody, 9) flat layout
                xmat_np = xmat_np.reshape(xmat_np.shape[0], xmat_np.shape[1], 3, 3)
            target_pnt = np.zeros((n_world, 3), dtype=np.float32)
            for w in range(n_world):
                v_idx = int(assignment[w])
                pcd_local = pcd_cache.get(v_idx, {}).get(body_name)
                if pcd_local is None or len(pcd_local) == 0:
                    target_pnt[w] = obj_xpos[w]
                    continue
                R_body    = xmat_np[w, self.obj_body_id]                # (3, 3)
                pcd_world = (R_body @ pcd_local.T).T + obj_xpos[w]      # (n_pts, 3)
                dists     = np.linalg.norm(pcd_world - wrist_xpos[w], axis=1)
                target_pnt[w] = pcd_world[int(np.argmin(dists))]
            if noise_amp > 0.0:
                target_pnt = target_pnt + rng.uniform(
                    low=-noise_amp, high=noise_amp,
                    size=target_pnt.shape,
                ).astype(np.float32)
            return target_pnt

        # ── Centroid + noise fallback ───────────────────────────────────
        offset = rng.uniform(
            low=-noise_amp, high=noise_amp, size=obj_xpos.shape,
        ).astype(np.float32)
        return obj_xpos.astype(np.float32, copy=False) + offset

    # ══════════════════════════════════════════════════════════════════════
    # Re-grasp retry / early success knobs (episode logic shared by teacher and student)
    # ══════════════════════════════════════════════════════════════════════
    CONFIG_KEYS_GROUPS = {
        **CONFIG_KEYS_GROUPS,
        "retry": (
            "RETRY_ENABLED", "RETRY_MAX", "RETRY_APPROACH_STEPS",
            "RETRY_MIN_LIFT_STEPS", "RETRY_MIN_HOLD_STEPS",
            "RETRY_DROP_CONTACT_STEPS", "RETRY_LIFT_THRESH_M",
            "RETRY_REQUIRE_CONTACT",
            "SUCCESS_EARLY_ENABLED", "SUCCESS_EARLY_LIFT_HOLD_STEPS",
            "OBS_EP_HORIZON_STEPS",
        ),
    }
    RETRY_ENABLED:            bool  = True
    RETRY_MAX:                int   = 1      # max retries per world
    RETRY_APPROACH_STEPS:     int   = 40     # ep_step := LIFT_STEP − this after a rewind
    RETRY_MIN_LIFT_STEPS:     int   = 15     # steps after LIFT_STEP before a drop can be declared
    RETRY_MIN_HOLD_STEPS:     int   = 20     # minimum hold steps a retry cycle must keep
    RETRY_DROP_CONTACT_STEPS: int   = 5      # consecutive no-contact & not-lifted steps → dropped
    RETRY_LIFT_THRESH_M:      float = -1.0   # <0 → LIFT_SUCCESS_THRESH
    RETRY_REQUIRE_CONTACT:    bool  = False  # True = only worlds that touched the object retry (train only)
    SUCCESS_EARLY_ENABLED:         bool = True
    SUCCESS_EARLY_LIFT_HOLD_STEPS: int  = 40   # hold-phase lifted streak → success (0 = strict only)
    OBS_EP_HORIZON_STEPS: int = 0              # ep_step_norm denominator; ≤0 → HOLD_STEP + SUCCESS_EARLY_LIFT_HOLD_STEPS
    _V2_SNAP_BUFS = ("ep_elapsed_wp", "drop_streak_wp", "grasp_progress_wp",
                     "retry_count_wp", "lift_hold_streak_wp")
    # BPS shape feature tail — RBF width (matches hand_utils.compute_bps_feature, γ=25)
    BPS_GAMMA: float = 25.0

    # ══════════════════════════════════════════════════════════════════════
    # Observation size / layout = core blocks + [student history] + BPS tail
    # ══════════════════════════════════════════════════════════════════════
    @property
    def obs_dim(self) -> int:
        return (int(self._core_obs_dim()) + int(getattr(self, "_n_bps", 0))
                + int(getattr(self, "_hist_w_total", 0)))

    def _bps_base_off(self) -> int:
        """Column where the BPS tail starts (after the core obs and, for the
        student, its history block)."""
        return int(self._core_obs_dim()) + int(getattr(self, "_hist_w_total", 0))

    @property
    def obs_term_layout(self):
        cached = getattr(self, "_obs_term_layout_cache_full", None)
        if cached is not None:
            return cached
        terms = list(self._core_obs_terms())
        off = int(terms[-1][2])
        nb = int(getattr(self, "_n_bps", 0))
        terms.append(("bps_feature", off, off + nb)); off += nb
        assert off == self.obs_dim, f"obs_term_layout total {off} != obs_dim {self.obs_dim}"
        self._obs_term_layout_cache_full = terms
        return terms

    def _load_bps_data(self) -> None:
        """Load the frozen hand-center-local basis points (``data/<Training.bps_data_path>``)."""
        cfg  = self.sampled_env.overall_cfg
        rel  = str(getattr(cfg.Training, "bps_data_path", "bps_basis_points.npz"))
        npts = int(getattr(cfg.Training, "n_bps", 1024))
        path = os.path.join(_HOME_DIR + "/data", rel)
        try:
            bps_np = np.load(path)["bps_data"]
        except Exception:
            print_red(f"[grasping] BPS data not found: {path} — using random ({npts}, 3) fallback")
            bps_np = np.random.uniform(-0.075, 0.15, size=(npts, 3))
        bps_np = np.ascontiguousarray(np.asarray(bps_np, dtype=np.float32).reshape(-1, 3))
        self._n_bps       = int(bps_np.shape[0])
        self.bps_local_np = bps_np
        self.bps_local_wp = wp.array(bps_np, dtype=wp.vec3, device=self.device)
        self.bps_gamma = float(getattr(cfg.Training, "bps_gamma", 25.0))
        # basis points below the support surface report 0 (unreachable region)
        self._bps_mask_below_z = float(getattr(self.sampled_env, "table_height", 0.0) or 0.0)

    def _setup_v2_buffers(self) -> None:
        z_i = lambda: wp.zeros((self.NWORLD,), dtype=int,   device=self.device)
        z_f = lambda: wp.zeros((self.NWORLD,), dtype=float, device=self.device)
        self.ep_elapsed_wp       = z_i()
        self.drop_streak_wp      = z_i()
        self.grasp_progress_wp   = z_i()
        self.retry_count_wp      = z_f();  self.retry_count_torch      = wp.to_torch(self.retry_count_wp)
        self.lift_hold_streak_wp = z_f();  self.lift_hold_streak_torch = wp.to_torch(self.lift_hold_streak_wp)
        self.retry_fired_wp      = z_f();  self.retry_fired_torch      = wp.to_torch(self.retry_fired_wp)
        self.early_success_wp    = z_f();  self.early_success_torch    = wp.to_torch(self.early_success_wp)
        self.ep_elapsed_torch    = wp.to_torch(self.ep_elapsed_wp)
        # diagnostics → reward_term/* (episode sums; retry_fired sum = #retries)
        self.metrics.update({
            "retry_count":      self.retry_count_torch,
            "retry_fired":      self.retry_fired_torch,
            "early_success":    self.early_success_torch,
            "lift_hold_streak": self.lift_hold_streak_torch,
        })
        self._v2_checked = False

    # v2 bookkeeping is part of the train state: eval must not leave
    # ``ep_elapsed`` at the eval-window end.
    def snapshot_train_state(self) -> None:
        super().snapshot_train_state()
        if not hasattr(self, "_v2_snap"):
            self._v2_snap = {n: wp.zeros_like(getattr(self, n)) for n in self._V2_SNAP_BUFS
                             if hasattr(self, n)}
        for n, buf in self._v2_snap.items():
            wp.copy(buf, getattr(self, n))

    # ══════════════════════════════════════════════════════════════════════
    # Observation collection — core blocks → tail → single-attempt clock
    # (teacher adds the critic obs after this; the student replaces the tail)
    # ══════════════════════════════════════════════════════════════════════
    def _collect_obs_kernel(self) -> None:
        self._before_core_obs()
        self._collect_core_obs_blocks()
        self._collect_obs_tail()
        self._apply_ep_norm_horizon()

    def _before_core_obs(self) -> None:
        """Hook run before the core obs blocks (no-op here; the student uses it)."""
        return

    def _collect_obs_tail(self) -> None:
        """Hook that fills the obs tail: full live BPS feature of the object."""
        self._launch_bps_tail(self.transformed_pcd_wp, self.obs_wp)

    def _launch_bps_tail(self, points_wp, out_wp) -> None:
        """BPS feature block = exp(-γ·min-dist) of each hand-centre basis point
        to the (world-frame) object PCD ``points_wp``, written at the tail of
        ``out_wp``. Reuses the reward pipeline's per-world PCD — no extra transform."""
        wp.launch(
            _obs_bps_feature_kernel,
            dim=(self.NWORLD, int(self._n_bps)),
            inputs=[
                self.d.xpos, self.d.xmat, points_wp, self.bps_local_wp,
                int(self.wrist_body_id), self._hand_center_offset_vec, self._hand_center_R_mat,
                int(self._pcd_n_pts), float(self.BPS_GAMMA), float(self._bps_mask_below_z),
                int(self._bps_base_off()), out_wp,
            ],
        )

    def obs_ep_horizon_steps(self) -> int:
        h = int(self.OBS_EP_HORIZON_STEPS)
        if h <= 0:
            h = int(self.HOLD_STEP) + int(self.SUCCESS_EARLY_LIFT_HOLD_STEPS)
        return max(h, 1)

    def _apply_ep_norm_horizon(self) -> None:
        """ep_step_norm := clamp(ep_step / T1, 0, 1) with T1 = one-attempt horizon
        (instead of ep_step / MAX_EPISODE_STEPS, whose scale depends on the retry budget)."""
        off = self._ep_stage_col()
        if off is None:
            return
        wp.launch(_obs_ep_norm_horizon_kernel, dim=self.NWORLD,
                  inputs=[self.ep_step_wp, int(self.obs_ep_horizon_steps()), int(off), self.obs_wp])

    def _ep_stage_col(self):
        """Column of ``ep_step_norm`` (first entry of the ``ep_stage`` block), or None."""
        if not self.INCLUDE_EP_STAGE_IN_OBS:
            return None
        off = getattr(self, "_ep_stage_off", None)
        if off is None:
            off = next((s for n, s, e in self.obs_term_layout if n == "ep_stage"), None)
            self._ep_stage_off = off
        return off

    # ══════════════════════════════════════════════════════════════════════
    # Re-grasp retry / early success / truncation
    # ══════════════════════════════════════════════════════════════════════
    def retry_budget_steps(self) -> int:
        """Smallest MAX_EPISODE_STEPS that fits RETRY_MAX full retry cycles."""
        lift_len = int(self.HOLD_STEP) - int(self.LIFT_STEP)
        cycle = int(self.RETRY_APPROACH_STEPS) + lift_len + int(self.RETRY_MIN_HOLD_STEPS)
        return int(self.LIFT_STEP) + lift_len + int(self.RETRY_MIN_HOLD_STEPS) \
            + int(self.RETRY_MAX) * cycle

    def _v2_check_once(self) -> None:
        if self._v2_checked:
            return
        self._v2_checked = True
        need = self.retry_budget_steps()
        if int(self.MAX_EPISODE_STEPS) < need:
            warnings.warn(
                f"[grasping] MAX_EPISODE_STEPS={self.MAX_EPISODE_STEPS} < retry budget {need} "
                f"(LIFT {self.LIFT_STEP} + lift {self.HOLD_STEP - self.LIFT_STEP} + hold {self.RETRY_MIN_HOLD_STEPS} "
                f"+ {self.RETRY_MAX}×(approach {self.RETRY_APPROACH_STEPS} + lift + hold)) — "
                f"the last retry may be cut short.")
        if float(self.DONE_OBJ_VEL_Z) > -3.0:
            warnings.warn(
                f"[grasping] DONE_OBJ_VEL_Z={self.DONE_OBJ_VEL_Z} is active — a drop terminates "
                f"the episode (obj_fell) before the retry can fire. Disable it (e.g. -15).")
        print(f"[grasping] retry ON={bool(self.RETRY_ENABLED)} max={self.RETRY_MAX} "
              f"approach={self.RETRY_APPROACH_STEPS} drop_k={self.RETRY_DROP_CONTACT_STEPS} "
              f"| early-success ON={bool(self.SUCCESS_EARLY_ENABLED)} lift_hold={self.SUCCESS_EARLY_LIFT_HOLD_STEPS} "
              f"strict={self.SUCCESS_STREAK_MIN} | budget need={need} / MAX={self.MAX_EPISODE_STEPS} "
              f"| obs ep_step_norm horizon T1={self.obs_ep_horizon_steps()}")

    def _launch_retry_early_success(self) -> None:
        self._v2_check_once()
        thr = float(self.RETRY_LIFT_THRESH_M)
        if thr < 0.0:
            thr = float(self.LIFT_SUCCESS_THRESH)
        # early success / contact-gated retry are TRAIN-only: eval keeps the
        # original protocol (play the whole window) so success rates stay comparable.
        early_on = bool(self.SUCCESS_EARLY_ENABLED) and not self.eval_mode
        wp.launch(
            _retry_early_success_kernel, dim=self.NWORLD,
            inputs=[
                self.d.xpos, self.cond.get("obj_p_init"),
                self.hand_obj_active_wp, self.bonus_streak_wp,
                int(self.obj_body_id),
                int(self.LIFT_STEP), int(self.HOLD_STEP), int(self.MAX_EPISODE_STEPS),
                int(1) if bool(self.RETRY_ENABLED) else int(0),
                int(self.RETRY_MAX), int(self.RETRY_APPROACH_STEPS),
                int(self.RETRY_MIN_LIFT_STEPS), int(self.RETRY_MIN_HOLD_STEPS),
                int(self.RETRY_DROP_CONTACT_STEPS), float(thr),
                int(1) if (bool(self.RETRY_REQUIRE_CONTACT) and not self.eval_mode) else int(0),
                int(1) if early_on else int(0),
                int(0) if self.eval_mode else int(1),
                int(self.SUCCESS_STREAK_MIN), float(self.LIFT_SUCCESS_THRESH),
                int(self.SUCCESS_EARLY_LIFT_HOLD_STEPS),
                float(self.LIFT_TARGET_M), float(self.SUCCESS_LIFT_TOL_FRAC),
                int(self.SUCCESS_REASON_CODE),
                self.ep_step_wp, self.ep_elapsed_wp,
                self.retry_count_wp, self.drop_streak_wp, self.grasp_progress_wp,
                self.lift_hold_streak_wp,
                self.lift_reached_wp,
                self.retry_fired_wp, self.early_success_wp,
                self.done_wp, self.done_mask_wp, self.done_reason_wp,
            ],
        )

    def _update_truncation(self) -> None:
        """Early success and elapsed-time timeout both bootstrap (time limit);
        only failure terminals (reason ∉ {success, timeout}) are true terminals."""
        done = self.done_mask_torch != 0
        reason = self.done_reason_torch
        is_trunc = done & ((reason == int(self.SUCCESS_REASON_CODE)) | (reason == 2)
                           | (self.ep_elapsed_torch >= int(self.MAX_EPISODE_STEPS)))
        self.truncation_torch.copy_(is_trunc.to(self.truncation_torch.dtype))
