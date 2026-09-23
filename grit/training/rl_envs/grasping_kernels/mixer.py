"""Reward mixer (weights · dt · clip in one pass) — warp kernels of the grasping task."""
import mujoco as _mj  # type: ignore
import numpy as np  # type: ignore
import warp as wp  # type: ignore


# ══════════════════════════════════════════════════════════════════════════
# Reward MIXER kernel — applies all weights, dt-scale, clip (single pass)
# ══════════════════════════════════════════════════════════════════════════
# Reads only per-world buffers — dominated by raw arithmetic. Keeping
# weights HERE lets the host tweak ``W_APPROACH / W_FINGER / W_MIMIC /
# W_WRIST_DIR / W_LIFT / W_SELF_COLL / W_TABLE /
# CTRL_COST / K_ACTION_SMOOTH / GRASP_BONUS_VALUE`` between iterations
# without touching upstream compute kernels.


@wp.kernel
def _object_grasping_reward_mix_kernel(
    # ── Raw inputs (per-world, weight-free) ───────────────────────────────
    raw_appraoch:         wp.array(dtype=float, ndim=1),  # type: ignore
    raw_finger:           wp.array(dtype=float, ndim=1),  # type: ignore
    mimic_qpos_err:       wp.array(dtype=float, ndim=1),  # type: ignore
    mimic_naive_err:      wp.array(dtype=float, ndim=1),  # type: ignore  Σ_rest|diff| without deadzone (constant naive pressure on rest ctrl)
    raw_mimic:            wp.array(dtype=float, ndim=1),  # type: ignore
    raw_wrist_dir:        wp.array(dtype=float, ndim=1),  # type: ignore
    raw_lift:             wp.array(dtype=float, ndim=1),  # type: ignore
    raw_vel:              wp.array(dtype=float, ndim=1),  # type: ignore  wrist stationarity coefficient
    raw_obj_vel:          wp.array(dtype=float, ndim=1),  # type: ignore  object stationarity coefficient
    raw_obj_xy_coeff:     wp.array(dtype=float, ndim=1),  # type: ignore  object xy-drift-vs-spawn coefficient
    raw_obj_R_coeff:      wp.array(dtype=float, ndim=1),  # type: ignore  object uprightness-vs-spawn coefficient
    raw_face_dir:         wp.array(dtype=float, ndim=1),  # type: ignore  active face↔object-look cosine ∈ [0, 1]
    raw_contact_dir_reward: wp.array(dtype=float, ndim=1),  # type: ignore  Σ frames·which_finger_touch / n_active
    raw_contact_ratio:    wp.array(dtype=float, ndim=1),  # type: ignore  (#active fingers touched)/(#active fingers) ∈ [0,1]
    torque_balance_coeff: wp.array(dtype=float, ndim=1),  # type: ignore  intra-finger motor torque balance ∈(0,1], 1 = uniform
    raw_contact_pos:      wp.array(dtype=float, ndim=1),  # type: ignore  Σ_active in_contact·cos⁺·w / Σ_active w ∈ [0,1]
    raw_contact_neg:      wp.array(dtype=float, ndim=1),  # type: ignore  Σ_rest scale·in_contact (penalty magnitude)
    raw_affordance_pos:   wp.array(dtype=float, ndim=1),  # type: ignore  Σ_active cos⁺·impulse·w / Σ_active w (weighted mean)
    raw_affordance_neg:   wp.array(dtype=float, ndim=1),  # type: ignore  Σ_rest impulse (penalty magnitude)
    raw_force_closure:    wp.array(dtype=float, ndim=1),  # type: ignore  exp(-k·‖G·f‖) ∈ [0,1] while in contact (force-closure residual)
    # slot-level taxonomy-compliance metrics (same formulas as the eval sem_contact_*), all ∈ [0,1]
    raw_sem_recall:       wp.array(dtype=float, ndim=1),  # type: ignore  |active∩contact|/|active|
    raw_sem_precision:    wp.array(dtype=float, ndim=1),  # type: ignore  |active∩contact|/|contact|
    raw_sem_f1:           wp.array(dtype=float, ndim=1),  # type: ignore  2PR/(P+R)
    raw_sem_iou:          wp.array(dtype=float, ndim=1),  # type: ignore  |∩|/|∪|
    action_sqnorm:        wp.array(dtype=float, ndim=1),  # type: ignore
    action_smooth_sqnorm: wp.array(dtype=float, ndim=1),  # type: ignore
    raw_self_contact_w:   wp.array(dtype=float, ndim=1),  # type: ignore  finger-weighted self-coll contact sum (drives W_SELF_COLL)
    raw_self_impulse_w:   wp.array(dtype=float, ndim=1),  # type: ignore  finger-weighted self-coll impulse mean (drives W_SELF_IMPULSE)
    raw_table_contact_w:  wp.array(dtype=float, ndim=1),  # type: ignore  finger-weighted obstacle (table+forearm) contact sum (drives W_TABLE_CONTACT)
    raw_table_impulse_w:  wp.array(dtype=float, ndim=1),  # type: ignore  finger-weighted obstacle impulse mean (drives W_TABLE_IMPULSE)
    obj_touch_force:      wp.array(dtype=float, ndim=1),  # type: ignore  Σ object touch sensors = force on object
    obj_weight_force:     wp.array(dtype=float, ndim=1),  # type: ignore  per-world (1−antigrav α)·m·g (force-scaled) — weight-based free allowance
    obj_ground_force:     wp.array(dtype=float, ndim=1),  # type: ignore  object↔table force estimate (aux kernel; total obj_touch − Σ hand slots)
    bonus_active:         wp.array(dtype=int,   ndim=1),  # type: ignore
    bonus_streak:         wp.array(dtype=int,   ndim=1),  # type: ignore READ — success kernel updates after
    ep_step:              wp.array(dtype=int,   ndim=1),  # type: ignore  for lift_step curriculum gate
    done_reason:          wp.array(dtype=int,   ndim=1),  # type: ignore  priority-resolved done reason (0=none); violation penalty gate
    # ── Curriculum + weights ──────────────────────────────────────────────
    lift_step:            int,
    w_approach:             float,
    w_finger:             float,
    w_mimic:              float,
    w_mimic_naive:        float,   # constant naive L1 pressure on rest ctrl, no deadzone (0 = off)
    w_wrist_dir:          float,
    w_lift:               float,
    w_self:               float,
    w_self_impulse:       float,
    w_tbl:                float,
    w_table_impulse:      float,
    w_obj_touch:          float,
    touch_pen_lift_scale: float,  # touch-penalty multiplier in the lift stage (ep_step>=lift_step) (1.0 = off, <1 → allows holding force). approach stays fully gentle
    # ── weight-based free allowance (fairer obj touch penalty) ────────────
    # Force that is 'legitimately needed' to lift the object is not penalized:
    #   free = k·(effective weight force) + alpha,  F_excess = max(0, ΣF − free)
    # k = margin on the required grip-force/weight ratio (friction, opposing-squeeze
    # geometry; 0 = filter off → legacy raw ΣF penalty), alpha = mass-independent
    # extra margin (force-scaled units).
    obj_touch_free_k:     float,
    obj_touch_free_alpha: float,
    w_obj_table_force:    float,  # object↔table collision penalty weight (0 = off) — separate axis from hand↔table (W_TABLE_*)
    obj_table_free_k:     float,  # free allowance = k·(object weight force) + alpha — the static load of resting on the table is free; only the "press/slam" excess is penalized
    obj_table_free_alpha: float,
    ctrl_cost:            float,
    k_action_smooth:      float,
    w_vel:                float,
    w_obj_vel:            float,
    w_obj_xy_coeff:       float,
    w_obj_R_coeff:        float,
    w_face_dir:           float,
    w_contact_dir:        float,
    w_contact_ratio:      float,
    contact_ratio_lift_boost: float,  # multiplier on the contact-ratio term of ineq_coeff in the lift stage (ep_step>=lift_step) (1.0 = off). approach is left as is (gentle); contact ratio is reinforced only while lifting
    w_torque_balance:     float,   # torque_regulate strength c ∈ [0,1] (0 = off)
    w_contact_pos:        float,
    w_contact_neg:        float,
    w_affordance_pos:     float,
    w_affordance_neg:     float,
    w_force_closure:      float,   # force-closure bonus weight (0 = off)
    # sem_contact reward weights (all 0 = off; raw ∈ [0,1], so W is the upper bound)
    w_sem_recall:         float,
    w_sem_precision:      float,
    w_sem_f1:             float,
    w_sem_iou:            float,
    w_bonus:              float,
    bonus_value:          float,
    bonus_streak_cap:     int,
    # Coverage coupling coefficient c ∈ [0,1]: the hold/lift bonus is multiplied by
    # [(1−c) + c·contact_ratio]. c=0 → unchanged (no coupling), c=1 → bonus fully
    # proportional to coverage. If the bonus that dominates the late reward is
    # independent of coverage, the ratio stalls at 2-3-finger grasps, so the
    # "more fingers on → larger bonus" pressure is applied directly in the hold
    # stage (a gate ramp from 0→c is recommended).
    w_cov_coupling:       float,
    # coverage exponent: cov = (1−c) + c·ratio^p. p=1 → linear (default). With p>1
    # the marginal gain of the last finger grows, so the bonus jumps when **all**
    # required fingers are on (e.g. p=3, c=1: ratio 0.8→0.51, 1.0→1.0, so 4/5→5/5
    # nearly doubles). Corrects the issue that with linear coupling alone the gain
    # of one extra finger is diluted by 1/n, leaving expensive-to-reach fingers
    # such as the little finger neglected.
    cov_exp:              float,
    # ── Lift-success bonus (JAX-convention: obj lifted > LIFT_SUCCESS_THRESH) ─
    w_lift_success_bonus: float,                          # per-step bonus when lifted past threshold
    lift_success_ratio:   float,                          # LIFT_SUCCESS_THRESH / LIFT_TARGET_M (raw_lift compare)
    # ── Violation terminal penalty (done reason ∉ {success, timeout}) ─────
    w_violation_pen:      float,
    timeout_reason_code:  int,
    success_reason_code:  int,
    # ── dt scale + clip ──────────────────────────────────────────────────
    dt:                   float,
    min_reward:           float,
    max_reward:           float,
    # ── Per-term outputs (signed contributions to ``out_reward``) ────────
    out_r_approach:       wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_finger:         wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_mimic:          wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_wrist_dir:      wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_lift:           wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_self_pen:       wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_self_impulse_pen: wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_table_pen:      wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_table_impulse_pen: wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_obj_touch_pen:  wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_obj_table_pen:  wp.array(dtype=float, ndim=1),  # type: ignore  object↔table collision penalty (≤0)
    out_r_violation_pen:  wp.array(dtype=float, ndim=1),  # type: ignore  dt-scaled penalty on violation-terminal steps
    out_r_bonus:          wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_lift_success_bonus: wp.array(dtype=float, ndim=1),  # type: ignore  +w when obj lifted past LIFT_SUCCESS_THRESH
    out_r_ctrl_cost:      wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_action_smooth:  wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_wrist_vel:            wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_obj_vel:        wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_obj_xy_coeff:   wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_obj_R_coeff:    wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_face_dir:       wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_contact_dir:    wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_contact_pos:    wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_contact_neg_pen: wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_affordance_pos: wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_affordance_neg_pen: wp.array(dtype=float, ndim=1),  # type: ignore
    out_r_force_closure:  wp.array(dtype=float, ndim=1),  # type: ignore  +w·exp(-k·‖Gf‖) (bonus group)
    out_r_sem_contact:    wp.array(dtype=float, ndim=1),  # type: ignore  Σ w_sem_*·sem_* (bonus group, one summed slot)
    # ── Composite / coefficient terms (the expressions that USE the per-term
    #    rewards above — exposed so each can be inspected/plotted) ──────────
    out_r_mimic_qpos_err:          wp.array(dtype=float, ndim=1),  # type: ignore  -w_mimic·mimic_qpos_err
    out_r_mimic_naive:             wp.array(dtype=float, ndim=1),  # type: ignore  -w_mimic_naive·mimic_naive_err
    out_hand_ineq_coeff:           wp.array(dtype=float, ndim=1),  # type: ignore  r_wrist_dir·r_wrist_vel·r_ctrl·r_smooth
    out_r_hand_process:            wp.array(dtype=float, ndim=1),  # type: ignore  (r_finger+r_face_dir+r_mimic_qpos_err)·hand_ineq_coeff
    out_obj_ineq_coeff:            wp.array(dtype=float, ndim=1),  # type: ignore  r_obj_vel·r_obj_xy_c·r_obj_R_c
    out_ineq_coeff:                wp.array(dtype=float, ndim=1),  # type: ignore  r_mimic·obj_ineq_coeff
    out_r_obj_process:             wp.array(dtype=float, ndim=1),  # type: ignore  r_lift·ineq_coeff
    out_r_hand_obj_bonus:          wp.array(dtype=float, ndim=1),  # type: ignore  (r_contact_dir+r_contact_pos+r_affordance_pos)·ineq_coeff
    out_r_hand_obj_penalty:        wp.array(dtype=float, ndim=1),  # type: ignore  Σ contact/impulse/touch penalties (subtracted)
    # ── Final per-step reward ────────────────────────────────────────────
    out_reward:           wp.array(dtype=float, ndim=1),  # type: ignore
):
    """Weighted sum + curriculum gating + dt-scale + clip.

    Lift reward is gated on ``ep_step >= lift_step`` (legacy curriculum).
    Bonus uses the same streak handling as the tracking task: the streak
    counter held *before* this step is bumped in the mixer's pre-success
    view so the per-step bonus already reflects the streak this step
    contributes to (capped at ``bonus_streak_cap``).

    ``r_wrist_vel = w_vel · raw_vel`` is a stationarity bonus — peaks at
    ``w_vel`` when the wrist is still and decays as the wrist moves
    (legacy multiplicative ``wrist_vel_coeff`` mapped to an additive term
    in this mixer's design).

    Violation terminal penalty: ``w_violation_pen`` is subtracted from
    ``total`` (before the dt-scale, like the other penalties) on a step whose
    ``done_reason`` is non-zero and ∉ {``timeout_reason_code``,
    ``success_reason_code``} — i.e. a violation done (obj_z / wrist_up / obj_xy /
    crush / obj_fell). Because the mixer runs BEFORE the success kernel, a
    success episode still reads ``timeout_reason_code`` here, so excluding
    timeout also excludes success.
    """
    w = wp.tid()

    # Hand process reward
    r_approach = w_approach * raw_appraoch[w] 
    r_finger   = w_finger   * raw_finger[w]
    r_face_dir = w_face_dir * raw_face_dir[w]          # +w_face_dir when faces point at obj
    r_contact_ratio = w_contact_ratio * raw_contact_ratio[w] 
    r_mimic_qpos_err = -w_mimic * mimic_qpos_err[w] 
    # naive mimic pressure (an additive simplification of the -0.1·Σ|diff| term in
    # JAX v2 controllable_process): provides a weak constant gradient toward the
    # template even 'inside' the deadzone band — a pose prior where the hinge term
    # is dormant.
    r_mimic_naive = -w_mimic_naive * mimic_naive_err[w]

        
    # Hand inequality constraint coefficient
    r_wrist_dir = w_wrist_dir * raw_wrist_dir[w]
    r_wrist_vel = w_vel       * raw_vel[w]         # +w_wrist_vel when wrist still
    r_ctrl     = wp.exp(-ctrl_cost       * action_sqnorm[w])
    r_smooth   = wp.exp(-k_action_smooth * action_smooth_sqnorm[w])
    hand_ineq_coeff = r_wrist_dir * r_wrist_vel * r_ctrl * r_smooth 

    # Object inequality constraint coefficient 
    r_obj_vel  = w_obj_vel        * raw_obj_vel[w]           # +w_obj_vel when object still
    r_obj_xy_c = w_obj_xy_coeff   * raw_obj_xy_coeff[w]      # +w when object near spawn xy
    r_obj_R_c  = w_obj_R_coeff    * raw_obj_R_coeff[w]       # +w when object upright as spawned
    obj_ineq_coeff = r_obj_vel * r_obj_xy_c * r_obj_R_c

    # intra-finger motor torque balance coefficient: when torque concentrates on one
    # joint, balance<1 → ineq_coeff is reduced, lowering that grasp's reward.
    # With c=0 the term is 1.0 (off).
    torque_regulate_term = (1.0 - w_torque_balance) + w_torque_balance * torque_balance_coeff[w]

    # Coverage coupling coefficient: hold/lift bonus ×[(1−c) + c·contact_ratio].
    # c=0 → 1.0 (default behaviour), c>0 → the bonus grows as more fingers make contact.
    cov_r = raw_contact_ratio[w]
    if cov_exp != 1.0:
        if cov_r < 0.0:
            cov_r = 0.0
        cov_r = wp.pow(cov_r, cov_exp)
    cov = (1.0 - w_cov_coupling) + w_cov_coupling * cov_r

    # Mimic inequality constraint coefficient.
    # # In the lift stage (ep_step>=lift_step) the contact-ratio term is scaled by
    # # boost — approach / initial grasp stay as is (gentle); only while lifting does
    # # the grasp reward grow with "more fingers on", maximizing the contact ratio.
    r_contact_ratio_eff = r_contact_ratio
    if ep_step[w] >= lift_step:
        r_contact_ratio_eff = r_contact_ratio * contact_ratio_lift_boost
    r_mimic     = raw_mimic[w]
    ineq_coeff  = r_mimic * obj_ineq_coeff * hand_ineq_coeff * r_contact_ratio_eff * torque_regulate_term
    
    r_hand_process = (r_approach + r_finger + r_face_dir + cov) * ineq_coeff 
    r_hand_reward = r_hand_process + r_mimic_qpos_err + r_mimic_naive + r_finger + r_wrist_dir

    # object process 
    # Lift gated by the curriculum step. Pre-lift_step the term is 0
    # regardless of weight — the policy first has to learn to approach
    # and close before being rewarded for the actual lift.
    r_lift = float(0.0)
    if ep_step[w] >= lift_step:
        r_lift = w_lift * raw_lift[w]
    r_obj_process = r_lift * r_hand_process 

    # Hand-Object Interaction bonus  
    r_contact_dir    = w_contact_dir    * raw_contact_dir_reward[w]   # signed face↔contact alignment

    r_contact_pos    = w_contact_pos    * raw_contact_pos[w]          # +active fingers in contact
    r_contact_neg    = w_contact_neg   * raw_contact_neg[w]          # -rest fingers in contact
    r_affordance_pos = w_affordance_pos * raw_affordance_pos[w]       # +active aligned impulse
    r_affordance_neg = w_affordance_neg * raw_affordance_neg[w]      # -rest impulse
    r_force_closure  = w_force_closure  * raw_force_closure[w]        # +wrench equilibrium (JAX v2 force_closure_reward)
    # taxonomy-compliance set metrics as a dense reward. Unlike contact_pos/ratio,
    # (a) they are slot-level, so they see link-level coverage directly, and (b) the
    # precision term penalizes rest contact as a 'ratio', keeping the scale invariant
    # for taxonomies with many links.
    r_sem_contact = (w_sem_recall    * raw_sem_recall[w]
                     + w_sem_precision * raw_sem_precision[w]
                     + w_sem_f1        * raw_sem_f1[w]
                     + w_sem_iou       * raw_sem_iou[w])

    # NOTE: these enter ``hand_object_interaction_penalty`` which is SUBTRACTED
    # from total, so each must be a POSITIVE magnitude (matching r_contact_neg /
    # r_affordance_neg, which are already positive via wft<0 × -w).
    r_self        = -w_self          * raw_self_contact_w[w]         # finger-weighted self-coll contact (penalty magnitude)
    r_self_imp    = -w_self_impulse  * raw_self_impulse_w[w]         # finger-weighted self-coll impulse  (penalty magnitude)
    r_table       = -w_tbl           * raw_table_contact_w[w]        # finger-weighted obstacle (table+forearm) contact (penalty magnitude)
    r_table_imp   = -w_table_impulse * raw_table_impulse_w[w]        # finger-weighted obstacle impulse   (penalty magnitude)

    # touch penalty: fully gentle during approach, ×touch_pen_lift_scale in the lift
    # stage (allows the force needed to hold the object against gravity, so the
    # gentle pressure does not block lift success).
    w_obj_touch_eff = w_obj_touch
    if ep_step[w] >= lift_step:
        w_obj_touch_eff = w_obj_touch * touch_pen_lift_scale
    # weight-based free allowance: force up to what lifting needs (k·mg_eff) + alpha
    # counts as legitimate grip force and is not penalized — only the excess gets the
    # gentle penalty (k=0 → legacy).
    f_touch = obj_touch_force[w]
    if obj_touch_free_k > 0.0:
        free = obj_touch_free_k * obj_weight_force[w] + obj_touch_free_alpha
        f_touch = f_touch - free
        if f_touch < 0.0:
            f_touch = 0.0
    r_obj_touch   = -w_obj_touch_eff * f_touch                       # excess force on object (penalty magnitude → subtracted; gentle grasp)

    # object↔table collision penalty — a **separate axis** from hand↔table
    # (r_table / r_table_imp). The signal is the aux kernel's obj_ground_force (total
    # object touch sensors − Σ hand slot forces) = force the table exerts on the
    # object. Penalizing the static load of resting on the table (≈mg) would make the
    # whole pre-grasp phase a constant penalty and push the object away, so, as with
    # the gentle penalty, the weight-based free allowance is subtracted and **only
    # the excess (pressing/slamming)** is penalized.
    f_ground = obj_ground_force[w]
    if obj_table_free_k > 0.0:
        free_g = obj_table_free_k * obj_weight_force[w] + obj_table_free_alpha
        f_ground = f_ground - free_g
        if f_ground < 0.0:
            f_ground = 0.0
    r_obj_table   = -w_obj_table_force * f_ground                    # object↔table collision (penalty magnitude → subtracted)
    hand_object_interaction_bonus = (r_contact_dir + r_contact_pos + r_affordance_pos + r_force_closure + r_sem_contact) * r_hand_process
    hand_object_interaction_penalty = r_contact_neg + r_affordance_neg + \
                                      r_self + r_self_imp + \
                                      r_table + r_table_imp + \
                                      r_obj_touch + r_obj_table
   


    streak_eff = bonus_streak[w] + 1
    if bonus_active[w] == 1:
        if streak_eff > bonus_streak_cap:
            streak_eff = bonus_streak_cap
    bonus = float(0.0)
    if bonus_active[w] > 0:
        bonus = w_bonus * bonus_value * float(streak_eff) * cov

    # Lift-success bonus (JAX-convention): flat +w per step once the object is
    # lifted past LIFT_SUCCESS_THRESH (raw_lift = clip(lift/target,0,1) >
    # threshold/target). Gated by the lift curriculum (same as r_lift) so it
    # does not reward incidental lifts during the approach stage.
    r_lift_success_bonus = float(0.0)
    if ep_step[w] >= lift_step:
        if raw_lift[w] > lift_success_ratio:
            r_lift_success_bonus = w_lift_success_bonus * cov

    out_r_approach[w]      = r_approach
    out_r_finger[w]        = r_finger
    out_r_face_dir[w]      = r_face_dir

    out_r_mimic[w]         = r_mimic
    out_r_wrist_dir[w]     = r_wrist_dir
    out_r_lift[w]          = r_lift
    out_r_self_pen[w]      = r_self
    out_r_self_impulse_pen[w] = r_self_imp
    out_r_table_pen[w]     = r_table
    out_r_table_impulse_pen[w] = r_table_imp
    out_r_obj_touch_pen[w] = r_obj_touch
    out_r_obj_table_pen[w] = r_obj_table
    out_r_bonus[w]         = bonus
    out_r_lift_success_bonus[w] = r_lift_success_bonus
    out_r_ctrl_cost[w]     = r_ctrl
    out_r_action_smooth[w] = r_smooth
    out_r_wrist_vel[w]     = r_wrist_vel
    out_r_obj_vel[w]       = r_obj_vel
    out_r_obj_xy_coeff[w]  = r_obj_xy_c
    out_r_obj_R_coeff[w]   = r_obj_R_c
    out_r_contact_dir[w]      = r_contact_dir
    out_r_contact_pos[w]      = r_contact_pos
    out_r_contact_neg_pen[w]  = r_contact_neg
    out_r_affordance_pos[w]   = r_affordance_pos
    out_r_affordance_neg_pen[w] = r_affordance_neg
    out_r_force_closure[w]    = r_force_closure
    out_r_sem_contact[w]      = r_sem_contact

    # Composite / coefficient terms (the expressions that USE the per-term
    # rewards) — exposed so each grouped contribution can be inspected.
    out_r_mimic_qpos_err[w]   = r_mimic_qpos_err
    out_r_mimic_naive[w]      = r_mimic_naive
    out_hand_ineq_coeff[w]    = hand_ineq_coeff
    out_r_hand_process[w]     = r_hand_process
    out_obj_ineq_coeff[w]     = obj_ineq_coeff
    out_ineq_coeff[w]         = ineq_coeff
    out_r_obj_process[w]      = r_obj_process
    out_r_hand_obj_bonus[w]   = hand_object_interaction_bonus
    out_r_hand_obj_penalty[w] = hand_object_interaction_penalty

    # Violation terminal penalty: subtract ``w_violation_pen`` on a done step
    # whose reason is NOT success and NOT timeout (i.e. obj_z / wrist_up /
    # obj_xy / crush / obj_fell). ``done_reason`` is non-zero only on a terminal
    # step; at mixer time a SUCCESS episode still reads ``timeout_reason_code``
    # (the success kernel promotes it afterwards), so excluding timeout also
    # excludes success. dt-scaled with the rest of ``total`` (per design).
    r_violation_pen = float(0.0)
    rc = done_reason[w]
    if rc != 0 and rc != timeout_reason_code and rc != success_reason_code:
        r_violation_pen = -w_violation_pen
    out_r_violation_pen[w] = r_violation_pen

    total = r_hand_reward + r_obj_process + hand_object_interaction_bonus + hand_object_interaction_penalty + bonus + r_lift_success_bonus + r_violation_pen
    total *= dt
    # NaN/Inf guard: an unstable contact (deep penetration → blown-up touch force,
    # or a NaN sensordata reading) can produce a non-finite ``total``. The min/max
    # comparisons below do NOT catch NaN (every NaN compare is False), so a NaN
    # would slip through into the reward → GAE → critic/advantages and poison the
    # whole rollout. Pin any non-finite reward to ``min_reward`` (treat as worst).
    if not (total == total) or total > 1.0e30 or total < -1.0e30:   # NaN or ±Inf
        total = min_reward
    if total < min_reward:
        total = min_reward
    if total > max_reward:
        total = max_reward
    out_reward[w] = total


@wp.kernel
def _object_grasping_taxonomy_gather_kernel(
    tax_row:       wp.array(dtype=int,   ndim=1),  # type: ignore (NWORLD,) -2 skip / -1 fallback / >=0 taxonomy row
    tax_qpos:      wp.array(dtype=float, ndim=2),  # type: ignore (n_tax, n_ctrl)
    tax_fmask:     wp.array(dtype=float, ndim=2),  # type: ignore (n_tax, n_sensors)
    tax_cmask:     wp.array(dtype=float, ndim=2),  # type: ignore (n_tax, n_ctrl)
    tax_fdir_idx:  wp.array(dtype=float, ndim=2),  # type: ignore (n_tax, n_sensors)
    tax_fdir_sgn:  wp.array(dtype=float, ndim=2),  # type: ignore (n_tax, n_sensors)
    def_fmask:     wp.array(dtype=float, ndim=1),  # type: ignore (n_sensors,)
    def_cmask:     wp.array(dtype=float, ndim=1),  # type: ignore (n_ctrl,)
    def_fdir_idx:  wp.array(dtype=float, ndim=1),  # type: ignore (n_sensors,)
    def_fdir_sgn:  wp.array(dtype=float, ndim=1),  # type: ignore (n_sensors,)
    n_ctrl:        int,
    n_sensors:     int,
    out_qpos:      wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, n_ctrl)
    out_fmask:     wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, n_sensors)
    out_cmask:     wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, n_ctrl)
    out_fdir_idx:  wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, n_sensors)
    out_fdir_sgn:  wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, n_sensors)
):
    """GPU taxonomy overlay — masked gather, replaces the host CPU↔GPU round-trip.

    Per world, ``tax_row[w]`` encodes the host's per-world taxonomy draw:
      * ``-2`` → world NOT being reset → leave all fields untouched.
      * ``-1`` → reset, "rolled uniform": write the DEFAULT masks/face-dirs;
                 ``target_qpos`` is left at its GPU-uniform sample.
      * ``>=0`` → reset, taxonomy row ``t``: gather row ``t`` of all 5 tables
                 into the cond fields (consistent active fingers + qpos
                 template + ctrl set + face dirs).

    The host only draws/uploads the small ``(NWORLD,)`` ``tax_row`` — no
    full-buffer mirror, no per-world Python loop, no forced GPU sync (replaces
    :meth:`GraspingHandler._apply_taxonomy_to_qpos_and_mask`'s old host
    overlay). Optional tables (ctrl-mask / face-dir) are pre-broadcast to the
    default row at setup, so the gather is always in-range.
    """
    w = wp.tid()
    t = tax_row[w]
    if t < -1:                               # -2 → not reset this step
        return
    if t >= 0:                               # taxonomy row → gather all 5 fields
        for i in range(n_ctrl):
            out_qpos[w, i]  = tax_qpos[t, i]
            out_cmask[w, i] = tax_cmask[t, i]
        for i in range(n_sensors):
            out_fmask[w, i]    = tax_fmask[t, i]
            out_fdir_idx[w, i] = tax_fdir_idx[t, i]
            out_fdir_sgn[w, i] = tax_fdir_sgn[t, i]
    else:                                    # -1 → uniform fallback: defaults (qpos untouched)
        for i in range(n_ctrl):
            out_cmask[w, i] = def_cmask[i]
        for i in range(n_sensors):
            out_fmask[w, i]    = def_fmask[i]
            out_fdir_idx[w, i] = def_fdir_idx[i]
            out_fdir_sgn[w, i] = def_fdir_sgn[i]
