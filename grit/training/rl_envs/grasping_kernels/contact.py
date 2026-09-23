"""Hand ↔ object contact OR-reduce (touch-gated) — warp kernels of the grasping task."""
import mujoco as _mj  # type: ignore
import numpy as np  # type: ignore
import warp as wp  # type: ignore


# ══════════════════════════════════════════════════════════════════════════
# Hand ↔ object contact OR-reduce (single family — touch-gated)
# ══════════════════════════════════════════════════════════════════════════
# Mirrors :func:`_hand_contact_active_kernel` but for the
# ``contact_sensor_for_object_index`` family alone. Kept inline here
# (instead of extending the shared kernel) so the tracking task isn't
# burdened with an obj family it doesn't read.


@wp.kernel
def _obj_contact_active_kernel(
    sensordata:            wp.array(dtype=float,    ndim=2),  # type: ignore
    n_obj_slots:           int,
    hand_obj_found_idx:         wp.array(dtype=int,      ndim=1),  # type: ignore
    hand_obj_touch_idx:         wp.array(dtype=int,      ndim=1),  # type: ignore
    touch_eps:             float,
    gate_by_touch:         int,
    force_scale:           float,                              # Training.force_scale_for_state
    out_obj_active:        wp.array(dtype=int,      ndim=1),  # type: ignore  1 if any slot fires
    out_obj_contact_impulse: wp.array(dtype=float,    ndim=1),  # type: ignore  max active-slot touch force
    out_obj_contact_pos:   wp.array(dtype=wp.vec3,  ndim=1),  # type: ignore  contact pos of the max-force slot (world frame)
    out_obj_contact_dir:   wp.array(dtype=wp.vec3,  ndim=1),  # type: ignore  normal of the max-force slot (world frame)
    out_obj_contact_per_slot: wp.array(dtype=float, ndim=2),  # type: ignore  (NWORLD, n_obj_slots) per-slot touch force
):
    """Hand ↔ object contact reduce + dominant-contact force / pos / direction.

    Beyond the boolean ``out_obj_active`` (1 iff any hand_part ↔ obj slot
    fires, touch-gated), this reports — for the single **highest-force
    accepted contact** this step (argmax over slots) — its force scalar,
    contact position and unit normal:

      * ``out_obj_contact_impulse`` — that contact's scalar normal force, read
        from the matching ``_touch`` sensor (``sensordata[w, touch_idx]``;
        MuJoCo touch = summed normal force). The contact-sensor packet
        itself carries only ``[found, pos, normal]`` (no force channel), so
        the force comes from the sibling touch sensor — the same one already
        used for ``gate_by_touch``. ``0`` when no contact is active (or when
        the firing slot has no paired touch sensor, ``touch_idx < 0``).
      * ``out_obj_contact_pos`` — that contact's world-frame position from
        the contact-sensor packet (``pos = sensordata[w, f_idx + 4 : +7]``).
        ``(0, 0, 0)`` when no contact is active.
      * ``out_obj_contact_dir`` — that contact's unit normal from the
        contact-sensor packet (``normal = sensordata[w, f_idx + 7 : +10]``),
        in the sensor's reported (world) frame. ``(0, 0, 0)`` when no
        contact is active.

    "Max force" is the per-world aggregation (the dominant contact); a
    ``has_pick`` guard lets the first accepted slot win outright so the
    pos / normal are still captured when every firing slot has zero / no
    touch force. Additionally ``out_obj_contact_per_slot[w, k]`` records the
    per-slot touch force individually (0 for inactive / gated-out slots), so
    the per-sensor hand↔object force breakdown is available, not just the
    dominant-contact aggregate.
    """
    w = wp.tid()
    active     = int(0)
    has_pick   = int(0)
    best_force = float(0.0)
    best_pos   = wp.vec3(0.0, 0.0, 0.0)
    best_dir   = wp.vec3(0.0, 0.0, 0.0)

    for k in range(n_obj_slots):
        slot_force = float(0.0)
        f_idx = hand_obj_found_idx[k]
        if sensordata[w, f_idx] > 0.5:
            t_idx = hand_obj_touch_idx[k]
            # Own contact force magnitude from the 10-float contact packet
            # [found, force(+1:4), pos(+4:7), normal(+7:10)] — each contact
            # type reports its OWN force (no shared touch sensor). The touch
            # sensor below is still consulted only as a garbage/uninit GATE.
            force = wp.length(wp.vec3(
                sensordata[w, f_idx + 1],
                sensordata[w, f_idx + 2],
                sensordata[w, f_idx + 3])) * force_scale

            accept = int(1)
            if gate_by_touch == 1:
                if t_idx >= 0:
                    if sensordata[w, t_idx] <= touch_eps:
                        accept = 0

            if accept == 1:
                active     = 1
                slot_force = force
                if (has_pick == 0) or (force > best_force):
                    best_force = force
                    # 10-float packet layout [found(0) force(1:4) pos(4:7) normal(7:10)]
                    best_pos   = wp.vec3(
                        sensordata[w, f_idx + 4],
                        sensordata[w, f_idx + 5],
                        sensordata[w, f_idx + 6],
                    )
                    best_dir   = wp.vec3(
                        sensordata[w, f_idx + 7],
                        sensordata[w, f_idx + 8],
                        sensordata[w, f_idx + 9],
                    )
                    has_pick = 1

        out_obj_contact_per_slot[w, k] = slot_force   # 0 for inactive / gated-out slots

    out_obj_active[w]        = active
    out_obj_contact_impulse[w] = best_force
    out_obj_contact_pos[w]   = best_pos
    out_obj_contact_dir[w]   = best_dir


@wp.kernel
def _contact_impulse_kernel(
    sensordata:    wp.array(dtype=float, ndim=2),     # type: ignore
    n_slots:       int,
    found_idx:     wp.array(dtype=int,   ndim=1),     # type: ignore
    touch_idx:     wp.array(dtype=int,   ndim=1),     # type: ignore
    touch_eps:     float,
    gate_by_touch: int,
    force_scale:   float,                              # Training.force_scale_for_state
    out_impulse:   wp.array(dtype=float, ndim=1),     # type: ignore  Σ active-slot scaled touch force
    out_per_slot:  wp.array(dtype=float, ndim=2),     # type: ignore  (NWORLD, n_slots) per-slot scaled touch force
):
    """Contact impulse for one sensor family (UNWEIGHTED diagnostic).

    Family-agnostic companion to the shared ``_hand_contact_active_kernel``
    (which produces only the binary ``*_active`` flags — and is shared with
    the tracking task, so its signature stays fixed). Launched once per
    family (self-collision, hand↔table, …). Produces BOTH:

      * ``out_impulse``  — per-world **sum** over all accepted slots (the
        total contact intensity this step).
      * ``out_per_slot`` — ``(NWORLD, n_slots)`` per-slot touch force, so each
        hand contact sensor's force is available individually (0 for inactive
        slots). Many downstream signals want the per-sensor breakdown, not
        just the aggregate.

    The contact-sensor packet (``[found, pos, normal]``) carries no force
    channel, so the scalar comes from the matching ``_touch`` sensor
    (``sensordata[w, touch_idx]``; MuJoCo touch = summed normal force). Accept
    / gate logic mirrors the shared kernel: a slot counts when its ``found``
    channel fires and (when ``gate_by_touch``) its touch reading exceeds
    ``touch_eps``. Slots with no paired touch sensor (``touch_idx < 0``)
    contribute 0. ``n_slots == 0`` writes only the (already-zeroed) sum.
    """
    w = wp.tid()
    total = float(0.0)
    for k in range(n_slots):
        slot_force = float(0.0)
        f_idx = found_idx[k]
        if sensordata[w, f_idx] > 0.5:
            t_idx = touch_idx[k]
            # Own contact force magnitude from the 10-float contact packet
            # [found, force(+1:4), pos(+4:7), normal(+7:10)] — each contact
            # type reports its OWN force (no shared touch sensor). The touch
            # sensor below is still consulted only as a garbage/uninit GATE.
            force = wp.length(wp.vec3(
                sensordata[w, f_idx + 1],
                sensordata[w, f_idx + 2],
                sensordata[w, f_idx + 3])) * force_scale

            accept = int(1)
            if gate_by_touch == 1:
                if t_idx >= 0:
                    if sensordata[w, t_idx] <= touch_eps:
                        accept = 0

            if accept == 1:
                slot_force = force
                total += force

        out_per_slot[w, k] = slot_force      # 0 for inactive / gated-out slots

    out_impulse[w] = total


@wp.kernel
def _touch_sum_kernel(
    sensordata:  wp.array(dtype=float, ndim=2),     # type: ignore
    n_slots:     int,
    force_scale: float,
    touch_adr:   wp.array(dtype=int,   ndim=1),     # type: ignore  sensordata addresses to sum
    out_touch:   wp.array(dtype=float, ndim=1),     # type: ignore  Σ touch-sensor scalars
):
    """Sum standalone touch-sensor scalars for one family (UNWEIGHTED).

    For sensor families that expose ONLY a touch sensor (no
    found/pos/normal contact sensor) — e.g. the fore-arm, whose touch
    sensor name ends with ``_arm_part_touch``. Each ``touch_adr[k]`` is a
    direct ``sensordata`` address (a MuJoCo touch sensor is a dim-1 scalar
    = summed normal force on its zone); we add them into a per-world total.
    No found/gate logic — the touch reading IS the contact magnitude, 0
    when nothing presses on the pad. ``n_slots == 0`` writes 0.
    """
    w = wp.tid()
    total = float(0.0)
    for k in range(n_slots):
        total += sensordata[w, touch_adr[k]] 
    out_touch[w] = total * force_scale


@wp.kernel
def _self_coll_weighted_penalty_kernel(
    sensordata:      wp.array(dtype=float, ndim=2),    # type: ignore
    n_slots:         int,
    found_idx:       wp.array(dtype=int,   ndim=1),    # type: ignore
    touch_idx:       wp.array(dtype=int,   ndim=1),    # type: ignore
    slot_weight:     wp.array(dtype=float, ndim=1),    # type: ignore  finger_contact_weights per slot
    touch_eps:       float,
    gate_by_touch:   int,
    force_scale:     float,                             # Training.force_scale_for_reward — scales the touch force
    out_contact_pen: wp.array(dtype=float, ndim=1),    # type: ignore  Σ accept·w   (weighted contact count)
    out_impulse_pen: wp.array(dtype=float, ndim=1),    # type: ignore  (Σ force·w)/n_slots  (weighted impulse mean)
):
    """Finger-weight-weighted self-collision contact + impulse penalties
    (UNWEIGHTED by the ``W_*`` knobs — mixer applies those).

    Warp port of the legacy JAX::

        self_collision_contact_penalty = sum(self_contacts_r_af · finger_weights_contact)
        self_collision_impulse_penalty = mean(self_impulses_r_af · finger_weights_contact)

    Per self-collision slot ``k`` (palm / fore-arm / finger link), weighted
    by that slot's ``finger_contact_weights`` value (sensor space — tip
    sites boosted, palm/fore-arm ×4) via the aligned ``slot_weight`` array:

      * ``out_contact_pen`` = Σ_k accept_k · w_k       — weighted count of
        active self-collision slots (legacy ``sum``).
      * ``out_impulse_pen`` = (Σ_k force_k · w_k) / n_slots — weighted mean
        self-collision impulse (legacy ``mean`` over the array — divides by
        the fixed slot count, not the active count).

    ``accept_k`` / ``force_k`` reuse the found + touch-gate logic of the
    other contact kernels: ``force_k = sensordata[touch_idx] · force_scale``
    (the matching ``_touch`` scalar scaled by ``force_scale_for_state`` so the
    penalty lives on the SAME force scale as the impulse state buffers / done
    crush-guard; 0 when no paired touch sensor). The gate compares the RAW
    touch reading — only the magnitude is scaled. Computed from this step's
    fresh ``sensordata`` (same timing as the contact-count term). The mixer
    applies the penalty sign + ``W_*``. ``n_slots == 0`` writes 0.
    """
    w = wp.tid()
    contact_sum = float(0.0)
    impulse_sum = float(0.0)
    for k in range(n_slots):
        f_idx = found_idx[k]
        if sensordata[w, f_idx] > 0.5:
            t_idx = touch_idx[k]
            # Own contact force magnitude from the 10-float contact packet
            # [found, force(+1:4), pos(+4:7), normal(+7:10)] — type-separated.
            # The touch sensor below is still consulted only as a GATE.
            force = wp.length(wp.vec3(
                sensordata[w, f_idx + 1],
                sensordata[w, f_idx + 2],
                sensordata[w, f_idx + 3])) * force_scale

            accept = int(1)
            if gate_by_touch == 1:
                if t_idx >= 0:
                    if sensordata[w, t_idx] <= touch_eps:
                        accept = 0

            if accept == 1:
                wv = slot_weight[k]
                contact_sum += wv
                impulse_sum += force * wv

    out_contact_pen[w] = contact_sum
    denom = float(n_slots)
    if denom < 1.0:
        denom = 1.0
    out_impulse_pen[w] = impulse_sum / denom


@wp.kernel
def _obstacle_coll_weighted_penalty_kernel(
    sensordata:        wp.array(dtype=float, ndim=2),    # type: ignore
    # ── table contact family (found / pos / normal + paired touch) ─────────
    n_table_slots:     int,
    table_found_idx:   wp.array(dtype=int,   ndim=1),    # type: ignore
    table_touch_idx:   wp.array(dtype=int,   ndim=1),    # type: ignore
    table_weight:      wp.array(dtype=float, ndim=1),    # type: ignore  finger_contact_weights per table slot
    # ── fore-arm touch-only family (touch sensor, no found channel) ────────
    n_forearm_slots:   int,
    forearm_touch_adr: wp.array(dtype=int,   ndim=1),    # type: ignore  sensordata addresses
    forearm_weight:    wp.array(dtype=float, ndim=1),    # type: ignore  finger_contact_weights per fore-arm slot
    force_scale:       float,                             # Training.force_scale_for_reward — scales both families' touch force
    touch_eps:         float,
    gate_by_touch:     int,
    out_contact_pen:   wp.array(dtype=float, ndim=1),    # type: ignore  Σ accept·w  (weighted contact count)
    out_impulse_pen:   wp.array(dtype=float, ndim=1),    # type: ignore  (Σ force·w)/n_total  (weighted impulse mean)
):
    """Finger-weight-weighted obstacle-collision penalties = hand↔table
    contact **plus** fore-arm touch, summed (UNWEIGHTED by ``W_*``).

    Sibling of :func:`_self_coll_weighted_penalty_kernel`, but the obstacle
    family spans two sensor kinds, both folded into one contact / impulse
    pair (legacy ``self_collision_*`` style applied to table + fore-arm):

      * **table contact slots** — found/pos/normal contact sensors with a
        paired ``_touch`` sensor. Same accept/gate logic as the other
        contact kernels; ``force = sensordata[touch_idx] · force_scale``
        (gate compares the RAW reading — only the magnitude is scaled).
      * **fore-arm touch slots** — touch-only (no found channel), so a
        contact is simply ``touch > touch_eps`` (RAW reading) and
        ``force = touch · force_scale``. Lets the fore-arm pad (which has no
        contact sensor) contribute to the obstacle penalty.

    Both families scale by ``force_scale_for_state`` so the obstacle impulse
    penalty lives on the same force scale as the impulse state buffers, and
    both read this step's fresh ``sensordata`` (same timing as the contact
    term — no obs-time lag).

    Each slot is weighted by its link's ``finger_contact_weights`` (the
    fore-arm slot uses the arm-row weight, ×4-boosted). Outputs::

        out_contact_pen = Σ_table accept·w  +  Σ_forearm (touch>eps)·w     (sum)
        out_impulse_pen = (Σ_table force·w + Σ_forearm force·w) / n_total  (mean)

    where ``n_total = n_table_slots + n_forearm_slots`` (fixed slot count,
    matching the legacy ``mean``). Either family may be empty (writes 0).
    """
    w = wp.tid()
    contact_sum = float(0.0)
    impulse_sum = float(0.0)

    # ── Table contact family (found + paired touch) ───────────────────────
    for k in range(n_table_slots):
        f_idx = table_found_idx[k]
        if sensordata[w, f_idx] > 0.5:
            t_idx = table_touch_idx[k]
            # Own contact force magnitude from the 10-float contact packet
            # [found, force(+1:4), pos(+4:7), normal(+7:10)] — type-separated.
            # The touch sensor below is still consulted only as a GATE.
            force = wp.length(wp.vec3(
                sensordata[w, f_idx + 1],
                sensordata[w, f_idx + 2],
                sensordata[w, f_idx + 3])) * force_scale

            accept = int(1)
            if gate_by_touch == 1:
                if t_idx >= 0:
                    if sensordata[w, t_idx] <= touch_eps:
                        accept = 0

            if accept == 1:
                wv = table_weight[k]
                contact_sum += wv
                impulse_sum += force * wv

    # ── Fore-arm touch-only family (contact = RAW touch > eps) ─────────────
    for k in range(n_forearm_slots):
        raw = sensordata[w, forearm_touch_adr[k]]      # gate on RAW reading
        if raw > touch_eps:
            wv = forearm_weight[k]
            contact_sum += wv
            # Scale the impulse magnitude to match the table family's scale.
            impulse_sum += raw * force_scale * wv

    out_contact_pen[w] = contact_sum
    denom = float(n_table_slots + n_forearm_slots)
    if denom < 1.0:
        denom = 1.0
    out_impulse_pen[w] = impulse_sum / denom

@wp.kernel
def _table_pen_grasp_gate_kernel(
    contact_ratio:  wp.array(dtype=float, ndim=1),  # type: ignore (NWORLD,) grasp coverage this step
    min_ratio:      float,                           # at or above this the grasp counts as "in progress"
    scale:          float,                           # table-penalty multiplier while grasping (0 = no penalty, 1 = gate off)
    contact_pen:    wp.array(dtype=float, ndim=1),  # type: ignore IN/OUT raw_table_contact_w
    impulse_pen:    wp.array(dtype=float, ndim=1),  # type: ignore IN/OUT raw_table_impulse_w
):
    """State-conditional table-penalty gate (allows the 'necessary contact' for flat objects).

    Table contact while a grasp is in progress (``contact_ratio >= min_ratio``) may be
    **necessary contact** for wrapping a flat object, so it is discounted by ``scale``;
    contact that uses the table without the object (grazing during approach) is kept in
    full. The raw_table_* buffers are reduced in place, so the mixer penalty,
    DONE_TABLE_IMPULSE and the lagrangian metric all see the discounted value
    **consistently**. The undiscounted diagnostics ``tbl_active``/``tbl_impulse`` from
    the separate kernel are left untouched.
    Note: to prevent the exploit of touching the object with one finger while pressing
    the table with the rest, the gate is based on the **ratio**, not on mere contact —
    min_ratio 0.5 = discount only when at least half of the active fingers are wrapped."""
    w = wp.tid()
    if contact_ratio[w] >= min_ratio:
        contact_pen[w] = contact_pen[w] * scale
        impulse_pen[w] = impulse_pen[w] * scale
