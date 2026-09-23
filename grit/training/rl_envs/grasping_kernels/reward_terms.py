"""Per-term reward kernels (weight-free — the mixer applies W_* / clip / dt) — warp kernels of the grasping task."""
import mujoco as _mj  # type: ignore
import numpy as np  # type: ignore
import warp as wp  # type: ignore
from .common import _grit_force_log1p


# ══════════════════════════════════════════════════════════════════════════
# Per-term compute kernels (UNWEIGHTED — mixer applies W_* / clip / dt)
# ══════════════════════════════════════════════════════════════════════════
#
# Each kernel computes a single geometric/physical signal and writes the
# raw RBF value (and an err diagnostic where useful). The mixer kernel is
# the sole place weights are multiplied, so `h.W_APPROACH = ...` etc. can
# be hot-swapped at any point without re-launching upstream compute.
# Same compute → mixer split as the shared tracking kernels
# (``_hand_pose_wrist_err_kernel`` / ``_hand_pose_qpos_err_kernel`` /
# ``_hand_af_xpos_compute_kernel``).
# ──────────────────────────────────────────────────────────────────────────


@wp.kernel
def _object_grasping_approach_err_kernel(
    xpos:                wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore (NWORLD, nbody)
    xmat:                wp.array(dtype=wp.mat33, ndim=2),  # type: ignore (NWORLD, nbody)
    wrist_body_id:       int,
    obj_body_id:         int,
    hand_center_offset:  wp.vec3,                            # rh_hand_center[:3, 3] — constant per hand
    k_approach:          float,
    out_approach_err:    wp.array(dtype=float, ndim=1),     # type: ignore center_dist (m)
    out_raw_approach:    wp.array(dtype=float, ndim=1),     # type: ignore exp(-k · center_dist)
):
    """Hand-center ↔ object xpos approach distance + RBF (UNWEIGHTED).

    Legacy (``grit_with_obj_coll_v2``)::

        hand_T   = wrist_T @ rh_hand_center           # SE3 composition
        center_dist = ||hand_T[:3, 3] - obj_xpos||    # L2 in world frame
        hand_approach_process = exp(-5 · center_dist · distance_smoothness_coeff)

    ``rh_hand_center`` is a 4×4 SE3 constant per hand (defined in
    ``hand_info/<hand>/rh_info.py`` — e.g. tesollo's
    ``rh_hand_center[:3, 3] = (0.066, 0.006, 0.163)``). The hand-center
    position in world frame is::

        p_hc(w) = R_wrist(w) · rh_hand_center[:3, 3] + wrist_p(w)

    See ``notebook/hand/01_hand_setup/10_viz_hand_center.ipynb`` for the
    CPU-side reference visualisation::

        rh_hand_center_T_calculated = wrist_T @ hand_util.rh_hand_center

    Output ``out_raw_approach = exp(-K_APPROACH · center_dist)`` — UNWEIGHTED
    (mixer applies ``W_APPROACH`` downstream). Distance is L2 in metres,
    so a typical ``K_APPROACH = 5`` gives ``raw ≈ 0.5`` at 14 cm.
    """
    w = wp.tid()
    R_wrist = xmat[w, wrist_body_id]
    wrist_p = xpos[w, wrist_body_id]
    obj_p   = xpos[w, obj_body_id]

    # World-frame hand-center position via wrist SE3 composition.
    hc_p = R_wrist * hand_center_offset + wrist_p

    err = wp.length(obj_p - hc_p)
    out_approach_err[w] = err
    out_raw_approach[w] = wp.exp(-k_approach * err)


@wp.kernel
def _object_grasping_transform_pcd_kernel(
    xpos:        wp.array(dtype=wp.vec3,  ndim=2),    # type: ignore (NWORLD, nbody)
    xmat:        wp.array(dtype=wp.mat33, ndim=2),    # type: ignore (NWORLD, nbody)
    pcd_local:   wp.array(dtype=wp.vec3,  ndim=3),    # type: ignore (n_var, n_obj, n_pts) body-local
    assignment:  wp.array(dtype=int,      ndim=1),    # type: ignore (NWORLD,)
    obj_idx:     int,                                  # graspable obj slot in renamed_obj_names
    obj_body_id: int,                                  # body id (constant across worlds — same skeleton)
    out_pcd_world: wp.array(dtype=wp.vec3, ndim=2),   # type: ignore (NWORLD, n_pts) world-frame PCD
):
    """Transform the per-world assigned object PCD from body-local → world.

    2D launch over ``(NWORLD, n_pts)``: thread ``(w, k)`` writes::

        out_pcd_world[w, k] = R_obj[w] · pcd_local[assignment[w], obj_idx, k] + p_obj[w]

    Hoists the rigid transform OUT of
    :func:`_object_grasping_nearest_pcd_kernel`. Previously each of that
    kernel's ``n_sensors`` per-world loops re-transformed every vertex, so the
    same point was transformed ``n_sensors`` times — now it is transformed
    ONCE per point and reused by all sensors (and by the observation kernel,
    which can read the world-frame PCD directly from ``out_pcd_world``).
    """
    w, k = wp.tid()
    R_obj = xmat[w, obj_body_id]
    p_obj = xpos[w, obj_body_id]
    v_idx = assignment[w]
    out_pcd_world[w, k] = R_obj * pcd_local[v_idx, obj_idx, k] + p_obj


@wp.kernel
def _object_grasping_nearest_pcd_hand_center_kernel(
    xpos:               wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore (NWORLD, nbody)
    xmat:               wp.array(dtype=wp.mat33, ndim=2),  # type: ignore (NWORLD, nbody)
    transformed_pcd:    wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore (NWORLD, n_pts) world-frame obj PCD
    wrist_body_id:      int,
    obj_body_id:        int,                                # best_pt default (p_obj) when n_pts == 0
    hand_center_offset: wp.vec3,                            # rh_hand_center[:3, 3] — constant per hand
    n_pts:              int,
    out_best_dist_hc:   wp.array(dtype=float, ndim=1),     # type: ignore (NWORLD,) L2 dist hand-center → nearest vertex
    out_best_pt_hc:     wp.array(dtype=wp.vec3, ndim=1),   # type: ignore (NWORLD,) nearest vertex (world)
):
    """Nearest object-PCD vertex to the HAND-CENTER (1 point per world).

    Mirror of :func:`_object_grasping_nearest_pcd_kernel`, but the query point
    is the hand-center (not the per-finger sites), so it is a 1D launch over
    ``NWORLD`` (no sensor dim). The hand-center world position is composed from
    the wrist pose + the constant ``rh_hand_center[:3, 3]`` offset — the same
    ``hc_p = R_wrist · offset + wrist_p`` used by
    :func:`_object_grasping_approach_err_kernel`. Reads the world-frame
    ``transformed_pcd`` already produced by
    :func:`_object_grasping_transform_pcd_kernel` (no re-transform).

    Writes TWO outputs:
      * ``out_best_pt_hc`` — nearest vertex world position → feeds the
        ``hand_center_pcd_err`` observation block (wrist-frame vector).
      * ``out_best_dist_hc`` — the L2 distance ``||best_pt − hc_p||`` (norm of
        the min squared distance) → consumed by the reward function.

    ``best`` / ``best_d2`` are seeded from the object centroid ``p_obj`` so a
    degenerate ``n_pts == 0`` still yields a meaningful distance
    (``||p_obj − hc_p||``) and a zero obs vector downstream."""
    w = wp.tid()
    R_wrist = xmat[w, wrist_body_id]
    wrist_p = xpos[w, wrist_body_id]
    hc_p = R_wrist * hand_center_offset + wrist_p
    best = xpos[w, obj_body_id]                  # best_pt default (n_pts == 0)
    d0 = best - hc_p
    best_d2 = wp.dot(d0, d0)                      # default dist² = ||p_obj − hc_p||²
    for k in range(n_pts):
        pcd_w = transformed_pcd[w, k]            # world-frame vertex (precomputed once)
        d = pcd_w - hc_p
        d2 = wp.dot(d, d)
        if d2 < best_d2:
            best_d2 = d2
            best = pcd_w
    
    out_best_dist_hc[w] = wp.sqrt(best_d2)       # L2 distance (norm) for the reward 
    out_best_pt_hc[w] = best
    

@wp.kernel
def _object_grasping_nearest_pcd_kernel(
    site_xpos:        wp.array(dtype=wp.vec3,  ndim=2),    # type: ignore (NWORLD, nsite)
    xpos:             wp.array(dtype=wp.vec3,  ndim=2),    # type: ignore (NWORLD, nbody) — p_obj default for best_pt
    transformed_pcd:  wp.array(dtype=wp.vec3,  ndim=2),    # type: ignore (NWORLD, n_pts) world-frame obj PCD
    finger_site_ids:  wp.array(dtype=int,      ndim=1),    # type: ignore (n_sensors,) palm + arm + fingers
    obj_body_id:      int,                                  # body id (for the p_obj best_pt default)
    n_sensors:        int,
    n_pts:            int,
    out_best_d2:      wp.array(dtype=float,   ndim=2),     # type: ignore (NWORLD, n_sensors) min squared dist
    out_best_pt:      wp.array(dtype=wp.vec3, ndim=2),     # type: ignore (NWORLD, n_sensors) nearest vertex (world)
):
    """Per-sensor nearest object-PCD vertex search — the SHARED scan.

    For each (world ``w``, sensor ``f``), find the object-PCD vertex closest to
    that sensor's finger **site** position and record BOTH the squared distance
    (``out_best_d2``) and the vertex world position (``out_best_pt``). Reads
    the **world-frame** PCD ``transformed_pcd[w, k]`` precomputed once by
    :func:`_object_grasping_transform_pcd_kernel` (so the rigid transform is no
    longer re-run per sensor). :func:`_object_grasping_finger_close_kernel`
    (consumes ``best_d2``) and :func:`_object_grasping_face_dir_kernel`
    (consumes ``best_pt``) then become cheap weight-free REDUCTIONS over these
    per-sensor results. ``transformed_pcd`` + the per-sensor buffers are also
    exposed for the observation kernel.

    Scans ALL ``n_sensors`` (no taxonomy-mask gating) so the buffers are valid
    for every sensor — active/rest masking happens in the downstream reduction
    kernels. ``best_pt`` defaults to ``p_obj`` (so a degenerate ``n_pts == 0``
    yields a zero look vector downstream).
    """
    # 2D launch over (NWORLD, n_sensors): thread (w, f) owns ONE sensor's
    # scan, so only the n_pts inner loop is serial (was n_sensors × n_pts per
    # world-thread under the old dim=NWORLD launch). Same min-search, byte-
    # identical output, ~n_sensors× more threads → much better GPU occupancy.
    w, f = wp.tid()
    p_obj   = xpos[w, obj_body_id]               # best_pt default (n_pts == 0)
    site_p  = site_xpos[w, finger_site_ids[f]]
    best_d2 = float(1.0e9)
    best_pt = p_obj
    for k in range(n_pts):
        pcd_w = transformed_pcd[w, k]            # world-frame vertex (precomputed once)
        diff  = pcd_w - site_p
        d2    = wp.dot(diff, diff)
        if d2 < best_d2:
            best_d2 = d2
            best_pt = pcd_w
    out_best_d2[w, f] = best_d2
    out_best_pt[w, f] = best_pt


@wp.kernel
def _object_grasping_finger_close_kernel(
    best_d2:          wp.array(dtype=float, ndim=2),       # type: ignore (NWORLD, n_sensors) min sq dist (from nearest_pcd)
    finger_weights:   wp.array(dtype=float, ndim=1),       # type: ignore (n_sensors,)
    site_mask:        wp.array(dtype=float, ndim=2),       # type: ignore (NWORLD, n_sensors)
    n_sensors:        int,
    k_finger:         float,
    out_finger_err:   wp.array(dtype=float, ndim=1),       # type: ignore link_dist (m²)
    out_raw_finger:   wp.array(dtype=float, ndim=1),       # type: ignore exp(-k · link_dist)
):
    """Palm + finger sites ↔ object PCD weighted MSE + RBF (UNWEIGHTED).

    Port of the JAX-era ``tip_link_process`` (from ``grit_with_obj_coll_v2``),
    with one deliberate change: the denominator includes the weights (proper
    WEIGHTED MEAN, like the face-dir kernel), where the legacy divided by the
    plain mask count::

        dist_info = per-link distances to nearest obj PCD vertex
        link_dist = sum(dist_info² · finger_weights · mask) / sum(finger_weights · mask)

    Why: ``finger_weights`` damps tip sites ×0.25–0.5 (``_build_finger_weights``),
    so a count denominator shrank link_dist 2–4× under tip-only
    (``do_finger_tip``) masks — ``raw_finger`` saturated with tips still far.
    The weighted mean is scale-invariant to the masked set's weight profile
    while keeping the proximal-dominates emphasis within the set.

    This is a cheap REDUCTION over the per-sensor nearest-PCD squared
    distances precomputed by :func:`_object_grasping_nearest_pcd_kernel` (the
    shared scan) — it no longer runs the ``n_pts`` brute-force search itself::

      link_dist = Σ_f best_d2[w, f] · finger_weights[f] · site_mask[w, f]
                / max(Σ_f finger_weights[f] · site_mask[w, f], 1e-6)
      out_raw_finger = exp(-K_FINGER · link_dist)   (UNWEIGHTED — mixer applies W_FINGER)

    The mask is sampled per world from
    ``taxonomy_specific_finger_body_mask_array`` (or all-ones fallback when
    the world's per-world taxonomy draw rolled "uniform"). Multiplying by
    ``mask`` zeros out fingers that aren't active for the chosen grasp
    taxonomy; the arm slot is also zero in every taxonomy mask, so the
    forearm site (sensor index 1) is automatically excluded without
    special-casing. ``best_d2`` is computed for ALL sensors upstream, so
    masked sensors contribute ``best_d2 · 0 = 0`` here.
    """
    w = wp.tid()

    sum_weighted_d2 = float(0.0)
    sum_w           = float(0.0)

    for f in range(n_sensors):
        mask_v = site_mask[w, f]
        w_v    = finger_weights[f] * mask_v
        sum_weighted_d2 += best_d2[w, f] * w_v   # nearest dist² from the shared scan kernel
        sum_w           += w_v                   # weighted-mean denominator (scale-invariant to tip damping)

    denom = sum_w
    if denom < 1.0e-6:
        denom = 1.0e-6
    link_dist = sum_weighted_d2 / denom

    out_finger_err[w] = link_dist
    out_raw_finger[w] = wp.exp(-k_finger * link_dist)


@wp.kernel
def _object_grasping_hand_center_close_kernel(
    best_dist_hc:        wp.array(dtype=float, ndim=1),   # type: ignore (NWORLD,) hand-center → nearest vertex L2 dist (m)
    k_hand_center:       float,
    out_raw_hand_center: wp.array(dtype=float, ndim=1),   # type: ignore exp(-k · dist) (UNWEIGHTED)
):
    """Hand-center ↔ nearest obj-PCD distance RBF (UNWEIGHTED — mixer applies W).

    Single-value analogue of :func:`_object_grasping_finger_close_kernel`: that
    kernel reduces the per-sensor squared distances to a weighted ``link_dist``
    before the RBF, whereas the hand-center distance is the already-computed
    scalar ``best_dist_hc`` (the L2 norm from
    :func:`_object_grasping_nearest_pcd_hand_center_kernel`), so this is a direct
    ``out_raw_hand_center = exp(-k_hand_center · best_dist_hc)`` ∈ (0, 1] — peaks
    at 1 when the hand-center sits on the object surface, decaying with distance.
    Distance is L2 in metres (like ``K_APPROACH``), so e.g. ``k = 5`` gives
    ``raw ≈ 0.5`` at 14 cm."""
    w = wp.tid()
    out_raw_hand_center[w] = wp.exp(-k_hand_center * best_dist_hc[w])


@wp.kernel
def _object_grasping_face_dir_kernel(
    site_xpos:        wp.array(dtype=wp.vec3,  ndim=2),    # type: ignore (NWORLD, nsite) finger site positions (look origin)
    xmat:             wp.array(dtype=wp.mat33, ndim=2),    # type: ignore (NWORLD, nbody) for R_body
    best_pt:          wp.array(dtype=wp.vec3,  ndim=2),    # type: ignore (NWORLD, n_sensors) nearest PCD vertex (from nearest_pcd)
    sensor_body_ids:  wp.array(dtype=int,      ndim=1),    # type: ignore (n_sensors,) palm + arm + fingers — for R_body
    finger_site_ids:  wp.array(dtype=int,      ndim=1),    # type: ignore (n_sensors,) parallel site ids — for site_p
    finger_weights:   wp.array(dtype=float,    ndim=1),    # type: ignore (n_sensors,)
    site_mask:        wp.array(dtype=float,    ndim=2),    # type: ignore (NWORLD, n_sensors) specific_finger_link_mask
    face_dir_idx:     wp.array(dtype=float,    ndim=2),    # type: ignore (NWORLD, n_sensors) col 0/1/2 (int-valued)
    face_dir_sign:    wp.array(dtype=float,    ndim=2),    # type: ignore (NWORLD, n_sensors) ±1
    n_sensors:        int,
    out_raw_face_dir: wp.array(dtype=float, ndim=1),       # type: ignore weighted mean max(cos, 0) ∈ [0, 1]
):
    """Active palm/finger face-direction ↔ object-look-direction cosine (UNWEIGHTED).

    Port of the legacy JAX ``hand_to_obj_cos_sim`` term. REDUCTION over the
    per-sensor nearest-PCD vertices precomputed by
    :func:`_object_grasping_nearest_pcd_kernel` (the shared scan) — no
    per-vertex search here. For every **active** sensor (palm + fingers; arm
    masked out), compare the body's taxonomy-selected "face" direction against
    the unit direction from the finger SITE toward its nearest object-PCD
    vertex (``best_pt``). ``R_body`` (face rotation) is the sensor **body**'s
    world rotation; the look origin ``site_p`` is the finger **site** — same
    site as :func:`_object_grasping_finger_close_kernel`. Rewards the hand
    presenting its active finger / palm faces *toward* the object::

        face_dir(w, f)  = sign · R_body[:, idx]                  # body rotation column
        look_dir(w, f)  = normalize(best_pt[w, f] - site_p)      # site → nearest vertex
        cos(w, f)       = max(<face_dir, look_dir>, 0)           # away → no credit
        raw_face_dir(w) = Σ_f cos·weight·mask / Σ_f weight·mask   ∈ [0, 1]

    ``idx`` / ``sign`` come from the per-world ``face_dir_idx_in_mat`` /
    ``face_dir_sign`` cond fields (sampled per taxonomy in
    :meth:`GraspingHandler._apply_taxonomy_to_qpos_and_mask`). The
    active set + per-sensor ``finger_weights`` reuse the SAME
    ``specific_finger_link_mask`` + weight arrays as the finger-close kernel,
    so the arm slot (``mask == 0``) and uniform-fallback worlds are handled
    identically. Sensors with ``weight <= 0`` ``continue`` — only the cheap
    per-active-sensor reduction runs here (the expensive scan was done once
    upstream for all sensors). The taxonomy face column is extracted as
    ``R_body · e_idx`` (one mat·vec, unit, no dynamic element index).
    """
    w = wp.tid()

    sum_cos  = float(0.0)
    sum_mask = float(0.0)

    for f in range(n_sensors):
        weight = finger_weights[f] * site_mask[w, f]
        if weight <= 0.0:
            continue                                    # arm + inactive fingers

        bid    = sensor_body_ids[f]
        site_p = site_xpos[w, finger_site_ids[f]]       # look origin (finger site)
        R_body = xmat[w, bid]                           # body rotation (taxonomy face column)

        # Face direction = signed column ``idx`` of the body rotation matrix.
        # Build the (signed) basis vector and multiply: R_body · (s·e_idx)
        # gives ``s · R_body[:, idx]`` — unit (rotation columns are unit),
        # no divergent dynamic element index.
        c  = int(face_dir_idx[w, f])
        s  = face_dir_sign[w, f]
        ex = float(0.0)
        ey = float(0.0)
        ez = float(0.0)
        if c == 0:
            ex = s
        elif c == 1:
            ey = s
        else:
            ez = s
        face = R_body * wp.vec3(ex, ey, ez)

        # Look direction: finger site → nearest PCD vertex (best_pt from the
        # shared _object_grasping_nearest_pcd_kernel — no per-vertex scan here).
        look = best_pt[w, f] - site_p
        ln   = wp.length(look)
        if ln > 1.0e-6:
            look = look / ln
            cos  = wp.dot(face, look)
            if cos < 0.0:
                cos = 0.0                               # face pointing away → no credit
            sum_cos  += cos * weight
            sum_mask += weight

    denom = sum_mask
    if denom < 1.0e-6:
        denom = 1.0e-6
    out_raw_face_dir[w] = sum_cos / denom


@wp.kernel
def _object_grasping_contact_dir_kernel(
    sensordata:       wp.array(dtype=float,    ndim=2),  # type: ignore (NWORLD, nsensordata)
    xmat:             wp.array(dtype=wp.mat33, ndim=2),  # type: ignore (NWORLD, nbody)
    n_obj_slots:      int,
    hand_obj_found_idx:    wp.array(dtype=int,      ndim=1),  # type: ignore (n_obj_slots,) hand↔obj contact-sensor found addr per slot
    hand_obj_touch_idx:    wp.array(dtype=int,      ndim=1),  # type: ignore (n_obj_slots,) hand↔obj contact-sensor paired touch addr (gate)
    hand_obj_sensor_idx:   wp.array(dtype=int,      ndim=1),  # type: ignore (n_obj_slots,) sensor-space index per slot (-1 = unmapped)
    sensor_body_ids:  wp.array(dtype=int,      ndim=1),  # type: ignore (n_sensors,) hand sensor body ids
    face_dir_idx:     wp.array(dtype=float,    ndim=2),  # type: ignore (NWORLD, n_sensors)
    face_dir_sign:    wp.array(dtype=float,    ndim=2),  # type: ignore (NWORLD, n_sensors)
    n_sensors:        int,
    touch_eps:        float,
    gate_by_touch:    int,
    out_cos:          wp.array(dtype=float, ndim=2),     # type: ignore (NWORLD, n_sensors) per-sensor signed cosine
):
    """Per-sensor face-direction ↔ object-contact-normal cosine (UNWEIGHTED).

    Port of the legacy JAX ``frames_r_sim`` term — sibling of
    :func:`_object_grasping_face_dir_kernel`, but the alignment target is the
    **actual contact direction** (object-contact sensor normal) at each
    finger. The result is kept as a **per-sensor array** (``out_cos[w, si]``,
    sensor space = palm / arm / fingers) instead of being reduced to a single
    scalar, so downstream reward terms can weight / mask / select sensors
    however they like::

        face_dir(w, si)    = sign · R_body[:, idx]            # taxonomy face-out
        contact_dir(w, si) = obj-contact sensor normal        # [found, pos, normal]
        out_cos[w, si]     = <face_dir, contact_dir>          # RAW signed (both unit)

    Each row is zeroed first, then every **active** obj-contact slot (found +
    optional touch-gate) writes its raw signed cosine at its sensor-space
    index ``hand_obj_sensor_idx`` (the firing body's ``face_dir_idx_in_mat`` /
    ``face_dir_sign`` column give the face). Non-contacting sensors and the
    arm slot (no obj sensor) stay 0. ``cos`` is RAW signed (the legacy
    ``frames_r_sim`` is an unclamped dot) — the contact-normal sign
    convention sets whether alignment reads + or −.
    """
    w = wp.tid()

    # Zero the per-sensor row first (non-contacting sensors → 0).
    for si in range(n_sensors):
        out_cos[w, si] = float(0.0)

    for k in range(n_obj_slots):
        f_idx = hand_obj_found_idx[k]
        if sensordata[w, f_idx] > 0.5:
            t_idx = hand_obj_touch_idx[k]
            accept = int(1)
            if gate_by_touch == 1:
                if t_idx >= 0:
                    if sensordata[w, t_idx] <= touch_eps:
                        accept = 0
            if accept == 1:
                si = hand_obj_sensor_idx[k]
                if si >= 0:
                    bid    = sensor_body_ids[si]
                    R_body = xmat[w, bid]
                    c  = int(face_dir_idx[w, si])
                    s  = face_dir_sign[w, si]
                    ex = float(0.0)
                    ey = float(0.0)
                    ez = float(0.0)
                    if c == 0:
                        ex = s
                    elif c == 1:
                        ey = s
                    else:
                        ez = s
                    face   = R_body * wp.vec3(ex, ey, ez)
                    # normal at packet offset +7..9 (10-float layout
                    # [found(0) force(1:4) pos(4:7) normal(7:10)]).
                    normal = wp.vec3(
                        sensordata[w, f_idx + 7],
                        sensordata[w, f_idx + 8],
                        sensordata[w, f_idx + 9],
                    )
                    out_cos[w, si] = wp.dot(face, normal)


@wp.kernel
def _object_grasping_contact_reward_kernel(
    sensordata:        wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, nsensordata)
    contact_dir_cos:   wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, n_sensors) per-sensor face↔contact cosine
    finger_link_mask:  wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, n_sensors) taxonomy active(1)/rest(0)
    n_obj_slots:       int,
    hand_obj_found_idx:     wp.array(dtype=int,   ndim=1),  # type: ignore (n_obj_slots,) hand↔obj contact-sensor found addr
    hand_obj_touch_idx:     wp.array(dtype=int,   ndim=1),  # type: ignore (n_obj_slots,) hand↔obj contact-sensor touch addr
    hand_obj_sensor_idx:    wp.array(dtype=int,   ndim=1),  # type: ignore (n_obj_slots,) sensor-space index per slot (-1 = unmapped)
    slot_weight:       wp.array(dtype=float, ndim=1),  # type: ignore (n_obj_slots,) per-slot contact weight (magnitude)
    slot_finger_id:    wp.array(dtype=int,   ndim=1),  # type: ignore (n_obj_slots,) finger-group id per slot (-1 = non-hand)
    n_finger_groups:   int,
    touch_eps:         float,
    gate_by_touch:     int,
    neg_contact_scale: float,                           # CONTACT_NEG_SCALE (legacy 3.0)
    force_scale:       float,                           # Training.force_scale_for_reward — scales the touch impulse
    pos_sum_mode:      int,                             # 1 → contact_pos/aff_pos as a SUM instead of a weighted mean (JAX v2)
    pos_cos_binary:    int,                             # 1 → [cos>0] binary gate instead of the continuous cos⁺ gate (JAX v2)
    out_dir_reward:        wp.array(dtype=float, ndim=1),  # type: ignore  Σ frames·which_finger_touch / use_finger_num
    out_contact_pos:       wp.array(dtype=float, ndim=1),  # type: ignore  Σ_active contact·[cos>0]
    out_contact_neg:       wp.array(dtype=float, ndim=1),  # type: ignore  Σ_rest scale·contact
    out_affordance_pos:    wp.array(dtype=float, ndim=1),  # type: ignore  Σ_active [cos>0]·impulse
    out_affordance_neg:    wp.array(dtype=float, ndim=1),  # type: ignore  Σ_rest impulse
    out_contact_ratio:     wp.array(dtype=float, ndim=1),  # type: ignore  (#active in_contact)/(#active) ∈ [0,1]
    # ── slot-level taxonomy-compliance semantic metrics (same formulas as the eval sem_contact_*) ──
    out_sem_recall:        wp.array(dtype=float, ndim=1),  # type: ignore  |active∩contact| / |active|
    out_sem_precision:     wp.array(dtype=float, ndim=1),  # type: ignore  |active∩contact| / |contact|
    out_sem_f1:            wp.array(dtype=float, ndim=1),  # type: ignore  2PR/(P+R)
    out_sem_iou:           wp.array(dtype=float, ndim=1),  # type: ignore  |active∩contact| / |active∪contact|
):
    """Taxonomy-aware contact-affordance rewards (UNWEIGHTED by ``W_*``).

    Port of the legacy JAX contact block. Per obj-contact slot (a palm /
    finger link in contact with the object), the taxonomy splits sensors into
    **active** (this grasp class SHOULD use them) vs **rest** (should NOT),
    via ``finger_link_mask`` — the sensor-space taxonomy mask (same role as
    ``specific_ctrl_mask`` in the mimic kernel, but in sensor space since the
    contact signals are sensor-indexed). ``which_finger_touch`` is then the
    signed magnitude ``±slot_weight`` (active → +, rest → −). Five separate
    reward components are produced (each weighted / summed by the caller)::

        which_finger_touch = (active ? +1 : -1) · slot_weight
        gate_k         = cos⁺_k                (default; pos_cos_binary=1 → [cos>0] binary)
        dir_reward     = Σ_k frames_k · which_finger_touch_k / use_finger_num
        contact_pos    = Σ_{active} in_contact_k · gate_k · w_k [/ Σ_{active} w_k]  (denominator only when pos_sum_mode=0)
        contact_neg    = Σ_{rest}    neg_contact_scale · in_contact_k
        affordance_pos = Σ_{active} gate_k · impulse_k · w_k [/ Σ_{active} w_k]     (likewise)
        affordance_neg = Σ_{rest}    impulse_k
        contact_ratio  = (#active FINGERS touched) / (#active FINGERS)   ∈ [0, 1]

    ``pos_sum_mode=1`` + ``pos_cos_binary=1`` = the original JAX v2 form: the full +w
    per contacting slot is summed, so the marginal gain of one 'additional' finger is
    not diluted (the weighted-mean paragraph below describes the default pos_sum_mode=0).

    ``contact_pos`` / ``affordance_pos`` are WEIGHTED MEANS over the active
    set (denominator Σ active slot weights), not sums: a raw sum scales with
    the taxonomy's active-slot count (power ≈ 16 slots vs tip_pinch = 2), so
    the attainable bonus ceiling differed per taxonomy. Normalising makes
    both taxonomy-scale-invariant (``contact_pos`` ∈ [0, 1]); the penalty
    sums (``contact_neg`` / ``affordance_neg``) stay per-event sums — each
    wrong contact is individually bad regardless of taxonomy size.

    ``contact_ratio`` is a plain coverage signal — no face-cosine gate, no
    ``slot_weight`` — counted in **finger units** via ``slot_finger_id``
    (palm / per-finger groups, int-bitmask accumulation, ≤ 30 groups): a unit
    is active when ANY of its slots is taxonomy-active, touched when ANY
    active slot is in contact. Finger units keep the ratio scale comparable
    across taxonomies with different active-link counts; for
    ``do_finger_tip`` taxonomies the mask itself is tip-restricted (see
    ``get_hand_sensor_site_id_list``), so touched ⇢ TIP contact only.

    where ``frames_k`` = per-sensor face↔contact-normal cosine
    (``contact_dir_cos``; 0 when not in contact), ``in_contact`` = obj-contact
    found (+ touch gate), ``impulse`` = obj-contact touch force × ``force_scale``, and
    ``use_finger_num`` = count of active sensors (≥ 1). ``slot_weight`` is the
    contact-reward weighting (``_hand_only_contact_weight_wp``); it enters only
    via ``which_finger_touch`` (dir_reward), matching the legacy — the other
    four use the active/rest split as a gate. The negative components are
    *penalty magnitudes* (the caller applies the minus sign).
    """
    w = wp.tid()
    dir_sum     = float(0.0)
    contact_pos = float(0.0)
    contact_neg = float(0.0)
    aff_pos     = float(0.0)
    aff_neg     = float(0.0)
    n_active    = float(0.0)
    n_active_w  = float(0.0)   # Σ active slot weights (contact_pos/aff_pos weighted-mean denominator)
    active_bits  = int(0)   # finger groups with ≥1 taxonomy-active slot
    touched_bits = int(0)   # finger groups with ≥1 active slot in contact
    # sem_contact: slot-level confusion counts (pure set operations without weight/cos
    # gates — intended to match the definitions in eval_semantic_metrics).
    n_act_slot   = float(0.0)   # |active|
    n_con_slot   = float(0.0)   # |contact|  (active + rest)
    n_inter_slot = float(0.0)   # |active ∩ contact|

    for k in range(n_obj_slots):
        si = hand_obj_sensor_idx[k]
        if si < 0:
            continue

        active = int(0)
        if finger_link_mask[w, si] > 0.5:
            active = 1

        weight = slot_weight[k]
        wft    = weight                                # which_finger_touch (signed)
        if active == 0:
            wft = -weight

        frames     = contact_dir_cos[w, si]
        frames_pos = float(0.0)
        if frames > 0.0:
            frames_pos = frames


        # contact flag (found + optional touch gate)
        in_contact = float(0.0)
        f_idx = hand_obj_found_idx[k]
        if sensordata[w, f_idx] > 0.5:
            acc   = int(1)
            t_idx = hand_obj_touch_idx[k]
            if gate_by_touch == 1:
                if t_idx >= 0:
                    if sensordata[w, t_idx] <= touch_eps:
                        acc = 0
            if acc == 1:
                in_contact = 1.0

        # impulse = this contact's OWN force magnitude from the 10-float packet
        # [found, force(+1:4), pos, normal], × force_scale so the affordance
        # impulse lives on the same scale as the self/table impulse penalties
        # and the hand↔obj impulse state. Gated by ``in_contact`` (found + touch
        # gate) so uninitialised/garbage packets contribute 0.
        impulse = float(0.0)
        if in_contact > 0.5:
            impulse = wp.length(wp.vec3(
                sensordata[w, f_idx + 1],
                sensordata[w, f_idx + 2],
                sensordata[w, f_idx + 3])) * force_scale

        # sem_contact confusion counts (no cos gate — contact only).
        if active == 1:
            n_act_slot += 1.0
        if in_contact > 0.5:
            n_con_slot += 1.0
            if active == 1:
                n_inter_slot += 1.0

        # dir_reward over all fingers (signed-weighted alignment).
        dir_sum += frames * wft

        # alignment gate: default = continuous cos⁺ (partial credit), JAX v2 mode =
        # [cos>0] binary (full credit once touching with the right direction — the
        # ring and little fingers, which are hard to align, get full credit on contact).
        pos_gate = frames_pos
        if pos_cos_binary == 1:
            pos_gate = float(0.0)
            if frames > 0.0:
                pos_gate = 1.0

        if active == 1:
            n_active    += 1.0
            n_active_w  += weight
            g = slot_finger_id[k]                      # finger-unit coverage (no cos gate / weight)
            if g >= 0:
                b = 1 << g
                active_bits = active_bits | b
                if in_contact > 0.5:
                    touched_bits = touched_bits | b
            contact_pos += in_contact * pos_gate * wft
            aff_pos     += impulse * pos_gate * wft
        else:
            contact_neg += neg_contact_scale * in_contact * wft
            aff_neg     += impulse * wft

    denom = n_active
    if denom < 1.0:
        denom = 1.0
    denom_w = n_active_w
    if denom_w < 1.0e-6:
        denom_w = 1.0e-6
    out_dir_reward[w]     = dir_sum / denom
    if pos_sum_mode == 1:
        # JAX v2 SUM mode: the marginal gain of one 'additional' finger stays at the
        # full +w — removes the 1/Σw dilution of the weighted mean (the main cause of
        # neglected ring/little fingers). The scale then depends on taxonomy size, so
        # W_* must be recalibrated (JAX parity is W_CONTACT_POS=5.0).
        out_contact_pos[w]    = contact_pos
        out_affordance_pos[w] = aff_pos
    else:
        out_contact_pos[w]    = contact_pos / denom_w   # weighted mean ∈ [0, 1] (taxonomy-scale-invariant)
        out_affordance_pos[w] = aff_pos / denom_w       # weighted mean impulse
    out_contact_neg[w]    = contact_neg
    out_affordance_neg[w] = aff_neg
    # fraction of active FINGERS (per-finger units, not link slots) in contact
    n_act_f = float(0.0)
    n_tch_f = float(0.0)
    for g in range(n_finger_groups):
        if ((active_bits >> g) & 1) == 1:
            n_act_f += 1.0
            if ((touched_bits >> g) & 1) == 1:
                n_tch_f += 1.0
    ratio = float(0.0)
    if n_act_f > 0.5:
        ratio = n_tch_f / n_act_f
    out_contact_ratio[w]  = ratio

    # ── sem_contact_* (same formulas/eps as eval_semantic_metrics) ──────
    # recall    = fraction of required links actually touched (neglected ring/little = recall↓)
    # precision = fraction of touched links that are required (rest-finger contact = precision↓)
    # f1 / iou  = harmonic / set summaries of the two. All ∈ [0,1], so invariant to taxonomy size.
    sem_eps = float(1.0e-6)
    sem_p   = n_inter_slot / (n_con_slot + sem_eps)
    sem_r   = n_inter_slot / (n_act_slot + sem_eps)
    sem_f1  = 2.0 * sem_p * sem_r / (sem_p + sem_r + sem_eps)
    sem_iou = n_inter_slot / (n_act_slot + n_con_slot - n_inter_slot + sem_eps)
    out_sem_precision[w] = sem_p
    out_sem_recall[w]    = sem_r
    out_sem_f1[w]        = sem_f1
    out_sem_iou[w]       = sem_iou


@wp.kernel
def _object_grasping_force_closure_kernel(
    sensordata:          wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, nsensordata)
    xpos:                wp.array(dtype=wp.vec3, ndim=2),  # type: ignore (NWORLD, nbody)
    n_obj_slots:         int,
    hand_obj_found_idx:  wp.array(dtype=int,   ndim=1),  # type: ignore (n_obj_slots,)
    hand_obj_touch_idx:  wp.array(dtype=int,   ndim=1),  # type: ignore (n_obj_slots,)
    obj_body_id:         int,
    touch_eps:           float,
    gate_by_touch:       int,
    force_scale:         float,                           # force_scale_for_reward
    k_force_closure:     float,
    out_fc_res:          wp.array(dtype=float, ndim=1),  # type: ignore ‖G·f‖ wrench residual (force-scaled)
    out_raw_fc:          wp.array(dtype=float, ndim=1),  # type: ignore exp(-k·res) gated on ≥1 contact
):
    """Force-closure wrench residual (port of the JAX v2 ``force_closure_reward``).

    For each hand↔obj contact slot, read the contact force vector F_k and contact
    point p_k from the 10-float packet [found, force(1:4), pos(4:7), normal(7:10)]
    and assemble the net wrench (G·f = [Σ F_k ; Σ (p_k − p_obj) × F_k])::

        res     = ‖[Σ F_k, Σ r_k × F_k]‖        (force/torque equilibrium residual; 0 = equilibrium)
        raw_fc  = exp(-k · res)   (when ≥ 1 contact; no contact → 0)

    Difference from the JAX version: JAX assembled voxel-deduplicated contact points
    with normal direction × force magnitude, whereas here the actual force vector of
    the contact packet is used directly (more accurate). Multiplied by ``force_scale``
    (=force_scale_for_reward 0.1) to share the scale of the other impulse terms —
    with k=1.0 this is equivalent to the JAX exp(-0.1·raw). Grasps that wrap the
    center of mass with cancelling forces (thumb opposing index/ring/little) have a
    smaller residual and a larger reward — grasps with force concentrated on one side
    (index/middle only) have a large residual."""
    w = wp.tid()
    F_sum = wp.vec3(0.0, 0.0, 0.0)
    T_sum = wp.vec3(0.0, 0.0, 0.0)
    n_con = int(0)
    p_obj = xpos[w, obj_body_id]
    for k in range(n_obj_slots):
        f_idx = hand_obj_found_idx[k]
        if sensordata[w, f_idx] > 0.5:
            accept = int(1)
            if gate_by_touch == 1:
                t_idx = hand_obj_touch_idx[k]
                if t_idx >= 0:
                    if sensordata[w, t_idx] <= touch_eps:
                        accept = 0
            if accept == 1:
                F = wp.vec3(
                    sensordata[w, f_idx + 1],
                    sensordata[w, f_idx + 2],
                    sensordata[w, f_idx + 3]) * force_scale
                p = wp.vec3(
                    sensordata[w, f_idx + 4],
                    sensordata[w, f_idx + 5],
                    sensordata[w, f_idx + 6])
                F_sum += F
                T_sum += wp.cross(p - p_obj, F)
                n_con += 1
    res = wp.sqrt(wp.dot(F_sum, F_sum) + wp.dot(T_sum, T_sum))
    out_fc_res[w] = res
    raw = float(0.0)
    if n_con > 0:
        raw = wp.exp(-k_force_closure * res)
    out_raw_fc[w] = raw


@wp.kernel
def _object_grasping_obj_contact_aux_kernel(
    sensordata:          wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, nsensordata)
    n_obj_slots:         int,
    hand_obj_found_idx:  wp.array(dtype=int,   ndim=1),  # type: ignore (n_obj_slots,)
    hand_obj_touch_idx:  wp.array(dtype=int,   ndim=1),  # type: ignore (n_obj_slots,)
    hand_obj_sensor_idx: wp.array(dtype=int,   ndim=1),  # type: ignore (n_obj_slots,) sensor-space idx (-1 = unmapped)
    finger_link_mask:    wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, n_sensors) taxonomy active(1)/rest(0)
    obj_touch_force:     wp.array(dtype=float, ndim=1),  # type: ignore (NWORLD,) Σ obj touch sensors (fresh, same step)
    touch_eps:           float,
    gate_by_touch:       int,
    force_scale:         float,                           # force_scale_for_state (same scale as obj_touch_force)
    out_rest_impulse_max: wp.array(dtype=float, ndim=1), # type: ignore max contact force over rest (non-taxonomy) slots
    out_obj_ground_force: wp.array(dtype=float, ndim=1), # type: ignore ≈ force the floor/table exerts on the object
):
    """Rest-crush / obj-ground force signals (consumed by done + obs; JAX v2 port).

    * ``rest_impulse_max`` — slot maximum of the contact force that hand bodies the
      taxonomy says not to use (rest) apply to the object. Corresponds to the JAX v2
      ``masked_impulse > 15 → done`` (immediate abort on pressing with the wrong
      fingers). Reads sensordata directly so it matches the same-step physics (the
      per-slot buffers are on the obs-collect timing and lag by one step).
    * ``obj_ground_force`` — residual ``max(0, obj_touch − Σ hand)`` of the total
      object touch sensor (all contact sources) minus the sum of hand↔obj slot forces.
      The Grit scene has no dedicated obj↔floor contact sensor, so this residual
      approximates the JAX v2 ``obj_table_impulse`` (spikes on a floor slam; the
      hand's pressing force also enters the hand sum and cancels). Both terms are on
      the force_scale_for_state scale."""
    w = wp.tid()
    rest_max = float(0.0)
    hand_sum = float(0.0)
    for k in range(n_obj_slots):
        f_idx = hand_obj_found_idx[k]
        if sensordata[w, f_idx] > 0.5:
            accept = int(1)
            if gate_by_touch == 1:
                t_idx = hand_obj_touch_idx[k]
                if t_idx >= 0:
                    if sensordata[w, t_idx] <= touch_eps:
                        accept = 0
            if accept == 1:
                f = wp.length(wp.vec3(
                    sensordata[w, f_idx + 1],
                    sensordata[w, f_idx + 2],
                    sensordata[w, f_idx + 3])) * force_scale
                hand_sum += f
                si = hand_obj_sensor_idx[k]
                if si >= 0:
                    if finger_link_mask[w, si] < 0.5:      # rest slot
                        if f > rest_max:
                            rest_max = f
    out_rest_impulse_max[w] = rest_max
    ground = obj_touch_force[w] - hand_sum
    if ground < 0.0:
        ground = 0.0
    out_obj_ground_force[w] = ground


@wp.kernel
def _obs_obj_ground_force_kernel(
    obj_ground_force: wp.array(dtype=float, ndim=1),  # type: ignore (NWORLD,)
    ep_step:          wp.array(dtype=int,   ndim=1),  # type: ignore (NWORLD,)
    lift_step:        int,                             # <0 → no gating (always raw)
    log1p_cap:        float,
    inv_log1p_cap:    float,
    offset:           int,
    out_obs:          wp.array(dtype=float, ndim=2),  # type: ignore
):
    """obj-ground (floor/table) force obs — optionally masked to 0 before the lift
    stage as in JAX v2 (so the ever-present support force during approach does not
    pollute the obs; after lift it signals 'hit the floor'). Log-compressed into
    [0,1] when log1p_cap > 0."""
    w = wp.tid()
    v = obj_ground_force[w]
    if lift_step >= 0:
        if ep_step[w] < lift_step:
            v = 0.0
    out_obs[w, offset] = _grit_force_log1p(v, log1p_cap, inv_log1p_cap)


@wp.kernel
def _obs_projected_gravity_kernel(
    xmat:          wp.array(dtype=wp.mat33, ndim=2),  # type: ignore
    wrist_body_id: int,
    offset:        int,
    out_obs:       wp.array(dtype=float, ndim=2),     # type: ignore
):
    """Projected gravity — world down (0,0,−1) expressed in the wrist frame (3ch).

    The standard legged-RL attitude cue. Gives the actor an absolute cue for "which
    way the hand is flipped", observable at deployment via IMU/FK. A unit vector,
    hence bounded (±1) — obs_norm passthrough."""
    w = wp.tid()
    g = wp.transpose(xmat[w, wrist_body_id]) * wp.vec3(0.0, 0.0, -1.0)
    out_obs[w, offset + 0] = g[0]
    out_obs[w, offset + 1] = g[1]
    out_obs[w, offset + 2] = g[2]


@wp.kernel
def _obs_lift_target_kernel(
    xpos:          wp.array(dtype=wp.vec3, ndim=2),   # type: ignore
    xmat:          wp.array(dtype=wp.mat33, ndim=2),  # type: ignore
    obj_p_init:    wp.array(dtype=float, ndim=2),     # type: ignore (NWORLD, 3)
    obj_body_id:   int,
    wrist_body_id: int,
    lift_target_m: float,
    offset:        int,
    out_obs:       wp.array(dtype=float, ndim=2),     # type: ignore
):
    """Always-on lift-target channel (3ch): ``R_wᵀ·(p_lift_target − p_obj)``,
    ``p_lift_target = obj spawn position + (0,0,LIFT_TARGET_M)`` (fixed per episode).

    Gives the remaining displacement of "where the object must go" directly in the
    wrist frame — always on, without phase multiplexing (no switching jumps, always
    non-zero so the running norm stays stable, and observable as a command at
    deployment). ~0 once inside the success band. During approach it stays near the
    constant (0,0,target), so it is harmless."""
    w = wp.tid()
    tgt = wp.vec3(obj_p_init[w, 0], obj_p_init[w, 1],
                  obj_p_init[w, 2] + lift_target_m)
    v = wp.transpose(xmat[w, wrist_body_id]) * (tgt - xpos[w, obj_body_id])
    out_obs[w, offset + 0] = v[0]
    out_obs[w, offset + 1] = v[1]
    out_obs[w, offset + 2] = v[2]


@wp.kernel
def _object_grasping_qpos_mimic_kernel(
    qpos:                wp.array(dtype=float, ndim=2),    # type: ignore (NWORLD, nq)
    ctrl:                wp.array(dtype=float, ndim=2),    # type: ignore (NWORLD, n_ctrl) post-clamp commanded setpoint
    finger_qpa:          wp.array(dtype=int,   ndim=1),    # type: ignore (n_ctrl,)
    target_qpos:         wp.array(dtype=float, ndim=2),    # type: ignore (NWORLD, n_ctrl)
    specific_ctrl_mask:  wp.array(dtype=float, ndim=2),    # type: ignore (NWORLD, n_ctrl) 1 = active, 0 = rest
    n_ctrl:              int,
    use_ctrl:            int,                               # 1 → command (ctrl) based mimic (JAX v2 hand_action semantics)
    k_mimic:             float,
    active_deadzone:     float,                             # MIMIC_ACTIVE_DEADZONE
    rest_deadzone:       float,                             # MIMIC_REST_DEADZONE
    out_mimic_qpos_err:        wp.array(dtype=float, ndim=1),    # type: ignore  active_mse + rest_mse
    out_raw_mimic:       wp.array(dtype=float, ndim=1),    # type: ignore  exp(-k · mimic_mse)
    out_naive_err:       wp.array(dtype=float, ndim=1),    # type: ignore  Σ_rest|diff| (rest ctrl only; no deadzone)
):
    """Taxonomy-mask-split L1 hinge mimic loss + RBF (UNWEIGHTED).

    Port of the JAX ``mimic_process`` from ``grit_with_obj_coll_v2``::

        jnt_diff       = |qpos - action_template|             # per-ctrl L1
        active_subset  = jnt_diff where specific_ctrl_mask     # active fingers
        rest_subset    = jnt_diff where rest_ctrl_mask         # other fingers
        active_jnt_mimic = sum(relu(active_subset - 0.6)) / (1 + sum(specific_ctrl_mask))
        rest_jnt_mimic   = sum(relu(rest_subset   - 0.2)) / (1 + sum(rest_ctrl_mask))
        mimic_mse        = active_jnt_mimic + rest_jnt_mimic
        mimic_process    = exp(-4 · mimic_mse)

    NOTE: the denominator is ``1 + count`` (a +1 Laplace smoothing), NOT the
    legacy ``max(count, 1)``. The +1 both guards against division by zero when
    a branch has no ctrls AND damps the per-step error for branches with few
    active/rest ctrls — a deliberate smoothing choice. Its magnitude therefore
    differs slightly from the legacy mean, so retune ``K_MIMIC`` if porting
    exact numbers.

    Why split? The chosen grasp taxonomy makes some fingers "active" (they
    move a lot to wrap around the object — tolerate 0.6 rad slack) and the
    rest "passive" (should stay close to template — tight 0.2 rad). A
    single mean-squared loss would over-penalise the active fingers'
    natural deviation and under-penalise the passive fingers' drift.

    Implementation:
      * L1 (``abs``), not L2 — matches JAX exactly.
      * Per-branch normalisation by ``1 +`` that branch's count (active count
        / rest count, NOT the total ctrl count) — see the +1 smoothing note
        above. ``n_active`` / ``n_rest`` are initialised to ``1.0`` for this.
      * ``specific_ctrl_mask`` is 0 / 1 (per world, sampled from the
        taxonomy in :meth:`_apply_taxonomy_to_qpos_and_mask`); the rest
        mask is its 1-complement, so the kernel only needs one array.
      * Tendon-driven ctrls (``finger_qpa[i] == -1``) are skipped entirely
        — they contribute 0 to neither branch nor its count.

    Output ``out_qpos_err = active_jnt_mimic + rest_jnt_mimic`` is the
    pre-RBF mimic error (logged to W&B); ``out_raw_mimic`` is the
    UNWEIGHTED RBF used by the mixer.

    ``out_naive_err`` (constant L1 pressure without deadzone, ``W_MIMIC_NAIVE``) sums
    **rest ctrls only** — active ctrls normally deviate far from the template to
    grasp, so a constant pressure would directly conflict with wrapping. Its scale
    is therefore based on n_rest, not n_ctrl (6 rest joints × 0.3rad ≈ 1.8).
    """
    w = wp.tid()

    sum_active = float(0.0)
    sum_rest   = float(0.0)
    n_active   = float(1.0)
    n_rest     = float(1.0)
    naive_sum  = float(0.0)

    for i in range(n_ctrl):
        qa = finger_qpa[i]
        # ── diff source: MIMIC_ON_CTRL=true → command (ctrl setpoint) based (JAX v2
        # ``hand_action − template``). Even when the object blocks the fingers during
        # a grasp the command can still reach the template, so the mimic error does
        # not saturate at a residual and the gradient stays alive throughout.
        # false → measured qpos based (legacy; error has a floor after contact).
        # In ctrl mode the ctrl-space comparison is also valid for tendons (qa<0).
        diff = float(0.0)
        if use_ctrl == 1:
            diff = ctrl[w, i] - target_qpos[w, i]
        else:
            if qa < 0:
                continue
            diff = qpos[w, qa] - target_qpos[w, i]
        if diff < 0.0:
            diff = -diff                                # |diff| (L1)

        mask_v = specific_ctrl_mask[w, i]
        if mask_v > 0.5:                                # active ctrl
            n_active += 1.0
            if diff > active_deadzone:
                sum_active += diff - active_deadzone
        else:                                           # rest ctrl
            n_rest += 1.0
            # the naive constant pressure accumulates **rest ctrls only** (no deadzone).
            # Active ctrls normally deviate far from the template to grasp, so a
            # constant L1 would suppress wrapping itself — the hinge term
            # (active_deadzone 0.6rad) suffices. Rest ctrls should stay near the
            # template, so a weak restoring gradient is kept even inside the band.
            naive_sum += diff
            if diff > rest_deadzone:
                sum_rest += diff - rest_deadzone

    active_mse = sum_active / n_active
    rest_mse   = sum_rest   / n_rest
    mimic_mse  = active_mse + rest_mse

    out_mimic_qpos_err[w]   = mimic_mse
    out_raw_mimic[w]  = wp.exp(-k_mimic * mimic_mse)
    out_naive_err[w]  = naive_sum


@wp.kernel
def _object_grasping_wrist_dir_kernel(
    xmat:           wp.array(dtype=wp.mat33, ndim=2),     # type: ignore (NWORLD, nbody)
    target_R:       wp.array(dtype=float,    ndim=3),     # type: ignore (NWORLD, 3, 3) — frozen target rot
    wrist_body_id:  int,
    hand_center_R:  wp.mat33,                             # rh_hand_center[:3, :3] — constant per hand
    k_dir:          float,                                # decay rate in exp(-k · mean per-axis angle [rad])
    out_raw_dir:    wp.array(dtype=float, ndim=1),        # type: ignore  exp(-k · mean axis angle) ∈ (0, 1]
):
    """Orientation alignment of the **hand-center** x- AND y-axes with the
    frozen target frame (UNWEIGHTED).

    Legacy: ``wrist_dir_coeff = exp(-0.01·angle_err)`` where
    ``hand_R = (wrist_T @ rh_hand_center)[:3, :3]``. The legacy term aligned
    the **x-axis only** (``angle_err = arccos(<hand_R[:, 0], target_R[:, 0]>)``),
    which leaves the roll about that axis unconstrained. This kernel
    additionally aligns the **y-axis**, so the FULL hand-center orientation
    is pinned — for orthonormal frames the z-axis then follows from
    ``x × y``, so the two-axis error fully determines the rotation.

    The reference signals are the hand-center x/y axes in world frame, not
    the wrist body's own axes. ``rh_hand_center[:3, :3]`` rotates the wrist
    frame into the hand-center frame (defined per hand in
    ``hand_info/<hand>/rh_info.py``), so::

        R_hc(w)    = R_wrist(w) · rh_hand_center[:3, :3]
        hc_x_world = R_hc(w)[:, 0]            # palm-out direction
        hc_y_world = R_hc(w)[:, 1]

    The frozen ``target_R`` field is sampled once at reset by
    :meth:`GraspingHandler._post_cond_update_hook` via
    :func:`hand_utils.sample_target_R_init_per_world`. Columns 0 / 1
    (``target_R[:, 0]`` / ``target_R[:, 1]``) are the unit target x / y
    axes (column 0 is the legacy ``target_dir`` heading). At reset
    ``hc_{x,y}_world ≈ target_R[:, {0,1}]`` so both per-axis angles ≈ 0;
    they grow as the wrist drifts and the reward keeps the hand-center
    frame aligned with its reset orientation.

    Per-axis angle error (radians = ``arccos`` of the dot of two unit
    columns, ``clamp``-ed to ``[-1, 1]`` for numerical safety against
    ``acos`` domain errors) is averaged over the two axes and mapped
    through the legacy decaying exponential::

        ang_x    = arccos(clamp(<hc_x_world, target_R[:, 0]>, -1, 1))
        ang_y    = arccos(clamp(<hc_y_world, target_R[:, 1]>, -1, 1))
        out_raw_dir = exp(-K_WRIST_DIR · (ang_x + ang_y) / 2)        ∈ (0, 1]

    ``out_raw_dir`` peaks at 1 when both axes match the target and decays
    as either axis drifts. With the default ``K_WRIST_DIR = 0.1`` the curve is
    still fairly shallow — the mean angle spans ``[0, π]`` so the term only
    drops to ``exp(-0.1·π) ≈ 0.73`` at full opposition (the legacy coefficient
    was ``0.01`` → ``exp(-0.01·π) ≈ 0.97``); raise ``K_WRIST_DIR`` for a sharper
    alignment gradient.

    See ``notebook/hand/01_hand_setup/10_viz_hand_center.ipynb`` for the
    visual reference — the rendered ``hand_center_calculated`` frame's
    x / y axes are exactly ``hc_{x,y}_world`` here.
    """
    w = wp.tid()
    R_wrist = xmat[w, wrist_body_id]
    # World-frame hand-center rotation via wrist SE3 composition.
    R_hc       = R_wrist * hand_center_R
    hc_x_world = R_hc * wp.vec3(1.0, 0.0, 0.0)        # = first  column of R_hc
    hc_y_world = R_hc * wp.vec3(0.0, 1.0, 0.0)        # = second column of R_hc

    # Frozen target x / y axes = columns 0 / 1 of target_R
    # (column 0 is the legacy ``target_dir`` heading).
    tgt_dir_x = wp.vec3(target_R[w, 0, 0], target_R[w, 1, 0], target_R[w, 2, 0])
    tgt_dir_y = wp.vec3(target_R[w, 0, 1], target_R[w, 1, 1], target_R[w, 2, 1])

    # Per-axis angle error (radians) — NOTE these are ANGLES (arccos), not
    # cosines. clamp guards acos against |dot| > 1 from float round-off.
    ang_x    = wp.acos(wp.clamp(wp.dot(hc_x_world, tgt_dir_x), -1.0, 1.0))
    ang_y    = wp.acos(wp.clamp(wp.dot(hc_y_world, tgt_dir_y), -1.0, 1.0))
    ang_mean = (ang_x + ang_y) / 2.0
    out_raw_dir[w] = wp.exp(-k_dir * ang_mean)

@wp.kernel
def _object_grasping_wrist_vel_kernel(
    fd_vel:         wp.array(dtype=float, ndim=2),       # type: ignore (NWORLD, 12) pose-FD velocity buffer
    ang_weight:     float,                                 # multiplier on angular-vel sq norm
    k_vel:          float,                                 # decay rate in exp(-k · total)
    out_total_vel:  wp.array(dtype=float, ndim=1),       # type: ignore  ‖vlin‖² + ang_w·‖vang‖²
    out_raw_vel:    wp.array(dtype=float, ndim=1),       # type: ignore  exp(-k · total)
):
    """Wrist stationarity coefficient (UNWEIGHTED) — pose-FD based.

    Legacy JAX ``wrist_vel_coeff``::

        total_vel       = sum(vlin²) + 0.1·sum(vang²)
        wrist_vel_coeff = exp(-0.1·total_vel)

    The velocity source is the **pose-FD buffer** (filled every step by
    ``_fd_vel_update_kernel``; same values as the fd_vel obs block), not ``d.qvel``.
    qvel/cvel pick up contact-solver jitter as instantaneous velocity and
    overestimate even at rest — since this term enters the **multiplicative
    hand_ineq_coeff**, that jitter was a noise source randomly attenuating the
    whole reward every step. FD averages over the control step so the jitter
    cancels, and obs and reward share the same velocity. The squared norm is
    rotation-invariant, so the prev-wrist-frame representation of fd_vel is used
    as is. The first step after a reset has fd=0 → coeff 1 (treated as at rest; harmless).
    """
    w = wp.tid()

    lin_sq = float(0.0)
    ang_sq = float(0.0)
    for i in range(3):
        v = fd_vel[w, 0 + i]
        lin_sq += v * v
        a = fd_vel[w, 3 + i]
        ang_sq += a * a

    total = lin_sq + ang_weight * ang_sq
    out_total_vel[w] = total
    out_raw_vel[w]   = wp.exp(-k_vel * total)


@wp.kernel
def _object_grasping_obj_vel_kernel(
    fd_vel:             wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, 12) pose-FD velocity buffer
    ang_weight:         float,                                       # multiplier on angular-vel sq norm
    k_vel:              float,                                       # decay rate in exp(-k · total)
    out_total_obj_vel:  wp.array(dtype=float, ndim=1),  # type: ignore  Σ clip(vlin)² + ang_w·Σ clip(vang)²
    out_raw_obj_vel:    wp.array(dtype=float, ndim=1),  # type: ignore  exp(-k · total) ∈ (0, 1]
):
    """Object stationarity coefficient — pose-FD, wrist-relative, clipped (UNWEIGHTED).

    Legacy JAX ``obj_vel_coeff``::

        obj_vel_in_wrist  = clip(Rᵀ·v_obj − wrist_vel, -1, 1)   # wrist-relative!
        obj_qvel_in_wrist = clip(Rᵀ·ω_obj − wrist_ang, -1, 1)
        total_vel       = Σ(vel²) + 0.01·Σ(qvel²)
        obj_vel_coeff   = exp(-0.1·total_vel)

    Velocity source = pose-FD buffer channels [6:9]/[9:12] (obj, prev-wrist frame,
    **wrist-relative**). Two things are corrected relative to the earlier
    (cvel absolute-velocity) implementation:

      1. **cvel contact jitter removed** — cvel picks up solver jitter as
         instantaneous velocity and overestimates even for an object at rest. This
         term is the multiplicative obj_ineq_coeff, so the jitter randomly
         attenuated the reward at every contact. FD is a per-step average.
      2. **Original JAX semantics (wrist-relative) restored** — JAX used the obs's
         wrist-relative velocity directly in the reward. With relative velocity an
         object "moving with the hand" during lift is not penalized; only relative
         motion slipping within the grasp is (absolute velocity attenuates a normal
         lift itself).

    Per-component clip ±1 → ``total ≤ 3 + w·3`` → coeff floor ≈ 0.74 is kept.
    The first step after a reset has fd=0 → coeff 1 (harmless).
    """
    w = wp.tid()

    lin_sq = float(0.0)
    ang_sq = float(0.0)
    for i in range(3):
        c = wp.clamp(fd_vel[w, 6 + i], -1.0, 1.0)
        lin_sq += c * c
        a = wp.clamp(fd_vel[w, 9 + i], -1.0, 1.0)
        ang_sq += a * a

    total = lin_sq + ang_weight * ang_sq
    out_total_obj_vel[w] = total
    out_raw_obj_vel[w]   = wp.exp(-k_vel * total)


@wp.kernel
def _object_grasping_lift_kernel(
    xpos:             wp.array(dtype=wp.vec3, ndim=2),    # type: ignore
    obj_p_init:       wp.array(dtype=float,   ndim=2),    # type: ignore (NWORLD, 3)
    obj_body_id:      int,
    lift_target_m:    float,
    out_lift_z:       wp.array(dtype=float, ndim=1),      # type: ignore signed obj z lift
    out_raw_lift:     wp.array(dtype=float, ndim=1),      # type: ignore clip(lift / target, 0, 1)
):
    """Object **lift (z)** signal (UNWEIGHTED).

    Legacy: ``obj_z_process = sum(where(obj_z >= 0.1, 1, clip(obj_z/0.1, 0, 1)))``
    — kept in normalised [0, 1] form so the mixer weight has an intuitive
    magnitude (``W_LIFT = 5`` → up to +5 per step once the object is lifted to
    ``LIFT_TARGET_M``).

    The object **xy-drift** signals are NOT computed here — the xy (and R)
    coefficients live in :func:`_grasping_obj_pose_coeff_kernel`, which
    is the single owner of the object-position-vs-spawn reward.
    """
    w = wp.tid()
    obj_p = xpos[w, obj_body_id]

    obj_p0 = wp.vec3(obj_p_init[w, 0], obj_p_init[w, 1], obj_p_init[w, 2])
    lift_z = obj_p[2] - obj_p0[2]

    lift_norm = lift_z / lift_target_m
    if lift_norm < 0.0:
        lift_norm = 0.0
    if lift_norm > 1.0:
        lift_norm = 1.0

    out_lift_z[w]   = lift_z
    out_raw_lift[w] = lift_norm


@wp.kernel
def _grasping_obj_pose_coeff_kernel(
    xpos:                 wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore (NWORLD, nbody)
    xmat:                 wp.array(dtype=wp.mat33, ndim=2),  # type: ignore (NWORLD, nbody)
    obj_p_init:           wp.array(dtype=float,    ndim=2),  # type: ignore (NWORLD, 3)
    obj_body_id:          int,
    dist_penalty_coeff:   float,                              # OBJ_DIST_PENALTY_COEFF (≤ 0)
    r_upright_clip_min:   float,                              # clip floor on obj_R[2, 2] (0.1)
    out_raw_obj_xy_coeff: wp.array(dtype=float, ndim=1),  # type: ignore exp(coeff · |Δxy|) ∈ (0, 1]
    out_raw_obj_R_coeff:  wp.array(dtype=float, ndim=1),  # type: ignore exp(coeff/2.5 · (1 - clip(R[2,2]))) ∈ (0, 1]
):
    """Object xy-drift + uprightness coefficients vs the reset spawn pose
    (UNWEIGHTED).

    Port of the legacy JAX object-pose coefficients from
    ``grit_with_obj_coll_v2``::

        obj_xy_dist  = ||grasping_obj_p_diff[:2]||            # |Δxy| from spawn
        obj_xy_coeff = exp(obj_dist_penalty_coeff · obj_xy_dist)
        obj_R_coeff  = exp(obj_dist_penalty_coeff/2.5 · (1 - clip(obj_R[2,2], 0.1, 1.0)))

    Both share the single ``obj_dist_penalty_coeff`` (the rotation term is
    scaled by ``1/2.5``), which is **negative** so each coefficient sits at
    1 when the object is undisturbed and decays toward 0 as it drifts /
    tips:

      * ``obj_xy_coeff`` — ``|Δxy|`` is the L2 distance of the object's
        current world xy from its reset ``obj_p_init`` xy (this kernel is the
        sole owner of the object xy-drift signal — the lift kernel no longer
        computes it). ``= 1`` at the spawn xy, decaying as the object slides
        across the table.
      * ``obj_R_coeff`` — ``obj_R[2, 2]`` is the world-z component of the
        object's body-z axis (its **uprightness**): ``1`` when the object
        stands exactly as spawned, dropping as it tips. Clipped to
        ``[0.1, 1.0]`` (legacy floor) so a fully toppled / inverted object
        saturates the penalty instead of exploding it; ``1 - clip`` is then
        the tip amount in ``[0, 0.9]``.

    The mixer multiplies each raw coefficient by its weight
    (``W_OBJ_XY_COEFF`` / ``W_OBJ_R_COEFF``) to fold them into the additive
    per-step reward — same compute→mix split as the velocity coefficients.
    """
    w = wp.tid()
    obj_p  = xpos[w, obj_body_id]
    obj_p0 = wp.vec3(obj_p_init[w, 0], obj_p_init[w, 1], obj_p_init[w, 2])

    # |Δxy| from the reset spawn position.
    dx = obj_p[0] - obj_p0[0]
    dy = obj_p[1] - obj_p0[1]
    xy_dist = wp.sqrt(dx * dx + dy * dy)

    # Uprightness = world-z component of the object body-z axis = R[2, 2].
    R_obj = xmat[w, obj_body_id]
    r22   = wp.clamp(R_obj[2, 2], r_upright_clip_min, 1.0)

    out_raw_obj_xy_coeff[w] = wp.exp(dist_penalty_coeff * xy_dist)
    out_raw_obj_R_coeff[w]  = wp.exp((dist_penalty_coeff / 2.5) * (1.0 - r22))
