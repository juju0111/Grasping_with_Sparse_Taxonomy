"""Per-world target rotation sampler (GPU) — warp kernels of the grasping task."""
import mujoco as _mj  # type: ignore
import numpy as np  # type: ignore
import warp as wp  # type: ignore


# ══════════════════════════════════════════════════════════════════════════
# Per-world target_R init — GPU port of hand_utils.sample_target_R_init_per_world
# ══════════════════════════════════════════════════════════════════════════


@wp.kernel
def _object_grasping_sample_target_R_kernel(
    xmat:          wp.array(dtype=wp.mat33, ndim=2),   # type: ignore (NWORLD, nbody) world-frame
    wrist_body_id: int,
    hc_col0:       wp.vec3,                            # rh_hand_center[:3, 0] (wrist→hand-center X col)
    hc_col2:       wp.vec3,                            # rh_hand_center[:3, 2] (Z col)
    mask:          wp.array(dtype=int, ndim=1),        # type: ignore (NWORLD,) 1 = (re)sample this world
    seed:          int,                                # per-call seed (host increments each reset)
    out_R:         wp.array(dtype=float, ndim=3),      # type: ignore (NWORLD, 3, 3) cond target_R gpu
):
    """GPU port of :func:`hand_utils.sample_target_R_init_per_world`.

    The CPU original looped over worlds doing numpy linalg + ``np.random``; it
    was ~79% of ``_post_cond_update_hook`` (the dominant training bottleneck).
    The target rotation depends ONLY on the wrist orientation + hand-center
    columns + per-world random noise (the PCD it was handed is unused — see the
    CPU docstring), so it ports to a pure per-world warp kernel with NO host
    sync. Reproduces the CPU math exactly (warp RNG replaces ``np.random`` — the
    noise distribution is identical, the exact stream differs):

        heading = normalize(R_wrist · hc_col0 + U(-0.2, 0.2)^3)
        z_world = normalize(R_wrist · hc_col2 + U(-0.2, 0.2)^3)
        y_new   = normalize(z_world × heading)
        z_new   = normalize(heading × y_new)
        target_R = [heading | y_new | z_new] · Rx(U(-pi/6, pi/6))

    ``Rx(a)`` post-multiply rotates the frame about its own X (heading) axis:
    col0' = heading; col1' = cos·y_new + sin·z_new; col2' = -sin·y_new + cos·z_new.
    """
    w = wp.tid()
    if mask[w] == 0:
        return
    Rw = xmat[w, wrist_body_id]
    heading = Rw * hc_col0
    z_world = Rw * hc_col2

    # 7 uniforms — distinct rand_init offsets per component (independent draws,
    # advance-semantics-agnostic). seed varies per reset for fresh noise.
    base = w * 8
    hx = (wp.randf(wp.rand_init(seed, base + 0)) * 2.0 - 1.0) * 0.2
    hy = (wp.randf(wp.rand_init(seed, base + 1)) * 2.0 - 1.0) * 0.2
    hz = (wp.randf(wp.rand_init(seed, base + 2)) * 2.0 - 1.0) * 0.2
    zx = (wp.randf(wp.rand_init(seed, base + 3)) * 2.0 - 1.0) * 0.2
    zy = (wp.randf(wp.rand_init(seed, base + 4)) * 2.0 - 1.0) * 0.2
    zz = (wp.randf(wp.rand_init(seed, base + 5)) * 2.0 - 1.0) * 0.2
    xn = (wp.randf(wp.rand_init(seed, base + 6)) * 2.0 - 1.0) * (3.14159265358979 / 6.0)

    heading = wp.normalize(heading + wp.vec3(hx, hy, hz))
    z_world = wp.normalize(z_world + wp.vec3(zx, zy, zz))
    y_new = wp.normalize(wp.cross(z_world, heading))
    z_new = wp.normalize(wp.cross(heading, y_new))

    c = wp.cos(xn)
    s = wp.sin(xn)
    col0 = heading
    col1 = y_new * c + z_new * s
    col2 = y_new * (-s) + z_new * c

    out_R[w, 0, 0] = col0[0]; out_R[w, 1, 0] = col0[1]; out_R[w, 2, 0] = col0[2]
    out_R[w, 0, 1] = col1[0]; out_R[w, 1, 1] = col1[1]; out_R[w, 2, 1] = col1[2]
    out_R[w, 0, 2] = col2[0]; out_R[w, 1, 2] = col2[1]; out_R[w, 2, 2] = col2[2]


@wp.kernel
def _object_grasping_sample_target_pnt_kernel(
    xpos:          wp.array(dtype=wp.vec3,  ndim=2),   # type: ignore (NWORLD, nbody)
    xmat:          wp.array(dtype=wp.mat33, ndim=2),   # type: ignore (NWORLD, nbody)
    pcd_local:     wp.array(dtype=wp.vec3,  ndim=3),   # type: ignore (n_var, n_obj, n_pts) body-local
    assignment:    wp.array(dtype=int,      ndim=1),   # type: ignore (NWORLD,)
    obj_idx:       int,
    obj_body_id:   int,
    wrist_body_id: int,
    n_pts:         int,
    noise_amp:     float,
    seed:          int,
    mask:          wp.array(dtype=int, ndim=1),        # type: ignore (NWORLD,)
    out_pnt:       wp.array(dtype=float, ndim=2),      # type: ignore (NWORLD, 3) cond target_pnt gpu
):
    """GPU port of :meth:`GraspingHandler._sample_target_pnt`.

    Per world: transform the assigned object PCD body-local → world, pick the
    vertex nearest the wrist, add U(-noise, noise)^3 jitter. The CPU original
    looped over worlds (each a 1024-vertex transform + argmin) ⇒ ~20% of the
    reset hook; this is the same nearest-vertex scan the reward path already
    uses, run sync-free for the wrist. Falls back to the object centroid when
    ``n_pts == 0`` (best stays at ``p_obj``)."""
    w = wp.tid()
    if mask[w] == 0:
        return
    R_obj = xmat[w, obj_body_id]
    p_obj = xpos[w, obj_body_id]
    wrist_p = xpos[w, wrist_body_id]
    v_idx = assignment[w]
    best_d2 = float(1.0e18)
    best = p_obj
    for k in range(n_pts):
        pw = R_obj * pcd_local[v_idx, obj_idx, k] + p_obj
        d = pw - wrist_p
        d2 = wp.dot(d, d)
        if d2 < best_d2:
            best_d2 = d2
            best = pw
    base = w * 4
    nx = (wp.randf(wp.rand_init(seed, base + 0)) * 2.0 - 1.0) * noise_amp
    ny = (wp.randf(wp.rand_init(seed, base + 1)) * 2.0 - 1.0) * noise_amp
    nz = (wp.randf(wp.rand_init(seed, base + 2)) * 2.0 - 1.0) * noise_amp
    out_pnt[w, 0] = best[0] + nx
    out_pnt[w, 1] = best[1] + ny
    out_pnt[w, 2] = best[2] + nz


@wp.kernel
def _object_grasping_snapshot_obj_pose_kernel(
    xpos:        wp.array(dtype=wp.vec3,  ndim=2),     # type: ignore (NWORLD, nbody)
    xmat:        wp.array(dtype=wp.mat33, ndim=2),     # type: ignore (NWORLD, nbody)
    obj_body_id: int,
    mask:        wp.array(dtype=int, ndim=1),          # type: ignore (NWORLD,)
    out_p_init:  wp.array(dtype=float, ndim=2),        # type: ignore (NWORLD, 3) cond obj_p_init gpu
    out_R_init:  wp.array(dtype=float, ndim=3),        # type: ignore (NWORLD, 3, 3) cond obj_R_init gpu
):
    """Snapshot the object's spawn pose (xpos / xmat) into the cond fields on an
    episode reset — GPU replacement for the ``d.xpos/xmat.numpy()`` host copy +
    masked CPU write. Only ``mask[w]==1`` worlds are (re)anchored."""
    w = wp.tid()
    if mask[w] == 0:
        return
    p = xpos[w, obj_body_id]
    out_p_init[w, 0] = p[0]; out_p_init[w, 1] = p[1]; out_p_init[w, 2] = p[2]
    R = xmat[w, obj_body_id]
    out_R_init[w, 0, 0] = R[0, 0]; out_R_init[w, 0, 1] = R[0, 1]; out_R_init[w, 0, 2] = R[0, 2]
    out_R_init[w, 1, 0] = R[1, 0]; out_R_init[w, 1, 1] = R[1, 1]; out_R_init[w, 1, 2] = R[1, 2]
    out_R_init[w, 2, 0] = R[2, 0]; out_R_init[w, 2, 1] = R[2, 1]; out_R_init[w, 2, 2] = R[2, 2]
