"""Single-arm · single-hand MuJoCo Warp environment.

This file covers the **single-arm, single-hand** scenario.
:class:`SingleHandSubEnv` is the base env builder that simulates one
hand × object pair across N worlds in parallel on the GPU.

The sibling module :mod:`grit.training.orchestrator.leader_follower`
(``LeaderFollowerSubEnv``) is also **single-arm code** in this classification:
two hands appear, but the structure is *single-hand leader + single-hand
follower*, each being an independent single arm with a single hand.

TODO: a **dual-arm (bimanual)** setup — e.g. the left and right arms of one
humanoid sharing a common root and manipulating simultaneously — does not
exist in the current codebase and remains future work.
"""
import os
import abc
import random
import numpy as np
from pathlib import Path
from typing import Optional
from etils import epath
import mujoco
import mujoco_warp as mjwarp
import warp as wp
import dataclasses

from grit.util.utils import print_red
from grit.util.hand_utils import merge_mjcfs   # routes outputs into XML_OUTPUT_DIR
from grit.util.sim_core.utils import farthest_point_sampling

from grit.util import hand_utils
from grit.util.hand_rl_parser import HandRLParserClass

_grit_dir = Path(__file__).resolve().parent  # walk up from THIS module, not cwd

def resolve_project_root() -> Path:
    for base in (_grit_dir, *_grit_dir.parents):
        if (
            (base / "README.md").is_file()
            and (base / "grit").is_dir()
            and (base / "config").is_dir()
        ):
            return base
    raise RuntimeError(
        f"Could not find project root from {_grit_dir}. "
        "cd into the repository and restart the kernel."
    )

PROJECT_ROOT = resolve_project_root()
home_dir = str(PROJECT_ROOT)


# ════════════════════════════════════════════════════════════════════════════
# GPU-resident pose-init kernels (test path, parallel to numpy roundtrip ones)
# ----------------------------------------------------------------------------
# These mirror the semantics of warp_obj_pose_init() / warp_wrist_pose_init()
# but run entirely on the GPU. Designed to be wp.ScopedCapture-friendly so a
# full "reset all worlds" can be a single graph launch.
#
# Convention checks (verified against installed mujoco_warp 1.x):
#   d.qpos       : wp.array(dtype=float,    ndim=2)  (nworld, nq)
#   d.qvel       : wp.array(dtype=float,    ndim=2)  (nworld, nv)
#   d.xpos       : wp.array(dtype=wp.vec3,  ndim=2)  (nworld, nbody)
#   d.xmat       : wp.array(dtype=wp.mat33, ndim=2)  (nworld, nbody)
#   d.mocap_pos  : wp.array(dtype=wp.vec3,  ndim=2)  (nworld, nmocap)
#   d.mocap_quat : wp.array(dtype=wp.quat,  ndim=2)  (nworld, nmocap)  wxyz
# wp.quat memory layout: q[0],q[1],q[2],q[3] = element-wise as constructed.
# MuJoCo / mujoco_warp convention is wxyz, so we construct as wp.quat(w,x,y,z).
# ════════════════════════════════════════════════════════════════════════════


@wp.func
def _grit_rpy2r(rpy: wp.vec3) -> wp.mat33:
    """ZYX intrinsic Euler → 3×3 rotation. Used only for ±0.3° wrist noise."""
    cr = wp.cos(rpy[0]); sr = wp.sin(rpy[0])
    cp = wp.cos(rpy[1]); sp = wp.sin(rpy[1])
    cy = wp.cos(rpy[2]); sy = wp.sin(rpy[2])
    return wp.mat33(
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp,     cp * sr,                cp * cr,
    )


@wp.func
def _grit_r2quat_wxyz(M: wp.mat33) -> wp.quat:
    """Rotation matrix → unit quaternion in wxyz layout (Shepherd's method)."""
    tr = M[0, 0] + M[1, 1] + M[2, 2]
    qw = float(0.0); qx = float(0.0); qy = float(0.0); qz = float(0.0)
    if tr > 0.0:
        s = wp.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (M[2, 1] - M[1, 2]) / s
        qy = (M[0, 2] - M[2, 0]) / s
        qz = (M[1, 0] - M[0, 1]) / s
    elif (M[0, 0] > M[1, 1]) and (M[0, 0] > M[2, 2]):
        s = wp.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2.0
        qw = (M[2, 1] - M[1, 2]) / s
        qx = 0.25 * s
        qy = (M[0, 1] + M[1, 0]) / s
        qz = (M[0, 2] + M[2, 0]) / s
    elif M[1, 1] > M[2, 2]:
        s = wp.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2.0
        qw = (M[0, 2] - M[2, 0]) / s
        qx = (M[0, 1] + M[1, 0]) / s
        qy = 0.25 * s
        qz = (M[1, 2] + M[2, 1]) / s
    else:
        s = wp.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2.0
        qw = (M[1, 0] - M[0, 1]) / s
        qx = (M[0, 2] + M[2, 0]) / s
        qy = (M[1, 2] + M[2, 1]) / s
        qz = 0.25 * s
    return wp.quat(qw, qx, qy, qz)


@wp.kernel
def _grit_set_seed_kernel(seed_arr: wp.array(dtype=int), value: int):  # type: ignore
    """Tiny helper: set seed_arr[0] = value on GPU. Used to vary the
    captured graph's RNG seed between launches without recapture."""
    seed_arr[0] = value


@wp.kernel
def _grit_finger_closest_pcd_kernel(
    site_xpos:       wp.array(dtype=wp.vec3,  ndim=2),   # type: ignore  (NWORLD, nsite)
    xpos:            wp.array(dtype=wp.vec3,  ndim=2),   # type: ignore  (NWORLD, nbody)
    xmat:            wp.array(dtype=wp.mat33, ndim=2),   # type: ignore  (NWORLD, nbody)
    finger_site_ids: wp.array(dtype=int,      ndim=1),   # type: ignore  (n_finger,)
    pcd_local:       wp.array(dtype=wp.vec3,  ndim=3),   # type: ignore  (n_var, n_obj, n_pts)
    assignment:      wp.array(dtype=int,      ndim=1),   # type: ignore  (NWORLD,)
    obj_idx:         int,                                  # which obj slot (graspable target)
    obj_body_id:     int,                                  # body id in d.xpos / d.xmat
    n_finger:        int,
    n_pts:           int,
    out_closest_p:   wp.array(dtype=wp.vec3, ndim=2),    # type: ignore  (NWORLD, n_finger) world-frame
    out_min_dist:    wp.array(dtype=float,   ndim=2),    # type: ignore  (NWORLD, n_finger)
):
    """Per-world, per-finger nearest-PCD-vertex search (warp port of the
    legacy CPU pattern used in
    ``notebook/hand/01_hand_setup/13_dyn_init_single_obj`` and
    ``notebook/hand/03_warp_parallel/04`` — finger site → obj PCD argmin).

    For each (world w, finger f):
      1. Pull the variant's body-local PCD ``pcd_local[v_idx, obj_idx, :]``.
      2. Transform every vertex into world frame: ``p_w = R_obj · p_local + obj_p``.
      3. Find the vertex with minimum squared distance to ``site_xpos[w, sid]``.
      4. Write the world-frame closest point and the L2 distance.

    Single 1D launch over ``NWORLD`` — each thread loops the small
    ``n_finger × n_pts`` work (≈ 5 × 1024 = 5 k comparisons / thread →
    ~10 µs at NWORLD=16). Output buffers are pre-allocated by the host
    helper so the kernel is allocation-free.
    """
    w = wp.tid()

    R_obj  = xmat[w, obj_body_id]
    p_obj  = xpos[w, obj_body_id]
    v_idx  = assignment[w]

    for f in range(n_finger):
        sid       = finger_site_ids[f]
        finger_p  = site_xpos[w, sid]

        best_d2 = float(1.0e9)
        best_p  = wp.vec3(0.0, 0.0, 0.0)
        for k in range(n_pts):
            pcd_w = R_obj * pcd_local[v_idx, obj_idx, k] + p_obj
            diff  = pcd_w - finger_p
            d2    = wp.dot(diff, diff)
            if d2 < best_d2:
                best_d2 = d2
                best_p  = pcd_w
        out_closest_p[w, f] = best_p
        out_min_dist[w, f]  = wp.sqrt(best_d2)


@wp.kernel
def _grit_warp_obj_pose_init_kernel(
    seed_arr:        wp.array(dtype=int),               # type: ignore  (1,)
    n_obj:           int,
    table_height:    float,
    xy_min:          float,
    xy_max:          float,
    z_clearance:     float,
    qpos_addrs:      wp.array(dtype=int),               # type: ignore  (n_obj,)
    assignment:      wp.array(dtype=int),               # type: ignore  (nworld,)
    var_obj_min_z:   wp.array(dtype=float, ndim=2),     # type: ignore  (n_variants, n_obj)
    qpos:            wp.array(dtype=float, ndim=2),     # type: ignore  (nworld, nq)
    qvel:            wp.array(dtype=float, ndim=2),     # type: ignore  (nworld, nv)
):
    w = wp.tid()
    state = wp.rand_init(seed_arr[0], w)
    v = assignment[w]

    # zero qvel for this world
    nv = qvel.shape[1]
    for i in range(nv):
        qvel[w, i] = 0.0

    for o in range(n_obj):
        qpa = qpos_addrs[o]
        x = wp.randf(state) * (xy_max - xy_min) + xy_min
        y = wp.randf(state) * (xy_max - xy_min) + xy_min
        z = table_height + var_obj_min_z[v, o] + z_clearance
        qpos[w, qpa + 0] = x
        qpos[w, qpa + 1] = y
        qpos[w, qpa + 2] = z
        qpos[w, qpa + 3] = 1.0  # qw
        qpos[w, qpa + 4] = 0.0  # qx
        qpos[w, qpa + 5] = 0.0  # qy
        qpos[w, qpa + 6] = 0.0  # qz


@wp.kernel
def _grit_warp_wrist_pose_init_kernel(
    seed_arr:        wp.array(dtype=int),               # type: ignore  (1,)
    obj_body_id:     int,
    obj_index:       int,
    n_pts:           int,
    wrist_qpa:       int,
    mocap_id:        int,
    noise_scale:     float,
    x_lo:            float,
    x_hi:            float,
    y_lo:            float,                              # y is independent of x (for azimuth-bin evaluation)
    y_hi:            float,
    z_lo:            float,
    z_hi:            float,
    roll_lo:         float,
    roll_hi:         float,
    p_init_range_lo: float,                              # default-branch hand-center→target offset (m), lo
    p_init_range_hi: float,                              # default-branch hand-center→target offset (m), hi
    p_init_z_min:    float,                              # default-branch minimum hand-center world z (prevents floor penetration)
    R_inv:           wp.mat33,
    p_inv:           wp.vec3,
    palm_a:          float,                              # palm_in_wrist[0] (rh_hand_center[0,0])
    palm_c:          float,                              # palm_in_wrist[2] (rh_hand_center[2,0])
    y_up_prob:       float,
    table_height:    float,
    y_up_z_lo:       float,                              # special-mode wrist z above table
    y_up_z_hi:       float,
    y_up_margin_lo:  float,                              # extra margin beyond obj XY radius
    y_up_margin_hi:  float,
    obj_xy_radius:   wp.array(dtype=float, ndim=2),     # type: ignore  (n_var, n_obj) body-local max XY distance from centroid
    pcd_local:       wp.array(dtype=wp.vec3, ndim=3),   # type: ignore  (n_var, n_obj, n_pts)
    centroid_local:  wp.array(dtype=wp.vec3, ndim=2),   # type: ignore  (n_var, n_obj)
    assignment:      wp.array(dtype=int),               # type: ignore  (nworld,)
    xpos:            wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore  (nworld, nbody)
    xmat:            wp.array(dtype=wp.mat33, ndim=2),  # type: ignore  (nworld, nbody)
    qpos:            wp.array(dtype=float,    ndim=2),  # type: ignore  (nworld, nq)
    mocap_pos:       wp.array(dtype=wp.vec3,  ndim=2),  # type: ignore  (nworld, nmocap)
    mocap_quat:      wp.array(dtype=wp.quat,  ndim=2),  # type: ignore  (nworld, nmocap)
):
    w = wp.tid()
    state = wp.rand_init(seed_arr[0] + 7919, w)   # offset so obj/wrist streams differ
    v = assignment[w]

    p_body = xpos[w, obj_body_id]
    R_body = xmat[w, obj_body_id]

    # sample target vertex from the body-local PCD, transform to world
    pt_idx = int(wp.randf(state) * float(n_pts))
    if pt_idx >= n_pts:
        pt_idx = n_pts - 1
    target_world = R_body * pcd_local[v, obj_index, pt_idx] + p_body

    # === Branch: special "Y up + palm facing object" with prob y_up_prob ===
    use_special = wp.randf(state) < y_up_prob

    wrist_R = wp.mat33(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    )
    wrist_p = wp.vec3(0.0, 0.0, 0.0)

    if use_special:
        # === Special: wrist Y == world up, palm faces target horizontally ===
        # object centroid_world ≈ object body origin (assumes obj quat=I at init)
        obj_xy_x = p_body[0]
        obj_xy_y = p_body[1]

        # Offset by the object XY radius (max about the body-local centroid) + margin.
        # At init the object rotation is identity, so body-local radius == world radius.
        obj_r = obj_xy_radius[v, obj_index]
        margin  = wp.randf(state) * (y_up_margin_hi - y_up_margin_lo) + y_up_margin_lo
        horiz_d = obj_r + margin

        azimuth = wp.randf(state) * (3.141592653589793 * 2.0) - 3.141592653589793
        wz      = table_height + wp.randf(state) * (y_up_z_hi - y_up_z_lo) + y_up_z_lo

        wrist_p = wp.vec3(
            obj_xy_x - horiz_d * wp.cos(azimuth),
            obj_xy_y - horiz_d * wp.sin(azimuth),
            wz,
        )

        # palm_world horizontal direction = (target_world - wrist_p) projected to XY
        palm_world = target_world - wrist_p
        px_raw = palm_world[0]
        py_raw = palm_world[1]
        horiz_norm = wp.sqrt(px_raw * px_raw + py_raw * py_raw)
        px = float(1.0)
        py = float(0.0)
        if horiz_norm > 1.0e-6:
            px = px_raw / horiz_norm
            py = py_raw / horiz_norm

        denom = palm_a * palm_a + palm_c * palm_c
        cos_a = float(1.0)
        sin_a = float(0.0)
        if denom > 1.0e-9:
            cos_a = (palm_a * px - palm_c * py) / denom
            sin_a = (palm_c * px + palm_a * py) / denom
            n2 = wp.sqrt(cos_a * cos_a + sin_a * sin_a)
            if n2 > 1.0e-9:
                cos_a = cos_a / n2
                sin_a = sin_a / n2

        # R_wrist columns: X=(c,s,0), Y=(0,0,1), Z=(s,-c,0)
        wrist_R = wp.mat33(
            cos_a, 0.0,  sin_a,
            sin_a, 0.0, -cos_a,
            0.0,   1.0,  0.0,
        )
        # no noise — preserves wrist Y == world up
    else:
        # === Default: R_init with the hand-center facing the object + roll noise ===
        cd = wp.vec3(
            wp.randf(state) * (x_hi - x_lo) + x_lo,
            wp.randf(state) * (y_hi - y_lo) + y_lo,
            wp.randf(state) * (z_hi - z_lo) + z_lo,
        )
        cd = wp.normalize(cd)
        # sample the offset distance from [p_init_range_lo, p_init_range_hi] each time instead of a fixed value
        offset_d = wp.randf(state) * (p_init_range_hi - p_init_range_lo) + p_init_range_lo
        p_init = target_world + offset_d * cd

        # Floor-clearance clamp: a bottom-surface PCD vertex combined with a
        # low-elevation direction can bring the hand-center down to 3-6 cm above
        # the floor, spawning the forearm/fingertips through it (~6.5% of resets
        # with roll ±π and ±30° noise). Clamp before computing x_axis so the hand
        # still faces the target.
        if p_init[2] < p_init_z_min:
            p_init = wp.vec3(p_init[0], p_init[1], p_init_z_min)

        x_axis = wp.normalize(target_world - p_init)

        world_up = wp.vec3(0.0, 0.0, 1.0)
        if wp.abs(wp.dot(world_up, x_axis)) > 0.999:
            world_up = wp.vec3(1.0, 0.0, 0.0)
        ref_y = wp.normalize(world_up - wp.dot(world_up, x_axis) * x_axis)
        ref_z = wp.normalize(wp.cross(x_axis, ref_y))

        roll = wp.randf(state) * (roll_hi - roll_lo) + roll_lo
        cr = wp.cos(roll)
        sr = wp.sin(roll)
        y_axis = cr * ref_y + sr * ref_z
        z_axis = -sr * ref_y + cr * ref_z

        R_init = wp.mat33(
            x_axis[0], y_axis[0], z_axis[0],
            x_axis[1], y_axis[1], z_axis[1],
            x_axis[2], y_axis[2], z_axis[2],
        )

        wrist_R = R_init * R_inv
        wrist_p = R_init * p_inv + p_init

        # ±noise_scale rpy noise → varied wrist orientation across resets
        nrx = (wp.randf(state) * 2.0 - 1.0) * noise_scale
        nry = (wp.randf(state) * 2.0 - 1.0) * noise_scale
        nrz = (wp.randf(state) * 2.0 - 1.0) * noise_scale
        wrist_R = wrist_R * _grit_rpy2r(wp.vec3(nrx, nry, nrz))

    quat_wxyz = _grit_r2quat_wxyz(wrist_R)

    qpos[w, wrist_qpa + 0] = wrist_p[0]
    qpos[w, wrist_qpa + 1] = wrist_p[1]
    qpos[w, wrist_qpa + 2] = wrist_p[2]
    qpos[w, wrist_qpa + 3] = quat_wxyz[0]   # w
    qpos[w, wrist_qpa + 4] = quat_wxyz[1]   # x
    qpos[w, wrist_qpa + 5] = quat_wxyz[2]   # y
    qpos[w, wrist_qpa + 6] = quat_wxyz[3]   # z

    mocap_pos[w, mocap_id]  = wrist_p
    mocap_quat[w, mocap_id] = quat_wxyz


@wp.kernel
def _grit_fix_wrist_pose_kernel(
    wrist_qpa:  int,
    mocap_id:   int,
    pos:        wp.vec3,
    quat:       wp.quat,                                 # wxyz (codebase convention)
    qpos:       wp.array(dtype=float,    ndim=2),        # type: ignore (nworld, nq)
    mocap_pos:  wp.array(dtype=wp.vec3,  ndim=2),        # type: ignore (nworld, nmocap)
    mocap_quat: wp.array(dtype=wp.quat,  ndim=2),        # type: ignore (nworld, nmocap)
):
    """GPU (sync-free) version of ``fix_wrist_pose``: overwrite every world's
    wrist free-joint qpos + mocap with a fixed pose. Used on the finger-only
    (``control_wrist=False``) reset path to restore the random wrist from
    ``warp_pose_init_capture_launch`` to the DEFAULT pose (same semantics as
    ``fix_wrist_pose``, without a numpy round trip)."""
    w = wp.tid()
    qpos[w, wrist_qpa + 0] = pos[0]
    qpos[w, wrist_qpa + 1] = pos[1]
    qpos[w, wrist_qpa + 2] = pos[2]
    qpos[w, wrist_qpa + 3] = quat[0]   # w
    qpos[w, wrist_qpa + 4] = quat[1]   # x
    qpos[w, wrist_qpa + 5] = quat[2]   # y
    qpos[w, wrist_qpa + 6] = quat[3]   # z
    mocap_pos[w, mocap_id]  = pos
    mocap_quat[w, mocap_id] = quat


@wp.kernel
def _grit_fill_zero_kernel(buf: wp.array(dtype=float)):  # type: ignore
    buf[wp.tid()] = 0.0


@wp.kernel
def _grit_hand_floor_pen_scan_kernel(
    nacon:        wp.array(dtype=int),                    # type: ignore (1,) active contact count
    con_geom:     wp.array(dtype=wp.vec2i),               # type: ignore (naconmax,)
    con_dist:     wp.array(dtype=float),                  # type: ignore (naconmax,)
    con_worldid:  wp.array(dtype=int),                    # type: ignore (naconmax,)
    geom_is_hand: wp.array(dtype=int),                    # type: ignore (ngeom,) 1 = hand geom
    floor_gid:    int,
    out_depth:    wp.array(dtype=float),                  # type: ignore (NWORLD,) deepest hand↔floor penetration (≤0)
):
    """Collect the per-world deepest hand↔floor penetration from the contact array.
    Launch right after forward() (dim=naconmax); out_depth must be zero-filled beforehand."""
    i = wp.tid()
    if i >= nacon[0]:
        return
    dist = con_dist[i]
    if dist >= 0.0:
        return
    g0 = con_geom[i][0]
    g1 = con_geom[i][1]
    if (g0 == floor_gid and geom_is_hand[g1] == 1) or \
       (g1 == floor_gid and geom_is_hand[g0] == 1):
        wp.atomic_min(out_depth, con_worldid[i], dist)


@wp.kernel
def _grit_wrist_floor_liftout_kernel(
    depth:      wp.array(dtype=float),                    # type: ignore (NWORLD,) scan result
    margin:     float,                                    # minimum clearance after resolution (m)
    wrist_qpa:  int,
    mocap_id:   int,
    qpos:       wp.array(dtype=float,   ndim=2),          # type: ignore
    mocap_pos:  wp.array(dtype=wp.vec3, ndim=2),          # type: ignore
):
    """Lift the wrist (+mocap) vertically by (depth + margin) only in penetrating
    worlds — orientation unchanged, no-op for non-penetrating worlds (minimal
    distortion of the spawn distribution)."""
    w = wp.tid()
    d = depth[w]
    if d < 0.0:
        lift = margin - d
        qpos[w, wrist_qpa + 2] += lift
        p = mocap_pos[w, mocap_id]
        mocap_pos[w, mocap_id] = wp.vec3(p[0], p[1], p[2] + lift)


class ObjPcdGpuMixin:
    """Three object-PCD GPU helpers shared by the sub-envs.

    Inherited by both ``SingleHandSubEnv`` and ``LeaderFollowerSubEnv``. The
    required surface is only ``variant_pcd_cache`` / ``renamed_obj_names`` /
    ``variants`` / ``assignment`` / ``d`` / ``mjm`` / ``NWORLD``, which both env
    classes share (the leader-follower env differs only in having 2·n_obj
    leader/follower slots, but the buffers are always sized to the full
    ``renamed_obj_names`` length, so slot indices stay global — LFHandView just
    picks its slots on top of that).
    """

    def _ensure_obj_pcd_gpu_buffers(self) -> None:
        """Lazily upload ``variant_pcd_cache`` (per-variant, per-object body-local
        **object** point clouds) to a GPU buffer ``_obj_pcd_gpu``, plus the
        per-world variant ``_obj_pcd_assignment_gpu``. Named for its contents
        (object PCD); the primary CONSUMER is the finger-closest kernel (each
        finger site → nearest object-PCD vertex), which is why the RL handler
        aliases these as ``_pcd_gpu`` / ``_pcd_assignment``. Allocated once per
        sub-env instance and reused — independent of
        ``setup_warp_pose_init_kernels`` so this also works in
        notebooks (``03_warp_parallel/04``) that never build an RL handler.
        """
        if getattr(self, '_obj_pcd_gpu', None) is not None:
            return
        if not getattr(self, 'variant_pcd_cache', None):
            raise RuntimeError(
                "Finger-closest viz requires variant_pcd_cache. "
                "Build it via build_variant_pcd_cache() during build_sub_env()."
            )
        if not self.renamed_obj_names:
            raise RuntimeError(
                "Finger-closest viz requires at least one object body in the scene."
            )

        device = self.d.qpos.device

        n_variants = len(self.variants)
        n_obj      = len(self.renamed_obj_names)
        # All variants share the same PCD sample size (build_variant_pcd_cache
        # uses a fixed n_sample per call). Pick the first non-empty entry to
        # discover n_pts; if any entry is missing we leave it zeros.
        n_pts = 0
        for vd in self.variant_pcd_cache.values():
            for pcd in vd.values():
                if pcd is not None and len(pcd) > 0:
                    n_pts = int(len(pcd))
                    break
            if n_pts:
                break
        if n_pts == 0:
            raise RuntimeError("variant_pcd_cache is empty — nothing to upload.")

        pcd_arr = np.zeros((n_variants, n_obj, n_pts, 3), dtype=np.float32)
        for v in range(n_variants):
            for o, body_name in enumerate(self.renamed_obj_names):
                pcd = self.variant_pcd_cache.get(v, {}).get(body_name)
                if pcd is None or len(pcd) == 0:
                    continue
                # Defensive: take the first n_pts rows (caches built with
                # different n_sample would otherwise corrupt the stride).
                pcd_arr[v, o] = np.asarray(pcd[:n_pts], dtype=np.float32)

        self._obj_pcd_gpu = wp.array(
            pcd_arr, dtype=wp.vec3, device=device,
        )
        self._obj_pcd_n_pts = int(n_pts)

        # Per-world variant assignment (same buffer used for the init kernel,
        # but allocated independently here so the helper has no implicit
        # dependency on ``setup_warp_pose_init_kernels``).
        self._obj_pcd_assignment_gpu = wp.array(
            np.asarray(self.assignment, dtype=np.int32),
            dtype=int, device=device,
        )

    def _resolve_obj_body_id(self, grasping_obj_name: str | None = None) -> tuple[int, int]:
        """Resolve ``(obj_slot_index, obj_body_id)`` for the target body."""
        if grasping_obj_name is None:
            grasping_obj_name = self.renamed_obj_names[0]
        obj_slot = self.renamed_obj_names.index(grasping_obj_name)
        obj_body_id = int(mujoco.mj_name2id(            # type: ignore
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, grasping_obj_name,  # type: ignore
        ))
        if obj_body_id == -1:
            raise ValueError(f"object body '{grasping_obj_name}' not found.")
        return obj_slot, obj_body_id

    def compute_finger_to_pcd_closest(
        self,
        d,
        finger_site_ids,
        grasping_obj_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-world, per-finger nearest object PCD vertex on the GPU.

        Args:
            d:               warp ``MjData`` (must have ``site_xpos`` / ``xpos`` / ``xmat``
                             populated — call ``mjwarp.forward`` first if necessary).
            finger_site_ids: 1D iterable of site ids (e.g.
                             ``sampled_orch.hand_sensor_site_id_list[2:]`` to skip the
                             arm + palm sites and only query finger-tip sites).
            grasping_obj_name: optional body name of the target object. Defaults to
                             ``renamed_obj_names[0]`` (the legacy graspable body).

        Returns:
            ``(closest_p, min_dist)`` numpy arrays of shapes
            ``(NWORLD, n_finger, 3)`` and ``(NWORLD, n_finger)``. The closest
            point is in **world frame** (no grid offset applied — caller
            adds the offset for rendering).
        """
        self._ensure_obj_pcd_gpu_buffers()

        device = self.d.qpos.device

        # Site ids: re-upload only when the requested set changes.
        site_ids_np = np.asarray(finger_site_ids, dtype=np.int32).reshape(-1)
        if (getattr(self, '_finger_site_ids_cache', None) is None
                or self._finger_site_ids_cache.shape != site_ids_np.shape
                or not np.array_equal(self._finger_site_ids_cache, site_ids_np)):
            self._finger_site_ids_cache = site_ids_np
            self._finger_site_ids_gpu   = wp.array(
                site_ids_np, dtype=int, device=device,
            )

        n_finger = int(site_ids_np.size)
        if n_finger == 0:
            return (np.zeros((self.NWORLD, 0, 3), dtype=np.float32),
                    np.zeros((self.NWORLD, 0),    dtype=np.float32))

        # (Re)allocate output buffers when shape changes.
        if (getattr(self, '_finger_closest_p_wp', None) is None
                or self._finger_closest_p_wp.shape != (self.NWORLD, n_finger)):
            self._finger_closest_p_wp = wp.zeros(
                (self.NWORLD, n_finger), dtype=wp.vec3, device=device,
            )
            self._finger_min_dist_wp  = wp.zeros(
                (self.NWORLD, n_finger), dtype=float,   device=device,
            )

        obj_slot, obj_body_id = self._resolve_obj_body_id(grasping_obj_name)

        wp.launch(
            _grit_finger_closest_pcd_kernel, dim=self.NWORLD,
            inputs=[
                d.site_xpos, d.xpos, d.xmat,
                self._finger_site_ids_gpu,
                self._obj_pcd_gpu,
                self._obj_pcd_assignment_gpu,
                int(obj_slot),
                int(obj_body_id),
                int(n_finger),
                int(self._obj_pcd_n_pts),
                self._finger_closest_p_wp,
                self._finger_min_dist_wp,
            ],
        )

        return (
            self._finger_closest_p_wp.numpy(),    # (NWORLD, n_finger, 3)
            self._finger_min_dist_wp.numpy(),     # (NWORLD, n_finger)
        )


class SingleHandSubEnv(ObjPcdGpuMixin, abc.ABC):
    """
    SingleHandSubEnv is a base class for single hand grasping policy training.
    It is used to create a single hand grasping policy training environment.

    Single-hand & Heterogeneous Object.
    """
    def __init__(self, overall_cfg, hand_util: hand_utils.HandUtils, verbose:bool = False, for_inference:bool = False):
        self.overall_cfg      = overall_cfg
        self.hand_util        = hand_util
        self.hand_name        = hand_util.hand_name
        self.hand_type        = hand_util.hand_type
        self.verbose          = verbose
        self.for_inference    = for_inference

        self.ri_package_asset_dir = epath.Path(home_dir) / "asset"
        self.ri_package_hand_asset_dir = self.ri_package_asset_dir / "dextrous_hand"
        
        if self.overall_cfg.Training.with_table:
            self.floor_xml_path = [
                self.ri_package_asset_dir / self.overall_cfg.Training.floor_xml_path,
                self.ri_package_asset_dir / self.overall_cfg.Training.table_xml_path,
            ]
            self.table_height = self.overall_cfg.Training.table_height
        else:
            self.floor_xml_path = [self.ri_package_asset_dir / self.overall_cfg.Training.floor_xml_path]
            self.table_height = 0.0
        self.use_simple = False if self.overall_cfg.Dataset.Objaverse.use else True
        
        self.NWORLD                  = self.overall_cfg.nworld
        self.CON_PER_ENV             = self.overall_cfg.con_per_env
        self.NJMAX_MUILTPLY          = self.overall_cfg.njmax_muiltiply
        self.N_STEPS                 = self.overall_cfg.n_steps
        self.ACTION_REPEAT           = self.overall_cfg.action_repeat
        # Forwarded to ``make_mjcf_from_spec(... mjcf_arena_memory=...)``
        # — controls the ``<size memory="..."/>`` directive baked into
        # every compiled MjModel. Default ``None`` keeps MuJoCo's auto
        # sizing (fine for single-obj scenes); set in base.yaml (e.g.
        # ``mjcf_arena_memory: "64M"``) to avoid solver-stack OOM in
        # dense multi-obj scenes.
        self.mjcf_arena_memory       = self.overall_cfg.get("mjcf_arena_memory", None)

    def clear_cache(self): 
        # TODO 
        pass  
    
    # Reset hand env
    def reset(self, obj_spec_lst, per_spec_ngeom, n_obj, with_mjwarp:bool = True):
        self.clear_cache()

        self.variants = []
        self.variants_geom_dataid_dict = {}
        self.n_obj = n_obj
        self.obj_spec_lst = obj_spec_lst

        # Route every XML artefact into the dedicated subdir so the notebook's
        # cwd doesn't accumulate scene_* clutter. ``XML_OUTPUT_DIR`` is the
        # single source of truth for that location (default ``xml``).
        _xml_dir = hand_utils.XML_OUTPUT_DIR
        os.makedirs(_xml_dir, exist_ok=True)

        self.parent_xml_path = self._merge_scene_xml(_xml_dir)   # hook (DualHandSubEnv adds the 2nd hand)
        self.parent_spec = mujoco.MjSpec.from_file(self.parent_xml_path) # type: ignore

        # Collision-hull vertex cap — spec stage (for the variant spec.compile() path).
        # See the hand_utils.apply_maxhullvert docstring for details.
        hand_utils.apply_maxhullvert(self.overall_cfg, spec=self.parent_spec)

        # ── contact-sensor match cap (yaml ``Opt.contact_sensor_maxmatch``) ──
        # mjwarp reads this value from an MJCF **custom numeric** (default 64 if
        # absent — mujoco_warp io.py). Hands without the numeric (e.g. robotis)
        # run on the default, and once the match candidates at grasp time (hand
        # geoms × object pieces) exceed 64, the excess contacts are dropped from
        # sensor matching with a "contact match overflow" warning (→ missed touch
        # gating). Inject/update the cfg value as the numeric.
        _maxmatch = int(getattr(self.overall_cfg.Opt, "contact_sensor_maxmatch", 0) or 0)
        if _maxmatch > 0:
            _mm = [n for n in self.parent_spec.numerics
                   if n.name == "contact_sensor_maxmatch"]
            _n = _mm[0] if _mm else self.parent_spec.add_numeric()
            _n.name = "contact_sensor_maxmatch"
            _n.size = 1
            _n.data = [float(_maxmatch)]

        (
            self.relative_xml_path,
            self.skeleton_spec,
            self.renamed_obj_names,
        ) = hand_utils.make_mjcf_from_spec(
            self.parent_spec,
            obj_spec_lst,
            per_spec_ngeom,
            n_obj_per_env=n_obj,
            verbose=True,
            # Inherits xml/ prefix from parent_xml_path (already there).
            output_xml_path=f"{self.parent_xml_path}_w_obj_{n_obj}.xml",
            # Pull from orchestrator cfg if set (yaml ``mjcf_arena_memory``).
            # Falls back to None → MuJoCo's auto-sized arena (fine for
            # single-obj; multi-obj can OOM the solver stack without this).
            mjcf_arena_memory=getattr(self, "mjcf_arena_memory", None),
        )

        # Collision-hull vertex cap — XML stage (for the HandRLParserClass compile path;
        # works around mujoco 3.10's spec.to_xml() not serializing maxhullvert).
        hand_utils.apply_maxhullvert(self.overall_cfg, xml_path=self.relative_xml_path)

        # Create model and data.
        self.make_model_and_data(with_mjwarp=with_mjwarp)

        # Hand sensor, site, joint, ctrl names
        self.get_hand_site_sensor_name()
        self.get_hand_sensor_site_id_list()
        self._build_touch_sensor_index()
        self._build_contact_sensor_index()
        self._build_finger_weights()

        self.mjwarp_step_fn = self.mjwarp_step_fn() # type: ignore


    def _merge_scene_xml(self, xml_dir: str) -> str:
        """Write the parent scene XML (floor/table + hand) and return its path.
        Hook — :class:`DualHandSubEnv` overrides it to include both hands."""
        return merge_mjcfs(
            included_mjcf_files=self.floor_xml_path
            + [self.ri_package_hand_asset_dir / self.hand_util.hand_xml_path],
            output_xml_path=os.path.join(
                xml_dir, f"scene_{self.hand_name}_{self.hand_type}.xml",
            ),
        )

    def _apply_model_driven_ctrl_joint_idx(self) -> None:
        """Recompute ``hand_util.ctrl_joint_idx`` from the compiled model and
        overwrite the rh_info-authored value.

        ``ctrl_joint_idx[a]`` is the finger-block position — index into
        ``qpos[7 : 7+finger_link_num]``, i.e. ``jnt_qposadr − 7`` — of the joint
        that actuator ``a`` directly drives. rh_info hardcodes this per hand, but
        for COUPLED hands whose passive joints are interspersed in qpos the
        hardcoded list can be wrong: e.g. shadow was identity ``[0..17]`` while
        the model needs ``[0,1,2,4,5,6,8,9,10,12,13,14,15,17,18,19,20,21]`` (the
        4 passive ``*J1`` joints sit at qpos-block 3/7/11/16). A wrong index
        scrambles the taxonomy actuator-remap (``_build_taxonomy_qpos_table``),
        finger init, and FK.

        Computing it from ``actuator_trnid → jnt_qposadr`` makes every consumer
        model-consistent — this generalizes the ``active_hand_joint_idx``
        fix so no per-hand hand_info authoring can drift out of sync. Skips hands
        with a non-JOINT (tendon/site) actuator (no 1:1 qpos — keep rh_info); the
        `−7` finger-block offset matches the ``qpos[7:...]`` convention used by
        obs / FK everywhere, and a range check falls back to rh_info if a hand's
        layout ever violates it.
        """
        mjm = self.mjm
        idx = []
        for a in range(int(mjm.nu)):
            if int(mjm.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):  # type: ignore
                return   # tendon/site actuator → no single driven joint; keep rh_info
            idx.append(int(mjm.jnt_qposadr[int(mjm.actuator_trnid[a, 0])]) - 7)
        model_idx = np.asarray(idx, dtype=int)
        fln = int(getattr(self.hand_util, "finger_link_num", int(model_idx.max()) + 1))
        if model_idx.size == 0 or model_idx.min() < 0 or model_idx.max() >= fln:
            return   # unexpected layout → keep rh_info (don't silently break it)
        old = np.asarray(getattr(self.hand_util, "ctrl_joint_idx_array", []), dtype=int)
        if old.size == model_idx.size and not np.array_equal(old, model_idx):
            print_red(
                f"[ctrl_joint_idx] '{self.hand_util.hand_name}': rh_info "
                f"{old.tolist()} disagrees with model {model_idx.tolist()} "
                f"→ using model-driven (rh_info is stale)."
            )
        self.hand_util.ctrl_joint_idx       = model_idx.tolist()
        self.hand_util.ctrl_joint_idx_array = model_idx

    def make_model_and_data(self, with_mjwarp:bool = True):
        self.mj_env:HandRLParserClass = HandRLParserClass(rel_xml_path=self.relative_xml_path, verbose=False)
        self.mj_env.reset() 
        # Test against the spec env built with the max geom count
        self.mj_env.build_obj_pcd_cache(self.renamed_obj_names, n_sample=1024, verbose=True) 
        
        # Model
        self.mjm = self.mj_env.model

        # ── SDF octree fix ────────────────────────────────────────────────
        # HandRLParserClass compiled ``self.mjm`` from the round-tripped XML
        # (``relative_xml_path``). ``needsdf`` is NOT an MJCF attribute, so the
        # XML load only rebuilds an SDF octree for the ONE mesh referenced by an
        # ``mjGEOM_SDF`` geom in the skeleton (the anchor variant). Every other
        # variant's object mesh — swapped in per-world via ``geom_dataid`` — then
        # has ``mesh_octadr = -1`` (no octree), so its SDF collider is inert and
        # the object tunnels straight through the floor. ``self.skeleton_spec``
        # still carries ``needsdf=True`` on every object mesh (propagated in
        # ``make_mjcf_from_spec``), so recompiling from the spec rebuilds octrees
        # for ALL of them. The spec is exactly what produced ``relative_xml_path``,
        # so the compiled model is structurally identical (same geom/mesh ids) —
        # only the octrees differ. No-op unless SDF meshes are present.
        _sdf_meshes = [m for m in self.skeleton_spec.meshes
                       if getattr(m, "needsdf", False)]
        if _sdf_meshes:
            self.mjm = self.skeleton_spec.compile()
            if self.verbose:
                print(f"[SDF] recompiled skeleton from spec → SDF octrees rebuilt "
                      f"for {len(_sdf_meshes)} object mesh(es)")

        self.nv = self.mjm.nv
        self.nbody = self.mjm.nbody
        self.nsensor = self.mjm.nsensor

        # Model-driven ctrl→finger-joint index (overrides mis-authored rh_info —
        # e.g. shadow's identity list; see _apply_model_driven_ctrl_joint_idx).
        self._apply_model_driven_ctrl_joint_idx()

        # ── Finger actuator gain override ────────────────────────────────────
        # Retune finger position-servo gains from ``cfg.finger_gain`` (persisted
        # in the config snapshot → reapplied at eval/deploy) with GRIT_FINGER_*
        # env vars overriding for quick experiments. No-op when neither is set.
        # (Logic in grit.util.hand_utils.apply_finger_gain_override.)
        hand_utils.apply_finger_gain_override_from_config(
            self.mjm, self.overall_cfg, hand_cfg=getattr(self.hand_util, "hand_cfg", None))

        self.mjm.opt.iterations      = self.overall_cfg.Opt.iterations
        self.mjm.opt.ccd_iterations  = self.overall_cfg.Opt.ccd_iterations
        self.mjm.opt.ls_iterations   = self.overall_cfg.Opt.ls_iterations
        # impratio >= 5 suppresses contact-force oscillation ("ringing") at rest.
        # The config floor is respected; raising it only when the config is too low.
        self.mjm.opt.impratio        = max(float(self.overall_cfg.Opt.impratio), 5.0)
        self.mjm.opt.gravity         = self.overall_cfg.Opt.gravity
        self.mjm.opt.cone            = self.overall_cfg.Opt.cone
        self.mjm.opt.solver          = self.overall_cfg.Opt.solver
        self.mjm.opt.timestep        = self.overall_cfg.sim_dt
        self.mj_env.dt               = self.mjm.opt.timestep

        # ── SDF contact-count tuning ──────────────────────────────────────
        # mujoco_warp's mesh-SDF narrowphase seeds ``opt.sdf_initpoints`` points
        # per collision PAIR and refines each with ``opt.sdf_iterations`` gradient
        # steps. MuJoCo's default 40 seeds is wildly too many here: a hand with
        # ~48 geoms touching ONE object SDF emits ~48×40 ≈ 1900 contacts (vs
        # ~130 for the convex-mesh path), flooding the contact/constraint buffers.
        # 6 seeds cuts that ~7× (→ ~260) while keeping a stable contact manifold;
        # more iterations tighten seed convergence. Config-/env-overridable.
        # NOTE: this only controls contact COUNT. It does NOT fix the ~2-3 cm
        # surface inflation of mujoco_warp's mesh-SDF (its octree is capped at
        # depth 4 and ``mesh.octree_maxdepth`` is ignored), so SDF contacts still
        # register slightly outside the visual mesh — an upstream SDF-fidelity
        # limitation, not tunable from here.
        if _sdf_meshes:
            import os as _os
            self.mjm.opt.sdf_initpoints = int(_os.environ.get(
                "GRIT_SDF_INITPOINTS",
                getattr(self.overall_cfg.Opt, "sdf_initpoints", 6)))
            self.mjm.opt.sdf_iterations = int(_os.environ.get(
                "GRIT_SDF_ITERATIONS",
                getattr(self.overall_cfg.Opt, "sdf_iterations", 20)))

        self.create_variants()
        self.build_variant_pcd_cache(n_sample=1024, verbose=self.verbose)

        # Data 
        if with_mjwarp: 
            self.mjd = mujoco.MjData(self.mjm) # type: ignore
            mujoco.mj_forward(self.mjm, self.mjd)  # type: ignore

            # NCONMAX: the larger of CON_PER_ENV and actual ncon + buffer (nworld not considered)
            # NJMAX: use max() instead of min(). In the initial rest state nefc is close
            #   to 0, so min() picked (nefc+50)*mult over NCONMAX*mult and overflowed.
            NCONMAX = max(self.CON_PER_ENV, int(self.mjd.ncon) + 100)
            NJMAX   = max(NCONMAX * self.NJMAX_MUILTPLY, int(self.mjd.nefc) + 200)
            if self.verbose:
                print(f"nworld={self.NWORLD}, nconmax={NCONMAX}, {self.mjd.ncon}, njmax={NJMAX}, {self.mjd.nefc}")
            
            # BUGFIX: the parallel-warp contact-pair list is pre-filtered ONCE
            # from the SHARED skeleton model's (1D) geom_contype/conaffinity at
            # put_model time — per-world override is impossible. The skeleton is
            # copied from a single argmax(per_spec_ngeom) anchor, so a slot's
            # collision flags reflect only that anchor; if the anchor's
            # bottom_watertight is a dummy/non-colliding region, NO variant's
            # bottom collides (sample-dependent on which object anchored).
            # Union the per-slot collision flags across all variants so every
            # slot any variant collides on is collision-enabled here.
            self._union_object_geom_collision_onto_skeleton()

            # Declare the MuJoCo Warp Model and Data
            self.m = mjwarp.put_model(self.mjm)
            self.d = mjwarp.make_data(
                            self.mjm, 
                            nworld=self.NWORLD, 
                            nconmax=NCONMAX, 
                            njmax=NJMAX,
                            )

            self.heterogeneous_env_setup()

            # Per-world finger gain handle (for DR) — it expands and replaces
            # gainprm/biasprm as (NWORLD, nu), so it must be created **before**
            # ScopedCapture (the capture bakes array pointers; later set()/randomize()
            # calls are in-place kernel writes to the same arrays, hence graph-safe
            # and callable every tick).
            # The GRIT_FINGER_KP env-var remains a static override of the mjm
            # defaults and is naturally reflected in this handle's kp0/kv0 base.
            from grit.util.warp_actuator_gain import PerWorldFingerGains
            try:
                self.finger_gains = PerWorldFingerGains(
                    self.m, self.mjm, self.NWORLD, verbose=self.verbose)
            except AssertionError:
                self.finger_gains = None      # scene without position servos (defensive)

            # The ScopedCapture graph must reference the correct GPU buffers.
            # Fix: the capture order was changed.
            # - Before: make_data → ScopedCapture → create_variants → heterogeneous_env_setup
            # - After: make_data → create_variants → heterogeneous_env_setup → reset_data → ScopedCapture
            if self.for_inference:
                mjwarp.reset_data(self.m, self.d)
                self.warp_obj_pose_init(
                    table_height    = self.table_height,
                    xy_offset_range = [-0.2, 0.2],
                )
                mjwarp.forward(self.m, self.d)  # update kinematics to new poses

                with wp.ScopedCapture() as self.capture_step:
                    mjwarp.step(self.m, self.d)

                with wp.ScopedCapture() as self.capture_forward:
                    mjwarp.forward(self.m, self.d)

    def get_hand_site_sensor_name(self): 
        self.hand_sensor_names = [n for n in self.mj_env.sensor_names if ("touch" in n) and ("rh_" in n or "lh_" in n or "arm_" in n)] 
        self.hand_site_name = [n for n in self.mj_env.site_names if "touch" in n and ("rh_" in n or "lh_" in n or "arm_" in n)]
        # ── opt-in: rh_info-declared canonical touch-sensor layout ──────────
        # The substring filter above takes EVERY "touch" sensor in XML order.
        # Everything downstream (palm_sensor_idx=0 / fore_arm_sensor_idx=1,
        # tip_body_idx, the 16-entry per-taxonomy face-dir tables, obs touch
        # block) assumes the canonical [palm, forearm, 5×(prox, mid, tip)]
        # layout — which the curated hand XMLs happen to satisfy. A vendor XML
        # with extra touch sites (e.g. yaw/roll/fingertip → 27) or a
        # different order silently scrambles those semantics and inflates obs.
        # Hands that set ``use_declared_touch_sensor_order = True`` in rh_info
        # get the layout rebuilt from ``hand_touch_sensors`` / ``fore_arm_touch_sensor``
        # instead; other hands are untouched (their checkpoints depend on it).
        if bool(getattr(self.hand_util, "use_declared_touch_sensor_order", False)):
            declared = list(getattr(self.hand_util, "hand_touch_sensors", []) or [])
            forearm  = getattr(self.hand_util, "fore_arm_touch_sensor", None)
            canonical = ([declared[0]] + ([forearm] if forearm else []) + declared[1:]) if declared else []
            missing = [n for n in canonical if n not in self.mj_env.sensor_names]
            if missing:
                raise ValueError(f"[{self.hand_name}] declared touch sensors missing in model: {missing}")
            mjm = self.mj_env.model
            sites = []
            for n in canonical:
                sid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SENSOR, n)                       # type: ignore
                assert mjm.sensor_objtype[sid] == mujoco.mjtObj.mjOBJ_SITE, f"{n}: touch sensor must target a site"  # type: ignore
                sites.append(mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_SITE, int(mjm.sensor_objid[sid])))  # type: ignore
            if self.verbose or len(canonical) != len(self.hand_sensor_names):
                print_red(f"[touch-layout] {self.hand_name}: XML filter gave {len(self.hand_sensor_names)} touch sensors "
                          f"→ using rh_info canonical order ({len(canonical)}): {canonical[:3]} …")
            self.hand_sensor_names = canonical
            self.hand_site_name    = sites
        self.hand_joint_names = [n for n in self.mj_env.active_joint_names if "rh_" in n.lower() or "lh_" in n.lower()]
        self.active_hand_joint_idx = np.array([self.hand_joint_names.index(n) for n in self.mj_env.active_joint_names]) 
        self.hand_ctrl_names = [n for n in self.mj_env.ctrl_names if "rh_" in n.lower() or "lh_" in n.lower()] 

    def get_hand_sensor_site_id_list(self): 
        self.hand_sensor_site_id_list = [] 
        self.hand_sensor_body_id_list = np.zeros(len(self.hand_sensor_names))
        self.hand_sensor_body_name_list = [] 
        for i, sensor_name in enumerate(self.hand_sensor_names):  
            sensor_id   = mujoco.mj_name2id(self.mj_env.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) # type: ignore
            obj_type    = self.mj_env.model.sensor_objtype[sensor_id]
            obj_id      = self.mj_env.model.sensor_objid[sensor_id]

            current_site_id = -1 # initial value (when the sensor is not site-based)
            if obj_type == mujoco.mjtObj.mjOBJ_SITE: # type: ignore 
                current_site_id = obj_id # ID of the site the sensor is attached to
                body_id = self.mj_env.model.site_bodyid[obj_id]
            elif obj_type == mujoco.mjtObj.mjOBJ_BODY: # type: ignore
                body_id = obj_id
            elif obj_type == mujoco.mjtObj.mjOBJ_GEOM: # type: ignore
                body_id = self.mj_env.model.geom_bodyid[obj_id]
            else:
                raise ValueError(f"Unsupported sensor object type: {obj_type}")

            self.hand_sensor_site_id_list.append(current_site_id)
            self.hand_sensor_body_id_list[i] = body_id 
            self.hand_sensor_body_name_list.append(self.mj_env.body_names[body_id]) 
        
        # Sensor-space taxonomy masks. do_finger_tip=True (precision grasps)
        # restricts the active slots to TIP links (+ explicit palm) — the non-tip
        # links of active fingers move to rest, so contact_neg penalizes wrapping.
        sensor_parts = [n.lower() for n in self.hand_sensor_body_name_list]
        tip_or_palm = np.array(["palm" in p for p in sensor_parts])
        tip_or_palm[np.asarray(getattr(self.hand_util, "tip_body_idx", []) or [], dtype=int)] = True
        finger_specific_body_mask_lst = []
        self.finger_specific_body_idx_lst = []
        for names, do_tip in zip(self.hand_util.taxonomy_specific_finger_names_array,
                                 self.hand_util.taxonomy_do_finger_tip_array):
            finger_specific_body_mask_lst.append(np.array(
                [any(f in p.lower() for f in names) for p in self.hand_util.rh_body_parts]))  # type: ignore
            mask = np.array([any(f in p for f in names) for p in sensor_parts])
            self.finger_specific_body_idx_lst.append(mask & tip_or_palm if do_tip else mask)

        # self.taxonomy_specific_finger_link_mask_array = np.array(finger_specific_body_mask_lst)
        self.taxonomy_specific_finger_body_mask_array = np.array(self.finger_specific_body_idx_lst)
        self.use_palm = self.taxonomy_specific_finger_body_mask_array[:,0]

        # Finger-group id per sensor slot (palm/each finger = one group, -1 = non-hand
        # slots such as the arm) — for per-finger contact_ratio counting. Group names
        # are the union of taxonomy specific_finger_names (same substring matching as the masks).
        group_names = list(dict.fromkeys(
            n for names in self.hand_util.taxonomy_specific_finger_names_array for n in names))
        self.sensor_finger_group_names = group_names
        self.sensor_finger_group_id = np.array(
            [next((g for g, gn in enumerate(group_names) if gn in p), -1) for p in sensor_parts],
            dtype=np.int32)

        # Ctrl(actuator)-space finger-group id — for per-finger motor torque
        # balancing (torque_regulate). Groups ``d.actuator_force[w, a]`` by finger
        # so torque does not concentrate on a single joint motor. ctrl index ==
        # actuator index (n_ctrl = mjm.nu), so it aligns directly with actuator_force.
        # Same substring matching as the sensor version (on ctrl_names); -1 = non-finger such as arm/wrist.
        ctrl_parts = [str(n).lower() for n in self.mj_env.ctrl_names]
        self.ctrl_finger_group_id = np.array(
            [next((g for g, gn in enumerate(group_names) if gn in p), -1) for p in ctrl_parts],
            dtype=np.int32)


        finger_specific_ctrl_name_lst = []  
        for specific_finger_names in self.hand_util.taxonomy_specific_ctrl_names_array:
            mask = np.array([any(finger in part.lower() for finger in specific_finger_names) for part in self.mj_env.ctrl_names])
            finger_specific_ctrl_name_lst.append(mask)
        self.taxonomy_specific_ctrl_mask_array = np.array(finger_specific_ctrl_name_lst)
        self.taxonomy_rest_ctrl_mask_array = np.array([~mask for mask in finger_specific_ctrl_name_lst]) 

        tip_body_idx       = self.hand_util.tip_body_idx # For contact in sensor_body_name_list
        self.tip_body_idx_arr   = np.array(tip_body_idx)
        self.tip_body_mask      = np.zeros(len(self.hand_sensor_body_name_list), dtype=bool)
        self.tip_body_mask[self.tip_body_idx_arr] = True

    # ──────────────────────────────────────────────────────────────────────
    # Touch sensor parsing
    # ──────────────────────────────────────────────────────────────────────
    def _build_touch_sensor_index(self):
        """Cache (sensor_adr, sensor_dim) for the forearm + per-link hand
        touch sensors declared in ``hand_util.fore_arm_touch_sensor`` and
        ``hand_util.hand_touch_sensors`` (both come from
        ``hand_info/{hand_name}/rh_info.py``).

        Populates:
            self.fore_arm_touch_sensor_name : str | None
            self.hand_touch_sensor_names    : list[str]
            self.touch_sensor_index         : dict[name → (adr, dim)]

        Sensor names that are missing from the compiled model (e.g. the
        chosen XML doesn't include the touch site) are silently skipped.
        Called from ``reset()`` after the model is compiled.
        """
        m = self.mjm
        forearm_name = getattr(self.hand_util, "fore_arm_touch_sensor", None)
        hand_names   = list(getattr(self.hand_util, "hand_touch_sensors", []) or [])

        index: dict[str, tuple[int, int]] = {}

        def _add(name: str):
            if not name: return
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)  # type: ignore
            if sid < 0:
                if self.verbose:
                    print_red(f"[touch_sensor_index] '{name}' not in model — skipped")
                return
            index[name] = (int(m.sensor_adr[sid]), int(m.sensor_dim[sid]))

        _add(forearm_name)
        for n in hand_names:
            _add(n)

        self.fore_arm_touch_sensor_name = forearm_name if forearm_name in index else None
        self.hand_touch_sensor_names    = [n for n in hand_names if n in index]
        self.touch_sensor_index         = index

    def _build_sensor_body_index(self):
        """Cache ``sensor_name → host_body_id`` for every touch + contact sensor.

        host body = whatever body owns the site/body/geom the sensor is attached
        to (matches the mapping used by ``HandRLParserClass._build_touch_color_cache``).
        Used by ``get_per_body_world_mask`` to ground-truth-gate sensor readings
        against ``d.contact`` per-world.
        """
        m = self.mjm
        index: dict[str, int] = {}

        def _resolve(name: str):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)  # type: ignore
            if sid < 0:
                return
            ot = int(m.sensor_objtype[sid])
            oi = int(m.sensor_objid[sid])
            if   ot == int(mujoco.mjtObj.mjOBJ_SITE): bid = int(m.site_bodyid[oi])  # type: ignore
            elif ot == int(mujoco.mjtObj.mjOBJ_BODY): bid = oi                       # type: ignore
            elif ot == int(mujoco.mjtObj.mjOBJ_GEOM): bid = int(m.geom_bodyid[oi])  # type: ignore
            else: return
            index[name] = bid

        for n in getattr(self, "hand_touch_sensor_names", []) or []:
            _resolve(n)
        for n in getattr(self, "hand_contact_sensor_names", []) or []:
            _resolve(n)
        for n in getattr(self, "hand_contact_sensor_for_object_names", []) or []:
            _resolve(n)
        for n in getattr(self, "hand_contact_sensor_for_table_names", []) or []:
            _resolve(n)
        if getattr(self, "fore_arm_touch_sensor_name", None):
            _resolve(self.fore_arm_touch_sensor_name)

        self.sensor_body_index = index

    def get_per_body_world_mask(self, data) -> "dict[int, np.ndarray] | None":
        """For every hand sensor's host body B, return ``(NWORLD,) bool`` where
        ``True`` means at least one contact in ``d.contact`` involves body B in
        that world.

        This is the most precise per-(world, body) gate: it bypasses
        ``d.sensordata`` entirely and reads the canonical per-world contact
        ledger ``(d.contact.geom, d.contact.worldid)``. Useful when mjwarp's
        sensor pipeline appears to alias values across worlds that share a
        variant — this method's output is independent of any sensor-buffer
        path because it consults the contact buffer directly.

        Returns ``None`` if the data object lacks a warp-style contact buffer
        (i.e. CPU MjData), or if the model has no hand sensors.
        """
        contact = getattr(data, "contact", None)
        nacon   = getattr(data, "nacon",   None)
        if contact is None or nacon is None:
            return None
        if not hasattr(nacon, "numpy"):
            return None
        body_index = getattr(self, "sensor_body_index", {}) or {}
        if not body_index:
            return None

        NW = int(self.NWORLD)
        unique_bids = sorted(set(int(b) for b in body_index.values()))
        out = {b: np.zeros(NW, dtype=bool) for b in unique_bids}

        nacon_val = int(nacon.numpy()[0])
        if nacon_val == 0:
            return out

        # geom array: (naconmax, 2) of geom-pair ids; worldid: (naconmax,)
        geom_arr  = contact.geom.numpy()[:nacon_val]
        wid_arr   = contact.worldid.numpy()[:nacon_val]
        # geom_bodyid is model-level (skeleton); shared across all variants
        # because variants only override OBJECT geoms, not hand bodies.
        geom_bodyid = np.asarray(self.mjm.geom_bodyid)

        valid = (wid_arr >= 0) & (wid_arr < NW)
        for c in np.where(valid)[0]:
            w = int(wid_arr[c])
            g1, g2 = int(geom_arr[c, 0]), int(geom_arr[c, 1])
            if 0 <= g1 < geom_bodyid.size:
                b = int(geom_bodyid[g1])
                if b in out:
                    out[b][w] = True
            if 0 <= g2 < geom_bodyid.size:
                b = int(geom_bodyid[g2])
                if b in out:
                    out[b][w] = True
        return out

    def get_world_active_mask(self, data) -> "np.ndarray | None":
        """Return ``(NWORLD,) bool`` — True for worlds with ≥1 active contact in
        ``d.contact`` (the canonical mjwarp per-world contact buffer).

        This is a ground-truth per-world signal independent of how
        ``d.sensordata`` was populated. Use it to mask out sensor readings
        for worlds where no contact is happening, defending against any
        sensor-buffer aliasing / misattribution upstream.

        Returns ``None`` for CPU ``MjData`` (single-world; the question is
        vacuous), and ``None`` if the data object lacks the expected fields.
        """
        contact = getattr(data, "contact", None)
        nacon   = getattr(data, "nacon",   None)
        if contact is None or nacon is None:
            return None
        if not hasattr(nacon, "numpy") or not hasattr(getattr(contact, "worldid", None), "numpy"):
            return None
        nacon_val = int(nacon.numpy()[0])
        mask = np.zeros(int(self.NWORLD), dtype=bool)
        if nacon_val > 0:
            wid = contact.worldid.numpy()[:nacon_val]
            valid = (wid >= 0) & (wid < int(self.NWORLD))
            if valid.any():
                mask[np.unique(wid[valid])] = True
        return mask

    def get_touch_sensor_values(
            self,
            data=None,
            gate_by_world_contact:    bool = False,
            gate_by_per_body_contact: bool = False,
        ) -> dict:
        """Return ``{sensor_name: reading}`` for forearm + hand touch sensors.

        Args:
            data: One of
                - ``None``         → defaults to ``self.mjd`` (CPU MjData).
                - CPU ``MjData``   → ``data.sensordata`` is ``(nsensor,)``.
                  Each entry returned is a Python ``float`` (dim==1) or 1-D
                  ``np.ndarray`` (dim>1, e.g. multi-channel sensors).
                - warp ``Data``    → ``data.sensordata.numpy()`` is
                  ``(NWORLD, nsensor)``. Each entry returned is a
                  ``(NWORLD,)`` array (dim==1) or ``(NWORLD, dim)``.
            gate_by_world_contact: warp only. When True, force every reading
                  in any world W with **zero** contacts in ``d.contact`` to
                  exactly 0. Defends against sensor-buffer aliasing where
                  e.g. world W might inherit a stale touch reading from
                  another world that shares the same variant.
            gate_by_per_body_contact: warp only. STRICTEST gate. For each
                  sensor S with host body B, force the reading on world W
                  to 0 whenever ``d.contact`` has no contact involving B in
                  world W. This bypasses ``d.sensordata`` entirely for the
                  gating decision and uses the per-world contact ledger
                  ``(d.contact.geom, d.contact.worldid)`` as ground truth —
                  immune to any aliasing in mjwarp's sensor accumulation.
        """
        if data is None:
            data = self.mjd
        is_warp = hasattr(getattr(data, "sensordata", None), "numpy")
        sd = data.sensordata.numpy() if is_warp else np.asarray(data.sensordata)

        out: dict = {}
        for name, (adr, dim) in self.touch_sensor_index.items():
            if is_warp:
                v = sd[:, adr:adr + dim]                # (NWORLD, dim)
                out[name] = np.array(v[:, 0]) if dim == 1 else np.array(v)
            else:
                v = sd[adr:adr + dim]                   # (dim,)
                out[name] = float(v[0]) if dim == 1 else np.asarray(v)

        if is_warp and gate_by_world_contact:
            mask = self.get_world_active_mask(data)
            if mask is not None:
                # mask True = world has contact; gate inverse worlds to 0.
                m_f = mask.astype(out[next(iter(out))].dtype) if out else None
                for name, val in out.items():
                    if val.ndim == 1:                       # (NWORLD,)
                        out[name] = val * m_f
                    else:                                    # (NWORLD, dim)
                        out[name] = val * m_f[:, None]

        if is_warp and gate_by_per_body_contact:
            body_masks = self.get_per_body_world_mask(data)
            body_index = getattr(self, "sensor_body_index", {}) or {}
            if body_masks is not None:
                for name, val in out.items():
                    bid = body_index.get(name)
                    if bid is None or bid not in body_masks:
                        continue
                    m_f = body_masks[bid].astype(val.dtype)
                    if val.ndim == 1:
                        out[name] = val * m_f
                    else:
                        out[name] = val * m_f[:, None]
        return out

    def get_touch_sensor_array(
            self,
            data=None,
            gate_by_world_contact:    bool = False,
            gate_by_per_body_contact: bool = False,
        ):
        """Convenience wrapper returning ``(forearm, hand)`` arrays.

        ``gate_by_world_contact`` / ``gate_by_per_body_contact`` are
        forwarded — see ``get_touch_sensor_values`` for semantics.

        - CPU input  → ``forearm`` is ``float`` (or ``None`` if absent),
                       ``hand``    is shape ``(n_hand_touch,)``.
        - warp input → ``forearm`` is shape ``(NWORLD,)`` (or ``None``),
                       ``hand``    is shape ``(NWORLD, n_hand_touch)``.
        Sensor order in ``hand`` matches ``self.hand_touch_sensor_names``
        (= ``hand_util.hand_touch_sensors`` minus any missing sensors).
        """
        values = self.get_touch_sensor_values(
            data,
            gate_by_world_contact    = gate_by_world_contact,
            gate_by_per_body_contact = gate_by_per_body_contact,
        )
        forearm = values.get(self.fore_arm_touch_sensor_name) \
                  if self.fore_arm_touch_sensor_name else None
        if not self.hand_touch_sensor_names:
            return forearm, np.zeros(0)
        first = values[self.hand_touch_sensor_names[0]]
        if isinstance(first, np.ndarray) and first.ndim >= 1 and first.shape[0] > 1:
            # warp path → stack along last axis to (NWORLD, n_hand)
            hand = np.stack([values[n] for n in self.hand_touch_sensor_names], axis=-1)
        else:
            hand = np.array([values[n] for n in self.hand_touch_sensor_names], dtype=float)
        return forearm, hand

    # ──────────────────────────────────────────────────────────────────────
    # Contact sensor parsing
    # ──────────────────────────────────────────────────────────────────────
    # MuJoCo ``<contact ... num="N" data="...">`` packs every contact as a
    # fixed-order record. Two layouts occur in this project:
    #
    #   data="found pos normal"        → 7-float record  [found(1) pos(3) normal(3)]
    #   data="found force pos normal"  → 10-float record [found(1) force(3) pos(3) normal(3)]
    #
    # The per-record width is ``PACKET = sensor_dim // num`` (num is the
    # sensor's intprm[2]). force lives at offset 1 (between found and pos) —
    # NOT at the end — because MuJoCo emits fields in the canonical order
    # found, force, torque, dist, pos, normal, tangent. So for the 10-float
    # layout the offsets are: found=0, force=1:4, pos=4:7, normal=7:10.
    # ``CONTACT_PACKET_SIZES`` enumerates the layouts we know how to parse.
    CONTACT_PACKET_SIZES = (7, 10)

    @staticmethod
    def _contact_packet_offsets(psize: int):
        """Return ``(force_off, pos_off, normal_off, has_force)`` for a record
        of width ``psize`` (``found`` is always at offset 0). ``force_off`` is
        ``None`` when the record has no force channel."""
        if psize >= 10:                       # [found, force, pos, normal]
            return 1, 4, 7, True
        return None, 1, 4, False              # [found, pos, normal]

    def _build_one_contact_index(self, names_attr: str, label: str):
        """Build ``{name → (adr, dim, num)}`` for one source list on
        ``hand_util``. ``num`` is the sensor's authoritative packet count
        (``intprm[2]``); the per-record width ``dim // num`` must be one of
        :attr:`CONTACT_PACKET_SIZES`. Sensors missing from the compiled model
        or with an unrecognised record width are silently skipped.
        """
        m = self.mjm
        contact_names = list(getattr(self.hand_util, names_attr, []) or [])
        index: dict[str, tuple[int, int, int]] = {}
        for name in contact_names:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)  # type: ignore
            if sid < 0:
                if self.verbose:
                    print_red(f"[{label}_index] '{name}' not in model — skipped")
                continue
            adr = int(m.sensor_adr[sid])
            dim = int(m.sensor_dim[sid])
            num = int(m.sensor_intprm[sid][2])          # authoritative packet count
            if dim <= 0 or num <= 0 or (dim % num) != 0 \
                    or (dim // num) not in self.CONTACT_PACKET_SIZES:
                if self.verbose:
                    print_red(
                        f"[{label}_index] '{name}' dim={dim} num={num} "
                        f"→ record width {dim // num if num else 0} unsupported "
                        f"(expected one of {self.CONTACT_PACKET_SIZES}) — skipped")
                continue
            index[name] = (adr, dim, num)
        ordered = [n for n in contact_names if n in index]
        return ordered, index

    def _build_contact_sensor_index(self):
        """Cache contact-sensor (adr, dim, num) for three families:

          * ``hand_util.hand_contact_sensors``            → self-collision
            (subtree2 reference; only fires when the hand collides with
            itself / wrist). The forearm is excluded. 
          * ``hand_util.hand_contact_sensors_for_object`` → object-restricted
            (``body2 = obj`` reference; one sensor per (hand_part, slot)).
          * ``hand_util.hand_contact_sensors_for_table``  → table/floor
            restricted (``body2 = base_table`` if present, else ``world``;
            one sensor per hand_part).

        Populates:
            self.hand_contact_sensor_names              : list[str]
            self.contact_sensor_index                   : dict[name → (adr, dim, num)]
            self.hand_contact_sensor_for_object_names   : list[str]
            self.contact_sensor_for_object_index        : dict[name → (adr, dim, num)]
            self.hand_contact_sensor_for_table_names    : list[str]
            self.contact_sensor_for_table_index         : dict[name → (adr, dim, num)]
        """
        # self-collision only (the forearm is not included)
        self.hand_contact_sensor_names, self.contact_sensor_index = \
            self._build_one_contact_index("hand_contact_sensors", "contact_sensor")
        # object contact only
        self.hand_contact_sensor_for_object_names, self.contact_sensor_for_object_index = \
            self._build_one_contact_index(
                "hand_contact_sensors_for_object", "contact_sensor_for_object",
            )
        # table contact only
        self.hand_contact_sensor_for_table_names, self.contact_sensor_for_table_index = \
            self._build_one_contact_index(
                "hand_contact_sensors_for_table", "contact_sensor_for_table",
            )
        # Now that all indexes exist, build the unified sensor → body_id map.
        self._build_sensor_body_index()

    # ──────────────────────────────────────────────────────────────────────
    # Per-hand finger weights (sensor space — palm + arm + fingers)
    # ──────────────────────────────────────────────────────────────────────
    def _build_finger_weights(self):
        """Build per-hand finger weight arrays in **sensor space**.

        Port of the JAX-era ``_get_finger_weights`` helper from
        ``grit_with_obj_coll_v2.py`` (and v3 / drill variants). Both arrays
        are parallel to ``self.hand_sensor_site_id_list`` /
        ``self.hand_sensor_body_name_list`` (palm = 0, fore_arm = 1,
        fingers = 2…), so a kernel that iterates sensors / sites can
        weight each slot directly without re-indexing.

        Two output arrays:

          * ``self.finger_weights``         (n_sensors,) float32
              Used by **distance-style** rewards (e.g. the object_grasping
              task's site→PCD closest-vertex weighted MSE). Tip sites are
              DAMPED here (multiply ≤ 1) so the non-tip joints dominate the
              weighted average — matches the JAX ``finger_weights``
              semantics where ``link_dist = sum(d² · w · mask) / sum(mask)``
              concentrates the reward signal on proximal joints once the
              tip is close to the object.

          * ``self.finger_contact_weights`` (n_sensors,) float32
              Used by **contact-style** rewards. Tip sites are DAMPED here
              too, but only mildly (×0.5, first tip ×0.75) versus the
              distance array's ×0.25 — so a full wrap (proximal + medial
              links) outweighs a fingertip-only touch. palm + fore_arm then
              get an extra ×4 boost after normalisation (the legacy
              ``finger_weights_contact.at[:, 0].multiply(4.0)`` /
              ``[:, 1].multiply(4.0)`` step) — strongly rewards palm /
              forearm contact even when most fingers are inactive.
              Resulting scale (tesollo/robotis/allex): palm 4.61,
              proximal/medial 1.15, tips 0.58 (thumb tip 0.86).

        Per-hand emphasis (verbatim port of JAX ``_get_finger_weights``):

        ============  ===================================================
        hand_name     distance / contact emphasis (indexed by tip_body_idx)
        ============  ===================================================
        tesollo /     ``finger_weights[tip[1:]] *= 0.25``,
        inspire /     ``finger_weights[tip[:1]] *= 0.5`` and
        allex /       ``finger_contact_weights[tip[1:]] *= 0.5``,
        shadow /      ``finger_contact_weights[tip[:1]] *= 0.75``
        robotis_sh5 /
        wuji_hand2 /  (5-finger, **thumb = first** tip slot — its touch-sensor order is
                       palm, th, ff, mf, rf, lf, so
                       ``tip_body_idx = [4,7,10,13,16]`` matches tesollo's
                       exactly.)
        allegro       ``finger_weights[tip[:3]] *= 0.25``,
                      ``finger_weights[tip[-1:]] *= 0.5`` and
                      ``finger_contact_weights[tip[:3]] *= 0.5``,
                      ``finger_contact_weights[tip[-1:]] *= 0.75``
                      (4-finger and **thumb = last** tip slot.)
        others        uniform 1.0 (equal-weight fallback)
        ============  ===================================================

        Both arrays are renormalised to ``sum == n_sensors`` after the
        per-hand multipliers, then the contact array applies its palm +
        arm ×4 boost. The arm slot (``[1]``) of ``finger_weights`` is
        **not** zeroed here — callers that want to exclude the forearm
        from a distance reward should AND with the taxonomy mask (which
        has a 0 at the arm slot by construction).
        """
        hand_name = getattr(self.hand_util, "hand_name", "")
        n_sensors = len(self.hand_sensor_body_name_list)
        tip_idx   = np.asarray(self.hand_util.tip_body_idx, dtype=int).reshape(-1)

        fw  = np.ones(n_sensors, dtype=np.float32)
        fcw = np.ones(n_sensors, dtype=np.float32)

        # 5-finger hands whose FIRST tip slot is the thumb. wuji_hand2
        # share tesollo's ``tip_body_idx = [4,7,10,13,16]`` (touch-sensor order
        # palm, th, ff, mf, rf, lf), so they take this branch — without it they
        # fell through to the uniform 1.0 fallback, which left their fingertips
        # at 1.74x tesollo's contact weight and 3.18x its distance weight and
        # made contact quality incomparable across hands.
        if hand_name in ("inspire", "tesollo", "wuji_hand2"):
            fw[tip_idx[1:]] *= 0.25
            fw[tip_idx[:1]] *= 0.5
            fcw[tip_idx[1:]] *= 0.5
            fcw[tip_idx[:1]] *= 0.75
        elif hand_name == "allex":
            fw[tip_idx[1:]] *= 0.25
            fw[tip_idx[:1]] *= 0.5
            fcw[tip_idx[1:]] *= 0.5
            fcw[tip_idx[:1]] *= 0.75
        elif hand_name == "allegro":
            fw[tip_idx[:3]] *= 0.25
            fw[tip_idx[-1:]] *= 0.5
            fcw[tip_idx[:3]] *= 0.5
            fcw[tip_idx[-1:]] *= 0.75
        elif hand_name == "shadow":
            fw[tip_idx[1:]] *= 0.25
            fw[tip_idx[:1]] *= 0.5
            fcw[tip_idx[1:]] *= 0.5
            fcw[tip_idx[:1]] *= 0.75
        elif hand_name == "robotis_sh5":
            fw[tip_idx[1:]] *= 0.25
            fw[tip_idx[:1]] *= 0.5
            fcw[tip_idx[1:]] *= 0.5
            fcw[tip_idx[:1]] *= 0.75

        s_fw  = float(fw.sum())
        if s_fw > 0.0 and n_sensors > 0:
            fw  = (fw  / s_fw)  * float(n_sensors)
        s_fcw = float(fcw.sum())
        if s_fcw > 0.0 and n_sensors > 0:
            fcw = (fcw / s_fcw) * float(n_sensors)
        # Contact-only palm + fore_arm boost (post normalisation, matches
        # JAX). Mirrors ``palm_sensor_idx = 0`` / ``fore_arm_sensor_idx = 1``
        # convention from ``hand_info/<hand>/rh_info.py``.
        if n_sensors >= 1:
            fcw[0] *= 4.0
        if n_sensors >= 2:
            fcw[1] *= 4.0

        self.finger_weights         = fw.astype(np.float32,  copy=False)
        self.finger_contact_weights = fcw.astype(np.float32, copy=False)

    def _get_contact_values_from_index(
            self,
            data,
            contact_index: dict,
            gate_by_touch:            bool = False,
            touch_eps:                float = 0.0,
            gate_by_world_contact:    bool = False,
            gate_by_per_body_contact: bool = False,
        ) -> dict:
        """Internal: parse ``found / pos / normal`` packets from a
        ``{name → (adr, dim, num)}`` index, with optional gating. Shared by
        the self-collision / ``_for_object`` / ``_for_table`` contact-sensor
        accessors.

        ``gate_by_touch`` maps each contact-sensor name back to its matching
        touch sensor by stripping the longest matching contact suffix
        (``_body_obj_contact`` → object, ``_table_contact`` → table,
        ``_self_coll`` → self-collision) and appending ``_touch``. Sensors
        without a matching touch entry are passed through unchanged.
        """
        if data is None:
            data = self.mjd
        is_warp = hasattr(getattr(data, "sensordata", None), "numpy")
        sd = data.sensordata.numpy() if is_warp else np.asarray(data.sensordata)

        out: dict = {}
        for name, (adr, dim, num) in contact_index.items():
            psize = dim // num
            f_off, p_off, n_off, has_force = self._contact_packet_offsets(psize)
            if is_warp:
                block = sd[:, adr:adr + dim].reshape(sd.shape[0], num, psize)
                out[name] = {
                    "found":  block[..., 0],
                    "pos":    block[..., p_off:p_off + 3],
                    "normal": block[..., n_off:n_off + 3],
                    "force":  (block[..., f_off:f_off + 3] if has_force
                               else np.zeros(block.shape[:-1] + (3,), dtype=block.dtype)),
                }
            else:
                block = sd[adr:adr + dim].reshape(num, psize)
                out[name] = {
                    "found":  np.asarray(block[:, 0]),
                    "pos":    np.asarray(block[:, p_off:p_off + 3]),
                    "normal": np.asarray(block[:, n_off:n_off + 3]),
                    "force":  (np.asarray(block[:, f_off:f_off + 3]) if has_force
                               else np.zeros((num, 3), dtype=block.dtype)),
                }

        if gate_by_touch:
            touch_values = self.get_touch_sensor_values(
                data,
                gate_by_world_contact    = (is_warp and gate_by_world_contact),
                gate_by_per_body_contact = (is_warp and gate_by_per_body_contact),
            )
            for name, info in out.items():
                # Try suffixes in **longest-first** order so that
                # ``_body_obj_contact`` is matched before the shorter
                # ``_obj_contact`` / ``_contact`` substrings.
                if name.endswith("_body_obj_contact"):
                    base = name[: -len("_body_obj_contact")]
                elif name.endswith("_obj_contact"):
                    base = name[: -len("_obj_contact")]
                elif name.endswith("_table_contact"):
                    base = name[: -len("_table_contact")]
                elif name.endswith("_self_coll"):
                    base = name[: -len("_self_coll")]
                elif name.endswith("_contact"):
                    base = name[: -len("_contact")]
                else:
                    continue
                # Two touch-naming conventions coexist (mirrors the handler
                # logic in ``object_grasping.py``): some hands name the touch
                # sensor ``<base>_touch`` (e.g. robotis_sh5), others insert an
                # extra ``_body`` → ``<base>_body_touch`` (e.g. allegro / tesollo).
                # Stripping ``_body_obj_contact`` already removed the ``_body``,
                # and ``_table_contact`` / ``_self_coll`` never carried it, so try
                # the plain form first and fall back to the ``_body_touch`` variant.
                # Trying only ``<base>_touch`` silently no-ops the gate on those
                # hands (touch_name absent → ``continue``).
                cands = [base + "_touch"]
                if not base.endswith("_body"):
                    cands.append(base + "_body_touch")
                touch_name = next((c for c in cands if c in touch_values), None)
                if touch_name is None:
                    continue
                tv = np.asarray(touch_values[touch_name]).reshape(-1)
                if is_warp:
                    valid = (tv > touch_eps).astype(info["found"].dtype)
                    info["found"]  = info["found"]  * valid[:, None]
                    info["pos"]    = info["pos"]    * valid[:, None, None]
                    info["normal"] = info["normal"] * valid[:, None, None]
                    info["force"]  = info["force"]  * valid[:, None, None]
                else:
                    if not (float(tv[0]) > touch_eps):
                        info["found"]  = np.zeros_like(info["found"])
                        info["pos"]    = np.zeros_like(info["pos"])
                        info["normal"] = np.zeros_like(info["normal"])
                        info["force"]  = np.zeros_like(info["force"])

        if is_warp and gate_by_world_contact:
            mask = self.get_world_active_mask(data)
            if mask is not None:
                m_f = mask.astype(np.float32)
                for name, info in out.items():
                    info["found"]  = np.asarray(info["found"])  * m_f[:, None]
                    info["pos"]    = np.asarray(info["pos"])    * m_f[:, None, None]
                    info["normal"] = np.asarray(info["normal"]) * m_f[:, None, None]
                    info["force"]  = np.asarray(info["force"])  * m_f[:, None, None]

        if is_warp and gate_by_per_body_contact:
            body_masks = self.get_per_body_world_mask(data)
            body_index = getattr(self, "sensor_body_index", {}) or {}
            if body_masks is not None:
                for name, info in out.items():
                    bid = body_index.get(name)
                    if bid is None or bid not in body_masks:
                        continue
                    m_f = body_masks[bid].astype(np.float32)
                    info["found"]  = np.asarray(info["found"])  * m_f[:, None]
                    info["pos"]    = np.asarray(info["pos"])    * m_f[:, None, None]
                    info["normal"] = np.asarray(info["normal"]) * m_f[:, None, None]
                    info["force"]  = np.asarray(info["force"])  * m_f[:, None, None]
        return out

    def get_contact_sensor_values(
            self,
            data=None,
            gate_by_touch:            bool = False,
            touch_eps:                float = 0.0,
            gate_by_world_contact:    bool = False,
            gate_by_per_body_contact: bool = False,
        ) -> dict:
        """Parse ``found / pos / normal`` blocks for every hand contact sensor.

        Args:
            data: ``None`` → ``self.mjd`` (CPU). CPU MjData → 1-world fields.
                  warp Data → fields are batched along the leading axis.
            gate_by_touch: If True, zero out ``found / pos / normal`` for
                  every (world, packet) whose matching touch sensor reads
                  ``<= touch_eps``. The match is name-based: ``X_contact``
                  is paired with ``X_touch``. Sensors without a touch
                  counterpart are passed through unchanged.

                  Why: parallel-warp ``d.sensordata`` can hold uninitialised
                  contact bytes before the first real touch event, which
                  surface as ghost markers / RL features. The MuJoCo touch
                  sensor (force magnitude) stays at exactly 0 in that
                  state, so it is a reliable gate.
            touch_eps: Force threshold (N). Default ``0.0`` = strictly > 0.
            gate_by_world_contact: warp only. When True, additionally force
                  every world W with **zero** contacts in ``d.contact`` to
                  read all-zero ``found / pos / normal``. Defense against
                  sensor-buffer aliasing — independent of how
                  ``d.sensordata`` was populated, because it consults the
                  canonical ``d.contact.worldid`` per-world ledger.
            gate_by_per_body_contact: warp only. STRICTEST gate. For each
                  contact sensor X_contact with host body B, force the
                  reading on world W to all-zero whenever ``d.contact``
                  contains no contact involving body B in world W. Uses
                  ``(d.contact.geom, d.contact.worldid)`` as ground truth —
                  fully independent of ``d.sensordata`` so any aliasing in
                  mjwarp's sensor pipeline cannot leak through.

        Returns:
            ``dict[name → {'found', 'pos', 'normal'}]`` with the following shapes:

            =========  ==============================  ====================================
            field      CPU (mjData)                    warp (Data, NWORLD worlds)
            =========  ==============================  ====================================
            found      ``(num,)``                      ``(NWORLD, num)``
            pos        ``(num, 3)``                    ``(NWORLD, num, 3)``
            normal     ``(num, 3)``                    ``(NWORLD, num, 3)``
            =========  ==============================  ====================================
        """
        return self._get_contact_values_from_index(
            data if data is not None else self.mjd,
            self.contact_sensor_index,
            gate_by_touch            = gate_by_touch,
            touch_eps                = touch_eps,
            gate_by_world_contact    = gate_by_world_contact,
            gate_by_per_body_contact = gate_by_per_body_contact,
        )

    def get_contact_sensor_for_object_values(
            self,
            data=None,
            gate_by_touch:            bool = False,
            touch_eps:                float = 0.0,
            gate_by_world_contact:    bool = False,
            gate_by_per_body_contact: bool = False,
        ) -> dict:
        """Same as ``get_contact_sensor_values`` but reads the
        ``hand_contact_sensors_for_object`` family — contact sensors whose
        ``refid`` is the graspable object body, so the packets only
        accumulate contacts between the hand link and that object (not e.g.
        floor / table / other-hand contacts).

        Sensor names follow the convention ``<body>_body_obj_contact``;
        the matching touch sensor for ``gate_by_touch`` is ``<body>_body_touch``
        (the ``_obj_contact`` suffix is stripped before matching).

        Returns the same dict shape as ``get_contact_sensor_values`` —
        one ``{found, pos, normal}`` block per sensor, keyed by sensor name.
        """
        return self._get_contact_values_from_index(
            data if data is not None else self.mjd,
            self.contact_sensor_for_object_index,
            gate_by_touch            = gate_by_touch,
            touch_eps                = touch_eps,
            gate_by_world_contact    = gate_by_world_contact,
            gate_by_per_body_contact = gate_by_per_body_contact,
        )

    def get_contact_sensor_for_table_values(
            self,
            data=None,
            gate_by_touch:            bool = False,
            touch_eps:                float = 0.0,
            gate_by_world_contact:    bool = False,
            gate_by_per_body_contact: bool = False,
        ) -> dict:
        """Same as ``get_contact_sensor_values`` but reads the
        ``hand_contact_sensors_for_table`` family — contact sensors whose
        ``refid`` is the table body (``base_table`` when present, else the
        ``world`` body that owns the floor geom). The packets thus only
        accumulate contacts between the hand link and the table/floor —
        the canonical signal for "hand pushed into the support surface".

        Sensor names follow the convention ``<body>_table_contact``; the
        matching touch sensor for ``gate_by_touch`` is ``<body>_touch``
        (the ``_table_contact`` suffix is stripped before matching).

        Returns the same dict shape as ``get_contact_sensor_values`` —
        one ``{found, pos, normal}`` block per sensor, keyed by sensor name.
        """
        return self._get_contact_values_from_index(
            data if data is not None else self.mjd,
            self.contact_sensor_for_table_index,
            gate_by_touch            = gate_by_touch,
            touch_eps                = touch_eps,
            gate_by_world_contact    = gate_by_world_contact,
            gate_by_per_body_contact = gate_by_per_body_contact,
        )

    def get_contact_sensor_array(
            self,
            data=None,
            gate_by_touch:            bool = False,
            touch_eps:                float = 0.0,
            gate_by_world_contact:    bool = False,
            gate_by_per_body_contact: bool = False,
        ):
        """Stacked-across-sensors variant of ``get_contact_sensor_values``.

        ``gate_by_touch`` / ``touch_eps`` / ``gate_by_world_contact`` /
        ``gate_by_per_body_contact`` are forwarded — see
        ``get_contact_sensor_values`` for the gating rationale.

        Returns ``(found, pos, normal)`` tuples whose sensor axis matches
        ``self.hand_contact_sensor_names``:

        =========  ==========================  ====================================
        field      CPU                         warp
        =========  ==========================  ====================================
        found      ``(n_sensors, num)``        ``(NWORLD, n_sensors, num)``
        pos        ``(n_sensors, num, 3)``     ``(NWORLD, n_sensors, num, 3)``
        normal     ``(n_sensors, num, 3)``     ``(NWORLD, n_sensors, num, 3)``
        =========  ==========================  ====================================
        """
        return self._stack_contact_values(
            self.get_contact_sensor_values(
                data,
                gate_by_touch            = gate_by_touch,
                touch_eps                = touch_eps,
                gate_by_world_contact    = gate_by_world_contact,
                gate_by_per_body_contact = gate_by_per_body_contact,
            ),
            self.hand_contact_sensor_names,
        )

    def get_contact_sensor_for_table_array(
            self,
            data=None,
            gate_by_touch:            bool = False,
            touch_eps:                float = 0.0,
            gate_by_world_contact:    bool = False,
            gate_by_per_body_contact: bool = False,
        ):
        """Stacked-across-sensors variant of
        ``get_contact_sensor_for_table_values``. Same shapes as
        ``get_contact_sensor_array``; sensor axis follows
        ``self.hand_contact_sensor_for_table_names``.
        """
        return self._stack_contact_values(
            self.get_contact_sensor_for_table_values(
                data,
                gate_by_touch            = gate_by_touch,
                touch_eps                = touch_eps,
                gate_by_world_contact    = gate_by_world_contact,
                gate_by_per_body_contact = gate_by_per_body_contact,
            ),
            self.hand_contact_sensor_for_table_names,
        )

    def get_contact_sensor_for_object_array(
            self,
            data=None,
            gate_by_touch:            bool = False,
            touch_eps:                float = 0.0,
            gate_by_world_contact:    bool = False,
            gate_by_per_body_contact: bool = False,
        ):
        """Stacked-across-sensors variant of
        ``get_contact_sensor_for_object_values``. Same shapes as
        ``get_contact_sensor_array``; sensor axis follows
        ``self.hand_contact_sensor_for_object_names``.
        """
        return self._stack_contact_values(
            self.get_contact_sensor_for_object_values(
                data,
                gate_by_touch            = gate_by_touch,
                touch_eps                = touch_eps,
                gate_by_world_contact    = gate_by_world_contact,
                gate_by_per_body_contact = gate_by_per_body_contact,
            ),
            self.hand_contact_sensor_for_object_names,
        )

    @staticmethod
    def _stack_contact_values(values: dict, order: list):
        """Stack ``{name → {found, pos, normal}}`` into ``(found, pos, normal)``
        with the sensor axis ordered by ``order``."""
        if not order:
            return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
        sample = values[order[0]]["found"]
        is_warp = sample.ndim == 2  # CPU has (num,), warp has (NWORLD, num)
        axis = 1 if is_warp else 0
        found  = np.stack([values[n]["found"]  for n in order], axis=axis)
        pos    = np.stack([values[n]["pos"]    for n in order], axis=axis)
        normal = np.stack([values[n]["normal"] for n in order], axis=axis)
        return found, pos, normal

    # ──────────────────────────────────────────────────────────────────────
    # Per-world contact-state visualisation
    # ──────────────────────────────────────────────────────────────────────
    # Per-category contact colours. Each contact family contributes its own
    # colour and overlapping families are mean-blended (see
    # ``compute_contact_state_rgba``):
    #
    #   obj   on alone        → green
    #   table on alone        → orange
    #   self  on alone        → red
    #   obj + table           → mean(green, orange)         (yellow-ish)
    #   obj + self            → mean(green, red)            (yellow / olive)
    #   table + self          → mean(orange, red)           (darker red-orange)
    #   obj + table + self    → mean of all three           (brownish)
    CONTACT_STATE_COLOR_OBJ       = (0.00, 0.90, 0.20, 1.0)  # green
    CONTACT_STATE_COLOR_TABLE     = (1.00, 0.55, 0.00, 1.0)  # orange
    CONTACT_STATE_COLOR_SELF      = (1.00, 0.10, 0.10, 1.0)  # red

    # Backward-compat aliases: the old code exposed BLUE/GREEN/RED constants
    # for the (obj on, self on)/(obj on)/(self on) tuple. Blending replaces
    # the special-case BLUE; GREEN/RED are aliased to the per-category
    # OBJ/SELF colours so existing callers that override
    # ``color_very_good`` / ``color_bad`` continue to work.
    CONTACT_STATE_COLOR_GOOD      = (0.15, 0.30, 1.00, 1.0)  # legacy blue (unused)
    CONTACT_STATE_COLOR_VERY_GOOD = CONTACT_STATE_COLOR_OBJ
    CONTACT_STATE_COLOR_BAD       = CONTACT_STATE_COLOR_SELF

    def _ensure_contact_color_state(self):
        """Lazy-build the caches needed by ``compute_contact_state_rgba``:
            - body_id → [geom_id ...]   (skeleton; shared across variants)
            - (NWORLD, ngeom, 4) per-world base rgba (variant-keyed)
            - body_id → [sensor names] for the general / for_object / touch
              families
            - sorted list of ``colorable`` body ids (have ≥1 contact sensor)
        """
        if hasattr(self, "_color_body_to_geoms"):
            return
        if not getattr(self, "variants", None):
            raise RuntimeError(
                "compute_contact_state_rgba requires variants/assignment to be "
                "built (call build_sub_env first)."
            )

        vm0 = self.variants[0]
        body_to_geoms: dict = {}
        for gid in range(int(vm0.ngeom)):
            body_to_geoms.setdefault(int(vm0.geom_bodyid[gid]), []).append(gid)
        self._color_body_to_geoms = body_to_geoms

        NW    = int(self.NWORLD)
        ngeom = int(vm0.ngeom)
        base  = np.empty((NW, ngeom, 4), dtype=np.float64)
        for w in range(NW):
            base[w] = self.variants[int(self.assignment[w])].geom_rgba
        self._color_per_world_base_rgba = base

        gen_pb:   dict = {}
        obj_pb:   dict = {}
        tbl_pb:   dict = {}
        touch_pb: dict = {}
        for n, b in (getattr(self, "sensor_body_index", {}) or {}).items():
            if n in self.contact_sensor_index:
                gen_pb.setdefault(int(b), []).append(n)
            if n in self.contact_sensor_for_object_index:
                obj_pb.setdefault(int(b), []).append(n)
            if n in getattr(self, "contact_sensor_for_table_index", {}) or {}:
                tbl_pb.setdefault(int(b), []).append(n)
            if n in self.touch_sensor_index:
                touch_pb.setdefault(int(b), []).append(n)
        self._color_general_per_body = gen_pb
        self._color_for_obj_per_body = obj_pb
        self._color_for_tbl_per_body = tbl_pb
        self._color_touch_per_body   = touch_pb
        # Touch-only bodies (e.g. forearm in every hand_info/*/rh_info.py: only
        # `*_arm_part_touch` is registered, no contact sensor) must still be
        # paintable so that hits against the floor / table show up. Without
        # them, contact sensors decide the colourable set and the forearm is
        # invisible to the visualiser even when its touch sensor reads > 0.
        self._color_colorable_body_ids = sorted(
            set(gen_pb) | set(obj_pb) | set(tbl_pb) | set(touch_pb)
        )
        self._color_touch_only_body_ids = (
            set(touch_pb) - set(gen_pb) - set(obj_pb) - set(tbl_pb)
        )

        # ── Body-category sets for touch-only (forearm) partner classification ──
        # The forearm carries only a touch sensor (no contact-family sensor), so
        # touch alone can't tell self vs obstacle vs object. ``compute_contact_
        # state_rgba`` classifies its ``d.contact`` partner instead:
        #   world / floor (+ ``base_table``) → obstacle (orange)
        #   object subtree                   → IGNORED  (forearm takes no object contact)
        #   any other hand body              → self-collision (red)
        mjm = self.mjm
        obstacle = {0}                                   # worldbody owns the floor geom
        _tb = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "base_table")  # type: ignore
        if _tb >= 0:
            obstacle.add(int(_tb))
        self._color_obstacle_bids = obstacle
        # object subtree body ids (top + bottom watertight, per slot)
        _children: dict = {}
        for _b in range(int(mjm.nbody)):
            _children.setdefault(int(mjm.body_parentid[_b]), []).append(_b)
        obj_bids: set = set()
        for _on in (getattr(self, "renamed_obj_names", []) or []):
            _rid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, _on)  # type: ignore
            if _rid < 0:
                continue
            _stack = [int(_rid)]
            while _stack:
                _cur = _stack.pop()
                obj_bids.add(_cur)
                _stack.extend(c for c in _children.get(_cur, []) if c != _cur)
        self._color_object_bids = obj_bids

    def _touch_only_partner_active(self, data):
        """Classify each touch-only body's ``d.contact`` partners per world.

        Bodies that carry only a touch sensor (the forearm) can't tell from
        touch alone what they hit. Read the per-world contact ledger
        ``(d.contact.geom, d.contact.worldid)`` and bucket each partner:
            * world / floor / ``base_table``  →  obstacle  (orange)
            * object subtree                   →  IGNORED   (no colour)
            * any other hand body              →  self-coll (red)

        Returns ``{bid: {"self": (NW,)bool, "tbl": (NW,)bool}}`` for every
        touch-only body, or ``None`` for CPU ``MjData`` (no warp contact
        buffer) so the caller falls back to the conservative red.
        """
        contact = getattr(data, "contact", None)
        nacon   = getattr(data, "nacon",   None)
        if contact is None or nacon is None or not hasattr(nacon, "numpy"):
            return None
        NW  = int(self.NWORLD)
        out = {int(b): {"self": np.zeros(NW, dtype=bool),
                        "tbl":  np.zeros(NW, dtype=bool)}
               for b in self._color_touch_only_body_ids}
        if not out:
            return out
        nv = int(nacon.numpy()[0])
        if nv == 0:
            return out
        geom = contact.geom.numpy()[:nv]
        wid  = contact.worldid.numpy()[:nv]
        gb   = np.asarray(self.mjm.geom_bodyid)
        obstacle = self._color_obstacle_bids
        objset   = self._color_object_bids
        valid = (wid >= 0) & (wid < NW)
        for c in np.where(valid)[0]:
            w = int(wid[c])
            g1, g2 = int(geom[c, 0]), int(geom[c, 1])
            b1 = int(gb[g1]) if 0 <= g1 < gb.size else -1
            b2 = int(gb[g2]) if 0 <= g2 < gb.size else -1
            for me, other in ((b1, b2), (b2, b1)):
                if me not in out:
                    continue
                if other in obstacle:
                    out[me]["tbl"][w] = True
                elif other in objset or other < 0:
                    continue                       # object / invalid → ignored
                else:
                    out[me]["self"][w] = True
        return out

    def _per_body_world_active(self, values_dict: dict, body_to_names: dict) -> dict:
        """For each body id in body_to_names, return ``(NWORLD,) bool``: True
        iff any sensor mapped to that body has ``found > 0.5`` in that world.
        """
        NW = int(self.NWORLD)
        out = {b: np.zeros(NW, dtype=bool) for b in self._color_colorable_body_ids}
        for bid, names in body_to_names.items():
            if bid not in out:
                continue
            for n in names:
                info = values_dict.get(n)
                if info is None:
                    continue
                f = np.asarray(info["found"])
                if f.ndim >= 2:
                    a = (f > 0.5).reshape(f.shape[0], -1).any(axis=-1)
                else:
                    a = np.full(NW, bool((f > 0.5).any()))
                out[bid] = out[bid] | a
        return out

    def _per_body_world_touch_value(self, touch_values: dict, body_to_touch_names: dict) -> dict:
        """For each body id, return ``(NWORLD,) float`` = max touch reading
        across all of that body's touch sensors per world. Drives both the
        garbage filter (``> eps`` ⇒ valid) and the colour blend factor.
        """
        NW = int(self.NWORLD)
        out = {b: np.zeros(NW, dtype=np.float64) for b in self._color_colorable_body_ids}
        for bid, names in body_to_touch_names.items():
            if bid not in out:
                continue
            for n in names:
                v = touch_values.get(n)
                if v is None:
                    continue
                v_arr = np.asarray(v).reshape(-1).astype(np.float64)
                if v_arr.size != NW:
                    continue
                out[bid] = np.maximum(out[bid], v_arr)
        return out

    def compute_contact_state_rgba(
            self,
            data,
            touch_eps:    float = 0.0,
            touch_scale:  float = 10.0,
            touch_gamma:  float = 0.5,
            gate_by_touch:            bool = True,
            gate_by_per_body_contact: bool = True,
            color_obj       = None,
            color_table     = None,
            color_self      = None,
            # ── legacy aliases (backward compat) ────────────────────────
            color_good      = None,   # unused under blending; kept for callers
            color_very_good = None,   # alias for ``color_obj``
            color_bad       = None,   # alias for ``color_self``
        ) -> np.ndarray:
        """Compute ``(NWORLD, ngeom, 4)`` per-world RGBA for the contact-state
        visualisation. Drop directly into
        ``HandRLParserClass.mjwarp_render(..., per_world_geom_rgba=rgba)``.

        Each (world, body) gets a target colour built by **blending** the
        active contact families:

            * obj   on  →  contributes :data:`CONTACT_STATE_COLOR_OBJ`   (green)
            * table on  →  contributes :data:`CONTACT_STATE_COLOR_TABLE` (orange)
            * self  on  →  contributes :data:`CONTACT_STATE_COLOR_SELF`  (red)

            target_rgb = mean(active category colours)        (per-world)

        The blended target is then blended into the variant's base rgba
        with intensity proportional to the body's max touch force::

            t   = clip(touch / touch_scale, 0, 1) ** touch_gamma
            rgb = (1 - t) * variant_base_rgb + t * target_rgb

        Examples:
            obj only          → green
            table only        → orange
            self only         → red
            obj + table       → mean(green, orange)  (yellow-ish)
            obj + self        → mean(green, red)     (yellow / olive)
            table + self      → mean(orange, red)    (darker red-orange)
            obj + table + self→ mean of all three    (brownish)

        On-bits are AND-filtered by ``touch > touch_eps`` so contact packets
        without a coincident touch reading do not paint anything (garbage
        filter on top of the upstream ``gate_by_touch`` already passed to
        ``get_contact_sensor_*_values``).

        Args:
            data: warp ``Data``. (CPU ``MjData`` is single-world; per-world
                  rgba does not apply.)
            touch_eps / touch_scale / touch_gamma: see formula above.
            gate_by_touch / gate_by_per_body_contact: forwarded to
                ``get_contact_sensor_values`` /
                ``get_contact_sensor_for_object_values`` /
                ``get_contact_sensor_for_table_values`` so the on-bits
                themselves are read after upstream gating.
            color_obj / color_table / color_self: optional 4-tuple
                overrides; default to the class constants.
            color_good / color_very_good / color_bad: legacy aliases. The
                old "BLUE for obj+self" override (``color_good``) is
                ignored under blending; ``color_very_good`` falls back to
                ``color_obj`` and ``color_bad`` to ``color_self`` for
                backward compatibility.
        """
        self._ensure_contact_color_state()
        NW = int(self.NWORLD)

        cv_g = self.get_contact_sensor_values(
            data,
            gate_by_touch            = gate_by_touch,
            touch_eps                = touch_eps,
            gate_by_per_body_contact = gate_by_per_body_contact,
        )
        cv_o = self.get_contact_sensor_for_object_values(
            data,
            gate_by_touch            = gate_by_touch,
            touch_eps                = touch_eps,
            gate_by_per_body_contact = gate_by_per_body_contact,
        )
        cv_t = self.get_contact_sensor_for_table_values(
            data,
            gate_by_touch            = gate_by_touch,
            touch_eps                = touch_eps,
            gate_by_per_body_contact = gate_by_per_body_contact,
        )
        tv = self.get_touch_sensor_values(
            data, gate_by_per_body_contact = gate_by_per_body_contact,
        )

        gen_on  = self._per_body_world_active(cv_g, self._color_general_per_body)
        obj_on  = self._per_body_world_active(cv_o, self._color_for_obj_per_body)
        tbl_on  = self._per_body_world_active(cv_t, self._color_for_tbl_per_body)
        touch_v = self._per_body_world_touch_value(tv, self._color_touch_per_body)

        # Touch garbage filter: contact must coincide with touch
        for bid in list(gen_on.keys()):
            if bid in touch_v:
                active = touch_v[bid] > touch_eps
                gen_on[bid] = gen_on[bid] & active
                obj_on[bid] = obj_on[bid] & active
                if bid in tbl_on:
                    tbl_on[bid] = tbl_on[bid] & active

        # Resolve per-category target colours (legacy aliases honoured).
        c_obj_src   = color_obj   if color_obj   is not None else color_very_good
        c_self_src  = color_self  if color_self  is not None else color_bad
        c_obj   = np.asarray(c_obj_src   or self.CONTACT_STATE_COLOR_OBJ,   dtype=np.float64)
        c_table = np.asarray(color_table or self.CONTACT_STATE_COLOR_TABLE, dtype=np.float64)
        c_self  = np.asarray(c_self_src  or self.CONTACT_STATE_COLOR_SELF,  dtype=np.float64)
        # ``color_good`` (legacy BLUE for the obj+self special case) is
        # intentionally unused under the blending model.
        del color_good

        # Per-world d.contact partner classification for touch-only bodies
        # (forearm): self (red) vs obstacle/table (orange); object ignored.
        partner_cls = self._touch_only_partner_active(data)

        out = self._color_per_world_base_rgba.copy()
        for bid in self._color_colorable_body_ids:
            gid_list = self._color_body_to_geoms.get(int(bid), [])
            if not gid_list:
                continue
            self_b = gen_on.get(int(bid),  np.zeros(NW, dtype=bool))
            obj_b  = obj_on.get(int(bid),  np.zeros(NW, dtype=bool))
            tbl_b  = tbl_on.get(int(bid),  np.zeros(NW, dtype=bool))
            tv_b   = touch_v.get(int(bid), np.zeros(NW, dtype=np.float64))

            if int(bid) in self._color_touch_only_body_ids:
                # No contact sensor wired for this body (the forearm). Classify
                # its actual d.contact partner: self-collision (red) vs
                # obstacle/table (orange). Object contact is intentionally
                # IGNORED — the forearm never takes object contact.
                if partner_cls is not None and int(bid) in partner_cls:
                    cls    = partner_cls[int(bid)]
                    obj_b  = np.zeros(NW, dtype=bool)
                    self_b = cls["self"]
                    tbl_b  = cls["tbl"]
                else:
                    # CPU MjData (no per-world contact ledger) → conservative red.
                    touch_only_active = tv_b > touch_eps
                    obj_b  = np.zeros(NW, dtype=bool)
                    tbl_b  = np.zeros(NW, dtype=bool)
                    self_b = touch_only_active

            # Mean-blend per-world: target = sum(active_color) / n_active.
            obj_f  = obj_b.astype(np.float64)
            tbl_f  = tbl_b.astype(np.float64)
            self_f = self_b.astype(np.float64)
            n_active = obj_f + tbl_f + self_f                    # (NW,)
            paint = n_active > 0
            if not np.any(paint):
                continue

            target = np.zeros((NW, 4), dtype=np.float64)
            # Sum active per-category colours per (world).
            target[:, :3] = (
                obj_f[:, None]  * c_obj[:3]   +
                tbl_f[:, None]  * c_table[:3] +
                self_f[:, None] * c_self[:3]
            )
            denom = np.where(paint, n_active, 1.0)               # avoid div-0
            target[:, :3] /= denom[:, None]

            t_lin = np.clip(tv_b / max(touch_scale, 1e-9), 0.0, 1.0)
            t = t_lin if touch_gamma == 1.0 else np.power(t_lin, touch_gamma)
            t = t * paint.astype(np.float64)
            if not np.any(t > 0):
                continue
            for gid in gid_list:
                base    = out[:, gid]
                blended = (1.0 - t)[:, None] * base[:, :3] + t[:, None] * target[:, :3]
                new_rgba = base.copy()
                new_rgba[:, :3] = blended
                out[:, gid] = new_rgba           # alpha preserved
        return out

    def mjwarp_step_fn(self):
        def mjwarp_step(
            ctrl: wp.array2d(dtype=float, ndim=2),                #     type: ignore
            qpos_in: wp.array2d(dtype=float, ndim=2),             #  type: ignore
            qvel_in: wp.array2d(dtype=float, ndim=2),             #  type: ignore
            qacc_in: wp.array2d(dtype=float, ndim=2),             #  type: ignore
            qacc_warmstart_in: wp.array2d(dtype=float, ndim=2),   #  type: ignore
            mocap_pos_in: wp.array2d(dtype=wp.vec3, ndim=2),      #  type: ignore
            mocap_quat_in: wp.array2d(dtype=wp.quat, ndim=2),     #  type: ignore
            qpos_out: wp.array(dtype=float, ndim=2),              #  type: ignore
            qvel_out: wp.array(dtype=float, ndim=2),              #  type: ignore
            xpos_out: wp.array2d(dtype=wp.vec3, ndim=2),          #  type: ignore
            xmat_out: wp.array2d(dtype=wp.mat33, ndim=2),         #  type: ignore
            qacc_out: wp.array2d(dtype=float, ndim=2),            #  type: ignore
            qacc_warmstart_out: wp.array2d(dtype=float, ndim=2),  #  type: ignore
            mocap_pos_out: wp.array(dtype=wp.vec3, ndim=2),       #  type: ignore
            mocap_quat_out: wp.array2d(dtype=wp.quat, ndim=2),    #  type: ignore
            sensordata_out: wp.array2d(dtype=float, ndim=2),      #  type: ignore
            worldid_out: wp.array(dtype=int),                     #  type: ignore
            ):
            # print_red(f'mjwarp class : {self}, env id : {self.env_id}')
            # jax.debug.breakpoint() 

            wp.copy(self.d.ctrl, ctrl)
            wp.copy(self.d.qpos, qpos_in)
            wp.copy(self.d.qvel, qvel_in)
            wp.copy(self.d.qacc, qacc_in)
            wp.copy(self.d.qacc_warmstart, qacc_warmstart_in)
            wp.copy(self.d.mocap_pos, mocap_pos_in)
            wp.copy(self.d.mocap_quat, mocap_quat_in) 
            # TODO(team): remove this hard coding substeps
            # To see the value, print it outside of JAX-traced/jitted functions.
            # For debugging, print here before entering JAX:
            
            for i in range(self.N_STEPS):
                mjwarp.step(self.m, self.d)
            
            wp.copy(qpos_out, self.d.qpos)
            wp.copy(qvel_out, self.d.qvel)
            wp.copy(xpos_out, self.d.xpos)
            wp.copy(xmat_out, self.d.xmat)
            wp.copy(qacc_out, self.d.qacc)
            wp.copy(qacc_warmstart_out, self.d.qacc_warmstart)
            wp.copy(mocap_pos_out, self.d.mocap_pos)
            wp.copy(mocap_quat_out, self.d.mocap_quat)
            wp.copy(sensordata_out, self.d.sensordata) # sensor data still needs proper parsing downstream  
            wp.copy(worldid_out, self.d.contact.worldid)
        return mjwarp_step

    # Variant creation: for each object in obj_spec_lst, replace the skeleton's geom slots to build a variant mjm.
    #
    # Bugs fixed:
    #   1. geom_spec_dict + intermediate compile → invalidated spec pointers, corrupted state
    #   2. mutating parent_spec cumulatively → variant i built on top of variant i-1
    #   3. variant_geom_dataid size = len(parent_spec.geoms) ≠ mjm.ngeom → np.where mismatch
    #   4. skel_body.pos/quat not updated → body-local geom frame mismatch
    #
    # Fix: a completely fresh spec via skeleton_spec.copy() per variant; update body pos/quat, then overwrite the geoms.

    def find_body_by_name(self, spec, name):
        for b in spec.bodies:
            if b.name == name:
                return b
        return None

    def override_geoms_for_body(self, skel_body, src_body):
        skel_geoms = list(skel_body.geoms)
        src_geoms  = list(src_body.geoms)
        for i in range(len(skel_geoms)):
            sg = skel_geoms[i]
            if i < len(src_geoms):
                og = src_geoms[i]
                sg.type = og.type
                # SDF geoms also reference a mesh asset; without preserving meshname the
                # compiler raises "mesh geom '' (id = N) must have valid meshid".
                if og.type in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SDF):  # type: ignore
                    sg.meshname = og.meshname
                else:
                    sg.meshname = ""
                sg.pos         = og.pos
                sg.quat        = og.quat
                sg.size        = og.size
                sg.rgba        = og.rgba
                sg.contype     = og.contype
                sg.conaffinity = og.conaffinity
                sg.group       = og.group
                sg.friction    = og.friction
                sg.density     = og.density
                sg.solref      = og.solref
                sg.solimp      = og.solimp
            else:
                # slot present only in the skeleton → disable collision, keep the skeleton mesh reference (compilable)
                sg.contype     = 0
                sg.conaffinity = 0
                sg.group       = 4  # marker for dataid=-1 handling during the warp override


    def override_bodies_recursive(self, working_spec, skel_body_name, src_body, slot_suffix):
        skel_body = self.find_body_by_name(working_spec, skel_body_name)
        if skel_body is None:
            print_red(f"[override] skeleton body '{skel_body_name}' not found!")
            return
        # update the body frame: geom pos/quat are body-local, so a different frame shifts the positions
        skel_body.pos  = src_body.pos
        skel_body.quat = src_body.quat

        # ── INERTIAL-FRAME FIX (per-variant body inertia leak) ────────────
        # ``copy_body_recursive`` (skeleton build) copies ipos/iquat/mass/
        # inertia from the *anchor* obj_spec — these stay frozen on the
        # skeleton spec. If override_bodies_recursive only patches pos/quat,
        # every non-anchor variant_mjm inherits the **anchor's** inertial
        # properties even though its visual mesh, dimensions, and COM are
        # totally different. ``heterogeneous_env_setup`` then copies
        # ``variant_mjm.body_ipos`` → ``m.body_ipos[w]`` per world, so every
        # world's object body is simulated with the anchor's COM / mass /
        # principal inertia. Asymmetric meshes (wine bottle, can, tall mug)
        # pivot around the wrong point and tip over despite phantom
        # collisions being fixed.
        #
        # ``build_obj_spec_lst`` explicitly sets ``body.ipos``, ``body.mass``,
        # ``body.inertia`` (and leaves ``body.iquat`` at identity — the
        # inertia tensor is already principal-axis diagonal). Copying all
        # four fields here is the symmetric counterpart to
        # ``copy_body_recursive`` and restores per-variant inertial frames.
        skel_body.ipos    = src_body.ipos
        skel_body.iquat   = src_body.iquat
        skel_body.mass    = src_body.mass
        skel_body.inertia = src_body.inertia

        self.override_geoms_for_body(skel_body, src_body)
        for src_child in src_body.bodies:
            self.override_bodies_recursive(working_spec, src_child.name + slot_suffix, src_child, slot_suffix)
    
    def _sample_variant_slot_specs(self, anchor_spec_idx: int) -> list:
        """Pick which ``obj_spec_lst`` index goes into each of the variant's
        ``self.n_obj`` slots.

        - Slot 0 always anchors on ``anchor_spec_idx`` so for ``n_obj == 1``
          the legacy "variant v == obj_spec_lst[v]" identity is preserved.
        - Slots 1..n-1 are sampled uniformly from the remaining specs WITHOUT
          replacement when ``n_specs > n_obj`` (so a single variant shows up
          to ``n_obj`` distinct objects); WITH replacement otherwise.
        """
        n_specs = len(self.obj_spec_lst)
        slot_indices = [int(anchor_spec_idx)]
        if self.n_obj <= 1 or n_specs == 0:
            return slot_indices

        remaining = [i for i in range(n_specs) if i != anchor_spec_idx]
        random.shuffle(remaining)
        for _ in range(self.n_obj - 1):
            if remaining:
                slot_indices.append(remaining.pop())
            else:
                # Fewer distinct specs than slots → fall back to random with replacement.
                slot_indices.append(random.randint(0, n_specs - 1))
        return slot_indices

    def create_variants(self):
        """Build one compiled MjModel per variant.

        - ``n_obj == 1``: variant *v* contains ``obj_spec_lst[v]`` (legacy).
        - ``n_obj > 1`` : variant *v* contains ``obj_spec_lst[v]`` in slot 0
          plus ``n_obj − 1`` randomly sampled (preferably distinct) specs in
          the remaining slots — so every variant spawns a heterogeneous mix.

        ``self.variant_slot_specs[spec_idx]`` records the per-slot
        ``obj_spec_lst`` indices used for each variant; downstream code
        (``build_variant_pcd_cache``, ``warp_obj_pose_init``, ...) keys off
        the compiled per-variant MjModel directly so no further changes are
        required for them.
        """
        self.variant_xml_paths   = []
        self.variant_slot_specs  = {}   # spec_idx → list[obj_spec_lst index] of length n_obj
        for spec_idx, obj_spec in enumerate(self.obj_spec_lst):
            working_spec = self.skeleton_spec.copy()  # fresh skeleton (no cumulative drift)

            src_root_body = obj_spec.worldbody.first_body()
            if src_root_body is None:
                print_red(f"[spec_idx={spec_idx}] no root body!")
                continue

            # ``n_obj == 0`` (hand-only scene): the skeleton carries NO object
            # body, so every override below would miss its target and only log
            # "[override] skeleton body ... not found!". Skip the slot pass —
            # the variant is then the bare hand/floor skeleton. Variant COUNT is
            # unchanged (still one per obj_spec) so ``assignment`` /
            # ``heterogeneous_env_setup`` behave exactly as before.
            if int(self.n_obj) <= 0:
                slot_spec_indices = []
            else:
                slot_spec_indices = self._sample_variant_slot_specs(spec_idx)
            self.variant_slot_specs[spec_idx] = slot_spec_indices

            for slot_idx, slot_spec_idx in enumerate(slot_spec_indices):
                slot_obj_spec = self.obj_spec_lst[slot_spec_idx]
                slot_root_body = slot_obj_spec.worldbody.first_body()
                if slot_root_body is None:
                    continue
                slot_suffix = f"_{slot_idx}"
                # Skeleton body names follow the *base* spec used by
                # make_mjcf_from_spec — every obj_spec in this dataset
                # shares the same root body name (e.g. "top_watertight_tiny"),
                # so ``slot_root_body.name + slot_suffix`` matches the
                # skeleton body. Fall back to the registered renamed name
                # when available for extra safety.
                if (slot_idx < len(self.renamed_obj_names)
                        and self.renamed_obj_names[slot_idx]):
                    skel_root_name = self.renamed_obj_names[slot_idx]
                else:
                    skel_root_name = slot_root_body.name + slot_suffix
                self.override_bodies_recursive(
                    working_spec, skel_root_name, slot_root_body, slot_suffix)

            if self.verbose and self.n_obj > 1:
                print(f"[create_variants] spec_idx={spec_idx} slot_specs={slot_spec_indices}")

            variant_mjm = working_spec.compile()
            variant_mjm.opt.iterations      = self.overall_cfg.Opt.iterations
            variant_mjm.opt.ccd_iterations  = self.overall_cfg.Opt.ccd_iterations
            variant_mjm.opt.ccd_tolerance   = self.overall_cfg.Opt.ccd_tolerance
            variant_mjm.opt.ls_iterations   = self.overall_cfg.Opt.ls_iterations
            variant_mjm.opt.impratio        = self.overall_cfg.Opt.impratio
            variant_mjm.opt.gravity         = self.overall_cfg.Opt.gravity
            variant_mjm.opt.cone            = self.overall_cfg.Opt.cone
            variant_mjm.opt.solver          = self.overall_cfg.Opt.solver
            variant_mjm.opt.tolerance       = self.overall_cfg.Opt.tolerance
            variant_mjm.opt.timestep        = self.overall_cfg.sim_dt     

            variant_geom_dataid = variant_mjm.geom_group == 4  # FIX: mask based on the compiled ngeom

            # save as an XML file for make_parser_from_variant()
            variant_xml_path = f"{self.relative_xml_path}_variant_{spec_idx}.xml"
            with open(variant_xml_path, 'w') as _f:
                _f.write(working_spec.to_xml())
            self.variant_xml_paths.append(variant_xml_path)

            self.variants.append(variant_mjm)
            self.variants_geom_dataid_dict[spec_idx] = variant_geom_dataid

            if self.verbose:
                n_active   = (~variant_geom_dataid).sum()
                n_disabled = variant_geom_dataid.sum()
                print(f"[spec_idx={spec_idx}] variant built  active={n_active}  disabled={n_disabled}")

    # Body-local offset that exiles disabled "phantom" geom slots out of the
    # active scene every step. See ``heterogeneous_env_setup`` below for the
    # rationale — short version: mujoco_warp's pre-filtered collision pair
    # list (``nxn_geom_pair_filtered``) is built from skeleton-level
    # ``geom_contype`` / ``geom_conaffinity`` (1D, ``(ngeom,)``), which can't
    # be per-world. Disabled slots in non-anchor variants therefore survive
    # filtering and would generate spurious GJK contacts unless we move them
    # outside any plausible broadphase overlap. 100 m clears every fixture
    # in the dexterous-hand scene with room to spare while staying well
    # within float32's mantissa-precision band.
    _PHANTOM_GEOM_BANISH_M = 100.0

    # Disabled slots whose geom is a PRIMITIVE (box/capsule/cylinder/sphere —
    # procedural objects, see grit.util.primitive_object) must NOT be banished:
    # see ``heterogeneous_env_setup``. They are shrunk to this half-size at the
    # body origin instead, which makes them inert without moving them.
    _PHANTOM_PRIM_HALFSIZE = 1.0e-6

    def _union_object_geom_collision_onto_skeleton(self):
        """Collision-enable every object-body geom slot that ANY variant uses
        as a collision geom, on the SHARED skeleton model (``self.mjm``).

        Why: ``mujoco_warp.put_model`` pre-filters the n×n contact-pair list
        ONCE from ``self.mjm.geom_contype`` / ``geom_conaffinity`` (1D
        ``(ngeom,)`` — per-world override is impossible). The skeleton is
        copied from a single ``argmax(per_spec_ngeom)`` anchor, so each slot's
        collision flags reflect ONLY that anchor. When the anchor's
        ``bottom_watertight`` is a dummy / non-colliding region, the bottom
        slots get ``contype = 0`` and NO variant's bottom collides in the warp
        sim — even objects (wine glass) whose bottom carries real collision
        geoms. Because the anchor depends on the sampled object set, the
        breakage was ``N_SUB_ENV``-dependent.

        Fix: per-slot bitwise-OR the collision flags across all variant models
        and write the union onto the skeleton's OBJECT-body slots. Slots no
        variant uses for collision keep their (0) flags; unused / dummy slots
        are still banished far away in ``heterogeneous_env_setup`` (geom_pos =
        FAR via ``geom_dataid = -1``), so enabling their contype is harmless.
        Bitwise-OR (not force-to-1) preserves the exact contype/conaffinity
        bitmasks the variants actually use. Must run BEFORE ``put_model``.
        """
        if not getattr(self, "variants", None) or not self.renamed_obj_names:
            return
        ngeom       = int(self.mjm.ngeom)
        geom_bodyid = np.asarray(self.mjm.geom_bodyid)

        # OBJECT-subtree geom slots (top + bottom, full extent — never touch
        # hand / table / floor collision flags).
        obj_geom_ids = set()
        for body_name in self.renamed_obj_names:
            for bname in self._variant_subtree_body_names(self.mjm, body_name, exclude_prefix=()):
                bid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, bname)  # type: ignore
                if bid < 0:
                    continue
                obj_geom_ids.update(int(g) for g in np.where(geom_bodyid == bid)[0])
        if not obj_geom_ids:
            return

        # Per-slot union (bitwise OR) of collision flags across all variants.
        union_contype     = np.zeros(ngeom, dtype=np.int64)
        union_conaffinity = np.zeros(ngeom, dtype=np.int64)
        for vm in self.variants:
            vc = np.asarray(vm.geom_contype,     dtype=np.int64)
            va = np.asarray(vm.geom_conaffinity, dtype=np.int64)
            if vc.shape[0] != ngeom or va.shape[0] != ngeom:
                continue                                    # ngeom mismatch → skip (defensive)
            union_contype     |= vc
            union_conaffinity |= va

        n_fixed = 0
        for gid in obj_geom_ids:
            new_ct = int(union_contype[gid])
            new_ca = int(union_conaffinity[gid])
            if (new_ct != int(self.mjm.geom_contype[gid])
                    or new_ca != int(self.mjm.geom_conaffinity[gid])):
                n_fixed += 1
            self.mjm.geom_contype[gid]     = new_ct
            self.mjm.geom_conaffinity[gid] = new_ca

        if self.verbose:
            print(
                f"[union collision] object geom slots: {n_fixed} slot(s) "
                f"collision-enabled via cross-variant union (of {len(obj_geom_ids)} object slots)"
            )

    def heterogeneous_env_setup(self):
        self.assignment = [int(i % self.NWORLD % len(self.variants)) for i in range(self.NWORLD)]

        ngeom, nbody = self.mjm.ngeom, self.mjm.nbody

        dataid           = np.tile(self.mjm.geom_dataid, (self.NWORLD, 1))
        geom_size        = np.zeros((self.NWORLD, ngeom, 3))
        geom_rbound      = np.zeros((self.NWORLD, ngeom))
        geom_aabb        = np.zeros((self.NWORLD, ngeom, 2, 3))
        geom_pos         = np.zeros((self.NWORLD, ngeom, 3))
        geom_quat        = np.zeros((self.NWORLD, ngeom, 4))  # FIX: per-world override of geom orientation
        geom_group       = np.zeros((self.NWORLD, ngeom), dtype=int)
        body_mass        = np.zeros((self.NWORLD, nbody))
        body_subtreemass = np.zeros((self.NWORLD, nbody))
        body_inertia     = np.zeros((self.NWORLD, nbody, 3))
        body_invweight0  = np.zeros((self.NWORLD, nbody, 2))
        body_ipos        = np.zeros((self.NWORLD, nbody, 3))
        body_iquat       = np.zeros((self.NWORLD, nbody, 4))
        body_pos         = np.zeros((self.NWORLD, nbody, 3))  # FIX: per-world override of body rest position
        body_quat        = np.zeros((self.NWORLD, nbody, 4))  # FIX: per-world override of body rest orientation

        # ── PHANTOM-COLLISION FIX ─────────────────────────────────────────────
        # Why we banish disabled slots' geom_pos:
        #
        #   - ``self.m = mjwarp.put_model(self.mjm)`` builds the warp Model
        #     from the *skeleton* MjModel. Object-body geom slots are
        #     collision-enabled there by ``_union_object_geom_collision_onto_skeleton``
        #     (per-slot bitwise-OR of every variant's contype/conaffinity),
        #     so any slot some variant collides on participates in the pair
        #     list — NOT just the anchor's collision geoms.
        #
        #   - mujoco_warp pre-filters the n×n geom-pair list ONCE at model
        #     build time, using these skeleton-level contype/conaffinity
        #     values (see ``mujoco_warp/_src/io.py``). Per-world override of
        #     ``contype`` / ``conaffinity`` is impossible — both arrays are
        #     1D ``(ngeom,)`` on the Model.
        #
        #   - In non-anchor variants, slots that this variant doesn't use
        #     get ``contype = 0, conaffinity = 0, group = 4`` set on the
        #     variant's spec (``override_geoms_for_body``) — but their
        #     ``geom_size``, ``geom_pos``, ``geom_aabb``, ``geom_rbound``,
        #     ``meshname`` all stay at the skeleton's anchor-mesh values
        #     because the spec override never touches those fields.
        #
        #   - Setting ``geom_dataid = -1`` on disabled slots short-circuits
        #     the per-pair mesh-data lookup in ``collision_core.py`` (vertadr
        #     becomes -1) but does NOT prune the pair from the broadphase
        #     pre-filtered list. Narrow-phase GJK then runs against garbage
        #     mesh data → spurious contacts.
        #
        #   - We can't safely zero ``rbound``: the broadphase filter routes
        #     ``rbound == 0`` into ``_plane_filter``, which treats a non-plane
        #     mesh as a plane and effectively passes everything through. Same
        #     story for ``geom_aabb`` (zero-aabb at xpos near the scene is
        #     still inside other AABBs).
        #
        #   - Cleanest fix that respects every mjwarp invariant: move the
        #     disabled slot's body-local ``geom_pos`` far away (and identity
        #     quat). World position is then ``body_xpos + R · (FAR,FAR,FAR)``
        #     ≈ ±FAR · √3, so SPHERE filter (distance >> rbound sum) and
        #     AABB filter (no overlap) both cull the pair every step. mesh
        #     data lookup remains short-circuited via dataid=-1. Disabled
        #     slots are still group=4 so the viewer renders them invisible.
        #
        # CPU ``make_parser_from_variant`` is unaffected: it compiles the
        # variant XML directly, where disabled slots already have
        # ``contype = conaffinity = 0`` baked in — MuJoCo prunes them at
        # compile time, so the CPU single-env path is robust by construction.

        FAR  = float(self._PHANTOM_GEOM_BANISH_M)
        TINY = float(self._PHANTOM_PRIM_HALFSIZE)
        IDQT = np.array([1.0, 0.0, 0.0, 0.0])
        n_phantom_per_variant = {}

        # ── PRIMITIVE phantoms must be SHRUNK, not banished ──────────────────
        # Banishing works for MESH slots because ``geom_dataid = -1`` (set just
        # below) short-circuits the mesh-data lookup, so the narrowphase never
        # actually collides them. A primitive geom ignores ``geom_dataid``: its
        # shape comes from ``geom_size``, so a banished primitive still
        # collides — and the banish offset is applied in the BODY frame, so as
        # soon as the object rotates far enough, ``body_xpos + R·(FAR,FAR,FAR)``
        # lands the phantom BELOW the floor plane. Plane collision then reports
        # a metre-deep penetration → the solver explodes → NaN qpos in that
        # world (observed: phantom at z = -0.56 m, contact dist = -0.557, NaN
        # ~35 steps later). Shrinking to a ~1 µm geom at the body origin is
        # inert instead: anything that could touch it is already deep inside
        # the object's real geoms, and same-body pairs never collide.
        _prim_slot = ~np.isin(
            np.asarray(self.mjm.geom_type),
            [int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_SDF)],  # type: ignore
        )

        for w, var_idx in enumerate(self.assignment):
            ref                 = self.variants[var_idx]
            variant_geom_dataid = self.variants_geom_dataid_dict[var_idx]  # True = disabled slot
            dataid[w]           = ref.geom_dataid
            dataid[w]           = np.where(variant_geom_dataid, -1, dataid[w])
            geom_size[w]        = ref.geom_size
            geom_rbound[w]      = ref.geom_rbound          # NB: do NOT zero — would trigger PLANE branch
            geom_aabb[w]        = ref.geom_aabb.reshape(ngeom, 2, 3)
            geom_pos[w]         = ref.geom_pos
            geom_quat[w]        = ref.geom_quat

            # Banish MESH phantoms: world pos = body_xpos + R·(FAR,FAR,FAR),
            # well outside any plausible scene → broadphase culls every pair.
            # Shrink PRIMITIVE phantoms in place instead (see the note above).
            if variant_geom_dataid.any():
                banish = variant_geom_dataid & ~_prim_slot
                shrink = variant_geom_dataid & _prim_slot
                if banish.any():
                    geom_pos[w][banish]  = FAR
                    geom_quat[w][banish] = IDQT
                if shrink.any():
                    geom_pos[w][shrink]    = 0.0
                    geom_quat[w][shrink]   = IDQT
                    geom_size[w][shrink]   = TINY
                    geom_rbound[w][shrink] = 2.0 * TINY   # NB: never 0 (plane filter)
                    geom_aabb[w][shrink]   = TINY
                    geom_aabb[w][shrink, 0, :] = 0.0      # aabb = (centre, half-extent)
                n_phantom_per_variant.setdefault(var_idx, int(variant_geom_dataid.sum()))

            geom_group[w]       = ref.geom_group           # 2D override is read by mjwarp_viewer
            body_mass[w]        = ref.body_mass
            body_subtreemass[w] = ref.body_subtreemass
            body_inertia[w]     = ref.body_inertia
            body_invweight0[w]  = ref.body_invweight0
            body_ipos[w]        = ref.body_ipos
            body_iquat[w]       = ref.body_iquat
            body_pos[w]         = ref.body_pos
            body_quat[w]        = ref.body_quat

        # ── per-world array override on the warp model ─────────────────────────
        self.m.geom_dataid     = wp.array(dataid,           dtype=int)
        self.m.geom_size       = wp.array(geom_size,        dtype=wp.vec3)
        self.m.geom_rbound     = wp.array(geom_rbound,      dtype=float)
        self.m.geom_aabb       = wp.array(geom_aabb,        dtype=wp.vec3)
        self.m.geom_pos        = wp.array(geom_pos,         dtype=wp.vec3)
        self.m.geom_quat       = wp.array(geom_quat,        dtype=wp.quat)
        self.m.geom_group      = wp.array(geom_group,       dtype=int)
        self.m.body_mass       = wp.array(body_mass,        dtype=float)
        self.m.body_subtreemass= wp.array(body_subtreemass, dtype=float)
        self.m.body_inertia    = wp.array(body_inertia,     dtype=wp.vec3)
        self.m.body_invweight0 = wp.array(body_invweight0,  dtype=wp.vec2)
        self.m.body_ipos       = wp.array(body_ipos,        dtype=wp.vec3)
        self.m.body_iquat      = wp.array(body_iquat,       dtype=wp.quat)
        self.m.body_pos        = wp.array(body_pos,         dtype=wp.vec3)
        self.m.body_quat       = wp.array(body_quat,        dtype=wp.quat)

        if self.verbose:
            print("assignment : ", self.assignment)
            print(f"nworld={self.NWORLD}  ngeom={ngeom}  nbody={nbody}")
            print(f"geom_dataid shape : {self.m.geom_dataid.numpy().shape}")
            if n_phantom_per_variant:
                total_phantoms = sum(n_phantom_per_variant.values())
                print(
                    f"[phantom-banish] banished {total_phantoms} disabled-slot "
                    f"geoms across {len(n_phantom_per_variant)} variant(s) "
                    f"to body-local ±{FAR:g} m (per-variant counts: "
                    f"{dict(sorted(n_phantom_per_variant.items()))})"
                )

    # ──────────────────────────────────────────────────────────────────────
    # Variant PCD cache
    # ──────────────────────────────────────────────────────────────────────

    def _variant_subtree_body_names(self, mjm, root_body_name, exclude_prefix=('bottom_',)):
        """Return all body names in the subtree rooted at root_body_name (root included)."""
        n_body = mjm.nbody
        body_names = [
            mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, i)  # type: ignore
            for i in range(n_body)
        ]
        children = [[] for _ in range(n_body)]
        for cid in range(1, n_body):
            children[int(mjm.body_parentid[cid])].append(cid)

        root_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, root_body_name)  # type: ignore
        if root_id == -1:
            return []

        result = [root_body_name]
        stack = [root_id]
        while stack:
            pid = stack.pop()
            for cid in children[pid]:
                cname = body_names[cid]
                if cname is None:
                    continue
                if exclude_prefix and any(cname.startswith(p) for p in exclude_prefix):
                    continue
                result.append(cname)
                stack.append(cid)
        return result

    def _build_variant_pcd(self, mjm, mjd, obj_body_name, n_sample=1024,
                           exclude_prefix=('bottom_',), DENSE_SAMPLE_FACTOR=5):
        """
        Build body-local canonical PCD for obj_body_name from a variant MjModel/MjData.
        Mirrors HandRLParserClass._build_body_local_pcd but operates on an external (mjm, mjd).

        Returns:
            np.ndarray [n_sample, 3] in body-local frame, or None if no mesh found.
        """
        subtree = self._variant_subtree_body_names(mjm, obj_body_name, exclude_prefix)

        # Collect geom indices belonging to the subtree
        geom_idxs = []
        for bname in subtree:
            bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, bname)  # type: ignore
            if bid == -1:
                continue
            for gid in range(mjm.ngeom):
                if mjm.geom_bodyid[gid] == bid:
                    geom_idxs.append(gid)
        geom_idxs = list(set(geom_idxs))

        body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, obj_body_name)  # type: ignore
        p_body  = mjd.xpos[body_id].copy()
        R_body  = mjd.xmat[body_id].reshape(3, 3).copy()

        vertices_tf_list = []
        # Primitive-collider samples are kept SEPARATE and used only when the
        # object has no mesh geom at all (procedurally generated objects — see
        # grit.util.primitive_object). Mesh-based objects therefore keep their
        # exact previous cloud even when their XML also carries a primitive
        # collider (some flat objects use a box collider alongside the mesh).
        prim_tf_list = []
        for geom_idx in geom_idxs:
            # group=4 marks skeleton placeholder slots: when the skeleton has more geoms
            # than the source object, the leftover slots get group=4, so this geom's mesh
            # belongs to the skeleton (a different object). Including it in the PCD would
            # mix in vertices unrelated to the actual object shape and cause misalignment.
            if int(mjm.geom_group[geom_idx]) == 4:
                continue
            n_sample_geom = max(1, int(np.ceil(
                n_sample * DENSE_SAMPLE_FACTOR / max(1, len(geom_idxs))
            )))

            # ── primitive colliders (box / capsule / cylinder / sphere / …) ──
            # No mesh asset to sample → analytic surface sampling, so
            # ``variant_pcd_cache`` / ``variant_obj_min`` work for primitive
            # objects exactly as they do for meshes.
            if int(mjm.geom_type[geom_idx]) not in (
                mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SDF):  # type: ignore
                from grit.util.primitive_object import sample_geom_surface
                pts = sample_geom_surface(
                    int(mjm.geom_type[geom_idx]), mjm.geom_size[geom_idx], n_sample_geom)
                if pts is None:
                    continue
                p_geom  = mjd.geom_xpos[geom_idx].copy()
                R_geom  = mjd.geom_xmat[geom_idx].reshape(3, 3).copy()
                R_off   = R_body.T @ R_geom
                p_off   = R_body.T @ (p_geom - p_body)
                prim_tf_list.append((R_off @ pts.T).T + p_off)
                continue

            # SDF-mode objects carry an ``mjGEOM_SDF`` collider that still points
            # at a real mesh asset (``geom_dataid``), so it must be included when
            # building the object PCD / min-z — otherwise ``variant_obj_min`` is
            # empty and the spawn kernel drops the object into the floor.
            mesh_idx = int(mjm.geom_dataid[geom_idx])
            if mesh_idx < 0:
                continue

            vert_adr = int(mjm.mesh_vertadr[mesh_idx])
            n_vert   = int(mjm.mesh_vertnum[mesh_idx])
            if n_vert <= 0:
                continue
            vertices = mjm.mesh_vert[vert_adr:vert_adr + n_vert].reshape(-1, 3)

            face_adr = int(mjm.mesh_faceadr[mesh_idx])
            n_face   = int(mjm.mesh_facenum[mesh_idx])
            if n_face <= 0:
                continue
            faces = mjm.mesh_face[face_adr:face_adr + n_face].reshape(-1, 3)

            v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
            valid = area > 1e-12
            if not np.any(valid):
                continue
            v0, v1, v2, area = v0[valid], v1[valid], v2[valid], area[valid]

            prob      = area / area.sum()
            fidxs     = np.random.choice(len(area), size=n_sample_geom, p=prob)
            sv0, sv1, sv2 = v0[fidxs], v1[fidxs], v2[fidxs]

            u       = np.random.rand(n_sample_geom, 1)
            v_r     = np.random.rand(n_sample_geom, 1)
            sqrt_u  = np.sqrt(u)
            pts     = (1 - sqrt_u) * sv0 + sqrt_u * (1 - v_r) * sv1 + sqrt_u * v_r * sv2

            p_geom  = mjd.geom_xpos[geom_idx].copy()
            R_geom  = mjd.geom_xmat[geom_idx].reshape(3, 3).copy()
            R_off   = R_body.T @ R_geom
            p_off   = R_body.T @ (p_geom - p_body)
            vertices_tf_list.append((R_off @ pts.T).T + p_off)

        if not vertices_tf_list:
            vertices_tf_list = prim_tf_list      # primitive-only object
        if not vertices_tf_list:
            return None

        pcd = np.concatenate(vertices_tf_list, axis=0)
        if len(pcd) >= n_sample:
            pcd = pcd[farthest_point_sampling(pcd, n_sample)]
        else:
            pcd = pcd[np.random.choice(len(pcd), n_sample, replace=True)]
        return pcd

    def warp_obj_pose_init(
        self,
        table_height:    float = 0.0,
        xy_offset_range: list  = [-0.2, 0.2],
        z_clearance:     float = 0.001,
    ) -> None:
        """
        Initialize object poses across all parallel warp worlds.
        Equivalent to hand_utils.obj_pose_init() for MuJoCo Warp.

        - Reads variant_obj_min for each world's variant → correct table clearance.
        - Applies per-world random XY offset within xy_offset_range.
        - Zeros all velocities.
        - Pushes modified qpos/qvel back to GPU.

        Call after mjwarp.reset_data() and before ScopedCapture / first step.
        """
        mjm = self.mjm

        # ── qpos address per object (skeleton is shared → same addr for every world) ──
        qpos_addrs: dict = {}
        for body_name in self.renamed_obj_names:
            body   = mjm.body(body_name)
            jntadr = int(body.jntadr[0])
            if jntadr == -1:
                body   = mjm.body("body_obj_" + body_name)
                jntadr = int(body.jntadr[0])
            qpos_addrs[body_name] = int(mjm.jnt_qposadr[jntadr])

        # ── GPU → CPU ──
        qpos_np = self.d.qpos.numpy()   # (nworld, nq)
        qvel_np = self.d.qvel.numpy()   # (nworld, nv)

        for w in range(self.NWORLD):
            variant_idx = int(self.assignment[w])
            xy_offsets  = np.random.uniform(
                xy_offset_range[0], xy_offset_range[1],
                (len(self.renamed_obj_names), 2),
            )

            for obj_idx, body_name in enumerate(self.renamed_obj_names):
                qpa = qpos_addrs[body_name]

                obj_pos    = np.zeros(3)
                obj_pos[:2] = xy_offsets[obj_idx]
                obj_pos[2]  = table_height

                # variant_obj_min[v][body] = -np.min(pcd, axis=0)
                # → rel_min[2] = -min_z_of_pcd  (positive for objects below origin)
                if hasattr(self, 'variant_obj_min'):
                    rel_min = self.variant_obj_min.get(variant_idx, {}).get(body_name)
                    if rel_min is not None:
                        obj_pos[2] += float(rel_min[2]) + z_clearance

                qpos_np[w, qpa    : qpa + 3] = obj_pos
                qpos_np[w, qpa + 3: qpa + 7] = [1.0, 0.0, 0.0, 0.0]

        qvel_np[:] = 0.0

        # ── CPU → GPU  (in-place: preserve the buffer that ScopedCapture baked in) ──
        # Using d.qpos = wp.array(...) would create a NEW buffer, breaking the captured
        # CUDA graph which still holds the OLD pointer. wp.copy writes into the existing
        # buffer so the graph continues to see the updated values.
        device = self.d.qpos.device
        dtype  = self.d.qpos.dtype
        wp.copy(self.d.qpos, wp.array(qpos_np, dtype=dtype, device=device))
        wp.copy(self.d.qvel, wp.array(qvel_np, dtype=dtype, device=device))

    def default_wrist_pose(self, hand_util=None):
        """Model-reference (``qpos0``) wrist free-joint pose.

        Returns ``(pos(3,), quat(4,) wxyz)`` as float32. This is the hand's
        "default" wrist pose — used by finger-only (``control_wrist=False``)
        setups that HOLD the wrist at a fixed pose while only the fingers move.
        """
        hand_util = hand_util if hand_util is not None else self.hand_util
        mjm  = self.mjm
        qpa  = int(mjm.jnt_qposadr[int(mjm.body(hand_util.rh_wrist_base_name).jntadr[0])])
        q0   = np.asarray(mjm.qpos0, dtype=np.float32)
        return q0[qpa:qpa + 3].copy(), q0[qpa + 3:qpa + 7].copy()

    def fix_wrist_pose(self, pos=None, quat=None, hand_util=None) -> None:
        """Pin the wrist (free-joint qpos + mocap) to a FIXED pose across ALL
        parallel worlds.

        For finger-only (``control_wrist=False``) setups: after the usual reset
        (``warp_pose_init_capture_launch`` etc.) call this to overwrite the
        randomized wrist with a fixed pose so the wrist is held there while the
        policy / sim only moves the fingers (the mocap weld keeps it put). Both
        the free-joint qpos and the mocap target are written so the weld is
        satisfied and the wrist does not drift. ``pos`` / ``quat`` default to
        :meth:`default_wrist_pose`.

        numpy round-trip — reset-time use, NOT the per-step hot path.
        """
        hand_util = hand_util if hand_util is not None else self.hand_util
        dpos, dquat = self.default_wrist_pose(hand_util)
        pos  = dpos  if pos  is None else np.asarray(pos,  dtype=np.float32).reshape(3)
        quat = dquat if quat is None else np.asarray(quat, dtype=np.float32).reshape(4)

        mjm = self.mjm
        d   = self.d
        qpa       = int(mjm.jnt_qposadr[int(mjm.body(hand_util.rh_wrist_base_name).jntadr[0])])
        mocap_bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, hand_util.rh_mocap_name)  # type: ignore
        mocap_id  = int(mjm.body_mocapid[mocap_bid])

        q = d.qpos.numpy()
        q[:, qpa:qpa + 3]     = pos
        q[:, qpa + 3:qpa + 7] = quat
        wp.copy(d.qpos, wp.array(q, dtype=float, device=d.qpos.device))
        mp = d.mocap_pos.numpy();  mp[:, mocap_id] = pos
        mq = d.mocap_quat.numpy(); mq[:, mocap_id] = quat
        wp.copy(d.mocap_pos,  wp.array(mp, dtype=float, device=d.mocap_pos.device))
        wp.copy(d.mocap_quat, wp.array(mq, dtype=float, device=d.mocap_quat.device))

    def fix_wrist_pose_gpu(self, hand_util=None) -> None:
        """GPU sync-free version of :meth:`fix_wrist_pose` — for the per-step hot path.

        Pins every world's wrist free-joint qpos + mocap to the DEFAULT pose
        (:meth:`default_wrist_pose` = model ``qpos0``). Handled by a single warp
        kernel launch, so calling it inside a sync-free path such as
        ``per_world_reset_if_done`` incurs no CPU↔GPU round trip. Addresses and
        pose constants are cached on the first call.
        """
        hand_util = hand_util if hand_util is not None else self.hand_util
        if not hasattr(self, "_fix_wrist_gpu_const"):
            mjm  = self.mjm
            qpa  = int(mjm.jnt_qposadr[int(mjm.body(hand_util.rh_wrist_base_name).jntadr[0])])
            mocap_bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, hand_util.rh_mocap_name)  # type: ignore
            mocap_id  = int(mjm.body_mocapid[mocap_bid])
            pos, quat = self.default_wrist_pose(hand_util)
            self._fix_wrist_gpu_const = (
                qpa, mocap_id,
                wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            )
        qpa, mocap_id, pos_wp, quat_wp = self._fix_wrist_gpu_const
        wp.launch(
            _grit_fix_wrist_pose_kernel, dim=self.NWORLD,
            inputs=[
                int(qpa), int(mocap_id), pos_wp, quat_wp,
                self.d.qpos, self.d.mocap_pos, self.d.mocap_quat,
            ],
        )

    def warp_wrist_pose_init(
        self,
        hand_util,
        grasping_obj_name: str | None = None,
        table_height: float = 0.0,
        palm_facing_y_up_prob: float = 0.2,
        y_up_z_range = (0.10, 0.25),
        y_up_margin_range = (0.03, 0.10),
    ) -> None:
        """
        Initialize wrist (free joint) pose and mocap target across all parallel warp worlds.
        Equivalent to hand_utils.wrist_pose_init() for MuJoCo Warp.

        Per world:
          1. Pull the grasping object's body-local PCD from variant_pcd_cache
             and transform it to world frame via d.xpos/d.xmat for that world.
          2. Sample a random wrist init via hand_utils.get_wrist_init_pos
             (hand-center +X aligned to point exactly at the object, roll about
             the X axis uniform in (-π, π)).
          3. Compose wrist transform: T_wrist = T_handcenter @ inv(rh_hand_center).
             Apply ±30° per-axis rpy noise (matches CPU wrist_pose_init) so
             the wrist orientation varies across resets.
          4. Write d.qpos[w, qpa:qpa+7]   (free joint: pos + quat wxyz)
                 d.mocap_pos[w, mocap_id]
                 d.mocap_quat[w, mocap_id] (wxyz)

        Call ORDER:
            mjwarp.reset_data(m, d)
            self.warp_obj_pose_init(...)
            mjwarp.forward(m, d)            # ← required so d.xpos reflects new obj poses
            self.warp_wrist_pose_init(hand_util)
            mjwarp.forward(m, d)            # ← refresh kinematics after wrist write
            wp.ScopedCapture(...)           # only after all init writes are done

        Args:
            hand_util:         HandUtils instance (provides rh_hand_center,
                               rh_wrist_base_name, rh_mocap_name).
            grasping_obj_name: Object body to sample around. Defaults to the first
                               body in self.renamed_obj_names.
        """
        from grit.util.sim_core.transforms import pr2t, t2pr, r2quat, rpy2r

        if grasping_obj_name is None:
            grasping_obj_name = self.renamed_obj_names[0]

        if not getattr(self, 'variant_pcd_cache', None):
            raise RuntimeError(
                "warp_wrist_pose_init requires variant_pcd_cache. "
                "Build it via build_variant_pcd_cache() during reset()."
            )

        mjm = self.mjm

        # ── wrist free-joint qpos addr (skeleton-shared → same addr across worlds) ──
        wrist_body_name = hand_util.rh_wrist_base_name
        wrist_jntadr    = int(mjm.body(wrist_body_name).jntadr[0])
        wrist_qpa       = int(mjm.jnt_qposadr[wrist_jntadr])

        # ── mocap idx for the hand mocap body ──
        mocap_body_id = mujoco.mj_name2id(  # type: ignore
            mjm, mujoco.mjtObj.mjOBJ_BODY, hand_util.rh_mocap_name  # type: ignore
        )
        if mocap_body_id == -1:
            raise ValueError(f"mocap body '{hand_util.rh_mocap_name}' not found.")
        mocap_id = int(mjm.body_mocapid[mocap_body_id])
        if mocap_id == -1:
            raise ValueError(
                f"body '{hand_util.rh_mocap_name}' is not a mocap body (body_mocapid=-1)."
            )

        # ── grasping object body id (skeleton model) ──
        obj_body_id = mujoco.mj_name2id(  # type: ignore
            mjm, mujoco.mjtObj.mjOBJ_BODY, grasping_obj_name  # type: ignore
        )
        if obj_body_id == -1:
            raise ValueError(f"object body '{grasping_obj_name}' not found.")

        # ── inverse of (wrist→hand_center) once ──
        p_w2c, R_w2c = t2pr(hand_util.rh_hand_center)
        R_inv = R_w2c.T
        p_inv = -R_inv @ p_w2c
        T_inv = pr2t(p_inv, R_inv)

        # ── GPU → CPU snapshots ──
        qpos_np      = self.d.qpos.numpy()         # (nworld, nq)
        mocap_pos_np = self.d.mocap_pos.numpy()    # (nworld, nmocap, 3)
        mocap_quat_np = self.d.mocap_quat.numpy()  # (nworld, nmocap, 4) wxyz
        xpos_np      = self.d.xpos.numpy()         # (nworld, nbody, 3)
        xmat_np      = self.d.xmat.numpy()         # (nworld, nbody, 3, 3) or (.., 9)

        for w in range(self.NWORLD):
            variant_idx = int(self.assignment[w])
            pcd_local = self.variant_pcd_cache.get(variant_idx, {}).get(grasping_obj_name)
            if pcd_local is None or len(pcd_local) == 0:
                continue

            # body-local PCD → world frame for THIS world
            p_body = np.asarray(xpos_np[w, obj_body_id]).ravel()[:3]
            R_body = np.asarray(xmat_np[w, obj_body_id]).reshape(3, 3)
            pcd_world = (R_body @ pcd_local.T).T + p_body

            if np.random.rand() < palm_facing_y_up_prob:
                # === Special: wrist Y == world up + palm-facing-object ===
                sample_idx   = np.random.randint(0, pcd_world.shape[0])
                target_world = pcd_world[sample_idx]

                obj_xy = pcd_world.mean(axis=0)[:2]
                # object XY radius (max distance from the centroid) + margin
                obj_xy_radius = float(
                    np.linalg.norm(pcd_world[:, :2] - obj_xy, axis=1).max()
                )
                azimuth = np.random.uniform(-np.pi, np.pi)
                margin  = np.random.uniform(y_up_margin_range[0], y_up_margin_range[1])
                horiz_d = obj_xy_radius + margin
                wrist_p_init = np.array([
                    obj_xy[0] - horiz_d * np.cos(azimuth),
                    obj_xy[1] - horiz_d * np.sin(azimuth),
                    float(table_height) + np.random.uniform(y_up_z_range[0], y_up_z_range[1]),
                ])
                wrist_R_init = hand_utils._make_wrist_pose_y_up_palm_facing(
                    target_world, wrist_p_init, hand_util,
                )
                # no noise — preserves wrist Y == world up
            else:
                # === Default: hand-center faces the object + ±30° rpy noise ===
                p_init, _, _, R_init = hand_utils.get_wrist_init_pos(
                    pcd_world,
                    xyz_range=np.array([[-0.4, 0.4], [-0.4, 0.4], [0.1, 0.5]]),
                )
                hand_center_T          = pr2t(p_init, R_init)
                wrist_T                = hand_center_T @ T_inv
                wrist_p_init, wrist_R_init = t2pr(wrist_T)

                noise_scale = (np.pi / 180.0) * 30.0
                wrist_R_init = wrist_R_init @ rpy2r(np.array([
                    np.random.uniform(-noise_scale, noise_scale),
                    np.random.uniform(-noise_scale, noise_scale),
                    np.random.uniform(-noise_scale, noise_scale),
                ]))

            wrist_R_quat = r2quat(wrist_R_init)  # wxyz
            qpos_np[w, wrist_qpa    : wrist_qpa + 3] = wrist_p_init
            qpos_np[w, wrist_qpa + 3: wrist_qpa + 7] = wrist_R_quat
            mocap_pos_np[w, mocap_id]  = wrist_p_init
            mocap_quat_np[w, mocap_id] = wrist_R_quat

        # ── CPU → GPU (in-place, preserves CUDA graph buffer pointers) ──
        device = self.d.qpos.device
        wp.copy(self.d.qpos,
                wp.array(qpos_np, dtype=self.d.qpos.dtype, device=device))
        wp.copy(self.d.mocap_pos,
                wp.array(mocap_pos_np, dtype=self.d.mocap_pos.dtype, device=device))
        wp.copy(self.d.mocap_quat,
                wp.array(mocap_quat_np, dtype=self.d.mocap_quat.dtype, device=device))

    # ────────────────────────────────────────────────────────────────────
    # GPU-resident pose-init path (parallel to warp_obj/wrist_pose_init).
    # Built around the module-level _grit_*_kernel functions defined above.
    # ────────────────────────────────────────────────────────────────────

    def setup_warp_pose_init_kernels(
        self,
        hand_util,
        grasping_obj_name: str | None = None,
    ) -> None:
        """
        One-time prep for the GPU-kernel pose-init path. Uploads PCD, centroids,
        per-variant obj-min, qpos addresses, and precomputed inv(rh_hand_center)
        as warp arrays / scalars. Allocates the seed buffer used by both kernels.

        Call ONCE after build_sub_env() (variant_pcd_cache must already exist).
        Re-call if hand_util / variants change.

        Stores on self:
            _gpu_obj_qpos_addrs, _gpu_assignment, _gpu_var_obj_min_z,
            _gpu_pcd_local, _gpu_centroid_local,
            _gpu_R_inv, _gpu_p_inv,
            _gpu_grasp_obj_idx, _gpu_obj_body_id,
            _gpu_wrist_qpa, _gpu_mocap_id,
            _gpu_n_pts, _n_obj, _gpu_init_seed_arr, _gpu_init_seed_value,
            _gpu_init_capture (None until capture_warp_pose_init_kernels is called)
        """
        from grit.util.sim_core.transforms import t2pr

        # ── n_obj = 0 fallback ─────────────────────────────────────────────
        # When there are no objects in the scene (hand-pose-only training),
        # the kernels still need *valid-shape* GPU arrays to reference even
        # though the object init loop is a no-op. We synthesize a single
        # phantom slot anchored at the origin with identity orientation.
        # Default object pose: (pos=0, quat=identity wxyz) = (0,0,0,1,0,0,0).
        no_object = (len(self.renamed_obj_names) == 0)

        mjm    = self.mjm
        device = self.d.qpos.device

        if no_object:
            n_variants  = max(1, len(self.variants))
            n_pts_dummy = 1
            self._gpu_obj_qpos_addrs = wp.array(
                np.zeros((1,), dtype=np.int32), dtype=int, device=device,
            )                                                 # phantom — never read (n_obj=0 loop is no-op)
            self._n_obj = 0                                   # obj init kernel still skips
            self._gpu_var_obj_min_z = wp.array(
                np.zeros((n_variants, 1), dtype=np.float32), dtype=float, device=device,
            )
            pcd_arr            = np.zeros((n_variants, 1, n_pts_dummy, 3), dtype=np.float32)
            centroid_arr       = np.zeros((n_variants, 1, 3),              dtype=np.float32)
            obj_xy_radius_arr  = np.zeros((n_variants, 1),                 dtype=np.float32)
            self._gpu_pcd_local      = wp.array(pcd_arr,           dtype=wp.vec3, device=device)
            self._gpu_centroid_local = wp.array(centroid_arr,      dtype=wp.vec3, device=device)
            self._gpu_obj_xy_radius  = wp.array(obj_xy_radius_arr, dtype=float,   device=device)
            self._gpu_n_pts          = n_pts_dummy

            self._gpu_assignment = wp.array(
                np.asarray(self.assignment, dtype=np.int32), dtype=int, device=device,
            )

            # Phantom grasping target: world body (id 0, xpos always (0,0,0))
            # so the wrist init kernel still has something to face.
            self._gpu_grasp_obj_name = None
            self._gpu_grasp_obj_idx  = 0
            self._gpu_obj_body_id    = 0
        else:
            if not getattr(self, 'variant_pcd_cache', None):
                raise RuntimeError(
                    "GPU init kernels require variant_pcd_cache. "
                    "Build it via build_variant_pcd_cache() during reset()."
                )

            if grasping_obj_name is None:
                grasping_obj_name = self.renamed_obj_names[0]

            # ── object qpos addresses (skeleton-shared across worlds) ──
            addr_list = []
            for body_name in self.renamed_obj_names:
                body   = mjm.body(body_name)
                jntadr = int(body.jntadr[0])
                if jntadr == -1:
                    body   = mjm.body("body_obj_" + body_name)
                    jntadr = int(body.jntadr[0])
                addr_list.append(int(mjm.jnt_qposadr[jntadr]))
            self._gpu_obj_qpos_addrs = wp.array(
                np.asarray(addr_list, dtype=np.int32), dtype=int, device=device,
            )
            self._n_obj = len(self.renamed_obj_names)

            # ── per-variant per-obj min z (for table clearance) ──
            n_variants = len(self.variants)
            var_min_z  = np.zeros((n_variants, self._n_obj), dtype=np.float32)
            for v in range(n_variants):
                for o, body_name in enumerate(self.renamed_obj_names):
                    rel_min = self.variant_obj_min.get(v, {}).get(body_name)
                    if rel_min is not None:
                        var_min_z[v, o] = float(rel_min[2])
            self._gpu_var_obj_min_z = wp.array(var_min_z, dtype=float, device=device)

            # ── PCD + centroid: shape (n_var, n_obj, n_pts, 3) → wp.vec3 ndim=3 ──
            #
            # Wrist target sampling uses the **graspable** PCD
            # (``variant_pcd_cache``, top_watertight only) so the wrist
            # never aims at a non-graspable area like a wine-glass base.
            #
            # obj_xy_radius / centroid drive the y_up wrist init mode
            # (palm-faces-object). These must reflect the **full** body
            # footprint (``variant_full_pcd_cache``) — otherwise the wrist
            # would be placed inside the wider base of a wine glass /
            # mug since the radius computed from top-only PCD is too
            # small. ``variant_full_pcd_cache`` collapses to the graspable
            # cache when no real bottom exists, so this is a no-op for
            # bottle-less / dummy-bottom objects.
            full_cache       = getattr(self, "variant_full_pcd_cache", self.variant_pcd_cache)
            any_variant_pcd  = next(iter(next(iter(self.variant_pcd_cache.values())).values()))
            n_pts            = int(any_variant_pcd.shape[0])
            pcd_arr          = np.zeros((n_variants, self._n_obj, n_pts, 3), dtype=np.float32)
            centroid_arr     = np.zeros((n_variants, self._n_obj, 3),        dtype=np.float32)
            obj_xy_radius_arr= np.zeros((n_variants, self._n_obj),           dtype=np.float32)
            for v in range(n_variants):
                for o, body_name in enumerate(self.renamed_obj_names):
                    pcd_grasp = self.variant_pcd_cache.get(v, {}).get(body_name)
                    pcd_full  = full_cache.get(v, {}).get(body_name)
                    if pcd_grasp is None or len(pcd_grasp) == 0:
                        continue
                    # GPU pcd_local stays graspable — kernel samples wrist
                    # target points from this buffer.
                    pcd_arr[v, o] = pcd_grasp
                    if pcd_full is None or len(pcd_full) == 0:
                        pcd_full = pcd_grasp
                    # Centroid + XY radius from the full-extent PCD so the
                    # y_up "wrist outside object footprint" init clears the
                    # bottom region too. body-local + rotation-invariant
                    # since the object quat is identity at reset.
                    centroid_arr[v, o] = pcd_full.mean(axis=0)
                    cxy = centroid_arr[v, o, :2]
                    dxy = pcd_full[:, :2] - cxy
                    obj_xy_radius_arr[v, o] = float(np.linalg.norm(dxy, axis=1).max())
            self._gpu_pcd_local        = wp.array(pcd_arr,           dtype=wp.vec3, device=device)
            self._gpu_centroid_local   = wp.array(centroid_arr,      dtype=wp.vec3, device=device)
            self._gpu_obj_xy_radius    = wp.array(obj_xy_radius_arr, dtype=float,   device=device)
            self._gpu_n_pts            = n_pts

            # ── assignment (world → variant) ──
            self._gpu_assignment = wp.array(
                np.asarray(self.assignment, dtype=np.int32), dtype=int, device=device,
            )

            # ── wrist + mocap addresses (single grasping body) ──
            self._gpu_grasp_obj_name = grasping_obj_name
            self._gpu_grasp_obj_idx  = self.renamed_obj_names.index(grasping_obj_name)
            self._gpu_obj_body_id    = int(mujoco.mj_name2id(  # type: ignore
                mjm, mujoco.mjtObj.mjOBJ_BODY, grasping_obj_name  # type: ignore
            ))
            if self._gpu_obj_body_id == -1:
                raise ValueError(f"object body '{grasping_obj_name}' not found.")

        wrist_jntadr        = int(mjm.body(hand_util.rh_wrist_base_name).jntadr[0])
        self._gpu_wrist_qpa = int(mjm.jnt_qposadr[wrist_jntadr])

        mocap_body_id = int(mujoco.mj_name2id(  # type: ignore
            mjm, mujoco.mjtObj.mjOBJ_BODY, hand_util.rh_mocap_name  # type: ignore
        ))
        if mocap_body_id == -1:
            raise ValueError(f"mocap body '{hand_util.rh_mocap_name}' not found.")
        self._gpu_mocap_id = int(mjm.body_mocapid[mocap_body_id])
        if self._gpu_mocap_id == -1:
            raise ValueError(
                f"body '{hand_util.rh_mocap_name}' is not a mocap body (body_mocapid=-1)."
            )

        # ── inv(rh_hand_center): R_inv, p_inv ──
        p_w2c, R_w2c = t2pr(hand_util.rh_hand_center)
        R_inv = R_w2c.T.astype(np.float32, copy=False)
        p_inv = (-R_inv @ p_w2c).astype(np.float32, copy=False)
        # row-major flatten → wp.mat33 ctor expects 9 floats row-major
        self._gpu_R_inv = wp.mat33(*R_inv.flatten().tolist())
        self._gpu_p_inv = wp.vec3(*p_inv.tolist())

        # ── palm direction in wrist frame (= R_wc[:,0]) for special Y-up init ──
        # assumes b ≈ 0 (rh_hand_center[1,0] is ~0 for every hand), so only (a, c) are kept.
        self._gpu_palm_a = float(R_w2c[0, 0])
        self._gpu_palm_c = float(R_w2c[2, 0])

        # ── Floor lift-out buffers: geom classification (hand / floor plane) +
        # per-world penetration depth. Used by the lift-out post-pass at the end
        # of the capture graph — resolves spawns where the forearm/fingertips start
        # inside the floor (~6% of cases; the weld keeps physics from pushing them
        # out) by the depth read from the contacts. ──
        obj_bodies: set[int] = set()
        for _name in self.renamed_obj_names:
            for _cand in (_name, "body_obj_" + _name):
                _bid = mujoco.mj_name2id(  # type: ignore
                    mjm, mujoco.mjtObj.mjOBJ_BODY, _cand)  # type: ignore
                if _bid != -1:
                    obj_bodies.add(int(_bid))
        _is_hand  = np.zeros(mjm.ngeom, dtype=np.int32)
        _floor_gid = -1
        for _g in range(mjm.ngeom):
            if mjm.geom_type[_g] == mujoco.mjtGeom.mjGEOM_PLANE:  # type: ignore
                if _floor_gid < 0:
                    _floor_gid = _g
                continue
            _b = int(mjm.geom_bodyid[_g])
            if _b == 0:
                continue
            _bb, _is_obj = _b, False
            while _bb != 0:
                if _bb in obj_bodies:
                    _is_obj = True
                    break
                _bb = int(mjm.body_parentid[_bb])
            if not _is_obj:
                _is_hand[_g] = 1
        self._gpu_geom_is_hand = wp.array(_is_hand, dtype=int, device=device)
        self._gpu_floor_gid    = int(_floor_gid)          # -1 = no floor plane → skip lift-out
        self._gpu_pen_depth    = wp.zeros((self.NWORLD,), dtype=float, device=device)

        # ── seed buffer (1-elem array on GPU, updated via _grit_set_seed_kernel) ──
        self._gpu_init_seed_arr   = wp.array(
            np.asarray([0], dtype=np.int32), dtype=int, device=device,
        )
        self._gpu_init_seed_value = 0
        self._gpu_init_capture    = None  # filled by capture_warp_pose_init_kernels()

    def _bump_init_seed(self) -> None:
        """Advance the GPU seed buffer (read inside both init kernels)."""
        self._gpu_init_seed_value = (self._gpu_init_seed_value + 1) & 0x7FFFFFFF
        wp.launch(
            _grit_set_seed_kernel, dim=1,
            inputs=[self._gpu_init_seed_arr, int(self._gpu_init_seed_value)],
        )

    def warp_obj_pose_init_kernel(
        self,
        table_height:    float = 0.0,
        xy_offset_range: list  = [-0.2, 0.2],
        z_clearance:     float = 0.001,
    ) -> None:
        """
        GPU-kernel version of warp_obj_pose_init() — no host roundtrip.

        Writes only:
            d.qpos[:, qpa:qpa+7] for every object slot
            d.qvel[:, :]           = 0
        Other state is untouched, so call AFTER mjwarp.reset_data(m, d) if
        you want a fully clean state.
        """
        if not hasattr(self, '_gpu_obj_qpos_addrs'):
            raise RuntimeError("Call setup_warp_pose_init_kernels() first.")

        self._bump_init_seed()
        wp.launch(
            _grit_warp_obj_pose_init_kernel, dim=self.NWORLD,
            inputs=[
                self._gpu_init_seed_arr,
                int(self._n_obj),
                float(table_height),
                float(xy_offset_range[0]), float(xy_offset_range[1]),
                float(z_clearance),
                self._gpu_obj_qpos_addrs,
                self._gpu_assignment,
                self._gpu_var_obj_min_z,
                self.d.qpos,
                self.d.qvel,
            ],
        )

    def warp_wrist_pose_init_kernel(
        self,
        xyz_range  = ((-0.4, 0.4), (-0.4, 0.4), (0.1, 0.5)),
        roll_range = (-np.pi, np.pi),
        noise_deg: float = 30.0,
        palm_facing_y_up_prob: float = 0.2,
        table_height: float = 0.0,
        y_up_z_range = (0.10, 0.25),
        y_up_margin_range = (0.03, 0.10),
        p_init_range = (0.2, 0.35),
        p_init_min_h: float = 0.12,
    ) -> None:
        """
        GPU-kernel version of warp_wrist_pose_init() — no host roundtrip.

        Pre-condition: d.xpos / d.xmat must reflect the CURRENT object poses
        (i.e. call mjwarp.forward(m, d) after warp_obj_pose_init_kernel before
        invoking this).

        Two init branches per world (sampled per-world from the GPU RNG):
          * **Default (1 - palm_facing_y_up_prob)**: hand-center +X facing
            the object + ±``noise_deg`` rpy noise.
          * **Special (palm_facing_y_up_prob)**: wrist Y axis pinned exactly to
            world up; only the yaw is chosen so the palm faces the object.
            wrist z = ``table_height`` + Uniform(``y_up_z_range``);
            wrist xy = obj_xy − (obj_xy_radius + Uniform(``y_up_margin_range``))
                       · [cos azim, sin azim] — slightly beyond the object AABB.
            No noise applied (preserves Y up).

        Writes:
            d.qpos       (wrist free-joint slot, 7 floats)
            d.mocap_pos  [w, mocap_id]
            d.mocap_quat [w, mocap_id]
        """
        if not hasattr(self, '_gpu_pcd_local'):
            raise RuntimeError("Call setup_warp_pose_init_kernels() first.")

        self._bump_init_seed()
        wp.launch(
            _grit_warp_wrist_pose_init_kernel, dim=self.NWORLD,
            inputs=[
                self._gpu_init_seed_arr,
                int(self._gpu_obj_body_id),
                int(self._gpu_grasp_obj_idx),
                int(self._gpu_n_pts),
                int(self._gpu_wrist_qpa),
                int(self._gpu_mocap_id),
                float(np.pi / 180.0 * noise_deg),
                float(xyz_range[0][0]), float(xyz_range[0][1]),  # x  lo/hi
                float(xyz_range[1][0]), float(xyz_range[1][1]),  # y  lo/hi
                float(xyz_range[2][0]), float(xyz_range[2][1]),  # z  lo/hi
                float(roll_range[0]),  float(roll_range[1]),
                float(p_init_range[0]), float(p_init_range[1]),
                float(table_height + p_init_min_h),
                self._gpu_R_inv,
                self._gpu_p_inv,
                float(self._gpu_palm_a), float(self._gpu_palm_c),
                float(palm_facing_y_up_prob),
                float(table_height),
                float(y_up_z_range[0]),      float(y_up_z_range[1]),
                float(y_up_margin_range[0]), float(y_up_margin_range[1]),
                self._gpu_obj_xy_radius,
                self._gpu_pcd_local,
                self._gpu_centroid_local,
                self._gpu_assignment,
                self.d.xpos,
                self.d.xmat,
                self.d.qpos,
                self.d.mocap_pos,
                self.d.mocap_quat,
            ],
        )

    def capture_warp_pose_init_kernels(
        self,
        table_height:    float = 0.0,
        xy_offset_range: list  = [-0.2, 0.2],
        z_clearance:     float = 0.001,
        wrist_xyz_range = ((-0.4, 0.4), (-0.4, 0.4), (0.1, 0.5)),   # (x, y, z) — all three independent
        wrist_roll_range = (-np.pi, np.pi),
        wrist_noise_deg: float = 30.0,
        palm_facing_y_up_prob: float = 0.1,
        y_up_z_range = (0.10, 0.20),
        y_up_margin_range = (0.10, 0.15),
        p_init_range = (0.2, 0.35),
        p_init_min_h: float = 0.12,
    ) -> None:
        """
        Capture a CUDA graph that runs the full reset sequence:
            obj_kernel → mjwarp.forward → wrist_kernel → mjwarp.forward

        After capture, call warp_pose_init_capture_launch() to re-init all
        worlds with new randoms in a single graph launch.

        The seed is read from a wp.array inside the captured kernels, so each
        launch sees a fresh value (we update it via the small set-seed kernel
        OUTSIDE the captured region — see warp_pose_init_capture_launch).

        Note: scalar args (table_height, xy_offset_range, z_clearance,
        xyz_range, noise_deg) are baked into the graph at capture time. To
        change them, call capture_warp_pose_init_kernels() again.
        """
        if not hasattr(self, '_gpu_pcd_local'):
            raise RuntimeError("Call setup_warp_pose_init_kernels() first.")

        # Initial forward to populate xpos/xmat consistent with current qpos
        # (so the very first captured-graph launch reads a sensible pose).
        mjwarp.forward(self.m, self.d)

        with wp.ScopedCapture() as cap:
            wp.launch(
                _grit_warp_obj_pose_init_kernel, dim=self.NWORLD,
                inputs=[
                    self._gpu_init_seed_arr,
                    int(self._n_obj),
                    float(table_height),
                    float(xy_offset_range[0]), float(xy_offset_range[1]),
                    float(z_clearance),
                    self._gpu_obj_qpos_addrs,
                    self._gpu_assignment,
                    self._gpu_var_obj_min_z,
                    self.d.qpos,
                    self.d.qvel,
                ],
            )
            mjwarp.forward(self.m, self.d)
            wp.launch(
                _grit_warp_wrist_pose_init_kernel, dim=self.NWORLD,
                inputs=[
                    self._gpu_init_seed_arr,
                    int(self._gpu_obj_body_id),
                    int(self._gpu_grasp_obj_idx),
                    int(self._gpu_n_pts),
                    int(self._gpu_wrist_qpa),
                    int(self._gpu_mocap_id),
                    float(np.pi / 180.0 * wrist_noise_deg),
                    float(wrist_xyz_range[0][0]), float(wrist_xyz_range[0][1]),  # x  lo/hi
                    float(wrist_xyz_range[1][0]), float(wrist_xyz_range[1][1]),  # y  lo/hi
                    float(wrist_xyz_range[2][0]), float(wrist_xyz_range[2][1]),  # z  lo/hi
                    float(wrist_roll_range[0]),   float(wrist_roll_range[1]),
                    float(p_init_range[0]),       float(p_init_range[1]),
                    float(table_height + p_init_min_h),
                    self._gpu_R_inv,
                    self._gpu_p_inv,
                    float(self._gpu_palm_a), float(self._gpu_palm_c),
                    float(palm_facing_y_up_prob),
                    float(table_height),
                    float(y_up_z_range[0]),      float(y_up_z_range[1]),
                    float(y_up_margin_range[0]), float(y_up_margin_range[1]),
                    self._gpu_obj_xy_radius,
                    self._gpu_pcd_local,
                    self._gpu_centroid_local,
                    self._gpu_assignment,
                    self.d.xpos,
                    self.d.xmat,
                    self.d.qpos,
                    self.d.mocap_pos,
                    self.d.mocap_quat,
                ],
            )
            mjwarp.forward(self.m, self.d)
            # ── Floor lift-out post-pass: read the per-world deepest hand↔floor
            # penetration from the contacts computed by the forward above and raise
            # the wrist (+mocap) z by that amount. Only penetrating worlds (~6%) are
            # touched, so the spawn distribution is barely distorted. ──
            if self._gpu_floor_gid >= 0:
                wp.launch(_grit_fill_zero_kernel, dim=self.NWORLD,
                          inputs=[self._gpu_pen_depth])
                wp.launch(
                    _grit_hand_floor_pen_scan_kernel, dim=int(self.d.naconmax),
                    inputs=[
                        self.d.nacon, self.d.contact.geom,
                        self.d.contact.dist, self.d.contact.worldid,
                        self._gpu_geom_is_hand, int(self._gpu_floor_gid),
                        self._gpu_pen_depth,
                    ],
                )
                wp.launch(
                    _grit_wrist_floor_liftout_kernel, dim=self.NWORLD,
                    inputs=[
                        self._gpu_pen_depth, 0.002,
                        int(self._gpu_wrist_qpa), int(self._gpu_mocap_id),
                        self.d.qpos, self.d.mocap_pos,
                    ],
                )
                mjwarp.forward(self.m, self.d)
        self._gpu_init_capture = cap

    def warp_pose_init_capture_launch(self) -> None:
        """Update seed and re-launch the captured pose-init graph."""
        if self._gpu_init_capture is None:
            raise RuntimeError(
                "No captured graph. Call capture_warp_pose_init_kernels() first."
            )
        self._bump_init_seed()
        wp.capture_launch(self._gpu_init_capture.graph)

    def make_parser_from_variant(self, variant_idx: int) -> HandRLParserClass:
        """
        Create a new HandRLParserClass from the XML of variant[variant_idx].
        If variant_pcd_cache has been built, the PCD cache is restored as well.

        - Call this to get a completely fresh env when closing the viewer and restarting the outer loop.
        - Unlike the in-place patch (apply_variant_to_cpu_model), the model state is fully reinitialized.
        """
        parser = HandRLParserClass(rel_xml_path=self.variant_xml_paths[variant_idx], verbose=False)

        # group=4 geoms are skeleton placeholder slots: the CPU viewer would render
        # the skeleton mesh and pollute the visual mesh, so disable them with dataid=-1.
        disabled_mask = parser.model.geom_group == 4
        parser.model.geom_dataid[disabled_mask] = -1
        mujoco.mj_forward(parser.model, parser.data)  # type: ignore

        if hasattr(self, 'variant_pcd_cache') and variant_idx in self.variant_pcd_cache:
            for body_name, pcd in self.variant_pcd_cache[variant_idx].items():
                parser.obj_pcd_local_cache[body_name] = pcd
            for body_name, v in self.variant_obj_min[variant_idx].items():
                parser.relative_obj_min[body_name] = v
            for body_name, v in self.variant_obj_max[variant_idx].items():
                parser.relative_obj_max[body_name] = v
        return parser

    def apply_variant_to_cpu_model(self, variant_idx: int):
        """
        Overwrite the geom/body fields of the CPU MjModel (mj_env.model) with variant[variant_idx].
        mujoco.mj_forward(env.model, env.data) must be called afterwards to refresh the kinematics.

        If variant_pcd_cache has been built, obj_pcd_local_cache / relative_obj_min/max are
        updated too so that get_obj_pcd() returns the new object shape.
        """
        ref  = self.variants[variant_idx]
        mask = self.variants_geom_dataid_dict[variant_idx]  # True = disabled slot
        mjm  = self.mj_env.model

        dataid        = ref.geom_dataid.copy()
        dataid[mask]  = -1
        mjm.geom_dataid[:]      = dataid
        mjm.geom_size[:]        = ref.geom_size
        mjm.geom_pos[:]         = ref.geom_pos
        mjm.geom_quat[:]        = ref.geom_quat
        mjm.geom_group[:]       = ref.geom_group
        mjm.geom_rbound[:]      = ref.geom_rbound
        mjm.body_mass[:]        = ref.body_mass
        mjm.body_subtreemass[:] = ref.body_subtreemass
        mjm.body_inertia[:]     = ref.body_inertia
        mjm.body_invweight0[:]  = ref.body_invweight0
        mjm.body_ipos[:]        = ref.body_ipos
        mjm.body_iquat[:]       = ref.body_iquat
        mjm.body_pos[:]         = ref.body_pos
        mjm.body_quat[:]        = ref.body_quat

        if hasattr(self, 'variant_pcd_cache') and variant_idx in self.variant_pcd_cache:
            for body_name, pcd in self.variant_pcd_cache[variant_idx].items():
                self.mj_env.obj_pcd_local_cache[body_name] = pcd
            for body_name, v in self.variant_obj_min[variant_idx].items():
                self.mj_env.relative_obj_min[body_name] = v
            for body_name, v in self.variant_obj_max[variant_idx].items():
                self.mj_env.relative_obj_max[body_name] = v

    def _has_real_bottom_watertight(
        self,
        mjm,
        root_body_name: str,
        prefix: str = "bottom_",
        min_collide_geoms: int = 2,
    ) -> bool:
        """Detect whether ``root_body_name``'s descendant subtree contains a
        non-graspable bottom area that is **physically real** (not a dummy).

        Why this matters for object reset:
            ``build_obj_spec_lst`` flags a ``bottom_*`` body that owns exactly
            one ``contype==1`` geom as a *dummy* and force-zeros its contype
            / conaffinity (visual-only placeholder, e.g. an Objaverse bottom
            cap that has no physical role). When ≥ ``min_collide_geoms``
            collision-active geoms remain, the bottom represents a real
            non-graspable region — wine-glass base, mug body below the rim,
            bottle stand — whose physical extent the table-clearance
            kernel must account for. The default
            ``variant_pcd_cache`` is built with ``exclude_prefix=('bottom_',)``
            (so wrist target sampling never tries to grasp the non-graspable
            region), but that exclusion also hides the bottom from the
            placement min/max — the object then spawns with its base partly
            inside the floor / table. This helper tells
            ``build_variant_pcd_cache`` when to rebuild a *full-extent* PCD
            for the placement math while keeping the graspable cache intact.

        Args:
            mjm:               compiled variant ``MjModel``.
            root_body_name:    obj-slot root body (e.g.
                               ``"top_watertight_tiny_0"``).
            prefix:            descendant body-name prefix that identifies
                               the non-graspable region.
            min_collide_geoms: how many ``contype != 0`` geoms a matching
                               body must own to count as "real".

        Returns:
            ``True`` iff at least one descendant body whose name starts with
            ``prefix`` has ≥ ``min_collide_geoms`` collision-active geoms.
        """
        root_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, root_body_name)  # type: ignore
        if root_id == -1:
            return False

        n_body   = int(mjm.nbody)
        children = [[] for _ in range(n_body)]
        for cid in range(1, n_body):
            children[int(mjm.body_parentid[cid])].append(cid)

        stack = list(children[root_id])    # descendants only (skip root itself)
        while stack:
            bid   = stack.pop()
            bname = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, bid) or ""  # type: ignore
            if bname.startswith(prefix):
                n_collide = sum(
                    1 for gid in range(mjm.ngeom)
                    if int(mjm.geom_bodyid[gid]) == bid and int(mjm.geom_contype[gid]) != 0
                )
                if n_collide >= min_collide_geoms:
                    return True
            for cid in children[bid]:
                stack.append(cid)
        return False

    def build_variant_pcd_cache(self, n_sample=1024, verbose=False):
        """
        For each variant MjModel build body-local PCD and bounding box for every object slot.
        Mirrors HandRLParserClass.build_obj_pcd_cache.

        Populates:
            self.variant_pcd_cache      : dict[spec_idx, dict[body_name, np.ndarray (n_sample, 3)]]
                                          — **graspable PCD** (top_watertight only). Drives wrist
                                          target sampling + finger→PCD nearest-vertex viz; wrist
                                          init must NEVER target a non-graspable area, so this
                                          stays top-only regardless of whether bottom is real.
            self.variant_full_pcd_cache : dict[spec_idx, dict[body_name, np.ndarray (n_sample, 3)]]
                                          — **full-extent PCD** (top + bottom when bottom is
                                          real per :meth:`_has_real_bottom_watertight`).
                                          Drives placement clearance (``variant_obj_min``) and
                                          XY radius (``setup_warp_pose_init_kernels``) — both
                                          of which must see the full physical footprint to
                                          avoid spawning wine-glass bases through the floor.
                                          When no real bottom exists, this is identical to
                                          ``variant_pcd_cache`` (no extra cost).
            self.variant_obj_min        : dict[spec_idx, dict[body_name, np.ndarray (3,)]]
                                          — negated min of the **full** PCD (placement).
            self.variant_obj_max        : dict[spec_idx, dict[body_name, np.ndarray (3,)]]
                                          — max of the **full** PCD.
            self.variant_has_real_bottom: dict[spec_idx, dict[body_name, bool]]
                                          — diagnostic; True iff bottom_* contributed extra
                                          extent for this (variant, body) pair.
        """
        self.variant_pcd_cache       = {}
        self.variant_full_pcd_cache  = {}
        self.variant_obj_min         = {}
        self.variant_obj_max         = {}
        self.variant_has_real_bottom = {}

        for spec_idx, variant_mjm in enumerate(self.variants):
            variant_mjd = mujoco.MjData(variant_mjm)  # type: ignore
            mujoco.mj_forward(variant_mjm, variant_mjd)  # type: ignore

            pcd_dict, full_pcd_dict, min_dict, max_dict, real_bottom_dict = {}, {}, {}, {}, {}
            for body_name in self.renamed_obj_names:
                # ── Graspable PCD: top_watertight subtree only ────────────
                # This is the cache that drives wrist target sampling. Even
                # when a real bottom exists we must NOT sample wrist
                # targets there (non-graspable by construction).
                pcd = self._build_variant_pcd(
                    variant_mjm, variant_mjd, body_name, n_sample,
                    exclude_prefix=('bottom_',),
                )

                # ── Full-extent PCD: ALWAYS sample the whole object subtree
                # (top + bottom) for placement min/max so the object rests on
                # its TRUE lowest face (e.g. a wine-glass base), not just the
                # graspable top. ``_build_variant_pcd`` already skips
                # placeholder (group==4) and non-mesh geoms, so dummy / absent
                # bottoms (geoms deleted or only padding slots) collapse to the
                # graspable extent automatically — no behaviour change there.
                #
                # BUGFIX: this used to be gated on ``_has_real_bottom_watertight``
                # (a ≥2-collision-geom heuristic). Combined with the old
                # skeleton geom-slot truncation, that made the *placement*
                # min-z sample-dependent: for some ``N_SUB_ENV`` draws a real
                # bottom (wine-glass base) was missed and the object spawned
                # using only the top_watertight PCD (base sunk into the table).
                # Sampling the full subtree unconditionally removes that
                # dependence. ``has_real_bottom`` is kept as a diagnostic only.
                has_real_bottom = self._has_real_bottom_watertight(variant_mjm, body_name)
                real_bottom_dict[body_name] = has_real_bottom
                full_pcd = self._build_variant_pcd(
                    variant_mjm, variant_mjd, body_name, n_sample,
                    exclude_prefix=(),  # include bottom subtree
                )

                if pcd is None:
                    if verbose:
                        print_red(f"[build_variant_pcd_cache] spec={spec_idx} body='{body_name}': no mesh")
                    pcd_dict[body_name]      = np.zeros((n_sample, 3))
                    full_pcd_dict[body_name] = np.zeros((n_sample, 3))
                    min_dict[body_name]      = np.zeros(3)
                    max_dict[body_name]      = np.zeros(3)
                    continue

                pcd_dict[body_name] = pcd

                # Placement min/max + (later) obj_xy_radius from the
                # full-extent PCD. Falls back to the graspable PCD only if the
                # full build returned nothing (shouldn't happen — the full
                # subtree is a superset of the graspable one).
                if full_pcd is not None and len(full_pcd) > 0:
                    full_pcd_dict[body_name] = full_pcd
                    min_dict[body_name]      = -np.min(full_pcd, axis=0)
                    max_dict[body_name]      =  np.max(full_pcd, axis=0)
                    if verbose:
                        z_top_only = float(-np.min(pcd, axis=0)[2])
                        z_full     = float(min_dict[body_name][2])
                        delta_z    = z_full - z_top_only
                        if abs(delta_z) > 1e-6:
                            print_blue(
                                f"[build_variant_pcd_cache] spec={spec_idx} "
                                f"body='{body_name}' (has_real_bottom={has_real_bottom}) → "
                                f"placement min_z {z_top_only:.4f} → {z_full:.4f} "
                                f"(+{delta_z*1000:.1f} mm from full-extent bottom geoms)"
                            )
                else:
                    full_pcd_dict[body_name] = pcd
                    min_dict[body_name]      = -np.min(pcd, axis=0)
                    max_dict[body_name]      =  np.max(pcd, axis=0)

                if verbose:
                    print(
                        f"[build_variant_pcd_cache] spec={spec_idx} "
                        f"body='{body_name}' shape={pcd.shape}  "
                        f"has_real_bottom={has_real_bottom}"
                    )

            self.variant_pcd_cache[spec_idx]       = pcd_dict
            self.variant_full_pcd_cache[spec_idx]  = full_pcd_dict
            self.variant_obj_min[spec_idx]         = min_dict
            self.variant_obj_max[spec_idx]         = max_dict
            self.variant_has_real_bottom[spec_idx] = real_bottom_dict

        if verbose:
            n_real = sum(
                1 for vd in self.variant_has_real_bottom.values()
                for v in vd.values() if v
            )
            print(
                f"[build_variant_pcd_cache] done. "
                f"{len(self.variants)} variants × {len(self.renamed_obj_names)} slots; "
                f"{n_real} (variant, body) pair(s) with real non-graspable bottom area."
            )

    # ──────────────────────────────────────────────────────────────────────
    # Ghost target-hand visualization (per-world overlay)
    # ──────────────────────────────────────────────────────────────────────
    # Same idea as `thirdParty/.../mujoco_warp_viewer.py`: keep a separate
    # `MjModel` for the bare hand (built from `parent_spec` so it's just
    # floor + hand, NO objects) and one `MjData` per world. Caller fills in
    # per-world target qpos via `set_ghost_targets()`, then passes the
    # `(model, data_list)` pair to the viewer's `mjwarp_render(...)` call.
    # `mjv_addGeoms(..., mjCAT_DYNAMIC, ...)` filters out the static floor,
    # so only the hand subtree renders.

    def setup_ghost_hand_model(
        self,
        rgba: tuple = (0.55, 0.55, 0.65, 0.40),
    ) -> None:
        """Build the ghost hand model + per-world MjData (floor + bare hand only).

        The ghost is rebuilt from the *pristine* floor+hand file on disk
        (``self.parent_xml_path``, the ``merge_mjcfs([floor, hand])`` output).

        ⚠️ Do NOT use ``self.parent_spec`` here: ``make_mjcf_from_spec`` mutates
        that in-memory spec **in place** during ``reset()`` (it appends object
        meshes, the dummy object skeleton body *with geoms*, and contact
        sensors — objects are written to a separate ``*_w_obj_*.xml`` file, but
        the spec object itself is modified). Compiling that mutated spec would
        bake a ghost-coloured object body into the ghost model. Since
        ``set_ghost_targets`` only positions the wrist free-joint (+ grid
        offset) and finger qpos — never the object free-joint — every world's
        ghost object would pile up at the same un-offset default pose, showing
        up as a single stray ghost-coloured object in one world. The on-disk
        ``parent_xml_path`` predates that mutation, so it stays clean.

        Args:
            rgba: uniform RGBA applied to every geom (cool grey + alpha by
                default). Tweak per-call before re-rendering if you want to
                A/B different ghost colours; this method is cheap to re-run.
        """
        if not getattr(self, 'parent_xml_path', None):
            raise RuntimeError("parent_xml_path missing — call reset() first.")

        # Rebuild from the clean floor+hand file (NOT self.parent_spec, which
        # make_mjcf_from_spec mutated to include the object skeleton). `.compile()`
        # is the standard mujoco API for spec→model on this codebase.
        ghost_spec               = mujoco.MjSpec.from_file(self.parent_xml_path)  # type: ignore
        self.ghost_hand_model    = ghost_spec.compile()  # type: ignore
        # Override every geom colour so the ghost is visually distinct.
        self.ghost_hand_model.geom_rgba[:] = np.asarray(rgba, dtype=np.float32)

        # qpos layout helpers — wrist free joint (7 floats) + hinge ctrl joints.
        wrist_jntadr = int(
            self.ghost_hand_model.body(self.hand_util.rh_wrist_base_name).jntadr[0]
        )
        self._ghost_wrist_qpa = int(self.ghost_hand_model.jnt_qposadr[wrist_jntadr])
        # Per-actuator driven-joint qpos addr. Guard by transmission type: only
        # JOINT actuators map to a single qpos (``trnid[0]`` is a joint id);
        # tendon/site actuators have no 1:1 qpos → ``-1`` (skipped when writing
        # the ghost qpos, then filled by the equality-coupling solver instead).
        _g = self.ghost_hand_model
        self._ghost_ctrl_to_qpos = np.array([
            int(_g.jnt_qposadr[int(_g.actuator_trnid[ai, 0])])
            if int(_g.actuator_trntype[ai]) == int(mujoco.mjtTrn.mjTRN_JOINT)  # type: ignore
            else -1
            for ai in range(_g.nu)
        ], dtype=int)

        # Cache hand-subtree geoms (anything not welded to the world body — i.e.
        # everything reachable from the wrist's free joint) for the floor-
        # clearance check in `set_ghost_targets`. We use bounding-sphere radii
        # (`geom_rbound`) since they're already pre-computed and conservative.
        _hand_geom_ids = []
        for _gi in range(int(self.ghost_hand_model.ngeom)):
            _bid = int(self.ghost_hand_model.geom_bodyid[_gi])
            if int(self.ghost_hand_model.body_weldid[_bid]) != 0:
                _hand_geom_ids.append(_gi)
        self._ghost_hand_geom_ids = np.asarray(_hand_geom_ids, dtype=int)
        self._ghost_geom_rbound   = np.asarray(
            self.ghost_hand_model.geom_rbound[self._ghost_hand_geom_ids],
            dtype=np.float32,
        )

        # One MjData per parallel world. Each gets its own qpos / xpos / xmat
        # buffer that the viewer reads in its `mjv_addGeoms` pass.
        self.ghost_hand_datas = [
            mujoco.MjData(self.ghost_hand_model)  # type: ignore
            for _ in range(int(self.NWORLD))
        ]

    # ── Viewer render interface ───────────────────────────────────────────
    # ``evaluate.py`` drives the parallel viewer through these hooks ONLY, so it
    # stays embodiment/task-agnostic: each sub-env class supplies its own hand
    # identity, per-world offset roots, ghost, and overlay capability. Overriding
    # classes (e.g. LeaderFollowerSubEnv) replace only what differs.
    @property
    def render_hand_util(self):
        """hand_util for the viewer title + interactive-handle mocap."""
        return self.hand_util

    @property
    def render_offset_body_names(self) -> list:
        """Free-joint root bodies XY-offset per world (hand wrist base + objects)."""
        return [self.hand_util.rh_wrist_base_name, *self.renamed_obj_names]

    @property
    def supports_contact_overlays(self) -> bool:
        """Whether the contact-state colour / contact-sensor markers apply."""
        return True

    def setup_render_ghost(self, handler, rgba) -> bool:
        """Build the target ghost; return True if the task exposes one.
        Single-hand → the handler's target pose (``eval_ghost_targets``)."""
        if handler.eval_ghost_targets() is None:
            return False
        self.setup_ghost_hand_model(rgba=rgba)
        return True

    def update_render_ghost(self, handler, grid_offsets) -> None:
        """Refresh the ghost to the current per-world target (per frame)."""
        g = handler.eval_ghost_targets()
        if g is None:
            return
        tp, tq, tqp = g
        self.set_ghost_targets(target_pos=tp, target_quat=tq, target_qpos=tqp,
                               grid_offsets=grid_offsets)

    @property
    def render_ghost_model(self):
        return getattr(self, "ghost_hand_model", None)

    @property
    def render_ghost_data(self):
        return getattr(self, "ghost_hand_datas", None)

    def render_ghost_geom_rgba(self, handler):
        """Optional per-world ghost geom colour (single-hand handler feature)."""
        fn = getattr(handler, "eval_ghost_geom_rgba", None)
        return fn() if fn is not None else None

    def set_ghost_targets(
        self,
        target_pos:      np.ndarray,    # (NWORLD, 3)
        target_quat:     np.ndarray,    # (NWORLD, 4) wxyz
        target_qpos:     np.ndarray,    # (NWORLD, n_ctrl)
        grid_offsets,                   # (NWORLD, 2) — XY world offset per world
        floor_clearance: Optional[float] = 0.005,   # min hand-geom z; None → disable
    ) -> None:
        """Fill each world's ghost MjData with the target wrist pose, target
        per-actuator qpos, and the grid XY offset; runs ``mj_forward`` so
        ``xpos / xmat / geom_xpos / geom_xmat`` are ready for the viewer.

        Mirrors the thirdParty pattern (``hand_data.qpos[0..6]`` for the wrist
        free joint + ``hand_data.qpos[7:]`` for the per-actuator qpos), but
        with the wrist-qpa / ctrl-to-qpos mapping resolved properly so the
        method works regardless of XML joint ordering.

        Args:
            floor_clearance: if not None, after the initial ``mj_forward`` we
                check the lowest hand-geom point per world (using bounding
                spheres: ``geom_xpos.z - geom_rbound``). If any geom dips
                below ``floor_clearance``, we lift the wrist's z by the
                deficit and re-forward. Set to ``None`` to disable when the
                caller already guarantees floor-safe targets (or when there
                is no floor in the scene).
        """
        if getattr(self, 'ghost_hand_model', None) is None:
            raise RuntimeError("Call setup_ghost_hand_model() first.")
        if target_pos.shape[0] != self.NWORLD:
            raise ValueError(
                f"target_pos has {target_pos.shape[0]} rows, expected NWORLD={self.NWORLD}"
            )

        wqpa = self._ghost_wrist_qpa
        cqi  = self._ghost_ctrl_to_qpos
        rbnd = self._ghost_geom_rbound
        gids = self._ghost_hand_geom_ids
        for w in range(int(self.NWORLD)):
            d = self.ghost_hand_datas[w]
            ox = float(grid_offsets[w][0])
            oy = float(grid_offsets[w][1])
            d.qpos[wqpa + 0] = float(target_pos[w, 0]) + ox
            d.qpos[wqpa + 1] = float(target_pos[w, 1]) + oy
            d.qpos[wqpa + 2] = float(target_pos[w, 2])
            d.qpos[wqpa + 3] = float(target_quat[w, 0])
            d.qpos[wqpa + 4] = float(target_quat[w, 1])
            d.qpos[wqpa + 5] = float(target_quat[w, 2])
            d.qpos[wqpa + 6] = float(target_quat[w, 3])
            for j, qaddr in enumerate(cqi):
                if qaddr >= 0:                      # skip non-JOINT actuators (-1)
                    d.qpos[qaddr] = float(target_qpos[w, j])
            # Coupled hands (allex/inspire/shadow): fill the passive
            # equality-coupled joints so the ghost FK is geometrically correct.
            # ``mj_forward`` does NOT project qpos onto equality constraints, so
            # without this the coupled distal joints stay at default and
            # ``target_af_xpos`` / the ghost render are wrong. No-op for serial
            # hands (no joint equality).
            hand_utils.fill_passive_joints_from_equality(self.ghost_hand_model, d.qpos)
            mujoco.mj_forward(self.ghost_hand_model, d)  # type: ignore

            # Floor-safety lift: invariant to the sampled quat / qpos.
            if floor_clearance is not None and gids.size > 0:
                geom_z   = np.asarray(d.geom_xpos)[gids, 2]
                lowest_z = float((geom_z - rbnd).min())
                if lowest_z < floor_clearance:
                    d.qpos[wqpa + 2] += float(floor_clearance) - lowest_z
                    mujoco.mj_forward(self.ghost_hand_model, d)  # type: ignore

    # ──────────────────────────────────────────────────────────────────────
    # Parallel PCD visualization
    # ──────────────────────────────────────────────────────────────────────

    def plot_parallel_pcd(
        self,
        d,
        n_pcd_subsample: int   = 64,
        pcd_r:           float = 0.003,
        alpha:           float = 0.8,
    ):
        """
        Add PCD sphere markers for every parallel world, colour-coded by variant index.
        Must be called BEFORE mjwarp_render() so markers are flushed inside the render.

        Args:
            d:               MuJoCo Warp Data (GPU) after simulation step.
            n_pcd_subsample: Points rendered per world per object body (0 = all).
            pcd_r:           Sphere radius for each point marker.
            alpha:           Marker opacity.
        """
        viewer = self.mj_env.viewer
        if getattr(viewer, '_offsets_for_parallel_render', None) is None:
            return  # grid offsets not yet initialised (skip first frame)

        if not getattr(self, 'variant_pcd_cache', None):
            return

        xpos_np = d.xpos.numpy()   # (nworld, nbody, 3)
        xmat_np = d.xmat.numpy()   # (nworld, nbody, 3, 3) or (nworld, nbody, 9)

        VARIANT_COLORS = [
            (0.0, 1.0, 1.0, alpha),   # cyan
            (1.0, 0.5, 0.0, alpha),   # orange
            (0.8, 0.0, 0.8, alpha),   # magenta
            (0.0, 0.9, 0.0, alpha),   # green
            (1.0, 1.0, 0.0, alpha),   # yellow
            (0.5, 0.5, 1.0, alpha),   # lavender
            (1.0, 0.3, 0.3, alpha),   # coral
            (0.3, 0.9, 0.3, alpha),   # lime
        ]

        for w in range(self.NWORLD):
            variant_idx = int(self.assignment[w])
            if variant_idx not in self.variant_pcd_cache:
                continue

            offset_xy = viewer._offsets_for_parallel_render[w] # type: ignore
            color     = VARIANT_COLORS[variant_idx % len(VARIANT_COLORS)]

            for body_name in self.renamed_obj_names:
                pcd_local = self.variant_pcd_cache[variant_idx].get(body_name)
                if pcd_local is None or len(pcd_local) == 0:
                    continue

                body_id = mujoco.mj_name2id(  # type: ignore
                    self.mj_env.model, mujoco.mjtObj.mjOBJ_BODY, body_name  # type: ignore
                )
                if body_id == -1:
                    continue

                p_body = np.array(xpos_np[w, body_id]).ravel()[:3]
                R_body = np.array(xmat_np[w, body_id]).reshape(3, 3)

                # Fixed subsample indices — computed once and cached to avoid per-frame flicker
                if not hasattr(self, '_parallel_pcd_idx_cache'):
                    self._parallel_pcd_idx_cache = {}
                cache_key = (variant_idx, body_name, n_pcd_subsample)
                if cache_key not in self._parallel_pcd_idx_cache:
                    n_pts = len(pcd_local)
                    if 0 < n_pcd_subsample < n_pts:
                        self._parallel_pcd_idx_cache[cache_key] = \
                            np.random.choice(n_pts, n_pcd_subsample, replace=False)
                    else:
                        self._parallel_pcd_idx_cache[cache_key] = np.arange(n_pts)
                pts_local = pcd_local[self._parallel_pcd_idx_cache[cache_key]]

                # body-local → world frame + grid XY offset
                pcd_world = (R_body @ pts_local.T).T + p_body
                pcd_world[:, 0] += offset_xy[0]
                pcd_world[:, 1] += offset_xy[1]

                self.mj_env.plot_spheres(p_list=pcd_world, r=pcd_r, rgba=color)


    def plot_parallel_object_com(
        self,
        d,
        com_r:        float = 0.008,
        com_rgba:     tuple = (1.0, 0.0, 1.0, 0.95),
        origin_r:     float = 0.004,
        origin_rgba: "tuple | None" = (0.2, 0.2, 0.2, 0.9),
        body_names:  "list[str] | None" = None,
        draw_offset_arrow: bool  = True,
        arrow_r:     float = 0.0007,
        arrow_rgba:  tuple = (1.0, 1.0, 0.0, 0.95),
        verbose_first_frame: bool = False,
    ):
        """Per-world object Center-of-Mass markers (+ body-origin marker
        and the origin → COM offset arrow).

        For each parallel world *w* and each object body, draws:

          * a magenta sphere at the COM world position
            ``d.xipos[w, body_id] + grid_offset[w]``
          * (optional) a small dark sphere at the body-origin world position
            ``d.xpos [w, body_id] + grid_offset[w]``
          * (optional) a yellow arrow from origin → COM (only when the
            body-local COM offset has appreciable magnitude — i.e. mass is
            not perfectly centered on the body frame, which is the usual
            case for asymmetric meshes like wine bottles or cans)

        Call BEFORE ``mjwarp_render(...)`` so the marker buffer is flushed
        inside the render call.

        Why this is useful here:
            ``heterogeneous_env_setup`` per-world overrides ``body_ipos``,
            ``body_iquat``, ``body_mass``, ``body_inertia``. If any of these
            disagree with the actual mesh (e.g. ``body_ipos`` left at the
            origin while the mesh is shifted, or ``body_iquat`` swapped
            principal axes), the body will tip in directions that look
            "wrong" relative to the geometry. Watching the magenta COM
            sphere relative to the visible mesh tells you immediately
            whether the inertial frame matches the shape MuJoCo is actually
            rendering.

        Args:
            d:             MuJoCo Warp Data (GPU) — read after ``mj_forward``
                           / ``capture_step.graph`` launch.
            com_r:         sphere radius for COM marker.
            com_rgba:      RGBA for COM marker.
            origin_r:      sphere radius for body-origin marker (set to 0
                           or origin_rgba=None to suppress).
            origin_rgba:   RGBA for body-origin marker, or ``None`` to skip.
            body_names:    Subset of body names to draw; ``None`` =
                           ``self.renamed_obj_names`` (every object body).
            draw_offset_arrow: if True, plot a yellow arrow origin → COM
                           per world (only when the offset is > origin_r).
            arrow_r:       arrow radius.
            arrow_rgba:    arrow RGBA.
            verbose_first_frame: print a one-shot summary of per-world
                           body-local ``body_ipos`` (CPU-side) for sanity
                           checking the inertial frame. Useful to detect
                           variants where ``body_ipos`` was left at
                           ``(0, 0, 0)`` while the visual mesh has an
                           obvious offset.
        """
        viewer = self.mj_env.viewer
        if getattr(viewer, '_offsets_for_parallel_render', None) is None:
            return  # grid offsets not yet initialised (skip first frame)

        names = body_names if body_names is not None else self.renamed_obj_names
        if not names:
            return

        # body id is a model-level (skeleton) attribute — variants share the
        # same body slots, only geom contents differ.
        body_ids = []
        for nm in names:
            bid = mujoco.mj_name2id(  # type: ignore
                self.mj_env.model, mujoco.mjtObj.mjOBJ_BODY, nm  # type: ignore
            )
            if bid != -1:
                body_ids.append((nm, bid))
        if not body_ids:
            return

        xipos_np = d.xipos.numpy()                      # (NWORLD, nbody, 3)
        xpos_np  = d.xpos.numpy()                       # (NWORLD, nbody, 3)

        if verbose_first_frame and not getattr(self, "_com_first_frame_dumped", False):
            self._com_first_frame_dumped = True
            # body_ipos is per-world (2D) after heterogeneous_env_setup;
            # this snapshot helps spot a variant whose inertial origin was
            # never moved off the body frame even though its mesh is
            # obviously offset (a common compile-side hazard).
            ipos_np = self.m.body_ipos.numpy()          # (n_ipos_set, nbody, 3)
            print("[plot_parallel_object_com] per-variant body_ipos snapshot:")
            seen = set()
            for w in range(self.NWORLD):
                v = int(self.assignment[w])
                if v in seen:
                    continue
                seen.add(v)
                row_idx = w % ipos_np.shape[0]          # mjwarp can store fewer rows
                for nm, bid in body_ids:
                    p = ipos_np[row_idx, bid]
                    n = float(np.linalg.norm(p))
                    flag = "" if n > 1e-6 else "  ← ZERO (suspicious for asymmetric meshes)"
                    print(f"  variant {v:2d}  body '{nm}'  body_ipos={p.round(4)}  |ipos|={n:.4f}{flag}")

        for w in range(self.NWORLD):
            offset_xy = viewer._offsets_for_parallel_render[w]  # type: ignore
            for nm, bid in body_ids:
                com_w   = np.array(xipos_np[w, bid]).ravel()[:3].copy()
                org_w   = np.array(xpos_np [w, bid]).ravel()[:3].copy()
                com_w[0] += offset_xy[0]; com_w[1] += offset_xy[1]
                org_w[0] += offset_xy[0]; org_w[1] += offset_xy[1]

                if origin_rgba is not None and origin_r > 0:
                    self.mj_env.plot_sphere(p=org_w, r=origin_r, rgba=origin_rgba)

                self.mj_env.plot_sphere(p=com_w, r=com_r, rgba=com_rgba)

                if draw_offset_arrow and np.linalg.norm(com_w - org_w) > max(origin_r, 1e-3):
                    self.mj_env.plot_arrow_fr2to(
                        p_fr=org_w, p_to=com_w,
                        r=arrow_r, rgba=arrow_rgba, label='',
                    )


    # ════════════════════════════════════════════════════════════════════
    # Per-world finger-site → object-PCD closest-vertex (warp kernel + viz)
    # ════════════════════════════════════════════════════════════════════
    # Warp port of the CPU pattern used in
    # ``01_hand_setup/13_dyn_init_single_obj`` (and the inline copy in
    # ``03_warp_parallel/04_warp_parallel_spawn``)::
    #
    #     site_obj_dist = np.linalg.norm(
    #         finger_site_xpos[:, None, :] - grasping_obj_pcd[None, :, :], axis=-1)
    #     pcd_min_idxs = np.argmin(site_obj_dist, axis=-1)
    #     for site_p, pcd_min_idx in zip(...):
    #         env.plot_arrow_fr2to(p_fr=site_p, p_to=grasping_obj_pcd[pcd_min_idx], ...)
    #
    # In a parallel warp env this would require copying every world's PCD
    # to CPU + a Python loop. The kernel below moves the
    # ``(NWORLD × n_finger × n_pts)`` distance comparison onto the GPU and
    # only round-trips the small (NWORLD, n_finger, 3) result back for
    # rendering.
    # ────────────────────────────────────────────────────────────────────

    def plot_parallel_finger_pcd_arrows(
        self,
        d,
        finger_site_ids,
        grasping_obj_name: str | None = None,
        arrow_r:    float = 0.0005,
        arrow_rgba                = (0.0, 1.0, 0.0, 0.95),
        sphere_r:   float = 0.0015,
        sphere_rgba              = (1.0, 1.0, 0.0, 0.95),
        draw_target_sphere: bool = True,
        max_dist:   float | None = None,
    ) -> None:
        """Draw a per-world arrow from each finger site to its closest
        object-PCD vertex. Must be called BEFORE ``mjwarp_render()`` so
        the markers are flushed during the render pass (same convention
        as ``plot_parallel_pcd``).

        Args:
            d:                  warp ``MjData`` (forward must have run so
                                ``site_xpos`` is current).
            finger_site_ids:    finger-site indices in the warp model
                                (e.g. ``sampled_orch.hand_sensor_site_id_list[2:]``
                                — arm + palm dropped, only finger touch sites).
            grasping_obj_name:  target body name; defaults to ``renamed_obj_names[0]``.
            arrow_r/rgba:       arrow shaft radius / colour.
            sphere_r/rgba:      marker on the closest PCD vertex (set
                                ``draw_target_sphere=False`` to skip).
            draw_target_sphere: toggle the closest-vertex sphere marker.
            max_dist:           if not None, suppress drawing when the
                                computed distance exceeds this threshold
                                (useful to hide arrows when the hand is
                                far away — keeps the viewer uncluttered).
        """
        viewer = self.mj_env.viewer
        if getattr(viewer, '_offsets_for_parallel_render', None) is None:
            return  # grid offsets not yet initialised (skip first frame)

        if not finger_site_ids or len(finger_site_ids) == 0:
            return

        closest_p_np, min_dist_np = self.compute_finger_to_pcd_closest(
            d, finger_site_ids, grasping_obj_name=grasping_obj_name,
        )
        # site_xpos round-trip (one CPU sync per frame — same cadence as
        # plot_parallel_pcd, so no extra sync on the render hot path).
        site_xpos_np = d.site_xpos.numpy()                  # (NWORLD, nsite, 3)
        site_ids_np  = self._finger_site_ids_cache          # already int32, (n_finger,)

        n_finger = int(site_ids_np.size)
        for w in range(self.NWORLD):
            offset_xy = viewer._offsets_for_parallel_render[w]  # type: ignore
            offset    = np.array([float(offset_xy[0]), float(offset_xy[1]), 0.0],
                                 dtype=np.float64)
            for f in range(n_finger):
                if max_dist is not None and float(min_dist_np[w, f]) > max_dist:
                    continue
                sid       = int(site_ids_np[f])
                site_p    = np.asarray(site_xpos_np[w, sid], dtype=np.float64) + offset
                closest_p = np.asarray(closest_p_np[w, f],   dtype=np.float64) + offset
                if not (np.isfinite(site_p).all() and np.isfinite(closest_p).all()):
                    continue
                self.mj_env.plot_arrow_fr2to(
                    p_fr=site_p, p_to=closest_p,
                    r=arrow_r, rgba=arrow_rgba, label='',
                )
                if draw_target_sphere:
                    self.mj_env.plot_sphere(
                        p=closest_p, r=sphere_r, rgba=sphere_rgba,
                    )



