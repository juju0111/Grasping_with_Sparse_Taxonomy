"""Partial-view PCD — batched warp HPR (Hidden Point Removal) for parallel envs.

Warp port of the single-env numpy HPR reference (``spherical_grid_hpr_numpy``:
Katz spherical flipping + spherical-grid depth buffer + cross dilation) that
**computes all worlds in a batch with 4 GPU kernel launches**.

Mathematical equivalence (closed form)
--------------------------------------
``p_flipped = pc + 2(R−‖pc‖)·pc/‖pc‖ = pc·(2R−‖pc‖)/‖pc‖`` (``pc = p−cam``),
so **the direction is that of pc** and the flipped norm has the closed form
``2R−n``. Hence (θ,φ) are computed directly from the camera-centred vector and
the flipped coordinates never need to be materialised.
Verified against the numpy reference with 0% mismatch (sphere/box/cylinder x
random poses x random cameras).

Pipeline (all in-place on preallocated buffers → safe under CUDA-graph capture)
-------------------------------------------------------------------------------
==  =========  ============================================================
K0  reset      ``maxd ← 0``, ``grid ← −1``
K1  points     (a) body-local → world transform + dist + per-world maxd (atomic)
               (b) dist + maxd only, from an external world-frame PCD buffer (RL path)
K2  scatter    ``fnorm = 2R−n``, (θ,φ) → grid cell, scatter-max depth buffer
K3  visible    cross (4-neighbour) dilation performed at read time → ``fnorm ≥ max₅−tol``
==  =========  ============================================================

Two consumers
-------------
* **Notebook viewer**:
  ``update(xpos, xmat)`` — transforms the ``variant_pcd_cache`` stack via path (a) and runs HPR.
* **RL task** (``rl_envs/object_grasping_w_partial_bps``):
  ``update_from_world_pcd(transformed_pcd_wp)`` — reuses the world-frame PCD
  already produced by the base reward pipeline via path (b) (no extra transform).

Camera sampling (same convention as the reference ``sample_partial_view_camera``)
---------------------------------------------------------------------------------
A position is drawn from a candidate box, **only its direction** is kept, and
the final camera is placed ``cam_dist`` away from the object centre.
* ``resample_cameras()``      — host numpy (for notebooks, event-driven, centre = PCD mean)
* ``resample_cameras_gpu()``  — GPU masked kernel (for RL, at the same time as the
  per-world done reset, without CPU sync; centre = object root body ``xpos``,
  which approximates the PCD mean).
  Seed handling follows the ``_gpu_seed_wp`` idiom of
  :class:`~grit.util.warp_per_world_condition.WarpPerWorldCondition`
  (the host bumps the seed buffer before every launch).

Frozen initial view + sensor noise (student observation in teacher-student)
---------------------------------------------------------------------------
When the student must act from **a single partial view captured at episode
start** (the default in ``object_grasping_w_partial_bps``), instead of live
recomputation every step:

* ``capture_masked(points, mask)`` — freezes the current PCD of the reset worlds
  only into ``frozen_pcd`` (+ ``pos_noise_std`` jitter, ``dropout_prob`` keep-mask).
  The noise realisation is fixed at capture time as well = fixed noise of one
  initial depth frame.
* ``refresh_from_frozen()`` — recomputes HPR visibility from ``frozen_pcd`` and
  ANDs the dropout mask. frozen_pcd and cam are constant over the episode, so
  ``visible`` is deterministically identical.

Noise knobs (all fixed at capture/resample time, constant within an episode):
``pos_noise_std`` (point position, m), ``dropout_prob`` (point dropout),
``cam_noise_std`` (camera position, m). With all zero this is pure noise-free HPR.
"""
from __future__ import annotations

import numpy as np
import warp as wp

_PV_PI = wp.constant(3.141592653589793)


# ══════════════════════════════════════════════════════════════════════════
# Kernels
# ══════════════════════════════════════════════════════════════════════════
@wp.kernel
def _pv_reset_maxd_kernel(
    maxd: wp.array(dtype=float),                       # type: ignore (NWORLD,)
):
    w = wp.tid()
    maxd[w] = 0.0


@wp.kernel
def _pv_reset_grid_kernel(
    grid: wp.array(dtype=float, ndim=2),               # type: ignore (NWORLD, G*G)
):
    w, c = wp.tid()
    grid[w, c] = -1.0


@wp.kernel
def _pv_transform_kernel(
    pcd_local:  wp.array(dtype=wp.vec3, ndim=2),       # type: ignore (V, N) body-local PCD per variant
    assignment: wp.array(dtype=int),                   # type: ignore (NWORLD,) world → variant idx
    xpos:       wp.array(dtype=wp.vec3, ndim=2),       # type: ignore (NWORLD, nbody) post-forward body pos
    xmat:       wp.array(dtype=wp.mat33, ndim=2),      # type: ignore (NWORLD, nbody) post-forward body rot
    body_id:    int,
    cam:        wp.array(dtype=wp.vec3),               # type: ignore (NWORLD,) camera world pos
    p_world:    wp.array(dtype=wp.vec3, ndim=2),       # type: ignore (NWORLD, N) out — world-frame PCD
    dist:       wp.array(dtype=float, ndim=2),         # type: ignore (NWORLD, N) out — ‖p−cam‖
    maxd:       wp.array(dtype=float),                 # type: ignore (NWORLD,)  out — per-world max dist
):
    """Path (a): body-local PCD → world frame + camera distance (+ per-world max)."""
    w, i = wp.tid()
    v  = assignment[w]
    pw = xmat[w, body_id] * pcd_local[v, i] + xpos[w, body_id]
    p_world[w, i] = pw
    dn = wp.max(wp.length(pw - cam[w]), 1.0e-6)
    dist[w, i] = dn
    wp.atomic_max(maxd, w, dn)


@wp.kernel
def _pv_dist_kernel(
    points: wp.array(dtype=wp.vec3, ndim=2),           # type: ignore (NWORLD, N) external world-frame PCD
    cam:    wp.array(dtype=wp.vec3),                   # type: ignore (NWORLD,)
    dist:   wp.array(dtype=float, ndim=2),             # type: ignore (NWORLD, N) out
    maxd:   wp.array(dtype=float),                     # type: ignore (NWORLD,)  out
):
    """Path (b): dist + maxd only, from a PCD buffer already in world frame (no transform)."""
    w, i = wp.tid()
    dn = wp.max(wp.length(points[w, i] - cam[w]), 1.0e-6)
    dist[w, i] = dn
    wp.atomic_max(maxd, w, dn)


@wp.kernel
def _pv_scatter_kernel(
    points:       wp.array(dtype=wp.vec3, ndim=2),     # type: ignore (NWORLD, N) world-frame PCD
    cam:          wp.array(dtype=wp.vec3),             # type: ignore (NWORLD,)
    dist:         wp.array(dtype=float, ndim=2),       # type: ignore (NWORLD, N)
    maxd:         wp.array(dtype=float),               # type: ignore (NWORLD,)
    radius_scale: float,
    grid_res:     int,
    grid:         wp.array(dtype=float, ndim=2),       # type: ignore (NWORLD, G*G) out — depth buffer
    fnorm:        wp.array(dtype=float, ndim=2),       # type: ignore (NWORLD, N)   out — flipped norm
    ucell:        wp.array(dtype=int, ndim=2),         # type: ignore (NWORLD, N)   out — azimuth cell
    vcell:        wp.array(dtype=int, ndim=2),         # type: ignore (NWORLD, N)   out — polar cell
):
    """Spherical flip (closed form) + (θ,φ) grid cell + scatter-max depth buffer."""
    w, i = wp.tid()
    pc = points[w, i] - cam[w]
    n  = dist[w, i]
    fn = 2.0 * (maxd[w] * radius_scale) - n            # ‖p_flipped‖ = 2R − n
    fnorm[w, i] = fn

    theta = wp.atan2(pc[1], pc[0])
    phi   = wp.acos(wp.clamp(pc[2] / n, -1.0, 1.0))
    u = wp.int32((theta + _PV_PI) / (2.0 * _PV_PI) * float(grid_res))
    v = wp.int32(phi / _PV_PI * float(grid_res))
    u = wp.min(wp.max(u, 0), grid_res - 1)
    v = wp.min(wp.max(v, 0), grid_res - 1)
    ucell[w, i] = u
    vcell[w, i] = v
    wp.atomic_max(grid, w, v * grid_res + u, fn)


@wp.kernel
def _pv_visible_kernel(
    grid:         wp.array(dtype=float, ndim=2),       # type: ignore (NWORLD, G*G)
    ucell:        wp.array(dtype=int, ndim=2),         # type: ignore (NWORLD, N)
    vcell:        wp.array(dtype=int, ndim=2),         # type: ignore (NWORLD, N)
    fnorm:        wp.array(dtype=float, ndim=2),       # type: ignore (NWORLD, N)
    maxd:         wp.array(dtype=float),               # type: ignore (NWORLD,)
    radius_scale: float,
    tol_scale:    float,
    grid_res:     int,
    visible:      wp.array(dtype=int, ndim=2),         # type: ignore (NWORLD, N) out — 1=visible
):
    """Cross (4-neighbour) dilation performed at read time → no separate dilation pass.

    Neighbours outside the grid are ignored, matching ``np.pad(constant=-1)`` in
    the numpy reference (out of bounds = −1, so they do not contribute to the max)."""
    w, i = wp.tid()
    u = ucell[w, i]
    v = vcell[w, i]
    c = v * grid_res + u
    m = grid[w, c]
    if v > 0:
        m = wp.max(m, grid[w, c - grid_res])
    if v < grid_res - 1:
        m = wp.max(m, grid[w, c + grid_res])
    if u > 0:
        m = wp.max(m, grid[w, c - 1])
    if u < grid_res - 1:
        m = wp.max(m, grid[w, c + 1])

    tol = maxd[w] * radius_scale * tol_scale
    vis = 0
    if fnorm[w, i] >= m - tol:
        vis = 1
    visible[w, i] = vis


@wp.kernel
def _pv_resample_camera_masked_kernel(
    seed_arr:     wp.array(dtype=int),                 # type: ignore (1,) shared GPU seed
    seed_offset:  int,
    xpos:         wp.array(dtype=wp.vec3, ndim=2),     # type: ignore (NWORLD, nbody)
    obj_body_id:  int,                                  # object centre approximation = obj root body xpos
    box_low:      wp.array(dtype=wp.vec3),             # type: ignore (K,) candidate box low corner
    box_high:     wp.array(dtype=wp.vec3),             # type: ignore (K,) candidate box high corner
    n_boxes:      int,
    cam_dist:     float,
    cam_noise_std: float,                               # Gaussian noise on camera position (m)
    cam:          wp.array(dtype=wp.vec3),             # type: ignore (NWORLD,) out
    apply_mask:   wp.array(dtype=int),                 # type: ignore (NWORLD,) 1=resample
    apply_to_all: int,
):
    """GPU masked version of the reference camera resampling (for per-world reset, sync-free).

    Pick one of the K candidate boxes, sample a position inside it → keep **only
    the direction**, and place the camera ``cam_dist`` away from the object centre
    (``xpos[w, obj_body_id]``, an approximation of the PCD mean). With
    ``apply_to_all == 0`` only worlds with ``apply_mask[w]==1`` are updated (new
    camera only for done-reset worlds). With ``cam_noise_std > 0`` isotropic
    Gaussian noise is added to the final camera position (mimics sensor
    calibration error)."""
    w = wp.tid()
    if apply_to_all == 0 and apply_mask[w] == 0:
        return
    state = wp.rand_init(seed_arr[0] + seed_offset, w)
    k  = wp.randi(state, 0, n_boxes)
    lo = box_low[k]
    hi = box_high[k]
    pos = wp.vec3(
        lo[0] + wp.randf(state) * (hi[0] - lo[0]),
        lo[1] + wp.randf(state) * (hi[1] - lo[1]),
        lo[2] + wp.randf(state) * (hi[2] - lo[2]),
    )
    center = xpos[w, obj_body_id]
    v = pos - center
    v = v / (wp.length(v) + 1.0e-6)
    c = center + v * cam_dist
    if cam_noise_std > 0.0:
        c = c + wp.vec3(wp.randn(state) * cam_noise_std,
                        wp.randn(state) * cam_noise_std,
                        wp.randn(state) * cam_noise_std)
    cam[w] = c


# ══════════════════════════════════════════════════════════════════════════
# Frozen initial-view capture (student sees only a single initial partial view)
# ══════════════════════════════════════════════════════════════════════════
@wp.kernel
def _pv_capture_copy_kernel(
    points:        wp.array(dtype=wp.vec3, ndim=2),    # type: ignore (NWORLD, N) live world-frame PCD
    mask:          wp.array(dtype=int),                # type: ignore (NWORLD,) 1=capture
    apply_to_all:  int,
    seed_arr:      wp.array(dtype=int),                # type: ignore (1,) shared GPU seed
    seed_offset:   int,
    n_point:       int,
    pos_noise_std: float,                              # Gaussian jitter on point position (m)
    dropout_prob:  float,                              # point dropout probability
    frozen_out:    wp.array(dtype=wp.vec3, ndim=2),    # type: ignore (NWORLD, N) out — frozen snapshot
    keep_out:      wp.array(dtype=int, ndim=2),        # type: ignore (NWORLD, N) out — 1=kept (survived dropout)
):
    """Copy the current PCD of flagged worlds into the frozen buffer (**freeze**), with sensor noise.

    The student sees only this single snapshot for the whole episode (not
    updated even when the object moves). The noise realisation (jitter offset +
    dropout keep-mask) is also fixed at capture time and constant over the
    episode, equivalent to observing a single real depth-camera frame."""
    w, i = wp.tid()
    if apply_to_all == 0 and mask[w] == 0:
        return
    state = wp.rand_init(seed_arr[0] + seed_offset, w * n_point + i)
    p = points[w, i]
    if pos_noise_std > 0.0:
        p = p + wp.vec3(wp.randn(state) * pos_noise_std,
                        wp.randn(state) * pos_noise_std,
                        wp.randn(state) * pos_noise_std)
    frozen_out[w, i] = p
    keep = int(1)
    if dropout_prob > 0.0:
        if wp.randf(state) < dropout_prob:
            keep = int(0)
    keep_out[w, i] = keep


@wp.kernel
def _pv_and_keep_kernel(
    keep:    wp.array(dtype=int, ndim=2),              # type: ignore (NWORLD, N) frozen dropout mask
    visible: wp.array(dtype=int, ndim=2),              # type: ignore (NWORLD, N) in/out
):
    """AND the frozen dropout keep-mask into HPR visibility so dropped points are invisible."""
    w, i = wp.tid()
    if keep[w, i] == 0:
        visible[w, i] = 0


# ══════════════════════════════════════════════════════════════════════════
# WarpPartialViewPCD
# ══════════════════════════════════════════════════════════════════════════
class WarpPartialViewPCD:
    """Partial-view (HPR) PCD computer for parallel warp envs; handles one object (body).

    For the object PCD of every world it computes the per-world camera
    visibility mask with 4 kernel launches. ``update*()`` works in place on
    preallocated buffers, so it is safe inside CUDA-graph capture.

    Two input modes:
      (a) **body-local** — given ``pcd_local_per_variant``, ``update(xpos, xmat)``
          also performs the transform (notebook viewer path).
      (b) **world-frame** — construct with ``pcd_local_per_variant=None`` +
          ``n_point=N`` and call ``update_from_world_pcd(points_wp)`` to consume
          a ``(NWORLD, N)`` vec3 buffer already in world frame (e.g. the base
          task's ``transformed_pcd_wp``) directly (RL obs path, no extra transform).

    Generality:
      - With several object slots, use one instance per slot (body).
      - PCDs may differ per variant (``assignment`` maps world → variant).
      - ``grid_res / radius_scale / tol_scale / cam_boxes / cam_dist`` are all knobs.

    Args:
        pcd_local_per_variant: (V, N, 3) body-local PCD per variant
                               (``variant_pcd_cache`` stack; N must be equal).
                               ``None`` → world-frame mode (``n_point`` required).
        assignment:            (NWORLD,) world → variant index.
                               In world-frame mode used only to determine the
                               world count and camera buffer size.
        body_id:               MuJoCo body id of the target object (shared across
                               variants); used for the (a) transform and as the
                               GPU camera-resampling centre.
        n_point:               per-world point count N in world-frame mode.
        grid_res:              spherical grid resolution (default 120, as in the reference).
        radius_scale:          flip radius = maxd × radius_scale (default 100).
        tol_scale:             visibility tolerance = radius × tol_scale (default 1e-3).
        cam_boxes:             (K, 3, 2) camera candidate boxes (default: left/right, as in the reference).
        cam_dist:              final camera = object centre + direction × cam_dist (m).
        seed:                  seed for GPU camera resampling (default 0).
        device:                warp device (None → current default device).
    """

    # Left/right candidate boxes of the reference ``sample_partial_view_camera`` (x/y/z × [low, high])
    DEFAULT_CAM_BOXES = np.array([
        [[0.6, 1.4], [ 0.5,  0.5], [0.4, 0.8]],
        [[0.6, 1.4], [-0.5, -0.5], [0.4, 0.8]],
    ])

    def __init__(
        self,
        pcd_local_per_variant,
        assignment,
        body_id: int,
        n_point: int | None = None,
        grid_res: int = 120,
        radius_scale: float = 100.0,
        tol_scale: float = 1e-3,
        cam_boxes=None,
        cam_dist: float = 0.3,
        pos_noise_std: float = 0.0,
        dropout_prob: float = 0.0,
        cam_noise_std: float = 0.0,
        seed: int = 0,
        device=None,
    ):
        self.nworld        = len(assignment)
        self.body_id       = int(body_id)
        self.grid_res      = int(grid_res)
        self.radius_scale  = float(radius_scale)
        self.tol_scale     = float(tol_scale)
        self.cam_boxes     = np.asarray(
            self.DEFAULT_CAM_BOXES if cam_boxes is None else cam_boxes, dtype=np.float64)
        self.cam_dist      = float(cam_dist)
        # ── sensor noise (all fixed at capture time = fixed noise of one initial depth frame) ──
        self.pos_noise_std = float(pos_noise_std)   # Gaussian jitter on point position (m)
        self.dropout_prob  = float(dropout_prob)    # point dropout probability
        self.cam_noise_std = float(cam_noise_std)   # Gaussian noise on camera position (m)

        # ── input mode selection ─────────────────────────────────────────
        if pcd_local_per_variant is not None:
            pcd = np.asarray(pcd_local_per_variant, dtype=np.float32)
            assert pcd.ndim == 3 and pcd.shape[-1] == 3, \
                f"pcd_local_per_variant must be (V, N, 3), got {pcd.shape}"
            self.n_variant, self.n_point = pcd.shape[0], pcd.shape[1]
            self.pcd_local = wp.array(pcd, dtype=wp.vec3, device=device)          # (V, N)
            self.device    = self.pcd_local.device
        else:
            assert n_point is not None, "world-frame mode requires n_point"
            self.n_variant, self.n_point = 0, int(n_point)
            self.pcd_local = None
            self.device    = wp.get_device(device)

        NW, N, G = self.nworld, self.n_point, self.grid_res
        dev = self.device
        self.assignment = wp.array(np.asarray(assignment, dtype=np.int32), dtype=int, device=dev)
        self.cam        = wp.zeros(NW,       dtype=wp.vec3, device=dev)            # (NW,)
        self.p_world    = wp.zeros((NW, N),  dtype=wp.vec3, device=dev)            # (NW, N) — path (a) output
        self.dist       = wp.zeros((NW, N),  dtype=float,   device=dev)
        self.fnorm      = wp.zeros((NW, N),  dtype=float,   device=dev)
        self.ucell      = wp.zeros((NW, N),  dtype=int,     device=dev)
        self.vcell      = wp.zeros((NW, N),  dtype=int,     device=dev)
        self.maxd       = wp.zeros(NW,       dtype=float,   device=dev)
        self.grid       = wp.zeros((NW, G * G), dtype=float, device=dev)
        self.visible    = wp.zeros((NW, N),  dtype=int,     device=dev)            # 1 = visible

        # ── Frozen initial-view buffers (student sees a single initial partial view) ──
        # ``frozen_pcd`` = world-frame PCD snapshot at capture time (+jitter), constant over the episode.
        # ``keep_mask``  = dropout result at capture time (1=kept), constant over the episode.
        self.frozen_pcd = wp.zeros((NW, N), dtype=wp.vec3, device=dev)
        self.keep_mask  = wp.zeros((NW, N), dtype=int,     device=dev)

        # World-frame point buffer used by the last update (for the host resample centre)
        self._last_points = self.p_world

        # GPU camera resampling: candidate box corners + shared seed buffer
        # (WarpPerWorldCondition._gpu_seed_wp idiom: the host bumps it before every launch)
        self._box_low_wp  = wp.array(
            np.ascontiguousarray(self.cam_boxes[:, :, 0], dtype=np.float32),
            dtype=wp.vec3, device=dev)
        self._box_high_wp = wp.array(
            np.ascontiguousarray(self.cam_boxes[:, :, 1], dtype=np.float32),
            dtype=wp.vec3, device=dev)
        self._gpu_seed_value = int(seed)
        self._gpu_seed_wp    = wp.array(
            np.array([self._gpu_seed_value], dtype=np.int32), dtype=int, device=dev)
        self._dummy_mask_wp  = wp.zeros((NW,), dtype=int, device=dev)

    # ── Core: 4-kernel HPR pipeline (graph-capturable, no CPU sync) ────────
    def update(self, xpos, xmat):
        """Path (a): transform the body-local stack and run HPR. xpos/xmat: ``d.xpos``/``d.xmat``."""
        assert self.pcd_local is not None, \
            "update() called without body-local PCD; use update_from_world_pcd() instead"
        NW, N = self.nworld, self.n_point
        self._reset_buffers()
        wp.launch(
            _pv_transform_kernel, dim=(NW, N), device=self.device,
            inputs=[self.pcd_local, self.assignment, xpos, xmat, self.body_id,
                    self.cam, self.p_world, self.dist, self.maxd],
        )
        self._last_points = self.p_world
        self._scatter_and_visible(self.p_world)

    def update_from_world_pcd(self, points):
        """Path (b): HPR on a ``(NWORLD, N)`` vec3 buffer already in world frame.

        Consumes a buffer that another pipeline updates every step, such as the
        base task's ``transformed_pcd_wp``, as-is (no transform re-run)."""
        assert points.shape == (self.nworld, self.n_point), \
            f"points shape {points.shape} != ({self.nworld}, {self.n_point})"
        self._reset_buffers()
        wp.launch(
            _pv_dist_kernel, dim=(self.nworld, self.n_point), device=self.device,
            inputs=[points, self.cam, self.dist, self.maxd],
        )
        self._last_points = points
        self._scatter_and_visible(points)

    def update_from_data(self, d):
        """Run path (a) directly from mjwarp Data (post-forward xpos/xmat, zero-copy)."""
        self.update(d.xpos, d.xmat)

    def _reset_buffers(self):
        NW, G = self.nworld, self.grid_res
        wp.launch(_pv_reset_maxd_kernel, dim=NW,          inputs=[self.maxd], device=self.device)
        wp.launch(_pv_reset_grid_kernel, dim=(NW, G * G), inputs=[self.grid], device=self.device)

    def _scatter_and_visible(self, points):
        NW, N, G = self.nworld, self.n_point, self.grid_res
        wp.launch(
            _pv_scatter_kernel, dim=(NW, N), device=self.device,
            inputs=[points, self.cam, self.dist, self.maxd,
                    self.radius_scale, G, self.grid, self.fnorm, self.ucell, self.vcell],
        )
        wp.launch(
            _pv_visible_kernel, dim=(NW, N), device=self.device,
            inputs=[self.grid, self.ucell, self.vcell, self.fnorm, self.maxd,
                    self.radius_scale, self.tol_scale, G, self.visible],
        )

    # ── Camera sampling — host (notebook, event-driven) ────────────────────
    def resample_cameras(self, rng=None):
        """Resample the cameras of all worlds following the reference scheme (host numpy).

        Draw a position from a candidate box, keep **only its direction**, and
        place the final camera ``cam_dist`` away from each world's object PCD centre.
        The centre is the mean of the point buffer used by the last ``update*()``,
        so call this **only after at least one update**. (Intended for key-press /
        reset events, not per-frame cost. Use ``resample_cameras_gpu`` in the RL
        training loop.)
        """
        rand = np.random if rng is None else rng
        centers = self._last_points.numpy().mean(axis=1)                 # (NW, 3)
        boxes   = self.cam_boxes[rand.randint(0, len(self.cam_boxes), size=self.nworld)]
        low, high = boxes[:, :, 0], boxes[:, :, 1]
        pos     = low + rand.uniform(0.0, 1.0, size=(self.nworld, 3)) * (high - low)
        vec     = pos - centers
        vec    /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6)
        cams    = centers + vec * self.cam_dist
        self.set_cameras(cams)
        return cams

    # ── Camera sampling — GPU masked (RL per-world reset, sync-free) ───────
    def resample_cameras_gpu(self, xpos, mask_wp=None):
        """Resample cameras on the GPU for done-reset worlds only (or all worlds).

        Args:
            xpos:    ``d.xpos``; object centre = ``xpos[w, body_id]`` (approximates the PCD mean).
            mask_wp: ``(NWORLD,)`` int wp.array, 1=resample (e.g. ``done_mask_wp``).
                     ``None`` → resample all worlds (full episode reset).
        """
        self._gpu_seed_value += 1
        self._gpu_seed_wp.assign(np.array([self._gpu_seed_value], dtype=np.int32))
        apply_to_all = 1 if mask_wp is None else 0
        wp.launch(
            _pv_resample_camera_masked_kernel, dim=self.nworld, device=self.device,
            inputs=[
                self._gpu_seed_wp, 0,
                xpos, self.body_id,
                self._box_low_wp, self._box_high_wp, int(len(self.cam_boxes)),
                self.cam_dist, self.cam_noise_std, self.cam,
                self._dummy_mask_wp if mask_wp is None else mask_wp,
                apply_to_all,
            ],
        )

    # ── Frozen initial-view capture (student obs path) ─────────────────────
    def capture_masked(self, points, mask_wp=None):
        """Freeze the current PCD of flagged worlds into the ``frozen_pcd`` snapshot (+ sensor noise).

        The student runs the whole episode from this single snapshot (not
        updated even when the object moves). The jitter offset and dropout
        keep-mask are drawn here too and stay fixed for the episode = a single
        initial frame of a real depth camera. Called at the same time as the
        per-world reset; ``mask_wp`` limits re-freezing to the reset worlds
        (no CPU sync).

        Args:
            points:  ``(NWORLD, N)`` world-frame PCD (base task ``transformed_pcd_wp``).
            mask_wp: ``(NWORLD,)`` int, 1=capture. ``None`` → all worlds (full reset).
        """
        assert points.shape == (self.nworld, self.n_point), \
            f"points shape {points.shape} != ({self.nworld}, {self.n_point})"
        self._gpu_seed_value += 1
        self._gpu_seed_wp.assign(np.array([self._gpu_seed_value], dtype=np.int32))
        apply_to_all = 1 if mask_wp is None else 0
        wp.launch(
            _pv_capture_copy_kernel, dim=(self.nworld, self.n_point), device=self.device,
            inputs=[
                points,
                self._dummy_mask_wp if mask_wp is None else mask_wp,
                apply_to_all,
                self._gpu_seed_wp, 0,
                int(self.n_point),
                self.pos_noise_std, self.dropout_prob,
                self.frozen_pcd, self.keep_mask,
            ],
        )

    def refresh_from_frozen(self):
        """Recompute HPR visibility from the frozen ``frozen_pcd`` and AND the dropout keep-mask.

        frozen_pcd and cam are constant over the episode, so the result
        (``visible``) is constant too: calling this every obs step
        deterministically reproduces the same initial partial view. (Even if
        the object moves and the real PCD changes, the student only sees the
        initial snapshot.)"""
        self.update_from_world_pcd(self.frozen_pcd)
        wp.launch(
            _pv_and_keep_kernel, dim=(self.nworld, self.n_point), device=self.device,
            inputs=[self.keep_mask, self.visible],
        )

    def set_cameras(self, cams_np):
        """Upload (NWORLD, 3) camera world positions to the GPU buffer."""
        self.cam.assign(np.ascontiguousarray(cams_np, dtype=np.float32))

    # ── Host-side conveniences (for viz; RL obs should use the wp buffers directly) ──
    def visible_np(self):
        return self.visible.numpy().astype(bool)                          # (NW, N)

    def p_world_np(self):
        return self.p_world.numpy()                                       # (NW, N, 3)

    def cam_np(self):
        return self.cam.numpy()                                           # (NW, 3)
