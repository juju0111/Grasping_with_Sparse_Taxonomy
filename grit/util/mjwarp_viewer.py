import time
import mujoco
import mujoco_warp as mjwarp
import warp as wp
import glfw
import pathlib

import numpy as np

from grit.util.sim_core.viewer import MuJoCoMinimalViewer

class MJWarpMinimalViewer(MuJoCoMinimalViewer):
    """
    Viewer for MuJoCo Warp parallel environments.

    Extends MuJoCoMinimalViewer with parallel_render(), which renders N warp
    worlds in a grid layout.  World-index labels, success/failure indicators,
    and arbitrary debug markers (via add_marker()) are all supported.
    """
    def __init__(
        self,
        model,
        data,
        mode              = 'window',
        title             = "MuJoCo Warp Minimal Viewer",
        width             = None,
        height            = None,
        maxgeom           = 50000,
        perturbation      = True,
        x_offset          = None,
        y_offset          = None,
    ):
        # sim_core reordered MuJoCoMinimalViewer.__init__ params
        # (x_offset/y_offset now precede maxgeom/perturbation) — pass by
        # keyword so this stays correct regardless of positional order.
        super().__init__(
            model,
            data,
            mode         = mode,
            title        = title,
            width        = width,
            height       = height,
            maxgeom      = maxgeom,
            perturbation = perturbation,
            x_offset     = x_offset,
            y_offset     = y_offset,
        )

        # cached per-frame state for parallel rendering
        self._offsets_for_parallel_render = None
        self._datas_for_parallel_render   = None
        self._target_env_for_parallel_render = None
        self._visual_geom_offsets         = None

    def set_contactpoint(self, contactpoint=True):
        """
        Set the contact point visibility.
        """
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contactpoint # type: ignore

    def set_shadow(self, shadow: bool = True):
        """
        Toggle shadow rendering for the current scene.
        Backed by MjvScene.flags[mjRND_SHADOW].
        """
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadow)  # type: ignore

    def _create_overlay(self):
        """Empty override — overlays are added externally via add_overlay()."""
        pass

    @staticmethod
    def compute_world_offsets(n_envs: int, spacing: float = 0.5,
                              cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
        """Return (n_envs, 2) XY grid offsets using the same layout as parallel_render."""
        positions = []
        gs        = int((n_envs - 1) ** 0.5) + 1
        half      = gs // 2
        done      = False
        for i in range(gs):
            for j in range(gs):
                positions.append((cx + (i - half) * spacing, cy + (j - half) * spacing))
                if len(positions) == n_envs:
                    done = True
                    break
            if done:
                break
        col_len  = min(gs, (n_envs + gs - 1) // gs)
        mid_idx  = (col_len // 2) * gs
        if mid_idx < len(positions):
            positions[0], positions[mid_idx] = positions[mid_idx], positions[0]
        return np.array(positions)  # (n_envs, 2)

    # ------------------------------------------------------------------
    # Parallel rendering
    # ------------------------------------------------------------------
    def parallel_render(self,
                        mjwarp_data,
                        offset_body_names: list,
                        viewer_model: list,
                        mjwarp_model,
                        target_model          = None,
                        target_data           = None,
                        offset                = 0.5,
                        label_z               = 3.0,
                        plot_success          = None,
                        plot_env_number       = True,
                        record                = False,
                        assignment            = None,
                        contact_marker_radius = 0.0015,
                        per_world_geom_rgba   = None,
                        target_geom_rgba      = None,
                        **kwargs,
    ):
        """
        Render all warp worlds side-by-side in a grid.

        Args:
            mjwarp_data:        MuJoCo Warp Data object (d).
            offset_body_names:  Per-sub_env list of body names to offset in XY grid.
            viewer_model:       List of mujoco.MjModel (one per sub_env).
            mjwarp_model:       MuJoCo Warp Model (m).
            target_model:       Optional MjModel used to draw a per-world target
                                pose overlay (e.g. a ghost hand at the desired
                                wrist pose / qpos). Provide ``target_data`` —
                                one ``MjData`` per world, pre-filled and
                                ``mj_forward``-ed by the caller — to inject
                                the geoms into the scene.
            target_data:        ``[MjData]`` of length ``≥ n_envs`` — see
                                ``SingleHandSubEnv.set_ghost_targets`` for the
                                fill helper.
            offset:             XY grid spacing between worlds.
            label_z:            Z height for per-world label/indicator spheres.
            plot_success:       Array-like of bool (length n_envs), or None.
                                True → green sphere, False → red sphere.
                                None → invisible anchor (label only).
            plot_env_number:    Annotate each world with its index label.
            record:             (reserved) capture frames.
            per_world_geom_rgba: Optional ``(n_envs, ngeom, 4)`` array — full
                                RGBA override applied to each world's
                                ``viewer_model[v].geom_rgba`` *before* it is
                                added to the scene. Lets two worlds sharing
                                the same variant render with **different**
                                colours (e.g. per-world touch-sensor heat).
                                Disabled padding slots still get alpha→0.
            target_geom_rgba:   Optional RGBA override for the ``target_model``
                                ghost geoms. Either ``(ngeom, 4)`` (same colour
                                in every world) or ``(n_envs, ngeom, 4)`` (a
                                distinct colour per world — e.g. tint each
                                world's ghost fingers by its sampled taxonomy
                                active/rest mask). Applied to
                                ``target_model.geom_rgba`` before each world's
                                ``mjv_addGeoms``; the model's original rgba is
                                restored afterwards.
        """
        def _grid_positions(cx, cy, n, spacing):
            positions = []
            gs        = int((n - 1) ** 0.5) + 1
            half      = gs // 2
            done      = False
            for i in range(gs):
                for j in range(gs):
                    positions.append((cx + (i - half) * spacing,
                                      cy + (j - half) * spacing))
                    if len(positions) == n:
                        done = True
                        break
                if done:
                    break
            col_len  = min(gs, (n + gs - 1) // gs)
            mid_idx  = (col_len // 2) * gs
            if mid_idx < len(positions):
                positions[0], positions[mid_idx] = positions[mid_idx], positions[0]
            return positions, gs

        # ---- extract GPU data to numpy once per render call ----
        qpos_np       = mjwarp_data.qpos.numpy()        # (nworld, nq)
        mocap_np      = mjwarp_data.mocap_pos.numpy()   # (nworld, nmocap, 3)
        mocap_quat_np = mjwarp_data.mocap_quat.numpy()  # (nworld, nmocap, 4)
        n_envs    = qpos_np.shape[0]
        n_sub_env = len(viewer_model)

        # build model index array: assignment[i] → which variant model world i uses.
        # must match how m.geom_dataid / geom_size / geom_pos were filled in Cell [19].
        if assignment is not None:
            _model_idx_arr = [int(assignment[i]) for i in range(n_envs)]
        else:
            _model_idx_arr = [int(i % n_sub_env) for i in range(n_envs)]

        # ---- lazy-initialise cached objects ----
        if self._offsets_for_parallel_render is None or \
                n_envs > len(self._offsets_for_parallel_render):
            self._offsets_for_parallel_render, self.grid_size = \
                _grid_positions(0.0, 0.0, n_envs, offset)
            self._visual_geom_offsets = \
                np.array(self._offsets_for_parallel_render)[:, np.newaxis, :]

        if self._datas_for_parallel_render is None or \
                n_envs > len(self._datas_for_parallel_render):
            self._datas_for_parallel_render = [
                mujoco._structs.MjData(viewer_model[int(i % int(n_envs) % n_sub_env)])
                for i in range(n_envs)
            ]

        if not self.is_alive:
            raise RuntimeError("GLFW window does not exist but render() was called.")
        if glfw.window_should_close(self.window):
            self.close()
            return

        # ---- per-world mesh selection setup (done once per render call) ----
        geom_dataid_raw  = mjwarp_model.geom_dataid.numpy()
        per_world_dataid = geom_dataid_raw.ndim == 2
        geom_group_raw   = mjwarp_model.geom_group.numpy()
        per_world_group  = geom_group_raw.ndim == 2
        base_rgba = [vm.geom_rgba.copy() for vm in viewer_model]

        # ----------------------------------------------------------------
        def _update_frame():
            self._create_overlay()
            render_start = time.time()
            width, height = glfw.get_framebuffer_size(self.window)
            self.viewport.width, self.viewport.height = width, height

            # World 0's mjv_updateScene call would also auto-draw CPU-side contact
            # points (from mj_forward run on a single CPU MjData), which (a) only
            # appear in world 0 and (b) don't match the GPU contacts of the actual
            # parallel sim. Temporarily disable the flag here so all worlds get
            # contact points drawn from the authoritative GPU buffer below.
            _cp_flag = mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
            _cp_flag_was_on = bool(self.vopt.flags[_cp_flag])
            if _cp_flag_was_on:
                self.vopt.flags[_cp_flag] = 0

            # ---- render each world ----
            # Optional per-world full-RGBA override (lets sibling worlds of the
            # same variant render with different colours, e.g. per-world touch
            # heat). Cached as ndarray once per render call.
            pwgr = (np.asarray(per_world_geom_rgba)
                    if per_world_geom_rgba is not None else None)

            for i in range(n_envs):
                model_idx = _model_idx_arr[i]

                # per-world mesh variant + padding transparency
                disabled_i = None
                if per_world_dataid:
                    dataid_i = geom_dataid_raw[i]
                    # group==<0 alone is wrong: primitive geoms (floor, plane, box)
                    # also h4 marks disabled padding slots (convention from Cell [19]).
                    # dataidave dataid=-1 but must stay visible.
                    if per_world_group:
                        disabled_i = geom_group_raw[i] == 4
                    else:
                        disabled_i = dataid_i < 0
                    viewer_model[model_idx].geom_dataid[:] = \
                        np.where(disabled_i, 0, dataid_i)
                    viewer_model[model_idx].geom_rgba[:, 3] = np.where(
                        disabled_i, 0.0, base_rgba[model_idx][:, 3])

                # per-world full-RGBA override (after alpha-mask logic above so
                # we keep its disabled-padding behaviour while letting the user
                # paint the visible geoms freely).
                if pwgr is not None:
                    viewer_model[model_idx].geom_rgba[:] = pwgr[i]
                    if disabled_i is not None:
                        viewer_model[model_idx].geom_rgba[disabled_i, 3] = 0.0

                # compute qpos indices for XY offset bodies
                idxs = []
                for b in offset_body_names:
                    jntadr  = int(viewer_model[model_idx].body(b).jntadr[0])
                    qposadr = int(viewer_model[model_idx].joint(jntadr).qposadr[0])
                    idxs.append(qposadr)
                idxs = np.array(idxs, dtype=np.intp)

                offset_xy = self._offsets_for_parallel_render[i]
                data      = self._datas_for_parallel_render[i]

                data.qpos[:] = qpos_np[i]
                if mocap_np.shape[1] > 0:
                    data.mocap_pos[:] = mocap_np[i]
                    data.mocap_pos[:, 0] += offset_xy[0]
                    data.mocap_pos[:, 1] += offset_xy[1]
                    data.mocap_quat[:] = mocap_quat_np[i]
                data.qpos[idxs]     += offset_xy[0]
                data.qpos[idxs + 1] += offset_xy[1]

                mujoco.mj_forward(viewer_model[model_idx], data)

                if i == 0:
                    mujoco.mjv_updateScene(
                        viewer_model[model_idx], data, self.vopt,
                        None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn,
                    )
                else:
                    mujoco.mjv_addGeoms(
                        viewer_model[model_idx], data, self.vopt,
                        mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, self.scn,
                    )

            # restore rgba after per-world masking
            for midx, vm in enumerate(viewer_model):
                vm.geom_rgba[:] = base_rgba[midx]

            # ---- per-world target overlay (e.g. ghost target hand) ----
            # Caller pre-fills `target_data[i]` (one MjData per world built
            # from `target_model`) with the per-world target qpos + grid offset
            # and runs `mj_forward` ahead of time — see
            # `SingleHandSubEnv.set_ghost_targets()`. Here we only inject the
            # geoms into the scene. `mjCAT_DYNAMIC` strips out the static
            # floor that ships with `parent_spec`, so only the hand subtree
            # actually renders.
            if target_model is not None and target_data is not None:
                _n_target = min(n_envs, len(target_data))
                # Optional ghost-geom RGBA override. (ngeom,4) → same colour in
                # every world; (n_envs,ngeom,4) → per-world (mjv_addGeoms reads
                # model.geom_rgba at call time, so we swap it per world and
                # restore the model's original rgba once the loop is done).
                _tgr = (np.asarray(target_geom_rgba, dtype=np.float32)
                        if target_geom_rgba is not None else None)
                _tgt_base_rgba = (target_model.geom_rgba.copy()
                                  if _tgr is not None else None)
                for i in range(_n_target):
                    if _tgr is not None:
                        target_model.geom_rgba[:] = _tgr[i] if _tgr.ndim == 3 else _tgr
                    mujoco.mjv_addGeoms(
                        target_model, target_data[i], self.vopt,
                        mujoco.MjvPerturb(),
                        mujoco.mjtCatBit.mjCAT_DYNAMIC, self.scn,
                    )
                if _tgt_base_rgba is not None:
                    target_model.geom_rgba[:] = _tgt_base_rgba

            # ---- per-world contact points (drawn from GPU contact buffer) ----
            # mjwarp keeps a single flat contact array shared across all worlds;
            # contact.worldid[k] tells which world each one belongs to. We restore
            # the user's flag first so the rest of the system still observes the
            # caller's intent.
            if _cp_flag_was_on:
                self.vopt.flags[_cp_flag] = 1
                n_active = int(mjwarp_data.nacon.numpy()[0])
                if n_active > 0:
                    c_worldid = mjwarp_data.contact.worldid.numpy()[:n_active]
                    c_pos     = mjwarp_data.contact.pos.numpy()[:n_active]
                    c_dist    = mjwarp_data.contact.dist.numpy()[:n_active]
                    # Note: model.vis.scale.contactwidth is a render-scale multiplier
                    # of the model extent, not a metric size — for hand-sized scenes
                    # it works out to a several-cm sphere, occluding the object. Use
                    # a metric radius (default 1.5 mm) instead so the dot is visible
                    # without hiding the geometry.
                    r = float(max(contact_marker_radius, 2e-4))
                    rgba = [float(x) for x in viewer_model[0].vis.rgba.contactpoint]
                    size_sphere = [r, r, r]
                    for k in range(n_active):
                        if c_dist[k] > 0:        # not actually penetrating
                            continue
                        wid = int(c_worldid[k])
                        if wid < 0 or wid >= n_envs:
                            continue
                        ox, oy = self._offsets_for_parallel_render[wid]
                        self._add_marker_to_scene({
                            'pos':   [float(c_pos[k, 0] + ox),
                                      float(c_pos[k, 1] + oy),
                                      float(c_pos[k, 2])],
                            'type':  mujoco.mjtGeom.mjGEOM_SPHERE,
                            'size':  size_sphere,
                            'rgba':  rgba,
                            'label': '',
                        })

            # ---- flush externally-added markers (add_marker() API) ----
            for marker in self._markers:
                self._add_marker_to_scene(marker)

            # ---- per-world indicator spheres + labels ----
            for i in range(n_envs):
                offset_xy = self._offsets_for_parallel_render[i]
                label_str = str(i) if plot_env_number else ''

                if plot_success is not None:
                    # default RED ball above every world; turns GREEN on success.
                    rgba = [0.0, 1.0, 0.0, 1.0] if plot_success[i] \
                           else [1.0, 0.0, 0.0, 1.0]
                    size = [0.04, 0.04, 0.04]     # bigger so the state pops out
                else:
                    rgba = [1.0, 1.0, 1.0, 0.0]   # invisible – label anchor only
                    size = [0.02,  0.02,  0.02]

                self._add_marker_to_scene({
                    'pos':   [offset_xy[0], offset_xy[1], label_z],
                    'type':  mujoco.mjtGeom.mjGEOM_SPHERE,
                    'size':  size,
                    'rgba':  rgba,
                    'label': label_str,
                })

            # ---- draw ----
            mujoco.mjr_render(self.viewport, self.scn, self.ctx)

            for gridpos, [t1, t2] in self._overlay.items():
                mujoco.mjr_overlay(
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    gridpos, self.viewport, t1, t2, self.ctx,
                )

            glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = (
                0.9 * self._time_per_render + 0.1 * (time.time() - render_start)
            )
        # ----------------------------------------------------------------

        if self._paused:
            while self._paused:
                _update_frame()
                if glfw.window_should_close(self.window):
                    self.close()
                    break
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += (
                viewer_model[0].opt.timestep / (self._time_per_render * self._run_speed)
            )
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                _update_frame()
                self._loop_count -= 1

        self._markers[:] = []
        self._overlay.clear()

        if self.perturbation:
            self.apply_perturbations()


# ──────────────────────────────────────────────────────────────────────────────
# Shared gizmo drawing — used by BodyHandle (CPU env) and WarpInteractiveHandle
# (parallel warp env). Pure drawing, no interaction state. Pass ``hover_mode``
# / ``hover_axis`` to highlight a specific handle (BodyHandle uses this);
# leave them ``None`` for a plain non-interactive display.
# ──────────────────────────────────────────────────────────────────────────────
GIZMO_AXIS_COLORS  = [(1.0, 0.20, 0.20, 1.00),
                      (0.20, 1.0,  0.20, 1.00),
                      (0.20, 0.55, 1.0,  1.00)]
GIZMO_PLANE_COLORS = [(1.0, 0.30, 0.30, 0.55),
                      (0.30, 1.0,  0.30, 0.55),
                      (0.30, 0.55, 1.0,  0.55)]
GIZMO_RING_COLORS  = [(1.0, 0.30, 0.30, 0.90),
                      (0.30, 1.0,  0.30, 0.90),
                      (0.30, 0.55, 1.0,  0.90)]
GIZMO_HI_COLOR        = (1.0, 0.90, 0.20, 1.0)
GIZMO_HI_PLANE_COLOR  = (1.0, 0.90, 0.20, 0.85)
GIZMO_AXIS_LABELS     = ('X', 'Y', 'Z')
GIZMO_CENTER_RGBA     = (1.0, 1.0, 0.2, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Exact viewer camera model — "where is the cursor, in world coordinates?"
#
# Every intuitive drag depends on this: the pixel the user is pointing at has
# to map to a world-space ray that matches what OpenGL actually drew. MuJoCo
# hands us the exact frustum it rendered with (``mjvGLCamera``: an asymmetric
# ``[center ± width/2] × [bottom, top]`` window at ``near``), so we invert that
# instead of assuming a symmetric frustum whose aspect matches the viewport —
# the assumption silently skews the ray on any window whose aspect drifts.
#
# These are module-level so BodyHandle (CPU) and WarpInteractiveHandle (warp)
# share one implementation; a mismatch between the two would make hover
# highlight one handle while the drag moves along another.
# ──────────────────────────────────────────────────────────────────────────────
def viewer_cam_basis(viewer):
    """Camera basis + exact frustum of ``viewer``'s current scene camera.

    Returns ``None`` when the scene has not been rendered yet (no frustum).
    Keys: ``pos / fwd / up / right`` (orthonormal, world), ``W / H`` (framebuffer
    pixels), ``f_left / f_right / f_bottom / f_top / near`` (frustum window in
    camera units at ``near``), ``ortho``, plus ``half_w / half_h`` kept for
    older callers.
    """
    try:
        cam = viewer.scn.camera[0]
    except Exception:
        return None
    near = float(cam.frustum_near)
    top  = float(cam.frustum_top)
    bot  = float(cam.frustum_bottom)
    if near <= 0.0 or (top - bot) <= 0.0:
        return None
    pos = np.array([cam.pos[0], cam.pos[1], cam.pos[2]], dtype=float)
    fwd = np.array([cam.forward[0], cam.forward[1], cam.forward[2]], dtype=float)
    up0 = np.array([cam.up[0], cam.up[1], cam.up[2]], dtype=float)
    nf  = float(np.linalg.norm(fwd))
    if nf < 1e-9:
        return None
    fwd = fwd / nf
    right = np.cross(fwd, up0)
    nr    = float(np.linalg.norm(right))
    if nr < 1e-9:
        return None
    right = right / nr
    up    = np.cross(right, fwd)

    vp = viewer.viewport
    W, H = int(vp.width), int(vp.height)
    if W <= 0 or H <= 0:
        return None

    # Horizontal window: prefer the frustum MuJoCo actually used; fall back to
    # the viewport aspect when the field is unset (older scenes).
    width = float(getattr(cam, 'frustum_width', 0.0) or 0.0)
    if width <= 0.0:
        width = (W / H) * (top - bot)
    center  = float(getattr(cam, 'frustum_center', 0.0) or 0.0)
    f_left  = center - 0.5 * width
    f_right = center + 0.5 * width
    return dict(pos=pos, fwd=fwd, up=up, right=right, W=W, H=H,
                f_left=f_left, f_right=f_right, f_bottom=bot, f_top=top,
                near=near, ortho=bool(getattr(cam, 'orthographic', 0)),
                half_w=0.5 * width, half_h=0.5 * (top - bot))


def cam_world_to_pixel(cb, p_world):
    """World point → framebuffer pixel ``(x, y)`` (y down), or ``None`` when
    the point is behind the camera."""
    if cb is None:
        return None
    rel = np.asarray(p_world, dtype=float) - cb['pos']
    x_c = float(rel @ cb['right']); y_c = float(rel @ cb['up']); z_c = float(rel @ cb['fwd'])
    if cb['ortho']:
        x_n, y_n = x_c, y_c
    else:
        if z_c <= 1e-6:
            return None
        x_n = x_c * cb['near'] / z_c
        y_n = y_c * cb['near'] / z_c
    px = (x_n - cb['f_left']) / (cb['f_right'] - cb['f_left']) * cb['W']
    py = (cb['f_top'] - y_n) / (cb['f_top'] - cb['f_bottom']) * cb['H']
    return (px, py)


def cam_mouse_ray(cb, mx, my):
    """Framebuffer pixel (y down) → world-space ray ``(origin, unit dir)``."""
    if cb is None:
        return None
    u = float(mx) / cb['W']
    v = float(my) / cb['H']
    x_n = cb['f_left'] + u * (cb['f_right'] - cb['f_left'])
    y_n = cb['f_top']  - v * (cb['f_top']   - cb['f_bottom'])
    if cb['ortho']:
        # Parallel projection: the pixel shifts the ray ORIGIN, not its direction.
        return cb['pos'] + x_n * cb['right'] + y_n * cb['up'], cb['fwd'].copy()
    d = x_n * cb['right'] + y_n * cb['up'] + cb['near'] * cb['fwd']
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return None
    return cb['pos'], d / n


def cam_pixels_per_meter(cb, depth):
    """Screen pixels spanned by one world metre at ``depth`` along the view axis."""
    if cb is None:
        return 0.0
    span = cb['f_top'] - cb['f_bottom']
    if span <= 0.0:
        return 0.0
    if cb['ortho']:
        return cb['H'] / span
    if depth <= 1e-6:
        return 0.0
    return cb['H'] * cb['near'] / (span * depth)


def ray_plane_intersect(ray_o, ray_d, plane_p, plane_n):
    """Where a ray meets a plane, or ``None`` (parallel, or behind the eye)."""
    denom = float(np.asarray(ray_d) @ np.asarray(plane_n))
    if abs(denom) < 1e-9:
        return None
    t = float((np.asarray(plane_p) - np.asarray(ray_o)) @ np.asarray(plane_n)) / denom
    if t < 0:
        return None
    return np.asarray(ray_o) + t * np.asarray(ray_d)


def ray_line_closest_t(ray_o, ray_d, line_p, line_d):
    """Parameter ``t`` of the point ``line_p + t·line_d`` closest to the ray.

    This is what makes an axis handle stick to the cursor: the grabbed point
    on the axis stays under the mouse instead of tracking a linearised pixel
    delta. Returns ``(t, sin_angle)`` where ``sin_angle`` is how far the axis
    is from being parallel to the view ray — near 0 the solution blows up and
    the caller should hold the previous value. ``None`` if degenerate.

    ``t`` is measured in metres along the **unit** direction of ``line_d``.
    """
    ld = np.asarray(line_d, dtype=float)
    rd = np.asarray(ray_d,  dtype=float)
    nl, nr = float(np.linalg.norm(ld)), float(np.linalg.norm(rd))
    if nl < 1e-12 or nr < 1e-12:
        return None
    ld = ld / nl; rd = rd / nr
    b   = float(ld @ rd)
    den = 1.0 - b * b                       # = sin²(angle between the lines)
    if den < 1e-12:
        return None
    w0 = np.asarray(line_p, dtype=float) - np.asarray(ray_o, dtype=float)
    d  = float(ld @ w0); e = float(rd @ w0)
    t  = (b * e - d) / den
    return float(t), float(np.sqrt(max(0.0, den)))


def draw_translate_gizmo(env, p, R,
                         hover_mode = None,
                         hover_axis = None,
                         axis_len:    float = 0.10,
                         plane_off:   float = 0.035,
                         plane_size:  float = 0.030,
                         plane_thick: float = 0.0008,
                         arrow_r:     float = 0.0035,
                         center_r:    float = 0.005,
                         axis_colors  = GIZMO_AXIS_COLORS,
                         plane_colors = GIZMO_PLANE_COLORS,
                         hi_color     = GIZMO_HI_COLOR,
                         hi_plane     = GIZMO_HI_PLANE_COLOR,
                         labels       = GIZMO_AXIS_LABELS,
                         hover_scale: float = 1.0,
                         show_center: bool = True):
    """RGB axis arrows + plane quads at world pose ``(p, R)``.

    ``hover_scale`` thickens the hovered/dragged handle (sim_core's
    InteractiveMarker convention) so the active one reads at a glance, not
    just by colour.
    """
    p = np.asarray(p)
    for k in range(3):
        hot = (hover_mode == 'translate' and hover_axis == k)
        c = hi_color if hot else axis_colors[k]
        env.plot_arrow_fr2to(
            p_fr=p, p_to=p + R[:, k] * axis_len,
            r=arrow_r * (hover_scale if hot else 1.0), rgba=c, label=labels[k],
        )
    for k in range(3):
        u = R[:, (k+1) % 3]; v = R[:, (k+2) % 3]; n = R[:, k]
        center = p + (plane_off + plane_size/2) * (u + v)
        R_box  = np.column_stack([u, v, n])
        hot = (hover_mode == 'plane' and hover_axis == k)
        c = hi_plane if hot else plane_colors[k]
        env.plot_box(
            p=center, R=R_box,
            xlen=plane_size, ylen=plane_size,
            zlen=plane_thick * (hover_scale if hot else 1.0),
            rgba=c, label='',
        )
    if show_center:
        env.plot_sphere(p=p, r=center_r, rgba=GIZMO_CENTER_RGBA)


def draw_rotate_gizmo(env, p, R,
                      hover_mode = None,
                      hover_axis = None,
                      ring_r:     float = 0.075,
                      ring_segs:  int   = 64,
                      ring_thick: float = 0.0022,
                      center_r:   float = 0.005,
                      ring_colors = GIZMO_RING_COLORS,
                      hi_color    = GIZMO_HI_COLOR,
                      hover_scale: float = 1.0,
                      show_center: bool = True):
    """RGB rotation rings (one per body axis) at world pose ``(p, R)``.

    ``hover_scale`` thickens the hovered/dragged ring — see
    :func:`draw_translate_gizmo`.
    """
    p = np.asarray(p)
    th = np.linspace(0.0, 2*np.pi, ring_segs, endpoint=False)
    cs, sn = np.cos(th), np.sin(th)
    for k in range(3):
        hot = (hover_mode == 'rotate' and hover_axis == k)
        c = hi_color if hot else ring_colors[k]
        u = R[:, (k+1) % 3]; v = R[:, (k+2) % 3]
        ring = p[None, :] + ring_r * (cs[:, None]*u + sn[:, None]*v)
        N = len(ring)
        for i in range(N):
            env.plot_cylinder_fr2to(
                p_fr=ring[i], p_to=ring[(i + 1) % N],
                r=ring_thick * (hover_scale if hot else 1.0), rgba=c,
            )
    if show_center:
        env.plot_sphere(p=p, r=center_r, rgba=GIZMO_CENTER_RGBA)


# ──────────────────────────────────────────────────────────────────────────────
# BodyHandle — interactive ImGuizmo-style 3D handle for a free-jointed body.
# ──────────────────────────────────────────────────────────────────────────────
class BodyHandle:
    """
    Interactive translate / rotate handles for a single MuJoCo body.

    Drives the body's pose via one of two backends, selected by ``mode``:

    * ``'free_joint'`` — edits ``data.qpos[adr:adr+7]`` of the body's free
      joint. Use under forward-kinematics demos / static manipulation.
    * ``'mocap'``      — edits ``data.mocap_pos[mid]`` /
      ``data.mocap_quat[mid]`` of the (mocap) body. Use under dynamic
      simulation where the controlled link is welded to a mocap.
    * ``'auto'`` (default) — picks ``mocap`` if the target body is itself a
      mocap body; otherwise falls back to ``free_joint`` if the body owns a
      free joint. Raises if neither applies.

    The visible handle center is taken from ``data.xpos`` / ``data.xmat`` of
    the supplied body. In mocap mode pass the *mocap* body itself — its
    xpos/xmat track ``mocap_pos`` / ``mocap_quat`` after ``mj_forward``, so
    the projection / hover / drawing logic is identical between modes.

    Renders RGB axis arrows + plane quads (TRANSLATE mode, default) or
    rotation rings (ROTATE mode, hold Shift). LMB drag on a hovered handle
    edits the pose directly; mouse outside any handle keeps the viewer's
    normal camera control. Ctrl+LMB / Ctrl+RMB still trigger MuJoCo's
    built-in perturb on the same body as a fallback.

    Usage::

        from grit.util.mjwarp_viewer import BodyHandle

        env.init_viewer(...)
        # Free-joint body (auto-detected):
        bh = BodyHandle(env, wrist_body_id).attach()
        # Or, mocap-driven body (auto-detected from body name):
        bh = BodyHandle(env, "rh_mocap").attach()
        while env.is_viewer_alive():
            # ... update other qpos (e.g. fingers) ...
            env.forward()
            bh.update()                    # project, hover, draw
            if env.is_key_pressed_once(glfw.KEY_R):
                bh.reset_pose(xyz=[0, 0, 1.0])
            env.render()
        bh.detach()

    The ``env`` argument must expose a MuJoCoParser-style API:
    ``env.data``, ``env.model``, ``env.viewer`` plus the plot helpers
    ``plot_arrow_fr2to``, ``plot_box``, ``plot_cylinder_fr2to``, ``plot_sphere``.
    """

    BASE_AXIS_COLORS  = [(1.0, 0.20, 0.20, 1.00),
                         (0.20, 1.0, 0.20, 1.00),
                         (0.20, 0.55, 1.0, 1.00)]
    BASE_PLANE_COLORS = [(1.0, 0.30, 0.30, 0.55),
                         (0.30, 1.0, 0.30, 0.55),
                         (0.30, 0.55, 1.0, 0.55)]
    BASE_RING_COLORS  = [(1.0, 0.30, 0.30, 0.90),
                         (0.30, 1.0, 0.30, 0.90),
                         (0.30, 0.55, 1.0, 0.90)]
    HI_COLOR          = (1.0, 0.90, 0.20, 1.0)
    HI_PLANE_COLOR    = (1.0, 0.90, 0.20, 0.85)
    AXIS_LABELS       = ('X', 'Y', 'Z')

    def __init__(self,
                 env,
                 body,                            # body_id (int) | body_name (str)
                 mode:           str   = 'auto',  # 'auto' | 'free_joint' | 'mocap'
                 label:          str   = None,    # display label (defaults to body name)
                 axis_len:       float = 0.10,
                 ring_r:         float = 0.075,
                 ring_segs:      int   = 64,
                 ring_thick:     float = 0.0022,
                 plane_off:      float = 0.035,
                 plane_size:     float = 0.030,
                 plane_thick:    float = 0.0008,
                 hover_px_axis:  float = 30.0,
                 hover_px_plane: float = 24.0,
                 hover_px_ring:  float = 22.0,
                 view_q_min:     float = 0.10,
                 # ── Pointer visuals (sim_core InteractiveMarker) ──
                 show_mouse_sphere: bool = True,
                 mouse_sphere_r: float = 0.005,
                 mouse_color           = (0.15, 0.45, 1.00, 1.0),   # blue 3D pointer
                 click_sphere_r: float = 0.007,
                 click_color           = (1.00, 0.20, 0.95, 1.0),   # magenta grab point
                 mouse_view_offset     = (0.0, 0.0),                # (right, up) metres
                 hover_scale:    float = 1.35,
                 # ── Grasp drag (Alt+LMB / Alt+wheel) ──
                 enable_grasp:   bool  = True,
                 ctrl_open             = None,   # (nu,) "open hand" baseline. Default: ctrl_min per joint.
                 ctrl_closed           = None,   # (nu,) "closed hand" baseline. Default: ctrl_max per joint.
                 grasp_drag_idxs       = None,   # ctrl indices the grasp drag controls (default: all).
                 grasp_sensitivity: float = 0.005,
                 slider_widget          = None,  # MultiSliderQtWidget-like: keep slider visually in sync AND
                                                 # ensure the notebook's sim_step (which usually reads sliders
                                                 # → env.data.ctrl) doesn't overwrite the grasp values.
                 slider_ctrl_offset: int = 0):   # index in slider.get_values() where actuator ctrls begin
                                                 # (e.g. 6 in 13/15 notebooks: [x,y,z,r,p,y, ctrl_0, ...])
        body_id = env.body_names.index(body) if isinstance(body, str) else int(body)

        self.env     = env
        self.viewer  = env.viewer
        self.body_id = body_id
        # Display label used by ``draw_overlay`` (e.g. "wrist", "obj_3").
        # Falls back to the body name if not provided.
        try:
            default_label = env.body_names[body_id]
        except Exception:
            default_label = str(body_id)
        self.label   = str(label) if label is not None else default_label

        self.AXIS_LEN       = float(axis_len)
        self.RING_R         = float(ring_r)
        self.RING_SEGS      = int(ring_segs)
        self.RING_THICK     = float(ring_thick)
        self.PLANE_OFFSET   = float(plane_off)
        self.PLANE_SIZE     = float(plane_size)
        self.PLANE_THICK    = float(plane_thick)
        self.HOVER_PX_AXIS  = float(hover_px_axis)
        self.HOVER_PX_PLANE = float(hover_px_plane)
        self.HOVER_PX_RING  = float(hover_px_ring)
        self.VIEW_Q_MIN     = float(view_q_min)
        # sim_core InteractiveMarker visuals (see MOUSE_* block below).
        self.MOUSE_VIEW_OFFSET = np.asarray(mouse_view_offset, dtype=float).reshape(2)
        self.MOUSE_SPHERE_R    = float(mouse_sphere_r)
        self.MOUSE_COLOR       = tuple(mouse_color)
        self.CLICK_SPHERE_R    = float(click_sphere_r)
        self.CLICK_COLOR       = tuple(click_color)
        self.HOVER_SCALE       = float(hover_scale)
        self.SHOW_MOUSE_SPHERE = bool(show_mouse_sphere)
        # Drag-robustness thresholds. Axis: sin(angle between axis and view
        # ray) below this makes the closest-point solution explode (you are
        # sighting straight down the axis) → freeze instead of teleport.
        # Ring: |axis . view_dir| below this means an edge-on ring, where the
        # plane hit slides to infinity → fall back to the screen angle.
        self.AXIS_DRAG_SIN_MIN = 0.12      # ~7 deg off the view ray
        self.RING_DRAG_COS_MIN = 0.15      # ~81 deg tilt

        # ── Pose backend (mocap vs free_joint) ────────────────────────────
        self._mode      = self._resolve_mode(env.model, body_id, mode)
        self._mocap_id  = -1
        self.qpos_xyz_adr  = -1
        self.qpos_quat_adr = -1
        if self._mode == 'mocap':
            self._mocap_id = int(env.model.body_mocapid[body_id])
        else:  # 'free_joint'
            adr_xyz, adr_quat = self._find_free_joint_qpos(env.model, body_id)
            if adr_xyz < 0:
                raise ValueError(
                    f"BodyHandle: body id {body_id} has no free joint and is "
                    f"not a mocap body — cannot drive it interactively.")
            self.qpos_xyz_adr  = adr_xyz
            self.qpos_quat_adr = adr_quat

        # public state
        self.shift_held    = False
        self.hover_mode    = None   # 'translate' | 'plane' | 'rotate' | None
        self.hover_axis    = None
        self.drag_mode     = None
        self.drag_axis     = None
        self.rot_sign      = 1.0
        self.rot_cum_angle = 0.0
        # Grabbed handle point, in the body frame, while a drag is live.
        self.last_click_p_local = None

        # private drag state
        self._start_mouse_xy      = None
        self._start_qpos_xyz      = None
        self._plane_normal        = None
        self._plane_origin        = None
        self._plane_start_isect   = None
        self._start_qpos_quat     = None
        self._start_axis_world    = None
        self._start_screen_center = None
        self._rot_prev_screen_a   = 0.0
        # sticky axis drag (ray <-> axis-line closest point), frozen at press
        self._axis_line_o         = None
        self._axis_line_d         = None
        self._axis_grab_t         = None
        self._axis_last_off       = 0.0
        # in-plane rotate (ray <-> ring-plane hit), frozen at press
        self._rot_plane_o         = None
        self._rot_plane_u         = None
        self._rot_plane_v         = None
        self._rot_prev_plane_a    = None

        # per-frame projections
        self._axis_pstart       = [None]*3
        self._axis_pend         = [None]*3
        self._axis_pix_dir      = [None]*3
        self._axis_pix_len      = [0.0]*3
        self._axis_world        = [None]*3
        self._ring_pts_pix      = [None]*3
        self._plane_corners_pix = [None]*3

        self._prev_btn_cb    = None
        self._prev_cur_cb    = None
        self._prev_scroll_cb = None
        self._attached       = False

        # ── Grasp drag state (Alt+LMB / Alt+wheel) ─────────────────────
        self._grasp_enabled   = bool(enable_grasp)
        self._grasp_dragging  = False
        self._grasp_last_xy   = None
        self._grasp_active_idx = -1                 # -1 = all, 0..n-1 = single joint
        self._grasp_t_per_joint = None
        self._grasp_idxs        = None
        self._ctrl_open  = None
        self._ctrl_closed = None
        self._ctrl_min   = None
        self._ctrl_max   = None
        self.GRASP_SENS  = float(grasp_sensitivity)
        # Optional slider integration so a notebook that drives ctrl from a
        # MultiSliderQtWidget stays in sync (otherwise its sim_step writes the
        # slider value over our grasp value every tick).
        self._slider_widget       = slider_widget
        self._slider_ctrl_offset  = int(slider_ctrl_offset)
        if self._grasp_enabled:
            self._init_grasp_buffers(ctrl_open, ctrl_closed, grasp_drag_idxs)

    def _init_grasp_buffers(self, ctrl_open, ctrl_closed, grasp_drag_idxs):
        try:
            nu = int(getattr(self.env, "n_ctrl", len(self.env.ctrl_names)))
        except Exception:
            nu = 0
        if nu == 0:
            self._grasp_enabled = False
            return
        try:
            ctrl_min_arr = np.asarray(self.env.ctrl_mins, dtype=float)
            ctrl_max_arr = np.asarray(self.env.ctrl_maxs, dtype=float)
        except Exception:
            self._grasp_enabled = False
            return
        if ctrl_open   is None: ctrl_open   = ctrl_min_arr.copy()
        if ctrl_closed is None: ctrl_closed = ctrl_max_arr.copy()
        self._ctrl_open   = np.asarray(ctrl_open,   dtype=float)[:nu]
        self._ctrl_closed = np.asarray(ctrl_closed, dtype=float)[:nu]
        self._ctrl_min    = ctrl_min_arr[:nu]
        self._ctrl_max    = ctrl_max_arr[:nu]
        self._grasp_idxs  = (np.arange(nu, dtype=int) if grasp_drag_idxs is None
                             else np.asarray(grasp_drag_idxs, dtype=int))
        self._grasp_t_per_joint = np.zeros(len(self._grasp_idxs), dtype=float)

    # ────────────────────────── public API ──────────────────────────
    def attach(self):
        """Hook GLFW mouse callbacks on the viewer's window.

        Uses the *return value* of ``glfw.set_*_callback`` (the previous GLFW
        callback) for the chain, NOT ``viewer._mouse_button_callback`` (a
        bound method that is identical no matter how many handlers are
        already stacked) — otherwise stacking another handler on top would
        silently bypass us.
        """
        if self._attached:
            return self
        self._prev_btn_cb = glfw.set_mouse_button_callback(
            self.viewer.window, self._on_button)
        self._prev_cur_cb = glfw.set_cursor_pos_callback(
            self.viewer.window, self._on_cursor)
        if self._grasp_enabled:
            self._prev_scroll_cb = glfw.set_scroll_callback(
                self.viewer.window, self._on_scroll)
        # Also pre-select for Ctrl+drag perturb fallback. Zero ``localpos`` so
        # ``mjv_initPerturb`` computes the reference point at the body origin
        # (otherwise stale values from a prior ``mjv_select`` make Ctrl+drag
        # pivot around an arbitrary far-away point).
        try:
            self.viewer.pert.select     = self.body_id
            self.viewer.pert.skinselect = -1
            self.viewer.pert.active     = 0
            self.viewer.pert.localpos[:] = 0.0
        except Exception:
            pass
        self._attached = True
        return self

    def detach(self):
        """Restore the viewer's original GLFW mouse callbacks."""
        if not self._attached:
            return self
        # Skip GLFW calls if the underlying window is already gone. Calling
        # ``glfw.set_*_callback`` after ``glfw.destroy_window`` (or with a
        # None handle from a closed viewer) can corrupt the GLFW callback
        # table and lead to occasional native crashes on subsequent renders.
        win = getattr(self.viewer, "window", None)
        if win is not None:
            try:
                glfw.set_mouse_button_callback(win, self._prev_btn_cb)
                glfw.set_cursor_pos_callback   (win, self._prev_cur_cb)
                if self._grasp_enabled and self._prev_scroll_cb is not None:
                    glfw.set_scroll_callback(win, self._prev_scroll_cb)
            except Exception:
                pass
        self._grasp_dragging = False
        self._grasp_last_xy  = None
        self._attached = False
        return self

    def reset_pose(self, xyz=None, quat=None):
        """Set the body's pose. Defaults to (0,0,0) and identity quat. Backend
        (free_joint qpos vs mocap_pos/mocap_quat) is chosen by ``mode``."""
        if xyz  is None: xyz  = [0.0, 0.0, 0.0]
        if quat is None: quat = [1.0, 0.0, 0.0, 0.0]
        self._write_xyz(xyz)
        self._write_quat(quat)

    @property
    def mode(self) -> str:
        """Active pose backend: ``'free_joint'`` or ``'mocap'``."""
        return self._mode

    # ────────────────────────── pose backend ──────────────────────────
    def _body_pose(self):
        """Return ``(wp, R)``: world position (3,) and rotation matrix (3,3)
        of the controlled body. Default reads CPU ``env.data``; subclasses
        may override (e.g. parallel warp env with grid offsets)."""
        wp = self.env.data.xpos[self.body_id].copy()
        R  = self.env.data.xmat[self.body_id].reshape(3, 3).copy()
        return wp, R

    @staticmethod
    def _resolve_mode(model, body_id: int, mode: str) -> str:
        if mode == 'auto':
            if int(model.body_mocapid[body_id]) >= 0:
                return 'mocap'
            return 'free_joint'
        if mode == 'mocap':
            if int(model.body_mocapid[body_id]) < 0:
                raise ValueError(f"body id {body_id} is not a mocap body")
            return 'mocap'
        if mode == 'free_joint':
            return 'free_joint'
        raise ValueError(f"unknown BodyHandle mode: {mode!r}")

    def _read_xyz(self):
        if self._mode == 'mocap':
            return self.env.data.mocap_pos[self._mocap_id].copy()
        return self.env.data.qpos[self.qpos_xyz_adr:self.qpos_xyz_adr+3].copy()

    def _read_quat(self):
        if self._mode == 'mocap':
            return self.env.data.mocap_quat[self._mocap_id].copy()
        return self.env.data.qpos[self.qpos_quat_adr:self.qpos_quat_adr+4].copy()

    def _write_xyz(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        if self._mode == 'mocap':
            self.env.data.mocap_pos[self._mocap_id] = xyz
        else:
            self.env.data.qpos[self.qpos_xyz_adr:self.qpos_xyz_adr+3] = xyz

    def _write_quat(self, quat):
        quat = np.asarray(quat, dtype=float)
        if self._mode == 'mocap':
            self.env.data.mocap_quat[self._mocap_id] = quat
        else:
            self.env.data.qpos[self.qpos_quat_adr:self.qpos_quat_adr+4] = quat

    def update(self):
        """Per-frame: poll Shift, refresh projections, run hover detection, draw handles.
        Call this AFTER ``env.forward()`` and BEFORE ``env.render()``."""
        try:
            self.shift_held = (
                glfw.get_key(self.viewer.window, glfw.KEY_LEFT_SHIFT)  == glfw.PRESS or
                glfw.get_key(self.viewer.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            )
        except Exception:
            self.shift_held = False

        if self.drag_mode is not None:
            active_set = 'rotate' if self.drag_mode == 'rotate' else 'translate'
        else:
            active_set = 'rotate' if self.shift_held else 'translate'

        wp, R = self._body_pose()

        self._update_projections(wp, R)
        if self.drag_mode is None:
            self._update_hover(wp, R, active_set)

        h_mode = self.drag_mode if self.drag_mode is not None else self.hover_mode
        h_axis = self.drag_axis if self.drag_axis is not None else self.hover_axis
        if active_set == 'translate':
            self._draw_translate(wp, R, h_mode, h_axis)
        else:
            self._draw_rotate(wp, R, h_mode, h_axis)
        self._draw_pointer(wp, R)

    def _draw_pointer(self, wp, R):
        """sim_core InteractiveMarker pointer visuals.

        * blue sphere  — the mouse as a 3D point, on the camera-facing plane
          through the marker (or through the drag's frozen reference point, so
          it does not slide as the body moves under it). This is the same
          point every hit test uses.
        * magenta sphere — where the current drag grabbed the handle, carried
          in the body frame so it rides along.
        """
        if not self.SHOW_MOUSE_SPHERE:
            return
        try:
            if self.last_click_p_local is not None and self.drag_mode is not None:
                self.env.plot_sphere(p=wp + R @ self.last_click_p_local,
                                     r=self.CLICK_SPHERE_R, rgba=self.CLICK_COLOR,
                                     label="")
            mp = self.pointer_world_point(self._drag_ref_point(wp))
            if mp is not None:
                self.env.plot_sphere(p=mp, r=self.MOUSE_SPHERE_R,
                                     rgba=self.MOUSE_COLOR, label="")
        except Exception:
            pass

    @property
    def status_text(self) -> str:
        """One-line status useful for ``env.viewer_text_overlay``."""
        set_lbl = 'ROTATE (Shift)' if (self.shift_held or self.drag_mode == 'rotate') else 'TRANSLATE'
        h_mode = self.drag_mode if self.drag_mode is not None else self.hover_mode
        h_axis = self.drag_axis if self.drag_axis is not None else self.hover_axis
        m = h_mode if h_mode is not None else '-'
        a = self.AXIS_LABELS[h_axis] if h_axis is not None else '-'
        extra = ''
        if self.drag_mode == 'rotate':
            # rot_cum_angle is already the signed world angle (ASCII only —
            # the viewer font has no glyph for a degree sign).
            extra = f"  d={np.degrees(self.rot_cum_angle):+.1f}deg"
        return f"[{set_lbl}] [{m} {a}]{extra}"

    HELP_TEXT = ("LMB=move handle | Shift+LMB=rotate ring | "
                 "Alt+LMB=grasp | Alt+wheel=joint sel | "
                 "RMB=pull obj | dblclick=switch | [W] wrist")

    def draw_overlay(self,
                     show_status: bool = True,
                     show_grasp:  bool = True,
                     show_help:   bool = True,
                     help_text:   str  = None,
                     help_loc:    str  = "top left",
                     status_loc:  str  = "bottom right"):
        """Push handle status lines to ``env.viewer_text_overlay``.

        Call once per render frame. Lines emitted (each toggleable):
          - ``handle target  : <self.label>``
          - ``handle status  : <self.status_text>``
          - ``grasp target [Alt+wheel]``  /  ``grasp_t [Alt+LMB]``  (if grasp enabled)
          - help text (default: see ``BodyHandle.HELP_TEXT``).
        """
        if show_status:
            self.env.viewer_text_overlay(
                text1="handle target", text2=str(self.label), loc=status_loc,
            )
            self.env.viewer_text_overlay(
                text1="handle status", text2=self.status_text, loc=status_loc,
            )
        if show_grasp and self._grasp_enabled and self._grasp_t_per_joint is not None:
            self.env.viewer_text_overlay(
                text1="grasp target [Alt+wheel]:",
                text2=self.grasp_active_label, loc=status_loc,
            )
            self.env.viewer_text_overlay(
                text1="grasp_t [Alt+LMB]:",
                text2=f"{self.grasp_t:.2f}", loc=status_loc,
            )
        if show_help:
            self.env.viewer_text_overlay(
                text1=help_text if help_text is not None else self.HELP_TEXT,
                loc=help_loc,
            )

    # ────────────────────────── geometry helpers ──────────────────────────
    @staticmethod
    def _find_free_joint_qpos(model, body_id: int):
        for jid in range(model.njnt):
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE \
               and int(model.jnt_bodyid[jid]) == body_id:
                adr = int(model.jnt_qposadr[jid])
                return adr, adr + 3
        return -1, -1

    # Camera model lives in the module-level helpers so this class and
    # WarpInteractiveHandle can never disagree about where the cursor points.
    def _cam_basis(self):
        return viewer_cam_basis(self.viewer)

    def _world_to_pixel(self, p_world):
        return cam_world_to_pixel(self._cam_basis(), p_world)

    def _mouse_world_ray(self, mx, my):
        return cam_mouse_ray(self._cam_basis(), mx, my)

    def _cursor_pixel(self):
        """Current cursor in framebuffer pixels (HiDPI-scaled, y down)."""
        x, y = glfw.get_cursor_pos(self.viewer.window)
        return self.viewer._scale * x, self.viewer._scale * y

    # ── The 3D pointer (sim_core InteractiveMarker) ───────────────
    # The marker's whole UX rests on ONE idea: show the mouse as a point in
    # the scene, at the handle's depth, and make every hit test use exactly
    # that point. Then "what you aim at" and "what you see" cannot disagree.
    #
    # ``MOUSE_VIEW_OFFSET`` shifts that virtual pointer by a world-space
    # camera-plane offset (metres at the reference depth, converted to a pixel
    # shift). sim_core defaults it to 3 cm right so the OS cursor arrow does
    # not cover the sphere; we default to 0 so the pointer is literally the
    # cursor, and leave the knob for callers who want the offset feel.
    def _pointer_pixel(self, p_ref=None, cb=None):
        """Cursor pixel, shifted by ``MOUSE_VIEW_OFFSET`` at ``p_ref``'s depth."""
        mx, my = self._cursor_pixel()
        off = self.MOUSE_VIEW_OFFSET
        if p_ref is None or (abs(off[0]) < 1e-12 and abs(off[1]) < 1e-12):
            return mx, my
        cb = self._cam_basis() if cb is None else cb
        if cb is None:
            return mx, my
        p_ref = np.asarray(p_ref, dtype=float)
        base  = cam_world_to_pixel(cb, p_ref)
        shift = cam_world_to_pixel(cb, p_ref + off[0]*cb['right'] + off[1]*cb['up'])
        if base is None or shift is None:
            return mx, my
        return mx + shift[0] - base[0], my + shift[1] - base[1]

    def _drag_ref_point(self, fallback=None):
        """Depth reference for the virtual pointer: the drag's frozen origin
        while one is live, else ``fallback`` (normally the gizmo centre).

        Frozen rather than live so the pointer plane cannot drift along with
        the body the drag is moving.
        """
        origin = {'translate': self._axis_line_o,
                  'plane':     self._plane_origin,
                  'rotate':    self._rot_plane_o}.get(self.drag_mode)
        return fallback if origin is None else origin

    def pointer_world_point(self, p_ref, cb=None):
        """Where the virtual pointer sits in 3D: the mouse ray meeting the
        camera-facing plane through ``p_ref``. ``None`` if it is behind us."""
        cb = self._cam_basis() if cb is None else cb
        if cb is None:
            return None
        mx, my = self._pointer_pixel(p_ref, cb)
        ray = cam_mouse_ray(cb, mx, my)
        if ray is None:
            return None
        return self._ray_plane_intersect(ray[0], ray[1],
                                         np.asarray(p_ref, dtype=float), cb['fwd'])

    def _handle_point(self, mode, axis, mx, my, wp, R):
        """World point on the handle that a click at ``(mx, my)`` grabs.

        Drawn as the magenta click sphere for the duration of the drag, so
        the user can see which part of the arrow / plane / ring they took
        hold of.
        """
        if mode == 'translate':
            ps, pe = self._axis_pstart[axis], self._axis_pend[axis]
            aw     = self._axis_world[axis]
            if ps is None or pe is None or aw is None:
                return None
            dpix = np.array([pe[0]-ps[0], pe[1]-ps[1]], dtype=float)
            L2   = float(dpix @ dpix)
            t    = 0.0 if L2 < 1e-9 else float(np.clip(
                (np.array([mx, my], dtype=float) - np.array(ps)) @ dpix / L2, 0.0, 1.0))
            return wp + aw * (t * self.AXIS_LEN)
        if mode == 'plane':
            aw  = self._axis_world[axis]
            ray = self._mouse_world_ray(mx, my)
            if aw is None or ray is None:
                return None
            return self._ray_plane_intersect(ray[0], ray[1], wp, aw)
        if mode == 'rotate':
            pts = self._ring_pts_pix[axis]
            if not pts:
                return None
            best, best_i = 1e18, None
            for i, q in enumerate(pts):
                if q is None:
                    continue
                dd = float(np.hypot(mx - q[0], my - q[1]))
                if dd < best:
                    best, best_i = dd, i
            if best_i is None:
                return None
            return self._ring_world_pts(wp, R, axis)[best_i]
        return None

    # Thin alias so the drag code reads the same in every handle class.
    _ray_plane_intersect = staticmethod(ray_plane_intersect)

    @staticmethod
    def _seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        L2 = dx*dx + dy*dy
        if L2 < 1e-9:
            return float(np.hypot(px - ax, py - ay))
        t = max(0.0, min(1.0, ((px - ax)*dx + (py - ay)*dy) / L2))
        return float(np.hypot(px - (ax + t*dx), py - (ay + t*dy)))

    @classmethod
    def _polyline_min_dist(cls, px, py, pts_pix):
        best = 1e9
        N = len(pts_pix)
        for i in range(N):
            a = pts_pix[i]; b = pts_pix[(i+1) % N]
            if a is None or b is None: continue
            d = cls._seg_dist(px, py, a[0], a[1], b[0], b[1])
            if d < best: best = d
        return best

    def _ring_world_pts(self, wp, R, k):
        u = R[:, (k+1) % 3]; v = R[:, (k+2) % 3]
        th = np.linspace(0.0, 2*np.pi, self.RING_SEGS, endpoint=False)
        return wp[None, :] + self.RING_R * (np.cos(th)[:, None]*u + np.sin(th)[:, None]*v)

    def _plane_corners_world(self, wp, R, k):
        u = R[:, (k+1) % 3]; v = R[:, (k+2) % 3]
        o, s = self.PLANE_OFFSET, self.PLANE_SIZE
        return [wp +  o      * u +  o      * v,
                wp + (o + s) * u +  o      * v,
                wp + (o + s) * u + (o + s) * v,
                wp +  o      * u + (o + s) * v]

    # ────────────────────────── per-frame logic ──────────────────────────
    def _update_projections(self, wp, R):
        p_start_pix = self._world_to_pixel(wp)
        for k in range(3):
            ax = R[:, k]
            an = float(np.linalg.norm(ax))
            ax = ax / an if an > 1e-9 else ax
            self._axis_world[k] = ax
            p_end_pix = self._world_to_pixel(wp + ax * self.AXIS_LEN)
            self._axis_pstart[k] = p_start_pix
            self._axis_pend[k]   = p_end_pix
            if (p_start_pix is not None) and (p_end_pix is not None):
                ddx = p_end_pix[0] - p_start_pix[0]; ddy = p_end_pix[1] - p_start_pix[1]
                L = float(np.hypot(ddx, ddy))
                self._axis_pix_len[k] = L
                self._axis_pix_dir[k] = (ddx/L, ddy/L) if L > 1e-3 else None
            else:
                self._axis_pix_len[k] = 0.0
                self._axis_pix_dir[k] = None
            self._plane_corners_pix[k] = [self._world_to_pixel(c)
                                          for c in self._plane_corners_world(wp, R, k)]
            self._ring_pts_pix[k]      = [self._world_to_pixel(p)
                                          for p in self._ring_world_pts(wp, R, k)]

    def _update_hover(self, wp, R, active_set):
        self.hover_mode = None
        self.hover_axis = None
        try:
            cb = self._cam_basis()
            if cb is None: return
            # Hit-test with the SAME virtual pointer that gets drawn.
            mx, my = self._pointer_pixel(wp, cb)
            ray = cam_mouse_ray(cb, mx, my)
            if ray is None: return
            ray_o, ray_d = ray

            candidates = []
            if active_set == 'translate':
                for k in range(3):
                    aw = self._axis_world[k]
                    if aw is None: continue
                    af = float(aw @ cb['fwd'])
                    vq_axis  = float(np.sqrt(max(0.0, 1.0 - af*af)))
                    vq_plane = abs(af)
                    if vq_plane >= self.VIEW_Q_MIN:
                        isect = self._ray_plane_intersect(ray_o, ray_d, wp, aw)
                        if isect is not None:
                            u = R[:, (k+1) % 3]; v = R[:, (k+2) % 3]
                            rel = isect - wp
                            uc = float(rel @ u); vc = float(rel @ v)
                            o, s = self.PLANE_OFFSET, self.PLANE_SIZE
                            du = max(o - uc, uc - (o + s), 0.0)
                            dv = max(o - vc, vc - (o + s), 0.0)
                            wd = float(np.hypot(du, dv))
                            depth = float((isect - cb['pos']) @ cb['fwd'])
                            score = (wd * cam_pixels_per_meter(cb, depth)
                                     / self.HOVER_PX_PLANE - 0.02 * vq_plane)
                            if score < 1.0:
                                candidates.append((score, 'plane', k))
                    if vq_axis >= self.VIEW_Q_MIN:
                        ps, pe = self._axis_pstart[k], self._axis_pend[k]
                        if ps is not None and pe is not None:
                            d_pix = self._seg_dist(mx, my, ps[0], ps[1], pe[0], pe[1])
                            score = d_pix / self.HOVER_PX_AXIS - 0.02 * vq_axis
                            if score < 1.0:
                                candidates.append((score, 'translate', k))
            else:  # rotate
                for k in range(3):
                    aw = self._axis_world[k]
                    if aw is None: continue
                    vq = abs(float(aw @ cb['fwd']))
                    if vq < 0.03: continue
                    ring_pix = self._ring_pts_pix[k]
                    if ring_pix is None: continue
                    d_pix = self._polyline_min_dist(mx, my, ring_pix)
                    score = d_pix / self.HOVER_PX_RING - 0.02 * vq
                    if score < 1.0:
                        candidates.append((score, 'rotate', k))

            if candidates:
                candidates.sort(key=lambda c: c[0])
                _, self.hover_mode, self.hover_axis = candidates[0]
        except Exception:
            self.hover_mode = None
            self.hover_axis = None

    def _start_drag(self, mode, axis, mx, my):
        self.drag_mode = mode
        self.drag_axis = axis
        self._start_mouse_xy = (mx, my)
        # Remember the grabbed point in the BODY frame so the click sphere
        # rides along with the body for the whole drag.
        try:
            _wp, _R = self._body_pose()
            cp = self._handle_point(mode, axis, mx, my, _wp, _R)
            self.last_click_p_local = None if cp is None else _R.T @ (cp - _wp)
        except Exception:
            self.last_click_p_local = None
        if mode == 'translate':
            self._start_qpos_xyz = self._read_xyz()
            # Sticky axis drag: remember WHERE on the axis the user grabbed
            # (the point on the axis line closest to the click ray) and keep
            # that point under the cursor for the rest of the drag. The line
            # is frozen at press time — re-deriving it from the moving body
            # each event would feed the body's own motion back into the
            # solution and the handle would run away.
            wp0, _ = self._body_pose()
            aw = self._axis_world[axis]
            self._axis_line_o   = wp0.copy() if wp0 is not None else None
            self._axis_line_d   = aw.copy() if aw is not None else None
            self._axis_grab_t   = None
            self._axis_last_off = 0.0
            ray = self._mouse_world_ray(mx, my)
            if ray is not None and self._axis_line_d is not None:
                got = ray_line_closest_t(ray[0], ray[1],
                                         self._axis_line_o, self._axis_line_d)
                if got is not None and got[1] >= self.AXIS_DRAG_SIN_MIN:
                    self._axis_grab_t = got[0]
        elif mode == 'plane':
            self._start_qpos_xyz = self._read_xyz()
            aw = self._axis_world[axis]
            wp, _ = self._body_pose()
            self._plane_normal = aw.copy() if aw is not None else None
            self._plane_origin = wp
            ray = self._mouse_world_ray(mx, my)
            if ray is not None and self._plane_normal is not None:
                self._plane_start_isect = self._ray_plane_intersect(
                    ray[0], ray[1], wp, self._plane_normal)
            else:
                self._plane_start_isect = None
            if self._plane_start_isect is None:
                self.drag_mode = None; self.drag_axis = None
        elif mode == 'rotate':
            self._start_qpos_quat = self._read_quat()
            aw = self._axis_world[axis]
            self._start_axis_world = aw.copy() if aw is not None else None
            wp, R = self._body_pose()
            sc = self._world_to_pixel(wp)
            self._start_screen_center = sc
            cb = self._cam_basis()
            if sc is None or aw is None or cb is None:
                self.drag_mode = None; self.drag_axis = None
            else:
                s = float(np.dot(aw, cb['fwd']))
                self.rot_sign           = 1.0 if s >= 0 else -1.0
                self._rot_prev_screen_a = float(np.arctan2(my - sc[1], mx - sc[0]))
                self.rot_cum_angle      = 0.0
                # In-plane rotate: measure the angle where the cursor ray hits
                # the RING'S OWN plane, so the ring follows the cursor instead
                # of tracking a screen-space angle about the projected centre
                # (which lags badly on a tilted ring). (u, v, aw) is
                # right-handed, so the angle grows counter-clockwise about +aw
                # and needs no sign correction.
                self._rot_plane_o = wp.copy()
                self._rot_plane_u = R[:, (axis + 1) % 3].copy()
                self._rot_plane_v = R[:, (axis + 2) % 3].copy()
                self._rot_prev_plane_a = None
                if abs(s) >= self.RING_DRAG_COS_MIN:
                    ray = self._mouse_world_ray(mx, my)
                    if ray is not None:
                        hit = self._ray_plane_intersect(ray[0], ray[1], wp, aw)
                        if hit is not None:
                            rel = hit - wp
                            self._rot_prev_plane_a = float(np.arctan2(
                                rel @ self._rot_plane_v, rel @ self._rot_plane_u))

    # ────────────────────────── GLFW callbacks ──────────────────────────
    def _on_button(self, window, button, act, mods):
        # Alt+LMB → grasp drag (interpolate env.data.ctrl between open/closed).
        is_alt_lmb = ((button == glfw.MOUSE_BUTTON_LEFT)
                      and (mods & glfw.MOD_ALT)
                      and not (mods & glfw.MOD_CONTROL))
        if is_alt_lmb and self._grasp_enabled:
            if act == glfw.PRESS:
                # Recover t from the current ctrl and continue from it — starting
                # from a stale t slams every joint to that t's ctrl on the first
                # cursor event, making the whole hand jump (hands with a heavy
                # forearm hanging on a weld, like shadow, visibly jitter).
                self._sync_grasp_t_from_ctrl()
                self._grasp_dragging = True
                x, y = glfw.get_cursor_pos(window)
                self._grasp_last_xy = (self.viewer._scale * x, self.viewer._scale * y)
                return
            if act == glfw.RELEASE and self._grasp_dragging:
                self._grasp_dragging = False
                self._grasp_last_xy  = None
                return

        # Plain LMB (no Ctrl, no Alt) → axis-constrained gizmo drag on hover.
        is_plain_lmb = ((button == glfw.MOUSE_BUTTON_LEFT)
                        and not (mods & (glfw.MOD_CONTROL | glfw.MOD_ALT)))
        if is_plain_lmb:
            if act == glfw.PRESS and self.hover_mode is not None and self.drag_mode is None:
                try:
                    wp0, _ = self._body_pose()
                except Exception:
                    wp0 = None
                mx, my = self._pointer_pixel(wp0)
                self._start_drag(self.hover_mode, self.hover_axis, mx, my)
                return
            if act == glfw.RELEASE and self.drag_mode is not None:
                self.drag_mode = None
                self.drag_axis = None
                self.last_click_p_local = None
                return
        if self._prev_btn_cb is not None:
            self._prev_btn_cb(window, button, act, mods)

    def _on_cursor(self, window, xpos, ypos):
        cx, cy = self.viewer._scale * xpos, self.viewer._scale * ypos

        # Grasp drag (Alt+LMB) takes priority over gizmo drag — Alt+LMB
        # cannot start a gizmo drag because is_plain_lmb excludes MOD_ALT.
        if self._grasp_dragging and self._grasp_last_xy is not None:
            dy = cy - self._grasp_last_xy[1]
            dt = -dy * self.GRASP_SENS  # mouse up → close
            if self._grasp_active_idx < 0:
                self._grasp_t_per_joint = np.clip(
                    self._grasp_t_per_joint + dt, 0.0, 1.0)
            else:
                i = int(self._grasp_active_idx)
                self._grasp_t_per_joint[i] = float(np.clip(
                    self._grasp_t_per_joint[i] + dt, 0.0, 1.0))
            self._apply_grasp()
            self._grasp_last_xy = (cx, cy)
            return  # consume; no fallthrough

        mode = self.drag_mode
        if mode is None:
            if self._prev_cur_cb is not None:
                self._prev_cur_cb(window, xpos, ypos)
            return
        a = self.drag_axis
        # Drive the drag from the virtual pointer — the blue sphere the user
        # is actually watching, not the raw cursor pixel.
        cx, cy = self._pointer_pixel(self._drag_ref_point())

        if mode == 'translate':
            aworld = self._axis_line_d if self._axis_line_d is not None else self._axis_world[a]
            if aworld is None or self._start_qpos_xyz is None:
                return
            # Preferred: the grabbed point on the axis stays under the cursor.
            if self._axis_grab_t is not None:
                ray = self._mouse_world_ray(cx, cy)
                got = (ray_line_closest_t(ray[0], ray[1], self._axis_line_o, aworld)
                       if ray is not None else None)
                if got is not None and got[1] >= self.AXIS_DRAG_SIN_MIN:
                    self._axis_last_off = got[0] - self._axis_grab_t
                # Axis nearly edge-on to the view ray → the closest-point
                # solution is numerically meaningless; hold the last offset
                # instead of letting the body shoot off to infinity.
                self._write_xyz(self._start_qpos_xyz + self._axis_last_off * aworld)
                return
            # Fallback (grab ray was degenerate at press): linearised pixel
            # projection along the axis' screen direction — the old behaviour.
            sx, sy = self._start_mouse_xy
            dx, dy = cx - sx, cy - sy
            adir   = self._axis_pix_dir[a]
            alen   = self._axis_pix_len[a]
            if (adir is not None) and (alen > 1e-3):
                proj_pix = dx * adir[0] + dy * adir[1]
                offset   = proj_pix * (self.AXIS_LEN / alen)
                self._write_xyz(self._start_qpos_xyz + offset * aworld)
            return

        if mode == 'plane':
            si = self._plane_start_isect
            pn = self._plane_normal
            po = self._plane_origin
            sq = self._start_qpos_xyz
            if si is None or pn is None or sq is None: return
            ray = self._mouse_world_ray(cx, cy)
            if ray is None: return
            ci = self._ray_plane_intersect(ray[0], ray[1], po, pn)
            if ci is None: return
            self._write_xyz(sq + (ci - si))
            return

        if mode == 'rotate':
            sc = self._start_screen_center
            aw = self._start_axis_world
            sq = self._start_qpos_quat
            if sc is None or aw is None or sq is None: return
            # In-plane angle when the ring faces us enough; screen angle
            # otherwise (edge-on ring → the plane hit is unusable).
            in_plane = None
            if self._rot_prev_plane_a is not None:
                ray = self._mouse_world_ray(cx, cy)
                hit = (self._ray_plane_intersect(ray[0], ray[1], self._rot_plane_o, aw)
                       if ray is not None else None)
                if hit is not None:
                    rel = hit - self._rot_plane_o
                    in_plane = float(np.arctan2(rel @ self._rot_plane_v,
                                                rel @ self._rot_plane_u))
            if in_plane is not None:
                da = in_plane - self._rot_prev_plane_a
                if da >  np.pi: da -= 2*np.pi
                elif da < -np.pi: da += 2*np.pi
                self.rot_cum_angle     += da
                self._rot_prev_plane_a  = in_plane
                # Keep the screen-angle reference live so a mid-drag fallback
                # (ring swinging edge-on) resumes without a jump.
                self._rot_prev_screen_a = float(np.arctan2(cy - sc[1], cx - sc[0]))
                angle = self.rot_cum_angle
            else:
                cur_a  = float(np.arctan2(cy - sc[1], cx - sc[0]))
                da = cur_a - self._rot_prev_screen_a
                if da >  np.pi: da -= 2*np.pi
                elif da < -np.pi: da += 2*np.pi
                self.rot_cum_angle      += self.rot_sign * da
                self._rot_prev_screen_a  = cur_a
                angle = self.rot_cum_angle
            q_axis = np.zeros(4, dtype=np.float64)
            mujoco.mju_axisAngle2Quat(q_axis, aw.astype(np.float64), float(angle))
            q_out  = np.zeros(4, dtype=np.float64)
            mujoco.mju_mulQuat(q_out, q_axis, sq.astype(np.float64))
            self._write_quat(q_out)
            return

    def _on_scroll(self, window, x_offset, y_offset):
        """Alt+wheel cycles the active grasp target through
        ``[all_joint, joint_0, joint_1, …, joint_last]``. Plain wheel falls
        through so the viewer's zoom keeps working."""
        try:
            alt_held = (
                glfw.get_key(window, glfw.KEY_LEFT_ALT)  == glfw.PRESS or
                glfw.get_key(window, glfw.KEY_RIGHT_ALT) == glfw.PRESS
            )
        except Exception:
            alt_held = False
        if alt_held and self._grasp_enabled:
            step = 1 if y_offset > 0 else (-1 if y_offset < 0 else 0)
            if step != 0:
                self.cycle_grasp_target(step)
                return  # consume; do NOT zoom
        if self._prev_scroll_cb is not None:
            self._prev_scroll_cb(window, x_offset, y_offset)

    # ────────────────────────── grasp drag (Alt+LMB / Alt+wheel) ──────────────────────────
    @property
    def grasp_t(self) -> float:
        """Active joint's grasp progress (or mean of all in 'all' mode)."""
        if self._grasp_t_per_joint is None or len(self._grasp_t_per_joint) == 0:
            return 0.0
        if self._grasp_active_idx < 0:
            return float(np.mean(self._grasp_t_per_joint))
        return float(self._grasp_t_per_joint[self._grasp_active_idx])

    @property
    def grasp_t_per_joint(self) -> np.ndarray:
        if self._grasp_t_per_joint is None:
            return np.zeros(0)
        return self._grasp_t_per_joint.copy()

    @property
    def grasp_active_idx(self) -> int:
        """-1 = all joints; otherwise index into ``grasp_drag_idxs``."""
        return int(self._grasp_active_idx)

    @property
    def grasp_active_label(self) -> str:
        if (not self._grasp_enabled) or self._grasp_idxs is None or self._grasp_active_idx < 0:
            return "all"
        ctrl_idx = int(self._grasp_idxs[self._grasp_active_idx])
        try:
            name = self.env.ctrl_names[ctrl_idx]
        except Exception:
            name = f"#{ctrl_idx}"
        return f"[{self._grasp_active_idx}] {name}"

    def set_grasp_t(self, t: float, idx=None):
        """Programmatically set grasp progress. ``idx=None`` broadcasts to all
        joints; otherwise sets only ``grasp_drag_idxs[idx]``."""
        if not self._grasp_enabled or self._grasp_t_per_joint is None: return
        t = float(np.clip(t, 0.0, 1.0))
        if idx is None:
            self._grasp_t_per_joint[:] = t
        else:
            self._grasp_t_per_joint[int(idx)] = t
        self._apply_grasp()

    def cycle_grasp_target(self, step: int = 1):
        """Cycle the active grasp target through ``[-1, 0, 1, …, n−1]``
        (i.e. ``[all, joint_0, joint_1, …, joint_last]``)."""
        if self._grasp_idxs is None: return
        n = len(self._grasp_idxs)
        if n == 0:
            self._grasp_active_idx = -1
            return
        cur = self._grasp_active_idx + 1     # 0..n
        cur = (cur + int(step)) % (n + 1)
        self._grasp_active_idx = cur - 1     # back to -1..n-1

    def _read_grasp_ctrl_cur(self):
        """Current ctrl values at ``self._grasp_idxs`` (1D, len == n_grasp).
        CPU backend reads ``env.data.ctrl``; WarpBodyHandle overrides to read
        the selected world's ``d.ctrl``. ``None`` → sync skipped."""
        try:
            return np.asarray(self.env.data.ctrl, dtype=float)[self._grasp_idxs]
        except Exception:
            return None

    def _sync_grasp_t_from_ctrl(self):
        """On Alt+LMB press, recover ``grasp_t_per_joint`` from the current ctrl.

        A drag must always **continue from the hand's current pose**. Reusing the
        previous drag's t (initial 0 = ctrl_min for all joints) snaps every joint
        to that t's ctrl on the first cursor event — in a headless reproduction
        the shadow fingertip jumped 33 mm per frame (9 mm when synced). Joints
        with span≈0 keep their previous t."""
        if self._grasp_idxs is None or len(self._grasp_idxs) == 0:
            return
        cur = self._read_grasp_ctrl_cur()
        if cur is None or len(cur) != len(self._grasp_idxs):
            return
        idxs = self._grasp_idxs
        span = self._ctrl_closed[idxs] - self._ctrl_open[idxs]
        safe = np.where(np.abs(span) > 1e-9, span, 1.0)
        t    = (np.asarray(cur, dtype=float) - self._ctrl_open[idxs]) / safe
        self._grasp_t_per_joint = np.where(
            np.abs(span) > 1e-9,
            np.clip(t, 0.0, 1.0),
            self._grasp_t_per_joint,
        )

    def _apply_grasp(self):
        """Write ``(1-t_i)·ctrl_open[i] + t_i·ctrl_closed[i]`` for every i in
        ``grasp_drag_idxs`` to ``env.data.ctrl``, clamped per-joint to the
        actuator's ctrl range.

        If ``slider_widget`` was provided, its corresponding entries are also
        updated (offset by ``slider_ctrl_offset``) — without this, a notebook
        that copies slider → ``env.data.ctrl`` every sim step would silently
        overwrite our grasp values, leaving the joints stationary even though
        ``grasp_t`` changes.

        Subclasses (e.g. WarpBodyHandle) may override to write to
        ``d.ctrl[w, ...]`` instead.
        """
        if (not self._grasp_enabled) or self._grasp_idxs is None: return
        idxs = self._grasp_idxs
        t    = self._grasp_t_per_joint
        ctrl_target = (1.0 - t) * self._ctrl_open[idxs] + t * self._ctrl_closed[idxs]
        ctrl_target = np.clip(ctrl_target, self._ctrl_min[idxs], self._ctrl_max[idxs])
        try:
            self.env.data.ctrl[idxs] = ctrl_target
        except Exception:
            pass
        # Mirror into slider widget so the notebook's sim step doesn't
        # immediately overwrite our values from stale slider state.
        # Throttling: only call ``set_values`` when at least one value
        # actually changes — calling Qt's set_values on every cursor event
        # forces a UI round-trip per pixel of mouse motion, which is both
        # expensive and a source of intermittent native crashes (Qt paint
        # events scheduling against the GLFW/GL render).
        if self._slider_widget is not None:
            try:
                vals = list(self._slider_widget.get_values())
                changed = False
                for i, ctrl_idx in enumerate(idxs):
                    sl_idx = self._slider_ctrl_offset + int(ctrl_idx)
                    if 0 <= sl_idx < len(vals):
                        new_v = float(ctrl_target[i])
                        if abs(float(vals[sl_idx]) - new_v) > 1e-6:
                            vals[sl_idx] = new_v
                            changed = True
                if changed:
                    self._slider_widget.set_values(vals)
            except Exception:
                pass

    # ────────────────────────── drawing ──────────────────────────
    def _draw_translate(self, p, R, hover_mode, hover_axis):
        draw_translate_gizmo(
            self.env, p, R,
            hover_mode=hover_mode, hover_axis=hover_axis,
            axis_len=self.AXIS_LEN, plane_off=self.PLANE_OFFSET,
            plane_size=self.PLANE_SIZE, plane_thick=self.PLANE_THICK,
            axis_colors=self.BASE_AXIS_COLORS,
            plane_colors=self.BASE_PLANE_COLORS,
            hi_color=self.HI_COLOR, hi_plane=self.HI_PLANE_COLOR,
            labels=self.AXIS_LABELS, hover_scale=self.HOVER_SCALE,
        )

    def _draw_rotate(self, p, R, hover_mode, hover_axis):
        draw_rotate_gizmo(
            self.env, p, R,
            hover_mode=hover_mode, hover_axis=hover_axis,
            ring_r=self.RING_R, ring_segs=self.RING_SEGS,
            ring_thick=self.RING_THICK,
            ring_colors=self.BASE_RING_COLORS,
            hi_color=self.HI_COLOR, hover_scale=self.HOVER_SCALE,
        )


# ──────────────────────────────────────────────────────────────────────────────
# WarpBodyHandle — BodyHandle subclass that drives a body in a parallel mjwarp
# env. Visualises at the body's xpos plus the per-world grid offset, and
# reads/writes ``d.qpos`` / ``d.mocap_pos`` (with ``wp.copy``) instead of the
# CPU env.data. Plain LMB-drag on the gizmo handles is fully interactive
# (axis / plane translate, ring rotate) — same hover & drag logic as the CPU
# BodyHandle, only the pose backend differs.
# ──────────────────────────────────────────────────────────────────────────────
class WarpBodyHandle(BodyHandle):
    """Interactive gizmo for a body in a specific parallel warp world."""

    def __init__(self,
                 env_warp,
                 mjwarp_data,
                 body,
                 world_idx:        int,
                 grid_offsets,
                 mode:             str = 'auto',
                 label:            str = None,
                 **kwargs):
        # Provide warp-aware members BEFORE calling super().__init__ because
        # _resolve_mode and the constructor body don't need them, but later
        # calls to _body_pose / _read_* / _write_* will.
        self.d            = mjwarp_data
        self.world_idx    = int(world_idx)
        self.grid_offsets = np.asarray(grid_offsets, dtype=float)
        super().__init__(env_warp, body, mode=mode, label=label, **kwargs)

    # ── body pose: include per-world grid offset in XY for the gizmo center ──
    def _body_pose(self):
        ox, oy = self.grid_offsets[self.world_idx]
        local  = self.d.xpos.numpy()[self.world_idx, self.body_id]
        wp     = np.array([local[0] + ox, local[1] + oy, local[2]], dtype=float)
        R      = np.asarray(
            self.d.xmat.numpy()[self.world_idx, self.body_id], dtype=float
        ).reshape(3, 3)
        return wp, R

    # ── pose I/O: use warp arrays via wp.copy. World ↔ logical (offset) frame
    #    conversion happens here so the shared drag math in BodyHandle (which
    #    operates on world coordinates) stays untouched. ────────────────────
    def _offset_xy(self):
        ox, oy = self.grid_offsets[self.world_idx]
        return np.array([ox, oy, 0.0], dtype=float)

    def _read_xyz(self):
        if self._mode == 'mocap':
            local = self.d.mocap_pos.numpy()[self.world_idx, self._mocap_id]
        else:
            local = self.d.qpos.numpy()[self.world_idx,
                                        self.qpos_xyz_adr:self.qpos_xyz_adr+3]
        return np.asarray(local, dtype=float) + self._offset_xy()

    def _read_quat(self):
        if self._mode == 'mocap':
            return self.d.mocap_quat.numpy()[self.world_idx, self._mocap_id].copy()
        return self.d.qpos.numpy()[self.world_idx,
                                   self.qpos_quat_adr:self.qpos_quat_adr+4].copy()

    def _read_grasp_ctrl_cur(self):
        """Warp backend: read the current grasp ctrl from the selected world's
        ``d.ctrl`` (for recovering grasp_t on Alt+LMB press — the base class uses env.data.ctrl)."""
        try:
            return self.d.ctrl.numpy()[self.world_idx, self._grasp_idxs].astype(float)
        except Exception:
            return None

    def _write_xyz(self, xyz_world):
        local = np.asarray(xyz_world, dtype=float) - self._offset_xy()
        if self._mode == 'mocap':
            mp = self.d.mocap_pos.numpy().copy()
            mp[self.world_idx, self._mocap_id] = local
            wp.copy(self.d.mocap_pos,
                    wp.array(mp,
                             dtype=self.d.mocap_pos.dtype,
                             device=self.d.mocap_pos.device))
        else:
            qp = self.d.qpos.numpy().copy()
            qp[self.world_idx, self.qpos_xyz_adr:self.qpos_xyz_adr+3] = local
            wp.copy(self.d.qpos,
                    wp.array(qp,
                             dtype=self.d.qpos.dtype,
                             device=self.d.qpos.device))
            # Zero linear qvel of the dragged free joint to prevent carrying
            # accumulated momentum into a kinematic teleport.
            qv = self.d.qvel.numpy().copy()
            qv[self.world_idx, self.qpos_xyz_adr:self.qpos_xyz_adr+6] = 0.0
            wp.copy(self.d.qvel,
                    wp.array(qv,
                             dtype=self.d.qvel.dtype,
                             device=self.d.qvel.device))

    def _write_quat(self, quat_wxyz):
        q = np.asarray(quat_wxyz, dtype=float)
        if self._mode == 'mocap':
            mq = self.d.mocap_quat.numpy().copy()
            mq[self.world_idx, self._mocap_id] = q
            wp.copy(self.d.mocap_quat,
                    wp.array(mq,
                             dtype=self.d.mocap_quat.dtype,
                             device=self.d.mocap_quat.device))
        else:
            qp = self.d.qpos.numpy().copy()
            qp[self.world_idx, self.qpos_quat_adr:self.qpos_quat_adr+4] = q
            wp.copy(self.d.qpos,
                    wp.array(qp,
                             dtype=self.d.qpos.dtype,
                             device=self.d.qpos.device))
            qv = self.d.qvel.numpy().copy()
            qv[self.world_idx, self.qpos_xyz_adr+3:self.qpos_xyz_adr+6] = 0.0
            wp.copy(self.d.qvel,
                    wp.array(qv,
                             dtype=self.d.qvel.dtype,
                             device=self.d.qvel.device))


# ──────────────────────────────────────────────────────────────────────────────
# RightDragForce — RMB drag → spring-style xfrc_applied on a candidate body.
# ──────────────────────────────────────────────────────────────────────────────
class RightDragForce:
    """
    Right-mouse-button drag on a candidate body applies a spring-style external
    force via ``data.xfrc_applied[body_id, :3]``.

    On RMB press, ``mujoco.mj_ray`` identifies the body under the cursor; if it
    is in ``candidate_body_names`` the press is consumed and a drag begins. The
    initial hit point and camera forward vector define a screen-aligned plane;
    each cursor update intersects the new mouse ray with that plane to produce
    a world-space target, and the applied force is ``K * (target - body_xpos)``
    (clipped to ``max_force``). On release the force is cleared.

    Chains with the viewer's existing GLFW callbacks (and with a ``BodyHandle``
    if attached first) so LMB/Ctrl+RMB pass through unchanged.

    Usage::

        rf = RightDragForce(env, candidate_body_names=obj_names).attach()
    """
    def __init__(self, env, candidate_body_names, k=200.0, max_force=300.0):
        self.env    = env
        self.viewer = env.viewer
        self.candidate_body_ids = {env.body_names.index(n) for n in candidate_body_names}
        self.K          = float(k)
        self.MAX_F      = float(max_force)
        self._dragging  = False
        self._body_id   = None
        self._hit_pt0   = None
        self._cam_n0    = None
        self._prev_btn_cb = None
        self._prev_cur_cb = None
        self._attached    = False

    def attach(self):
        if self._attached: return self
        # Use return value of glfw.set_*_callback to chain to whatever handler
        # is currently registered (could be the viewer or another stacked
        # handle), not ``viewer._mouse_button_callback`` which is just the
        # bound method and would silently skip already-attached handlers.
        self._prev_btn_cb = glfw.set_mouse_button_callback(
            self.viewer.window, self._on_button)
        self._prev_cur_cb = glfw.set_cursor_pos_callback(
            self.viewer.window, self._on_cursor)
        self._attached = True
        return self

    def detach(self):
        if not self._attached: return self
        win = getattr(self.viewer, "window", None)
        if win is not None:
            try:
                glfw.set_mouse_button_callback(win, self._prev_btn_cb)
                glfw.set_cursor_pos_callback   (win, self._prev_cur_cb)
            except Exception:
                pass
            try:
                self._clear_force()
            except Exception:
                pass
        self._attached = False
        return self

    # Exact camera model shared with BodyHandle — see viewer_cam_basis().
    def _cam_basis(self):
        return viewer_cam_basis(self.viewer)

    def _ray_from_mouse(self, mx, my):
        return cam_mouse_ray(self._cam_basis(), mx, my)

    # Thin alias so the drag code reads the same in every handle class.
    _ray_plane_intersect = staticmethod(ray_plane_intersect)

    def _clear_force(self):
        if self._body_id is not None:
            self.env.data.xfrc_applied[self._body_id] = 0.0

    def _on_button(self, window, button, act, mods):
        if button == glfw.MOUSE_BUTTON_RIGHT and not (mods & glfw.MOD_CONTROL):
            mx_raw, my_raw = glfw.get_cursor_pos(window)
            mx = self.viewer._scale * mx_raw
            my = self.viewer._scale * my_raw
            if act == glfw.PRESS:
                ray = self._ray_from_mouse(mx, my)
                if ray is None:
                    if self._prev_btn_cb is not None:
                        self._prev_btn_cb(window, button, act, mods)
                    return
                ro, rd = ray
                geomid = np.array([-1], dtype=np.int32)
                t = mujoco.mj_ray(
                    self.env.model, self.env.data,
                    ro.astype(np.float64), rd.astype(np.float64),
                    None, 1, -1, geomid,
                )
                if geomid[0] < 0 or t < 0:
                    if self._prev_btn_cb is not None:
                        self._prev_btn_cb(window, button, act, mods)
                    return
                body_id = int(self.env.model.geom_bodyid[geomid[0]])
                if body_id not in self.candidate_body_ids:
                    if self._prev_btn_cb is not None:
                        self._prev_btn_cb(window, button, act, mods)
                    return
                cb = self._cam_basis()
                if cb is None: return
                self._dragging = True
                self._body_id  = body_id
                self._hit_pt0  = ro + t * rd
                self._cam_n0   = cb['fwd'].copy()
                return  # consume
            if act == glfw.RELEASE and self._dragging:
                self._clear_force()
                self._dragging = False
                self._body_id  = None
                return
        if self._prev_btn_cb is not None:
            self._prev_btn_cb(window, button, act, mods)

    def _on_cursor(self, window, xpos, ypos):
        if self._dragging and self._body_id is not None:
            cx = self.viewer._scale * xpos
            cy = self.viewer._scale * ypos
            ray = self._ray_from_mouse(cx, cy)
            if ray is not None:
                ro, rd = ray
                target = self._ray_plane_intersect(ro, rd, self._hit_pt0, self._cam_n0)
                if target is not None:
                    bxp = self.env.data.xpos[self._body_id]
                    f = self.K * (target - bxp)
                    fn = float(np.linalg.norm(f))
                    if fn > self.MAX_F:
                        f = f * (self.MAX_F / fn)
                    self.env.data.xfrc_applied[self._body_id, :3] = f
                    self.env.data.xfrc_applied[self._body_id, 3:] = 0.0
        if self._prev_cur_cb is not None:
            self._prev_cur_cb(window, xpos, ypos)


def find_object_body_under_xyz(env, xyz, candidate_body_names, max_dist=0.15):
    """Return the candidate body name whose ``xpos`` is closest to ``xyz``
    (within ``max_dist``), or ``None``. Useful for "double-click to switch
    interactive-handle target" flows."""
    if xyz is None or len(candidate_body_names) == 0:
        return None
    xpos = np.array([env.data.xpos[env.body_names.index(n)] for n in candidate_body_names])
    d = np.linalg.norm(xpos - np.asarray(xyz), axis=1)
    i = int(np.argmin(d))
    return candidate_body_names[i] if d[i] <= max_dist else None


# ──────────────────────────────────────────────────────────────────────────────
# WarpInteractiveHandle — pick a (world, body) via double-click, RMB-drag force
# ──────────────────────────────────────────────────────────────────────────────
class WarpInteractiveHandle:
    """
    Click-to-select + RMB-drag external-force interaction for a body in a
    parallel MuJoCo Warp env.

    Single- vs multi-hand
    ---------------------
    Single-hand scenes bind one wrist mocap (``wrist_mocap_name="…:mocap"``)
    and let the grasp drag act on every ctrl. Two-hand (leader-follower)
    scenes pass **per-hand** info so each hand is driven independently:

      - ``wrist_mocap_name`` accepts a ``str`` OR a ``list[str]``. A list binds
        every named mocap body (``body_id → mocap_id`` map), so ANY of the
        wrists can be Ctrl-dragged once selected.
      - ``grasp_idxs_by_body`` (``{body_name: ctrl_idxs}``) scopes the Alt+LMB
        grasp to the SELECTED hand — see *Grasp drag* below.

    Selection
    ---------
    Driven by an externally-supplied world XYZ (typically from
    ``HandRLParserClass.get_xyz_left_double_click_from_framebuffer``):
      - The closest grid-offset row of ``grid_offsets`` determines
        ``world_idx``.
      - Subtracting that offset yields the click in the body's logical frame.
      - The closest candidate body in ``d.xpos[world_idx]`` becomes
        ``body_id`` (rejected if farther than ``max_pick_dist``).

    Kinematic translate (Ctrl + RMB drag)
    -------------------------------------
    With a body selected and either a free joint or a bound mocap
    (``wrist_mocap_name``; str or list — the selected body's own mocap_id is
    looked up in ``_mocap_id_by_body``), **Ctrl + RMB**-press starts a
    translate drag. Cursor pixel deltas accumulate into ``kin_refpos`` along
    the camera-aligned plane at the body depth, and each event writes the new
    world position to ``d.mocap_pos`` (wrist) or ``d.qpos[w, qpa:qpa+3]``
    (free joint). ``qvel`` is zeroed on every cursor event for free-joint
    targets so prior momentum cannot contaminate the kinematic teleport.

    Kinematic rotate (Ctrl + LMB drag)
    ----------------------------------
    With a body selected (same conditions as above), **Ctrl + LMB**-press
    starts a trackball-style rotation drag. Each cursor delta produces a
    small rotation around an axis combining camera-up (driven by horizontal
    cursor) and camera-right (vertical cursor); the resulting incremental
    quaternion is composed onto the body's current quaternion (mocap_quat
    or qpos[qpa+3:qpa+7]). Sensitivity is ``ROTATE_SENS`` rad/pixel.

    These two bindings (Ctrl+LMB rotate, Ctrl+RMB translate) match MuJoCo's
    standard perturb convention and the gizmo convention used by 13 / 15
    notebooks (``BodyHandle`` + ``RightDragForce``).

    Grasp drag (Alt + LMB drag)
    ---------------------------
    With a *world* selected, **Alt + LMB**-drag changes a per-joint
    interpolation parameter ``grasp_t_per_joint[i] ∈ [0, 1]`` driven by
    vertical mouse delta (up = close). The selected world's
    ``d.ctrl[w, grasp_drag_idxs[i]]`` is set to
    ``(1 − t_i)·ctrl_open[i] + t_i·ctrl_closed[i]`` and then clamped to
    each actuator's own ctrlrange. By default ``ctrl_open`` / ``ctrl_closed``
    are the per-joint ``ctrl_min`` / ``ctrl_max`` so a t-sweep covers the
    full range of every joint independently; pass explicit arrays for
    taxonomy-shaped grasps. Grasp progress persists across drag releases.

    **Per-hand scoping** (``grasp_idxs_by_body``): without it the grasp acts on
    every ctrl in ``grasp_drag_idxs`` — in a two-hand scene that closes BOTH
    hands. Pass ``{wrist_body_name: ctrl_idxs}`` to restrict the drag (both the
    ``t`` update and the ctrl write) to the SELECTED hand's actuators, so only
    the picked hand closes. When the map is set and the selection is not a
    mapped hand (an object, or nothing), the grasp is a no-op. See
    ``_active_grasp_positions``.

    Grasp target selector (Alt + mouse wheel)
    -----------------------------------------
    **Alt + wheel** cycles the active grasp target through
    ``[all_joint, joint_0, joint_1, …, joint_last]`` (``grasp_active_idx``).
    When set to ``-1`` (``all``) Alt+LMB drag updates every joint; otherwise
    it updates only that single joint. Plain wheel still zooms the camera.

    Render-loop integration::

        ih = WarpInteractiveHandle(
                env_warp, d, candidate_body_names=[wrist_body] + obj_names,
                grid_offsets=grid_offsets,
             ).attach()
        try:
            while env_warp.is_viewer_alive():
                ...
                if tmr_render.do_run():
                    ih.draw_highlight()                  # before mjwarp_render
                    env_warp.mjwarp_render(...)
                    xyz, flag = env_warp.get_xyz_left_double_click_from_framebuffer()
                    if flag:
                        ih.update_selection(xyz)
        finally:
            ih.detach()
    """

    def __init__(self,
                 env_warp,
                 mjwarp_data,
                 candidate_body_names,
                 grid_offsets,
                 wrist_mocap_name = None,    # str OR list[str] — mocap body(ies) for Ctrl kinematic drag (e.g. leader+follower).
                 ctrl_open        = None,    # (nu,) "open hand" ctrl baseline. Default: ctrl_min per joint.
                 ctrl_closed      = None,    # (nu,) "closed hand" ctrl baseline. Default: ctrl_max per joint.
                 grasp_drag_idxs  = None,    # ctrl indices the grasp drag controls (default: all).
                 grasp_idxs_by_body = None,  # {body_name: ctrl idxs} — scope Alt+LMB grasp to the selected hand (e.g. leader/follower).
                 grasp_sensitivity:float = 0.005,  # Δ_t per pixel of vertical mouse drag.
                 max_pick_dist:   float = 0.20,
                 highlight_radius:float = 0.020,
                 highlight_rgba   = (1.0, 1.0, 0.2, 0.55),
                 # ── Gizmo display (uses shared draw_translate_gizmo /
                 #     draw_rotate_gizmo helpers, decorative only) ──
                 gizmo_draw_mode: str  = 'both',   # 'translate' | 'rotate' | 'both' | 'off'
                 gizmo_axis_len:  float = 0.10,
                 gizmo_plane_off: float = 0.035,
                 gizmo_plane_size:float = 0.030,
                 gizmo_ring_r:    float = 0.075,
                 gizmo_ring_thick:float = 0.0022):
        self.env    = env_warp
        self.viewer = env_warp.viewer
        self.d      = mjwarp_data
        self.candidate_body_names = list(candidate_body_names)
        self.candidate_body_ids   = [env_warp.body_names.index(n) for n in candidate_body_names]
        self.grid_offsets         = np.asarray(grid_offsets, dtype=float)  # (NWORLD, 2)
        self.MAX_PICK_DIST  = float(max_pick_dist)
        self.HIGHLIGHT_R    = float(highlight_radius)
        self.HIGHLIGHT_RGBA = tuple(highlight_rgba)
        # Gizmo geometry params
        self.GIZMO_DRAW_MODE  = str(gizmo_draw_mode)
        self.GIZMO_AXIS_LEN   = float(gizmo_axis_len)
        self.GIZMO_PLANE_OFF  = float(gizmo_plane_off)
        self.GIZMO_PLANE_SIZE = float(gizmo_plane_size)
        self.GIZMO_RING_R     = float(gizmo_ring_r)
        self.GIZMO_RING_THICK = float(gizmo_ring_thick)
        # Optional wrist mocap binding(s) for kinematic Ctrl drag. Accepts a
        # single name (single-hand) OR a list of names (e.g. leader + follower
        # two-hand scene) — every named mocap body becomes independently
        # Ctrl-draggable. ``_mocap_id_by_body`` (body_id → mocap_id) is the
        # source of truth; the legacy scalar attrs mirror the first binding.
        self._mocap_id_by_body = {}
        if wrist_mocap_name is not None:
            _names = ([wrist_mocap_name] if isinstance(wrist_mocap_name, str)
                      else list(wrist_mocap_name))
            for _nm in _names:
                if _nm is None or _nm not in env_warp.body_names:
                    continue
                wbid = env_warp.body_names.index(_nm)
                mid  = int(env_warp.model.body_mocapid[wbid])
                if mid >= 0:
                    self._mocap_id_by_body[wbid] = mid
        if self._mocap_id_by_body:
            _first_bid = next(iter(self._mocap_id_by_body))
            self.wrist_mocap_body_id = _first_bid
            self.wrist_mocap_id      = self._mocap_id_by_body[_first_bid]
        else:
            self.wrist_mocap_body_id = -1
            self.wrist_mocap_id      = -1
        # Free-joint qposadr per candidate body (-1 if none).
        self._qposadr_per_body = {}
        for bid in self.candidate_body_ids:
            self._qposadr_per_body[bid] = self._find_free_joint_qposadr(env_warp.model, bid)
        # selection state
        self.world_idx = None
        self.body_id   = None
        # ── inner BodyHandle for axis-constrained gizmo drag ─────────────
        # Attached lazily on selection (see _reattach_body_handle).
        self._body_handle = None
        # ── drag state — Ctrl+RMB (kinematic translate of mocap/qpos) ───
        self._translate_dragging = False
        self._kin_refpos         = None      # world-space target position
        # Sticky-drag anchors, frozen at press (see _on_button / _on_cursor).
        self._kin_plane_n        = None      # camera-facing plane normal
        self._kin_plane_o        = None      # plane origin (body pos at press)
        self._kin_grab_isect     = None      # where the click ray met that plane
        self._kin_start_pos      = None      # body position at press
        # ── drag state — Ctrl+LMB (kinematic rotate of mocap/qpos) ──────
        self._rotate_dragging    = False
        # ── drag state — Alt+LMB (grasp ctrl interpolation) ─────────────
        self._grasp_dragging     = False
        # ── rotation sensitivity ────────────────────────────────────────
        self.ROTATE_SENS         = float(np.deg2rad(0.4))   # rad per pixel
        # ── grasp configuration ────────────────────────────────────────
        nu = int(env_warp.n_ctrl) if hasattr(env_warp, "n_ctrl") else len(env_warp.ctrl_names)
        ctrl_min_arr = np.asarray(env_warp.ctrl_mins, dtype=float)
        ctrl_max_arr = np.asarray(env_warp.ctrl_maxs, dtype=float)
        # Default open/closed track each joint's own ctrlrange so a t∈[0,1]
        # sweep traverses the full range *per joint*. (A global "0.7·ctrl_max"
        # baseline collapsed to a single value for joints whose range was
        # negative or zero-crossing, leaving them stuck.) Pass explicit arrays
        # for taxonomy-shaped grasps.
        if ctrl_open is None:
            ctrl_open = ctrl_min_arr.copy()
        if ctrl_closed is None:
            ctrl_closed = ctrl_max_arr.copy()
        self._ctrl_open   = np.asarray(ctrl_open,   dtype=float)[:nu]
        self._ctrl_closed = np.asarray(ctrl_closed, dtype=float)[:nu]
        self._ctrl_min    = ctrl_min_arr[:nu]
        self._ctrl_max    = ctrl_max_arr[:nu]
        self._grasp_idxs  = (np.arange(nu, dtype=int) if grasp_drag_idxs is None
                             else np.asarray(grasp_drag_idxs, dtype=int))
        self.GRASP_SENS   = float(grasp_sensitivity)
        # Optional per-body grasp scoping: {body_id -> positions into
        # ``_grasp_idxs``}. When the selected body is in this map, the Alt+LMB
        # grasp drag (update AND apply) touches only that hand's ctrls, so a
        # two-hand (leader/follower) scene closes just the selected hand instead
        # of both. Empty → legacy behaviour (grasp acts on every ctrl).
        self._grasp_pos_by_body = {}
        if grasp_idxs_by_body:
            _pos_of_ctrl = {int(c): p for p, c in enumerate(self._grasp_idxs)}
            for _nm, _cidxs in grasp_idxs_by_body.items():
                if _nm is None or _nm not in env_warp.body_names:
                    continue
                _bid = env_warp.body_names.index(_nm)
                _pos = [_pos_of_ctrl[int(c)] for c in np.asarray(_cidxs).ravel()
                        if int(c) in _pos_of_ctrl]
                if _pos:
                    self._grasp_pos_by_body[_bid] = np.asarray(_pos, dtype=int)
        # Per-joint grasp progress (one t per ctrl in self._grasp_idxs).
        # Persists across drag releases.
        self._grasp_t_per_joint = np.zeros(len(self._grasp_idxs), dtype=float)
        # Active selector. -1 → all joints; 0..n_grasp-1 → that single joint.
        # Cycled by Alt + mouse wheel scroll.
        self._grasp_active_idx = -1
        # shared cursor tracking
        self._last_cursor_xy = None
        # callback bookkeeping
        self._prev_btn_cb    = None
        self._prev_cur_cb    = None
        self._prev_scroll_cb = None
        self._attached       = False

    @staticmethod
    def _find_free_joint_qposadr(model, body_id):
        for jid in range(model.njnt):
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE \
               and int(model.jnt_bodyid[jid]) == body_id:
                return int(model.jnt_qposadr[jid])
        return -1

    # ────────────────────────── lifecycle ──────────────────────────
    def attach(self):
        if self._attached: return self
        self._prev_btn_cb = glfw.set_mouse_button_callback(
            self.viewer.window, self._on_button)
        self._prev_cur_cb = glfw.set_cursor_pos_callback(
            self.viewer.window, self._on_cursor)
        self._prev_scroll_cb = glfw.set_scroll_callback(
            self.viewer.window, self._on_scroll)
        self._attached = True
        return self

    def detach(self):
        if not self._attached: return self
        # Detach the inner gizmo handle FIRST so its callbacks unhook before
        # we restore ours (preserves the GLFW callback chain order).
        if self._body_handle is not None:
            self._body_handle.detach()
            self._body_handle = None
        try:
            glfw.set_mouse_button_callback(self.viewer.window, self._prev_btn_cb)
            glfw.set_cursor_pos_callback   (self.viewer.window, self._prev_cur_cb)
            glfw.set_scroll_callback       (self.viewer.window, self._prev_scroll_cb)
        except Exception:
            pass
        self._translate_dragging = False
        self._rotate_dragging    = False
        self._grasp_dragging     = False
        self._kin_refpos         = None
        self._last_cursor_xy     = None
        self._attached = False
        return self

    # ────────────────────────── selection ──────────────────────────
    def clear_selection(self):
        """Drop the current (world_idx, body_id) selection."""
        if self._body_handle is not None:
            self._body_handle.detach()
            self._body_handle = None
        self.world_idx = None
        self.body_id   = None

    def update_selection(self, xyz_click_world):
        """Pick (world_idx, body_id) from a world-space click. Returns True on hit."""
        if xyz_click_world is None:
            return False
        click = np.asarray(xyz_click_world, dtype=float)
        if click.shape != (3,) or not np.all(np.isfinite(click)):
            return False
        # closest world by grid offset
        d2  = np.sum((self.grid_offsets - click[:2])**2, axis=1)
        w   = int(np.argmin(d2))
        # logical-frame click (offset removed)
        local = click - np.array([self.grid_offsets[w, 0], self.grid_offsets[w, 1], 0.0])
        # closest candidate body in that world
        xpos_np   = self.d.xpos.numpy()  # (NWORLD, nbody, 3)
        cand_ids  = np.asarray(self.candidate_body_ids, dtype=int)
        body_xpos = xpos_np[w, cand_ids]
        bd2       = np.sum((body_xpos - local)**2, axis=1)
        i         = int(np.argmin(bd2))
        if float(np.sqrt(bd2[i])) > self.MAX_PICK_DIST:
            return False
        self.world_idx = w
        self.body_id   = int(cand_ids[i])
        # (Re)attach an interactive WarpBodyHandle for this selection.
        # Plain LMB-drag on the gizmo handles will translate / rotate the
        # body via the shared BodyHandle hover & drag logic.
        self._reattach_body_handle()
        return True

    @property
    def selected_body_name(self):
        if self.body_id is None: return None
        return self.env.body_names[self.body_id]

    def _reattach_body_handle(self):
        """Detach any existing inner WarpBodyHandle and create a new one
        for the current selection (silently skip if it has no compatible
        pose backend, e.g. a sensor-only body)."""
        if self._body_handle is not None:
            self._body_handle.detach()
            self._body_handle = None
        if self.world_idx is None or self.body_id is None:
            return
        try:
            label = self.env.body_names[self.body_id]
        except Exception:
            label = str(self.body_id)
        try:
            self._body_handle = WarpBodyHandle(
                self.env, self.d, self.body_id,
                world_idx    = self.world_idx,
                grid_offsets = self.grid_offsets,
                label        = label,
                axis_len     = self.GIZMO_AXIS_LEN,
                plane_off    = self.GIZMO_PLANE_OFF,
                plane_size   = self.GIZMO_PLANE_SIZE,
                ring_r       = self.GIZMO_RING_R,
                ring_thick   = self.GIZMO_RING_THICK,
                # WarpInteractiveHandle owns Alt+LMB grasp / Alt+wheel cycle
                # for warp env (writes to d.ctrl[w] not env.data.ctrl).
                enable_grasp = False,
            ).attach()
        except ValueError:
            # Body has no free joint and no mocap binding → no kinematic
            # backend available; gizmo stays decorative.
            self._body_handle = None

    # ────────────────────────── visual ──────────────────────────
    def _body_world_R(self, w, body_id):
        """3×3 world rotation matrix of body in world *w*. ``d.xmat`` is
        ``wp.mat33`` typed → ``.numpy()`` returns shape (NWORLD, nbody, 3, 3)."""
        return np.asarray(self.d.xmat.numpy()[w, body_id], dtype=float).reshape(3, 3)

    def draw_highlight(self):
        """Highlight + translate / rotate gizmos at the selected body.

        If an inner :class:`WarpBodyHandle` is attached for the selection
        (the common path), we delegate to its ``update()`` which runs hover
        detection and draws the gizmo with the highlighted handle. That
        means plain LMB-drag on a gizmo arrow / plane / ring directly moves
        the body in the warp env (axis-constrained interaction).

        If no inner handle exists (e.g. selection has no kinematic
        backend), we fall back to drawing a *decorative* gizmo via the
        shared helpers so the user still sees the body's frame.

        Call before ``mjwarp_render`` so the markers are flushed in that
        render call.
        """
        if self.world_idx is None or self.body_id is None:
            return
        wp_p = self._body_world_pos(self.world_idx, self.body_id)

        if self._body_handle is not None:
            # Interactive: hover detection + axis-constrained drag.
            self._body_handle.update()
        else:
            # Decorative fallback (no kinematic backend on this body).
            R = self._body_world_R(self.world_idx, self.body_id)
            if self.GIZMO_DRAW_MODE in ('translate', 'both'):
                draw_translate_gizmo(
                    self.env, wp_p, R,
                    axis_len=self.GIZMO_AXIS_LEN,
                    plane_off=self.GIZMO_PLANE_OFF,
                    plane_size=self.GIZMO_PLANE_SIZE,
                )
            if self.GIZMO_DRAW_MODE in ('rotate', 'both'):
                draw_rotate_gizmo(
                    self.env, wp_p, R,
                    ring_r=self.GIZMO_RING_R,
                    ring_thick=self.GIZMO_RING_THICK,
                )

        # Optional pick highlight on top.
        if self.HIGHLIGHT_R > 0:
            self.env.plot_sphere(wp_p, r=self.HIGHLIGHT_R, rgba=self.HIGHLIGHT_RGBA)

    HELP_TEXT = ("dblclick=pick | LMB=drag axis/plane | Shift+LMB=ring | "
                 "Ctrl+RMB=move | Ctrl+LMB=rotate | Alt+LMB=grasp | Alt+wheel=joint sel")

    def draw_overlay(self,
                     show_picked: bool = True,
                     show_grasp:  bool = True,
                     show_help:   bool = True,
                     show_cursor: bool = True,
                     help_loc:    str  = "top left",
                     status_loc:  str  = "bottom right"):
        """Push handle status lines to ``env.viewer_text_overlay``.

        Call once per render frame (typically right after
        ``draw_highlight()``). Each flag toggles a section so callers can
        keep just the parts they want.

        ``show_cursor`` adds the world xyz of whatever is under the mouse
        (depth-buffer unprojection) plus the gizmo's hover/drag state — the
        two things you need to know to aim a drag. It reads one pixel of the
        CURRENT framebuffer, so call it after the frame has been rendered.
        """
        if show_picked:
            sel_w = self.world_idx
            sel_b = self.selected_body_name or "-"
            self.env.viewer_text_overlay(
                text1="picked [dblclick]:",
                text2=f"world={sel_w}  body={sel_b}",
                loc=status_loc,
            )
        if show_cursor:
            # One 1-pixel depth read of the frame on screen — see
            # HandRLParserClass.get_xyz_at_pixel_from_framebuffer.
            try:
                xyz = self.env.get_xyz_at_cursor_from_framebuffer()
            except Exception:
                xyz = None
            self.env.viewer_text_overlay(
                text1="cursor world xyz:",
                text2=(f"{xyz[0]:+.3f} {xyz[1]:+.3f} {xyz[2]:+.3f}" if xyz is not None
                       else "(nothing under cursor)"),
                loc=status_loc,
            )
            bh = self._body_handle
            if bh is not None:
                mode = bh.drag_mode or bh.hover_mode
                ax   = bh.drag_axis if bh.drag_mode is not None else bh.hover_axis
                lbl  = GIZMO_AXIS_LABELS[ax] if ax is not None else '-'
                self.env.viewer_text_overlay(
                    text1="gizmo [hover/drag]:",
                    text2=(f"{'DRAG' if bh.drag_mode else 'hover'} {mode} {lbl}"
                           if mode is not None else "none (point at an arrow/plane/ring)"),
                    loc=status_loc,
                )
        if show_grasp:
            self.env.viewer_text_overlay(
                text1="grasp target [Alt+wheel]:",
                text2=self.grasp_active_label,
                loc=status_loc,
            )
            self.env.viewer_text_overlay(
                text1="grasp_t [Alt+LMB]:",
                text2=f"{self.grasp_t:.2f}",
                loc=status_loc,
            )
        if show_help:
            self.env.viewer_text_overlay(text1=self.HELP_TEXT, loc=help_loc)

    def _body_world_pos(self, w, body_id):
        local = self.d.xpos.numpy()[w, body_id]
        return np.array([
            local[0] + self.grid_offsets[w, 0],
            local[1] + self.grid_offsets[w, 1],
            local[2],
        ])

    # ────────────────────────── refpos / cursor delta ──────────────────────────
    def _cursor_world_delta(self, dx_pix, dy_pix, anchor_world):
        """Convert a screen-space cursor delta to a camera-plane world delta
        at the depth of ``anchor_world`` (right · dx + up · -dy)."""
        cb = self._cam_basis()
        if cb is None: return np.zeros(3)
        depth = float(np.dot(np.asarray(anchor_world) - cb['pos'], cb['fwd']))
        if depth <= 1e-6: return np.zeros(3)
        # World units per pixel at this depth (perspective).
        wpp = (2.0 * depth * cb['half_h']) / (cb['near'] * cb['H'])
        return float(dx_pix) * wpp * cb['right'] - float(dy_pix) * wpp * cb['up']

    # ────────────────────────── grasp drag (Alt+LMB) ──────────────────────────
    @property
    def grasp_t(self) -> float:
        """Active joint's grasp progress (or mean of all in 'all' mode)."""
        if self._grasp_active_idx < 0:
            return float(np.mean(self._grasp_t_per_joint)) if len(self._grasp_t_per_joint) else 0.0
        return float(self._grasp_t_per_joint[self._grasp_active_idx])

    @property
    def grasp_t_per_joint(self) -> np.ndarray:
        """Per-joint grasp progress (length matches ``grasp_drag_idxs``)."""
        return self._grasp_t_per_joint.copy()

    @property
    def grasp_active_idx(self) -> int:
        """-1 = all joints; otherwise index into ``grasp_drag_idxs``."""
        return int(self._grasp_active_idx)

    @property
    def grasp_active_label(self) -> str:
        """Human-readable label for the current active joint selection."""
        if self._grasp_active_idx < 0:
            return "all"
        ctrl_idx = int(self._grasp_idxs[self._grasp_active_idx])
        try:
            name = self.env.ctrl_names[ctrl_idx]
        except Exception:
            name = f"#{ctrl_idx}"
        return f"[{self._grasp_active_idx}] {name}"

    def set_grasp_t(self, t: float, idx: int | None = None):
        """Programmatically set grasp progress.

        ``idx=None`` sets every joint to ``t`` (broadcast). Otherwise sets only
        ``grasp_drag_idxs[idx]``.
        """
        t = float(np.clip(t, 0.0, 1.0))
        if idx is None:
            self._grasp_t_per_joint[:] = t
        else:
            self._grasp_t_per_joint[int(idx)] = t
        self._apply_grasp()

    def cycle_grasp_target(self, step: int = 1):
        """Cycle the active grasp target through [-1, 0, 1, …, n−1] (i.e.
        [all, joint_0, joint_1, …, joint_last])."""
        n = len(self._grasp_idxs)
        if n == 0:
            self._grasp_active_idx = -1
            return
        # Map -1..n-1 onto 0..n with -1 ↔ slot n.
        cur = self._grasp_active_idx + 1               # 0..n
        cur = (cur + int(step)) % (n + 1)
        self._grasp_active_idx = cur - 1               # back to -1..n-1

    def _sync_grasp_t_from_ctrl(self):
        """On Alt+LMB press, recover ``grasp_t_per_joint`` from the selected
        world's current ``d.ctrl`` and continue from it.

        Starting from the previous drag's t (initial 0 = ctrl_min for all joints)
        snaps every joint to that t's ctrl on the first cursor event — in a
        headless reproduction the shadow fingertip jumped 33 mm per frame and the
        whole hand jittered (9 mm when synced). Joints with span≈0 keep their previous t."""
        if self.world_idx is None or len(self._grasp_idxs) == 0:
            return
        try:
            cur = self.d.ctrl.numpy()[self.world_idx, self._grasp_idxs].astype(float)
        except Exception:
            return
        idxs = self._grasp_idxs
        span = self._ctrl_closed[idxs] - self._ctrl_open[idxs]
        safe = np.where(np.abs(span) > 1e-9, span, 1.0)
        t    = (cur - self._ctrl_open[idxs]) / safe
        self._grasp_t_per_joint = np.where(
            np.abs(span) > 1e-9,
            np.clip(t, 0.0, 1.0),
            self._grasp_t_per_joint,
        )

    def _active_grasp_positions(self):
        """Positions into ``_grasp_idxs`` / ``_grasp_t_per_joint`` the grasp drag
        should touch for the current selection.

        * No per-body map (single-hand / legacy) → ``None`` = all positions.
        * Per-body map set → restrict to the selected hand's ctrls; if the
          selection is not a mapped hand (an object, or nothing selected),
          return an EMPTY set so the grasp is a no-op (prevents closing every
          hand when no hand is picked)."""
        if not self._grasp_pos_by_body:
            return None
        return self._grasp_pos_by_body.get(self.body_id, np.empty(0, dtype=int))

    def _apply_grasp(self):
        """Write ``(1-t_i)·ctrl_open[i] + t_i·ctrl_closed[i]`` for every i in
        ``self._grasp_idxs`` to ``d.ctrl[world_idx]``, clamped per-joint to
        the actuator's ctrl range. Other worlds untouched. When a hand is
        selected under a per-body grasp map, only that hand's ctrls are written
        so the other hand is left untouched."""
        if self.world_idx is None: return
        pos  = self._active_grasp_positions()
        idxs = self._grasp_idxs if pos is None else self._grasp_idxs[pos]
        t    = self._grasp_t_per_joint if pos is None else self._grasp_t_per_joint[pos]
        ctrl_target = (1.0 - t) * self._ctrl_open[idxs] + t * self._ctrl_closed[idxs]
        # Defensive: keep within each joint's own ctrlrange even if user-
        # supplied open/closed went outside.
        ctrl_target = np.clip(ctrl_target, self._ctrl_min[idxs], self._ctrl_max[idxs])
        ctrl_np = self.d.ctrl.numpy().copy()
        ctrl_np[self.world_idx, idxs] = ctrl_target
        wp.copy(self.d.ctrl,
                wp.array(ctrl_np,
                         dtype=self.d.ctrl.dtype,
                         device=self.d.ctrl.device))

    # ───────────────────── kinematic drag (Ctrl+LMB / Ctrl+RMB) ─────────────────────
    def _can_kin_drag(self):
        """Return True if the current selection supports direct kinematic drag
        (it has a mocap mapping or a free joint we can write to)."""
        if self.world_idx is None or self.body_id is None: return False
        if self.body_id in self._mocap_id_by_body: return True
        return self._qposadr_per_body.get(self.body_id, -1) >= 0

    def _read_kin_quat(self):
        """Current world-frame quaternion (wxyz) of the selected body."""
        if self.body_id in self._mocap_id_by_body:
            mid = self._mocap_id_by_body[self.body_id]
            return self.d.mocap_quat.numpy()[self.world_idx, mid].copy()
        qpa = self._qposadr_per_body[self.body_id]
        return self.d.qpos.numpy()[self.world_idx, qpa+3:qpa+7].copy()

    def _write_kin_quat(self, quat_wxyz):
        if self.body_id in self._mocap_id_by_body:
            mid = self._mocap_id_by_body[self.body_id]
            mq = self.d.mocap_quat.numpy().copy()
            mq[self.world_idx, mid] = quat_wxyz
            wp.copy(self.d.mocap_quat,
                    wp.array(mq,
                             dtype=self.d.mocap_quat.dtype,
                             device=self.d.mocap_quat.device))
        else:
            qpa = self._qposadr_per_body[self.body_id]
            qp = self.d.qpos.numpy().copy()
            qp[self.world_idx, qpa+3:qpa+7] = quat_wxyz
            wp.copy(self.d.qpos,
                    wp.array(qp,
                             dtype=self.d.qpos.dtype,
                             device=self.d.qpos.device))
            # zero angular velocity on teleport
            qv = self.d.qvel.numpy().copy()
            qv[self.world_idx, qpa+3:qpa+6] = 0.0
            wp.copy(self.d.qvel,
                    wp.array(qv,
                             dtype=self.d.qvel.dtype,
                             device=self.d.qvel.device))

    def _apply_kin_translate(self):
        """Write ``self._kin_refpos`` to mocap_pos (wrist) or qpos (free-joint)
        of the selected (world, body)."""
        if not self._can_kin_drag() or self._kin_refpos is None: return
        # Convert world refpos → logical (offset-removed) frame for that world.
        ox, oy = self.grid_offsets[self.world_idx]
        local = np.array([self._kin_refpos[0] - ox,
                          self._kin_refpos[1] - oy,
                          self._kin_refpos[2]], dtype=float)
        if self.body_id in self._mocap_id_by_body:
            mid = self._mocap_id_by_body[self.body_id]
            mp = self.d.mocap_pos.numpy().copy()
            mp[self.world_idx, mid] = local
            wp.copy(self.d.mocap_pos,
                    wp.array(mp,
                             dtype=self.d.mocap_pos.dtype,
                             device=self.d.mocap_pos.device))
        else:
            qpa = self._qposadr_per_body[self.body_id]
            qp = self.d.qpos.numpy().copy()
            qp[self.world_idx, qpa:qpa+3] = local
            wp.copy(self.d.qpos,
                    wp.array(qp,
                             dtype=self.d.qpos.dtype,
                             device=self.d.qpos.device))
            # zero linear velocity to avoid carrying over momentum on teleport
            qv = self.d.qvel.numpy().copy()
            qv[self.world_idx, qpa:qpa+6] = 0.0
            wp.copy(self.d.qvel,
                    wp.array(qv,
                             dtype=self.d.qvel.dtype,
                             device=self.d.qvel.device))

    def _apply_kin_rotate(self, dx_pix, dy_pix):
        """Trackball-style incremental rotation of the selected body driven
        by cursor pixel delta.

        Horizontal cursor (dx) → rotate around camera-up axis.
        Vertical   cursor (dy) → rotate around camera-right axis (negated so
        dragging up rotates the body's top toward the viewer).
        """
        if not self._can_kin_drag(): return
        cb = self._cam_basis()
        if cb is None: return
        axis = -float(dy_pix) * cb['right'] + float(dx_pix) * cb['up']
        n = float(np.linalg.norm(axis))
        if n < 1e-9: return
        axis = axis / n
        angle = n * self.ROTATE_SENS
        q_axis = np.zeros(4, dtype=np.float64)
        mujoco.mju_axisAngle2Quat(q_axis, axis.astype(np.float64), float(angle))
        cur_q = self._read_kin_quat().astype(np.float64)
        new_q = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(new_q, q_axis, cur_q)
        self._write_kin_quat(new_q)

    # ────────────────────────── camera helpers ──────────────────────────
    # Exact camera model shared with BodyHandle — see viewer_cam_basis().
    def _cam_basis(self):
        return viewer_cam_basis(self.viewer)

    def _ray_from_mouse(self, mx, my):
        return cam_mouse_ray(self._cam_basis(), mx, my)

    # Thin alias so the drag code reads the same in every handle class.
    _ray_plane_intersect = staticmethod(ray_plane_intersect)

    def cursor_world_on_body_plane(self, cx, cy, plane_o):
        """World point where the cursor ray meets the camera-facing plane
        through ``plane_o``.

        This is the grab point sticky translate freezes at press time so the
        clicked spot stays under the mouse for the whole drag. ``None`` when
        the camera is not ready yet or the ray misses the plane.
        """
        cb = self._cam_basis()
        if cb is None or plane_o is None:
            return None
        ray = cam_mouse_ray(cb, cx, cy)
        if ray is None:
            return None
        plane_n = cb['fwd'] if self._kin_plane_n is None else self._kin_plane_n
        return self._ray_plane_intersect(ray[0], ray[1],
                                         np.asarray(plane_o, dtype=float), plane_n)

    # ────────────────────────── GLFW callbacks ──────────────────────────
    def _on_button(self, window, button, act, mods):
        # Match notebook 13's MuJoCo-perturb-style convention:
        #   Ctrl + LMB  → kinematic ROTATE   of the selected body
        #   Ctrl + RMB  → kinematic TRANSLATE of the selected body
        #   Alt  + LMB  → grasp drag (interpolate d.ctrl between open/closed)
        # Plain LMB/RMB pass through so the viewer's camera rotate/pan work.
        is_ctrl_rmb = button == glfw.MOUSE_BUTTON_RIGHT and (mods & glfw.MOD_CONTROL)
        is_ctrl_lmb = button == glfw.MOUSE_BUTTON_LEFT  and (mods & glfw.MOD_CONTROL) and not (mods & glfw.MOD_ALT)
        is_alt_lmb  = button == glfw.MOUSE_BUTTON_LEFT  and (mods & glfw.MOD_ALT)     and not (mods & glfw.MOD_CONTROL)

        if is_ctrl_rmb:
            if act == glfw.PRESS and self._can_kin_drag():
                self._translate_dragging = True
                self._kin_refpos = self._body_world_pos(self.world_idx, self.body_id).copy()
                mx, my = glfw.get_cursor_pos(window)
                cx, cy = self.viewer._scale * mx, self.viewer._scale * my
                self._last_cursor_xy = (cx, cy)
                # Sticky translate: freeze the camera-facing plane through the
                # body at press time and remember where on it the user grabbed.
                # Moving by (hit - grab) keeps the grabbed point under the
                # cursor exactly, instead of integrating per-event pixel deltas
                # whose world scale is only correct near the start.
                cb = self._cam_basis()
                self._kin_plane_n     = cb['fwd'].copy() if cb is not None else None
                self._kin_plane_o     = self._kin_refpos.copy()
                self._kin_start_pos   = self._kin_refpos.copy()
                self._kin_grab_isect  = self.cursor_world_on_body_plane(
                    cx, cy, self._kin_plane_o) if cb is not None else None
                return
            if act == glfw.RELEASE and self._translate_dragging:
                self._translate_dragging = False
                self._kin_refpos = None
                self._last_cursor_xy = None
                self._kin_grab_isect = None
                return

        if is_ctrl_lmb:
            if act == glfw.PRESS and self._can_kin_drag():
                self._rotate_dragging = True
                mx, my = glfw.get_cursor_pos(window)
                self._last_cursor_xy = (self.viewer._scale * mx, self.viewer._scale * my)
                return
            if act == glfw.RELEASE and self._rotate_dragging:
                self._rotate_dragging = False
                self._last_cursor_xy = None
                return

        if is_alt_lmb:
            if act == glfw.PRESS and self.world_idx is not None:
                # Recover grasp_t from the current ctrl — prevents the snap from
                # starting at a stale t (all joints jumping at drag start).
                self._sync_grasp_t_from_ctrl()
                self._grasp_dragging = True
                mx, my = glfw.get_cursor_pos(window)
                self._last_cursor_xy = (self.viewer._scale * mx, self.viewer._scale * my)
                return
            if act == glfw.RELEASE and self._grasp_dragging:
                self._grasp_dragging = False
                self._last_cursor_xy = None
                return

        if self._prev_btn_cb is not None:
            self._prev_btn_cb(window, button, act, mods)

    def _on_cursor(self, window, xpos, ypos):
        cx = self.viewer._scale * xpos
        cy = self.viewer._scale * ypos
        if (self._translate_dragging or self._rotate_dragging or self._grasp_dragging) \
                and self._last_cursor_xy is not None:
            dx = cx - self._last_cursor_xy[0]
            dy = cy - self._last_cursor_xy[1]
            if self._translate_dragging and self._kin_refpos is not None:
                if self._kin_grab_isect is not None and self._kin_plane_n is not None:
                    ray = self._ray_from_mouse(cx, cy)
                    hit = (self._ray_plane_intersect(ray[0], ray[1],
                                                     self._kin_plane_o, self._kin_plane_n)
                           if ray is not None else None)
                    if hit is not None:
                        self._kin_refpos = self._kin_start_pos + (hit - self._kin_grab_isect)
                        self._apply_kin_translate()
                else:
                    # No usable grab point (camera not ready at press) — keep
                    # the incremental camera-plane delta as a fallback.
                    world_d = self._cursor_world_delta(dx, dy, self._kin_refpos)
                    self._kin_refpos = self._kin_refpos + world_d
                    self._apply_kin_translate()
            if self._rotate_dragging:
                self._apply_kin_rotate(dx, dy)
            if self._grasp_dragging:
                # Mouse up (negative dy) closes the hand; down opens it.
                dt = -dy * self.GRASP_SENS
                if self._grasp_active_idx < 0:
                    # "all joints" — but only the SELECTED hand's ctrls when a
                    # per-body grasp map is active (leader/follower isolation).
                    pos = self._active_grasp_positions()
                    if pos is None:
                        self._grasp_t_per_joint = np.clip(
                            self._grasp_t_per_joint + dt, 0.0, 1.0)
                    else:
                        self._grasp_t_per_joint[pos] = np.clip(
                            self._grasp_t_per_joint[pos] + dt, 0.0, 1.0)
                else:
                    i = int(self._grasp_active_idx)
                    self._grasp_t_per_joint[i] = float(np.clip(
                        self._grasp_t_per_joint[i] + dt, 0.0, 1.0))
                self._apply_grasp()
        self._last_cursor_xy = (cx, cy)

        if self._prev_cur_cb is not None:
            self._prev_cur_cb(window, xpos, ypos)

    def _on_scroll(self, window, x_offset, y_offset):
        # Alt + wheel cycles the active grasp target through
        # [all_joint, joint_0, joint_1, …, joint_last]. Plain wheel
        # falls through so the viewer's zoom keeps working.
        try:
            alt_held = (
                glfw.get_key(window, glfw.KEY_LEFT_ALT)  == glfw.PRESS or
                glfw.get_key(window, glfw.KEY_RIGHT_ALT) == glfw.PRESS
            )
        except Exception:
            alt_held = False
        if alt_held:
            step = 1 if y_offset > 0 else (-1 if y_offset < 0 else 0)
            if step != 0:
                self.cycle_grasp_target(step)
                return  # consume; do NOT zoom
        if self._prev_scroll_cb is not None:
            self._prev_scroll_cb(window, x_offset, y_offset)
