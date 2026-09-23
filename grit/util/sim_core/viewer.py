"""``MuJoCoMinimalViewer`` — a small GLFW window around ``mujoco.MjvScene``.

Responsibilities (kept deliberately minimal — the parallel-world rendering
lives in :class:`grit.util.mjwarp_viewer.MJWarpMinimalViewer`, a subclass):

* window + OpenGL context creation, HiDPI-aware framebuffer scale
* mouse camera control (LMB rotate / RMB pan / wheel zoom, Shift variants),
  Ctrl+mouse body perturbation, ESC-to-close, key press / repeat sets that
  the parser polls (``is_key_pressed_once`` …), double-click detection
* a per-frame **marker queue** (``add_marker(pos=, type=, size=, mat=, rgba=, label=)``)
  flushed into the scene as decorative geoms on every ``render()``
* a per-frame **text overlay** queue (``add_overlay(loc, text1, text2)``)
  and optional RGB image overlays in the four corners

The ``MjrContext`` (``self.ctx``) is created by the owner *after*
construction so the font scale can be chosen there.
"""
from __future__ import annotations

import time
from threading import Lock

import glfw
import mujoco
import numpy as np


_GRIDPOS = {
    "top": mujoco.mjtGridPos.mjGRID_TOP,
    "top right": mujoco.mjtGridPos.mjGRID_TOPRIGHT,
    "top left": mujoco.mjtGridPos.mjGRID_TOPLEFT,
    "bottom": mujoco.mjtGridPos.mjGRID_BOTTOM,
    "bottom right": mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
    "bottom left": mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
}



def _move_camera(model, action, reldx, reldy, scn, cam):
    """mjv_moveCamera across MuJoCo versions: ≤3.3 takes (m, action, dx, dy, scn, cam), 3.4+ dropped ``scn``."""
    try:
        mujoco.mjv_moveCamera(model, action, reldx, reldy, cam)
    except TypeError:
        mujoco.mjv_moveCamera(model, action, reldx, reldy, scn, cam)

class MinimalCallbacks:
    """GLFW input callbacks + the state they maintain."""

    def __init__(self):
        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        self._last_left_click_time = None
        self._last_right_click_time = None
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._paused = False
        self._render_every_frame = True
        self._time_per_render = 1.0 / 60.0
        self._run_speed = 1.0
        self._loop_count = 0
        self._advance_by_one_step = False
        self._scale = 1.0
        # key buffers polled by the parser
        self._key_pressed_set = set()
        self._key_repeated_set = set()

    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self._key_pressed_set.add(key)
        elif action == glfw.REPEAT:
            self._key_repeated_set.add(key)
        elif action == glfw.RELEASE:
            self._key_pressed_set.discard(key)
            self._key_repeated_set.discard(key)
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    def _cursor_pos_callback(self, window, xpos, ypos):
        if not (self._button_left_pressed or self._button_right_pressed):
            return
        shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                 or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self._button_right_pressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        _, height = glfw.get_framebuffer_size(window)
        with self._gui_lock:
            if self.pert.active:
                mujoco.mjv_movePerturb(self.model, self.data, action, dx / height, dy / height,
                                       self.scn, self.pert)
            else:
                _move_camera(self.model, action, dx / height, dy / height, self.scn, self.cam)
        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left_pressed = (button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS)
        self._button_right_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS)
        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        now = glfw.get_time()
        if self._button_left_pressed:
            if self._last_left_click_time is None:
                self._last_left_click_time = now
            dt = now - self._last_left_click_time
            if 0.01 < dt < 0.3:
                self._left_double_click_pressed = True
            self._last_left_click_time = now
        if self._button_right_pressed:
            if self._last_right_click_time is None:
                self._last_right_click_time = now
            dt = now - self._last_right_click_time
            if 0.01 < dt < 0.3:
                self._right_double_click_pressed = True
            self._last_right_click_time = now
        # Ctrl + mouse on a selected body → MuJoCo perturbation
        newperturb = 0
        if mods == glfw.MOD_CONTROL and self.pert.select > 0:
            if self._button_right_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if self._button_left_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_ROTATE
            if newperturb and not self.pert.active:
                mujoco.mjv_initPerturb(self.model, self.data, self.scn, self.pert)
        self.pert.active = newperturb
        if act == glfw.RELEASE:
            self.pert.active = 0

    def _scroll_callback(self, window, x_offset, y_offset):
        with self._gui_lock:
            _move_camera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y_offset, self.scn, self.cam)


class MuJoCoMinimalViewer(MinimalCallbacks):
    def __init__(self, model, data, mode="window", title="MuJoCo Minimal Viewer",
                 width=None, height=None, maxgeom=50000, perturbation=True,
                 x_offset=None, y_offset=None):
        super().__init__()
        self.model = model
        self.data = data
        self.render_mode = mode
        self.is_alive = True
        if not glfw.init():
            raise RuntimeError("glfw.init() failed — is a display available?")
        vm = glfw.get_video_mode(glfw.get_primary_monitor())
        monitor_w, monitor_h = int(vm.size.width), int(vm.size.height)
        width = int(width or monitor_w)
        height = int(height or monitor_h)
        if mode == "offscreen":
            glfw.window_hint(glfw.VISIBLE, 0)
        self.maxgeom = maxgeom
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            raise RuntimeError("glfw.create_window() failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        if mode == "window":
            win_w, _ = glfw.get_window_size(self.window)
            self._scale = fb_w / float(win_w)      # HiDPI: framebuffer px per window px
            glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
            glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
            glfw.set_key_callback(self.window, self._key_callback)
        self.vopt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.scn = mujoco.MjvScene(self.model, maxgeom=self.maxgeom)
        self.pert = mujoco.MjvPerturb()
        self.ctx = None                                    # owner sets MjrContext(model, fontscale)
        # window placement: centred by default, offsets as monitor ratio (0~1) or pixels
        win_w, win_h = glfw.get_window_size(self.window)
        x_px = int((monitor_w - win_w) * 0.5)
        y_px = int((monitor_h - win_h) * 0.5)
        if x_offset is not None:
            x_px = int(monitor_w * float(x_offset)) if 0.0 <= x_offset <= 1.0 else int(x_offset)
        if y_offset is not None:
            y_px = int(monitor_h * float(y_offset)) if 0.0 <= y_offset <= 1.0 else int(y_offset)
        x_px = max(0, min(x_px, monitor_w - win_w))
        y_px = max(0, min(y_px, monitor_h - win_h))
        glfw.set_window_pos(self.window, x_px, y_px)
        self.viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)
        self._overlay = {}
        self._markers = []
        self.rgb_overlay_top_right = None
        self.rgb_overlay_top_left = None
        self.rgb_overlay_bottom_right = None
        self.rgb_overlay_bottom_left = None
        self.perturbation = perturbation

    # ── markers / overlays ─────────────────────────────────────────────
    def add_marker(self, **marker_params):
        """Queue a decorative geom for the next frame (``pos``, ``type``, ``size``, ``mat``, ``rgba``, ``label``)."""
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(f"ran out of scene geoms (maxgeom={self.scn.maxgeom})")
        g = self.scn.geoms[self.scn.ngeom]
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.matid = -1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:] = 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = 1.0
        for key, value in marker.items():
            if isinstance(value, str):
                g.label = value
            elif isinstance(value, (int, float, np.integer, np.floating, mujoco.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value, dtype=np.float64).reshape(attr.shape)
            elif value is None and key == "label":
                g.label = ""
            else:
                raise ValueError(f"marker field {key!r} has unsupported type {type(value)}")
        self.scn.ngeom += 1

    def apply_perturbations(self):
        self.data.xfrc_applied[:] = 0.0
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)

    def add_overlay(self, loc="bottom left", gridpos=mujoco.mjtGridPos.mjGRID_TOPLEFT, text1="", text2=""):
        if loc is not None:
            gridpos = _GRIDPOS.get(loc, gridpos)
        if gridpos not in self._overlay:
            self._overlay[gridpos] = [text1, text2]
        else:
            self._overlay[gridpos][0] += "\n" + text1
            self._overlay[gridpos][1] += "\n" + text2

    def plot_rgb_overlay(self, rgb=None, loc="top right"):
        """Show an RGB image in a corner (quarter of the window, aspect preserved)."""
        import cv2   # lazy — only the RGB-overlay path needs it
        w_win, h_win = glfw.get_framebuffer_size(self.window)
        h_ov, w_ov = h_win // 4, w_win // 4
        h_raw, w_raw = rgb.shape[:2]
        s = min(w_ov / w_raw, h_ov / h_raw)
        w_new, h_new = int(w_raw * s), int(h_raw * s)
        rgb_rs = cv2.resize(rgb, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        pad = np.zeros((h_ov, w_ov, 3), dtype=np.uint8)
        x0, y0 = (w_ov - w_new) // 2, (h_ov - h_new) // 2
        pad[y0:y0 + h_new, x0:x0 + w_new] = rgb_rs
        attr = {"top right": "rgb_overlay_top_right", "top left": "rgb_overlay_top_left",
                "bottom right": "rgb_overlay_bottom_right", "bottom left": "rgb_overlay_bottom_left"}.get(loc)
        if attr is None:
            print("invalid RGB overlay location:", loc)
            return
        setattr(self, attr, pad)

    def reset_rgb_overlay(self, loc=None):
        names = {"top right": "rgb_overlay_top_right", "top left": "rgb_overlay_top_left",
                 "bottom right": "rgb_overlay_bottom_right", "bottom left": "rgb_overlay_bottom_left"}
        for k, a in names.items():
            if loc is None or loc == k:
                setattr(self, a, None)

    def _draw_rgb_overlays(self):
        for img, (cx, cy) in ((self.rgb_overlay_top_right, (3, 3)), (self.rgb_overlay_top_left, (0, 3)),
                              (self.rgb_overlay_bottom_right, (3, 0)), (self.rgb_overlay_bottom_left, (0, 0))):
            if img is None:
                continue
            h, w = img.shape[:2]
            rect = mujoco.MjrRect(left=cx * w, bottom=cy * h, width=w, height=h)
            mujoco.mjr_drawPixels(rgb=np.flipud(img).flatten(), depth=None, viewport=rect, con=self.ctx)

    # ── frame ─────────────────────────────────────────────────────────
    def update_render_scene(self):
        """Render exactly one frame: scene + queued markers + overlays, then swap."""
        t0 = time.time()
        w, h = glfw.get_framebuffer_size(self.window)
        self.viewport.width, self.viewport.height = w, h
        with self._gui_lock:
            mujoco.mjv_updateScene(self.model, self.data, self.vopt, self.pert, self.cam,
                                   mujoco.mjtCatBit.mjCAT_ALL.value, self.scn)
            for m in self._markers:
                self._add_marker_to_scene(m)
            mujoco.mjr_render(self.viewport, self.scn, self.ctx)
            for gridpos, (t1, t2) in self._overlay.items():
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100, gridpos, self.viewport, t1, t2, self.ctx)
            self._draw_rgb_overlays()
            glfw.swap_buffers(self.window)
        glfw.poll_events()
        self._time_per_render = 0.9 * self._time_per_render + 0.1 * (time.time() - t0)

    def render(self, reset_rgb_overlay=True):
        if not self.is_alive:
            raise RuntimeError("GLFW window does not exist but render() was called")
        if glfw.window_should_close(self.window):
            self.close()
            return
        if self._paused:
            while self._paused:
                self.update_render_scene()
                if glfw.window_should_close(self.window):
                    self.close()
                    break
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                self.update_render_scene()
                self._loop_count -= 1
        self._markers[:] = []
        self._overlay.clear()
        if reset_rgb_overlay:
            self.reset_rgb_overlay()
        if self.perturbation:
            self.apply_perturbations()

    def close(self):
        if not self.is_alive:
            return
        self.is_alive = False
        for fn in (
            lambda: (glfw.set_window_should_close(self.window, True), glfw.hide_window(self.window),
                     glfw.make_context_current(self.window)),
            lambda: self.ctx.free() if self.ctx is not None else None,
            lambda: glfw.make_context_current(None),
            lambda: glfw.destroy_window(self.window),
            glfw.poll_events,
        ):
            try:
                fn()
            except Exception:
                pass
        self.window = None

    # ── settings ──────────────────────────────────────────────────────
    def set_transparency(self, transparent=True):
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent

    def get_cam_info(self):
        return float(self.cam.azimuth), float(self.cam.distance), float(self.cam.elevation), self.cam.lookat.copy()

    def set_cam_info(self, azimuth=None, distance=None, elevation=None, lookat=None):
        if azimuth is not None:
            self.cam.azimuth = azimuth
        if distance is not None:
            self.cam.distance = distance
        if elevation is not None:
            self.cam.elevation = elevation
        if lookat is not None:
            self.cam.lookat[:] = np.asarray(lookat, dtype=np.float64).reshape(3)

    def set_geomgroup(self, group_0=True, group_1=True, group_2=True, group_3=False, group_4=None, group_5=None):
        for i, g in enumerate((group_0, group_1, group_2, group_3, group_4, group_5)):
            if g is not None:
                self.vopt.geomgroup[i] = g

    def set_sitegroup(self, group_0=None, group_1=None, group_2=None, group_3=None, group_4=None, group_5=None):
        for i, g in enumerate((group_0, group_1, group_2, group_3, group_4, group_5)):
            if g is not None:
                self.vopt.sitegroup[i] = g

    def set_convexhull(self, convexhull=False):
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = convexhull
