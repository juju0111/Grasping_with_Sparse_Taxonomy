import os
import time
import mujoco
import mujoco_warp as mjwarp
import warp as wp
import glfw

import cv2
import numpy as np

from typing import Optional, Union

from .mjwarp_viewer import MJWarpMinimalViewer, viewer_cam_basis, cam_mouse_ray

from grit.util.sim_core.parser import MuJoCoParser
from grit.util.sim_core.viewer import MuJoCoMinimalViewer
from grit.util.sim_core.transforms import t2p, t2r, pr2t, r2quat, rpy2r
from grit.util.sim_core.utils import trim_scale, get_idxs, get_monitor_size, TicToc, farthest_point_sampling
from grit.util.sim_core.viz import get_colors, print_red

class HandRLParserClass(MuJoCoParser):
    def __init__(
            self,
            name          = None,
            rel_xml_path  = None,
            xml_string    = None,
            assets        = None,
            hand_name     = None, 
            for_render    = False, 
            verbose       = True,
        ):
        """
        Initialize the MuJoCo parser.

        Parameters:
            name (str, optional): Name of the model/environment.
            rel_xml_path (str, optional): Relative path to the XML file.
            xml_string (str, optional): MuJoCo model XML content as a string.
            assets (dict, optional): Dictionary of additional assetsㅋ for MuJoCo.
            verbose (bool, optional): Whether to print detailed information.

        Returns:
            None
        """
        # Store parameters 
        self.name         = name
        self.rel_xml_path = rel_xml_path
        self.xml_string   = xml_string
        self.assets       = assets
        self.verbose      = verbose
        self.is_running   = True
        self.viewer: Optional[Union[MJWarpMinimalViewer, MuJoCoMinimalViewer]] = None
   

        # Parse XML
        self._parse_xml(rel_xml_path=self.rel_xml_path, xml_string=self.xml_string)

        # Get monitor size 
        self.monitor_width,self.monitor_height = get_monitor_size() 

        # Flags
        self.use_mujoco_viewer = False

        # Timing variables used in self.is_viewer_alive() method
        self._last_alive_t: Optional[float] = None
        self.loop_dt: Optional[float]       = 0.0
        self.loop_hz: Optional[float]       = 0.0
        self.loop_hz_ema: Optional[float]   = 0.0
        self._loop_hz_ema_alpha: float      = 0.05

        # Initial joint positions
        self.q0       = self.get_qpos()
        self.qrev0    = self.get_qpos(joint_names=self.rev_joint_names,flatten=True)
        self.qactive0 = self.get_qpos(joint_names=self.active_joint_names,flatten=True)


        # Final RGB image
        self.final_rgb_img = None

        # Object point cloud cache (body-local canonical pcd)
        self.obj_pcd_local_cache = {}

        # TicToc instance
        self.tt = TicToc()

        # Reset
        self.reset()

        # Print information
        if self.verbose:
            self.print_info()
        else:
            if self.rel_xml_path is not None:
                print ("MuJoCo model loaded from [%s]."%(self.rel_xml_path))
            elif self.xml_string is not None:
                print ("MuJoCo model loaded from [%s]."%(self.xml_string))

        # Set object mesh information and body & geom relative pose 
        self.obj_names = [] 
        self.obj_mesh_idx = [] 
        self.obj_body_geom_relative_pos = [] 
        self.relative_obj_min = {} 
        self.relative_obj_max = {} 

        # Tick
        self.tick = 0

        # mjwarp mode flag — set True by init_mjwarp_viewer
        self._mjwarp_mode = False

        # Ghost overlay alpha ([8] / [9]) — seeded from the ghost model's own
        # alpha on the first render that passes one.
        self._ghost_alpha         = None
        self._ghost_alpha_ramping = False

        # Contact overlays: real-hand force arrows [K] / ghost collision [J],
        # shared arrow scale [6] / [7]. Both on by default.
        self._contact_force_vis   = True
        self._ghost_contact_vis   = True
        self._contact_force_scale = self.CONTACT_FORCE_SCALE
        # Last frame's drawn-contact counts (shown by viewer_contact_vis_overlay).
        self._n_contact_arrows    = 0
        self._n_ghost_contacts    = 0


    # ------------------------------------------------------------------
    # Visual keybinding helpers (CPU + Warp viewer)
    # ------------------------------------------------------------------
    def init_visual_state(
        self,
        transparency: bool = True,
        contactpoint: bool = True,
        shadow:       bool = False,
        geomgroups:   tuple = (True, True, True, False, False),
    ):
        """One-shot init: store visual flags on the env and apply to the
        active viewer. Subsequent `handle_visual_keys()` calls toggle the
        same flags via `[0..4] / [T] / [C] / [S]`.

        Works with both CPU `MuJoCoMinimalViewer` and `MJWarpMinimalViewer`.
        Contactpoint/shadow are skipped if the active viewer doesn't expose
        the corresponding setter (CPU viewer has neither).
        """
        gs = list(geomgroups)
        while len(gs) < 5:
            gs.append(False)
        self._vis_transparency = bool(transparency)
        self._vis_contactpoint = bool(contactpoint)
        self._vis_shadow       = bool(shadow)
        self._vis_geomgroups   = [bool(g) for g in gs[:5]]

        self.viewer.set_transparency(self._vis_transparency) # type: ignore 
        if hasattr(self.viewer, 'set_contactpoint'):
            self.viewer.set_contactpoint(self._vis_contactpoint) # type: ignore
        if hasattr(self.viewer, 'set_shadow'):
            self.viewer.set_shadow(self._vis_shadow) # type: ignore
        self._apply_geomgroups()

    def _apply_geomgroups(self):
        g = self._vis_geomgroups
        self.viewer.set_geomgroup( # type: ignore
            group_0=g[0], group_1=g[1], group_2=g[2],
            group_3=g[3], group_4=g[4],
        )

    # Valid MuJoCo font-atlas scales (mjFONTSCALE_*): the label glyphs are
    # baked into the MjrContext at creation, so "text size" is only adjustable
    # by rebuilding the context at one of these scales — there is no
    # per-marker font size in MuJoCo label rendering.
    FONT_SCALES = (50, 100, 150, 200, 250, 300)

    @property
    def viewer_fontscale(self):
        """Current label font scale, or None before ``init_mjwarp_viewer``."""
        return getattr(self, '_viewer_fontscale', None)

    def set_viewer_fontscale(self, fontscale):
        """Rebuild the render context at a new font scale (snapped to
        :attr:`FONT_SCALES`) — resizes ALL viewer text (marker labels, world
        indices). Safe to call from the render loop: the GLFW context is
        current there, same as at init time. Returns the applied scale."""
        fs  = min(self.FONT_SCALES, key=lambda v: abs(v - int(fontscale)))
        cur = getattr(self, '_viewer_fontscale', None)
        if fs == cur or not hasattr(self, 'viewer'):
            return cur
        old = getattr(self.viewer, 'ctx', None)
        if old is not None:
            try:
                old.free()      # drop the old GL font atlas / buffers
            except Exception:
                pass
        self.viewer.ctx = mujoco.MjrContext(self.model, fs)  # type: ignore
        self._viewer_fontscale = fs
        return fs

    def _step_viewer_fontscale(self, direction: int):
        """Move one notch along :attr:`FONT_SCALES` (`direction` = ±1)."""
        cur = getattr(self, '_viewer_fontscale', None)
        if cur is None:
            return
        try:
            i = self.FONT_SCALES.index(cur)
        except ValueError:
            i = min(range(len(self.FONT_SCALES)),
                    key=lambda k: abs(self.FONT_SCALES[k] - cur))
        j = int(np.clip(i + direction, 0, len(self.FONT_SCALES) - 1))
        self.set_viewer_fontscale(self.FONT_SCALES[j])

    def handle_visual_keys(self):
        """Poll `[0..4] / [T] / [C] / [S]` once and toggle visual flags
        previously set via `init_visual_state(...)`. No-op until that init.
        `[-] / [=]` step the label font scale down / up (needs only
        ``init_mjwarp_viewer``, not ``init_visual_state``).
        """
        # Font-scale keys — independent of the init_visual_state flags.
        if self.is_key_pressed_once(glfw.KEY_MINUS): self._step_viewer_fontscale(-1)
        if self.is_key_pressed_once(glfw.KEY_EQUAL): self._step_viewer_fontscale(+1)

        if not hasattr(self, '_vis_geomgroups'):
            return

        changed = False
        for i, key in enumerate(
            (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4)
        ):
            if self.is_key_pressed_once(key):
                self._vis_geomgroups[i] = not self._vis_geomgroups[i]
                changed = True
        if changed:
            self._apply_geomgroups()

        if self.is_key_pressed_once(glfw.KEY_T):
            self._vis_transparency = not self._vis_transparency
            self.viewer.set_transparency(self._vis_transparency) # type: ignore
        if hasattr(self.viewer, 'set_contactpoint') and self.is_key_pressed_once(glfw.KEY_C):
            self._vis_contactpoint = not self._vis_contactpoint
            self.viewer.set_contactpoint(self._vis_contactpoint) # type: ignore
        if hasattr(self.viewer, 'set_shadow') and self.is_key_pressed_once(glfw.KEY_S):
            self._vis_shadow = not self._vis_shadow
            self.viewer.set_shadow(self._vis_shadow) # type: ignore

    @property
    def vis_transparency(self): return getattr(self, '_vis_transparency', None)

    @property
    def vis_contactpoint(self): return getattr(self, '_vis_contactpoint', None)

    @property
    def vis_shadow(self):       return getattr(self, '_vis_shadow', None)

    @property
    def vis_geomgroups(self):
        return tuple(getattr(self, '_vis_geomgroups', ()))

    # ------------------------------------------------------------------
    # Ghost overlay alpha keybinding ([8] dimmer / [9] brighter)
    # ------------------------------------------------------------------
    # Lives here rather than in the caller (evaluate.py) so every script that
    # renders a ghost through ``mjwarp_render(target_model=...)`` gets the
    # binding for free — the poll runs inside that call.
    GHOST_ALPHA_KEY_DOWN = glfw.KEY_8      # dimmer
    GHOST_ALPHA_KEY_UP   = glfw.KEY_9      # brighter
    GHOST_ALPHA_STEP     = 0.05            # per key press
    GHOST_ALPHA_HOLD     = 0.02            # per frame while held (key repeat)
    GHOST_ALPHA_MIN      = 0.0
    GHOST_ALPHA_MAX      = 1.0

    @property
    def ghost_alpha(self):
        """Current ghost alpha override, or ``None`` while untouched (the
        ghost model's own alpha is in effect)."""
        return getattr(self, '_ghost_alpha', None)

    def set_ghost_alpha(self, alpha, target_model=None):
        """Set the ghost overlay alpha (clamped to [MIN, MAX]).

        Writes through to ``target_model.geom_rgba[:, 3]`` when a ghost model
        is given, and remembers the value so per-world RGBA overrides
        (``target_geom_rgba``) get the same alpha each frame.
        """
        a = float(np.clip(float(alpha), self.GHOST_ALPHA_MIN, self.GHOST_ALPHA_MAX))
        self._ghost_alpha = a
        if target_model is not None:
            target_model.geom_rgba[:, 3] = a
        return a

    def _handle_ghost_alpha_keys(self, target_model=None, target_geom_rgba=None):
        """Poll ``[8]``/``[9]`` and apply the resulting alpha to the ghost.

        Tap = ``GHOST_ALPHA_STEP`` per press; hold = ``GHOST_ALPHA_HOLD`` per
        frame (GLFW key-repeat), so a long ramp doesn't need 12 taps. Returns
        ``target_geom_rgba`` with its alpha channel synced — the per-world
        override array is rebuilt by the handler each frame from the ghost
        model's *original* colours, so the override has to be re-applied here
        rather than only on the model.

        No-op without a ghost model: the keys stay unconsumed so nothing
        surprising happens while the overlay is toggled off ([H]).
        """
        if target_model is None:
            return target_geom_rgba

        if self._ghost_alpha is None:
            # Seed from the model's own alpha (setup_ghost_hand_model paints
            # every geom the same RGBA, so max == the uniform value).
            self._ghost_alpha = float(np.max(target_model.geom_rgba[:, 3]))

        try:
            held = self.get_key_repeated_list()
        except Exception:
            held = []
        ramping = (self.GHOST_ALPHA_KEY_DOWN in held) or (self.GHOST_ALPHA_KEY_UP in held)

        delta = 0.0
        if self.is_key_pressed_once(self.GHOST_ALPHA_KEY_DOWN): delta -= self.GHOST_ALPHA_STEP
        if self.is_key_pressed_once(self.GHOST_ALPHA_KEY_UP):   delta += self.GHOST_ALPHA_STEP
        if self.GHOST_ALPHA_KEY_DOWN in held:                   delta -= self.GHOST_ALPHA_HOLD
        if self.GHOST_ALPHA_KEY_UP   in held:                   delta += self.GHOST_ALPHA_HOLD

        if delta != 0.0:
            self.set_ghost_alpha(self._ghost_alpha + delta, target_model)
            # Print on a tap, and once when a hold-ramp ends — a per-frame
            # print while ramping would flood the console.
            if not ramping:
                print(f'[ghost] alpha = {self._ghost_alpha:.2f}')
        elif self._ghost_alpha_ramping and not ramping:
            print(f'[ghost] alpha = {self._ghost_alpha:.2f}')
        self._ghost_alpha_ramping = ramping

        # Keep the model in sync even on frames with no key event (the handler
        # may have rebuilt / recoloured the ghost model in between).
        target_model.geom_rgba[:, 3] = self._ghost_alpha
        if target_geom_rgba is not None:
            target_geom_rgba = np.array(target_geom_rgba, dtype=np.float32, copy=True)
            target_geom_rgba[..., 3] = self._ghost_alpha
        return target_geom_rgba

    # ------------------------------------------------------------------
    # Contact visualisation — real-hand force arrows + ghost collision
    # ------------------------------------------------------------------
    # Two independent overlays, both polled/drawn inside ``mjwarp_render`` so
    # any ghost-rendering caller gets them without its own bookkeeping:
    #
    #   [K] real hand — one arrow per ACTIVE contact in the warp sim, drawn
    #       along the **world-frame contact force** (``mujoco_warp.contact_force``
    #       with ``to_world_frame=True``: normal + friction, not just the
    #       normal), length ∝ ‖f‖. This is ground truth from ``d.contact`` /
    #       ``d.efc_force`` — independent of the contact *sensors* that drive
    #       the existing per-type markers ([O]/[B]/[N], which draw the sensor
    #       NORMAL because the sensor's force channel is expressed in the
    #       contact frame and its tangent axes are not emitted).
    #
    #   [J] ghost — the ghost is a kinematic FK overlay, so its "contacts" are
    #       whatever ``mj_forward`` (already run by ``set_ghost_targets``)
    #       finds inside the ghost model = **self-collision + floor** of the
    #       TARGET pose. Colliding ghost geoms turn red and each contact gets
    #       a sphere + force arrow. The ghost model holds no object, so
    #       ghost↔object interpenetration is out of scope here (would need the
    #       object compiled into the ghost model + its pose copied per frame).
    #
    # [6]/[7] halve / double the shared arrow length scale (m per N).
    CONTACT_FORCE_KEY      = glfw.KEY_K
    GHOST_CONTACT_KEY      = glfw.KEY_J
    CONTACT_SCALE_DOWN_KEY = glfw.KEY_6
    CONTACT_SCALE_UP_KEY   = glfw.KEY_7

    CONTACT_FORCE_SCALE       = 0.005      # metres of arrow per Newton
    CONTACT_ARROW_MIN_LEN     = 0.008
    CONTACT_ARROW_MAX_LEN     = 0.080      # ≈ hand scale; 16 N saturates
    CONTACT_FORCE_EPS         = 1e-5       # N — below this a contact is "not pushing"
    CONTACT_FORCE_SPHERE_R    = 0.0040
    CONTACT_FORCE_SPHERE_RGBA = (0.10, 0.90, 1.00, 0.95)   # cyan  — real hand
    CONTACT_FORCE_ARROW_RGBA  = (0.10, 0.90, 1.00, 0.95)
    CONTACT_FORCE_ARROW_R     = 0.0016
    MAX_CONTACT_ARROWS        = 512        # strongest-first cap across all worlds

    GHOST_CONTACT_RGB         = (1.00, 0.15, 0.10)         # colliding ghost geoms
    GHOST_CONTACT_SPHERE_R    = 0.0045
    GHOST_CONTACT_SPHERE_RGBA = (1.00, 0.35, 0.95, 0.95)   # magenta — ghost
    GHOST_CONTACT_ARROW_RGBA  = (1.00, 0.35, 0.95, 0.95)
    GHOST_CONTACT_ARROW_R     = 0.0016
    GHOST_DEPTH_TO_FORCE      = 1000.0     # N per metre of penetration (arrow length)
    GHOST_MAX_CONTACTS        = 32         # per world

    @property
    def contact_force_vis(self):
        """Real-hand contact-force arrow overlay on/off ([K])."""
        return bool(getattr(self, '_contact_force_vis', True))

    @property
    def ghost_contact_vis(self):
        """Ghost collision overlay on/off ([J])."""
        return bool(getattr(self, '_ghost_contact_vis', True))

    @property
    def contact_force_scale(self):
        """Arrow length per Newton ([6] / [7])."""
        return float(getattr(self, '_contact_force_scale', self.CONTACT_FORCE_SCALE))

    def _handle_contact_vis_keys(self):
        """Poll ``[J] / [K] / [6] / [7]`` for the contact overlays."""
        if self.is_key_pressed_once(self.CONTACT_FORCE_KEY):
            self._contact_force_vis = not self.contact_force_vis
            print(f'[contact] hand force arrows: {"ON" if self._contact_force_vis else "OFF"}')
        if self.is_key_pressed_once(self.GHOST_CONTACT_KEY):
            self._ghost_contact_vis = not self.ghost_contact_vis
            print(f'[contact] ghost collision  : {"ON" if self._ghost_contact_vis else "OFF"}')
        scale = self.contact_force_scale
        if self.is_key_pressed_once(self.CONTACT_SCALE_DOWN_KEY): scale *= 0.5
        if self.is_key_pressed_once(self.CONTACT_SCALE_UP_KEY):   scale *= 2.0
        if scale != self.contact_force_scale:
            self._contact_force_scale = float(np.clip(scale, 1e-5, 10.0))
            print(f'[contact] arrow scale = {self._contact_force_scale:.4g} m/N')

    def _arrow_len(self, magnitude: float) -> float:
        """Force magnitude (N) → arrow length (m), scaled and clamped."""
        return float(np.clip(magnitude * self.contact_force_scale,
                             self.CONTACT_ARROW_MIN_LEN, self.CONTACT_ARROW_MAX_LEN))

    # ── Real hand: world-frame contact forces from the warp sim ────────
    def get_parallel_contact_forces(self, mjwarp_model, mjwarp_data):
        """Active contacts of the warp sim as ``(pos, force, worldid, geom)``.

        ``force`` is the **world-frame** linear contact force (N) from
        ``mujoco_warp.contact_force(..., to_world_frame=True)`` — the full
        cone force, so sliding contacts point along friction+normal rather
        than along the normal alone. Returns ``None`` when the data object has
        no warp contact buffer (CPU ``MjData``) or when nothing is touching.
        """
        contact = getattr(mjwarp_data, "contact", None)
        nacon   = getattr(mjwarp_data, "nacon",   None)
        if contact is None or nacon is None or not hasattr(nacon, "numpy"):
            return None
        n = int(nacon.numpy()[0])
        if n <= 0:
            return None

        # Sized exactly to the live contact count: the kernel launches over
        # ``contact_ids``, so a padded/cached array would evaluate stale slots.
        ids = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32)
        out = wp.zeros(n, dtype=wp.spatial_vectorf)
        mjwarp.contact_force(mjwarp_model, mjwarp_data,   # type: ignore[attr-defined]
                             ids, True, out)
        force = out.numpy()[:n, :3].astype(np.float32)    # spatial top = linear force
        pos   = contact.pos.numpy()[:n].astype(np.float32)
        wid   = contact.worldid.numpy()[:n].astype(int)
        geom  = contact.geom.numpy()[:n].astype(int)
        return pos, force, wid, geom

    def plot_parallel_contact_force_arrows(
            self,
            mjwarp_model,
            mjwarp_data,
            n_worlds:  int,
            grid_offsets = None,
            sphere_r:    float = None,      # type: ignore[assignment]
            sphere_rgba        = None,
            arrow_r:     float = None,      # type: ignore[assignment]
            arrow_rgba         = None,
        ):
        """Draw one sphere + world-frame force arrow per active sim contact.

        Contacts are drawn strongest-first and capped at
        ``MAX_CONTACT_ARROWS`` so a pile-up can't exhaust the scene's geom
        budget; the first time the cap bites it says so on stdout.
        """
        got = self.get_parallel_contact_forces(mjwarp_model, mjwarp_data)
        if got is None:
            return 0
        pos, force, wid, _geom = got

        mag  = np.linalg.norm(force, axis=1)
        keep = (mag > self.CONTACT_FORCE_EPS) & (wid >= 0) & (wid < int(n_worlds))
        keep &= np.isfinite(pos).all(1) & np.isfinite(force).all(1)
        idx  = np.where(keep)[0]
        if idx.size == 0:
            return 0
        if idx.size > self.MAX_CONTACT_ARROWS:
            idx = idx[np.argsort(-mag[idx])[:self.MAX_CONTACT_ARROWS]]
            if not getattr(self, "_contact_arrow_cap_warned", False):
                self._contact_arrow_cap_warned = True
                print(f'[contact] >{self.MAX_CONTACT_ARROWS} active contacts — '
                      f'drawing the strongest {self.MAX_CONTACT_ARROWS} only.')

        grid = np.asarray(grid_offsets, dtype=float) if grid_offsets is not None else None
        s_r    = self.CONTACT_FORCE_SPHERE_R    if sphere_r    is None else sphere_r
        s_rgba = self.CONTACT_FORCE_SPHERE_RGBA if sphere_rgba is None else sphere_rgba
        a_r    = self.CONTACT_FORCE_ARROW_R     if arrow_r     is None else arrow_r
        a_rgba = self.CONTACT_FORCE_ARROW_RGBA  if arrow_rgba  is None else arrow_rgba

        for c in idx:
            w = int(wid[c])
            p = pos[c].astype(float)
            if grid is not None and w < len(grid):
                p = p + np.array([grid[w][0], grid[w][1], 0.0])
            f = force[c].astype(float)
            m = float(mag[c])
            self.plot_sphere(p=p, r=s_r, rgba=s_rgba)
            self.plot_arrow_fr2to(p_fr=p, p_to=p + f * (self._arrow_len(m) / m),
                                  r=a_r, rgba=a_rgba, label="")
        return int(idx.size)

    # ── Ghost: collision state of the kinematic target pose ───────────
    def get_ghost_contacts(self, target_model, target_data, n_worlds: int):
        """Per-world contacts of the ghost (target-pose) model.

        ``set_ghost_targets`` already ran ``mj_forward`` on each world's ghost
        ``MjData`` (with the grid offset baked into the wrist pose), so the
        contact buffer is current and its positions are already in the
        rendered frame. Only penetrating contacts (``dist <= 0``) count.

        The force comes from ``mj_contactForce`` (the constraint solve inside
        that ``mj_forward``) rotated into the world frame. A kinematic pose
        can leave that at ~0, so we fall back to a pseudo-force
        ``depth × GHOST_DEPTH_TO_FORCE`` along the normal — the arrow then
        reads as "how deep", which is the meaningful quantity for a ghost.

        Returns ``[{pos, normal, force, mag, depth, geoms}]`` — one dict of
        arrays per world (empty arrays when that world's target pose is clean).
        """
        out = []
        empty = {"pos":   np.zeros((0, 3), np.float32),
                 "normal": np.zeros((0, 3), np.float32),
                 "force":  np.zeros((0, 3), np.float32),
                 "mag":    np.zeros((0,),   np.float32),
                 "depth":  np.zeros((0,),   np.float32),
                 "geoms":  np.zeros((0, 2), int)}
        if target_model is None or target_data is None:
            return out
        res = np.zeros(6, dtype=np.float64)
        for w in range(min(int(n_worlds), len(target_data))):
            gd = target_data[w]
            ncon = int(getattr(gd, "ncon", 0))
            if ncon <= 0:
                out.append(dict(empty))
                continue
            pos, nrm, frc, mags, depths, geoms = [], [], [], [], [], []
            for j in range(min(ncon, self.GHOST_MAX_CONTACTS)):
                con   = gd.contact[j]
                depth = -float(con.dist)          # >0 → penetrating
                if depth < 0.0:
                    continue
                frame = np.asarray(con.frame, dtype=np.float64).reshape(3, 3)
                normal = frame[0]                 # row 0 = contact normal (world)
                mujoco.mj_contactForce(target_model, gd, j, res)  # type: ignore
                f_world = frame.T @ res[:3]       # contact frame → world
                m = float(np.linalg.norm(f_world))
                if m < self.CONTACT_FORCE_EPS:    # kinematic pose → depth pseudo-force
                    m       = depth * self.GHOST_DEPTH_TO_FORCE
                    f_world = normal * m
                pos.append(np.asarray(con.pos, dtype=np.float32))
                nrm.append(normal.astype(np.float32))
                frc.append(f_world.astype(np.float32))
                mags.append(m)
                depths.append(depth)
                geoms.append([int(con.geom[0]), int(con.geom[1])])
            if not pos:
                out.append(dict(empty))
                continue
            out.append({
                "pos":    np.asarray(pos,    np.float32),
                "normal": np.asarray(nrm,    np.float32),
                "force":  np.asarray(frc,    np.float32),
                "mag":    np.asarray(mags,   np.float32),
                "depth":  np.asarray(depths, np.float32),
                "geoms":  np.asarray(geoms,  int),
            })
        return out

    def ghost_contact_geom_rgba(self, target_model, target_geom_rgba,
                                ghost_contacts, n_worlds: int):
        """Paint every ghost geom involved in a contact ``GHOST_CONTACT_RGB``.

        Composes on top of whatever colouring the handler already supplied
        (e.g. object_grasping's green/red taxonomy fingers) and keeps each
        geom's alpha, so the ``[8]/[9]`` alpha setting survives. Returns a
        ``(n_worlds, ngeom, 4)`` array, allocating one only when there is at
        least one contact to mark.
        """
        if not ghost_contacts or not any(len(c["pos"]) for c in ghost_contacts):
            return target_geom_rgba
        ngeom = int(target_model.ngeom)
        if target_geom_rgba is None:
            rgba = np.tile(np.asarray(target_model.geom_rgba, np.float32)[None],
                           (int(n_worlds), 1, 1))
        else:
            rgba = np.asarray(target_geom_rgba, dtype=np.float32)
            rgba = (np.tile(rgba[None], (int(n_worlds), 1, 1)) if rgba.ndim == 2
                    else rgba.copy())
        rgb = np.asarray(self.GHOST_CONTACT_RGB, np.float32)
        for w, c in enumerate(ghost_contacts[:len(rgba)]):
            g = np.asarray(c["geoms"], int).reshape(-1)
            g = g[(g >= 0) & (g < ngeom)]
            if g.size:
                rgba[w, g, :3] = rgb
        return rgba

    def plot_ghost_contact_markers(self, ghost_contacts):
        """Sphere + force arrow at each ghost contact (positions already carry
        the per-world grid offset via the ghost ``MjData``).

        The arrow **direction** is the solver's contact force, but its
        **length follows penetration depth** (``depth × GHOST_DEPTH_TO_FORCE``)
        rather than ‖f‖: a kinematically forced pose makes the solver report
        hundreds of newtons (measured: 100–580 N on a fully curled hand), which
        would peg every arrow at the clamp. Depth is what actually separates
        "grazing" from "finger through finger" on a ghost.
        """
        n = 0
        for c in ghost_contacts:
            for i in range(len(c["pos"])):
                p = c["pos"][i].astype(float)
                if not np.isfinite(p).all():
                    continue
                f = c["force"][i].astype(float)
                m = float(np.linalg.norm(f))
                self.plot_sphere(p=p, r=self.GHOST_CONTACT_SPHERE_R,
                                 rgba=self.GHOST_CONTACT_SPHERE_RGBA)
                if m > self.CONTACT_FORCE_EPS:
                    length = self._arrow_len(
                        float(c["depth"][i]) * self.GHOST_DEPTH_TO_FORCE)
                    self.plot_arrow_fr2to(
                        p_fr=p, p_to=p + f * (length / m),
                        r=self.GHOST_CONTACT_ARROW_R,
                        rgba=self.GHOST_CONTACT_ARROW_RGBA, label="")
                n += 1
        return n

    def viewer_contact_vis_overlay(self, loc: str = 'bottom left'):
        """Emit the on-screen legend for the ghost-alpha ([8]/[9]) and contact
        ([J]/[K]/[6]/[7]) overlays, including the live contact counts of the
        last rendered frame.

        Lives with the features rather than in the caller's overlay block, so
        the text can never drift from the actual keybindings. Call it from a
        render loop's overlay section (``evaluate.py`` does, gated by [M]).

        ASCII only: the viewer's bitmap font has no glyphs past ASCII, so a
        stray Hangul / math symbol renders as garbage boxes on screen.
        """
        a = self.ghost_alpha
        self.viewer_text_overlay(
            loc=loc,
            text1="ghost alpha [8] dim / [9] bright:",
            text2=(f"{a:.2f}" if a is not None else "n/a (no ghost)")
                  + "   (tap = +/-0.05, hold = ramp)")
        self.viewer_text_overlay(
            loc=loc,
            text1="hand force [K] / ghost collision [J]:",
            text2=(f"K={'on' if self.contact_force_vis else 'off'} "
                   f"({self._n_contact_arrows} contacts, cyan)   "
                   f"J={'on' if self.ghost_contact_vis else 'off'} "
                   f"({self._n_ghost_contacts} penetrating, magenta + red geoms)"))
        self.viewer_text_overlay(
            loc=loc,
            text1="  force arrow scale [6] half / [7] double:",
            text2=(f"{self.contact_force_scale:.4g} m/N   "
                   f"(hand: world force, len ~ |f|;  ghost: len ~ penetration depth)"))

    def _handle_contact_overlays(self, mjwarp_model, mjwarp_data,
                                 target_model, target_data, target_geom_rgba,
                                 world_spacing: float):
        """Poll the overlay keys and draw both layers; returns the (possibly
        recoloured) ``target_geom_rgba``. Called from ``mjwarp_render``.

        Both layers are best-effort: a viewer overlay must never take the
        rollout down, so failures disable nothing and raise nothing.
        """
        self._handle_contact_vis_keys()

        n_worlds = int(getattr(mjwarp_data, "nworld", 0) or 0)
        # Prefer the offsets parallel_render actually cached (they were built
        # with the spacing of the FIRST render call); recompute only as a
        # fallback so markers can never land in a different grid than the hands.
        cached = getattr(self.viewer, "_offsets_for_parallel_render", None)
        if cached is not None and len(cached) >= n_worlds > 0:
            grid = np.asarray(cached, dtype=float)[:n_worlds, :2]
        elif n_worlds > 0:
            grid = MJWarpMinimalViewer.compute_world_offsets(n_worlds, spacing=world_spacing)
        else:
            grid = None

        self._n_contact_arrows = 0
        self._n_ghost_contacts = 0
        if self.contact_force_vis and n_worlds > 0:
            try:
                self._n_contact_arrows = self.plot_parallel_contact_force_arrows(
                    mjwarp_model, mjwarp_data, n_worlds, grid_offsets=grid)
            except Exception as e:
                if not getattr(self, "_contact_force_vis_warned", False):
                    self._contact_force_vis_warned = True
                    print(f'[contact] hand force arrows disabled ({type(e).__name__}: {e})')

        if self.ghost_contact_vis and target_model is not None and target_data is not None:
            try:
                gc = self.get_ghost_contacts(target_model, target_data,
                                             n_worlds or len(target_data))
                target_geom_rgba = self.ghost_contact_geom_rgba(
                    target_model, target_geom_rgba, gc, n_worlds or len(target_data))
                self._n_ghost_contacts = self.plot_ghost_contact_markers(gc)
            except Exception as e:
                if not getattr(self, "_ghost_contact_vis_warned", False):
                    self._ghost_contact_vis_warned = True
                    print(f'[contact] ghost collision overlay disabled ({type(e).__name__}: {e})')
        return target_geom_rgba

    def _parse_xml(self,rel_xml_path, xml_string=None):
        """
        Parse the MuJoCo XML file

        Parameters:
            rel_xml_path (str): Relative path to the MuJoCo XML file
        """
        
        if self.xml_string is not None:
            self.model = mujoco.MjModel.from_xml_string(xml=self.xml_string,assets=self.assets)
            
        elif self.rel_xml_path is not None:
            self.full_xml_path = os.path.abspath(os.path.join(os.getcwd(),self.rel_xml_path))
            self.model         = mujoco.MjModel.from_xml_path(self.full_xml_path)

        # Parse xml model name
        parsed_strings = [s for s in self.model.names.split(b'\x00') if s] 
        parsed_strings = [s.decode('utf-8') for s in parsed_strings]
        self.model_name = parsed_strings[0]

        # Create MjData instance
        self.data = mujoco.MjData(self.model)
        self.dt   = self.model.opt.timestep
        self.HZ   = int(1/self.dt)

        # Parse integrator type
        """
        MuJoCo integrator details can be found here:
        https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjtintegrator
        """
        self.integrator = self.model.opt.integrator
        if self.integrator == mujoco.mjtIntegrator.mjINT_EULER:
            self.integrator_name = 'EULER'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            self.integrator_name = 'RK4'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT:
            self.integrator_name = 'IMPLICIT'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
            self.integrator_name = 'IMPLICITFAST'
        else:
            self.integrator_name = 'UNKNOWN'

        # Parse model dimensions (I prefer to use snake_case for variable names)
        self.n_q = self.model.nq  # number of generalized coordinates
        self.n_v = self.model.nv  # number of generalized velocities
        self.n_u = self.model.nu  # number of actuators

        # Geometry information
        self.n_geom           = self.model.ngeom # number of geometries
        self.geom_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,geom_idx)
                                 for geom_idx in range(self.n_geom)]
        self.geom_name_to_id  = {name:idx for idx,name in enumerate(self.geom_names) if name is not None}

        # Mesh information
        self.n_mesh           = self.model.nmesh # number of meshes
        self.mesh_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_MESH,mesh_idx)
                              for mesh_idx in range(self.n_mesh)]
        self.mesh_name_to_id  = {name:idx for idx,name in enumerate(self.mesh_names) if name is not None}

        # Body information
        self.n_body           = self.model.nbody # number of bodies
        self.body_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,body_idx)
                                 for body_idx in range(self.n_body)]
        self.body_name_to_id  = {name:idx for idx,name in enumerate(self.body_names) if name is not None}
        self.body_masses      = self.model.body_mass
        self.body_total_mass  = self.body_masses.sum()
        # Keep name-level parent lookup because many helpers operate on body
        # names rather than integer body ids.
        self.parent_body_names = [] # parent body names
        for b_idx in range(self.n_body):
            parent_id = self.model.body_parentid[b_idx]
            parent_body_name = self.body_names[parent_id]
            self.parent_body_names.append(parent_body_name)

        # Joint information
        self.n_joint          = self.model.njnt # number of joints
        self.joint_names      = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,joint_idx)
                                 for joint_idx in range(self.n_joint)]
        self.joint_name_to_id = {name:idx for idx,name in enumerate(self.joint_names) if name is not None}
        self.joint_types      = self.model.jnt_type # joint types
        self.joint_ranges     = self.model.jnt_range # joint ranges
        self.joint_mins       = self.joint_ranges[:,0]
        self.joint_maxs       = self.joint_ranges[:,1]

        # Free joint
        self.free_joint_idxs  = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_FREE)[0].astype(np.int32)
        self.free_joint_names = [self.joint_names[joint_idx] for joint_idx in self.free_joint_idxs]
        self.n_free_joint     = len(self.free_joint_idxs)
        self.free_dof_idxs    = []
        for jidx in self.free_joint_idxs:
            dofadr = int(self.model.jnt_dofadr[jidx])
            self.free_dof_idxs.extend(range(dofadr,dofadr+6))
        self.free_dof_idxs = np.asarray(self.free_dof_idxs,dtype=np.int32)

        # Revolute Joint
        self.rev_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_HINGE)[0].astype(np.int32)
        self.rev_joint_names  = [self.joint_names[joint_idx] for joint_idx in self.rev_joint_idxs]
        self.n_rev_joint      = len(self.rev_joint_idxs)
        self.rev_joint_mins   = self.joint_ranges[self.rev_joint_idxs,0]
        self.rev_joint_maxs   = self.joint_ranges[self.rev_joint_idxs,1]
        self.rev_joint_ranges = self.rev_joint_maxs - self.rev_joint_mins
        self.rev_qpos_idxs    = self.model.jnt_qposadr[self.rev_joint_idxs].astype(np.int32)
        self.rev_qpos_set     = set(int(idx) for idx in self.rev_qpos_idxs)

        # Prismatic Joint
        self.pri_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_SLIDE)[0].astype(np.int32)
        self.pri_joint_names  = [self.joint_names[joint_idx] for joint_idx in self.pri_joint_idxs]
        self.n_pri_joint      = len(self.pri_joint_idxs)
        self.pri_joint_mins   = self.joint_ranges[self.pri_joint_idxs,0]
        self.pri_joint_maxs   = self.joint_ranges[self.pri_joint_idxs,1]
        self.pri_joint_ranges = self.pri_joint_maxs - self.pri_joint_mins
        self.pri_qpos_idxs    = self.model.jnt_qposadr[self.pri_joint_idxs].astype(np.int32)
        self.pri_qpos_set     = set(int(idx) for idx in self.pri_qpos_idxs)

        # Revolute + prismatic Joint Information
        self.n_rev_pri_joint      = self.n_rev_joint + self.n_pri_joint
        self.rev_pri_joint_idxs   = np.concatenate([self.rev_joint_idxs,self.pri_joint_idxs])
        self.rev_pri_joint_names  = self.rev_joint_names + self.pri_joint_names
        self.rev_pri_joint_mins   = np.concatenate([self.rev_joint_mins,self.pri_joint_mins])
        self.rev_pri_joint_maxs   = np.concatenate([self.rev_joint_maxs,self.pri_joint_maxs])
        self.rev_pri_joint_ranges = self.rev_pri_joint_maxs - self.rev_pri_joint_mins
        self.rev_pri_qpos_idxs    = self.model.jnt_qposadr[self.rev_pri_joint_idxs].astype(np.int32)

        # Equality-constrained revolute joints are treated as passive so
        # downstream helpers can target only independently driven joints.
        self.n_eq = self.model.eq_data.shape[0] # number of equality constraints
        self.active_joint_names = self.rev_pri_joint_names.copy()
        self.passive_joint_names = []
        for eq_idx in range(self.n_eq): # for each joint equality constraints
            if self.model.eq_type[eq_idx] == mujoco.mjtEq.mjEQ_JOINT: # joint equality constraint
                joint1 = self.model.joint(self.model.eq_obj1id[eq_idx]).name # passive joint name
                # If the passive joint is in the active joint list, move it to the passive joint list
                if joint1 in self.active_joint_names:
                    self.active_joint_names.remove(joint1)
                    self.passive_joint_names.append(joint1)
        self.active_rev_joint_names = [j for j in self.active_joint_names if j in self.rev_joint_names]
        self.active_rev_pri_joint_names = self.active_joint_names.copy()
        self.n_active_joint  = len(self.active_joint_names)
        self.active_joint_mins = self.joint_mins[get_idxs(self.joint_names,self.active_joint_names)]
        self.active_joint_maxs = self.joint_maxs[get_idxs(self.joint_names,self.active_joint_names)]
        self.active_joint_ranges = self.active_joint_maxs - self.active_joint_mins
        self.n_passive_joint = len(self.passive_joint_names)
        self.passive_joint_mins = self.joint_mins[get_idxs(self.joint_names,self.passive_joint_names)]
        self.passive_joint_maxs = self.joint_maxs[get_idxs(self.joint_names,self.passive_joint_names)]
        self.passive_joint_ranges = self.passive_joint_maxs - self.passive_joint_mins

        # Actuator (or control) information
        self.n_ctrl           = self.model.nu # number of actuators (or controls)
        self.ctrl_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,ctrl_idx)
                                 for ctrl_idx in range(self.n_ctrl)]
        self.ctrl_name_to_id  = {name:idx for idx,name in enumerate(self.ctrl_names) if name is not None}
        self.ctrl_ranges      = self.model.actuator_ctrlrange # control range
        self.ctrl_mins        = self.ctrl_ranges[:,0]
        self.ctrl_maxs        = self.ctrl_ranges[:,1]
        self.ctrl_gears       = self.model.actuator_gear[:,0] # gears
        self.ctrl_trntypes    = self.model.actuator_trntype.copy()
        self.ctrl_gaintypes   = self.model.actuator_gaintype.copy()
        self.ctrl_biastypes   = self.model.actuator_biastype.copy()
        self.actuator_gainprm_0    = self.model.actuator_gainprm.copy()
        self.actuator_biasprm_0    = self.model.actuator_biasprm.copy()
        self.actuator_ctrlrange_0  = self.model.actuator_ctrlrange.copy()
        self.actuator_forcerange_0 = self.model.actuator_forcerange.copy()

        # Camera information
        self.n_cam            = self.model.ncam
        self.cam_names        = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_CAMERA,cam_idx)
                                 for cam_idx in range(self.n_cam)]
        self.cam_name_to_id   = {name:idx for idx,name in enumerate(self.cam_names) if name is not None}
        self.cams             = []
        self.cam_fovs         = []
        self.cam_viewports    = []
        for cam_idx in range(self.n_cam):
            cam_name = self.cam_names[cam_idx]
            cam      = mujoco.MjvCamera()
            cam.fixedcamid = self.model.cam(cam_name).id
            cam.type       = mujoco.mjtCamera.mjCAMERA_FIXED
            cam_fov        = self.model.cam_fovy[cam_idx]
            viewport       = mujoco.MjrRect(0,0,800,600) # SVGA?
            # Append
            self.cams.append(cam)
            self.cam_fovs.append(cam_fov)
            self.cam_viewports.append(viewport)

        # Control to joint mapping
        self.ctrl_qpos_idxs  = [] # qpos index attached to control, -1 if not attached to any joint
        self.ctrl_qpos_names = [] # qpos name attached to control, empty string if not attached to any joint
        self.ctrl_qpos_mins  = [] # minimum qpos value attached to control
        self.ctrl_qpos_maxs  = [] # maximum qpos value attached to control
        self.ctrl_qvel_idxs  = [] # qvel index attached to control, -1 if not attached to any joint
        self.ctrl_types      = [] # type of control (JOINT, TENDON, UNKNOWN)

        for ctrl_idx in range(self.n_ctrl):
            trntype = self.model.actuator_trntype[ctrl_idx]

            if trntype == mujoco.mjtTrn.mjTRN_JOINT:
                joint_idx = self.model.actuator(self.ctrl_names[ctrl_idx]).trnid[0]

                self.ctrl_qpos_idxs.append(self.model.jnt_qposadr[joint_idx])
                self.ctrl_qpos_names.append(self.joint_names[joint_idx])
                self.ctrl_qpos_mins.append(self.joint_ranges[joint_idx,0])
                self.ctrl_qpos_maxs.append(self.joint_ranges[joint_idx,1])
                self.ctrl_qvel_idxs.append(self.model.jnt_dofadr[joint_idx])
                self.ctrl_types.append('JOINT')

            elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
                self.ctrl_qpos_idxs.append(-1)
                self.ctrl_qpos_names.append('')
                self.ctrl_qpos_mins.append(np.nan)
                self.ctrl_qpos_maxs.append(np.nan)
                self.ctrl_qvel_idxs.append(-1)
                self.ctrl_types.append('TENDON')

            else:
                self.ctrl_qpos_idxs.append(-1)
                self.ctrl_qpos_names.append('')
                self.ctrl_qpos_mins.append(np.nan)
                self.ctrl_qpos_maxs.append(np.nan)
                self.ctrl_qvel_idxs.append(-1)
                self.ctrl_types.append('UNKNOWN (trntype=%d)'%(trntype))

        # Semantic aliases for control to joint mapping
        self.ctrl_joint_names = self.ctrl_qpos_names.copy()
        self.ctrl_joint_mins  = self.ctrl_qpos_mins.copy()
        self.ctrl_joint_maxs  = self.ctrl_qpos_maxs.copy()
        self.ctrl_qpos_name_to_id = {
            name:idx for idx,name in enumerate(self.ctrl_qpos_names) if name != ''
        }

        # Mirror the mapping in the opposite direction because notebook code
        # often starts from a joint selection and asks which actuators drive it.
        self.joint_ctrl_idxs   = [] # control indices attached to each joint
        self.joint_ctrl_names  = [] # control names attached to each joint
        self.joint_ctrl_qpos_idxs = [] # qpos indices attached to each joint through controls
        self.joint_ctrl_qvel_idxs = [] # qvel indices attached to each joint through controls
        self.joint_ctrl_types  = [] # control types attached to each joint

        for joint_idx in range(self.n_joint):
            ctrl_idxs_for_joint      = []
            ctrl_names_for_joint     = []
            ctrl_qpos_idxs_for_joint = []
            ctrl_qvel_idxs_for_joint = []
            ctrl_types_for_joint     = []

            for ctrl_idx in range(self.n_ctrl):
                trntype = self.model.actuator_trntype[ctrl_idx]

                # Only joint transmission is directly mapped to a joint
                if trntype != mujoco.mjtTrn.mjTRN_JOINT:
                    continue

                attached_joint_idx = self.model.actuator(self.ctrl_names[ctrl_idx]).trnid[0]

                if attached_joint_idx == joint_idx:
                    ctrl_idxs_for_joint.append(ctrl_idx)
                    ctrl_names_for_joint.append(self.ctrl_names[ctrl_idx])
                    ctrl_qpos_idxs_for_joint.append(self.model.jnt_qposadr[joint_idx])
                    ctrl_qvel_idxs_for_joint.append(self.model.jnt_dofadr[joint_idx])
                    ctrl_types_for_joint.append('JOINT')

            self.joint_ctrl_idxs.append(ctrl_idxs_for_joint)
            self.joint_ctrl_names.append(ctrl_names_for_joint)
            self.joint_ctrl_qpos_idxs.append(ctrl_qpos_idxs_for_joint)
            self.joint_ctrl_qvel_idxs.append(ctrl_qvel_idxs_for_joint)
            self.joint_ctrl_types.append(ctrl_types_for_joint)

        # Position-style actuator subset
        # Practical rule:
        # - joint transmission only
        # - non-zero gain term
        # - non-zero q-position bias term
        # This excludes common velocity actuators that usually use only qvel bias.
        self.position_ctrl_idxs        = []
        self.position_ctrl_names       = []
        self.position_ctrl_joint_names = []
        for ctrl_idx in range(self.n_ctrl):
            if self.ctrl_types[ctrl_idx] != 'JOINT':
                continue

            gainprm = np.asarray(self.model.actuator_gainprm[ctrl_idx],dtype=np.float64)
            biasprm = np.asarray(self.model.actuator_biasprm[ctrl_idx],dtype=np.float64)

            is_position_style = (
                np.isfinite(gainprm[0]) and
                (np.abs(gainprm[0]) > 0.0) and
                (np.abs(biasprm[1]) > 0.0)
            )
            if not is_position_style:
                continue

            self.position_ctrl_idxs.append(ctrl_idx)
            self.position_ctrl_names.append(self.ctrl_names[ctrl_idx])
            self.position_ctrl_joint_names.append(self.ctrl_joint_names[ctrl_idx])

        # Sensor information
        self.n_sensor      = self.model.nsensor
        self.sensor_names  = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SENSOR,sensor_idx)
                              for sensor_idx in range(self.n_sensor)]
        self.sensor_name_to_id = {name:idx for idx,name in enumerate(self.sensor_names) if name is not None}

        # Site information
        self.n_site        = self.model.nsite
        self.site_names    = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SITE,site_idx)
                              for site_idx in range(self.n_site)]
        self.site_name_to_id = {name:idx for idx,name in enumerate(self.site_names) if name is not None}

        # Geometry original parameters
        self.geom_friction_0 = self.model.geom_friction.copy()
        self.geom_rgba_0     = self.model.geom_rgba.copy()
        
        # Build children adjacency list for bodies
        self.body_children_list = [[] for _ in range(self.model.nbody)]
        for cid in range(1,self.model.nbody):
            pid = int(self.model.body_parentid[cid])
            self.body_children_list[pid].append(cid)
        

    def reset(self,forward=True,step=False,clear_cache=True):
        super().reset( forward=forward, step=step, clear_cache=clear_cache )
        if clear_cache:
            self.relative_obj_min = {}
            self.relative_obj_max = {}
        self.tick = 0

    # ──────────────────────────────────────────────────────────────────────
    # Touch-sensor → body geom colorization
    # ──────────────────────────────────────────────────────────────────────
    def _build_touch_color_cache(self, sensor_names):
        """Cache, for each touch sensor name, the list of geom IDs belonging
        to the body that hosts the sensor's site (or geom/body, if the
        sensor is attached differently). Also snapshots ``geom_rgba`` so
        ``reset_touch_color()`` can restore.
        """
        cache: dict[str, tuple[int, list[int]]] = {}
        m = self.model
        # build body → geom_ids once (reused across sensors).
        body_to_geoms: dict[int, list[int]] = {}
        for gid in range(m.ngeom):
            body_to_geoms.setdefault(int(m.geom_bodyid[gid]), []).append(gid)
        for name in sensor_names:
            if not name: continue
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)  # type: ignore
            if sid < 0: continue
            objtype = m.sensor_objtype[sid]
            objid   = int(m.sensor_objid[sid])
            if   objtype == mujoco.mjtObj.mjOBJ_SITE:  body_id = int(m.site_bodyid[objid])  # type: ignore
            elif objtype == mujoco.mjtObj.mjOBJ_BODY:  body_id = objid                       # type: ignore
            elif objtype == mujoco.mjtObj.mjOBJ_GEOM:  body_id = int(m.geom_bodyid[objid])   # type: ignore
            else:                                        continue
            cache[name] = (body_id, body_to_geoms.get(body_id, []))
        self._touch_color_cache: dict = cache
        self._touch_base_rgba = m.geom_rgba.copy()

    def _get_touch_color_cache(self, sensor_names):
        if (not hasattr(self, "_touch_color_cache")
                or any(n not in self._touch_color_cache for n in sensor_names if n)):
            self._build_touch_color_cache(sensor_names)
        return self._touch_color_cache

    def colorize_touch_sensors(
            self,
            sensor_values,
            scale:        float = 5.0,
            gamma:        float = 0.5,
            hot_rgba              = (1.0, 0.1, 0.1, 1.0),
            preserve_alpha: bool  = True,
            reset_first:    bool  = True,
        ):
        """Recolor the geom of every body that hosts a touch sensor,
        proportional to the latest touch reading.

        Args:
            sensor_values:  dict ``{sensor_name → scalar}``  (e.g. the output
                            of ``SingleHandSubEnv.get_touch_sensor_values``).
            scale:          touch value at which color saturates to ``hot_rgba``.
                            Smaller ``scale`` → more sensitive.
            gamma:          non-linear emphasis on small touches.
                            ``t = clip(value/scale, 0, 1) ** gamma``.
                            ``gamma=0.5`` (default, sqrt) makes light contacts
                            visible; ``gamma=1.0`` is linear; ``gamma=2.0``
                            highlights only firm contacts.
            hot_rgba:       RGBA at full touch (t=1).
            preserve_alpha: keep each geom's original alpha (so transparent
                            visual meshes stay transparent).
            reset_first:    restore each cached geom to its base rgba before
                            applying. Set False to overlay multiple sensor
                            sources without flicker.

        Notes:
            - Updates ``model.geom_rgba`` *in-place*; reflected immediately by
              the next ``mjv_updateScene`` / ``render()`` call.
            - Geoms that share materials may not visibly change (MuJoCo
              prefers material color when ``geom_matid >= 0``); this method
              is most useful for geoms whose color is encoded in
              ``geom_rgba`` directly.
            - Use ``reset_touch_color()`` to wipe all changes.
        """
        names = list(sensor_values.keys()) if isinstance(sensor_values, dict) \
                else [n for n, _ in sensor_values]
        try:
            cache = self._get_touch_color_cache(names)
        except Exception:
            return  # cache build failed; bail safely (e.g. malformed sensor names)

        # Sanity check: model may have changed since the cache was built
        # (e.g. variant reload). If shapes drift, drop the cache and rebuild
        # next call. Doing this synchronously (vs blindly indexing) prevents
        # an out-of-bounds geom_rgba write that can corrupt MuJoCo's C-level
        # buffer and surface as an occasional native crash.
        try:
            ngeom_now = int(self.model.ngeom)
            if (self._touch_base_rgba is None
                    or self._touch_base_rgba.shape[0] != ngeom_now):
                self._build_touch_color_cache(names)
                cache = self._touch_color_cache
        except Exception:
            return

        hot = np.asarray(hot_rgba, dtype=float)

        if reset_first:
            for _, (_, geom_ids) in cache.items():
                for gid in geom_ids:
                    if 0 <= gid < ngeom_now:
                        self.model.geom_rgba[gid] = self._touch_base_rgba[gid]

        items = sensor_values.items() if isinstance(sensor_values, dict) else sensor_values
        for name, value in items:
            if name not in cache: continue
            try:
                v = float(np.asarray(value).reshape(-1)[0])
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            t_lin = float(np.clip(v / scale, 0.0, 1.0))
            if t_lin <= 0.0:
                continue
            t = t_lin ** float(gamma) if gamma != 1.0 else t_lin
            _, geom_ids = cache[name]
            for gid in geom_ids:
                if not (0 <= gid < ngeom_now):
                    continue
                base = self._touch_base_rgba[gid]
                blended = (1.0 - t) * base + t * hot
                if preserve_alpha:
                    blended[3] = base[3]
                try:
                    self.model.geom_rgba[gid] = blended
                except Exception:
                    pass

    def reset_touch_color(self):
        """Restore ``model.geom_rgba`` to the base snapshot taken at the
        first ``colorize_touch_sensors`` call."""
        if hasattr(self, "_touch_base_rgba") and hasattr(self, "_touch_color_cache"):
            for _, (_, geom_ids) in self._touch_color_cache.items():
                for gid in geom_ids:
                    self.model.geom_rgba[gid] = self._touch_base_rgba[gid]

    # ──────────────────────────────────────────────────────────────────────
    # Contact-sensor visualization (works for both CPU mjData and warp Data)
    # ──────────────────────────────────────────────────────────────────────
    def plot_contact_sensor_markers(
            self,
            contact_values: dict,
            sphere_r:    float = 0.005,
            sphere_rgba        = (1.0, 0.0, 0.0, 0.9),
            arrow_len:   float = 0.025,
            arrow_r:     float = 0.0015,
            arrow_rgba         = (1.0, 0.5, 0.0, 0.95),
            normal_flip: bool  = False,
            world_offset_xy    = None,
        ):
        """Render per-contact markers (sphere + normal arrow) for a single
        env's worth of contact-sensor data.

        Unlike ``plot_contact_info`` (which queries CPU
        ``data.contact[*]``), this works with **arbitrary** sensor packets
        — including ones extracted from MuJoCo Warp's ``d.sensordata`` —
        because it consumes the structured dict produced by
        ``SingleHandSubEnv.get_contact_sensor_values``. That makes it usable
        in parallel-warp render loops where ``data.contact`` is GPU-side
        and ``plot_contact_info`` would not see anything.

        Args:
            contact_values:   dict ``{name → {'found', 'pos', 'normal'}}`` for a
                              single world. ``found`` is ``(num,)``; ``pos``
                              and ``normal`` are ``(num, 3)``. (For warp
                              parallel rendering, use
                              ``plot_parallel_contact_sensor_markers`` below
                              or pre-slice the per-world fields yourself.)
            sphere_r/rgba:    contact-point sphere radius and colour.
            arrow_len/r/rgba: normal arrow length / shaft radius / colour.
            normal_flip:      negate ``normal`` (helpful when a particular
                              site's outward direction is reversed).
            world_offset_xy:  optional 2-vec; added to ``(x,y)`` of every
                              marker so a per-world marker can be placed
                              at the right grid cell in ``mjwarp_render``.

        Sensors with ``found < 0.5`` are skipped — no idle clutter when no
        contact is reported.
        """
        if world_offset_xy is None:
            offset = np.zeros(3)
        else:
            offset = np.array([float(world_offset_xy[0]),
                               float(world_offset_xy[1]), 0.0])

        for name, v in contact_values.items():
            try:
                found  = np.asarray(v["found"]).reshape(-1)
                pos    = np.asarray(v["pos"]).reshape(-1, 3)
                normal = np.asarray(v["normal"]).reshape(-1, 3)
            except Exception:
                continue
            if normal_flip:
                normal = -normal
            for i in range(len(found)):
                if not (found[i] > 0.5):
                    continue
                p = pos[i] + offset
                if not np.isfinite(p).all():
                    continue
                self.plot_sphere(p=p, r=sphere_r, rgba=sphere_rgba)
                n = normal[i]
                n_norm = float(np.linalg.norm(n))
                if n_norm > 1e-9:
                    p_to = p + n * (arrow_len / n_norm)
                    self.plot_arrow_fr2to(
                        p_fr=p, p_to=p_to,
                        r=arrow_r, rgba=arrow_rgba, label="",
                    )

    def plot_parallel_contact_sensor_markers(
            self,
            contact_values: dict,
            grid_offsets,
            n_worlds:    int,
            sphere_r:    float = 0.005,
            sphere_rgba        = (1.0, 0.0, 0.0, 0.9),
            arrow_len:   float = 0.025,
            arrow_r:     float = 0.0015,
            arrow_rgba         = (1.0, 0.5, 0.0, 0.95),
            normal_flip: bool  = False,
        ):
        """Apply ``plot_contact_sensor_markers`` to every parallel world.

        Args:
            contact_values: dict whose values have warp-shaped batches:
                            ``{'found': (NWORLD, num), 'pos': (NWORLD, num, 3),
                              'normal': (NWORLD, num, 3)}``
                            (the direct output of
                            ``SingleHandSubEnv.get_contact_sensor_values`` when
                            called with a warp ``Data``).
            grid_offsets:   ``(NWORLD, 2)`` — same layout used by
                            ``MJWarpMinimalViewer.compute_world_offsets`` /
                            ``mjwarp_render``.
            n_worlds:       number of worlds to draw (clamped to grid_offsets).
        """
        grid = np.asarray(grid_offsets, dtype=float) if grid_offsets is not None else None
        for w in range(int(n_worlds)):
            per_world = {}
            for name, v in contact_values.items():
                try:
                    per_world[name] = {
                        "found":  v["found"][w],
                        "pos":    v["pos"][w],
                        "normal": v["normal"][w],
                    }
                except Exception:
                    continue
            offset = grid[w] if (grid is not None and w < len(grid)) else None
            self.plot_contact_sensor_markers(
                per_world,
                sphere_r=sphere_r, sphere_rgba=sphere_rgba,
                arrow_len=arrow_len, arrow_r=arrow_r, arrow_rgba=arrow_rgba,
                normal_flip=normal_flip,
                world_offset_xy=offset,
            )

    # ── Generic task-supplied debug markers ─────────────────────────────
    # ``SubEnvHandler.eval_debug_markers`` returns world-frame geometry only;
    # these two draw it. The viewer therefore needs no knowledge of what the
    # markers MEAN (force target, contact normal, gaze ray, …) — every task
    # gets 3-D overlays by overriding one handler method.
    @staticmethod
    def _marker_rgba(group, w: int, i: int, default=(1.0, 1.0, 1.0, 0.9)):
        """Resolve a group's ``rgba`` for item ``i`` of world ``w``.

        Accepts a single colour ``(4,)``, per-item ``(K, 4)``, or fully
        per-world ``(NWORLD, K, 4)`` — so a task can colour by magnitude
        without the viewer branching on which form it used."""
        c = group.get("rgba", None)
        if c is None:
            return list(default)
        c = np.asarray(c, dtype=float)
        if c.ndim == 1:
            return c.reshape(-1)[:4].tolist()
        if c.ndim == 2:
            return c[min(i, len(c) - 1)].reshape(-1)[:4].tolist()
        return c[min(w, len(c) - 1)][min(i, c.shape[1] - 1)].reshape(-1)[:4].tolist()

    def plot_debug_markers(self, groups, world_idx: int = 0, world_offset_xy=None):
        """Draw one world's worth of ``eval_debug_markers`` groups.

        Args:
            groups:          list of marker-group dicts — schema documented on
                             ``SubEnvHandler.eval_debug_markers`` (the single
                             source of truth for the contract).
            world_idx:       which world's row to slice out of the per-world
                             arrays.
            world_offset_xy: optional 2-vec added to every marker's ``(x, y)``
                             so it lands in the right ``mjwarp_render`` grid
                             cell (same convention as
                             ``plot_contact_sensor_markers``).

        Malformed / short groups are skipped rather than raised on: this runs
        inside the render loop, where one bad frame of task data must never
        take the viewer down.
        """
        if not groups:
            return
        if world_offset_xy is None:
            offset = np.zeros(3)
        else:
            offset = np.array([float(world_offset_xy[0]),
                               float(world_offset_xy[1]), 0.0])

        for g in groups:
            try:
                kind = str(g.get("kind", "sphere"))
                pos  = np.asarray(g["pos"], dtype=float)
                if pos.ndim != 3 or world_idx >= len(pos):
                    continue
                p_w   = pos[world_idx]                     # (K, 3)
                K     = len(p_w)
                vec   = g.get("vec", None)
                vec_w = (np.asarray(vec, dtype=float)[world_idx]
                         if vec is not None else None)
                found = g.get("found", None)
                fnd_w = (np.asarray(found, dtype=float).reshape(len(pos), -1)[world_idx]
                         if found is not None else None)
                label = g.get("label", None)
                lab_w = (np.asarray(label, dtype=object).reshape(len(pos), -1)[world_idx]
                         if label is not None else None)
                r     = float(g.get("r", 0.002))
            except Exception:
                continue

            for i in range(K):
                if fnd_w is not None and not (fnd_w[i] > 0.5):
                    continue
                p = p_w[i] + offset
                if not np.isfinite(p).all():
                    continue
                rgba = self._marker_rgba(g, world_idx, i)
                lab  = "" if lab_w is None else str(lab_w[i])
                if kind == "text":
                    if lab:
                        self.plot_text(p=p, text=lab)
                    continue
                if kind == "sphere":
                    self.plot_sphere(p=p, r=r, rgba=rgba, label=lab)
                    continue
                # arrow / line — need a finite, non-degenerate vec
                if vec_w is None or i >= len(vec_w):
                    continue
                v = vec_w[i]
                if not np.isfinite(v).all() or float(np.linalg.norm(v)) < 1e-6:
                    continue
                p_to = p + v
                if kind == "line":
                    self.plot_line_fr2to(p_fr=p, p_to=p_to, rgba=rgba)
                else:
                    self.plot_arrow_fr2to(p_fr=p, p_to=p_to, r=r,
                                          rgba=rgba, label=lab)

    def plot_parallel_debug_markers(self, groups, grid_offsets, n_worlds: int):
        """Apply :meth:`plot_debug_markers` to every parallel world.

        Mirrors ``plot_parallel_contact_sensor_markers``: ``grid_offsets`` is
        the ``(NWORLD, 2)`` layout from ``MJWarpMinimalViewer.compute_world_offsets``
        / ``mjwarp_render``, and ``n_worlds`` is clamped to it."""
        if not groups:
            return
        grid = np.asarray(grid_offsets, dtype=float) if grid_offsets is not None else None
        for w in range(int(n_worlds)):
            offset = grid[w] if (grid is not None and w < len(grid)) else None
            self.plot_debug_markers(groups, world_idx=w, world_offset_xy=offset)

    # overrides sim_core.parser
    def build_obj_pcd_cache( # type: ignore
                    self, 
                    obj_body_names, 
                    n_sample=256, 
                    pcd_mode="face_centroid", 
                    exclude_prefix=['bottom_',], 
                    save_post_fix:str='',
                    verbose=False, 
                    overwrite=True,
                ):
        """
        Build object-local point cloud cache for the specified bodies.

        Parameters:
            obj_body_names (list): List of object body names.
            n_sample (int): Number of sampled points.
            pcd_mode (str): Point generation mode. One of
                {"face_centroid","vertex","mesh"}.
            exclude_prefix (list of str, optional): Prefixes to exclude from the returned list.
            save_post_fix (str, optional): Postfix to add to the cache attribute name.
            verbose (bool): Verbosity flag.
            overwrite (bool): If False, keep existing cache entries.

        Returns:
            dict: Internal cache dictionary.
        """
        cache_attr = "obj_pcd_local_cache" if pcd_mode=="face_centroid" \
            else "obj_pcd_local_cache_%s"%(pcd_mode)
        if not hasattr(self,cache_attr):
            setattr(self,cache_attr,{})
        obj_pcd_local_cache = getattr(self,cache_attr)

        for obj_idx,obj_body_name in enumerate(obj_body_names):
            if (not overwrite) and (obj_body_name in obj_pcd_local_cache):
                if verbose:
                    print("[%d/%d] skip cached obj body:[%s]"%
                        (obj_idx+1,len(obj_body_names),obj_body_name))
                continue

            pcd_local = self._build_body_local_pcd(
                obj_body_name = obj_body_name,
                n_sample      = n_sample,
                pcd_mode      = pcd_mode,
                exclude_prefix = exclude_prefix,
            )

            if pcd_local is None:
                if verbose:
                    print(" No valid mesh geom found.")
                continue

            # object body name with save_post_fix 
            save_name = obj_body_name + save_post_fix
            obj_pcd_local_cache[save_name] = pcd_local

            if verbose:
                print("[build_obj_pcd_cache] [%d/%d] build obj body:[%s] mode:[%s] cached pcd:%s exclude_prefix:%s save_post_fix:%s"%
                    (obj_idx+1, len(obj_body_names), obj_body_name, pcd_mode, pcd_local.shape, exclude_prefix, save_post_fix))
               
            if (not overwrite) and (obj_body_name in self.relative_obj_min):
                continue
            if save_name in obj_pcd_local_cache:
                pcd = obj_pcd_local_cache[save_name]
                self.relative_obj_min[save_name] = -np.min(pcd, axis=0)
                self.relative_obj_max[save_name] =  np.max(pcd, axis=0)
            else:
                self.relative_obj_min[save_name] = np.zeros(3)
                self.relative_obj_max[save_name] = np.zeros(3)
        return obj_pcd_local_cache

    # overrides sim_core.parser  
    def get_obj_pcd( # type: ignore  
            self,
            obj_body_name,
            n_sample   = 256,
            pcd_mode   = "face_centroid",
            auto_build = True,
            verbose    = True,
            pcd_postfix:str='',
        ):
        """
        Get current world-frame object point cloud.

        Parameters:
            obj_body_name (str): Object body name.
            n_sample (int): Used only when auto_build=True and cache is missing.
            pcd_mode (str): Point generation mode. One of
                {"face_centroid","vertex","mesh"}.
            auto_build (bool): If True, lazily build cache if missing.

        Returns:
            np.ndarray: [N x 3]
        """
        cache_attr = "obj_pcd_local_cache" if pcd_mode=="face_centroid" \
            else "obj_pcd_local_cache_%s"%(pcd_mode)
        if not hasattr(self,cache_attr):
            setattr(self,cache_attr,{})
        obj_pcd_local_cache = getattr(self,cache_attr)

        if obj_body_name+pcd_postfix not in obj_pcd_local_cache:
            if not auto_build: # don't build cache, just raise error
                raise KeyError(
                    "[get_obj_pcd] obj_body_name:[%s] mode:[%s] not found in cache."%
                    (obj_body_name,pcd_mode)
                )
            
            # Build cache for this body (and skip if already built by another thread in the meantime)
            print_red ("[get_obj_pcd] cache miss for obj_body_name:[%s] mode:[%s]."%(
                obj_body_name+pcd_postfix,pcd_mode))
            # self.build_obj_pcd_cache(
            #     obj_body_names = [obj_body_name],
            #     n_sample       = n_sample,
            #     pcd_mode       = pcd_mode,
            #     verbose        = verbose,
            #     overwrite      = False,
            # )
            if verbose:
                print ("[get_obj_pcd] built cache for obj_body_name:[%s] mode:[%s]."%(
                    obj_body_name,pcd_mode))

        if obj_body_name not in obj_pcd_local_cache:
            raise ValueError(
                "[get_obj_pcd] failed to build object pcd for [%s] mode:[%s]."%(
                    obj_body_name,pcd_mode)
            )

        pcd_local = obj_pcd_local_cache[obj_body_name+pcd_postfix]
        p_body    = self.get_p(obj_body_name,'body')
        R_body    = self.get_R(obj_body_name,'body')
        pcd_world = self._transform_pcd(pcd_local,p_body,R_body)
        return pcd_world
    

    def get_mesh_names(self,including='',excluding='collision'):
        """
        Get a list of mesh names filtered by substring criteria.

        Parameters:
            including (str): Include only names containing this substring.
            excluding (str): Exclude names containing this substring.

        Returns:
            list: Filtered list of mesh names.
        """
        if excluding is None:
            mesh_names = [x for x in self.mesh_names if x is not None and including in x]    
        else:
            mesh_names = [x for x in self.mesh_names if x is not None and including in x and excluding not in x]
        return mesh_names
    
    def set_p_mocap(self,mocap_name='',p=np.array([0,0,0])):
        """
        Set the position of a mocap body.

        Parameters:
            mocap_name (str): Name of the mocap body.
            p (np.array): Position (3D).

        Returns:
            None
        """
        mocap_idx = self.model.body_mocapid[self.body_names.index(mocap_name)]
        self.data.mocap_pos[mocap_idx] = p
        
    def set_R_mocap(self,mocap_name='',R=np.eye(3)):
        """
        Set the orientation of a mocap body.

        Parameters:
            mocap_name (str): Name of the mocap body.
            R (np.array): 3x3 rotation matrix.

        Returns:
            None
        """
        mocap_idx = self.model.body_mocapid[self.body_names.index(mocap_name)]
        self.data.mocap_quat[mocap_idx] = r2quat(R)

    def set_pR_mocap(self,mocap_name='',p=np.array([0,0,0]),R=np.eye(3)):
        """
        Set the full pose (position + orientation) of a mocap body.

        Parameters:
            mocap_name (str): Name of the mocap body.
            p (np.array): Position.
            R (np.array): Rotation matrix.

        Returns:
            None
        """
        self.set_p_mocap(mocap_name=mocap_name,p=p)
        self.set_R_mocap(mocap_name=mocap_name,R=R)

    def step(
            self,
            ctrl          = None,
            ctrl_idxs     = None,
            ctrl_names    = None,
            joint_names   = None,
            nstep         = 1,
            step_flag     = True,
            increase_tick    = True,
        ):
        """ 
        Advance the simulation by a certain number of steps, optionally applying control inputs.
        """
        super().step(ctrl=ctrl, ctrl_idxs=ctrl_idxs, ctrl_names=ctrl_names, joint_names=joint_names, nstep=nstep, step_flag=step_flag)
        
        # if increase_tick:
        #     self.tick += 1

    def forward(
            self,
            q             = None,
            joint_names   = None,
            joint_idxs    = None,
            clip_position = True,
            increase_tick = True,
        ):
        """
        Perform forward kinematics, optionally updating the state qpos.

        Parameters:
            q (np.array, optional): New joint positions to set.
            joint_names (list, optional): Names of joints to update (overrides joint_idxs if not None).
            joint_idxs (list, optional): Indices of joints to update.

        Returns:
            None
        """
        super().forward( q=q, joint_names=joint_names, joint_idxs=joint_idxs, clip_position=clip_position )
        # if increase_tick:
        #     self.tick += 1

    # overrides sim_core.parser 
    def get_subtree_body_names(self,root_body_name,include_root=False,exclude_prefix=['bottom_',]):
        """
        Get the names of all bodies in the subtree rooted at the specified body.

        Parameters:
            root_body_name (str): Name of the root body.
            include_root (bool): Whether to include the root body in the returned list.
            exclude_prefix (list of str, optional): Prefixes to exclude from the returned list.
        """
        root_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,root_body_name)
        if root_id==-1:
            raise ValueError(f"Body '{root_body_name}' not found.")

        names=[]
        stack=[root_id]
        while stack:
            pid=stack.pop()
            for cid in self.body_children_list[pid]:
                cname=self.body_names[cid]
                if cname is None:
                    continue
                # Check for prefix exclusion
                if exclude_prefix is not None and any([cname.startswith(pref) for pref in exclude_prefix]):
                    continue
                names.append(cname)
                stack.append(cid)

        if include_root:
            # For root_body_name, also check prefix exclusion if necessary
            if exclude_prefix is None or not any([root_body_name.startswith(pref) for pref in exclude_prefix]):
                names = [root_body_name] + names
        return names
    
    # overrides sim_core.parser  
    def get_geom_idxs_from_root_body_name(
            self,
            root_body_name,
            more_body_names = None,
            include_root    = True,
            exclude_prefix  = ['bottom_',], 
        ):
        """
        Get all geometry indices in the subtree rooted at 'root_body_name',
        optionally including additional body names.

        Parameters:
            root_body_name (str): Name of the root body.
            more_body_names (list of str, optional): Additional body names to include.
            include_root (bool, optional): Whether to include the root body itself.

        Returns:
            list of int: Indices of geometries in the subtree and additional bodies.
        """
        
        # Get all body names in the subtree rooted at 'root_body_name'
        body_names = self.get_subtree_body_names(root_body_name,include_root,exclude_prefix)
        if more_body_names is not None:
            body_names += more_body_names
        
        # Aggregate geom indices
        geom_idxs = []
        for body_name in body_names:
            geom_idxs = geom_idxs + self.get_geom_idxs_from_body_name(body_name)

        # Remove duplications
        geom_idxs = list(set(geom_idxs))

        # Return
        return geom_idxs

    # overrides sim_core.parser   
    def _build_body_local_pcd( # type: ignore 
            self,
            obj_body_name,
            n_sample = 256,
            pcd_mode = "face_centroid",
            exclude_prefix=['bottom_',], 
            DENSE_SAMPLE_FACTOR=5, 
        ):
        """
        Build body-local canonical point cloud for an object body.

        Parameters:
            obj_body_name (str): Root body name of the object.
            n_sample (int): Number of sampled points.
            DENSE_SAMPLE_FACTOR (int): Factor to increase the number of samples.
            
            pcd_mode (str): Point generation mode. One of
                {"face_centroid","vertex","mesh"}.

        Returns:
            np.ndarray or None:
                [N x 3] body-local point cloud for the object. Returns None if
                no valid mesh-based point source can be constructed.
        """
        geom_idxs = self.get_geom_idxs_from_root_body_name(
            root_body_name = obj_body_name,
            include_root   = True,
            exclude_prefix = exclude_prefix,
        )

        p_body = self.get_p(obj_body_name, 'body')
        R_body = self.get_R(obj_body_name, 'body')

        vertices_tf_list = []
        prim_tf_list     = []      # primitive colliders — fallback only (see below)

        for geom_idx in geom_idxs:
            geom_type = int(self.model.geom_type[geom_idx])
            # Include SDF colliders too — they still reference a mesh asset via
            # geom_dataid, so the object PCD (used by BPS / wrist-init / min-z)
            # must build from them in SDF mode (else "No valid mesh geom found").
            if geom_type not in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SDF):
                # Primitive collider (procedural objects — see
                # grit.util.primitive_object): no mesh asset, so sample the
                # analytic surface. Kept in a SEPARATE list and used only when
                # the object has no mesh geom at all, so mesh objects that also
                # carry a primitive collider keep their previous cloud exactly.
                from grit.util.primitive_object import sample_geom_surface
                n_prim = max(1, int(np.ceil(
                    n_sample * DENSE_SAMPLE_FACTOR / max(1, len(geom_idxs)))))
                prim_pts = sample_geom_surface(
                    geom_type, self.model.geom_size[geom_idx], n_prim)
                if prim_pts is None:
                    continue
                p_geom = self.data.geom_xpos[geom_idx].copy()
                R_geom = self.data.geom_xmat[geom_idx].reshape(3, 3).copy()
                prim_tf_list.append(self._transform_pcd(
                    prim_pts, R_body.T @ (p_geom - p_body), R_body.T @ R_geom))
                continue

            mesh_idx = int(self.model.geom_dataid[geom_idx])
            if mesh_idx < 0:
                continue

            # vertices
            vert_adr = int(self.model.mesh_vertadr[mesh_idx])
            n_vert = int(self.model.mesh_vertnum[mesh_idx])
            if n_vert <= 0:
                continue
            vertices = self.model.mesh_vert[vert_adr:vert_adr + n_vert].reshape(-1, 3)

            # faces
            face_adr = int(self.model.mesh_faceadr[mesh_idx])
            n_face = int(self.model.mesh_facenum[mesh_idx])
            if n_face <= 0:
                continue
            faces = self.model.mesh_face[face_adr:face_adr + n_face].reshape(-1, 3)

            # triangle vertices
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]

            # face areas
            cross = np.cross(v1 - v0, v2 - v0)
            area = 0.5 * np.linalg.norm(cross, axis=1)

            valid_mask = area > 1e-12
            if not np.any(valid_mask):
                continue

            v0 = v0[valid_mask]
            v1 = v1[valid_mask]
            v2 = v2[valid_mask]
            area = area[valid_mask]

            # oversample beyond n_sample to get a denser point set
            n_sample_geom = max(1, int(np.ceil(n_sample * DENSE_SAMPLE_FACTOR / max(1, len(geom_idxs)))))
            prob = area / np.sum(area)
            face_idxs = np.random.choice(len(area), size=n_sample_geom, p=prob)

            sv0 = v0[face_idxs]
            sv1 = v1[face_idxs]
            sv2 = v2[face_idxs]

            # uniform barycentric sampling on triangles
            u = np.random.rand(n_sample_geom, 1)
            v = np.random.rand(n_sample_geom, 1)
            sqrt_u = np.sqrt(u)

            w0 = 1.0 - sqrt_u
            w1 = sqrt_u * (1.0 - v)
            w2 = sqrt_u * v

            vertices_sampled = w0 * sv0 + w1 * sv1 + w2 * sv2

            p_geom = self.data.geom_xpos[geom_idx].copy()
            R_geom = self.data.geom_xmat[geom_idx].reshape(3, 3).copy()

            # geom-local -> body-local
            R_offset = R_body.T @ R_geom
            p_offset = R_body.T @ (p_geom - p_body)

            vertices_tf = self._transform_pcd(vertices_sampled, p_offset, R_offset)
            vertices_tf_list.append(vertices_tf)

        if len(vertices_tf_list) == 0:
            vertices_tf_list = prim_tf_list      # primitive-only object
        if len(vertices_tf_list) == 0:
            return None

        pcd_concat = np.concatenate(vertices_tf_list, axis=0)

        # always fix the count to n_sample
        if len(pcd_concat) >= n_sample:
            idxs = farthest_point_sampling(pcd_concat, n_sample)
            pcd_sample = pcd_concat[idxs]
        else:
            # if there are too few samples, fill up with replacement
            idxs = np.random.choice(len(pcd_concat), n_sample, replace=True)
            pcd_sample = pcd_concat[idxs]

        return pcd_sample

    def loop_every(self,HZ=None,tick_every=None):
        """
        Check whether the current tick meets a particular frequency (HZ) or interval (tick_every).

        Parameters:
            HZ (float, optional): Frequency in Hz at which to return True.
            tick_every (int, optional): Tick interval at which to return True.

        Returns:
            bool: True if the condition is met, otherwise False.
        """
        # tick = int(self.get_sim_time()/self.dt)
        FLAG = False
        if HZ is not None:
            FLAG = (self.tick-1)%(int(1/self.dt/HZ))==0
        if tick_every is not None:
            FLAG = (self.tick-1)%(tick_every)==0
        return FLAG

    def init_mjwarp_viewer(
            self,
            title            = None,
            width            = 0.6,
            height           = 1.0,
            maxgeom          = 50000,
            perturbation     = True,
            fontscale        = 200,
            x_offset         = None,
            y_offset         = None,
            azimuth          = None,
            distance         = None,
            elevation        = None,
            lookat           = None,
            transparent      = None,
            contactpoint     = None,
            joint            = None,
            geomgroup_0      = None,
            geomgroup_1      = None,
            geomgroup_2      = None,
            geomgroup_3      = None,
            geomgroup_4      = None,
            geomgroup_5      = None,
            black_sky        = None,
            convex_hull      = None,
        ):
        """
        Initialize the MuJoCo Warp viewer and optionally configure the camera.

        Parameters:
            title (str):        Window title.
            width, height:      Window size in pixels or as fraction of monitor (≤1.0).
            maxgeom (int):      Maximum scene geometries.
            perturbation (bool): Enable interactive perturbation.
            fontscale (int):    Font scale (e.g. 150, 200).
            x_offset, y_offset: Window position offsets.
            azimuth, distance, elevation, lookat: Camera pose.
            transparent (bool): Transparent dynamic geoms.
            contactpoint (bool): Show contact points.
            joint (bool):       Show joint frames.
            geomgroup_0..5:     Visibility of geom groups.
            black_sky (bool):   Disable skybox.
            convex_hull (bool): Show convex hull.

        Geom group conventions:
            0 – floor/sky,  1 – collision+visual,  2 – visual mesh,
            3 – collision mesh (off by default),  4/5 – custom.
        """
        self.use_mujoco_viewer = True
        self._mjwarp_mode = True
        if title is None:
            title = self.name

        if width <= 1.0 and height <= 1.0:
            w_monitor, h_monitor = get_monitor_size()
            width  = int(width  * w_monitor)
            height = int(height * h_monitor)

        self.viewer = MJWarpMinimalViewer(
            self.model,
            self.data,
            mode         = 'window',
            title        = str(title),
            width        = width,
            height       = height,
            maxgeom      = maxgeom,
            perturbation = perturbation,
            x_offset     = x_offset,
            y_offset     = y_offset,
        )
        self.viewer.ctx = mujoco.MjrContext(self.model, fontscale) # type: ignore
        # Runtime-adjustable via [-]/[=] (set_viewer_fontscale) — keep the
        # applied value so the key handler can step from it.
        self._viewer_fontscale = int(fontscale)

        self.set_viewer(
            azimuth      = azimuth,
            distance     = distance,
            elevation    = elevation,
            lookat       = lookat,
            transparent  = transparent,
            contactpoint = contactpoint,
            joint        = joint,
            geomgroup_0  = geomgroup_0,
            geomgroup_1  = geomgroup_1,
            geomgroup_2  = geomgroup_2,
            geomgroup_3  = geomgroup_3,
            geomgroup_4  = geomgroup_4,
            geomgroup_5  = geomgroup_5,
            black_sky    = black_sky,
            convex_hull  = convex_hull,
        )

    def set_viewer(
            self,
            azimuth      = None,
            distance     = None,
            elevation    = None,
            lookat       = None,
            transparent  = None,
            contactpoint = None,
            contactwidth  = None,
            contactheight = None,
            contactrgba   = None,
            joint        = None,
            jointlength  = None,
            jointwidth   = None,
            jointrgba    = None,
            geomgroup_0  = None,
            geomgroup_1  = None,
            geomgroup_2  = None,
            geomgroup_3  = None,
            geomgroup_4  = None,
            geomgroup_5  = None,
            black_sky    = None,
            convex_hull  = None,
            update       = False,
        ):
        """
        Configure camera and visualisation options on the active viewer.

        All parameters are optional — only non-None values are applied.
        Pass update=True to immediately push the changes to the rendered scene.
        """
        if self.viewer is None:
            return

        if azimuth   is not None: self.viewer.cam.azimuth   = azimuth
        if distance  is not None: self.viewer.cam.distance  = distance
        if elevation is not None: self.viewer.cam.elevation = elevation
        if lookat    is not None: self.viewer.cam.lookat    = lookat

        if transparent is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
        if contactpoint is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contactpoint
        if contactwidth  is not None: self.model.vis.scale.contactwidth  = contactwidth
        if contactheight is not None: self.model.vis.scale.contactheight = contactheight
        if contactrgba   is not None: self.model.vis.rgba.contactpoint   = contactrgba

        if joint       is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = joint
        if jointlength is not None: self.model.vis.scale.jointlength = jointlength
        if jointwidth  is not None: self.model.vis.scale.jointwidth  = jointwidth
        if jointrgba   is not None: self.model.vis.rgba.joint        = jointrgba

        if geomgroup_0 is not None: self.viewer.vopt.geomgroup[0] = geomgroup_0
        if geomgroup_1 is not None: self.viewer.vopt.geomgroup[1] = geomgroup_1
        if geomgroup_2 is not None: self.viewer.vopt.geomgroup[2] = geomgroup_2
        if geomgroup_3 is not None: self.viewer.vopt.geomgroup[3] = geomgroup_3
        if geomgroup_4 is not None: self.viewer.vopt.geomgroup[4] = geomgroup_4
        if geomgroup_5 is not None: self.viewer.vopt.geomgroup[5] = geomgroup_5

        if black_sky is not None:
            self.viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = not black_sky
        if convex_hull is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = convex_hull

        if update:
            mujoco.mj_forward(self.model, self.data)
            mujoco.mjv_updateScene(
                self.model, self.data,
                self.viewer.vopt, self.viewer.pert, self.viewer.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value, self.viewer.scn,
            )
            mujoco.mjr_render(self.viewer.viewport, self.viewer.scn, self.viewer.ctx) # type: ignore

    def mjwarp_render(self,
                      mjwarp_data,
                      offset_body_names : list,
                      viewer_model      : list,
                      mjwarp_model,
                      label_z               : float = 3.0,
                      plot_success                  = None,
                      plot_env_number       : bool  = True,
                      assignment                    = None,
                      world_spacing         : float = 0.5,
                      contact_marker_radius : float = 0.0015,
                      per_world_geom_rgba           = None,
                      target_model                  = None,
                      target_data                   = None,
                      target_geom_rgba              = None,
        ):
        """
        Render the warp parallel environments to the viewer.

        Args:
            mjwarp_data:        MuJoCo Warp Data (d).
            offset_body_names:  Per-sub_env body names for grid offsetting.
            viewer_model:       List of CPU MjModel (one per sub_env).
            mjwarp_model:       MuJoCo Warp Model (m).
            label_z:            Z height for world-index labels.
            plot_success:       Array-like of bool (n_envs), or None.
            plot_env_number:    Show world index labels (default True).
            assignment:         List[int] of length n_envs mapping world i → variant index.
                                Must match the assignment used to fill m.geom_dataid etc.
            world_spacing:      XY grid spacing between parallel worlds (default 0.5 m).
            per_world_geom_rgba: Optional ``(n_envs, ngeom, 4)`` array — full
                                RGBA override applied per-world before each
                                world is added to the scene. Allows two
                                worlds sharing the same variant model to
                                render with **different** geom colours.
                                Disabled padding slots still get alpha→0.
            target_model:       Optional ghost (target-pose) MjModel.
            target_data:        Per-world MjData list for ``target_model``.
            target_geom_rgba:   Optional ``(n_envs, ngeom, 4)`` per-world ghost
                                colour override. Its alpha channel is driven by
                                the ``[8]`` / ``[9]`` keys (see
                                :meth:`_handle_ghost_alpha_keys`).
        """
        # Ghost alpha keys ([8] dimmer / [9] brighter). Polled here so every
        # ghost-rendering caller gets the binding without its own key block.
        target_geom_rgba = self._handle_ghost_alpha_keys(
            target_model=target_model, target_geom_rgba=target_geom_rgba)
        # Contact overlays: real-hand force arrows [K] + ghost collision [J].
        # Markers queue into ``_markers`` and are flushed by parallel_render
        # below, so they must be added before it runs.
        target_geom_rgba = self._handle_contact_overlays(
            mjwarp_model, mjwarp_data, target_model, target_data,
            target_geom_rgba, world_spacing)

        if self.use_mujoco_viewer:
            self.viewer.parallel_render(  # type: ignore
                mjwarp_data           = mjwarp_data,
                offset_body_names     = offset_body_names,
                viewer_model          = viewer_model,
                mjwarp_model          = mjwarp_model,
                label_z               = label_z,
                plot_success          = plot_success,
                plot_env_number       = plot_env_number,
                assignment            = assignment,
                offset                = world_spacing,
                contact_marker_radius = contact_marker_radius,
                per_world_geom_rgba   = per_world_geom_rgba,
                target_model          = target_model,
                target_data           = target_data,
                target_geom_rgba      = target_geom_rgba,
            )
            self.is_running = not self.viewer._paused # type: ignore
        else:
            print("[%s] Viewer NOT initialized." % (self.name))

    def grab_image(self, rsz_rate=None, interpolation=cv2.INTER_NEAREST):
        """
        Capture the current rendered frame as an RGB image.

        Parameters:
            rsz_rate (float, optional): Scale factor to resize.
            interpolation: OpenCV interpolation mode.

        Returns:
            np.ndarray: RGB image (H x W x 3, uint8).
        """
        img = np.zeros(
            (self.viewer.viewport.height, self.viewer.viewport.width, 3), # type: ignore
            dtype=np.uint8,
        )
        mujoco.mjr_render(self.viewer.viewport, self.viewer.scn, self.viewer.ctx) # type: ignore
        mujoco.mjr_readPixels(img, None, self.viewer.viewport, self.viewer.ctx) # type: ignore
        img = np.flipud(img)
        if rsz_rate is not None:
            h = int(img.shape[0] * rsz_rate)
            w = int(img.shape[1] * rsz_rate)
            img = cv2.resize(img, (w, h), interpolation=interpolation)
        return img

    def increase_tick(self, increment=1):
        """Increment the internal simulation tick counter."""
        self.tick += increment

    def get_sim_time(self, init_flag=False):
        """In mjwarp mode, sim time is derived from tick * dt (CPU data.time never advances).
        Falls back to parent implementation otherwise."""
        if self._mjwarp_mode:
            if init_flag:
                self.tick = 0
            return self.tick * self.dt
        return super().get_sim_time(init_flag=init_flag)

    def get_xyz_left_double_click_from_framebuffer(self, fovy=None):
        """Like ``get_xyz_left_double_click`` but reads depth from the *current*
        framebuffer instead of re-rendering the CPU scene.

        Use this in parallel-warp render loops where re-rendering the CPU model
        would (a) clobber the displayed parallel scene and (b) only contain
        world #0's geometry. Call this **after** ``mjwarp_render`` so the
        framebuffer holds the parallel scene whose depth corresponds to what
        the user actually clicked on.

        Returns:
            (xyz_world: np.ndarray | None, flag_click: bool)
        """
        flag_click = False
        if not self.viewer._left_double_click_pressed: # type: ignore
            return self.xyz_left_double_click, flag_click

        mx_my = self.get_viewer_mouse_xy().astype(int)
        mx, my = int(mx_my[0]), int(mx_my[1])

        # One pixel is all a click needs. This used to read the WHOLE viewport
        # (mjr_readPixels over ~2300x2100) and unproject every pixel with a
        # ``fovy``-derived pinhole; the single-pixel + exact-frustum path is
        # both cheaper and measurably more accurate (floor pixels land within
        # ~0.8 mm of z=0 vs ~1.7 mm, and the click now agrees exactly with the
        # ray the gizmo drags along).
        xyz = self.get_xyz_at_pixel_from_framebuffer(mx, my, fovy=fovy)
        if xyz is None:
            # Clicked the sky: hand back a far point on that ray so the caller's
            # distance test rejects it (= deselect), as the old far-plane depth
            # value did.
            cb  = viewer_cam_basis(self.viewer)
            ray = cam_mouse_ray(cb, mx + 0.5, my + 0.5)
            if ray is not None:
                far = self.model.stat.extent * self.model.vis.map.zfar
                xyz = ray[0] + ray[1] * float(far)
        if xyz is not None:
            self.xyz_left_double_click = np.asarray(xyz, dtype=float)
        self.viewer._left_double_click_pressed = False # type: ignore
        flag_click = True
        return self.xyz_left_double_click, flag_click

    # ------------------------------------------------------------------
    # Where is the cursor, in world coordinates?
    # ------------------------------------------------------------------
    # ``get_xyz_left_double_click_from_framebuffer`` answers this only on a
    # double click, and pays for a FULL-viewport ``mjr_readPixels`` +
    # per-pixel unprojection to do it. For a live read-out (and for
    # "point at a spot" interactions) we only ever need ONE pixel, so read a
    # 1x1 depth rect at the cursor and unproject just that — same maths as
    # sim_core's ``get_xyz_from_depth_pixel`` (pinhole from ``fovy`` +
    # ``get_T_viewer``), a few microseconds instead of a few milliseconds.
    CURSOR_XYZ_FAR_FRAC = 0.98      # depth beyond this fraction of zfar = sky

    def get_xyz_at_pixel_from_framebuffer(self, mx, my, fovy=None):
        """World xyz of the geometry under framebuffer pixel ``(mx, my)``.

        ``(mx, my)`` is in framebuffer pixels with y measured from the TOP
        (what GLFW's cursor callback gives, times ``viewer._scale``).
        Returns ``None`` when the viewer is not up, the pixel is off-screen,
        or the ray hit no geometry (background / far plane).

        Must be called AFTER the frame is rendered — it reads the depth
        buffer of what is currently on screen, so in a parallel-warp loop
        that means after ``mjwarp_render``.
        """
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return None
        vp = viewer.viewport
        W, H = int(vp.width), int(vp.height)
        mx, my = int(round(float(mx))), int(round(float(my)))
        if not (0 <= mx < W and 0 <= my < H):
            return None

        # GL pixel rows count from the BOTTOM; the cursor's y counts from the
        # top. Read a 1x1 rect so the transfer is a single pixel.
        rect = mujoco.MjrRect(vp.left + mx, vp.bottom + (H - 1 - my), 1, 1)  # type: ignore
        rgb   = np.zeros((1, 1, 3), dtype=np.uint8)
        depth = np.zeros((1, 1),    dtype=np.float32)
        try:
            mujoco.mjr_readPixels(rgb, depth, rect, viewer.ctx)  # type: ignore
        except Exception:
            return None

        extent = self.model.stat.extent
        near   = self.model.vis.map.znear * extent
        far    = self.model.vis.map.zfar  * extent
        d_buf  = float(depth[0, 0])
        depth_m = near / (1.0 - d_buf * (1.0 - near / far))     # buffer → metres
        if not np.isfinite(depth_m) or depth_m >= far * self.CURSOR_XYZ_FAR_FRAC:
            return None                                          # nothing there

        # Preferred: walk the SAME ray the gizmo drags along, out to the depth
        # we just read (``depth_m`` is eye-space z, hence the 1/cos). Using the
        # frustum MuJoCo actually rendered with — rather than a pinhole rebuilt
        # from ``model.vis.global_.fovy`` — keeps this point and the drag ray
        # exactly consistent; the two disagreed by ~4 px worth of parallax when
        # the viewer's frustum did not match that fovy.
        cb  = viewer_cam_basis(viewer)
        ray = cam_mouse_ray(cb, mx + 0.5, my + 0.5)   # centre of the read pixel
        if cb is not None and ray is not None:
            o, dvec = ray
            cos = float(dvec @ cb['fwd'])
            if cos > 1e-6:
                return o + dvec * (depth_m / cos)

        # Fallback (scene not rendered yet → no frustum): sim_core's
        # pinhole + ``get_T_viewer`` reconstruction.
        if fovy is None:
            try:
                fovy = float(self.model.vis.global_.fovy)
            except Exception:
                fovy = 45.0
        focal_scaling = 0.5 * H / np.tan(fovy * np.pi / 360.0)
        # Camera convention of get_T_viewer: +X forward, +Y left, +Z up.
        p_cam = np.array([
            depth_m,
            -(mx - W / 2.0) * depth_m / focal_scaling,
            -(my - H / 2.0) * depth_m / focal_scaling,
        ], dtype=np.float64)
        T = self.get_T_viewer()
        return T[:3, :3] @ p_cam + T[:3, 3]

    def get_xyz_at_cursor_from_framebuffer(self, fovy=None):
        """:meth:`get_xyz_at_pixel_from_framebuffer` at the current cursor."""
        try:
            mx, my = self.get_viewer_mouse_xy()
        except Exception:
            return None
        return self.get_xyz_at_pixel_from_framebuffer(mx, my, fovy=fovy)
