"""``MuJoCoParser`` — CPU ``MjModel``/``MjData`` wrapper with viewer helpers.

This is the base class of :class:`grit.util.hand_rl_parser.HandRLParserClass`.
It provides:

* model loading / name tables (``_parse_xml``; the subclass overrides this
  with a much richer version) and ``reset`` / ``step`` / ``forward``
* sim-time and wall-time bookkeeping used by the viewer overlays
* pose getters ``get_p`` / ``get_R`` / ``get_T`` for body / geom / site /
  camera / joint / sensor
* viewer lifecycle (``init_viewer`` / ``is_viewer_alive`` / ``close_viewer`` /
  ``render``), key polling, text / RGB overlays, screen capture
* marker plotting (``plot_T`` / ``plot_sphere`` / ``plot_arrow_fr2to`` /
  ``plot_line_fr2to`` / ``plot_box`` / ``plot_cylinder_fr2to`` …)
* cursor → world-point helpers (``get_T_viewer``, ``get_xyz_left_double_click``)
"""
from __future__ import annotations

import os
import time
from typing import Optional

import glfw
import mujoco
import numpy as np

from .transforms import t2p, t2r, pr2t, rpy2r, get_R_from_twopoints
from .utils import get_idxs, get_monitor_size, TicToc
from .viewer import MuJoCoMinimalViewer
from .viz import meters2xyz


class MuJoCoParser:
    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __init__(self, name=None, rel_xml_path=None, xml_string=None, assets=None, verbose=True):
        self.name = name
        self.rel_xml_path = rel_xml_path
        self.xml_string = xml_string
        self.assets = assets
        self.verbose = verbose
        self.is_running = True
        self.viewer: Optional[MuJoCoMinimalViewer] = None
        self.use_mujoco_viewer = False
        self._parse_xml(rel_xml_path=rel_xml_path)
        self.monitor_width, self.monitor_height = get_monitor_size()
        self._last_alive_t = None
        self.loop_dt = 0.0
        self.loop_hz = 0.0
        self.loop_hz_ema = 0.0
        self._loop_hz_ema_alpha = 0.05
        self.q0 = self.get_qpos()
        self.final_rgb_img = None
        self.obj_pcd_local_cache = {}
        self.tt = TicToc()
        self.reset()
        if self.verbose:
            self.print_info()

    def _parse_xml(self, rel_xml_path, xml_string=None):
        """Load the model and build the basic name tables."""
        if self.xml_string is not None:
            self.model = mujoco.MjModel.from_xml_string(self.xml_string, assets=self.assets)
        elif rel_xml_path is not None:
            self.full_xml_path = os.path.abspath(os.path.join(os.getcwd(), rel_xml_path))
            self.model = mujoco.MjModel.from_xml_path(self.full_xml_path)
        else:
            raise ValueError("MuJoCoParser needs rel_xml_path or xml_string")
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.HZ = int(round(1.0 / self.dt))
        m = self.model
        self.n_q, self.n_v, self.n_u = m.nq, m.nv, m.nu
        self.body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        self.geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
        self.joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        self.site_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
        self.sensor_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)]
        self.ctrl_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        self.n_ctrl = m.nu
        self.ctrl_ranges = m.actuator_ctrlrange
        self.ctrl_mins, self.ctrl_maxs = self.ctrl_ranges[:, 0], self.ctrl_ranges[:, 1]
        self.joint_types = m.jnt_type
        self.rev_joint_idxs = np.where(self.joint_types == mujoco.mjtJoint.mjJNT_HINGE)[0].astype(np.int32)
        self.rev_joint_names = [self.joint_names[i] for i in self.rev_joint_idxs]
        self.pri_joint_idxs = np.where(self.joint_types == mujoco.mjtJoint.mjJNT_SLIDE)[0].astype(np.int32)
        self.pri_joint_names = [self.joint_names[i] for i in self.pri_joint_idxs]
        self.rev_pri_joint_names = self.rev_joint_names + self.pri_joint_names
        self.active_joint_names = list(self.rev_pri_joint_names)
        self.ctrl_qpos_names = []
        for i in range(m.nu):
            if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                self.ctrl_qpos_names.append(self.joint_names[int(m.actuator_trnid[i, 0])])
            else:
                self.ctrl_qpos_names.append("")

    def print_info(self):
        m = self.model
        print(f"[{self.name}] nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} "
              f"njnt={m.njnt} nsite={m.nsite} nsensor={m.nsensor} dt={self.dt:g}")
        print(f"  bodies   : {self.body_names}")
        print(f"  joints   : {self.joint_names}")
        print(f"  actuators: {self.ctrl_names}")

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def reset(self, forward=True, step=False, clear_cache=True):
        mujoco.mj_resetData(self.model, self.data)
        if forward:
            self.data.qpos[:] = self.q0
            mujoco.mj_forward(self.model, self.data)
        if step:
            mujoco.mj_step(self.model, self.data)
        self.init_sim_time = self.data.time
        now = time.time()
        self.init_wall_time = now
        self.accum_wall_time = 0.0
        self.last_wall_update = now
        self.xyz_left_double_click = None
        self.xyz_right_double_click = None
        if clear_cache:
            self.clear_obj_pcd_cache()
        self.final_rgb_img = None

    def get_idxs_fwd(self, joint_names):
        return [int(self.model.joint(j).qposadr[0]) for j in joint_names]

    def get_idxs_step(self, joint_names):
        return [self.ctrl_qpos_names.index(j) for j in joint_names]

    def step(self, ctrl=None, ctrl_idxs=None, ctrl_names=None, joint_names=None, nstep=1, step_flag=True):
        if step_flag:
            if ctrl is not None:
                if ctrl_names is not None:
                    ctrl_idxs = get_idxs(self.ctrl_names, ctrl_names)
                elif joint_names is not None:
                    ctrl_idxs = self.get_idxs_step(joint_names)
                if ctrl_idxs is None:
                    self.data.ctrl[:] = ctrl
                else:
                    self.data.ctrl[ctrl_idxs] = ctrl
            mujoco.mj_step(self.model, self.data, nstep=nstep)
        self.increase_wall_time(step_flag=step_flag)

    def forward(self, q=None, joint_names=None, joint_idxs=None, clip_position=True):
        if q is not None:
            if joint_names is not None:
                joint_idxs = self.get_idxs_fwd(joint_names)
            if joint_idxs is not None:
                self.data.qpos[joint_idxs] = q
            else:
                self.data.qpos[:] = q
        if clip_position and self.rev_pri_joint_names:
            jidxs = [self.model.joint(j).id for j in self.rev_pri_joint_names]
            qidxs = [int(self.model.jnt_qposadr[j]) for j in jidxs]
            self.data.qpos[qidxs] = np.clip(self.data.qpos[qidxs],
                                            self.model.jnt_range[jidxs, 0], self.model.jnt_range[jidxs, 1])
        mujoco.mj_forward(self.model, self.data)
        self.increase_wall_time()

    def set_zero_qvel(self):
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # time bookkeeping
    # ------------------------------------------------------------------
    def get_sim_time(self, init_flag=False):
        if init_flag:
            self.init_sim_time = self.data.time
        return self.data.time - self.init_sim_time

    def reset_sim_time(self):
        self.init_sim_time = self.data.time

    def reset_wall_time(self):
        now = time.time()
        self.init_wall_time = now
        self.accum_wall_time = 0.0
        self.last_wall_update = now

    def reset_sim_wall_time(self):
        self.reset_sim_time()
        self.reset_wall_time()

    def increase_wall_time(self, step_flag=True):
        now = time.time()
        if step_flag:
            self.accum_wall_time += now - self.last_wall_update
        self.last_wall_update = now

    def get_wall_time(self, init_flag=False):
        if init_flag:
            self.accum_wall_time = 0.0
            self.last_wall_update = time.time()
        return self.accum_wall_time

    def sync_sim_wall_time(self, rate=1.0):
        diff = self.get_sim_time() - self.get_wall_time()
        if diff > 0:
            time.sleep(rate * diff)

    # ------------------------------------------------------------------
    # keys
    # ------------------------------------------------------------------
    def get_key_pressed_list(self):
        return [] if self.viewer is None else list(self.viewer._key_pressed_set)

    def get_key_repeated_list(self):
        return [] if self.viewer is None else list(self.viewer._key_repeated_set)

    def pop_key_pressed_list(self, key=None):
        if key is not None and self.viewer is not None:
            self.viewer._key_pressed_set.discard(key)

    def is_key_pressed_once(self, key=None, key_list=None):
        keys = [key] if key is not None else (list(key_list) if key_list is not None else [])
        pressed = self.get_key_pressed_list()
        for k in keys:
            if k in pressed:
                self.pop_key_pressed_list(k)
                return True
        return False

    def is_key_pressed_repeat(self, key=None, key_list=None):
        keys = [key] if key is not None else (list(key_list) if key_list is not None else [])
        held = self.get_key_pressed_list() + self.get_key_repeated_list()
        return any(k in held for k in keys)

    def is_space_clicked_once(self, button="left"):
        return self.is_key_pressed_once(glfw.KEY_SPACE)

    # ------------------------------------------------------------------
    # state getters
    # ------------------------------------------------------------------
    def get_qpos(self, joint_names=None, ctrl_names=None, flatten=True):
        if joint_names is not None and ctrl_names is not None:
            raise ValueError("provide only one of joint_names / ctrl_names")
        if joint_names is None and ctrl_names is None:
            return self.data.qpos.copy()
        if ctrl_names is not None:
            joint_names = []
            for c in ctrl_names:
                jid = int(self.model.actuator(c).trnid[0])
                if jid < 0:
                    raise ValueError(f"actuator {c!r} is not attached to a joint")
                joint_names.append(self.joint_names[jid])
        qs = []
        for j in joint_names:
            jj = self.model.joint(j)
            a, L = int(jj.qposadr[0]), int(len(jj.qpos0))
            qs.append(self.data.qpos[a:a + L].copy())
        if flatten:
            return np.concatenate(qs) if qs else np.array([])
        return np.array(qs, dtype=object)

    def get_ctrl(self, ctrl_names=None):
        if ctrl_names is None:
            return self.data.ctrl.copy()
        return self.data.ctrl[get_idxs(self.ctrl_names, list(ctrl_names))].copy()

    def set_ctrl(self, ctrl, ctrl_names=None, ctrl_idxs=None):
        if ctrl_names is not None:
            ctrl_idxs = get_idxs(self.ctrl_names, list(ctrl_names))
        if ctrl_idxs is None:
            self.data.ctrl[:] = ctrl
        else:
            self.data.ctrl[ctrl_idxs] = ctrl

    def _sensor_target(self, name):
        sid = self.model.sensor(name).id
        objtype = int(self.model.sensor_objtype[sid])
        objid = int(self.model.sensor_objid[sid])
        if objtype == mujoco.mjtObj.mjOBJ_JOINT:
            return "body", int(self.model.jnt_bodyid[objid])
        if objtype == mujoco.mjtObj.mjOBJ_SITE:
            return "site", objid
        raise ValueError(f"unsupported sensor object type {objtype} for sensor {name!r}")

    def get_p(self, name, type):
        d = self.data
        if type == "body":
            p = d.xpos[self.model.body(name).id]
        elif type == "geom":
            p = d.geom_xpos[self.model.geom(name).id]
        elif type == "site":
            p = d.site_xpos[self.model.site(name).id]
        elif type == "camera":
            p = d.cam_xpos[self.model.cam(name).id]
        elif type == "joint":
            p = d.xpos[int(self.model.jnt_bodyid[self.model.joint(name).id])]
        elif type == "sensor":
            kind, idx = self._sensor_target(name)
            p = d.xpos[idx] if kind == "body" else d.site_xpos[idx]
        else:
            raise ValueError(f"unknown type {type!r}")
        return np.array(p, copy=True)

    def get_R(self, name, type):
        d = self.data
        if type == "body":
            R = d.xmat[self.model.body(name).id]
        elif type == "geom":
            R = d.geom_xmat[self.model.geom(name).id]
        elif type == "site":
            R = d.site_xmat[self.model.site(name).id]
        elif type == "camera":
            R = d.cam_xmat[self.model.cam(name).id]
        elif type == "joint":
            R = d.xmat[int(self.model.jnt_bodyid[self.model.joint(name).id])]
        elif type == "sensor":
            kind, idx = self._sensor_target(name)
            R = d.xmat[idx] if kind == "body" else d.site_xmat[idx]
        else:
            raise ValueError(f"unknown type {type!r}")
        return np.array(R, copy=True).reshape(3, 3)

    def get_T(self, name, type):
        return pr2t(self.get_p(name, type), self.get_R(name, type))

    def get_geom_idxs_from_body_name(self, body_name):
        bid = self.body_names.index(body_name)
        return [int(i) for i in np.where(self.model.geom_bodyid == bid)[0]]

    def set_p(self, name, type, p, forward=True):
        if type == "body":
            self.model.body_pos[self.model.body(name).id] = p
        elif type == "geom":
            self.model.geom_pos[self.model.geom(name).id] = p
        elif type == "site":
            self.model.site_pos[self.model.site(name).id] = p
        else:
            raise ValueError(f"set_p: unsupported type {type!r}")
        if forward:
            mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # point clouds
    # ------------------------------------------------------------------
    @staticmethod
    def _transform_pcd(pcd, p, R):
        pcd = np.asarray(pcd, dtype=np.float64).reshape(-1, 3)
        p = np.asarray(p, dtype=np.float64).reshape(3)
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        return (R @ pcd.T).T + p

    def clear_obj_pcd_cache(self, obj_body_names=None):
        for attr in [a for a in dir(self) if a.startswith("obj_pcd_local_cache")]:
            cache = getattr(self, attr)
            if obj_body_names is None:
                setattr(self, attr, {})
            else:
                for n in obj_body_names:
                    cache.pop(n, None)
        if not hasattr(self, "obj_pcd_local_cache"):
            self.obj_pcd_local_cache = {}

    # ------------------------------------------------------------------
    # viewer lifecycle
    # ------------------------------------------------------------------
    def init_viewer(self, title=None, width=0.6, height=1.0, maxgeom=50000, perturbation=True,
                    fontscale=200, x_offset=None, y_offset=None):
        self.use_mujoco_viewer = True
        if title is None:
            title = self.name
        if width <= 1.0 and height <= 1.0:
            mw, mh = get_monitor_size()
            width, height = int(width * mw), int(height * mh)
        self.viewer = MuJoCoMinimalViewer(self.model, self.data, mode="window", title=str(title),
                                          width=width, height=height, maxgeom=maxgeom,
                                          perturbation=perturbation, x_offset=x_offset, y_offset=y_offset)
        self.viewer.ctx = mujoco.MjrContext(self.model, fontscale)
        self.viewer.set_transparency(transparent=False)
        self.viewer.set_cam_info(azimuth=173, distance=3.56, elevation=-22.56, lookat=[0, 0, 0.59])
        self.viewer.set_geomgroup(group_0=True, group_1=True, group_2=True, group_3=False)
        self.viewer.set_sitegroup(group_0=False, group_1=False)

    def is_viewer_alive(self):
        if self.viewer is None:
            return False
        try:
            if (getattr(self.viewer, "is_alive", False) and getattr(self.viewer, "window", None) is not None
                    and glfw.window_should_close(self.viewer.window)):
                self.close_viewer()
                return False
        except Exception:
            pass
        now = time.perf_counter()
        if self._last_alive_t is not None:
            self.loop_dt = now - self._last_alive_t
            if self.loop_dt > 0.0:
                self.loop_hz = 1.0 / self.loop_dt
                a = self._loop_hz_ema_alpha
                self.loop_hz_ema = self.loop_hz if self.loop_hz_ema == 0.0 else (1 - a) * self.loop_hz_ema + a * self.loop_hz
        self._last_alive_t = now
        return bool(self.viewer.is_alive)

    def close_viewer(self):
        self.use_mujoco_viewer = False
        v = getattr(self, "viewer", None)
        if v is None or not getattr(v, "is_alive", False):
            return
        if self.final_rgb_img is None:
            try:
                self.final_rgb_img, _ = self.capture_clean_viewer_rgbd_imgs()
            except Exception:
                pass
        v.close()

    def grab_current_rgbd_imgs(self):
        vp = self.viewer.viewport
        rgb = np.zeros((vp.height, vp.width, 3), dtype=np.uint8)
        depth = np.zeros((vp.height, vp.width, 1), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, vp, self.viewer.ctx)
        rgb, depth = np.flipud(rgb), np.flipud(depth)
        extent = self.model.stat.extent
        near, far = self.model.vis.map.znear * extent, self.model.vis.map.zfar * extent
        depth = (near / (1.0 - depth * (1.0 - near / far))).squeeze()
        return rgb, depth

    def capture_clean_viewer_rgbd_imgs(self):
        v = self.viewer
        if v is None or not getattr(v, "is_alive", False):
            raise RuntimeError("viewer is not alive")
        win = v.window
        try:
            should_close_prev = glfw.window_should_close(win)
        except Exception:
            should_close_prev = False
        v._markers[:] = []
        v._overlay.clear()
        v.reset_rgb_overlay()
        try:
            glfw.set_window_should_close(win, False)
        except Exception:
            pass
        v.render()
        out = self.grab_current_rgbd_imgs()
        try:
            glfw.set_window_should_close(win, should_close_prev)
        except Exception:
            pass
        return out

    def viewer_text_overlay(self, text1="", text2="", loc="bottom left"):
        self.viewer.add_overlay(loc=loc, text1=text1, text2=text2)

    def viewer_rgb_overlay(self, rgb=None, loc="top right"):
        rgb = np.asarray(rgb)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[:, :, None], 3, axis=2)
        self.viewer.plot_rgb_overlay(rgb=rgb, loc=loc)

    def render(self, reset_rgb_overlay=True):
        if glfw.KEY_ESCAPE in self.get_key_pressed_list() and self.final_rgb_img is None:
            try:
                self.final_rgb_img, _ = self.capture_clean_viewer_rgbd_imgs()
            except Exception:
                pass
        if self.use_mujoco_viewer:
            assert self.viewer is not None and getattr(self.viewer, "is_alive", False), \
                f"[{self.name}] viewer is not alive"
            self.viewer.render(reset_rgb_overlay=reset_rgb_overlay)
        else:
            print(f"[{self.name}] viewer NOT initialized")

    def update_render_settings(self):
        mujoco.mj_forward(self.model, self.data)
        mujoco.mjv_updateScene(self.model, self.data, self.viewer.vopt, self.viewer.pert, self.viewer.cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, self.viewer.scn)
        mujoco.mjr_render(self.viewer.viewport, self.viewer.scn, self.viewer.ctx)

    # ------------------------------------------------------------------
    # markers
    # ------------------------------------------------------------------
    def _label_marker(self, p, text, alpha=0.01):
        self.viewer.add_marker(pos=np.asarray(p), size=[1e-4, 1e-4, 1e-4], rgba=[1, 1, 1, alpha],
                               type=mujoco.mjtGeom.mjGEOM_SPHERE, label=text)

    def plot_T(self, T=None, p=np.array([0, 0, 0]), R=np.eye(3), plot_axis=True, axis_len=1.0,
               axis_width=0.005, axis_rgba=None, axis_alpha=None, plot_sphere=False, sphere_r=0.05,
               sphere_rgba=(1, 0, 0, 0.5), label=None, print_xyz=False):
        if T is not None:
            p, R = t2p(T), t2r(T)
        p, R = np.asarray(p, dtype=np.float64), np.asarray(R, dtype=np.float64)
        if plot_axis:
            a = 0.9 if axis_alpha is None else axis_alpha
            cols = ([1, 0, 0, a], [0, 1, 0, a], [0, 0, 1, a]) if axis_rgba is None else (axis_rgba,) * 3
            R0 = R @ rpy2r(np.deg2rad([0, 0, 90]))
            for k, (axis, col, txt) in enumerate(zip(np.eye(3), cols, ("X-axis", "Y-axis", "Z-axis"))):
                Rk = R0 @ rpy2r(np.pi / 2 * axis)
                self.viewer.add_marker(pos=p + Rk[:, 2] * axis_len / 2, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                       size=[axis_width, axis_width, axis_len / 2], mat=Rk, rgba=col,
                                       label=txt if print_xyz else "")
        if plot_sphere:
            self.viewer.add_marker(pos=p, size=[sphere_r] * 3, rgba=sphere_rgba,
                                   type=mujoco.mjtGeom.mjGEOM_SPHERE, label="")
        if label is not None:
            self._label_marker(p, label)

    def plot_sphere(self, p, r=0.1, rgba=(1, 1, 1, 1), label="", show_xyz=False):
        if p is None:
            return
        p = np.asarray(p, dtype=np.float64)
        if show_xyz:
            label = "(%.2f, %.2f, %.2f)" % tuple(p[:3])
        if p.shape[0] == 2:
            p = np.append(p, 0.0)
        self.viewer.add_marker(pos=p, size=[r, r, r], rgba=rgba, type=mujoco.mjtGeom.mjGEOM_SPHERE, label=label)

    def plot_spheres(self, p_list, r=0.1, rgba=(1, 1, 1, 1), label=""):
        for p in p_list:
            self.plot_sphere(p=p, r=r, rgba=rgba, label=label)

    def plot_text(self, p, text):
        if p is not None:
            self._label_marker(p, text)

    def plot_box(self, T=None, p=np.array([0, 0, 0]), R=np.eye(3), xlen=0.1, ylen=0.1, zlen=0.1,
                 xyz_len=None, rgba=(0.5, 0.5, 0.5, 0.5), label="", plot_axes=False, axis_len=0.05,
                 axis_width=0.002, axis_alpha=0.9):
        if T is not None:
            p, R = t2p(T), t2r(T)
        if xyz_len is not None:
            xlen, ylen, zlen = np.asarray(xyz_len, dtype=np.float64)[:3]
        self.viewer.add_marker(pos=np.asarray(p), mat=np.asarray(R, dtype=np.float64).reshape(9),
                               type=mujoco.mjtGeom.mjGEOM_BOX, size=[xlen / 2, ylen / 2, zlen / 2],
                               rgba=rgba, label=label)
        if plot_axes:
            self.plot_T(p=p, R=R, axis_len=axis_len, axis_width=axis_width, axis_alpha=axis_alpha)

    def plot_arrow(self, T=None, p=np.array([0, 0, 0]), R=np.eye(3), r=0.01, h=0.1, rgba=(1, 0, 0, 1),
                   label="", plot_axes=False, axis_len=0.05, axis_width=0.002, axis_alpha=0.9):
        if T is not None:
            p, R = t2p(T), t2r(T)
        self.viewer.add_marker(pos=np.asarray(p), mat=np.asarray(R), type=mujoco.mjtGeom.mjGEOM_ARROW,
                               size=[r, r, h * 2], rgba=rgba, label=label)
        if plot_axes:
            self.plot_T(p=p, R=R, axis_len=axis_len, axis_width=axis_width, axis_alpha=axis_alpha)

    def plot_arrow_fr2to(self, p_fr, p_to, r=1.0, rgba=(0.5, 0.5, 0.5, 0.5), label="", label_to=None,
                         plot_axes=False, axis_len=0.05, axis_width=0.002, axis_alpha=0.9):
        p_fr, p_to = np.asarray(p_fr, dtype=np.float64), np.asarray(p_to, dtype=np.float64)
        R = get_R_from_twopoints(p_fr, p_to)
        self.viewer.add_marker(pos=p_fr, mat=R, type=mujoco.mjtGeom.mjGEOM_ARROW,
                               size=[r, r, np.linalg.norm(p_to - p_fr) * 2], rgba=rgba, label=label)
        if label_to is not None:
            self._label_marker(p_to, label_to, alpha=0.0)
        if plot_axes:
            self.plot_T(p=p_fr, R=R, axis_len=axis_len, axis_width=axis_width, axis_alpha=axis_alpha)

    def plot_line_fr2to(self, p_fr, p_to, rgba=(0.5, 0.5, 0.5, 0.5), label=""):
        p_fr, p_to = np.asarray(p_fr, dtype=np.float64), np.asarray(p_to, dtype=np.float64)
        L = np.linalg.norm(p_to - p_fr)
        if L < 1e-6:
            return
        self.viewer.add_marker(pos=p_fr, mat=get_R_from_twopoints(p_fr, p_to), type=mujoco.mjtGeom.mjGEOM_LINE,
                               size=float(L), rgba=rgba, label=label)

    def plot_cylinder_fr2to(self, p_fr, p_to, r=0.01, rgba=(0.5, 0.5, 0.5, 0.5), label="", label_to=None):
        p_fr, p_to = np.asarray(p_fr, dtype=np.float64), np.asarray(p_to, dtype=np.float64)
        L = np.linalg.norm(p_to - p_fr)
        if L < 1e-6:
            return
        self.viewer.add_marker(pos=(p_fr + p_to) / 2, mat=get_R_from_twopoints(p_fr, p_to),
                               type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, r, L / 2], rgba=rgba, label=label)
        if label_to is not None:
            self._label_marker(p_to, label_to, alpha=0.0)

    def plot_time(self, loc="bottom left", plot_sim_time=True, plot_wall_time=True, plot_loop_freq=False,
                  plot_time_diff=True):
        if plot_sim_time:
            self.viewer_text_overlay("sim time", "[%.2f]sec" % self.get_sim_time(), loc=loc)
        if plot_wall_time:
            self.viewer_text_overlay("wall time", "[%.2f]sec" % self.get_wall_time(), loc=loc)
        if plot_loop_freq:
            self.viewer_text_overlay("Loop freq", "[%.1f]Hz" % self.loop_hz, loc=loc)
            self.viewer_text_overlay("Loop freq (ema)", "[%.1f]Hz" % self.loop_hz_ema, loc=loc)
        if plot_time_diff:
            self.viewer_text_overlay("Time diff", "[%.2f]sec" % (self.get_wall_time() - self.get_sim_time()), loc=loc)

    # ------------------------------------------------------------------
    # cursor / camera geometry
    # ------------------------------------------------------------------
    def get_viewer_mouse_xy(self):
        return np.array([self.viewer._last_mouse_x, self.viewer._last_mouse_y])

    def get_T_viewer(self):
        """4x4 pose of the free camera: +X looks at ``lookat``, +Z up (Z-up world)."""
        cam = self.viewer.cam
        R = rpy2r(np.deg2rad([0, -cam.elevation, cam.azimuth]))
        T_lookat = pr2t(np.array(cam.lookat, dtype=np.float64), R)
        return T_lookat @ pr2t(np.array([-cam.distance, 0, 0]), np.eye(3))

    def get_pcd_from_depth_img(self, depth_img, fovy=45):
        T = self.get_T_viewer()
        h, w = depth_img.shape[:2]
        f = 0.5 * h / np.tan(fovy * np.pi / 360.0)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        xyz_img = meters2xyz(depth_img, K)
        pts = xyz_img.reshape(-1, 3)
        world = (T[:3, :3] @ pts.T).T + T[:3, 3]
        return world, xyz_img, world.reshape(h, w, 3)

    def get_xyz_left_double_click(self, verbose=False, fovy=45):
        """World point under the cursor on a left double-click (re-renders the CPU scene)."""
        flag = False
        if self.viewer is not None and self.viewer._left_double_click_pressed:
            mx, my = self.get_viewer_mouse_xy().astype(int)
            self.viewer.render()
            _, depth = self.grab_current_rgbd_imgs()
            _, _, xyz_world = self.get_pcd_from_depth_img(depth, fovy=fovy)
            my = int(np.clip(my, 0, depth.shape[0] - 1))
            mx = int(np.clip(mx, 0, depth.shape[1] - 1))
            self.xyz_left_double_click = xyz_world[my, mx]
            self.viewer._left_double_click_pressed = False
            flag = True
            if verbose:
                print("left double click: (%.3f, %.3f, %.3f)" % tuple(self.xyz_left_double_click))
        return self.xyz_left_double_click, flag

    def plot_left_double_click(self, r=0.01, rgba=(1, 0, 0, 1), label=""):
        if self.xyz_left_double_click is not None:
            self.plot_sphere(p=self.xyz_left_double_click, r=r, rgba=rgba, label=label)
