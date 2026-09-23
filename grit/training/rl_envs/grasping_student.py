"""grit/training/rl_envs/grasping_student.py — deployable student (``env_name: grasping_student``).

:class:`GraspingStudent` = :class:`GraspingCore` with the observation a real robot
can produce: contact / clock / object channels masked (``cfg.Training.student_*``),
a **frozen partial-view** BPS feature captured once per episode (HPR from a random
camera, with depth-sensor noise), an obs-driven stage machine, a proprioceptive
history block, object obs blanked after the grasp ("blind") and a re-grasp label /
prediction hook. ``scripts/distill.py`` trains it from a :mod:`grasping_teacher`
checkpoint (DAgger + hybrid RL); ``collect_teacher_obs`` / ``collect_critic_obs``
give the privileged views for that.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import torch  # type: ignore
import warp as wp  # type: ignore

from grit.training.rl_env_base import BaseRLEnv, register_rl_env
from grit.training.rl_envs.grasping_core import GraspingCore
from grit.training.rl_envs.grasping_kernels import *  # noqa: F401,F403
from grit.util.warp_partial_view import WarpPartialViewPCD

# obs terms blanked once the student is "blind" (proximity latch / stage ≥ 1)
BLIND_TERMS_DEFAULT = ("site_pcd_err", "hand_center_pcd_err",
                       "lift_target_vec", "bps_feature")
# obs terms blanked for the WHOLE episode (no external object tracking deployed)
ALWAYS_BLIND_TERMS = ("obj_pose_diff",)


class GraspingStudent(GraspingCore):
    """Core task + deployable (partial-view · masked · history · blind) observation."""

    CONFIG_KEYS_GROUPS = {
        **GraspingCore.CONFIG_KEYS_GROUPS,
        "regrasp_blind": ("REGRASP_PRED_CONSEC", "REGRASP_STAGE_RESET_SOURCE",
                          "W_REGRASP_BONUS", "W_HOLD_CONTACT"),
    }

    # ── Student observation — cfg.Training keys override ──
    PV_GRID_RES: int   = 120
    PV_CAM_DIST: float = 0.30
    PV_CAM_BOXES       = None   # None → WarpPartialViewPCD.DEFAULT_CAM_BOXES
    PV_FREEZE_INITIAL_VIEW: bool  = True    # one frozen partial view per episode
    STUDENT_DROP_CONTACT:   bool  = True    # zero the contact obs (no tactile sensors deployed)
    STUDENT_STAGE_FROM_OBS: bool  = False   # obs-driven stage machine instead of the env tick
    STAGE_HC_DIST_M:        float = 0.06
    STAGE_NEAR_TICKS:       int   = 5
    STAGE_LIFT_FRAC:        float = 1.0
    STUDENT_STAGE_TOUCH_GATE: bool  = False
    STAGE_TOUCH_THRESH:       float = 0.02
    STUDENT_DROP_EP_STEP:   bool  = False   # zero ep_step_norm (no episode clock deployed)
    STUDENT_TRACKING_FREE:  bool  = False
    STUDENT_DROP_PCD_ERR:   bool  = False
    STUDENT_TRACK_APPROACH_ONLY: bool = False
    STUDENT_TRACK_RIGID_ATTACH:  bool = False
    STUDENT_RATTACH_BPS:         bool = False
    STUDENT_HIST_K: int = 0                 # history stacking frames
    STUDENT_HIST_SKIP_PROB: float = 0.0
    STUDENT_HIST_NOISE_STD: float = 0.0
    STUDENT_HIST_INCLUDE_ACT: bool = False
    STUDENT_ACT_DELAY_PROB: float = 0.0
    STUDENT_HIST_LAGS: str = ""
    PV_POS_NOISE_STD:       float = 0.0
    PV_DROPOUT_PROB:        float = 0.0
    PV_CAM_NOISE_STD:       float = 0.0
    RETRY_RESET_STUDENT_STAGE: bool = True  # rewind the obs-stage machine when a retry fires

    # ── Blind-after-grasp + re-grasp label ───────────────────────
    STUDENT_BLIND_AFTER_GRASP:    bool = True
    STUDENT_BLIND_FD_VEL_OBJ:     bool = True
    STUDENT_BLIND_NEAR_M:         float = 0.03   # proximity latch (≤0 → legacy stage≥1 gate)
    STUDENT_BLIND_MODE:           str  = "zero"  # zero | attach
    STUDENT_DROP_OBJ_GROUND_FORCE: bool = True
    REGRASP_PRED_CONSEC:          int  = 10
    REGRASP_STAGE_RESET_SOURCE:   str  = "gt"    # gt | pred | both
    W_REGRASP_BONUS: float = 0.0
    W_HOLD_CONTACT:  float = 0.0

    # ── obs layout: core + history + partial-view BPS tail ────────────────
    @property
    def obs_term_layout(self):
        cached = getattr(self, "_obs_term_layout_cache_full", None)
        if cached is not None:
            return cached
        terms = list(self._core_obs_terms())
        off = int(terms[-1][2])
        hist_w = int(getattr(self, "_hist_w_total", 0))
        if hist_w > 0:
            terms.append(("student_hist", off, off + hist_w)); off += hist_w
        nb = int(getattr(self, "_n_bps", 0))
        terms.append(("partial_bps_feature", off, off + nb)); off += nb
        assert off == self.obs_dim, f"obs_term_layout total {off} != obs_dim {self.obs_dim}"
        self._obs_term_layout_cache_full = terms
        return terms

    # ── buffers ───────────────────────────────────────────────────────────
    def _setup_obs_buffers(self) -> None:
        self._setup_student_hist_dims()      # obs_dim reads the history width
        super()._setup_obs_buffers()
        if int(getattr(self, "_hist_k", 0)) > 0:
            self._setup_student_hist_buffers()

    def _setup_reward_done_buffers(self) -> None:
        super()._setup_reward_done_buffers()     # core buffers + retry bookkeeping
        self._setup_student_buffers()
        self._ep_norm_col = self._ep_stage_col()
        if getattr(self, "_stage_from_obs", False):
            self._stu_near_cnt_torch = wp.to_torch(self._stu_near_cnt_wp)
        self._setup_blind_buffers()

    def _reset_task_buffers(self) -> None:
        super()._reset_task_buffers()
        for nm in ("regrasp_label_torch", "regrasp_pred_torch",
                   "_regrasp_pred_streak", "regrasp_pred_fired_torch"):
            if hasattr(self, nm):
                getattr(self, nm).zero_()
        if hasattr(self, "_blind_latch"):
            self._blind_latch.zero_()

    # ── observation: hooks of GraspingCore._collect_obs_kernel ────────────
    def _before_core_obs(self) -> None:
        self._student_pre_obs()

    def _collect_obs_tail(self) -> None:
        self._student_post_obs()

    def _apply_ep_norm_horizon(self) -> None:
        """Teacher snapshot always carries the single-attempt clock; the student
        obs only when the clock is not masked out."""
        off = self._ep_norm_col
        if off is None:
            return
        T1 = int(self.obs_ep_horizon_steps())
        wp.launch(_obs_ep_norm_horizon_kernel, dim=self.NWORLD,
                  inputs=[self.ep_step_wp, T1, int(off), self._teacher_obs_wp])
        if not getattr(self, "_drop_ep_step", False):
            wp.launch(_obs_ep_norm_horizon_kernel, dim=self.NWORLD,
                      inputs=[self.ep_step_wp, T1, int(off), self.obs_wp])

    def _collect_obs_kernel(self) -> None:
        super()._collect_obs_kernel()      # pre-hook · core blocks · student tail · clock
        self._blind_post_obs()

    def _collect_success_kernel(self) -> None:
        super()._collect_success_kernel()  # core success + retry
        self._blind_reward_extras()

    def _post_cond_update_hook(self, world_mask_np=None, is_episode_reset=True) -> None:
        super()._post_cond_update_hook(world_mask_np, is_episode_reset)
        self._student_post_cond(world_mask_np, is_episode_reset)
        if hasattr(self, "_blind_latch"):
            # proximity latch is per-episode state — clear it for reset worlds only
            if world_mask_np is None:
                self._blind_latch.zero_()
            else:
                self._blind_latch[self.done_mask_torch.bool()] = False

    def step(self, action_torch: torch.Tensor):
        # record the COMMANDED action for the history block (before obs collection)
        if getattr(self, "_hist_incl_act", False):
            self._hist_act_torch.copy_(action_torch)
        # 1-tick action delay DR: with prob p apply the previous command instead
        p = float(getattr(self, "_act_delay_prob", 0.0))
        if p > 0.0:
            if getattr(self, "_prev_cmd_torch", None) is None:
                self._prev_cmd_torch = torch.zeros_like(action_torch)
            m = torch.rand(self.NWORLD, 1, device=action_torch.device) < p
            applied = torch.where(m, self._prev_cmd_torch, action_torch)
            self._prev_cmd_torch.copy_(action_torch)
            return super().step(applied)
        return super().step(action_torch)

    # ══════════════════════════════════════════════════════════════════════
    # Student observation pipeline
    # ══════════════════════════════════════════════════════════════════════
    def _setup_student_hist_dims(self) -> None:
        cfg_tr = self.sampled_env.overall_cfg.Training
        k = int(getattr(cfg_tr, "student_hist_k", self.STUDENT_HIST_K))
        _lags_s = str(getattr(cfg_tr, "student_hist_lags", self.STUDENT_HIST_LAGS) or "")
        self._hist_lags = tuple(int(x) for x in _lags_s.split(",") if x.strip())
        if self._hist_lags:
            assert all(l > 0 for l in self._hist_lags) and k > 0
            k = len(self._hist_lags)          # multi-scale: K = number of lags
        self._hist_k = k
        self._hist_incl_act = bool(getattr(
            cfg_tr, "student_hist_include_act", self.STUDENT_HIST_INCLUDE_ACT)) and k > 0
        self._hist_obs_src_w = 2 * int(self.n_ctrl) if k > 0 else 0
        self._hist_act_w = int(self.action_dim) if self._hist_incl_act else 0
        self._hist_src_w = self._hist_obs_src_w + self._hist_act_w
        self._hist_w_total = k * self._hist_src_w

    def _setup_student_hist_buffers(self) -> None:
        cfg_tr = self.sampled_env.overall_cfg.Training
        k = int(self._hist_k)
        _t = {nm: (s, e) for nm, s, e in self.obs_term_layout}
        cols = list(range(*_t["joint_qpos"])) + list(range(*_t["torque_proxy"]))
        assert len(cols) == self._hist_obs_src_w
        self._hist_src_cols_wp = wp.array(
            np.asarray(cols, dtype=np.int32), dtype=int, device=self.device)
        self._hist_c0 = int(_t["student_hist"][0])
        self._hist_act_wp = wp.zeros(
            (self.NWORLD, max(self._hist_act_w, 1)), dtype=float, device=self.device)
        self._hist_act_torch = wp.to_torch(self._hist_act_wp)
        if self._hist_lags:
            L = max(self._hist_lags) + 1
            self._hist_ring_L = L
            self._hist_ring_wp = wp.zeros(
                (self.NWORLD, L, self._hist_src_w), dtype=float, device=self.device)
            self._hist_upd_wp  = wp.zeros(self.NWORLD, dtype=int, device=self.device)
            self._hist_skip_wp = wp.zeros(self.NWORLD, dtype=int, device=self.device)
            self._hist_upd_torch = wp.to_torch(self._hist_upd_wp)
            self._hist_lags_wp = wp.array(
                np.asarray(self._hist_lags, dtype=np.int32), dtype=int, device=self.device)
            print(f"[student-obs] multi-scale history — lags {self._hist_lags} (ring L={L})")
        self._hist_skip_prob = float(getattr(cfg_tr, "student_hist_skip_prob", self.STUDENT_HIST_SKIP_PROB))
        self._hist_noise_std = float(getattr(cfg_tr, "student_hist_noise_std", self.STUDENT_HIST_NOISE_STD))
        self._act_delay_prob = float(getattr(cfg_tr, "student_act_delay_prob", self.STUDENT_ACT_DELAY_PROB))
        self._hist_tick = 0
        print(f"[student-obs] history stacking ON — K={k} × "
              f"(joint_qpos+torque_proxy = {self._hist_obs_src_w}"
              + (f" + act = {self._hist_act_w}" if self._hist_incl_act else "")
              + f") = {self._hist_w_total} cols (obs {self.obs_dim})"
              + (f"  DR: skip_p={self._hist_skip_prob:.2f} noise_std={self._hist_noise_std:.3f}"
                 if (self._hist_skip_prob > 0 or self._hist_noise_std > 0) else ""))

    def _resolve_contact_cols(self):
        """Contiguous column range of the contact obs (per_slot_contact + obj_touch_force)."""
        terms = {nm: (s, e) for nm, s, e in self.obs_term_layout}
        c0 = terms["per_slot_contact"][0]
        c1 = terms["obj_touch_force"][1]
        assert c1 > c0, f"contact cols not contiguous: {terms}"
        return int(c0), int(c1 - c0)

    def _setup_student_buffers(self) -> None:
        """Partial-view computer + sensor masking caches + obs-driven stage machine."""
        cfg_tr    = self.sampled_env.overall_cfg.Training
        grid_res  = int(getattr(cfg_tr, "pv_grid_res", self.PV_GRID_RES))
        cam_dist  = float(getattr(cfg_tr, "pv_cam_dist", self.PV_CAM_DIST))
        self._pv_freeze   = bool(getattr(cfg_tr, "pv_freeze_initial_view", self.PV_FREEZE_INITIAL_VIEW))
        self._drop_contact = bool(getattr(cfg_tr, "student_drop_contact", self.STUDENT_DROP_CONTACT))
        pos_noise = float(getattr(cfg_tr, "pv_pos_noise_std", self.PV_POS_NOISE_STD))
        dropout   = float(getattr(cfg_tr, "pv_dropout_prob",  self.PV_DROPOUT_PROB))
        cam_noise = float(getattr(cfg_tr, "pv_cam_noise_std", self.PV_CAM_NOISE_STD))
        self.pv = WarpPartialViewPCD(
            pcd_local_per_variant=None,
            assignment=np.zeros(self.NWORLD, dtype=np.int32),   # world-frame mode: unused
            body_id=int(self.obj_body_id),
            n_point=int(self._pcd_n_pts),
            grid_res=grid_res, cam_dist=cam_dist, cam_boxes=self.PV_CAM_BOXES,
            pos_noise_std=pos_noise, dropout_prob=dropout, cam_noise_std=cam_noise,
            device=self.device,
        )
        self._pv_capture_mask_wp    = wp.zeros(self.NWORLD, dtype=int, device=self.device)
        self._pv_capture_mask_torch = wp.to_torch(self._pv_capture_mask_wp)
        self._contact_c0, self._contact_w = self._resolve_contact_cols()
        self._contact_cache_wp = wp.zeros(
            (self.NWORLD, max(int(self._contact_w), 1)), dtype=float, device=self.device)
        self._drop_ep_step = bool(getattr(cfg_tr, "student_drop_ep_step", self.STUDENT_DROP_EP_STEP))
        if self._drop_ep_step:
            _terms = {nm: (s, e) for nm, s, e in self.obs_term_layout}
            self._ep_col = int(_terms["ep_stage"][0])
            self._ep_cache_wp = wp.zeros((self.NWORLD, 1), dtype=float, device=self.device)
            print("[student-obs] ep_step_norm masked (teacher keeps it)")
        self._tracking_free = bool(getattr(cfg_tr, "student_tracking_free", self.STUDENT_TRACKING_FREE))
        self._drop_pcd_err = bool(getattr(cfg_tr, "student_drop_pcd_err", self.STUDENT_DROP_PCD_ERR))
        if self._tracking_free:
            _t = {nm: (s, e) for nm, s, e in self.obs_term_layout}
            self._site_err_c0 = int(_t["site_pcd_err"][0])
            self._hc_err_c0   = int(_t["hand_center_pcd_err"][0])
            self._objpd_c0, self._objpd_w = int(_t["obj_pose_diff"][0]), 6
            self._objpd_cache_wp   = wp.zeros((self.NWORLD, 6), dtype=float, device=self.device)
            self._pcderr_cache_wp  = wp.zeros(
                (self.NWORLD, 3 * (int(self._n_sensors) - 1) + 3), dtype=float, device=self.device)
            self._frozen_hc_dist_wp = wp.zeros(self.NWORLD, dtype=float, device=self.device)
            mode = "B(drop pcd_err)" if self._drop_pcd_err else "A(frozen pcd_err)"
            print(f"[student-obs] tracking-free ON [{mode}] — obj_pose_diff masked, "
                  f"stage: frozen-dist + wrist-z proxy")
        self._track_approach_only = bool(getattr(
            cfg_tr, "student_track_approach_only", self.STUDENT_TRACK_APPROACH_ONLY))
        if self._track_approach_only:
            assert not self._tracking_free, \
                "student_track_approach_only and student_tracking_free are mutually exclusive"
            _t2 = {nm: (s, e) for nm, s, e in self.obs_term_layout}
            s0, e0 = _t2["site_pcd_err"]
            s1, e1 = _t2["hand_center_pcd_err"]
            assert e0 == s1, "site_pcd_err and hand_center_pcd_err must be contiguous"
            self._pg_pcd_c0, self._pg_pcd_w = int(s0), int(e1 - s0)
            self._pg_objpd_c0 = int(_t2["obj_pose_diff"][0])
            self._pg_pcd_cache_wp = wp.zeros((self.NWORLD, self._pg_pcd_w), dtype=float, device=self.device)
            self._pg_objpd_cache_wp = wp.zeros((self.NWORLD, 6), dtype=float, device=self.device)
            self._track_rigid_attach = bool(getattr(
                cfg_tr, "student_track_rigid_attach", self.STUDENT_TRACK_RIGID_ATTACH))
            if self._track_rigid_attach:
                self._ra_rel_p_wp = wp.zeros(self.NWORLD, dtype=wp.vec3,  device=self.device)
                self._ra_rel_R_wp = wp.zeros(self.NWORLD, dtype=wp.mat33, device=self.device)
                self._ra_wg_p_wp  = wp.zeros(self.NWORLD, dtype=wp.vec3,  device=self.device)
                self._ra_wg_R_wp  = wp.zeros(self.NWORLD, dtype=wp.mat33, device=self.device)
                self._rattach_bps = bool(getattr(cfg_tr, "student_rattach_bps", self.STUDENT_RATTACH_BPS))
                if self._rattach_bps:
                    self._ra_pcd_scratch_wp = wp.zeros(
                        (self.NWORLD, int(self._pcd_n_pts)), dtype=wp.vec3, device=self.device)
                print("[student-obs] approach-only tracking ON [rigid-attach"
                      + ("+BPS]" if self._rattach_bps else "]")
                      + " — stage≥1: obj_pose_diff/site_pcd_err propagated by wrist FK")
            else:
                print("[student-obs] approach-only tracking ON — stage≥1: pcd_err/obj_pose_diff "
                      "frozen at last-known values + wrist-z proxy stage")
        self._stage_from_obs = bool(getattr(cfg_tr, "student_stage_from_obs", self.STUDENT_STAGE_FROM_OBS))
        if self._stage_from_obs:
            terms = {nm: (s, e) for nm, s, e in self.obs_term_layout}
            self._stage_col = int(terms["ep_stage"][0]) + 1   # [ep_step_norm, stage]
            self._stu_stage_wp     = wp.zeros(self.NWORLD, dtype=int, device=self.device)
            self._stu_near_cnt_wp  = wp.zeros(self.NWORLD, dtype=int, device=self.device)
            self._stu_stage_torch    = wp.to_torch(self._stu_stage_wp)
            self._stu_near_cnt_torch = wp.to_torch(self._stu_near_cnt_wp)
            if getattr(self, "_stu_wristz_anchor_wp", None) is None:
                self._stu_wristz_anchor_wp = wp.zeros(self.NWORLD, dtype=float, device=self.device)
                self._stu_wristz_anchor_torch = wp.to_torch(self._stu_wristz_anchor_wp)
            self._stage_hc_dist  = float(getattr(cfg_tr, "stage_hc_dist_m", self.STAGE_HC_DIST_M))
            self._stage_near_ticks = int(getattr(cfg_tr, "stage_near_ticks", self.STAGE_NEAR_TICKS))
            self._stage_touch_gate = bool(getattr(
                cfg_tr, "student_stage_touch_gate", self.STUDENT_STAGE_TOUCH_GATE))
            self._stage_touch_thresh = float(getattr(cfg_tr, "stage_touch_thresh", self.STAGE_TOUCH_THRESH))
            _tt = {nm: (s, e) for nm, s, e in self.obs_term_layout}
            self._tq_c0, self._tq_c1 = map(int, _tt["torque_proxy"])
            if self._stage_touch_gate:
                print(f"[student-stage] contact gate ON — near = dist AND "
                      f"mean|torque_proxy| > {self._stage_touch_thresh:.3f}")
            self._stage_lift_m = (float(getattr(cfg_tr, "stage_lift_frac", self.STAGE_LIFT_FRAC))
                                  * float(self.LIFT_TARGET_M))
            print(f"[student-stage] obs-driven stage ON — hc_dist<{self._stage_hc_dist:.3f}m "
                  f"x{self._stage_near_ticks}tick → lift, obj z +{self._stage_lift_m:.3f}m → hold")

    def _student_post_cond(self, world_mask_np=None, is_episode_reset=True) -> None:
        """Episode reset: clear history / stage machine, resample partial-view cameras."""
        if not is_episode_reset:
            return          # mid-episode target switch keeps the camera
        if getattr(self, "pv", None) is None:
            return
        if int(getattr(self, "_hist_k", 0)) > 0 and getattr(self, "obs_torch", None) is not None:
            _h0, _h1 = int(self._hist_c0), int(self._hist_c0 + self._hist_w_total)
            if world_mask_np is None:
                self.obs_torch[:, _h0:_h1] = 0.0
                if getattr(self, "_hist_lags", ()):
                    self._hist_upd_torch.zero_()
            else:
                _dm = self.done_mask_torch.bool()
                self.obs_torch[_dm, _h0:_h1] = 0.0
                if getattr(self, "_hist_lags", ()):
                    self._hist_upd_torch[_dm] = 0
        if getattr(self, "_stage_from_obs", False):
            if world_mask_np is None:
                self._stu_stage_torch.zero_(); self._stu_near_cnt_torch.zero_()
                self._stu_wristz_anchor_torch.zero_()
            else:
                _dm = self.done_mask_torch.bool()
                self._stu_stage_torch[_dm] = 0; self._stu_near_cnt_torch[_dm] = 0
                self._stu_wristz_anchor_torch[_dm] = 0.0
        if world_mask_np is None:
            self.pv.resample_cameras_gpu(self.d.xpos)
            self._pv_capture_mask_torch.fill_(1)
        else:
            self.pv.resample_cameras_gpu(self.d.xpos, self.done_mask_wp)
            self._pv_capture_mask_torch[self.done_mask_torch.bool()] = 1

    def _student_pre_obs(self) -> None:
        """Before the core obs: prediction-driven re-grasp rewind, retry
        rewind of the obs-stage machine, and a fresh world-frame PCD."""
        src = self._regrasp_src
        self.RETRY_RESET_STUDENT_STAGE = src in ("gt", "both")
        self.regrasp_pred_fired_torch.zero_()
        if src in ("pred", "both"):
            p = self.regrasp_pred_torch > 0.5
            self._regrasp_pred_streak = torch.where(
                p, self._regrasp_pred_streak + 1, torch.zeros_like(self._regrasp_pred_streak))
            fire = (self._regrasp_pred_streak >= int(self.REGRASP_PRED_CONSEC)) \
                & (self._stu_stage_torch >= 1)
            if bool(fire.any()):
                self._stu_stage_torch[fire] = 0
                self._stu_near_cnt_torch[fire] = 0
                self._regrasp_pred_streak[fire] = 0
                self.regrasp_pred_fired_torch[fire] = 1.0
        if self._blind_mode == "attach" and self._blind_near_m > 0.0:
            self._blind_latch |= (self.best_dist_hc_torch < self._blind_near_m)
            self._pg_gate_torch.copy_(self._blind_latch.int())
        # a retry that just fired rewinds the obs-stage machine before the
        # partial-view stage reads it (retry_fired is written by the success kernel)
        if (bool(self.RETRY_RESET_STUDENT_STAGE)
                and getattr(self, "_stage_from_obs", False)
                and getattr(self, "retry_fired_torch", None) is not None):
            m = self.retry_fired_torch > 0.5
            if bool(m.any()):
                self._stu_stage_torch[m] = 0
                self._stu_near_cnt_torch[m] = 0
        # world-frame PCD at the CURRENT pose (the reward path only refreshes it on step)
        wp.launch(
            _object_grasping_transform_pcd_kernel,
            dim=(self.NWORLD, int(self._pcd_n_pts)),
            inputs=[
                self.d.xpos, self.d.xmat, self._pcd_gpu, self._pcd_assignment,
                int(self._grasp_obj_slot), int(self._grasp_obj_body_id),
                self.transformed_pcd_wp,
            ],
        )

    def _student_post_obs(self) -> None:
        """After the core obs: teacher snapshot (unmasked), student masking,
        frozen partial-view BPS tail and history stacking."""
        _pg_gate = getattr(self, "_pg_gate_wp", None)
        if _pg_gate is None:
            _pg_gate = self._stu_stage_wp
        # teacher snapshot of the core blocks BEFORE any masking (never masked)
        if getattr(self, "_teacher_obs_wp", None) is None:
            self._teacher_obs_wp    = wp.zeros_like(self.obs_wp)
            self._teacher_obs_torch = wp.to_torch(self._teacher_obs_wp)
        wp.copy(self._teacher_obs_wp, self.obs_wp)
        tracking_free = getattr(self, "_tracking_free", False)
        if tracking_free:
            if self._drop_pcd_err:
                wp.launch(
                    _obs_cache_zero_cols_kernel,
                    dim=(self.NWORLD, 3 * (int(self._n_sensors) - 1) + 3),
                    inputs=[self.obs_wp, int(self._site_err_c0),
                            3 * (int(self._n_sensors) - 1) + 3, self._pcderr_cache_wp],
                )
            else:
                wp.launch(
                    _frozen_site_pcd_err_kernel,
                    dim=(self.NWORLD, int(self._n_sensors)),
                    inputs=[self.d.xmat, self.d.site_xpos,
                            self.pv.frozen_pcd, self.pv.visible,
                            self._finger_site_ids_wp, int(self.wrist_body_id),
                            int(self._n_sensors), int(self._pcd_n_pts),
                            int(self.hand_util.fore_arm_sensor_idx),
                            int(self._site_err_c0), self.obs_wp],
                )
            wp.launch(
                _frozen_hand_center_pcd_err_kernel, dim=self.NWORLD,
                inputs=[self.d.xpos, self.d.xmat,
                        self.pv.frozen_pcd, self.pv.visible,
                        int(self.wrist_body_id), self._hand_center_offset_vec,
                        int(self._pcd_n_pts), int(self._hc_err_c0),
                        self.obs_wp, self._frozen_hc_dist_wp],
            )
            if self._drop_pcd_err:
                wp.launch(
                    _obs_cache_zero_cols_kernel, dim=(self.NWORLD, 3),
                    inputs=[self.obs_wp, int(self._hc_err_c0), 3, self._pcderr_cache_wp],
                )
            wp.launch(
                _obs_cache_zero_cols_kernel, dim=(self.NWORLD, 6),
                inputs=[self.obs_wp, int(self._objpd_c0), 6, self._objpd_cache_wp],
            )
        approach_only = getattr(self, "_track_approach_only", False)
        if getattr(self, "_stage_from_obs", False):
            _dist_src = (self._frozen_hc_dist_wp if tracking_free else self.best_dist_hc_wp)
            _use_proxy = tracking_free or approach_only
            wp.launch(
                _student_stage_kernel, dim=self.NWORLD,
                inputs=[
                    _dist_src, self.d.xpos, self.cond.get("obj_p_init"),
                    int(self.obj_body_id),
                    float(self._stage_hc_dist), int(self._stage_near_ticks),
                    int(1) if getattr(self, "_stage_touch_gate", False) else int(0),
                    int(self._tq_c0), int(self._tq_c1 - self._tq_c0),
                    float(getattr(self, "_stage_touch_thresh", 0.0)),
                    float(self._stage_lift_m),
                    int(self.wrist_body_id),
                    int(1) if _use_proxy else int(0),
                    self._stu_wristz_anchor_wp,
                    self._stu_stage_wp, self._stu_near_cnt_wp,
                    int(self._stage_col), self.obs_wp,
                ],
            )
        if approach_only:
            if getattr(self, "_track_rigid_attach", False):
                wp.launch(
                    _phase_gate_rigid_attach_kernel, dim=self.NWORLD,
                    inputs=[_pg_gate, self.d.xpos, self.d.xmat,
                            self.d.site_xpos, self._finger_site_ids_wp,
                            self.cond.get("obj_p_init"), self.cond.get("obj_R_init"),
                            int(self.wrist_body_id), int(self.obj_body_id),
                            int(self._n_sensors),
                            int(self.hand_util.fore_arm_sensor_idx),
                            int(self._pg_pcd_c0), int(self._pg_objpd_c0),
                            self.obs_wp, self._pg_pcd_cache_wp,
                            self._ra_rel_p_wp, self._ra_rel_R_wp,
                            self._ra_wg_p_wp, self._ra_wg_R_wp],
                )
            else:
                wp.launch(
                    _phase_gate_freeze_kernel, dim=(self.NWORLD, int(self._pg_pcd_w)),
                    inputs=[_pg_gate, int(self._pg_pcd_c0),
                            int(self._pg_pcd_w), self.obs_wp, self._pg_pcd_cache_wp],
                )
                wp.launch(
                    _phase_gate_freeze_kernel, dim=(self.NWORLD, 6),
                    inputs=[_pg_gate, int(self._pg_objpd_c0),
                            6, self.obs_wp, self._pg_objpd_cache_wp],
                )
        if getattr(self, "_drop_ep_step", False):
            wp.launch(
                _obs_cache_zero_cols_kernel, dim=(self.NWORLD, 1),
                inputs=[self.obs_wp, int(self._ep_col), 1, self._ep_cache_wp],
            )
        if self._drop_contact:
            wp.launch(
                _obs_cache_zero_cols_kernel, dim=(self.NWORLD, int(self._contact_w)),
                inputs=[self.obs_wp, int(self._contact_c0), int(self._contact_w),
                        self._contact_cache_wp],
            )
        # frozen initial partial view (capture flagged worlds) → visibility mask
        if self._pv_freeze:
            self.pv.capture_masked(self.transformed_pcd_wp, self._pv_capture_mask_wp)
            self._pv_capture_mask_torch.zero_()
            self.pv.refresh_from_frozen()
            pv_points = self.pv.frozen_pcd
            if getattr(self, "_rattach_bps", False):
                wp.launch(
                    _rattach_bps_pcd_kernel,
                    dim=(self.NWORLD, int(self._pcd_n_pts)),
                    inputs=[_pg_gate, self.d.xpos, self.d.xmat,
                            int(self.wrist_body_id), self.pv.frozen_pcd,
                            self._ra_wg_p_wp, self._ra_wg_R_wp,
                            self._ra_pcd_scratch_wp],
                )
                pv_points = self._ra_pcd_scratch_wp
        else:
            self.pv.update_from_world_pcd(self.transformed_pcd_wp)   # live visibility
            pv_points = self.transformed_pcd_wp
        # visible-masked BPS tail
        wp.launch(
            _obs_partial_bps_feature_kernel,
            dim=(self.NWORLD, int(self._n_bps)),
            inputs=[
                self.d.xpos, self.d.xmat, pv_points, self.pv.visible, self.bps_local_wp,
                int(self.wrist_body_id), self._hand_center_offset_vec, self._hand_center_R_mat,
                int(self._pcd_n_pts), float(self.BPS_GAMMA), float(self._bps_mask_below_z),
                int(self._bps_base_off()), self.obs_wp,
            ],
        )
        # history block (after the obs is complete — frame 0 = current values)
        if int(getattr(self, "_hist_k", 0)) > 0:
            self._hist_tick = int(getattr(self, "_hist_tick", 0)) + 1
            if getattr(self, "_hist_lags", ()):
                wp.launch(
                    _student_hist_gate_kernel, dim=self.NWORLD,
                    inputs=[float(getattr(self, "_hist_skip_prob", 0.0)),
                            int(self._hist_tick),
                            self._hist_upd_wp, self._hist_skip_wp],
                )
                wp.launch(
                    _student_hist_ms_kernel, dim=(self.NWORLD, int(self._hist_src_w)),
                    inputs=[self._hist_src_cols_wp, int(self._hist_obs_src_w),
                            self._hist_act_wp, int(self._hist_act_w),
                            int(self._hist_src_w),
                            self._hist_lags_wp, int(self._hist_k),
                            int(self._hist_ring_L),
                            self._hist_upd_wp, self._hist_skip_wp,
                            int(self._hist_c0),
                            float(getattr(self, "_hist_noise_std", 0.0)),
                            int(self._hist_tick),
                            self._hist_ring_wp, self.obs_wp],
                )
            else:
                wp.launch(
                    _student_hist_kernel, dim=(self.NWORLD, int(self._hist_src_w)),
                    inputs=[self._hist_src_cols_wp, int(self._hist_obs_src_w),
                            self._hist_act_wp, int(self._hist_act_w),
                            int(self._hist_src_w),
                            int(self._hist_c0), int(self._hist_k),
                            float(getattr(self, "_hist_skip_prob", 0.0)),
                            float(getattr(self, "_hist_noise_std", 0.0)),
                            int(self._hist_tick), self.obs_wp],
                )

    def collect_teacher_obs(self) -> torch.Tensor:
        """Privileged teacher obs for the CURRENT state (distillation only):
        unmasked core blocks + full live BPS tail, with the student-only history
        block removed so the layout is the teacher's ``[core | bps]``."""
        assert getattr(self, "_teacher_obs_wp", None) is not None, (
            "collect_teacher_obs needs a prior _collect_obs_kernel (step/reset)")
        self._launch_bps_tail(self.transformed_pcd_wp, self._teacher_obs_wp)
        hist_w = int(getattr(self, "_hist_w_total", 0))
        if hist_w == 0:
            return self._teacher_obs_torch
        b0 = int(self._hist_c0)
        if getattr(self, "_teacher_obs_slim_torch", None) is None:
            self._teacher_obs_slim_torch = torch.zeros(
                (self.NWORLD, int(self.obs_dim) - hist_w),
                dtype=self._teacher_obs_torch.dtype, device=self._teacher_obs_torch.device)
        self._teacher_obs_slim_torch[:, :b0].copy_(self._teacher_obs_torch[:, :b0])
        self._teacher_obs_slim_torch[:, b0:].copy_(self._teacher_obs_torch[:, b0 + hist_w:])
        return self._teacher_obs_slim_torch

    def collect_critic_obs(self) -> torch.Tensor:
        """Student hybrid-RL critic obs = ``[teacher obs | wrist/obj velocity (12)]``."""
        t = self.collect_teacher_obs()
        n = int(t.shape[1])
        if getattr(self, "_critic_obs_asym_wp", None) is None:
            self._critic_obs_asym_wp = wp.zeros((self.NWORLD, n + 12), dtype=float, device=self.device)
            self._critic_obs_asym_torch = wp.to_torch(self._critic_obs_asym_wp)
        self._critic_obs_asym_torch[:, :n].copy_(t)
        wp.launch(
            _obs_wrist_obj_vel_kernel, dim=self.NWORLD,
            inputs=[self.d.cvel, self.d.xmat,
                    int(self.wrist_body_id), int(self.obj_body_id),
                    n, self._critic_obs_asym_wp],
        )
        return self._critic_obs_asym_torch

    # ══════════════════════════════════════════════════════════════════════
    # Blind-after-grasp + re-grasp label / prediction
    # ══════════════════════════════════════════════════════════════════════
    def _setup_blind_buffers(self) -> None:
        cfg_tr = self.sampled_env.overall_cfg.Training
        self._blind_after_grasp = bool(getattr(cfg_tr, "student_blind_after_grasp", self.STUDENT_BLIND_AFTER_GRASP))
        self._blind_fd_vel_obj = bool(getattr(cfg_tr, "student_blind_fd_vel_obj", self.STUDENT_BLIND_FD_VEL_OBJ))
        self._drop_obj_ground = bool(getattr(cfg_tr, "student_drop_obj_ground_force", self.STUDENT_DROP_OBJ_GROUND_FORCE))
        terms = {nm: (s, e) for nm, s, e in self.obs_term_layout}
        cols: list = [c for nm in BLIND_TERMS_DEFAULT if nm in terms for c in range(*terms[nm])]
        self._blind_cols = torch.as_tensor(sorted(set(cols)), dtype=torch.long, device=self.torch_device)
        acols: list = [c for nm in ALWAYS_BLIND_TERMS if nm in terms for c in range(*terms[nm])]
        if self._blind_fd_vel_obj and "fd_vel" in terms:
            a0, _ = terms["fd_vel"]
            acols += list(range(a0 + 6, a0 + 12))         # obj lin/ang velocity
        self._always_blind_cols = torch.as_tensor(sorted(set(acols)), dtype=torch.long, device=self.torch_device)
        self._blind_near_m = float(getattr(cfg_tr, "student_blind_near_m", self.STUDENT_BLIND_NEAR_M))
        self._blind_latch = torch.zeros(int(self.NWORLD), dtype=torch.bool, device=self.torch_device)
        self._blind_mode = str(getattr(cfg_tr, "student_blind_mode", self.STUDENT_BLIND_MODE)).lower()
        assert self._blind_mode in ("zero", "attach"), self._blind_mode
        if self._blind_mode == "attach":
            self._pg_gate_wp = wp.zeros(int(self.NWORLD), dtype=int, device=self.device)
            self._pg_gate_torch = wp.to_torch(self._pg_gate_wp)
            _skip = {"site_pcd_err", "hand_center_pcd_err", "obj_pose_diff"}
            if bool(getattr(cfg_tr, "student_rattach_bps", False)):
                _skip.add("bps_feature")
            cols = [c for nm in BLIND_TERMS_DEFAULT if nm in terms and nm not in _skip
                    for c in range(*terms[nm])]
            self._blind_cols = torch.as_tensor(sorted(set(cols)), dtype=torch.long, device=self.torch_device)
            acols = [c for nm in ALWAYS_BLIND_TERMS if nm in terms and nm not in _skip
                     for c in range(*terms[nm])]
            if self._blind_fd_vel_obj and "fd_vel" in terms:
                a0, _ = terms["fd_vel"]
                acols += list(range(a0 + 6, a0 + 12))
            self._always_blind_cols = torch.as_tensor(sorted(set(acols)), dtype=torch.long, device=self.torch_device)
        self._ogf_col = int(terms["obj_ground_force"][0]) if "obj_ground_force" in terms else None
        N = int(self.NWORLD)
        self.regrasp_label_torch = torch.zeros(N, device=self.torch_device)
        self.regrasp_pred_torch  = torch.zeros(N, device=self.torch_device)
        self._regrasp_pred_streak = torch.zeros(N, dtype=torch.long, device=self.torch_device)
        self.regrasp_pred_fired_torch = torch.zeros(N, device=self.torch_device)
        self._xpos_torch = wp.to_torch(self.d.xpos)
        self._objp0_torch = wp.to_torch(self.cond.get("obj_p_init"))
        self._regrasp_src = str(self.REGRASP_STAGE_RESET_SOURCE)
        self.regrasp_bonus_torch = torch.zeros(N, device=self.torch_device)
        self.metrics.update({"regrasp_label": self.regrasp_label_torch,
                             "regrasp_pred_fired": self.regrasp_pred_fired_torch,
                             "regrasp_bonus": self.regrasp_bonus_torch})
        assert getattr(self, "_stage_from_obs", False), \
            "BLIND needs student_stage_from_obs=true (stage-based masking / rewind)"
        _gate = (f"proximity latch <{self._blind_near_m*100:.1f}cm[{self._blind_mode}]"
                 if self._blind_near_m > 0.0 else "stage>=1(legacy)")
        print(f"[student-obs][blind] gate={_gate} blind={self._blind_after_grasp} "
              f"({int(self._blind_cols.numel())} cols: {', '.join(BLIND_TERMS_DEFAULT)})  "
              f"always masked {int(self._always_blind_cols.numel())} cols: {', '.join(ALWAYS_BLIND_TERMS)}"
              f"{' + fd_vel[obj]' if self._blind_fd_vel_obj else ''}  "
              f"obj_ground_force masked={self._drop_obj_ground}  "
              f"regrasp: source={self._regrasp_src} consec={int(self.REGRASP_PRED_CONSEC)}")

    def set_regrasp_source(self, src: str) -> None:
        assert src in ("gt", "pred", "both"), src
        self._regrasp_src = src
        self._regrasp_pred_streak.zero_()

    def _blind_reward_extras(self) -> None:
        """Optional shaping for the re-grasp student (default weights 0 = off)."""
        wb, wh = float(self.W_REGRASP_BONUS), float(self.W_HOLD_CONTACT)
        if wb <= 0.0 and wh <= 0.0:
            return
        contact = self.hand_obj_active_torch > 0
        lift_z = self._xpos_torch[:, int(self.obj_body_id), 2] - self._objp0_torch[:, 2]
        held = contact & (lift_z >= float(self.LIFT_SUCCESS_THRESH))
        extra = torch.zeros_like(self.reward_torch)
        if wb > 0.0:
            extra += wb * ((self.retry_count_torch > 0.5) & held).float()
        if wh > 0.0:
            extra += wh * ((self._stu_stage_torch >= 1) & contact).float()
        self.reward_torch += extra
        self.regrasp_bonus_torch.copy_(extra)

    def _blind_post_obs(self) -> None:
        """Re-grasp label ("lost the object") + blanking of the object obs."""
        stage = self._stu_stage_torch
        contact = self.hand_obj_active_torch > 0
        lift_z = self._xpos_torch[:, int(self.obj_body_id), 2] - self._objp0_torch[:, 2]
        lab = ((stage >= 1) & ~contact) | ((stage >= 2) & (lift_z < float(self.LIFT_SUCCESS_THRESH)))
        self.regrasp_label_torch.copy_(lab.float())
        if self._blind_after_grasp and self._blind_mode == "zero":
            if self._blind_near_m > 0.0:
                self._blind_latch |= (self.best_dist_hc_torch < self._blind_near_m)
                keep = (~self._blind_latch).float().unsqueeze(1)
            else:
                keep = (stage < 1).float().unsqueeze(1)
            self.obs_torch[:, self._blind_cols] *= keep
        elif self._blind_after_grasp and int(self._blind_cols.numel()) > 0:
            self.obs_torch[:, self._blind_cols] *= (~self._blind_latch).float().unsqueeze(1)
        if int(self._always_blind_cols.numel()) > 0:
            self.obs_torch[:, self._always_blind_cols] = 0.0
        if self._drop_obj_ground and self._ogf_col is not None:
            self.obs_torch[:, self._ogf_col] = 0.0


@register_rl_env("grasping_student")
class GraspingStudentEnv(BaseRLEnv):
    """Grasp-and-lift with the deployable student observation (``env_name: grasping_student``)."""
    HANDLER_CLS = GraspingStudent
