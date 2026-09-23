"""grit/training/rl_envs/grasping_teacher.py — privileged teacher (``env_name: grasping_teacher``).

:class:`GraspingTeacher` = :class:`GraspingCore` + an asymmetric critic observation
(actor obs + 20 privileged floats about the object) + Lagrangian dual ascent on the
constrained penalty weights. This is the policy that ``scripts/train.py`` trains
and that ``scripts/distill.py`` distils into the student.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import torch  # type: ignore
import warp as wp  # type: ignore

from grit.training.rl_env_base import BaseRLEnv, register_rl_env
from grit.training.rl_envs.grasping_core import GraspingCore
from grit.training.rl_envs.grasping_kernels import _obs_privileged_gt_kernel


class GraspingTeacher(GraspingCore):
    """Core task + privileged critic obs + Lagrangian penalty constraints."""

    CONFIG_KEYS_GROUPS = {
        **GraspingCore.CONFIG_KEYS_GROUPS,
        "obs_priv": ("OBS_CURR_PROGRESS",),
        "reward_lagrangian": (
            "LAG_ETA", "LAG_ETA_DOWN", "LAG_UPDATE_EVERY",
            "LAG_WARMUP_STEPS", "LAG_VIOL_CLIP", "LAG_CONSTRAINTS",
            "LAG_INIT_LAMBDA",
        ),
    }

    # ── Privileged critic ─────────────────────────────────────────────────
    # Width of the privileged GT block written by ``_obs_privileged_gt_kernel``.
    _N_PRIV: int = 20
    # Curriculum-pressure scalar (curriculum_pressure ∈ [0,1]) appended to the
    # critic obs so the value function can observe the current penalty regime.
    # obs-dim knob applied after buffer allocation → the setter re-allocates.
    _obs_curr_progress: bool = False

    @property
    def OBS_CURR_PROGRESS(self) -> bool:
        return self._obs_curr_progress

    @OBS_CURR_PROGRESS.setter
    def OBS_CURR_PROGRESS(self, v) -> None:
        v = bool(v)
        if v == self._obs_curr_progress:
            return
        self._obs_curr_progress = v
        if hasattr(self, "_critic_obs_wp"):
            self._setup_obs_buffers()   # critic_obs_dim changed (init-time only)

    # ── Lagrangian penalty constraints ─────────────────────
    LAG_ETA:          float = 0.05
    LAG_ETA_DOWN:     float = 0.01
    LAG_UPDATE_EVERY: int   = 10
    LAG_WARMUP_STEPS: float = 5e6
    LAG_VIOL_CLIP:    float = 5.0
    LAG_INIT_LAMBDA:  Dict[str, float] = {}          # λ warm start {knob: λ0}
    # knob: [metric, per-step target, λ_max (total weight cap)[, warmup_steps]]
    LAG_CONSTRAINTS: Dict[str, list] = {
        "W_OBJ_TOUCH":     ["obj_touch_force",     0.08,  8.0],
        "W_CONTACT_NEG":   ["contact_neg_mag",     0.10,  1.5],
        "W_TABLE_CONTACT": ["raw_table_contact_w", 0.05,  8.0],
        "W_TABLE_IMPULSE": ["raw_table_impulse_w", 0.02, 10.0],
        "W_SELF_COLL":     ["raw_self_contact_w",  0.02,  5.0],
        "W_SELF_IMPULSE":  ["raw_self_impulse_w",  0.005, 5.0],
    }
    _LAG_EMA_BETA: float = 0.99

    # ── critic obs ────────────────────────────────────────────────────────
    @property
    def critic_obs_dim(self) -> int:
        return (int(self.obs_dim) + int(self._N_PRIV)
                + (1 if self.OBS_CURR_PROGRESS else 0))

    @property
    def critic_obs_torch(self) -> torch.Tensor:
        return self._critic_obs_torch

    def critic_obs_norm_passthrough_mask(self):
        """[actor mask | privileged GT (standardised) | pressure scalar (raw)]."""
        parts = [self.obs_norm_passthrough_mask(), np.zeros(int(self._N_PRIV), dtype=bool)]
        if self.OBS_CURR_PROGRESS:
            parts.append(np.ones(1, dtype=bool))
        return np.concatenate(parts)

    def _setup_obs_buffers(self) -> None:
        super()._setup_obs_buffers()
        self._critic_obs_wp    = wp.zeros(
            (self.NWORLD, self.critic_obs_dim), dtype=float, device=self.device)
        self._critic_obs_torch = wp.to_torch(self._critic_obs_wp)

    def _collect_obs_kernel(self) -> None:
        super()._collect_obs_kernel()      # core blocks + BPS tail + single-attempt clock
        self._collect_critic_obs()

    def _collect_critic_obs(self) -> None:
        """critic obs = [actor obs copy | privileged GT (20) | curriculum pressure (opt.)]."""
        actor_dim = int(self.obs_dim)
        self._critic_obs_torch[:, :actor_dim].copy_(self.obs_torch)
        wp.launch(
            _obs_privileged_gt_kernel, dim=self.NWORLD,
            inputs=[
                self.d.xpos, self.d.xmat, self.d.cvel,
                self.cond.get("obj_p_init"), self.cond.get("target_pnt"),
                int(self.obj_body_id), int(self.wrist_body_id),
                self._hand_center_offset_vec, float(self.LIFT_TARGET_M),
                int(actor_dim), self._critic_obs_wp,
            ],
        )
        if self.OBS_CURR_PROGRESS:
            self._critic_obs_torch[:, actor_dim + int(self._N_PRIV)] = \
                float(self.curriculum_pressure())
        torch.nan_to_num_(self._critic_obs_torch, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Lagrangian dual ascent on the penalty weights ─────────────────────
    def on_train_progress(self, cur_step: int, total_steps: int) -> None:
        super().on_train_progress(cur_step, total_steps)
        self._lag_update(int(cur_step))

    def curriculum_pressure(self) -> float:
        """mean(λ / λ_max) ∈ [0, 1] — how hard the constraints currently push."""
        cons = getattr(self, "_lag_cons", None)
        if not cons:
            return 0.0
        fr = [self._lag_lambda[k] / max(1e-8, lmax - self._lag_base[k])
              for k, (_m, _t, lmax, _wu) in cons.items()]
        return float(min(1.0, max(0.0, sum(fr) / max(1, len(fr)))))

    def _lag_metric_mean(self, name: str) -> float:
        if name == "contact_neg_mag":
            return float((-self.raw_contact_neg_torch).clamp(min=0.0).mean().item())
        t = getattr(self, f"{name}_torch", None)
        if t is None:
            raise ValueError(f"LAG_CONSTRAINTS metric {name!r}: handler has no {name}_torch buffer")
        return float(t.mean().item())

    def _lag_init_if_needed(self) -> None:
        if getattr(self, "_lag_lambda", None) is not None:
            return
        cons: Dict[str, Tuple[str, float, float, float]] = {}
        for knob, spec in dict(self.LAG_CONSTRAINTS).items():
            seq = list(spec)
            assert len(seq) in (3, 4), (
                f"LAG_CONSTRAINTS.{knob}: expected [metric, target, lambda_max[, warmup_steps]] — {spec!r}")
            assert hasattr(self, knob), f"LAG_CONSTRAINTS: unknown knob {knob!r}"
            _wu = float(seq[3]) if len(seq) == 4 else float(self.LAG_WARMUP_STEPS)
            cons[knob] = (str(seq[0]), float(seq[1]), float(seq[2]), _wu)
            self._lag_metric_mean(str(seq[0]))   # fail fast if the metric buffer is missing
        self._lag_cons   = cons
        self._lag_base   = {k: float(getattr(self, k)) for k in cons}
        self._lag_lambda = {k: 0.0 for k in cons}
        self._lag_ema    = {k: None for k in cons}
        self._lag_iter   = 0
        init = dict(getattr(self, "LAG_INIT_LAMBDA", {}) or {})
        for k, v in init.items():
            if k not in cons:
                raise ValueError(f"LAG_INIT_LAMBDA: knob {k!r} has no constraint")
            self._lag_lambda[k] = min(max(float(v), 0.0), max(0.0, cons[k][2] - self._lag_base[k]))
            setattr(self, k, self._lag_base[k] + self._lag_lambda[k])
        tag = getattr(getattr(self, "hand_util", None), "hand_name", "?")
        for k, (m, tgt, lmax, wu) in cons.items():
            w0 = (f", λ0={self._lag_lambda[k]:g} → W0={self._lag_base[k] + self._lag_lambda[k]:g}"
                  if self._lag_lambda[k] else "")
            _wus = f", warmup={wu:.0e}" if wu != float(self.LAG_WARMUP_STEPS) else ""
            print(f"[lagrangian] {tag}: {k} ← λ(base={self._lag_base[k]:g}, "
                  f"metric={m}, target={tgt:g}/step, W_max={lmax:g}{_wus}{w0})")

    def _lag_update(self, cur_step: int) -> None:
        self._lag_init_if_needed()
        self._lag_iter += 1
        for k, (m, _tgt, _lmax, _wu) in self._lag_cons.items():
            v = self._lag_metric_mean(m)
            e = self._lag_ema[k]
            self._lag_ema[k] = v if e is None else \
                self._LAG_EMA_BETA * e + (1.0 - self._LAG_EMA_BETA) * v
        if self._lag_iter % max(1, int(self.LAG_UPDATE_EVERY)) != 0:
            return
        changed: List[str] = []
        for k, (m, tgt, lmax, wu) in self._lag_cons.items():
            if cur_step < int(wu):
                continue
            viol = (float(self._lag_ema[k]) - tgt) / max(tgt, 1e-8)
            viol = min(max(viol, -1.0), float(self.LAG_VIOL_CLIP))
            eta  = float(self.LAG_ETA) if viol > 0.0 else float(self.LAG_ETA_DOWN)
            lam_cap = max(0.0, lmax - self._lag_base[k])
            lam = min(max(self._lag_lambda[k] + eta * viol, 0.0), lam_cap)
            if lam != self._lag_lambda[k]:
                self._lag_lambda[k] = lam
                setattr(self, k, self._lag_base[k] + lam)
                changed.append(f"{k}={self._lag_base[k] + lam:.3g}(λ{lam:+.3g}, "
                               f"ema={float(self._lag_ema[k]):.3g}/{tgt:g})")
        if changed:
            print(f"[lagrangian] {'  '.join(changed)}")

    def curriculum_state(self) -> Dict[str, Any]:
        st = dict(super().curriculum_state())
        self._lag_init_if_needed()
        st["lag_lambda"] = {k: float(v) for k, v in self._lag_lambda.items()}
        st["lag_ema"]    = {k: (None if v is None else float(v)) for k, v in self._lag_ema.items()}
        return st

    def load_curriculum_state(self, state: Optional[Dict[str, Any]]) -> None:
        super().load_curriculum_state(state)
        if not state:
            return
        self._lag_init_if_needed()
        lam = (state or {}).get("lag_lambda") or {}
        ema = (state or {}).get("lag_ema") or {}
        restored = []
        for k, (_m, _t, lmax, _wu) in self._lag_cons.items():
            if k in lam:
                v = min(max(float(lam[k]), 0.0), max(0.0, lmax - self._lag_base[k]))
                self._lag_lambda[k] = v
                setattr(self, k, self._lag_base[k] + v)
                restored.append(f"{k}={self._lag_base[k] + v:g}")
            if k in ema and ema[k] is not None:
                self._lag_ema[k] = float(ema[k])
        if restored:
            print(f"[lagrangian] resume — λ restored: {'  '.join(restored)}")


@register_rl_env("grasping_teacher")
class GraspingTeacherEnv(BaseRLEnv):
    """Grasp-and-lift a tabletop object — privileged teacher (``env_name: grasping_teacher``).

    One :class:`GraspingTeacher` handler per hand in ``using_hand_name_list``.
    Pair with ``training.async_ppo: true`` and a critic-aware policy
    (``policy.name: bps_encoder_mlp``; the builder injects ``critic_obs_dim``).
    """
    HANDLER_CLS = GraspingTeacher
