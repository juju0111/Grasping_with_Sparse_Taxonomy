"""Runtime per-world modification of finger position-servo gains (kp/kv) for DR.

In mujoco_warp, ``Model.actuator_gainprm`` / ``actuator_biasprm`` are batched
fields of shape ``(*, nu, vec10)`` (same convention ``heterogeneous_env_setup``
uses for geom/body fields: expanding the leading dim from 1 to NWORLD makes the
values per-world). This module applies that expansion to the actuator fields and
provides a sync-free GPU write path that may be called **every tick**:

    gains = PerWorldFingerGains(m, mjm, nworld)      # <- create **before** graph capture
    gains.set(kp=45.0)                               # same value for all worlds (replaces env-var)
    gains.set(kp=rng.uniform(25, 80, nworld))        # per-world DR (on every reset)
    gains.set_from_wp(kp_wp, kv_wp)                  # pure GPU path (per-tick DR)

Design constraints (important):
  * **Must be created before CUDA graph capture.** Construction replaces
    (reallocates) the gainprm/biasprm arrays as (NWORLD, nu); since capture bakes
    array pointers, replacing them after capture leaves the graph reading the old
    arrays. All subsequent ``set*()`` calls are **in-place kernel writes** to the
    same arrays and therefore graph-safe.
  * Only the position-servo convention is handled: ``gainprm[0]=kp, biasprm=[0,-kp,-kv]``
    (joint transmission + affine bias). Other actuators are excluded automatically.
  * By default kv keeps critical damping: ``kv = kv0*sqrt(kp/kp0)``.
    ``kv="fixed"`` keeps the original kv; raising only kp by a large factor causes overshoot.
  * Note for grasping tasks: contact force = kp x penetration, so kp DR is also
    contact-force DR.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src.types import vec10f   # same dtype as the mjwarp model fields


@wp.kernel
def _write_pos_servo_gains_kernel(
    kp:       wp.array(dtype=float, ndim=2),   # type: ignore (NWORLD, n_act)
    kv:       wp.array(dtype=float, ndim=2),   # type: ignore (NWORLD, n_act)
    act_ids:  wp.array(dtype=int,   ndim=1),   # type: ignore (n_act,) scene actuator idx
    n_act:    int,
    gainprm:  wp.array(dtype=vec10f, ndim=2),  # type: ignore (NWORLD, nu) IN/OUT
    biasprm:  wp.array(dtype=vec10f, ndim=2),  # type: ignore (NWORLD, nu) IN/OUT
):
    """position servo: gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kv.
    Touches only the selected actuator columns (other actuators/params preserved)."""
    w = wp.tid()
    for i in range(n_act):
        a = act_ids[i]
        g = gainprm[w, a]
        b = biasprm[w, a]
        g[0] = kp[w, i]
        b[1] = -kp[w, i]
        b[2] = -kv[w, i]
        gainprm[w, a] = g
        biasprm[w, a] = b


class PerWorldFingerGains:
    """Handle for per-world modification of finger position-servo kp/kv in one mjwarp model.

    Args:
        m:        mjwarp Model (result of ``put_model``); its gainprm/biasprm are
                  expanded/replaced as (NWORLD, nu) once.
        mjm:      original mujoco.MjModel (used to scan base kp0/kv0).
        nworld:   number of worlds.
        act_idxs: target actuator indices (None -> auto-detect all position servos;
                  leader-follower setups can pass a per-hand scope for independent DR).
        device:   warp device (None -> device of m.actuator_gainprm).
    """

    def __init__(self, m, mjm, nworld: int,
                 act_idxs: Optional[Sequence[int]] = None,
                 device=None, verbose: bool = False):
        self.m = m
        self.nworld = int(nworld)
        self.device = device or m.actuator_gainprm.device
        nu = int(mjm.nu)

        # ── position-servo actuator detection (same as the env-var guard in orchestrator/base) ─
        cand = range(nu) if act_idxs is None else [int(a) for a in act_idxs]
        sel, kp0, kv0 = [], [], []
        for a in cand:
            if int(mjm.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            kp = float(mjm.actuator_gainprm[a, 0])
            if kp <= 0.0 or abs(float(mjm.actuator_biasprm[a, 1]) + kp) > 1e-6:
                continue                      # not a position servo → skip
            sel.append(a)
            kp0.append(kp)
            kv0.append(-float(mjm.actuator_biasprm[a, 2]))   # biasprm[2] = -kv
        assert sel, "no position-servo finger actuator found"
        self.act_idxs = np.asarray(sel, dtype=int)
        self.n_act = len(sel)
        self.kp0 = np.asarray(kp0, dtype=np.float64)          # (n_act,)
        self.kv0 = np.asarray(kv0, dtype=np.float64)

        # ── expand gainprm/biasprm to (NWORLD, nu) (batch dim 1 -> NWORLD) ──
        # Must run before graph capture; subsequent set*() calls are in-place writes.
        for name in ("actuator_gainprm", "actuator_biasprm"):
            arr = getattr(m, name)
            host = arr.numpy()                                # (1|NWORLD, nu, 10)
            if host.shape[0] != self.nworld:
                host = np.broadcast_to(host[:1], (self.nworld,) + host.shape[1:]).copy()
                setattr(m, name, wp.array(host, dtype=vec10f, device=self.device))

        self._act_ids_wp = wp.array(self.act_idxs.astype(np.int32), dtype=int,
                                    device=self.device)
        self._kp_wp = wp.zeros((self.nworld, self.n_act), dtype=float, device=self.device)
        self._kv_wp = wp.zeros((self.nworld, self.n_act), dtype=float, device=self.device)
        if verbose:
            print(f"[per-world-gains] {self.n_act} pos-servo actuator(s), "
                  f"kp0∈[{self.kp0.min():.1f},{self.kp0.max():.1f}] × {self.nworld} worlds")

    # ── host convenience path (reset-time DR / constant setting) ────────────
    def set(self, kp: Union[float, np.ndarray],
            kv: Union[None, str, float, np.ndarray] = None) -> None:
        """kp: scalar | (NWORLD,) | (NWORLD, n_act). kv: None -> keep critical
        damping (kv0*sqrt(kp/kp0)) / "fixed" -> original kv / scalar or array -> explicit."""
        kp_arr = np.broadcast_to(
            np.asarray(kp, dtype=np.float64).reshape(-1, 1) if np.ndim(kp) == 1
            else np.asarray(kp, dtype=np.float64),
            (self.nworld, self.n_act)).copy()
        if kv is None:                                        # critical-damping scale
            kv_arr = self.kv0[None, :] * np.sqrt(kp_arr / self.kp0[None, :])
        elif isinstance(kv, str) and kv == "fixed":
            kv_arr = np.broadcast_to(self.kv0[None, :], kp_arr.shape).copy()
        else:
            kv_arr = np.broadcast_to(
                np.asarray(kv, dtype=np.float64).reshape(-1, 1) if np.ndim(kv) == 1
                else np.asarray(kv, dtype=np.float64),
                (self.nworld, self.n_act)).copy()
        wp.copy(self._kp_wp, wp.array(kp_arr.astype(np.float32), dtype=float,
                                      device=self.device))
        wp.copy(self._kv_wp, wp.array(kv_arr.astype(np.float32), dtype=float,
                                      device=self.device))
        self._launch()

    def randomize(self, rng: np.random.Generator, kp_low: float, kp_high: float,
                  per_actuator: bool = False, log_uniform: bool = True) -> np.ndarray:
        """Sample and apply per-world kp DR. Log-uniform by default (gains are multiplicative).
        Returns the sampled kp, (NWORLD,) or (NWORLD, n_act), for logging/reproducibility."""
        shape = (self.nworld, self.n_act) if per_actuator else (self.nworld,)
        if log_uniform:
            kp = np.exp(rng.uniform(np.log(kp_low), np.log(kp_high), shape))
        else:
            kp = rng.uniform(kp_low, kp_high, shape)
        self.set(kp)
        return kp

    # ── pure GPU path (per-tick DR; no host sync) ───────────────────────────
    def set_from_wp(self, kp_wp, kv_wp) -> None:
        """Use (NWORLD, n_act) warp arrays directly; only a kernel launch (graph-safe)."""
        wp.copy(self._kp_wp, kp_wp)
        wp.copy(self._kv_wp, kv_wp)
        self._launch()

    def _launch(self) -> None:
        wp.launch(_write_pos_servo_gains_kernel, dim=self.nworld,
                  inputs=[self._kp_wp, self._kv_wp, self._act_ids_wp, self.n_act,
                          self.m.actuator_gainprm, self.m.actuator_biasprm])

    # ── verification / debugging ────────────────────────────────────────────
    def readback(self) -> dict:
        """Current GPU-side (kp, kv) as (NWORLD, n_act). For verification/notebooks (host sync)."""
        g = self.m.actuator_gainprm.numpy()[:, self.act_idxs, 0]
        b = self.m.actuator_biasprm.numpy()[:, self.act_idxs, :]
        return dict(kp=g, kp_bias=-b[..., 1], kv=-b[..., 2])
