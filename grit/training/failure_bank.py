"""failure_bank.py — **per-world state bank** for focused exploration of tail failures (opt-in helper).

Snapshots / restores the per-world GPU buffers the handler already owns, from the
outside, without touching env/handler code (the same principle by which the base
``per_world_reset_if_done`` snapshots and restores qpos/qvel/ctrl/mocap, generalised to
**every per-world buffer**).

Operation
---------
* ``tick()``            : every step, record the state s_t **before** env.step in a
                          τ+1 ring (small buffers only).
* ``push_failures(m)``  : **after** env.step, for worlds with a failure event (failure
                          terminal / re-grasp retry) store s_{t-τ} (from the ring) + the
                          large buffers (current values; constant within an episode, e.g.
                          frozen PCD) in the bank (ring, cap).
* ``restore(done)``     : **after** ``per_world_reset_if_done``, restore a bank entry into
                          each just-reset world with probability p → forward / setpoint
                          sync / FD velocity invalidation / obs re-collection. That world
                          continues its episode from "τ steps before the failure" (clocks
                          such as ep_step are restored too).

Collection rule (generic): every handler attribute ``*_wp`` that is a ``wp.array`` with
shape[0]==NWORLD + ``cond._fields[*]['gpu']`` + the per-world arrays of ``pv`` (partial
view) + ``d.{qpos,qvel,ctrl,mocap_pos,mocap_quat}``. Static buffers mixed in are harmless
(the same values are written back). Arrays with >= LARGE_COLS columns are classified as
'large': they are not put in the ring; their value at failure time is stored instead
(assumed constant within an episode, e.g. frozen partial PCD).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch  # type: ignore
import warp as wp  # type: ignore

LARGE_COLS = 1024
SKIP_SUFFIX = ("reward", "done", "_snap_")     # exclude transient / base snapshot buffers


def _per_world_arrays(h) -> Dict[str, torch.Tensor]:
    N = int(h.NWORLD)
    out: Dict[str, torch.Tensor] = {}

    def add(name, arr):
        if not isinstance(arr, wp.array) or arr.ndim < 1 or int(arr.shape[0]) != N:
            return
        if any(s in name for s in SKIP_SUFFIX):
            return
        try:
            out[name] = wp.to_torch(arr)
        except Exception:
            pass

    d = h.d
    for nm in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "act"):
        if hasattr(d, nm):
            add("d." + nm, getattr(d, nm))
    for nm, v in vars(h).items():
        if nm.endswith("_wp"):
            add("h." + nm, v)
    for nm, f in getattr(h.cond, "_fields", {}).items():
        add("cond." + nm, f["gpu"])
    pv = getattr(h, "pv", None)
    if pv is not None:
        for nm, v in vars(pv).items():
            add("pv." + nm, v)
    return out


class FailureStateBank:
    def __init__(self, h, tau: int = 15, cap: int = 4096, p_restore: float = 0.3,
                 min_bank: int = 256, device=None, verbose: bool = True):
        self.h = h; self.N = int(h.NWORLD); self.tau = int(tau)
        self.cap = int(cap); self.p = float(p_restore); self.min_bank = int(min_bank)
        self.dev = device or h.obs_torch.device
        views = _per_world_arrays(h)
        # some torch-tensor state (not warp) — known ones only
        for nm in ("_prev_cmd_torch", "_regrasp_pred_streak", "regrasp_pred_torch"):
            t = getattr(h, nm, None)
            if isinstance(t, torch.Tensor) and t.shape[0] == self.N:
                views["t." + nm] = t
        self.small: List[Tuple[str, torch.Tensor]] = []
        self.large: List[Tuple[str, torch.Tensor]] = []
        for nm, v in views.items():
            cols = int(v[0].numel())
            (self.large if cols >= LARGE_COLS else self.small).append((nm, v))
        L = self.tau + 1
        self.ring = {nm: torch.zeros((L,) + tuple(v.shape), dtype=v.dtype, device=self.dev)
                     for nm, v in self.small}
        self.bank = {nm: torch.zeros((self.cap,) + tuple(v.shape[1:]), dtype=v.dtype, device=self.dev)
                     for nm, v in self.small + self.large}
        self.ptr = 0; self.n_ticks = 0
        self.bank_n = 0; self.bank_ptr = 0
        self.age = torch.zeros(self.N, dtype=torch.long, device=self.dev)   # steps elapsed in the current episode
        self.n_pushed = 0; self.n_restored = 0
        self._fd_off = next((s for (n, s, e) in h.obs_term_layout if n == "fd_vel"), -1)
        if verbose:
            ns = sum(int(v[0].numel()) for _, v in self.small)
            nl = sum(int(v[0].numel()) for _, v in self.large)
            print(f"[fail-bank] τ={self.tau} cap={self.cap} p={self.p}  small {len(self.small)} arrs "
                  f"({ns} vals/world, ring {L}×{ns*self.N*4/1e6:.0f}MB)  large {len(self.large)} arrs "
                  f"({nl} vals/world, bank {nl*self.cap*4/1e6:.0f}MB)")

    # ── every step: before env.step ─────────────────────────────────
    @torch.no_grad()
    def tick(self):
        for nm, v in self.small:
            self.ring[nm][self.ptr].copy_(v)
        self.ptr = (self.ptr + 1) % (self.tau + 1); self.n_ticks += 1

    # ── after env.step: bank s_{t-τ} of the worlds with a failure event ─
    @torch.no_grad()
    def push_failures(self, fail_mask: torch.Tensor, done_mask: torch.Tensor):
        ok = fail_mask & (self.age >= self.tau) & (self.n_ticks > self.tau)
        idx = ok.nonzero().squeeze(1)
        # update age (0 on done; otherwise +1)
        self.age = torch.where(done_mask, torch.zeros_like(self.age), self.age + 1)
        k = int(idx.numel())
        if k == 0:
            return 0
        slot = (self.ptr - 1 - self.tau) % (self.tau + 1)          # s_{t-τ} (tick records before the step)
        dst = (torch.arange(k, device=self.dev) + self.bank_ptr) % self.cap
        for nm, _ in self.small:
            self.bank[nm][dst] = self.ring[nm][slot][idx]
        for nm, v in self.large:
            self.bank[nm][dst] = v[idx]
        self.bank_ptr = (self.bank_ptr + k) % self.cap
        self.bank_n = min(self.bank_n + k, self.cap); self.n_pushed += k
        return k

    # ── after per_world_reset_if_done: restore bank states into some reset worlds ─
    @torch.no_grad()
    def restore(self, done_mask: torch.Tensor) -> int:
        if self.bank_n < self.min_bank or self.p <= 0.0:
            return 0
        pick = done_mask & (torch.rand(self.N, device=self.dev) < self.p)
        widx = pick.nonzero().squeeze(1)
        k = int(widx.numel())
        if k == 0:
            return 0
        bidx = torch.randint(0, self.bank_n, (k,), device=self.dev)
        for nm, v in self.small + self.large:
            v[widx] = self.bank[nm][bidx]
        h = self.h
        mask_wp = wp.from_torch(pick.to(torch.int32).contiguous(), dtype=wp.int32)
        wp.capture_launch(h.capture_forward.graph)
        if hasattr(h, "applier") and hasattr(h.applier, "sync_setpoint_to_qpos"):
            h.applier.sync_setpoint_to_qpos(mask_wp)
        # invalidate pose-FD velocities (avoid spurious velocity right after the teleport) — reuses the handler kernel
        if hasattr(h, "fd_valid_wp"):
            from grit.training.rl_envs.object_grasping import _fd_vel_invalidate_masked_kernel  # noqa: PLC0415
            wp.launch(_fd_vel_invalidate_masked_kernel, dim=self.N,
                      inputs=[mask_wp, int(self._fd_off), h.fd_valid_wp, h.fd_vel_wp,
                              h.fd_obj_vz_wp, h.obs_wp])
        h._collect_obs_kernel(); h._sanitize_obs()
        self.age[widx] = self.tau               # restored worlds are already τ steps into the episode
        self.n_restored += k
        return k
