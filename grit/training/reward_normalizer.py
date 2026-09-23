"""Running-statistics reward normalization for the PPO rollout.

Why
---
``ppo_update`` regresses the critic on RAW discounted returns
(``vf_loss = MSE(value, return)``). When per-step rewards are large — or their
scale *shifts mid-run* (reward-knob curriculum / success-gate weight ramps) —
the value-loss gradient dominates the global grad norm (chronic pre-clip
``grad_norm`` ≫ ``max_grad_norm``) and the critic chases a moving scale.
Dividing rollout rewards by a running scale keeps returns O(1) for the critic
without touching env code, eval metrics, or per-term reward logging (those all
stay RAW).

What
----
:class:`RunningMeanStd`
    Shape-agnostic Welford/Chan parallel running mean+variance over torch
    tensors. Sync-free: ``update()`` does no ``.item()`` — statistics live as
    GPU tensors (float64 accumulators for numerical safety).

:class:`RewardNormalizer`
    Per-tag (per-handler) reward transform applied between ``handler.step()``
    and ``buffer.add()``. Two modes:

    * ``mode="return"`` (default, gym ``NormalizeReward`` / VecNormalize
      convention): track the per-env running **discounted return**
      ``ret ← γ·(1−done)·ret + r`` and divide the reward by the running *std of
      that return*. No centering → reward SIGN and relative structure are
      preserved; only the scale is stabilised. Recommended.
    * ``mode="reward"``: running mean/var of the raw per-step reward;
      ``r / std``, optionally centered (``center=true`` → ``(r − mean)/std``).
      NOTE: centering flips the sign of below-average rewards — it changes
      what "0 reward" means and can reward-shape termination avoidance.
      Keep ``center=false`` unless you know you want that.

Config (``cfg.training.reward_norm`` — defaults in ``config/training/base.yaml``)::

    training:
      reward_norm:
        enabled: false     # master switch
        mode:    return    # return | reward
        center:  false     # reward mode only
        clip:    10.0      # post-normalization |r| clip; 0/null = off
        eps:     1.0e-8    # variance floor
        gamma:   null      # discount for return mode; null → training.gamma

Checkpointing: ``state_dict()`` / ``load_state_dict()`` round-trip through the
checkpoint blob (``extra`` payload) so a resumed run keeps the reward scale the
saved critic was trained against. The per-env return carrier is restored only
when ``num_envs`` matches (else it re-warms in a few episodes).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

__all__ = ["RunningMeanStd", "RewardNormalizer"]


class RunningMeanStd:
    """Running mean/variance over batches (Chan et al. parallel update).

    ``shape`` is the per-sample statistic shape (``()`` for scalar rewards);
    ``update(x)`` treats ``x``'s leading dim as the batch. All state stays on
    ``device`` as float64 tensors — no host sync in the hot path.
    """

    def __init__(self, shape=(), device="cpu", eps: float = 1e-4):
        self.mean  = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var   = torch.ones(shape,  dtype=torch.float64, device=device)
        self.count = float(eps)          # small init count avoids div-by-0 on the first batch

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, *self.mean.shape).to(dtype=torch.float64)
        batch_count = x.shape[0]
        if batch_count == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var  = x.var(dim=0, unbiased=False)

        delta     = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / tot_count)
        m_a  = self.var * self.count
        m_b  = batch_var * batch_count
        m2   = m_a + m_b + delta.square() * (self.count * batch_count / tot_count)
        self.var   = m2 / tot_count
        self.count = tot_count

    def std(self, eps: float = 1e-8) -> torch.Tensor:
        return torch.sqrt(self.var + eps)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean":  self.mean.detach().cpu(),
            "var":   self.var.detach().cpu(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        device = self.mean.device
        self.mean  = state["mean"].to(dtype=torch.float64, device=device)
        self.var   = state["var"].to(dtype=torch.float64, device=device)
        self.count = float(state["count"])


class RewardNormalizer:
    """Normalize a per-step reward batch by running statistics (see module doc).

    Call as ``rew_n = normalizer(rew, done)`` between ``handler.step()`` and
    ``buffer.add()``. ``done`` is the same float (0/1) tensor the discount uses;
    it zeroes the discounted-return carrier AFTER this step's reward is folded
    in (gym ``NormalizeReward`` ordering). Returns a NEW tensor — the handler's
    reward buffer is never mutated, so raw-metric consumers stay untouched.
    """

    MODES = ("return", "reward")

    def __init__(
        self,
        num_envs: int,
        device,
        *,
        gamma:  float = 0.99,
        mode:   str   = "return",
        center: bool  = False,
        clip:   float = 10.0,
        eps:    float = 1e-8,
    ):
        if mode not in self.MODES:
            raise ValueError(f"RewardNormalizer: unknown mode {mode!r} (expected one of {self.MODES})")
        if center and mode != "reward":
            raise ValueError("RewardNormalizer: center=True only applies to mode='reward'")
        self.mode   = mode
        self.gamma  = float(gamma)
        self.center = bool(center)
        self.clip   = float(clip) if clip else 0.0
        self.eps    = float(eps)
        self.rms    = RunningMeanStd((), device=device)
        # Per-env discounted-return carrier (return mode only; inert otherwise).
        self.ret    = torch.zeros(num_envs, dtype=torch.float64, device=device)

    @classmethod
    def from_config(
        cls,
        cfg: Optional[Dict[str, Any]],
        *,
        num_envs: int,
        device,
        default_gamma: float,
    ) -> Optional["RewardNormalizer"]:
        """Build from a ``cfg.training.reward_norm`` dict; ``None`` when the
        section is absent or ``enabled`` is falsy. ``gamma: null`` inherits
        ``default_gamma`` (= ``training.gamma``)."""
        if not cfg or not bool(cfg.get("enabled", False)):
            return None
        gamma = cfg.get("gamma", None)
        return cls(
            num_envs, device,
            gamma  = float(gamma) if gamma is not None else float(default_gamma),
            mode   = str(cfg.get("mode", "return")),
            center = bool(cfg.get("center", False)),
            clip   = float(cfg.get("clip") or 0.0),
            eps    = float(cfg.get("eps", 1e-8)),
        )

    @torch.no_grad()
    def __call__(self, reward: torch.Tensor, done: Optional[torch.Tensor] = None) -> torch.Tensor:
        r64 = reward.detach().to(dtype=torch.float64)
        if self.mode == "return":
            # ret ← γ·ret + r, stats over the return, THEN zero finished envs —
            # the terminal step's reward still counts toward the return scale.
            self.ret.mul_(self.gamma).add_(r64)
            self.rms.update(self.ret)
            out = r64 / self.rms.std(self.eps)
            if done is not None:
                self.ret.mul_(1.0 - done.detach().to(dtype=torch.float64))
        else:  # "reward"
            self.rms.update(r64)
            if self.center:
                out = (r64 - self.rms.mean) / self.rms.std(self.eps)
            else:
                out = r64 / self.rms.std(self.eps)
        if self.clip > 0.0:
            out = out.clamp_(-self.clip, self.clip)
        return out.to(dtype=reward.dtype)

    @property
    def scale(self) -> float:
        """Current divisor (running std) — host sync; use in logging cadence only."""
        return float(self.rms.std(self.eps).item())

    def reset_returns(self) -> None:
        """Zero the per-env return carrier (call after an env rebuild/resample
        where every episode restarts). Running mean/var statistics are kept."""
        self.ret.zero_()

    # ── Checkpoint round-trip ────────────────────────────────────────────
    def state_dict(self) -> Dict[str, Any]:
        return {
            "mode":   self.mode,
            "gamma":  self.gamma,
            "center": self.center,
            "clip":   self.clip,
            "eps":    self.eps,
            "rms":    self.rms.state_dict(),
            "ret":    self.ret.detach().cpu(),
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore running statistics. Tolerant: ``None``/missing → no-op
        (older ckpt without the payload); knob mismatches keep the CURRENT
        config (only statistics are restored); a ``ret`` carrier with a
        different ``num_envs`` is skipped (re-warms in a few episodes)."""
        if not state:
            return
        if state.get("mode") != self.mode:
            print(f"  [reward_norm] ckpt mode {state.get('mode')!r} != cfg mode "
                  f"{self.mode!r} — restoring statistics into the cfg mode anyway.")
        self.rms.load_state_dict(state["rms"])
        ret = state.get("ret")
        if isinstance(ret, torch.Tensor) and ret.shape == self.ret.shape:
            self.ret.copy_(ret.to(dtype=self.ret.dtype, device=self.ret.device))
        else:
            self.ret.zero_()
