"""Running mean/std observation normalizer (stabilizes PPO training).

Applies ``obs_n = clip((obs - mean) / sqrt(var + eps), -clip, clip)``, where
mean/var are updated **online (parallel Welford)** from the batches observed
during rollout.

Design notes
------------
* Implemented as a :class:`torch.nn.Module` with the statistics stored as
  **buffers**, so they are automatically included in ``policy.state_dict()``
  and checkpoint save/restore (resume, evaluate.py) works without extra code.
* Statistics updates are **explicit** (``forward(x, update=True)`` or
  ``update(x)``) — since nothing updates automatically, the PPO update's
  ``evaluate()`` (which re-passes the same rollout data epoch × minibatch
  times) cannot double-count into the statistics.
  Calling convention: update **only in rollout ``act()``** with
  ``update=self.training`` (``run_eval`` calls ``policy.eval()``, so the
  statistics are frozen automatically during evaluation).
* Precision: with 1e8+ samples accumulated online, mean/var/count are kept as
  float64 buffers and cast to the input dtype at normalization time.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class RunningObsNormalizer(nn.Module):
    """Standardize + clip obs with per-dimension running mean/std.

    Args:
        dim:   observation dimension.
        clip:  clamp range after normalization (±clip). 5-10 by PPO convention.
        eps:   denominator stabilization term.
    """

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-8,
                 passthrough=None):
        super().__init__()
        self.dim  = int(dim)
        self.clip = float(clip)
        self.eps  = float(eps)
        # float64: with hundreds of millions of samples accumulated online, fp32 lets the mean drift.
        self.register_buffer("running_mean", torch.zeros(self.dim, dtype=torch.float64))
        self.register_buffer("running_var",  torch.ones(self.dim,  dtype=torch.float64))
        # A tiny initial count (effectively 0) lets the first batch dominate the statistics.
        self.register_buffer("count", torch.full((1,), 1e-4, dtype=torch.float64))
        # ── Passthrough mask ──────────────────────────────────────────────
        # Dimensions marked True pass through raw, without standardization or
        # clipping. Sparse (contact impulse, BPS), discrete (masks, one-hot) and
        # bounded ([0,1], ±1) channels are excluded because their running σ
        # collapses toward 0, so a rare activation becomes z = tens and
        # saturates at ±clip (losing the event magnitude). Registered as a
        # buffer so it travels in the checkpoint and training / evaluation /
        # deployment all use the same mask. None → standardize every dimension.
        if passthrough is not None:
            pt = torch.as_tensor(passthrough, dtype=torch.bool).reshape(-1)
            assert pt.numel() == self.dim, (
                f"passthrough mask {pt.numel()} != dim {self.dim}")
            self.register_buffer("passthrough", pt)
            n_std = int((~pt).sum())
            print(f"[obs-norm] dim={self.dim}: standardize {n_std} / "
                  f"passthrough {self.dim - n_std}")
        else:
            self.passthrough = None

    # ── Statistics update (parallel Welford / Chan et al.) ──────────────
    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running mean/var from a (B, dim) batch."""
        x = x.detach().reshape(-1, self.dim).to(torch.float64)
        b_count = x.shape[0]
        if b_count == 0:
            return
        b_mean = x.mean(dim=0)
        b_var  = x.var(dim=0, unbiased=False)

        delta     = b_mean - self.running_mean
        tot_count = self.count + b_count

        new_mean = self.running_mean + delta * (b_count / tot_count)
        m_a = self.running_var * self.count
        m_b = b_var * b_count
        m_2 = m_a + m_b + delta.square() * (self.count * b_count / tot_count)

        self.running_mean.copy_(new_mean)
        self.running_var.copy_(m_2 / tot_count)
        self.count.copy_(tot_count)

    # ── Apply ────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
        """Normalize (optionally updating the statistics first).

        ``update`` is **explicit at the call site** — only rollout act() passes True.
        """
        if update:
            self.update(x)
        mean = self.running_mean.to(x.dtype)
        std  = self.running_var.add(self.eps).sqrt().to(x.dtype)
        y = ((x - mean) / std).clamp_(-self.clip, self.clip)
        if self.passthrough is not None:
            # Passthrough dimensions pass raw (no standardization/clip). Statistics
            # are still updated for every dimension but ignored on apply, so the
            # running values survive a mask change.
            y = torch.where(self.passthrough, x, y)
        return y

    def extra_repr(self) -> str:
        n_pt = (int(self.passthrough.sum()) if self.passthrough is not None else 0)
        return (f"dim={self.dim}, clip={self.clip}, passthrough={n_pt}, "
                f"count={float(self.count.item()):.0f}")


def maybe_normalizer(enabled: bool, dim: int, clip: float = 10.0,
                     passthrough=None) -> Optional[RunningObsNormalizer]:
    """Create a normalizer only when ``enabled`` (helper for policy __init__)."""
    return (RunningObsNormalizer(dim, clip=clip, passthrough=passthrough)
            if enabled else None)
