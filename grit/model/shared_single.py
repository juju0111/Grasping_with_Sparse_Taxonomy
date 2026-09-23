"""shared_single_mlp — a fully shared single policy (for multi-leader LF).

:class:`SharedCrossEmbodimentMLPPolicy` keeps a per-tag adapter/head, but when
**every tag has identical obs/action dimensions** (as in multi-leader
leader-follower) and the tag identity is already encoded in the observation
(e.g. a conditioning vector), per-tag parameters hurt generalization: a new
leader is a new tag with no adapter, so zero-shot evaluation is impossible.
This class holds **a single NormalTanhMLPPolicy** internally and:

  * ``tag_view(tag)`` returns that core module **as is** — the call surface seen
    by the train loop / run_eval / ppo_update_multi is exactly that of a single
    policy.
  * Its name starts with ``shared_``, so the existing shared machinery (single
    optimizer in builders, gradient accumulation + single step in
    ``ppo_update_multi``, ``save/load_shared_checkpoint``) is reused unchanged.
  * Supports ``critic_obs_dim`` (asymmetric critic) — usable with async_ppo as
    long as all tags share the same dimension (builders allow this only for
    shared_single).

Zero-shot new leaders: the tag is only a buffer/logging key and is unrelated to
the parameters, so a new leader handler can be evaluated immediately as long as
it produces the same obs layout (including the conditioning vector).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from grit.model.base_networks import register_policy, NormalTanhMLPPolicy


@register_policy("shared_single_mlp")
class SharedSingleMLPPolicy(nn.Module):
    def __init__(self,
                 obs_dims: Mapping[str, int],
                 action_dims: Mapping[str, int],
                 critic_obs_dim: Optional[int] = None,
                 **kw):
        super().__init__()
        assert set(obs_dims) == set(action_dims) and len(obs_dims) >= 1
        o_set = set(int(v) for v in obs_dims.values())
        a_set = set(int(v) for v in action_dims.values())
        assert len(o_set) == 1 and len(a_set) == 1, (
            f"shared_single_mlp requires identical dims across tags: obs={dict(obs_dims)} "
            f"action={dict(action_dims)} — use shared_cross_embodiment_mlp for differing dims")
        self.tags        = sorted(obs_dims)
        self.obs_dims    = {t: int(obs_dims[t]) for t in obs_dims}
        self.action_dims = {t: int(action_dims[t]) for t in action_dims}
        extra = {"critic_obs_dim": int(critic_obs_dim)} if critic_obs_dim else {}
        # Single core — every tag shares this one module (no adapter/head).
        self.core = NormalTanhMLPPolicy(
            obs_dim=int(next(iter(o_set))),
            action_dim=int(next(iter(a_set))),
            **extra, **kw)
        # Compatibility with the shared-mode diagnostic print in train.py (no adapter, so latent = obs)
        self.latent_dim  = int(next(iter(o_set)))
        self.hidden_dims = list(kw.get("hidden_dim", []))

    # ── Shared-machinery contract ───────────────────────────────────────
    def tag_view(self, tag: str):
        """Ignores ``tag`` — the core itself is the view (call surface = NormalTanhMLPPolicy)."""
        assert tag in self.obs_dims, f"unknown tag {tag!r} (known: {self.tags})"
        return self.core

    # (Delegation for direct-call paths — consumers normally go through tag_view.)
    def forward(self, obs: torch.Tensor, tag: str = ""):
        return self.core(obs)

    def act(self, obs, tag: str = "", **kw) -> Dict[str, torch.Tensor]:
        return self.core.act(obs, **kw)

    def evaluate(self, obs, pre_tanh, tag: str = "", critic_obs=None):
        return self.core.evaluate(obs, pre_tanh, critic_obs=critic_obs)

    def _value(self, obs, tag: str = "", critic_obs=None):
        return self.core._value(obs, critic_obs)
