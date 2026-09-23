"""Shared-trunk + per-tag head cross-embodiment policy.

Used when training one **unified model** that can act on multiple hand
morphologies (different ``obs_dim`` / ``action_dim`` per tag). The shared
trunk learns a cross-task representation; per-tag input adapters and
output heads specialise to each hand.

Architecture (actor / critic each mirror :class:`NormalTanhMLPPolicy`'s
structure — separate trunks)::

    obs_tag → adapter[tag] (Linear: obs_dim_tag → latent_dim)
            → trunk        (MLP   : latent_dim → ... → trunk_out)         ← SHARED
            → head[tag]    (Linear: trunk_out → 2 × action_dim_tag        [actor]
                                              → 1                          [critic])

Train-loop integration: see :class:`_TagView` below. A view binds a
single ``tag`` to the shared module so ``policies[tag].act(obs)`` /
``policies[tag].evaluate(obs, pre_tanh)`` work unchanged — all views
share the **same** :class:`nn.Module` parameters, so a single
:class:`torch.optim.Adam` over ``shared.parameters()`` updates every
tag's adapter + head + the shared trunk.

YAML example (drop into ``cfg.policy``)::

    policy:
      name:          shared_cross_embodiment_mlp
      latent_dim:    128
      hidden_dim:    [256, 256, 128, 64]
      min_std:       0.1
      var_scale:     0.5
      use_layernorm: true

``obs_dims`` and ``action_dims`` are auto-derived from the handlers at
build time — the YAML only carries the architecture dims.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grit.model.base_networks import register_policy


HiddenDims = Union[int, Sequence[int]]


@register_policy("shared_cross_embodiment_mlp")
class SharedCrossEmbodimentMLPPolicy(nn.Module):
    """Squashed-Gaussian actor + critic with **shared trunk** and
    **per-tag input adapter + output head**.

    Methods take an explicit ``tag`` arg identifying which embodiment is
    being acted on. Use :meth:`tag_view` to bind a single tag and obtain
    a drop-in replacement for :class:`NormalTanhMLPPolicy` (no ``tag``
    arg needed at call sites).
    """
    LOG2 = float(np.log(2.0))

    def __init__(
        self,
        obs_dims:    Mapping[str, int],
        action_dims: Mapping[str, int],
        latent_dim:  int           = 128,
        hidden_dim:  HiddenDims    = (256, 256),
        min_std:     float         = 0.001,
        var_scale:   float         = 1.0,
        use_layernorm: bool        = True,
        obs_norm:      bool        = False,
        obs_norm_clip: float       = 10.0,
    ):
        super().__init__()
        if set(obs_dims) != set(action_dims):
            raise ValueError(
                f"obs_dims tags {sorted(obs_dims)} must equal action_dims "
                f"tags {sorted(action_dims)}"
            )
        self.tags        = list(obs_dims.keys())
        self.obs_dims    = dict(obs_dims)
        self.action_dims = dict(action_dims)
        self.latent_dim  = int(latent_dim)
        self.min_std     = float(min_std)
        self.var_scale   = float(var_scale)
        self.use_layernorm = bool(use_layernorm)

        # hidden_dim accepts int or sequence (matches NormalTanhMLPPolicy).
        if isinstance(hidden_dim, int):
            hidden_dims = (int(hidden_dim),)
        else:
            hidden_dims = tuple(int(h) for h in hidden_dim)
        if len(hidden_dims) == 0 or any(h <= 0 for h in hidden_dims):
            raise ValueError(
                f"hidden_dim must be a positive int or non-empty sequence; "
                f"got {hidden_dim!r}"
            )
        self.hidden_dims = hidden_dims

        # ── Actor: per-tag adapter → shared trunk → per-tag head ──────
        self.actor_adapter = nn.ModuleDict({
            tag: nn.Linear(self.obs_dims[tag], self.latent_dim)
            for tag in self.tags
        })
        self.actor_trunk = self._build_trunk(self.latent_dim, hidden_dims, self.use_layernorm)
        self.actor_head  = nn.ModuleDict({
            tag: nn.Linear(hidden_dims[-1], 2 * self.action_dims[tag])
            for tag in self.tags
        })

        # ── Critic: mirror structure (separate trunk, like NormalTanhMLPPolicy) ─
        self.critic_adapter = nn.ModuleDict({
            tag: nn.Linear(self.obs_dims[tag], self.latent_dim)
            for tag in self.tags
        })
        self.critic_trunk = self._build_trunk(self.latent_dim, hidden_dims, self.use_layernorm)
        self.critic_head  = nn.ModuleDict({
            tag: nn.Linear(hidden_dims[-1], 1)
            for tag in self.tags
        })

        # ── Per-tag observation normalizer (obs_dim differs per tag) ───────
        # Same convention as NormalTanhMLPPolicy: statistics are updated only in
        # rollout ``act()`` (update=self.training); evaluate/_value/forward
        # normalize with frozen statistics.
        self.obs_normalizer: Optional[nn.ModuleDict] = None
        if obs_norm:
            from grit.model.obs_normalizer import RunningObsNormalizer
            self.obs_normalizer = nn.ModuleDict({
                tag: RunningObsNormalizer(self.obs_dims[tag], clip=obs_norm_clip)
                for tag in self.tags
            })

    def _norm_obs(self, obs, tag: str, update: bool = False):
        if self.obs_normalizer is None:
            return obs
        return self.obs_normalizer[tag](obs, update=update)

    @staticmethod
    def _build_trunk(in_dim: int, hidden_dims: Sequence[int],
                     use_layernorm: bool = True) -> nn.Sequential:
        """Same Linear[→LayerNorm]→ReLU stack as ``NormalTanhMLPPolicy._build_trunk``."""
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        return nn.Sequential(*layers)

    # ── Internal helpers (matches NormalTanhMLPPolicy contract) ────────
    def _params(self, obs, tag: str):
        """Return (loc, scale) for the policy distribution at ``tag``."""
        z = self.actor_adapter[tag](obs)
        h = self.actor_trunk(z)
        loc, scale_raw = self.actor_head[tag](h).chunk(2, dim=-1)
        scale = (F.softplus(scale_raw) + self.min_std) * self.var_scale
        return loc, scale

    def _value_core(self, obs_n, tag: str):
        z = self.critic_adapter[tag](obs_n)
        h = self.critic_trunk(z)
        return self.critic_head[tag](h).squeeze(-1)

    def _value(self, obs, tag: str):
        """Public value entry point taking RAW obs (called by the GAE bootstrap
        via TagView) — normalizes with frozen statistics only, then computes value."""
        return self._value_core(self._norm_obs(obs, tag, update=False), tag)

    @classmethod
    def _tanh_log_jacobian(cls, x):
        return 2.0 * (cls.LOG2 - x - F.softplus(-2.0 * x))

    @classmethod
    def _log_prob(cls, loc, scale, pre_tanh):
        normal = torch.distributions.Normal(loc, scale)
        lp_pre = normal.log_prob(pre_tanh).sum(-1)
        jac    = cls._tanh_log_jacobian(pre_tanh).sum(-1)
        return lp_pre - jac

    # ── Forward / act / evaluate (with explicit ``tag``) ───────────────
    def forward(self, obs, tag: str):
        loc, _ = self._params(self._norm_obs(obs, tag, update=False), tag)
        return torch.tanh(loc)

    def act(self, obs, tag: str, deterministic: bool = False, critic_obs=None,
            inference_only: bool = False):
        # ``critic_obs`` accepted for call-surface parity with the single-task
        # policies; shared cross-embodiment does NOT support async actor-critic
        # (builders block it) so it is ignored here. ``inference_only=True`` skips
        # the critic (value=None) — actor-only forward for deployment / eval.
        # obs-norm statistics are updated only here (in train mode; frozen in eval).
        obs        = self._norm_obs(obs, tag, update=self.training)
        loc, scale = self._params(obs, tag)
        pre_tanh   = (loc if deterministic
                      else torch.distributions.Normal(loc, scale).rsample())
        action     = torch.tanh(pre_tanh)
        log_prob   = self._log_prob(loc, scale, pre_tanh)
        value      = None if inference_only else self._value_core(obs, tag)
        return dict(action=action, pre_tanh=pre_tanh, log_prob=log_prob,
                    value=value, loc=loc, scale=scale)

    def evaluate(self, obs, pre_tanh, tag: str, critic_obs=None):
        # RAW obs from the buffer → frozen normalization (so PPO re-passes do not pollute the statistics).
        obs        = self._norm_obs(obs, tag, update=False)
        loc, scale = self._params(obs, tag)        # critic_obs ignored (see ``act``)
        log_prob   = self._log_prob(loc, scale, pre_tanh)
        entropy    = torch.distributions.Normal(loc, scale).entropy().sum(-1)
        value      = self._value_core(obs, tag)
        return log_prob, entropy, value

    # ── Bind a tag → drop-in single-tag policy ─────────────────────────
    def tag_view(self, tag: str) -> "_TagView":
        if tag not in self.tags:
            raise KeyError(f"unknown tag {tag!r}. registered: {self.tags}")
        return _TagView(self, tag)


class _TagView:
    """Bind a single ``tag`` to a :class:`SharedCrossEmbodimentMLPPolicy`
    so the training loop can call ``view.act(obs)`` /
    ``view.evaluate(obs, pre_tanh)`` / ``view._value(obs)`` unchanged.

    All TagViews of the same shared policy reference the **same
    :class:`nn.Module` parameters** — a single
    :class:`torch.optim.Adam` over ``shared.parameters()`` updates every
    tag's adapter + head + the shared trunk. The view does NOT make
    copies; ``parameters()`` / ``state_dict()`` / ``load_state_dict()``
    delegate directly to the shared module.
    """

    def __init__(self, shared: SharedCrossEmbodimentMLPPolicy, tag: str):
        self._shared = shared
        self._tag    = tag

    # ── Identity ──────────────────────────────────────────────────────
    @property
    def shared(self) -> SharedCrossEmbodimentMLPPolicy: return self._shared
    @property
    def tag(self)    -> str:                            return self._tag
    @property
    def obs_dim(self)    -> int: return self._shared.obs_dims[self._tag]
    @property
    def action_dim(self) -> int: return self._shared.action_dims[self._tag]
    @property
    def training(self)   -> bool: return self._shared.training

    def __repr__(self) -> str:
        return (f"_TagView(tag={self._tag!r}, "
                f"shared={type(self._shared).__name__}, "
                f"obs_dim={self.obs_dim}, action_dim={self.action_dim})")

    # ── Forward proxies (tag-bound) ───────────────────────────────────
    def forward(self, obs):
        return self._shared.forward(obs, self._tag)

    def act(self, obs, deterministic: bool = False, critic_obs=None,
            inference_only: bool = False):
        return self._shared.act(obs, self._tag, deterministic=deterministic,
                                inference_only=inference_only)

    def evaluate(self, obs, pre_tanh, critic_obs=None):
        return self._shared.evaluate(obs, pre_tanh, self._tag)

    def _value(self, obs, critic_obs=None):
        return self._shared._value(obs, self._tag)

    # ── nn.Module surface (delegates to shared so optim + ckpt work) ──
    def parameters(self, recurse: bool = True):
        return self._shared.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self._shared.named_parameters(prefix=prefix, recurse=recurse)

    def train(self, mode: bool = True):
        self._shared.train(mode); return self

    def eval(self):
        self._shared.eval(); return self

    def to(self, *args, **kwargs):
        self._shared.to(*args, **kwargs); return self

    def state_dict(self, *args, **kwargs):
        return self._shared.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self._shared.load_state_dict(state_dict, strict=strict)
