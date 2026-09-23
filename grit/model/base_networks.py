"""Policy networks + registry-based factory.

Provides a name-keyed registry so notebooks can construct policies by string
identifier::

    from grit.model.networks import make_policy

    policy = make_policy(
        "normal_tanh_mlp",
        obs_dim=OBS_DIM, action_dim=applier.action_dim,
        hidden_dim=256, min_std=0.001, var_scale=1.0,
    ).to(torch_device).train()

Register a new policy by decorating its class::

    @register_policy("my_policy")
    class MyPolicy(nn.Module):
        ...

``list_policies()`` returns the currently registered names.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Type, Union

import os

import numpy as np
import torch
import torch.nn as nn

# ``GRIT_DEBUG_NAN=1`` → enables the loc NaN check in act() (same flag as rl_env_base).
# OFF by default — the check itself forces a GPU sync per call, so it cannot live on the hot path.
_GRIT_DEBUG_NAN = os.environ.get("GRIT_DEBUG_NAN", "") == "1"

HiddenDims = Union[int, Sequence[int]]


# ──────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────

_POLICY_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_policy(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def deco(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _POLICY_REGISTRY:
            raise ValueError(f"policy '{name}' already registered")
        _POLICY_REGISTRY[name] = cls
        return cls
    return deco


def make_policy(name: str, **kwargs) -> nn.Module:
    if name not in _POLICY_REGISTRY:
        raise KeyError(
            f"unknown policy '{name}'. registered: {sorted(_POLICY_REGISTRY)}"
        )
    return _POLICY_REGISTRY[name](**kwargs)


def list_policies() -> list[str]:
    return sorted(_POLICY_REGISTRY)

# ──────────────────────────────────────────────────────────────────────────
# NormalTanhMLPPolicy: PPO-compatible squashed-Gaussian actor + critic
# ──────────────────────────────────────────────────────────────────────────
# Mirrors brax's NormalTanhDistribution:
#   loc, scale_raw  ←  actor_fc2(obs)  (split last dim → 2 × action_dim)
#   scale           = (softplus(scale_raw) + MIN_STD) × VAR_SCALE
#   pre_tanh        ~ Normal(loc, scale)            ← reparameterized sample
#   action          = tanh(pre_tanh)                ← bounded to (-1, 1)
#   log_prob        = log p(pre_tanh) - sum log|dtanh/dx|
#                   = log p(pre_tanh) - 2·sum(log 2 - pre_tanh - softplus(-2·pre_tanh))
#                                                   ← numerically stable Jacobian
# To avoid tanh saturation in PPO updates we operate on PRE-TANH actions
# (the trajectory buffer stores `pre_tanh`, not `action`); the apply kernel
# is fed the post-tanh `action` since that's what's bounded to ±1.

@register_policy("normal_tanh_mlp")
class NormalTanhMLPPolicy(nn.Module):
    LOG2 = float(np.log(2.0))

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dim: HiddenDims = 256,
                 min_std: float = 0.001,
                 var_scale: float = 1.0,
                 use_layernorm: bool = True,
                 state_dependent_std: bool = False,
                 init_std: Optional[float] = None,
                 critic_hidden_dim: Optional[HiddenDims] = None,
                 critic_obs_dim: Optional[int] = None,
                 obs_norm: bool = False,
                 obs_norm_clip: float = 10.0,
                 obs_norm_passthrough=None,
                 critic_obs_norm_passthrough=None,
                 phase_std_idx: Optional[int] = None,
                 phase_std_thresh: float = 0.02,
                 phase_std_scale_hi: float = 1.0,
                 phase_std_scale_lo: float = 1.0,
                 # The yaml names the phase channel by **obs term name**, not
                 # by index (so it follows layout changes). builders resolves
                 # it through handler.obs_term_layout() and passes
                 # phase_std_idx. Accepted but unused here — so eval/probe
                 # paths that forward cfg.policy straight into the ctor keep
                 # working (they run deterministic inference, where σ is moot).
                 phase_std_obs: Optional[str] = None,
                 phase_std_obs_comp: int = 0):
        super().__init__()
        self.action_dim = int(action_dim)
        self.min_std = float(min_std)
        self.var_scale = float(var_scale)
        self.use_layernorm = bool(use_layernorm)
        # std parameterization. ``state_dependent_std=False`` (DEFAULT) mirrors
        # brax's ``state_dependent_std=False`` / ``init_noise_std``: the actor head
        # emits ``loc`` only and std is a SINGLE global learned parameter per action
        # dim (``std_param``), which is far less prone to premature exploration
        # collapse — this is what the JAX codebase used. ``init_std`` (state-INDEP
        # only) sets the initial std via the inverse of the scale formula below.
        # ``True`` restores the older behavior where the head emits
        # ``[loc | scale_raw]`` so std is a function of the observation (set this
        # explicitly when resuming a checkpoint trained that way).
        self.state_dependent_std = bool(state_dependent_std)
        self.init_std = None if init_std is None else float(init_std)
        # ── Phase-dependent exploration σ ─────────────────────────────────
        # A single global σ cannot explore a manipulation task: the grasp pose
        # (rotation) is a **discrete event during approach** that a small σ
        # never finds, while a larger σ drops an already-held object during
        # hold. → gate σ on **one obs channel**. Since it is a function of obs
        #   alone, ``evaluate()`` reproduces the same value from the stored
        #   obs — no need to carry σ in the buffer (log_prob stays consistent).
        # Defaults (idx=None, scale 1.0) are bit-identical to the ungated behaviour.
        #   phase_std_idx    : obs index used for the phase test (builders resolves
        #                      name → index; usually the leader object's z
        #                      displacement: not lifted yet ⇒ approach)
        #   phase_std_thresh : threshold on that value (below = approach)
        #   hi / lo          : σ multiplier during approach / elsewhere
        self.phase_std_idx = None if phase_std_idx is None else int(phase_std_idx)
        self.phase_std_thresh = float(phase_std_thresh)
        # The multiplier is a scalar or a **per-dimension vector** (action_dim,)
        # so wrist and fingers can get different exploration widths: the grasp
        # pose (rotation) is decided by the wrist DOFs, whereas the same σ on
        # the fingers (finger_scale ~0.2 rad/step) drops a held object at once.
        def _sc(v, nm):
            t = torch.as_tensor(v, dtype=torch.float32)
            if t.ndim == 0:
                return t.reshape(1)
            if t.numel() != self.action_dim:
                raise ValueError(f"{nm} must be a scalar or a vector of length {self.action_dim} "
                                 f"(got {tuple(t.shape)})")
            return t.reshape(-1)
        # persistent=False — the multipliers are **configuration**, not learned values.
        # Putting them in the state_dict would break the loader when warm-starting
        # from a checkpoint trained without the phase gate (missing keys), and the
        # checkpoint would override the yaml.
        self.register_buffer("phase_std_hi", _sc(phase_std_scale_hi, "phase_std_scale_hi"),
                             persistent=False)
        self.register_buffer("phase_std_lo", _sc(phase_std_scale_lo, "phase_std_scale_lo"),
                             persistent=False)
        if self.phase_std_idx is not None:
            if not (0 <= self.phase_std_idx < int(obs_dim)):
                raise ValueError(
                    f"phase_std_idx={self.phase_std_idx} out of range for "
                    f"obs_dim={obs_dim}")
        # Asymmetric ("async") actor-critic: when ``critic_obs_dim`` is given the
        # critic trunk is built over a DIFFERENT (privileged) obs width than the
        # actor. ``None`` → critic shares the actor's ``obs_dim`` (symmetric — the
        # default, byte-identical to before).
        self.critic_obs_dim = int(critic_obs_dim) if critic_obs_dim is not None else int(obs_dim)

        # hidden_dim accepts int (single hidden layer) or sequence of ints
        # (e.g. [256, 256] → two hidden layers, each 256 wide).
        hidden_dims = self._normalize_hidden_dim(hidden_dim, "hidden_dim")
        self.hidden_dims = hidden_dims
        # Critic trunk widths — independent of the actor's when ``critic_hidden_dim``
        # is given (e.g. a deeper value net, mirroring brax's larger value_net).
        # ``None`` → reuse ``hidden_dims`` (byte-identical to before).
        self.critic_hidden_dims = (
            hidden_dims if critic_hidden_dim is None
            else self._normalize_hidden_dim(critic_hidden_dim, "critic_hidden_dim"))

        # actor: trunk → head. state-dependent → [loc | scale_raw] (2×action_dim);
        # state-independent → loc only (action_dim) + a global ``std_param``.
        self.actor_trunk = self._build_trunk(obs_dim, hidden_dims, self.use_layernorm)
        self.actor_head  = nn.Linear(
            hidden_dims[-1], (2 if self.state_dependent_std else 1) * action_dim)
        if not self.state_dependent_std:
            self.std_param = nn.Parameter(self._init_std_param(
                action_dim, self.init_std, self.min_std, self.var_scale))
        # critic: trunk over ``critic_obs_dim`` (== obs_dim unless async) → scalar value
        self.critic_trunk = self._build_trunk(self.critic_obs_dim, self.critic_hidden_dims, self.use_layernorm)
        self.critic_head  = nn.Linear(self.critic_hidden_dims[-1], 1)

        # ── Observation normalization (running mean/std, stored as buffers in the ckpt) ─
        # Statistics are updated only in rollout ``act()`` (update=self.training).
        # The PPO update's ``evaluate()`` and the GAE bootstrap's ``_value()``
        # always normalize with frozen statistics → rollout and update see the
        # same transform within one iteration. In async mode a separate
        # normalizer handles the privileged critic obs (different dimension).
        # ``*_passthrough`` (optional): True dimensions pass raw — protects
        # sparse/discrete/bounded channels (see obs_normalizer.py). Derived by
        # the handler from obs_term_layout and injected by builders (the code
        # that builds the obs is the single source of truth, not the yaml).
        from grit.model.obs_normalizer import maybe_normalizer
        self.obs_normalizer = maybe_normalizer(
            obs_norm, int(obs_dim), obs_norm_clip,
            passthrough=obs_norm_passthrough)
        self.critic_obs_normalizer = maybe_normalizer(
            obs_norm and critic_obs_dim is not None,
            self.critic_obs_dim, obs_norm_clip,
            passthrough=critic_obs_norm_passthrough)

    # ── obs-norm helpers (None-safe; ``update`` is explicit at the call site) ─
    def _norm_obs(self, obs, update: bool = False):
        if self.obs_normalizer is None:
            return obs
        return self.obs_normalizer(obs, update=update)

    def _norm_critic_obs(self, critic_obs, update: bool = False):
        if critic_obs is None:
            return None
        if self.critic_obs_normalizer is not None:      # async (separate dimension)
            return self.critic_obs_normalizer(critic_obs, update=update)
        if self.obs_normalizer is not None:              # symmetric fallback
            return self.obs_normalizer(critic_obs, update=False)
        return critic_obs

    @staticmethod
    def _init_std_param(action_dim: int, init_std: Optional[float],
                        min_std: float, var_scale: float) -> torch.Tensor:
        """Initial value for the global ``std_param`` (state-independent std).

        ``scale = (softplus(std_param) + min_std) · var_scale`` is the same
        formula used in :meth:`_params`. To make the initial scale equal
        ``init_std`` we invert it: ``softplus(p) = init_std/var_scale - min_std``
        → ``p = log(exp(s) - 1)`` (softplus inverse). ``init_std=None`` →
        ``p = 0`` (matches the state-dependent head's ``scale_raw ≈ 0`` init).
        """
        if init_std is None:
            raw = 0.0
        else:
            s = init_std / var_scale - min_std
            if s <= 0.0:
                raise ValueError(
                    f"init_std={init_std} too small for min_std={min_std}, "
                    f"var_scale={var_scale} (need init_std > min_std·var_scale)")
            # softplus inverse, numerically stable: log(expm1(s)).
            raw = float(np.log(np.expm1(s)))
        return torch.full((int(action_dim),), raw)

    @staticmethod
    def _normalize_hidden_dim(hidden_dim: HiddenDims, what: str = "hidden_dim") -> tuple:
        """int → (int,); sequence → tuple of positive ints. Shared by the actor
        and critic so ``critic_hidden_dim`` validates identically."""
        if isinstance(hidden_dim, int):
            dims = (int(hidden_dim),)
        else:
            dims = tuple(int(h) for h in hidden_dim)
        if len(dims) == 0 or any(h <= 0 for h in dims):
            raise ValueError(f"{what} must be a positive int or non-empty sequence "
                             f"of positive ints, got {hidden_dim!r}")
        return dims

    @staticmethod
    def _build_trunk(in_dim: int, hidden_dims: Sequence[int],
                     use_layernorm: bool = True) -> nn.Sequential:
        # Per hidden layer: Linear [→ LayerNorm] → ReLU.
        # LayerNorm sits BEFORE the nonlinearity so each pre-activation is
        # unit-variance regardless of weight scale — stabilises PPO when the
        # obs distribution drifts mid-training (a common cause of collapsing
        # entropy / exploding loss on long single-task runs). Disable via
        # ``use_layernorm=False`` to fall back to plain Linear→ReLU (e.g.
        # when resuming a legacy checkpoint trained without LayerNorm).
        # Heads stay plain Linear (loc / scale_raw / value should not be
        # re-centred per-sample).
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        return nn.Sequential(*layers)

    # ── Internal helpers ─────────────────────────────────────────────────
    def _params(self, obs):
        h = self.actor_trunk(obs)
        if self.state_dependent_std:
            loc, scale_raw = self.actor_head(h).chunk(2, dim=-1)
        else:
            loc = self.actor_head(h)
            scale_raw = self.std_param          # (action_dim,) → broadcasts over batch
        scale = (torch.nn.functional.softplus(scale_raw) + self.min_std) * self.var_scale
        # getattr fallback: subclasses with their own __init__ (e.g.
        # BPSEncoderMLPPolicy) lack this attribute — treat the phase gate as off.
        _pidx = getattr(self, "phase_std_idx", None)
        if _pidx is not None:
            # ``obs`` has already passed the normalization stage — LF tasks use
            # obs_norm=false so it is still in raw metres (with obs_norm on, the
            # threshold must be given in the normalized scale).
            # gate (N,1) × multiplier (1,) or (action_dim,) → (N, action_dim).
            gate = (obs[..., _pidx] < self.phase_std_thresh).unsqueeze(-1)
            scale = scale * torch.where(gate, self.phase_std_hi.to(scale.dtype),
                                        self.phase_std_lo.to(scale.dtype))
        return loc, scale

    def _value_core(self, obs_n, critic_obs_n=None):
        # Async actor-critic: the critic reads ``critic_obs`` (privileged) when
        # provided, else falls back to ``obs`` (symmetric — identical to before).
        # Inputs are already normalized (act/evaluate/_value call this after normalizing).
        x = obs_n if critic_obs_n is None else critic_obs_n
        return self.critic_head(self.critic_trunk(x)).squeeze(-1)

    def _value(self, obs, critic_obs=None):
        """Public value entry point taking RAW obs — called directly by the GAE
        bootstrap (scripts/train.py). With obs-norm enabled it normalizes with
        **frozen statistics** only (no update) before computing the value."""
        return self._value_core(self._norm_obs(obs, update=False),
                                self._norm_critic_obs(critic_obs, update=False))

    @classmethod
    def _tanh_log_jacobian(cls, x):
        # log|d tanh(x)/dx| = log(1 - tanh²(x))
        # Stable form: 2·(log 2 - x - softplus(-2x))   (avoids saturating log)
        return 2.0 * (cls.LOG2 - x - torch.nn.functional.softplus(-2.0 * x))

    @classmethod
    def _log_prob(cls, loc, scale, pre_tanh):
        normal = torch.distributions.Normal(loc, scale)
        lp_pre = normal.log_prob(pre_tanh).sum(-1)            # (batch,)
        jac = cls._tanh_log_jacobian(pre_tanh).sum(-1)        # (batch,)
        return lp_pre - jac

    # ── Deterministic inference (nn.Module entry point) ──────────────────
    def forward(self, obs):
        """Deterministic action: ``tanh(loc)`` — no sampling, no log_prob.

        Use ``act()`` during rollouts (stochastic) and ``evaluate()`` during
        PPO updates. ``forward`` is for deployment/eval where you want the
        policy mean only.
        """
        loc, _ = self._params(self._norm_obs(obs, update=False))
        return torch.tanh(loc)

    # ── Rollout: sample squashed action ──────────────────────────────────
    def act(self, obs, deterministic: bool = False, critic_obs=None,
            inference_only: bool = False):
        """``inference_only=True`` → actor-only forward: skip the critic entirely
        (``value=None``). For deployment / evaluation of an ASYNC policy, where
        the privileged critic obs is unavailable and the value head is unused.
        Rollout / PPO paths leave it False and get the value as before.

        obs-norm: statistics are updated **only here**, and only in train mode
        (``update=self.training``) — evaluation rollouts, where ``run_eval``
        calls ``policy.eval()``, normalize with frozen statistics."""
        obs        = self._norm_obs(obs, update=self.training)
        critic_obs = self._norm_critic_obs(critic_obs, update=self.training)
        loc, scale = self._params(obs)

        # loc NaN debug — only with ``GRIT_DEBUG_NAN=1``. Left ungated, the CUDA
        # tensor truthiness test in ``torch.isnan(loc).any()`` forces a **GPU→CPU
        # sync on every act() call** and breaks the sync-free rollout design
        # (LF: twice per step including the leader tick). The always-on defence
        # is obs sanitization plus the non-finite step-skip in ppo.py.
        if _GRIT_DEBUG_NAN and (torch.isnan(loc).any() or torch.isinf(loc).any()):
            print("[Error] loc contains NaN or Inf!")
            print("Is loc NaN?:", torch.isnan(loc).any().item())
            print("Is loc Inf?:", torch.isinf(loc).any().item())
            print("Input obs NaN?:", torch.isnan(obs).any().item())
            # breakpoint() # uncomment for interactive debugging at this point.


        pre_tanh = loc if deterministic else torch.distributions.Normal(loc, scale).rsample()
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob(loc, scale, pre_tanh)
        # ``obs``/``critic_obs`` were already normalized above → straight into the core.
        value = None if inference_only else self._value_core(obs, critic_obs)
        return dict(action=action, pre_tanh=pre_tanh, log_prob=log_prob,
                    value=value, loc=loc, scale=scale)

    # ── PPO update: recompute on stored pre-tanh (NOT on saturated action) ─
    def evaluate(self, obs, pre_tanh, critic_obs=None):
        # The buffer stores RAW obs → normalize with frozen statistics (no update,
        # so the epoch × minibatch re-passes do not pollute them).
        obs        = self._norm_obs(obs, update=False)
        critic_obs = self._norm_critic_obs(critic_obs, update=False)
        loc, scale = self._params(obs)
        log_prob = self._log_prob(loc, scale, pre_tanh)
        # Pre-tanh entropy in closed form (squashed entropy lacks closed form;
        # this matches brax — used for the bonus term). Subtract Jacobian if
        # you want post-tanh entropy estimate (omitted here).
        entropy = torch.distributions.Normal(loc, scale).entropy().sum(-1)
        value = self._value_core(obs, critic_obs)
        return log_prob, entropy, value

    # # ── 6D-rotation-aware bias init ──────────────────────────────────────
    # @torch.no_grad()
    # def init_rot6d_identity_bias(self, rot6d_off: int, scale_raw_bias: float = -1.0):
    #     """Initialize the actor bias so the post-tanh 6D rotation defaults to identity.

    #     ``post-tanh action[3] ≈ tanh(loc[3])``; Gram-Schmidt normalizes direction
    #     so any ``loc[3] >> 0`` makes 6D[0] dominant → ``R_delta ≈ identity``.
    #     Also tightens initial exploration by setting scale_raw bias = -1
    #     (softplus(-1) ≈ 0.31 → std ≈ 0.31).
    #     """
    #     self.actor_head.bias.zero_()
    #     self.actor_head.bias[rot6d_off + 0] = 1.0       # loc for action[3] (= 6D[0])
    #     self.actor_head.bias[rot6d_off + 4] = 1.0       # loc for action[7] (= 6D[4])
    #     self.actor_head.bias[self.action_dim:].fill_(scale_raw_bias)


# ──────────────────────────────────────────────────────────────────────────
# Per-finger weight-shared encoder
# ──────────────────────────────────────────────────────────────────────────
class _PerFingerTrunk(nn.Module):
    """Thin **shared** per-finger encoder → concat with global features → regular trunk.

    Motivation: hand-tracking obs have a **repeated structure** across the five
    fingers, yet a monolithic MLP receives them flat and learns the same rule
    ("this joint configuration yields this fingertip position") separately per
    finger. Sharing weights trains one encoder on data from all five fingers —
    **5× the effective sample size** — and the rule transfers even when a new
    pose combines fingers differently from anything seen before.

    Note: the thumb is not symmetric with the others (different kinematics, and
    it is the hub of every pinch). A finger-ID one-hot is therefore appended to
    the input so the **shared encoder can specialize where needed** — the data
    decides how much sharing pays off.

    The index groups are derived and passed in by the **same builder** the
    handler uses to construct the obs (``per_finger_obs_groups``). Writing them
    by hand silently groups the wrong fingers.
    """

    def __init__(self, finger_idx, global_idx, enc_dim: int,
                 hidden_dims, use_layernorm: bool, trunk_builder,
                 finger_id_onehot: bool = True):
        super().__init__()
        n_f = len(finger_idx)
        lens = [len(g) for g in finger_idx]
        per  = max(lens)
        # Note: the per-finger dimension **differs between hands**. Some hands
        # have a uniform 3 joints per finger (roll+MCP+PIP), while others (e.g.
        # inspire) have the roll (abduction) joint only on some fingers or not
        # at all. Assuming uniformity would crash on such hands — fatal for a
        # cross-embodiment project.
        # → **pad to the max width + validity mask**. The mask distinguishes
        #   "value is 0" from "slot does not exist", so the encoder does not
        #   misread missing entries. Without any padding (uniform hand) the mask
        #   would be a constant 1 = dead dimension, so it is omitted → identical
        #   to the mask-free path on uniform hands.
        self.ragged = bool(min(lens) != per)
        pad_idx  = [list(g) + [0] * (per - len(g)) for g in finger_idx]
        pad_mask = [[1.0] * len(g) + [0.0] * (per - len(g)) for g in finger_idx]
        if self.ragged:
            print(f"[per-finger] warning: ragged per-finger dims {lens} → padding to {per} + "
                  f"adding a {per}-channel validity mask (missing slots are flagged as absent, not 0)")
        self.n_fingers = n_f
        self.use_onehot = bool(finger_id_onehot)
        self.register_buffer("fidx", torch.as_tensor(pad_idx, dtype=torch.long))
        self.register_buffer("fmask", torch.as_tensor(pad_mask, dtype=torch.float32))
        self.register_buffer("gidx", torch.as_tensor(global_idx, dtype=torch.long))
        if self.use_onehot:
            self.register_buffer("onehot", torch.eye(n_f))
        in_dim = per + (per if self.ragged else 0) + (n_f if self.use_onehot else 0)
        enc: list[nn.Module] = [nn.Linear(in_dim, enc_dim)]
        if use_layernorm:
            enc.append(nn.LayerNorm(enc_dim))
        enc += [nn.ReLU(), nn.Linear(enc_dim, enc_dim)]
        if use_layernorm:
            enc.append(nn.LayerNorm(enc_dim))
        enc.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)          # the **same** weights for all fingers
        self.trunk = trunk_builder(n_f * enc_dim + len(global_idx),
                                   hidden_dims, use_layernorm)

    def forward(self, obs):
        f = obs[..., self.fidx]                              # (B, n_f, per)
        if self.ragged:
            m = self.fmask.expand(f.shape[0], -1, -1)        # (B, n_f, per)
            f = torch.cat([f * m, m], dim=-1)                # missing slots zeroed + mask
        if self.use_onehot:
            oh = self.onehot.expand(f.shape[0], -1, -1)      # (B, n_f, n_f)
            f = torch.cat([f, oh], dim=-1)
        e = self.encoder(f)                                  # (B, n_f, enc)
        x = torch.cat([e.flatten(start_dim=1), obs[..., self.gidx]], dim=-1)
        return self.trunk(x)


@register_policy("per_finger_mlp")
class PerFingerMLPPolicy(NormalTanhMLPPolicy):
    """``normal_tanh_mlp`` with **only the actor trunk** replaced by the shared per-finger encoder.

    The act / evaluate / critic / obs-normalizer paths are inherited unchanged —
    changing a single axis keeps results interpretable and limits risk. The
    critic stays a plain MLP (value regression carries no IK-decoding burden).

    ``finger_groups`` is obtained from the handler by builders. When absent
    (a task without known finger structure) this behaves exactly like the parent.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 finger_groups=None, finger_enc_dim: int = 48,
                 finger_id_onehot: bool = True, **kw):
        super().__init__(obs_dim, action_dim, **kw)
        if not finger_groups:
            print("[per-finger] finger_groups not provided → running as a monolithic MLP (same as parent)")
            return
        finger_idx, global_idx = finger_groups
        self.actor_trunk = _PerFingerTrunk(
            finger_idx, global_idx, int(finger_enc_dim),
            self.hidden_dims, self.use_layernorm, self._build_trunk,
            finger_id_onehot=bool(finger_id_onehot))
        n_enc = sum(p.numel() for p in self.actor_trunk.encoder.parameters())
        print(f"[per-finger] actor: {len(finger_idx)} fingers × {len(finger_idx[0])} dims "
              f"→ shared encoder({finger_enc_dim}) {n_enc:,} params + {len(global_idx)} global "
              f"→ trunk input {len(finger_idx) * int(finger_enc_dim) + len(global_idx)}")
