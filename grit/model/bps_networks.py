"""BPS-encoder policy networks.

A policy variant for the ``grasping`` task (BPS shape obs): the observation is
``[ base_obs (D_base) | bps_feature (M) ]`` (the BPS block is appended at the
TAIL by :class:`grit.training.rl_envs.grasping.GraspingHandler`).
Feeding the raw ``M``-dim BPS vector (e.g. 1024) straight into a wide MLP
dwarfs the ``D_base`` proprioceptive/contact signal and blows up the first
layer. Instead this policy:

    1. **encodes** the BPS sub-vector with its own small MLP
       ``M → ... → bps_latent_dim`` (default 1024 → 256 → 128 → 64),
    2. **concatenates** the BPS latent back with the (untouched) base obs
       → ``[ base_obs | bps_latent ]`` of width ``D_base + bps_latent_dim``,
    3. runs that through the **main trunk MLP → action head** (squashed
       Gaussian) — exactly the actor/critic head + PPO machinery inherited
       from :class:`NormalTanhMLPPolicy`.

Only the trunk changes; ``act`` / ``evaluate`` / ``forward`` / value head /
tanh-squashed log-prob are reused verbatim from the base policy, so it is a
drop-in PPO actor-critic.

Registered as ``bps_encoder_mlp`` — select it in the training YAML::

    policy:
      name:            bps_encoder_mlp
      bps_dim:         1024            # MUST equal the env's n_bps (BPS tail width)
      bps_encoder_dim: [256, 128]      # BPS encoder hidden layers
      bps_latent_dim:  64              # BPS encoder output width
      hidden_dim:      [256, 256, 128, 64]   # main trunk (over [base | bps_latent])
      min_std:         0.001
      var_scale:       0.5
      use_layernorm:   false

``bps_dim`` is the width of the BPS tail; it MUST match the env's ``n_bps``
(``obs_dim = D_base + bps_dim``). It is passed via config rather than derived
because the policy factory only receives ``obs_dim`` / ``action_dim`` — an
assert below fails fast if it is inconsistent with ``obs_dim``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from grit.model.base_networks import (
    register_policy,
    NormalTanhMLPPolicy,
    HiddenDims,
)


def _as_dims(x: HiddenDims, what: str) -> tuple[int, ...]:
    """Normalize an int-or-sequence hidden-dim spec to a tuple of positive ints."""
    dims = (int(x),) if isinstance(x, int) else tuple(int(h) for h in x)
    if len(dims) == 0 or any(h <= 0 for h in dims):
        raise ValueError(f"{what} must be a positive int or non-empty sequence "
                         f"of positive ints, got {x!r}")
    return dims


class BPSSplitEncoderTrunk(nn.Module):
    """Trunk that encodes the BPS obs tail, concats it with the base obs, then
    runs a main MLP. Output width = ``hidden_dims[-1]`` so the downstream
    actor/critic heads from :class:`NormalTanhMLPPolicy` plug in unchanged.

    forward(obs):
        base    = obs[..., :base_dim]                 # proprioception / contact / etc.
        bps     = obs[..., base_dim:base_dim+bps_dim] # BPS feature tail
        z       = bps_encoder(bps)                    # (..., bps_latent_dim)
        return main_trunk( cat([base, z], -1) )       # (..., hidden_dims[-1])
    """

    def __init__(self, base_dim: int, bps_dim: int,
                 bps_encoder_dims: Sequence[int], bps_latent_dim: int,
                 hidden_dims: Sequence[int], use_layernorm: bool = True):
        super().__init__()
        self.base_dim = int(base_dim)
        self.bps_dim = int(bps_dim)
        self.bps_latent_dim = int(bps_latent_dim)

        # BPS encoder: M → bps_encoder_dims... → bps_latent_dim.
        # Same Linear[→LayerNorm]→ReLU block as the base trunk, plus a final
        # Linear→ReLU projection to the latent width.
        enc: list[nn.Module] = []
        prev = int(bps_dim)
        for h in bps_encoder_dims:
            enc.append(nn.Linear(prev, int(h)))
            if use_layernorm:
                enc.append(nn.LayerNorm(int(h)))
            enc.append(nn.ReLU())
            prev = int(h)
        enc.append(nn.Linear(prev, self.bps_latent_dim))
        enc.append(nn.ReLU())
        self.bps_encoder = nn.Sequential(*enc)

        # Main trunk over [base_obs | bps_latent]. Reuses the base policy's
        # trunk builder so LayerNorm placement / nonlinearity match exactly.
        self.main_trunk = NormalTanhMLPPolicy._build_trunk(                # type: ignore 
            self.base_dim + self.bps_latent_dim, hidden_dims, use_layernorm,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        base = obs[..., :self.base_dim]
        bps  = obs[..., self.base_dim:self.base_dim + self.bps_dim]
        z    = self.bps_encoder(bps)
        return self.main_trunk(torch.cat([base, z], dim=-1))


@register_policy("bps_encoder_mlp")
class BPSEncoderMLPPolicy(NormalTanhMLPPolicy):
    """PPO squashed-Gaussian actor-critic with a BPS-encoding trunk.

    Identical to :class:`NormalTanhMLPPolicy` except the actor and critic
    trunks split off the BPS obs tail, encode it, and concatenate the latent
    with the base obs before the main MLP. Actor / critic each get their OWN
    BPS encoder + trunk (mirroring the base policy's separate trunks).
    """

    def __init__(self, obs_dim: int, action_dim: int, bps_dim: int,
                 hidden_dim: HiddenDims = 256,
                 bps_encoder_dim: HiddenDims = (256, 128),
                 bps_latent_dim: int = 64,
                 min_std: float = 0.001,
                 var_scale: float = 1.0,
                 use_layernorm: bool = False,
                 state_dependent_std: bool = False,
                 init_std: Optional[float] = None,
                 critic_hidden_dim: Optional[HiddenDims] = None,
                 critic_obs_dim: Optional[int] = None,
                 obs_norm: bool = False,
                 obs_norm_clip: float = 10.0,
                 obs_norm_passthrough=None,
                 critic_obs_norm_passthrough=None):
        # Bypass NormalTanhMLPPolicy.__init__ (it builds plain trunks over the
        # FULL obs_dim); set up the shared attributes its methods rely on, then
        # build the BPS-split trunks. All distribution / head / PPO logic is
        # inherited unchanged.
        nn.Module.__init__(self)
        self.action_dim    = int(action_dim)
        self.min_std       = float(min_std)
        self.var_scale     = float(var_scale)
        self.use_layernorm = bool(use_layernorm)
        # std parameterization — see NormalTanhMLPPolicy. ``_params`` (inherited)
        # reads ``state_dependent_std`` / ``std_param``, so we only need to set
        # these attrs and size the actor head accordingly here.
        self.state_dependent_std = bool(state_dependent_std)
        self.init_std = None if init_std is None else float(init_std)
        # Async actor-critic: ``critic_obs_dim`` (privileged critic obs width) ≠
        # actor ``obs_dim`` → the critic is a PLAIN MLP over the larger critic obs
        # (its privileged-GT tail has no BPS layout to split-encode). ``None`` →
        # symmetric: the critic shares the actor's BPS-split trunk (unchanged).
        self.critic_obs_dim = int(critic_obs_dim) if critic_obs_dim is not None else int(obs_dim)

        obs_dim = int(obs_dim)
        bps_dim = int(bps_dim)
        if not (0 < bps_dim < obs_dim):
            raise ValueError(
                f"bps_dim ({bps_dim}) must satisfy 0 < bps_dim < obs_dim ({obs_dim}); "
                f"set policy.bps_dim to the env's n_bps (BPS tail width)."
            )
        self.bps_dim  = bps_dim
        self.base_dim = obs_dim - bps_dim

        hidden_dims     = _as_dims(hidden_dim, "hidden_dim")
        bps_enc_dims    = _as_dims(bps_encoder_dim, "bps_encoder_dim")
        # Critic trunk widths — independent of the actor when ``critic_hidden_dim``
        # is given (deeper value net). ``None`` → reuse the actor's (unchanged).
        critic_hidden_dims = (hidden_dims if critic_hidden_dim is None
                              else _as_dims(critic_hidden_dim, "critic_hidden_dim"))
        self.hidden_dims        = hidden_dims
        self.critic_hidden_dims = critic_hidden_dims
        self.bps_latent_dim = int(bps_latent_dim)
        if self.bps_latent_dim <= 0:
            raise ValueError(f"bps_latent_dim must be > 0, got {bps_latent_dim!r}")

        # actor: BPS-split trunk → head. state-dependent → [loc | scale_raw]
        # (2×action_dim); state-independent → loc only + global ``std_param``.
        self.actor_trunk = BPSSplitEncoderTrunk(
            self.base_dim, bps_dim, bps_enc_dims, self.bps_latent_dim,
            hidden_dims, self.use_layernorm,
        )
        self.actor_head = nn.Linear(
            hidden_dims[-1], (2 if self.state_dependent_std else 1) * self.action_dim)
        if not self.state_dependent_std:
            self.std_param = nn.Parameter(NormalTanhMLPPolicy._init_std_param(
                self.action_dim, self.init_std, self.min_std, self.var_scale))
        # critic: symmetric → own BPS-split trunk (same obs as actor); async →
        # plain MLP over the wider privileged critic obs (base+BPS+GT).
        if critic_obs_dim is None:
            self.critic_trunk = BPSSplitEncoderTrunk(
                self.base_dim, bps_dim, bps_enc_dims, self.bps_latent_dim,
                critic_hidden_dims, self.use_layernorm,
            )
        else:
            self.critic_trunk = NormalTanhMLPPolicy._build_trunk(
                self.critic_obs_dim, critic_hidden_dims, self.use_layernorm,
            )
        self.critic_head = nn.Linear(critic_hidden_dims[-1], 1)

        # This __init__ bypasses NormalTanhMLPPolicy.__init__, so it must set up
        # the obs normalizers its inherited ``_norm_obs`` / ``_norm_critic_obs``
        # rely on (otherwise ``act`` raises AttributeError). Mirrors the base
        # class (base_networks.py) — ``obs_norm`` defaults False → None → obs
        # passes through unchanged (the split-encoder normalizes the full obs
        # before splitting off the BPS tail when enabled).
        from grit.model.obs_normalizer import maybe_normalizer
        self.obs_normalizer = maybe_normalizer(
            obs_norm, obs_dim, obs_norm_clip,
            passthrough=obs_norm_passthrough)
        self.critic_obs_normalizer = maybe_normalizer(
            obs_norm and critic_obs_dim is not None,
            self.critic_obs_dim, obs_norm_clip,
            passthrough=critic_obs_norm_passthrough)
