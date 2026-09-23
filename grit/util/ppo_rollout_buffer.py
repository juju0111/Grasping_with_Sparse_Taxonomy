"""On-policy rollout buffer for PPO (brax-style ``Transition``).

Provides:

  * ``Transition`` — NamedTuple matching the brax convention. Fields are
    torch tensors (jax arrays in brax). When produced by the buffer, each
    field has shape ``(T, N, *)`` (full rollout) or ``(B, *)`` (minibatch).

  * ``PPORolloutBuffer`` — pre-allocated GPU buffer for one PPO iteration.
    ``(T, N, *)`` torch tensors live on the same device as the warp env
    (so zero-copy ``wp.to_torch`` views write directly into / out of the
    buffer with no host roundtrip). Supports GAE-λ and minibatch
    iteration with optional shuffling.

  * ``MultiHandRolloutBuffer`` — composite buffer keyed by name tag (e.g.
    ``hand_name``). Wraps **one ``PPORolloutBuffer`` per tag**, so the
    cost of single-hand training is unchanged (the inner buffer is
    untouched). Used for cross-embodiment training where different hands
    can have different ``obs_dim`` / ``action_dim`` / ``NWORLD``.

Typical loop (single-hand):

    buffer = PPORolloutBuffer(
        num_envs=NWORLD, obs_dim=OBS_DIM, action_dim=A,
        batch_size=256, num_minibatch=8, device=torch_device,
        extras_spec=dict(log_prob=(), value=(), pre_tanh=(A,)),
    )

    for _ in range(buffer.unroll_length):
        out = policy.act(prev_obs)
        ...
        buffer.add(observation=prev_obs, action=out['action'],
                   reward=r, discount=GAMMA*(1-d), next_observation=next_obs,
                   log_prob=out['log_prob'], value=out['value'],
                   pre_tanh=out['pre_tanh'])

    buffer.compute_gae(last_value, gamma, lam)
    for mb in buffer.iter_minibatches(num_epochs=4, shuffle=True):
        ppo_update(mb)
    buffer.reset()

Typical loop (multi-hand cross-embodiment):

    buffer = MultiHandRolloutBuffer.from_handlers(
        env.handlers, unroll_length=T, num_minibatch=8,
        device=env.torch_device,
    )
    for _ in range(buffer.unroll_length):
        for tag, h in zip(buffer.tags, env.handlers):
            out = policies[tag].act(prev_obs[tag])
            obs, r, d, _ = h.step(out['action'])
            buffer.add(tag, observation=prev_obs[tag], action=out['action'],
                       reward=r.clone(), discount=GAMMA*(1-d), next_observation=obs.clone(),
                       log_prob=out['log_prob'], value=out['value'], pre_tanh=out['pre_tanh'])

    buffer.compute_gae({tag: ... for tag in buffer.tags}, gamma=GAMMA, lam=LAM)
    for mb_dict in buffer.iter_minibatches(num_epochs=4, shuffle=True):
        for tag, mb in mb_dict.items():
            ppo_update(policies[tag], mb)
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Iterator, List, Mapping, NamedTuple, Optional

import torch


class Transition(NamedTuple):
    """One step (or minibatch) of (s, a, r, γ·(1-d), s', extras).

    brax-compatible signature with torch tensors.

    Shapes:
      * full rollout view  : ``(T, N, *)``  (T=unroll_length, N=num_envs)
      * minibatch view     : ``(B, *)``     where B=batch_size

    ``discount`` follows the brax convention ``γ * (1 - done)`` so GAE can
    consume it directly.
    """
    observation:      torch.Tensor
    action:           torch.Tensor
    reward:           torch.Tensor
    discount:         torch.Tensor
    next_observation: torch.Tensor
    extras:           Dict[str, torch.Tensor]


class PPORolloutBuffer:
    """On-policy rollout buffer for PPO.

    The buffer holds ``unroll_length × num_envs`` transitions per PPO
    iteration. Minibatches are sampled (without replacement within each
    epoch) from this pool; ``batch_size × num_minibatch`` controls how
    many transitions are *consumed* per epoch — it does **not** have to
    equal the buffer size, so e.g. ``unroll_length=10`` lets you collect
    a wide rollout while training on a smaller random subset.

    Args:
        num_envs:      parallel env count (= NWORLD).
        unroll_length: number of control steps to collect per PPO
                       iteration. Sets the buffer size to
                       ``unroll_length × num_envs``. Required (no longer
                       derived from ``batch_size × num_minibatch``).
        obs_dim:       observation dim.
        action_dim:    action dim (post-tanh action is stored).
        batch_size:    minibatch size used by ``iter_minibatches``.
        num_minibatch: minibatches per epoch.
        device:        torch device. Use the same GPU as the warp env so
                       ``wp.to_torch`` zero-copy views match.
        extras_spec:   ``{key: shape_after_TN}`` dict — e.g.
                       ``dict(log_prob=(), value=(), pre_tanh=(A,))`` for PPO.

    Invariants:
        ``total_samples == unroll_length × num_envs``
        ``batch_size × num_minibatch ≤ total_samples``  (subset sampling).

    Backward compat:
        If ``unroll_length`` is left ``None``, the legacy formula
        ``unroll_length = (batch_size × num_minibatch) // num_envs`` is
        used (and the divisibility check is enforced) so existing
        callers continue to work without modification.
    """

    def __init__(
        self,
        num_envs:       int,
        obs_dim:        int,
        action_dim:     int,
        batch_size:     int,
        num_minibatch:  int,
        device:         torch.device,
        extras_spec:    Optional[Dict[str, tuple]] = None,
        unroll_length:  Optional[int] = None,
    ):
        self.num_envs       = int(num_envs)
        self.obs_dim        = int(obs_dim)
        self.action_dim     = int(action_dim)
        self.batch_size     = int(batch_size)
        self.num_minibatch  = int(num_minibatch)
        self.device         = device

        if unroll_length is None:
            # Legacy mode — derive T from batch sizes (back-compat).
            total = int(batch_size) * int(num_minibatch)
            if total % num_envs != 0:
                raise ValueError(
                    f"batch_size*num_minibatch={total} must be divisible by "
                    f"num_envs={num_envs}; either fix the sizes or pass "
                    f"`unroll_length=...` explicitly."
                )
            self.unroll_length = total // num_envs
        else:
            self.unroll_length = int(unroll_length)
            mb_total = int(batch_size) * int(num_minibatch)
            buf_total = self.unroll_length * self.num_envs
            if mb_total > buf_total:
                raise ValueError(
                    f"batch_size*num_minibatch={mb_total} cannot exceed "
                    f"unroll_length*num_envs={buf_total} — the buffer holds "
                    f"only {buf_total} transitions per iteration."
                )

        self.total_samples = self.unroll_length * self.num_envs

        T, N = self.unroll_length, self.num_envs
        self.observation       = torch.zeros((T, N, obs_dim),    device=device)
        self.action            = torch.zeros((T, N, action_dim), device=device)
        self.reward            = torch.zeros((T, N),             device=device)
        self.discount          = torch.zeros((T, N),             device=device)
        self.next_observation  = torch.zeros((T, N, obs_dim),    device=device)
        self.extras: Dict[str, torch.Tensor] = {}
        if extras_spec is not None:
            for k, shape in extras_spec.items():
                self.extras[k] = torch.zeros((T, N) + tuple(shape), device=device)

        self.advantage: Optional[torch.Tensor] = None   # (T, N) — set by compute_gae
        self.return_:   Optional[torch.Tensor] = None   # (T, N)
        self._t = 0

    # ──────────────────────────────────────────────────────────────────
    # Basic state
    # ──────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the write cursor and clear cached GAE results."""
        self._t = 0
        self.advantage = None
        self.return_   = None

    @property
    def is_full(self) -> bool:
        return self._t >= self.unroll_length

    # ──────────────────────────────────────────────────────────────────
    # Add a (parallel) transition
    # ──────────────────────────────────────────────────────────────────

    def add(
        self,
        observation:       torch.Tensor,
        action:            torch.Tensor,
        reward:            torch.Tensor,
        discount:          torch.Tensor,
        next_observation:  torch.Tensor,
        **extras:          torch.Tensor,
    ) -> None:
        """Append one parallel transition (covers all N envs).

        Each tensor has leading dim N. Unknown extras are silently dropped
        (only registered keys from ``extras_spec`` are stored).
        """
        if self.is_full:
            raise RuntimeError("buffer full; call reset() first")
        t = self._t
        self.observation[t]     .copy_(observation)
        self.action[t]          .copy_(action)
        self.reward[t]          .copy_(reward)
        self.discount[t]        .copy_(discount)
        self.next_observation[t].copy_(next_observation)
        for k, v in extras.items():
            if k in self.extras:
                self.extras[k][t].copy_(v)
        self._t += 1

    # ──────────────────────────────────────────────────────────────────
    # Brax-style Transition view
    # ──────────────────────────────────────────────────────────────────

    def get_transition(self) -> Transition:
        """Full rollout as a brax-style ``Transition`` (shapes ``(T, N, *)``)."""
        T = self._t
        return Transition(
            observation       = self.observation[:T],
            action            = self.action[:T],
            reward            = self.reward[:T],
            discount          = self.discount[:T],
            next_observation  = self.next_observation[:T],
            extras            = {k: v[:T] for k, v in self.extras.items()},
        )

    # ──────────────────────────────────────────────────────────────────
    # Generalized Advantage Estimation
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma:      float = 0.99,
        lam:        float = 0.95,
    ):
        """brax-faithful GAE-λ.

        Mirrors ``brax.training.agents.ppo.losses.compute_gae``: separates
        truncation (bootstrap, but cut the trajectory at that step) from
        termination (no bootstrap), produces a value target ``vs`` and an
        advantage that is *recomputed* from ``vs`` rather than the raw
        recursion accumulator.

        Inputs (read from ``self``):
          * ``self.reward``                — ``(T, N)`` rewards.
          * ``self.discount``              — ``(T, N)``: ``γ · (1 - done_any)``.
            Legacy γ-scaled convention preserved (so notebook 62 keeps working).
            ``done_any = 1`` if the episode ended this step for *any* reason.
          * ``self.extras['value']``       — ``(T, N)`` value estimates.
          * ``self.extras['truncation']``  — ``(T, N)`` ∈ {0,1}; 1 if the step
            was truncated (e.g. timeout). **Optional**: if absent, defaults to
            zeros (all dones treated as real terminations — equivalent to the
            previous behavior).

        Args:
            last_value: ``(N,)`` bootstrap value at step T  (i.e. ``V(obs_T)``).
            gamma:      scalar discount γ. Must match the γ used to populate
                        ``self.discount``.
            lam:        GAE λ.

        Recursion (per env):
            done_any        = (discount == 0)
            termination     = done_any · (1 - truncation)
            factor          = γ · (1 - termination)        # γ if alive or truncated, 0 if terminated
            truncation_mask = 1 - truncation
            deltas[t]       = (r[t] + factor[t]·V_{t+1} - V[t]) · truncation_mask[t]
            acc[t]          = deltas[t] + factor[t]·truncation_mask[t]·λ·acc[t+1]
            vs[t]           = acc[t] + V[t]                # value target
            advantages[t]   = (r[t] + factor[t]·vs_{t+1} - V[t]) · truncation_mask[t]

        Stores:
            ``self.advantage = advantages``  ``(T, N)``
            ``self.return_   = vs``          ``(T, N)``   — critic regression target.

        Returns:
            ``(advantage, return_)``  (kept in this order for back-compat).

        Requires ``'value'`` in ``extras_spec``.
        """
        if 'value' not in self.extras:
            raise KeyError("compute_gae requires 'value' in extras (shape (T,N))")

        T, N        = self.unroll_length, self.num_envs
        values      = self.extras['value']                  # (T, N)
        rewards     = self.reward                           # (T, N)
        discount_tn = self.discount                         # (T, N)  γ·(1-done_any)

        truncation = self.extras.get('truncation')          # (T, N) or None
        if truncation is None:
            truncation = torch.zeros_like(rewards)
        else:
            truncation = truncation.to(rewards.dtype)

        # Recover the binary done-any mask from the γ-scaled discount.
        done_any        = (discount_tn == 0).to(rewards.dtype)
        termination     = done_any * (1.0 - truncation)
        factor          = gamma   * (1.0 - termination)     # (T, N)
        truncation_mask = 1.0 - truncation                  # (T, N)

        # V_{t+1}: shift values by one timestep, last slot = bootstrap.
        values_next = torch.cat(
            [values[1:], last_value.to(values.dtype).unsqueeze(0)], dim=0,
        )

        deltas = (rewards + factor * values_next - values) * truncation_mask

        # Backward recursion (brax: ``acc == vs - V``).
        acc      = torch.zeros((T, N), device=self.device, dtype=rewards.dtype)
        acc_next = torch.zeros((N,),    device=self.device, dtype=rewards.dtype)
        for t in reversed(range(T)):
            acc_next = deltas[t] + factor[t] * truncation_mask[t] * lam * acc_next
            acc[t]   = acc_next

        vs = acc + values                                   # value target

        # Recompute advantages with ``vs_{t+1}`` as bootstrap (brax form).
        vs_next    = torch.cat(
            [vs[1:], last_value.to(values.dtype).unsqueeze(0)], dim=0,
        )
        advantages = (rewards + factor * vs_next - values) * truncation_mask

        self.advantage = advantages
        self.return_   = vs
        return advantages, vs

    # ──────────────────────────────────────────────────────────────────
    # Minibatch iterator
    # ──────────────────────────────────────────────────────────────────

    def iter_minibatches(
        self,
        num_epochs: int  = 1,
        shuffle:    bool = True,
    ) -> Iterator[Transition]:
        """Yields exactly ``num_epochs * num_minibatch`` Transitions of size
        ``batch_size``.

        After ``compute_gae`` has been called, every yielded Transition's
        ``extras`` carries ``'advantage'`` and ``'return'`` (shape ``(B,)``).
        """
        T, N = self.unroll_length, self.num_envs
        flat_obs      = self.observation.reshape(T * N, -1)
        flat_act      = self.action.reshape(T * N, -1)
        flat_rew      = self.reward.reshape(T * N)
        flat_disc     = self.discount.reshape(T * N)
        flat_next_obs = self.next_observation.reshape(T * N, -1)
        flat_extras   = {k: v.reshape(T * N, *v.shape[2:])
                         for k, v in self.extras.items()}
        if self.advantage is not None:
            flat_extras['advantage'] = self.advantage.reshape(T * N)
            flat_extras['return']    = self.return_.reshape(T * N)

        total = T * N
        # ── sample validity masking ──────────────────────────────────────
        # Tasks that register ``sample_mask`` train only on transitions with
        # mask>0.5. GAE has already been computed over the **full trajectory**
        # (bootstrapping intact), so filtering here only removes gradient
        # contributions: segments without a demonstration do not pull on the
        # policy. If fewer valid samples than batch_size remain, that pool is
        # used as-is (smaller minibatches); if it is empty entirely, masking is
        # abandoned so training does not stall.
        pool = None
        if 'sample_mask' in self.extras:
            valid = (self.extras['sample_mask'].reshape(total) > 0.5).nonzero(as_tuple=True)[0]
            if valid.numel() >= max(1, self.batch_size // 8):
                pool = valid
            else:
                # count fallbacks and report with exponentially decreasing frequency
                n = int(getattr(self, "_mask_fallback_n", 0)) + 1
                self._mask_fallback_n = n
                if n & (n - 1) == 0:      # print only on occurrences 1,2,4,8,...
                    print(f"[buffer] sample_mask valid {valid.numel()}/{total} "
                          f"-> proceeding without masking (cumulative {n})")
        for _ in range(num_epochs):
            if pool is not None:
                perm = torch.randperm(pool.numel(), device=self.device) if shuffle \
                    else torch.arange(pool.numel(), device=self.device)
                idx = pool[perm]
            else:
                idx = (torch.randperm(total, device=self.device) if shuffle
                       else torch.arange(total, device=self.device))
            n_idx = idx.numel()
            for k in range(self.num_minibatch):
                lo = (k * self.batch_size) % max(1, n_idx)
                mb = idx[lo: lo + self.batch_size]
                if mb.numel() == 0:
                    mb = idx[:self.batch_size]
                yield Transition(
                    observation       = flat_obs[mb],
                    action            = flat_act[mb],
                    reward            = flat_rew[mb],
                    discount          = flat_disc[mb],
                    next_observation  = flat_next_obs[mb],
                    extras            = {k_: v[mb] for k_, v in flat_extras.items()},
                )

    # ──────────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        T, N = self.unroll_length, self.num_envs
        ext  = list(self.extras.keys())
        return (f"PPORolloutBuffer(T={T}, N={N}, obs_dim={self.obs_dim}, "
                f"action_dim={self.action_dim}, "
                f"batch_size={self.batch_size}×num_minibatch={self.num_minibatch}, "
                f"extras={ext}, device={self.device})")


# ──────────────────────────────────────────────────────────────────────────
# MultiHandRolloutBuffer — composite buffer for cross-embodiment training
# ──────────────────────────────────────────────────────────────────────────


def _default_extras_spec_fn(handler: Any) -> Dict[str, tuple]:
    """Default per-handler PPO extras: log_prob, value, pre_tanh, truncation.

    Async (asymmetric) actor-critic: when the handler exposes a critic obs of a
    DIFFERENT width than the actor obs (``critic_obs_dim != obs_dim``), also
    register a ``critic_observation`` extra so the privileged critic obs is
    stored per step (consumed by ``policy.evaluate(..., critic_obs=...)`` in the
    PPO update). Symmetric handlers (the default) register nothing extra → zero
    allocation, byte-identical behaviour.
    """
    # Sample validity mask: registered only when the task can express "do not
    # train on this step" (LFOI: worlds where the leader failed to lift = no
    # demonstration). Handlers that do not declare it have no key at all, so
    # behaviour is byte-identical to before.
    _mask = dict(sample_mask=()) if bool(getattr(handler, "provides_sample_mask", False)) else {}
    # DAgger BC label (teacher action); registered only when the task declares it
    if bool(getattr(handler, "provides_bc_action", False)):
        _mask["bc_action"] = (int(handler.action_dim),)
    spec = dict(
        **_mask,
        log_prob   = (),
        value      = (),
        pre_tanh   = (int(handler.action_dim),),
        truncation = (),
    )
    critic_dim = int(getattr(handler, "critic_obs_dim", handler.obs_dim))
    if critic_dim != int(handler.obs_dim):
        spec["critic_observation"] = (critic_dim,)
    return spec


def _default_tag_fn(handler: Any, idx: int, used: set) -> str:
    """Default tag = ``hand_util.hand_name``; disambiguate duplicates with ``#i``."""
    name = getattr(handler.hand_util, 'hand_name', None) or f"handler_{idx}"
    if name not in used:
        return name
    return f"{name}#{idx}"


def derive_handler_tags(env) -> List[str]:
    """Reproduce :func:`MultiHandRolloutBuffer._default_tag_fn` so callers know
    the canonical tag ordering **before** the buffer is built.

    Useful when the SAVE_DIR / wandb run name / policies dict / optimizers
    dict need to be keyed by tag before any rollout has happened. Used by
    ``scripts/train.py`` and inference scripts.
    """
    tags: List[str] = []
    used: set = set()
    for i, h in enumerate(env.handlers):
        tag = _default_tag_fn(h, i, used)
        # ``_default_tag_fn`` uses ``idx`` for #-suffixes, but the legacy
        # train.py form used a 1-based local counter. Preserve the same
        # disambiguation shape ("name#1", "name#2", …) here.
        base = getattr(h.hand_util, "hand_name", None) or f"handler_{i}"
        if tag != base:
            k = 1
            tag = f"{base}#{k}"
            while tag in used:
                k += 1
                tag = f"{base}#{k}"
        used.add(tag)
        tags.append(tag)
    return tags


class MultiHandRolloutBuffer:
    """Composite rollout buffer keyed by name tag.

    Wraps **one ``PPORolloutBuffer`` per tag** — the inner buffers are
    untouched, so single-hand performance is unaffected. The wrapper only
    adds dispatch (``add(tag, ...)``) and grouping (``compute_gae`` /
    ``iter_minibatches`` over all tags).

    Designed for cross-embodiment training where each tag (typically a
    hand name) may have its own ``obs_dim`` / ``action_dim`` / ``NWORLD``.
    All tags must share the same ``unroll_length`` so the rollout loop
    advances in lockstep.

    Args:
        specs: ``{tag: kwargs}`` — kwargs forwarded to ``PPORolloutBuffer``.

    Example:
        buffer = MultiHandRolloutBuffer(specs={
            "tesollo":  dict(num_envs=8, obs_dim=O1, action_dim=A1, ...),
            "robotis":  dict(num_envs=8, obs_dim=O2, action_dim=A2, ...),
        })
    """

    def __init__(self, specs: Mapping[str, Mapping[str, Any]]):
        if len(specs) == 0:
            raise ValueError("specs must contain at least one tag")
        self.buffers: Dict[str, PPORolloutBuffer] = {
            tag: PPORolloutBuffer(**dict(spec)) for tag, spec in specs.items()
        }
        self.tags: List[str] = list(self.buffers.keys())

        ulens = {tag: b.unroll_length for tag, b in self.buffers.items()}
        if len(set(ulens.values())) > 1:
            raise ValueError(
                f"per-tag unroll_length must agree (so the outer rollout "
                f"loop is well-defined); got {ulens}"
            )
        self.unroll_length: int = next(iter(ulens.values()))
        # Sanity: device must agree across tags (all buffers on one GPU).
        devs = {tag: b.device for tag, b in self.buffers.items()}
        if len(set(str(d) for d in devs.values())) > 1:
            raise ValueError(f"per-tag device must agree; got {devs}")
        self.device = next(iter(devs.values()))

    # ── Construction helpers ─────────────────────────────────────────────

    @classmethod
    def from_handlers(
        cls,
        handlers: List[Any],
        *,
        unroll_length: int,
        num_minibatch: Optional[int] = None,
        device: torch.device,
        batch_size: Optional[int] = None,
        extras_spec_fn: Optional[Callable[[Any], Dict[str, tuple]]] = None,
        tag_fn: Optional[Callable[[Any, int, set], str]] = None,
    ) -> "MultiHandRolloutBuffer":
        """Build from a list of ``SubEnvHandler`` (or any object exposing
        ``NWORLD / obs_dim / action_dim / hand_util.hand_name``).

        Exactly which of ``batch_size`` / ``num_minibatch`` you pin selects the
        sizing mode (``buf_total = NWORLD_tag × unroll_length``):

        * **num_minibatch pinned, batch_size=None** — FULL COVERAGE, dynamic
          ``batch_size = buf_total // num_minibatch``. The whole rollout is
          consumed every epoch, but ``batch_size`` GROWS with the rollout
          (more envs / longer unroll → bigger minibatches).
        * **batch_size pinned, num_minibatch=None** — FULL COVERAGE, dynamic
          ``num_minibatch = buf_total // batch_size``. The minibatch SIZE stays
          fixed (bounded gradient-step cost) and the NUMBER of minibatches grows
          with the rollout — usually the better default for scaling ``nworld``.
        * **both pinned** — subset sampling: only ``batch_size × num_minibatch``
          transitions are drawn per epoch (must be ≤ ``buf_total``).
        * **both None** — error.

        Args:
            handlers:        list of handler objects.
            unroll_length:   shared T (rollout length) across all tags.
            num_minibatch:   shared minibatch count, OR ``None`` to derive it
                             per tag from ``batch_size`` (dynamic).
            device:          torch device.
            batch_size:      explicit minibatch size, OR ``None`` to derive it
                             per tag from ``num_minibatch`` (dynamic).
            extras_spec_fn:  per-handler ``extras_spec`` factory (defaults
                             to PPO's ``log_prob/value/pre_tanh``).
            tag_fn:          per-handler tag function (defaults to
                             ``handler.hand_util.hand_name``).
        """
        extras_spec_fn = extras_spec_fn or _default_extras_spec_fn
        tag_fn         = tag_fn         or _default_tag_fn

        if batch_size is None and num_minibatch is None:
            raise ValueError(
                "from_handlers: set at least one of batch_size / num_minibatch "
                "(both None — cannot size the minibatches).")

        used: set = set()
        specs: Dict[str, Dict[str, Any]] = {}
        derived_nmb: Dict[str, int] = {}     # per-tag minibatch count (for lockstep check)
        for i, h in enumerate(handlers):
            tag = tag_fn(h, i, used)
            if tag in specs:
                raise ValueError(f"duplicate tag '{tag}' — provide a custom tag_fn")
            used.add(tag)

            buf_total = int(h.NWORLD) * int(unroll_length)
            if num_minibatch is None:
                # batch_size pinned → derive num_minibatch (bounded batch, full
                # coverage). num_minibatch GROWS with the rollout.
                bs = int(batch_size)        # type: ignore[arg-type]
                if bs <= 0:
                    raise ValueError(f"tag '{tag}': batch_size must be > 0, got {bs}.")
                if bs > buf_total:
                    raise ValueError(
                        f"tag '{tag}': batch_size={bs} exceeds NWORLD*unroll_length="
                        f"{buf_total} — lower batch_size or raise nworld/unroll_length.")
                tag_num_minibatch = buf_total // bs
                tag_batch_size    = bs
                if buf_total % bs != 0:
                    warnings.warn(
                        f"tag '{tag}': NWORLD*T={buf_total} not divisible by "
                        f"batch_size={bs}; using num_minibatch={tag_num_minibatch} "
                        f"→ {tag_num_minibatch * bs}/{buf_total} consumed/epoch "
                        f"(remainder reshuffled away each epoch).")
            elif batch_size is None:
                # num_minibatch pinned → derive batch_size (full coverage).
                if buf_total % num_minibatch != 0:
                    raise ValueError(
                        f"tag '{tag}': NWORLD*T={buf_total} not divisible by "
                        f"num_minibatch={num_minibatch} — pin batch_size instead, "
                        f"or use an nworld/unroll that divides evenly."
                    )
                tag_batch_size    = buf_total // int(num_minibatch)
                tag_num_minibatch = int(num_minibatch)
            else:
                # Both pinned → explicit subset sampling. Buffer holds
                # `buf_total` transitions per iteration; only
                # `batch_size × num_minibatch` are consumed per epoch.
                tag_batch_size    = int(batch_size)
                tag_num_minibatch = int(num_minibatch)
                if tag_batch_size * tag_num_minibatch > buf_total:
                    raise ValueError(
                        f"tag '{tag}': batch_size*num_minibatch="
                        f"{tag_batch_size * tag_num_minibatch} cannot exceed "
                        f"NWORLD*unroll_length={buf_total}."
                    )

            derived_nmb[tag] = tag_num_minibatch
            specs[tag] = dict(
                num_envs       = int(h.NWORLD),
                unroll_length  = int(unroll_length),
                obs_dim        = int(h.obs_dim),
                action_dim     = int(h.action_dim),
                batch_size     = tag_batch_size,
                num_minibatch  = tag_num_minibatch,
                device         = device,
                extras_spec    = extras_spec_fn(h),
            )

        # ``iter_minibatches`` aligns tags in LOCKSTEP (zip), so every tag must
        # produce the SAME number of minibatches per epoch. The dynamic
        # (num_minibatch=None) path can derive different counts when tags have
        # different NWORLD — reject that with a clear message.
        if len(set(derived_nmb.values())) > 1:
            raise ValueError(
                f"per-tag num_minibatch disagree {derived_nmb} — multi-hand "
                f"lockstep updates need equal minibatch counts. Pin "
                f"`num_minibatch` explicitly, or use equal NWORLD across tags."
            )
        return cls(specs)

    # ── Tag-keyed access ─────────────────────────────────────────────────

    def __getitem__(self, tag: str) -> PPORolloutBuffer:
        return self.buffers[tag]

    def __contains__(self, tag: str) -> bool:
        return tag in self.buffers

    def __len__(self) -> int:
        return len(self.buffers)

    def keys(self):
        return self.buffers.keys()

    def values(self):
        return self.buffers.values()

    def items(self):
        return self.buffers.items()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def reset(self) -> None:
        for b in self.buffers.values():
            b.reset()

    @property
    def is_full(self) -> bool:
        return all(b.is_full for b in self.buffers.values())

    # ── Add (per tag) ────────────────────────────────────────────────────

    def add(self, tag: str, **kwargs) -> None:
        """Append a transition to the buffer keyed by ``tag``."""
        if tag not in self.buffers:
            raise KeyError(f"unknown tag '{tag}'. registered: {self.tags}")
        self.buffers[tag].add(**kwargs)

    def add_all(self, per_tag_data: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        """Bulk add: ``{tag: kwargs_for_buffer.add}`` for every tag at once."""
        for tag, data in per_tag_data.items():
            self.add(tag, **data)

    # ── GAE (per tag) ────────────────────────────────────────────────────
    # With multiple hands, rollout data is computed per hand.
    def compute_gae(
        self,
        last_values: Mapping[str, torch.Tensor],
        gamma: float = 0.99,
        lam:   float = 0.95,
    ) -> Dict[str, tuple]:
        """Run GAE on every tag. ``last_values[tag]`` shape is ``(NWORLD_tag,)``.

        Returns ``{tag: (advantage, return_)}`` for inspection.
        """
        out: Dict[str, tuple] = {}
        for tag, b in self.buffers.items():
            if tag not in last_values:
                raise KeyError(f"last_values missing tag '{tag}'")
            out[tag] = b.compute_gae(last_values[tag], gamma=gamma, lam=lam)
        return out

    # ── Minibatch iter — yields {tag: Transition} ────────────────────────

    def iter_minibatches(
        self,
        num_epochs: int  = 1,
        shuffle:    bool = True,
    ) -> Iterator[Dict[str, Transition]]:
        """Yields ``num_epochs * num_minibatch`` dicts ``{tag: Transition}``.

        Each tag's :meth:`PPORolloutBuffer.iter_minibatches` runs
        independently; the wrapper aligns them via ``zip``. All tags must
        share the same ``num_minibatch`` (and produce the same number of
        minibatches per epoch) for the alignment to be meaningful.
        """
        gens = {tag: iter(b.iter_minibatches(num_epochs=num_epochs, shuffle=shuffle))
                for tag, b in self.buffers.items()}
        while True:
            mb_dict: Dict[str, Transition] = {}
            for tag, g in gens.items():
                try:
                    mb_dict[tag] = next(g)
                except StopIteration:
                    return
            yield mb_dict

    # ── Introspection ────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Per-tag spec for logging."""
        out = {}
        for tag, b in self.buffers.items():
            out[tag] = dict(
                T=b.unroll_length, N=b.num_envs,
                obs_dim=b.obs_dim, action_dim=b.action_dim,
                batch_size=b.batch_size, num_minibatch=b.num_minibatch,
                extras=list(b.extras.keys()),
            )
        return out

    def __repr__(self) -> str:
        parts = ", ".join(f"{t}(N={b.num_envs}, obs={b.obs_dim}, act={b.action_dim})"
                          for t, b in self.buffers.items())
        return (f"MultiHandRolloutBuffer(T={self.unroll_length}, "
                f"tags={len(self.buffers)}, "
                f"device={self.device}, [{parts}])")
