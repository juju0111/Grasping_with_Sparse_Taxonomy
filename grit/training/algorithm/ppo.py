"""PPO update step(s).

Two flavours:

  * :func:`ppo_update`        — single (policy, buffer) pair. Each minibatch
    does one ``zero_grad → backward → step``. Used for independent
    per-tag training (one network per hand) or single-hand training.

  * :func:`ppo_update_multi`  — shared-policy cross-embodiment update.
    The buffer is a :class:`MultiHandRolloutBuffer`; minibatches across
    tags are aligned in lockstep. For each (epoch, mb_idx) we:

        zero_grad                       # once
        for tag in tags:
            loss(view[tag], mb_dict[tag]).backward()   # accumulates
        clip_grad_norm_(shared.parameters())
        step                            # once per macro-minibatch

    so **one** optimizer step covers gradients from every tag, against
    the SAME pre-step policy. This matches the standard PPO update
    semantics (ratio = new_pi(...) / old_pi(...)) — running per-tag
    sequential updates would shift the policy mid-iter and bias every
    later tag's ratio computation.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import torch


def _adapt_lr(optimizer, kl: float, *, desired_kl: Optional[float],
              lr_min: float, lr_max: float, factor: float) -> float:
    """KL-based adaptive lr (rsl_rl style) — **once per iteration**; returns the current lr.

    ``kl > 2×desired_kl`` → ÷factor, ``kl < ½×desired_kl`` → ×factor
    (dead band in between). ``desired_kl`` None/0 → no-op (just reads and returns lr).
    The lr is written directly into ``param_groups``, so there is no separate
    state and it carries over through ``optim_state_dict`` on resume.

    Why once per iteration rather than per minibatch as in rsl_rl: here
    epoch×minibatch is ~256 (rsl_rl ~20), so a ×1.5 step per minibatch would
    swing the lr by orders of magnitude within one iteration. factor=1.5 moves
    KL by ~2.25× (KL ∝ lr²), matching the dead band width (2×) — one adjustment
    usually lands inside the target band.
    """
    lr = float(optimizer.param_groups[0]['lr'])
    if not desired_kl:
        return lr
    if kl > desired_kl * 2.0:
        lr = max(lr_min, lr / factor)
    elif 0.0 < kl < desired_kl * 0.5:
        lr = min(lr_max, lr * factor)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def ppo_update(
    policy,
    optimizer,
    buffer,
    *,
    n_epochs:        int   = 10,
    clip_eps:        float = 0.2,
    vf_coef:         float = 0.5,
    ent_coef:        float = 0.0,
    max_grad_norm:   float = 1.0,
    desired_kl:      Optional[float] = None,
    lr_min:          float = 1.0e-6,
    lr_max:          float = 1.0e-3,
    kl_adapt_factor: float = 1.5,
    adv_norm:        str   = "minibatch",
    # Auxiliary loss hook (DAgger/BC etc.). Default None → identical to plain PPO.
    aux_loss_fn            = None,
    aux_coef:        float = 0.0,
    # Policy-gradient coefficient. During DAgger (β-mixing) the **executed
    # actions are not the policy's own**, so PG is turned off (0) and only
    # BC + critic are trained — left on, the teacher's returns get attributed
    # to actions the policy did not choose, fighting the BC loss head-on.
    pol_coef:        float = 1.0,
) -> dict:
    """Run ``n_epochs`` epochs of minibatch PPO updates over the full
    buffer. Returns mean loss components averaged across all minibatches
    consumed — ``pol/vf/ent/ratio/grad_norm/adv_std`` plus the KL-drift
    diagnostics ``approx_kl`` (k3), ``approx_kl_first/last`` (first/last epoch
    means — last ≫ first suggests too many n_epochs) and ``clip_frac``.

    Hyperparameters (all keyword-only — no module globals):

    * ``n_epochs``: number of passes over the buffer (PPO ``K`` in
      Schulman et al. 2017).
    * ``clip_eps``: PPO clip range ε (typical 0.1–0.2).
    * ``vf_coef``: value-function loss weight (typical 0.5–1.0).
    * ``ent_coef``: entropy bonus weight (typical 0.0–0.01); subtracted
      from the loss to encourage exploration.
    * ``max_grad_norm``: per-step gradient clip applied AFTER backward
      to all of ``policy.parameters()`` (skipped when None / non-positive).
    * ``desired_kl``: None (default) → fixed lr. When set (typically 0.01),
      the lr is adjusted from ``approx_kl`` at the end of each iteration
      (:func:`_adapt_lr`); clamped to ``lr_min`` / ``lr_max``, with
      ``kl_adapt_factor`` as the per-adjustment ratio. Observe it via ``lr``
      in the returned dict.

    Assumes ``buffer`` has already been GAE'd (``compute_gae`` called):
    minibatches expose ``mb.observation``, ``mb.extras['pre_tanh']``,
    ``mb.extras['log_prob']``, ``mb.extras['advantage']``,
    ``mb.extras['return']``. The policy must expose
    ``evaluate(obs, pre_tanh) -> (log_prob, entropy, value)``.
    """
    # ``grad_norm``  : global L2 norm of the policy gradient (PRE-clip) — the
    #                  direct "is the gradient vanishing?" diagnostic.
    # ``adv_std``    : std of the RAW advantage (BEFORE normalization). A true
    #                  zero-gradient trap shows ``grad_norm → 0`` AND
    #                  ``adv_std → 0`` while reward is flat; if ``adv_std → 0``
    #                  but ``grad_norm`` stays up, the normalizer is just
    #                  rescaling tiny advantages.
    # ``approx_kl``  : k3 estimator E[(ratio−1) − log ratio] ≥ 0 — old→new
    #                  policy KL. ``approx_kl_first/last`` are the first/last
    #                  epoch means — last ≫ first signals ``n_epochs`` is too high.
    # ``clip_frac``  : fraction of samples with |ratio−1| > ε (clipping rate).
    losses = dict(pol=0.0, vf=0.0, ent=0.0, ratio=0.0, grad_norm=0.0, adv_std=0.0,
                  approx_kl=0.0, clip_frac=0.0)
    n_mb   = 0
    clip_grad = max_grad_norm is not None and max_grad_norm > 0
    mb_per_epoch = int(buffer.num_minibatch)
    kl_first = kl_last = 0.0
    n_first  = n_last  = 0

    # Advantage normalization mode (``adv_norm``):
    #   * "rollout"   — normalized once, in place, with whole-rollout (T×N)
    #     statistics (brax ``normalize_advantage``).
    #   * "minibatch" — re-normalized (adv−mean)/std per minibatch. Under
    #     heavy-tailed advantages (e.g. streak bonuses or periodic re-sampling
    #     shocks) the per-minibatch re-centering/re-scaling acts as a strong
    #     stabilizer and can clearly outperform "rollout". Chosen per task in yaml.
    if adv_norm not in ("rollout", "minibatch"):
        raise ValueError(f"adv_norm={adv_norm!r} — 'rollout' | 'minibatch'")
    if adv_norm == "minibatch":
        adv_std_raw = float(buffer.advantage.std().item())     # RAW std for logging
    else:
        adv_std_raw = _normalize_buffer_advantage(buffer)

    for i_mb, mb in enumerate(buffer.iter_minibatches(num_epochs=n_epochs, shuffle=True)):
        # Async actor-critic: ``critic_observation`` is present in extras only for
        # asymmetric envs; ``.get`` → None for the symmetric path (identical math).
        new_logp, entropy, value = policy.evaluate(
            mb.observation, mb.extras['pre_tanh'],
            critic_obs=mb.extras.get('critic_observation'))
        log_ratio = new_logp - mb.extras['log_prob']
        ratio = torch.exp(log_ratio)

        adv_n = mb.extras['advantage']            # rollout mode: already normalized
        if adv_norm == "minibatch":
            adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

        pol_loss = -torch.min(
            ratio * adv_n,
            torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_n,
        ).mean()
        vf_loss  = ((value - mb.extras['return']) ** 2).mean()
        ent_loss = entropy.mean()
        loss     = (pol_coef * pol_loss + vf_coef * vf_loss
                    - pol_coef * ent_coef * ent_loss)
        if aux_loss_fn is not None and aux_coef > 0.0:
            aux = aux_loss_fn(mb)
            if aux is not None:
                loss = loss + float(aux_coef) * aux

        with torch.no_grad():
            approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())   # k3, ≥ 0
            clip_frac = float(((ratio - 1.0).abs() > clip_eps).float().mean().item())

        # NaN/Inf guard: a single non-finite obs/reward sample produces a
        # non-finite loss; stepping on it would write NaN into EVERY parameter
        # via the optimizer and permanently kill the run. Skip such minibatches.
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            losses['nonfinite_skips'] = losses.get('nonfinite_skips', 0.0) + 1.0
            continue

        optimizer.zero_grad()
        loss.backward()
        # Always measure the global grad norm (the zero-gradient/collapse
        # diagnostic). ``clip_grad_norm_`` returns the PRE-clip norm; pass
        # max_norm=inf when clipping is off so it only measures (no modification).
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), max_grad_norm if clip_grad else float('inf'))
        # Non-finite grads (e.g. from a non-finite advantage that slipped
        # through) → skip the step instead of poisoning the weights.
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            losses['nonfinite_skips'] = losses.get('nonfinite_skips', 0.0) + 1.0
            continue
        optimizer.step()

        # ── Adaptation (RMA/CSE) substep — opt-in, requested by the policy ──
        # Only policies that expose ``adaptation_loss`` and have
        # ``adaptation_substeps > 0`` (e.g. ``normal_tanh_mlp_cse``) enter here.
        # For other policies the getattr default is 0, so the branch is a no-op
        # and existing tasks are unaffected.
        #
        # Same structure as the original (``b1_gym_learn/ppo_cse/ppo.py``): a
        # **separate optimizer step** from the policy loss, and the loss graph
        # only passes through the estimator, so no gradient reaches the
        # actor/critic parameters. The supervision label is the privileged tail
        # of the raw critic obs stored in the buffer.
        _n_sub = int(getattr(policy, "adaptation_substeps", 0))
        _cobs  = mb.extras.get('critic_observation')
        if _n_sub > 0 and _cobs is not None and hasattr(policy, "adaptation_loss"):
            _opt_a = policy.adaptation_optimizer()
            _priv  = _cobs[:, int(policy.priv_offset):].detach()
            for _ in range(_n_sub):
                a_loss, a_info = policy.adaptation_loss(mb.observation, _priv)
                if not torch.isfinite(a_loss):
                    _opt_a.zero_grad(set_to_none=True)
                    break
                _opt_a.zero_grad()
                a_loss.backward()
                _opt_a.step()
            for _k, _v in a_info.items():                      # per-block diagnostics
                losses[_k] = losses.get(_k, 0.0) + float(_v)

        losses['pol']   += float(pol_loss.item())
        losses['vf']    += float(vf_loss.item())
        losses['ent']   += float(ent_loss.item())
        losses['ratio'] += float(ratio.mean().item())
        losses['grad_norm'] += float(grad_norm)
        losses['adv_std']   += adv_std_raw
        losses['approx_kl'] += approx_kl
        losses['clip_frac'] += clip_frac
        # Separate first/last epoch KL sums — diagnoses whether n_epochs is too high.
        if i_mb // mb_per_epoch == 0:
            kl_first += approx_kl; n_first += 1
        if i_mb // mb_per_epoch == n_epochs - 1:
            kl_last += approx_kl; n_last += 1
        n_mb += 1

    skips = losses.pop('nonfinite_skips', 0.0)   # raw count — not an average
    if n_mb > 0:
        for k in losses:
            losses[k] /= n_mb
    losses['approx_kl_first'] = kl_first / max(n_first, 1)
    losses['approx_kl_last']  = kl_last  / max(n_last, 1)
    # The control signal is the overall mean ``approx_kl``, not the last-epoch
    # mean — it averages 8× more minibatches (lower variance) and matches the
    # meaning of rsl_rl ``desired_kl``. (n_mb==0 → kl=0.0 → inside dead band → no-op.)
    losses['lr'] = _adapt_lr(optimizer, losses['approx_kl'],
                             desired_kl=desired_kl, lr_min=lr_min,
                             lr_max=lr_max, factor=kl_adapt_factor)
    if skips:
        losses['nonfinite_skips'] = skips
    return losses


def _ppo_loss_terms(policy, mb, clip_eps: float, adv_norm: str = "rollout"):
    """Compute ``(pol_loss, vf_loss, ent_loss, ratio, approx_kl, clip_frac)``
    for one minibatch. With ``adv_norm="rollout"``, ``mb.extras['advantage']``
    must already be normalized with whole-rollout statistics
    (:func:`_normalize_buffer_advantage`); with ``"minibatch"`` it is
    re-normalized here with minibatch statistics (see :func:`ppo_update` for
    the mode descriptions).

    Shared between :func:`ppo_update` and :func:`ppo_update_multi` so the
    two paths cannot diverge in their loss math. ``policy`` must expose
    ``evaluate(obs, pre_tanh) -> (log_prob, entropy, value)``.
    """
    new_logp, entropy, value = policy.evaluate(
        mb.observation, mb.extras['pre_tanh'],
        critic_obs=mb.extras.get('critic_observation'))
    log_ratio = new_logp - mb.extras['log_prob']
    ratio = torch.exp(log_ratio)

    adv_n = mb.extras['advantage']
    if adv_norm == "minibatch":
        adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

    pol_loss = -torch.min(
        ratio * adv_n,
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_n,
    ).mean()
    vf_loss  = ((value - mb.extras['return']) ** 2).mean()
    ent_loss = entropy.mean()
    with torch.no_grad():
        approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())   # k3, ≥ 0
        clip_frac = float(((ratio - 1.0).abs() > clip_eps).float().mean().item())
    return pol_loss, vf_loss, ent_loss, ratio, approx_kl, clip_frac


def _normalize_buffer_advantage(buffer) -> float:
    """Normalize the advantage in place with whole-rollout (T×N) statistics and
    return the RAW std (brax ``normalize_advantage`` style — statistically more
    stable than per-minibatch normalization)."""
    adv = buffer.advantage
    if adv is None:
        raise RuntimeError("ppo_update: buffer.compute_gae() must run first (advantage is None)")
    raw_std = float(adv.std().item())
    buffer.advantage = (adv - adv.mean()) / (adv.std() + 1e-8)
    return raw_std


def ppo_update_multi(
    policies:  Mapping[str, object],        # {tag: _TagView (shared) OR nn.Module}
    optimizer: torch.optim.Optimizer,       # ONE optimizer over shared params
    buffer,                                 # MultiHandRolloutBuffer
    *,
    n_epochs:        int   = 10,
    clip_eps:        float = 0.2,
    vf_coef:         float = 0.5,
    ent_coef:        float = 0.0,
    max_grad_norm:   float = 1.0,
    desired_kl:      Optional[float] = None,
    lr_min:          float = 1.0e-6,
    lr_max:          float = 1.0e-3,
    kl_adapt_factor: float = 1.5,
    adv_norm:        str   = "rollout",
) -> Dict[str, dict]:
    """Cross-embodiment PPO update — **gradient accumulation across tags**,
    one ``optimizer.step()`` per macro-minibatch.

    Why this instead of looping ``ppo_update`` per tag?

    Looping ``ppo_update`` per tag does ``n_epochs × num_minibatch`` steps
    PER TAG, sequentially. After tag-A finishes its full update, the
    shared policy has shifted; tag-B's ratio = ``exp(new_logp - old_logp)``
    then compares an A-updated policy against rollout-time logp's,
    inflating ratios and biasing the gradient. With N hands this gives
    ``N × n_epochs × num_minibatch`` total steps and N−1 stale-policy
    sub-updates per iter.

    This function instead iterates the buffer once (yielding one minibatch
    per tag in lockstep), accumulating each tag's loss-gradient onto the
    shared parameters, then taking a single step. Total step count is
    ``n_epochs × num_minibatch`` — identical to single-hand training —
    and every tag sees the same "old" policy on every minibatch.

    The aggregate loss is the **sum** of per-tag losses (each is already a
    mean over its NWORLD_tag × batch_size_tag transitions). If hands have
    different NWORLDs and you want strict equal weighting per transition,
    pre-scale by 1/N_tag in your reward setup — this routine treats every
    tag equally per minibatch by design (one head, one bonus signal).

    Returns ``{tag: {pol, vf, ent, ratio}}`` averaged over the
    minibatches that tag participated in (= ``n_epochs × num_minibatch``).

    ``desired_kl`` (adaptive lr): there is a single optimizer, so the **max**
    over per-tag KLs is the control signal — no hand may leave the trust region.
    """
    tags = list(buffer.tags)
    losses: Dict[str, dict] = {
        tag: dict(pol=0.0, vf=0.0, ent=0.0, ratio=0.0, grad_norm=0.0, adv_std=0.0,
                  approx_kl=0.0, clip_frac=0.0)
        for tag in tags
    }
    counts: Dict[str, int]  = {tag: 0 for tag in tags}
    clip_grad = max_grad_norm is not None and max_grad_norm > 0
    mb_per_epoch = {tag: int(buffer[tag].num_minibatch) for tag in tags}
    kl_first = {tag: [0.0, 0] for tag in tags}     # [sum, n]
    kl_last  = {tag: [0.0, 0] for tag in tags}

    # Per-tag advantage normalization (see ppo_update for the modes; RAW std is for logging).
    if adv_norm not in ("rollout", "minibatch"):
        raise ValueError(f"adv_norm={adv_norm!r} — 'rollout' | 'minibatch'")
    if adv_norm == "minibatch":
        adv_std_raw = {tag: float(buffer[tag].advantage.std().item()) for tag in tags}
    else:
        adv_std_raw = {tag: _normalize_buffer_advantage(buffer[tag]) for tag in tags}

    # ``MultiHandRolloutBuffer.iter_minibatches`` yields ``{tag: Transition}``
    # — one minibatch per tag, lockstep-aligned. ``shuffle=True`` runs an
    # independent randperm per tag, which is fine: the aggregate gradient
    # at each step is the SUM of independent per-tag estimates.
    for i_mb, mb_dict in enumerate(buffer.iter_minibatches(num_epochs=n_epochs, shuffle=True)):
        optimizer.zero_grad()

        # Accumulate each tag's gradient onto the shared parameters.
        # Adapter[tag] + head[tag] receive grad only from this tag; the
        # shared trunk receives grad from EVERY tag's loss → cross-task
        # representation learning lives in this sum.
        for tag, mb in mb_dict.items():
            pol_loss, vf_loss, ent_loss, ratio, approx_kl, clip_frac = _ppo_loss_terms(
                policies[tag], mb, clip_eps, adv_norm=adv_norm,
            )
            loss = pol_loss + vf_coef * vf_loss - ent_coef * ent_loss
            loss.backward()

            losses[tag]['pol']       += float(pol_loss.item())
            losses[tag]['vf']        += float(vf_loss.item())
            losses[tag]['ent']       += float(ent_loss.item())
            losses[tag]['ratio']     += float(ratio.mean().item())
            losses[tag]['approx_kl'] += approx_kl
            losses[tag]['clip_frac'] += clip_frac
            epoch = i_mb // mb_per_epoch[tag]
            if epoch == 0:
                kl_first[tag][0] += approx_kl; kl_first[tag][1] += 1
            if epoch == n_epochs - 1:
                kl_last[tag][0]  += approx_kl; kl_last[tag][1]  += 1
            counts[tag] += 1

        # Global grad norm over the SHARED params (one accumulated gradient from
        # all tags). ``parameters()`` from any view yields the same params; pass
        # max_norm=inf when clipping is off so we only measure.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            next(iter(policies.values())).parameters(),
            max_grad_norm if clip_grad else float('inf'),
        )
        gn = float(grad_norm)
        for tag in mb_dict:               # same (shared) grad_norm per tag
            losses[tag]['grad_norm'] += gn
            losses[tag]['adv_std']   += adv_std_raw[tag]
        # ONE step per macro-minibatch (after every tag has contributed).
        optimizer.step()

    for tag in tags:
        n = max(counts[tag], 1)
        for k in losses[tag]:
            losses[tag][k] /= n
        losses[tag]['approx_kl_first'] = kl_first[tag][0] / max(kl_first[tag][1], 1)
        losses[tag]['approx_kl_last']  = kl_last[tag][0]  / max(kl_last[tag][1], 1)

    # One shared optimizer → adapt on the **max** per-tag KL (a mean would dilute a runaway tag).
    lr = _adapt_lr(optimizer,
                   max((losses[t]['approx_kl'] for t in tags), default=0.0),
                   desired_kl=desired_kl, lr_min=lr_min,
                   lr_max=lr_max, factor=kl_adapt_factor)
    for tag in tags:
        losses[tag]['lr'] = lr         # shared optimizer → same value for every tag
    return losses
