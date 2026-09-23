"""Evaluation helpers — deterministic rollout + per-term return stats.

Use :func:`run_eval` for periodic eval during training: it disables
per-world auto-reset + success done-promotion (via ``h.eval_context()``)
so each world plays out a full ``eval_length`` window, and returns a
rich dict of stats (mean return, success rate, first-failure histogram,
per-term episode returns, throughput).
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Optional

import numpy as np
import torch
import warp as wp

from grit.training.rl_env_base   import SubEnvHandler
from grit.training.reward_tracker import RewardTermReturnTracker


def run_eval(
    policy,
    h,
    eval_length:    Optional[int] = None,
    deterministic:  bool          = True,
) -> dict:
    """Deterministic rollout for evaluation. Records:

      * **per-world cumulative return** over the full eval window
        (no resets) — the absolute-reward signal for comparing
        checkpoints. Reported as ``mean_return`` / ``min_return`` /
        ``max_return``.
      * **first-failure stats** — for each world, the FIRST done event
        (obj_z / timeout / wrist_up) is recorded; subsequent done flags
        are ignored since no reset means they keep firing.
      * **success rate** — worlds whose ``bonus_streak`` at the FINAL
        step of the eval window is ≥ ``SUCCESS_STREAK_MIN`` / NWORLD.
        i.e. the policy must be holding the bonus continuously for the
        last ``SUCCESS_STREAK_MIN`` steps — losing tracking mid-window
        and recovering later does NOT count as success.
      * **per-term episode-return statistics** — sum of each metric
        term over the full eval window per world (one "episode" per
        world). Cross-env mean / variance / std / count.
      * ``eval_sps`` — control steps / sec measured around the rollout.

    Eval rollout runs inside ``h.eval_context()`` so:
      * ``per_world_reset_if_done()`` is a no-op (no auto-reset).
      * Success-based done promotion is suppressed — meeting the bonus
        condition does NOT terminate the episode, so the policy must
        MAINTAIN tracking to keep earning reward over the full window.

    Other done sources (obj_z drop, timeout, wrist_up) still flag
    ``done_mask`` for diagnostics but do not reset the world.

    ``eval_length=None`` (default) → ``h.MAX_EPISODE_STEPS``, i.e. one full
    episode window — the same value ``scripts/train.py`` uses when
    ``cfg.training.eval_length`` is null. Never pass a window shorter than
    ``SUCCESS_STREAK_MIN``: the streak cannot fill, so ``success_rate`` /
    ``success_rate_strict`` come back 0 and read as a broken policy rather
    than as a too-short window (grasping: STREAK_MIN=80 vs the old hard-coded
    default of 64 — that default silently reported 0% and is why this is now
    handler-derived). A shorter window warns on stdout.

    Reproducibility: with ``EVAL_RESET_SEED >= 0`` the reset draw is exact
    (qpos / ctrl / cond fields / taxonomy come back bit-identical), but the
    rollout is NOT bit-reproducible — mujoco-warp's contact solver is
    nondeterministic at ~1e-6 in the contact-impulse observations, and 160
    steps of contact-rich dynamics amplify that. Measured on a grasping teacher
    @ NWORLD=900: ~3.7% of worlds flip their success bit between identical
    runs, giving a 1-sigma of ~0.17 pp on the aggregate success rate. Treat
    per-cell rates from small samples accordingly.
    """
    was_training = policy.training
    policy.eval()

    eval_length = (int(h.MAX_EPISODE_STEPS) if eval_length is None
                   else int(eval_length))
    SUCCESS_STREAK_MIN = int(getattr(h, 'SUCCESS_STREAK_MIN', 30))
    if eval_length < SUCCESS_STREAK_MIN:
        print(f'[run_eval] ⚠ eval_length={eval_length} < SUCCESS_STREAK_MIN='
              f'{SUCCESS_STREAK_MIN} → success_rate / success_rate_strict will be '
              f'0 by construction (window too short to fill the streak).')
    rew_per_step: list[float] = []

    # Per-world accumulators (GPU-resident; one sync at end).
    cumulative_rew    = torch.zeros(h.NWORLD, device=h.torch_device)
    first_done_step   = torch.full((h.NWORLD,), -1, device=h.torch_device, dtype=torch.long)
    first_done_reason = torch.zeros(h.NWORLD, device=h.torch_device, dtype=torch.long)
    has_done          = torch.zeros(h.NWORLD, device=h.torch_device, dtype=torch.bool)

    # Per-term episode-return statistics tracker — one "episode" per world,
    # spanning the full eval window (no resets). Stats are cross-env over
    # all NWORLD episodes.
    term_tracker = RewardTermReturnTracker(h.metrics, h.NWORLD, h.torch_device)
    term_tracker.reset()
    # Eval has NO per-world reset → ONE episode per world spanning the full
    # window. Feed ``update`` a zero done-mask so it only ACCUMULATES (never
    # flushes mid-window on a re-firing failure), then ``flush_all`` once at the
    # end so each world contributes exactly one full-window episode-return.
    _eval_no_done = torch.zeros(h.NWORLD, device=h.torch_device, dtype=torch.long)

    # ── Eval rollout (no auto-reset; success doesn't end episodes) ──────
    wp.synchronize_device()
    t_eval_start = time.perf_counter()

    # Reproducible eval-reset protocol (opt-in; e.g. grasping EVAL_RESET_SEED): the reset
    # RNG is seeded with a fixed value so every eval sees the same spawn/target/taxonomy
    # draws — removes reset-draw variance from the eval metrics. The training RNG stream
    # is restored after the rollout (equivalent of use_eval_protocol in JAX).
    _reset_rng_saved = (h.seed_eval_reset()
                        if hasattr(h, "seed_eval_reset") else None)

    with torch.no_grad(), h.eval_context():
        obs = h.reset()
        prev_obs = obs.clone()
        for t in range(eval_length):
            # Async actor-critic: ``act`` always computes the value head, so an
            # asymmetric policy's critic trunk needs the (privileged) critic obs
            # even though eval ignores ``value``. ``critic_obs_torch`` defaults to
            # ``obs_torch`` for symmetric handlers, so this is safe either way.
            out = policy.act(prev_obs, deterministic=deterministic,
                             critic_obs=h.critic_obs_torch)
            next_obs, rew, done, info = h.step(out['action'])

            # Per-world reward accumulation (no resets → spans full window).
            cumulative_rew.add_(rew)
            rew_per_step.append(float(rew.mean().item()))

            done_mask   = info['done_mask']
            done_reason = info['done_reason']

            # First-failure tracking (sync-free): record (step, reason) the
            # first time each world flags done. Subsequent done flags are
            # ignored because no reset means they may keep firing.
            new_done = done_mask.bool() & (~has_done)
            first_done_step   = torch.where(new_done, torch.full_like(first_done_step, t), first_done_step)
            first_done_reason = torch.where(new_done, done_reason, first_done_reason)
            has_done = has_done | new_done

            # Accumulate only (zero done-mask) — no mid-window flush; see above.
            term_tracker.update(h.metrics, _eval_no_done)
            # NOTE: no h.per_world_reset_if_done() call — eval_mode disables it.
            prev_obs.copy_(next_obs)

        # One flush at the window end → n == NWORLD (one episode per world).
        term_tracker.flush_all()

    # Restore the training RNG stream (only when the eval-seed protocol was used).
    if _reset_rng_saved is not None:
        h.restore_reset_rng(_reset_rng_saved)

    # Success criterion: read ``bonus_streak`` AT THE END of the eval window.
    # Eval_context disables done promotion + auto-reset, so this counter
    # reflects the number of consecutive control steps for which the policy
    # has been holding the bonus condition right up to the final step. A
    # world only counts as success if that streak is >= SUCCESS_STREAK_MIN
    # — losing tracking mid-window and recovering later does NOT qualify
    # (the streak gets zeroed whenever bonus_active drops to 0).
    final_bonus_streak = h.bonus_streak_torch.clone()

    # Optional task-defined extra success masks, evaluated at the FINAL step
    # (e.g. grasping's JAX-convention ``lift`` = obj lifted > threshold from
    # spawn). Reported as ``success_rate_<name>`` for cross-codebase comparison;
    # the streak-based ``success_rate`` above is kept unchanged.
    extra_masks: dict = {}
    if hasattr(h, "eval_success_masks"):
        extra_masks = {k: v.detach().clone()
                       for k, v in h.eval_success_masks().items()}

    # Optional task-defined SEMANTIC metrics at the final eval step (e.g.
    # grasping's taxonomy contact IoU/precision/recall/F1 + face-angle error).
    # Reported as ``sem_<name>`` scalars alongside the success metrics.
    sem_metrics: dict = {}
    if hasattr(h, "eval_semantic_metrics"):
        sem_metrics = {f"sem_{k}": float(v)
                       for k, v in h.eval_semantic_metrics().items()}

    wp.synchronize_device()
    t_eval_end   = time.perf_counter()
    eval_elapsed = max(t_eval_end - t_eval_start, 1e-9)
    eval_steps   = int(eval_length) * int(h.NWORLD)
    eval_sps     = eval_steps / eval_elapsed

    if was_training:
        policy.train()

    # One CPU sync — pulls per-term {mean, std, n} off the GPU.
    term_stats = term_tracker.stats()

    # Per-world reduction (single sync block).
    cum_rew_np      = cumulative_rew.cpu().numpy()
    final_streak_np = final_bonus_streak.cpu().numpy()
    fd_step_np      = first_done_step.cpu().numpy()
    fd_reason_np    = first_done_reason.cpu().numpy()
    has_done_np     = has_done.cpu().numpy()

    n_world      = int(h.NWORLD)
    n_success    = int((final_streak_np >= SUCCESS_STREAK_MIN).sum())
    success_rate = n_success / n_world

    # Extra (task-defined) success metrics → ``success_rate_<name>`` /
    # ``n_success_<name>``. ``success_rate_strict`` aliases the streak-based one.
    extra_success: dict = {"success_rate_strict": success_rate}
    for name, mask in extra_masks.items():
        n_hit = int(mask.bool().cpu().numpy().sum())
        extra_success[f"n_success_{name}"]    = n_hit
        extra_success[f"success_rate_{name}"] = n_hit / n_world

    # First-failure reason histogram (only for worlds that had a done event).
    reasons = Counter()
    for w in range(n_world):
        if has_done_np[w]:
            reasons[SubEnvHandler.REASON_NAME[int(fd_reason_np[w])]] += 1
    n_failed = int(has_done_np.sum())     # worlds whose ep ended early (not the full window)
    n_intact = n_world - n_failed         # worlds that survived the full eval window

    # FULL reason set (every REASON_NAME except "none"=0), zeros included.
    # ``reasons`` above is nonzero-only (compact for stdout); this dense version
    # gives wandb a continuous per-reason series even on evals where a given
    # reason never fires (so a reason dropping to 0 logs 0 instead of a gap).
    reasons_full = {name: int(reasons.get(name, 0))
                    for code, name in SubEnvHandler.REASON_NAME.items() if code != 0}

    return dict(
        **extra_success,        # success_rate_strict + success_rate_<name> / n_success_<name>
        **sem_metrics,          # sem_<name> — task-defined semantic metrics (final step)
        # Absolute-reward summary (the user-facing eval signal).
        mean_return          = float(cum_rew_np.mean()),
        median_return        = float(np.median(cum_rew_np)),
        min_return           = float(cum_rew_np.min()),
        max_return           = float(cum_rew_np.max()),
        std_return           = float(cum_rew_np.std()),
        mean_reward_per_step = float(np.mean(rew_per_step)) if rew_per_step else 0.0,
        # Episode lifecycle stats.
        eval_length          = int(eval_length),
        n_world              = n_world,
        n_success            = n_success,
        n_failed             = n_failed,
        n_intact             = n_intact,
        success_rate         = success_rate,
        max_final_streak     = int(final_streak_np.max()),
        mean_final_streak    = float(final_streak_np.mean()),
        mean_first_done_step = float(fd_step_np[has_done_np].mean()) if has_done_np.any() else float(eval_length),
        first_done_reasons   = dict(reasons),
        # Dense per-reason counts (all reasons, 0 included) for wandb continuity.
        reset_reasons_full   = reasons_full,
        # Per-term episode-return statistics (one episode per world).
        reward_term_stats    = term_stats,
        # Throughput.
        eval_steps           = eval_steps,
        eval_elapsed_s       = float(eval_elapsed),
        eval_sps             = float(eval_sps),
        # Backwards-compat aliases for older callers.
        n_completed_episodes = n_world,
        n_success_episodes   = n_success,
        mean_episode_length  = float(eval_length),
        reset_reasons        = dict(reasons),
    )
