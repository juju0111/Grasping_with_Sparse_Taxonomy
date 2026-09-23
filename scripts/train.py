"""scripts/train.py — Single- or multi-hand PPO training entry point.

Everything is driven by a single YAML under ``config/training/``. The
script generalises to **cross-embodiment training** by treating every
handler in ``env.handlers`` symmetrically:

  * 1 handler  (single-hand)  — a yaml like ``grasping_policy_teacher.yaml``
    with ``using_hand_name_list: [tesollo]``.
  * N handlers (multi-hand)   — yaml inheriting from ``grit_multi_hand``
    with ``using_hand_name_list: [tesollo, robotis_sh5, ...]``.

Both paths use the same code: one policy / optimizer / term tracker per
**tag** (where ``tag = handler.hand_util.hand_name``), and a single
:class:`MultiHandRolloutBuffer` that wraps one inner buffer per tag.

Usage::

    python scripts/train.py -c grasping_policy_teacher
    python scripts/train.py -c grasping_policy_async_best_w_ground_force --overrides 'using_hand_name_list=[tesollo]'

    # programmatic
    from scripts.train import main
    main(config_name="grasping_policy_teacher")
"""
from __future__ import annotations

# === stdlib =============================================================
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ── Bootstrap project root + sys.path (runnable from any cwd) ────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_pr   = _HERE
while _pr not in ("/", "") and not (
    os.path.isdir(os.path.join(_pr, "grit"))
    and os.path.isdir(os.path.join(_pr, "config"))
):
    _pr = os.path.dirname(_pr)
if _pr in ("/", ""):
    raise RuntimeError(f"could not locate project root from {_HERE}")
PROJECT_ROOT = _pr
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Headless guard: MUST run before any grit import. Those pull in a
# module-level ``import pyautogui`` that crashes on a server with no DISPLAY
# (KeyError: 'DISPLAY' inside mouseinfo). Importing this stubs the GUI deps when
# headless. (grit.util.* __init__ is empty → this import is light.)
from grit.util.headless_guard import ensure_headless_gui_stubs  # noqa: E402
ensure_headless_gui_stubs()

# === third-party ========================================================
import torch
import warp as wp
from omegaconf import OmegaConf


# === project ============================================================
# Train-loop-only symbols (these stay in train.py).
from grit.training.rl_env_base       import SubEnvHandler, list_rl_envs   # SubEnvHandler.REASON_NAME used inside train_loop
from grit.training.checkpoint        import (
    save_checkpoint, save_shared_checkpoint,
    # Re-exported high-level orchestration:
    resolve_save_dir, maybe_resume, load_weights_from,
)
from grit.training.reward_tracker    import RewardTermReturnTracker
from grit.training.reward_normalizer import RewardNormalizer
from grit.training.evaluation        import run_eval
from grit.training.algorithm.ppo               import ppo_update, ppo_update_multi
from grit.model.base_networks        import list_policies
from grit.util.ppo_rollout_buffer    import MultiHandRolloutBuffer        # type hint inside train_loop

# Setup helpers — moved to dedicated modules; train.py is now a thin driver
# that wires them together and runs the PPO loop.
from grit.training.builders          import (
    build_orchestrator,
    obj_spec_provider_from_config,
    build_rl_env_and_handlers,
    build_eval_env,
    build_policies_and_optimizers,
    build_rollout_buffer,
)
from grit.training.run_config        import save_run_snapshot
from grit.training.wandb_setup       import setup_wandb
from grit.training.resample          import ObjectResampler, is_oom_error, free_gpu_memory

# ──────────────────────────────────────────────────────────────────────────
# PPO training loop — handles 1+ handlers via per-tag dicts
# ──────────────────────────────────────────────────────────────────────────

def bc_pretrain(env, policies, tags, pre_cfg, device) -> bool:
    """DAgger (BC) pretrain — runs **in the same process and on the same network as PPO**.

    Active only when the handler provides ``expert_action()`` (otherwise silently
    skipped — the same hasattr-gate convention as ``resample_target_qpos_sources``).
    For LF same-embodiment the frozen leader teacher is queried with "target = current
    leader pose" and the follower mirroring action is used as the label.

    This replaces the path of training BC separately and handing over a ckpt, which
    had two pitfalls: (1) the obs convention easily diverged from the env obs, and
    (2) ``std_param``, which BC never trains, was carried in the ckpt and made the
    exploration std jump. Reusing the same policy object rules out both at the source
    (BC only uses ``forward()``, so std_param / critic are untouched).

    Returns True when pretraining actually ran.
    """
    if not bool(pre_cfg.get("enabled", False)):
        return False
    hs = [env.handlers[i] for i in range(len(tags))]
    if not all(hasattr(h, "expert_action") for h in hs):
        print("[pretrain] handler does not provide expert_action() → skip")
        return False

    n_iter   = int(pre_cfg.get("n_iter", 6))
    ticks    = int(pre_cfg.get("ticks_per_iter", 64))
    ep_len   = int(pre_cfg.get("ep_len", 64))
    beta0    = float(pre_cfg.get("beta0", 1.0))
    decay    = float(pre_cfg.get("beta_decay", 0.5))
    epochs   = int(pre_cfg.get("bc_epochs", 4))
    mb       = int(pre_cfg.get("batch_size", 4096))
    lr       = float(pre_cfg.get("lr", 1.0e-3))
    cap      = int(pre_cfg.get("buffer_cap", 150_000))

    print(f"\n=== BC pretrain (DAgger) : {n_iter} iter × {ticks} tick "
          f"× {sum(h.NWORLD for h in hs)} world, β {beta0}→{beta0*decay**(n_iter-1):.3f} ===")
    for i, tag in enumerate(tags):
        h, pol = hs[i], policies[tag]
        opt = torch.optim.Adam(pol.parameters(), lr=lr)
        buf_o: List[torch.Tensor] = []
        buf_a: List[torch.Tensor] = []
        n_drop = 0                       # transitions dropped for non-finite values
        pol.train()
        for it in range(n_iter):
            beta = beta0 * (decay ** it)
            obs = h.reset()
            for tstep in range(ticks):
                if tstep and tstep % ep_len == 0:
                    obs = h.reset()
                a_exp = h.expert_action()
                with torch.no_grad():
                    a_st = pol(obs)
                use_e = (torch.rand(h.NWORLD, 1, device=a_exp.device) < beta)
                # Non-finite transitions are **not pushed to the buffer** — the aggregate
                # buffer persists across iterations, so a single corrupted row would pin
                # the loss of every later epoch at NaN.
                _ok = (torch.isfinite(obs).all(dim=1)
                       & torch.isfinite(a_exp).all(dim=1))
                if bool(_ok.all()):
                    buf_o.append(obs.detach().clone())
                    buf_a.append(a_exp.detach().clone())
                else:
                    n_drop += int((~_ok).sum())
                    if _ok.any():
                        buf_o.append(obs[_ok].detach().clone())
                        buf_a.append(a_exp[_ok].detach().clone())
                obs = h.step(torch.where(use_e, a_exp, a_st))[0]
            O = torch.cat(buf_o)[-cap:]
            A = torch.cat(buf_a)[-cap:]
            buf_o, buf_a = [O], [A]                 # keep the memory cap
            losses, n_skip = [], 0
            for _ep in range(epochs):
                idx = torch.randperm(len(O), device=O.device)
                for k in range(0, len(O), mb):
                    b = idx[k:k + mb]
                    loss = torch.nn.functional.mse_loss(pol(O[b]), A[b])
                    if not torch.isfinite(loss):
                        n_skip += 1      # do not step on a non-finite loss
                        opt.zero_grad(); continue
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                    opt.step()
                    losses.append(float(loss))
            _bl = float(np.mean(losses[-20:])) if losses else float("nan")
            print(f"  [{tag}] iter {it}  β={beta:.3f}  buf={len(O):,}  "
                  f"bc_loss={_bl:.5f}"
                  + (f"  ⚠ drop={n_drop} skip={n_skip}" if (n_drop or n_skip) else ""))
        pol.eval()
        del buf_o, buf_a
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("=== BC pretrain done → starting PPO (continuing on the same network) ===\n")
    return True


def train_loop(
    *,
    env,
    eval_env       = None,                  # fixed-object eval env; None → eval on the training env

    tags:       List[str],
    policies:   Dict[str, torch.nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    buffer:     MultiHandRolloutBuffer,
    train_cfg:  dict,
    save_dir:   Path,
    task_name:  str,
    env_name:   str,
    seen_steps: int,
    log_metrics,
    print_every:   int   = 10,
    shared_module        = None,            # set when policy.name starts with "shared_"
    resampler:     Optional[ObjectResampler] = None,  # None = no object resampling
    env_box:       Optional[list] = None,   # [env] handle so the caller's ref can be released on rebuild
    reward_normalizers: Optional[Dict[str, RewardNormalizer]] = None,  # cfg.training.reward_norm; None = raw rewards,
    video_recorders: Optional[Dict[str, Any]] = None,
) -> int:
    """Run the PPO loop until ``cfg.training.total_new_steps`` is hit.
    Generalises across ``len(tags) >= 1`` by keeping every tag's
    rollout / GAE / eval / ckpt symmetric (dict-keyed).

    PPO update path branches on ``shared_module``:

      * ``shared_module is None`` (independent per-tag): loop
        :func:`ppo_update` once per tag — each tag's network has its own
        optimiser, no cross-tag interference. ``n_epochs × num_minibatch``
        optimiser steps **per tag**.

      * ``shared_module is not None`` (shared cross-embodiment): call
        :func:`ppo_update_multi` once — gradients from every tag's
        minibatches accumulate onto the **same** shared parameters
        before a single ``optimizer.step()``. Total step count is
        ``n_epochs × num_minibatch`` (NOT multiplied by ``n_tags``),
        and every tag's PPO ratio sees the same "old" policy. See
        :mod:`grit.training.algorithm.ppo` for the rationale.

    Returns the final ``seen_steps`` cursor.
    """
    # ── Hyperparameters (all from cfg.training) ─────────────────────────
    GAMMA           = float(train_cfg["gamma"])
    GAE_LAMBDA      = float(train_cfg["gae_lambda"])
    N_PPO_EPOCHS    = int(train_cfg["n_ppo_epochs"])
    CLIP_EPS        = float(train_cfg["clip_eps"])
    ENT_COEF        = float(train_cfg["ent_coef"])
    VF_COEF         = float(train_cfg["vf_coef"])
    MAX_GRAD_NORM   = float(train_cfg["max_grad_norm"])
    # KL-based adaptive lr — None/0 → OFF (fixed lr). Typically 0.01. Even as σ
    # decreases monotonically (KL ∝ (Δμ/σ)²) the lr is lowered to keep KL near the target.
    # See the grit.training.algorithm.ppo._adapt_lr docstring for details.
    _desired_kl     = train_cfg.get("desired_kl", None)
    DESIRED_KL      = float(_desired_kl) if _desired_kl else None
    LR_MIN          = float(train_cfg.get("lr_min", 1.0e-6))
    LR_MAX          = float(train_cfg.get("lr_max", 1.0e-3))
    KL_ADAPT_FACTOR = float(train_cfg.get("kl_adapt_factor", 1.5))
    # ── lr cap step decay — late annealing independent of KL (0 = off) ──
    # KL adaptation only lowers the lr when approx_kl exceeds the desired value. On
    # tasks where training converges and kl stays below desired, the adapter keeps
    # pushing the lr up to the cap, full-rate updates continue and the policy drifts
    # around the optimum (performance slowly declines after the peak).
    # Lowering the cap by log-interpolation to lr_max_final over the [start, end]
    # window leaves the KL adapter intact while forcing smaller late steps.
    LR_MAX_FINAL       = float(train_cfg.get("lr_max_final", 0.0))
    LR_MAX_DECAY_START = int(train_cfg.get("lr_max_decay_start", 0))
    LR_MAX_DECAY_END   = int(train_cfg.get("lr_max_decay_end", 0))
    # Advantage normalisation mode: "rollout" (whole-rollout statistics once; default) |
    # "minibatch" (re-normalise per minibatch — for heavy-tailed advantages; see ppo.py).
    ADV_NORM        = str(train_cfg.get("adv_norm", "rollout"))
    TOTAL_NEW_STEPS = int(train_cfg["total_new_steps"])
    N_EVALS         = int(train_cfg["n_evals"])
    # Async (asymmetric) actor-critic: thread a separate (privileged) critic obs
    # through act / buffer / bootstrap. OFF (default) → current symmetric path.
    ASYNC_PPO       = bool(train_cfg.get("async_ppo", False))
    # Two distinct per-iteration step counts:
    #
    #  * ``STEPS_PER_ITER``        — PHYSICAL env steps the rollout SIMULATES per
    #    iteration (sum NWORLD × T over all tags). Used only for throughput /
    #    SPS reporting (the sim genuinely advances this many control steps).
    #
    #  * ``UPDATE_STEPS_PER_ITER`` — env steps actually CONSUMED by the PPO
    #    update per iteration. ``iter_minibatches`` draws ``batch_size ×
    #    num_minibatch`` transitions per epoch out of the ``NWORLD × T`` the
    #    rollout collected; when both are pinned (subset sampling) this is
    #    SMALLER than ``STEPS_PER_ITER`` and the remainder is reshuffled away
    #    unused each epoch. ``total_new_steps`` is the BUDGET against the steps
    #    that drove a gradient update, so all budget bookkeeping below counts
    #    ``UPDATE_STEPS_PER_ITER`` (summed across tags). For full-coverage
    #    configs (``num_minibatch`` derived) the two are equal → no change.
    STEPS_PER_ITER  = sum(h.NWORLD for h in env.handlers) * buffer.unroll_length
    UPDATE_STEPS_PER_ITER = sum(
        min(buffer[tag].batch_size * buffer[tag].num_minibatch,
            buffer[tag].num_envs   * buffer[tag].unroll_length)
        for tag in tags
    )
    EVAL_EVERY      = max(1, TOTAL_NEW_STEPS // UPDATE_STEPS_PER_ITER // N_EVALS)
    # Handler target-source resample cadence: 0 = off, N > 0 = once **every N eval periods**.
    # Do not couple it to the eval period itself — raising n_evals would then change the
    # target distribution just as often, creating a moving target and slowing training
    # considerably. N keeps the two axes independent.
    # Active only when the handler implements ``resample_target_qpos_sources``.
    RESAMPLE_QPOS_EVERY = max(0, int(train_cfg.get("resample_qpos_sources_every", 0)))
    EVAL_LENGTH     = (int(train_cfg["eval_length"])
                       if train_cfg.get("eval_length") is not None
                       else env.handlers[0].MAX_EPISODE_STEPS)
    N_ITERATIONS    = max(1, math.ceil(TOTAL_NEW_STEPS / UPDATE_STEPS_PER_ITER))

    print(f'TOTAL_NEW_STEPS  : {TOTAL_NEW_STEPS:,}   (across {len(tags)} handler(s))')
    print(f'STEPS_PER_ITER   : {STEPS_PER_ITER:,}   (sum NWORLD × T over all tags — simulated)')
    print(f'UPDATE_STEPS/ITER: {UPDATE_STEPS_PER_ITER:,}   '
          f'(sum batch_size × num_minibatch — consumed by PPO update; counts toward budget)')
    print(f'N_ITERATIONS     : {N_ITERATIONS:,}')
    print(f'EVAL_EVERY       : {EVAL_EVERY}')
    print('adaptive lr      : '
          + (f'desired_kl={DESIRED_KL:g}  factor={KL_ADAPT_FACTOR:g}  '
             f'lr∈[{LR_MIN:g}, {LR_MAX:g}]'
             + (f'  cap→{LR_MAX_FINAL:g} @ {LR_MAX_DECAY_START:,}~{LR_MAX_DECAY_END:,}'
                if LR_MAX_FINAL > 0.0 else '') if DESIRED_KL else
             f'OFF (lr={float(train_cfg["lr"]):g}; enable with training.desired_kl=0.01)'))
    print(f'adv_norm         : {ADV_NORM}'
          + ('   (per-minibatch re-normalisation — stabilises heavy-tailed advantages)'
             if ADV_NORM == 'minibatch' else '   (whole-rollout statistics once)'))
    if RESAMPLE_QPOS_EVERY > 0:
        _rs_iters = max(1, EVAL_EVERY) * RESAMPLE_QPOS_EVERY
        print(f'qpos-src resample: every {RESAMPLE_QPOS_EVERY} EVAL period(s) '
              f'= {_rs_iters} iters ≈ {_rs_iters * UPDATE_STEPS_PER_ITER:,} steps '
              f'(handler.resample_target_qpos_sources)')
    print(f'SAVE_DIR         : {save_dir}')

    # ── Per-tag persistent state ────────────────────────────────────────
    N_REASONS    = len(SubEnvHandler.REASON_NAME)
    discount_buf:        Dict[str, torch.Tensor] = {}
    iter_reason_hist:    Dict[str, torch.Tensor] = {}
    train_term_tracker:  Dict[str, RewardTermReturnTracker] = {}
    prev_obs:            Dict[str, torch.Tensor] = {}
    prev_critic_obs:     Dict[str, torch.Tensor] = {}   # async only (privileged critic obs)

    # Raw-reward per-iter accumulator (GPU scalar; sync-free adds) — only used
    # when reward normalization is on, so ``reward_mean_raw`` can be logged
    # alongside the (normalized) ``reward_mean`` the buffer sees.
    raw_rew_accum:       Dict[str, torch.Tensor] = {}

    obs_list = env.reset()          # List[obs] aligned with env.handlers
    for i, tag in enumerate(tags):
        h = env.handlers[i]
        prev_obs[tag]           = obs_list[i].clone()
        if ASYNC_PPO:
            prev_critic_obs[tag] = h.critic_obs_torch.clone()
        discount_buf[tag]       = torch.empty(h.NWORLD, device=env.torch_device)
        iter_reason_hist[tag]   = torch.zeros(N_REASONS, dtype=torch.long, device=env.torch_device)
        train_term_tracker[tag] = RewardTermReturnTracker(h.metrics, h.NWORLD, env.torch_device)
        raw_rew_accum[tag]      = torch.zeros((), device=env.torch_device)

    log_lines: list[str] = []
    remaining_steps = max(0, TOTAL_NEW_STEPS - seen_steps)
    n_iter_to_run   = max(1, math.ceil(remaining_steps / UPDATE_STEPS_PER_ITER)) if remaining_steps > 0 else 0
    iter_offset     = N_ITERATIONS - n_iter_to_run
    print(f'iterations to run: {n_iter_to_run:,} '
          f'(starting from logical iter {iter_offset+1}/{N_ITERATIONS})')

    # Let each handler's curriculum (on_train_progress) reach its policy — the
    # geom-tracking curriculum anneals ``policy.min_std`` (σ floor) alongside its
    # reward-weight ramps. No-op for handlers/policies that don't use it.
    for i, tag in enumerate(tags):
        if hasattr(env.handlers[i], "bind_policy"):
            env.handlers[i].bind_policy(policies[tag])

    wp.synchronize_device()
    t_train_start  = time.perf_counter()
    steps_at_start = seen_steps
    phase_totals   = dict(rollout=0.0, gae=0.0, update=0.0, total=0.0)

    # ── Outer iteration loop ────────────────────────────────────────────
    for it_local in range(n_iter_to_run):
        it = iter_offset + it_local

        # ── Periodic object resample (once per EVAL period; see ObjectResampler) ──
        # Rebuilds the env with a fresh object draw and re-wires per-tag rollout
        # state. Tied to the eval cadence (not per-iter — a full rebuild is far
        # too costly and churns warp's pool into OOM) and OOM-safe (keeps the
        # current objects on failure).
        if resampler is not None and resampler.due(it, it_local, EVAL_EVERY):
            env = resampler.resample(
                it, env, tags, prev_obs, prev_critic_obs, train_term_tracker,
                async_ppo=ASYNC_PPO, env_box=env_box)
            # Env rebuild → every episode restarts; drop the normalizers'
            # per-env discounted-return carriers (running stats are kept).
            if reward_normalizers is not None:
                for _rn in reward_normalizers.values():
                    _rn.reset_returns()

        # ── Periodic target source resample (handler-defined; opt-in) ──
        # Lightweight path that swaps only the handler's gather tables for a new seed,
        # without rebuilding the env. cadence = once every ``resample_qpos_sources_every``
        # **eval periods** (0 = off) — see the RESAMPLE_QPOS_EVERY comment above for the
        # pitfall of coupling the resample cadence to the eval period (= n_evals).
        if (RESAMPLE_QPOS_EVERY > 0 and it_local > 0
                and (it % (max(1, EVAL_EVERY) * RESAMPLE_QPOS_EVERY) == 0)):
            for _h in env.handlers:
                if hasattr(_h, "resample_target_qpos_sources"):
                    _h.resample_target_qpos_sources()

        # Report training progress to each handler (generic hook — the env
        # drives any step-based curriculum itself; no task logic in train.py).
        # Progress is measured in BUDGET (consumed) steps so the curriculum
        # schedule (e.g. LIFT_DECAY_*) stays aligned with ``total_new_steps``.
        _cur_step = steps_at_start + it_local * UPDATE_STEPS_PER_ITER
        for _h in env.handlers:
            _h.on_train_progress(_cur_step, TOTAL_NEW_STEPS)

        wp.synchronize_device()
        t_iter_start = time.perf_counter()

        # ── Rollout (per-tag, sync-free inner loop) ─────────────────────
        # OOM-guarded: a freshly-resampled HEAVY object set can exceed the
        # (rebuild-fragmented) warp memory pool when mujoco_warp allocates
        # collision buffers during stepping. Rather than crash the whole run,
        # skip THIS iteration's rollout+update and continue — the next resample
        # draws a (usually lighter) set. No state leaks: buffer.reset() re-inits
        # the buffer next iteration and seen_steps is not advanced for the skip.
        try:
            buffer.reset()
            for tag in tags:
                iter_reason_hist[tag].zero_()
                train_term_tracker[tag].reset()
                raw_rew_accum[tag].zero_()

            with torch.no_grad():
                for t in range(buffer.unroll_length):
                    for i, tag in enumerate(tags):
                        h    = env.handlers[i]
                        # Async: the critic reads the privileged ``prev_critic_obs``;
                        # symmetric → None (the value is computed from ``prev_obs``).
                        crit = prev_critic_obs[tag] if ASYNC_PPO else None
                        out  = policies[tag].act(prev_obs[tag], critic_obs=crit)  # type: ignore
                        next_obs, rew, done, info = h.step(out['action'])

                        # Optional reward normalization (cfg.training.reward_norm):
                        # the buffer (→ GAE → critic) sees the normalized reward;
                        # raw stats are accumulated separately for logging. All
                        # other consumers (handler.metrics, eval) stay raw.
                        if reward_normalizers is not None:
                            raw_rew_accum[tag] += rew.mean()
                            rew = reward_normalizers[tag](rew, done)

                        # discount = γ · (1 − done) — in-place to skip alloc.
                        torch.sub(1.0, done, out=discount_buf[tag])
                        discount_buf[tag].mul_(GAMMA)

                        # Store the critic obs only in async mode (the extra is
                        # registered in the buffer only then; otherwise dropped).
                        async_extra = {"critic_observation": prev_critic_obs[tag]} if ASYNC_PPO else {}
                        # only when the task declares it (otherwise the buffer silently drops it)
                        if 'sample_mask' in info:
                            async_extra["sample_mask"] = info['sample_mask']
                        if 'bc_action' in info:
                            async_extra["bc_action"] = info['bc_action']
                        buffer.add(
                            tag,
                            observation      = prev_obs[tag],
                            action           = out['action'],
                            reward           = rew,
                            discount         = discount_buf[tag],
                            next_observation = next_obs,
                            log_prob         = out['log_prob'],
                            value            = out['value'],
                            pre_tanh         = out['pre_tanh'],
                            # Time-limit truncation flag → GAE bootstraps V at the
                            # time limit instead of treating timeout as a hard terminal.
                            truncation       = info['truncation'],
                            **async_extra,
                        )

                        # GPU-side done-reason histogram (single bincount/tag).
                        done_mask_t   = info['done_mask']
                        done_reason_t = info['done_reason']
                        iter_reason_hist[tag] += torch.bincount(
                            done_reason_t,
                            weights=done_mask_t.float(),
                            minlength=N_REASONS,
                        ).long()

                        # Per-term episode-return accumulator (sync-free).
                        # MUST run BEFORE per_world_reset_if_done because the
                        # next step's metrics reflect the freshly-reset state.
                        train_term_tracker[tag].update(h.metrics, done_mask_t)

                    # All handlers async-reset done worlds (sync-free).
                    env.per_world_reset_if_done()
                    for i, tag in enumerate(tags):
                        prev_obs[tag].copy_(env.handlers[i].obs_torch)
                        if ASYNC_PPO:
                            prev_critic_obs[tag].copy_(env.handlers[i].critic_obs_torch)

                # Bootstrap value at the rollout boundary, per tag.
                last_values = {
                    tag: policies[tag]._value(                                  # type: ignore
                        prev_obs[tag], prev_critic_obs[tag] if ASYNC_PPO else None)
                    for tag in tags
                }
            wp.synchronize_device()
        except RuntimeError as _e:
            if not is_oom_error(_e):
                raise
            free_gpu_memory()
            print(f'[iter {it}] ⚠ rollout OOM (heavy object set) — skipped update, continuing')
            continue
        t_after_rollout = time.perf_counter()

        # ── GAE (per-tag) ───────────────────────────────────────────────
        buffer.compute_gae(last_values, gamma=GAMMA, lam=GAE_LAMBDA)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_after_gae = time.perf_counter()

        # ── lr cap step decay (see the LR_MAX_* comment above; 0 = off) ─────
        _lr_max_now = LR_MAX
        if LR_MAX_FINAL > 0.0 and LR_MAX_DECAY_END > LR_MAX_DECAY_START:
            _f = (_cur_step - LR_MAX_DECAY_START) \
                 / float(LR_MAX_DECAY_END - LR_MAX_DECAY_START)
            _f = min(max(_f, 0.0), 1.0)
            _lr_max_now = float(LR_MAX * (LR_MAX_FINAL / LR_MAX) ** _f)
            # When the cap drops, push the current lr down with it (the adapter keeps
            # working below the cap). Otherwise the lr stays above the cap while kl is low.
            for _opt in optimizers.values():
                for _g in _opt.param_groups:
                    if _g["lr"] > _lr_max_now:
                        _g["lr"] = _lr_max_now

        # ── PG lr warm-start ramp ───────────────────────────────────────
        # Switching PG on at the target lr **all at once** right after the critic warm-up
        # makes the KL of the first update overshoot the target by orders of magnitude and
        # costs a one-off performance drop. Instead, open the lr **cap** geometrically from
        # pg_start_lr to lr_max and let the KL adapter find its own scale underneath — the
        # lr follows the ramp while KL is low and the adapter pulls it back as soon as KL
        # exceeds the target. The ramp only **permits**; the adapter decides.
        # 0 = off (legacy behaviour).
        _pgw   = int(train_cfg.get("pg_lr_warmup_iters", 0))
        _warm0 = int(train_cfg.get("critic_warmup_iters", 0))
        if _pgw > 0 and it_local >= _warm0:
            _p = (it_local - _warm0 + 1) / float(_pgw)
            if _p < 1.0:
                _lo = float(train_cfg.get("pg_start_lr", train_cfg["lr"]))
                if _lo > 0.0 and _lr_max_now > _lo:
                    _cap = float(_lo * (_lr_max_now / _lo) ** max(_p, 0.0))
                    _lr_max_now = min(_lr_max_now, _cap)
                    for _opt in optimizers.values():
                        for _g in _opt.param_groups:
                            if _g["lr"] > _lr_max_now:
                                _g["lr"] = _lr_max_now
            # Fixed-lr mode (no desired_kl): without the KL adapter the lr never rises on
            # its own — the ramp above only tightens the **cap**, so the lr would stay at
            # pg_start_lr forever. In this mode the ramp value is used directly as the lr,
            # and lr_max is kept as the fixed lr once the ramp ends.
            if not DESIRED_KL:
                for _opt in optimizers.values():
                    for _g in _opt.param_groups:
                        _g["lr"] = _lr_max_now

        # ── PPO update ──────────────────────────────────────────────────
        # Shared mode: one ``ppo_update_multi`` call accumulates per-tag
        # gradients onto the shared params and steps the (single) optimizer
        # once per macro-minibatch. Independent mode: per-tag loop.
        if shared_module is not None:
            losses = ppo_update_multi(
                policies,
                next(iter(optimizers.values())),    # one shared optimizer
                buffer,
                n_epochs        = N_PPO_EPOCHS,
                clip_eps        = CLIP_EPS,
                vf_coef         = VF_COEF,
                ent_coef        = ENT_COEF,
                max_grad_norm   = MAX_GRAD_NORM,
                desired_kl      = DESIRED_KL,
                lr_min          = LR_MIN,
                lr_max          = _lr_max_now,
                kl_adapt_factor = KL_ADAPT_FACTOR,
                adv_norm        = ADV_NORM,
            )
        else:
            losses: Dict[str, dict] = {}
            for tag in tags:
                # DAgger BC auxiliary loss (only when the handler has BC_COEF>0; decays on
                # the same schedule as β → pure RL later on)
                _h0    = env.handlers[0]
                _bc_c  = float(getattr(_h0, "BC_COEF", 0.0)) * \
                         float(getattr(_h0, "_dagger_scale", 0.0))
                _pol_t = policies[tag]
                def _bc_aux(mb, _p=_pol_t):
                    tgt = mb.extras.get("bc_action")
                    if tgt is None:
                        return None
                    out = _p.act(mb.observation, deterministic=True,
                                 inference_only=True)
                    return torch.nn.functional.mse_loss(torch.tanh(out["loc"]), tgt)
                # KL-adaptive lr is disabled during the BC phase: BC moves the policy a lot,
                # KL grows and the adaptation rule drives the lr down to 1e-6, stalling
                # training (offline BC converges fine at 3e-4). KL adaptation is a
                # policy-gradient device, not a supervised-learning one.
                # Without β-mixing (DAGGER_BETA=0) the executed action is the policy action,
                # so PG is valid → BC serves only as an **anchor** and PG stays on.
                # (PG is switched off only in the DAgger phase with β>0.)
                _beta_on = float(getattr(_h0, "DAGGER_BETA", 0.0)) > 0.0
                _pol_c = (1.0 if (_bc_c <= 0.0 or not _beta_on)
                          else 1.0 - float(getattr(_h0, "_dagger_scale", 0.0)))
                _bc_phase = (_bc_c > 0.0) and _beta_on
                # The value loss is also disabled during the BC phase: vf ≈ 1e3~1e4 while
                # BC ≈ 0.02, so when max_grad_norm clips the total norm the critic gradient
                # monopolises it and BC is crushed. It is brought back on the same ramp as
                # pol_coef (the critic switches on together with the BC→RL transition and
                # acts as a warm-up).
                # With BC off (plain RL run) PG is used as is — _dagger_scale defaults to 1.0
                # in the ctor, so without this guard the early PG would be 0 when finishing
                # RL from a DAgger ckpt.
                # critic warm-up: when RL takes over from a BC/DAgger ckpt the critic is
                # untrained, the advantages are meaningless and their gradient destroys the
                # policy within a few iterations. PG stays off until the critic settles.
                _warm = int(train_cfg.get("critic_warmup_iters", 0))
                _in_warm = (_warm > 0 and it_local < _warm)
                if _in_warm:
                    _pol_c = 0.0        # switch off PG only (the critic must keep training)
                # **Freeze the KL-adaptive lr during the critic warm-up.** The warm-up turns
                # PG off, so approx_kl drops to floating-point noise (≈1e-9); since the
                # increase condition of `_adapt_lr` is `0.0 < kl < desired_kl/2`, that noise
                # **passes the check** → the lr is multiplied by 1.5 every iteration and hits
                # the cap (lr_max=1e-3) within a few iterations. When the warm-up then ends,
                # the first PG update runs with a 10-50x oversized lr and kills the policy
                # in one shot (KL orders of magnitude above target, no recovery). With BC on
                # (BC_COEF>0) the same inflated lr let **BC** wreck the actor.
                # → adaptation is disabled (lr fixed) during the warm-up and the lr is reset
                #   explicitly to the PG start lr when it ends.
                if _in_warm:
                    _cw_lr = float(train_cfg.get("critic_warmup_lr",
                                                 train_cfg["lr"]))
                    for _g in optimizers[tag].param_groups:
                        _g["lr"] = _cw_lr
                elif _warm > 0 and it_local == _warm:
                    _pg_lr = float(train_cfg.get("pg_start_lr",
                                                 train_cfg["lr"]))
                    for _g in optimizers[tag].param_groups:
                        _g["lr"] = _pg_lr
                    print(f'  [{tag}] critic warm-up finished ({_warm} it) → '
                          f'PG start lr={_pg_lr:g}')
                if _bc_phase:
                    _bc_lr = float(train_cfg.get("bc_lr", 3.0e-4))
                    for _g in optimizers[tag].param_groups:
                        _g["lr"] = _bc_lr
                losses[tag] = ppo_update(
                    policies[tag], optimizers[tag], buffer[tag],
                    n_epochs        = N_PPO_EPOCHS,
                    clip_eps        = CLIP_EPS,
                    vf_coef         = 0.0 if _bc_phase else VF_COEF,
                    ent_coef        = ENT_COEF,
                    max_grad_norm   = MAX_GRAD_NORM,
                    # during warm-up KL is noise and the adapter misbehaves (see the comment above).
                    desired_kl      = (None if (_bc_phase or _in_warm)
                                       else DESIRED_KL),
                    lr_min          = LR_MIN,
                    lr_max          = _lr_max_now,
                    kl_adapt_factor = KL_ADAPT_FACTOR,
                    adv_norm        = ADV_NORM,
                    aux_loss_fn     = _bc_aux if _bc_c > 0.0 else None,
                    aux_coef        = _bc_c,
                    # PG off in the DAgger phase (β>0) → BC + critic only
                    pol_coef        = _pol_c,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_iter_end = time.perf_counter()

        # ── Per-iter throughput + phase share ───────────────────────────
        iter_time         = max(t_iter_end - t_iter_start, 1e-9)
        rollout_time      = t_after_rollout - t_iter_start
        gae_time          = t_after_gae     - t_after_rollout
        update_time       = t_iter_end      - t_after_gae
        training_sps_iter = STEPS_PER_ITER / iter_time

        phase_totals['rollout'] += rollout_time
        phase_totals['gae']     += gae_time
        phase_totals['update']  += update_time
        phase_totals['total']   += iter_time

        # Single CPU sync per iter — read histograms + per-term stats.
        reason_counts_full: Dict[str, Dict[str, int]] = {}
        reason_counts_nz:   Dict[str, Dict[str, int]] = {}
        for tag in tags:
            full = {
                SubEnvHandler.REASON_NAME[i]: int(c)
                for i, c in enumerate(iter_reason_hist[tag].cpu().tolist())
            }
            reason_counts_full[tag] = full
            reason_counts_nz[tag]   = {k: v for k, v in full.items() if v > 0}

        train_term_stats: Dict[str, dict] = {
            tag: train_term_tracker[tag].stats() for tag in tags
        }

        # ── Bookkeeping ─────────────────────────────────────────────────
        # Budget cursor advances by the steps CONSUMED by the update, not the
        # (larger) number simulated — see UPDATE_STEPS_PER_ITER above.
        seen_steps += UPDATE_STEPS_PER_ITER
        iter_mean_rew = {tag: float(buffer[tag].reward.mean().item()) for tag in tags}
        # With normalisation ON the buffer (= iter_mean_rew) is on the normalised scale — the
        # raw mean is computed separately from raw_rew_accum accumulated during the rollout
        # (reported alongside in wandb + stdout).
        iter_mean_rew_raw: Optional[Dict[str, float]] = (
            {tag: float(raw_rew_accum[tag].item()) / buffer.unroll_length for tag in tags}
            if reward_normalizers is not None else None)

        # ── W&B per-iter scalars (per-tag namespacing) ──────────────────
        train_log_iter = {
            'train/iter'           : it + 1,
            'train/sps'            : training_sps_iter,
            'train/iter_time_ms'   : iter_time     * 1000.0,
            'train/rollout_time_ms': rollout_time  * 1000.0,
            'train/gae_time_ms'    : gae_time      * 1000.0,
            'train/update_time_ms' : update_time   * 1000.0,
        }
        for tag in tags:
            pfx = f'train/{tag}'
            train_log_iter[f'{pfx}/reward_mean']    = iter_mean_rew[tag]
            # Reward normalization diagnostics: ``reward_mean`` above is the
            # NORMALIZED value the buffer/critic saw; log the raw-scale mean and
            # the current divisor so scale drift stays visible.
            if iter_mean_rew_raw is not None:
                train_log_iter[f'{pfx}/reward_mean_raw']   = iter_mean_rew_raw[tag]
                train_log_iter[f'{pfx}/reward_norm_scale'] = reward_normalizers[tag].scale  # type: ignore[index]
            train_log_iter[f'{pfx}/loss/policy']    = losses[tag]['pol']
            train_log_iter[f'{pfx}/loss/value']     = losses[tag]['vf']
            train_log_iter[f'{pfx}/loss/entropy']   = losses[tag]['ent']
            train_log_iter[f'{pfx}/ppo_ratio_mean'] = losses[tag]['ratio']
            # Zero-gradient / collapse diagnostics: ``grad_norm`` (pre-clip policy
            # gradient L2 norm) and ``adv_std`` (RAW pre-normalization advantage
            # std). True trap ⇔ both → 0 with flat reward.
            train_log_iter[f'{pfx}/grad_norm'] = losses[tag].get('grad_norm', 0.0)
            train_log_iter[f'{pfx}/adv_std']   = losses[tag].get('adv_std', 0.0)
            # KL drift diagnostics: approx_kl (k3 estimator) overall mean + first/last epoch
            # means. last ≫ first (e.g. last > 0.02 persistently) signals too many n_ppo_epochs.
            # clip_frac is the rate of |ratio−1| > ε (0.1~0.3 is usually healthy).
            # ``lr`` — the controller output with adaptive lr ON, the fixed value when OFF.
            train_log_iter[f'{pfx}/lr']              = losses[tag].get('lr', 0.0)
            train_log_iter[f'{pfx}/approx_kl']       = losses[tag].get('approx_kl', 0.0)
            train_log_iter[f'{pfx}/approx_kl_first'] = losses[tag].get('approx_kl_first', 0.0)
            train_log_iter[f'{pfx}/approx_kl_last']  = losses[tag].get('approx_kl_last', 0.0)
            train_log_iter[f'{pfx}/clip_frac']       = losses[tag].get('clip_frac', 0.0)
            # Passthrough for auxiliary losses emitted by algorithm extensions — the mapping
            # above lists keys one by one, so new terms would silently vanish. Extra scalars
            # from ppo_update such as ``adapt/*`` (per-block MSE of the RMA/CSE adaptation
            # module) are forwarded under their own names. Tasks without such terms leave
            # the loop empty (no-op).
            for _k, _v in losses[tag].items():
                if _k.startswith('adapt/'):
                    train_log_iter[f'{pfx}/{_k}'] = _v
            for r_name, c in reason_counts_full[tag].items():
                train_log_iter[f'{pfx}/reset/{r_name}'] = c
            for term, st in train_term_stats[tag].items():
                train_log_iter[f'{pfx}/reward_term/{term}/mean'] = st['mean']
                train_log_iter[f'{pfx}/reward_term/{term}/std']  = st['std']
            n_eps = (next(iter(train_term_stats[tag].values()))['n']
                     if train_term_stats[tag] else 0)
            train_log_iter[f'{pfx}/reward_term/n_episodes'] = int(n_eps)
        log_metrics(train_log_iter, step=seen_steps)

        # ── stdout iter line ────────────────────────────────────────────
        if (it_local % print_every == 0) or (it_local + 1 == n_iter_to_run):
            header = (
                f'iter {it+1:>5d}/{N_ITERATIONS}  seen={seen_steps:>12,}  '
                f'sps={training_sps_iter:>10,.0f}  '
                f't={iter_time*1000:>6.1f}ms (r={rollout_time*1000:>5.1f}/'
                f'g={gae_time*1000:>4.1f}/u={update_time*1000:>5.1f})'
            )
            log_lines.append(header)
            print(header)
            for tag in tags:
                # normalisation ON → report the raw scale next to the normalised mean seen by the buffer.
                _raw_str = (f'(raw={iter_mean_rew_raw[tag]:+.3f})'
                            if iter_mean_rew_raw is not None else '')
                line = (
                    f'  [{tag:>15s}]  reward={iter_mean_rew[tag]:+.3f}{_raw_str}  '
                    f'pol={losses[tag]["pol"]:+.4f}  vf={losses[tag]["vf"]:+.4f}  '
                    f'ent={losses[tag]["ent"]:+.4f}  ratio={losses[tag]["ratio"]:.3f}  '
                    f'kl={losses[tag].get("approx_kl", 0.0):.4f}'
                    f'(→{losses[tag].get("approx_kl_last", 0.0):.4f})  '
                    f'clip={losses[tag].get("clip_frac", 0.0):.2f}  '
                    f'lr={losses[tag].get("lr", 0.0):.2e}  '
                    f'grad={losses[tag].get("grad_norm", 0.0):.3f}  '
                    f'adv_std={losses[tag].get("adv_std", 0.0):.3g}  '
                    f'resets={reason_counts_nz[tag]}'
                )
                log_lines.append(line)
                print(line)

        # ── Eval + checkpoint every EVAL_EVERY iterations ───────────────
        if (it + 1) % EVAL_EVERY == 0 or (it_local + 1) == n_iter_to_run:
            # Eval runs on the FIXED-object ``eval_env`` when provided (so metrics
            # stay comparable across training regardless of object resampling);
            # otherwise on the training env itself. Only the training-env path
            # needs snapshot/restore — run_eval hard-resets its handlers, so we
            # save/restore the rollout state to keep eval transparent to training.
            # ``eval_env`` is separate, so evaluating it never touches training.
            _eval_handlers = eval_env.handlers if eval_env is not None else env.handlers
            if eval_env is None:
                for h in env.handlers:
                    h.snapshot_train_state()
            ev_per_tag: Dict[str, dict] = {}
            for i, tag in enumerate(tags):
                ev_per_tag[tag] = run_eval(
                    policies[tag], _eval_handlers[i],
                    eval_length=EVAL_LENGTH,
                )
                # Success-gate: pass the eval result to the TRAINING handler (the curriculum
                # weight belongs to the training handler, so this applies even with eval_env).
                if hasattr(env.handlers[i], "on_eval_metrics"):
                    env.handlers[i].on_eval_metrics(ev_per_tag[tag])

            # curriculum state (gate progress etc.) → ckpt (restored on resume with consider_seen_step).
            _curr_states: Dict[str, Optional[dict]] = {
                tag: (env.handlers[i].curriculum_state()
                      if hasattr(env.handlers[i], "curriculum_state") else None)
                for i, tag in enumerate(tags)
            }
            # Reward-normalizer state (running mean/var) → ckpt. The critic is fitted on the
            # normalised scale, so every path that restores the weights must restore this
            # too (see maybe_resume).
            _rn_states: Dict[str, Optional[dict]] = {
                tag: (reward_normalizers[tag].state_dict()
                      if reward_normalizers is not None else None)
                for tag in tags
            }

            # Checkpoint: shared mode saves ONE file containing the full
            # shared state + per-tag dim metadata (no per-tag duplication).
            # Independent mode keeps the legacy one-file-per-tag layout.
            ckpt_paths: Dict[str, Path] = {}
            if shared_module is not None:
                hands_joined  = "+".join(sorted(tags))
                obs_dims_dict    = {tag: env.handlers[i].obs_dim
                                    for i, tag in enumerate(tags)}
                action_dims_dict = {tag: env.handlers[i].action_dim
                                    for i, tag in enumerate(tags)}
                shared_path = save_shared_checkpoint(
                    save_dir     = save_dir,
                    policy       = shared_module,
                    optimizer    = next(iter(optimizers.values())),  # all same instance
                    seen_steps   = seen_steps,
                    hands_joined = hands_joined,
                    task_name    = task_name,
                    env_name     = env_name,
                    obs_dims     = obs_dims_dict,
                    action_dims  = action_dims_dict,
                    extra        = {"curriculum_states": _curr_states,
                                    "reward_norm_states": _rn_states},
                )
                # All tags point at the same on-disk file (one shared blob).
                ckpt_paths = {tag: shared_path for tag in tags}
            else:
                for i, tag in enumerate(tags):
                    ckpt_paths[tag] = save_checkpoint(
                        save_dir   = save_dir,
                        policy     = policies[tag],
                        optimizer  = optimizers[tag],
                        seen_steps = seen_steps,
                        hand_name  = tag,
                        task_name  = task_name,
                        env_name   = env_name,
                        obs_dim    = env.handlers[i].obs_dim,
                        action_dim = env.handlers[i].action_dim,
                        extra      = {"curriculum_state": _curr_states[tag],
                                      "reward_norm_state": _rn_states[tag]},
                    )

            print()
            for tag in tags:
                ev = ev_per_tag[tag]
                # Task-defined EXTRA success metrics: any ``success_rate_<suffix>``
                # key run_eval returned (task-specific success variants)
                # is printed generically — new tasks' metrics show up without
                # editing this driver (mirrors the W&B startswith loop below).
                extra_sr = ''.join(
                    f'{k[len("success_rate_"):]}={v*100:5.1f}%  '
                    for k, v in sorted(ev.items())
                    if k.startswith('success_rate_'))
                # Task-defined semantic metrics (``sem_*`` — e.g. taxonomy
                # contact IoU/F1) — printed compactly on the same line.
                extra_sr += ''.join(
                    f'{k[len("sem_"):]}={v:.3f}  '
                    for k, v in sorted(ev.items()) if k.startswith('sem_'))
                eval_line = (
                    f'  ↳ [{tag:>15s}] EVAL @ seen={seen_steps:>12,}  '
                    f'mean_return={ev["mean_return"]:+.3f}  '
                    f'ep_step@done={ev["mean_first_done_step"]:.1f}  '   # avg ep_step at first term/trunc
                    f'success_rate={ev["success_rate"]*100:5.1f}%  '
                    f'(n={ev["n_success_episodes"]}/{ev["n_completed_episodes"]})  '
                    + extra_sr
                    + f'eval_sps={ev["eval_sps"]:>10,.0f}  '
                    f'reasons={ev["reset_reasons"]}  '
                    f'→ saved {ckpt_paths[tag].name}'
                )
                log_lines.append(eval_line)
                print(eval_line)
                # Record a short rollout video in the ckpt folder so the eval result can be
                # inspected **visually** (headless EGL; the training state is preserved via
                # snapshot/restore inside the recorder).
                _rec = (video_recorders or {}).get(tag)
                if _rec is not None:
                    _rec.capture(policies[tag], seen_steps, tag=tag)
                for k, st in ev['reward_term_stats'].items():
                    term_line = (f'      {k:18s}: mean={st["mean"]:+.4f}  '
                                 f'std={st["std"]:.4f}  n={st["n"]}')
                    log_lines.append(term_line)
                    print(term_line)

            # Restore the pre-eval training rollout state (see snapshot above):
            # ep_step / physics / per-world targets carry over so the next
            # rollout continues EXACTLY where it left off. ``prev_obs`` /
            # ``prev_critic_obs`` are the matching post-rollout obs (stable
            # buffers that eval does NOT touch) — do NOT overwrite them.
            # Skipped when a dedicated eval_env was used — training was untouched.
            if eval_env is None:
                for h in env.handlers:
                    h.restore_train_state()

            # W&B per-eval scalars (per-tag namespaced).
            eval_log: Dict[str, float] = {}
            for i, tag in enumerate(tags):
                ev  = ev_per_tag[tag]
                pfx = f'eval/{tag}'
                # Success-gate curriculum progress (0=base, 1=final weight).
                eval_log[f'{pfx}/curriculum_gate_progress'] = float(
                    getattr(env.handlers[i], "_gate_progress", 0.0))
                # object anti-gravity assist α (0 = off / fully decayed).
                eval_log[f'{pfx}/curriculum_antigrav_alpha'] = float(
                    getattr(env.handlers[i], "_antigrav_alpha", 0.0) or 0.0)
                # current xyz translation scale (approach-speed curriculum; base→base·frac).
                _appl = getattr(env.handlers[i], "applier", None)
                if _appl is not None:
                    eval_log[f'{pfx}/curriculum_xyz_scale'] = float(_appl.xyz_scale)
                eval_log[f'{pfx}/sps']                  = ev['eval_sps']
                eval_log[f'{pfx}/elapsed_s']            = ev['eval_elapsed_s']
                eval_log[f'{pfx}/mean_return']          = ev['mean_return']
                eval_log[f'{pfx}/mean_episode_length']  = ev['mean_episode_length']
                # avg ep_step (over worlds) at FIRST termination/truncation — the
                # episode length each world reaches before it first ends.
                eval_log[f'{pfx}/mean_first_done_step']  = ev['mean_first_done_step']
                eval_log[f'{pfx}/mean_reward_per_step'] = ev['mean_reward_per_step']
                eval_log[f'{pfx}/n_completed_episodes'] = ev['n_completed_episodes']
                eval_log[f'{pfx}/n_success_episodes']   = ev['n_success_episodes']
                eval_log[f'{pfx}/success_rate']         = ev['success_rate']
                # Extra task-defined success metrics (e.g. JAX-convention
                # ``success_rate_lift``) + the ``success_rate_strict`` alias.
                for k, v in ev.items():
                    if (k.startswith('success_rate_') or k.startswith('n_success_')
                            or k.startswith('sem_')):
                        eval_log[f'{pfx}/{k}'] = v
                # Log the DENSE reason set (all reasons, 0 included) so each
                # ``eval/<tag>/reset/<reason>`` wandb series is continuous even
                # on evals where a reason never fires. Falls back to the
                # nonzero-only Counter for older run_eval returns.
                for r_name, count in ev.get('reset_reasons_full',
                                            ev['reset_reasons']).items():
                    eval_log[f'{pfx}/reset/{r_name}'] = count
                for term, st in ev['reward_term_stats'].items():
                    eval_log[f'{pfx}/reward_term/{term}/mean'] = st['mean']
                    eval_log[f'{pfx}/reward_term/{term}/std']  = st['std']
            log_metrics(eval_log, step=seen_steps)

    # ── Final stats ─────────────────────────────────────────────────────
    print()
    wp.synchronize_device()
    t_train_end = time.perf_counter()
    elapsed     = t_train_end - t_train_start

    new_steps_this_run = max(1, seen_steps - steps_at_start)        # BUDGET (consumed) steps
    # Throughput is reported in PHYSICAL env steps: the rollout simulates
    # STEPS_PER_ITER control steps every iteration regardless of how many the
    # update consumes, so SPS reflects real sim speed (decoupled from the budget
    # cursor, which now advances by the smaller UPDATE_STEPS_PER_ITER).
    physical_steps_this_run = max(1, n_iter_to_run * STEPS_PER_ITER)
    ctrl_sps = physical_steps_this_run / max(elapsed, 1e-9)
    sim_sps  = ctrl_sps * env.sim_nstep

    n_iter_done = max(1, n_iter_to_run)
    phase_lines = []
    for k in ('rollout', 'gae', 'update'):
        avg_ms = phase_totals[k] / n_iter_done * 1000.0
        share  = phase_totals[k] / max(phase_totals['total'], 1e-9) * 100.0
        phase_lines.append(f'    {k:7s}: avg {avg_ms:>6.1f} ms / iter   ({share:>5.1f} % of train time)')
    phase_breakdown = '\n'.join(phase_lines)

    summary = (
        f'\n=== Training summary ===\n'
        f'  handlers (tags)         : {len(tags)} — {tags}\n'
        f'  iterations this run     : {n_iter_to_run:,}\n'
        f'  new steps this run      : {new_steps_this_run:,}\n'
        f'  total seen control steps: {seen_steps:,}\n'
        f'  wall-clock elapsed      : {elapsed:.1f} s  ({elapsed/60:.1f} min)\n'
        f'  control steps / sec     : {ctrl_sps:>14,.0f}\n'
        f'  sim    steps / sec      : {sim_sps:>14,.0f}\n'
        f'  per-iter wall time avg  : {(phase_totals["total"]/n_iter_done)*1000:>6.1f} ms\n'
        f'  phase breakdown:\n{phase_breakdown}\n'
        f'  checkpoints saved to    : {save_dir}\n'
    )
    log_lines.append(summary)
    print(summary)

    log_metrics({
        'summary/total_iters'      : n_iter_done,
        'summary/seen_steps'       : seen_steps,
        'summary/elapsed_s'        : elapsed,
        'summary/control_sps_avg'  : ctrl_sps,
        'summary/sim_sps_avg'      : sim_sps,
        'summary/avg_iter_time_ms' : (phase_totals['total'] / n_iter_done) * 1000.0,
        'summary/share/rollout'    : phase_totals['rollout'] / max(phase_totals['total'], 1e-9),
        'summary/share/gae'        : phase_totals['gae']     / max(phase_totals['total'], 1e-9),
        'summary/share/update'     : phase_totals['update']  / max(phase_totals['total'], 1e-9),
    }, step=seen_steps)

    # Append to (or create) the persistent training log.
    log_path = save_dir / f'{save_dir.name}_train_log.txt'
    with open(log_path, 'a') as f:
        f.write('\n'.join(log_lines) + '\n')
    print(f'  log appended            : {log_path}')

    return seen_steps


# ──────────────────────────────────────────────────────────────────────────
# Metrics persistence — tee every logged dict to a local JSONL file
# ──────────────────────────────────────────────────────────────────────────

def make_metrics_file_logger(
    base_log_metrics: Callable[..., None],
    path: Path,
    config: Optional[dict] = None,
) -> Tuple[Callable[..., None], Callable[[], None]]:
    """Wrap ``log_metrics`` so every logged dict is ALSO appended to a local
    JSONL file (one record per call: ``{"step": step, **metrics}``).

    Fully general / task-agnostic: it persists whatever ``train_loop`` already
    sends to W&B — per-iter train scalars (``train/<tag>/loss/{policy,value,
    entropy}``, ``reward_mean``, ``reward_term/<term>/{mean,std}``, ``grad_norm``,
    ``adv_std``, reset reasons) and per-eval scalars (``eval/<tag>/success_rate``,
    ``success_rate_lift``, ``reward_term/<term>/{mean,std}`` …) plus the run
    ``summary/*``. JSONL handles the per-task varying key sets without a fixed
    schema, so it works for any hand / task / reward layout.

    ``config``: when given (the FULLY RESOLVED run cfg), it is written as the
    FIRST record ``{"record": "config", "config": {...}}`` so the metrics file is
    SELF-CONTAINED — the reward-term weights (``handler.W_*``) and every other
    hyperparameter used for the run live in the log itself, and analysis tools
    need not depend on a separate ``config.yaml`` snapshot. Sent to the file
    only, NOT to W&B (it is not a scalar metric).

    Returns ``(logger, close)``; call ``close`` once at the end (the run's
    ``finally``) to flush + close the handle.
    """
    path = Path(path)
    fh = open(path, "a")
    if config is not None:
        fh.write(json.dumps({"record": "config", "config": config},
                            ensure_ascii=False, default=str) + "\n")
        fh.flush()

    def _json_safe(v):
        if isinstance(v, bool) or v is None or isinstance(v, (int, float, str)):
            return v
        if hasattr(v, "item"):          # numpy / torch scalar
            try:
                return v.item()
            except Exception:
                pass
        return float(v) if hasattr(v, "__float__") else str(v)

    def logger(metrics: dict, step=None) -> None:
        base_log_metrics(metrics, step=step)           # original sink (W&B / no-op)
        rec = {"step": (int(step) if step is not None else None)}
        rec.update({k: _json_safe(v) for k, v in metrics.items()})
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()                                     # crash-safe: survive a killed run

    def close() -> None:
        try:
            fh.close()
        except Exception:
            pass

    return logger, close


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def _opt_cfg_dict(cfg, key: str) -> dict:
    """``cfg[key]`` as a plain dict — **an empty dict when the key is absent**.

    The ``OmegaConf.to_container(cfg.get(key, {}))`` pattern breaks when the key is
    missing: ``cfg.get`` returns a **plain dict** rather than a DictConfig and
    ``to_container`` rejects it ("Input cfg is not an OmegaConf config object"),
    so every config that omits the section dies right at startup.
    Always use this helper when reading optional config sections.
    """
    v = cfg.get(key, None)
    if v is None:
        return {}
    if OmegaConf.is_config(v):
        return OmegaConf.to_container(v, resolve=True)      # type: ignore[return-value]
    return dict(v) if v else {}


def main(
    config_name: str = "grasping_policy_teacher",
    overrides=None,
    load_from: Optional[str] = None,
    keep_ckpt_std: bool = False,
    load_optimizer: bool = False,
) -> None:
    """Build everything from ``config/training/<config_name>.yaml`` and
    run the PPO training loop. Generalises to single-hand and multi-hand
    (cross-embodiment) automatically.

    ``overrides``: optional list of Hydra-style ``key=value`` strings
    (e.g. ``["nworld=32", "Training.n_obj=3"]``). Baked into the
    resolved cfg before the orchestrator builds, so the SAVE_DIR snapshot
    captures them — inference reproduces the same env automatically.

    ``load_from``: optional arbitrary checkpoint dir or ``.pt`` file for a
    **weight-only warm start** — decoupled from the config-derived SAVE_DIR.
    Loads weights (NOT seen_steps; fresh optimizer unless ``load_optimizer``),
    so training proceeds from step 0 for a fresh ``total_new_steps`` budget,
    writing checkpoints to the normal config SAVE_DIR. Takes precedence over
    the config ``resume.*`` block (which is the "continue the SAME run after a
    crash" path). ``load_optimizer=True`` also restores the optimizer state.
    """

    # 1) Orchestrator + yaml ────────────────────────────────────────────
    orchestrator, cfg = build_orchestrator(config_name, overrides=overrides)
    if overrides:
        print(f'overrides  : {list(overrides)}')

    seed = int(cfg.training.get("seed", 42))              # type: ignore
    random.seed(seed); np.random.seed(seed)

    hands     = list(orchestrator.hand_names)        # e.g. ["tesollo"] or ["tesollo","robotis_sh5"]
    task_name = str(cfg.task_name)                        # type: ignore
    env_name  = str(cfg.env_name)                         # type: ignore
    geom_collision_type = str(cfg.get("geom_collision_type", "mesh")) # type: ignore

    hands_joined = "+".join(sorted(hands))
    print(f'config     : {config_name}.yaml')
    print(f'hands      : {hands_joined}   (n_handlers={len(hands)})')
    print(f'task / env : {task_name} / {env_name}')

    # Single-arm setup.
    if cfg.env_type == "single_hand":                  # type: ignore
        orchestrator.build_sub_env(
        with_mjwarp=True, for_inference=True,
            geom_collision_type=geom_collision_type,
            # Training.object_source: 'dataset' (mesh) | 'primitive' (procedural)
            obj_spec_provider=obj_spec_provider_from_config(cfg, verbose=True),
        )
    else:
        raise NotImplementedError(
            f"env_type={cfg.env_type!r} is not included in grit_share "
            "(single_hand only)")
    print('rl_envs    :', list_rl_envs())
    print('policies   :', list_policies())

    # 2) RL env + handlers (knobs applied to ALL handlers) ──────────────
    env, rl_env_cfg, _handler_cfg, tags = build_rl_env_and_handlers(orchestrator, cfg, env_name)

    # 3) Per-tag policies + optimizers + rollout buffer ─────────────────
    # ``shared_module`` is None in independent mode, or the underlying
    # cross-embodiment :class:`nn.Module` when ``cfg.policy.name``
    # starts with ``"shared_"``. All ``policies[tag]`` values are then
    # lightweight TagViews of that single module.
    policies, optimizers, policy_cfg, shared_module = build_policies_and_optimizers(cfg, env, tags)
    buffer    = build_rollout_buffer(cfg, env)
    train_cfg = OmegaConf.to_container(cfg.training, resolve=True) # type: ignore

    if shared_module is not None:
        n_params = sum(p.numel() for p in shared_module.parameters())
        print(f'policy mode  : SHARED cross-embodiment '
              f'({type(shared_module).__name__})')
        print(f'  total params (single shared module) : {n_params:,}')
        print(f'  per-tag adapters / heads — input_dim={shared_module.latent_dim}, '
              f'hidden_dims={shared_module.hidden_dims}')
    else:
        # Independent per-tag — sum across each tag's own module.
        n_params = sum(sum(p.numel() for p in pol.parameters()) for pol in policies.values()) # type: ignore
        print(f'policy mode  : independent per-tag '
              f'({len(policies)} network(s))')
        print(f'  total params (sum of {len(policies)} networks) : {n_params:,}')
    print(buffer)

    # 4) Baseline eval (per-tag sanity) ─────────────────────────────────
    print('eval (untrained policies):')
    for i, tag in enumerate(tags):
        baseline = run_eval(
            policies[tag], env.handlers[i],
            eval_length=env.handlers[i].MAX_EPISODE_STEPS,
        )
        # Validate the reward_gate settings (fail fast on metric typos; no bump) — done
        # once here so errors surface at startup rather than at the first periodic eval.
        if hasattr(env.handlers[i], "on_eval_metrics"):
            env.handlers[i].on_eval_metrics(baseline, validate_only=True)
        print(f'  [{tag:>15s}]  mean_return={baseline["mean_return"]:+.3f}  '
              f'success_rate={baseline["success_rate"]:.3f}  '
              f'(n={baseline["n_success_episodes"]}/{baseline["n_completed_episodes"]})  '
              f'eval_sps={baseline["eval_sps"]:,.0f}')

    # 4b) Reward normalizer (optional; cfg.training.reward_norm) ────────
    # Per-tag running-statistics reward scaling for the PPO buffer — see
    # grit/training/reward_normalizer.py and the knob block in base.yaml.
    _rn_cfg = train_cfg.get("reward_norm") or {}                        # type: ignore
    reward_normalizers: Optional[Dict[str, RewardNormalizer]] = None
    if bool(_rn_cfg.get("enabled", False)):
        reward_normalizers = {
            tag: RewardNormalizer.from_config(
                _rn_cfg,
                num_envs      = env.handlers[i].NWORLD,
                device        = env.torch_device,
                default_gamma = float(train_cfg["gamma"]),              # type: ignore
            )
            for i, tag in enumerate(tags)
        }  # type: ignore[assignment]
        _rn0 = next(iter(reward_normalizers.values()))                  # type: ignore
        print(f'reward norm  : ON  (mode={_rn0.mode}  clip={_rn0.clip}  '
              f'gamma={_rn0.gamma}  center={_rn0.center})')
    else:
        print('reward norm  : off  (training.reward_norm.enabled=false)')

    # 5) Save dir + resume ──────────────────────────────────────────────
    resume_cfg = OmegaConf.to_container(cfg.resume, resolve=True) # type: ignore
    # Optional folder-name disambiguation (cfg.output.*):
    #   include_sim_in_name=True  → append ``_nse{N_SUB_ENV}_nw{NWORLD}``
    #   run_tag=<str>             → append ``_{run_tag}``
    # so otherwise-identical {hand}_{task}_{env} runs land in distinct dirs.
    # Inference reconstructs the SAME tail via ``--name-suffix`` (see the
    # printed hint below). resume forking still appends ``_resume`` last.
    out_cfg = (OmegaConf.to_container(cfg.output, resolve=True)            # type: ignore
               if cfg.get("output") is not None else {})
    assert isinstance(out_cfg, dict)
    _suffix_parts: List[str] = []
    if bool(out_cfg.get("include_sim_in_name", False)):
        _suffix_parts.append(f'_nse{int(orchestrator.N_SUB_ENV)}_nw{int(orchestrator.NWORLD)}')
    _run_tag = out_cfg.get("run_tag", None)
    if _run_tag is not None and str(_run_tag).strip() not in ("", "null", "None"):
        _suffix_parts.append(f'_{str(_run_tag).strip()}')
    name_suffix = "".join(_suffix_parts)

    save_dir_base, load_dir, save_dir, resume_enabled = resolve_save_dir(
        orchestrator, hands, task_name, env_name, rl_env_cfg, resume_cfg, # type: ignore
        name_suffix=name_suffix,
    )
    print(f'SAVE_DIR     : {save_dir}'
          + ('' if save_dir == load_dir else '   (resumed → fork)'))
    # ── overwrite warning ──────────────────────────────────────────────
    # SAVE_DIR is **shared by all runs of the same config** and file names carry only
    # the step, so a new run on the same step grid silently overwrites the previous
    # run's ckpts. Since ckpts with a different obs_dim cannot be resumed from anyway,
    # warn **before starting** when such ckpts remain, giving a chance to move them.
    try:
        import torch as _t
        _cur_obs = int(env.handlers[0].obs_dim)
        _stale = []
        for _f in sorted(Path(save_dir).glob("*.pt")):
            if _f.stat().st_size < 1_000_000:
                continue
            try:
                _o = int(_t.load(_f, map_location="cpu", weights_only=False)["obs_dim"])
            except Exception:
                continue
            if _o != _cur_obs:
                _stale.append((_f.name, _o))
        if _stale:
            print(f'  WARNING: this folder holds {len(_stale)} ckpt(s) with a different obs_dim '
                  f'(current obs={_cur_obs}, existing obs={sorted({o for _n, o in _stale})}).')
            print(f'    Files with the same step name will be overwritten by this run — move them now to keep them.')
    except Exception as _e:                       # a failed warning must not block training
        print(f'  (ckpt overwrite check skipped: {type(_e).__name__})')
    if name_suffix:
        print(f'  name_suffix : {name_suffix}   '
              f'(select the same folder in evaluate.py with --name-suffix "{name_suffix}")')

    # ── warm start declared in yaml (CLI --load-from takes precedence) ─────
    # When "this config always starts from that ckpt" is part of the recipe (e.g.
    # DAgger (BC) pretrain → post-RL fine-tune), passing a long path on the CLI every
    # time is easy to forget (and forgetting silently trains from scratch).
    # Relative paths are resolved from the project root — works from scripts/ too.
    if not load_from:
        _cfg_lf = cfg.get("load_from", None)                     # type: ignore
        if _cfg_lf:
            _p = Path(str(_cfg_lf))
            load_from = str(_p if _p.is_absolute() else Path(PROJECT_ROOT) / _p)
            print(f'\nload_from (yaml): {_cfg_lf}')

    if load_from:
        # Explicit warm start from an arbitrary path wins over config resume.
        if resume_enabled:
            print('  (note: --load-from given → ignoring config resume.enabled=true)')
        print(f'\n--load-from  : {load_from}   '
              f'(weight-only warm start; seen_steps=0, '
              f'optimizer={"loaded" if load_optimizer else "fresh"})')
        load_weights_from(
            load_from,
            tags               = tags,
            env                = env,
            policies           = policies,  # type: ignore
            task_name          = task_name,
            env_name           = env_name,
            shared_module      = shared_module,
            optimizers         = optimizers,
            load_optimizer     = load_optimizer,
            reward_normalizers = reward_normalizers,
        )
        # ── restore the exploration std on a BC (DAgger) ckpt warm start ──────
        # BC trains only with an MSE on ``forward()`` (= tanh(loc)), so ``std_param``
        # stays **at its initial value** (raw≈0). Loading it verbatim from the state_dict
        # overrides ``policy.init_std`` from the yaml and the effective exploration std
        # jumps to ``softplus(0)+min_std`` (~0.89), which is far too large for a precise
        # tracking fine-tune: performance rises after the warm start and then regresses.
        # → when the yaml specifies ``init_std``, reset to that value after loading.
        _init_std = cfg.policy.get("init_std", None)             # type: ignore
        if _init_std is not None and not keep_ckpt_std:
            import torch as _t
            for _tag, _pol in policies.items():
                _sp = getattr(_pol, "std_param", None)
                if _sp is None:                                   # state-dependent std
                    continue
                _before = float((_t.nn.functional.softplus(_sp)
                                 + _pol.min_std).mean() * _pol.var_scale)
                with _t.no_grad():
                    _sp.copy_(_pol._init_std_param(
                        _pol.action_dim, float(_init_std),
                        _pol.min_std, _pol.var_scale).to(_sp.device))
                _after = float((_t.nn.functional.softplus(_sp)
                                + _pol.min_std).mean() * _pol.var_scale)
                print(f'  [{_tag}] exploration std restored: {_before:.3f} → {_after:.3f} '
                      f'(policy.init_std; disable with --keep-ckpt-std)')
                # Separate σ for the finger dimensions: action = [wrist 9 | finger n_ctrl].
                # With finger_scale at 0.2 rad/step, the same σ produces finger noise that
                # breaks the grasp (much higher drop rate in stochastic rollouts than with
                # deterministic actions). Keep the wrist σ and lower only the fingers.
                # std_param is per-dim (action_dim,) anyway.
                _fs = cfg.policy.get("init_std_finger", None)          # type: ignore
                if _fs is not None:
                    _nf = int(getattr(env.handlers[0], "n_ctrl", 0))
                    if 0 < _nf < int(_pol.action_dim):
                        with _t.no_grad():
                            _full = _pol._init_std_param(
                                _pol.action_dim, float(_fs),
                                _pol.min_std, _pol.var_scale).to(_sp.device)
                            _sp[-_nf:] = _full[-_nf:]
                        _sig = ((_t.nn.functional.softplus(_sp) + _pol.min_std)
                                * _pol.var_scale)
                        print(f'  [{_tag}] separate finger σ: wrist '
                              f'{float(_sig[:-_nf].mean()):.3f} / finger '
                              f'{float(_sig[-_nf:].mean()):.3f} '
                              f'(policy.init_std_finger={float(_fs):g}, n_ctrl={_nf})')
        seen_steps = 0
    else:
        seen_steps = maybe_resume(
            resume             = resume_enabled,
            consider_seen_step = bool(resume_cfg.get("consider_seen_step", False)), # type: ignore
            load_dir           = load_dir,
            save_dir           = save_dir,
            tags               = tags,
            env                = env,
            policies           = policies, # type: ignore
            optimizers         = optimizers,
            task_name          = task_name,
            env_name           = env_name,
            shared_module      = shared_module,        # None in independent mode
            reward_normalizers = reward_normalizers,
        )

    # 6) W&B + snapshot ─────────────────────────────────────────────────
    log_metrics, finish_wandb = setup_wandb(
        cfg,
        hands_joined = hands_joined,
        tags         = tags,
        task_name    = task_name,
        env_name     = env_name,
        save_dir     = save_dir,
        env          = env,
        policies     = policies, # type: ignore 
        rl_env_cfg   = rl_env_cfg, # type: ignore
        train_cfg    = train_cfg, # type: ignore
        policy_cfg   = policy_cfg, # type: ignore
    )
    cfg_path = save_run_snapshot(cfg, env, save_dir)
    print(f'config saved : {cfg_path}')

    # Persist every logged metric (train per-iter + eval + summary) to a local
    # JSONL alongside the checkpoints — independent of W&B, for offline analysis.
    # The fully-resolved cfg (incl. handler.W_* reward weights) is written as the
    # FIRST record so the log is self-contained — analysis reads weights from the
    # metrics file itself, no dependence on the config.yaml snapshot.
    metrics_path = save_dir / f'{save_dir.name}_metrics.jsonl'
    _resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    log_metrics, close_metrics_file = make_metrics_file_logger(
        log_metrics, metrics_path, config=_resolved_cfg)
    print(f'metrics log  : {metrics_path}')

    # 6b) Object resampler ──────────────────────────────────────────────
    # Enabled only when the object set is NOT fully pinned AND cfg.training.
    # resample_objects_every > 0. Rebuilds the env with a fresh object draw once
    # per eval period (see grit.training.resample.ObjectResampler). ``None`` when
    # disabled → train_loop skips all resample logic.
    resampler = ObjectResampler.from_config(orchestrator, cfg, env_name, env, train_cfg)

    # 6c) Fixed-object eval env ──────────────────────────────────────────
    # ``eval_obj_idxs`` (top-level cfg): when set, eval runs on this FIXED object
    # set so metrics stay comparable across training regardless of object
    # resampling. ``null`` → eval falls back to the (possibly resampled) training
    # env. ``eval_nworld`` (null → min(nworld, 256)) keeps the second env small.
    eval_obj_idxs = cfg.get("eval_obj_idxs", None)                      # type: ignore
    eval_env = None
    if eval_obj_idxs:
        eval_obj_idxs = [int(i) for i in eval_obj_idxs]
        eval_nworld   = int(cfg.get("eval_nworld") or min(int(orchestrator.NWORLD), 256))  # type: ignore
        eval_env      = build_eval_env(orchestrator, env_name, eval_obj_idxs, nworld=eval_nworld)
        print(f'eval env       : FIXED objects {eval_obj_idxs} @ nworld={eval_nworld}')
    else:
        print('eval env       : (none) → eval uses the training env objects')

    # 7) Train ──────────────────────────────────────────────────────────
    # Hand the env to train_loop through a 1-element box and drop main's own
    # reference, so a rebuild can actually free the *original* env (otherwise
    # main's frame pins it for the whole run → 2× GPU on the first resample).
    # 7a) BC (DAgger) pretrain — directly on the same network, no ckpt round trip.
    #     Skipped when trained weights are already loaded:
    #       * ``--load-from`` (explicit warm start)
    #       * config ``resume.enabled`` (continuing the same run after a crash)
    #     Without the resume check, BC would **overwrite** the weights restored by
    #     ``maybe_resume`` and the resumed run would silently start from scratch.
    if not load_from and not resume_enabled:
        bc_pretrain(env, policies, tags,                       # type: ignore
                    _opt_cfg_dict(cfg, "pretrain"), env.torch_device)

    # 7b) Eval rollout video (cfg.eval_video) ───────────────────────────
    _vid_cfg = _opt_cfg_dict(cfg, "eval_video")
    video_recorders = None
    if _vid_cfg.get("enabled", False):
        from grit.training.eval_video import EvalVideoRecorder
        video_recorders = {
            tag: EvalVideoRecorder(
                env.handlers[i], Path(save_dir) / "eval_videos",
                n_worlds=int(_vid_cfg.get("n_worlds", 4)),
                width=int(_vid_cfg.get("width", 320)),
                height=int(_vid_cfg.get("height", 240)),
                n_frames=int(_vid_cfg.get("n_frames", 150)),
                fps=int(_vid_cfg.get("fps", 30)),
                every=int(_vid_cfg.get("every", 1)),
                camera=_vid_cfg.get("camera") or None,
                distance=float(_vid_cfg.get("distance", 1.2)),
                elevation=float(_vid_cfg.get("elevation", -20.0)),
                azimuth=float(_vid_cfg.get("azimuth", 150.0)))
            for i, tag in enumerate(tags)}
        print(f'eval videos    : {Path(save_dir) / "eval_videos"}  '
              f'(nworld={_vid_cfg.get("n_worlds", 4)}, '
              f'{_vid_cfg.get("n_frames", 150)} frame, every {_vid_cfg.get("every", 1)} eval)')

    _env_box = [env]
    del env
    try:
        train_loop(
            env           = _env_box[0],
            eval_env      = eval_env,
            tags          = tags,
            policies      = policies,  # type: ignore
            optimizers    = optimizers,
            buffer        = buffer,
            train_cfg     = train_cfg, # type: ignore
            video_recorders = video_recorders,
            save_dir      = save_dir,
            task_name     = task_name,
            env_name      = env_name,
            seen_steps    = seen_steps,
            log_metrics   = log_metrics,
            print_every   = int(train_cfg.get("print_every", 10)), # type: ignore
            shared_module = shared_module,        # None in independent mode
            resampler     = resampler,
            env_box       = _env_box,
            reward_normalizers = reward_normalizers,
        )
    finally:
        close_metrics_file()
        finish_wandb()


# ──────────────────────────────────────────────────────────────────────────
# python scripts/train.py -c <yaml> 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPO training driven by a single YAML under config/training/. "
                    "Supports single-hand and multi-hand (cross-embodiment) automatically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/train.py -c grasping_policy_teacher\n"
            "  python scripts/train.py -c grasping_policy_async_best_w_ground_force --overrides 'using_hand_name_list=[tesollo]'\n"
            "  python scripts/train.py -c grasping_policy_teacher --overrides nworld=1024 handler.termination.MAX_EPISODE_STEPS=300\n"
        ),
    )
    parser.add_argument(
        "-c", "--config",
        default="grasping_policy_teacher",
        help="YAML name under config/training/ (without .yaml). "
             "Default: grasping_policy_teacher",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="Hydra-style cfg overrides applied at compose() time. "
             "Dotted keys (``Training.n_obj=3``) and list values "
             "(``'policy.hidden_dim=[256,256]'``) supported — quote the "
             "argument when it contains characters the shell would expand "
             "(``[``, ``]``, ``*``, …). Resolved values are baked into "
             "``SAVE_DIR/config.yaml`` so inference reproduces the run.",
    )
    parser.add_argument(
        "--load-from",
        default=None,
        metavar="PATH",
        help="Weight-only warm start from an ARBITRARY checkpoint dir or .pt "
             "file, independent of the config-derived SAVE_DIR folder name. "
             "Loads weights only (seen_steps=0, fresh optimizer) and saves new "
             "checkpoints to the normal config SAVE_DIR. Use this to initialise "
             "a run from another run's weights. (For continuing the SAME run "
             "after a crash, use the config resume.enabled / consider_seen_step "
             "block instead.)",
    )
    parser.add_argument(
        "--keep-ckpt-std",
        action="store_true",
        default=False,
        help="With --load-from, keep the checkpoint's std_param as-is. "
             "By default a BC/DAgger ckpt's untrained std_param is reset to "
             "policy.init_std (BC never trains std → exploration std would "
             "blow up to softplus(0)+min_std).",
    )
    parser.add_argument(
        "--load-optimizer",
        action="store_true",
        help="With --load-from, also restore the optimizer state (default: "
             "fresh optimizer). seen_steps still starts at 0.",
    )
    args = parser.parse_args()
    main(
        config_name    = args.config,
        keep_ckpt_std  = args.keep_ckpt_std,
        overrides      = args.overrides,
        load_from      = args.load_from,
        load_optimizer = args.load_optimizer,
    )
