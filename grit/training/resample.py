"""Object-resample support for the PPO training loop (``scripts/train.py``).

Keeps the "rebuild the RL env with a FRESH object draw once per evaluation
period" feature out of the training loop so ``train.py`` stays a thin driver.
Everything memory-related — OOM detection, GPU cache hygiene, warp mempool
release, and the OOM-safe *build → reset → commit / rollback* dance — lives here.

Usage (in ``scripts/train.py``)::

    resampler = ObjectResampler.from_config(orchestrator, cfg, env_name, env, train_cfg)
    ...
    if resampler is not None and resampler.due(it, it_local, EVAL_EVERY):
        env = resampler.resample(it, env, tags, prev_obs, prev_critic_obs,
                                 train_term_tracker, async_ppo=ASYNC_PPO, env_box=env_box)
"""
from __future__ import annotations

import contextlib
import gc
import io
from typing import Dict, List, Optional, Sequence

import torch
import warp as wp

from grit.training.builders      import build_rl_env_and_handlers
from grit.training.reward_tracker import RewardTermReturnTracker


# ──────────────────────────────────────────────────────────────────────────
# GPU-memory helpers (shared by the loop's rollout OOM guard too)
# ──────────────────────────────────────────────────────────────────────────
def is_oom_error(e: BaseException) -> bool:
    """True if ``e`` is a GPU out-of-memory. warp raises a plain ``RuntimeError``
    whose message is ``"Failed to allocate N bytes on device 'cuda:0'"`` — it does
    NOT contain the string "out of memory" (that only appears in warp's separate
    stderr banner), so matching on that alone silently misses every warp OOM."""
    m = str(e).lower()
    return ("out of memory" in m
            or "failed to allocate" in m
            or "cuda error" in m
            or "cudamalloc" in m)


def free_gpu_memory(sync: bool = True) -> None:
    """Release cached device memory: Python gc → torch caching allocator →
    (optional) warp device sync so its stream-ordered pool reclaims freed blocks.
    torch and warp hold SEPARATE device pools, so both must be nudged."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if sync:
        try:
            wp.synchronize_device()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────
# ObjectResampler
# ──────────────────────────────────────────────────────────────────────────
class ObjectResampler:
    """Rebuilds ``orchestrator``'s variants with a fresh object draw once per
    evaluation period. hand / n_obj / obs_dim / action_dim stay fixed, so the
    policy / optimizer / rollout buffer remain valid — only the env and the
    per-tag rollout state (``prev_obs`` / ``prev_critic_obs`` / trackers) are
    re-wired. Rebuilds are OOM-safe: on failure the previous objects are kept.
    """

    def __init__(self, orchestrator, cfg, env_name: str,
                 obs_dims: Sequence[int], action_dims: Sequence[int]):
        self.orch         = orchestrator
        self.cfg          = cfg
        self.env_name     = env_name
        self._obs_dims    = list(obs_dims)
        self._action_dims = list(action_dims)

    # ── factory: decide enabled + prime warp's mempool ──────────────────
    @classmethod
    def from_config(cls, orchestrator, cfg, env_name: str, env,
                    train_cfg: dict, *, verbose: bool = True) -> "Optional[ObjectResampler]":
        """Return a resampler, or ``None`` when resampling is disabled.

        Enabled only when the object set is NOT fully pinned (``obj_idxs`` unset
        or a pool larger than ``n_sub_env``) AND ``training.resample_objects_every
        > 0``. The knob is an on/off flag — the actual cadence is the eval period
        (see :meth:`due`).
        """
        knob = (0 if orchestrator.objects_fully_pinned
                else int(train_cfg.get("resample_objects_every", 0)))
        if knob <= 0:
            if verbose:
                print("object resample : off (objects fully pinned or knob=0)")
            return None
        # warp mempool release-threshold 0 → freed blocks return to the driver on
        # sync, limiting cross-rebuild fragmentation that otherwise OOMs.
        try:
            dev = env.device
            if wp.is_mempool_supported(dev):
                wp.set_mempool_release_threshold(dev, 0)
        except Exception as e:  # non-fatal
            if verbose:
                print(f"  (mempool release-threshold set skipped: {e})")
        if verbose:
            pool = ("unset" if orchestrator.obj_idxs is None
                    else f"pool={len(orchestrator.obj_idxs)}")
            print(f"object resample : once per EVAL period  "
                  f"(obj_idxs {pool} > n_sub_env={orchestrator.N_SUB_ENV} → not fully "
                  f"pinned; rebuild amortised over EVAL_EVERY iters)")
        return cls(orchestrator, cfg, env_name, env.obs_dims, env.action_dims)

    # ── cadence: eval period (skip it_local==0 — env already fresh) ─────
    def due(self, it: int, it_local: int, eval_every: int) -> bool:
        return it_local > 0 and (it % max(1, eval_every) == 0)

    # ── quiet rebuild + dim guard ───────────────────────────────────────
    def _build_fresh_env(self):
        # The build helpers print a full env-init block every call; redirect it
        # so the training log shows only the concise resample line + real errors.
        with contextlib.redirect_stdout(io.StringIO()):
            self.orch.resample_objects()
            env, *_ = build_rl_env_and_handlers(self.orch, self.cfg, self.env_name)
        if (list(env.obs_dims) != self._obs_dims
                or list(env.action_dims) != self._action_dims):
            raise RuntimeError(
                f"resample changed obs/action dims (obs {self._obs_dims}→"
                f"{list(env.obs_dims)}, action {self._action_dims}→"
                f"{list(env.action_dims)}); keep hand / n_obj / task fixed.")
        return env

    # ── OOM-safe rebuild + re-wire ──────────────────────────────────────
    def resample(self, it: int, env, tags: List[str],
                 prev_obs: Dict[str, torch.Tensor],
                 prev_critic_obs: Dict[str, torch.Tensor],
                 train_term_tracker: Dict[str, RewardTermReturnTracker],
                 *, async_ppo: bool, env_box: Optional[list] = None):
        """Rebuild ``env`` with a fresh object draw and re-wire per-tag rollout
        state in place. Builds into TEMP vars and commits only on FULL success
        (the heavy-set OOM usually surfaces in ``reset()``), so an OOM leaves the
        previous env + state untouched. Returns the (new or unchanged) env.
        """
        prev_env = env
        try:
            free_gpu_memory()                       # give warp room before the rebuild
            new_env  = self._build_fresh_env()
            obs_list = new_env.reset()
            n_obs, n_crit, n_trk = {}, {}, {}
            for i, tag in enumerate(tags):
                h = new_env.handlers[i]
                n_obs[tag] = obs_list[i].clone()
                if async_ppo:
                    n_crit[tag] = h.critic_obs_torch.clone()
                n_trk[tag] = RewardTermReturnTracker(h.metrics, h.NWORLD, new_env.torch_device)
            # ── commit (nothing above threw) ──
            env = new_env
            for tag in tags:
                prev_obs[tag] = n_obs[tag]
                if async_ppo:
                    prev_critic_obs[tag] = n_crit[tag]
                train_term_tracker[tag] = n_trk[tag]
            prev_env = None                         # release the old env
            if env_box is not None:
                env_box[0] = None                   # drop the caller's ref to the original env too
            free_gpu_memory()
            print(f'[iter {it}] resampled objects → '
                  f'{getattr(env.orchestrator, "sampled_obj_indices", "?")}')
        except RuntimeError as e:
            if not is_oom_error(e):
                raise
            # OOM during rebuild/reset: nothing was committed, so env + prev_obs /
            # trackers still point at the intact previous env. Keep them, skip.
            env = prev_env
            free_gpu_memory()
            print(f'[iter {it}] ⚠ resample skipped (GPU OOM during rebuild) — '
                  f'kept current objects, will retry next eval period')
        return env
