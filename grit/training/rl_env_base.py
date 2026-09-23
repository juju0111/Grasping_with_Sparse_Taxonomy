"""Base RL environment for warp-based parallel hand-grasping tasks.

Mirrors the structural pattern of
``thirdParty/taxonomy_guided_RL/package/mjx_helper/hand_rl_envs/base_env_0.py``
(``Warp_General_Single_Obj_Base_Env``), but adapted to this project's
**warp + torch** stack and to ``orchestrator.env_list``.

Why the multi-handler design
----------------------------
``Env_Orchestrator.env_list`` may hold **multiple sub-envs** (one per hand
type — see ``grit_multi_hand`` config). Each sub-env has its
own ``m`` / ``d`` / ``NWORLD`` / ``hand_util`` and they can have *different*
obs/action dims.

The RL env therefore is composed of two layers:

* :class:`SubEnvHandler` — owns ONE sub-env's resources:
  ``WarpActionApplier``, ``WarpPerWorldCondition``, captures, ep counters,
  snapshot buffers, and the task-specific obs / reward / done kernels.
  Subclass per task.
* :class:`BaseRLEnv` — wraps ``list[SubEnvHandler]`` (one per
  ``orchestrator.env_list`` entry) and dispatches ``reset`` / ``step`` /
  ``per_world_reset_if_done`` over them.

  * ``num_envs`` = ``sum(handler.NWORLD for handler in handlers)``.
  * single-handler convenience: ``env.handlers[0]`` exposes the same
    single-tensor API the previous version of this module had.

Registry::

    @register_rl_env("grasping")
    class GraspingEnv(BaseRLEnv):
        HANDLER_CLS = GraspingHandler

    env = make_rl_env(
        "grasping",
        orchestrator=orchestrator,            # may have N>=1 sub-envs
        sim_nstep=5,
    )
    obs_list = env.reset()                    # list of length len(handlers)
    obs_list, rew_list, done_list, info_list = env.step(actions_list)
    env.per_world_reset_if_done()             # sync-free async reset

For the single-hand case (most common), call the handler directly to keep
single-tensor returns::

    h = env.handlers[0]
    obs = h.reset()
    obs, rew, done, info = h.step(action_torch)
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import mujoco as _mj
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from grit.training.orchestrator.base import _grit_r2quat_wxyz
from grit.util.warp_action import WarpActionApplier
from grit.util.warp_per_world_condition import (
    WarpPerWorldCondition,
    register_hand_pose_target,
)
from grit.training.hand_init import HandFingerInitializer


# ──────────────────────────────────────────────────────────────────────────
# NaN / Inf hardening (see _sanitize_obs / _flag_nonfinite_worlds /
# _debug_check_finite). A single non-finite obs would otherwise poison the
# replay buffer and, via loss.backward()/optimizer.step(), every network
# parameter — permanently killing the run. Set GRIT_DEBUG_NAN=1 to trace the
# FIRST non-finite tensor (action vs physics-state vs obs) per step.
# ──────────────────────────────────────────────────────────────────────────
_GRIT_DEBUG_NAN   = bool(int(os.environ.get("GRIT_DEBUG_NAN", "0") or "0"))
_OBS_SANITIZE_BIG = 1.0e4   # ±Inf / overflow obs values are clamped to ± this before the policy

# Test hook: with GRIT_DEBUG_NAN=1, inject a NaN into obs[world 0, this column]
# ONCE on the first step to validate the detection + .log path end-to-end.
# Empty string → disabled (normal runs). Not used in production.
_GRIT_NAN_INJECT_COL = os.environ.get("GRIT_NAN_INJECT_COL", "")


def _nan_log_path() -> Path:
    """Resolve the NaN log file path. Override with ``GRIT_NAN_LOG`` (abs or rel);
    default ``<repo>/output/nan_debug.log``."""
    override = os.environ.get("GRIT_NAN_LOG", "")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "output" / "nan_debug.log"


def _write_nan_log(message: str) -> None:
    """Append a one-line NaN event record to the log file (best-effort)."""
    try:
        p = _nan_log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(message + "\n")
    except Exception as e:   # logging must never crash training
        print(f"[nan-log] failed to write {message!r}: {e!r}", flush=True)


# ──────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────

_RL_ENV_REGISTRY: Dict[str, Type["BaseRLEnv"]] = {}


def register_rl_env(name: str) -> Callable[[Type["BaseRLEnv"]], Type["BaseRLEnv"]]:
    def deco(cls: Type["BaseRLEnv"]) -> Type["BaseRLEnv"]:
        if name in _RL_ENV_REGISTRY:
            raise ValueError(f"rl_env '{name}' already registered")
        _RL_ENV_REGISTRY[name] = cls
        return cls
    return deco


def make_rl_env(name: str, **kwargs) -> "BaseRLEnv":
    if name not in _RL_ENV_REGISTRY:
        raise KeyError(
            f"unknown rl_env '{name}'. registered: {sorted(_RL_ENV_REGISTRY)}"
        )
    return _RL_ENV_REGISTRY[name](**kwargs)


def list_rl_envs() -> list[str]:
    return sorted(_RL_ENV_REGISTRY)


def get_rl_env_cls(name: str) -> Type["BaseRLEnv"]:
    """Resolve a registered RL-env class by name. Raises ``KeyError`` if
    the name isn't registered. Used by ``grit.training.run_config`` to
    introspect ``HANDLER_CLS.BUILDER_KEYS`` / ``CONFIG_KEYS`` when
    rebuilding env+handler from a saved config (e.g. inference)."""
    if name not in _RL_ENV_REGISTRY:
        raise KeyError(
            f"unknown rl_env '{name}'. registered: {sorted(_RL_ENV_REGISTRY)}"
        )
    return _RL_ENV_REGISTRY[name]


def get_handler_cls(name: str) -> Type[SubEnvHandler]:
    """Return the ``HANDLER_CLS`` of a registered RL env name. Convenience
    wrapper used to feed ``run_config.rl_env_kwargs_from_config(...)``."""
    return get_rl_env_cls(name).HANDLER_CLS


def _flatten_groups(groups: Dict[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    """Concatenate the value tuples of a ``{group_name: (keys, ...)}``
    dict, preserving insertion order.

    Used to build the legacy flat ``BUILDER_KEYS`` / ``CONFIG_KEYS``
    aliases from the canonical grouped declarations on each
    :class:`SubEnvHandler` subclass.

    Duplicate keys across groups raise ``ValueError`` — knobs should
    live in exactly one group so handler config snapshots are
    deterministic. (Same key under two groups would silently overwrite
    on apply / produce ambiguous YAML on save.)
    """
    seen: Dict[str, str] = {}
    out:  list           = []
    for group_name, keys in groups.items():
        for k in keys:
            if k in seen:
                raise ValueError(
                    f"_flatten_groups: duplicate knob {k!r} — already in "
                    f"group {seen[k]!r}, repeated in group {group_name!r}"
                )
            seen[k] = group_name
            out.append(k)
    return tuple(out)


# ──────────────────────────────────────────────────────────────────────────
# Generic reward-knob curriculum — shared by all handlers (yaml: handler.curriculum.reward)
# ──────────────────────────────────────────────────────────────────────────


def apply_reward_curriculum(handler, cur_step: int, total_steps: int) -> None:
    """Generic reward-knob curriculum — scheduler shared by all tasks.

    Interpolates every schedule entry ``{knob: (final, start_step, end_step[, mode])}``
    of ``handler.reward`` (yaml ``handler.curriculum.reward``) each iteration
    and setattr's the knob. It is a **module-level function** so that not only
    the :class:`SubEnvHandler` family but also standalone handlers (e.g.
    ``LeaderFollowerGeomTrackingHandler``, which shares the call surface without
    inheriting SubEnvHandler) can call it directly from their own
    ``on_train_progress`` and get the same curriculum.

      * ``knob``  — only keys declared in a ``reward_*`` group of the handler's
        ``CONFIG_KEYS_GROUPS`` are allowed. Typos / non-reward keys raise a
        **ValueError (fail-fast)** so a silently ignored schedule cannot skew an
        experiment.
      * The **base value** is captured on the first call (= after
        ``apply_handler_knobs``, i.e. the yaml value) and interpolated
        ``base → final``. Progress is step-based, so a resume continues from the
        same schedule position (same convention as the geom curriculum).
      * ``start_step`` / ``end_step`` refer to the global control-step (budget
        cursor) passed by train.py. ``end_step <= 0`` is read as ``total_steps``.
      * **On/off is decided by the window**: ``start_step < 0`` → that knob's
        ramp is OFF (base value kept, not an error). An entry can be disabled by
        setting only its window to -1 while leaving it in the yaml; a value(-1)
        sentinel is not used.
      * ``mode`` (optional 4th element): ``"linear"`` (default) or ``"log"``
        (geometric / log-space interpolation, natural for ``K_*``-style exp
        rates; only when base and final are both > 0, otherwise falls back to linear).
      * **success-gate** (when ``handler.reward_gate`` is set): effective
        progress = ``max(step-schedule progress, gate progress)``. Good eval
        results approach final earlier than the schedule, and if the gate never
        opens the step schedule acts as fallback. A knob with ``start_step < 0``
        becomes gate-only instead of OFF. Gate progress is advanced by
        :func:`apply_reward_gate` after every eval.
      * **global start** (``handler.reward_start_step``, default 0): before this
        step the whole curriculum is frozen. The effective start per knob is
        ``max(knob_start, reward_start_step)``, and gate progress is neither
        applied (treated as 0) nor accumulated (:func:`apply_reward_gate`) before it.

    No-op if ``handler.reward`` is empty. Progress is logged to stdout only, in
    25% increments (per-step values are visible through the existing
    reward_term tracking).
    """
    sched = getattr(handler, "reward", None)
    if not sched:
        return
    # Global curriculum start (handler.reward_start_step; 0 = original behaviour).
    # Stashed on every call so apply_reward_gate knows the step at eval time.
    g_start = int(float(getattr(handler, "reward_start_step", 0) or 0))
    handler._rew_curr_step = int(cur_step)
    # ── First call: validate + capture base values (yaml-applied) ──────────
    if getattr(handler, "_rew_curr_spec", None) is None:
        allowed: set = set()
        for g, keys in getattr(type(handler), "CONFIG_KEYS_GROUPS", {}).items():
            if g.startswith("reward"):
                allowed.update(keys)
        # Allow explicit schedules for knobs outside the reward_* groups (e.g.
        # W_BONUS; _flatten_groups forbids duplicate group membership, so this
        # is handled as a whitelist extension).
        allowed.update(getattr(type(handler), "CURRICULUM_EXTRA_KNOBS", ()))
        spec_clean: Dict[str, Tuple[float, int, int, str]] = {}
        base_vals:  Dict[str, float] = {}
        for name, spec in dict(sched).items():
            name = str(name)
            # null = "no schedule for this knob", so an ablation can disable it
            # from the CLI with just `handler.curriculum.reward.X=null`
            # (convention for running a control without editing the yaml).
            if spec is None:
                continue
            if name not in allowed:
                raise ValueError(
                    f"curriculum.reward: {name!r} is not a "
                    f"reward_* group knob of {type(handler).__name__}.\n"
                    f"  allowed keys: {sorted(allowed)}")
            seq = list(spec) if isinstance(spec, (list, tuple)) else None
            if seq is None or len(seq) not in (3, 4):
                raise ValueError(
                    f"curriculum.reward.{name}: must be of the form (final, start_step, end_step"
                    f"[, 'log']) — got {spec!r}")
            mode = str(seq[3]).lower() if len(seq) == 4 else "linear"
            if mode not in ("linear", "log"):
                raise ValueError(
                    f"curriculum.reward.{name}: mode must be 'linear'|'log' — got {seq[3]!r}")
            try:
                cur_val = float(getattr(handler, name))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"curriculum.reward.{name}: current value is not numeric "
                    f"({getattr(handler, name)!r}) — cannot schedule") from e
            spec_clean[name] = (float(seq[0]), int(float(seq[1])),
                                int(float(seq[2])), mode)
            base_vals[name]  = cur_val
        handler._rew_curr_spec       = spec_clean
        handler._rew_curr_base       = base_vals
        handler._rew_curr_last_print = -1.0
        tag = getattr(getattr(handler, "hand_util", None), "hand_name", "?")
        _gate = getattr(handler, "reward_gate", None)
        if g_start > 0:
            print(f"[reward-curr] {tag}: global start = {g_start:,} step "
                  f"(schedule and gate both frozen before it — base weights kept)")
        for n, (fv, s, e, m) in spec_clean.items():
            if s < 0:
                print(f"[reward-curr] {tag}: {n} "
                      + (f"{base_vals[n]:g} → {fv:g} ({m}, gate-only)" if _gate
                         else "— OFF (no window, start<0)"))
                continue
            s_eff = max(s, g_start)
            e_str = f"{e:,}" if e > 0 else f"total({total_steps:,})"
            print(f"[reward-curr] {tag}: {n} {base_vals[n]:g} → {fv:g} "
                  f"({m}, {s_eff:,} → {e_str})")
        if _gate:
            print(f"[reward-curr] {tag}: success-gate ON — {dict(_gate)}")
    # ── Every call: interpolate each knob on its own window ────────────────
    # progress: 0 = base weight, 1 = final weight. Effective progress per knob
    # = max(step-schedule progress, success-gate progress).
    gate_on       = bool(getattr(handler, "reward_gate", None))
    gate_progress = float(getattr(handler, "_gate_progress", 0.0)) if gate_on else 0.0
    # Before the global start, gate progress is not applied either (including
    # progress restored on resume): the whole curriculum stays at base weights.
    # Accumulation itself is blocked by apply_reward_gate.
    if cur_step < g_start:
        gate_progress = 0.0
    top_progress  = 0.0
    for name, (final, start, end, mode) in handler._rew_curr_spec.items():
        if start < 0 and not gate_on:
            continue                            # no window (-1) and no gate → OFF
        if start < 0:
            progress = gate_progress            # gate-only knob (no step window)
        else:
            start = max(start, g_start)         # global start is the floor of the knob start
            if end <= 0:
                end = int(total_steps)
            sched = min(max(float(cur_step - start) / float(max(1, end - start)), 0.0), 1.0)
            progress = max(sched, gate_progress)
        b = handler._rew_curr_base[name]
        if mode == "log" and b > 0.0 and final > 0.0:
            val = b * (final / b) ** progress   # geometric interpolation (for K_* exp rates)
        else:
            val = b + (final - b) * progress    # linear interpolation
        setattr(handler, name, float(val))
        top_progress = max(top_progress, progress)
    # progress log (25% increments, stdout only)
    if top_progress - handler._rew_curr_last_print >= 0.25 or (
            top_progress >= 1.0 > handler._rew_curr_last_print):
        handler._rew_curr_last_print = top_progress
        vals = "  ".join(f"{n}={float(getattr(handler, n)):.4g}"
                         for n in handler._rew_curr_spec)
        gate_str = f"  gate={gate_progress:.2f}" if gate_on else ""
        print(f"[reward-curr] progress={top_progress:.2f}{gate_str}  {vals}")


def apply_reward_gate(handler, ev: Dict[str, Any], *, validate_only: bool = False) -> bool:
    """Success-gate: advance the reward curriculum early when eval results are good.

    yaml ``handler.curriculum.reward_gate``::

        reward_gate:
          metric:            success_rate_lift  # metric to read from the run_eval result
          threshold:         0.75               # gate passes at or above this value
          progress_per_eval: 0.05               # progress +5 percentage points per pass
          max_progress:      1.0                # progress cap (>1 → extrapolate beyond final)
          margin_band:       0.10               # (optional) P-control: scale the step by the
                                                # metric's margin above threshold — ~0 just
                                                # above threshold, full speed at threshold+band
                                                # and beyond. Unset (0) → bang-bang behaviour.

    Every pass raises ``handler._gate_progress`` (curriculum progress: 0 = base
    weight, 1 = final weight) by ``progress_per_eval``. It never decreases
    (monotone: kept even if the metric drops again). The effective progress is
    combined by :func:`apply_reward_curriculum` as ``max(step-schedule progress,
    gate progress)``, so the step schedule proceeds as fallback even if the gate
    never opens. ``max_progress > 1`` extrapolates beyond the final weight (a
    linear down-ramp may flip sign, so design it together with the final value).

    No-op without a ``reward`` schedule. A metric typo raises ValueError
    (fail-fast: train.py's baseline eval validates once right after start with
    ``validate_only=True``). Module-level for the same reason as
    :func:`apply_reward_curriculum`."""
    gate = getattr(handler, "reward_gate", None)
    if not gate or not getattr(handler, "reward", None):
        return False
    metric = str(gate.get("metric", "success_rate"))
    if metric not in ev:
        raise ValueError(
            f"curriculum.reward_gate.metric {metric!r} not found in the eval result. "
            f"available keys: {sorted(k for k, v in ev.items() if isinstance(v, (int, float)))}")
    threshold = float(gate.get("threshold", 0.75))
    progress  = float(getattr(handler, "_gate_progress", 0.0))
    limit     = float(gate.get("max_progress", 1.0))
    if validate_only or float(ev[metric]) < threshold or progress >= limit:
        return False
    # Before the global curriculum start (handler.reward_start_step) a pass does
    # not accumulate: prevents "doing well early" from pulling the penalty ramp
    # forward. cur_step is stashed by apply_reward_curriculum every iteration
    # (0 before the first eval).
    g_start = int(float(getattr(handler, "reward_start_step", 0) or 0))
    if int(getattr(handler, "_rew_curr_step", 0)) < g_start:
        tag = getattr(getattr(handler, "hand_util", None), "hand_name", "?")
        print(f"[reward-gate] {tag}: {metric}={float(ev[metric]):.3f} ≥ {threshold:g} "
              f"passed, but before global start ({g_start:,} step) — accumulation frozen")
        return False
    step = float(gate.get("progress_per_eval", 0.05))
    band = float(gate.get("margin_band", 0.0) or 0.0)
    if band > 0.0:
        # P-control: barely advance just above threshold, full speed from
        # threshold+band; removes the bang-bang cliff that pushes the success
        # rate down to the threshold.
        step *= min(1.0, (float(ev[metric]) - threshold) / band)
        if step <= 0.0:
            return False
    handler._gate_progress = min(limit, progress + step)
    tag = getattr(getattr(handler, "hand_util", None), "hand_name", "?")
    print(f"[reward-gate] {tag}: {metric}={float(ev[metric]):.3f} ≥ {threshold:g} → "
          f"curriculum progress {progress:.2f} → {handler._gate_progress:.2f} (0=base, 1=final)")
    return True


# ──────────────────────────────────────────────────────────────────────────
# Shared warp kernels (handler-private state, same code for every handler)
# ──────────────────────────────────────────────────────────────────────────


@wp.func
def _grit_quat_wxyz_to_R(q: wp.quat) -> wp.mat33:                            # type: ignore
    """Convert a wxyz unit quaternion to a 3×3 rotation matrix (column-major
    semantics — same convention used by ``data.xmat``)."""
    w = q[0]; x = q[1]; y = q[2]; z = q[3]
    xx = x * x; yy = y * y; zz = z * z # type: ignore
    xy = x * y; xz = x * z; yz = y * z # type: ignore
    wx = w * x; wy = w * y; wz = w * z # type: ignore
    return wp.mat33(
        1.0 - 2.0 * (yy + zz),       2.0 * (xy - wz),       2.0 * (xz + wy), # type: ignore
              2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz),       2.0 * (yz - wx), # type: ignore
              2.0 * (xz - wy),       2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), # type: ignore
    )


# ──────────────────────────────────────────────────────────────────────────
# Rotation 6D representation (Zhou et al. 2019)
# ──────────────────────────────────────────────────────────────────────────
# A 3×3 rotation matrix is encoded as its **first two columns** — a 6-vector
# ``(R[:,0], R[:,1])``. The decoder (``_gram_schmidt_6d`` in
# ``warp_action.py``) recovers the third column by orthonormalisation +
# cross product, and is continuous everywhere on SO(3) — unlike quat /
# euler / axis-angle. We use this representation in TWO places so the
# policy sees a single, continuous orientation format throughout:
#
#   (1) **Action output** (``WarpActionApplier``): Δrot residual is a 6D
#       vector around identity ``[1,0,0,0,1,0]`` — the apply kernel adds
#       the bias and Gram-Schmidts to get R_delta.
#
#   (2) **Observation**: ``_hand_pose_obs_kernel`` emits the wrist-local
#       target rotation ``(R_wrist^T · R_target)[:,:2]`` (6 floats) — the
#       only rotation slot in the obs vector. The absolute wrist
#       orientation used to be emitted alongside it (also 6D) but was
#       removed: the policy never needed it (wrist-local error already
#       captures all task-relevant rotation info) and providing it forced
#       the network to learn world↔body covariates that hurt translation
#       tracking.
#
# Cond / reward / ghost still use 4-float wxyz quat (canonical, more
# compact, ghost MJCF requires quat). The 6D conversion happens *only* at
# the obs/action boundary the policy sees.


@wp.kernel
def _restore_non_done_worlds(
    qpos_snap:   wp.array(dtype=float,   ndim=2),  # type: ignore
    qvel_snap:   wp.array(dtype=float,   ndim=2),  # type: ignore
    ctrl_snap:   wp.array(dtype=float,   ndim=2),  # type: ignore
    mp_snap:     wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
    mq_snap:     wp.array(dtype=wp.quat, ndim=2),  # type: ignore
    qpos_curr:   wp.array(dtype=float,   ndim=2),  # type: ignore
    qvel_curr:   wp.array(dtype=float,   ndim=2),  # type: ignore
    ctrl_curr:   wp.array(dtype=float,   ndim=2),  # type: ignore
    mp_curr:     wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
    mq_curr:     wp.array(dtype=wp.quat, ndim=2),  # type: ignore
    done_mask:   wp.array(dtype=int,     ndim=1),  # type: ignore
    nq: int, nv: int, nu: int, n_mocap: int,
):
    w = wp.tid()
    if done_mask[w] == 0:
        for i in range(nq):
            qpos_curr[w, i] = qpos_snap[w, i]
        for i in range(nv):
            qvel_curr[w, i] = qvel_snap[w, i]
        for i in range(nu):
            ctrl_curr[w, i] = ctrl_snap[w, i]
        for j in range(n_mocap):
            mp_curr[w, j] = mp_snap[w, j]
            mq_curr[w, j] = mq_snap[w, j]


@wp.kernel
def _flag_nonfinite_worlds(
    qpos:        wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, nq)
    qvel:        wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, nv)
    nq: int, nv: int, big: float,
    done_mask:   wp.array(dtype=int, ndim=1),    # type: ignore (NWORLD,)
    done_reason: wp.array(dtype=int, ndim=1),    # type: ignore (NWORLD,)
):
    """Flag worlds whose physics state went non-finite (mjwarp solver blow-up).

    A blown-up world's ``qpos/qvel`` becomes NaN/Inf, which flows unguarded into
    the obs. The done kernel can't catch it (NaN threshold comparisons are all
    ``False``), so without this it persists every step (and the snapshot/restore
    in ``per_world_reset_if_done`` would copy the NaN back). Marking it done
    (reason 8 = ``nonfinite``) routes it to a FRESH init instead of the NaN
    snapshot restore — breaking both persistence paths. ``v != v`` catches NaN;
    ``|v| > big`` catches Inf / overflow.
    """
    w = wp.tid()
    bad = int(0)
    for i in range(nq):
        v = qpos[w, i]
        if (v != v) or (v > big) or (v < -big):
            bad = 1
    for i in range(nv):
        v = qvel[w, i]
        if (v != v) or (v > big) or (v < -big):
            bad = 1
    if bad == 1:
        done_mask[w] = 1
        if done_reason[w] == 0:          # don't clobber a real terminal reason
            done_reason[w] = 8           # REASON_NAME[8] = "nonfinite"


@wp.kernel
def _update_ep_counters_kernel(
    done_mask: wp.array(dtype=int, ndim=1),  # type: ignore
    ep_step:   wp.array(dtype=int, ndim=1),  # type: ignore
    ep_idx:    wp.array(dtype=int, ndim=1),  # type: ignore
):
    w = wp.tid()
    if done_mask[w] == 1:
        ep_idx[w]  = ep_idx[w] + 1
        ep_step[w] = 0


# ──────────────────────────────────────────────────────────────────────────
# Shared action kernels — reusable across tasks
# ──────────────────────────────────────────────────────────────────────────

@wp.kernel
def _action_sqnorm_kernel(
    action:     wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, action_dim)
    action_dim: int,
    out_sqnorm: wp.array(dtype=float, ndim=1),  # type: ignore (NWORLD,)
):
    """Compute ``||action||²`` per world (raw control-cost driver)."""
    w = wp.tid()
    sqnorm = float(0.0)
    for i in range(action_dim):
        av = action[w, i]
        sqnorm += av * av
    out_sqnorm[w] = sqnorm


@wp.kernel
def _action_smooth_sqnorm_kernel(
    action:       wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, action_dim)
    last_action:  wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, action_dim) IN/OUT
    action_dim:   int,
    out_sqnorm:   wp.array(dtype=float, ndim=1),  # type: ignore (NWORLD,)
):
    """Compute the **mean** squared Δaction ``||action - last_action||² /
    action_dim`` per world and update ``last_action``.

    Runs before the mixer so ``action_smooth_sqnorm`` is ready for
    the ``-W_ACTION_SMOOTH · mean(Δa²)`` smoothness penalty. Updates
    ``last_action[w] = action[w]`` in place so the next step automatically
    sees the correct previous action.

    The per-component MEAN (divide by ``action_dim``) makes the penalty
    magnitude independent of the action dimensionality (so ``W_ACTION_SMOOTH``
    transfers across hands with different ``n_ctrl``).

    On episode reset ``last_action`` is zeroed by
    ``_zero_masked_action_kernel``, so the first-step penalty equals
    ``mean(a_0²)``.
    """
    w = wp.tid()
    sqnorm = float(0.0)
    for i in range(action_dim):
        da = action[w, i] - last_action[w, i]
        sqnorm += da * da
    # MEAN (÷action_dim) is required: with a SUM, exploration noise (std≈1)
    # gives a Δa² sum of ~n_ctrl (≈13+), so exp(-W·13)≈0 and the whole process
    # reward is gated to 0 the moment the policy explores, killing the learning
    # signal (with the mean it is ~0.46 → gate 0.63). If a dimension correction
    # is needed, adjust W_ACTION_SMOOTH rather than the kernel.
    denom = float(action_dim)
    if denom < 1.0:
        denom = 1.0
    out_sqnorm[w] = sqnorm / denom              # MEAN of squared Δaction
    for i in range(action_dim):
        last_action[w, i] = action[w, i]


@wp.kernel
def _zero_masked_action_kernel(
    done_mask:    wp.array(dtype=int,   ndim=1),  # type: ignore (NWORLD,)
    last_action:  wp.array(dtype=float, ndim=2),  # type: ignore (NWORLD, action_dim) OUT
    action_dim:   int,
):
    """Zero ``last_action[w]`` for done worlds after per-world reset.

    Called from ``per_world_reset_if_done`` so the first step of the
    new episode computes action smoothness against zero (no prior action).
    """
    w = wp.tid()
    if done_mask[w] == 1:
        for i in range(action_dim):
            last_action[w, i] = 0.0


@wp.kernel
def _episode_success_kernel(
    bonus_active:        wp.array(dtype=int,   ndim=1),    # type: ignore  (NWORLD,) >=1 = bonus condition met this step
    ep_step:             wp.array(dtype=int,   ndim=1),    # type: ignore  (NWORLD,) post-increment counter
    bonus_streak:        wp.array(dtype=int,   ndim=1),    # type: ignore  IN/OUT consecutive-bonus counter
    success_streak_min:  int,                              # consecutive bonus steps required for success
    max_ep_steps:        int,
    success_reason_code: int,
    disable_done_promotion: int,                           # 1 → eval mode: only update streak, don't promote done
    out_done:            wp.array(dtype=float, ndim=1),    # type: ignore  IN/OUT
    out_mask:            wp.array(dtype=int,   ndim=1),    # type: ignore  IN/OUT
    out_reason:          wp.array(dtype=int,   ndim=1),    # type: ignore  IN/OUT
):
    """Generic episode-success kernel — streak bookkeeping + done promotion.

    TASK-AGNOSTIC: each task only has to produce ``bonus_active[w]`` (>=1 when
    its per-task success/bonus condition holds this step — via whatever bonus
    gate it defines); this kernel handles consecutive-streak counting and the
    success done-promotion uniformly. Shared by every ``SubEnvHandler``
    subclass through its ``_collect_success_kernel`` (so the per-task handler
    only decides HOW ``bonus_active`` is computed).

    Success criterion: ``ep_step[w] >= max_ep_steps  AND  bonus_streak[w] >=
    success_streak_min`` — the episode reached timeout while the bonus has been
    held long enough; the timeout (``reason=2`` truncation) is overwritten with
    ``success_reason_code`` so a satisfied success at the last step reads as a
    terminal success, not truncation. (The streak-only criterion is kept
    commented below for tasks that want to promote success before timeout.)

    On any done (success or otherwise) the streak counter resets to ``0`` so
    the next episode starts clean.

    **Eval mode** (``disable_done_promotion == 1``): only the streak counter
    updates (the mixer needs it for the streak-scaled bonus reward); no done
    promotion / streak-on-done reset happens, so the policy must MAINTAIN its
    bonus condition to keep earning reward across the full eval window.
    """
    w = wp.tid()

    # ① Update the consecutive-bonus streak counter (always — mixer reads it).
    #    ``bonus_active`` is 0 / 1 / 2 (one count per enabled gate that passed);
    #    treat ANY active gate (>= 1) as "bonus on" for the streak.
    if bonus_active[w] >= 1:
        bonus_streak[w] = bonus_streak[w] + 1
    else:
        bonus_streak[w] = 0

    # Eval mode: stop here. Other done sources stay (for diagnostics); success
    # doesn't promote done, and the streak is NOT reset on those other done
    # sources (lets the user track "max streak ever achieved per world").
    if disable_done_promotion == 1:
        return

    # ② Evaluate success (training only).
    success = int(0)
    # if bonus_streak[w] >= success_streak_min:
    #     success = 1
    if (ep_step[w] >= max_ep_steps) and (bonus_streak[w] >= success_streak_min):
        success = 1

    # ③ On success, fire (or override) done with the success reason — converts
    #    a coincident timeout (reason=2 truncation) into a terminal success.
    if success == 1:
        out_done[w]   = 1.0
        out_mask[w]   = 1
        out_reason[w] = success_reason_code

    # ④ Streak reset — applies to *any* done. Next episode starts streak = 0.
    if out_done[w] > 0.5:
        bonus_streak[w] = 0


@wp.kernel
def _dynamic_target_switch_mask_kernel(
    ep_step:             wp.array(dtype=int, ndim=1),   # type: ignore
    target_switch_step:  wp.array(dtype=int, ndim=1),   # type: ignore (IN/OUT)
    out_switch_mask:     wp.array(dtype=int, ndim=1),   # type: ignore
):
    """Per-world dynamic-target trigger (shared, task-agnostic).

    Writes ``out_switch_mask[w] = 1`` and clears ``target_switch_step[w] = -1``
    when this world's ``ep_step`` just reached its scheduled switch tick
    (``target_switch_step[w] >= 0``). Otherwise writes ``out_switch_mask[w] = 0``
    and leaves both buffers untouched. The ``-1`` sentinel guarantees at-most-one
    switch per episode — re-sampled by ``SubEnvHandler._resample_switch_step`` at
    episode reset. Each task's ``_maybe_dynamic_target_switch`` consumes the mask.
    """
    w = wp.tid()
    s = target_switch_step[w]
    if s >= 0 and ep_step[w] == s:
        out_switch_mask[w] = 1
        target_switch_step[w] = -1
    else:
        out_switch_mask[w] = 0


# ──────────────────────────────────────────────────────────────────────────
# SubEnvHandler — owns one warp sub-env
# ──────────────────────────────────────────────────────────────────────────


class SubEnvHandler:
    """One sub-env's complete RL state (action, condition, kernels, buffers).

    Subclass per task and override:

      * ``obs_dim`` (property)
      * ``_setup_obs_buffers``
      * ``_collect_obs_kernel``
      * ``_collect_reward_done_kernel``
      * (optional) ``_setup_condition`` — defaults to a uniform target
        wrist position.

    All buffers are zero-copy torch views of warp arrays. The handler is
    fully self-contained: ``step`` / ``reset`` / ``per_world_reset_if_done``
    don't touch any other handler.
    """

    # 5 (obj_xy) / 6 (hand_object_touch) / 7 (obj_fell) are object_grasping-specific
    # done reasons — listed here so any consumer of ``REASON_NAME`` can label them
    # without a KeyError (the tracking task simply never emits them).
    REASON_NAME = {0: "none", 1: "success", 2: "timeout", 3: "obj_z", 4: "wrist_up",
                   5: "obj_xy", 6: "hand_object_touch", 7: "obj_fell", 8: "nonfinite",
                   9: "hand_obj_far", 10: "table_crush",
                   11: "rest_crush",    # non-taxonomy (rest) body↔obj contact force exceeded (object_grasping)
                   12: "obj_ground",    # object presses the floor/table beyond threshold (object_grasping)
                   13: "hold_far"}      # hand↔object surface distance exceeded during hold — early recovery of failed episodes (object_grasping)

    # ── Dynamic mid-episode target switch (shared base config) ───────────
    # When enabled, each episode schedules a single ``target_switch_step[w]`` in
    # ``[floor(MIN_FRAC·MAX_EPISODE_STEPS), ceil(MAX_FRAC·MAX_EPISODE_STEPS))``;
    # when ``ep_step[w]`` reaches it during ``step()`` the task's
    # ``_maybe_dynamic_target_switch`` re-samples that world's target cond in
    # place and re-collects obs. Default False → no behaviour change. Subclasses
    # that want the feature must (a) allocate ``target_switch_step`` / ``switch_mask``
    # (NWORLD,) int buffers, (b) call ``_resample_switch_step`` at episode reset,
    # and (c) implement ``_maybe_dynamic_target_switch``.
    DYNAMIC_TARGET_ENABLED:  bool  = False
    DYNAMIC_TARGET_MIN_FRAC: float = 0.3
    DYNAMIC_TARGET_MAX_FRAC: float = 0.7

    # Type hints for the dynamic-switch buffers — declared here so Pylance
    # resolves ``self.target_switch_step_torch`` / ``switch_mask_torch`` in
    # ``_resample_switch_step`` without a red underline. The actual wp.array /
    # torch.Tensor objects are allocated by each subclass that opts in
    # (``target_switch_step`` / ``switch_mask`` in their INT_BUFS); the
    # Optional annotation reflects that the base never allocates them.
    target_switch_step_wp:    Optional[wp.array]       = None  # type: ignore
    target_switch_step_torch: Optional[torch.Tensor]   = None
    switch_mask_wp:           Optional[wp.array]       = None  # type: ignore
    switch_mask_torch:        Optional[torch.Tensor]   = None

    # ── Run-config introspection (consumed by grit.training.run_config) ─
    #
    # Knobs are declared as **dicts of (group_name → tuple of names)** so
    # the surface area is organised by RL concept and stays grokkable as
    # more tasks are added. Two flat aliases (``BUILDER_KEYS`` /
    # ``CONFIG_KEYS``) are auto-derived in ``__init_subclass__`` for
    # compatibility with code that reads the legacy flat tuple.
    #
    # *Groups owned by the base class* — every handler subclass inherits
    # these and can extend by composing with ``**SubEnvHandler.X_GROUPS``:
    #
    #   ``simulation``     — control loop / physics step setup
    #   ``action_applier`` — Δaction scaling + deadzone (WarpActionApplier)
    #   ``reset``          — per-world finger init / RNG seed
    #   ``condition``      — per-world cond field bounds (used by the
    #                        default :meth:`_setup_condition`; tasks that
    #                        register their own cond may ignore this group)
    #
    # Task-specific handlers (e.g. :class:`HandPoseTrackingHandler`) add
    # their own groups (reward weights, termination thresholds, success
    # gates, …) via :attr:`CONFIG_KEYS_GROUPS` — see that class for the
    # canonical example.
    BUILDER_KEYS_GROUPS: Dict[str, Tuple[str, ...]] = {
        "simulation": (
            "sim_nstep",
        ),
        "action_applier": (
            "xyz_scale", "rot_scale", "finger_scale",
            "control_wrist", "finger_abs_action", "finger_action_mode", "finger_lead_max_rad", "finger_leak",
            "finger_hist_len", "finger_hist_mean",
            "use_deadzone",
            "xyz_deadzone_m", "rot_deadzone_deg", "finger_deadzone_deg",
        ),
        "reset": (
            "randomize_finger_init", "finger_init_seed",
        ),
        "condition": (
            "cond_pos_low", "cond_pos_high",
        ),
    }
    CONFIG_KEYS_GROUPS: Dict[str, Tuple[str, ...]] = {
        # base has no tunable instance attrs; task handlers declare their own.
    }

    # Flat aliases — auto-rebuilt in __init_subclass__ for each subclass.
    # The base-class flat aliases are filled in immediately after the
    # ``class SubEnvHandler:`` body (see ``_init_base_keys`` below).
    BUILDER_KEYS: Tuple[str, ...] = ()
    CONFIG_KEYS:  Tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Universal reward-curriculum knobs: EVERY task's ``curriculum`` group
        # carries the ``reward`` schedule dict (see ``apply_reward_curriculum``)
        # and the ``reward_gate`` success-gate dict (see ``apply_reward_gate``)
        # without each handler having to declare them. Copy-on-write + idempotent
        # so subclasses that inherit the parent's (already-injected) dict are
        # untouched and re-registration can't duplicate the keys.
        groups = dict(cls.CONFIG_KEYS_GROUPS)
        _cur = tuple(groups.get("curriculum", ()))
        _add = tuple(k for k in ("reward", "reward_gate", "reward_start_step")
                     if k not in _cur)
        if _add:
            groups["curriculum"] = _cur + _add
            cls.CONFIG_KEYS_GROUPS = groups
        cls.BUILDER_KEYS = _flatten_groups(cls.BUILDER_KEYS_GROUPS)
        cls.CONFIG_KEYS  = _flatten_groups(cls.CONFIG_KEYS_GROUPS)

    # Subclass-tunable defaults (from the v2 kernel) ──────────────
    DONE_OBJ_Z_MIN: float    = -0.10
    DONE_WRIST_X_Z: float    = 1.0
    MAX_EPISODE_STEPS: int   = 256
    CTRL_COST: float         = 1.0

    # Final reward post-process: ``out_reward = clip(out_reward * REWARD_DT,
    # MIN_REWARD, MAX_REWARD)``. Applied AFTER the contact penalty kernel
    # so the clip bounds the full per-step reward (tracking + bonus +
    # ctrl-cost + contact penalties). Mirrors the JAX-era convention
    # ``reward = jnp.clip(reward * self.dt, self.min_reward, self.max_reward)``.
    #
    # ``REWARD_DT`` defaults to the env's control timestep
    # (``sim_nstep * mjm.opt.timestep``) and is auto-initialised from the
    # sampled env at handler init; subclasses / callers may override it as
    # an instance attr (e.g. ``h.REWARD_DT = 1.0`` to disable the dt scaling).
    REWARD_DT:  float = 1.0
    MIN_REWARD: float = -10.0
    MAX_REWARD: float = 50.0

    #: Randomize finger qpos / ctrl on every reset (full + per-world async).
    #: Translates legacy JAX ``_finger_init_fn`` per ``hand_util.hand_name``.
    RANDOMIZE_FINGER_INIT: bool = True

    #: Evaluation mode — when True, **disables auto-reset and the
    #: success-based done promotion** so the rollout runs uninterrupted
    #: for a fixed window:
    #:
    #:   * ``per_world_reset_if_done()`` becomes a no-op (no per-world
    #:     reset on done — worlds keep their state, target, and continue
    #:     accumulating reward).
    #:   * ``_collect_success_kernel()`` becomes a no-op in subclasses
    #:     that override it (e.g. ``HandPoseTrackingHandler``) — meeting
    #:     the bonus condition no longer terminates the episode, so the
    #:     policy must MAINTAIN tracking to keep earning reward.
    #:
    #: Other done sources (``obj_z`` drop, ``timeout``, ``wrist_up``)
    #: still fire (useful for diagnostics) but trigger no reset.
    #:
    #: Use ``with h.eval_context(): ...`` for scoped switching.
    eval_mode: bool = False

    #: Optional per-substep hook — :meth:`step` calls it **right after every**
    #: physics substep graph launch (``sim_nstep`` times per control tick). Use
    #: case: sampling and accumulating contact sensors at sim rate (reading
    #: sensordata once per control tick folds sim-rate contact chatter in as
    #: **aliasing** — Nyquist; observed in practice with force-aware collection).
    #: ``None`` (default) = zero overhead beyond one check. Subclasses override
    #: it as a method and check the on/off knob inside the hook (knobs are
    #: overwritten by ``apply_handler_knobs`` after init, so they must not be
    #: frozen at binding time).
    _substep_sensor_hook = None

    #: Timeout-only termination — when True, **only ``timeout`` (ep_step >=
    #: MAX_EPISODE_STEPS) sets ``done_mask`` / ``done`` so each world plays
    #: out a FULL ``MAX_EPISODE_STEPS`` window before being recycled.
    #:
    #: This is distinct from :attr:`eval_mode` (which disables resets entirely):
    #: here per-world reset still fires, but only at the natural episode end.
    #: Non-timeout failure states (``obj_z`` drop, ``obj_xy`` drift, crush,
    #: ``obj_fell``, ``wrist_up``) are still written to ``done_reason`` for
    #: diagnostics / success-marker use, but they NO LONGER set ``done_mask``.
    #:
    #: Why this matters (the "zombie" bug): ``_update_ep_counters`` zeroes
    #: ``ep_step`` whenever ``done_mask == 1``. State-based failure conditions
    #: (e.g. a dropped object resting below the table) hold True every step, so
    #: without this flag they zero ``ep_step`` every step — the world's clock
    #: never reaches ``MAX_EPISODE_STEPS``, ``timeout`` never fires, and an
    #: evaluate.py loop that recycles on timeout-only leaves the world frozen
    #: forever. Gating ``done_mask`` to timeout lets ``ep_step`` advance to the
    #: window end where it recycles cleanly.
    #:
    #: Task handlers whose done kernel reads this flag (e.g.
    #: ``object_grasping``) honour it; tasks that don't (e.g. tracking, whose
    #: only non-timeout done is wrist_up — disabled in eval) are unaffected.
    #: Defaults to False so training keeps its multi-reason early termination.
    timeout_only_done: bool = False

    # Buffers populated by ``_setup_obs_buffers`` / ``_setup_reward_done_buffers``
    # (declared here so static checkers see the attribute on the base class).
    obs_torch:    torch.Tensor
    reward_torch: torch.Tensor
    done_torch:   torch.Tensor
    obs_wp:       wp.array
    reward_wp:    wp.array
    done_wp:      wp.array

    def __init__(
        self,
        sampled_env,
        hand_util,
        sim_nstep: int = 5,
        xyz_scale: float = 0.005,
        finger_scale: float = 0.02,
        rot_scale: float = 0.1,
        cond_seed: int = 42,
        cond_pos_low: tuple = (-0.20, -0.20, 0.10),
        cond_pos_high: tuple = (0.20, 0.20, 0.50),
        handler_idx: int = 0,
        randomize_finger_init: Optional[bool] = None,
        finger_init_seed: int = 42,
        # ``True`` (default) → policy commands wrist + fingers. ``False`` →
        # finger-only action (``action_dim = n_ctrl``); the wrist is held at
        # its reset pose. Tasks that read it (e.g. hand_pose_tracking) also
        # pin the wrist target so only finger pose is tracked.
        control_wrist:       bool  = True,
        # ── Deadzone pass-through (forwarded to ``WarpActionApplier``) ───
        # ``use_deadzone=False`` keeps the legacy behaviour. See
        # ``WarpActionApplier`` docstring for unit semantics.
        use_deadzone:        bool  = False,
        xyz_deadzone_m:      float = 0.0,
        rot_deadzone_deg:    float = 0.0,
        finger_deadzone_deg: float = 0.0,
        # finger action = absolute-normalized setpoint (see WarpActionApplier).
        finger_abs_action:   bool  = False,
        finger_action_mode:  str   = "delta",
        finger_lead_max_rad: float = 0.2,
        finger_leak:         float = 1.0,
        finger_hist_len:     int   = 4,
        finger_hist_mean:    bool  = True,
    ):
        self.sampled_env = sampled_env
        self.hand_util   = hand_util
        self.sim_nstep   = int(sim_nstep)
        self.handler_idx = int(handler_idx)
        # Per-instance reward post-process knob: control dt = sim_nstep · sim_dt.
        # Mirrors the JAX-era ``self.dt`` used in
        # ``reward = jnp.clip(reward * self.dt, ...)``. Override on the
        # instance (``h.REWARD_DT = 1.0``) to disable the dt scaling.
        self.REWARD_DT = float(self.sim_nstep) * float(sampled_env.mjm.opt.timestep)
        # Resolve randomize_finger_init: ctor arg overrides the class-attr default.
        self.randomize_finger_init = (
            bool(randomize_finger_init)
            if randomize_finger_init is not None
            else bool(self.RANDOMIZE_FINGER_INIT)
        )

        self.m   = sampled_env.m
        self.d   = sampled_env.d
        self.mjm = sampled_env.mjm

        self.NWORLD = int(sampled_env.NWORLD)
        self.device = self.d.qpos.device

        # Action applier (writes mocap_pos / mocap_quat / ctrl).
        # Deadzone parameters are forwarded as-is; they can be toggled at
        # runtime via ``self.applier.set_deadzone(...)`` without rebuilding
        # the handler (kernel reads the scalars at launch time).
        self._control_wrist = bool(control_wrist)
        self.applier = WarpActionApplier(
            env=sampled_env,
            hand_util=hand_util,
            xyz_scale=xyz_scale,
            finger_scale=finger_scale,
            rot_scale=rot_scale,
            control_wrist=control_wrist,
            use_deadzone=use_deadzone,
            xyz_deadzone_m=xyz_deadzone_m,
            rot_deadzone_deg=rot_deadzone_deg,
            finger_deadzone_deg=finger_deadzone_deg,
            finger_abs_action=finger_abs_action,
            finger_action_mode=finger_action_mode,
            finger_lead_max_rad=finger_lead_max_rad,
            finger_leak=finger_leak,
            finger_hist_len=finger_hist_len,
            finger_hist_mean=finger_hist_mean,
            # Multi-hand scenes: a per-hand env view (LFHandView) exposes its
            # actuator-id subset here so the applier drives ONLY that hand's
            # ctrl columns. Single-hand envs don't define it → None → all
            # actuators (bit-identical legacy behaviour).
            act_ids=getattr(sampled_env, "applier_act_ids", None),
        )
        self.action_dim = int(self.applier.action_dim)
        self.n_ctrl     = int(self.applier.n_ctrl)
        self.mocap_id   = int(self.applier.mocap_id)
        # Policy-facing action tensor (the applier exposes the correct writable
        # view — full ``9+n_ctrl`` buffer, or only the finger columns when
        # ``control_wrist=False``).
        self.action_torch = self.applier.action_torch
        # torch-side device handle (``self.device`` is a warp Device, which
        # is not accepted by torch.rand / torch.zeros etc.).
        self.torch_device = self.action_torch.device

        # GPU init kernels (reset_data + warp_pose_init_capture_launch).
        sampled_env.setup_warp_pose_init_kernels(hand_util)
        sampled_env.capture_warp_pose_init_kernels(
            table_height=sampled_env.table_height
        )

        self.capture_step    = sampled_env.capture_step
        self.capture_forward = sampled_env.capture_forward

        # Common body ids (subclass kernels read these).
        # n_obj = 0 sentinel: no object in scene — kernels fall back to a
        # default pose (pos=0, quat=identity) and skip obj-z done logic.
        self.has_object = int(len(sampled_env.renamed_obj_names) > 0)
        if self.has_object:
            self.obj_body_id = int(_mj.mj_name2id( # type: ignore
                self.mjm, _mj.mjtObj.mjOBJ_BODY,  # type: ignore
                sampled_env.renamed_obj_names[0],
            ))
        else:
            self.obj_body_id = 0     # world body — xpos[w, 0] is always (0,0,0)
        self.wrist_body_id = int(_mj.mj_name2id( # type: ignore
            self.mjm, _mj.mjtObj.mjOBJ_BODY,  # type: ignore
            hand_util.rh_wrist_base_name,
        ))

        # Per-world condition (subclass-extensible).
        self.cond = WarpPerWorldCondition(
            NWORLD=self.NWORLD, device=self.device,
            seed=cond_seed + self.handler_idx,
        )
        self._cond_pos_low  = tuple(cond_pos_low)
        self._cond_pos_high = tuple(cond_pos_high)
        self._setup_condition()

        # Episode counters + done bookkeeping.
        self.ep_step_wp     = wp.zeros((self.NWORLD,), dtype=int, device=self.device)
        self.ep_idx_wp      = wp.zeros((self.NWORLD,), dtype=int, device=self.device)
        self.done_mask_wp   = wp.zeros((self.NWORLD,), dtype=int, device=self.device)
        self.done_reason_wp = wp.zeros((self.NWORLD,), dtype=int, device=self.device)
        self.ep_step_torch     = wp.to_torch(self.ep_step_wp)
        self.ep_idx_torch      = wp.to_torch(self.ep_idx_wp)
        self.done_mask_torch   = wp.to_torch(self.done_mask_wp)
        self.done_reason_torch = wp.to_torch(self.done_reason_wp)
        # Truncation flag (1.0 = this step ended the episode by the TIME LIMIT —
        # timeout OR success, both at ``ep_step >= MAX_EPISODE_STEPS`` — as
        # opposed to a failure terminal). Consumed by GAE to BOOTSTRAP the value
        # at the time limit instead of treating it as a hard terminal. Refreshed
        # every ``step()`` (``_update_truncation``) just BEFORE
        # ``_update_ep_counters`` zeros ``ep_step``.
        self.truncation_wp     = wp.zeros((self.NWORLD,), dtype=float, device=self.device)
        self.truncation_torch  = wp.to_torch(self.truncation_wp)

        # Snapshot buffers (for non-done world restore in async reset).
        self._snap_qpos       = wp.zeros_like(self.d.qpos)
        self._snap_qvel       = wp.zeros_like(self.d.qvel)
        self._snap_ctrl       = wp.zeros_like(self.d.ctrl)
        self._snap_mocap_pos  = wp.zeros_like(self.d.mocap_pos)
        self._snap_mocap_quat = wp.zeros_like(self.d.mocap_quat)
        self._nq     = int(self.d.qpos.shape[1])
        self._nv     = int(self.d.qvel.shape[1])
        self._nu     = int(self.d.ctrl.shape[1])
        self._nmocap = int(self.d.mocap_pos.shape[1])

        # Per-hand finger qpos / ctrl randomizer (legacy JAX
        # ``_finger_init_fn`` ported to a single warp kernel).
        self.finger_initializer: Optional[HandFingerInitializer] = None
        if self.randomize_finger_init:
            self.finger_initializer = HandFingerInitializer(
                sampled_env=sampled_env,
                hand_util=hand_util,
                seed=int(finger_init_seed),
                handler_idx=self.handler_idx,
            )
            # Prefer a self-collision-free INIT bank when the task exposes one
            # (``_setup_condition`` above already built it) — so reset never
            # starts in a self-penetrating pose. Falls back to uniform otherwise.
            _cf_table, _cf_n, _cf_qpa = self._collision_free_init_bank()
            if _cf_table is not None and int(_cf_n) > 0:
                self.finger_initializer.attach_collision_free_bank(
                    _cf_table, int(_cf_n), _cf_qpa)

        # Subclass-specific obs / reward setup.
        self._setup_obs_buffers()
        self._setup_reward_done_buffers()

        # Cached info dict — same reference returned every step (values are
        # zero-copy torch views, so they always reflect the latest GPU state).
        # Avoids one Python dict allocation per step in the rollout hot path.
        self._info = {
            "done_mask":   self.done_mask_torch,
            "done_reason": self.done_reason_torch,
            "ep_step":     self.ep_step_torch,
            "ep_idx":      self.ep_idx_torch,
            "truncation":  self.truncation_torch,
        }

    # ── Hooks subclasses must override ───────────────────────────────────

    @property
    def obs_dim(self) -> int:
        raise NotImplementedError

    # ── Async (asymmetric) actor-critic hooks ────────────────────────────
    # Default = symmetric: the critic reads the SAME obs as the actor. An
    # async env overrides these to expose a larger privileged critic obs +
    # its own buffer. Generic code (rollout buffer / train loop / builders)
    # reads these without a getattr guard; symmetric handlers report
    # ``critic_obs_dim == obs_dim`` so nothing downstream changes.
    @property
    def critic_obs_dim(self) -> int:
        return self.obs_dim

    @property
    def critic_obs_torch(self) -> torch.Tensor:
        return self.obs_torch

    def _collision_free_init_bank(self):
        """Hook: return ``(table_wp, n_rows, ctrl_qpa_wp)`` of self-collision-free
        qpos rows (actuator order, with the qpos-address mapping the table was
        built with) for random INIT sampling, or ``(None, 0, None)`` to use the
        uniform sampler. Tasks that build a collision-free qpos bank
        (taxonomy/pinch/bank) override this to reuse it for reset poses too."""
        return None, 0, None

    def _setup_condition(self) -> None:
        """Default: register a uniform target wrist position."""
        register_hand_pose_target(
            self.cond, self.sampled_env, self.hand_util,
            pos_low=self._cond_pos_low, pos_high=self._cond_pos_high,
        )

    def _setup_obs_buffers(self) -> None:
        self.obs_wp    = wp.zeros((self.NWORLD, self.obs_dim),
                                  dtype=float, device=self.device)
        self.obs_torch = wp.to_torch(self.obs_wp)

    def _setup_reward_done_buffers(self) -> None:
        # default: shared (NWORLD,) reward/done float buffers
        self.reward_wp, self.reward_torch = self._alloc_world(float)
        self.done_wp,   self.done_torch   = self._alloc_world(float)

    # ── Allocation helper ───────────────────────────────────────────────
    def _alloc_world(self, dtype=float):
        """Allocate one ``(NWORLD,)`` warp buffer + its zero-copy torch view.
        Avoids the repeated ``wp.zeros + wp.to_torch`` boilerplate."""
        arr = wp.zeros((self.NWORLD,), dtype=dtype, device=self.device)
        return arr, wp.to_torch(arr)

    def eval_ghost_targets(self):
        """Per-world (pos, quat, qpos) for the evaluate.py ghost-hand overlay,
        or ``None`` when this task has no target hand pose to ghost.

        Returns a tuple of numpy arrays
        ``(target_pos (N,3), target_quat (N,4 wxyz), target_qpos (N,n_ctrl))``
        consumed by ``orchestrator.set_ghost_targets(...)``. Task-agnostic
        default is ``None`` — evaluate.py then skips the ghost overlay instead
        of hard-coding any task's cond-field names. Tasks that define a target
        hand pose (e.g. ``hand_pose_tracking``) override this.
        """
        return None

    def eval_ghost_geom_rgba(self):
        """Per-world RGBA override for the evaluate.py ghost-hand geoms, or
        ``None`` when this task gives the ghost no per-geom colour meaning.

        Returns a ``(NWORLD, ghost_ngeom, 4)`` float32 array consumed by
        ``mjwarp_render(target_geom_rgba=...)`` — it recolours the ghost hand
        per world right before it is drawn. Task-agnostic default is ``None``:
        evaluate.py then leaves the ghost in its plain ``setup_ghost_hand_model``
        grey. Tasks whose ghost carries per-finger state (e.g. object_grasping's
        taxonomy active/rest fingers) override this. Mirrors the
        ``eval_ghost_targets`` contract so evaluate.py stays task-agnostic (no
        per-task cond-field lookups in the viewer).
        """
        return None

    def eval_debug_markers(self):
        """Task-supplied 3-D debug markers for the evaluate.py viewer, or
        ``None`` when this task defines no markers **at all**.

        ⚠ ``None`` means "this task never draws markers", NOT "nothing to draw
        this frame" — evaluate.py probes this ONCE at startup (before the first
        ``reset()``, so targets are still zero) to decide whether to bind the
        ``[J]`` toggle. A task that returns ``None`` on an idle frame would
        disable its own overlay for the whole session; express "nothing active
        right now" with an all-False ``found`` instead.

        Third member of the ``eval_ghost_*`` family: the handler owns ALL task
        semantics (which quantity, where it lives in the scene, how it scales)
        and returns plain world-frame geometry; evaluate.py only forwards the
        result to ``mj_env.plot_parallel_debug_markers`` and toggles it with a
        key. No task ever needs a line of viewer code — e.g. force tracking
        draws its per-finger target/measured contact force as arrows along the
        pinch axis purely by overriding this method.

        Returns a **list of marker groups**, each a dict::

            {
              "kind":  "arrow" | "line" | "sphere" | "text",   # required
              "pos":   (NWORLD, K, 3) float,   # anchor, WORLD frame, no grid
                                               #   offset (viewer adds it)
              "vec":   (NWORLD, K, 3) float,   # arrow/line only: tip = pos+vec
              "found": (NWORLD, K)   float/bool,  # optional; <=0.5 → skip item
              "rgba":  (4,) | (K, 4) | (NWORLD, K, 4) float,   # optional
              "r":     float,                  # shaft / sphere radius [m]
              "label": (NWORLD, K) str array,  # optional text at the anchor
                                               #   ("text" kind draws only this)
              "name":  str,                    # optional, for diagnostics
            }

        ``K`` is arbitrary and may differ per group (per-finger, per-contact,
        …). Groups are drawn in list order, so put translucent/large things
        first. ``"text"``-kind groups and ``label`` fields together form the
        **label layer**: evaluate.py's ``[U]`` key hides just that layer
        (keeping the 3-D geometry), so put every textual readout there rather
        than baking numbers into geometry. All arrays are numpy on the HOST — a handler that keeps its
        state on the GPU should sync here (once per rendered frame at viewer
        NWORLD ≈ 16, so the copy is free); returning ``None`` costs nothing for
        every task that does not override this.
        """
        return None

    def eval_overlay_lines(self, world_idx=None):
        """Task-supplied text rows for the evaluate.py viewer overlay, or
        ``None`` (default) when the task adds nothing to the generic readout.

        Returns a list of ``(text1, text2)`` pairs handed verbatim to
        ``viewer_text_overlay(text1=..., text2=...)`` — the same left-label /
        right-value layout the viewer already uses. ``world_idx`` is the
        double-click-selected world (``None`` when nothing is selected), so a
        task can report per-world numbers (e.g. that world's target vs measured
        per-finger force) instead of only aggregates.

        Keeps the numeric readout in the same place as the 3-D markers
        (:meth:`eval_debug_markers`): the handler owns the units and the
        formatting, evaluate.py just prints the rows.
        """
        return None

    # ── Generic reward-weight curriculum (yaml: handler.curriculum.reward) ──
    #: Schedule dict ``{knob_name: (final_value, decay_start_step,
    #: decay_end_step[, "log"])}`` consumed by :func:`apply_reward_curriculum`
    #: every ``on_train_progress`` call. ``knob_name`` must belong to one of
    #: this handler's ``reward_*`` CONFIG groups (fail-fast otherwise). The
    #: knob is auto-registered into every subclass's ``curriculum`` CONFIG
    #: group by ``__init_subclass__``, so ANY task accepts::
    #:
    #:     handler:
    #:       curriculum:
    #:         reward:
    #:           W_MIMIC:   [0.5, 10_000_000, 40_000_000]         # linear
    #:           K_FINGER:  [50.0, 20_000_000, 0, log]            # geometric interpolation, end<=0 → total
    #:
    #: ``None`` (default) → off (byte-identical to before). Per entry,
    #: ``start_step < 0`` (no window) is that knob's OFF switch (except that
    #: with :attr:`reward_gate` set it becomes a gate-only knob).
    reward: Optional[Dict[str, Any]] = None

    #: Success-gate (yaml ``handler.curriculum.reward_gate``):
    #: ``{metric, threshold, progress_per_eval[, max_progress]}`` — every time an
    #: eval reaches ``metric ≥ threshold``, the reward-curriculum progress
    #: (0=base, 1=final) advances by ``progress_per_eval``. See :func:`apply_reward_gate`.
    #: ``None`` (default) → off; no-op without a :attr:`reward` schedule. The
    #: progress is state, so it is stored in the ckpt (:meth:`curriculum_state`)
    #: and restored on resume.
    reward_gate: Optional[Dict[str, Any]] = None

    #: Global curriculum start step (yaml ``handler.curriculum.reward_start_step``):
    #: Before this global control-step the reward curriculum is **frozen entirely**:
    #: every knob keeps its base (yaml) value. Both paths are blocked:
    #:   * step schedule — effective start per knob = ``max(knob_start, reward_start_step)``
    #:   * success-gate — no progress accumulation from eval passes before this step
    #:     (and progress restored on resume is not applied to the weights before it either)
    #: 0 (default) → byte-identical to before. E.g. ``4e+7`` → penalty ramps/gate
    #: start at 40M steps (learn grasping first with base weights).
    reward_start_step: float = 0.0

    def on_train_progress(self, cur_step: int, total_steps: int) -> None:
        """Hook called once per training iteration by the training loop with
        the current global control-step count and the total budget.

        The BASE implementation applies the generic reward curriculum
        (:attr:`reward` / yaml ``handler.curriculum.reward``) — task overrides
        MUST call ``super().on_train_progress(cur_step, total_steps)`` so the
        shared scheduler keeps running alongside their task-specific curricula
        (e.g. object_grasping's rule-based wrist-z lift decay). The training
        loop just reports progress; the env decides what to do.
        """
        apply_reward_curriculum(self, cur_step, total_steps)

    def on_eval_metrics(self, ev: Dict[str, Any], *, validate_only: bool = False) -> None:
        """Hook: the train loop passes the ``run_eval`` result dict right after every eval.
        The base implementation handles the success-gate (:func:`apply_reward_gate`);
        it takes effect on the weights from the next iteration's ``on_train_progress``.
        Keep the super() call when overriding. ``validate_only=True`` only validates
        the configuration (fail-fast; train.py baseline eval)."""
        apply_reward_gate(self, ev, validate_only=validate_only)

    def curriculum_state(self) -> Dict[str, Any]:
        """Curriculum state for the ckpt — only what is not a pure function of the
        step (currently just the gate progress). Step-based progress is
        recomputed from cur_step on resume."""
        return {"gate_progress": float(getattr(self, "_gate_progress", 0.0))}

    def load_curriculum_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore the :meth:`curriculum_state` payload (resume path only)."""
        self._gate_progress = float((state or {}).get("gate_progress", 0.0))
        if self._gate_progress:
            print(f"[reward-gate] resume — restored progress {self._gate_progress:.2f} (0=base, 1=final)")

    def bind_policy(self, policy) -> None:
        """Bridge this handler's policy (train.py calls it once before the loop).

        Kept ``None`` by default; a curriculum that needs to touch a POLICY
        parameter (e.g. the geom-tracking σ-floor ``min_std`` ramp in
        ``on_train_progress``) reads ``self._curr_policy``. Tasks that don't
        need it simply never dereference it."""
        self._curr_policy = policy

    def _collect_obs_kernel(self) -> None:
        """Collect the task-specific observation."""
        raise NotImplementedError

    def _collect_reward_done_kernel(self) -> None:
        """Collect the task-specific reward and done."""
        raise NotImplementedError

    def _collect_success_kernel(self) -> None:
        """Evaluate the task-specific success condition and (optionally)
        promote ``done`` / ``done_mask`` / ``done_reason`` accordingly.

        **Every** :class:`SubEnvHandler` subclass MUST override this hook —
        the contract is enforced via the ``NotImplementedError`` below so
        new tasks cannot silently fall back to "no success ever". If your
        task genuinely has no success criterion, override with an empty
        body to make that intent explicit.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement `_collect_success_kernel`. "
            "Override with the task's success criterion (or an explicit "
            "empty body if the task has no success notion)."
        )

    def _reset_task_buffers(self) -> None:
        """Hook for zeroing task-specific buffers (e.g. success-streak
        counters) on a hard ``reset()``. Default is a no-op.
        """
        pass

    # ── Public API (single-handler view; called from BaseRLEnv too) ─────

    def snapshot_train_state(self) -> None:
        """Save the full per-world ROLLOUT state so a periodic eval — which hard-
        resets this SHARED env (``run_eval`` → ``h.reset()``) — can be made
        TRANSPARENT to training.

        ``run_eval`` restarts every world at ``ep_step=0`` (no stagger) and runs
        a fresh ``eval_length`` window, clobbering the training sim + episode
        clocks + per-world targets. Without restoring them the training rollout
        does NOT continue where it left off: ``ep_step`` is reset (→ all worlds
        re-synchronized → timeout BURSTS once the policy survives the eval
        window), the object is re-posed, and the per-world targets are
        re-sampled mid-episode. Snapshot here, ``restore_train_state`` after eval,
        and the rollout's step index / physics / targets all carry over intact.

        Captures the same physics roots the per-world reset already trusts
        (``qpos/qvel/ctrl/mocap``; ``forward`` rebuilds the rest) plus episode
        clocks, bonus streak, last action, and every per-world condition field.
        ``prev_obs`` in the train loop is the matching post-rollout obs and is
        preserved separately (NOT overwritten), so obs need not be snapshotted."""
        d = self.d
        if not hasattr(self, "_train_snap"):
            snap = dict(
                qpos=wp.zeros_like(d.qpos),           qvel=wp.zeros_like(d.qvel),
                ctrl=wp.zeros_like(d.ctrl),           mocap_pos=wp.zeros_like(d.mocap_pos),
                mocap_quat=wp.zeros_like(d.mocap_quat),
                ep_step=wp.zeros_like(self.ep_step_wp), ep_idx=wp.zeros_like(self.ep_idx_wp),
            )
            if hasattr(self, "bonus_streak_wp"):
                snap["bonus_streak"] = wp.zeros_like(self.bonus_streak_wp)              # type: ignore
            if hasattr(self, "last_action_wp"):
                snap["last_action"] = wp.zeros_like(self.last_action_wp)                # type: ignore
            snap["cond"] = {n: wp.zeros_like(f["gpu"])                                  # type: ignore
                            for n, f in self.cond._fields.items()}
            self._train_snap = snap
        s = self._train_snap
        wp.copy(s["qpos"], d.qpos);             wp.copy(s["qvel"], d.qvel)
        wp.copy(s["ctrl"], d.ctrl);             wp.copy(s["mocap_pos"], d.mocap_pos)
        wp.copy(s["mocap_quat"], d.mocap_quat)
        wp.copy(s["ep_step"], self.ep_step_wp); wp.copy(s["ep_idx"], self.ep_idx_wp)
        if "bonus_streak" in s: wp.copy(s["bonus_streak"], self.bonus_streak_wp)        # type: ignore
        if "last_action"  in s: wp.copy(s["last_action"],  self.last_action_wp)         # type: ignore
        for n, buf in s["cond"].items():                                                # type: ignore
            wp.copy(buf, self.cond._fields[n]["gpu"])                                  

    def restore_train_state(self) -> None:
        """Inverse of :meth:`snapshot_train_state` — copy the saved rollout state
        back and rebuild derived physics with ``forward``. ``prev_obs`` (the
        matching post-rollout obs) is preserved by the train loop, so no obs
        recompute is needed here."""
        if not hasattr(self, "_train_snap"):
            return
        d = self.d; s = self._train_snap
        wp.copy(d.qpos, s["qpos"]);             wp.copy(d.qvel, s["qvel"])
        wp.copy(d.ctrl, s["ctrl"]);             wp.copy(d.mocap_pos, s["mocap_pos"])
        wp.copy(d.mocap_quat, s["mocap_quat"])
        wp.copy(self.ep_step_wp, s["ep_step"]); wp.copy(self.ep_idx_wp, s["ep_idx"])
        if "bonus_streak" in s: wp.copy(self.bonus_streak_wp, s["bonus_streak"])      # type: ignore
        if "last_action"  in s: wp.copy(self.last_action_wp,  s["last_action"])       # type: ignore
        for n, buf in s["cond"].items():                                              # type: ignore
            wp.copy(self.cond._fields[n]["gpu"], buf)
        mjwarp.forward(self.m, self.d)

    def reset(self) -> torch.Tensor:
        """Hard reset of all worlds in this sub-env."""
        mjwarp.reset_data(self.m, self.d)
        self._reset_hand_init()
        return self._reset_finish()

    def _reset_hand_init(self) -> None:
        """Finger init + this hand's pose-init graph (after ``reset_data``)."""
        # Per-hand random finger qpos / ctrl init (legacy JAX ``_finger_init_fn``).
        # Runs **before** the graph launch so the floor lift-out at the end of
        # the pose-init graph sees contacts that include the random finger pose
        # and resolves penetration (if finger init came after, fingertips could
        # start penetrating the floor again).
        if self.finger_initializer is not None:
            self.finger_initializer.apply_all()
        self.sampled_env.warp_pose_init_capture_launch()
        # Finger-only (``control_wrist=False``): the pose-init graph above
        # randomizes the wrist per world (grasping-style spawn); overwrite it
        # with the model-default (``qpos0``) wrist pose so the held wrist sits
        # at the canonical pose in every world (mirrors fix_wrist_pose).
        if not self._control_wrist:
            self.sampled_env.fix_wrist_pose_gpu(self.hand_util)

    def _reset_finish(self) -> torch.Tensor:
        """``forward`` + episode buffers + task buffers + initial obs."""
        mjwarp.forward(self.m, self.d)

        self.ep_step_wp.zero_()   # all worlds start the episode clock at 0 (no random stagger)
        self.ep_idx_wp.zero_()
        self.done_mask_wp.zero_()
        self.done_reason_wp.zero_()
        self.truncation_wp.zero_()
        # Task-specific buffer reset (e.g. success-streak counters).
        self._reset_task_buffers()

        self.cond.reset()
        # Optional post-cond hook (subclasses may need cond fields to derive
        # extra per-world quantities — e.g. FK on target_qpos for af-xpos
        # tracking). Default no-op.
        self._post_cond_update_hook(world_mask_np=None)
        # integ action: setpoint := qpos for all worlds (see applier comment; no-op in other modes)
        self.applier.sync_setpoint_to_qpos(None)
        self._collect_obs_kernel()
        self._sanitize_obs()
        return self.obs_torch

    # ── Dynamic target switch (shared helpers; impl is task-specific) ─────
    def _resample_switch_step(self, world_mask_np=None) -> None:
        """Schedule a new mid-episode switch tick for the affected worlds.

        Each selected world draws ``target_switch_step[w] ~ Uniform[lo, hi)``
        with ``lo = floor(DYNAMIC_TARGET_MIN_FRAC·MAX_EPISODE_STEPS)`` and
        ``hi = ceil(DYNAMIC_TARGET_MAX_FRAC·MAX_EPISODE_STEPS)``. ``world_mask_np
        = None`` → all worlds (hard reset). No-op when the feature is disabled or
        the subclass never allocated ``target_switch_step``. Task-agnostic —
        shared by every subclass that opts into dynamic target switching.
        """
        if not self.DYNAMIC_TARGET_ENABLED:
            return
        if self.target_switch_step_torch is None:   # type-narrowing guard (Optional → Tensor)
            return
        max_ep = int(self.MAX_EPISODE_STEPS)
        lo = max(0,      int(np.floor(self.DYNAMIC_TARGET_MIN_FRAC * max_ep)))
        hi = max(lo + 1, int(np.ceil( self.DYNAMIC_TARGET_MAX_FRAC * max_ep)))
        hi = min(hi, max_ep)
        if hi <= lo:
            hi = lo + 1
        dev = self.target_switch_step_torch.device
        if world_mask_np is None:
            new_vals = np.random.randint(lo, hi, size=self.NWORLD).astype(np.int32)
            self.target_switch_step_torch.copy_(torch.from_numpy(new_vals).to(dev))
        else:
            mask = np.asarray(world_mask_np, dtype=bool)
            n_active = int(mask.sum())
            if n_active == 0:
                return
            new_vals = np.random.randint(lo, hi, size=n_active).astype(np.int32)
            mask_t   = torch.from_numpy(mask).to(dev)
            self.target_switch_step_torch[mask_t] = torch.from_numpy(new_vals).to(dev)

    def _maybe_dynamic_target_switch(self) -> None:
        """Fire a mid-episode target switch for worlds whose ``ep_step`` reached
        their scheduled ``target_switch_step``.

        Called from ``step()`` (after success, before obs) when
        ``DYNAMIC_TARGET_ENABLED``. **Task-specific** — each subclass re-samples
        its own target cond fields for the fired worlds (use
        ``_dynamic_target_switch_mask_kernel`` for the per-world trigger mask)
        and re-collects obs. The base provides no implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} sets DYNAMIC_TARGET_ENABLED but does not "
            "implement _maybe_dynamic_target_switch()."
        )

    def step(
        self,
        action_torch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Phases are split so a shared-scene env (DualHandSubEnv: several handlers
        # on ONE mjwarp data) can apply every hand's action, step the physics ONCE
        # and then observe every hand — see ``BaseRLEnv.step``.
        self._step_apply(action_torch)
        self._step_physics()
        return self._step_observe()

    def _step_apply(self, action_torch: torch.Tensor) -> None:
        """Upload + apply this hand's Δ-action to mocap/ctrl (no physics)."""
        # 1 zero-copy upload action (no-op if caller wrote into the action view).
        if action_torch.data_ptr() != self.action_torch.data_ptr():
            self.action_torch.copy_(action_torch)
        # 2 apply Δ-action → mocap/ctrl, advance physics ``sim_nstep`` times.
        # ``_substep_sensor_hook`` (if any) is called every substep: an extension
        # point for sim-rate sampling of contact sensors to avoid Nyquist aliasing
        # (see the class attribute comment).
        self.applier.apply()

    def _step_physics(self) -> None:
        """Advance the (possibly shared) scene ``sim_nstep`` sub-steps."""
        _sub_hook = self._substep_sensor_hook
        if _sub_hook is None:
            for _ in range(self.sim_nstep):
                wp.capture_launch(self.capture_step.graph)
        else:
            for _ in range(self.sim_nstep):
                wp.capture_launch(self.capture_step.graph)
                _sub_hook()

    def _step_observe(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Reward / done / success / obs / counters for this hand."""
        # 3. reward+done / success / ep counters. Success may override
        # done/mask/reason (e.g. promote a timeout into a terminal success),
        # so it MUST run before ``_update_ep_counters``.
        self._collect_reward_done_kernel()
        self._collect_success_kernel()
        # 4. Mid-episode dynamic target switch (no-op unless the subclass sets
        # ``DYNAMIC_TARGET_ENABLED`` and implements ``_maybe_dynamic_target_switch``).
        # Runs BEFORE obs collection so a switched world's obs reflects the new
        # target in the tuple returned this step.
        if self.DYNAMIC_TARGET_ENABLED:
            self._maybe_dynamic_target_switch()
        self._collect_obs_kernel() # 5. collect obs
        if _GRIT_DEBUG_NAN:
            self._maybe_inject_test_nan()          # one-shot test NaN (validates .log path)
            self._debug_check_finite(tag="step")   # locate FIRST non-finite (pre-sanitize)
        self._sanitize_obs()       # 5b. NaN/Inf safety net before obs reaches policy/buffer
        self._update_truncation()  # 5c. flag time-limit truncation (reads ep_step BEFORE it is zeroed)
        self._update_ep_counters() # 6. update ep counters (zeros ep_step on done)
        return self.obs_torch, self.reward_torch, self.done_torch, self._info

    def _maybe_inject_test_nan(self) -> None:
        """Test hook: with ``GRIT_NAN_INJECT_COL=<col>`` set (and GRIT_DEBUG_NAN=1),
        poke a NaN into ``obs[world 0, col]`` exactly ONCE so the detection +
        ``.log`` path can be validated end-to-end. No-op in normal runs (the env
        var is unset)."""
        if _GRIT_NAN_INJECT_COL == "" or getattr(self, "_nan_injected", False):
            return
        try:
            col = int(_GRIT_NAN_INJECT_COL) % int(self.obs_dim)
            self.obs_torch[0, col] = float("nan")
            self._nan_injected = True
            print(f"[NAN-DEBUG] injected test NaN at obs[world 0, col {col}]", flush=True)
        except Exception as e:
            print(f"[nan-inject] failed: {e!r}", flush=True)

    def _sanitize_obs(self) -> None:
        """In-place clean the obs view so a blown-up world can never feed NaN/Inf
        into the policy / replay buffer (which would poison every parameter via
        ``loss.backward()``/``optimizer.step()``).

        ``obs_torch`` is a zero-copy view of ``obs_wp`` so this fixes the buffer
        the policy reads. ``nan_to_num_`` is a cheap elementwise op (no host
        sync), so it stays on the per-step hot path. Root-cause visibility is
        provided separately by the ``nonfinite`` (reason 8) done counter and the
        ``GRIT_DEBUG_NAN`` tracer — this is only the safety net.
        """
        torch.nan_to_num_(self.obs_torch, nan=0.0,
                          posinf=_OBS_SANITIZE_BIG, neginf=-_OBS_SANITIZE_BIG)

    def _debug_check_finite(self, tag: str = "") -> Optional[str]:
        """Diagnostic (GRIT_DEBUG_NAN=1): report the FIRST non-finite tensor in
        the order action → physics-state → obs, with offending world indices.

        Lets us tell apart the root cause: a physics blow-up shows up in
        ``qpos``/``qvel`` BEFORE obs; an obs-term math bug shows up only in
        ``obs`` (with the offending columns). Forces GPU→CPU syncs, so it is
        gated behind the env flag and off the production path.
        """
        checks = [
            ("action", self.action_torch),
            ("qpos",   wp.to_torch(self.d.qpos)),
            ("qvel",   wp.to_torch(self.d.qvel)),
            ("xpos",   wp.to_torch(self.d.xpos)),
            ("xmat",   wp.to_torch(self.d.xmat)),
            ("cvel",   wp.to_torch(self.d.cvel)),
            ("obs",    self.obs_torch),
        ]
        for name, t in checks:
            bad = ~torch.isfinite(t)
            if not bool(bad.any()):
                continue
            flat   = bad.reshape(bad.shape[0], -1).any(dim=1)
            widx   = torch.nonzero(flat).flatten()[:8].tolist()
            nbad_w = int(flat.sum())
            detail = ""
            if name == "obs":
                # Which obs columns went non-finite → map to the obs term name(s)
                # via the handler's ``obs_term_layout`` (e.g. wrist_obj_vel, rot6d).
                cols   = torch.nonzero(bad.any(dim=0)).flatten().tolist()
                layout = getattr(self, "obs_term_layout", None)
                if layout:
                    terms = []
                    for nm, s, e in layout:
                        hit = [c - s for c in cols if s <= c < e]
                        if hit:
                            terms.append(f"{nm}[{hit[0]}..{hit[-1]}]" if len(hit) > 1
                                         else f"{nm}[{hit[0]}]")
                    detail = f"  obs_terms={terms}  obs_cols={cols[:24]}"
                else:
                    detail = f"  obs_cols={cols[:24]}"
            ts  = time.strftime("%Y-%m-%d %H:%M:%S")
            tg  = getattr(self, "tag", getattr(self, "name", "?"))
            msg = (f"[{ts}] tag={tg} {tag}: FIRST non-finite tensor='{name}' "
                   f"n_worlds={nbad_w} worlds={widx}{detail}")
            print("[NAN-DEBUG] " + msg, flush=True)
            _write_nan_log(msg)              # persist to <repo>/output/nan_debug.log
            return name
        return None

    @contextmanager
    def eval_context(self):
        """Scoped switch to evaluation mode (``self.eval_mode = True``).

        Inside the ``with`` block:
          * ``per_world_reset_if_done()`` is a no-op (no auto-reset).
          * ``_collect_success_kernel()`` is a no-op (success no longer
            terminates the episode, so target tracking continues).

        Other done sources (obj_z / timeout / wrist_up) still fire on
        ``done_mask`` for diagnostics but trigger no reset.

        Usage::

            with h.eval_context():
                obs = h.reset()
                for t in range(eval_length):
                    obs, rew, done, info = h.step(action)
                    # No per-world reset; episodes play out for the full
                    # eval window — measures absolute reward stability.
        """
        prev = self.eval_mode
        self.eval_mode = True
        try:
            yield self
        finally:
            self.eval_mode = prev

    def per_world_reset_if_done(self) -> None:
        """Reset only the worlds in this sub-env with ``done_mask == 1``.

        **No-op when ``self.eval_mode`` is True** — eval rollouts measure
        absolute reward over a fixed window, so worlds must keep their
        state / target / ep_step instead of being recycled.

        **Sync-free hot path** — every step of the rollout invokes this
        unconditionally, so the implementation must avoid CPU↔GPU syncs
        that block the python thread on async warp launches:

          1. Snapshot all 5 state tensors (cheap ``wp.copy``).
          2. ``mjwarp.reset_data`` + GPU init kernel + finger init —
             touch all worlds.
          3. ``_restore_non_done_worlds`` (consumes ``done_mask_wp`` on
             GPU) — restores worlds with ``done_mask[w]==0`` from the
             snapshot, leaving only ``done`` worlds with the fresh init.
          4. ``capture_forward.graph`` to recompute derived state.
          5. ``cond.sample_all_gpu(mask_wp=done_mask_wp)`` — re-samples
             per-world targets only on done worlds, sync-free thanks to
             the GPU samplers installed by ``register_hand_pose_target``.
          6. ``_collect_obs_kernel`` to refresh ``obs_torch``.

        No return value — the rollout loop reads done counts off the
        zero-copy ``done_mask_torch`` view at iteration end (or via a
        per-iter GPU histogram) instead of paying a per-step ``.item()``.
        """
        if self.eval_mode:
            return  # eval mode: keep worlds intact for absolute-reward measurement.
        self._pwr_flag_and_snapshot()
        mjwarp.reset_data(self.m, self.d)
        self._pwr_reinit()
        self._pwr_restore()
        wp.capture_launch(self.capture_forward.graph)
        self._pwr_finish()

    def _pwr_flag_and_snapshot(self) -> None:
        """Flag non-finite worlds as done, snapshot the FULL scene state."""
        # Root-cause containment: flag worlds whose physics state went non-finite
        # (mjwarp blow-up) as done BEFORE the snapshot, so they route to a fresh
        # init instead of having their NaN state snapshot-restored below. The
        # done kernel cannot catch these (NaN threshold comparisons are False).
        wp.launch(
            _flag_nonfinite_worlds, dim=self.NWORLD,
            inputs=[self.d.qpos, self.d.qvel, self._nq, self._nv, 1.0e30,
                    self.done_mask_wp, self.done_reason_wp],
        )

        wp.copy(self._snap_qpos,       self.d.qpos)
        wp.copy(self._snap_qvel,       self.d.qvel)
        wp.copy(self._snap_ctrl,       self.d.ctrl)
        wp.copy(self._snap_mocap_pos,  self.d.mocap_pos)
        wp.copy(self._snap_mocap_quat, self.d.mocap_quat)

    def _pwr_reinit(self) -> None:
        """Masked finger init + this hand's pose-init graph (after ``reset_data``)."""
        # Per-hand random finger init runs against ``done_mask_wp`` on the
        # GPU — only worlds flagged as done get fresh finger qpos / ctrl.
        # ``apply_all`` is the all-worlds variant (kept for hard reset).
        # Runs **before** the graph launch so the floor lift-out at the end of
        # the graph sees contacts that include the random finger pose (same
        # order as reset()).
        if self.finger_initializer is not None:
            self.finger_initializer.apply_masked(self.done_mask_wp)
        self.sampled_env.warp_pose_init_capture_launch()
        # Finger-only (``control_wrist=False``): pin the wrist back to the
        # model-default pose after the randomized pose-init graph. Launched
        # over ALL worlds (sync-free GPU kernel) — non-done worlds are
        # restored from the snapshot right below, so only done worlds keep
        # this write (and in finger-only mode every world's wrist is the
        # same default pose anyway).
        if not self._control_wrist:
            self.sampled_env.fix_wrist_pose_gpu(self.hand_util)

    def _pwr_restore(self) -> None:
        """Restore every non-done world from the snapshot (whole scene)."""
        wp.launch(
            _restore_non_done_worlds, dim=self.NWORLD,
            inputs=[
                self._snap_qpos, self._snap_qvel, self._snap_ctrl,
                self._snap_mocap_pos, self._snap_mocap_quat,
                self.d.qpos, self.d.qvel, self.d.ctrl,
                self.d.mocap_pos, self.d.mocap_quat,
                self.done_mask_wp,
                self._nq, self._nv, self._nu, self._nmocap,
            ],
        )

    def _pwr_finish(self) -> None:
        """Per-hand post-reset: cond resample, hooks, setpoint sync, obs."""
        # Per-world cond re-sampling. ``cond.reset`` runs GPU samplers
        # first (sync-free, masked by ``done_mask_wp``) and then CPU
        # samplers if any field registered one (e.g. taxonomy injection
        # over uniform target_qpos). The CPU pass derives a host bool
        # mask from ``done_mask_wp`` — costs one sync per step ONLY when
        # CPU samplers are present; the all-GPU path stays sync-free.
        self.cond.reset(mask_wp=self.done_mask_wp)

        # Post-cond hook: only the worlds that just reset. Sync the GPU
        # mask once when the subclass needs it (default = no-op, no sync).
        if self._has_post_cond_update_hook():
            mask_np = self.done_mask_torch.cpu().numpy().astype(bool)
            self._post_cond_update_hook(world_mask_np=mask_np)

        # integ action: align the setpoint of reset worlds with qpos (stale ctrl=0 → prevents force spikes)
        self.applier.sync_setpoint_to_qpos(self.done_mask_wp)
        self._collect_obs_kernel()
        self._sanitize_obs()

    def _has_post_cond_update_hook(self) -> bool:
        """Override to return True if your subclass needs ``done_mask_np`` in
        ``_post_cond_update_hook``. Defaults to False so the base class skips
        the per-step CPU sync for hand-tasks that don't need it."""
        return False

    def _post_cond_update_hook(self, world_mask_np=None) -> None:
        """Optional hook fired right after ``cond`` has been (re)sampled,
        before ``_collect_obs_kernel``. ``world_mask_np`` is ``None`` for
        full resets and a ``(NWORLD,) bool`` array for per-world resets
        (only worlds with ``True`` were just re-sampled)."""
        pass

    def _update_truncation(self) -> None:
        """Flag time-limit truncation for GAE bootstrapping.

        ``truncation[w] = 1`` iff world ``w`` is done THIS step AND it ended by
        reaching the time limit (``ep_step >= MAX_EPISODE_STEPS`` — i.e. timeout
        or a timeout-promoted success), as opposed to a failure terminal
        (obj_z / wrist_up / obj_xy / crush / obj_fell, which fire at
        ``ep_step < MAX_EPISODE_STEPS``). GAE uses this to BOOTSTRAP ``V(s_T)`` at
        the time limit instead of treating it as a hard terminal (zero future
        value). MUST run before :meth:`_update_ep_counters` zeros ``ep_step``."""
        is_trunc = (self.done_mask_torch != 0) & \
                   (self.ep_step_torch >= int(self.MAX_EPISODE_STEPS))
        self.truncation_torch.copy_(is_trunc.to(self.truncation_torch.dtype))

    def _update_ep_counters(self) -> None:
        wp.launch(
            _update_ep_counters_kernel, dim=self.NWORLD,
            inputs=[self.done_mask_wp, self.ep_step_wp, self.ep_idx_wp],
        )


# Initialise the base class's flat aliases from its own ``*_KEYS_GROUPS``.
# ``__init_subclass__`` only fires for *sub*classes, so the base needs an
# explicit nudge to keep the legacy flat-tuple API working for code that
# reads ``SubEnvHandler.BUILDER_KEYS`` directly.
SubEnvHandler.BUILDER_KEYS = _flatten_groups(SubEnvHandler.BUILDER_KEYS_GROUPS)
SubEnvHandler.CONFIG_KEYS  = _flatten_groups(SubEnvHandler.CONFIG_KEYS_GROUPS)


# ──────────────────────────────────────────────────────────────────────────
# BaseRLEnv — list-of-handlers wrapper
# ──────────────────────────────────────────────────────────────────────────


class BaseRLEnv:
    """Wraps ``orchestrator.env_list`` as a list of :class:`SubEnvHandler`.

    Subclasses set the ``HANDLER_CLS`` class attribute to the concrete
    :class:`SubEnvHandler` subclass for the task.

    Convention:

      * ``handlers[i]`` is the handler for ``orchestrator.env_list[i]``.
        For single-hand setups (``len(env_list) == 1``) this list has one
        entry; users typically grab ``handler = env.handlers[0]`` and use
        the per-handler API directly to get single tensors back.
      * ``num_envs`` = ``sum(h.NWORLD for h in handlers)`` — the total
        number of parallel worlds across all sub-envs.
      * ``reset`` / ``step`` / ``per_world_reset_if_done`` dispatch over
        all handlers and return per-handler lists. ``num_envs`` is
        proportional to ``len(env_list)`` as requested.

    Multi-hand sub-envs (``grit_multi_hand`` config) typically have
    different ``obs_dim`` / ``action_dim`` per handler, so the API
    returns lists rather than concatenated tensors.
    """

    #: Concrete :class:`SubEnvHandler` subclass for this task.
    HANDLER_CLS: Type[SubEnvHandler] = SubEnvHandler

    def __init__(
        self,
        orchestrator,
        hand_util_list: Optional[Sequence] = None,
        sim_nstep: int = 5,
        xyz_scale: float = 0.005,
        finger_scale: float = 0.02,
        rot_scale: float = 0.1,
        cond_seed: int = 42,
        cond_pos_low: tuple = (-0.20, -0.20, 0.10),
        cond_pos_high: tuple = (0.20, 0.20, 0.50),
        randomize_finger_init: Optional[bool] = None,
        finger_init_seed: int = 42,
        # ── Deadzone pass-through (per-handler, forwarded to applier) ──
        # All four default to a no-op so existing call sites keep their
        # behaviour. Set ``use_deadzone=True`` + per-component thresholds
        # to make the wrist / fingers "hold" once the policy command
        # decays below the configured magnitude.
        use_deadzone:        bool  = False,
        xyz_deadzone_m:      float = 0.0,
        rot_deadzone_deg:    float = 0.0,
        finger_deadzone_deg: float = 0.0,
        finger_abs_action:   bool  = False,
        finger_action_mode:  str   = "delta",
        finger_lead_max_rad: float = 0.2,
        finger_leak:         float = 1.0,
        finger_hist_len:     int   = 4,
        finger_hist_mean:    bool  = True,
        **handler_kwargs,
    ):
        self.orchestrator = orchestrator

        # Resolve per-sub-env hand_util.
        if hand_util_list is None:
            hand_util_list = self._derive_hand_util_list(orchestrator)
        if len(hand_util_list) != len(orchestrator.env_list):
            raise ValueError(
                f"hand_util_list length ({len(hand_util_list)}) does not match "
                f"orchestrator.env_list length ({len(orchestrator.env_list)})"
            )
        self.hand_util_list = list(hand_util_list)

        # Build one handler per sub-env.
        self.handlers: List[SubEnvHandler] = []
        for i, (sampled_env, hu) in enumerate(
            zip(orchestrator.env_list, self.hand_util_list)
        ):
            handler = self.HANDLER_CLS(
                sampled_env=sampled_env,
                hand_util=hu,
                sim_nstep=sim_nstep,
                xyz_scale=xyz_scale,
                finger_scale=finger_scale,
                rot_scale=rot_scale,
                cond_seed=cond_seed,
                cond_pos_low=cond_pos_low,
                cond_pos_high=cond_pos_high,
                handler_idx=i,
                randomize_finger_init=randomize_finger_init,
                finger_init_seed=finger_init_seed,
                use_deadzone=use_deadzone,
                xyz_deadzone_m=xyz_deadzone_m,
                rot_deadzone_deg=rot_deadzone_deg,
                finger_deadzone_deg=finger_deadzone_deg,
                finger_abs_action=finger_abs_action,
                finger_action_mode=finger_action_mode,
                finger_lead_max_rad=finger_lead_max_rad,
                finger_leak=finger_leak,
                finger_hist_len=finger_hist_len,
                finger_hist_mean=finger_hist_mean,
                **handler_kwargs,
            )
            self.handlers.append(handler)

        # ── shared scene (DualHandSubEnv views): all handlers drive ONE mjwarp
        # data → apply all actions, step physics once, observe all; resets are
        # coordinated with a common (OR-ed) done mask. See step()/reset().
        self.shared_scene: bool = (
            len(self.handlers) > 1
            and all(h.d is self.handlers[0].d for h in self.handlers[1:]))
        self.nworld_per_handler: List[int] = [h.NWORLD for h in self.handlers]
        self.num_envs: int                 = int(sum(self.nworld_per_handler))
        self.sim_nstep                     = int(sim_nstep)
        self.torch_device                  = self.handlers[0].torch_device
        self.device                        = self.handlers[0].device

    # ── Public API (lists across handlers) ───────────────────────────────

    def reset(self) -> List[torch.Tensor]:
        """Hard reset every handler. Returns per-handler initial obs."""
        if not self.shared_scene:
            return [h.reset() for h in self.handlers]
        h0 = self.handlers[0]
        mjwarp.reset_data(h0.m, h0.d)              # once for the shared scene
        for h in self.handlers:
            h._reset_hand_init()                   # each hand: finger init + its wrist/obj pose graph
        return [h._reset_finish() for h in self.handlers]

    def step(
        self,
        actions,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[dict]]:
        """Step every handler.

        Args:
            actions: either a list of length ``len(handlers)`` (one
                ``(NWORLD_i, action_dim_i)`` tensor per handler) **or**, when
                there is a single handler, a single tensor of shape
                ``(NWORLD, action_dim)`` (auto-wrapped).

        Returns:
            ``(obs_list, reward_list, done_list, info_list)`` — each a list
            of length ``len(handlers)``.
        """
        actions = self._coerce_action_list(actions)

        obs_list, rew_list, done_list, info_list = [], [], [], []
        if self.shared_scene:
            for h, a in zip(self.handlers, actions):
                h._step_apply(a)                   # every hand's Δ-action into the shared ctrl/mocap
            self.handlers[0]._step_physics()       # physics ONCE
            for h in self.handlers:
                o, r, d, info = h._step_observe()
                obs_list.append(o); rew_list.append(r)
                done_list.append(d); info_list.append(info)
            return obs_list, rew_list, done_list, info_list
        for h, a in zip(self.handlers, actions):
            o, r, d, info = h.step(a)
            obs_list.append(o); rew_list.append(r)
            done_list.append(d); info_list.append(info)
        return obs_list, rew_list, done_list, info_list

    def per_world_reset_if_done(self) -> None:
        """Per-handler async reset (sync-free; sum across handlers).

        Per-handler ``per_world_reset_if_done`` is now sync-free, so this
        returns ``None`` instead of a count — read counts off the
        ``handler.done_mask_torch`` view (or a per-iter GPU histogram)
        when needed, **after** the rollout loop finishes.
        """
        if not self.shared_scene:
            for h in self.handlers:
                h.per_world_reset_if_done()
            return
        if all(h.eval_mode for h in self.handlers):
            return
        # Shared scene: an episode ends for the WHOLE world when ANY hand is done
        # → OR the masks into every handler, then snapshot / reset_data / restore
        # once and re-init each hand on the freshly reset worlds.
        hs = self.handlers
        common = hs[0].done_mask_torch.clone()
        for h in hs[1:]:
            torch.maximum(common, h.done_mask_torch, out=common)
        for h in hs:
            h.done_mask_torch.copy_(common)
        h0 = hs[0]
        h0._pwr_flag_and_snapshot()
        for h in hs[1:]:                             # propagate non-finite flags raised by h0
            h.done_mask_torch.copy_(h0.done_mask_torch)
        mjwarp.reset_data(h0.m, h0.d)
        for h in hs:
            h._pwr_reinit()
        h0._pwr_restore()
        wp.capture_launch(h0.capture_forward.graph)
        for h in hs:
            h._pwr_finish()

    # ── Convenience proxies for the single-handler case ──────────────────

    @property
    def is_single_handler(self) -> bool:
        return len(self.handlers) == 1

    @property
    def obs_dims(self) -> List[int]:
        return [h.obs_dim for h in self.handlers]

    @property
    def action_dims(self) -> List[int]:
        return [h.action_dim for h in self.handlers]

    def __len__(self) -> int:
        return len(self.handlers)

    def __getitem__(self, idx: int) -> SubEnvHandler:
        return self.handlers[idx]

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _derive_hand_util_list(orchestrator) -> list:
        """Pull ``{hand_key}_util`` for each entry of ``hand_type_name_list``.

        ``Env_Orchestrator.build_sub_env`` builds one ``SingleHandSubEnv`` per
        ``hand_type_name_list`` entry, in order, so this matches
        ``orchestrator.env_list``.
        """
        try:
            hand_keys = orchestrator.hand_type_name_list
        except AttributeError as e:
            raise AttributeError(
                "orchestrator has no `hand_type_name_list`. Pass "
                "`hand_util_list=[...]` explicitly to BaseRLEnv."
            ) from e
        return [getattr(orchestrator, f"{k}_util") for k in hand_keys]

    def _coerce_action_list(self, actions):
        if isinstance(actions, (list, tuple)):
            if len(actions) != len(self.handlers):
                raise ValueError(
                    f"expected {len(self.handlers)} action tensors, "
                    f"got {len(actions)}"
                )
            return list(actions)
        # Single tensor: only valid when there is a single handler.
        if not self.is_single_handler:
            raise TypeError(
                f"env has {len(self.handlers)} handlers — pass a list of "
                f"actions, not a single tensor."
            )
        return [actions]

# ──────────────────────────────────────────────────────────────────────────
# Concrete task implementations live in ``grit/training/rl_envs/<task>.py``
# and self-register via the ``@register_rl_env`` decorator. The
# ``grit.training`` package __init__ eagerly imports them, so the registry
# is populated by the time any caller hits ``make_rl_env(...)``.
# ──────────────────────────────────────────────────────────────────────────
