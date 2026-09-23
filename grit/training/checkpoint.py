"""Checkpoint save / load / discovery helpers for the training pipeline.

Centralises the on-disk format used by ``scripts/train.py`` (and the
notebook variant ``notebook/hand/04_rl_training/09_basic_tracking_RL_Training``)
so callers do not duplicate the serialization layout.

The on-disk blob is a single ``torch.save`` dict with these keys::

    model_state_dict   policy ``state_dict()``
    optim_state_dict   optimizer ``state_dict()``  (optional but recommended)
    seen_steps         int      cumulative *new* env-steps trained on so far
    hand_name          str
    task_name          str
    env_name           str      registered RL env name (e.g. "grasping")
    obs_dim            int
    action_dim         int

Filename convention (built by :func:`make_checkpoint_name`)::

    {hand_name}_{task_name}_{env_name}_{seen_steps}.pt

So ``find_latest_checkpoint`` can sort by ``int(stem.rsplit('_', 1)[-1])``
to pick up where the last training run left off.

Typical resume-friendly training loop::

    from grit.training.checkpoint import (
        save_checkpoint, load_checkpoint, find_latest_checkpoint,
    )

    seen_steps = 0
    latest = find_latest_checkpoint(SAVE_DIR, hand_name, TASK_NAME, ENV_NAME)
    if latest is not None and RESUME:
        blob = load_checkpoint(
            latest, policy=policy, optimizer=optimizer,
            expected_obs_dim=h.obs_dim,
            expected_action_dim=h.action_dim,
            expected_hand_name=hand_name,
            map_location=env.torch_device,
        )
        seen_steps = blob['seen_steps']

    while seen_steps < TOTAL_NEW_STEPS:
        ...
        seen_steps += STEPS_PER_ITER
        if (it + 1) % EVAL_EVERY == 0:
            save_checkpoint(
                save_dir=SAVE_DIR, policy=policy, optimizer=optimizer,
                seen_steps=seen_steps,
                hand_name=hand_name, task_name=TASK_NAME, env_name=ENV_NAME,
                obs_dim=h.obs_dim, action_dim=h.action_dim,
            )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


__all__ = (
    "make_checkpoint_name",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    "list_checkpoints",
    "parse_checkpoint_name",
    # Shared cross-embodiment policy (single file with per-tag dims).
    "save_shared_checkpoint",
    "load_shared_checkpoint",
    "is_shared_checkpoint",
    # High-level orchestration (SAVE_DIR derivation + resume entry point).
    "resolve_save_dir",
    "maybe_resume",
    "load_weights_from",
)


# ──────────────────────────────────────────────────────────────────────────
# Naming convention
# ──────────────────────────────────────────────────────────────────────────


def make_checkpoint_name(
    hand_name: str,
    task_name: str,
    env_name:  str,
    seen_steps: int,
) -> str:
    """Return the canonical on-disk filename for a checkpoint
    (``{hand}_{env}_{seen_steps}.pt``; ``task_name`` is kept for API compatibility)."""
    return f"{hand_name}_{env_name}_{int(seen_steps)}.pt"


def parse_checkpoint_name(path) -> Optional[Dict[str, Any]]:
    """Best-effort parse of the canonical filename
    ``{hand_name}_{env_name}_{seen_steps}.pt``.

    Returns ``None`` when the trailing token does not parse as an int — the
    seen-steps suffix is the only invariant we rely on, so callers can use
    this to filter foreign files in a directory without stricter parsing.
    """
    p = Path(path)
    stem = p.stem
    head, _, tail = stem.rpartition("_")
    if not tail:
        return None
    try:
        seen_steps = int(tail)
    except ValueError:
        return None
    return {"path": p, "stem": stem, "seen_steps": seen_steps}


# ──────────────────────────────────────────────────────────────────────────
# Save / load
# ──────────────────────────────────────────────────────────────────────────


def save_checkpoint(
    save_dir,
    *,
    policy:     torch.nn.Module,
    optimizer:  Optional[torch.optim.Optimizer],
    seen_steps: int,
    hand_name:  str,
    task_name:  str,
    env_name:   str,
    obs_dim:    int,
    action_dim: int,
    extra:      Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a training checkpoint to ``save_dir``.

    The directory is created on demand. The file is named via
    :func:`make_checkpoint_name` so :func:`find_latest_checkpoint` can
    later locate the most recent one.

    Args:
        save_dir:    target directory (str or Path); created if missing.
        policy:      torch ``nn.Module`` whose ``state_dict()`` is saved.
        optimizer:   optional torch ``Optimizer`` whose ``state_dict()`` is
                     saved alongside the policy. Pass ``None`` if you do
                     not plan to resume training.
        seen_steps:  cumulative new env-steps seen so far. This is the
                     resume cursor used by ``load_checkpoint``.
        hand_name / task_name / env_name:  identity tags written into the
                     blob *and* the filename.
        obs_dim / action_dim:  policy I/O dims (asserted on load).
        extra:       optional extra payload merged into the blob (must
                     not collide with the canonical keys).

    Returns:
        Path to the saved file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / make_checkpoint_name(hand_name, task_name, env_name, seen_steps)

    blob: Dict[str, Any] = {
        "model_state_dict": policy.state_dict(),
        "optim_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "seen_steps":       int(seen_steps),
        "hand_name":        str(hand_name),
        "task_name":        str(task_name),
        "env_name":         str(env_name),
        "obs_dim":          int(obs_dim),
        "action_dim":       int(action_dim),
    }
    if extra:
        overlap = set(extra) & set(blob)
        if overlap:
            raise ValueError(
                f"checkpoint extra keys collide with canonical keys: {sorted(overlap)}"
            )
        blob.update(extra)

    torch.save(blob, path)
    return path


def load_checkpoint(
    path,
    *,
    policy:              torch.nn.Module,
    optimizer:           Optional[torch.optim.Optimizer]    = None,
    expected_obs_dim:    Optional[int]                      = None,
    expected_action_dim: Optional[int]                      = None,
    expected_hand_name:  Optional[str]                      = None,
    expected_task_name:  Optional[str]                      = None,
    expected_env_name:   Optional[str]                      = None,
    map_location:        Any                                 = None,
    strict_dim:          bool                                = True,
) -> Dict[str, Any]:
    """Load a checkpoint into ``policy`` (+ ``optimizer`` if given) and
    verify dim / identity compatibility before applying weights.

    The returned dict is the full on-disk blob — most importantly
    ``blob['seen_steps']`` so the training loop can resume the
    ``seen_steps`` counter at exactly the value the saving run left off.

    Args:
        path:                file path to load.
        policy / optimizer:  loaded in-place (optimizer skipped when its
                             state dict in the blob is ``None`` or absent).
        expected_obs_dim:    when ``strict_dim=True``, raises
                             ``AssertionError`` if the blob's ``obs_dim``
                             does not match.
        expected_action_dim: same, for ``action_dim``.
        expected_hand_name / expected_task_name / expected_env_name:
                             optional identity guards (also assert on
                             mismatch when ``strict_dim=True``). Useful
                             when loading into a freshly-built env to
                             catch hand/task/env drift early.
        map_location:        passed straight to ``torch.load``.
        strict_dim:          when False every assertion above is downgraded
                             to a no-op (handy for partial / debug loads).

    Returns:
        The full blob (Dict[str, Any]).

    Raises:
        FileNotFoundError:   if ``path`` does not exist.
        KeyError:            if the blob is missing a required key.
        AssertionError:      on any ``strict_dim`` compat check failure.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(blob, dict):
        raise TypeError(
            f"unexpected checkpoint payload type {type(blob).__name__} at {path}"
        )

    for k in ("model_state_dict", "seen_steps", "obs_dim", "action_dim"):
        if k not in blob:
            raise KeyError(f"checkpoint at '{path}' missing required key '{k}'")

    # Identity / dim asserts (the headline guard: dim
    # mismatch must abort load before we apply incompatible weights).
    if strict_dim:
        ckpt_obs    = int(blob["obs_dim"])
        ckpt_action = int(blob["action_dim"])
        if expected_obs_dim is not None:
            assert ckpt_obs == int(expected_obs_dim), (
                f"obs_dim mismatch: ckpt={ckpt_obs} vs env={expected_obs_dim} "
                f"({path.name})"
            )
        if expected_action_dim is not None:
            assert ckpt_action == int(expected_action_dim), (
                f"action_dim mismatch: ckpt={ckpt_action} vs env={expected_action_dim} "
                f"({path.name})"
            )
        if expected_hand_name is not None and "hand_name" in blob:
            assert blob["hand_name"] == expected_hand_name, (
                f"hand_name mismatch: ckpt={blob['hand_name']!r} vs "
                f"env={expected_hand_name!r} ({path.name})"
            )
        if expected_task_name is not None and "task_name" in blob:
            assert blob["task_name"] == expected_task_name, (
                f"task_name mismatch: ckpt={blob['task_name']!r} vs "
                f"env={expected_task_name!r} ({path.name})"
            )
        if expected_env_name is not None and "env_name" in blob:
            assert blob["env_name"] == expected_env_name, (
                f"env_name mismatch: ckpt={blob['env_name']!r} vs "
                f"env={expected_env_name!r} ({path.name})"
            )

    # Apply weights only after every guard passed.
    # Allow promotion to an asymmetric critic: when a ckpt trained with a symmetric
    # critic (critic_trunk input = obs_dim) warm-starts an ``async_ppo`` run (input =
    # critic_obs_dim > obs_dim), only the first critic_trunk layer has a shape mismatch.
    # The actor is inherited as is and **only the critic parameters with a different
    # shape are skipped** — nothing is lost since the critic is retrained during
    # ``critic_warmup_iters`` anyway.
    _sd  = blob["model_state_dict"]
    _cur = policy.state_dict()
    _skip = [k for k, v in _sd.items()
             if k in _cur and tuple(_cur[k].shape) != tuple(v.shape)]
    if _skip:
        _sd = {k: v for k, v in _sd.items() if k not in _skip}
        print(f"  [ckpt] skipped {len(_skip)} parameter(s) with a shape mismatch "
              f"(critic retrained): {_skip[:4]}{' …' if len(_skip) > 4 else ''}")
        policy.load_state_dict(_sd, strict=False)
    else:
        policy.load_state_dict(_sd)
    if optimizer is not None:
        opt_sd = blob.get("optim_state_dict")
        if opt_sd is not None:
            optimizer.load_state_dict(opt_sd)

    return blob


# ──────────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────────


def list_checkpoints(
    save_dir,
    hand_name: str,
    task_name: str,
    env_name:  str,
) -> list[Path]:
    """All checkpoints in ``save_dir`` matching the given identity, sorted
    by ``seen_steps`` ascending."""
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return []
    pattern = f"{hand_name}_{env_name}_*.pt"

    out: list[tuple[int, Path]] = []
    for p in save_dir.glob(pattern):
        info = parse_checkpoint_name(p)
        if info is None:
            continue
        out.append((info["seen_steps"], p))
    out.sort(key=lambda t: t[0])
    return [p for _, p in out]


def find_latest_checkpoint(
    save_dir,
    hand_name: str,
    task_name: str,
    env_name:  str,
) -> Optional[Path]:
    """Latest checkpoint (highest ``seen_steps``) for the given identity,
    or ``None`` when no matching files exist."""
    ckpts = list_checkpoints(save_dir, hand_name, task_name, env_name)
    return ckpts[-1] if ckpts else None


def find_checkpoint_by_steps(
    save_dir,
    hand_name:  str,
    task_name:  str,
    env_name:   str,
    seen_steps: int,
) -> Path:
    """The checkpoint whose ``seen_steps`` equals ``seen_steps`` exactly.

    Unlike :func:`find_latest_checkpoint` this raises rather than returning
    ``None`` — an eval run asking for a specific checkpoint must never
    silently fall back to a different one. The error lists the nearest
    available step counts so a typo is easy to fix.
    """
    ckpts = list_checkpoints(save_dir, hand_name, task_name, env_name)
    if not ckpts:
        raise FileNotFoundError(
            f"no checkpoints matching "
            f"{hand_name}_{env_name}_*.pt under {save_dir}"
        )
    want  = int(seen_steps)
    steps = [parse_checkpoint_name(p)["seen_steps"] for p in ckpts]  # type: ignore[index]
    for s, p in zip(steps, ckpts):
        if s == want:
            return p
    # Nearest few (by |Δsteps|) to make the message actionable.
    near = sorted(steps, key=lambda s: abs(s - want))[:5]
    raise FileNotFoundError(
        f"no checkpoint with seen_steps={want:,} under {save_dir} "
        f"({len(ckpts)} found, range {steps[0]:,}..{steps[-1]:,}). "
        f"Nearest available: {', '.join(f'{s:,}' for s in sorted(near))}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Shared cross-embodiment policy — one file, per-tag dim metadata
# ──────────────────────────────────────────────────────────────────────────
#
# A shared policy (see :class:`grit.model.shared_cross_embodiment.
# SharedCrossEmbodimentMLPPolicy`) holds **one** ``nn.Module`` whose
# ``state_dict`` covers every tag's adapter + head plus the shared trunk.
# Saving one file per tag (the naive train.py path) duplicates the same
# tensors N times on disk for an N-hand run. These helpers save / load
# the shared module **once** per checkpoint and embed the per-tag
# ``obs_dim`` / ``action_dim`` dicts in the blob so the loader can
# verify each tag independently.
#
# On-disk extras (in addition to the usual single-hand keys)::
#
#     is_shared      : True                 (sentinel — checked by is_shared_checkpoint)
#     tags           : list[str]            (sorted tag list)
#     obs_dims       : dict[str, int]       per-tag obs dimensions
#     action_dims    : dict[str, int]       per-tag action dimensions
#
# ``obs_dim`` / ``action_dim`` (the legacy scalar fields) are set to ``-1``
# as sentinels so that any caller of the single-hand :func:`load_checkpoint`
# would trip the dim assert immediately rather than silently mismatch.
# ``hand_name`` carries the joined-tag string (e.g. ``"tesollo+robotis_sh5"``)
# so ``find_latest_checkpoint`` / ``list_checkpoints`` work with that as
# the identity argument.


SHARED_CKPT_OBS_SENTINEL    = -1
SHARED_CKPT_ACTION_SENTINEL = -1


def save_shared_checkpoint(
    save_dir,
    *,
    policy:        torch.nn.Module,        # the SharedCrossEmbodimentMLPPolicy itself, NOT a TagView
    optimizer:     Optional[torch.optim.Optimizer],
    seen_steps:    int,
    hands_joined:  str,                    # e.g. "tesollo+robotis_sh5"
    task_name:     str,
    env_name:      str,
    obs_dims:      Dict[str, int],         # {tag: obs_dim}
    action_dims:   Dict[str, int],         # {tag: action_dim}
    extra:         Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a single .pt for a shared cross-embodiment policy.

    Filename: ``{hands_joined}_{task_name}_{env_name}_{seen_steps}.pt``.
    The single file contains the FULL ``policy.state_dict()`` once (not
    duplicated per tag) + per-tag dim metadata for load-time verification.

    Use :func:`load_shared_checkpoint` to load. Use
    :func:`find_latest_checkpoint(save_dir, hands_joined, ...)` for
    discovery — the existing scanner works since the filename convention
    is identical.
    """
    if set(obs_dims) != set(action_dims):
        raise ValueError(
            f"obs_dims tags {sorted(obs_dims)} must match action_dims tags "
            f"{sorted(action_dims)}"
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / make_checkpoint_name(hands_joined, task_name, env_name, seen_steps)

    blob: Dict[str, Any] = {
        "model_state_dict": policy.state_dict(),
        "optim_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "seen_steps":       int(seen_steps),
        "hand_name":        str(hands_joined),
        "task_name":        str(task_name),
        "env_name":         str(env_name),
        "obs_dim":          SHARED_CKPT_OBS_SENTINEL,
        "action_dim":       SHARED_CKPT_ACTION_SENTINEL,
        # Shared-only extensions
        "is_shared":   True,
        "tags":        sorted(obs_dims.keys()),
        "obs_dims":    {str(k): int(v) for k, v in obs_dims.items()},
        "action_dims": {str(k): int(v) for k, v in action_dims.items()},
    }
    if extra:
        overlap = set(extra) & set(blob)
        if overlap:
            raise ValueError(
                f"shared checkpoint extra keys collide with canonical keys: "
                f"{sorted(overlap)}"
            )
        blob.update(extra)

    torch.save(blob, path)
    return path


def is_shared_checkpoint(blob_or_path) -> bool:
    """Return True if the ckpt blob (or .pt file) was written by
    :func:`save_shared_checkpoint`. Cheap probe — only reads the
    ``is_shared`` key; callers usually want this before deciding which
    loader to call."""
    if isinstance(blob_or_path, dict):
        return bool(blob_or_path.get("is_shared", False))
    p = Path(blob_or_path)
    if not p.exists():
        return False
    blob = torch.load(p, map_location="cpu", weights_only=False)
    return bool(isinstance(blob, dict) and blob.get("is_shared", False))


def load_shared_checkpoint(
    path,
    *,
    policy:                torch.nn.Module,        # the SharedCrossEmbodimentMLPPolicy
    optimizer:             Optional[torch.optim.Optimizer]   = None,
    expected_obs_dims:     Optional[Dict[str, int]]          = None,
    expected_action_dims:  Optional[Dict[str, int]]          = None,
    expected_hands_joined: Optional[str]                     = None,
    expected_task_name:    Optional[str]                     = None,
    expected_env_name:     Optional[str]                     = None,
    map_location:          Any                               = None,
    strict_dim:            bool                              = True,
    strict_state:          bool                              = True,
) -> Dict[str, Any]:
    """Load a shared cross-embodiment checkpoint. Mirrors
    :func:`load_checkpoint` but verifies dims **per tag**.

    Args:
        expected_obs_dims:     when ``strict_dim=True``, every key listed
                               here must match the ckpt's per-tag value.
                               Unknown ckpt tags (in ``ckpt.tags`` but not
                               in ``expected_obs_dims``) are tolerated —
                               that's the "load a subset of tags" use
                               case. Missing keys (in ``expected_obs_dims``
                               but not in ``ckpt.tags``) → assert.
        expected_action_dims:  same, for action dims.
        expected_hands_joined: optional identity guard. When given,
                               asserts an exact string match (so a
                               ``"tesollo"``-trained ckpt won't silently
                               load into a ``"tesollo+robotis_sh5"`` env
                               and vice versa).
        strict_dim:            False → all asserts above become no-ops.
        strict_state:          False → pass ``strict=False`` to
                               ``policy.load_state_dict`` so a ckpt with
                               adapters/heads for tags missing in the
                               current policy still loads (only the
                               matching keys are applied).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(blob, dict):
        raise TypeError(
            f"unexpected checkpoint payload type {type(blob).__name__} at {path}"
        )

    for k in ("model_state_dict", "seen_steps", "obs_dims", "action_dims", "tags"):
        if k not in blob:
            raise KeyError(
                f"shared checkpoint at '{path}' missing required key '{k}'. "
                f"Was this file written by save_shared_checkpoint?"
            )
    if not blob.get("is_shared", False):
        # Allow but warn — the data still has the shared structure; user
        # may have written it with a custom path.
        pass

    ckpt_obs_dims:    Dict[str, int] = dict(blob["obs_dims"])
    ckpt_action_dims: Dict[str, int] = dict(blob["action_dims"])

    if strict_dim:
        if expected_hands_joined is not None and "hand_name" in blob:
            assert blob["hand_name"] == expected_hands_joined, (
                f"hands_joined mismatch: ckpt={blob['hand_name']!r} vs "
                f"env={expected_hands_joined!r} ({path.name})"
            )
        if expected_task_name is not None and "task_name" in blob:
            assert blob["task_name"] == expected_task_name, (
                f"task_name mismatch: ckpt={blob['task_name']!r} vs "
                f"env={expected_task_name!r} ({path.name})"
            )
        if expected_env_name is not None and "env_name" in blob:
            assert blob["env_name"] == expected_env_name, (
                f"env_name mismatch: ckpt={blob['env_name']!r} vs "
                f"env={expected_env_name!r} ({path.name})"
            )

        if expected_obs_dims is not None:
            for tag, want in expected_obs_dims.items():
                if tag not in ckpt_obs_dims:
                    raise AssertionError(
                        f"shared ckpt is missing tag {tag!r} (ckpt tags: "
                        f"{sorted(ckpt_obs_dims)}) — {path.name}"
                    )
                assert int(ckpt_obs_dims[tag]) == int(want), (
                    f"obs_dim mismatch for tag {tag!r}: ckpt="
                    f"{ckpt_obs_dims[tag]} vs env={want} ({path.name})"
                )
        if expected_action_dims is not None:
            for tag, want in expected_action_dims.items():
                if tag not in ckpt_action_dims:
                    raise AssertionError(
                        f"shared ckpt is missing tag {tag!r} (ckpt tags: "
                        f"{sorted(ckpt_action_dims)}) — {path.name}"
                    )
                assert int(ckpt_action_dims[tag]) == int(want), (
                    f"action_dim mismatch for tag {tag!r}: ckpt="
                    f"{ckpt_action_dims[tag]} vs env={want} ({path.name})"
                )

    # Apply weights only after every guard passed.
    policy.load_state_dict(blob["model_state_dict"], strict=bool(strict_state))
    if optimizer is not None:
        opt_sd = blob.get("optim_state_dict")
        if opt_sd is not None:
            optimizer.load_state_dict(opt_sd)

    return blob


# ──────────────────────────────────────────────────────────────────────────
# High-level orchestration — SAVE_DIR layout + resume entry point
# ──────────────────────────────────────────────────────────────────────────

def resolve_save_dir(
    orchestrator,
    hands:        List[str],
    task_name:    str,
    env_name:     str,
    rl_env_cfg:   dict,
    resume_cfg:   dict,
    name_suffix:  str = "",
):
    """Derive ``(save_dir_base, load_dir, save_dir, resume)`` from cfg.

    * Single hand → ``<hand>_<env>/``.
    * Multi hand  → ``<hand1>+<hand2>+..._<env>/`` (sorted hand
      names joined by ``+`` so the folder name is stable regardless of
      yaml ordering).
    * ``use_deadzone=True`` in ``rl_env_cfg`` → appends ``_use_deadzone``
      to the base name so deadzone runs don't clobber non-deadzone ones.
    * ``name_suffix`` (caller-computed, e.g. ``"_nse30_nw4096_for_approach_test"``)
      → appended after the deadzone tag so otherwise-identical hand/task/env
      runs land in distinct folders. Inference must reconstruct the SAME
      tail (``resolve_inference_save_dir(..., name_suffix=...)``).
    * ``resume.enabled=True`` + ``resume.save_suffix`` (default
      ``"_resume"``) → forks a sibling dir so the original run's
      checkpoints stay intact while the resumed run writes new ones.
    """
    hands_joined = "+".join(sorted(hands))
    use_deadzone = bool(rl_env_cfg.get("use_deadzone", False))

    # ``orchestrator.home_dir`` is a string (set as ``str(PROJECT_ROOT)`` for
    # easy interop with ``os.path``/``str.format`` callers). Wrap in Path so
    # the ``/`` operator works for the rest of this function.
    save_dir_base = (
        Path(orchestrator.home_dir) / "output" / "checkpoints"
        / f'{hands_joined}_{env_name}'
    )
    if use_deadzone:
        save_dir_base = save_dir_base.parent / f"{save_dir_base.name}_use_deadzone"
    if name_suffix:
        save_dir_base = save_dir_base.parent / f"{save_dir_base.name}{name_suffix}"

    resume      = bool(resume_cfg.get("enabled", False))
    save_suffix = str(resume_cfg.get("save_suffix", "_resume"))

    load_dir = save_dir_base
    save_dir = (
        save_dir_base.parent / f'{save_dir_base.name}{save_suffix}'
        if resume and save_suffix
        else save_dir_base
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir_base, load_dir, save_dir, resume


def maybe_resume(
    *,
    resume:             bool,
    consider_seen_step: bool,
    load_dir:           Path,
    save_dir:           Path,
    tags:               List[str],
    env,
    policies:           Dict[str, torch.nn.Module],
    optimizers:         Dict[str, torch.optim.Optimizer],
    task_name:          str,
    env_name:           str,
    shared_module             = None,
    reward_normalizers: Optional[Dict[str, Any]] = None,
) -> int:
    """Load the latest matching checkpoint(s) and (optionally) return the
    seen_steps cursor to resume from.

    ``reward_normalizers`` (optional ``{tag: RewardNormalizer}``): when given,
    each normalizer's running statistics are restored from the blob
    (``reward_norm_state`` / shared ``reward_norm_states``). Restored whenever
    weights load — REGARDLESS of ``consider_seen_step`` — because the critic
    weights are fitted to the normalized reward scale, not to the step budget.
    Older checkpoints without the payload are a silent no-op (stats re-warm).

    Two paths:

      * **Independent per-tag** (``shared_module=None``): loop over tags;
        each ``policies[tag]`` loads its own ``<tag>_<task>_<env>_*.pt``.
        Returned cursor is the min across tags so the training budget
        still covers every tag fully.

      * **Shared cross-embodiment** (``shared_module`` provided): one
        ``<hands_joined>_<task>_<env>_*.pt`` file holds the full shared
        state. We load it once into ``shared_module`` (which every tag
        view points at) and into the single shared optimizer (any
        ``optimizers[tag]`` instance does — they're all the same).
    """
    if not resume:
        print("\nRESUME=False — training from scratch.")
        return 0

    # ── Shared cross-embodiment ───────────────────────────────────────
    if shared_module is not None:
        hands_joined = "+".join(sorted(tags))
        latest = find_latest_checkpoint(load_dir, hands_joined, task_name, env_name)
        if latest is None:
            print(f"\nno matching shared ckpt under {load_dir} (hands_joined="
                  f"{hands_joined!r}) — training from scratch.")
            return 0

        obs_dims    = {tag: env.handlers[i].obs_dim    for i, tag in enumerate(tags)}
        action_dims = {tag: env.handlers[i].action_dim for i, tag in enumerate(tags)}
        blob = load_shared_checkpoint(
            latest,
            policy                = shared_module,
            optimizer             = next(iter(optimizers.values())),
            expected_obs_dims     = obs_dims,
            expected_action_dims  = action_dims,
            expected_hands_joined = hands_joined,
            expected_task_name    = task_name,
            expected_env_name     = env_name,
            map_location          = env.torch_device,
        )
        print(f'\nresumed (shared) from {latest.parent.name}/{latest.name}')
        print(f'  hands_joined : {hands_joined}')
        print(f'  seen_steps   : {blob["seen_steps"]:,}')
        if save_dir != load_dir:
            print(f'  ↳ saving forked checkpoints under {save_dir.name}/')
        # Restore the curriculum state (success-gate progress etc.) — only when the step
        # cursor is continued. consider_seen_step=False means a "fresh budget", so the
        # gate also starts from 0.
        if consider_seen_step:
            _states = blob.get("curriculum_states") or {}
            for i, tag in enumerate(tags):
                h = env.handlers[i]
                if hasattr(h, "load_curriculum_state"):
                    h.load_curriculum_state(_states.get(tag))
        # Restore the reward-normalizer statistics — always when weights were loaded (the
        # critic is fitted on the normalised scale; see the docstring).
        if reward_normalizers:
            _rn_states = blob.get("reward_norm_states") or {}
            for tag in tags:
                if reward_normalizers.get(tag) is not None:
                    reward_normalizers[tag].load_state_dict(_rn_states.get(tag))
        return int(blob["seen_steps"]) if consider_seen_step else 0

    # ── Independent per-tag (legacy) ──────────────────────────────────
    seen_per_tag: List[int] = []
    any_loaded = False
    for i, tag in enumerate(tags):
        latest = find_latest_checkpoint(load_dir, tag, task_name, env_name)
        if latest is None:
            print(f"  [{tag}] no matching ckpt under {load_dir} — starts from scratch.")
            seen_per_tag.append(0)
            continue
        h = env.handlers[i]
        blob = load_checkpoint(
            latest,
            policy              = policies[tag],
            optimizer           = optimizers[tag],
            expected_obs_dim    = h.obs_dim,
            expected_action_dim = h.action_dim,
            expected_hand_name  = tag,
            expected_task_name  = task_name,
            expected_env_name   = env_name,
            map_location        = env.torch_device,
        )
        seen_per_tag.append(int(blob["seen_steps"]))
        any_loaded = True
        print(f'  [{tag}] resumed from {latest.name}  (seen_steps={blob["seen_steps"]:,})')
        # Restore the curriculum state (success-gate progress etc.) — only when the step
        # cursor is continued (consider_seen_step=False is a fresh budget → gate from 0).
        if consider_seen_step and hasattr(h, "load_curriculum_state"):
            h.load_curriculum_state(blob.get("curriculum_state"))
        # Restore the reward-normalizer statistics — always when weights were loaded (the
        # critic is fitted on the normalised scale; see the docstring).
        if reward_normalizers and reward_normalizers.get(tag) is not None:
            reward_normalizers[tag].load_state_dict(blob.get("reward_norm_state"))

    if not any_loaded:
        return 0
    if not consider_seen_step:
        return 0

    # Use the MIN so the training budget still covers every tag fully.
    cursor = min(seen_per_tag) if seen_per_tag else 0
    if save_dir != load_dir:
        print(f'  ↳ saving forked checkpoints under {save_dir.name}/')
    return cursor


# ──────────────────────────────────────────────────────────────────────────
# Arbitrary-path warm start (decoupled from the config-derived SAVE_DIR)
# ──────────────────────────────────────────────────────────────────────────

def _latest_pt_in_dir(d: Path) -> Optional[Path]:
    """Latest ``*.pt`` (highest trailing ``seen_steps``) in ``d``, ignoring the
    ``{hand}_{task}_{env}`` identity prefix. Used for arbitrary load dirs where
    the folder name need not match the current run's hand/task/env."""
    cands: list[tuple[int, Path]] = []
    for p in Path(d).glob("*.pt"):
        info = parse_checkpoint_name(p)
        if info is not None:
            cands.append((info["seen_steps"], p))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    return cands[-1][1]


def load_weights_from(
    path,
    *,
    tags:           List[str],
    env,
    policies:       Dict[str, torch.nn.Module],
    task_name:      str,
    env_name:       str,
    shared_module             = None,
    optimizers:     Optional[Dict[str, torch.optim.Optimizer]] = None,
    load_optimizer: bool       = False,
    strict_dim:     bool       = True,
    reward_normalizers: Optional[Dict[str, Any]] = None,
) -> None:
    """Weight-only warm start from an **arbitrary** checkpoint path — fully
    decoupled from the config-derived ``SAVE_DIR``. Unlike :func:`maybe_resume`
    (which only looks in ``load_dir = save_dir_base`` and continues
    ``seen_steps``), this loads weights from any directory or ``.pt`` file you
    point at and leaves the step counter at 0 (the caller keeps ``seen_steps=0``
    → a fresh ``total_new_steps`` budget). The optimizer is left freshly
    initialised unless ``load_optimizer=True``.

    ``path`` resolution:

      * **file** (``.../foo_123.pt``): used verbatim. For independent per-tag
        policies this is only valid with a single tag.
      * **directory**: per tag, the latest ``{tag}_{task}_{env}_*.pt`` is used;
        if none match (foreign folder name) and there is a single tag, the
        latest ``*.pt`` of any name is taken. Shared policies take the single
        shared ``.pt`` (or the latest in the dir).

    Dim compatibility (``obs_dim`` / ``action_dim``) is still asserted when
    ``strict_dim=True`` — that's a real shape guard. The hand/task/env *identity*
    asserts are intentionally relaxed (the whole point is loading from a
    differently-named source), so loading e.g. a checkpoint trained under a
    different ``run_tag`` / ``nworld`` folder just works as long as the network
    shapes match.
    """
    src = Path(path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"--load-from path does not exist: {src}")
    is_file = src.is_file()

    # ── Shared cross-embodiment ───────────────────────────────────────────
    if shared_module is not None:
        ckpt = src if is_file else _latest_pt_in_dir(src)
        if ckpt is None:
            raise FileNotFoundError(f"no *.pt checkpoint found under {src}")
        obs_dims    = {tag: env.handlers[i].obs_dim    for i, tag in enumerate(tags)}
        action_dims = {tag: env.handlers[i].action_dim for i, tag in enumerate(tags)}
        opt = next(iter(optimizers.values())) if (load_optimizer and optimizers) else None
        blob = load_shared_checkpoint(
            ckpt,
            policy                = shared_module,
            optimizer             = opt,
            expected_obs_dims     = obs_dims,
            expected_action_dims  = action_dims,
            expected_hands_joined = None,        # relax identity — arbitrary source
            expected_task_name    = None,
            expected_env_name     = None,
            map_location          = env.torch_device,
            strict_dim            = strict_dim,
            strict_state          = False,       # tolerate tag subset/superset
        )
        print(f'  warm-started (shared) from {ckpt}  '
              f'(ckpt seen_steps={blob.get("seen_steps", "?")}, '
              f'optimizer={"loaded" if opt is not None else "fresh"})')
        # Restore the reward-normalizer statistics (if present) — a warm start also takes
        # the critic weights, so the reward scale is carried along. Older ckpts have no
        # payload → no-op.
        if reward_normalizers:
            _rn_states = blob.get("reward_norm_states") or {}
            for tag in tags:
                if reward_normalizers.get(tag) is not None:
                    reward_normalizers[tag].load_state_dict(_rn_states.get(tag))
        return

    # ── Independent per-tag ───────────────────────────────────────────────
    for i, tag in enumerate(tags):
        h = env.handlers[i]
        if is_file:
            if len(tags) != 1:
                raise ValueError(
                    f"--load-from is a single .pt file but there are {len(tags)} "
                    f"tags ({tags}); pass a directory so each tag can be matched."
                )
            ckpt: Optional[Path] = src
        else:
            ckpt = find_latest_checkpoint(src, tag, task_name, env_name)
            if ckpt is None and len(tags) == 1:
                ckpt = _latest_pt_in_dir(src)     # foreign folder name fallback
            if ckpt is None:
                raise FileNotFoundError(
                    f"no checkpoint for tag {tag!r} under {src} "
                    f"(looked for {tag}_{task_name}_{env_name}_*.pt)"
                )
        opt = optimizers[tag] if (load_optimizer and optimizers) else None
        blob = load_checkpoint(
            ckpt,
            policy              = policies[tag],
            optimizer           = opt,
            expected_obs_dim    = h.obs_dim,
            expected_action_dim = h.action_dim,
            expected_hand_name  = None,           # relax identity — arbitrary source
            expected_task_name  = None,
            expected_env_name   = None,
            map_location        = env.torch_device,
            strict_dim          = strict_dim,
        )
        print(f'  [{tag}] warm-started from {ckpt}  '
              f'(ckpt seen_steps={blob.get("seen_steps", "?")}, '
              f'optimizer={"loaded" if opt is not None else "fresh"})')
        # Restore the reward-normalizer statistics (if present; older ckpts have no payload → no-op).
        if reward_normalizers and reward_normalizers.get(tag) is not None:
            reward_normalizers[tag].load_state_dict(blob.get("reward_norm_state"))
