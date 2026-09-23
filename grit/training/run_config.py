"""Per-run config snapshot for RL training/inference.

Saves the full hyperparameter set (orchestrator / rl_env builder kwargs /
handler attribute knobs / PPO training params / policy network spec) to
``<save_dir>/config.yaml`` so that:

  * **Training** can resume with the exact same configuration.
  * **Inference (nb73)** can rebuild the env / handler / policy from the
    same config without duplicating hardcoded values.

The schema is intentionally a plain nested ``dict`` (no pydantic / dataclass
dependency) so it survives codebase churn — the loader is **lenient**:
missing keys fall back to the current code defaults, unknown keys emit a
warning. RL-env-specific knob names are discovered automatically from
each handler class's ``CONFIG_KEYS`` / ``BUILDER_KEYS`` tuples — to add a
new task, just declare those tuples on the new handler subclass.

Schema (yaml top-level)::

    schema_version: 1
    hand_name: <str>
    task_name: <str>
    env_name:  <str>
    orchestrator: {config_name, with_table, n_obj, N_SUB_ENV, nworld, ...}
    rl_env:       {<BUILDER_KEYS of handler class>}
    handler:      {<CONFIG_KEYS  of handler class>}
    training:     {gamma, gae_lambda, lr, ... — opaque dict}
    policy:       {name, hidden_dim, min_std, var_scale, ...}
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import yaml


SCHEMA_VERSION = 1
CONFIG_FILENAME = "config.yaml"


# ────────────────────────────────────────────────────────────────────────
# Save / load
# ────────────────────────────────────────────────────────────────────────

def save_run_config(save_dir: Path | str, config: Mapping[str, Any]) -> Path:
    """Write ``config`` (a plain dict) to ``<save_dir>/config.yaml``.

    Always overwrites any prior config.yaml — the user is expected to fork
    ``save_dir`` (e.g. ``..._v2`` suffix) when resuming with new knobs, so
    the snapshot always reflects the *current* run.

    Accepts plain dicts OR an OmegaConf ``DictConfig``; the latter is
    resolved + converted to a plain dict so the on-disk YAML is portable.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / CONFIG_FILENAME

    payload = _to_plain_dict(config)
    payload.setdefault("schema_version", SCHEMA_VERSION)

    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return path


def save_run_snapshot(cfg, env, save_dir: Path | str) -> Path:
    """Snapshot the resolved cfg → ``<save_dir>/config.yaml`` with the
    **live** handler attrs (post-``apply_handler_knobs``) substituted in.

    The same yaml is applied to every handler in :func:`build_rl_env_and_handlers`,
    so ``env.handlers[0]`` is representative. ``resume`` is dropped from the
    snapshot since it describes how *this* run started, not how the saved
    one should be replayed.

    Used by ``scripts/train.py`` at training start so inference scripts
    (``scripts/evaluate.py``, ``nb11``) can rebuild the env / handler /
    policy from the same config without duplicating hardcoded values.
    """
    # Imported lazily to avoid pulling OmegaConf into the public symbol
    # set of this module — only callers that actually run training need it.
    from omegaconf import OmegaConf
    raw = OmegaConf.to_container(cfg, resolve=True)
    # ``to_container`` returns ``Dict | List | str | None`` in general; for
    # a training-cfg DictConfig it's always a dict. Assert + cast to narrow.
    assert isinstance(raw, dict), (
        f"save_run_snapshot: expected cfg to convert to a dict, got "
        f"{type(raw).__name__}. Pass an OmegaConf DictConfig."
    )
    full_cfg: Dict[str, Any] = raw                           # type: ignore[assignment]
    full_cfg["handler"] = extract_handler_knobs(env.handlers[0])
    full_cfg.pop("resume", None)
    return save_run_config(save_dir, full_cfg)


def load_run_config_as_omegaconf(save_dir: Path | str):
    """Load ``<save_dir>/config.yaml`` and return it as an OmegaConf
    ``DictConfig`` — convenient for feeding straight back into
    ``Env_Orchestrator(overall_cfg=...)`` at inference time so nb73 reads
    the exact same env definition that produced the checkpoint.

    Returns ``None`` if no config.yaml exists (legacy checkpoint).
    """
    from omegaconf import OmegaConf
    raw = load_run_config(save_dir)
    if raw is None:
        return None
    return OmegaConf.create(raw)


def _to_plain_dict(cfg: Any) -> Dict[str, Any]:
    """Convert OmegaConf ``DictConfig`` (or anything dict-like) to a plain
    ``dict`` so PyYAML can ``safe_dump`` it without omegaconf-specific tags.
    """
    try:
        from omegaconf import DictConfig, OmegaConf
        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except ImportError:
        pass
    return dict(cfg)


def load_run_config(save_dir: Path | str) -> Optional[Dict[str, Any]]:
    """Return the parsed config dict, or ``None`` if no ``config.yaml``
    exists under ``save_dir``. Caller decides whether the absence is fatal
    (inference) or OK (legacy checkpoint).
    """
    save_dir = Path(save_dir)
    path = save_dir / CONFIG_FILENAME
    if not path.is_file():
        return None
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(
            f"{path} did not parse to a dict (got {type(cfg).__name__})"
        )
    return cfg


# ────────────────────────────────────────────────────────────────────────
# Handler-class introspection (CONFIG_KEYS / BUILDER_KEYS)
# ────────────────────────────────────────────────────────────────────────

def _handler_keys(handler_or_cls, attr_name: str) -> Tuple[str, ...]:
    """Return ``cls.<attr_name>`` if declared, else ``()``."""
    cls = handler_or_cls if isinstance(handler_or_cls, type) else type(handler_or_cls)
    return tuple(getattr(cls, attr_name, ()) or ())


def _handler_groups(handler_or_cls, attr_name: str) -> Dict[str, Tuple[str, ...]]:
    """Return ``cls.<attr_name>`` (a ``{group: keys}`` dict) if declared,
    else ``{}``. Used to access the grouped representation of
    ``BUILDER_KEYS_GROUPS`` / ``CONFIG_KEYS_GROUPS``."""
    cls = handler_or_cls if isinstance(handler_or_cls, type) else type(handler_or_cls)
    groups = getattr(cls, attr_name, None) or {}
    return {g: tuple(keys) for g, keys in groups.items()}


def _flatten_grouped_knobs(d: Optional[Mapping]) -> Dict[str, Any]:
    """Normalise a knob dict that may be either flat or grouped.

      * flat:    ``{key: value, ...}``           → returned as-is (copy)
      * grouped: ``{group: {key: value, ...}}``  → keys merged across groups

    Detection rule: if EVERY top-level value is a ``Mapping``, treat as
    grouped (RL knobs are scalars / lists / bools / strings — never dicts).
    Duplicate keys across groups raise ``ValueError`` (matches the
    handler-class side of ``_flatten_groups``).

    Supports the user writing the YAML ``handler:`` (or ``rl_env:``)
    section either way without changing any caller code.
    """
    if not d:
        return {}
    is_grouped = all(isinstance(v, Mapping) for v in d.values())
    if not is_grouped:
        return dict(d)

    flat: Dict[str, Any] = {}
    for group_name, group in d.items():
        for k, v in group.items():
            if k in flat:
                raise ValueError(
                    f"_flatten_grouped_knobs: duplicate key {k!r} across "
                    f"groups (already present, second occurrence in "
                    f"group {group_name!r})"
                )
            flat[k] = v
    return flat


def extract_handler_knobs(handler, *, grouped: bool = False) -> Dict[str, Any]:
    """Snapshot the **current** values of all keys declared as tunable
    on ``handler``'s class.

    ``grouped=False`` (default) returns a flat ``{key: value}`` dict
    matching the legacy YAML schema (``handler: {W_POS: 4.0, ...}``).

    ``grouped=True`` returns a nested ``{group: {key: value}}`` dict
    matching the class-level ``CONFIG_KEYS_GROUPS`` organisation —
    useful when you want the saved YAML to be self-documenting by
    grouping reward / termination / success knobs visually.

    Anything not declared in ``CONFIG_KEYS`` / ``CONFIG_KEYS_GROUPS`` is
    silently skipped, so the YAML stays in sync with the explicit
    "tunable" surface area.
    """
    if grouped:
        out: Dict[str, Dict[str, Any]] = {}
        for group, keys in _handler_groups(handler, "CONFIG_KEYS_GROUPS").items():
            out[group] = {
                k: _yaml_safe(getattr(handler, k))
                for k in keys if hasattr(handler, k)
            }
        return out
    keys = _handler_keys(handler, "CONFIG_KEYS")
    return {k: _yaml_safe(getattr(handler, k)) for k in keys if hasattr(handler, k)}

def apply_handler_knobs(
    handler,
    knob_dict: Optional[Mapping[str, Any]],
    *,
    lenient: bool = True,
    rebuild_cond: bool = True,
) -> Dict[str, list]:
    """Write ``knob_dict`` values onto ``handler`` (lenient by default).

    Behaviour:
      * **Missing keys** (in ``CONFIG_KEYS`` but not in ``knob_dict``):
        left at the handler's current value → log under ``"missing"``.
      * **Null values** (``key: null`` in YAML, i.e. ``None`` in dict):
        treated as "explicit leave at default" — handler attribute is
        NOT touched (useful for knobs auto-computed in ``__init__``,
        e.g. ``REWARD_DT = sim_nstep × sim_dt``). Logged under ``"missing"``.
      * **Unknown keys** (in ``knob_dict`` but not in ``CONFIG_KEYS``):
        skipped + ``UserWarning`` → log under ``"unknown"``.
      * Successfully applied keys → log under ``"applied"``.

    When ``rebuild_cond=True`` and the handler exposes ``_setup_condition``
    + ``cond`` (the per-world condition registry), the cond buffers are
    torn down and rebuilt — this is required for keys like
    ``TARGET_QUAT_MODE`` / ``TAXONOMY_PROB`` that influence the sampler.

    Returns a small report dict so the caller can decide whether to abort
    or just print it.
    """
    report = dict(applied=[], missing=[], unknown=[])
    # Accept both flat ``{key: value}`` and grouped ``{group: {key: value}}``
    # representations transparently — the on-disk YAML may use either.
    knob_dict = _flatten_grouped_knobs(knob_dict)

    valid_keys = set(_handler_keys(handler, "CONFIG_KEYS"))

    # Apply known keys. ``None`` (YAML ``null``) is treated as "explicit
    # leave at default" — useful for auto-computed knobs (e.g. REWARD_DT).
    for k, v in knob_dict.items():
        if k not in valid_keys:
            report["unknown"].append(k)
            continue
        if v is None:
            # falls through to the missing-keys accounting below
            continue
        setattr(handler, k, v)
        report["applied"].append(k)

    # Track which expected keys were absent (or explicitly null) in the YAML.
    applied_set = set(report["applied"])
    for k in valid_keys:
        if k not in applied_set:
            report["missing"].append(k)

    if not lenient and (report["unknown"] or report["missing"]):
        raise ValueError(
            f"apply_handler_knobs: strict mode failed — "
            f"unknown={report['unknown']}  missing={report['missing']}"
        )
    if report["unknown"]:
        warnings.warn(
            f"apply_handler_knobs: ignored unknown keys "
            f"{report['unknown']} (not in {type(handler).__name__}.CONFIG_KEYS)",
            stacklevel=2,
        )

    # Rebuild condition samplers (TARGET_QUAT_MODE / TAXONOMY_PROB / pos
    # bounds etc. only take effect after _setup_condition reruns).
    if rebuild_cond and hasattr(handler, "cond") and hasattr(handler, "_setup_condition"):
        for f in list(handler.cond._fields.keys()):
            handler.cond.unregister(f)
        handler._setup_condition()

    # ── rebuild the obs buffers (apply obs-layout knobs) ───────────────────
    # Knobs like ``INCLUDE_*_IN_OBS`` change ``obs_dim``, but the obs buffers were already
    # allocated by the handler ``__init__`` with the **class defaults**, so changing such a
    # knob via yaml leaves ``obs_dim`` (new value) and ``obs_wp`` (old size) inconsistent.
    # The policy is then built with ``obs_dim`` and dies on the first forward with
    # "mat1 and mat2 shapes cannot be multiplied (N×old and new×hidden)".
    # → re-allocate the buffers here so the yaml value is the source of truth (same reason
    #   and same place as the cond rebuild). This happens **before** the policy / rollout
    #   buffers are created, so no consumer holds ``obs_torch`` yet and it is safe.
    if hasattr(handler, "_setup_obs_buffers") and hasattr(handler, "obs_wp"):
        try:
            want = int(handler.obs_dim)
            have = int(handler.obs_wp.shape[1]) if handler.obs_wp.ndim == 2 else -1
        except Exception:
            want = have = -1
        # Comparing widths alone misses the case **same width, different layout**.
        # Example: class default = EMA on / prev off, yaml = EMA off / prev on → both are
        # 4 blocks × 5 = 128, so no rebuild runs and the block offsets keep the old layout
        # (width matches, column order is wrong).
        # Hence the knob combination (layout signature) is checked as well.
        layout_changed = False
        if hasattr(handler, "_obs_layout_now") and hasattr(handler, "_obs_layout_lock"):
            try:
                layout_changed = (handler._obs_layout_now() != handler._obs_layout_lock)
            except Exception:
                layout_changed = False
        if want > 0 and (want != have or layout_changed):
            handler._setup_obs_buffers()
            why = "width changed" if want != have else "same width, layout changed"
            print(f"[obs-rebuild] obs_dim {have} → {want} ({why} — "
                  f"buffers re-created to apply the obs-layout knobs)")

    # ── re-bind the applier hooks (same reason as the obs buffers) ─────────
    # Hooks like ``_bind_action_ema`` are decided once inside the handler ``__init__``
    # from the **class defaults**. Enabling β via yaml comes too late and it silently
    # stays off (worse than a crash, because nothing reacts).
    if hasattr(handler, "_bind_action_ema"):
        handler._bind_action_ema()

    return report


def extract_rl_env_kwargs(handler) -> Dict[str, Any]:
    """Snapshot the **builder kwargs** that produced this handler.

    Reads attributes named in ``handler.BUILDER_KEYS``. Useful when you
    want to round-trip a config (currently nb72 stores them explicitly
    via ``CFG['rl_env']``; this is the symmetric reader for completeness).
    """
    keys = _handler_keys(handler, "BUILDER_KEYS")
    out: Dict[str, Any] = {}
    for k in keys:
        # Builder kwargs may live on the handler OR on its applier (e.g.
        # ``xyz_scale`` is forwarded to ``WarpActionApplier``). Try both.
        if hasattr(handler, k):
            out[k] = _yaml_safe(getattr(handler, k))
        elif hasattr(handler, "applier") and hasattr(handler.applier, k):
            out[k] = _yaml_safe(getattr(handler.applier, k))
    return out


def rl_env_kwargs_from_config(
    cfg: Mapping[str, Any],
    handler_cls,
    *,
    lenient: bool = True,
) -> Dict[str, Any]:
    """Filter ``cfg['rl_env']`` to just the keys accepted by ``handler_cls``
    (i.e. ``handler_cls.BUILDER_KEYS``). Unknown keys are warned + dropped.

    Use this to feed ``make_rl_env(env_name, orchestrator=..., **kwargs)``
    safely from a YAML that may have been saved by a different code version.

    The ``rl_env`` section may be either flat or grouped (see
    :func:`_flatten_grouped_knobs`).
    """
    rl_env_cfg = _flatten_grouped_knobs(cfg.get("rl_env"))
    valid_keys = set(_handler_keys(handler_cls, "BUILDER_KEYS"))

    out: Dict[str, Any] = {}
    unknown: list = []
    for k, v in rl_env_cfg.items():
        if k in valid_keys:
            out[k] = v
        else:
            unknown.append(k)

    if unknown:
        if not lenient:
            raise ValueError(
                f"rl_env_kwargs_from_config: unknown keys {unknown} "
                f"for {handler_cls.__name__}.BUILDER_KEYS"
            )
        warnings.warn(
            f"rl_env_kwargs_from_config: dropped unknown keys "
            f"{unknown} (not in {handler_cls.__name__}.BUILDER_KEYS)",
            stacklevel=2,
        )
    return out


# ────────────────────────────────────────────────────────────────────────
# YAML coercion helpers
# ────────────────────────────────────────────────────────────────────────

def _yaml_safe(v: Any) -> Any:
    """Coerce common numpy / torch / tuple types to plain Python so PyYAML
    safe_dump never bails out on a value we'd silently rely on."""
    # Lazy imports — keep run_config.py importable even if numpy/torch
    # aren't present in some narrow env.
    try:
        import numpy as _np
        if isinstance(v, _np.generic):
            return v.item()
        if isinstance(v, _np.ndarray):
            return v.tolist()
    except ImportError:
        pass
    try:
        import torch as _t
        if isinstance(v, _t.Tensor):
            return v.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(v, tuple):
        return [_yaml_safe(x) for x in v]
    if isinstance(v, list):
        return [_yaml_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _yaml_safe(x) for k, x in v.items()}
    return v


# ────────────────────────────────────────────────────────────────────────
# Convenience: training/policy dict builders (no introspection — flat)
# ────────────────────────────────────────────────────────────────────────

def build_default_training_dict(**overrides) -> Dict[str, Any]:
    """Common PPO defaults; override per-run as needed. The dict is opaque
    to the rest of the system — callers consume their own keys."""
    d: Dict[str, Any] = dict(
        gamma                = 0.99,
        gae_lambda           = 0.95,
        unroll_length        = 10,
        batch_size           = 1024,
        num_minibatch        = 16,
        n_ppo_epochs         = 10,
        clip_eps             = 0.2,
        ent_coef             = 0.0,
        vf_coef              = 0.5,
        max_grad_norm        = 1.0,
        lr                   = 3.0e-4,
        # KL-based adaptive lr — None → OFF (fixed lr, legacy behaviour). 0.01 is typical.
        desired_kl           = None,
        lr_min               = 1.0e-6,
        lr_max               = 1.0e-3,
        kl_adapt_factor      = 1.5,
        total_new_steps      = int(1e8),
        n_evals              = 100,
        eval_length          = None,    # None → handler's MAX_EPISODE_STEPS
        print_every          = 10,
        consider_seen_step   = False,
        seed                 = 42,
    )
    d.update(overrides)
    return d


def build_default_policy_dict(**overrides) -> Dict[str, Any]:
    d: Dict[str, Any] = dict(
        name          = "normal_tanh_mlp",
        hidden_dim    = [256, 256, 128, 64],
        min_std       = 0.001,
        var_scale     = 0.5,
        use_layernorm = True,
    )
    d.update(overrides)
    return d
