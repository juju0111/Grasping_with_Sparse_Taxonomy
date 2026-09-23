"""Training-time builders — one-call helpers that assemble the major
components of a PPO training run from a resolved ``OmegaConf`` cfg.

Pulled out of ``scripts/train.py`` so the same builders can drive other
entrypoints (notebooks, sweeps, ablations) without copy-pasting.

Order of usage (single- or multi-hand, independent or shared cross-embodiment):

    orchestrator, cfg            = build_orchestrator(config_name, overrides=...)
    env, rl_env_cfg, handler_cfg, tags = build_rl_env_and_handlers(orchestrator, cfg, env_name)
    policies, optimizers, policy_cfg, shared_module = build_policies_and_optimizers(cfg, env, tags)
    buffer                       = build_rollout_buffer(cfg, env)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from omegaconf import OmegaConf

from grit.training.orchestrator.orchestrator import Env_Orchestrator
from grit.training.rl_env_base               import get_handler_cls, make_rl_env
from grit.training.run_config                import (
    apply_handler_knobs, rl_env_kwargs_from_config, load_run_config_as_omegaconf,
)
from grit.training.checkpoint                import (
    find_latest_checkpoint, find_checkpoint_by_steps, load_checkpoint,
    load_shared_checkpoint, is_shared_checkpoint,
)
from grit.util.ppo_rollout_buffer            import (
    MultiHandRolloutBuffer, derive_handler_tags,
)
from grit.model.base_networks                import make_policy


# ──────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────

def build_orchestrator(
    config_name: str,
    overrides: Optional[Sequence[str]] = None,
) -> Tuple[Env_Orchestrator, OmegaConf]:
    """Load ``config/training/<config_name>.yaml`` → ``(orchestrator, cfg)``.

    ``overrides`` is a list of Hydra-style ``key=value`` strings (or None).
    Forwarded to :class:`Env_Orchestrator` → ``compose(... overrides=...)``,
    so any cfg key can be tweaked from the CLI without editing yaml.
    """
    orch = Env_Orchestrator(
        config_name=config_name, test_mode=False, overrides=overrides,
    )
    return orch, orch.overall_cfg # type: ignore


# ──────────────────────────────────────────────────────────────────────────
# Object source (mesh dataset ↔ procedural primitives)
# ──────────────────────────────────────────────────────────────────────────

def obj_spec_provider_from_config(cfg, *, seed=None, verbose: bool = False):
    """``Training.object_source`` → an ``obj_spec_provider`` for ``build_sub_env``.

    * ``"dataset"`` (default, or key absent) → ``None``: the mesh dataset path
      is used exactly as before.
    * ``"primitive"`` → a procedural provider built from
      ``Training.primitive_object`` (:mod:`grit.util.primitive_object`), so the
      run never touches ``obj_xml_path_lst``. ``obj_idxs`` is meaningless then
      and must be ``null``.

    Every env-build entry point (train / inference / eval / distill) goes
    through this, so one yaml key switches the whole pipeline.
    """
    try:
        src = str(cfg.Training.get("object_source", "dataset") or "dataset").lower()
    except Exception:
        src = "dataset"
    if src in ("dataset", "mesh"):
        return None
    if src not in ("primitive", "mixed"):
        raise ValueError(
            f"Training.object_source={src!r} is not supported "
            f"(expected 'dataset', 'primitive' or 'mixed')")
    from grit.util import primitive_object as prim
    opts = prim.primitive_options_from_config(cfg)
    if src == "mixed":
        # ``obj_idxs`` IS meaningful here — it pools the mesh half.
        provider = prim.make_mixed_obj_spec_provider(cfg, seed=seed, verbose=verbose)
        if verbose:
            mf = float(dict(cfg.Training.get("primitive_object", {}) or {}).get("mesh_frac", 0.5))
            print(f"object source   : MIXED — mesh_frac={mf} (the rest is procedurally generated), "
                  f"layout={opts['layout']}, seed={seed if seed is not None else opts['seed']}")
        return provider
    if cfg.get("obj_idxs", None):
        raise ValueError(
            "Training.object_source='primitive' but obj_idxs is set — those are "
            "indices into the MESH dataset and are never used by the procedural "
            "path. Set `obj_idxs: null` (or use object_source='mixed', where "
            "obj_idxs pools the mesh half).")
    provider = prim.make_obj_spec_provider_from_config(cfg, seed=seed, verbose=verbose)
    if verbose:
        print(f"object source   : PRIMITIVE (procedural) — layout={opts['layout']}, "
              f"seed={seed if seed is not None else opts['seed']}")
    return provider


# ──────────────────────────────────────────────────────────────────────────
# RL env + handlers
# ──────────────────────────────────────────────────────────────────────────

def build_rl_env_and_handlers(orchestrator, cfg, env_name: str):
    """Build the RL env (which wraps *all* sub-envs as handlers) and apply
    ``cfg.handler`` knobs to **every** handler. The same handler section
    is applied to all hands — same yaml drives every handler's reward /
    termination / success / contact / target knobs.

    Returns ``(env, rl_env_cfg, handler_cfg, tags)`` where ``tags`` is the
    canonical per-handler tag list (used as keys throughout the script).
    """
    rl_env_cfg  = OmegaConf.to_container(cfg.rl_env,  resolve=True)
    handler_cfg = OmegaConf.to_container(cfg.handler, resolve=True)

    handler_cls = get_handler_cls(env_name)
    kw          = rl_env_kwargs_from_config({"rl_env": rl_env_cfg}, handler_cls, lenient=True)
    env         = make_rl_env(env_name, orchestrator=orchestrator, **kw)

    # ⚠ Apply knobs to EVERY handler — the previous single-handler script
    # only patched env.handlers[0], silently leaving the rest at class
    # defaults in multi-hand setups. Same handler_cfg drives all hands.
    first_report = None
    for h in env.handlers:
        report = apply_handler_knobs(h, handler_cfg, lenient=True, rebuild_cond=True) # type: ignore
        if first_report is None:
            first_report = report
    print(f'handler knobs applied  : {len(first_report["applied"])} per handler '  # type: ignore
          f'× {len(env.handlers)} handler(s)') 
    if first_report["missing"]: # type: ignore
        print(f'  (left at class default : {len(first_report["missing"])} key(s) — not in cfg.handler)') # type: ignore
    if first_report["unknown"]: # type: ignore
        print(f'  ⚠ unknown cfg.handler keys ignored : {first_report["unknown"]}') # type: ignore

    tags  = derive_handler_tags(env)
    multi = len(env.handlers) > 1
    print(f'handlers : {len(env.handlers)} '
          f'({"cross-embodiment" if multi else "single-hand"})')
    for i, (tag, h) in enumerate(zip(tags, env.handlers)):
        print(f'  [{i}] tag={tag:20s}  NWORLD={h.NWORLD:>5}  '
              f'obs={h.obs_dim:>3}  action={h.action_dim:>3}')
    print(f'sim_nstep    : {env.sim_nstep}')
    print(f'cond fields  : {env.handlers[0].cond.names()}')
    return env, rl_env_cfg, handler_cfg, tags


def build_eval_env(orchestrator, env_name: str, eval_obj_idxs: Sequence[int], *,
                   nworld: int, geom_collision_type: Optional[str] = None):
    """Build a dedicated **fixed-object** eval env pinned to ``eval_obj_idxs``.

    Reuses the TRAINING ``orchestrator``'s already-resolved config (just clones
    it and overrides ``n_sub_env`` / ``nworld``) instead of re-composing Hydra —
    so hand / task / dataset / collision type all match the training env for
    free, and the trained policies evaluate on it unchanged. Fully pinned
    (``n_sub_env == len(eval_obj_idxs)``) → never resampled. It gets its own
    orchestrator, so eval never perturbs the training rollout; keep ``nworld``
    modest to bound the extra GPU memory of a second resident env.
    """
    eval_obj_idxs = [int(i) for i in eval_obj_idxs]
    eval_cfg = orchestrator.overall_cfg.copy()            # independent DictConfig copy
    eval_cfg.n_sub_env = len(eval_obj_idxs)                # all listed objects → fully pinned
    eval_cfg.obj_idxs  = None                              # pool is passed explicitly below
    eval_cfg.nworld    = int(nworld)
    gct = geom_collision_type or getattr(orchestrator, "geom_collision_type", "mesh")

    # Procedural objects: ``eval_obj_idxs`` has no meaning (there is no dataset
    # to index), so it only fixes HOW MANY held-out objects to draw. A separate
    # seed keeps that eval set distinct from the training draw and constant for
    # the whole run (the training set is what gets resampled).
    eval_provider = obj_spec_provider_from_config(
        eval_cfg, seed=int(eval_cfg.get("training", {}).get("seed", 42)) + 10_000)
    eval_orch = Env_Orchestrator(
        overall_cfg=eval_cfg,
        obj_idxs=None if eval_provider is not None else eval_obj_idxs,
    )
    eval_orch.build_sub_env(with_mjwarp=True, for_inference=True, geom_collision_type=gct,
                            obj_spec_provider=eval_provider)
    eval_env, *_ = build_rl_env_and_handlers(eval_orch, eval_cfg, env_name)
    return eval_env


# ──────────────────────────────────────────────────────────────────────────
# Policies + optimizers (independent vs shared cross-embodiment)
# ──────────────────────────────────────────────────────────────────────────

def build_policies_and_optimizers(cfg, env, tags: List[str]):
    """Build ``{tag: policy}`` + ``{tag: optimizer}`` — one entry per handler.

    Two modes (selected by ``cfg.policy.name``):

      * **Independent** (default, e.g. ``normal_tanh_mlp``): one full
        policy network per tag, each with its own optimizer. Each tag's
        ``policies[tag]`` is a standalone :class:`nn.Module`.

      * **Shared cross-embodiment** (any policy name starting with
        ``"shared_"``, currently ``shared_cross_embodiment_mlp``): a
        SINGLE :class:`nn.Module` is built with per-tag input adapters
        + per-tag output heads + a shared trunk. Every ``policies[tag]``
        is a lightweight :class:`_TagView` of that shared module — all
        views reference the same parameters, and every ``optimizers[tag]``
        is the same :class:`torch.optim.Adam` over
        ``shared.parameters()``. Per-tag PPO updates accumulate
        gradients through tag-specific adapters/heads + the shared
        trunk, naturally producing cross-task transfer through the
        trunk.

    Returns ``(policies, optimizers, policy_cfg, shared_module)`` —
    ``shared_module`` is ``None`` in independent mode, or the underlying
    shared :class:`nn.Module` in shared mode (used for the snapshot +
    diagnostics).
    """
    policy_cfg  = OmegaConf.to_container(cfg.policy, resolve=True)
    kw_template = dict(policy_cfg)                           # type: ignore
    name        = kw_template.pop("name", "normal_tanh_mlp") # type: ignore
    # train.py-only knob (finger-dimension σ init) — not a policy ctor argument.
    kw_template.pop("init_std_finger", None)
    # Phase-dependent exploration σ: the yaml names the obs **term**, and the index is
    # resolved here via the handler's obs_term_layout() (the obs producer is the single
    # source of truth — a hand-written index silently reads the wrong channel once the
    # layout changes).
    _ph_name = kw_template.pop("phase_std_obs", None)
    _ph_comp = int(kw_template.pop("phase_std_obs_comp", 0) or 0)
    # Optional finger-only scale — splits action = [wrist 9 | finger n_ctrl] so only the
    # wrist explores widely. When given, the scalar scale is broadcast to a per-dim vector.
    _ph_hi_f = kw_template.pop("phase_std_scale_hi_finger", None)
    _ph_lo_f = kw_template.pop("phase_std_scale_lo_finger", None)
    # The scales/threshold are popped from kw_template as well — they are put into
    # `extra` per tag below (broadcast to per-dim vectors if needed), so leaving them
    # would pass the same key to make_policy twice.
    _ph_hi = float(kw_template.pop("phase_std_scale_hi", 1.0) or 1.0)
    _ph_lo = float(kw_template.pop("phase_std_scale_lo", 1.0) or 1.0)
    _ph_thr = kw_template.pop("phase_std_thresh", None)
    lr          = float(cfg.training.lr)

    def _resolve_phase_std_idx(h) -> Optional[int]:
        if not _ph_name:
            return None
        assert hasattr(h, "obs_term_layout"), (
            "policy.phase_std_obs requires the handler to provide "
            "obs_term_layout()")
        lay = {n: (a, b) for n, a, b in h.obs_term_layout()}
        assert _ph_name in lay, (
            f"policy.phase_std_obs={_ph_name!r} is not in the obs layout — "
            f"available terms: {sorted(lay)}")
        a, b = lay[_ph_name]
        assert 0 <= _ph_comp < (b - a), (
            f"phase_std_obs_comp={_ph_comp} is out of range for {_ph_name}(width {b-a})")
        return a + _ph_comp

    is_shared = name.startswith("shared_")
    # Async (asymmetric) actor-critic: when ``training.async_ppo`` is set, the
    # critic trunk is built over the env's (larger, privileged) ``critic_obs_dim``
    # instead of the actor ``obs_dim``. Default (off) → ``critic_obs_dim`` is NOT
    # passed → policies stay symmetric (byte-identical).
    async_ppo = bool(cfg.training.get("async_ppo", False))
    # shared_single_* is a single core with identical dims across tags, so it can be
    # combined with an asymmetric critic (critic_obs_dim must also match across tags).
    # The adapter-based shared_cross_embodiment stays blocked as before.
    if async_ppo and is_shared and not name.startswith("shared_single"):
        raise ValueError(
            "async_ppo is only supported for INDEPENDENT per-tag policies "
            "or 'shared_single_*' (uniform dims), not adapter-based shared "
            "cross-embodiment.")

    if is_shared:
        # ── Shared cross-embodiment policy ───────────────────────────────
        obs_dims    = {tag: env.handlers[i].obs_dim    for i, tag in enumerate(tags)}
        action_dims = {tag: env.handlers[i].action_dim for i, tag in enumerate(tags)}
        shared_extra = {}
        if async_ppo:
            crit_dims = {int(h.critic_obs_dim) for h in env.handlers}
            assert len(crit_dims) == 1, (
                f"shared_single + async_ppo assumes the same critic_obs_dim across tags: {crit_dims}")
            shared_extra["critic_obs_dim"] = crit_dims.pop()
        assert not _ph_name, (
            "policy.phase_std_obs is supported only for independent per-tag policies "
            "(shared_* tags have different obs layouts, so no single index applies)")
        shared_module = make_policy(
            name,
            obs_dims    = obs_dims,
            action_dims = action_dims,
            **shared_extra, # type: ignore
            **kw_template, # type: ignore
        ).to(env.torch_device).train()
        shared_optimizer = torch.optim.Adam(shared_module.parameters(), lr=lr)

        # Every tag view + every optimizer entry points at the SAME
        # underlying module / optimizer — keeps the train-loop call
        # surface (``policies[tag].act(...)`` /
        # ``optimizers[tag].step()``) unchanged, while a single Adam
        # state coordinates the cross-task update.
        policies:   Dict[str, object]                  = {tag: shared_module.tag_view(tag) for tag in tags}
        optimizers: Dict[str, torch.optim.Optimizer]   = {tag: shared_optimizer            for tag in tags}
        return policies, optimizers, policy_cfg, shared_module

    # ── Independent per-tag policies (legacy / default) ──────────────────
    policies   = {}
    optimizers = {}
    for tag, h in zip(tags, env.handlers):
        extra = {"critic_obs_dim": int(h.critic_obs_dim)} if async_ppo else {}
        # Structure-aware policies (per_finger_*) receive the per-finger obs index groups
        # from the handler — derived by the same builder that produces the obs, so they
        # follow layout changes automatically (hand-written groups would silently bind the
        # wrong fingers). Handlers without that method pass nothing → the policy falls
        # back to a monolithic MLP.
        if name.startswith("per_finger") and hasattr(h, "per_finger_obs_groups"):
            extra["finger_groups"] = h.per_finger_obs_groups()
        # obs running-norm passthrough mask — the handler that produces the obs is the
        # single source of truth (based on obs_term_layout). Injected only when obs_norm is
        # on and the handler provides the hook; otherwise all dims are normalised (legacy).
        if bool(kw_template.get("obs_norm", False)):
            if hasattr(h, "obs_norm_passthrough_mask"):
                extra["obs_norm_passthrough"] = h.obs_norm_passthrough_mask()
            if async_ppo and hasattr(h, "critic_obs_norm_passthrough_mask"):
                extra["critic_obs_norm_passthrough"] = h.critic_obs_norm_passthrough_mask()
        _pidx = _resolve_phase_std_idx(h)
        if _pidx is not None:
            extra["phase_std_idx"] = _pidx
            _nf = int(getattr(h, "n_ctrl", 0))
            _nw = int(h.action_dim) - _nf
            if _ph_thr is not None:
                extra["phase_std_thresh"] = float(_ph_thr)
            _desc = []
            for _key, _base, _fin in (("phase_std_scale_hi", _ph_hi, _ph_hi_f),
                                      ("phase_std_scale_lo", _ph_lo, _ph_lo_f)):
                if _fin is None:
                    extra[_key] = _base
                    _desc.append(f"{_base:g}")
                    continue
                assert 0 < _nf < int(h.action_dim), (
                    f"{_key}_finger requires the handler to provide n_ctrl "
                    f"(n_ctrl={_nf}, action_dim={h.action_dim})")
                extra[_key] = [_base] * _nw + [float(_fin)] * _nf
                _desc.append(f"wrist {_base:g} / finger {float(_fin):g}")
            print(f'  [{tag}] phase-dependent σ: obs[{_pidx}] (={_ph_name}[{_ph_comp}]) '
                  f'< {float(_ph_thr) if _ph_thr is not None else 0.02:g} → σ×'
                  f'[{_desc[0]}],  otherwise σ×[{_desc[1]}]')
        policy = make_policy(
            name,
            obs_dim    = h.obs_dim,
            action_dim = h.action_dim,
            **extra,       # type: ignore  critic_obs_dim only when async_ppo
            **kw_template, # type: ignore
        ).to(env.torch_device).train()
        policies[tag]   = policy
        optimizers[tag] = torch.optim.Adam(policy.parameters(), lr=lr)
    return policies, optimizers, policy_cfg, None


# ──────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ──────────────────────────────────────────────────────────────────────────

def build_rollout_buffer(cfg, env) -> MultiHandRolloutBuffer:
    """One :class:`MultiHandRolloutBuffer` wraps all per-tag inner buffers.
    Single-hand and multi-hand share the exact same buffer type; the
    single-hand path just has a 1-element tag dict (zero overhead).
    """
    tr = cfg.training
    # ``batch_size`` / ``num_minibatch`` — null/missing/<=0 means "derive me".
    # Pin exactly one for full coverage (see from_handlers for the 3 modes):
    #   * num_minibatch pinned, batch_size null → batch_size grows with rollout.
    #   * batch_size pinned, num_minibatch null → minibatch SIZE fixed, count
    #     grows with rollout (bounded gradient-step cost — recommended when
    #     scaling nworld). Both auto-track ``nworld`` overrides.
    #   * both pinned → subset sampling.
    bs            = tr.get("batch_size", None)
    nmb           = tr.get("num_minibatch", None)
    batch_size    = int(bs)  if (bs  is not None and int(bs)  > 0) else None
    num_minibatch = int(nmb) if (nmb is not None and int(nmb) > 0) else None
    return MultiHandRolloutBuffer.from_handlers(
        env.handlers,
        unroll_length = int(tr.unroll_length),
        num_minibatch = num_minibatch,
        batch_size    = batch_size,
        device        = env.torch_device,
    )


# ──────────────────────────────────────────────────────────────────────────
# Inference-side builders (driven by SAVE_DIR/config.yaml snapshot)
# ──────────────────────────────────────────────────────────────────────────

def resolve_inference_save_dir(
    project_root:    Union[str, Path],
    hand:            str,
    task:            str,
    env_name:        str,
    deadzone_suffix: str = "",
    name_suffix:     str = "",
    resume_suffix:   str = "",
) -> Path:
    """Locate an existing SAVE_DIR for inference. Unlike the training-side
    :func:`grit.training.checkpoint.resolve_save_dir` (which **creates** the
    dir), this one **asserts** the dir exists so a typo fails fast instead
    of silently rolling out an untrained policy.

    Layout (matches training output)::

        <project_root>/output/checkpoints/{hand}_{env_name}{deadzone_suffix}{name_suffix}{resume_suffix}/

    * Single-hand runs   : ``hand = "tesollo"``
    * Shared multi-hand  : ``hand = "robotis_sh5+tesollo"`` (sorted, ``+``-joined)
    * ``name_suffix``     : the disambiguation tail the training run appended
      via ``cfg.output`` (e.g. ``"_nse30_nw4096_for_approach_test"``). Must
      match exactly — training prints it as a ``--name-suffix`` hint.
    """
    save_dir = (
        Path(project_root) / "output" / "checkpoints"
        / f"{hand}_{env_name}{deadzone_suffix}{name_suffix}{resume_suffix}"
    )
    assert save_dir.is_dir(), (
        f"\nSAVE_DIR not found: {save_dir}\n"
        f"  hand={hand!r}  task={task!r}  env_name={env_name!r}\n"
        f"  deadzone_suffix={deadzone_suffix!r}  name_suffix={name_suffix!r}  "
        f"resume_suffix={resume_suffix!r}\n"
        f"→ check output/checkpoints/ for the actual folder name."
    )
    return save_dir


def build_inference_env(save_dir: Path, render_nworld: int,
                        overrides: Optional[Sequence[str]] = None):
    """Inference entry: load ``<save_dir>/config.yaml`` snapshot, build
    orchestrator + env at ``render_nworld`` (viewer-friendly size).

    Reuses :func:`build_rl_env_and_handlers` for the env / handler-knobs leg —
    the only inference-specific bits here are (a) loading the snapshot cfg
    instead of composing a yaml, (b) overriding ``cfg.nworld``, and (c)
    surfacing ``sampled_orch`` (the first sub-env) which the warp viewer
    needs.

    ``overrides``: optional list of Hydra-style ``key=value`` dotlist strings
    (same syntax as ``--overrides`` in train.py — dotted keys / list values supported)
    applied ON TOP of the loaded snapshot cfg via ``OmegaConf.from_dotlist``
    merge. Applied AFTER the ``render_nworld`` write, so ``nworld=...`` in the
    dotlist wins over the ``--nworld`` viewer arg. Orchestrator / env / handler
    knobs / policy are all built from the merged cfg, so any snapshot key can be
    changed at evaluation time (keys that change the obs/action dim — policy
    architecture, Training.n_bps, ... — are rejected by the strict ckpt dim check).

    Returns ``(orchestrator, env, sampled_orch, cfg)``.
    """
    cfg = load_run_config_as_omegaconf(save_dir)
    if cfg is None:
        raise FileNotFoundError(
            f"no config.yaml under {save_dir} — every run directory written by train.py / "
            f"distill.py carries the resolved config snapshot the env is rebuilt from")
    print(f'config loaded     : {save_dir / "config.yaml"}')

    print(f'  yaml nworld     : {cfg.nworld}  →  viewer override : {render_nworld}')
    cfg.nworld = int(render_nworld)

    # ── CLI overrides (same Hydra-style dotlist as train.py) ───────────────
    # Merged on top of the snapshot values — dotted keys (``handler.termination.
    # MAX_EPISODE_STEPS=300``) / list values (``obj_idxs=[0,2,3]``) supported. nworld
    # given in the dotlist takes precedence over the viewer override above.
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
        print(f'  eval overrides  : {list(overrides)}')
        print(f'  effective nworld: {cfg.nworld}')

    seed = int(cfg.get("training", {}).get("seed", 42)) if cfg.get("training") else 42
    random.seed(seed); np.random.seed(seed)

    orchestrator = Env_Orchestrator(overall_cfg=cfg, test_mode=False) # type: ignore
    
    if cfg.env_type == "single_hand":                  # type: ignore
        orchestrator.build_sub_env(
        with_mjwarp=True, for_inference=True,
            geom_collision_type=str(cfg.get("geom_collision_type", "mesh")),
            obj_spec_provider=obj_spec_provider_from_config(cfg, verbose=True),
        )
    else:
        raise NotImplementedError(
            f"env_type={cfg.env_type!r} is not included in grit_share "
            "(single_hand only)")
    # Re-use the training-time builder for env + handler knobs application.
    # We don't need its (rl_env_cfg, handler_cfg, tags) return — inference
    # rebuilds those from cfg as needed.
    env, _rl_env_cfg, _handler_cfg, _tags = build_rl_env_and_handlers(
        orchestrator, cfg, str(cfg.env_name),
    )

    sampled_orch = orchestrator.env_list[0]
    return orchestrator, env, sampled_orch, cfg


def resolve_eval_checkpoint(
    save_dir,
    hand_name: str,
    task_name: str,
    env_name:  str,
    ckpt: Optional[Union[str, int]] = None,
) -> Optional[Path]:
    """Pick which checkpoint an eval run loads.

    ``ckpt`` is the user-facing selector:
      * ``None``          — latest (highest ``seen_steps``); may be ``None``
                            when the dir holds no matching file.
      * ``12288000``      — that exact ``seen_steps`` (int or digit string).
      * ``"....pt"``      — an explicit file; relative paths resolve against
                            ``save_dir``.

    Anything explicit raises when it doesn't exist — an eval must not
    silently roll out a checkpoint the caller didn't ask for.
    """
    if ckpt is None:
        return find_latest_checkpoint(save_dir, hand_name, task_name, env_name)

    s = str(ckpt).strip()
    if s.endswith(".pt"):
        p = Path(s)
        if not p.is_absolute():
            p = Path(save_dir) / p
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return p

    # Accept '12_288_000' / '12,288,000' typed for readability.
    digits = s.replace("_", "").replace(",", "")
    if not digits.isdigit():
        raise ValueError(
            f"--ckpt must be a seen_steps integer or a '*.pt' path, got {ckpt!r}"
        )
    return find_checkpoint_by_steps(
        save_dir, hand_name, task_name, env_name, int(digits))


def build_policy_and_load(
    cfg,
    env,
    save_dir: Path,
    hand:     str,
    task:     str,
    env_name: str,
    rollout_tag: Optional[str] = None,
    ckpt: Optional[Union[str, int]] = None,
):
    """Inference-side policy build + checkpoint load.

    Auto-detects shared vs independent mode by ``cfg.policy.name`` and
    returns a **single-tag** policy view in both cases (so the caller can
    treat the result uniformly with ``policy.act(obs)``).

    Returns ``(policy, policy_loaded, h_used)`` where:
      * ``policy``        — the rollout-ready policy view.
      * ``policy_loaded`` — False when no matching ckpt found (rolling out
                            the untrained baseline for sanity comparisons).
      * ``h_used``        — the handler whose obs feeds the policy each
                            tick. In shared mode this picks the handler
                            matching ``rollout_tag``; in independent mode
                            it's always ``env.handlers[0]``.

    ``ckpt`` selects which checkpoint to load (``None`` = latest); see
    :func:`resolve_eval_checkpoint` for the accepted forms.
    """
    pol_raw = OmegaConf.to_container(cfg.policy, resolve=True)
    assert isinstance(pol_raw, dict), (
        f"build_policy_and_load: cfg.policy must convert to a dict, got "
        f"{type(pol_raw).__name__}."
    )
    # Re-key as a plain Dict[str, Any] so ``**pol_kw`` type-checks (OmegaConf
    # to_container returns ``DictKeyType`` which the typechecker won't accept
    # for kwarg expansion).
    pol_kw: Dict[str, object] = {str(k): v for k, v in pol_raw.items()}
    pol_name  = str(pol_kw.pop("name", "normal_tanh_mlp"))
    pol_kw.pop("init_std_finger", None)   # train.py-only knob (not a policy ctor argument)
    is_shared = pol_name.startswith("shared_")

    if is_shared:
        training_tags    = list(cfg.using_hand_name_list)
        obs_dims_dict    = {t: env.handlers[i].obs_dim    for i, t in enumerate(training_tags)}
        action_dims_dict = {t: env.handlers[i].action_dim for i, t in enumerate(training_tags)}
        shared = make_policy(
            pol_name, obs_dims=obs_dims_dict, action_dims=action_dims_dict, **pol_kw,
        ).to(env.torch_device).eval()
        hands_joined = "+".join(sorted(training_tags))
        if hand != hands_joined:
            print(f'  ⚠ hand={hand!r} differs from snapshot hands_joined={hands_joined!r}')

        ckpt = resolve_eval_checkpoint(
            save_dir, hands_joined, task, env_name, ckpt=ckpt)
        if ckpt is None:
            print(f'[!] no shared ckpt under {save_dir} — using UNTRAINED policy.')
            policy_loaded = False
        else:
            if not is_shared_checkpoint(ckpt):
                print(f'  ⚠ {ckpt.name} does not look like a shared ckpt — load may fail.')
            blob = load_shared_checkpoint(
                ckpt, policy=shared,
                expected_obs_dims=obs_dims_dict, expected_action_dims=action_dims_dict,
                expected_hands_joined=hands_joined,
                expected_task_name=task, expected_env_name=env_name,
                map_location=env.torch_device,
            )
            print(f'loaded SHARED  : {ckpt.name}  (seen_steps={blob["seen_steps"]:,})')
            policy_loaded = True

        tag = rollout_tag or env.handlers[0].hand_util.hand_name
        # ``make_policy(...)`` returns the registered class; Pylance can't
        # introspect through the registry, so the .tags / .tag_view symbols
        # need explicit attr access.
        shared_tags = getattr(shared, "tags")        # type: ignore[arg-type]
        if tag not in shared_tags:
            raise ValueError(
                f"rollout_tag={tag!r} not in shared policy tags {shared_tags}. "
                f"Pass a valid tag from the trained set."
            )
        h_used = next(env.handlers[i] for i, t in enumerate(training_tags) if t == tag)
        policy = getattr(shared, "tag_view")(tag)    # type: ignore[arg-type]
        print(f'policy mode    : SHARED — tag_view({tag!r})  '
              f'(backing module {sum(p.numel() for p in shared.parameters()):,} params)')
        return policy, policy_loaded, h_used

    # ── Independent (legacy) ─────────────────────────────────────────
    h = env.handlers[0]
    # Async actor-critic: rebuild the critic trunk over the env's privileged
    # ``critic_obs_dim`` when the snapshot trained with ``training.async_ppo`` —
    # otherwise the checkpoint's critic ``state_dict`` won't match. Off → omitted
    # (symmetric, unchanged).
    async_ppo = bool(cfg.training.get("async_ppo", False))
    extra = {"critic_obs_dim": int(h.critic_obs_dim)} if async_ppo else {}
    # obs_norm passthrough mask — injected from the handler exactly as in the training
    # builder. Without it the normalizer has no ``passthrough`` buffer and loading the
    # state_dict of a ckpt trained with obs_norm=True fails with an unexpected key. With it,
    # the ckpt's (training-time) mask overwrites the buffer on load and is restored exactly.
    if bool(pol_kw.get("obs_norm", False)):
        if hasattr(h, "obs_norm_passthrough_mask"):
            extra["obs_norm_passthrough"] = h.obs_norm_passthrough_mask()
        if async_ppo and hasattr(h, "critic_obs_norm_passthrough_mask"):
            extra["critic_obs_norm_passthrough"] = h.critic_obs_norm_passthrough_mask()
    policy = make_policy(
        pol_name, obs_dim=h.obs_dim, action_dim=h.action_dim, **extra, **pol_kw,
    ).to(env.torch_device).eval()
    # ckpt files are tagged with the LEARNING handler's hand_name (save_checkpoint
    # uses ``hand_name=tag`` = ``handler.hand_util.hand_name``). For leader-follower
    # that is the FOLLOWER (e.g. 'allegro') — distinct from the joined SAVE_DIR tag
    # passed as ``hand`` ('allegro+tesollo'). Use the handler tag so LF ckpts load;
    # for single-hand runs the two are identical (no behavior change).
    ckpt_hand = str(getattr(h.hand_util, "hand_name", hand))
    ckpt = resolve_eval_checkpoint(
        save_dir, ckpt_hand, task, env_name, ckpt=ckpt)
    if ckpt is None:
        print(f'[!] no ckpt under {save_dir} matching {ckpt_hand!r} — using UNTRAINED policy.')
        return policy, False, h

    blob = load_checkpoint(
        ckpt, policy=policy,
        expected_obs_dim=h.obs_dim, expected_action_dim=h.action_dim,
        expected_hand_name=ckpt_hand, expected_task_name=task, expected_env_name=env_name,
        map_location=env.torch_device,
    )
    print(f'loaded         : {ckpt.name}  (seen_steps={blob["seen_steps"]:,})')
    print(f'policy mode    : independent ({pol_name})')
    return policy, True, h
