"""Weights & Biases setup helper for training entrypoints.

Pulled out of ``scripts/train.py`` so any training driver (CLI script,
notebook, sweep agent) can wire wandb the same way:

    log_metrics, finish_wandb = setup_wandb(
        cfg, hands_joined=..., tags=tags, task_name=..., env_name=...,
        save_dir=..., env=env, policies=policies,
        rl_env_cfg=..., train_cfg=..., policy_cfg=...,
    )
    ...
    log_metrics({"train/<tag>/reward_mean": ...}, step=seen_steps)
    ...
    finish_wandb()

``cfg.wandb`` schema (all optional)::

    enabled:        bool
    project:        str
    entity:         str | null
    reinit:         bool
    name_template:  "{save_dir_name}"      # {hand_name} {hands_joined} {hand_type} {task_name} {env_name} {save_dir_name}
    group_template: "{hands_joined}"
    tags:           [ "{task_name}", "{env_name}" ]
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from omegaconf import OmegaConf

from grit.training.run_config import extract_handler_knobs
from grit.util.utils         import fmt_template, flatten_dict_with_prefix


def setup_wandb(
    cfg,
    *,
    hands_joined: str,
    tags:         List[str],
    task_name:    str,
    env_name:     str,
    save_dir:     Path,
    env,
    policies:     Dict[str, torch.nn.Module],
    rl_env_cfg:   dict,
    train_cfg:    dict,
    policy_cfg:   dict,
):
    """Single W&B run for the whole training; metrics are per-tag namespaced
    by the training loop. Returns ``(log_metrics, finish_wandb)`` — both
    are no-ops when wandb is disabled or not installed, so the caller can
    invoke them unconditionally.

    Templates can use the following ``{var}`` substitutions:
        ``{hand_name}``  (alias of ``{hands_joined}`` for backward compat),
        ``{hands_joined}``, ``{hand_type}``, ``{task_name}``, ``{env_name}``,
        ``{save_dir_name}``.
    """
    wandb_cfg = OmegaConf.to_container(cfg.wandb, resolve=True) if cfg.get("wandb") else {}
    assert isinstance(wandb_cfg, dict)
    enabled = bool(wandb_cfg.get("enabled", True))
    project = str(wandb_cfg.get("project", "grit-tracking-rl"))
    entity  = wandb_cfg.get("entity")
    reinit  = bool(wandb_cfg.get("reinit", True))

    tmpl_vars = dict(
        hand_name     = hands_joined,        # backward-compat alias for old templates
        hands_joined  = hands_joined,
        hand_type     = env.handlers[0].hand_util.hand_type,
        task_name     = task_name,
        env_name      = env_name,
        save_dir_name = save_dir.name,
    )
    run_name = fmt_template(wandb_cfg.get("name_template") or "{save_dir_name}", **tmpl_vars)
    group    = fmt_template(wandb_cfg.get("group_template", "{hands_joined}"),  **tmpl_vars)
    tags_wb  = [fmt_template(t, **tmpl_vars) for t in (wandb_cfg.get("tags") or [])]

    try:
        import wandb
    except ImportError:
        wandb = None  # type: ignore
        print("wandb not installed — running with a no-op logger.")

    run = None
    if enabled and wandb is not None:
        wandb_config = dict(
            hands_joined  = hands_joined,
            tags_list     = list(tags),
            task_name     = task_name,
            env_name      = env_name,
            save_dir_name = save_dir.name,
            save_dir      = str(save_dir),
            n_handlers    = len(env.handlers),
            num_envs      = int(env.num_envs),
            sim_nstep     = int(env.sim_nstep),
        )
        # Per-tag obs/action dims + total params.
        for tag in tags:
            h = env.handlers[tags.index(tag)]
            wandb_config[f"handler/{tag}/NWORLD"]      = h.NWORLD
            wandb_config[f"handler/{tag}/obs_dim"]     = h.obs_dim
            wandb_config[f"handler/{tag}/action_dim"]  = h.action_dim
            wandb_config[f"policy/{tag}/n_params"]     = sum(
                p.numel() for p in policies[tag].parameters()
            )
        # Same yaml drives every handler — so the handler knob snapshot for
        # tag-0 is representative of all tags.
        flatten_dict_with_prefix("rl_env/",   rl_env_cfg,                              wandb_config)
        flatten_dict_with_prefix("handler/",  extract_handler_knobs(env.handlers[0]),  wandb_config)
        flatten_dict_with_prefix("training/", train_cfg,                               wandb_config)
        flatten_dict_with_prefix("policy/",   policy_cfg,                              wandb_config)

        run = wandb.init(
            project = project,
            entity  = entity,
            name    = run_name,
            group   = group,
            tags    = tags_wb,
            config  = wandb_config,
            reinit  = reinit,
        )
        print(f'wandb run : {run.name}  →  {run.url}')
    else:
        print('wandb logging disabled.')

    def log_metrics(metrics: dict, step: int) -> None:
        if run is None:
            return
        wandb.log(metrics, step=int(step))     # type: ignore[union-attr]

    def finish_wandb() -> None:
        if run is not None:
            wandb.finish()                     # type: ignore[union-attr]

    return log_metrics, finish_wandb
