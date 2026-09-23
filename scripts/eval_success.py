"""scripts/eval_success.py — headless success-rate evaluation of a checkpoint.

    python scripts/eval_success.py --save-dir pretrained/<hand>_grasping_teacher --hand tesollo \\
        --env grasping_teacher --nworld 1024

Builds the env from the run's config snapshot at ``--nworld`` worlds (objects
are the run's ``obj_idxs`` set, spread over the worlds), rolls the policy out
deterministically for one full episode window and prints the same metrics
``train.py`` logs at its eval cadence (``success_rate``, ``success_rate_lift``,
``success_rate_strict``, first-failure reasons, …). ``--json`` writes them out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
os.environ.setdefault("GRIT_FORCE_HEADLESS", "1")

from grit.util.headless_guard import ensure_headless_gui_stubs  # noqa: E402
ensure_headless_gui_stubs()

from grit.training.builders import build_inference_env, build_policy_and_load  # noqa: E402
from grit.training.evaluation import run_eval  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--save-dir", required=True)
    p.add_argument("--hand", required=True)
    p.add_argument("--task", default="grasping")
    p.add_argument("--env", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--nworld", type=int, default=1024)
    p.add_argument("--eval-length", type=int, default=None, help="default: MAX_EPISODE_STEPS")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--json", default=None, help="write metrics to this file")
    p.add_argument("--overrides", nargs="*", default=None)
    args = p.parse_args()

    from pathlib import Path
    save_dir = Path(args.save_dir).expanduser().resolve()
    orchestrator, env, so, cfg = build_inference_env(save_dir, args.nworld, overrides=args.overrides)
    policy, loaded, h = build_policy_and_load(cfg, env, save_dir, hand=args.hand, task=args.task,
                                              env_name=args.env, ckpt=args.ckpt)
    assert loaded, "no checkpoint loaded — check --save-dir/--hand/--task/--env"
    if hasattr(h, "lift_curriculum_scale"):
        h.lift_curriculum_scale = 0.0      # final-curriculum dynamics (no scripted lift assist)

    class _ActorOnly:
        """run_eval feeds ``critic_obs``; skip the critic (its input differs for
        distilled students) — only the actor matters for the rollout."""
        def __init__(self, pol): self.pol = pol
        def act(self, obs, deterministic=True, **_kw):
            return self.pol.act(obs, deterministic=deterministic, inference_only=True)
        def eval(self): self.pol.eval()
        def train(self): self.pol.train()
        @property
        def training(self): return self.pol.training

    res = run_eval(_ActorOnly(policy), h, eval_length=args.eval_length or int(h.MAX_EPISODE_STEPS),
                   deterministic=not args.stochastic)
    keys = ["success_rate", "success_rate_lift", "success_rate_strict", "mean_return",
            "mean_first_done_step", "n_world", "eval_length", "eval_sps"]
    print(f"\n== {save_dir.name} ==")
    for k in keys:
        if k in res:
            v = res[k]
            print(f"  {k:22s}: {v:.4f}" if isinstance(v, float) else f"  {k:22s}: {v}")
    print(f"  first_done_reasons    : {res.get('first_done_reasons')}")
    if args.json:
        slim = {k: v for k, v in res.items() if isinstance(v, (int, float, str, dict, list))}
        slim["save_dir"] = save_dir.name; slim["hand"] = args.hand
        with open(args.json, "w") as f:
            json.dump(slim, f, indent=1, default=str)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
