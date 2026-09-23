"""scripts/distill.py — Teacher→Student distillation (DAgger + hybrid RL).

General-purpose distillation runner in the style of train.py::

    python scripts/distill.py -c distill_grasping_tf_b \\
        --overrides distill.n_iters=4000 wandb.enabled=true

Overview
--------
1. **teacher resolve** — locate the existing training SAVE_DIR from
   ``teacher.{hand,task,env_name,name_suffix}`` (``resolve_inference_save_dir``
   convention) and load its config.yaml snapshot + latest ckpt. The critic
   input dimension is read directly from the ckpt state_dict (works for both
   asym and sym teachers).
2. **student env** — take the teacher snapshot cfg, replace only ``env_name``
   with ``student.env_name``, inject ``student.training_overrides`` into
   ``cfg.Training`` and rebuild. All observation constraints (masking /
   partial view / stage state machine, ...) are read by the handler from
   cfg.Training.
3. **DAgger** — per-episode driver curriculum (teacher/student/blend modes,
   p_teacher annealing) + GPU ring buffer + Gaussian KL distribution matching.
4. **hybrid RL** (optional) — true-reward PPO + teacher-KL anchor (annealed).
   The critic uses the handler's ``collect_critic_obs()`` (privileged
   observation) when ``rl_finetune.critic: asym``, otherwise the student's
   own obs.
5. **save** — ckpt + config snapshot to ``{hand}_{task}_{student_env}_{run_tag}``
   (DAgger) and ``..._rl`` (hybrid). The snapshot records
   ``policy.critic_obs_dim`` explicitly and sets ``training.async_ppo=false``
   so that **evaluate.py can load / visualise it as is**.

Handler generalisation contract (capability probes with automatic fallback):
  * ``h.collect_teacher_obs()``  missing -> teacher obs = student obs (symmetric distill)
  * ``h.collect_critic_obs()``   missing -> critic obs = student obs
  * ``h.lift_curriculum_scale``  present -> pinned to 0 (late-teacher dynamics)
  * success/crush reason codes come from a reverse lookup of ``SubEnvHandler.REASON_NAME``
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from omegaconf import OmegaConf  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Headless guard: MUST run before any grit import. Those pull in a
# module-level ``import pyautogui`` that crashes on a server with no DISPLAY
# (KeyError: 'DISPLAY' inside mouseinfo). Importing this stubs the GUI deps when
# headless. (grit.util.* __init__ is empty → this import is light.)
from grit.util.headless_guard import ensure_headless_gui_stubs  # noqa: E402
ensure_headless_gui_stubs()

from grit.training.orchestrator.orchestrator import Env_Orchestrator  # noqa: E402
from grit.training.builders import (  # noqa: E402
    build_rl_env_and_handlers, resolve_inference_save_dir,
    obj_spec_provider_from_config,
)
from grit.training.run_config import load_run_config_as_omegaconf  # noqa: E402
from grit.training.checkpoint import (  # noqa: E402
    find_latest_checkpoint, load_checkpoint, save_checkpoint)
from grit.training.evaluation import run_eval  # noqa: E402
from grit.training.rl_env_base import SubEnvHandler  # noqa: E402
from grit.training.algorithm.ppo import _adapt_lr  # noqa: E402
from grit.model.base_networks import make_policy  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────

def gaussian_kl(mu_p, sig_p, mu_q, sig_q, eps=1e-6):
    """KL( N(mu_p,sig_p) ‖ N(mu_q,sig_q) ) — diagonal, summed over the action dim."""
    sig_p = sig_p.clamp_min(eps); sig_q = sig_q.clamp_min(eps)
    var_p, var_q = sig_p ** 2, sig_q ** 2
    return (torch.log(sig_q / sig_p)
            + (var_p + (mu_p - mu_q) ** 2) / (2.0 * var_q) - 0.5).sum(-1)


@torch.no_grad()
def compute_gae(rew, val, done, last_val, gamma, lam):
    T, N = rew.shape
    adv = torch.zeros_like(rew)
    gae = torch.zeros(N, device=rew.device)
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        nextval = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nextval * nonterm - val[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv, adv + val


class ReplayBuffer:
    """GPU ring buffer of (student_obs, teacher_loc, teacher_scale)."""

    def __init__(self, cap, obs_dim, act_dim, device, aux_dim=0, with_lab=False):
        self.cap = int(cap); self.n = 0; self.ptr = 0
        self.obs = torch.zeros(self.cap, obs_dim, device=device)
        self.loc = torch.zeros(self.cap, act_dim, device=device)
        self.scale = torch.zeros(self.cap, act_dim, device=device)
        # privileged reconstruction target (e.g. obj_pose_diff from the teacher obs) — unused when aux_dim=0
        self.aux = torch.zeros(self.cap, aux_dim, device=device) if aux_dim > 0 else None
        # re-grasp label (blind env) — unused when with_lab=False
        self.lab = torch.zeros(self.cap, device=device) if with_lab else None

    def push(self, obs, loc, scale, aux=None, lab=None):
        b = obs.shape[0]
        idx = (torch.arange(b, device=obs.device) + self.ptr) % self.cap
        self.obs[idx] = obs; self.loc[idx] = loc; self.scale[idx] = scale
        if self.aux is not None:
            self.aux[idx] = aux
        if self.lab is not None:
            self.lab[idx] = lab
        self.ptr = int((self.ptr + b) % self.cap); self.n = min(self.n + b, self.cap)

    def sample(self, bs):
        idx = torch.randint(0, self.n, (min(bs, self.n),), device=self.obs.device)
        aux = self.aux[idx] if self.aux is not None else None
        lab = self.lab[idx] if self.lab is not None else None
        return self.obs[idx], self.loc[idx], self.scale[idx], aux, lab


class ObsSwapActor:
    """run_eval adapter: swaps the obs source and bypasses the critic (inference_only)."""

    def __init__(self, p, get_obs=None):
        self.p = p; self.get_obs = get_obs

    @property
    def training(self):
        return self.p.training

    def eval(self):
        self.p.eval(); return self

    def train(self, mode=True):
        self.p.train(mode); return self

    #: set a LongTensor here to select the actor input columns (the head keeps the full obs)
    actor_cols = None

    def act(self, obs, deterministic=True, critic_obs=None, **kw):
        o = self.get_obs() if self.get_obs is not None else obs
        if getattr(self, "actor_cols", None) is not None:
            o = o.index_select(1, self.actor_cols)   # the actor sees the obs without the hist block
        return self.p.act(o, deterministic=deterministic, inference_only=True)


class RegraspPredActor(ObsSwapActor):
    """run_eval adapter that also feeds the re-grasp head prediction into the env
    (``regrasp_pred_torch``). With ``REGRASP_STAGE_RESET_SOURCE=pred`` the env rewinds the
    stage after ``consec`` consecutive positive predictions."""

    def __init__(self, p, head, cols, h):
        super().__init__(p)
        self.head, self.cols, self.h = head, cols, h
        self.n_fired = 0; self.n_pred = 0; self.n_lab = 0; self.n_tp = 0; self.n_steps = 0

    def act(self, obs, deterministic=True, critic_obs=None, **kw):
        with torch.no_grad():
            pred = (torch.sigmoid(self.head(obs[:, self.cols]).squeeze(1)) > 0.5)
            self.h.regrasp_pred_torch.copy_(pred.float())
            lab = self.h.regrasp_label_torch > 0.5
            self.n_fired += int(self.h.regrasp_pred_fired_torch.sum())
            self.n_pred += int(pred.sum()); self.n_lab += int(lab.sum())
            self.n_tp += int((pred & lab).sum()); self.n_steps += 1
        o = (obs if getattr(self, "actor_cols", None) is None
             else obs.index_select(1, self.actor_cols))
        return self.p.act(o, deterministic=deterministic, inference_only=True)

    def stats(self):
        return dict(prec=self.n_tp / max(self.n_pred, 1), rec=self.n_tp / max(self.n_lab, 1),
                    fired=self.n_fired, lab_rate=self.n_lab / max(self.n_steps * self.h.NWORLD, 1))


@torch.no_grad()
def _first_sustained(mask_TN: torch.Tensor, k: int) -> torch.Tensor:
    """(T,N) bool -> per world, the first step t at which True has held for k consecutive steps (-1 if never)."""
    T, N = mask_TN.shape
    run = torch.zeros(N, device=mask_TN.device, dtype=torch.long)
    out = torch.full((N,), -1, device=mask_TN.device, dtype=torch.long)
    for t in range(T):
        run = torch.where(mask_TN[t], run + 1, torch.zeros_like(run))
        hit = (run >= k) & (out < 0)
        out[hit] = t
    return out


@torch.no_grad()
def autocal_stage_thresholds(h, env, teacher, get_teacher_obs, action_dim, cfg_ac):
    """Demonstration-based auto-calibration of the stage thresholds (tracking-free state machine).

    Rolls the teacher for one episode in the student env (all worlds, no reset), recording
    per step (frozen hand-center distance, wrist z, GT contact, GT lift). Then grid-searches
    (dist_thresh, near_ticks, lift_proxy_m) to minimise the error between the GT transition
    times (t1* = contact established, t2* = GT lift target reached) and the transitions
    predicted by the state machine. GT (hand_obj_active, obj lift) is used **only at
    calibration time** — the deployed student stays tracking-free.
    Returns (dist, ticks, lift_m), or None when not applicable."""
    if not getattr(h, "_stage_from_obs", False):
        return None
    tracking_free = bool(getattr(h, "_tracking_free", False))
    dist_wp = (getattr(h, "_frozen_hc_dist_wp", None) if tracking_free
               else getattr(h, "best_dist_hc_wp", None))
    if dist_wp is None:
        return None
    import warp as wp  # noqa: PLC0415
    DEV = env.torch_device
    T = int(h.MAX_EPISODE_STEPS)
    N = int(h.NWORLD)
    dist_t = wp.to_torch(dist_wp)
    xpos_t = wp.to_torch(h.d.xpos)
    objp0_t = wp.to_torch(h.cond.get("obj_p_init"))
    contact_wp = getattr(h, "hand_obj_active_wp", None)
    contact_t = wp.to_torch(contact_wp) if contact_wp is not None else None

    dist_buf = torch.zeros(T, N, device=DEV)
    wz_buf = torch.zeros(T, N, device=DEV)
    ct_buf = torch.zeros(T, N, device=DEV, dtype=torch.bool)
    lift_buf = torch.zeros(T, N, device=DEV)
    touch_gate = bool(getattr(h, "_stage_touch_gate", False))
    tq_buf = torch.zeros(T, N, device=DEV) if touch_gate else None

    with h.eval_context():
        env.reset()
        env.step(torch.zeros(N, action_dim, device=DEV))   # priming (frozen capture)
        for t in range(T):
            out = teacher.act(get_teacher_obs(), deterministic=True, inference_only=True)
            env.step(out["action"])
            dist_buf[t] = dist_t
            wz_buf[t] = xpos_t[:, int(h.wrist_body_id), 2]
            lift_buf[t] = xpos_t[:, int(h.obj_body_id), 2] - objp0_t[:, 2]
            if contact_t is not None:
                ct_buf[t] = contact_t > 0
            if tq_buf is not None:
                tq_buf[t] = h.obs_torch[:, h._tq_c0:h._tq_c1].abs().mean(1)

    # GT transition times
    lift_target = float(h.LIFT_TARGET_M)
    t1_gt = (_first_sustained(ct_buf, 3) if contact_t is not None
             else _first_sustained(lift_buf > 0.005, 3))
    t2_gt = _first_sustained(lift_buf > lift_target, 1)
    valid = (t1_gt >= 0) & (t2_gt >= 0) & (t2_gt > t1_gt)
    n_valid = int(valid.sum())
    if n_valid < max(32, N // 20):
        print(f"[autocal] too few valid trajectories ({n_valid}/{N}) — keeping default thresholds")
        return None

    # (dist, ticks) grid fit: median |error| between predicted t1 and GT t1 (+ penalty for never firing)
    dist_grid = [float(x) for x in cfg_ac.get("dist_grid",
                 [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12])]
    ticks_grid = [int(x) for x in cfg_ac.get("ticks_grid", [1, 2, 3, 5, 7, 10])]
    # touch gate: near = (dist < dth) AND (mean|torque_proxy| > tth) — tth is grid-fitted as well
    touch_grid = ([float(x) for x in cfg_ac.get("touch_grid",
                   [0.005, 0.01, 0.02, 0.03, 0.05, 0.08])] if touch_gate else [None])
    best = None
    for tth in touch_grid:
        base_mask = (dist_buf < 1e9)
        if tth is not None:
            base_mask = tq_buf > tth
        for dth in dist_grid:
            below = (dist_buf < dth) & base_mask
            for k in ticks_grid:
                t1p = _first_sustained(below, k)
                miss = ((t1p < 0) & valid).float().mean()
                err = (t1p - t1_gt).abs().float()
                med = err[valid & (t1p >= 0)].median() if bool((valid & (t1p >= 0)).any()) else torch.tensor(1e9)
                score = float(med) + 200.0 * float(miss)
                if best is None or score < best[0]:
                    best = (score, dth, k, tth)
    _, d_star, k_star, t_star = best
    if touch_gate and t_star is not None:
        h._stage_touch_thresh = float(t_star)
        print(f"[autocal] touch gate threshold: mean|torque_proxy| > {t_star:.3f}")

    # Whether to fit the wrist-z proxy: needed for tracking-free / approach-only
    # (no obj z after grasp); full-tracked keeps using obj z, so not needed.
    fit_proxy = tracking_free or bool(getattr(h, "_track_approach_only", False))
    if not fit_proxy:
        print(f"[autocal] (tracked) n_valid={n_valid}/{N}  "
              f"dist={d_star:.3f}m x{k_star}tick (err_med≈{best[0]:.1f})  "
              f"lift threshold keeps the existing obj-z value")
        return d_star, k_star, None

    # wrist-z proxy threshold fit (anchor = wrist z at GT t1)
    idx = torch.arange(N, device=DEV)
    anchor = wz_buf[t1_gt.clamp(min=0), idx]
    lift_grid = [float(x) for x in cfg_ac.get("lift_grid",
                 [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15])]
    bestL = None
    tt = torch.arange(T, device=DEV).unsqueeze(1)
    after_t1 = tt >= t1_gt.clamp(min=0).unsqueeze(0)
    for L in lift_grid:
        t2p = _first_sustained((wz_buf - anchor.unsqueeze(0) > L) & after_t1, 1)
        miss = ((t2p < 0) & valid).float().mean()
        err = (t2p - t2_gt).abs().float()
        med = err[valid & (t2p >= 0)].median() if bool((valid & (t2p >= 0)).any()) else torch.tensor(1e9)
        score = float(med) + 200.0 * float(miss)
        if bestL is None or score < bestL[0]:
            bestL = (score, L)
    _, L_star = bestL

    print(f"[autocal] n_valid={n_valid}/{N}  dist={d_star:.3f}m x{k_star}tick "
          f"(err_med≈{best[0]:.1f})  wrist-z proxy=+{L_star:.3f}m (err_med≈{bestL[0]:.1f})")
    return d_star, k_star, L_star


def read_adaptive_lr_cfg(section) -> dict:
    """Read the KL-based adaptive-lr knobs from a yaml section (distill / rl_finetune).

    Same convention as the train.py path: ``desired_kl`` null/0 -> OFF (fixed lr).
    When set (typically 0.01), :func:`_adapt_lr` scales the lr once per iteration by
    ``kl_adapt_factor`` (up or down; dead band = 0.5x .. 2x desired_kl)."""
    dk = section.get("desired_kl", None)
    return dict(desired_kl=(float(dk) if dk else None),
                lr_min=float(section.get("lr_min", 1.0e-6)),
                lr_max=float(section.get("lr_max", 1.0e-3)),
                factor=float(section.get("kl_adapt_factor", 1.5)))


def reason_code(name: str) -> int:
    """Reverse lookup in REASON_NAME (-999 if absent — never matches)."""
    for k, v in SubEnvHandler.REASON_NAME.items():
        if v == name:
            return int(k)
    return -999


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-c", "--config", required=True,
                   help="config/training/<name>.yaml (extension optional)")
    p.add_argument("--overrides", nargs="*", default=[], metavar="KEY=VALUE",
                   help="OmegaConf dotlist overrides (same syntax as train.py)")
    return p.parse_args()


def load_cfg(args):
    name = args.config if args.config.endswith(".yaml") else args.config + ".yaml"
    path = PROJECT_ROOT / "config" / "training" / name
    assert path.is_file(), f"config not found: {path}"
    cfg = OmegaConf.load(path)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))
    return cfg


def main():
    args = parse_args()
    dcfg = load_cfg(args)
    print(f"[distill] config: {args.config}  overrides: {args.overrides}")

    # ── 1. teacher resolve + snapshot load ──────────────────────────────
    t = dcfg.teacher
    if t.get("save_dir", None):
        teacher_dir = Path(str(t.save_dir)).expanduser()
        if not teacher_dir.is_absolute():
            teacher_dir = PROJECT_ROOT / teacher_dir
        assert teacher_dir.is_dir(), f"teacher.save_dir not found: {teacher_dir}"
    else:
        teacher_dir = resolve_inference_save_dir(
            PROJECT_ROOT, hand=str(t.hand), task=str(t.task),
            env_name=str(t.env_name), name_suffix=str(t.get("name_suffix", "")))
    cfg = load_run_config_as_omegaconf(teacher_dir)
    HAND, TASK = str(t.hand), str(t.task)
    TEACHER_ENV = str(cfg.env_name)
    print(f"[distill] teacher: {teacher_dir.name}")

    # ── 2. rewrite the student env cfg ──────────────────────────────────
    s = dcfg.student
    OmegaConf.set_struct(cfg, False)
    cfg.env_name = str(s.env_name)
    cfg.nworld = int(s.nworld)
    cfg.n_sub_env = int(s.n_sub_env)
    cfg.wandb.enabled = False                      # env-side wandb is always off
    for k, v in dict(s.get("training_overrides", {}) or {}).items():
        setattr(cfg.Training, str(k), v)
    for k, v in dict(s.get("handler_overrides", {}) or {}).items():
        setattr(cfg.handler, str(k), v)          # handler-level overrides (reward knobs etc.)
        print(f"[distill] handler override: {k} = {v}")
    print(f"[distill] student env: {cfg.env_name}  nworld={cfg.nworld} "
          f"n_sub_env={cfg.n_sub_env}")

    orch = Env_Orchestrator(overall_cfg=cfg, test_mode=False)
    orch.build_sub_env(
        with_mjwarp=True, for_inference=True,
        geom_collision_type=str(cfg.get("geom_collision_type", "mesh")),
        # Training.object_source: 'dataset' (mesh) | 'primitive' (procedural)
        obj_spec_provider=obj_spec_provider_from_config(cfg, verbose=True))
    env, *_ = build_rl_env_and_handlers(orch, cfg, str(cfg.env_name))
    h = env.handlers[0]
    DEVICE = env.torch_device
    OBS_DIM, ACTION_DIM = int(h.obs_dim), int(h.action_dim)
    MAX_EP = int(h.MAX_EPISODE_STEPS)
    if hasattr(h, "lift_curriculum_scale"):
        h.lift_curriculum_scale = 0.0              # late-teacher dynamics
    print(f"[distill] obs={OBS_DIM} act={ACTION_DIM} max_ep={MAX_EP}")

    # Priming: initial partial-view capture + teacher obs buffer creation (one obs
    # collection must precede the critic probe calling collect_*_obs).
    env.reset()
    env.step(torch.zeros(h.NWORLD, ACTION_DIM, device=env.torch_device))

    # capability probes (handler generalisation)
    get_teacher_obs = (h.collect_teacher_obs if hasattr(h, "collect_teacher_obs")
                       else (lambda: h.obs_torch))

    # ── 3. teacher policy (critic dim read from the ckpt) ───────────────
    pol_kw = {str(k): v for k, v in OmegaConf.to_container(cfg.policy, resolve=True).items()}
    POL_NAME = str(pol_kw.pop("name"))
    pol_kw.pop("critic_obs_dim", None)
    teacher_ckpt = find_latest_checkpoint(teacher_dir, HAND, TASK, TEACHER_ENV)
    assert teacher_ckpt is not None, f"no teacher ckpt in {teacher_dir}"
    _ck = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
    _sd = _ck["model_state_dict"]
    t_critic_in = int(_sd["critic_trunk.0.weight"].shape[1]) if "critic_trunk.0.weight" in _sd else None
    # The teacher obs dimension is read from the ckpt metadata — even when the student
    # obs is wider (student-only tail such as history), the teacher receives only the
    # leading [0, t_actor_in) slice, **exactly the observation it was trained on**
    # (never edit the teacher obs).
    t_actor_in = int(_ck.get("obs_dim") or OBS_DIM)
    # obs_norm passthrough mask — taken verbatim from the ckpt normalizer buffer (None if absent).
    # Building the policy without the mask makes ``*.passthrough`` an unexpected key at load time.
    _t_pt = _sd.get("obs_normalizer.passthrough", None)
    _t_cpt = _sd.get("critic_obs_normalizer.passthrough", None)
    t_pt_kw = ({"obs_norm_passthrough": _t_pt.bool().cpu().numpy()} if _t_pt is not None else {})
    if _t_cpt is not None:
        t_pt_kw["critic_obs_norm_passthrough"] = _t_cpt.bool().cpu().numpy()
    del _sd, _ck
    assert t_actor_in <= OBS_DIM, f"teacher obs {t_actor_in} > student obs {OBS_DIM}?"
    if t_actor_in != OBS_DIM:
        _gto_raw = get_teacher_obs
        get_teacher_obs = (lambda f=_gto_raw, d=t_actor_in: f()[:, :d])
        print(f"[distill] teacher obs = student obs[:, :{t_actor_in}] "
              f"(student-only tail of {OBS_DIM - t_actor_in} cols is hidden from the teacher)")
    # ── restore the async_v2 teacher time-channel convention ─────────────
    # The v2 handler overwrites ep_step_norm := clamp(ep_step / T1, 0, 1) with
    # T1 = OBS_EP_HORIZON_STEPS (<=0 -> HOLD_STEP + SUCCESS_EARLY_LIFT_HOLD_STEPS),
    # whereas the base kernel of the student env (w_partial_bps) uses
    # ep_step / MAX_EPISODE_STEPS. The env is left untouched; only that one column of
    # the teacher obs is put back on the v2 scale (irrelevant to the student, which
    # zero-masks the column via student_drop_ep_step).
    # teacher.ep_horizon_steps: null/unset -> derived automatically when the teacher env
    # name contains "async_v2", otherwise OFF. Positive -> forced value. 0 -> forced OFF.
    _eh = t.get("ep_horizon_steps", None)
    if hasattr(h, "obs_ep_horizon_steps"):
        # The student env itself is a v2 variant (e.g. w_partial_bps_v2) — the teacher
        # snapshot already carries the T1 scale, so do not overwrite it here.
        print(f"[distill] student env already has the v2 time scale (T1={h.obs_ep_horizon_steps()}) "
              f"— skipping the distill-side ep_step_norm patch")
        _eh = 0
    if _eh is None and "async_v2" in TEACHER_ENV:
        _hs = int(getattr(h, "HOLD_STEP"))
        _el = int(cfg.handler.get("SUCCESS_EARLY_LIFT_HOLD_STEPS", None) or 40)
        _oh = int(cfg.handler.get("OBS_EP_HORIZON_STEPS", None) or 0)
        _eh = _oh if _oh > 0 else _hs + _el
    if _eh and int(_eh) > 0 and getattr(h, "INCLUDE_EP_STAGE_IN_OBS", False):
        _ep_col = int(next(s_ for n_, s_, e_ in h.obs_term_layout if n_ == "ep_stage"))
        import warp as _wp  # noqa: PLC0415
        _ep_step_t = _wp.to_torch(h.ep_step_wp)
        _T1 = float(int(_eh))

        def _patch_ep_norm(f=get_teacher_obs, c=_ep_col, T1=_T1):
            o = f()
            o[:, c] = (_ep_step_t.float() / T1).clamp_(0.0, 1.0)
            return o
        get_teacher_obs = _patch_ep_norm
        print(f"[distill] teacher ep_step_norm := clamp(ep_step/{int(_eh)}, 0, 1) "
              f"(async_v2 convention; env MAX_EPISODE_STEPS={MAX_EP}) col={_ep_col}")

    teacher = make_policy(POL_NAME, obs_dim=t_actor_in, action_dim=ACTION_DIM,
                          **({"critic_obs_dim": t_critic_in} if t_critic_in else {}),
                          **t_pt_kw, **pol_kw).to(DEVICE).eval()
    load_checkpoint(teacher_ckpt, policy=teacher,
                    expected_obs_dim=t_actor_in, expected_action_dim=ACTION_DIM,
                    map_location=DEVICE)
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    print(f"[distill] teacher loaded: {teacher_ckpt.name} (critic_in={t_critic_in})")

    # ── 4. student policy (optional asym critic + filtered warm-start) ──
    rl = dcfg.rl_finetune
    asym = str(rl.get("critic", "asym")) == "asym" and hasattr(h, "collect_critic_obs")
    CRITIC_DIM = int(h.collect_critic_obs().shape[1]) if asym else OBS_DIM
    get_critic_obs = (h.collect_critic_obs if asym else (lambda: h.obs_torch))

    # student actor passthrough = handler-derived mask (same path as the evaluate.py builder).
    # The critic gets no mask (evaluate.py does not inject a critic passthrough for an
    # async_ppo=false snapshot, so the buffer layout must match).
    _pt_mask_full = (h.obs_norm_passthrough_mask()
                     if (bool(pol_kw.get("obs_norm", False))
                         and hasattr(h, "obs_norm_passthrough_mask")) else None)

    # ── drop the hist block from the actor input (opt-in) ─────────────────
    #   student.actor_drop_hist: true -> **the actor does not see student_hist**;
    #   history is used only as input to the re-grasp head. Rationale: a longer history
    #   did not help the actor measurably, while the head needs the extra frames — the
    #   hist block is redundant for the actor. The actor obs then has the same dimension
    #   as the teacher obs.
    _tl_all = {nm: (a_, b_) for nm, a_, b_ in h.obs_term_layout}
    ACTOR_DROP_HIST = bool(s.get("actor_drop_hist", False)) and "student_hist" in _tl_all
    if ACTOR_DROP_HIST:
        _h0, _h1 = _tl_all["student_hist"]
        ACTOR_COLS = torch.as_tensor([c for c in range(OBS_DIM) if not (_h0 <= c < _h1)],
                                     dtype=torch.long, device=DEVICE)
        ACTOR_DIM = int(ACTOR_COLS.numel())
        print(f"[distill] actor hist drop ON — actor obs {OBS_DIM} → {ACTOR_DIM} "
              f"(student_hist {_h1-_h0} cols go to the re-grasp head only)")
    else:
        ACTOR_COLS, ACTOR_DIM = None, OBS_DIM

    def A(o):
        """obs -> actor input (applies the optional hist drop)."""
        return o if ACTOR_COLS is None else o.index_select(1, ACTOR_COLS)

    s_pt_kw = ({} if _pt_mask_full is None else
               {"obs_norm_passthrough": (_pt_mask_full if ACTOR_COLS is None
                                         else _pt_mask_full[ACTOR_COLS.cpu()])})

    def make_student():
        return make_policy(POL_NAME, obs_dim=ACTOR_DIM, action_dim=ACTION_DIM,
                           critic_obs_dim=CRITIC_DIM, **s_pt_kw, **pol_kw).to(DEVICE)

    student = make_student().train()
    ssd = student.state_dict()
    tsd = {k: v for k, v in teacher.state_dict().items()
           if k in ssd and ssd[k].shape == v.shape}
    student.load_state_dict({**ssd, **tsd})
    print(f"[distill] student warm-start {len(tsd)}/{len(ssd)} tensors "
          f"(critic={'asym' if asym else 'sym'} {CRITIC_DIM}-D fresh)")

    # ── (optional) initialise the student from an existing DAgger ckpt — overrides the
    # teacher warm-start. Combined with distill.n_iters=0 this goes straight to RL
    # without re-running DAgger.
    init_ck = s.get("init_ckpt", None)
    if init_ck:
        ip = Path(str(init_ck))
        ckpt_path = (find_latest_checkpoint(ip, HAND, TASK, str(cfg.env_name))
                     if ip.is_dir() else ip)
        assert ckpt_path is not None and Path(ckpt_path).is_file(), \
            f"student.init_ckpt not found: {init_ck}"
        load_checkpoint(ckpt_path, policy=student,
                        expected_obs_dim=ACTOR_DIM, expected_action_dim=ACTION_DIM,
                        map_location=DEVICE)
        print(f"[distill] student init_ckpt loaded: {Path(ckpt_path).name}")

    # ── wandb ────────────────────────────────────────────────────────────
    wb = None
    if bool(dcfg.wandb.get("enabled", False)):
        import wandb as wb  # type: ignore
        wb.init(project=str(dcfg.wandb.project), entity=str(dcfg.wandb.entity),
                name=str(dcfg.wandb.get("name", f"{HAND}_distill_{dcfg.output.run_tag}")),
                config=OmegaConf.to_container(dcfg, resolve=True))

    SUCCESS_CODE = reason_code("success")
    CRUSH_CODE = reason_code("hand_object_touch")

    # ── DR curriculum setup — capture targets, then run autocal with DR OFF (clean fit) ─
    _dr_t = {"skip":  float(getattr(h, "_hist_skip_prob", 0.0)),
             "noise": float(getattr(h, "_hist_noise_std", 0.0)),
             "delay": float(getattr(h, "_act_delay_prob", 0.0))}
    DR_RAMP = float(dcfg.distill.get("dr_ramp_frac", 0.0))

    def set_dr(frac):
        if int(getattr(h, "_hist_k", 0)) > 0:
            h._hist_skip_prob = _dr_t["skip"] * frac
            h._hist_noise_std = _dr_t["noise"] * frac
        h._act_delay_prob = _dr_t["delay"] * frac

    if DR_RAMP > 0.0:
        set_dr(0.0)
        print(f"[distill] DR curriculum ON — over the first {DR_RAMP:.0%} of n_iters, linear ramp 0→"
              f"(skip {_dr_t['skip']:.2f}, noise {_dr_t['noise']:.3f}, "
              f"delay {_dr_t['delay']:.2f}) (autocal runs with DR OFF)")

    # ── 4.5 stage threshold auto-calibration (demonstration-based) ──────
    ac_cfg = dict(dcfg.get("stage_autocal", {}) or {})
    if bool(ac_cfg.get("enabled", False)):
        cal = autocal_stage_thresholds(h, env, teacher, get_teacher_obs, ACTION_DIM, ac_cfg)
        if cal is not None:
            d_star, k_star, L_star = cal
            h._stage_hc_dist = float(d_star)
            h._stage_near_ticks = int(k_star)
            if L_star is not None:                 # proxy replaced only for tracking-free
                h._stage_lift_m = float(L_star)
            # Persist the calibrated thresholds in the config snapshot so that
            # evaluate.py / eval_success.py rebuild the SAME stage machine from
            # config.yaml (the handler reads cfg.Training.stage_* at setup).
            cfg.Training.stage_hc_dist_m = float(d_star)
            cfg.Training.stage_near_ticks = int(k_star)
            if L_star is not None:
                cfg.Training.stage_lift_frac = float(L_star) / float(h.LIFT_TARGET_M)
            if wb is not None:
                wb.log({"autocal/dist_m": d_star, "autocal/near_ticks": k_star,
                        **({"autocal/lift_proxy_m": L_star} if L_star is not None else {})})

    # ── 5. DAgger (per-episode driver curriculum) ───────────────────────
    d = dcfg.distill
    N_ITERS = int(d.n_iters); LOG_EVERY = int(d.log_every)
    BETA_S, BETA_E, P_BLEND = float(d.beta_start), float(d.beta_end), float(d.p_blend)

    # ── privileged reconstruction auxiliary loss (supervised obj_pose_diff) ──
    # An aux head predicts the GT obj_pose_diff of the teacher obs from the student trunk
    # feature — representation-learning pressure to "estimate the object state from the
    # observation". Removed at deployment.
    AUX_COEF = float(d.get("aux_objpd_coef", 0.0))
    aux_head, AUX_RANGES, AUX_W = None, [], 0
    if AUX_COEF > 0.0:
        _tl = {nm: (s, e) for nm, s, e in h.obs_term_layout}
        _aux_terms = [t.strip() for t in
                      str(d.get("aux_terms", "obj_pose_diff")).split(",") if t.strip()]
        AUX_RANGES = [tuple(map(int, _tl[t])) for t in _aux_terms]
        AUX_W = sum(b - a for a, b in AUX_RANGES)
        with torch.no_grad():
            feat_dim = int(student.actor_trunk(
                torch.zeros(2, ACTOR_DIM, device=DEVICE)).shape[1])
        aux_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, AUX_W)).to(DEVICE)
        print(f"[distill] aux privileged-reconstruction head ON — feat {feat_dim} → "
              f"{AUX_W} ({', '.join(_aux_terms)}), coef {AUX_COEF}")

    def aux_tgt(tobs):
        # extract and concatenate the reconstruction target slices from the (unmasked GT) teacher obs
        return torch.cat([tobs[:, a:b] for a, b in AUX_RANGES], dim=1)

    def aux_loss_on(obs_mb, tgt_mb):
        return F.mse_loss(aux_head(student.actor_trunk(A(obs_mb))), tgt_mb)

    # ── re-grasp signal head (blind env only) ────────────────────────────
    # Input = raw proprioceptive slices only: joint_qpos + torque_proxy (current) +
    # the student_hist block ([qpos, torque_proxy, last action] x K). Independent of the
    # trunk — detached and reused as is at deployment. Label = the env's GT "object
    # dropped" signal (regrasp_label_torch).
    RG_COEF = float(d.get("aux_regrasp_coef", 0.0))
    rg_head, RG_COLS = None, None
    RG_SRC_TRAIN = str(d.get("regrasp_train_source", "gt"))
    if RG_COEF > 0.0 and hasattr(h, "regrasp_label_torch"):
        _tl = {nm: (s, e) for nm, s, e in h.obs_term_layout}
        _rg_terms = [t.strip() for t in str(d.get(
            "regrasp_input_terms", "joint_qpos,torque_proxy,student_hist")).split(",")
            if t.strip() and t.strip() in _tl]
        _cols = [c for t in _rg_terms for c in range(*_tl[t])]
        RG_COLS = torch.as_tensor(_cols, dtype=torch.long, device=DEVICE)
        rg_head = torch.nn.Sequential(
            torch.nn.Linear(len(_cols), 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 1)).to(DEVICE)
        RG_POSW = torch.tensor(float(d.get("aux_regrasp_pos_weight", 3.0)), device=DEVICE)
        h.set_regrasp_source(RG_SRC_TRAIN)
        _ick = s.get("init_ckpt", None)
        if _ick and (Path(str(_ick)) / "regrasp_head.pt").is_file():
            _hd = torch.load(Path(str(_ick)) / "regrasp_head.pt", map_location=DEVICE, weights_only=False)
            assert list(_hd["input_cols"]) == _cols, "regrasp_head input column mismatch (check the env/hist settings)"
            rg_head.load_state_dict(_hd["state_dict"])
            print(f"[distill] re-grasp head loaded: {Path(str(_ick)) / 'regrasp_head.pt'}")
        print(f"[distill] re-grasp head ON — in {len(_cols)} ({', '.join(_rg_terms)}) "
              f"→ 1 logit, coef {RG_COEF}, pos_weight {float(RG_POSW):g}, "
              f"train stage-reset source={RG_SRC_TRAIN}, "
              f"consec={int(getattr(h, 'REGRASP_PRED_CONSEC', 0))}")

    def rg_logit(obs_b):
        return rg_head(obs_b[:, RG_COLS]).squeeze(1)

    def rg_loss_on(obs_b, lab_b):
        return F.binary_cross_entropy_with_logits(rg_logit(obs_b), lab_b, pos_weight=RG_POSW)

    _train_params = (list(student.parameters())
                     + (list(aux_head.parameters()) if aux_head is not None else [])
                     + (list(rg_head.parameters()) if rg_head is not None else []))

    buffer = ReplayBuffer(int(d.buffer_cap), OBS_DIM, ACTION_DIM, DEVICE,
                          aux_dim=AUX_W, with_lab=rg_head is not None)
    opt = torch.optim.Adam(_train_params, lr=float(d.lr))
    loss_kind = str(d.get("loss", "kl"))
    # KL-based adaptive lr — DAgger has no PPO ratio, so the control signal is the
    # update-KL: KL(old‖new) of the student distribution before/after the grad step on
    # the same minibatch. Equivalent in meaning to approx_kl ("how far did the policy
    # move in one iteration"). Adjusted once per iteration (same rationale as ppo.py).
    d_alr = read_adaptive_lr_cfg(d)
    if d_alr["desired_kl"]:
        print(f"[distill] adaptive lr ON — desired_kl={d_alr['desired_kl']} "
              f"lr∈[{d_alr['lr_min']:.1e},{d_alr['lr_max']:.1e}] ×÷{d_alr['factor']}")

    # ── critic pre-fit (critic_prefit) — DAgger trains only the actor, so the critic is
    # essentially random when RL starts -> noisy initial advantages. Since DAgger already
    # steps the env once per iteration, accumulate (obs, critic_obs, rew, done) segments
    # and regress only the critic on GAE returns (separate optimizer from the actor — no
    # interference with the DAgger distribution matching). The target is the value of the
    # mixed teacher/blend driver; the RL-side critic_warmup refines it on-policy.
    PREFIT = bool(d.get("critic_prefit", False))
    if PREFIT:
        PF_LEN = int(d.get("critic_prefit_len", 32))
        PF_EPOCHS = int(d.get("critic_prefit_epochs", 1))
        _critic_params = [p for n, p in student.named_parameters()
                          if n.startswith("critic")]
        pf_opt = torch.optim.Adam(_critic_params,
                                  lr=float(d.get("critic_prefit_lr", 1.0e-3)))
        pf_obs = torch.zeros(PF_LEN, h.NWORLD, OBS_DIM, device=DEVICE)
        pf_cobs = torch.zeros(PF_LEN, h.NWORLD, CRITIC_DIM, device=DEVICE)
        pf_rew = torch.zeros(PF_LEN, h.NWORLD, device=DEVICE)
        pf_done = torch.zeros(PF_LEN, h.NWORLD, device=DEVICE)
        pf_t = 0; pf_vloss = float("nan")
        print(f"[distill] critic prefit ON — seg {PF_LEN} it, {PF_EPOCHS} epoch, "
              f"lr {float(d.get('critic_prefit_lr', 1.0e-3)):.1e} "
              f"({len(_critic_params)} critic tensors, γ/λ from rl_finetune)")

    def distill_loss(s_out, loc_t, scale_t):
        loss = s_out["loc"].new_zeros(())
        if loss_kind in ("mse_loc", "mse+kl"):
            loss = loss + F.mse_loss(s_out["loc"], loc_t)
        if loss_kind in ("kl", "mse+kl"):
            loss = loss + gaussian_kl(loc_t, scale_t, s_out["loc"], s_out["scale"]).mean()
        return loss

    env.reset()
    env.step(torch.zeros(h.NWORLD, ACTION_DIM, device=DEVICE))
    world_mode = torch.full((h.NWORLD,), 1, device=DEVICE, dtype=torch.long)
    _r0 = torch.rand(h.NWORLD, device=DEVICE)
    world_mode[_r0 < BETA_S * (1.0 - P_BLEND)] = 0
    world_mode[_r0 >= 1.0 - P_BLEND] = 2

    run_loss = run_l1 = run_kl = 0.0
    rg_tp = rg_np = rg_nl = rg_fire = 0; last_rg = 0.0
    cur_lr = float(d.lr)
    ep_done = ep_succ = 0
    t0 = time.perf_counter()
    for it in range(1, N_ITERS + 1):
        beta = BETA_S + (BETA_E - BETA_S) * (it - 1) / max(N_ITERS - 1, 1)
        if DR_RAMP > 0.0:
            set_dr(min(1.0, it / max(DR_RAMP * N_ITERS, 1.0)))
        student_obs = h.obs_torch
        teacher_obs = get_teacher_obs()
        with torch.no_grad():
            t_out = teacher.act(teacher_obs, deterministic=True, inference_only=True)
        buffer.push(student_obs.detach(), t_out["loc"].detach(), t_out["scale"].detach(),
                    aux=(aux_tgt(teacher_obs).detach()
                         if aux_head is not None else None),
                    lab=(h.regrasp_label_torch.detach().clone()
                         if rg_head is not None else None))
        if rg_head is not None:             # inject the prediction (drives the rewind when source=pred/both) + metrics
            with torch.no_grad():
                _pred = torch.sigmoid(rg_logit(student_obs)) > 0.5
                _lab = h.regrasp_label_torch > 0.5
                h.regrasp_pred_torch.copy_(_pred.float())
                rg_tp += int((_pred & _lab).sum()); rg_np += int(_pred.sum())
                rg_nl += int(_lab.sum()); rg_fire += int(h.regrasp_pred_fired_torch.sum())
        if PREFIT:                          # capture the pre-step state (indexed copy)
            pf_obs[pf_t] = student_obs.detach()
            pf_cobs[pf_t] = get_critic_obs().detach()
            if student.critic_obs_normalizer is not None:   # stands in for the rollout-side statistics update
                student.critic_obs_normalizer(pf_cobs[pf_t], update=True)

        last_loss = last_aux = 0.0
        upd_kl_sum = 0.0; upd_kl_n = 0
        for _ in range(int(d.grad_steps_per_iter)):
            obs_b, loc_b, scale_b, aux_b, lab_b = buffer.sample(int(d.batch_size))
            s_out = student.act(A(obs_b), deterministic=True, inference_only=True)
            loss = distill_loss(s_out, loc_b, scale_b)
            if aux_head is not None:
                al = aux_loss_on(obs_b, aux_b)
                loss = loss + AUX_COEF * al
                last_aux = float(al.item())
            if rg_head is not None:
                rl_ = rg_loss_on(obs_b, lab_b)
                loss = loss + RG_COEF * rl_
                last_rg = float(rl_.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_train_params, 1.0)
            opt.step()
            last_loss = float(loss.item())
            if d_alr["desired_kl"]:
                with torch.no_grad():
                    s_new = student.act(A(obs_b), deterministic=True, inference_only=True)
                    upd_kl_sum += float(gaussian_kl(
                        s_out["loc"].detach(), s_out["scale"].detach(),
                        s_new["loc"], s_new["scale"]).mean())
                    upd_kl_n += 1
        cur_lr = _adapt_lr(opt, upd_kl_sum / max(upd_kl_n, 1),
                           desired_kl=d_alr["desired_kl"], lr_min=d_alr["lr_min"],
                           lr_max=d_alr["lr_max"], factor=d_alr["factor"])

        with torch.no_grad():
            a_teacher = t_out["action"]
            a_student = student.act(A(student_obs), deterministic=False,
                                    inference_only=True)["action"]
            coin = torch.rand(h.NWORLD, 1, device=DEVICE) < 0.5
            is_t = (world_mode == 0).unsqueeze(1)
            is_b = (world_mode == 2).unsqueeze(1)
            drive = torch.where(is_t | (is_b & coin), a_teacher, a_student)
            s_now = student.act(A(student_obs), deterministic=True, inference_only=True)
            act_l1 = (torch.tanh(s_now["loc"]) - torch.tanh(t_out["loc"])).abs().mean().item()
        env.step(drive.detach())
        done_mask = h.done_mask_torch.bool()
        if PREFIT:                          # capture the transition (rew, done) before the reset
            pf_rew[pf_t] = h.reward_torch.detach()
            pf_done[pf_t] = h.done_torch.float().detach()
        if bool(done_mask.any()):
            ep_done += int(done_mask.sum())
            ep_succ += int((h.done_reason_torch[done_mask] == SUCCESS_CODE).sum())
        env.per_world_reset_if_done()
        if bool(done_mask.any()):
            n_new = int(done_mask.sum())
            r = torch.rand(n_new, device=DEVICE)
            nm = torch.full((n_new,), 1, device=DEVICE, dtype=torch.long)
            nm[r < beta * (1.0 - P_BLEND)] = 0
            nm[r >= 1.0 - P_BLEND] = 2
            world_mode[done_mask] = nm
        if PREFIT:
            pf_t += 1
            if pf_t == PF_LEN:              # segment complete -> regress the critic on GAE returns
                with torch.no_grad():
                    seg_v = student._value(
                        pf_obs.reshape(-1, OBS_DIM),
                        pf_cobs.reshape(-1, CRITIC_DIM)).reshape(PF_LEN, h.NWORLD)
                    last_v = student._value(h.obs_torch.detach(), get_critic_obs())
                    _, pf_ret = compute_gae(pf_rew, seg_v, pf_done, last_v,
                                            float(rl.gamma), float(rl.gae_lambda))
                b_o = pf_obs.reshape(-1, OBS_DIM)
                b_c = pf_cobs.reshape(-1, CRITIC_DIM)
                b_r = pf_ret.reshape(-1)
                vl_acc = 0.0; n_vb = 0
                for _ in range(PF_EPOCHS):
                    perm = torch.randperm(b_o.shape[0], device=DEVICE)
                    for s0 in range(0, b_o.shape[0], int(d.batch_size)):
                        mb = perm[s0:s0 + int(d.batch_size)]
                        vloss = F.mse_loss(student._value(b_o[mb], b_c[mb]), b_r[mb])
                        pf_opt.zero_grad(set_to_none=True)
                        vloss.backward()
                        torch.nn.utils.clip_grad_norm_(_critic_params, 1.0)
                        pf_opt.step()
                        vl_acc += float(vloss.item()); n_vb += 1
                pf_vloss = vl_acc / max(n_vb, 1)
                pf_t = 0

        run_loss += last_loss; run_l1 += act_l1
        run_kl += upd_kl_sum / max(upd_kl_n, 1)
        if it % LOG_EVERY == 0:
            succ = (ep_succ / ep_done) if ep_done else float("nan")
            sps = LOG_EVERY * h.NWORLD / max(time.perf_counter() - t0, 1e-9)
            lr_str = (f" kl={run_kl/LOG_EVERY:.4f} lr={cur_lr:.1e}"
                      if d_alr["desired_kl"] else "")
            pf_str = f" vpre={pf_vloss:.3f}" if PREFIT else ""
            rg_str = ""
            if rg_head is not None:
                rg_str = (f" rg[bce={last_rg:.3f} P={rg_tp/max(rg_np,1):.2f} "
                          f"R={rg_tp/max(rg_nl,1):.2f} lab={rg_nl/(LOG_EVERY*h.NWORLD):.3f} "
                          f"fired={rg_fire}]")
            print(f"[dagger {it:5d}/{N_ITERS}] loss={run_loss/LOG_EVERY:.4f} "
                  f"act_L1={run_l1/LOG_EVERY:.4f} succ={succ:.3f} "
                  f"β={beta:.2f} buf={buffer.n} sps={sps:,.0f}{lr_str}{pf_str}{rg_str}")
            if wb is not None:
                wb.log({"distill/loss": run_loss / LOG_EVERY,
                        "distill/aux_objpd_mse": last_aux,
                        "distill/act_l1": run_l1 / LOG_EVERY,
                        "distill/succ": succ, "distill/beta": beta,
                        "distill/update_kl": run_kl / LOG_EVERY,
                        "distill/lr": cur_lr,
                        **({"distill/critic_vloss": pf_vloss} if PREFIT else {}),
                        **({"distill/rg_bce": last_rg,
                            "distill/rg_prec": rg_tp / max(rg_np, 1),
                            "distill/rg_rec": rg_tp / max(rg_nl, 1),
                            "distill/rg_label_rate": rg_nl / (LOG_EVERY * h.NWORLD),
                            "distill/rg_fired": rg_fire} if rg_head is not None else {}),
                        "distill/it": it})
            run_loss = run_l1 = run_kl = 0.0
            rg_tp = rg_np = rg_nl = rg_fire = 0
            ep_done = ep_succ = 0
            t0 = time.perf_counter()

    dagger_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
    if PREFIT:                              # release the segment buffers (avoid coexisting with the RL buffers)
        del pf_obs, pf_cobs, pf_rew, pf_done, pf_opt
        torch.cuda.empty_cache()

    # ── 6. evaluation (same as training eval: run_eval streak criterion) ─
    EVAL_LEN = int(dcfg.eval.get("length") or MAX_EP)

    def eval3(tag):
        rt = run_eval(ObsSwapActor(teacher, get_teacher_obs), h, eval_length=EVAL_LEN)
        _sa = ObsSwapActor(student); _sa.actor_cols = ACTOR_COLS
        rs = run_eval(_sa, h, eval_length=EVAL_LEN)
        for nm, r in [("teacher", rt), (tag, rs)]:
            print(f"[eval] {nm:14s} strict={r['success_rate_strict']:.3f} "
                  f"lift={r.get('success_rate_lift', float('nan')):.3f} "
                  f"return={r['mean_return']:.2f}")
        if wb is not None:
            wb.log({f"eval/{tag}_strict": rs["success_rate_strict"],
                    f"eval/{tag}_lift": rs.get("success_rate_lift", float("nan")),
                    f"eval/{tag}_return": rs["mean_return"],
                    "eval/teacher_strict": rt["success_rate_strict"],
                    "eval/teacher_lift": rt.get("success_rate_lift", float("nan"))})
        if rg_head is not None:
            # deployment mode: drive the stage rewind **only from the predicted signal (consec run)**
            h.set_regrasp_source("pred")
            ra = RegraspPredActor(student, rg_head, RG_COLS, h)
            ra.actor_cols = ACTOR_COLS
            rp = run_eval(ra, h, eval_length=EVAL_LEN)
            h.set_regrasp_source(RG_SRC_TRAIN)
            st = ra.stats()
            print(f"[eval] {tag + '+predRG':14s} strict={rp['success_rate_strict']:.3f} "
                  f"lift={rp.get('success_rate_lift', float('nan')):.3f} "
                  f"return={rp['mean_return']:.2f}  "
                  f"(head P={st['prec']:.2f} R={st['rec']:.2f} lab={st['lab_rate']:.3f} "
                  f"fired={st['fired']})")
            if wb is not None:
                wb.log({f"eval/{tag}_predrg_strict": rp["success_rate_strict"],
                        f"eval/{tag}_predrg_lift": rp.get("success_rate_lift", float("nan")),
                        f"eval/{tag}_rg_prec": st["prec"], f"eval/{tag}_rg_rec": st["rec"],
                        f"eval/{tag}_rg_fired": st["fired"]})
        return rs

    set_dr(1.0)                      # all later eval/RL runs with full DR (representative of deployment)
    res_dagger = eval3("dagger")

    # ── 7. hybrid RL (PPO + teacher-KL anchor) ──────────────────────────
    if bool(rl.get("enabled", True)):
        RL_ITERS = int(rl.iters); ROLLOUT = int(rl.rollout_len)
        # The re-grasp head is frozen during RL by default — do not perturb the DAgger-fitted
        # predictor with the RL on-policy distribution (sparse / biased labels). false -> keep training.
        RG_FREEZE = bool(rl.get("freeze_regrasp_head", True)) and rg_head is not None
        # ── head-only mode: freeze the policy (+aux) and update only the re-grasp head on
        #   the RL distribution. Rationale: post-RL does not improve strict success across
        #   hands (perturbing the policy brings no gain), whereas the head's low precision
        #   (many false positives) is the main source of loss in deployment mode (predRG).
        #   Leaving the policy untouched avoids that failure pattern while fixing the detector.
        #   Note: if freeze_regrasp_head is also true there is nothing left to train.
        POLICY_FREEZE = bool(rl.get("freeze_policy", False))
        assert not (POLICY_FREEZE and RG_FREEZE), \
            "freeze_policy and freeze_regrasp_head are both true — no parameters left to train"
        _rl_params = (([] if POLICY_FREEZE else list(student.parameters()))
                      + ([] if (aux_head is None or POLICY_FREEZE) else list(aux_head.parameters()))
                      + ([] if (rg_head is None or RG_FREEZE) else list(rg_head.parameters())))
        if POLICY_FREEZE:
            for p_ in student.parameters():
                p_.requires_grad_(False)
            if aux_head is not None:
                for p_ in aux_head.parameters():
                    p_.requires_grad_(False)
            student.eval()
            print("[rl] POLICY FROZEN — training the re-grasp head only (head-only post-RL)")
        if RG_FREEZE:
            for p_ in rg_head.parameters():
                p_.requires_grad_(False)
            rg_head.eval()
            print("[rl] re-grasp head FROZEN (excluded from the optimizer and the BCE loss)")
        rl_opt = torch.optim.Adam(_rl_params, lr=float(rl.lr))
        # ── reward normalization (same RewardNormalizer as train.py; opt-in) ──
        # OFF: the critic regresses raw returns (tens) directly -> large vloss, and in the joint
        # actor+critic clip_grad_norm_ the critic grad eats into the actor step.
        # ON: divide by the running std of the discounted return (mode=return, clip 10) -> same
        # scale as the teacher PPO.
        _rnc = dict(rl.get("reward_norm", {}) or {})
        rew_norm = None
        if bool(_rnc.get("enabled", False)):
            from grit.training.reward_normalizer import RewardNormalizer  # noqa: PLC0415
            rew_norm = RewardNormalizer(int(h.NWORLD), DEVICE, gamma=float(rl.gamma),
                                        mode=str(_rnc.get("mode", "return")),
                                        center=bool(_rnc.get("center", False)),
                                        clip=float(_rnc.get("clip", 10.0)))
            print(f"[rl] reward norm ON (mode={rew_norm.mode} clip={rew_norm.clip} γ={float(rl.gamma)})")
        # ── focused exploration of tail failures (all opt-in, OFF by default) ─
        # (A) fail_bank : bank the state s_{t-τ} preceding a failure/retry and resume a
        #                 fraction of the reset worlds from it
        # (A') fail_tail: up-weight (>1) the PPO loss of the τ steps preceding a failure event
        # (D) rank_adv  : terminal bonus coef·(2q−1) from the quantile q of the successful
        #                 episode's return
        _fb = dict(rl.get("fail_bank", {}) or {})
        fail_bank = None
        if bool(_fb.get("enabled", False)):
            from grit.training.failure_bank import FailureStateBank  # noqa: PLC0415
            fail_bank = FailureStateBank(
                h, tau=int(_fb.get("tau", 15)), cap=int(_fb.get("cap", 4096)),
                p_restore=float(_fb.get("p_restore", 0.3)),
                min_bank=int(_fb.get("min_bank", 256)), device=DEVICE)
        _ft = dict(rl.get("fail_tail", {}) or {})
        FT_TAU, FT_W = int(_ft.get("tau", 0)), float(_ft.get("weight", 1.0))
        _ra = dict(rl.get("rank_adv", {}) or {})
        RANK_COEF = float(_ra.get("coef", 0.0))
        RANK_BANK = torch.zeros(int(_ra.get("bank", 8192)), device=DEVICE); rank_n = 0; rank_ptr = 0
        RANK_MIN = int(_ra.get("min_bank", 256))
        ep_ret = torch.zeros(int(h.NWORLD), device=DEVICE)
        FAIL_REASONS = torch.tensor([c for c in SubEnvHandler.REASON_NAME
                                     if c not in (0, SUCCESS_CODE, 2)], device=DEVICE)
        if FT_TAU > 0 or RANK_COEF > 0.0 or fail_bank is not None:
            print(f"[rl] tail-failure exploration: fail_bank={'ON' if fail_bank else 'OFF'} "
                  f"fail_tail(τ={FT_TAU},w={FT_W}) rank_adv(coef={RANK_COEF})")
        # KL-based adaptive lr — same as ppo.py: adjusted once per iteration from the
        # iteration mean of the minibatch approx_kl (k3).
        rl_alr = read_adaptive_lr_cfg(rl)
        rl_lr = float(rl.lr)
        if rl_alr["desired_kl"]:
            print(f"[rl] adaptive lr ON — desired_kl={rl_alr['desired_kl']} "
                  f"lr∈[{rl_alr['lr_min']:.1e},{rl_alr['lr_max']:.1e}] ×÷{rl_alr['factor']}")
        # critic warm-up — the first N iterations freeze the policy (loss = vf only) so that
        # the under-fitted critic right after DAgger does not push the actor with noisy
        # advantages. Warm-up iterations are **added** in front of rl.iters; bc_kl annealing
        # and adaptive lr follow their usual schedule once the warm-up ends.
        WARMUP = int(rl.get("critic_warmup_iters", 0))
        if WARMUP > 0:
            print(f"[rl] critic warm-up ON — first {WARMUP} iters are vf-only (policy frozen)")
        N = h.NWORLD
        env.reset(); env.step(torch.zeros(N, ACTION_DIM, device=DEVICE))
        student.train()
        for it in range(1, WARMUP + RL_ITERS + 1):
            warm = it <= WARMUP
            obs_buf = torch.zeros(ROLLOUT, N, OBS_DIM, device=DEVICE)
            cobs_buf = torch.zeros(ROLLOUT, N, CRITIC_DIM, device=DEVICE)
            pth_buf = torch.zeros(ROLLOUT, N, ACTION_DIM, device=DEVICE)
            logp_buf = torch.zeros(ROLLOUT, N, device=DEVICE)
            val_buf = torch.zeros(ROLLOUT, N, device=DEVICE)
            rew_buf = torch.zeros(ROLLOUT, N, device=DEVICE)
            done_buf = torch.zeros(ROLLOUT, N, device=DEVICE)
            tloc_buf = torch.zeros(ROLLOUT, N, ACTION_DIM, device=DEVICE)
            tscl_buf = torch.zeros(ROLLOUT, N, ACTION_DIM, device=DEVICE)
            aux_buf = (torch.zeros(ROLLOUT, N, AUX_W, device=DEVICE)
                       if aux_head is not None else None)
            lab_buf = (torch.zeros(ROLLOUT, N, device=DEVICE)
                       if rg_head is not None else None)
            fail_ev_buf = torch.zeros(ROLLOUT, N, device=DEVICE)
            ep_done = ep_succ = crush = 0
            touchF = 0.0
            tblP = 0.0; tblA = 0.0   # table penalty / number of fingers in contact (table-min monitor)
            for tt in range(ROLLOUT):
                # NOTE: obs_torch / critic obs are **views** of warp buffers that env.step()
                #   overwrites in place, so they must be cloned before the step. Copying after
                #   the step stored obs_{t+1} and made PPO train on (obs_{t+1}, a_t, logp_t),
                #   giving a large k3 even with a frozen actor.
                obs = h.obs_torch.detach().clone()
                # The critic obs is not covered by the env's _sanitize_obs (separate buffer) —
                # non-finite velocity channels right after a restore/teleport would make the loss
                # and the whole grad NaN, so sanitise here.
                cobs = torch.nan_to_num(get_critic_obs().clone(), nan=0.0, posinf=0.0, neginf=0.0)
                with torch.no_grad():
                    out = student.act(A(obs), deterministic=False, critic_obs=cobs)
                    tob = get_teacher_obs()
                    t_rl = teacher.act(tob, deterministic=True, inference_only=True)
                if aux_buf is not None:
                    aux_buf[tt] = aux_tgt(tob)
                if lab_buf is not None:
                    lab_buf[tt] = h.regrasp_label_torch      # (copy — indexed assignment)
                    with torch.no_grad():
                        h.regrasp_pred_torch.copy_((torch.sigmoid(rg_logit(obs)) > 0.5).float())
                if fail_bank is not None:
                    fail_bank.tick()
                env.step(out["action"].detach())
                obs_buf[tt] = obs; cobs_buf[tt] = cobs
                pth_buf[tt] = out["pre_tanh"]; logp_buf[tt] = out["log_prob"]
                val_buf[tt] = out["value"]
                done_buf[tt] = h.done_torch.float().detach()
                rew_buf[tt] = (rew_norm(h.reward_torch.detach(), done_buf[tt]) if rew_norm is not None
                               else h.reward_torch.detach())
                tloc_buf[tt] = t_rl["loc"]; tscl_buf[tt] = t_rl["scale"]
                dm = h.done_mask_torch.bool()
                # failure event = failure terminal (excluding success/timeout) ∪ re-grasp retry fired
                fail_ev = dm & torch.isin(h.done_reason_torch, FAIL_REASONS)
                if hasattr(h, "retry_fired_torch"):
                    fail_ev = fail_ev | (h.retry_fired_torch > 0.5)
                fail_ev_buf[tt] = fail_ev.float()
                if fail_bank is not None:
                    fail_bank.push_failures(fail_ev, dm)
                if RANK_COEF > 0.0:                    # (D) terminal bonus from the success-return quantile
                    ep_ret += rew_buf[tt]
                    sd_ = dm & (h.done_reason_torch == SUCCESS_CODE)
                    if bool(sd_.any()):
                        r_ = ep_ret[sd_]
                        if rank_n >= RANK_MIN:
                            q = (RANK_BANK[:rank_n].unsqueeze(0) < r_.unsqueeze(1)).float().mean(1)
                            rew_buf[tt][sd_] += RANK_COEF * (2.0 * q - 1.0)
                        k_ = int(r_.numel()); di = (torch.arange(k_, device=DEVICE) + rank_ptr) % RANK_BANK.numel()
                        RANK_BANK[di] = r_; rank_ptr = (rank_ptr + k_) % RANK_BANK.numel()
                        rank_n = min(rank_n + k_, RANK_BANK.numel())
                    ep_ret[dm] = 0.0
                if bool(dm.any()):
                    ep_done += int(dm.sum())
                    ep_succ += int((h.done_reason_torch[dm] == SUCCESS_CODE).sum())
                    crush += int((h.done_reason_torch[dm] == CRUSH_CODE).sum())
                if hasattr(h, "obj_touch_force_torch"):
                    touchF += float(h.obj_touch_force_torch.mean())
                if hasattr(h, "r_table_pen_torch"):
                    tblP += float(h.r_table_pen_torch.mean())
                if hasattr(h, "tbl_active_torch"):
                    tblA += float(h.tbl_active_torch.float().mean())
                env.per_world_reset_if_done()
                if fail_bank is not None:
                    fail_bank.restore(dm)
            with torch.no_grad():
                last_val = student.act(A(h.obs_torch.detach()), deterministic=True,
                                       critic_obs=torch.nan_to_num(get_critic_obs(), nan=0.0,
                                                                   posinf=0.0, neginf=0.0))["value"]
            adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_val,
                                   float(rl.gamma), float(rl.gae_lambda))
            b_obs = obs_buf.reshape(-1, OBS_DIM)
            b_cobs = cobs_buf.reshape(-1, CRITIC_DIM)
            b_pth = pth_buf.reshape(-1, ACTION_DIM)
            b_logp = logp_buf.reshape(-1)
            b_adv = adv.reshape(-1); b_ret = ret.reshape(-1)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            b_tloc = tloc_buf.reshape(-1, ACTION_DIM)
            b_tscl = tscl_buf.reshape(-1, ACTION_DIM)
            b_aux = (aux_buf.reshape(-1, AUX_W)
                     if aux_buf is not None else None)
            b_lab = lab_buf.reshape(-1) if lab_buf is not None else None
            # (A') weights for the τ steps preceding a failure, (T,N) -> (M,)
            if FT_TAU > 0 and FT_W != 1.0:
                wmask = torch.zeros(ROLLOUT, N, device=DEVICE)
                for t_ in range(ROLLOUT):
                    ev = fail_ev_buf[t_] > 0.5
                    if bool(ev.any()):
                        wmask[max(0, t_ - FT_TAU + 1):t_ + 1][:, ev] = 1.0
                b_w = torch.where(wmask.reshape(-1) > 0.5, torch.full((ROLLOUT * N,), FT_W, device=DEVICE),
                                  torch.ones(ROLLOUT * N, device=DEVICE))
                b_w = b_w / b_w.mean()
            else:
                b_w = None
            it_eff = max(it - WARMUP, 1)          # bc annealing starts after the warm-up
            bc_coef = (0.0 if warm else
                       float(rl.bc_kl_start)
                       + (float(rl.bc_kl_end) - float(rl.bc_kl_start))
                       * (it_eff - 1) / max(RL_ITERS - 1, 1))
            M = b_obs.shape[0]
            vacc = pacc = kacc = 0.0; nb = 0; n_skip_nonfinite = 0
            kl_first = float("nan")   # k3 of the first minibatch before any update — should be 0
                                      # (>0 means a rollout/evaluate mismatch: diagnostic signal)
            for _ in range(int(rl.ppo_epochs)):
                perm = torch.randperm(M, device=DEVICE)
                for s0 in range(0, M, int(rl.minibatch)):
                    mb = perm[s0:s0 + int(rl.minibatch)]
                    new_logp, ent, val = student.evaluate(
                        A(b_obs[mb]), b_pth[mb], critic_obs=b_cobs[mb])
                    log_ratio = new_logp - b_logp[mb]
                    ratio = torch.exp(log_ratio)
                    with torch.no_grad():   # k3 estimator (same as ppo.py)
                        _k3 = float(((ratio - 1.0) - log_ratio).mean().item())
                        kacc += _k3
                        if nb == 0:
                            kl_first = _k3
                    s1 = ratio * b_adv[mb]
                    s2 = torch.clamp(ratio, 1 - float(rl.clip), 1 + float(rl.clip)) * b_adv[mb]
                    if b_w is not None:
                        ploss = -(torch.min(s1, s2) * b_w[mb]).mean()
                        vloss = ((val - b_ret[mb]) ** 2 * b_w[mb]).mean()
                    else:
                        ploss = -torch.min(s1, s2).mean()
                        vloss = F.mse_loss(val, b_ret[mb])
                    if POLICY_FREEZE:
                        # head-only: policy, critic and aux are all frozen, so the PPO/BC/critic
                        # terms have no grad_fn. Only the re-grasp head BCE is back-propagated
                        # (the rollout still runs, providing labels from the on-policy distribution).
                        loss = RG_COEF * rg_loss_on(b_obs[mb], b_lab[mb])
                    elif warm:                     # policy frozen — train the critic only
                        loss = float(rl.vf_coef) * vloss
                    else:
                        s_da = student.act(A(b_obs[mb]), deterministic=True,
                                           inference_only=True)
                        bc = gaussian_kl(b_tloc[mb], b_tscl[mb],
                                         s_da["loc"], s_da["scale"]).mean()
                        loss = (ploss + float(rl.vf_coef) * vloss
                                - float(rl.ent_coef) * ent.mean() + bc_coef * bc)
                        if aux_head is not None:
                            loss = loss + AUX_COEF * aux_loss_on(b_obs[mb], b_aux[mb])
                        if rg_head is not None and not RG_FREEZE:
                            loss = loss + RG_COEF * rg_loss_on(b_obs[mb], b_lab[mb])
                    if not bool(torch.isfinite(loss)):
                        n_skip_nonfinite += 1          # non-finite loss -> skip this minibatch (clip_grad would NaN everything)
                        continue
                    rl_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(_rl_params, 1.0)
                    if not bool(torch.isfinite(gn)):
                        n_skip_nonfinite += 1; rl_opt.zero_grad(set_to_none=True)
                        continue
                    rl_opt.step()
                    vacc += float(vloss.item()); pacc += float(ploss.item()); nb += 1
            if not warm:                    # policy is frozen during warm-up (kl≈0) -> hold off lr adaptation
                rl_lr = _adapt_lr(rl_opt, kacc / max(nb, 1),
                                  desired_kl=rl_alr["desired_kl"], lr_min=rl_alr["lr_min"],
                                  lr_max=rl_alr["lr_max"], factor=rl_alr["factor"])
            if it % int(rl.get("log_every", 20)) == 0 or (warm and it == WARMUP):
                succ = (ep_succ / ep_done) if ep_done else float("nan")
                lr_str = (f" kl={kacc/max(nb,1):.4f} kl0={kl_first:.4f} lr={rl_lr:.1e}")
                warm_str = " [warm]" if warm else ""
                if fail_bank is not None:
                    warm_str += (f" bank={fail_bank.bank_n} pushed={fail_bank.n_pushed} "
                                 f"restored={fail_bank.n_restored}")
                if RANK_COEF > 0.0:
                    warm_str += f" rankN={rank_n}"
                if n_skip_nonfinite:
                    warm_str += f" skipNaN={n_skip_nonfinite}"
                print(f"[rl {it:4d}/{WARMUP + RL_ITERS}]{warm_str} "
                      f"rew={float(rew_buf.mean()):.3f} "
                      f"succ={succ:.3f} crush={crush} touchF={touchF/ROLLOUT:.2f} "
                      f"tblPen={tblP/ROLLOUT:.3f} tblAct={tblA/ROLLOUT:.3f} "
                      f"bcKL={bc_coef:.2f} vloss={vacc/nb:.3f} ploss={pacc/nb:.4f}{lr_str}")
                if wb is not None:
                    wb.log({"rl/rew": float(rew_buf.mean()), "rl/succ": succ,
                            "rl/crush": crush, "rl/touchF": touchF / ROLLOUT,
                            "rl/tblPen": tblP / ROLLOUT, "rl/tblAct": tblA / ROLLOUT,
                            "rl/bc_coef": bc_coef, "rl/vloss": vacc / nb,
                            "rl/ploss": pacc / nb, "rl/approx_kl": kacc / max(nb, 1),
                            "rl/approx_kl_first": kl_first,
                            "rl/lr": rl_lr, "rl/warmup": int(warm), "rl/it": it})
        eval3("postrl")

    # ── 8. save (evaluate.py-compatible snapshot) ───────────────────────
    OmegaConf.set_struct(cfg, False)
    cfg.policy.critic_obs_dim = int(CRITIC_DIM)   # recorded explicitly for the evaluate.py rebuild
    if "training" in cfg:
        cfg.training.async_ppo = False            # bypass the builder's handler-injection path
    run_tag = str(dcfg.output.run_tag)
    dagger_pol = make_student(); dagger_pol.load_state_dict(dagger_state)
    for suffix, pol, tag in [(f"_{run_tag}", dagger_pol, "dagger"),
                             (f"_{run_tag}_rl", student, "dagger+RL hybrid")]:
        if tag != "dagger" and not bool(rl.get("enabled", True)):
            continue
        sd = PROJECT_ROOT / "output" / "checkpoints" / f"{HAND}_{cfg.env_name}{suffix}"
        sd.mkdir(parents=True, exist_ok=True)
        path = save_checkpoint(
            save_dir=sd, policy=pol, optimizer=opt,
            seen_steps=int(N_ITERS) * int(h.NWORLD),
            hand_name=HAND, task_name=TASK, env_name=str(cfg.env_name),
            obs_dim=ACTOR_DIM, action_dim=ACTION_DIM)
        OmegaConf.save(cfg, sd / "config.yaml")
        if rg_head is not None:
            torch.save({"state_dict": rg_head.state_dict(),
                        "input_cols": RG_COLS.cpu().tolist(),
                        "input_terms": str(d.get("regrasp_input_terms",
                                                 "joint_qpos,torque_proxy,student_hist")),
                        "consec": int(getattr(h, "REGRASP_PRED_CONSEC", 10)),
                        "threshold": 0.5, "env_name": str(cfg.env_name), "obs_dim": OBS_DIM},
                       sd / "regrasp_head.pt")
            print(f"[save] [{tag}] re-grasp head → {sd / 'regrasp_head.pt'}")
        print(f"[save] [{tag}] {path}")
        print(f"       → python scripts/evaluate.py --hand {HAND} --env {cfg.env_name} --name-suffix {suffix}")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
