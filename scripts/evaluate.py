"""scripts/evaluate.py — Render a trained policy in the warp viewer.

Loads a checkpoint directory (``--save-dir`` or ``output/checkpoints/<hand>_<task>_<env><suffix>``),
builds the env and policy from its ``config.yaml`` snapshot + latest checkpoint, and runs the warp
and policy (independent or shared cross-embodiment), and runs the warp
viewer with policy rollout + ghost target overlay + per-world contact-
state colour + interactive gizmo (so future object-in-the-scene tests
keep the same UX).

Usage::

    # single-hand ckpt — --task / --env are task_name / env_name from the training yaml
    python scripts/evaluate.py --hand <HAND> --task <TASK> --env <ENV>

    # shared cross-embodiment ckpt (--hand is the '+'-joined SAVE_DIR id)
    python scripts/evaluate.py --hand "<HAND_A>+<HAND_B>" --rollout-tag <HAND_A>

    # when SAVE_DIR is split by output.run_tag etc. (training prints the exact value)
    python scripts/evaluate.py --hand <HAND> --task <TASK> --env <ENV> \\
        --name-suffix _nse1_nw4096_<RUN_TAG>

    # override the snapshot cfg from the CLI (same syntax as train.py --overrides;
    # dotted keys / lists supported, nworld=... takes precedence over --nworld)
    python scripts/evaluate.py --hand <HAND> --overrides \\
        handler.termination.MAX_EPISODE_STEPS=300 nworld=64

    # fork-dir suffixes
    python scripts/evaluate.py --hand <HAND> --resume-suffix _resume
    python scripts/evaluate.py --hand <HAND> --deadzone-suffix _use_deadzone

Key bindings (warp viewer)
==========================
**Sim / policy**

    [SPACE]  pause / resume simulation
    [A]      toggle policy ↔ random action
    [D]      deterministic ↔ stochastic policy
    [R]      hard reset (env init + new targets)
    [X]      resample ghost targets only (env state preserved)
    [E]      auto reset on / off — when OFF a world is not recycled on timeout;
             it keeps playing with the same target (for observation)
    [Q]/ESC  quit

**Ghost target overlay**

    [H]      ghost target hand on / off. What is drawn as the ghost is decided by
             the handler via ``set_ghost_targets`` — disabled automatically when
             no binding exists.
    [8]/[9]  lower / raise the ghost alpha (tap = ±0.05, hold for a continuous
             ramp; the current value is printed to the console as
             ``[ghost] alpha = ...``). Implemented in
             ``HandRLParserClass._handle_ghost_alpha_keys`` — polled inside
             ``mjwarp_render``, which draws the ghost, so other scripts work too.
    [J]      ghost collision overlay on / off (default ON). The ghost is a
             kinematic FK overlay, so the contacts caught by ``mj_forward`` in
             ``set_ghost_targets`` = **self-collision of the target pose + floor**.
             Penetrating ghost geoms turn red; each contact point gets a magenta
             sphere + force arrow. (The ghost model has no object, so
             ghost↔object penetration is not shown yet.)

**Contact force arrows** (``HandRLParserClass``, handled inside mjwarp_render)

    [K]      contact force arrows of the real hand on / off (default ON).
             ``d.contact`` + ``mujoco_warp.contact_force(to_world_frame=True)``
             give the **world-frame force vector** (normal + friction), drawn as
             a cyan sphere + arrow. Unlike the normal direction drawn by the
             [O]/[B]/[N] sensor markers, this is the actual force direction, and
             the length ∝ ‖f‖.
    [6]/[7]  arrow length scale ÷2 / ×2 (default 0.01 m/N, shared by hand and ghost).

**Task debug markers** (``handler.eval_debug_markers`` — task-agnostic here)

    [J]      toggle the TASK's own 3-D markers on / off (auto-"n/a" when the
             handler defines none). The handler returns world-frame geometry
             only (arrow / line / sphere / text groups; schema documented on
             ``SubEnvHandler.eval_debug_markers``) and this viewer forwards it
             to ``plot_parallel_debug_markers`` — no task ever adds code here.
             ``handler.eval_overlay_lines(selected_world)`` likewise appends
             numeric rows to the overlay. Whatever a task draws, this file does
             not change — implement new visualisations in the handler-side hooks.

**Action pre-processing** (env rule-based curriculum)

    [L]      toggle the env's rule-based action pre-processor on / off.
             Default OFF in eval (raw trained policy); ON applies whatever
             assist the env bound. Handlers with no pre-processor show "n/a".

**Per-world success marker**

    A ball floats above each world (z = label_z): RED by default, turns GREEN the
    moment that world enters a success state. The handler defines success
    (the viewer only reads the mask). No key.

**Contact-state visualisation** (red=self / green=obj / orange=table)

    [V]      per-world hand-geom colour on / off
    [B]      object-contact markers on / off
    [N]      table-contact markers on / off
    [O]      self-collision markers on / off
    [F]      strict gate (per-body contact AND)
    [G]      touch-sensor gate (touch>EPS AND contact)

**Overlay text**

    [M]      toggle all overlay text on / off

**Interactive handle** (mjwarp viewer — nb33 / nb73 pattern)

    Double-click on a body  → (world, body) select + axis-constrained gizmo
    Plain LMB on axis/plane → translate / Shift+ring rotate
    Ctrl + LMB drag         → free trackball rotate (mocap_quat / qpos quat)
    Ctrl + RMB drag         → free camera-plane translate (mocap_pos / qpos xyz)
    Alt  + LMB drag         → grasp t interpolation (per-joint)
    Alt  + wheel            → cycle active grasp joint

    gizmo display (port of the sim_core InteractiveMarker, ``BodyHandle``):
      * blue sphere = 3D mouse position. It lies on the camera plane through the
        gizmo origin and is **the very point used for hover/drag tests**, so
        "where you see it = where you grab it". During a drag it is pinned to the
        plane of the drag start point. Shown only while the gizmo is up.
      * magenta sphere = the point where the handle was grabbed in this drag
        (body frame, so it moves along).
      * a hovered arrow/plane/ring grows in thickness as well as colour
        (``hover_scale``).
      * ``mouse_view_offset=(0.03, 0)`` lifts the pointer 3 cm to the side so the
        cursor arrow does not hide it, as in sim_core (default 0 = at the cursor).
      * the ``cursor world xyz`` overlay is the cursor pixel back-projected from
        the depth buffer (``get_xyz_at_pixel_from_framebuffer``) — text only, no marker.

**Generic viewer** (handled by ``env_warp.handle_visual_keys``)

    [T]      transparency toggle
    [P]      PCD overlay toggle
    [0]–[4]  geomgroup 0–4 toggle
"""
from __future__ import annotations

# === stdlib =============================================================
import argparse
import os
import sys
from typing import Optional

import numpy as np


# ── Bootstrap project root + sys.path (runnable from any cwd) ────────────
# Match nb72/nb73's cwd convention so the relative dataset paths in
# ``config/training/base.yaml`` resolve correctly.
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
_cwd = os.path.join(PROJECT_ROOT, "notebook", "hand")
if not os.path.isdir(_cwd):
    _cwd = PROJECT_ROOT
os.chdir(_cwd)

# ── Headless guard: MUST run before any grit import (module-level
# ``import pyautogui`` crashes on a server with no DISPLAY). (grit.util.*
# __init__ is empty → this import is light.)
from grit.util.headless_guard import ensure_headless_gui_stubs  # noqa: E402
ensure_headless_gui_stubs()


# === third-party ========================================================
import glfw                                                                  # noqa: E402
import torch                                                                 # noqa: E402


# === project ============================================================
from grit.util.mjwarp_viewer        import (                                 # noqa: E402
    MJWarpMinimalViewer, WarpInteractiveHandle,
)
from grit.training.rl_env_base      import SubEnvHandler                     # noqa: E402

# Setup helpers — moved to grit/training/builders.py so evaluate.py is now
# a thin viewer driver. resolve_inference_save_dir / build_inference_env /
# build_policy_and_load encapsulate all the snapshot-driven setup logic
# (load SAVE_DIR/config.yaml, build orchestrator+env, build policy, load
# matching ckpt with shared vs independent auto-dispatch).
from grit.training.builders         import (                                 # noqa: E402
    resolve_inference_save_dir,
    build_inference_env,
    build_policy_and_load,
)

from grit.util.sim_core.utils import SimpleTimer                              # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0]) # type: ignore
    p.add_argument(
        "--hand", default="tesollo",
        help="Hand name embedded in the checkpoint directory / file name "
             "(e.g. 'tesollo'). Default: tesollo.",
    )
    p.add_argument("--task", default="grasping", help=argparse.SUPPRESS)
    p.add_argument("--env",  default="grasping", help="Registered RL env name (grasping | grasping_student, or the alias stored in the checkpoint name). Default: grasping.")
    p.add_argument(
        "--deadzone-suffix", default="", metavar="SUFFIX",
        help="Append to SAVE_DIR when the training run used use_deadzone=True (e.g. '_use_deadzone').",
    )
    p.add_argument(
        "--name-suffix", default="", metavar="SUFFIX",
        help="Append to SAVE_DIR to match a training run's cfg.output disambiguation "
             "tail (e.g. '_nse30_nw4096_for_approach_test'). Training prints the exact "
             "value to copy. Inserted between deadzone-suffix and resume-suffix.",
    )
    p.add_argument(
        "--resume-suffix", default="", metavar="SUFFIX",
        help="Append to SAVE_DIR for forked resume runs (e.g. '_resume').",
    )
    p.add_argument(
        "--ckpt", default=None, metavar="STEPS|FILE",
        help="Which checkpoint to load. Default: the latest (highest "
             "seen_steps). Pass the seen_steps number printed in the "
             "filename (e.g. 12288000, or 12_288_000) to pin one exactly, "
             "or a '*.pt' filename/path (relative paths resolve against "
             "SAVE_DIR). A missing explicit ckpt is an error — it never "
             "falls back to the latest.",
    )
    p.add_argument(
        "--save-dir", default=None, metavar="DIR",
        help="Explicit checkpoint directory (contains config.yaml + *.pt). When given, "
             "--hand/--task/--env/--*-suffix are only used to pick the ckpt file name "
             "and the SAVE_DIR lookup under output/checkpoints is skipped. Use this for "
             "the bundled pretrained/ runs.",
    )
    p.add_argument(
        "--max-ticks", type=int, default=None, metavar="N",
        help="Quit automatically after N simulation ticks (headless / CI smoke runs; "
             "default: run until the window is closed).",
    )
    p.add_argument(
        "--list-ckpts", action="store_true",
        help="Print the checkpoints available in SAVE_DIR and exit.",
    )
    p.add_argument(
        "--nworld", type=int, default=16,
        help="NWORLD override for the viewer (training value is much bigger). Default: 16.",
    )
    p.add_argument(
        "--rollout-tag", default=None, metavar="TAG",
        help="Shared mode only: which training tag the policy rolls out as. "
             "Default: handlers[0]'s hand_name.",
    )
    p.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True,
        help="Start in deterministic eval mode (toggle at runtime with [D]). Default: True.",
    )
    p.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="Hydra-style cfg overrides applied ON TOP of the loaded "
             "SAVE_DIR/config.yaml snapshot (same syntax as train.py). Dotted "
             "keys (``handler.termination.MAX_EPISODE_STEPS=300``) and list "
             "values (``'obj_idxs=[0,2,3]'``) supported — quote arguments "
             "containing shell-special characters (``[``, ``]``, ``*``, …). "
             "``nworld=...`` here wins over --nworld. NOTE: keys that change "
             "obs/action dims (policy.*, Training.n_bps, …) will fail the "
             "checkpoint's strict dim validation.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
# Setup helpers moved to grit/training/builders.py:
#   resolve_inference_save_dir  — assert + build SAVE_DIR path
#   build_inference_env         — load snapshot cfg + build orch / env
#   build_policy_and_load       — make_policy(...) + load ckpt (shared/indep)
# ──────────────────────────────────────────────────────────────────────────


def print_available_checkpoints(save_dir) -> None:
    """List every ``*.pt`` in SAVE_DIR with its ``seen_steps`` (``--list-ckpts``).

    Deliberately identity-agnostic (no hand/task/env filter) so it works
    without building the env — the filenames carry the identity anyway.
    """
    from pathlib import Path
    from grit.training.checkpoint import parse_checkpoint_name

    rows = []
    for p in sorted(Path(save_dir).glob("*.pt")):
        info = parse_checkpoint_name(p)
        if info is not None:
            rows.append((info["seen_steps"], p.name))
    if not rows:
        print(f'[!] no checkpoints under {save_dir}')
        return
    rows.sort()
    print(f'{len(rows)} checkpoints (seen_steps → file):')
    for steps, name in rows:
        print(f'  {steps:>14,}  {name}')
    print(f'latest = {rows[-1][0]:,}   → --ckpt {rows[-1][0]}')



# ──────────────────────────────────────────────────────────────────────────
# Viewer rollout
# ──────────────────────────────────────────────────────────────────────────

# Visual constants — tuned in nb38 / nb73.
WORLD_SPACING         = 1.5
PCD_SUBSAMPLE         = 32
PCD_RADIUS            = 0.003
GHOST_RGBA            = (0.55, 0.55, 0.65, 0.40)
# Contact-state colour params (intensity ∝ touch sensor reading).
TOUCH_EPS    = 0.0
TOUCH_SCALE  = 10.0
TOUCH_GAMMA  = 0.5
# Contact-marker visuals (sphere @ contact pos + arrow along normal).
CONTACT_SPHERE_R    = 0.005
CONTACT_ARROW_LEN   = 0.025
CONTACT_ARROW_R     = 0.0015
CONTACT_NORMAL_FLIP = False
# Marker colours: obj=red, table=orange, self=red-ish.
SPHERE_RGBA_OBJ    = (1.00, 0.00, 0.00, 0.95)
ARROW_RGBA_OBJ     = (1.00, 0.50, 0.00, 0.95)
SPHERE_RGBA_TABLE  = (1.00, 0.55, 0.00, 0.95)
ARROW_RGBA_TABLE   = (1.00, 0.55, 0.00, 0.95)
SPHERE_RGBA_SELF   = (1.00, 0.10, 0.10, 0.95)
ARROW_RGBA_SELF    = (1.00, 0.10, 0.10, 0.95)


def viewer_rollout(env, sampled_orch, h, policy, policy_loaded: bool, *,
                   has_table: bool, deterministic_init: bool = True,
                   max_ticks: Optional[int] = None):
    """nb73-style render loop: per-tick policy rollout + ghost targets +
    contact-state colour + obj/table/self contact markers + interactive
    gizmo handle (double-click pick, Ctrl/Alt modifier drags). Done-mask
    is masked to ``timeout``-only so each world plays out a full
    ``MAX_EPISODE_STEPS`` window before being recycled."""
    env_warp         = sampled_orch.mj_env
    m, d             = sampled_orch.m, sampled_orch.d
    NWORLD           = sampled_orch.NWORLD
    assignment       = sampled_orch.assignment
    variants         = sampled_orch.variants
    added_body_names = sampled_orch.renamed_obj_names
    HAS_OBJECT       = bool(added_body_names)
    HAS_TABLE        = bool(has_table)
    sim_dt           = variants[0].opt.timestep
    # Sub-env supplies the hand identity used for the title + handle mocap
    # (single-hand → its hand; leader-follower → the follower it trains).
    hand_util        = sampled_orch.render_hand_util

    # ── Toggles (mutable in the key-handling block below) ────────────
    sim_step                 = True
    plot_pcd                 = True
    use_policy               = bool(policy_loaded)
    deterministic            = bool(deterministic_init)
    # [E] auto-reset gate: when False a world is not recycled even on a timeout done —
    # it keeps tracking the same target so the pose can be observed
    # (ep_step returns to 0 on done and counts the next window again).
    auto_reset               = True
    show_ghost               = True
    show_contact_state_color = True
    show_obj_markers         = True
    show_table_markers       = True
    show_self_markers        = True
    gate_strict              = True
    gate_by_touch            = True
    show_overlay             = True
    # [J] task-supplied 3-D markers (handler.eval_debug_markers) — generic:
    # this viewer never learns what they mean. Auto-disabled below when the
    # handler does not define any.
    show_task_markers        = True
    # [U] text-label layer WITHIN the task markers. Still task-agnostic: the
    # marker schema already types labels ("text"-kind groups / "label" field),
    # so this just filters that layer out — e.g. force tracking's per-tip
    # "f=…N" (embodiment) / "f*=…N" (ghost) readouts, without touching the
    # arrows/spheres. No-op for tasks whose markers carry no labels.
    show_task_labels         = True

    if not sampled_orch.supports_contact_overlays:
        # Sub-env without single-hand contact/pcd helpers → skip those overlays
        # (policy rollout + parallel render only).
        show_contact_state_color = False
        plot_pcd                 = False
        show_obj_markers = show_table_markers = show_self_markers = False

    # Rule-based action pre-processor (task-bound curriculum assist, if any).
    # In eval there is no training-step decay, so it would run at FULL strength —
    # default it OFF so the raw trained policy is shown; toggle with [L].
    # Handlers without a bound pre-processor just show "n/a".
    # The LF handler has no applier (follower ctrl is applied by its own warp kernel) → guard.
    _applier      = getattr(h, "applier", None)
    _has_preproc  = bool(_applier is not None and _applier.has_action_preprocessor)
    preprocess_on = True
    if _applier is not None:
        _applier.set_action_preprocessor_enabled(preprocess_on)

    REASON       = SubEnvHandler.REASON_NAME
    SUCCESS_CODE = int(getattr(h, "SUCCESS_REASON_CODE", 1))
    TIMEOUT_CODE = 2

    # Timeout-only termination: only ``timeout`` (ep_step >= MAX_EPISODE_STEPS)
    # sets done_mask, so each world plays out a full episode window before being
    # recycled. Without this, state-based failures (dropped object, etc.) keep
    # zeroing ep_step every step → the world never reaches timeout, never resets,
    # and freezes in the failed pose ("zombie"). Non-timeout reasons still land
    # in done_reason for the diagnostics counter + success marker below — this
    # base flag alone covers what the old per-knob overrides (e.g. forcing
    # ``DONE_WRIST_X_Z``) did, without clobbering any task's configured values.
    h.timeout_only_done = True

    # ── Ghost target overlay + grid offsets ──────────────────────────
    # The sub-env owns the ghost (single-hand target hand or leader-follower
    # leader ghost); the viewer only asks whether one exists and drives it
    # through the uniform render hooks. Tasks without a target ghost return
    # False here and the overlay stays off.
    _ghost_available = sampled_orch.setup_render_ghost(h, GHOST_RGBA)
    show_ghost = show_ghost and _ghost_available
    if not _ghost_available:
        print('[ghost] no target ghost for this task — overlay disabled.')
    _ghost_colored = _ghost_available and (sampled_orch.render_ghost_geom_rgba(h) is not None)
    if _ghost_colored:
        print('[ghost] per-world geom colour ON (handler-provided): active=green / rest=red.')

    # ── Task-supplied debug markers ([J]) ────────────────────────────
    # Same contract as the ghost: the HANDLER owns every task semantic
    # (task markers defined by the handler — this file knows nothing about the task)
    # and returns plain world-frame geometry; this loop just forwards it. A
    # probe call decides availability so tasks without markers cost nothing.
    try:
        _markers_available = h.eval_debug_markers() is not None
    except Exception as _e:
        print(f'[markers] eval_debug_markers() raised ({_e}) — overlay disabled.')
        _markers_available = False
    show_task_markers = show_task_markers and _markers_available
    if not _markers_available:
        print('[markers] no task debug markers for this task — overlay disabled.')

    grid_offsets          = MJWarpMinimalViewer.compute_world_offsets(NWORLD, spacing=WORLD_SPACING)
    # Per-world XY offset roots — the sub-env owns the full list (its floating
    # hand(s)' wrist bases + objects), so leader-follower offsets both hands.
    offset_body_names_lst = list(sampled_orch.render_offset_body_names)
    candidate_body_names  = [hand_util.rh_mocap_name]      + added_body_names

    env_warp.init_mjwarp_viewer(
        title     = f"evaluate — {hand_util.hand_name}",
        width     = 0.6, height = 1.0, fontscale = 200,
        azimuth   = 170, distance = 4.0, elevation = -20,
        lookat    = [0.0, 0.0, 1.0],
    )
    env_warp.init_visual_state(
        transparency=True, contactpoint=True, shadow=False,
        geomgroups=(True, True, True, False, False),
    )
    env_warp.is_running = True
    env_warp.tick       = 0

    # ── Interactive handle (double-click pick + axis-constrained gizmo) ─
    # Useful even in tracking-only setups; essential for future object-in-
    # the-scene tests so the same UX (Ctrl/Alt drags + Alt+wheel grasp
    # cycle) is reachable from this script.
    warp_handle = WarpInteractiveHandle(
        env_warp, d,
        candidate_body_names = candidate_body_names,
        grid_offsets         = grid_offsets,
        wrist_mocap_name     = hand_util.rh_mocap_name,
        max_pick_dist        = 0.20,
    ).attach()
    print('viewer + interactive handle attached.')

    # Initial reset (samples cond targets + obs).
    obs      = h.reset()
    prev_obs = obs.clone()

    reason_counter = {v: 0 for v in REASON.values()}
    n_completed = n_success = n_resets = 0
    world_success  = np.zeros(NWORLD, dtype=bool)   # per-world: currently in a success state (reason==SUCCESS)

    if SimpleTimer is None:
        def _do_run(): return True
        def _end():    pass
        class _T:
            start = staticmethod(lambda *_: None)
            do_run = staticmethod(_do_run)
            end    = staticmethod(_end)
        tmr_sim = _T(); tmr_render = _T()
    else:
        tmr_sim    = SimpleTimer(name="Sim",    Hz=1.0 / (sim_dt * env.sim_nstep), verbose=False)
        tmr_render = SimpleTimer(name="Render", Hz=25,                              verbose=False)
    tmr_sim.start(); tmr_render.start(); env_warp.reset_wall_time()

    def _resample_ghost_targets():
        """nb73 [X] — resample the target(s). Handlers exposing an env-state-
        preserving cond hook use it; others fall back to a full reset."""
        if hasattr(h, "_post_cond_update_hook"):
            h.cond.reset(mask_wp=None)
            h._post_cond_update_hook(world_mask_np=None)
        else:
            h.reset()

    try:
        while env_warp.is_viewer_alive():
            if max_ticks is not None and env_warp.tick >= max_ticks:
                print(f'[--max-ticks] reached {max_ticks} ticks — quitting.')
                break
            # ── Sim + policy step ────────────────────────────────────
            if tmr_sim.do_run():
                env_warp.increase_wall_time()
                if sim_step:
                    with torch.no_grad():
                        if use_policy:
                            # Actor-only forward: skip the critic so an async policy
                            # needs no privileged critic obs at inference (value unused here).
                            out    = policy.act(prev_obs, deterministic=deterministic,
                                                inference_only=True)
                            action = out['action']
                        else:
                            action = (torch.rand(NWORLD, h.action_dim,
                                                 device=env.torch_device) * 2.0 - 1.0)
                        next_obs, _rew, _done, info = h.step(action)

                    done_mask_t   = info['done_mask']
                    done_reason_t = info['done_reason']
                    # Per-world success state for the green success marker
                    # (success worlds hold reason==SUCCESS; others 0 / reset).
                    world_success = (done_reason_t == SUCCESS_CODE).cpu().numpy()

                    # ``timeout_only_done`` (set above) already gates done_mask to
                    # the episode-window end, so done_mask carries exactly the
                    # worlds that reached timeout (or a timeout promoted to
                    # success). Recycle directly off it — no extra masking, and
                    # success-promoted worlds (reason==1) recycle too (the old
                    # ``done_reason==2``-only mask left them frozen). done_reason
                    # still reports the real terminal state for the counter below.
                    n_done = int(done_mask_t.sum().item())
                    if n_done > 0 and auto_reset:
                        for r in done_reason_t[done_mask_t == 1].cpu().tolist():
                            reason_counter[REASON[r]] += 1
                            if r == SUCCESS_CODE:
                                n_success += 1
                        n_completed += n_done
                        h.per_world_reset_if_done()
                        prev_obs = h.obs_torch.clone()
                        n_resets += n_done
                    else:
                        # auto_reset OFF: not recycled even on done (= timeout) —
                        # only ep_step returns to 0; the world keeps playing with the
                        # same target (statistics counters pause as well).
                        prev_obs = next_obs.clone()
                    env_warp.increase_tick()
                tmr_sim.end()

            # ── Render frame ─────────────────────────────────────────
            if tmr_render.do_run():
                # 1) Per-world contact-state RGBA (hand geom colour by contact source)
                per_world_rgba = None
                if show_contact_state_color:
                    try:
                        per_world_rgba = sampled_orch.compute_contact_state_rgba(
                            d, touch_eps=TOUCH_EPS,
                            touch_scale=TOUCH_SCALE, touch_gamma=TOUCH_GAMMA,
                            gate_by_touch=gate_by_touch,
                            gate_by_per_body_contact=gate_strict,
                        )
                    except Exception:
                        per_world_rgba = None

                if plot_pcd:
                    sampled_orch.plot_parallel_pcd(d, n_pcd_subsample=PCD_SUBSAMPLE, pcd_r=PCD_RADIUS)

                # 2) Contact markers (sphere @ contact pos + arrow along normal)
                n_active_self = n_active_obj = n_active_table = 0
                try:
                    cv_self = sampled_orch.get_contact_sensor_values(
                        d, gate_by_touch=gate_by_touch, touch_eps=TOUCH_EPS,
                        gate_by_per_body_contact=gate_strict,
                    )
                    n_active_self = sum(int((np.asarray(v["found"]) > 0.5).sum())
                                        for v in cv_self.values())
                    if show_self_markers:
                        env_warp.plot_parallel_contact_sensor_markers(
                            contact_values=cv_self, grid_offsets=grid_offsets, n_worlds=NWORLD,
                            sphere_r=CONTACT_SPHERE_R, sphere_rgba=SPHERE_RGBA_SELF,
                            arrow_len=CONTACT_ARROW_LEN, arrow_r=CONTACT_ARROW_R,
                            arrow_rgba=ARROW_RGBA_SELF, normal_flip=CONTACT_NORMAL_FLIP,
                        )
                except Exception:
                    pass

                if HAS_OBJECT:
                    try:
                        cv_obj = sampled_orch.get_contact_sensor_for_object_values(
                            d, gate_by_touch=gate_by_touch, touch_eps=TOUCH_EPS,
                            gate_by_per_body_contact=gate_strict,
                        )
                        n_active_obj = sum(int((np.asarray(v["found"]) > 0.5).sum())
                                           for v in cv_obj.values())
                        if show_obj_markers:
                            env_warp.plot_parallel_contact_sensor_markers(
                                contact_values=cv_obj, grid_offsets=grid_offsets, n_worlds=NWORLD,
                                sphere_r=CONTACT_SPHERE_R, sphere_rgba=SPHERE_RGBA_OBJ,
                                arrow_len=CONTACT_ARROW_LEN, arrow_r=CONTACT_ARROW_R,
                                arrow_rgba=ARROW_RGBA_OBJ, normal_flip=CONTACT_NORMAL_FLIP,
                            )
                    except Exception:
                        pass

                if HAS_TABLE:
                    try:
                        cv_table = sampled_orch.get_contact_sensor_for_table_values(
                            d, gate_by_touch=gate_by_touch, touch_eps=TOUCH_EPS,
                            gate_by_per_body_contact=gate_strict,
                        )
                        n_active_table = sum(int((np.asarray(v["found"]) > 0.5).sum())
                                             for v in cv_table.values())
                        if show_table_markers:
                            env_warp.plot_parallel_contact_sensor_markers(
                                contact_values=cv_table, grid_offsets=grid_offsets, n_worlds=NWORLD,
                                sphere_r=CONTACT_SPHERE_R, sphere_rgba=SPHERE_RGBA_TABLE,
                                arrow_len=CONTACT_ARROW_LEN, arrow_r=CONTACT_ARROW_R,
                                arrow_rgba=ARROW_RGBA_TABLE, normal_flip=CONTACT_NORMAL_FLIP,
                            )
                    except Exception:
                        pass

                # 2b) Task-supplied debug markers — the handler returns
                #     world-frame geometry only (schema on
                #     SubEnvHandler.eval_debug_markers); nothing here knows
                #     which task or quantity it is drawing.
                if show_task_markers:
                    try:
                        _groups = h.eval_debug_markers()
                        if _groups and not show_task_labels:
                            # [U] off → drop the label layer only: "text"-kind
                            # groups vanish, other kinds keep geometry but lose
                            # their "label" field.
                            _groups = [
                                {k: v for k, v in g.items() if k != "label"}
                                for g in _groups
                                if str(g.get("kind", "sphere")) != "text"
                            ]
                        env_warp.plot_parallel_debug_markers(
                            _groups,
                            grid_offsets=grid_offsets, n_worlds=NWORLD,
                        )
                    except Exception:
                        pass

                # 3) Ghost target overlay — the sub-env owns the FK and the
                #    model/data (single-hand target hand or LF leader ghost).
                if show_ghost:
                    sampled_orch.update_render_ghost(h, grid_offsets)
                    _tgt_model     = sampled_orch.render_ghost_model
                    _tgt_data      = sampled_orch.render_ghost_data
                    _tgt_geom_rgba = sampled_orch.render_ghost_geom_rgba(h)
                else:
                    _tgt_model = _tgt_data = _tgt_geom_rgba = None

                # 4) Selection highlight + main parallel render. ``plot_success``
                #    draws a green sphere above each True world (at ``label_z``);
                #    feed the live per-world success so only successes show green.
                warp_handle.draw_highlight()
                env_warp.mjwarp_render(
                    mjwarp_data         = d,
                    offset_body_names   = offset_body_names_lst,
                    viewer_model        = variants,
                    mjwarp_model        = m,
                    label_z             = 1.5,
                    plot_success        = world_success,
                    plot_env_number     = True,
                    assignment          = assignment,
                    world_spacing       = WORLD_SPACING,
                    per_world_geom_rgba = per_world_rgba,
                    # General ghost: single-hand target ghost or LF leader ghost
                    # (resolved above). target_geom_rgba is single-hand-only.
                    target_model        = _tgt_model,
                    target_data         = _tgt_data,
                    target_geom_rgba    = _tgt_geom_rgba,
                )
                xyz, flag = env_warp.get_xyz_left_double_click_from_framebuffer()
                if flag:
                    if not warp_handle.update_selection(xyz):
                        warp_handle.clear_selection()
                env_warp.plot_time(loc='bottom left')

                # ── Key bindings ─────────────────────────────────────
                env_warp.handle_visual_keys()
                if env_warp.is_key_pressed_once(glfw.KEY_SPACE): sim_step                 = not sim_step
                if env_warp.is_key_pressed_once(glfw.KEY_P):     plot_pcd                 = not plot_pcd
                if env_warp.is_key_pressed_once(glfw.KEY_A):     use_policy               = (not use_policy) and policy_loaded
                if env_warp.is_key_pressed_once(glfw.KEY_D):     deterministic            = not deterministic
                if env_warp.is_key_pressed_once(glfw.KEY_H):     show_ghost               = (not show_ghost) and _ghost_available
                if env_warp.is_key_pressed_once(glfw.KEY_V):     show_contact_state_color = not show_contact_state_color
                if env_warp.is_key_pressed_once(glfw.KEY_B):     show_obj_markers         = not show_obj_markers
                if env_warp.is_key_pressed_once(glfw.KEY_N):     show_table_markers       = not show_table_markers
                if env_warp.is_key_pressed_once(glfw.KEY_M):     show_overlay             = not show_overlay
                if env_warp.is_key_pressed_once(glfw.KEY_F):     gate_strict              = not gate_strict
                if env_warp.is_key_pressed_once(glfw.KEY_G):     gate_by_touch            = not gate_by_touch
                if env_warp.is_key_pressed_once(glfw.KEY_O):     show_self_markers        = not show_self_markers
                if env_warp.is_key_pressed_once(glfw.KEY_J):     show_task_markers        = (not show_task_markers) and _markers_available
                if env_warp.is_key_pressed_once(glfw.KEY_U):     show_task_labels         = (not show_task_labels) and _markers_available
                if env_warp.is_key_pressed_once(glfw.KEY_L) and _has_preproc:
                    preprocess_on = not preprocess_on
                    h.applier.set_action_preprocessor_enabled(preprocess_on)
                if env_warp.is_key_pressed_once(glfw.KEY_E):     auto_reset               = not auto_reset
                if env_warp.is_key_pressed_once(glfw.KEY_X):     _resample_ghost_targets()
                if env_warp.is_key_pressed_once(glfw.KEY_R):
                    obs            = h.reset()
                    prev_obs       = obs.clone()
                    reason_counter = {v: 0 for v in REASON.values()}
                    n_completed = n_success = n_resets = 0
                    env_warp.tick = 0
                    env_warp.reset_wall_time()
                if (env_warp.is_key_pressed_once(glfw.KEY_Q)
                        or env_warp.is_key_pressed_once(glfw.KEY_ESCAPE)):
                    break

                # ── Overlay (always show the [O] hint, rest gated) ───
                env_warp.viewer_text_overlay(text1="text overlay [M]:", text2=str(show_overlay))
                if show_overlay:
                    warp_handle.draw_overlay()
                    sr = (n_success / max(n_completed, 1)) if n_completed > 0 else 0.0
                    # Selected-world readout (double-click a world to select):
                    # ep_step, done reason (+name), and the manual-lift rule state
                    # (latch / curriculum scale / obj lift_z) so it's clear WHY the
                    # rule-based wrist lift is or isn't firing for that world.
                    _sel_w = getattr(warp_handle, "world_idx", None)
                    if _sel_w is not None:
                        _w  = int(_sel_w)
                        _ep = int(h.ep_step_torch[_w].item())
                        env_warp.viewer_text_overlay(
                            text1=f"selected world [{_w}] ep_step:",
                            text2=f"{_ep} / {h.MAX_EPISODE_STEPS}")
                        _rc = int(h.done_reason_torch[_w].item())
                        try:
                            _rname = REASON[_rc]
                        except Exception:
                            _rname = "none" if _rc == 0 else str(_rc)
                        env_warp.viewer_text_overlay(
                            text1=f"selected world [{_w}] done reason:",
                            text2=f"{_rc} ({_rname})")
                        # Handler-specific action-preprocessor diagnostics.
                        _diag = []
                        if hasattr(h, "lift_reached_torch"):
                            _diag.append(f"latch={int(h.lift_reached_torch[_w].item())}")
                        if hasattr(h, "lift_curriculum_scale"):
                            _diag.append(f"scale={float(h.lift_curriculum_scale):.2f}")
                        if hasattr(h, "LIFT_STEP") and hasattr(h, "HOLD_STEP"):
                            _in_win = int(h.LIFT_STEP) <= _ep < int(h.HOLD_STEP)
                            _diag.append(f"in_lift_window={_in_win}")
                        if hasattr(h, "obj_lift_z_torch"):
                            _diag.append(f"lift_z={float(h.obj_lift_z_torch[_w].item()):.3f}")
                        if _diag:
                            env_warp.viewer_text_overlay(
                                text1="  manual-lift [latch/scale/window/lift_z]:",
                                text2="  ".join(_diag))
                    else:
                        env_warp.viewer_text_overlay(
                            text1="selected world:",
                            text2="(double-click a world to select)")
                    # Task-supplied readout rows (handler owns units/format —
                    # e.g. force tracking prints its per-finger f* vs f). Same
                    # task-agnostic contract as the 3-D markers above.
                    try:
                        _task_rows = h.eval_overlay_lines(_sel_w)
                    except Exception:
                        _task_rows = None
                    for _t1, _t2 in (_task_rows or []):
                        env_warp.viewer_text_overlay(text1=str(_t1), text2=str(_t2))
                    env_warp.viewer_text_overlay(
                        text1="policy [A]:",
                        text2=f"{'on' if use_policy else 'random'}  (loaded={policy_loaded})")
                    env_warp.viewer_text_overlay(
                        text1="deterministic [D] / ghost [H] / sim [SPACE] / pcd [P]:",
                        text2=f"D={deterministic}  H={show_ghost}  SPACE={sim_step}  P={plot_pcd}"
                              + ("  (ghost: active=green/rest=red)" if _ghost_colored else ""))
                    env_warp.viewer_text_overlay(
                        text1="action pre-proc rule [L]:",
                        text2=(f"{'ON' if preprocess_on else 'OFF'}" if _has_preproc
                               else "n/a (no rule bound)"))
                    env_warp.viewer_text_overlay(
                        text1="resample ghost [X] / hard reset [R]:",
                        text2="env-state preserved on [X]")
                    env_warp.viewer_text_overlay(
                        text1="reset policy / auto reset [E]:",
                        text2=f"timeout-only (every {h.MAX_EPISODE_STEPS} steps)  /  "
                              f"{'ON' if auto_reset else 'OFF (hold)'}")
                    env_warp.viewer_text_overlay(
                        text1="completed / success / world-resets:",
                        text2=f"{n_completed}  /  {n_success}  /  {n_resets}  (rate={sr*100:.1f}%)")
                    env_warp.viewer_text_overlay(
                        text1="reset reasons:",
                        text2="  ".join(f"{k}={v}" for k, v in reason_counter.items() if v > 0) or "(none)")
                    env_warp.viewer_text_overlay(
                        text1="contact-state colour [V]:",
                        text2=f"{'on' if show_contact_state_color else 'off'}  "
                              f"green=obj  orange=table  red=self  (shade ~ touch)")
                    env_warp.viewer_text_overlay(
                        text1="obj [B] / table [N] / self [O] markers:",
                        text2=(f"obj=({'on' if show_obj_markers else 'off'},{n_active_obj})  "
                               f"table=({'on' if show_table_markers else 'off'},{n_active_table})  "
                               f"self=({'on' if show_self_markers else 'off'},{n_active_self})"))
                    env_warp.viewer_text_overlay(
                        text1="strict gate [F] / touch->contact gate [G]:",
                        text2=f"F={'on' if gate_strict else 'off'}  G={'on' if gate_by_touch else 'off'}")
                    env_warp.viewer_text_overlay(
                        text1="task markers [J] / labels [U] / font [-][=]:",
                        text2=(f"{'on' if show_task_markers else 'off'} / "
                               f"{'on' if show_task_labels else 'off'} / "
                               f"{env_warp.viewer_fontscale}"
                               if _markers_available else "n/a (task has none)"))
                    # Ghost alpha [8]/[9] + contact-force / ghost-collision
                    # overlays [K]/[J]/[6]/[7] — the rows are built by the
                    # parser that owns those keys, so they can't drift.
                    env_warp.viewer_contact_vis_overlay()
                tmr_render.end()
    finally:
        warp_handle.detach()
        env_warp.close_viewer()

    print(f'\nDone. ticks={env_warp.tick}  completed={n_completed}  '
          f'success={n_success}  world_resets={n_resets}')
    print(f'reset reasons: {reason_counter}')


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.save_dir:
        from pathlib import Path as _P
        save_dir = _P(args.save_dir).expanduser().resolve()
        assert save_dir.is_dir(), f'--save-dir not found: {save_dir}'
    else:
        save_dir = resolve_inference_save_dir(
            project_root    = PROJECT_ROOT,
            hand            = args.hand,
            task            = args.task,
            env_name        = args.env,
            deadzone_suffix = args.deadzone_suffix,
            name_suffix     = args.name_suffix,
            resume_suffix   = args.resume_suffix,
        )
    print(f'SAVE_DIR : {save_dir}')

    if args.list_ckpts:
        print_available_checkpoints(save_dir)
        return

    orchestrator, env, sampled_orch, cfg = build_inference_env(
        save_dir, args.nworld, overrides=args.overrides)
    policy, policy_loaded, h = build_policy_and_load(
        cfg, env, save_dir,
        hand=args.hand, task=args.task, env_name=args.env,
        rollout_tag=args.rollout_tag, ckpt=args.ckpt,
    )
    has_table = bool(cfg.Training.get("with_table", False))
    viewer_rollout(env, sampled_orch, h, policy, policy_loaded,
                   has_table=has_table, deterministic_init=args.deterministic,
                   max_ticks=args.max_ticks)


if __name__ == "__main__":
    main()
