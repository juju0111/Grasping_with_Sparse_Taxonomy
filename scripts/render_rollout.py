"""scripts/render_rollout.py — offscreen video of a policy rollout (no viewer window).

Renders the first ``--nworld`` worlds of a deterministic rollout side by side
and writes an MP4 (+ optional GIF). Each tile is one world with its own object;
a green frame marks worlds that reached the task's success state.

    python scripts/render_rollout.py --save-dir pretrained/<hand>_grasping_teacher --hand tesollo \\
        --env grasping_teacher \\
        --nworld 6 --cols 3 --n-frames 220 --out docs/media/tesollo.mp4 --gif

Rendering backend: ``MUJOCO_GL`` (default ``glfw`` → invisible window on the
current DISPLAY; use ``egl`` on headless machines with EGL support, or run
under ``xvfb-run``).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np


def _even(frames):
    """Pad frames to even width/height (libx264 yuv420p requirement)."""
    import numpy as _np
    h, w = frames[0].shape[:2]
    if h % 2 == 0 and w % 2 == 0:
        return frames
    return [_np.pad(f, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge") for f in frames]

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("GRIT_FORCE_HEADLESS", "1")

from grit.util.headless_guard import ensure_headless_gui_stubs  # noqa: E402
ensure_headless_gui_stubs()

import mujoco  # noqa: E402
import torch   # noqa: E402
import warp as wp  # noqa: E402

from grit.training.builders import build_inference_env, build_policy_and_load  # noqa: E402
from grit.training.rl_env_base import SubEnvHandler  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--save-dir", required=True, help="checkpoint dir (config.yaml + *.pt)")
    p.add_argument("--hand", required=True)
    p.add_argument("--task", default="grasping")
    p.add_argument("--env", required=True, help="registered env name of the checkpoint")
    p.add_argument("--ckpt", default=None, help="seen_steps or *.pt (default: latest)")
    p.add_argument("--nworld", type=int, default=6)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--n-frames", type=int, default=220, help="control steps to roll out")
    p.add_argument("--every", type=int, default=2, help="render every k-th control step")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--distance", type=float, default=0.42)
    p.add_argument("--elevation", type=float, default=-18.0)
    p.add_argument("--azimuth", type=float, default=140.0, help="fallback azimuth (used when --palm-view is off or the palm faces down)")
    p.add_argument("--palm-view", action=argparse.BooleanOptionalAction, default=True,
                   help="per-world camera on the palm side: a first pass (same eval seed) records the wrist "
                        "orientation in the hold phase, the render pass looks at the palm from --palm-offset° aside")
    p.add_argument("--palm-offset", type=float, default=0.0, help="extra azimuth offset from the palm normal (deg)")
    p.add_argument("--tip-tilt", type=float, default=0.15,
                   help="palm-view: tilt the camera from the palm normal towards the fingertips by atan(this) "
                        "(0.45 ≈ 24°), so the fronts of the fingers are seen instead of the fingertips pointing at the lens")
    p.add_argument("--el-min", type=float, default=-35.0, help="palm-view: lowest elevation (camera above the hand)")
    p.add_argument("--el-max", type=float, default=0.0,
                   help="palm-view: highest elevation (0 = camera at hand height; never looks up from the table)")
    p.add_argument("--cam-min-above", type=float, default=0.03,
                   help="palm-view: the camera never drops below the table plane + this height (m); the elevation is "
                        "lowered while the object is still on the table and rises with the lift")
    p.add_argument("--az-search", type=float, nargs=2, default=(20.0, 10.0), metavar=("MAX", "STEP"),
                   help="palm-view: also try azimuths ±MAX° (every STEP°) around the palm direction and keep the one "
                        "with most fingertips in front of the object (fingers not hidden behind it)")
    p.add_argument("--avoid-objects", nargs="*", default=["wood block", "wood block oriented"], metavar="NAME",
                   help="objects (pretty names) that hide the fingers from the palm side; such worlds rank behind others")
    p.add_argument("--hero-taxonomy", default=None, metavar="NAME",
                   help="--save-frames picks a clean world of this grasp taxonomy (falls back to the best world)")
    p.add_argument("--wrist-rise-cap", type=float, default=0.12, metavar="M",
                   help="rule lift stops (and holds) once the wrist rose this far above its height at "
                        "LIFT_STEP, so a slipped object never sends the wrist to the top of the lift "
                        "window (handler knob LIFT_WRIST_RISE_CAP_M; 0 = training behaviour)")
    p.add_argument("--polish", action="store_true",
                   help="brighter lighting + shadows + recoloured object (default: the scene exactly as trained)")
    p.add_argument("--object-color", type=float, nargs=4, default=(-1, -1, -1, -1), metavar="C",
                   help="with --polish: RGBA for the object's visual mesh (default: keep the asset colour)")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--out", required=True, help="output .mp4 path")
    p.add_argument("--gif", action="store_true", help="also write a palette-optimised GIF next to the MP4")
    p.add_argument("--gif-width", type=int, default=0, help="GIF width (0 = same as MP4)")
    p.add_argument("--pick", type=int, default=0, metavar="N",
                   help="roll out --nworld worlds but show only N tiles, preferring worlds that "
                        "reached success (0 = show all worlds)")
    p.add_argument("--save-frames", default=None, metavar="NPY",
                   help="also save the tiles of one successful world as (T,H,W,3) uint8")
    p.add_argument("--hero-index", type=int, default=0,
                   help="which successful world --save-frames stores (0 = first; wraps around)")
    p.add_argument("--label", action=argparse.BooleanOptionalAction, default=True,
                   help="burn 'hand · taxonomy · object' into each tile (default on)")
    p.add_argument("--clip-width", type=int, default=320, help="width the --taxonomy-out clips are stored at")
    p.add_argument("--taxonomy-out", default=None, metavar="DIR",
                   help="save one clip per grasp taxonomy seen in the rollout (a successful world when "
                        "there is one) as DIR/<hand>__<taxonomy>.npy + DIR/<hand>.json, for make_taxonomy_grid.py")
    p.add_argument("--overrides", nargs="*", default=None)
    p.add_argument("--stochastic", action="store_true")
    return p.parse_args()


HAND_LABEL = {
    "tesollo": "Tesollo DG-5F", "robotis_sh5": "ROBOTIS SH5",
    "allex": "Allex", "allegro": "Allegro", "inspire": "Inspire RH56", "shadow": "Shadow Hand",
    "wuji_hand2": "Wuji Hand 2",
}


def pretty_object_name(dirname: str) -> str:
    """'002_master_chef_can' → 'master chef can', 'Basket_<hash>' → 'Basket'."""
    import re
    n = re.sub(r"^\d+_", "", dirname)
    n = re.sub(r"_[0-9a-f]{24,}$", "", n)
    return n.replace("_", " ")


def _label_tile(tile, lines, font):
    """Burn text lines into the top-left corner of an (H, W, 3) uint8 tile."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(tile); dr = ImageDraw.Draw(im)
    y = 4
    for txt in lines:
        w = dr.textlength(txt, font=font); h = font.size + 4
        dr.rectangle([4, y, 10 + w, y + h], fill=(0, 0, 0))
        dr.text((7, y + 1), txt, fill=(255, 255, 255), font=font)
        y += h + 2
    return np.asarray(im)


def _frame_tile(tile, color, px=4):
    tile = tile.copy()
    tile[:px, :] = color; tile[-px:, :] = color; tile[:, :px] = color; tile[:, -px:] = color
    return tile


def _grid(tiles, cols):
    n = len(tiles); rows = (n + cols - 1) // cols
    h, w, c = tiles[0].shape
    canvas = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, q = divmod(i, cols)
        canvas[r * h:(r + 1) * h, q * w:(q + 1) * w] = t
    return canvas


def main():
    args = parse_args()
    from pathlib import Path
    save_dir = Path(args.save_dir).expanduser().resolve()
    orchestrator, env, so, cfg = build_inference_env(save_dir, args.nworld, overrides=args.overrides)
    policy, loaded, h = build_policy_and_load(cfg, env, save_dir, hand=args.hand, task=args.task,
                                              env_name=args.env, ckpt=args.ckpt)
    assert loaded, "no checkpoint loaded — check --save-dir/--hand/--task/--env"
    SUCCESS = int(getattr(h, "SUCCESS_REASON_CODE", 1))
    h.timeout_only_done = True            # let every world play its whole episode (no mid-episode recycling)
    h.eval_mode = True                    # evaluation protocol: no early-success termination (in train mode
                                          # a success resets the world's clock → the policy re-approaches,
                                          # i.e. a visible second lift + wrist rotation late in the clip)
    if hasattr(h, "LIFT_WRIST_RISE_CAP_M"):
        h.LIFT_WRIST_RISE_CAP_M = float(args.wrist_rise_cap)

    variants, assignment = so.variants, so.assignment
    obj_names = list(getattr(so, "renamed_obj_names", []) or [])
    nworld = int(so.NWORLD)
    renderers = {}
    datas = [mujoco.MjData(variants[int(assignment[w])]) for w in range(nworld)]

    def polish_model(m):
        """Friendlier lighting + object colour for the offscreen render (visual only)."""
        m.vis.headlight.ambient[:] = (0.35, 0.35, 0.38)
        m.vis.headlight.diffuse[:] = (0.55, 0.55, 0.55)
        m.vis.headlight.specular[:] = (0.15, 0.15, 0.15)
        m.vis.map.shadowclip = 2.0
        m.vis.quality.shadowsize = 4096
        for i in range(m.nlight):
            m.light_castshadow[i] = 1
            m.light_diffuse[i] = (0.55, 0.55, 0.6)
            m.light_ambient[i] = (0.05, 0.05, 0.05)
        # the fore-arm block of every hand asset is a bare dark box — render it light grey
        for b in range(m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if "fore_arm" in nm or "forearm" in nm:
                for g in range(m.ngeom):
                    if int(m.geom_bodyid[g]) == b and int(m.geom_group[g]) in (1, 2):
                        m.geom_rgba[g] = (0.82, 0.82, 0.85, 1.0)
        if args.object_color[0] >= 0 and obj_names:
            body_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in obj_names}
            for g in range(m.ngeom):
                if int(m.geom_bodyid[g]) in body_ids and int(m.geom_group[g]) == 2:
                    m.geom_rgba[g] = args.object_color

    if args.polish:
        for v in set(int(a) for a in assignment):
            polish_model(variants[v])

    def renderer_for(v):
        if v not in renderers:
            r = mujoco.Renderer(variants[v], height=args.height, width=args.width)
            if args.polish:
                r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
                r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
            renderers[v] = r
        return renderers[v]

    cams = [None] * nworld
    table_z = [None]
    tax_names = list(getattr(h.hand_util, "taxonomy_name_list", []) or [])
    obj_dirs = getattr(orchestrator, "obj_path_dir_lst", None)
    obj_idx = getattr(orchestrator, "sampled_obj_indices", None)
    def world_object(w):
        try:
            return pretty_object_name(os.path.basename(obj_dirs[obj_idx[int(assignment[w])]]))
        except Exception:
            return obj_names[0] if obj_names else "object"
    font = None
    if args.label or args.taxonomy_out:
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(11, args.height // 16))
        except Exception:
            font = ImageFont.load_default()

    def push_state(w):
        m, dd = variants[int(assignment[w])], datas[w]
        dd.qpos[:] = qpos_np[w]
        if mocap_np.shape[1] > 0:
            dd.mocap_pos[:] = mocap_np[w]; dd.mocap_quat[:] = mocap_quat_np[w]
        mujoco.mj_forward(m, dd)
        target = (dd.body(obj_names[0]).xpos + np.array([0.0, 0.0, 0.05]) if obj_names
                  else (dd.mocap_pos[0] if mocap_np.shape[1] > 0 else np.array([0, 0, 0.8])))
        if cams[w] is None:
            cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance, cam.elevation, cam.azimuth = args.distance, world_el[w], world_az[w]
            cam.lookat[:] = target
            cams[w] = cam
            if table_z[0] is None:   # highest plane geom = table / floor surface
                planes = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
                table_z[0] = float(max(dd.geom_xpos[g][2] for g in planes)) if planes else 0.0
        else:   # follow the object smoothly so the lift stays in frame
            cams[w].lookat[:] = 0.85 * np.asarray(cams[w].lookat) + 0.15 * target
        if world_el[w] > 0.0:   # camera below the lookat: keep it above the table, so it rises with the lift
            room = (float(cams[w].lookat[2]) - table_z[0] - float(args.cam_min_above)) / float(args.distance)
            cams[w].elevation = float(min(world_el[w], np.degrees(np.arcsin(np.clip(room, -1.0, 1.0)))))
        return m, dd

    per_world = [[] for _ in range(nworld)]
    world_az = np.full(nworld, float(args.azimuth))
    world_el = np.full(nworld, float(args.elevation))
    palm_vis = np.zeros(nworld)                   # cos(angle between palm normal and camera ray); higher = fingers face the camera
    tip_front = np.zeros(nworld)                  # fraction of fingertips nearer the camera than the object (not hidden by it)
    hold_probe = int(getattr(h, "HOLD_STEP", 140)) + 10
    pre_ok = np.ones(nworld, dtype=bool); pre_clean = np.ones(nworld, dtype=bool)
    if args.palm_view and hasattr(h, "seed_eval_reset") and args.n_frames > hold_probe:
        # pass 1 (no rendering, same eval seed → same objects / taxonomies / initial poses): the whole window,
        # to know each world's outcome and its wrist orientation in the hold phase before anything is rendered
        saved = h.seed_eval_reset(); obs = h.reset()
        wz1 = np.zeros((args.n_frames, nworld), dtype=np.float32)
        with torch.no_grad():
            for t in range(args.n_frames):
                obs, _r, _d, info1 = h.step(policy.act(obs, deterministic=not args.stochastic, inference_only=True)["action"])
                wz1[t] = wp.to_torch(h.d.xpos)[:, int(h.wrist_body_id), 2].cpu().numpy()
                if t == hold_probe:
                    R_w = wp.to_torch(h.d.xmat)[:, int(h.wrist_body_id)].cpu().numpy().reshape(nworld, 3, 3)
                    X_w = wp.to_torch(h.d.xpos).cpu().numpy().reshape(nworld, -1, 3)
        try:
            m1 = h.eval_success_masks(); pre_ok = m1["strict" if "strict" in m1 else next(iter(m1))].cpu().numpy().astype(bool)
        except Exception:
            pre_ok = (info1["done_reason"] == SUCCESS).cpu().numpy()
        _hold = int(getattr(h, "HOLD_STEP", 140))
        _rt = getattr(h, "retry_count_torch", None)
        _rt = _rt.cpu().numpy() > 0.5 if _rt is not None else np.zeros(nworld, dtype=bool)
        pre_clean = pre_ok & ~_rt & ((wz1[_hold:].max(0) - wz1[_hold]) < 0.02)
        palm = R_w[:, :, 0]                       # hand assets: palm normal = wrist-frame +X, fingers +Z
        fing = R_w[:, :, 2]
        # fingertips = leaf bodies of the hand subtree below the wrist; object = its (slot) body
        m0 = variants[0]; wrist = int(h.wrist_body_id)
        def _under_wrist(b):
            while b > 0:
                if b == wrist:
                    return True
                b = int(m0.body_parentid[b])
            return False
        parents = set(int(x) for x in m0.body_parentid)
        tip_ids = [b for b in range(m0.nbody) if b != wrist and _under_wrist(b) and b not in parents]
        obj_bid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, obj_names[0]) if obj_names else -1
        n_side = 0
        az_max, az_step = float(args.az_search[0]), max(float(args.az_search[1]), 1.0)
        offsets = [0.0] + [s_ * k for k in np.arange(az_step, az_max + 1e-6, az_step) for s_ in (1.0, -1.0)]
        for w in range(nworld):
            n = palm[w]
            # camera direction: palm normal tilted towards the fingertips, so the fronts of the fingers are seen
            d = n + float(args.tip_tilt) * fing[w]; d = d / max(np.linalg.norm(d), 1e-6)
            # MuJoCo free camera: pos = lookat − dist·[cos(el)cos(az), cos(el)sin(az), sin(el)]
            # → (pos − lookat) ∥ d ⇒ az = atan2(−dʸ, −dˣ), el = asin(−dᶻ). The elevation is clamped: the camera never
            # looks up from the table, so a palm facing the table is seen from the side (such worlds rank low).
            dh = d[:2]
            if np.linalg.norm(dh) >= 0.25:
                az0 = np.degrees(np.arctan2(-dh[1], -dh[0])) + float(args.palm_offset); n_side += 1
            else:
                df = fing[w, :2]
                az0 = np.degrees(np.arctan2(-df[1], -df[0])) if np.linalg.norm(df) > 1e-3 else float(args.azimuth)
            el = float(np.clip(np.degrees(np.arcsin(np.clip(-d[2], -1.0, 1.0))), args.el_min, args.el_max))
            # azimuth search: keep the fingertips between the camera and the object (not hidden behind it)
            rel = (X_w[w, tip_ids] - X_w[w, obj_bid]) if (tip_ids and obj_bid >= 0) else np.zeros((0, 3))
            best = None
            for off in offsets:
                az = az0 + off
                az_r, el_r = np.radians(az), np.radians(el)
                ray = -np.array([np.cos(el_r) * np.cos(az_r), np.cos(el_r) * np.sin(az_r), np.sin(el_r)])   # camera → lookat
                pv = float(np.dot(n, ray))
                tf = float((rel @ (-ray) > 0.0).mean()) if len(rel) else 0.0                                  # tips nearer the camera than the object
                # palm squarely facing the camera (index…little finger in a row across the frame) first,
                # then fingertips not hidden behind the object
                score = pv + 0.5 * tf - 0.002 * abs(off)
                if best is None or score > best[0]:
                    best = (score, az, pv, tf)
            world_az[w], world_el[w], palm_vis[w], tip_front[w] = best[1], el, best[2], best[3]
        h.restore_reset_rng(saved)
        print(f"palm-view: pass 1 success {int(pre_ok.sum())}/{nworld} (clean {int(pre_clean.sum())}); camera set from the "
              f"hold-phase wrist orientation ({n_side}/{nworld} side views; {len(tip_ids)} fingertip bodies; "
              f"tips-in-front mean {tip_front.mean():.2f}, palm·camera mean {palm_vis.mean():.2f})")
        h.seed_eval_reset()
    obs = h.reset()
    tax_row = getattr(h, "_tax_row_host", None)
    tax_row = tax_row.numpy().copy() if tax_row is not None else np.full(nworld, -1)
    world_tax = [tax_names[int(r)] if 0 <= int(r) < len(tax_names) else "uniform" for r in tax_row]
    world_obj = [world_object(w) for w in range(nworld)]
    labels = [[HAND_LABEL.get(args.hand, args.hand), world_tax[w].replace("_", " "), world_obj[w]] for w in range(nworld)]
    print("worlds:", ", ".join(f"[{w}] {world_tax[w]} / {world_obj[w]}" for w in range(nworld)))
    # memory guard: with --taxonomy-out only keep the frames of the first 2 worlds per taxonomy
    # (+ the first worlds a --pick grid may choose from); otherwise every world is kept
    store = set(range(nworld))
    avoid = np.array([world_obj[w] in set(args.avoid_objects or []) for w in range(nworld)])
    def pre_rank(w):    # pass-1 outcome first (bulky objects half a class down), then the palm facing the camera most
        return ((0 if pre_clean[w] else (1 if pre_ok[w] else 2)) + (0.5 if avoid[w] else 0.0),
                -round(float(palm_vis[w]), 1), -float(tip_front[w]))
    if args.taxonomy_out:
        order = sorted(range(nworld), key=pre_rank)
        store = set(order[:max(args.pick, 0) * 3])
        seen = {}
        for w in order:
            if seen.get(world_tax[w], 0) < (6 if world_tax[w] == args.hero_taxonomy else 3):
                store.add(w); seen[world_tax[w]] = seen.get(world_tax[w], 0) + 1
        print(f"storing frames of {len(store)} worlds (best per taxonomy by pass-1 outcome and palm visibility)")
    wrist_z = np.zeros((args.n_frames, nworld), dtype=np.float32)
    with torch.no_grad():
        for t in range(args.n_frames):
            out = policy.act(obs, deterministic=not args.stochastic, inference_only=True)
            obs, _r, _d, info = h.step(out["action"])
            wrist_z[t] = wp.to_torch(h.d.xpos)[:, int(h.wrist_body_id), 2].cpu().numpy()
            if t % args.every:
                continue
            qpos_np = h.d.qpos.numpy(); mocap_np = h.d.mocap_pos.numpy(); mocap_quat_np = h.d.mocap_quat.numpy()
            for w in range(nworld):
                if w not in store:
                    continue
                m, dd = push_state(w)
                r = renderer_for(int(assignment[w]))
                r.update_scene(dd, cams[w])
                tile = r.render()
                if args.label:
                    tile = _label_tile(tile, labels[w], font)
                per_world[w].append(tile)
    # success = the evaluation criterion (strict success at the end of the window), as eval_success.py
    try:
        masks = h.eval_success_masks()
        key = "strict" if "strict" in masks else next(iter(masks))
        ever_success = masks[key].cpu().numpy().astype(bool)
    except Exception:
        ever_success = (info["done_reason"] == SUCCESS).cpu().numpy()
    hold = int(getattr(h, "HOLD_STEP", 140))
    late_rise_w = (wrist_z[hold:].max(0) - wrist_z[hold]) if args.n_frames > hold else np.zeros(nworld)
    retried = getattr(h, "retry_count_torch", None)
    retried = retried.cpu().numpy() > 0.5 if retried is not None else np.zeros(nworld, dtype=bool)
    # "clean" success = no re-grasp retry and no wrist motion after HOLD_STEP (preferred for clips)
    clean = ever_success & ~retried & (late_rise_w < 0.02)
    def rank(w):        # lower is better: clean > success > failure (bulky objects half a class down), then palm visibility
        return ((0 if clean[w] else (1 if ever_success[w] else 2)) + (0.5 if avoid[w] else 0.0),
                -round(float(palm_vis[w]), 1), -float(tip_front[w]))
    cand = [w for w in range(nworld) if per_world[w]]
    print(f"rollout done: {len(per_world[cand[0]])} frames, success worlds = {int(ever_success.sum())}/{nworld} "
          f"(strict, end of window), clean (no retry, still in hold) = {int(clean.sum())}, retried = {int(retried.sum())}")
    if args.pick and args.pick < len(cand):
        order = sorted(cand, key=rank)
        keep = sorted(order[:args.pick])
        print(f"--pick {args.pick}: showing worlds {keep}")
    else:
        keep = cand
    T = len(per_world[keep[0]])
    frames = [_grid([_frame_tile(per_world[w][t], (40, 200, 80)) if ever_success[w] else per_world[w][t]
                     for w in keep], args.cols) for t in range(T)]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    import imageio.v2 as imageio
    imageio.mimsave(args.out, _even(frames), fps=args.fps, macro_block_size=1)
    print("wrote", args.out)
    if args.gif:
        gif = os.path.splitext(args.out)[0] + ".gif"
        w = args.gif_width or frames[0].shape[1]
        vf = f"fps={args.fps},scale={w}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.out, "-vf", vf, gif], check=True)
        print("wrote", gif, f"({os.path.getsize(gif)/1e6:.1f} MB)")
    if args.taxonomy_out:
        import json
        os.makedirs(args.taxonomy_out, exist_ok=True)
        summary = {}
        for tax in sorted(set(world_tax)):
            ws_all = [w for w in range(nworld) if world_tax[w] == tax]
            ws = [w for w in ws_all if per_world[w]]
            pick = sorted(ws, key=rank)[0]
            frames_w = per_world[pick]
            if args.clip_width and args.clip_width < frames_w[0].shape[1]:
                from PIL import Image
                cw = int(args.clip_width); ch = int(round(frames_w[0].shape[0] * cw / frames_w[0].shape[1]))
                frames_w = [np.asarray(Image.fromarray(f).resize((cw, ch), Image.LANCZOS)) for f in frames_w]
            np.save(os.path.join(args.taxonomy_out, f"{args.hand}__{tax}.npy"), np.asarray(frames_w, dtype=np.uint8))
            summary[tax] = {"world": int(pick), "success": bool(ever_success[pick]), "clean": bool(clean[pick]), "palm_vis": round(float(palm_vis[pick]), 3), "tip_front": round(float(tip_front[pick]), 2),
                            "object": world_obj[pick],
                            "n_worlds": len(ws_all), "n_success": int(sum(ever_success[w] for w in ws_all))}
        with open(os.path.join(args.taxonomy_out, f"{args.hand}.json"), "w") as f:
            json.dump({"hand": args.hand, "label": HAND_LABEL.get(args.hand, args.hand), "taxonomies": summary}, f, indent=1)
        print(f"taxonomy clips: {len(summary)} taxonomies → {args.taxonomy_out} "
              f"(success {sum(v['success'] for v in summary.values())}/{len(summary)})")
    if args.save_frames:
        pool = [w for w in range(nworld) if per_world[w]]
        if args.hero_taxonomy:
            sub = [w for w in pool if world_tax[w] == args.hero_taxonomy and clean[w]]
            if sub:
                pool = sub
            else:
                print(f"[hero] no clean stored world with taxonomy {args.hero_taxonomy!r} — using the best available")
        ok = sorted(pool, key=rank)
        n_best = sum(1 for w in ok if rank(w)[0] == rank(ok[0])[0])
        w_pick = ok[args.hero_index % max(n_best, 1)]
        print(f"[hero] world {w_pick}: {world_tax[w_pick]} / {world_obj[w_pick]} rank {rank(w_pick)}")
        np.save(args.save_frames, np.asarray(per_world[w_pick], dtype=np.uint8))
        print("saved frames of world", w_pick, "→", args.save_frames)


if __name__ == "__main__":
    main()
