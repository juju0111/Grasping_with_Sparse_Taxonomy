import os
import random
import copy
from typing import Any 

import numpy as np 
import re
import mujoco
import trimesh

from ..util.utils import print_blue, print_green, print_yellow, print_red
from ..util.utils import quat2r 

from hydra import initialize, compose
from etils import epath

from grit.util.sim_core.transforms import r2t, t2p, t2r, pr2t, t2pr, r2quat, rpy2r
from grit.util.sim_core.mjcf import merge_mjcfs as _raw_merge_mjcfs

# == XML output routing ====================================================
# All env-build XML artefacts (merged scene, _w_obj_N skeleton, _variant_*
# variants) are written under ``XML_OUTPUT_DIR`` (relative to cwd) so the
# notebook directory does not get cluttered. Override at runtime via
# ``hand_utils.XML_OUTPUT_DIR = "..."``.
XML_OUTPUT_DIR = "xml"

def _parse_memory_size(spec) -> int:
    """Convert a MuJoCo-style memory size string (e.g. ``"64M"``) → bytes.

    Accepts:
      * int / float / numpy scalar → returned as ``int(bytes)`` unchanged
      * str with optional K / M / G suffix (case-insensitive) — multiplied
        by 1024 / 1024² / 1024³ respectively.  No suffix → raw bytes.

    Used to feed ``mjSpec.memory`` (which is typed as int(bytes)) from the
    convenient YAML suffix form (``mjcf_arena_memory: "64M"``).

    Raises ``ValueError`` on unparseable input — callers should pass
    ``None`` / empty string to skip the directive entirely.
    """
    if spec is None:
        raise ValueError("_parse_memory_size: spec is None — caller must skip instead")
    if isinstance(spec, (int, float)) or hasattr(spec, "__index__"):
        return int(spec)
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"_parse_memory_size: cannot parse {spec!r}")
    s = spec.strip().upper()
    mult = 1
    if s.endswith("K"):
        mult, s = 1024, s[:-1]
    elif s.endswith("M"):
        mult, s = 1024 * 1024, s[:-1]
    elif s.endswith("G"):
        mult, s = 1024 * 1024 * 1024, s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError as e:
        raise ValueError(
            f"_parse_memory_size: cannot parse {spec!r} — expected forms "
            f"like '64M', '512K', '2G' or a raw byte count."
        ) from e


def _resolve_output_xml_path(output_xml_path: str) -> str:
    """Route bare filenames into ``XML_OUTPUT_DIR``; honour explicit dirs.

    - ``"scene.xml"``        → ``"xml/scene.xml"``  (auto-prefixed, dir created)
    - ``"xml/scene.xml"``    → unchanged             (dir created if absent)
    - ``"/abs/foo.xml"``     → unchanged             (dir created if absent)
    - ``""`` / ``None``      → unchanged
    """
    if not output_xml_path:
        return output_xml_path
    head, _ = os.path.split(output_xml_path)
    if head == "":
        output_xml_path = os.path.join(XML_OUTPUT_DIR, output_xml_path)
        head = XML_OUTPUT_DIR
    os.makedirs(head, exist_ok=True)
    return output_xml_path

def _absolutize_includes_in_xml(xml_path: str) -> None:
    """In-place: rewrite every ``<include file="..."/>`` in ``xml_path`` to use
    an absolute path. MuJoCo's relative-path bookkeeping when collapsing
    includes loaded from a deep sub-directory drops one ``..`` segment, so
    using absolute include paths is the simplest stable fix when the merged
    XML lives under ``XML_OUTPUT_DIR``.
    """
    import re as _re
    if not os.path.exists(xml_path):
        return
    with open(xml_path, "r", encoding="utf-8") as f:
        text = f.read()
    base = os.path.dirname(os.path.abspath(xml_path))
    def _abs(m):
        path = m.group(1)
        if os.path.isabs(path):
            return m.group(0)
        return f'<include file="{os.path.normpath(os.path.join(base, path))}"/>'
    new_text = _re.sub(r'<include\s+file="([^"]+)"\s*/>', _abs, text)
    if new_text != text:
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(new_text)

def merge_mjcfs(*args, **kwargs):
    """Drop-in wrapper around ``grit.util.sim_core.mjcf.merge_mjcfs`` that
    routes bare-filename outputs into ``XML_OUTPUT_DIR``, creates the
    directory on demand, and rewrites the resulting ``<include>`` paths to
    absolute so the spec loads cleanly from the sub-directory.
    """
    if "output_xml_path" in kwargs:
        kwargs["output_xml_path"] = _resolve_output_xml_path(kwargs["output_xml_path"])
    out = _raw_merge_mjcfs(*args, **kwargs)
    if isinstance(out, str):
        _absolutize_includes_in_xml(out)
    return out

def absolutize_mesh_paths(spec) -> None:
    """In-place: rewrite every ``MjsMesh.file`` in ``spec`` to an absolute
    path. Resolves relative paths against ``modelfiledir + meshdir`` so the
    spec can be safely serialised to ``MjSpec.to_xml()`` and loaded from a
    different directory (e.g. ``xml/`` subdir) without breaking mesh lookup.

    Idempotent — already-absolute paths are passed through unchanged.
    Required because MuJoCo writes mesh ``file=`` attributes verbatim, so
    moving the output XML invalidates relative paths.
    """
    base = os.path.abspath(spec.modelfiledir or ".")
    if spec.meshdir:
        base = os.path.normpath(os.path.join(base, spec.meshdir))
    for m in spec.meshes:
        if m.file and not os.path.isabs(m.file):
            m.file = os.path.normpath(os.path.join(base, m.file))


def apply_maxhullvert(overall_cfg, spec=None, xml_path=None) -> int:
    """Collision-hull vertex cap (yaml ``maxhullvert``; env ``GRIT_MAXHULLVERT``
    takes precedence when set). Visual rendering keeps the original mesh; the
    hull is used only for the convex narrowphase. mjwarp's GJK scans every hull
    vertex per iteration, so hands whose colliders are raw visual meshes
    (hulls of 358-663 verts) dominate the cost. A cap of 64 gave +32% SPS on
    contact-heavy grasping with unchanged niter and contact-force distributions.

    Both stages are needed:
      * ``spec``     — in-place setting for the spec.compile() path (e.g. variant recompiles).
      * ``xml_path`` — mujoco 3.10's ``spec.to_xml()`` does not serialize
        maxhullvert, so for the HandRLParserClass path (which compiles from XML)
        the attribute is injected directly into the ``<mesh>`` tags of the generated XML.

    Returns the applied N (0 = off).
    """
    mhv_cfg = overall_cfg.get("maxhullvert", 0) or 0
    mhv = int(os.environ.get("GRIT_MAXHULLVERT", str(int(mhv_cfg))))
    if mhv <= 0:
        return 0
    if spec is not None:
        for m in spec.meshes:
            m.maxhullvert = mhv
    if xml_path is not None:
        import re
        with open(xml_path) as f:
            xml = f.read()
        xml, n_sub = re.subn(r"<mesh (?!maxhullvert)",
                             f'<mesh maxhullvert="{mhv}" ', xml)
        with open(xml_path, "w") as f:
            f.write(xml)
        print(f"[collision] maxhullvert={mhv} → {xml_path}: injected into {n_sub} mesh tags")
    return mhv


def assert_unique_mesh_names(spec) -> None:
    """Detect duplicate mesh names up front and raise clearly (call before ``MjSpec.copy()``).

    MuJoCo's ``MjSpec.copy()`` **corrupts the heap and segfaults without any
    validation** on a spec with duplicate mesh names (a normal ``compile()``
    raises a "repeated name … in mesh" ValueError, but this code path calls
    copy first). Duplicates arise when two different hands are merged into one
    leader-follower scene and both produce the same mesh name — e.g. one hand's
    mesh **auto-named from the file** ``base_link.stl`` collides with another
    hand's mesh **explicitly named** ``name="base_link"`` (allegro vs. inspire).

    Effective names follow the MuJoCo convention: the explicit ``mesh.name`` if
    present, otherwise the file basename without extension. On duplicates a
    ValueError listing the clashing names is raised, turning a silent crash
    into an actionable error.
    """
    from collections import Counter
    names = []
    for m in spec.meshes:
        nm = m.name or (os.path.splitext(os.path.basename(m.file))[0] if m.file else "")
        names.append(nm)
    dups = sorted(n for n, c in Counter(names).items() if c > 1 and n)
    if dups:
        raise ValueError(
            f"Duplicate mesh names {dups} — this makes MjSpec.copy() segfault (MuJoCo bug). "
            "The two merged hands produce the same mesh name. Rename the mesh in the hand "
            "asset XML to be unique (e.g. inspire 'base_link' → 'inspire_base_link').")

rh_rgba_lst = [
    (1,0.5,0.,0.5),
    (1,0.5,0.1,0.5),
    (1,0.5,0.2,0.5),
    (1,0.5,0.3,0.5),
    (1,0.5,0.4,0.5),
    (1,0.5,0.5,0.5),
    (1,0.5,0.6,0.5),
    (1,0.5,0.7,0.5),
    (1,0.5,0.8,0.5),
    (1,0.5,0.9,0.5),
    (1,0.5,1.,0.5),
    (1,0.3,0.,0.5),
    (1,0.3,0.1,0.5),
    (1,0.3,0.2,0.5),
    (1,0.3,0.3,0.5),
    (1,0.3,0.4,0.5),
    (1,0.3,0.5,0.5),
    (1,0.3,0.6,0.5),
    (1,0.3,0.7,0.5),
    (1,0.3,0.8,0.5),
    (1,0.3,0.9,0.5),
    (1,0.2,0.1,0.5),
    (1,0.2,0.2,0.5),
    (1,0.2,0.3,0.5),
    (1,0.2,0.4,0.5),
    (1,0.2,0.5,0.5),
    (1,0.2,0.6,0.5),
    (1,0.2,0.7,0.5),
    (1,0.2,0.8,0.5),
    (1,0.2,0.9,0.5),
    (1,0.1,0.1,0.5),
    (1,0.1,0.2,0.5),
    (1,0.1,0.3,0.5),
    (1,0.1,0.4,0.5),
    (1,0.1,0.5,0.5),
    (1,0.1,0.6,0.5),
    (1,0.1,0.7,0.5),
    (1,0.1,0.8,0.5),
    (1,0.1,0.9,0.5),
]

non_rh_rgba_lst = [
    (0,0.5,0.,0.5),
    (0,0.5,0.1,0.5),
    (0,0.5,0.2,0.5),
    (0,0.5,0.3,0.5),
    (0,0.5,0.4,0.5),
    (0,0.5,0.5,0.5),
    (0,0.5,0.6,0.5),
    (0,0.5,0.7,0.5),
    (0,0.5,0.8,0.5),
    (0,0.5,0.9,0.5),
    (0,0.5,1.,0.5),
    (0,0.3,0.,0.5),
    (0,0.3,0.1,0.5),
    (0,0.3,0.2,0.5),
    (0,0.3,0.3,0.5),
    (0,0.3,0.4,0.5),
    (0,0.3,0.5,0.5),
    (0,0.3,0.6,0.5),
    (0,0.3,0.7,0.5),
    (0,0.3,0.8,0.5),
    (0,0.3,0.9,0.5),
    (0,0.2,0.1,0.5),
    (0,0.2,0.2,0.5),
    (0,0.2,0.3,0.5),
    (0,0.2,0.4,0.5),
    (0,0.2,0.5,0.5),
    (0,0.2,0.6,0.5),
    (0,0.2,0.7,0.5),
    (0,0.2,0.8,0.5),
    (0,0.2,0.9,0.5),
    (0,0.1,0.1,0.5),
    (0,0.1,0.2,0.5),
    (0,0.1,0.3,0.5),
    (0,0.1,0.4,0.5),
    (0,0.1,0.5,0.5),
    (0,0.1,0.6,0.5),
    (0,0.1,0.7,0.5),
    (0,0.1,0.8,0.5),
    (0,0.1,0.9,0.5),
]

def get_hand_cfg(hand_name, hand_type):
    config_path = f"../../config/{hand_name}"
    with initialize(version_base='1.3', config_path=config_path):
        cfg = compose(config_name=f"grit_{hand_type}")
    return cfg


# ──────────────────────────────────────────────────────────────────────────
# Finger actuator gain override (servo tuning on a compiled MjModel)
# ──────────────────────────────────────────────────────────────────────────

def apply_finger_gain_override(mjm, *, kp=None, kv=None, damping_scale=None,
                               kv_mode="critical", forcerange=None,
                               verbose=True) -> int:
    """Rewrite finger position-servo gains on a compiled ``MjModel`` in place.

    Every JOINT-transmission position servo (affine bias
    ``force = kp·ctrl − kp·qpos − kv·qvel``, i.e. ``biasprm=[0,-kp,-kv]``) is
    retuned. Non-servo actuators are skipped. Returns the number modified.

    Args (all optional; no-op when kp/kv/damping_scale all None):
      * ``kp``            — new proportional gain (position stiffness).
      * ``kv``            — explicit velocity gain (absolute). Preferred: it is
        the empirically-tuned critical damping **including** joint damping.
      * ``damping_scale`` — multiply each finger joint's ``dof_damping``.
      * ``kv_mode`` (when ``kv`` is None): ``"critical"`` → ``2√(kp·armature)``
        (analytic; ignores joint damping → can over-damp), ``"scale"`` →
        ``kv_old·√(kp/kp_old)`` (fails if kv_old=0), ``"fixed"`` → keep kv_old.
      * ``forcerange``    — cap the actuator force at ``±forcerange`` (Nm).
        **The key knob for reducing penetration**: the soft-contact penetration
        equilibrium scales with the pushing force, and the ±40 Nm in the factory
        XML is tens of times a real finger (0.5-2 Nm), so once the setpoint
        moves inside the object no contact stiffness can hold (measured: 20 mm
        penetration on a robotis squeeze, identical on CPU and warp → a force
        problem, not a solver problem).

    ⚠ Objectless tracking tasks only — raising kp on grasping tasks inflates
    contact force (force = kp·penetration); see docs/experiments/lf_nan_and_robotis_gain.md.
    """
    if kp is None and kv is None and damping_scale is None and forcerange is None:
        return 0
    n_mod = 0
    for a in range(int(mjm.nu)):
        if int(mjm.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        kp_old = float(mjm.actuator_gainprm[a, 0])
        # position-servo guard: bias must be [0, -kp, -kv]
        if kp_old <= 0.0 or abs(float(mjm.actuator_biasprm[a, 1]) + kp_old) > 1e-6:
            continue
        jid = int(mjm.actuator_trnid[a, 0])
        dof = int(mjm.jnt_dofadr[jid])
        if damping_scale is not None:
            mjm.dof_damping[dof] *= damping_scale
        # total damping = joint_damping + actuator_kv, so kv alone doesn't set the
        # damping ratio — explicit ``kv`` (critical incl. joint damping) preferred.
        if kp is not None:
            kv_old = float(mjm.actuator_biasprm[a, 2])   # ≤ 0
            mjm.actuator_gainprm[a, 0] = kp
            mjm.actuator_biasprm[a, 1] = -kp
            if kv is not None:
                mjm.actuator_biasprm[a, 2] = -abs(kv)
            elif kv_mode == "fixed":
                pass
            elif kv_mode == "scale":
                mjm.actuator_biasprm[a, 2] = kv_old * (kp / kp_old) ** 0.5
            else:  # "critical" (analytic; ignores joint damping)
                arm = float(mjm.dof_armature[dof])
                mjm.actuator_biasprm[a, 2] = -2.0 * (kp * max(arm, 1e-6)) ** 0.5
        elif kv is not None:
            mjm.actuator_biasprm[a, 2] = -abs(kv)
        if forcerange is not None:
            mjm.actuator_forcerange[a, 0] = -abs(forcerange)
            mjm.actuator_forcerange[a, 1] = abs(forcerange)
            mjm.actuator_forcelimited[a] = 1
        n_mod += 1
    if verbose and n_mod:
        print(f"[finger-gain] kp={kp} "
              f"kv={('expl ' + str(kv)) if kv is not None else kv_mode} "
              f"damp×{damping_scale} forcerange={forcerange} "
              f"applied to {n_mod} finger actuator(s)")
    return n_mod


def apply_finger_gain_override_from_config(mjm, cfg=None, verbose=True, hand_cfg=None) -> int:
    """Config + env-var driven :func:`apply_finger_gain_override`.

    Precedence per key (kp / kv / damping_scale / forcerange):
      1. env var ``GRIT_FINGER_*`` (quick experiments, not persisted)
      2. training cfg ``finger_gain.<key>`` when **non-null** — old run snapshots
         carry e.g. ``forcerange: 50.0`` and must keep reproducing at eval
      3. per-hand yaml ``config/<hand>/grit_<type>.yaml`` ``finger_gain.<key>``
         (``hand_cfg`` = ``HandUtils.hand_cfg``) — the place for hand-specific
         values such as the real-robot torque cap (shadow ±10, wuji ±2 …).

    Reads the gain from ``cfg.finger_gain`` (``kp`` / ``kv`` / ``damping_scale``
    / ``kv_mode``) so the setting is **persisted in the run's config snapshot and
    REAPPLIED at eval/deploy** (an env-var-only override would not survive into
    ``build_inference_env`` — the trained-on gains would silently revert to the
    factory XML gains at eval). Env vars ``GRIT_FINGER_*`` take precedence over
    the config so quick experiments still override without editing the yaml.
    No-op when neither config nor env provides anything.
    """
    def _cfg(key, src):
        if src is None:
            return None
        try:
            fg = src.get("finger_gain", None)          # OmegaConf DictConfig.get
        except Exception:
            fg = getattr(src, "finger_gain", None)
        if fg is None:
            return None
        try:
            v = fg.get(key, None)
        except Exception:
            v = getattr(fg, key, None)
        return v

    def _pick(env_key, cfg_key):
        e = os.environ.get(env_key, "")
        if e != "":
            return float(e)
        c = _cfg(cfg_key, cfg)
        if c is not None:
            return float(c)
        h = _cfg(cfg_key, hand_cfg)
        return None if h is None else float(h)

    kp = _pick("GRIT_FINGER_KP", "kp")
    kv = _pick("GRIT_FINGER_KV", "kv")
    ds = _pick("GRIT_FINGER_DAMPING_SCALE", "damping_scale")
    fr = _pick("GRIT_FINGER_FORCERANGE", "forcerange")
    kv_mode = os.environ.get("GRIT_FINGER_KV_MODE", "") or (_cfg("kv_mode", cfg) or _cfg("kv_mode", hand_cfg) or "critical")
    if kp is None and kv is None and ds is None and fr is None:
        return 0
    return apply_finger_gain_override(
        mjm, kp=kp, kv=kv, damping_scale=ds, kv_mode=str(kv_mode),
        forcerange=fr, verbose=verbose)


def apply_finger_gain_override_from_env(mjm, verbose=True) -> int:
    """Env-var-only override (``GRIT_FINGER_*``); thin wrapper over
    :func:`apply_finger_gain_override_from_config` with no config source."""
    return apply_finger_gain_override_from_config(mjm, cfg=None, verbose=verbose)


def get_obj_path_dir_lst(dataset_cfgs, verbose=False, project_root=None):
    """Resolve every active dataset's object paths.

    ``project_root``: when given, ``obj_asset_dir_pre_fix`` is interpreted
    relative to the project root rather than ``os.getcwd()``. This makes
    callers cwd-independent — e.g. ``scripts/train.py`` no longer needs
    the historical ``chdir(notebook/hand)`` trick that existed solely to
    let the legacy yaml prefix ``'../../asset/object'`` resolve correctly.

    The yaml prefix may still contain leading ``../`` walk-ups (legacy
    artifact from when configs were authored from ``notebook/hand/``).
    Those are stripped before joining with ``project_root`` so the path
    always lands inside the repo.
    """
    obj_path_dir_lst, obj_name_lst, non_contact_obj_name_lst, obj_xml_path_lst = [], [], [], []

    for dataset_name, dataset_cfg in dataset_cfgs.items():
        if verbose:
            print_blue(f'Dataset Name : {dataset_name}, cfg : {dataset_cfg}')

        if not dataset_cfg.get('use', False):
            continue

        obj_asset_dir = dataset_cfg['obj_asset_dir_pre_fix']
        obj_folder = dataset_cfg['obj_folder_name']
        if project_root is not None and not os.path.isabs(obj_asset_dir):
            # Strip legacy ``../`` walk-ups (cwd=notebook/hand artifact) and
            # anchor at the project root. Result is cwd-independent.
            stripped = re.sub(r'^(\.\./)+', '', obj_asset_dir)
            obj_path_dir = os.path.abspath(os.path.join(project_root, stripped, obj_folder))
        else:
            obj_path_dir = os.path.abspath(os.path.join(obj_asset_dir, obj_folder))
        obj_xml = dataset_cfg.get('obj_xml_path', '')

        if 'Objaverse' in dataset_name:
            # when use_simple is False
            obj_folders = sorted(n for n in os.listdir(obj_path_dir) if "DS" not in n)
            for folder in obj_folders:
                folder_path = os.path.join(obj_path_dir, folder)
                inner_folders = [d for d in os.listdir(folder_path)
                                if os.path.isdir(os.path.join(folder_path, d))]
                n_inner = len(inner_folders)
                if n_inner == 0:
                    continue
                obj_path_dir_lst.extend(os.path.join(folder_path, d) for d in inner_folders)
                # when use_simple is False
                if dataset_cfg.use_simple:
                    obj_name_lst.extend(['body_obj_'+inner_folder for inner_folder in inner_folders])
                else:
                    obj_name_lst.extend(inner_folders)

                non_contact_obj_name_lst.extend([dataset_cfg['non_contact_obj_name']] * n_inner)
                obj_xml_path_lst.extend(os.path.join(folder_path, d, obj_xml) for d in inner_folders)
        elif dataset_name == 'RLWRLD_Proj':
            # No subfolders; direct append
            obj_path_dir_lst.append(obj_path_dir)
            obj_name_lst.append(dataset_cfg['obj_name'])
            non_contact_obj_name_lst.append(dataset_cfg['non_contact_obj_name'])
            obj_xml_path_lst.append(os.path.join(obj_path_dir, obj_xml))
        else:
            subfolders = sorted(
                d for d in os.listdir(obj_path_dir)
                if os.path.isdir(os.path.join(obj_path_dir, d))
            )
            n_sub = len(subfolders)
            if n_sub == 0:
                continue
            obj_path_dir_lst.extend(os.path.join(obj_path_dir, d) for d in subfolders)
            obj_name_lst.extend([dataset_cfg['obj_name']] * n_sub)
            non_contact_obj_name_lst.extend([dataset_cfg['non_contact_obj_name']] * n_sub)
            obj_xml_path_lst.extend(os.path.join(obj_path_dir, d, obj_xml) for d in subfolders)

    return obj_path_dir_lst, obj_name_lst, non_contact_obj_name_lst, obj_xml_path_lst

def get_sampled_indices(n_obj, total_objs):
    """Return a list of n_obj unique random indices from total_objs."""
    if n_obj > total_objs:
        raise ValueError("n_obj cannot be greater than the total number of objects.")
    return random.sample(range(total_objs), n_obj)


def _read_obj_vertices(obj_path: str) -> np.ndarray:
    """Return (N,3) vertex array from an OBJ file."""
    verts = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float64) if verts else np.empty((0, 3))


def build_obj_spec_lst(
        obj_xml_path_set,
        obj_names,
        friction=[0.5, 0.05, 0.01],
        exclude_body_names=['bottom_watertight_tiny'],
        rename_body_name='top_watertight_tiny',
        use_simple=False,
        target_mass=None,
        geom_collision_type: str = "mesh",
        sim_dt=None,
        verbose=False,
    ):
    """
    Build a list of MjSpec objects from object XML paths.
    Managed by Env_Orchestrator; result is passed into make_mjcf_from_spec.
    Returns (obj_spec_lst, per_spec_ngeom, has_non_collide_obj)

    geom_collision_type:
        "mesh" - default. Use all decomposed collision mesh geoms (mjGEOM_MESH).
        "sdf"  - decomposed mesh geoms whose meshname contains "_collision_" are disabled (contype=0).
                 The remaining watertight visual mesh geoms are switched to type=mjGEOM_SDF and enabled (contype=1).
                 The corresponding mesh asset gets needsdf=True to enable SDF precomputation.
        Note: geom.name is always '' here, so it cannot be used as the discriminator → use geom.meshname.
    """
    if geom_collision_type not in ("mesh", "sdf"):
        raise ValueError(f"geom_collision_type must be 'mesh' or 'sdf', got {geom_collision_type!r}")

    has_non_collide_obj = []
    renamed_obj_names = copy.deepcopy(obj_names)

    touch_site_name = 'top_watertight_tiny_touch_site'
    dummy_touch_site_name = 'bottom_watertight_tiny_touch_site'
    # ── object contact timeconst — default tied to sim_dt ─────────────────
    # Stability rule: timeconst ≥ 2×sim_dt (violating it causes under-damped
    # oscillation and worse penetration; see docs for probe measurements).
    # Without sim_dt the legacy 0.01 is kept.
    # env GRIT_OBJ_SOLREF_T is an experimental override with top priority.
    _env_t = os.environ.get("GRIT_OBJ_SOLREF_T", "")
    if _env_t:
        _solref_t = float(_env_t)
    elif sim_dt is not None:
        _solref_t = max(0.01, 2.0 * float(sim_dt))
    else:
        _solref_t = 0.01
    if verbose or _env_t or (sim_dt is not None and _solref_t != 0.01):
        print(f"[obj-solref] timeconst={_solref_t} "
              f"(sim_dt={sim_dt}, env={'set' if _env_t else 'unset'})")

    obj_spec_lst = []
    per_spec_ngeom = []
    for idx, obj_xml in enumerate[Any](obj_xml_path_set):
        if verbose:
            print_blue(f"obj_xml : {obj_xml}")

        obj_name = obj_names[idx]

        dummy_flag = False
        is_objaverse = False
        is_rlwrl_proj = False
        is_there_body_site = False
        ngeom_per_obj = 0

        obj_spec = mujoco.MjSpec.from_file(obj_xml) # type: ignore

        for body in obj_spec.bodies:
            if body.name == obj_names[idx]:
                for b_s in body.sites:
                    if b_s.name == touch_site_name:
                        is_there_body_site = True

            if 'objaverse' in obj_xml:
                is_objaverse = True
                if obj_name in body.name:
                    if verbose: print_red(f"body (before): {body.name}")
                    body.name = body.name.replace(obj_name, rename_body_name)
                    renamed_obj_names[idx] = rename_body_name
                    if verbose: print_red(f"Rename body (after): {body.name}")

            if 'bmw' in obj_xml:
                is_rlwrl_proj = True
                if obj_name in body.name:
                    if verbose: print_red(f"body (before): {body.name}")
                    body.name = body.name.replace(obj_name, rename_body_name)
                    renamed_obj_names[idx] = rename_body_name
                    if verbose: print_red(f"Rename body (after): {body.name}")

            if body.name not in exclude_body_names: continue

            if verbose: print_green(f"This is exclude body : {body.name}")
            geom_num = 0
            for geom in body.geoms:
                if geom.contype == 1:
                    geom_num += 1
            if verbose: print_green(f"geom_num : {geom_num}")
            if geom_num == 1:
                dummy_flag = True
                if verbose: print_yellow(f"body : {body.name}. This is Dummy object")
                # ── Dummy bottom_watertight: strip it from the dynamics ────────
                # A "dummy" bottom carries a single placeholder collision geom plus
                # a visual mesh that add nothing graspable, yet their density (set
                # to 250 below) skews the object's center-of-mass downward/sideways
                # so small objects (mouse, …) lean or topple. Per request, build the
                # spec with the bottom geoms REMOVED so the object's collision AND
                # inertia come from the top_watertight mesh ALONE.
                #
                # Deleting the geoms alone is NOT enough on the parallel-warp path:
                # ``override_geoms_for_body`` (orchestrator/base.py) maps a variant's
                # geoms onto the shared skeleton slots and the leftover skeleton slots
                # keep the anchor's density (its else-branch clears only contype/group,
                # never density). The compiled variant would therefore still pick up
                # mass from those phantom slots. Pinning an EXPLICIT ZERO inertial on
                # the body makes the compiler ignore geom mass entirely (verified:
                # explicit inertial wins over density), so body_mass/body_ipos stay 0
                # in both the CPU variant model and the baked warp model. The body
                # itself is kept (welded, zero-mass) so ``override_bodies_recursive``
                # still finds it by name and copies the zeroed inertial per world.
                for geom in list(body.geoms):
                    obj_spec.delete(geom)
                body.mass    = 0.0
                body.ipos    = [0.0, 0.0, 0.0]
                body.inertia = [0.0, 0.0, 0.0]

        if not is_there_body_site:
            if verbose: print_green("Add body site")
            for body in obj_spec.bodies:
                if body.name == renamed_obj_names[idx]:
                    site = body.add_site()
                    site.name = touch_site_name
                    site.type = mujoco.mjtGeom.mjGEOM_SPHERE # type: ignore
                    site.size = [0.3]*3
                    site.rgba = [0,0,1,0.0]

                    site = body.add_site()
                    site.name = dummy_touch_site_name
                    site.type = mujoco.mjtGeom.mjGEOM_SPHERE # type: ignore
                    site.size = [0.3]*3
                    site.rgba = [0,1,0,0.0]

                    if verbose: print_yellow(f"{body.name} sites : {body.sites}")

            if verbose: print_blue(f"Add sensor")
            sensor1 = obj_spec.add_sensor(
                name='top_watertight_tiny_touch',
                type=mujoco.mjtSensor.mjSENS_TOUCH, # type: ignore
                objname=touch_site_name,
                objtype=mujoco.mjtObj.mjOBJ_SITE, # type: ignore
            )
            sensor2 = obj_spec.add_sensor(name='bottom_watertight_tiny_touch')
            sensor2.type = mujoco.mjtSensor.mjSENS_TOUCH # type: ignore
            sensor2.objname = dummy_touch_site_name
            sensor2.objtype = mujoco.mjtObj.mjOBJ_SITE # type: ignore

            if verbose: print("spec sensors : ", obj_spec.sensors)

        if dummy_flag or is_objaverse:
            has_non_collide_obj.append(False)
        else:
            has_non_collide_obj.append(True)

        # ── SDF mode: drop convex-decomposition geoms + their mesh assets entirely ──
        # Compiling every decomposed piece as an SDF would be slow and pointless —
        # the watertight visual mesh alone serves as the SDF collider. Removing them
        # before the physics loop also keeps make_mjcf_from_spec from copying the
        # orphan mesh assets into the parent spec.
        removed_decomp_geoms = 0
        removed_decomp_meshes = 0
        if geom_collision_type == "sdf":
            geoms_to_delete = []
            for body in obj_spec.bodies:
                for geom in body.geoms:
                    if "_collision_" in geom.meshname.lower():
                        geoms_to_delete.append(geom)
            for g in geoms_to_delete:
                obj_spec.delete(g)
            removed_decomp_geoms = len(geoms_to_delete)

            # mesh.name may still be '' at this stage (assigned from filename later),
            # so match by file basename too.
            meshes_to_delete = []
            for m in obj_spec.meshes:
                key = (m.name or os.path.basename(m.file)).lower()
                if "_collision_" in key:
                    meshes_to_delete.append(m)
            for m in meshes_to_delete:
                obj_spec.delete(m)
            removed_decomp_meshes = len(meshes_to_delete)

        # obj geom / joint physics
        sdf_meshnames = set()  # mesh asset names that need needsdf=True
        sdf_geom_count = 0
        for body in obj_spec.bodies:
            geom_num = 0
            is_exclude_body = body.name in exclude_body_names
            for geom in body.geoms:
                geom_num += 1
                geom.solimp = np.array([0.975, 0.999, 0.001, 0.5, 2.0])
                geom.solref = np.array([_solref_t, 1.0])
                geom.friction = np.array(friction)
                geom.density = 100
                # Only mesh/SDF geoms reference a mesh asset whose name needs the
                # per-object ``_{idx}`` suffix. Primitive geoms (box/sphere/…)
                # carry an empty meshname; suffixing it to "_{idx}" makes the
                # compiler look up a non-existent mesh ("mesh '_0' not found").
                # Guarding on a non-empty meshname lets object XMLs use a
                # primitive collider (e.g. a box for flat objects that jitter on
                # the convex-mesh-vs-plane degenerate 2-point contact).
                if geom.meshname:
                    geom.meshname = geom.meshname + f"_{idx}"
                if (geom_collision_type == "sdf"
                        and not is_exclude_body
                        and geom.type == mujoco.mjtGeom.mjGEOM_MESH):  # type: ignore
                    # Only the watertight visual mesh remains here; promote it to SDF.
                    geom.type = mujoco.mjtGeom.mjGEOM_SDF  # type: ignore
                    geom.contype = 1
                    geom.conaffinity = 1
                    sdf_meshnames.add(geom.meshname)
                    sdf_geom_count += 1
            ngeom_per_obj += geom_num
        per_spec_ngeom.append(ngeom_per_obj)

        for mesh in obj_spec.meshes:
            mesh.file = os.path.abspath(os.path.join(os.path.dirname(obj_xml), mesh.file))
            file_name = os.path.basename(mesh.file)
            if mesh.name == '':
                mesh.name = os.path.splitext(os.path.basename(file_name))[0] + f"_{idx}"
            else:
                mesh.name = mesh.name+f"_{idx}"
            if mesh.name in sdf_meshnames:
                mesh.needsdf = True

        if geom_collision_type == "sdf" and verbose:
            print_blue(
                f"[SDF] obj#{idx} {obj_name}: sdf_geoms={sdf_geom_count}, "
                f"removed_decomp_geoms={removed_decomp_geoms}, "
                f"removed_decomp_meshes={removed_decomp_meshes}, "
                f"remaining_geoms={ngeom_per_obj}"
            )

        # ── Explicit inertial from visual mesh bbox ──
        # Identify the watertight visual mesh by meshname (no "_collision_") rather than
        # by (contype==0 & type==mesh) — in SDF mode the visual geom has been promoted to
        # mjGEOM_SDF with contype=1, so the old filter would miss it.
        _vis_mesh_path = None
        for _body in obj_spec.bodies:
            if _body.name == renamed_obj_names[idx]:
                for _g in _body.geoms:
                    if "_collision_" in _g.meshname.lower():
                        continue
                    if _g.type not in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SDF):  # type: ignore
                        continue
                    for _mesh in obj_spec.meshes:
                        if _mesh.name == _g.meshname:
                            _vis_mesh_path = _mesh.file
                            break
                    if _vis_mesh_path:
                        break
                break

        if _vis_mesh_path and os.path.isfile(_vis_mesh_path):
            _verts = _read_obj_vertices(_vis_mesh_path)
            if len(_verts) > 0:
                _vmin, _vmax = _verts.min(axis=0), _verts.max(axis=0)
                _center      = (_vmin + _vmax) / 2.0
                _dims        = _vmax - _vmin

                offset_ratio = 0.05
                _center_shifted = np.copy(_center)
                _center_shifted[2] -= _dims[2] * offset_ratio

                _OBJ_DENSITY = 100.0
                if target_mass is not None:
                    _mass = float(target_mass)
                else:
                    try:
                        _tm = trimesh.load(_vis_mesh_path, force='mesh', process=False)
                        if isinstance(_tm, trimesh.Trimesh) and _tm.is_watertight and float(_tm.volume) > 1e-10:
                            _mass = _OBJ_DENSITY * float(_tm.volume)
                        else:
                            _mass = _OBJ_DENSITY * float(_tm.convex_hull.volume) * 0.5 # type: ignore
                    except Exception:
                        _mass = _OBJ_DENSITY * float(np.prod(_dims)) * 0.3
                    _mass = float(np.clip(_mass, 0.05, 1.0))

                _Ixx = (1/12) * _mass * (_dims[1]**2 + _dims[2]**2)
                _Iyy = (1/12) * _mass * (_dims[0]**2 + _dims[2]**2)
                _Izz = (1/12) * _mass * (_dims[0]**2 + _dims[1]**2)
                for _body in obj_spec.bodies:
                    if _body.name == renamed_obj_names[idx]:
                        _body.ipos    = _center_shifted
                        _body.mass    = _mass
                        _body.inertia = np.array([_Ixx, _Iyy, _Izz])
                        break

        for texture in obj_spec.textures:
            texture.file = os.path.abspath(os.path.join(os.path.dirname(obj_xml), texture.file))

        obj_spec_lst.append(obj_spec.copy())
        renamed_obj_names[idx] = renamed_obj_names[idx] + '_' + str(idx)

    return obj_spec_lst, per_spec_ngeom, has_non_collide_obj


def make_mjcf_from_spec(
        parent_spec,
        obj_spec_lst,
        per_spec_ngeom,
        n_obj_per_env=1,
        verbose=False,
        output_xml_path="output.xml",
        mjcf_arena_memory=None,
    ):
    """
    Build skeleton MJCF from parent_spec and pre-built obj_spec_lst.
    obj_spec_lst and per_spec_ngeom are produced by build_obj_spec_lst.
    Returns (relative_xml_path, skeleton_spec, added_body_names)

    Args:
        mjcf_arena_memory: optional ``<size memory="..."/>`` directive string
            (e.g. ``"128M"``, ``"512M"``). Pre-allocates the MuJoCo CPU
            solver stack arena to this size on every compiled MjModel.
            Required to avoid ``mj_stackAlloc: out of memory`` overflow
            in dense multi-obj scenes (n_obj ≥ 2 typically triggers it).
            Pass ``None`` / empty string to leave MuJoCo's auto-sizing
            in place (single-obj is generally fine without it).
    """
    # CPU solver arena sizing — see docstring above.
    if mjcf_arena_memory:
        # mjSpec exposes the ``<size memory="..."/>`` directive as
        # ``spec.memory`` — typed as ``int`` (bytes), NOT the string form
        # the XML directive accepts. We parse the convenient YAML suffix
        # syntax (e.g. ``"64M"``, ``"512K"``, ``"2G"``) here and convert
        # to bytes so users keep the readable form in cfg.
        parent_spec.memory = _parse_memory_size(mjcf_arena_memory)
    # Add every mesh of the obj_specs to parent_spec.
    _fallback_obj_meshname = None   # any registered object mesh — used for placeholder geoms
    for idx, obj_spec in enumerate(obj_spec_lst):
        for obj_mesh in obj_spec.meshes:
            added_mesh = parent_spec.add_mesh()
            added_mesh.name    = obj_mesh.name
            if _fallback_obj_meshname is None:
                _fallback_obj_meshname = obj_mesh.name
            added_mesh.file    = obj_mesh.file
            added_mesh.scale   = obj_mesh.scale
            added_mesh.refquat = obj_mesh.refquat
            # SDF mode: propagate the needsdf flag so the parent spec precomputes the SDF.
            if getattr(obj_mesh, "needsdf", False):
                added_mesh.needsdf = True

    # Add skeleton bodies based on the spec with the largest ngeom.
    # (Tree structure/names come from this anchor, but below each body's geom-slot
    #  count is padded to the per-body maximum over all specs — fixes the
    #  single-anchor truncation bug.)
    obj_spec_idx = np.argmax(per_spec_ngeom)
    obj_spec     = obj_spec_lst[obj_spec_idx]

    added_body_names  = []
    added_root_bodies = []   # (skeleton_root_body, suffix) — padding targets
    geom_spec_dict    = {}
    for i in range(n_obj_per_env):
        src_root_body = obj_spec.worldbody.first_body()
        if src_root_body:
            added_body, geom_spec_dict = copy_body_recursive(
                parent_spec.worldbody, src_root_body,
                suffix=f"_{i}", geom_spec_dict=geom_spec_dict)
            added_root_bodies.append((added_body, f"_{i}"))
        added_body_names.append(src_root_body.name + "_" + str(i))

    # ── BUGFIX: per-body geom-slot provisioning ──────────────────────────
    # Pad each skeleton object body's geom slots with disabled placeholders up to
    # the maximum geom count of that body over all specs. This way no variant
    # (e.g. a wine glass with several geoms in bottom_watertight) gets its geoms
    # truncated at the override stage, and the result no longer depends on the
    # anchor choice (i.e. the sample set). The cost is paid once at build time.
    max_geoms_per_body = _collect_max_geoms_per_body(obj_spec_lst)
    for _added_body, _sfx in added_root_bodies:
        _pad_skeleton_body_geoms(
            _added_body, _sfx, max_geoms_per_body, _fallback_obj_meshname, verbose=verbose,
        )

    # After all bodies are added, copy the sensors in one pass → compile
    for i in range(n_obj_per_env):
        copy_sensors(parent_spec, obj_spec, suffix=f"_{i}")

    # Using the self-collision sensors predefined in parent_spec (subtree2 +
    # ``_self_coll``) as templates, dynamically add two kinds of contact sensors:
    #
    # 1) ``add_object_contact_sensors``: ``..._body_obj_contact`` pointing at each
    #    object body (per-(hand_part, slot) contact).
    #    No-op when n_obj_per_env=0.
    # 2) ``add_table_contact_sensors``: ``..._table_contact`` pointing at the table
    #    or world (floor) body (per-hand_part contact).
    #    In the (pathological) setup with neither table nor floor, falls back to ``body2="world"``.
    #
    # Baking these into the XML up front would break compilation in setups without
    # object/table, so they are created right after the bodies are added (once the
    # parent_spec including the table is final).
    add_object_contact_sensors(parent_spec, added_body_names, verbose=verbose)
    add_table_contact_sensors(parent_spec, verbose=verbose)

    # Enable the FORCE channel on all contact sensors (self / object / table) so
    # each contact type reports its own contact force, removing the
    # cross-contamination of sharing a single type-blind touch sensor.
    # (Must be called before compile.)
    ensure_contact_force_channel(parent_spec, verbose=verbose)

    # Skeleton spec complete.
    # MjSpec.copy() segfaults on duplicate mesh names, so validate explicitly first.
    assert_unique_mesh_names(parent_spec)
    skeleton_spec = parent_spec.copy()
    # Absolutize mesh paths so ``to_xml()`` produces an output that resolves
    # meshes correctly regardless of where it lands (e.g. ``xml/`` subdir).
    absolutize_mesh_paths(skeleton_spec)
    xml_string = _to_xml_with_shell_inertia_fix(skeleton_spec, verbose=verbose)

    # Spec-related error introduced in MuJoCo >= 3.3.7: strip leading slashes from name/mesh etc.
    xml_string = re.sub(r'(name\s*=\s*")\s*/+', r'\1', xml_string)
    xml_string = re.sub(r'(mesh\s*=\s*")\s*/+', r'\1', xml_string)
    xml_string = re.sub(r'(site\s*=\s*")\s*/+', r'\1', xml_string)
    xml_string = re.sub(r'(class\s*=\s*")\s*/+', r'\1', xml_string)
    xml_string = re.sub(r'(joint\s*=\s*")\s*/+', r'\1', xml_string)
    xml_string = re.sub(r'(body1\s*=\s*")\s*/+', r'\1', xml_string)
    xml_string = re.sub(r'(body2\s*=\s*")\s*/+', r'\1', xml_string)
    xml_string = re.sub(r'(joint1\s*=\s*")\s*/+', r'\1', xml_string)
    xml_string = re.sub(r'(joint2\s*=\s*")\s*/+', r'\1', xml_string)
    xml_string = re.sub(r'(texture\s*=\s*")\s*/+', r'\1', xml_string)
    xml_string = re.sub(r'(material\s*=\s*")\s*/+', r'\1', xml_string)

    output_xml_path = _resolve_output_xml_path(output_xml_path)
    output_xml_path = os.path.abspath(output_xml_path)
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_string)

    relative_xml_path = os.path.relpath(output_xml_path, os.getcwd())
    return relative_xml_path, skeleton_spec, added_body_names


def _set_shell_inertia_for_mesh(spec, mesh_name):
    """Set shellinertia=True on every geom referencing mesh_name."""
    def _walk(body):
        for g in body.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname == mesh_name: # type: ignore
                g.shellinertia = True
        for child in body.bodies:
            _walk(child)
    _walk(spec.worldbody)


def _to_xml_with_shell_inertia_fix(spec, verbose=False, max_retries=50):
    """If to_xml() fails with 'mesh volume is too small', set shellinertia=True on
    that mesh's geoms and retry."""
    fixed_meshes = set()
    for _ in range(max_retries):
        try:
            return spec.to_xml()
        except ValueError as e:
            m = re.search(r"mesh volume is too small:\s*(\S+)", str(e))
            if m:
                bad_mesh = m.group(1).rstrip('.')
                if bad_mesh in fixed_meshes:
                    raise  # failing twice on the same mesh means a different cause
                fixed_meshes.add(bad_mesh)
                if verbose:
                    print(f"[WARNING] Small mesh volume for '{bad_mesh}', setting shellinertia=True")
                _set_shell_inertia_for_mesh(spec, bad_mesh)
            else:
                raise
    return spec.to_xml()  # final attempt


def copy_body_recursive(dest_parent, src_body, suffix="", geom_spec_dict={}):
    geom_list = []
    new_body          = dest_parent.add_body(name=src_body.name + suffix)
    new_body.pos      = src_body.pos
    new_body.quat     = src_body.quat
    new_body.ipos     = src_body.ipos
    new_body.iquat    = src_body.iquat
    new_body.mass     = src_body.mass
    new_body.inertia  = src_body.inertia

    for j in src_body.joints:
        new_j           = new_body.add_joint(name=j.name + suffix)
        new_j.type      = j.type
        new_j.armature  = j.armature
        # new_j.damping   = j.damping
        new_j.damping   = [0.005, 0., 0.] # Don't change this value. 
        # new_j.damping   = [0.1, 0., 0.]
        new_j.limited   = j.limited
        new_j.range     = j.range

    for g in src_body.geoms:
        # FIX: append the suffix to avoid geom name clashes when n_obj>1
        geom_name = (g.name + suffix) if g.name else g.name
        new_g      = new_body.add_geom(name=geom_name)
        new_g.type = g.type
        # SDF geoms also reference a mesh asset — without copying meshname the compiler
        # fails with "mesh geom '' (id = N) must have valid meshid".
        if g.type in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SDF):  # type: ignore
            new_g.meshname = g.meshname
        # FIX: copy pos/quat/size (otherwise collision geom position/orientation/size all fall back to defaults)
        new_g.pos         = g.pos
        new_g.quat        = g.quat
        new_g.size        = g.size
        new_g.rgba        = g.rgba
        new_g.friction    = g.friction
        new_g.group       = g.group
        new_g.density     = g.density
        new_g.solref      = g.solref
        new_g.solimp      = g.solimp
        new_g.contype     = g.contype
        new_g.conaffinity = g.conaffinity
        if hasattr(g, 'shellinertia'):
            new_g.shellinertia = g.shellinertia
        geom_list.append(new_g)

    geom_spec_dict[new_body.name] = geom_list

    for s in src_body.sites:
        new_s       = new_body.add_site(name=s.name + suffix)
        new_s.pos   = s.pos
        new_s.size  = s.size
        new_s.rgba  = s.rgba

    # FIX: do not overwrite new_body with the return value
    # (previously the child body replaced new_body, so later siblings attached to the wrong parent)
    for child_b in src_body.bodies:
        _, geom_spec_dict = copy_body_recursive(new_body, child_b, suffix, geom_spec_dict)

    return new_body, geom_spec_dict


def _collect_max_geoms_per_body(obj_spec_lst):
    """Per body-NAME max geom count across ALL object specs.

    The shared skeleton must reserve, for EVERY body, as many geom slots as
    the object that owns the MOST geoms in that body. ``make_mjcf_from_spec``
    used to copy the skeleton from a single ``argmax(per_spec_ngeom)`` anchor
    (the spec with the most *total* geoms), but ``override_geoms_for_body``
    fills a variant's geoms into the skeleton slots and **truncates at the
    slot count** — so any variant whose per-body geom count exceeds the
    anchor's for that body silently loses geoms (collision + visual + extent).

    A global-total anchor does NOT guarantee per-body coverage: e.g. a
    wine-glass ``bottom_watertight`` may carry many base geoms while the
    total-geom anchor's bottom is a 1-geom dummy. Because the anchor is chosen
    among the *sampled* specs, the truncation was sample-dependent (changing
    ``N_SUB_ENV`` changed which object anchored the skeleton). Provisioning
    each body to the per-body max removes that dependence entirely.
    """
    max_map = {}

    def _walk(b):
        n = len(list(b.geoms))
        if n > max_map.get(b.name, 0):
            max_map[b.name] = n
        for cb in b.bodies:
            _walk(cb)

    for spec in obj_spec_lst:
        root = spec.worldbody.first_body()
        if root is not None:
            _walk(root)
    return max_map


def _pad_skeleton_body_geoms(body, suffix, max_map, fallback_meshname, verbose=False):
    """Pad a copied skeleton ``body`` (name = ``<base><suffix>``) with disabled
    placeholder geoms up to ``max_map[base]`` so every variant's geoms fit
    without truncation in ``override_geoms_for_body``.

    Placeholders are inert: ``contype/conaffinity = 0`` (no collision),
    ``group = 4`` (the warp / CPU override path marks group-4 slots as
    ``dataid = -1`` / transparent), ``density = 0`` (no phantom mass/inertia),
    alpha 0. They reference a valid mesh so the skeleton compiles. A variant
    that USES a padded slot overwrites all of this via
    ``override_geoms_for_body``; a variant that doesn't keeps the disabled
    state — identical to the pre-existing "extra skeleton slot" convention.
    """
    base  = body.name[:-len(suffix)] if (suffix and body.name.endswith(suffix)) else body.name
    want  = int(max_map.get(base, 0))
    geoms = list(body.geoms)
    have  = len(geoms)
    if want > have:
        # Reuse an existing geom's mesh as the template when present (keeps a
        # same-object valid mesh); else fall back to any registered object mesh.
        tmpl = geoms[-1] if geoms else None
        tmpl_mesh = (tmpl.meshname if (tmpl is not None
                     and tmpl.type in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SDF)  # type: ignore
                     and tmpl.meshname) else fallback_meshname)
        for k in range(want - have):
            ng = body.add_geom(name=f"{body.name}_padslot_{have + k}")
            if tmpl_mesh:
                ng.type     = mujoco.mjtGeom.mjGEOM_MESH  # type: ignore
                ng.meshname = tmpl_mesh
            else:
                # No registered object mesh to reference → use a tiny sphere so
                # the skeleton still compiles (override replaces it for any
                # variant that actually uses this slot).
                ng.type = mujoco.mjtGeom.mjGEOM_SPHERE    # type: ignore
                ng.size = [1.0e-4, 0.0, 0.0]
            ng.contype     = 0
            ng.conaffinity = 0
            ng.group       = 4            # override → dataid=-1 / transparent
            ng.density     = 0.0          # no phantom mass / inertia
            ng.rgba        = [0.0, 0.0, 0.0, 0.0]
        if verbose:
            print_yellow(f"[skeleton pad] body '{body.name}': {have} → {want} geom slots")
    for cb in body.bodies:
        _pad_skeleton_body_geoms(cb, suffix, max_map, fallback_meshname, verbose=verbose)

def copy_sensors(parent_spec, src_spec, suffix=""):
    for src_sensor in src_spec.sensors:
        new_sensor          = parent_spec.add_sensor()
        new_sensor.type     = src_sensor.type
        new_sensor.name     = src_sensor.name + suffix
        new_sensor.objtype  = src_sensor.objtype
        new_sensor.objname  = src_sensor.objname + suffix
        if hasattr(src_sensor, 'cutoff'):
            new_sensor.cutoff = src_sensor.cutoff
        print(f"Verified: {new_sensor.name} (Type: {new_sensor.type}) -> {new_sensor.objname}")


_SELF_COLL_SUFFIX = "_self_coll"


def _gather_self_coll_templates(parent_spec):
    """Collect every self-collision template sensor from ``parent_spec``.

    A *self-collision template* is a contact sensor whose ``reftype`` is
    ``XBODY`` (i.e. uses ``subtree2`` as the second reference) and whose
    name ends in :data:`_SELF_COLL_SUFFIX` (``"_self_coll"``). Each hand's
    XML must declare one per body part; ``add_*_contact_sensors`` re-uses
    the template's ``objtype/objname/intprm/datatype/...`` so the only
    XML difference between generated sensors is the second reference
    (``subtree2`` → ``body2``).
    """
    templates = []
    # parent_spec.sensors mutates on every add → snapshot first.
    for s in parent_spec.sensors:
        if s.type != mujoco.mjtSensor.mjSENS_CONTACT:  # type: ignore
            continue
        if s.reftype != mujoco.mjtObj.mjOBJ_XBODY:  # type: ignore
            continue
        # Note: in a two-hand leader-follower scene the follower sensors carry a
        # **hand discriminator after the suffix**, e.g. ``..._self_coll_follower``.
        # Matching with endswith alone would drop every follower template, so no
        # obj/table contact sensors would be created for it (observed: zero
        # follower obj_contact sensors → the grasping handler's "contact families
        # must be parallel" assert prevented f_driver construction).
        if _SELF_COLL_SUFFIX not in s.name:
            continue
        templates.append({
            "name":      s.name,
            "objtype":   s.objtype,
            "objname":   s.objname,
            "intprm":    np.array(s.intprm).copy(),
            "datatype":  s.datatype,
            "needstage": s.needstage,
            "cutoff":    float(s.cutoff),
        })
    return templates


def _split_template_name(template_name: str):
    """``<stem>_self_coll[<hand_sfx>]`` → ``(stem, hand_sfx)``.

    ``hand_sfx`` is the tail that distinguishes hands in a leader-follower scene
    (e.g. ``_follower``); for a single hand or the leader it is empty, so the
    generated names are **byte-identical to the original ones**."""
    i = template_name.rindex(_SELF_COLL_SUFFIX)
    return template_name[:i], template_name[i + len(_SELF_COLL_SUFFIX):]


def _stem_from_template_name(template_name: str) -> str:
    """Strip the trailing ``_self_coll`` suffix from a template sensor name."""
    return _split_template_name(template_name)[0]


def _emit_contact_sensor(parent_spec, *, name, tpl, ref_body_name):
    """Helper: add a contact sensor with the same primary reference as ``tpl``
    but with ``body2 = ref_body_name`` (vs. the template's ``subtree2``).
    """
    new_s            = parent_spec.add_sensor()
    new_s.type       = mujoco.mjtSensor.mjSENS_CONTACT  # type: ignore
    new_s.name       = name
    new_s.objtype    = tpl["objtype"]
    new_s.objname    = tpl["objname"]
    new_s.reftype    = mujoco.mjtObj.mjOBJ_BODY  # type: ignore
    new_s.refname    = ref_body_name
    new_s.intprm     = tpl["intprm"]
    new_s.datatype   = tpl["datatype"]
    new_s.needstage  = tpl["needstage"]
    new_s.cutoff     = tpl["cutoff"]
    return new_s


# mjCONDATA datakind bit for the contact FORCE field (verified empirically:
# data="found pos normal" → intprm[0]=49; adding force → 51, i.e. +2). The
# enum is preferred when available; the literal is the compile-confirmed value.
try:
    _CONDATA_FORCE_BIT = int(mujoco.mjtConDataField.mjCONDATA_FORCE)  # type: ignore
except Exception:
    _CONDATA_FORCE_BIT = 2


def ensure_contact_force_channel(parent_spec, verbose=False):
    """Enable the FORCE data channel on **every** ``mjSENS_CONTACT`` sensor.

    MuJoCo packs a contact sensor as a fixed-order record
    ``[found, force, torque, dist, pos, normal, tangent]`` — only the enabled
    fields are emitted, in that canonical order. Our sensors ship with
    ``data="found pos normal"`` (``intprm[0]=49`` → 7 floats). Setting the
    FORCE bit (``intprm[0] |= 2`` → 51) makes the compiler emit
    ``[found(1) force(3) pos(3) normal(3)]`` = 10 floats, so each contact
    sensor now reports **its own** contact force — object / table / self each
    get a type-separated force instead of sharing one type-blind touch sensor.

    Idempotent: sensors that already have the FORCE bit are left unchanged.
    Must run BEFORE the spec is compiled / copied (``sensor_dim`` is derived
    from ``intprm`` at compile time).
    """
    n_set = 0
    for s in parent_spec.sensors:
        if int(s.type) != int(mujoco.mjtSensor.mjSENS_CONTACT):  # type: ignore
            continue
        ip = list(s.intprm)
        if (int(ip[0]) & _CONDATA_FORCE_BIT) == 0:
            ip[0] = int(ip[0]) | _CONDATA_FORCE_BIT
            s.intprm = ip
            n_set += 1
    if verbose:
        print_blue(f"[contact_force] FORCE channel enabled on {n_set} contact sensor(s)")
    return n_set


def add_object_contact_sensors(parent_spec, obj_body_names, verbose=False):
    """Based on the self-collision templates already defined in parent_spec
    (<contact ... subtree2="..." name="..._self_coll">), add a
    <contact ... body2="..."> sensor pointing at each object body.

    naming rule
        ``..._self_coll``  →  ``..._body_obj_contact``      (i == 0)
                           →  ``..._body_obj_contact_<i>``  (i >= 1)

    The only XML difference is ``subtree2`` (mjOBJ_XBODY) → ``body2`` (mjOBJ_BODY);
    the remaining attributes such as ``intprm`` (datakind/reduce/num),
    ``datatype`` and ``cutoff`` are copied verbatim from the source sensor.

    No-op if obj_body_names is empty (`n_obj_per_env=0`) or there are no
    template sensors.
    """
    if not obj_body_names:
        return

    templates = _gather_self_coll_templates(parent_spec)
    if not templates:
        if verbose:
            print_yellow(
                f"[obj_contact] no subtree2 + '{_SELF_COLL_SUFFIX}' contact "
                f"template found — skip"
            )
        return

    for i, body_name in enumerate(obj_body_names):
        name_suffix = "" if i == 0 else f"_{i}"
        for tpl in templates:
            _stem, _hand_sfx = _split_template_name(tpl["name"])
            new_name = _stem + "_body_obj_contact" + _hand_sfx + name_suffix
            _emit_contact_sensor(parent_spec, name=new_name, tpl=tpl, ref_body_name=body_name)
            if verbose:
                print_blue(
                    f"[obj_contact] add '{new_name}' "
                    f"site/body1='{tpl['objname']}' body2='{body_name}'"
                )


_ARM_PART_BODY_SUFFIX = "_arm_part"

def _walk_bodies(spec):
    """Yield every body in ``spec`` (recursively under worldbody).

    ``spec.bodies`` already returns bodies recursively, but we wrap it for
    forward-compatibility in case that semantics changes.
    """
    for body in spec.bodies:
        yield body

def disable_arm_part_collisions(parent_spec, verbose: bool = False) -> int:
    """Force every geom inside ``*_arm_part`` bodies to be **non-colliding**.

    Why:
        Each hand XML carries a coarse box geom on the forearm body
        (``<body name="..._arm_part"><geom type="box" contype="1"
        conaffinity="1" rgba="0 0 0 0.5"/>``). It's intended only as a
        rough touch-volume proxy for the forearm site, but with
        ``contype/conaffinity = 1`` it actively participates in
        collision detection. The box (~12×12×16 cm) sits ~10–20 cm away
        from the palm and can clip into objects/table while the visible
        fingers are far from any contact — producing the
        "hand grasps invisible object far from the fingers" symptom
        observed in `notebook/hand/03_warp_parallel/08_warp_contact_sensor.ipynb`.

    Fix:
        Set ``contype = 0`` and ``conaffinity = 0`` on every geom inside
        any body whose name ends with ``_arm_part``. The arm-touch
        ``<site>`` is kept untouched so the arm-touch sensor (which is
        fired by neighbouring contacts) still works as before.

    Returns:
        The number of geoms whose collision flags were cleared.
    """
    cleared = 0
    for body in _walk_bodies(parent_spec):
        if not body.name.endswith(_ARM_PART_BODY_SUFFIX):
            continue
        for geom in body.geoms:
            if int(geom.contype) == 0 and int(geom.conaffinity) == 0:
                continue
            geom.contype     = 0
            geom.conaffinity = 0
            cleared += 1
            if verbose:
                gname = geom.name or "<unnamed>"
                print_yellow(
                    f"[arm_part] disabled collision on body='{body.name}' "
                    f"geom='{gname}' (contype/conaffinity → 0)"
                )
    return cleared


def _resolve_table_target_body(parent_spec) -> str:
    """Pick the body name that the per-finger table-contact sensor should
    reference.

    Priority (matches scenes built by ``Env_Orchestrator``):
      1. ``base_table`` if present (``with_table=True`` floor + table scene).
      2. fallback to ``world`` (table-less scene where the floor geom is
         attached directly to the world body).
    """
    table_body_name = "base_table"
    has_table = False
    for body in parent_spec.worldbody.bodies:
        if body.name == table_body_name:
            has_table = True
            break
    return table_body_name if has_table else "world"


def add_table_contact_sensors(parent_spec, verbose=False):
    """In the same way as ``add_object_contact_sensors``, add sensors that
    capture hand ↔ table/floor contact.

    naming rule
        ``..._self_coll``  →  ``..._table_contact``  (single, no index)

    body2 selection rule
      * if the scene has a ``base_table`` body → ``body2="base_table"``
        (the ``with_table=True`` case; contact with the floor is not reported,
        but during actual grasping the hand only touches the table)
      * otherwise → ``body2="world"`` (table-less scene where the floor geom is
        attached to the world body)

    No-op if there are no template sensors (no ``_self_coll`` contact in the
    XML at all) or the target body cannot be identified.
    """
    templates = _gather_self_coll_templates(parent_spec)
    if not templates:
        if verbose:
            print_yellow(
                f"[table_contact] no subtree2 + '{_SELF_COLL_SUFFIX}' contact "
                f"template found — skip"
            )
        return

    target_body = _resolve_table_target_body(parent_spec)
    for tpl in templates:
        _stem, _hand_sfx = _split_template_name(tpl["name"])

        new_name = _stem + "_table_contact" + _hand_sfx
        _emit_contact_sensor(parent_spec, name=new_name, tpl=tpl, ref_body_name=target_body)
        if verbose:
            print_blue(
                f"[table_contact] add '{new_name}' "
                f"site/body1='{tpl['objname']}' body2='{target_body}'"
            )

def fill_passive_joints_from_equality(model, qpos, n_iter: int = 8):
    """In-place: fill equality-coupled **passive** joint qpos so that every
    ``mjEQ_JOINT`` constraint holds, given the **actuated** joints are already
    set. Returns ``qpos``.

    Needed because ``mj_forward`` does NOT project qpos onto equality
    constraints (it only computes constraint forces) — so any FK path that
    sets qpos directly and forwards (ghost / af-target target FK, taxonomy
    pose viz) leaves coupled distal joints (allex DIP/IP, inspire ``_2`` /
    thumb, shadow ``*J1``) un-driven and geometrically wrong for coupled hands.

    Handles BOTH authoring directions: the passive joint is whichever side of
    the equality has **no direct JOINT actuator**. When the passive joint is
    the polynomial's output side (``eq_obj1``) it is evaluated directly from
    the quartic ``eq_data[:5]``; when it is the input side (``eq_obj2``) a
    **linear** polycoef is inverted (nonlinear-inverse is skipped with the
    passive left as-is). Chained couplings (a passive joint referencing another
    passive joint — e.g. inspire ``thumb_3 = f(thumb_4 = g(thumb_2))``) converge
    via fixed-point iteration.

    Args:
        model:  ``mujoco.MjModel``.
        qpos:   ``np.ndarray`` shape ``(nq,)`` — modified in place. Actuated
                joint entries must already hold the target values.
        n_iter: fixed-point sweeps (default 8 — ample for the ≤2-deep chains
                in the current hands).
    """
    import numpy as _np
    actuated = set()
    for a in range(int(model.nu)):
        if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):  # type: ignore
            actuated.add(int(model.actuator_trnid[a, 0]))
    eqs = []
    for e in range(int(model.neq)):
        if int(model.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT):  # type: ignore
            continue
        eqs.append((int(model.eq_obj1id[e]), int(model.eq_obj2id[e]),
                    _np.asarray(model.eq_data[e, :5], dtype=_np.float64)))
    if not eqs:
        return qpos
    qa = model.jnt_qposadr
    q0 = model.qpos0
    for _ in range(int(n_iter)):
        for j1, j2, c in eqs:
            a1, a2 = j1 in actuated, j2 in actuated
            if a1 and a2:
                continue                                 # both actuated → nothing passive
            if a2 and not a1:
                passive, ref, direct = j1, j2, True      # passive = obj1 → direct eval
            elif a1 and not a2:
                passive, ref, direct = j2, j1, False     # passive = obj2 → invert
            else:
                passive, ref, direct = j1, j2, True       # both passive → obj1 is the output side
            pj, rj = int(qa[passive]), int(qa[ref])
            if direct:
                d = qpos[rj] - q0[rj]
                qpos[pj] = q0[pj] + c[0] + c[1]*d + c[2]*d*d + c[3]*d**3 + c[4]*d**4
            else:
                if abs(c[1]) < 1e-9 or max(abs(c[2]), abs(c[3]), abs(c[4])) > 1e-9:
                    continue                              # nonlinear inverse unsupported → leave
                qpos[pj] = q0[pj] + (qpos[rj] - q0[rj] - c[0]) / c[1]
    return qpos


def obj_pose_init(
    env,
    obj_names,
    obj_pose_criteria=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    xy_offset_range=[-0.10, 0.10],
    table_height=0.0,
    z_rot_sample: bool = False,
):
    """
    Optimized object pose initializer. Fast, clear, and minimizes memory usage.
    """
    obj_p_init = {}
    n_objs = len(obj_names)
    rng = np.random  # alias for speed

    xy_offsets = rng.uniform(xy_offset_range[0], xy_offset_range[1], (n_objs, 2))

    for idx, obj_n in enumerate(obj_names):
        # Copy base pose and compute position
        obj_pose = obj_pose_criteria
        obj_pos = np.copy(obj_pose[:3])
        rel_min = env.relative_obj_min[obj_n][:2]
        obj_pos[:2] += rel_min + xy_offsets[idx]
        obj_pos[2] = table_height

        # Use only valid verts for table Z alignment
        verts_main = env.obj_pcd_local_cache[obj_n]
        verts_nc = env.obj_pcd_local_cache.get(obj_n + '_non_collide', np.empty((0, 3)))
        # Mask and combine verts in place, avoiding unnecessary copies
        verts_concat = np.concatenate([verts_main, verts_nc], axis=0)
        valid_z = verts_concat[:, 2] > -10
        if np.any(valid_z):
            min_z = np.min(verts_concat[valid_z, 2])
            obj_pos[2] += abs(min_z) + 0.001  # 1mm clearance

        obj_p_init[obj_n] = obj_pos

        # Get joint address efficiently
        body = env.model.body(obj_n)
        jntadr = body.jntadr[0]
        if jntadr == -1:
            body = env.model.body("body_obj_" + obj_n)
            jntadr = body.jntadr[0]
        qposadr = env.model.jnt_qposadr[jntadr]

        env.data.qpos[qposadr:qposadr + 3] = obj_pos

        # Orientation: z-rot sampled if needed
        if z_rot_sample:
            z_rotation = rng.uniform(-np.pi, np.pi)
            obj_quat = r2quat(rpy2r(np.array([0, 0, z_rotation])))
        else:
            obj_quat = obj_pose_criteria[3:7]
        env.data.qpos[qposadr + 3 : qposadr + 7] = obj_quat

    env.data.qvel[:] = 0.0
    env.forward()
    return env, obj_p_init

def generate_target_axis_and_point(point_cloud, heading_direction=None,hand_center_T=None):
    """
    Generate a target axis and a corresponding midpoint for a grasp task based on the point cloud.

    Args:
        point_cloud (numpy.ndarray): Input point cloud of shape (N, 3).
        heading_direction (numpy.ndarray or None): Optional heading direction vector (3,).
            If None, a random normalized vector will be generated.

    Returns:
        dict: A dictionary containing:
            - "midpoint": Midpoint of the object (numpy.ndarray, shape (3,)).
            - "target_axis": Target heading axis (numpy.ndarray, shape (3,)).
            - "rotation_axis": Axis of rotation perpendicular to the target axis (numpy.ndarray, shape (3,)).
            - "sample_point": A randomly chosen point on the object (numpy.ndarray, shape (3,)).
    """
    assert len(point_cloud.shape) == 2 and point_cloud.shape[1] == 3, "Point cloud must have shape (N, 3)."

    # Generate or perturb the heading direction
    if heading_direction is None:
        heading_direction = np.random.uniform(-1, 1, 3)
    else:
        perturbation = np.random.uniform(-0.2, 0.2, 3)
        heading_direction = heading_direction + perturbation
    target_heading_direction = heading_direction / np.linalg.norm(heading_direction)
    
    # for target
    if hand_center_T is None:
        z_world = np.array([0, 0, 1])
    else:
        z_world = hand_center_T[:3,2]
        z_world += np.random.uniform(-0.2, 0.2, 3)
        z_world = z_world / np.linalg.norm(z_world)

    y_new = np.cross(z_world, target_heading_direction)
    y_new = y_new / np.linalg.norm(y_new)  # normalize
    z_new = np.cross(target_heading_direction, y_new)
    z_new = z_new / np.linalg.norm(z_new)  # normalize
    target_R = np.stack([target_heading_direction, y_new, z_new], axis=1)
    
    # Apply wrist rotation noise around X-axis using numpy
    wrist_rotation_noise = np.random.uniform(-np.pi / 6, np.pi / 6)
    target_R = target_R @ rpy2r([wrist_rotation_noise , 0, 0])
    # Generate a random rotation angle
    rotation_angle = np.random.uniform(0, 2 * np.pi)

    # Choose a random point from the point cloud
    sample_idx = np.random.randint(0, point_cloud.shape[0])
    sample_point = point_cloud[sample_idx]

    # Return the computed values
    return target_R, rotation_angle, sample_point, sample_idx


def sample_target_R_init_per_world(
    wrist_xpos:        np.ndarray,                     # (N, 3)  world-frame wrist xpos
    wrist_xmat:        np.ndarray,                     # (N, 3, 3) world-frame wrist xmat
    assignment,                                        # (N,) variant idx per world
    variant_pcd_cache: dict,                           # {variant_idx: {body_name: (n_pts, 3)}}
    grasping_obj_name: str,
    rh_hand_center:    np.ndarray,                     # (4, 4) wrist→hand_center
    out:               np.ndarray | None = None,       # (N, 3, 3) optional pre-alloc
    world_mask_np                       = None,        # (N,) bool; None → all worlds
) -> np.ndarray:
    """Per-world target rotation init for parallel grasp tasks.

    Mirrors the target sampling inside :func:`wrist_pose_init`, applied
    to a parallel-warp env. For each world ``w`` in ``world_mask_np`` (or
    all worlds when ``world_mask_np is None``)::

        wrist_T(w)       = pr2t(wrist_xpos[w], wrist_xmat[w])
        hand_center_T(w) = wrist_T(w) @ rh_hand_center
        heading_dir(w)   = hand_center_T(w)[:3, 0]            # palm/+X
        target_R(w), *_ = generate_target_axis_and_point(
                              variant_pcd_cache[asg[w]][obj],
                              heading_direction = heading_dir(w),
                              hand_center_T     = hand_center_T(w))

    The returned rotation is the *world-frame* target at sample time. To
    keep it attached to the object body during a roll, multiply by the
    live object rotation::

        target_R_world(w) = d.xmat[w, obj_body_id] @ target_R_init(w)

    Args:
        wrist_xpos:        ``(N, 3)`` world wrist xpos per world.
        wrist_xmat:        ``(N, 3, 3)`` world wrist xmat per world.
        assignment:        per-world variant index (used to pick the
                           correct body-local PCD).
        variant_pcd_cache: ``{variant_idx: {body_name: ndarray(n_pts, 3)}}``
                           — body-local PCD cache built by
                           ``SingleHandSubEnv.build_variant_pcd_cache``.
        grasping_obj_name: object body name to look up in the cache.
        rh_hand_center:    ``(4, 4)`` wrist→hand_center transform
                           (``HandUtils.rh_hand_center``).
        out:               optional pre-allocated ``(N, 3, 3)`` buffer to
                           write into. Cuts per-reset allocation in the RL
                           hot path. When ``None`` a fresh array is made.
        world_mask_np:     ``(N,)`` bool mask. Only ``True`` rows get
                           re-sampled; other rows are left at their
                           current value (must be valid in ``out``).

    Returns:
        ``(N, 3, 3)`` per-world target rotation. Identical to ``out`` when
        provided (in-place write).

    Notes:
      * ``generate_target_axis_and_point`` uses the **module-level**
        ``np.random`` for its noise / sample-point draws. Pass a seeded
        rng up-stream (e.g. ``np.random.seed`` once at handler init) if
        deterministic resets matter.
      * Worlds whose ``variant_pcd_cache`` row is missing fall back to a
        single-vertex PCD at the origin — the resulting ``target_R`` is
        still valid (the function only uses the PCD for the
        ``sample_point`` return which we discard).
    """
    N = int(wrist_xpos.shape[0])
    if out is None:
        out = np.zeros((N, 3, 3), dtype=np.float64)
    if world_mask_np is None:
        idxs = range(N)
    else:
        mask_np = np.asarray(world_mask_np, dtype=bool)
        if mask_np.size != N:
            raise ValueError(
                f"world_mask_np size {mask_np.size} != NWORLD {N}"
            )
        if not mask_np.any():
            return out
        idxs = np.where(mask_np)[0].tolist()

    # Reused fallback when a variant has no PCD cached. The PCD only
    # affects the discarded ``sample_point`` return; ``target_R`` itself
    # depends solely on ``heading_dir`` + ``hand_center_T``.
    pcd_fallback = np.zeros((1, 3), dtype=np.float32)

    for w in idxs:
        wrist_p = np.asarray(wrist_xpos[w]).reshape(3)
        wrist_R = np.asarray(wrist_xmat[w]).reshape(3, 3)
        wrist_T = pr2t(wrist_p, wrist_R)
        hand_center_T = wrist_T @ rh_hand_center
        heading_dir   = hand_center_T[:3, 0]

        variant_idx = int(assignment[w])
        pcd_local = variant_pcd_cache.get(variant_idx, {}).get(grasping_obj_name)
        if pcd_local is None or len(pcd_local) == 0:
            pcd_local = pcd_fallback

        target_R, _rot, _pt, _idx = generate_target_axis_and_point(
            pcd_local,
            heading_direction = heading_dir,
            hand_center_T     = hand_center_T,
        )
        out[w] = target_R
    return out


# Hand-side agnostic: it only reads ``rh_hand_center[:3, :3]``, and a mirrored
# left hand (hand_info/mirror.py) keeps the same rotation block (only the Y
# translation flips), so the same law applies to ``hand_type == 'left'``.
def _make_wrist_pose_y_up_palm_facing(target_world, wrist_pos, hand_util):
    """Special-init wrist rotation:
        - wrist +Y == world +Z (the wrist Y axis points up)
        - palm direction (hand-center +X in wrist frame, ≈ rh_hand_center[:3,0])
          points toward ``target_world`` (in terms of its horizontal component).

    In palm_in_wrist = (a, b, c), b ≈ 0 (the Y component of rh_hand_center[:3,0]
    is ~0 for every hand). The palm direction therefore lies almost in the wrist
    XZ plane, so even with wrist Y pinned to world up the palm can be pointed in
    any horizontal direction.

    R_wrist has columns = [X, Y=worldUp, Z] with X/Z both in the world XY plane:
    a one-DoF (α) rotation. α is chosen so the palm's horizontal direction
    matches the wrist→target horizontal direction.
    """
    R_wc = hand_util.rh_hand_center[:3, :3]
    palm_in_wrist = R_wc[:, 0]      # palm direction (wrist frame)
    a, _b, c = palm_in_wrist        # b assumed ~0 (vertical leak ignored)

    palm_world = np.asarray(target_world) - np.asarray(wrist_pos)
    px, py = palm_world[0], palm_world[1]
    horiz_norm = np.sqrt(px * px + py * py)
    if horiz_norm < 1e-6:
        # object directly above/below the wrist → default to +X
        px, py = 1.0, 0.0
    else:
        px /= horiz_norm
        py /= horiz_norm

    denom = a * a + c * c
    if denom < 1e-9:
        # palm parallel to wrist Y — the assumption breaks. fallback: identity-ish
        cos_a, sin_a = 1.0, 0.0
    else:
        cos_a = (a * px - c * py) / denom
        sin_a = (c * px + a * py) / denom
        norm = np.sqrt(cos_a * cos_a + sin_a * sin_a)
        if norm > 1e-9:
            cos_a /= norm
            sin_a /= norm

    # R_wrist columns: X = (cos α, sin α, 0), Y = world up, Z = X × Y
    R_wrist = np.column_stack([
        np.array([cos_a,  sin_a, 0.0]),
        np.array([0.0,    0.0,   1.0]),
        np.array([sin_a, -cos_a, 0.0]),
    ])
    return R_wrist


def wrist_pose_init(env, hand_util, hand_wrist_name, grasping_obj_name,
                    return_simple=False, table_height=0.0,
                    palm_facing_y_up_prob=0.2,
                    y_up_margin_range=(0.03, 0.10)):
    """Wrist (free joint) + mocap initialization.

    Two init branches:
      * **Default (1 - palm_facing_y_up_prob)**: sample with ``get_wrist_init_pos``
        so the hand-center faces the object, plus ±30° rpy noise.
      * **Special (palm_facing_y_up_prob)**: pin the wrist Y axis exactly to
        world up [0,0,1] and choose only the yaw so the palm faces the object.
        Wrist xy = obj_xy − (obj_xy_aabb_radius + Uniform(``y_up_margin_range``))·
        [cos azim, sin azim] — placed slightly beyond the object AABB.
        Wrist z = ``table_height`` + Uniform(0.10, 0.25). No noise is applied
        (preserves the Y-up constraint).
    """
    hand_center = hand_util.rh_hand_center[:3,3]
    wrist_T_center = hand_util.rh_hand_center
    
    if grasping_obj_name:
        transformed_vertics = env.get_obj_pcd(grasping_obj_name)
    else:
        transformed_vertics = np.array([[0,0,0]])

    p_init, target_center, heading_dir, R_init = get_wrist_init_pos(transformed_vertics,
                                                xyz_range=np.array([[-0.4, 0.4], [-0.4, 0.4], [0.1, 0.5]]),
                                                )



    hand_center_T = pr2t(p_init, R_init)
    p_w2c,R_w2c = t2pr(hand_util.rh_hand_center)
    R_inv = R_w2c.T
    p_inv = -R_inv @ p_w2c
    T_inv = pr2t(p_inv, R_inv)
    wrist_T = hand_center_T @ T_inv
    wirst_p_init, wrist_R_init = t2pr(wrist_T)

    use_special = (np.random.rand() < palm_facing_y_up_prob)
    if use_special:
        # sample a point on the object surface as target → the palm faces that point
        sample_idx = np.random.randint(0, transformed_vertics.shape[0])
        target_world = transformed_vertics[sample_idx]

        obj_center_xy = transformed_vertics.mean(axis=0)[:2]

        # Half-diagonal of the object XY AABB: a conservative radius that keeps the
        # palm from being placed inside the object from any approach direction
        # (covers the worst case, the full diagonal).
        xy = transformed_vertics[:, :2]
        xy_min = xy.min(axis=0)
        xy_max = xy.max(axis=0)
        obj_xy_aabb_radius = float(np.linalg.norm((xy_max - xy_min) * 0.5))

        azimuth = np.random.uniform(-np.pi, np.pi)
        margin  = np.random.uniform(y_up_margin_range[0], y_up_margin_range[1])
        horiz_dist = obj_xy_aabb_radius + margin
        wirst_p_init = np.array([
            obj_center_xy[0] - horiz_dist * np.cos(azimuth),
            obj_center_xy[1] - horiz_dist * np.sin(azimuth),
            float(table_height) + np.random.uniform(0.10, 0.25),
        ])
        wrist_R_init = _make_wrist_pose_y_up_palm_facing(
            target_world, wirst_p_init, hand_util,
        )
        # downstream consistency: hand_center_T = wrist_T @ rh_hand_center
        hand_center_T = pr2t(wirst_p_init, wrist_R_init) @ wrist_T_center
        heading_dir   = hand_center_T[:3, 0]   # palm direction (world)
        # no noise — preserves wrist Y == world up
    else:
        # ±30° per-axis uniform noise → variety across resets (the X-axis facing
        # only wobbles within ±30°, so the hand-center X still roughly faces the object)
        wrist_noise_max_rad = (np.pi / 180.0) * 30.0
        wrist_R_init = wrist_R_init @ rpy2r(np.array([
            np.random.uniform(-wrist_noise_max_rad, wrist_noise_max_rad),
            np.random.uniform(-wrist_noise_max_rad, wrist_noise_max_rad),
            np.random.uniform(-wrist_noise_max_rad, wrist_noise_max_rad),
        ]))

    wrist_R_quat = r2quat(wrist_R_init)

    # Only the upper point cloud extracted from transformed_vertics needs to be passed here.
    target_R_init, target_rot_wrist, target_p_init, target_sample_idx  \
        = generate_target_axis_and_point(transformed_vertics, 
                                        heading_direction=heading_dir,
                                        hand_center_T=hand_center_T,
                                        )
    # hand qpos idx hard coding 
    jntadr  = env.model.body(hand_wrist_name).jntadr[0]
    qposadr = env.model.jnt_qposadr[jntadr]
    env.data.qpos[qposadr:qposadr+3] = wirst_p_init 
    env.data.qpos[qposadr+3:qposadr+7] = wrist_R_quat
    env.set_p_mocap(mocap_name=hand_util.rh_mocap_name,p=wirst_p_init)
    env.set_R_mocap(mocap_name=hand_util.rh_mocap_name,R=wrist_R_init)

    env.forward() 
    if return_simple:
        return env, wrist_T_center, wirst_p_init, wrist_R_init, target_p_init, target_R_init, target_sample_idx
    else:
        return env, wrist_T_center, wirst_p_init, wrist_R_init, target_center, R_init, hand_center, \
                target_R_init, target_rot_wrist, target_p_init, target_sample_idx

# Right-hand convention
def get_wrist_init_pos(vertics:np.ndarray,
                             xyz_range=np.array([[-1, 1], [-1, 1], [0.65, 1]]),
                             roll_range=(-np.pi, np.pi),
                             wrist_T_center=np.eye(4),
                             ):
    """Hand-center pose sampler.

    - **X axis**: aligned from the hand-center position toward a sampled point
      on the object (i.e. +X of the hand-center frame always points at the object).
    - **Y/Z axes**: the roll angle about the X axis is sampled uniformly from
      ``roll_range``. The previous implementation forced ``z_axis[2] >= 0``,
      locking the hand into a top-down pose and ruling out side/inverted poses.
      Freeing the roll as a self-rotation covers every wrist orientation.
    """
    origin = vertics.mean(axis=0)

    current_direction_x = np.random.uniform(origin[0] + xyz_range[0][0], origin[0] + xyz_range[0][1], (1, 1))
    current_direction_y = np.random.uniform(origin[1] + xyz_range[1][0], origin[1] + xyz_range[1][1], (1, 1))
    current_direction_z = np.random.uniform(origin[2] + xyz_range[2][0], origin[2] + xyz_range[2][1], (1, 1))
    current_direction = np.concatenate([current_direction_x, current_direction_y, current_direction_z], axis=1)
    current_direction = current_direction - origin
    current_direction = current_direction / np.linalg.norm(current_direction, axis=1, keepdims=True)

    sample_idx = np.random.randint(0, vertics.shape[0])
    target_point = vertics[sample_idx]

    pos = (target_point + 0.25*current_direction).reshape(3,)
    bias = target_point

    # X axis: hand-center → target (looks exactly at the object)
    dir_ = target_point - pos
    x_axis = dir_ / np.linalg.norm(dir_)

    # Helper vector for building a reference orthonormal frame orthogonal to X.
    # If X is nearly parallel to world up, switch reference to avoid degeneracy.
    world_up = np.array([0., 0., 1.])
    if abs(np.dot(world_up, x_axis)) > 0.999:
        world_up = np.array([1., 0., 0.])
    ref_y = world_up - np.dot(world_up, x_axis) * x_axis
    ref_y = ref_y / np.linalg.norm(ref_y)
    ref_z = np.cross(x_axis, ref_y)
    ref_z = ref_z / np.linalg.norm(ref_z)
    R_base = np.column_stack([x_axis, ref_y, ref_z])

    # Roll about the X axis: uniform sample within roll_range → free wrist orientation
    roll = np.random.uniform(roll_range[0], roll_range[1])
    c, s = np.cos(roll), np.sin(roll)
    Rx = np.array([[1, 0,  0],
                   [0, c, -s],
                   [0, s,  c]])
    R_init = R_base @ Rx

    R_init = R_init @ wrist_T_center[:3,:3].T
    return pos, bias, dir_, R_init

def compute_bps_feature(basis_points, obj_pcd, wrist_p, wrist_R): 
    """ 
    basis_points : naive BPS basis points loaded from data (M,3)
    obj_pcd : point cloud in global coordinates (N,3)
    wrist_p : wrist center p (3)
    wrist_R : wrist center R (3,3)

    return : bps_feature in wrist frame (M,3)
    """
    obj_pcd_in_wrist = np.dot(obj_pcd - wrist_p, wrist_R)  

    diff = basis_points[:, None, :] - obj_pcd_in_wrist[None, :, :] 

    # 2. sum of squared distances (M, N)
    dist_sq = np.sum(diff**2, axis=-1)
    
    # 4. minimum distance
    min_dist = np.sqrt(np.min(dist_sq, axis=-1))

    # 5. distance normalization
    gamma= 25.0 
    bps_features = np.exp(-gamma * min_dist) 

    return bps_features

def update_hand_pose(env, slider, dummy_val, wrist_name, mocap_name, p_init, R_init, ctrl_idxs, ctrl_names, ctrl_mins, ctrl_maxs):
    slider_vals = slider.get_values()
    slider_vals[:6] -= dummy_val 
    xyz, rpy, hand_ctrl = slider_vals[:3].copy(), slider_vals[3:6].copy(), slider_vals[6:].copy()
    
    # remove tiny residual noise
    xyz[np.abs(xyz) < 1e-4] = 0.0
    rpy[np.abs(rpy) < 1e-4] = 0.0

    wrist_p, wrist_R = t2pr(env.get_T(wrist_name, type="body"))

    delta_xyz = np.dot(wrist_R,xyz) 
    p_init = delta_xyz + wrist_p 

    delta_r_mat = rpy2r(rpy) 
    R_init = R_init @ delta_r_mat

    env.set_pR_mocap(mocap_name=mocap_name, p=p_init, R=R_init)
    hand_ctrl = np.clip(hand_ctrl, ctrl_mins, ctrl_maxs)
    env.data.ctrl[ctrl_idxs] = hand_ctrl

    slider_vals[:6] = 0.0
    slider.set_values(slider_vals) 

    return p_init, R_init, slider

# Single Hand Utils Class
class HandUtils: 
    # Container class for hand information
    def __init__(self, hand_name, hand_type, home_dir): 
        self.hand_name = hand_name 
        self.hand_cfg = get_hand_cfg(hand_name, hand_type) 
        self.hand_type = self.hand_cfg.Training.hand_type 

        self.hand_xml_path = self.hand_cfg.Training.hand_xml_path
        
        self.home_dir = home_dir
        self.ri_package_asset_dir = epath.Path(home_dir) / "asset"
        self.ri_package_hand_asset_dir = self.ri_package_asset_dir / "dextrous_hand"

        # Hand info import — every hand authors ``rh_info`` (+ taxonomy). A
        # left hand additionally needs ``lh_info`` in the same package (see
        # hand_info/mirror.py); we fail loudly instead of silently running a
        # left asset with right-hand names.
        import importlib
        _pkg = {"inspire": "inspire", "tesollo": "tesollo",
                "allegro": "allegro", "shadow": "shadow", "robotis_sh5": "robotis",
                "wuji_hand2": "wuji_hand2"}.get(hand_name)
        if _pkg is None:
            raise ValueError(f"hand_info for '{hand_name}' is not available")
        rh_taxonomy_annotation = importlib.import_module(
            f"grit.util.hand_info.{_pkg}.{hand_name}_taxonomy_annotation")
        rh_info = importlib.import_module(f"grit.util.hand_info.{_pkg}.rh_info")

        if self.hand_type == "left":
            try:
                lh_info = importlib.import_module(f"grit.util.hand_info.{_pkg}.lh_info")
            except ModuleNotFoundError as e:
                raise NotImplementedError(
                    f"hand_type='left' for '{hand_name}' needs "
                    f"grit/util/hand_info/{_pkg}/lh_info.py (see hand_info/mirror.py)") from e
            from grit.util.hand_info import mirror as _mirror
            rh_taxonomy_annotation = _mirror.mirror_taxonomy(
                rh_taxonomy_annotation, lh_info.NAME_SUBS,
                qpos_sign=getattr(lh_info, "QPOS_MIRROR_SIGN", None),
                mirror_axis=getattr(lh_info, "MIRROR_AXIS", 1))   # type: ignore
            rh_info = lh_info
        elif self.hand_type != "right":
            raise ValueError(f"unknown hand_type '{self.hand_type}' (expected 'right'/'left')")

        self.rh_info = rh_info # type: ignore
        for attr in dir(rh_info): # type: ignore
            if not attr.startswith("__") and not callable(getattr(rh_info, attr)): # type: ignore
                setattr(self, attr, getattr(rh_info, attr)) # type: ignore

        # anchor_body_name: use the value if rh_info specifies one (e.g. robotis =
        # 'finger_r_link3_2'), otherwise default to the palm/base link (rh_af_parts[0]).
        # af_geom.build_anchor_tip_pairs resolves this name to an af_names index.
        if getattr(self, "anchor_body_name", None) is None:
            _af_parts = getattr(self, "rh_af_parts", None)
            self.anchor_body_name = _af_parts[0] if _af_parts else None

        self.rh_taxonomy_annotation = rh_taxonomy_annotation  # type: ignore
        self.set_hand_taxonomy_info() 

        # Use palm sensor   
        self.palm_sensor_idx = self.palm_sensor_idx 
        self.fore_arm_sensor_idx = self.fore_arm_sensor_idx  
        self.finger_motor_num = self.finger_motor_num  
        self.finger_link_num = self.finger_link_num 
        self.hand_qvel_num = self.hand_qvel_num  
       
        self.hand_qpos_dim = 7 + self.finger_link_num  # 7 (mocap) + finger joints
        
        self.tip_body_idx = self.tip_body_idx 
        self.tip_end_name = self.tip_end_name  
        self.ctrl_joint_idx = self.ctrl_joint_idx 
        self.ctrl_joint_idx_array = np.array(self.ctrl_joint_idx)

        self.wrist_transl_scale = self.wrist_transl_scale 
        self.wrist_rot_scale = self.wrist_rot_scale 
        self.finger_action_scale = self.finger_action_scale 


    def set_hand_taxonomy_info(self):
        self.taxonomy_name_list                     = self.rh_taxonomy_annotation.taxonomy_name_list # type: ignore
        self.taxonomy_qpos_array                    = np.asarray([getattr(self.rh_taxonomy_annotation, i)['qpos'] for i in self.taxonomy_name_list])
        self.taxonomy_do_finger_tip_array           = np.asarray([getattr(self.rh_taxonomy_annotation, i)['do_finger_tip'] for i in self.taxonomy_name_list])
        self.taxonomy_specific_finger_names_array   = [getattr(self.rh_taxonomy_annotation, i)['specific_finger_names'] for i in self.taxonomy_name_list]
        self.taxonomy_specific_ctrl_names_array     = [getattr(self.rh_taxonomy_annotation, i)['specific_ctrl_names'] for i in self.taxonomy_name_list]
        self.taxonomy_specific_sensor_names_array   = [getattr(self.rh_taxonomy_annotation, i)['specific_sensor_names'] for i in self.taxonomy_name_list]
        self.taxonomy_n_specific_sensor_array       = np.array([len(getattr(self.rh_taxonomy_annotation, i)['specific_sensor_names']) for i in self.taxonomy_name_list])

        # Important: contact direction
        self.taxonomy_hand_face_dir_idx_in_mat_array    = np.array([getattr(self.rh_taxonomy_annotation, i)['hand_face_dir_idx_in_mat'] for i in self.taxonomy_name_list])
        self.taxonomy_hand_face_dir_sign_array          = np.array([getattr(self.rh_taxonomy_annotation, i)['hand_face_dir_sign'] for i in self.taxonomy_name_list])
        self.taxonomy_hand_af_dir_idx_in_mat_array      = np.array([getattr(self.rh_taxonomy_annotation, i)['hand_af_dir_idx_in_mat'] for i in self.taxonomy_name_list])
        self.taxonomy_hand_af_dir_sign_array            = np.array([getattr(self.rh_taxonomy_annotation, i)['hand_af_dir_sign'] for i in self.taxonomy_name_list]) 




# ──────────────────────────────────────────────────────────────────────────
# Two hands (right + left) in ONE notebook scene
# ──────────────────────────────────────────────────────────────────────────
# ``merge_mjcfs`` only writes ``<include>`` lines, so a right- and a left-hand
# XML of the same hand collide on shared names (default classes, mesh /
# material / texture assets, the ``track`` camera). The training pipeline never
# needs this (one scene per hand), but the 01_hand_setup notebooks want both
# hands side by side. ``merge_hand_mjcfs`` writes a suffixed copy of every hand
# XML after the first (shared names → ``<name>_h<k>``; body / joint / site /
# sensor names are already side-specific) and offsets that hand's mocap + palm
# root along +Y so the hands do not overlap even after ``env.reset()``.

def suffix_mjcf_shared_names(xml_text: str, suffix: str) -> str:
    """Rename default classes, assets (mesh/material/texture) and cameras of an
    MJCF text with ``suffix`` and patch every reference to them."""
    import re as _re
    names = {"class": set(), "mesh": set(), "material": set(), "texture": set(), "camera": set()}
    for m in _re.finditer(r'<default\s+class="([^"]+)"', xml_text):
        names["class"].add(m.group(1))
    for tag in ("mesh", "material", "texture"):
        for m in _re.finditer(r'<%s\b([^>]*)>' % tag, xml_text):
            attrs = m.group(1)
            nm = _re.search(r'\bname="([^"]+)"', attrs)
            if nm:
                names[tag].add(nm.group(1))
            elif tag == "mesh":
                f = _re.search(r'\bfile="([^"]+)"', attrs)
                if f:   # implicit mesh name = file stem
                    stem = os.path.splitext(os.path.basename(f.group(1)))[0]
                    names[tag].add(stem)
                    xml_text = xml_text.replace(m.group(0), m.group(0)[:-1 - (1 if m.group(0).endswith("/>") else 0)]
                                                + f' name="{stem}"' + ("/>" if m.group(0).endswith("/>") else ">"), 1)
    for m in _re.finditer(r'<camera\s[^>]*\bname="([^"]+)"', xml_text):
        names["camera"].add(m.group(1))

    def ren(attr_names, pool):
        nonlocal xml_text
        for nm in sorted(pool, key=len, reverse=True):
            for attr in attr_names:
                xml_text = _re.sub(r'\b%s="%s"' % (attr, _re.escape(nm)), f'{attr}="{nm}{suffix}"', xml_text)
    ren(("class", "childclass"), names["class"])
    ren(("name", "mesh"), names["mesh"])            # <mesh name=…> and geom mesh=…
    ren(("name", "material"), names["material"])
    ren(("name", "texture"), names["texture"])
    ren(("name",), names["camera"])
    # ``name="X_h2"`` renames above also hit bodies/sites that happen to share a
    # mesh/material name — undo for tags that are not asset declarations.
    for tag in ("body", "site", "joint", "sensor", "touch", "contact", "actuator", "position", "general", "motor", "weld", "fixed"):
        xml_text = _re.sub(r'(<%s\s[^>]*\bname=")([^"]+)%s"' % (tag, _re.escape(suffix)), r'\1\2"', xml_text)
    return xml_text


def suffix_colliding_names(xml_text: str, other_xml_text: str, suffix: str) -> str:
    """Rename every ``name="X"`` of ``xml_text`` that also appears in
    ``other_xml_text`` (side-agnostic leftovers such as inspire's
    ``wrist_axis_align`` body or shadow's ``grasp_site``) and patch the
    reference attributes (body1/body2/site/joint/joint1/joint2/subtree2/target)."""
    import re as _re
    mine = set(_re.findall(r'\bname="([^"]+)"', xml_text))
    other = set(_re.findall(r'\bname="([^"]+)"', other_xml_text))
    for nm in sorted(mine & other, key=len, reverse=True):
        if nm.endswith(suffix):
            continue
        for attr in ("name", "body1", "body2", "site", "joint", "joint1", "joint2", "subtree2", "target", "body"):
            xml_text = _re.sub(r'\b%s="%s"' % (attr, _re.escape(nm)), f'{attr}="{nm}{suffix}"', xml_text)
    return xml_text


def offset_hand_root_in_mjcf(xml_text: str, dy: float) -> str:
    """Shift the mocap body and the welded palm root of a Grit hand XML by
    ``dy`` along Y (pos ``x y z`` → ``x y+dy z``)."""
    import re as _re
    w = _re.search(r'<weld\s+body1="([^"]+)"\s+body2="([^"]+)"', xml_text)
    if not w:
        return xml_text
    for body in (w.group(1), w.group(2)):
        def _shift(m):
            x, y, z = (float(v) for v in m.group(2).split())
            return f'{m.group(1)}pos="{x:g} {y + dy:g} {z:g}"'
        xml_text = _re.sub(r'(<body\s+name="%s"[^>]*?\s)pos="([^"]+)"' % _re.escape(body), _shift, xml_text)   # all occurrences (comments too)
    return xml_text


def merge_hand_mjcfs(included_mjcf_files, output_xml_path, hand_y_gap: float = 0.4, **merge_kw):
    """Drop-in for ``merge_mjcfs`` that accepts several hand XMLs (right + left).

    Non-hand files (floor, table) pass through. The first hand is included
    verbatim; each further hand is written as ``_multi_h<k>_<file>`` next to
    its original (mesh paths stay valid) with shared names suffixed and its
    root shifted by ``k * hand_y_gap`` along +Y.
    """
    from grit.util.sim_core.mjcf import merge_mjcfs as _merge
    files, k, first_txt = [], 0, ""
    for f in included_mjcf_files:
        f = str(f)
        is_hand = "dextrous_hand" in f.replace("\\", "/")
        if not is_hand or k == 0:
            files.append(f)
            if is_hand:
                first_txt = open(f).read(); k += 1
            continue
        txt = open(f).read()
        txt = suffix_mjcf_shared_names(txt, f"_h{k + 1}")
        txt = suffix_colliding_names(txt, first_txt, f"_h{k + 1}")
        txt = offset_hand_root_in_mjcf(txt, k * hand_y_gap)
        out = os.path.join(os.path.dirname(f), f"_multi_h{k + 1}_" + os.path.basename(f))
        open(out, "w").write(txt)
        files.append(out); k += 1
    return _merge(included_mjcf_files=files, output_xml_path=output_xml_path, **merge_kw)
