"""Left-hand ``*_info`` / taxonomy derivation from the right-hand source.

Every ``hand_info/<hand>/rh_info.py`` is the single hand-authored source of
truth (body / site / sensor names, hand-center pose, index tables). A left
hand asset that is a *mirror* of the right one only differs by

  1. a name convention (``_r_`` → ``_l_``, ``_rh_`` → ``_lh_`` …), and
  2. the hand-center pose, whose translation is reflected across the
     mirror plane (rotation stays: the labeling recipe in
     ``notebook/hand/01_hand_setup/09_label_hand_center.ipynb`` zeroes the
     lateral (Y) component of the heading, so the frame is mirror-invariant).

Rather than hand-copying 150 lines of names, ``hand_info/<hand>/lh_info.py``
calls :func:`mirror_info` with the hand's substitution rules and overrides
anything that is not a pure rename. The attribute names keep their ``rh_*``
prefix on purpose — ``HandUtils`` copies them verbatim and the whole
codebase reads ``hand_util.rh_mocap_name`` etc. as *generic* names.
"""
from __future__ import annotations

import types
from typing import Iterable, Sequence, Tuple

import numpy as np

def substitute(value, subs: Sequence[Tuple[str, str]]):
    """Apply ordered ``(old, new)`` substring rules to a str / nested list."""
    if isinstance(value, str):
        for old, new in subs:
            value = value.replace(old, new)
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(substitute(v, subs) for v in value)
    return value


def mirror_info(rh_info: types.ModuleType,
                subs: Sequence[Tuple[str, str]],
                *,
                mirror_axis: int = 1,
                overrides: dict | None = None) -> types.SimpleNamespace:
    """Build a left-hand info namespace from ``rh_info``.

    Args:
        rh_info:      the right-hand module.
        subs:         ordered substring rules applied to every str /
                      list-of-str attribute (numbers / arrays untouched).
        mirror_axis:  wrist-frame axis the left asset is reflected across
                      (robotis: Y). ``rh_hand_center[mirror_axis, 3]`` is
                      negated; the rotation block is kept (see module doc).
        overrides:    explicit attribute values that win over the derivation.
    """
    ns = types.SimpleNamespace()
    for attr in dir(rh_info):
        if attr.startswith("__"):
            continue
        val = getattr(rh_info, attr)
        if callable(val) or isinstance(val, types.ModuleType):
            continue
        setattr(ns, attr, substitute(val, subs))

    hc = np.array(getattr(rh_info, "rh_hand_center"), dtype=float, copy=True)
    hc[mirror_axis, 3] *= -1.0
    ns.rh_hand_center = hc

    ns.NAME_SUBS = tuple(subs)
    ns.MIRROR_AXIS = int(mirror_axis)
    for k, v in (overrides or {}).items():
        setattr(ns, k, v)
    return ns


_TAX_NAME_KEYS = ("specific_finger_names", "specific_sensor_names", "specific_ctrl_names")


def mirror_taxonomy(rh_tax: types.ModuleType,
                    subs: Sequence[Tuple[str, str]],
                    qpos_sign: Sequence[float] | None = None,
                    mirror_axis: int = 1) -> types.SimpleNamespace:
    """Mirror a ``*_taxonomy_annotation`` module.

    * name lists (``specific_*_names``) are renamed with ``subs``;
    * ``qpos`` is multiplied element-wise (on its trailing ``len(qpos_sign)``
      entries) by ``qpos_sign`` — the per-joint sign convention of the
      mirrored asset, authored in ``lh_info.QPOS_MIRROR_SIGN`` and verified by
      an FK sweep (a mirrored joint has a mirrored range, e.g. R[-pi,0] →
      L[0,pi]). ``None`` keeps qpos verbatim;
    * ``hand_face_dir_* / hand_af_dir_*`` index a body-frame axis + sign.
      A mirrored body keeps the two in-plane axes but its ``mirror_axis``
      column is reflected, so every entry whose idx == ``mirror_axis`` gets
      its sign negated (robotis: the ``lateral`` family points index/middle
      contact faces along body Y).
    """
    sign = None if qpos_sign is None else np.asarray(qpos_sign, dtype=float)
    ns = types.SimpleNamespace()
    names: Iterable[str] = getattr(rh_tax, "taxonomy_name_list")
    ns.taxonomy_name_list = list(names)
    for attr in dir(rh_tax):
        if attr.startswith("__"):
            continue
        val = getattr(rh_tax, attr)
        if isinstance(val, dict):
            val = dict(val)
            for k in _TAX_NAME_KEYS:
                if k in val:
                    val[k] = substitute(val[k], subs)
            if sign is not None and "qpos" in val:
                q = np.array(val["qpos"], dtype=float, copy=True)
                n = min(len(sign), q.shape[0])      # tolerate qpos/sign length mismatch (e.g. shadow 20 vs 22)
                q[-n:] *= sign[-n:]
                val["qpos"] = q
            for stem in ("hand_face_dir", "hand_af_dir"):
                ik, sk = f"{stem}_idx_in_mat", f"{stem}_sign"
                if ik in val and sk in val:
                    val[sk] = [(-sg if ix == mirror_axis else sg)
                               for ix, sg in zip(val[ik], val[sk])]
        setattr(ns, attr, val)
    return ns
