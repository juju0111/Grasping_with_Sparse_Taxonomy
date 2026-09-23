"""Procedurally generated **primitive objects** (Play2Perfect-style).

Motivation
----------
The dataset path (:func:`grit.util.hand_utils.build_obj_spec_lst`) materialises
objects from mesh XMLs under ``asset/object``. For *play pretraining* we instead
want an unbounded supply of cheap, procedurally generated objects.

Recipe — Play2Perfect (arXiv 2606.26428, App. D "Pretraining Environment and
Procedural Objects")::

    Each object is formed by rigidly combining two cuboid or capsule
    primitives. The primary component defines the graspable region … A
    secondary component is attached near one end … We independently randomize
    the component densities … This produces broad variation in geometry, mass,
    center of mass, and inertia.

The paper's published ranges (primary length/cross-section, secondary
length/cross-section, densities 300–600 / 300–2000 kg/m³) are the defaults in
:class:`PrimitiveSampleCfg`, with the *lengths rescaled to the dexterous hand in
this repo* — the paper's arm+hand system handles much larger parts, and its PDF
text renders decimal points inconsistently, so treat its cm figures as a shape
recipe rather than exact numbers. Every range is a constructor argument.

Why a fixed slot layout
-----------------------
``mujoco_warp`` keeps ``Model.geom_type`` as a **1-D ``(ngeom,)`` array** — it is
NOT per-world overridable (unlike ``geom_dataid`` / ``geom_size`` / ``geom_pos``,
which are ``(nworld, ngeom, …)``). The dataset path sidesteps this by making
every object a MESH geom and swapping ``geom_dataid`` per world. Native
primitives cannot do that: a slot's *type* is baked into the shared skeleton.

So every generated spec exposes the **same ordered slot layout** — by default
``("box", "capsule", "box", "capsule")`` = (primary-box, primary-capsule,
secondary-box, secondary-capsule) — and each object activates only the slots
matching the kinds it drew. The unused slots are emitted as inert placeholders
(``contype=conaffinity=0``, ``group=4``, ``density=0``, tiny size), which is
exactly the convention ``SingleHandSubEnv`` already uses: ``group == 4`` marks a
slot as disabled, and ``heterogeneous_env_setup`` banishes it per world.
Result: per-variant primitive **type** diversity with native primitive
collisions and zero changes to the skeleton / variant / warp override pipeline.

Usage::

    from grit.util import primitive_object as prim

    provider = prim.make_obj_spec_provider(seed=0, cfg=prim.PrimitiveSampleCfg())
    orchestrator.build_sub_env(with_mjwarp=True, for_inference=True,
                               obj_spec_provider=provider)

The returned ``(obj_spec_lst, per_spec_ngeom, meta)`` triple matches
``build_obj_spec_lst``'s contract, so the specs drop straight into
``SingleHandSubEnv.reset``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import Callable, Optional, Sequence

import numpy as np
import mujoco

# ── MuJoCo geom-size semantics ───────────────────────────────────────────────
#   box      : size = (hx, hy, hz)          half extents
#   capsule  : size = (r, half_len, 0)      half_len = half length of the
#                                           CYLINDRICAL part (caps add r each)
#   cylinder : size = (r, half_len, 0)
#   sphere   : size = (r, 0, 0)
# Capsules / cylinders are Z-aligned; we rotate them onto +X so every component
# shares one "long axis" convention.
KIND_TO_MJGEOM = {
    "box":      mujoco.mjtGeom.mjGEOM_BOX,       # type: ignore
    "capsule":  mujoco.mjtGeom.mjGEOM_CAPSULE,   # type: ignore
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,  # type: ignore
    "sphere":   mujoco.mjtGeom.mjGEOM_SPHERE,    # type: ignore
}
MJGEOM_TO_KIND = {int(v): k for k, v in KIND_TO_MJGEOM.items()}

# Which local axis is a kind's "long" axis (the one we aim along a direction):
# a box is built with size = (L/2, cy, cz) so it is +X; capsule/cylinder are +Z.
_LONG_AXIS_OF_KIND = {"box": "x", "capsule": "z", "cylinder": "z", "sphere": None}

# Z→X rotation (90° about +Y), wxyz — puts a capsule/cylinder's axis on +X.
_QUAT_Z_TO_X = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])


def _quat_long_axis(kind: str, axis_dir, roll: float = 0.0) -> np.ndarray:
    """Quaternion (wxyz) that aims ``kind``'s long axis along ``axis_dir``.

    ``roll`` additionally spins the geom about that axis — which matters for
    boxes (it turns the rectangular cross-section) and is free for capsules.
    This is what lets a secondary component sit **perpendicular** to the
    primary (hammer / T / cross) instead of only extending it end-to-end.
    """
    v = np.asarray(axis_dir, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    v = v / n

    q_align = np.zeros(4)
    mujoco.mju_quatZ2Vec(q_align, v)                 # +Z → v
    if _LONG_AXIS_OF_KIND.get(kind) == "x":          # box: +X → +Z → v
        q_x2z = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_x2z, np.array([0.0, 1.0, 0.0]), -np.pi / 2)
        tmp = np.zeros(4)
        mujoco.mju_mulQuat(tmp, q_align, q_x2z)
        q_align = tmp

    q_roll = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_roll, v, float(roll))
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q_roll, q_align)
    return out


def _basis_about(axis) -> tuple:
    """Orthonormal ``(u, e1, e2)`` with ``u`` along ``axis``."""
    u = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(u))
    u = np.array([1.0, 0.0, 0.0]) if n < 1e-12 else u / n
    t = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u, t); e1 /= max(np.linalg.norm(e1), 1e-12)
    return u, e1, np.cross(u, e1)


def _perp_dir(rng, tilt_rad: float, axis=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Unit vector roughly ⟂ to ``axis``: random roll about it, tilted by up to
    ``tilt_rad`` off the perpendicular so joints are not always exactly 90°."""
    u, e1, e2 = _basis_about(axis)
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    t   = float(rng.uniform(-tilt_rad, tilt_rad))
    return np.sin(t) * u + np.cos(t) * (np.cos(phi) * e1 + np.sin(phi) * e2)


def _part_frame(c: "PrimitiveComponent") -> tuple:
    """``(centre, long_axis_unit, half_length, cross_radius)`` of a component in
    the OBJECT body frame — what a child part needs to attach to it."""
    sz = np.asarray(c.size, dtype=float)
    R  = _quat_to_mat(c.quat)
    if c.kind == "box":
        axis, half, r = R @ np.array([1.0, 0.0, 0.0]), sz[0], float(max(sz[1], sz[2]))
    elif c.kind == "capsule":
        axis, half, r = R @ np.array([0.0, 0.0, 1.0]), sz[1] + sz[0], float(sz[0])
    elif c.kind == "cylinder":
        axis, half, r = R @ np.array([0.0, 0.0, 1.0]), sz[1], float(sz[0])
    else:                                   # sphere
        axis, half, r = np.array([1.0, 0.0, 0.0]), float(sz[0]), float(sz[0])
    return np.asarray(c.pos, dtype=float), axis, float(half), r

# ── Secondary-component shape archetypes ─────────────────────────────────────
# Sampling the secondary's size independently of the primary (the paper's
# literal recipe) mostly yields "long rod + tiny nub" — visually monotonous.
# Instead we draw an ARCHETYPE, which fixes *where* the secondary attaches and
# *how* it is oriented, and size it RELATIVE to the primary. Each archetype has
# a size bias so it reads as its name (a hammer head is short and fat, a cross
# bar is long and similar-width, …):
#
#   rod       bare primary, no secondary
#   hammer    fat head, ⟂ at one end, centred on the shaft   →  T at the tip
#   ell       ⟂ at one end but offset to one side            →  L
#   tee       ⟂ at mid-shaft, one-sided                      →  T
#   cross     ⟂ near the middle, centred (sticks out both)   →  +  /  X
#   dumbbell  collinear knob past one end (thicker)          →  dumbbell / mushroom
#   offset    parallel but laterally offset along the shaft  →  stepped / stacked slab
SECONDARY_SHAPES = ("rod", "hammer", "ell", "tee", "cross", "dumbbell", "offset")

# (length_mul, cross_mul) applied on top of the sampled size ratios.
_SHAPE_SIZE_BIAS = {
    "hammer":   (0.55, 1.35),
    "ell":      (0.75, 1.05),
    "tee":      (0.90, 1.05),
    "cross":    (1.00, 1.00),
    "dumbbell": (0.45, 1.45),
    "offset":   (0.80, 1.00),
}

DEFAULT_SHAPE_WEIGHTS = {
    "rod":      0.10,
    "hammer":   0.22,
    "ell":      0.14,
    "tee":      0.16,
    "cross":    0.14,
    "dumbbell": 0.14,
    "offset":   0.10,
}

# Enough slots for ``n_parts`` (default max 4) of EITHER kind — a slot's geom
# type is fixed for every world (mjwarp Model.geom_type is 1-D), so each
# drawable kind needs as many slots as an object could possibly use.
DEFAULT_SLOT_LAYOUT = ("box", "capsule") * 4
DEFAULT_ROOT_BODY   = "top_watertight_tiny"      # cfg ``Training.obj_name``

# YAML keys that live in the ``Training.primitive_object`` block but are NOT
# ``PrimitiveSampleCfg`` fields (they configure spec emission / the provider).
_NON_CFG_KEYS = ("slot_layout", "seed", "mesh_frac")

# Site / sensor names the rest of the pipeline expects on an object spec
# (mirrors the fallback branch of ``hand_utils.build_obj_spec_lst``).
_TOUCH_SITE       = "top_watertight_tiny_touch_site"
_DUMMY_TOUCH_SITE = "bottom_watertight_tiny_touch_site"
_TOUCH_SENSOR     = "top_watertight_tiny_touch"
_DUMMY_SENSOR     = "bottom_watertight_tiny_touch"


# ─────────────────────────────────────────────────────────────────────────────
# Surface sampling (also used by the PCD builders for primitive colliders)
# ─────────────────────────────────────────────────────────────────────────────
def sample_geom_surface(geom_type: int, size, n: int, rng=None) -> Optional[np.ndarray]:
    """Uniform-ish surface samples of a primitive geom, in its LOCAL frame.

    ``geom_type`` is an ``mjtGeom`` value, ``size`` the raw ``geom_size`` row.
    Returns ``(n, 3)`` or ``None`` for a type we do not handle (mesh / plane /
    hfield / …) so callers can fall through to their mesh path.

    Area-weighted per face/section so the density matches a mesh-sampled cloud
    (the wrist-init kernel and ``variant_obj_min`` both assume a surface cloud).
    """
    rng  = np.random.default_rng() if rng is None else rng
    s    = np.asarray(size, dtype=float).reshape(-1)
    gt   = int(geom_type)
    n    = int(max(1, n))

    if gt == int(mujoco.mjtGeom.mjGEOM_BOX):          # type: ignore
        h = s[:3]
        # 6 faces, area-weighted: face k (axis a, sign ±) has area 4*h[i]*h[j]
        areas, faces = [], []
        for a in range(3):
            i, j = [k for k in range(3) if k != a]
            for sgn in (-1.0, 1.0):
                areas.append(4.0 * h[i] * h[j])
                faces.append((a, i, j, sgn))
        areas = np.asarray(areas, dtype=float)
        if areas.sum() <= 0:
            return np.zeros((n, 3))
        pick = rng.choice(len(faces), size=n, p=areas / areas.sum())
        pts  = np.zeros((n, 3))
        u    = rng.uniform(-1.0, 1.0, size=(n, 2))
        for k, (a, i, j, sgn) in enumerate(faces):
            m = pick == k
            if not np.any(m):
                continue
            pts[m, a] = sgn * h[a]
            pts[m, i] = u[m, 0] * h[i]
            pts[m, j] = u[m, 1] * h[j]
        return pts

    if gt == int(mujoco.mjtGeom.mjGEOM_SPHERE):       # type: ignore
        r = float(s[0])
        v = rng.normal(size=(n, 3))
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        return v * r

    if gt == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):    # type: ignore
        v = rng.normal(size=(n, 3))
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        return v * s[:3]

    if gt in (int(mujoco.mjtGeom.mjGEOM_CAPSULE),     # type: ignore
              int(mujoco.mjtGeom.mjGEOM_CYLINDER)):   # type: ignore
        r, hl = float(s[0]), float(s[1])
        is_cap = gt == int(mujoco.mjtGeom.mjGEOM_CAPSULE)  # type: ignore
        a_side = 2.0 * np.pi * r * (2.0 * hl)
        a_end  = (4.0 * np.pi * r * r) if is_cap else (2.0 * np.pi * r * r)
        tot    = a_side + a_end
        if tot <= 0:
            return np.zeros((n, 3))
        n_side = int(rng.binomial(n, a_side / tot))
        n_end  = n - n_side
        out = []
        if n_side:
            th = rng.uniform(0.0, 2.0 * np.pi, size=n_side)
            z  = rng.uniform(-hl, hl, size=n_side)
            out.append(np.stack([r * np.cos(th), r * np.sin(th), z], axis=1))
        if n_end:
            sgn = np.where(rng.random(n_end) < 0.5, -1.0, 1.0)
            if is_cap:                       # hemispherical caps
                v = rng.normal(size=(n_end, 3))
                v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
                v[:, 2] = np.abs(v[:, 2]) * sgn
                out.append(v * r + np.stack([np.zeros(n_end), np.zeros(n_end), sgn * hl], axis=1))
            else:                            # flat disc caps
                rad = r * np.sqrt(rng.random(n_end))
                th  = rng.uniform(0.0, 2.0 * np.pi, size=n_end)
                out.append(np.stack([rad * np.cos(th), rad * np.sin(th), sgn * hl], axis=1))
        return np.concatenate(out, axis=0) if out else np.zeros((n, 3))

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Mass properties
# ─────────────────────────────────────────────────────────────────────────────
def _volume_and_inertia(kind: str, size: np.ndarray, density: float):
    """``(mass, I_local)`` about the component's own COM, in its local frame."""
    s = np.asarray(size, dtype=float)
    if kind == "box":
        hx, hy, hz = s[:3]
        m = density * 8.0 * hx * hy * hz
        I = (m / 3.0) * np.diag([hy**2 + hz**2, hx**2 + hz**2, hx**2 + hy**2])
        return m, I
    if kind == "sphere":
        r = s[0]
        m = density * (4.0 / 3.0) * np.pi * r**3
        return m, np.eye(3) * (0.4 * m * r**2)
    if kind == "cylinder":
        r, hl = s[0], s[1]
        m = density * np.pi * r**2 * (2.0 * hl)
        ip = m * (3.0 * r**2 + 4.0 * hl**2) / 12.0
        return m, np.diag([ip, ip, 0.5 * m * r**2])
    if kind == "capsule":
        r, hl = s[0], s[1]
        mc = density * np.pi * r**2 * (2.0 * hl)          # cylindrical part
        ms = density * (4.0 / 3.0) * np.pi * r**3         # both caps = 1 sphere
        m  = mc + ms
        izz = 0.5 * mc * r**2 + 0.4 * ms * r**2
        # Each hemisphere: transverse inertia about its own COM, then shifted to
        # the capsule centre (COM sits 3r/8 from the flat face, which is at ±hl).
        mh  = 0.5 * ms
        i_h_about_center = 0.4 * mh * r**2 - mh * (3.0 * r / 8.0) ** 2
        d   = hl + 3.0 * r / 8.0
        ixx = mc * (3.0 * r**2 + 4.0 * hl**2) / 12.0 + 2.0 * (i_h_about_center + mh * d**2)
        return m, np.diag([ixx, ixx, izz])
    raise ValueError(f"unsupported primitive kind: {kind!r}")


def _quat_to_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=np.float64))
    return R.reshape(3, 3)


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return q


# ─────────────────────────────────────────────────────────────────────────────
# Object model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PrimitiveComponent:
    """One rigid primitive inside a compound object (body-frame placement)."""
    kind:    str
    size:    np.ndarray          # raw mjGEOM size row (3,)
    pos:     np.ndarray          # (3,) in the object body frame
    quat:    np.ndarray          # (4,) wxyz in the object body frame
    density: float
    role:    str = "primary"     # "primary" | "secondary"

    def mass_inertia_body(self):
        """``(mass, I)`` about the component COM, rotated into the BODY frame."""
        m, I_local = _volume_and_inertia(self.kind, self.size, self.density)
        R = _quat_to_mat(self.quat)
        return m, R @ I_local @ R.T

    @property
    def extent(self) -> np.ndarray:
        """Half-extent of the component's AABB in its own local frame."""
        s = np.asarray(self.size, dtype=float)
        if self.kind == "box":
            return s[:3].copy()
        if self.kind == "sphere":
            return np.full(3, s[0])
        if self.kind == "capsule":
            return np.array([s[0], s[0], s[1] + s[0]])
        if self.kind == "cylinder":
            return np.array([s[0], s[0], s[1]])
        raise ValueError(self.kind)


@dataclass
class PrimitiveObject:
    """A compound object = the components that make it up."""
    components: list           # list[PrimitiveComponent]
    seed_id:    int = -1
    shape:      str = ""       # archetype label (see SECONDARY_SHAPES)

    # ── inertial of the assembly (about the combined COM, principal axes) ──
    def inertial(self):
        """``(mass, com(3), iquat(4) wxyz, principal_inertia(3))``."""
        m_tot, com = 0.0, np.zeros(3)
        for c in self.components:
            m, _ = c.mass_inertia_body()
            m_tot += m
            com   += m * np.asarray(c.pos, dtype=float)
        com = com / max(m_tot, 1e-12)

        I = np.zeros((3, 3))
        for c in self.components:
            m, I_c = c.mass_inertia_body()
            d = np.asarray(c.pos, dtype=float) - com
            I += I_c + m * (float(d @ d) * np.eye(3) - np.outer(d, d))

        evals, evecs = np.linalg.eigh(I)             # symmetric → orthonormal
        if np.linalg.det(evecs) < 0:                 # keep it a proper rotation
            evecs[:, 0] *= -1.0
        return m_tot, com, _mat_to_quat(evecs), np.maximum(evals, 1e-9)

    def rescale(self, k: float) -> None:
        """Scale the assembly's geometry in place (mass scales with ``k³``)."""
        k = float(k)
        for c in self.components:
            c.size = np.asarray(c.size, dtype=float) * k
            c.pos  = np.asarray(c.pos,  dtype=float) * k

    def inflate_thin_parts(self, min_thickness: float) -> None:
        """Raise every component's CROSS dimensions to ``min_thickness``.

        Graspability guard: a 5 mm slab or wire is valid geometry but the hand
        cannot pinch it. Only the cross-section grows — the long axis (and so
        the overall silhouette) is preserved.
        """
        r_min = 0.5 * float(min_thickness)
        for c in self.components:
            sz = np.asarray(c.size, dtype=float)
            if c.kind == "box":                       # (hx, hy, hz) — hx is long
                sz[1] = max(sz[1], r_min)
                sz[2] = max(sz[2], r_min)
            elif c.kind in ("capsule", "cylinder"):   # (r, half_len, 0)
                sz[0] = max(sz[0], r_min)
            elif c.kind == "sphere":
                sz[0] = max(sz[0], r_min)
            c.size = sz

    def scale_density(self, k: float) -> None:
        """Scale every component's density by ``k`` (mass ×k, COM unchanged)."""
        for c in self.components:
            c.density = float(c.density) * float(k)

    def aabb(self):
        """``(min(3), max(3))`` of the assembly in the object body frame."""
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for c in self.components:
            R = np.abs(_quat_to_mat(c.quat))          # AABB of a rotated box
            e = R @ c.extent
            lo = np.minimum(lo, np.asarray(c.pos) - e)
            hi = np.maximum(hi, np.asarray(c.pos) + e)
        return lo, hi

    def describe(self) -> str:
        m, com, _, inr = self.inertial()
        lo, hi = self.aabb()
        parts = " + ".join(
            f"{c.kind}({'×'.join(f'{v:.3f}' for v in np.asarray(c.size)[:2 if c.kind in ('capsule','cylinder') else 3])}"
            f"@{c.density:.0f})" for c in self.components)
        return (f"[{len(self.components)}p {self.shape or '?'}] {parts}  mass={m*1e3:.0f} g  "
                f"com=({com[0]:+.3f},{com[1]:+.3f},{com[2]:+.3f})  "
                f"bbox={np.round(hi-lo, 3).tolist()}  Iprincipal={np.round(inr*1e4, 2).tolist()}e-4")


@dataclass
class PrimitiveSampleCfg:
    """Sampling ranges for :func:`sample_primitive_object` (metres, kg/m³).

    Structure follows Play2Perfect (2 components, independently sampled
    densities) but the secondary is placed by ARCHETYPE and sized RELATIVE to
    the primary — see :data:`SECONDARY_SHAPES` — so hammers / Ts / crosses / Ls
    / dumbbells appear instead of "long rod + tiny nub".

    Defaults are sized to stay **pinchable**: no part of an assembly is thinner
    than ``min_thickness`` (3 cm), objects run ≈7–23 cm long, and mass is
    ≈24 g / 129 g / 466 g at the 5th / 50th / 95th percentile (capped at
    ``max_mass``). ``secondary_cross_ratio`` spans 0.55–1.8 so the two arms of
    an L / T / cross differ in thickness (median contrast ≈1.35×, 56% above
    1.3×) rather than reading as two identical bars. The mesh dataset for comparison is 50 g–1 kg over ~9–24 cm,
    built at 100 kg/m³ — the thickness floor makes these assemblies chunky, so
    the densities sit below the paper's to keep the hand able to lift them; the
    density *ratio* between the two components (up to ~12×) is what drives the
    COM / inertia diversity and is untouched.

    Every field is settable from YAML — ``cfg.Training.primitive_object`` — see
    :meth:`from_config`. ``size_scale`` is the one-knob "spawn bigger" lever: it
    multiplies every length / cross-section range (densities untouched, so mass
    grows with the cube of the scale).
    """
    primary_kinds:     Sequence[str] = ("box", "capsule")
    secondary_kinds:   Sequence[str] = ("box", "capsule")
    # How many primitives make up one object (inclusive int range). Every part
    # after the first attaches to an already-placed part, so 3–4 parts give
    # branched / chained assemblies rather than just "bar + head".
    # ⚠ ``slot_layout`` must hold ``n_parts[1]`` slots of EVERY drawable kind.
    n_parts:           tuple = (2, 4)
    primary_length:    tuple = (0.04, 0.22)     # full length along +X
    primary_cross:     tuple = (0.032, 0.070)   # full cross-section (per axis)
    primary_density:   tuple = (55.0, 200.0)    # paper: 300–600 (lowered because the volume is large)
    secondary_density: tuple = (70.0, 700.0)    # paper: 300–2000

    # ── secondary: sized RELATIVE to the primary ──────────────────────────
    # Absolute independent ranges (the paper's literal recipe) mostly produce
    # "long rod + tiny nub". Ratios keep the two components comparable, which
    # is what makes a hammer / T / cross actually read as that shape. The
    # per-archetype bias in ``_SHAPE_SIZE_BIAS`` is applied on top.
    secondary_length_ratio: tuple = (0.35, 1.10)   # × primary length
    # Also the ARM-THICKNESS CONTRAST for L / T / cross: a wide range makes one
    # arm visibly chunkier than the other instead of two look-alike bars.
    secondary_cross_ratio:  tuple = (0.55, 1.80)   # × primary cross-section
    # How often each archetype is drawn (see SECONDARY_SHAPES). Weights are
    # normalised; drop a shape by setting it to 0.
    shape_weights: dict = field(default_factory=lambda: dict(DEFAULT_SHAPE_WEIGHTS))

    # Attach point along the primary, as a fraction of its half-length:
    # ``attach_frac`` for end-attached shapes (hammer / ell / dumbbell / offset),
    # ``mid_attach_frac`` for mid-shaft ones (tee / cross).
    attach_frac:       tuple = (0.72, 1.00)
    mid_attach_frac:   tuple = (0.0, 0.55)
    lateral_frac:      tuple = (0.35, 1.10)     # 'offset' side-shift, × cross radius
    tilt_deg:          tuple = (0.0, 30.0)      # deviation from the ideal joint angle
    cross_aniso:       tuple = (0.75, 1.35)     # hy/hz ratio for cuboids (1 = square)
    size_scale:        float = 1.0              # ×all lengths / cross-sections

    # ── keep the assembly graspable ───────────────────────────────────────
    # A comparable-size secondary can blow the object up (a fat hammer head on
    # a long handle). Instead of narrowing the shape family we clamp the
    # RESULT: geometry is scaled down until it fits the extents, then the
    # densities are scaled until the mass fits — the density *contrast*
    # between the two components (and hence the COM offset) is preserved.
    # ``size_scale`` multiplies these caps too, so "spawn bigger" still works.
    # Set any of them to 0 to disable that clamp.
    max_extent:        float = 0.26             # longest bbox dim  (m)
    max_cross_extent:  float = 0.14             # other two bbox dims (m)
    max_mass:          float = 0.50             # (kg)

    # ── keep every part pinchable ─────────────────────────────────────────
    # A razor-thin slab or a wire-thin bar is valid geometry but miserable to
    # grasp (and looks wrong). After the extent clamp, every component's CROSS
    # dimensions are raised to at least this, so no part of the assembly is
    # thinner than a finger pad. Length is untouched — long rods stay long,
    # they just stop being wires. Scales with ``size_scale``; 0 = off.
    min_thickness:     float = 0.030            # (m) smallest cross dim of any part

    # Range fields validated + coerced to tuple[float, float] on construction.
    _RANGE_FIELDS = ("primary_length", "primary_cross",
                     "primary_density", "secondary_density",
                     "secondary_length_ratio", "secondary_cross_ratio",
                     "attach_frac", "mid_attach_frac", "lateral_frac",
                     "tilt_deg", "cross_aniso")

    def __post_init__(self):
        for name in self._RANGE_FIELDS:
            v = getattr(self, name)
            try:
                lo, hi = (float(v[0]), float(v[1]))
            except Exception as exc:
                raise ValueError(
                    f"PrimitiveSampleCfg.{name} must be a [lo, hi] pair, got {v!r}") from exc
            if hi < lo:
                raise ValueError(f"PrimitiveSampleCfg.{name}: lo > hi ({lo} > {hi})")
            if name.endswith(("_length", "_cross", "_density")) and lo <= 0.0:
                raise ValueError(f"PrimitiveSampleCfg.{name}: lo must be > 0 (got {lo})")
            object.__setattr__(self, name, (lo, hi))
        for name in ("primary_kinds", "secondary_kinds"):
            kinds = tuple(str(k) for k in getattr(self, name))
            if not kinds:
                raise ValueError(f"PrimitiveSampleCfg.{name} is empty")
            bad = [k for k in kinds if k not in KIND_TO_MJGEOM]
            if bad:
                raise ValueError(f"PrimitiveSampleCfg.{name}: unknown kind(s) {bad} "
                                 f"(known: {sorted(KIND_TO_MJGEOM)})")
            object.__setattr__(self, name, kinds)
        try:
            np_lo, np_hi = int(self.n_parts[0]), int(self.n_parts[1])
        except Exception as exc:
            raise ValueError(
                f"PrimitiveSampleCfg.n_parts must be an [lo, hi] int pair, "
                f"got {self.n_parts!r}") from exc
        if np_lo < 1 or np_hi < np_lo:
            raise ValueError(f"PrimitiveSampleCfg.n_parts invalid: [{np_lo}, {np_hi}]")
        object.__setattr__(self, "n_parts", (np_lo, np_hi))

        w = {str(k): float(v) for k, v in dict(self.shape_weights).items()}
        bad_w = [k for k in w if k not in SECONDARY_SHAPES]
        if bad_w:
            raise ValueError(f"PrimitiveSampleCfg.shape_weights: unknown shape(s) {bad_w} "
                             f"(known: {list(SECONDARY_SHAPES)})")
        if any(v < 0 for v in w.values()):
            raise ValueError(f"PrimitiveSampleCfg.shape_weights must be ≥ 0, got {w}")
        if sum(w.values()) <= 0:
            raise ValueError("PrimitiveSampleCfg.shape_weights sum to 0 — nothing can be sampled")
        object.__setattr__(self, "shape_weights", w)
        object.__setattr__(self, "size_scale",  float(self.size_scale))
        if self.size_scale <= 0.0:
            raise ValueError(f"PrimitiveSampleCfg.size_scale must be > 0 (got {self.size_scale})")
        for name in ("max_extent", "max_cross_extent", "max_mass", "min_thickness"):
            v = float(getattr(self, name))
            if v < 0.0:
                raise ValueError(f"PrimitiveSampleCfg.{name} must be ≥ 0 (0 = no clamp), got {v}")
            object.__setattr__(self, name, v)

    # ── YAML ───────────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, node=None, **overrides):
        """Build from an OmegaConf node / plain dict (e.g. the YAML block
        ``Training.primitive_object``), with keyword overrides on top.

        Keys outside the dataclass that belong to the same YAML block
        (``slot_layout``, ``seed``) are ignored here — read them with
        :func:`primitive_options_from_config`. Any OTHER unknown key raises,
        so a typo in the YAML fails loudly instead of being silently dropped.
        """
        data = {}
        if node is not None:
            try:
                from omegaconf import OmegaConf
                if OmegaConf.is_config(node):
                    node = OmegaConf.to_container(node, resolve=True)
            except ImportError:
                pass
            if not isinstance(node, dict):
                raise TypeError(f"primitive_object config must be a mapping, got {type(node)}")
            known   = {f.name for f in fields(cls)}
            unknown = set(node) - known - set(_NON_CFG_KEYS)
            if unknown:
                raise ValueError(
                    f"unknown primitive_object key(s): {sorted(unknown)}  "
                    f"(known: {sorted(known)} + {list(_NON_CFG_KEYS)})")
            data = {k: v for k, v in node.items() if k in known}
        data.update(overrides)
        return cls(**data)


def _fit_limits(obj: PrimitiveObject, cfg: PrimitiveSampleCfg) -> PrimitiveObject:
    """Shrink the assembly until it satisfies the extent / mass caps.

    Geometry first (so shape is preserved exactly — every component scales by
    the same factor), then density (so the two components keep their relative
    densities, and with them the COM offset that makes the object interesting).
    """
    sc = float(cfg.size_scale)
    lo, hi = obj.aabb()
    dims   = np.sort(hi - lo)[::-1]                  # longest first
    k = 1.0
    if cfg.max_extent > 0 and dims[0] > cfg.max_extent * sc:
        k = min(k, cfg.max_extent * sc / float(dims[0]))
    if cfg.max_cross_extent > 0 and dims[1] > cfg.max_cross_extent * sc:
        k = min(k, cfg.max_cross_extent * sc / float(dims[1]))
    if k < 1.0:
        obj.rescale(k)
    if cfg.min_thickness > 0:
        obj.inflate_thin_parts(cfg.min_thickness * sc)
    if cfg.max_mass > 0:
        mass = obj.inertial()[0]
        if mass > cfg.max_mass:
            obj.scale_density(cfg.max_mass / float(mass))
    return obj


def _make_component(kind, length, cross, aniso, pos, axis_dir, roll, density, role):
    """One primitive whose long axis points along ``axis_dir``.

    ``length`` / ``cross`` are FULL extents (m); the MuJoCo size row is derived
    per kind. A capsule shorter than its own diameter degenerates to a sphere,
    which is fine — it just widens the shape family.
    """
    if kind == "box":
        size = np.array([length / 2.0, cross / 2.0, cross / 2.0 * aniso])
    elif kind in ("capsule", "cylinder"):
        r    = cross / 2.0
        size = np.array([r, max(1e-4, length / 2.0 - (r if kind == "capsule" else 0.0)), 0.0])
    elif kind == "sphere":
        size = np.array([cross / 2.0, 0.0, 0.0])
    else:
        raise ValueError(f"unsupported primitive kind: {kind!r}")
    return PrimitiveComponent(kind, size, np.asarray(pos, dtype=float),
                              _quat_long_axis(kind, axis_dir, roll), float(density), role)


def _attach_part(rng, cfg, parent, kind, shape, seed_len_ratio, seed_cross_ratio,
                 len_cap=None, cross_cap=None):
    """Size + place ONE child primitive on ``parent`` following ``shape``.

    All placement is expressed in the parent's own frame (its long axis, its
    half-length, its cross radius), so a part can hang off any earlier part —
    that is what turns "primary + secondary" into an arbitrary multi-part
    assembly. Returns the new :class:`PrimitiveComponent`.
    """
    u          = rng.uniform
    p_c, p_ax, p_half, p_r = _part_frame(parent)
    bias_l, bias_c = _SHAPE_SIZE_BIAS[shape]

    Ls = (2.0 * p_half) * seed_len_ratio   * bias_l
    cs = (2.0 * p_r)    * seed_cross_ratio * bias_c
    # Chains compound: a child of a child of a child could otherwise grow by
    # ratio³ and swallow the whole assembly. Cap every part against the PRIMARY
    # so deep branches stay proportionate.
    if len_cap   is not None: Ls = min(Ls, float(len_cap))
    if cross_cap is not None: cs = min(cs, float(cross_cap))
    Ls = max(Ls, 0.6 * cs)                       # never absurdly stubby
    rs = 0.5 * cs

    sign = 1.0 if rng.random() < 0.5 else -1.0
    roll = float(u(0.0, 2.0 * np.pi))
    tilt = np.deg2rad(float(u(*cfg.tilt_deg)))

    if shape in ("hammer", "ell"):
        # ⟂ head at one end of the parent ('ell' pushed to one side → L).
        axis = _perp_dir(rng, tilt, p_ax)
        pos  = p_c + p_ax * (sign * float(u(*cfg.attach_frac)) * p_half)
        if shape == "ell":
            pos = pos + axis * (0.5 * Ls - 0.5 * p_r)
    elif shape in ("tee", "cross"):
        # ⟂ bar on the shaft: 'cross' passes through, 'tee' sticks out one side.
        axis = _perp_dir(rng, tilt, p_ax)
        pos  = p_c + p_ax * (sign * float(u(*cfg.mid_attach_frac)) * p_half)
        if shape == "tee":
            pos = pos + axis * (0.5 * Ls - 0.5 * p_r)
    elif shape == "dumbbell":
        # collinear knob just past one end (thicker → dumbbell / mushroom).
        axis    = p_ax
        overlap = min(p_r, 0.35 * Ls)
        pos     = p_c + p_ax * (sign * (p_half + 0.5 * Ls - overlap))
    else:  # "offset" — parallel, shifted sideways → stepped / stacked slab
        u_, e1, e2 = _basis_about(p_ax)
        axis = np.cos(tilt) * u_ + np.sin(tilt) * (np.cos(roll) * e1 + np.sin(roll) * e2)
        lat  = float(u(*cfg.lateral_frac)) * (p_r + rs)
        phi  = float(u(0.0, 2.0 * np.pi))
        pos  = (p_c + p_ax * (sign * float(u(0.0, 1.0)) * p_half)
                + (np.cos(phi) * e1 + np.sin(phi) * e2) * lat)

    return _make_component(kind, Ls, cs, float(u(*cfg.cross_aniso)), pos, axis, roll,
                           float(u(*cfg.secondary_density)), "part")


def sample_primitive_object(rng, cfg: PrimitiveSampleCfg, seed_id: int = -1) -> PrimitiveObject:
    """Draw one multi-part object (the first part's long axis = +X).

    Play2Perfect's structure — a graspable primary plus rigidly attached parts
    with independently sampled densities — generalised on three axes so the
    shape family is genuinely broad:

    1. **part count**: ``n_parts`` parts (default 2–4), each attached to an
       already-placed part (volume-weighted choice), so the assembly is always
       connected but can branch or chain;
    2. **relative sizing**: a part is sized against ITS PARENT, so nothing
       degenerates into "big rod + tiny nub";
    3. **archetype placement** (:data:`SECONDARY_SHAPES`): hammer / T / cross /
       L / dumbbell / stepped, each with its own attachment point and angle.

    The result is finally clamped to stay graspable — see :func:`_fit_limits`.
    """
    u  = rng.uniform
    sc = float(cfg.size_scale)      # one-knob "spawn bigger" (densities unchanged)

    # ── part 0 (primary): graspable region, at the body origin, long axis +X ──
    kind_p = str(rng.choice(list(cfg.primary_kinds)))
    L      = float(u(*cfg.primary_length)) * sc
    c0     = float(u(*cfg.primary_cross))  * sc
    comps  = [_make_component(
        kind_p, L, c0, float(u(*cfg.cross_aniso)), np.zeros(3),
        np.array([1.0, 0.0, 0.0]), float(u(0.0, 2.0 * np.pi)),
        float(u(*cfg.primary_density)), "primary")]

    # ── how many parts, and (for 2 parts) which archetype ────────────────────
    lo_n, hi_n = (int(cfg.n_parts[0]), int(cfg.n_parts[1]))
    n_parts    = int(rng.integers(lo_n, hi_n + 1))

    shapes  = [k for k in SECONDARY_SHAPES if cfg.shape_weights.get(k, 0.0) > 0]
    weights = np.array([cfg.shape_weights[k] for k in shapes], dtype=float)
    weights = weights / weights.sum()

    # "rod" is only meaningful as the FIRST attachment decision (= no parts at
    # all); once we are adding part k>1 it is excluded and renormalised.
    first = str(rng.choice(shapes, p=weights))
    if first == "rod" or n_parts < 2:
        return _fit_limits(PrimitiveObject(comps, seed_id=seed_id, shape="rod"), cfg)

    sub        = [(k, w) for k, w in zip(shapes, weights) if k != "rod"]
    sub_names  = [k for k, _ in sub]
    sub_w      = np.array([w for _, w in sub], dtype=float)
    sub_w      = sub_w / sub_w.sum()

    labels = []
    for i in range(1, n_parts):
        shape = first if i == 1 else str(rng.choice(sub_names, p=sub_w))
        # Parent choice weighted by volume → big parts host more children,
        # which keeps small twigs from growing their own twigs.
        vols   = np.array([max(_volume_and_inertia(c.kind, c.size, 1.0)[0], 1e-12)
                           for c in comps], dtype=float)
        parent = comps[int(rng.choice(len(comps), p=vols / vols.sum()))]
        kind_s = str(rng.choice(list(cfg.secondary_kinds)))
        comps.append(_attach_part(
            rng, cfg, parent, kind_s, shape,
            float(u(*cfg.secondary_length_ratio)),
            float(u(*cfg.secondary_cross_ratio)),
            len_cap   = L  * float(cfg.secondary_length_ratio[1]),
            cross_cap = c0 * float(cfg.secondary_cross_ratio[1])))
        labels.append(shape)

    shape_label = "+".join(labels) if len(labels) > 1 else labels[0]
    return _fit_limits(PrimitiveObject(comps, seed_id=seed_id, shape=shape_label), cfg)


# ─────────────────────────────────────────────────────────────────────────────
# MjSpec emission
# ─────────────────────────────────────────────────────────────────────────────
_PLACEHOLDER_SIZE = np.array([1.0e-4, 1.0e-4, 1.0e-4])
_RGBA_PRIMARY     = (0.35, 0.62, 0.90, 1.0)
_RGBA_SECONDARY   = (0.92, 0.55, 0.25, 1.0)
# part 0 = blue (graspable primary), the rest cycle so multi-part assemblies
# are readable at a glance in the viewer / render script.
_RGBA_PARTS = (
    (0.92, 0.55, 0.25, 1.0),   # orange
    (0.45, 0.75, 0.42, 1.0),   # green
    (0.80, 0.45, 0.75, 1.0),   # violet
    (0.90, 0.80, 0.30, 1.0),   # yellow
)


def validate_layout(cfg: "PrimitiveSampleCfg", layout: Sequence[str]) -> None:
    """Fail early (build time, not mid-sampling) if ``layout`` cannot host every
    kind the sampler may draw.

    A kind that can appear as BOTH primary and secondary needs two slots — the
    two components are placed in different slots.
    """
    # Worst case: every one of the ``n_parts[1]`` parts draws the same kind.
    n_max = int(cfg.n_parts[1])
    need  = {k: n_max for k in set(cfg.primary_kinds) | set(cfg.secondary_kinds)}
    have = {k: list(layout).count(k) for k in need}
    short = {k: (need[k], have[k]) for k in need if have[k] < need[k]}
    if short:
        raise ValueError(
            "primitive slot_layout cannot host the sampled kinds: "
            + ", ".join(f"{k} needs {n} slot(s), layout has {h}" for k, (n, h) in short.items())
            + f".  layout={tuple(layout)}, primary_kinds={tuple(cfg.primary_kinds)}, "
              f"secondary_kinds={tuple(cfg.secondary_kinds)}"
        )


def _assign_slots(obj: PrimitiveObject, layout: Sequence[str]) -> dict:
    """Map each component onto a slot index of ``layout`` (greedy, in order).

    Every component takes the first still-free slot of its own kind. Raises
    when the layout cannot host the drawn parts — a config error (layout too
    small for ``n_parts``), not a sampling accident; :func:`validate_layout`
    catches it at build time.
    """
    used, out = set(), {}
    for c in obj.components:
        cand = [i for i, k in enumerate(layout) if k == c.kind and i not in used]
        if not cand:
            raise ValueError(
                f"slot layout {tuple(layout)} has no free '{c.kind}' slot for a "
                f"{len(obj.components)}-part object — every drawable kind needs "
                f"n_parts[1] slots."
            )
        used.add(cand[0])
        out[cand[0]] = c
    return out


def make_primitive_obj_spec(
    obj: PrimitiveObject,
    idx: int = 0,
    layout: Sequence[str] = DEFAULT_SLOT_LAYOUT,
    root_body_name: str = DEFAULT_ROOT_BODY,
    friction=(1.2, 0.5, 0.4),
    sim_dt: Optional[float] = None,
    solref_timeconst: Optional[float] = None,
    mesh_slots: int = 0,
    mesh_slot_name: Optional[str] = None,
    child_tree: Optional[list] = None,
    child_geom_max: Optional[dict] = None,
):
    """Emit one ``MjSpec`` for ``obj`` shaped like a ``build_obj_spec_lst`` spec.

    * root body ``root_body_name`` with a free joint (the scene builder renames
      it to ``<root>_<slot>``),
    * ``len(layout)`` geoms in canonical slot order — active ones carry the
      component, unused ones are inert ``group=4`` placeholders,
    * the two touch sites + touch sensors the handler indices expect.
    """
    t = (solref_timeconst if solref_timeconst is not None
         else (max(0.01, 2.0 * float(sim_dt)) if sim_dt is not None else 0.01))
    slots = _assign_slots(obj, layout)
    # ordinal per component (identity map — dataclass __eq__ compares arrays)
    order = {id(c): k for k, c in enumerate(obj.components)}

    spec = mujoco.MjSpec()                                    # type: ignore
    spec.modelname = f"primitive_obj_{idx}"
    body = spec.worldbody.add_body(name=root_body_name)

    jnt = body.add_joint()
    jnt.name = f"{root_body_name}_freejoint"
    jnt.type = mujoco.mjtJoint.mjJNT_FREE                     # type: ignore

    # MIXED mode: the mesh slots come FIRST and stay disabled here, so a
    # primitive variant never lands a primitive collider on a mesh slot
    # (see build_mixed_obj_spec_lst).
    for i in range(int(mesh_slots)):
        if not mesh_slot_name:
            raise ValueError("mesh_slots>0 requires mesh_slot_name")
        _add_mesh_placeholder(body, mesh_slot_name, f"{root_body_name}_meshslot_{i}")

    for i, kind in enumerate(layout):
        g = body.add_geom(name=f"{root_body_name}_prim_{i}")
        g.type   = KIND_TO_MJGEOM[kind]
        g.solimp = np.array([0.975, 0.999, 0.001, 0.5, 2.0])
        g.solref = np.array([t, 1.0])
        g.friction = np.array(friction, dtype=float)
        c = slots.get(i)
        if c is None:                       # inert placeholder slot
            g.size        = _PLACEHOLDER_SIZE.copy()
            g.pos         = np.zeros(3)
            g.quat        = np.array([1.0, 0.0, 0.0, 0.0])
            g.rgba        = np.zeros(4)
            g.density     = 0.0
            g.contype     = 0
            g.conaffinity = 0
            g.group       = 4               # → variant marks it disabled/banished
        else:
            g.size        = np.asarray(c.size, dtype=float).reshape(3)
            g.pos         = np.asarray(c.pos, dtype=float)
            g.quat        = np.asarray(c.quat, dtype=float)
            _k = order[id(c)]
            g.rgba        = np.array(_RGBA_PRIMARY if _k == 0
                                     else _RGBA_PARTS[(_k - 1) % len(_RGBA_PARTS)])
            g.density     = float(c.density)
            g.contype     = 1
            g.conaffinity = 1
            g.group       = 0

    # Explicit inertial from the analytic assembly (the variant-override path
    # copies ipos/iquat/mass/inertia body-to-body; geom densities reproduce the
    # same numbers when MuJoCo recomputes, so the two agree either way).
    mass, com, iquat, inertia = obj.inertial()
    body.ipos    = com
    body.iquat   = iquat
    body.mass    = float(mass)
    body.inertia = inertia
    body.explicitinertial = True

    # MIXED mode: mirror the dataset body tree with inert children, otherwise
    # ``override_bodies_recursive`` never visits them for a primitive variant
    # and they keep the ANCHOR's geoms + inertial (a phantom mesh bottom).
    def _mirror(parent, nodes):
        for name, sub in (nodes or []):
            cb = parent.add_body(name=name)
            cb.mass = 0.0; cb.ipos = np.zeros(3); cb.inertia = np.zeros(3)
            for i in range(int((child_geom_max or {}).get(name, 0))):
                _add_mesh_placeholder(cb, mesh_slot_name, f"{name}_meshslot_{i}")
            _mirror(cb, sub)
    _mirror(body, child_tree)

    # Touch sites + sensors (same names/fallback as build_obj_spec_lst).
    for sname, rgba in ((_TOUCH_SITE, (0, 0, 1, 0.0)), (_DUMMY_TOUCH_SITE, (0, 1, 0, 0.0))):
        s = body.add_site()
        s.name = sname
        s.type = mujoco.mjtGeom.mjGEOM_SPHERE                 # type: ignore
        s.size = [0.3] * 3
        s.rgba = list(rgba)
    for sensor_name, site_name in ((_TOUCH_SENSOR, _TOUCH_SITE),
                                   (_DUMMY_SENSOR, _DUMMY_TOUCH_SITE)):
        sen = spec.add_sensor()
        sen.name    = sensor_name
        sen.type    = mujoco.mjtSensor.mjSENS_TOUCH           # type: ignore
        sen.objname = site_name
        sen.objtype = mujoco.mjtObj.mjOBJ_SITE                # type: ignore
    return spec


def build_primitive_obj_spec_lst(
    n_specs: int,
    cfg: Optional[PrimitiveSampleCfg] = None,
    layout: Sequence[str] = DEFAULT_SLOT_LAYOUT,
    rng=None,
    seed: Optional[int] = None,
    root_body_name: str = DEFAULT_ROOT_BODY,
    friction=(1.2, 0.5, 0.4),
    sim_dt: Optional[float] = None,
    verbose: bool = False,
):
    """``(obj_spec_lst, per_spec_ngeom, objects)`` — the dataset-path contract.

    ``objects`` is the list of :class:`PrimitiveObject` samples (the third slot
    is ``has_non_collide_obj`` in ``build_obj_spec_lst``; the orchestrator
    ignores it, and the samples are far more useful for introspection).
    """
    cfg = cfg or PrimitiveSampleCfg()
    validate_layout(cfg, layout)
    if rng is None:
        rng = np.random.default_rng(seed)

    specs, ngeoms, objects = [], [], []
    for i in range(int(n_specs)):
        obj  = sample_primitive_object(rng, cfg, seed_id=i)
        spec = make_primitive_obj_spec(obj, idx=i, layout=layout,
                                       root_body_name=root_body_name,
                                       friction=friction, sim_dt=sim_dt)
        specs.append(spec)
        ngeoms.append(len(layout))
        objects.append(obj)
        if verbose:
            print(f"[primitive obj {i}] {obj.describe()}")
    return specs, ngeoms, objects


def make_obj_spec_provider(
    cfg: Optional[PrimitiveSampleCfg] = None,
    layout: Sequence[str] = DEFAULT_SLOT_LAYOUT,
    seed: Optional[int] = None,
    friction=(1.2, 0.5, 0.4),
    sim_dt: Optional[float] = None,
    root_body_name: str = DEFAULT_ROOT_BODY,
    verbose: bool = False,
) -> Callable[[int], tuple]:
    """A provider for ``Env_Orchestrator.build_sub_env(obj_spec_provider=…)``.

    The RNG is created once and kept, so a later ``resample_objects()`` draws a
    FRESH object set (pass ``seed`` for a reproducible stream).
    """
    rng = np.random.default_rng(seed)

    def _provider(n_specs: int):
        return build_primitive_obj_spec_lst(
            n_specs, cfg=cfg, layout=layout, rng=rng, root_body_name=root_body_name,
            friction=friction, sim_dt=sim_dt, verbose=verbose,
        )
    _provider.rng = rng                     # type: ignore[attr-defined]
    return _provider


# ─────────────────────────────────────────────────────────────────────────────
# YAML-driven entry points
# ─────────────────────────────────────────────────────────────────────────────
def primitive_options_from_config(overall_cfg, **cfg_overrides) -> dict:
    """Read the ``Training.primitive_object`` YAML block off a full config.

    Returns ``{"cfg": PrimitiveSampleCfg, "layout": tuple[str, ...],
    "seed": int | None}``. A missing block (older YAMLs) simply yields the
    dataclass defaults, so nothing has to be added to a config that does not
    use procedural objects. ``cfg_overrides`` are applied on top of the YAML
    (e.g. ``size_scale=1.5`` from a notebook).
    """
    node = None
    try:
        node = overall_cfg.Training.get("primitive_object", None)   # OmegaConf
    except Exception:
        try:
            node = overall_cfg["Training"]["primitive_object"]
        except Exception:
            node = None

    raw = {}
    if node is not None:
        try:
            from omegaconf import OmegaConf
            raw = OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else dict(node)
        except ImportError:
            raw = dict(node)

    layout = tuple(str(k) for k in raw.get("slot_layout", DEFAULT_SLOT_LAYOUT))
    bad    = [k for k in layout if k not in KIND_TO_MJGEOM]
    if bad:
        raise ValueError(f"primitive_object.slot_layout: unknown kind(s) {bad} "
                         f"(known: {sorted(KIND_TO_MJGEOM)})")
    seed = raw.get("seed", None)
    cfg  = PrimitiveSampleCfg.from_config(raw, **cfg_overrides)
    validate_layout(cfg, layout)
    return {"cfg": cfg, "layout": layout,
            "seed": None if seed is None else int(seed)}


def make_obj_spec_provider_from_config(
    overall_cfg,
    seed: Optional[int] = None,
    verbose: bool = False,
    **cfg_overrides,
) -> Callable[[int], tuple]:
    """``make_obj_spec_provider`` wired entirely from a training config.

    Pulls the sampling ranges / slot layout / seed from
    ``cfg.Training.primitive_object`` and the shared physics knobs
    (``Training.friction``, ``sim_dt``, ``Training.obj_name``) from the same
    config, so a caller only needs::

        orch.build_sub_env(with_mjwarp=True, for_inference=True,
            obj_spec_provider=prim.make_obj_spec_provider_from_config(orch.overall_cfg))

    ``seed`` (arg) overrides the YAML ``seed``; ``cfg_overrides`` override any
    individual range (``size_scale=1.5``, ``primary_length=[0.1, 0.3]``, …).
    """
    opts = primitive_options_from_config(overall_cfg, **cfg_overrides)
    try:
        friction = tuple(float(f) for f in overall_cfg.Training.friction)
        sim_dt   = float(overall_cfg.sim_dt)
        root     = str(overall_cfg.Training.obj_name)
    except Exception:                      # plain dict / partial config
        friction, sim_dt, root = (1.2, 0.5, 0.4), None, DEFAULT_ROOT_BODY
    return make_obj_spec_provider(
        cfg=opts["cfg"], layout=opts["layout"],
        seed=opts["seed"] if seed is None else int(seed),
        friction=friction, sim_dt=sim_dt, root_body_name=root, verbose=verbose,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MIXED source — mesh dataset objects and procedural primitives in ONE scene
# ─────────────────────────────────────────────────────────────────────────────
# ⚠ Why this needs slot surgery: ``mujoco_warp.Model.geom_type`` is 1-D, so a
# geom SLOT has one type for every world. A mesh variant and a primitive
# variant therefore cannot share a slot — if they did, warp would simulate one
# of them with the other's collider type (physics/visual mismatch).
#
# The fix is a **disjoint slot layout** that every spec presents in the same
# order, so ``override_geoms_for_body`` (which maps src geoms onto skeleton
# slots by index) always lands on a slot of the right type:
#
#     root body : [ mesh slots 0 … M-1 ][ primitive slots M … M+P-1 ]
#     child body: [ mesh slots 0 … C-1 ]
#
#   * a MESH object fills its real geoms into the mesh slots (padded with
#     disabled mesh placeholders to exactly M) and leaves every primitive slot
#     disabled;
#   * a PRIMITIVE object leaves all mesh slots disabled and fills the primitive
#     slots it needs.
#
# Disabled slots carry ``group = 4`` → the existing per-world machinery marks
# them ``geom_dataid = -1`` and either banishes (mesh) or shrinks (primitive)
# them, so nothing collides. Mesh specs come FIRST so ``argmax(per_spec_ngeom)``
# (all equal after padding) selects a mesh spec as the skeleton anchor — sites,
# sensors and the body tree then come from the dataset layout as usual.

def _add_mesh_placeholder(body, meshname: str, name: str):
    g = body.add_geom(name=name)
    g.type        = mujoco.mjtGeom.mjGEOM_MESH          # type: ignore
    g.meshname    = meshname
    g.contype     = 0
    g.conaffinity = 0
    g.group       = 4
    g.density     = 0.0
    g.rgba        = np.zeros(4)
    return g


def _subtree_geom_counts(body, out=None):
    out = {} if out is None else out
    out[body.name] = max(out.get(body.name, 0), len(list(body.geoms)))
    for c in body.bodies:
        _subtree_geom_counts(c, out)
    return out


def _subtree_children(body):
    """``[(child_name, grandchildren…)]`` — the body tree below ``body``."""
    return [(c.name, _subtree_children(c)) for c in body.bodies]


def build_mixed_obj_spec_lst(
    n_specs: int,
    mesh_specs: list,
    mesh_ngeom: list,
    cfg: Optional[PrimitiveSampleCfg] = None,
    layout: Sequence[str] = DEFAULT_SLOT_LAYOUT,
    rng=None,
    seed: Optional[int] = None,
    root_body_name: str = DEFAULT_ROOT_BODY,
    friction=(1.2, 0.5, 0.4),
    sim_dt: Optional[float] = None,
    verbose: bool = False,
):
    """``(obj_spec_lst, per_spec_ngeom, meta)`` mixing ``mesh_specs`` (from
    :func:`hand_utils.build_obj_spec_lst`) with ``n_specs - len(mesh_specs)``
    procedural primitive objects, using the disjoint slot layout above.

    ``mesh_specs`` are MUTATED in place (padded) — pass freshly built ones.
    """
    cfg = cfg or PrimitiveSampleCfg()
    validate_layout(cfg, layout)
    if rng is None:
        rng = np.random.default_rng(seed)
    n_mesh = len(mesh_specs)
    n_prim = int(n_specs) - n_mesh
    if n_prim < 0:
        raise ValueError(f"mixed: {n_mesh} mesh specs > n_specs={n_specs}")
    if n_mesh == 0:
        return build_primitive_obj_spec_lst(
            n_specs, cfg=cfg, layout=layout, rng=rng, root_body_name=root_body_name,
            friction=friction, sim_dt=sim_dt, verbose=verbose)

    # ── slot budget from the mesh objects ────────────────────────────────
    roots      = [sp.worldbody.first_body() for sp in mesh_specs]
    geom_max: dict = {}
    for r in roots:
        for k, v in _subtree_geom_counts(r).items():
            geom_max[k] = max(geom_max.get(k, 0), v)
    tree      = _subtree_children(roots[int(np.argmax([len(_subtree_geom_counts(r)) for r in roots]))])
    fallback  = list(mesh_specs[0].meshes)[0].name
    root_name = roots[0].name

    # ── pad every mesh spec to the canonical layout ──────────────────────
    def _pad(body):
        want = int(geom_max.get(body.name, 0))
        for i in range(len(list(body.geoms)), want):
            _add_mesh_placeholder(body, fallback, f"{body.name}_meshpad_{i}")
        for c in body.bodies:
            _pad(c)
    for sp, r in zip(mesh_specs, roots):
        _pad(r)
        for i, kind in enumerate(layout):           # primitive slots, all disabled
            g = r.add_geom(name=f"{r.name}_primslot_{i}")
            g.type        = KIND_TO_MJGEOM[kind]
            g.size        = _PLACEHOLDER_SIZE.copy()
            g.contype     = 0
            g.conaffinity = 0
            g.group       = 4
            g.density     = 0.0
            g.rgba        = np.zeros(4)

    total_geoms = sum(geom_max.values()) + len(layout)

    # ── procedural specs mirroring that layout ───────────────────────────
    prim_specs, objects = [], []
    for i in range(n_prim):
        obj  = sample_primitive_object(rng, cfg, seed_id=i)
        spec = make_primitive_obj_spec(
            obj, idx=i, layout=layout, root_body_name=root_name,
            friction=friction, sim_dt=sim_dt,
            mesh_slots=int(geom_max.get(root_name, 0)), mesh_slot_name=fallback,
            child_tree=tree, child_geom_max=geom_max)
        prim_specs.append(spec)
        objects.append(obj)
        if verbose:
            print(f"[mixed primitive {i}] {obj.describe()}")

    specs   = list(mesh_specs) + prim_specs         # mesh first → skeleton anchor
    meta    = [MeshObjectRef(sp) for sp in mesh_specs] + objects
    ngeoms  = [total_geoms] * len(specs)
    if verbose:
        print(f"[mixed] {n_mesh} mesh + {n_prim} primitive specs | slots/root: "
              f"{geom_max.get(root_name, 0)} mesh + {len(layout)} primitive "
              f"| per_spec_ngeom={total_geoms}")
    return specs, ngeoms, meta


class MeshObjectRef:
    """Marker for a dataset-mesh variant in a mixed ``meta`` list, so callers
    can print / group both kinds uniformly (mirrors :class:`PrimitiveObject`)."""
    shape = "mesh"
    def __init__(self, spec):
        self.spec = spec
        root = spec.worldbody.first_body()
        self.name = root.name
        self.n_geom = len([g for g in root.geoms if int(g.group) != 4])
    @property
    def components(self):
        return []
    def describe(self) -> str:
        return f"[mesh] {self.name}  active_geoms={self.n_geom}  (dataset asset)"


def _project_root() -> str:
    """Repo root (has ``grit/`` + ``config/``), walking up from this file."""
    from pathlib import Path
    here = Path(__file__).resolve()
    for base in (here, *here.parents):
        if (base / "grit").is_dir() and (base / "config").is_dir():
            return str(base)
    raise RuntimeError(f"could not locate project root from {here}")


def make_mixed_obj_spec_provider(
    overall_cfg,
    seed: Optional[int] = None,
    verbose: bool = False,
    **cfg_overrides,
) -> Callable[[int], tuple]:
    """Provider for ``Training.object_source: mixed`` — dataset MESH objects and
    procedural PRIMITIVE objects as variants of ONE scene.

    ``Training.primitive_object.mesh_frac`` (0…1, default 0.5) is the share of
    variants taken from the mesh dataset; the rest are generated. The mesh half
    is drawn from ``obj_idxs`` when that pool is set, else uniformly from the
    active ``Dataset`` sections. The RNG is kept, so ``resample_objects()``
    redraws BOTH halves.
    """
    from grit.util import hand_utils                      # local: avoids a cycle
    opts   = primitive_options_from_config(overall_cfg, **cfg_overrides)
    raw    = {}
    try:
        raw = dict(overall_cfg.Training.get("primitive_object", {}) or {})
    except Exception:
        pass
    mesh_frac = float(raw.get("mesh_frac", 0.5))
    if not (0.0 <= mesh_frac <= 1.0):
        raise ValueError(f"Training.primitive_object.mesh_frac must be in [0,1], got {mesh_frac}")
    rng  = np.random.default_rng(seed if seed is not None else opts["seed"])
    root = _project_root()

    def _provider(n_specs: int):
        n_specs = int(n_specs)
        n_mesh  = int(round(n_specs * mesh_frac))
        n_mesh  = max(0, min(n_specs, n_mesh))
        mesh_specs, mesh_ngeom = [], []
        if n_mesh:
            _, obj_names, _, obj_xmls = hand_utils.get_obj_path_dir_lst(
                overall_cfg.Dataset, project_root=root)
            pool = list(overall_cfg.get("obj_idxs", None) or range(len(obj_xmls)))
            if len(pool) < n_mesh:
                raise ValueError(
                    f"mixed: mesh pool has {len(pool)} entries but {n_mesh} mesh "
                    f"variants are requested (n_sub_env={n_specs} × mesh_frac={mesh_frac})")
            idxs = [int(i) for i in rng.choice(len(pool), size=n_mesh, replace=False)]
            idxs = [pool[i] for i in idxs]
            mesh_specs, mesh_ngeom, _ = hand_utils.build_obj_spec_lst(
                [obj_xmls[i] for i in idxs], [obj_names[i] for i in idxs],
                friction=list(overall_cfg.Training.friction),
                use_simple=not bool(overall_cfg.Dataset.Objaverse.use),
                geom_collision_type=str(overall_cfg.get("geom_collision_type", "mesh")),
                rename_body_name=str(overall_cfg.Training.obj_name),
                sim_dt=float(overall_cfg.sim_dt), verbose=False)
            if verbose:
                print(f"[mixed] mesh dataset indices: {idxs}")
        return build_mixed_obj_spec_lst(
            n_specs, mesh_specs, mesh_ngeom, cfg=opts["cfg"], layout=opts["layout"],
            rng=rng, root_body_name=str(overall_cfg.Training.obj_name),
            friction=tuple(float(f) for f in overall_cfg.Training.friction),
            sim_dt=float(overall_cfg.sim_dt), verbose=verbose)

    _provider.rng = rng                                   # type: ignore[attr-defined]
    return _provider
