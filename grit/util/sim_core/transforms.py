"""Minimal rigid-transform helpers (numpy).

Conventions
-----------
* ``T``   : 4x4 homogeneous transform
* ``p``   : (3,) position
* ``R``   : 3x3 rotation matrix
* ``rpy`` : roll / pitch / yaw, applied as ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``
* quaternions are ``[w, x, y, z]`` (MuJoCo order)
"""
from __future__ import annotations

import numpy as np


def t2p(T):
    """Position part of a 4x4 transform (a view, not a copy)."""
    return T[:3, 3]


def t2r(T):
    """Rotation part of a 4x4 transform (a view, not a copy)."""
    return T[:3, :3]


def t2pr(T):
    """``(p, R)`` of a 4x4 transform."""
    return T[:3, 3], T[:3, :3]


def pr2t(p=(0, 0, 0), R=np.eye(3)):
    """Build a 4x4 transform from position ``p`` and rotation ``R``."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(p, dtype=np.float64).ravel()[:3]
    return T


def p2t(p=(0, 0, 0)):
    """4x4 pure-translation transform."""
    return pr2t(p, np.eye(3))


def r2t(R=np.eye(3)):
    """4x4 pure-rotation transform."""
    return pr2t((0, 0, 0), R)


def rpy2r(rpy, unit="rad"):
    """Roll/pitch/yaw → rotation matrix, ``R = Rz(yaw) Ry(pitch) Rx(roll)``."""
    rpy = np.asarray(rpy, dtype=np.float64).ravel()
    if unit == "deg":
        rpy = np.deg2rad(rpy)
    elif unit != "rad":
        raise ValueError(f"[rpy2r] unknown unit: {unit!r}")
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    return np.array([
        [cy * cp, -sy * cr + cy * sp * sr,  sy * sr + cy * sp * cr],
        [sy * cp,  cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr],
        [-sp,      cp * sr,                 cp * cr],
    ], dtype=np.float64)


def r2quat(R):
    """Rotation matrix → unit quaternion ``[w, x, y, z]`` with ``w >= 0``.

    Supports a leading batch dimension (``(..., 3, 3)`` → ``(..., 4)``).
    Uses the symmetric-matrix eigenvector method, which is robust for all
    rotations (no branch on the trace).
    """
    R = np.asarray(R, dtype=np.float64)
    Qxx, Qyx, Qzx = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    Qxy, Qyy, Qzy = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    Qxz, Qyz, Qzz = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    K = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # eigh works on the lower triangle, so the upper half can stay zero.
    vals, vecs = np.linalg.eigh(K)
    idx = np.argmax(vals, axis=-1)
    q = np.take_along_axis(vecs, idx[..., None, None], axis=-1)[..., 0]   # (..., 4) as [x, y, z, w]
    q = q[..., [3, 0, 1, 2]]
    q = np.where((q[..., :1] < 0), -q, q)
    return q


def quat2r(q, order="wxyz"):
    """Unit quaternion → rotation matrix."""
    q = np.asarray(q, dtype=np.float64).ravel()
    if order == "wxyz":
        w, x, y, z = q
    elif order == "xyzw":
        x, y, z, w = q
    else:
        raise ValueError(f"[quat2r] invalid order: {order!r}")
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def np_uv(vec, eps=1e-12):
    """Unit vector (returns the input unchanged when its norm is below ``eps``)."""
    vec = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(vec)
    return vec if n < eps else vec / n


def get_R_from_twopoints(p_fr, p_to):
    """Rotation whose local +Z axis points from ``p_fr`` to ``p_to``.

    Used to orient MuJoCo arrow / cylinder / line markers (their length runs
    along local +Z). Rodrigues rotation from a nearly-+Z reference direction;
    the tiny X/Y perturbation in the reference avoids the degenerate
    anti-parallel case.
    """
    p_fr = np.asarray(p_fr, dtype=np.float64)
    p_to = np.asarray(p_to, dtype=np.float64)
    d = p_to - p_fr
    n = np.linalg.norm(d)
    if n < 1e-8:
        return np.eye(3)
    a = np.array([1e-10, -1e-10, 1.0])
    b = d / n
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s == 0.0:
        return np.eye(3)
    S = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + S + S @ S * (1.0 - float(a @ b)) / (s * s)
