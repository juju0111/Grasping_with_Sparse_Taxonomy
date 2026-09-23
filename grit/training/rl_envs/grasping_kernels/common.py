"""Shared warp helper functions (rotation / log-compression / …) used by the kernels."""
import mujoco as _mj  # type: ignore
import numpy as np  # type: ignore
import warp as wp  # type: ignore
from grit.training.orchestrator.base import _grit_r2quat_wxyz


@wp.func
def _grit_force_log1p(f: float, cap: float, inv_log1p_cap: float) -> float:
    """Compress a contact force into [0,1]: ``log1p(clip(f,0,cap)) / log1p(cap)``.
    Bounds heavy-tail spikes (thousands of N) while preserving the slope near 0
    (resolution for light contacts). ``cap <= 0`` → raw passthrough."""
    if cap <= 0.0:
        return f
    c = f
    if c < 0.0:
        c = 0.0
    if c > cap:
        c = cap
    return wp.log(1.0 + c) * inv_log1p_cap


@wp.func
def _grit_rot_to_rotvec(R_rel: wp.mat33):  # type: ignore
    """Relative rotation matrix → axis-angle vector (log map; in the previous body frame).

    Via quaternion: ``q = quat_from_matrix(R_rel)`` (x,y,z,w), sign-normalized to the
    shortest arc, then ``rotvec = 2·atan2(|v|, w)·v/|v|`` (``2·v`` for small angles)."""
    q = wp.quat_from_matrix(R_rel)
    qw = q[3]
    v = wp.vec3(q[0], q[1], q[2])
    if qw < 0.0:                     # shortest arc (q ≡ -q)
        qw = -qw
        v = -v
    s = wp.length(v)
    if s < 1.0e-8:
        return v * 2.0               # small-angle approximation: rotvec ≈ 2·v
    return v * (2.0 * wp.atan2(s, qw) / s)


@wp.func
def _grit_r2rpy(R: wp.mat33):   # type: ignore
    """Rotation matrix → (roll, pitch, yaw) euler (extrinsic XYZ; atan2-based,
    singularity-robust). Matches the standard MuJoCo-style rpy used by the
    legacy ``mjwarp_utils.r2rpy`` (roll about x, pitch about y, yaw about z)."""
    roll  = wp.atan2(R[2, 1], R[2, 2])
    pitch = wp.atan2(-R[2, 0], wp.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2]))
    yaw   = wp.atan2(R[1, 0], R[0, 0])
    return wp.vec3(roll, pitch, yaw)
