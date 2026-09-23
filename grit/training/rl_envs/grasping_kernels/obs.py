"""Observation block kernels (one per obs term) — warp kernels of the grasping task."""
import mujoco as _mj  # type: ignore
import numpy as np  # type: ignore
import warp as wp  # type: ignore
from .common import _grit_force_log1p


# ══════════════════════════════════════════════════════════════════════════
# Observation — split into per-component kernels. Each writes its own block at
# a host-computed ``offset``; the slot layout / offset accumulation lives in a
# single place (``_collect_obs_kernel``) so blocks are independently reusable
# and a base-accounting mistake can't silently clobber a neighbour.
# ══════════════════════════════════════════════════════════════════════════


@wp.kernel
def _obs_joint_qpos_kernel(
    qpos:       wp.array(dtype=float, ndim=2),   # type: ignore
    finger_qpa: wp.array(dtype=int,   ndim=1),   # type: ignore
    n_ctrl:     int,
    offset:     int,
    out_obs:    wp.array(dtype=float, ndim=2),   # type: ignore
):
    """finger_qpos: current hand joint qpos at the ctrl-addressed joints."""
    w = wp.tid()
    for i in range(n_ctrl):
        qa = finger_qpa[i]
        if qa >= 0:
            out_obs[w, offset + i] = qpos[w, qa]
        else:
            out_obs[w, offset + i] = 0.0


@wp.kernel
def _obs_qpos_tax_err_kernel(
    qpos:        wp.array(dtype=float, ndim=2),  # type: ignore
    finger_qpa:  wp.array(dtype=int,   ndim=1),  # type: ignore
    target_qpos: wp.array(dtype=float, ndim=2),  # type: ignore
    n_ctrl:      int,
    deadzone:    float,                           # |err| < deadzone → 0 (JAX v2 hand_pose_error; 0 = raw)
    offset:      int,
    out_obs:     wp.array(dtype=float, ndim=2),  # type: ignore
):
    """qpos_taxonomy_err: target_qpos - qpos[ctrl_adr] (per-joint mimic err).

    Deadzone masking following the JAX v2 (``hand_pose_error``) convention: write 0
    when ``|err| < deadzone``, otherwise the **raw error** (no err−dz shrinkage).
    This aligns the obs signal with the mimic hinge reward's "inside the band = no
    penalty", so the policy does not waste actions chasing tiny errors inside the band."""
    w = wp.tid()
    for i in range(n_ctrl):
        qa = finger_qpa[i]
        if qa >= 0:
            err = target_qpos[w, i] - qpos[w, qa]
            if wp.abs(err) < deadzone:
                err = 0.0
            out_obs[w, offset + i] = err
        else:
            out_obs[w, offset + i] = 0.0


@wp.kernel
def _obs_torque_proxy_kernel(
    qpos:       wp.array(dtype=float, ndim=2),   # type: ignore
    finger_qpa: wp.array(dtype=int,   ndim=1),   # type: ignore
    ctrl:       wp.array(dtype=float, ndim=2),   # type: ignore
    n_ctrl:     int,
    use_log1p:  int,                              # 1 → signed log1p compression: sign(e)·log1p(|e|)
    offset:     int,
    out_obs:    wp.array(dtype=float, ndim=2),   # type: ignore
):
    """torque_proxy: ctrl_target - qpos[ctrl_adr] (position err ~ torque)."""
    w = wp.tid()
    for i in range(n_ctrl):
        qa  = finger_qpa[i]
        cur = float(0.0)
        if qa >= 0:
            cur = qpos[w, qa]
        e = ctrl[w, i] - cur
        # Signed log1p (OBS_TORQUE_PROXY_LOG1P): compresses the tail when contact
        # makes the setpoint run far ahead of the joint (saturated pressing), while
        # preserving the sign (flex/extend) and the slope near 0. Passthrough channel
        # in obs_norm.
        if use_log1p == 1:
            s = float(1.0)
            if e < 0.0:
                s = -1.0
            e = s * wp.log(1.0 + wp.abs(e))
        out_obs[w, offset + i] = e


@wp.kernel
def _obs_finger_link_mask_kernel(
    finger_link_mask: wp.array(dtype=float, ndim=2),  # type: ignore
    n_sensors:        int,
    offset:           int,
    out_obs:          wp.array(dtype=float, ndim=2),  # type: ignore
):
    """specific_finger_link_mask: taxonomy active(1)/rest(0), sensor space."""
    w = wp.tid()
    for i in range(n_sensors):
        out_obs[w, offset + i] = finger_link_mask[w, i]


@wp.kernel
def _obs_per_slot_contact_kernel(
    obj_impulse_per_slot:  wp.array(dtype=float, ndim=2),  # type: ignore
    self_impulse_per_slot: wp.array(dtype=float, ndim=2),  # type: ignore
    tbl_impulse_per_slot:  wp.array(dtype=float, ndim=2),  # type: ignore
    n_contact_slots:       int,
    log1p_cap:             float,                           # >0 → log1p/[0,1] compression of the impulse channels (bool channels unchanged)
    inv_log1p_cap:         float,                           # 1 / log1p(cap) (precomputed on the host)
    offset:                int,
    out_obs:               wp.array(dtype=float, ndim=2),  # type: ignore
):
    """Per-slot contact (REUSED reward impulse buffers):
    obj_bool(n) | obj_impulse(n) | obstacle_bool(n) | obstacle_impulse(n).
    Impulse channels are log-compressed into [0,1] when ``log1p_cap > 0`` (OBS_FORCE_LOG1P)."""
    w = wp.tid()
    for k in range(n_contact_slots):
        oi = obj_impulse_per_slot[w, k]
        ob = float(0.0)
        if oi > 0.0:
            ob = 1.0
        out_obs[w, offset + k]                   = ob
        out_obs[w, offset + n_contact_slots + k] = _grit_force_log1p(oi, log1p_cap, inv_log1p_cap)
        si = self_impulse_per_slot[w, k]
        ti = tbl_impulse_per_slot[w, k]
        obstacle = float(0.0)
        if si > 0.0:
            obstacle = 1.0
        if ti > 0.0:
            obstacle = 1.0
        out_obs[w, offset + 2 * n_contact_slots + k] = obstacle
        out_obs[w, offset + 3 * n_contact_slots + k] = _grit_force_log1p(si + ti, log1p_cap, inv_log1p_cap)


@wp.kernel
def _obs_obj_touch_force_kernel(
    obj_touch_force: wp.array(dtype=float, ndim=1),  # type: ignore
    log1p_cap:       float,
    inv_log1p_cap:   float,
    offset:          int,
    out_obs:         wp.array(dtype=float, ndim=2),  # type: ignore
):
    """Total force applied to the object (Sum obj touch sensors, force-scaled;
    log-compressed into [0,1] when log1p_cap > 0)."""
    w = wp.tid()
    out_obs[w, offset] = _grit_force_log1p(obj_touch_force[w], log1p_cap, inv_log1p_cap)


@wp.kernel
def _obs_wrist_obj_vel_kernel(
    cvel:          wp.array(dtype=wp.spatial_vector, ndim=2),  # type: ignore
    xmat:          wp.array(dtype=wp.mat33, ndim=2),           # type: ignore
    wrist_body_id: int,
    obj_body_id:   int,
    offset:        int,
    out_obs:       wp.array(dtype=float, ndim=2),             # type: ignore
):
    """wrist/object linear+angular velocity in the wrist frame (obj relative).
    cvel: spatial_top=angular, spatial_bottom=linear (MuJoCo convention)."""
    w = wp.tid()
    Rw_T = wp.transpose(xmat[w, wrist_body_id])
    wcv  = cvel[w, wrist_body_id]
    wrist_vel  = Rw_T * wp.spatial_bottom(wcv)  # type: ignore
    wrist_qvel = Rw_T * wp.spatial_top(wcv)  # type: ignore
    ocv = cvel[w, obj_body_id]
    obj_vel  = Rw_T * wp.spatial_bottom(ocv) - wrist_vel  # type: ignore
    obj_qvel = Rw_T * wp.spatial_top(ocv)    - wrist_qvel  # type: ignore
    out_obs[w, offset + 0]  = wrist_vel[0]
    out_obs[w, offset + 1]  = wrist_vel[1]
    out_obs[w, offset + 2]  = wrist_vel[2]
    out_obs[w, offset + 3]  = wrist_qvel[0]
    out_obs[w, offset + 4]  = wrist_qvel[1]
    out_obs[w, offset + 5]  = wrist_qvel[2]
    out_obs[w, offset + 6]  = obj_vel[0]
    out_obs[w, offset + 7]  = obj_vel[1]
    out_obs[w, offset + 8]  = obj_vel[2]
    out_obs[w, offset + 9]  = obj_qvel[0]
    out_obs[w, offset + 10] = obj_qvel[1]
    out_obs[w, offset + 11] = obj_qvel[2]
