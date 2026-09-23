"""Wuji Hand 2 **left** hand info — derived from ``rh_info`` by mirroring.

Asset: ``asset/dextrous_hand/wuji_hand2/wuji_hand2_lh_w_fore_arm_touch_contact.xml``
(built from the official left MJCF; in the palm frame it is the Y-mirror of the
right hand and the same qpos gives the mirrored pose → ``QPOS_MIRROR_SIGN`` all +1).

Name rules: ``wuji_rh:`` → ``wuji_lh:``, ``wuji_rh_`` → ``wuji_lh_``, ``wuji_r_`` → ``wuji_l_``.
``rh_hand_center`` keeps its rotation and flips the Y translation (provisional,
see rh_info TODO).
"""
from grit.util.hand_info import mirror as _mirror
from grit.util.hand_info.wuji_hand2 import rh_info as _rh

NAME_SUBS = (
    ("wuji_rh:", "wuji_lh:"),
    ("wuji_rh_", "wuji_lh_"),
    ("wuji_r_", "wuji_l_"),
)
MIRROR_AXIS = 1   # Y
QPOS_MIRROR_SIGN = [1] * 20

_ns = _mirror.mirror_info(_rh, NAME_SUBS, mirror_axis=MIRROR_AXIS)
globals().update({k: v for k, v in vars(_ns).items()})
del _ns, _mirror, _rh
