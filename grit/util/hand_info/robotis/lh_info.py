"""ROBOTIS SH5 **left** hand info — derived from ``rh_info`` by mirroring.

Asset: ``asset/dextrous_hand/robotis_sh5/ffw_sh5_lh_w_fore_arm_touch_contact.xml``
(a Y-mirror of the right-hand XML; identical joint axes / ordering).

Name rules (right → left), applied in order::

    sh5_rh:   → sh5_lh:      (mocap)
    _rh_      → _lh_         (sites / sensors / joints / actuators / *_end bodies)
    _r_       → _l_          (hx5_r_base, finger_r_link*, sh5_r_arm_part*)
    r_link    → l_link       (taxonomy specific_finger_names, no leading '_')

``rh_hand_center`` keeps its rotation and flips the Y translation (the
labeling recipe is mirror-invariant in rotation — see hand_info/mirror.py).
Attribute names stay ``rh_*`` on purpose: the codebase reads them generically.
"""
from grit.util.hand_info import mirror as _mirror
from grit.util.hand_info.robotis import rh_info as _rh

NAME_SUBS = (
    ("sh5_rh:", "sh5_lh:"),
    ("_rh_", "_lh_"),
    ("_r_", "_l_"),
    ("r_link", "l_link"),
)
MIRROR_AXIS = 1   # Y

# Joint-sign convention of the mirrored asset (verified by per-joint FK sweep,
# output/logs/left_hand/01_lh_mirror_fk.txt): abduction (``*_1``) and the
# thumb ``1_2 / 1_3 / 1_4`` flip sign on the left (their ranges are mirrored,
# e.g. R[-pi,0] → L[0,pi]); flexion ``*_2 / *_3 / *_4`` of the four fingers keep
# the same sign. Applied to the tail ``finger_link_num`` entries of every
# taxonomy ``qpos`` (the 27-dim 'none' entry keeps its leading free-joint part).
QPOS_MIRROR_SIGN = [-1, -1, -1, -1,   # thumb   1_1 .. 1_4
                    -1,  1,  1,  1,   # index   2_1 .. 2_4
                    -1,  1,  1,  1,   # middle  3_1 .. 3_4
                    -1,  1,  1,  1,   # ring    4_1 .. 4_4
                    -1,  1,  1,  1]   # pinky   5_1 .. 5_4

_ns = _mirror.mirror_info(_rh, NAME_SUBS, mirror_axis=MIRROR_AXIS)
globals().update({k: v for k, v in vars(_ns).items()})
del _ns, _mirror, _rh   # keep only data attributes (HandUtils copies dir())
