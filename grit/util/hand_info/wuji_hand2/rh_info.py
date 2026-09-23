"""Wuji Hand 2 **right** hand info (Grit format).

Asset: ``asset/dextrous_hand/wuji_hand2/wuji_hand2_rh_w_fore_arm_touch_contact.xml``
(built by ``scripts/asset_tools/build_wuji_hand2_xml.py`` from the official
wuji-description hand2_beta1 MJCF). Palm frame: fingers +Z, palm +X, thumb +Y.

Naming: bodies ``wuji_r_<finger>_<link>`` (finger ∈ thumb / index_finger /
middle_finger / ring_finger / pinky, link ∈ proximal / proximal_abd / middle /
distal), fingertip bodies ``wuji_r_<th|ff|mf|rf|lf>_end`` / ``_face_end``,
joints / actuators / sites / sensors ``wuji_rh_*``.

JOINT ORDER (differs from every other hand — abduction is the 2nd joint of
each finger, not the 1st, and lives in its own ``*_proximal_abd`` body)::

    qpos / ctrl index : finger : joint (body)
      0  1  2  3      thumb   cmc_flex (proximal) | cmc_abd (proximal_abd) | mcp (middle) | ip (distal)
      4  5  6  7      index   mcp_flex (proximal) | mcp_abd (proximal_abd) | pip (middle) | dip (distal)
      8  9 10 11      middle  "
     12 13 14 15      ring    "
     16 17 18 19      pinky   "

    robotis / tesollo / allegro / allex : [abd, flex, flex, flex] per finger.
    Nothing in grit/ hard-codes the per-finger order (joints are a flat vector,
    finger grouping is name-based), but every taxonomy ``qpos`` MUST follow this
    XML order, and the "mcp" role in ``rh_af_parts`` is the *flexion* knuckle
    body ``*_proximal`` (the ``*_proximal_abd`` body sits 14.5 mm distal of it).

TODO(hand_center): ``rh_hand_center`` below is a PROVISIONAL value computed with
the nb09 recipe (thumb distal → mean(ff,mf,rf tips) ×0.33, heading from the palm
site, Y zeroed) at the REST pose — relabel with
notebook/hand/01_hand_setup/09_label_hand_center.ipynb.
"""
import numpy as np

rh_mocap_name       = "wuji_rh:mocap"
rh_wrist_base_name  = "wuji_rh_palm"
rh_palm_site_name   = "wuji_rh_palm_touch"
# Hand-center keypoint (thumb distal link)
rh_thumb_link1_name = "wuji_r_thumb_distal"
rh_th_tip_name = ["wuji_r_th_end", "wuji_r_th_face_end", "wuji_rh_th_distal_touch"]
rh_ff_tip_name = ["wuji_r_ff_end", "wuji_r_ff_face_end", "wuji_rh_ff_distal_touch"]
rh_mf_tip_name = ["wuji_r_mf_end", "wuji_r_mf_face_end", "wuji_rh_mf_distal_touch"]
rh_rf_tip_name = ["wuji_r_rf_end", "wuji_r_rf_face_end", "wuji_rh_rf_distal_touch"]
rh_lf_tip_name = ["wuji_r_lf_end", "wuji_r_lf_face_end", "wuji_rh_lf_distal_touch"]

rh_hand_center = np.array([[ 0.96288,  0.     , -0.26994,  0.02541],
                            [ 0.00002,  1.     ,  0.00002,  0.00124],
                            [ 0.26994, -0.00003,  0.96288,  0.075  ],
                            [ 0.     ,  0.     ,  0.     ,  1.     ]])

rh_body_parts = ['wuji_rh_palm',
                'wuji_r_thumb_proximal', 'wuji_r_thumb_proximal_abd', 'wuji_r_thumb_middle', 'wuji_r_thumb_distal', 'wuji_r_th_end', 'wuji_r_th_face_end',
                'wuji_r_index_finger_proximal', 'wuji_r_index_finger_proximal_abd', 'wuji_r_index_finger_middle', 'wuji_r_index_finger_distal', 'wuji_r_ff_end', 'wuji_r_ff_face_end',
                'wuji_r_middle_finger_proximal', 'wuji_r_middle_finger_proximal_abd', 'wuji_r_middle_finger_middle', 'wuji_r_middle_finger_distal', 'wuji_r_mf_end', 'wuji_r_mf_face_end',
                'wuji_r_ring_finger_proximal', 'wuji_r_ring_finger_proximal_abd', 'wuji_r_ring_finger_middle', 'wuji_r_ring_finger_distal', 'wuji_r_rf_end', 'wuji_r_rf_face_end',
                'wuji_r_pinky_proximal', 'wuji_r_pinky_proximal_abd', 'wuji_r_pinky_middle', 'wuji_r_pinky_distal', 'wuji_r_lf_end', 'wuji_r_lf_face_end',
                ] # 1 + 5×6 = 31
# af_parts convention: base + 3 per finger = (mcp knuckle, pip, tip). For wuji the
# knuckle is the FLEXION body ``*_proximal`` (abduction body is distal of it).
rh_af_parts   = ['wuji_rh_palm',
                'wuji_r_thumb_proximal',         'wuji_r_thumb_middle',         'wuji_r_th_end',
                'wuji_r_index_finger_proximal',  'wuji_r_index_finger_middle',  'wuji_r_ff_end',
                'wuji_r_middle_finger_proximal', 'wuji_r_middle_finger_middle', 'wuji_r_mf_end',
                'wuji_r_ring_finger_proximal',   'wuji_r_ring_finger_middle',   'wuji_r_rf_end',
                'wuji_r_pinky_proximal',         'wuji_r_pinky_middle',         'wuji_r_lf_end',
                ] # 16

# Sensor order in the XML: palm, forearm, then (proximal_abd, middle, distal) × (th ff mf rf lf)
fore_arm_touch_sensor = "wuji_r_arm_part_touch"
hand_touch_sensors = [
    "wuji_rh_palm_body_touch",
    "wuji_rh_th_proximal_abd_body_touch", "wuji_rh_th_middle_body_touch", "wuji_rh_th_distal_body_touch",
    "wuji_rh_ff_proximal_abd_body_touch", "wuji_rh_ff_middle_body_touch", "wuji_rh_ff_distal_body_touch",
    "wuji_rh_mf_proximal_abd_body_touch", "wuji_rh_mf_middle_body_touch", "wuji_rh_mf_distal_body_touch",
    "wuji_rh_rf_proximal_abd_body_touch", "wuji_rh_rf_middle_body_touch", "wuji_rh_rf_distal_body_touch",
    "wuji_rh_lf_proximal_abd_body_touch", "wuji_rh_lf_middle_body_touch", "wuji_rh_lf_distal_body_touch",
]
_stems = ["palm"] + [f"{f}_{l}" for f in ("th", "ff", "mf", "rf", "lf") for l in ("proximal_abd", "middle", "distal")]
hand_contact_sensors_for_object = [f"wuji_rh_{s}_body_obj_contact" for s in _stems]
hand_contact_sensors_for_table  = [f"wuji_rh_{s}_table_contact"    for s in _stems]
hand_contact_sensors            = [f"wuji_rh_{s}_self_coll"        for s in _stems]
del _stems

anchor_body_name = "wuji_r_middle_finger_proximal"   # middle-finger knuckle (cf. robotis finger_r_link3_2)
palm_sensor_idx = 0
fore_arm_sensor_idx = 1
finger_motor_num = 20
finger_link_num = 20
hand_qvel_num = 26

ctrl_joint_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
tip_body_idx = [4, 7, 10, 13, 16]   # distal links in hand_sensor_body_name_list (palm, arm, then 3 per finger)
tip_end_name = "_end"

# Action scale
wrist_transl_scale = 80.0
wrist_rot_scale = 40.0
finger_action_scale = 5.0
