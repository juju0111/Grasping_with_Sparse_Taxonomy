"""Wuji Hand 2 grasp-taxonomy annotation (30 Feix taxonomies, robotis_sh5 layout).

qpos: 20 finger joints in XML order — **abduction is the 2nd joint of each finger**
(thumb cmc_flex, cmc_abd, mcp, ip | index mcp_flex, mcp_abd, pip, dip | middle | ring | pinky).
Poses were transferred from the tesollo taxonomy with the learned tesollo->wuji_hand2 IK
(ri_motion_v5 geoRT_refector: geort/taxonomy_data.py, tools/gen_taxonomy_transfer.py,
2026-08-31) and are therefore APPROXIMATE (mean normalised flexion error 0.122). Re-label any
pose with notebook/hand/01_hand_setup/05~08_fk_taxonomy*.ipynb (never copy a qpos vector from
another hand — the joint order differs).

Layout conventions
  * ``specific_finger_names`` are lowercase substrings of body names (palm sensor body =
    ``wuji_r_wrist`` -> 'wrist'; use 'middle_finger', not 'middle', which would also match every
    ``*_middle`` link). ``specific_ctrl_names`` are substrings of actuator names
    (``wuji_rh_THJ0`` ... -> 'thj', 'ffj', 'mfj', 'rfj', 'lfj').
  * ``hand_face_dir_*`` (16 = palm + 3 links x 5 fingers) follow the TOUCH-SENSOR order of the
    XML: palm, th, ff, mf, rf, lf (link = proximal_abd, middle, distal). ``hand_af_dir_*``
    (11 = palm + 2 per finger) follow ``rh_af_parts``.
  * Pad normal = **+Y** in every link/wrist frame (official convention; the left hand's -Y is
    handled by hand_info/mirror.py). ``lateral`` / ``writing_tripod`` put the index / middle
    contact face on the radial side (**+X**); ``palmar`` keeps the thumb pad (+Y already points
    along the palm normal) — axes chosen by matching the robotis_sh5 annotation in a canonical
    hand frame.
"""
import numpy as np

# 1
large_diameter = {
        'qpos' : np.array([ 1.103247, -0.712446,  0.085025,  1.060944,  0.692893, -0.289903,  0.706879,  0.701553,
                            0.537300, -0.137401,  0.826067,  0.516561,  0.548657, -0.025003,  0.787982,  0.534585,
                            0.563091,  0.135822,  0.824998,  0.765393]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 2
small_diameter = {
        'qpos' : np.array([ 0.961274, -0.449353,  0.266037,  1.279427,  0.514720, -0.252621,  1.294006,  1.231658,
                            0.644723,  0.016096,  1.340214,  1.264556,  0.639747,  0.086143,  1.539145,  1.070134,
                            0.959617,  0.235189,  1.360704,  1.385257]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 3
medium_diameter = {
        'qpos' : np.array([ 1.039671, -0.531693,  0.175144,  1.175295,  0.683736, -0.226386,  0.918275,  1.264047,
                            0.666685, -0.096612,  1.246304,  0.925246,  0.657548, -0.011164,  1.191154,  1.273107,
                            0.669410,  0.247256,  1.213996,  1.276109]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 4
adducted_thumb = {
        'qpos' : np.array([ 0.300248, -0.497595,  0.016626,  1.062981,  0.539690, -0.069830,  1.380728,  1.458082,
                            0.657643, -0.078882,  1.438000,  1.420532,  0.570089, -0.059524,  1.331191,  1.374054,
                            0.558499,  0.082321,  1.279263,  1.458320]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 5
light_tool = {
        'qpos' : np.array([ 0.066132,  0.034446,  0.357581,  1.224972,  0.653193, -0.109249,  1.446678,  1.444645,
                            0.828678, -0.062808,  1.574883,  1.408864,  0.751600, -0.055616,  1.628015,  1.255788,
                            0.954982,  0.144004,  1.370198,  1.400110]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 6
prismatic_4_finger = {
        'qpos' : np.array([ 1.127522, -0.679484,  0.095877,  1.059030,  0.657005, -0.214224,  1.097598,  0.571871,
                            0.673721, -0.117429,  1.375206,  0.015412,  0.658289, -0.001352,  1.163139,  0.076884,
                            0.557410,  0.238953,  1.075839,  1.007816]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 7
prismatic_3_finger = {
        'qpos' : np.array([ 1.114100, -0.675577,  0.086486,  1.069489,  0.771871, -0.189978,  1.111896,  0.081774,
                            0.686723, -0.087991,  1.120466,  0.045093,  0.757405,  0.022758,  1.282312,  0.052296,
                            1.392215,  0.348030,  1.770153,  1.178499]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 8
prismatic_2_finger = {
        'qpos' : np.array([ 0.888614, -0.493892,  0.095588,  0.954884,  0.841323, -0.271399,  0.782915,  0.148314,
                            0.817638, -0.118493,  1.145800,  0.024608,  0.625849, -0.039757,  1.790425,  1.332571,
                            1.392215,  0.348030,  1.770153,  1.178499]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 9
palmar_pinch = {
        'qpos' : np.array([ 0.916909, -0.620436,  0.042674,  0.896792,  0.941262, -0.098983,  0.750934,  0.115103,
                            0.604866, -0.113918, -0.010520,  0.222322,  0.574687, -0.056305,  0.057279,  0.234899,
                            0.210124,  0.108699,  0.434216,  0.115431]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger'],
        'specific_ctrl_names' : ['thj', 'ffj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 10
power_disk = {
        'qpos' : np.array([ 1.200294, -0.721661,  0.609636,  1.315008,  0.166584, -0.060538,  1.571149,  1.299410,
                            0.278939,  0.057832,  1.661722,  1.255575,  0.318151,  0.072636,  1.758096,  1.210480,
                            1.072673,  0.258683,  1.426763,  1.387308]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 11
power_sphere = {
        'qpos' : np.array([ 1.112923, -0.780372,  0.318898,  1.172253,  0.852504, -0.211991,  1.059892,  1.047516,
                            0.803211, -0.082658,  1.249831,  0.844810,  0.762541,  0.058818,  1.232103,  0.904560,
                            0.978911,  0.330857,  1.339678,  1.277843]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 12
precision_disk = {
        'qpos' : np.array([ 0.727601, -0.388341,  0.256545,  1.094835,  0.343795, -0.253756,  1.309953,  0.612677,
                            0.186672,  0.020676,  1.675322,  0.378313,  0.491571,  0.178325,  1.383838,  0.949606,
                            0.837087,  0.422828,  1.317980,  1.274219]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 13
precision_sphere = {
        'qpos' : np.array([ 1.066362, -0.737010,  0.114511,  0.993084,  0.635085, -0.233493,  1.361400,  0.083918,
                            0.683815, -0.011804,  1.061145,  0.055406,  0.513440,  0.179341,  1.694091,  0.032933,
                            0.987649,  0.382446,  1.212360,  0.709900]),
        'specific_finger_names' : ['thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 14
tripod = {
        'qpos' : np.array([ 0.981912, -0.596419,  0.116092,  0.957860,  0.603415, -0.215495,  1.437703,  0.024047,
                            0.856612, -0.097402,  1.206527,  0.050623,  0.796089,  0.091288,  1.852948,  1.382543,
                            1.406645,  0.361006,  1.744561,  1.113925]),
        'specific_finger_names' : ['thumb', 'index_finger', 'middle_finger'],
        'specific_sensor_names' : ['thumb', 'index_finger', 'middle_finger'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 15
fixed_hook = {
        'qpos' : np.array([ 0.300248, -0.497595,  0.016626,  1.062981,  0.539690, -0.069830,  1.380728,  1.458082,
                            0.657643, -0.078882,  1.438000,  1.420532,  0.570089, -0.059524,  1.331191,  1.374054,
                            0.558499,  0.082321,  1.279263,  1.458320]),
        'specific_finger_names' : ['wrist', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['wrist', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 16
lateral = {
        'qpos' : np.array([ 0.389484, -0.261999,  0.330043,  1.276322,  0.540626, -0.197701,  1.619945,  1.434500,
                            0.805545, -0.071976,  1.839840,  1.419139,  0.794542, -0.095696,  1.818208,  1.331220,
                            1.261031,  0.243743,  1.730885,  1.337943]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger'],
        'specific_ctrl_names' : ['thj', 'ffj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 0,0,0, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 0,0, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 17
index_finger_extension = {
        'qpos' : np.array([ 0.470581, -0.146771,  0.217042,  1.236619,  0.224357, -0.431580, -0.044285,  0.246880,
                            0.661395, -0.149006,  1.433761,  1.425295,  0.764941, -0.052305,  1.574850,  1.255281,
                            0.966226,  0.143581,  1.375090,  1.402400]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 18
extension_type = {
        'qpos' : np.array([ 0.717991, -0.251757,  0.420232,  1.323575,  0.015777, -0.503337, -0.101209,  0.551622,
                            0.089459, -0.141090,  0.071329,  0.430781,  0.005339,  0.083826,  0.026471,  0.574001,
                           -0.014971,  0.391147,  0.297134,  0.604402]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 19
writing_tripod = {
        'qpos' : np.array([ 0.582068, -0.249236,  0.202629,  1.191072,  0.810692, -0.077519,  1.171543,  0.779210,
                            1.050909, -0.011629,  1.878902,  0.547481,  1.127397, -0.020820,  1.910036,  1.100910,
                            1.384801,  0.355223,  1.713381,  1.151539]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 0,0,0, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 0,0, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 20
parallel_extension = {
        'qpos' : np.array([ 1.171616, -0.803729,  0.095587,  1.095293,  0.844809, -0.255625,  0.745382,  0.146139,
                            0.681394, -0.146705,  1.105790,  0.082239,  0.693873, -0.004155,  0.937141,  0.140384,
                            0.439880,  0.074434,  1.066224,  1.135471]),
        'specific_finger_names' : ['thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 21
tip_pinch = {
        'qpos' : np.array([ 0.867282, -0.548662,  0.185202,  1.040301,  0.757811, -0.067788,  1.012033,  0.489813,
                            0.376675,  0.057270,  0.074835,  0.312406,  0.415318,  0.106973,  0.092169,  0.251553,
                            0.163468,  0.187374,  0.648767,  0.955604]),
        'specific_finger_names' : ['thumb', 'index_finger'],
        'specific_sensor_names' : ['thumb', 'index_finger'],
        'specific_ctrl_names' : ['thj', 'ffj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 22
lateral_tripod = {
        'qpos' : np.array([ 0.874683, -0.683544,  0.410081,  1.182910,  0.777082, -0.097760,  1.190328,  1.120354,
                            1.048579, -0.023832,  1.720396,  0.470586,  1.259987,  0.029877,  1.889941,  1.054508,
                            1.385526,  0.336007,  1.797259,  1.196873]),
        'specific_finger_names' : ['thumb', 'index_finger', 'middle_finger'],
        'specific_sensor_names' : ['thumb', 'index_finger', 'middle_finger'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 23
shpere_4_finger = {
        'qpos' : np.array([ 1.035201, -0.747796,  0.063310,  0.983970,  0.938222, -0.132008,  0.979765,  0.189894,
                            0.809085, -0.097144,  1.011416,  0.511696,  0.807081,  0.062377,  1.103313,  0.371296,
                            1.407681,  0.361346,  1.745227,  1.108245]),
        'specific_finger_names' : ['thumb', 'index_finger', 'middle_finger', 'ring_finger'],
        'specific_sensor_names' : ['thumb', 'index_finger', 'middle_finger', 'ring_finger'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 24
quadpod = {
        'qpos' : np.array([ 1.090271, -0.737424,  0.018480,  1.009206,  0.934162, -0.117260,  1.019125,  0.172249,
                            0.884887, -0.041404,  1.107660,  0.097961,  0.914534,  0.056294,  1.035479,  0.195255,
                            1.407429,  0.361461,  1.744526,  1.108545]),
        'specific_finger_names' : ['thumb', 'index_finger', 'middle_finger', 'ring_finger'],
        'specific_sensor_names' : ['thumb', 'index_finger', 'middle_finger', 'ring_finger'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 25
shpere_3_finger = {
        'qpos' : np.array([ 0.938466, -0.509569,  0.199924,  1.133806,  0.712523, -0.179361,  1.094272,  0.583308,
                            0.812152,  0.079947,  1.227672,  0.426088,  1.328436,  0.008652,  1.845361,  1.300801,
                            1.407429,  0.361461,  1.744526,  1.108545]),
        'specific_finger_names' : ['thumb', 'index_finger', 'middle_finger'],
        'specific_sensor_names' : ['thumb', 'index_finger', 'middle_finger'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 26
stick = {
        'qpos' : np.array([ 0.534948, -0.124643,  0.083952,  1.168755,  0.236442, -0.058224,  1.545915,  1.205540,
                            0.676646, -0.020484,  1.386711,  1.376787,  0.683853,  0.019868,  1.502071,  1.279905,
                            1.112163,  0.236247,  1.471667,  1.382342]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 27
palmar = {
        'qpos' : np.array([-0.595045,  0.249716,  0.457593, -0.324820,  1.150392, -0.070540,  2.027902,  0.273266,
                            1.205187, -0.013127,  1.967404,  0.294794,  1.083924, -0.051713,  1.992260,  0.449287,
                            1.354725,  0.344277,  1.745457,  1.200822]),
        'specific_finger_names' : ['thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 28
ring = {
        'qpos' : np.array([ 0.873747, -0.565576,  0.193952,  1.009844,  0.839980, -0.065424,  1.034264,  0.164059,
                            0.276069, -0.023631,  0.045289,  0.389727,  0.043530,  0.013154,  0.264566,  0.266046,
                            0.108623,  0.210671,  0.371885,  0.195182]),
        'specific_finger_names' : ['thumb', 'index_finger'],
        'specific_sensor_names' : ['thumb', 'index_finger'],
        'specific_ctrl_names' : ['thj', 'ffj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 29
ventral = {
        'qpos' : np.array([ 0.657862, -0.073944,  0.124456,  1.202992,  0.544430, -0.060261,  0.222272,  0.265519,
                            0.702365, -0.037755,  1.025203,  1.505604,  0.572023, -0.067511,  1.471876,  1.311518,
                            1.093854,  0.172237,  1.569676,  1.384218]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky'],
        'specific_ctrl_names' : ['thj', 'ffj', 'mfj', 'rfj', 'lfj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 30
inferior_pincer = {
        'qpos' : np.array([ 0.944950, -0.680818,  0.005680,  0.883813,  1.224947, -0.034996,  0.174830,  0.401094,
                            0.301085,  0.064259, -0.019675,  0.256272,  0.259657,  0.083598, -0.016093,  0.215817,
                            0.011306,  0.433002,  0.099249,  0.239135]),
        'specific_finger_names' : ['wrist', 'thumb', 'index_finger'],
        'specific_sensor_names' : ['wrist', 'thumb', 'index_finger'],
        'specific_ctrl_names' : ['thj', 'ffj'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

taxonomy_name_list = [
        'large_diameter',
        'small_diameter',
        'medium_diameter',
        'adducted_thumb',
        'light_tool',
        'prismatic_4_finger',
        'prismatic_3_finger',
        'prismatic_2_finger',
        'palmar_pinch',
        'power_disk',
        'power_sphere',
        'precision_disk',
        'precision_sphere',
        'tripod',
        'fixed_hook',
        'lateral',
        'index_finger_extension',
        'extension_type',
        'writing_tripod',
        'parallel_extension',
        'tip_pinch',
        'lateral_tripod',
        'shpere_4_finger',
        'quadpod',
        'shpere_3_finger',
        'stick',
        'palmar',
        'ring',
        'ventral',
        'inferior_pincer',
]
