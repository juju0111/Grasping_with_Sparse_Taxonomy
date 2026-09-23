"""Shadow Hand grasp-taxonomy annotation (30 Feix taxonomies, robotis_sh5 layout).

qpos: 22 finger joints in XML order (FFJ4 FFJ3 FFJ2 FFJ1 | MFJ4..1 | RFJ4..1 | LFJ5 LFJ4..1 |
THJ5 THJ4 THJ3 THJ2 THJ1); the passive *J1 joints are filled by the coupling equality.
Poses were transferred from the tesollo taxonomy with the learned tesollo->shadow IK
(ri_motion_v5 geoRT_refector: geort/taxonomy_data.py, tools/gen_taxonomy_transfer.py,
2026-08-31) and are therefore APPROXIMATE (mean normalised flexion error 0.174; worst
``palmar`` 0.27). Re-label any pose with notebook/hand/01_hand_setup/05~08_fk_taxonomy*.ipynb.

Layout conventions
  * ``specific_finger_names`` / ``specific_sensor_names``: lowercase substrings of body /
    sensor-body names (``shadow_rh_palm``, ``shadow_rh_ff`` ...) — the ``shadow_rh_`` prefix is
    what lh_info.NAME_SUBS rewrites to ``shadow_lh_`` for the mirrored hand.
  * ``specific_ctrl_names``: substrings of actuator names (``rh_A_FFJ3`` -> 'rh_a_ff').
  * ``hand_face_dir_*`` (16 = palm + 3 links x 5 fingers) follow the TOUCH-SENSOR order of the
    XML: palm, ff, mf, rf, lf, **th (thumb LAST)**. ``hand_af_dir_*`` (11 = palm + 2 per finger)
    follow ``rh_af_parts`` (same finger order).
  * Pad normal = site **-Y** for the palm and the finger links; thumb proximal/middle = **-X**,
    thumb distal = -Y. ``lateral`` / ``writing_tripod`` put the index / middle contact face on
    the radial side (**+X**), ``palmar`` turns the thumb distal pad to +Z (palm normal) —
    axes chosen by matching the robotis_sh5 annotation in a canonical hand frame.
"""
import numpy as np

# 1
large_diameter = {
        'qpos' : np.array([-0.255029,  0.629642,  0.707675,  0.707675, -0.127130,  0.529671,  0.644965,  0.644965,
                            0.041180,  0.558945,  0.618015,  0.618015,  0.378728, -0.270863,  0.352016,  0.748054,
                            0.748054,  0.487966,  1.207557, -0.010898,  0.269207,  0.210375]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 2
small_diameter = {
        'qpos' : np.array([-0.252996,  0.562526,  1.045526,  1.045526, -0.003047,  0.542633,  1.142097,  1.142097,
                           -0.017501,  0.599478,  1.111903,  1.111903,  0.478630, -0.346579,  0.510522,  1.280842,
                            1.280842,  0.402487,  1.147305, -0.139468,  0.413658,  0.441250]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 3
medium_diameter = {
        'qpos' : np.array([-0.242081,  0.609815,  0.959888,  0.959888, -0.159003,  0.601504,  1.007870,  1.007870,
                            0.082345,  0.579754,  1.031428,  1.031428,  0.319038, -0.302137,  0.486435,  1.090754,
                            1.090754,  0.491566,  1.188429, -0.110030,  0.293393,  0.309707]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 4
adducted_thumb = {
        'qpos' : np.array([-0.051194,  0.521158,  1.154432,  1.154432, -0.095812,  0.515058,  1.240127,  1.240127,
                            0.114613,  0.507607,  1.109621,  1.109621,  0.210835, -0.138344,  0.489181,  1.187509,
                            1.187509, -0.433963,  0.732023, -0.024998,  0.176353,  0.767386]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 5
light_tool = {
        'qpos' : np.array([-0.079426,  0.629420,  1.255336,  1.255336, -0.050416,  0.706787,  1.328964,  1.328964,
                            0.026699,  0.697252,  1.304672,  1.304672,  0.457028, -0.336611,  0.527640,  1.271242,
                            1.271242, -0.470115,  0.343645, -0.028444,  0.403927,  1.089004]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 6
prismatic_4_finger = {
        'qpos' : np.array([-0.250265,  0.656523,  0.849656,  0.849656, -0.163546,  0.686648,  0.822920,  0.822920,
                            0.072157,  0.677090,  0.688852,  0.688852,  0.212692, -0.303911,  0.560006,  0.889458,
                            0.889458,  0.510630,  1.204681,  0.008366,  0.270395,  0.222900]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : True,
}

# 7
prismatic_3_finger = {
        'qpos' : np.array([-0.240982,  0.751347,  0.732081,  0.732081, -0.095079,  0.708715,  0.685622,  0.685622,
                            0.095414,  0.717393,  0.781349,  0.781349,  0.213484, -0.348856,  1.342174,  1.506997,
                            1.506997,  0.514702,  1.206612, -0.010030,  0.251766,  0.219171]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : True,
}

# 8
prismatic_2_finger = {
        'qpos' : np.array([-0.246395,  0.758229,  0.576449,  0.576449, -0.175963,  0.750656,  0.744206,  0.744206,
                           -0.065904,  0.651319,  1.442333,  1.442333,  0.213484, -0.348856,  1.342174,  1.506997,
                            1.506997,  0.324976,  1.059866, -0.125757,  0.323507,  0.275325]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : True,
}

# 9
palmar_pinch = {
        'qpos' : np.array([-0.129446,  0.835547,  0.607169,  0.607169, -0.063511,  0.454304,  0.279126,  0.279126,
                            0.011371,  0.466957,  0.309078,  0.309078,  0.211233, -0.216104,  0.004642,  0.250304,
                            0.250304,  0.263387,  1.097611, -0.136959,  0.336063,  0.250715]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : True,
}

# 10
power_disk = {
        'qpos' : np.array([-0.099877,  0.270498,  1.283820,  1.283820, -0.025050,  0.369958,  1.260168,  1.260168,
                           -0.022234,  0.373014,  1.323006,  1.323006,  0.502544, -0.348833,  0.595230,  1.327018,
                            1.327018,  0.530510,  1.216709,  0.169276,  0.453950,  0.426031]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 11
power_sphere = {
        'qpos' : np.array([-0.183870,  0.734686,  0.938672,  0.938672, -0.124542,  0.700479,  0.977196,  0.977196,
                            0.008332,  0.663668,  0.990306,  0.990306,  0.459602, -0.348797,  0.543688,  1.233833,
                            1.233833,  0.473616,  1.215229,  0.096240,  0.414316,  0.234164]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 12
precision_disk = {
        'qpos' : np.array([-0.228571,  0.503206,  0.917774,  0.917774,  0.001714,  0.399572,  1.011510,  1.011510,
                           -0.100111,  0.536698,  1.042991,  1.042991,  0.451117, -0.348011,  0.386581,  1.202333,
                            1.202333,  0.139163,  0.908034, -0.091617,  0.458613,  0.512040]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : True,
}

# 13
precision_sphere = {
        'qpos' : np.array([-0.263296,  0.678709,  0.837515,  0.837515, -0.005484,  0.693342,  0.662581,  0.662581,
                           -0.062350,  0.618661,  0.957412,  0.957412,  0.522960, -0.348763,  0.438371,  0.996579,
                            0.996579,  0.386579,  1.196301, -0.006724,  0.362890,  0.205184]),
        'specific_finger_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : True,
}

# 14
tripod = {
        'qpos' : np.array([-0.229701,  0.681021,  0.847975,  0.847975, -0.159728,  0.774573,  0.798764,  0.798764,
                           -0.124354,  0.756447,  1.478586,  1.478586,  0.266126, -0.348919,  1.290372,  1.512258,
                            1.512258,  0.363996,  1.129685, -0.112060,  0.323038,  0.221492]),
        'specific_finger_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf'],
        'specific_sensor_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : True,
}

# 15
fixed_hook = {
        'qpos' : np.array([-0.051194,  0.521158,  1.154432,  1.154432, -0.095812,  0.515058,  1.240127,  1.240127,
                            0.114613,  0.507607,  1.109621,  1.109621,  0.210835, -0.138344,  0.489181,  1.187509,
                            1.187509, -0.433963,  0.732023, -0.024998,  0.176353,  0.767386]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 16
lateral = {
        'qpos' : np.array([-0.142760,  0.578682,  1.322216,  1.322216, -0.105515,  0.716023,  1.476073,  1.476073,
                           -0.018973,  0.738988,  1.471846,  1.471846,  0.263453, -0.348142,  1.264969,  1.445148,
                            1.445148, -0.182220,  0.621547, -0.040375,  0.482535,  0.891364]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff'],
        'hand_face_dir_idx_in_mat' : [1, 0,0,0, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, 1,1,1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 0,0, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 17
index_finger_extension = {
        'qpos' : np.array([-0.306333,  0.235389,  0.307508,  0.307508, -0.133969,  0.512888,  1.240333,  1.240333,
                            0.043155,  0.668912,  1.254074,  1.254074,  0.459690, -0.338035,  0.536094,  1.274347,
                            1.274347, -0.045561,  0.551192, -0.002284,  0.443894,  0.812881]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 18
extension_type = {
        'qpos' : np.array([-0.301590,  0.092192,  0.371755,  0.371755, -0.139563,  0.140871,  0.373539,  0.373539,
                           -0.097269,  0.045876,  0.447814,  0.447814,  0.033733, -0.320649, -0.092815,  0.533695,
                            0.533695,  0.249918,  0.959180, -0.158256,  0.519734,  0.715343]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 19
writing_tripod = {
        'qpos' : np.array([-0.079154,  0.732052,  0.925805,  0.925805, -0.051722,  0.971291,  1.256760,  1.256760,
                            0.012000,  1.092820,  1.452732,  1.452732,  0.297025, -0.348935,  1.243930,  1.498211,
                            1.498211,  0.006000,  0.675542, -0.006275,  0.463720,  0.677633]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 0,0,0, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, 1,1,1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 0,0, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, 1,1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : True,
}

# 20
parallel_extension = {
        'qpos' : np.array([-0.227991,  0.757447,  0.558070,  0.558070, -0.173239,  0.698619,  0.677694,  0.677694,
                            0.061820,  0.684802,  0.596732,  0.596732,  0.165315, -0.206025,  0.535462,  0.931846,
                            0.931846,  0.530070,  1.217787,  0.113269,  0.280949,  0.191169]),
        'specific_finger_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 21
tip_pinch = {
        'qpos' : np.array([-0.092147,  0.720395,  0.803547,  0.803547,  0.054656,  0.303650,  0.336668,  0.336668,
                           -0.120167,  0.387535,  0.331316,  0.331316,  0.054426, -0.224270,  0.315994,  0.836595,
                            0.836595,  0.293730,  1.099372, -0.173134,  0.373052,  0.282122]),
        'specific_finger_names' : ['shadow_rh_th', 'shadow_rh_ff'],
        'specific_sensor_names' : ['shadow_rh_th', 'shadow_rh_ff'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : True,
}

# 22
lateral_tripod = {
        'qpos' : np.array([-0.076591,  0.683493,  0.993432,  0.993432, -0.089727,  0.960691,  1.123669,  1.123669,
                            0.000450,  1.144026,  1.451649,  1.451649,  0.178469, -0.348797,  1.367178,  1.505920,
                            1.505920,  0.237284,  1.140969, -0.106520,  0.531926,  0.376747]),
        'specific_finger_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf'],
        'specific_sensor_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 23
shpere_4_finger = {
        'qpos' : np.array([-0.187805,  0.838514,  0.733517,  0.733517, -0.143453,  0.716893,  0.813122,  0.813122,
                            0.023042,  0.707273,  0.816177,  0.816177,  0.268067, -0.348921,  1.287789,  1.512990,
                            1.512990,  0.334674,  1.197135, -0.063415,  0.353650,  0.190848]),
        'specific_finger_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf'],
        'specific_sensor_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 24
quadpod = {
        'qpos' : np.array([-0.165773,  0.843361,  0.742179,  0.742179, -0.083097,  0.786291,  0.762308,  0.762308,
                            0.030405,  0.760398,  0.772319,  0.772319,  0.268446, -0.348922,  1.287011,  1.512653,
                            1.512653,  0.420612,  1.204225, -0.013466,  0.291309,  0.208020]),
        'specific_finger_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf'],
        'specific_sensor_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : True,
}

# 25
shpere_3_finger = {
        'qpos' : np.array([-0.236293,  0.689234,  0.863920,  0.863920,  0.052036,  0.743598,  0.874613,  0.874613,
                           -0.038627,  1.105381,  1.477327,  1.477327,  0.268446, -0.348922,  1.287011,  1.512653,
                            1.512653,  0.379478,  1.140145, -0.152309,  0.358127,  0.331444]),
        'specific_finger_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf'],
        'specific_sensor_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 26
stick = {
        'qpos' : np.array([-0.090485,  0.373017,  1.219453,  1.219453, -0.029778,  0.539974,  1.182972,  1.182972,
                            0.055461,  0.580507,  1.173124,  1.173124,  0.481782, -0.348805,  0.702668,  1.323711,
                            1.323711,  0.045196,  0.587539,  0.021611,  0.362705,  0.716735]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 27
palmar = {
        'qpos' : np.array([-0.061480,  1.118613,  1.285550,  1.285550, -0.064001,  1.108042,  1.268931,  1.268931,
                            0.005336,  1.131269,  1.318820,  1.318820,  0.198561, -0.348916,  1.315470,  1.461948,
                            1.461948, -1.006099,  0.386618, -0.163520, -0.117845,  0.836914]),
        'specific_finger_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,2],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 28
ring = {
        'qpos' : np.array([-0.085563,  0.786938,  0.739162,  0.739162, -0.020287,  0.267136,  0.338767,  0.338767,
                           -0.042212,  0.178469,  0.368395,  0.368395,  0.161135, -0.272343, -0.084509,  0.275646,
                            0.275646,  0.283061,  1.102844, -0.165734,  0.387096,  0.271928]),
        'specific_finger_names' : ['shadow_rh_th', 'shadow_rh_ff'],
        'specific_sensor_names' : ['shadow_rh_th', 'shadow_rh_ff'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 29
ventral = {
        'qpos' : np.array([-0.111184,  0.494048,  0.343557,  0.343557, -0.107187,  0.519148,  1.077062,  1.077062,
                            0.104010,  0.500778,  1.166691,  1.166691,  0.419659, -0.347030,  0.880936,  1.340053,
                            1.340053,  0.221793,  0.759174, -0.054036,  0.398677,  0.680610]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff', 'shadow_rh_mf', 'shadow_rh_rf', 'shadow_rh_lf'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff', 'rh_a_mf', 'rh_a_rf', 'rh_a_lf'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
        'do_finger_tip' : False,
}

# 30
inferior_pincer = {
        'qpos' : np.array([ 0.014864,  0.946565,  0.484614,  0.484614,  0.069654,  0.294014,  0.272830,  0.272830,
                           -0.115649,  0.239840,  0.313071,  0.313071,  0.131000, -0.303360, -0.201352,  0.172072,
                            0.172072,  0.247008,  1.118765, -0.126356,  0.327178,  0.248736]),
        'specific_finger_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff'],
        'specific_sensor_names' : ['shadow_rh_palm', 'shadow_rh_th', 'shadow_rh_ff'],
        'specific_ctrl_names' : ['rh_a_th', 'rh_a_ff'],
        'hand_face_dir_idx_in_mat' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,1],
        'hand_face_dir_sign' : [-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1],
        'hand_af_dir_idx_in_mat' : [1, 1,1, 1,1, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [-1, -1,-1, -1,-1, -1,-1, -1,-1, -1,-1],
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
