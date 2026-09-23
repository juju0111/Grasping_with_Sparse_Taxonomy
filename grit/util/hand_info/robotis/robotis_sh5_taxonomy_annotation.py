# import jax.numpy as jp 
import numpy as np 

######################
# 0 
none = {
        # Dummy qpos
        'qpos' : np.array([-0.00314, -0.00314,  1.00066,  1.     , -0.00157, -0.00157, -0.00157,  0.25578, -1.63828,
        0.52883,  0.28874, -0.29843,  0.77998,  0.4896 ,  0.70929, -0.13121,  0.59952,  0.64966,
        0.58846, -0.01037,  0.65968,  0.52883,  0.64966,  0.24229,  0.5113 ,  0.68732,  1.1016 ]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        # TODO
        # 'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        # 'hand_face_dir_sign' : [1, -1,-1,-1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        # 'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        # 'hand_af_dir_sign' : [1, -1,-1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 1 
large_diameter = {
        'qpos' : np.array([ 0.25578, -1.63828,
        0.52883,  0.28874, -0.29843,  0.77998,  0.4896 ,  0.70929, -0.13121,  0.59952,  0.64966,
        0.58846, -0.01037,  0.65968,  0.52883,  0.64966,  0.24229,  0.5113 ,  0.68732,  1.1016 ]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 2 
small_diameter = {
        'qpos' : np.array([0.18046, -1.63828,
        0.36406,  0.63554, -0.29477,  1.12486,  0.90074,  0.84425, -0.1007 ,  1.07674,  0.98705,
        0.83483,  0.15929,  1.09077,  0.82855,  0.89917,  0.24229,  0.99653,  1.09846,  1.04982]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 3
medium_diameter = {
        'qpos' : np.array([0.22126, -1.49077,
        0.19615,  0.76422, -0.29477,  0.95844,  0.79403,  0.84425, -0.1007 ,  0.91833,  0.82385,
        0.83483,  0.15929,  0.99453,  0.73597,  0.89917,  0.24229,  0.84214,  1.02157,  0.84111]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 4
adducted_thumb = {
        'qpos' : np.array([ 0.06434, -0.29188,
        0.16948,  0.16948, -0.07873,  1.07072,  0.96978,  1.03412, -0.11046,  1.18702,  1.08905,
        1.11258, -0.0714 ,  1.02059,  1.02942,  1.11258, -0.0177 ,  0.6697 ,  1.23028,  1.2036 ]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 5
light_tool = {
        'qpos' : np.array([ 0.06434, -0.29188,
        0.49745,  0.94311, -0.07873,  1.07072,  0.96978,  1.03412,  0.01037,  1.18702,  1.08905,
        1.11258,  0.0299 ,  1.02059,  1.02942,  1.11258,  0.10558,  0.6697 ,  1.23028,  1.2036 ]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 6
prismatic_4_finger = {
        'qpos' : np.array([ 0.38132, -1.67594,
        0.64966,  0.29188, -0.28989,  0.77998,  0.85052,  0.22283, -0.0714 ,  0.73587,  0.90858,
        0.14437,  0.13243,  0.83612,  0.98234,  0.23538,  0.27524,  0.85618,  0.816  ,  0.23538]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 7 
prismatic_3_finger = {
        'qpos' : np.array([0.15535, -1.67594,
        0.64966,  0.29188, -0.28989,  0.77998,  0.85052,  0.22283, -0.0714 ,  0.73587,  0.90858,
        0.14437,  0.13243,  0.83612,  0.98234,  0.23538,  0.27524,  1.50583,  1.5708 ,  0.88975]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 8
prismatic_2_finger = {
        'qpos' : np.array([ 0.00157, -1.67594,
        0.64966,  0.23225, -0.28989,  0.77998,  0.85052,  0.22283, -0.0714 ,  0.8281 ,  0.90858,
        0.14437,  0.13243,  1.60608,  1.5708 ,  0.65437,  0.27524,  1.50583,  1.5708 ,  0.88975]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 9
palmar_pinch = {
        'qpos' : np.array([0.0016, -1.4908,  0.3107,
        0.408 , -0.1446,  1.1229,  0.5163,  0.    , -0.014 ,  0.6597,  0.1287,  0.3295,  0.0482,
        0.5995,  0.3295,  0.4488,  0.1227,  0.1283,  0.5445,  0.5288]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 10
power_disk = {
        'qpos' : np.array([ 1.1251, -1.2836,  0.1962,
        1.4296, -0.1361,  0.    ,  1.3495,  1.45  ,  0.0665,  0.1203,  1.3888,  1.3699,  0.1507,
        0.0802,  1.5692,  1.3699,  0.2276,  0.2085,  1.2899,  1.5692]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 11 
power_sphere = {
        'qpos' : np.array([0.6544, -1.5441,  0.4708,
        0.8505, -0.1422,  0.8522,  1.0216,  0.7062,  0.0665,  0.7018,  1.1644,  0.8772,  0.1507,
        0.7519,  1.0216,  0.7862,  0.2276,  0.9203,  1.0467,  0.681]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 12
precision_disk = {
        'qpos' : np.array([0.5226, -0.8129,  0.328 ,
        0.6026, -0.1385,  0.2406,  0.9086,  0.7093, -0.0104,  0.1203,  1.009 ,  0.7093,  0.3656,
        0.4812,  0.9698,  0.8286,  0.5804,  0.8522,  1.1   ,  0.6026]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 13 
precision_sphere = {
        'qpos' : np.array([ 0.8898, -1.3338,  0.3138,
        0.2354, -0.2899,  0.78  ,  0.8505,  0.2228, -0.0714,  0.7359,  0.9086,  0.1444,  0.346 ,
        0.8361,  1.0859,  0.2087,  0.5194,  1.1369,  0.9431,  0.091 ]),
        'specific_finger_names' : ['r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# defualt
#14
tripod = {
        'qpos' : np.array([0.3154, -1.3056,  0.51  ,
        0.2322, -0.2899,  0.78  ,  0.8505,  0.2228, -0.2325,  1.0848,  0.9086,  0.1444,  0.1324,
        1.6061,  1.5708,  0.6544,  0.2752,  1.5058,  1.5708,  0.8898]),
        'specific_finger_names' : ['r_link1', 'r_link2', 'r_link3'] ,
        'specific_sensor_names' : ['r_link1', 'r_link2', 'r_link3'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,

}

# 15
fixed_hook = {
        'qpos' : np.array([ 0.06434, -0.29188,
        0.16948,  0.16948, -0.07873,  1.07072,  0.96978,  1.03412, -0.11046,  1.18702,  1.08905,
        1.11258, -0.0714 ,  1.02059,  1.02942,  1.11258, -0.0177 ,  0.6697 ,  1.23028,  1.2036 ]),
        'specific_finger_names' : ['base', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 16
lateral = {
        'qpos' : np.array([ 0.215 , -0.6026,  0.6669,
        0.5634, -0.1178,  1.1208,  1.2099,  1.5488, -0.0006,  1.5379,  1.5692,  0.838 , -0.0043,
        1.556 ,  1.5692,  0.838 ,  0.0031,  1.6061,  1.4688,  0.8113]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2'] ,
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 1,1,1, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 1,1, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 17
index_finger_extension = {
        'qpos' : np.array([0.4441, -1.3087,  0.1444,
        0.6355, -0.6109,  0.8361,  0.3138,  0.419 , -0.1007,  1.0767,  0.987 ,  0.8348,  0.1593,
        1.0908,  0.8286,  0.8992,  0.2423,  0.9965,  1.0985,  1.0498]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 18
extension_type = {
        'qpos' : np.array([ 0.237 , -1.1801,  0.6685,
        1.1487, -0.2984,  0.    ,  0.4582,  0.8772, -0.1312,  0.0602,  0.51  ,  0.8113,  0.1092,
        0.    ,  0.6026,  0.9557,  0.2447,  0.3208,  0.3405,  0.7595]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 19 (scissors) is skipped

# 20 # 21 
writing_tripod = {
        'qpos' : np.array([ 0.7485, -1.1173,  0.    ,
        1.0074, -0.1935,  0.7018,  0.8772,  0.4582, -0.0201,  1.2371,  1.257 ,  0.4315,  0.1324,
        1.6061,  1.5708,  0.6936,  0.2752,  1.5058,  1.5708,  0.8898]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 1,1,1, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 1,1, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 22 
parallel_extension = {
        'qpos' : np.array([ 0.1302, -1.701 ,  0.3138,
        0.2479, -0.2789,  0.9003,  0.8254,  0.1302, -0.191 ,  0.7519,  0.9823,  0.1836, -0.0104,
        0.8702,  0.7297,  0.2291,  0.0421,  0.8361,  0.7203,  0.1695]),
        'specific_finger_names' : ['r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 23 
adduction_grip = {
        'qpos' : np.array([ 0.156028  , -0.36159   ,  0.0892    ,  0.5292    , -0.418879  ,
                        0.78      ,  0.7092    ,  0.7092    , -0.110865  ,  0.66      ,
                        0.9092    ,  0.7092    , -0.010865  ,  1.08000006,  1.4292    ,
                        1.3492    , -0.0174533 ,  0.141121  ,  1.2092    ,  1.3892    ]),
        'specific_finger_names' : ['r_link2', 'r_link3'],
        'specific_sensor_names' : ['r_link2', 'r_link3'], 
        'specific_ctrl_names' : ['h_joint2', 'h_joint3'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 1,1,1, 1,1,1, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, -1,-1,-1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 1,0, 1,1, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, -1,-1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# Remaining taxonomies
# 24 
tip_pinch= {
        'qpos' : np.array([0.1428, -1.3872,  0.2479,
        0.7862,  0.0409,  0.9003,  0.6293,  0.6497,  0.0043,  0.3509,  0.2354,  0.1695,  0.1349,
        0.3509,  0.4064,  0.3798,  0.0641,  0.399 ,  0.4582,  0.2746]),
        'specific_finger_names' : ['r_link1', 'r_link2',],
        'specific_sensor_names' : ['r_link1', 'r_link2'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 25
lateral_tripod = {
        'qpos' : np.array([ 0.7579, -1.3872,  0.1962,
        1.257 , -0.0995,  0.9604,  1.2428,  0.9431, -0.0311,  1.6061,  1.2821,  0.6151,  0.0238,
        1.6562,  1.4437,  0.7328,  0.0201,  1.5058,  1.5692,  0.7721]),
        'specific_finger_names' : ['r_link1', 'r_link2', 'r_link3'] ,
        'specific_sensor_names' : ['r_link1', 'r_link2', 'r_link3'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,

}

# 26
shpere_4_finger = {
        'qpos' : np.array([0.3154, -1.723 ,  0.3295,
        0.7878, -0.119 ,  0.9003,  0.9086,  0.5367, -0.0507,  0.5995,  0.9886,  0.5696,  0.0092,
        0.9203,  0.9086,  0.6418,  0.2337,  1.2873,  1.5692,  1.1487]),
        'specific_finger_names' : ['r_link1', 'r_link2', 'r_link3', 'r_link4'] ,
        'specific_sensor_names' : ['r_link1', 'r_link2', 'r_link3', 'r_link4'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,

}

# 27 precision_quadpod
quadpod = {
        'qpos' : np.array([ 0.08  , -1.723 ,  0.2354,
        0.6151, -0.119 ,  0.9003,  0.7987,  0.0659, -0.0507,  0.5995,  0.9886,  0.2479,  0.0092,
        0.9203,  0.7328,  0.2479,  0.2337,  1.5058,  1.5692,  1.1126]),
        'specific_finger_names' : ['r_link1', 'r_link2', 'r_link3', 'r_link4', ],
        'specific_sensor_names' : ['r_link1', 'r_link2', 'r_link3', 'r_link4', ], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', ], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 28 precision_quadpod
shpere_3_finger = {
        'qpos' : np.array([0.4441, -1.3244,  0.2354,
        0.6669, -0.119 ,  0.9003,  0.7987,  0.6418,  0.1227,  1.0326,  1.1251,  0.6622,  0.1031,
        1.4898,  1.5049,  0.6748,  0.2337,  1.6061,  1.5692,  0.5492]),
        'specific_finger_names' : ['r_link1', 'r_link2', 'r_link3',],
        'specific_sensor_names' : ['r_link1', 'r_link2', 'r_link3',], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3',], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 29
stick = {
        'qpos' : np.array([0.2087, -0.4959,  0.51  ,
        0.5634, -0.0311,  0.5855,  1.2303,  1.0859, -0.0104,  1.3554,  1.5708,  0.3672,  0.1324,
        1.6061,  1.2177,  0.6544,  0.2752,  1.5058,  1.2428,  0.8898]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 30 
palmar = {
        'qpos' : np.array([-0.1836,  0.    ,  0.8097,
        0.8772, -0.1996,  1.572 ,  1.3354,  0.091 , -0.0861,  1.6061,  1.2303,  0.3531, -0.0067,
        1.3875,  1.4657,  0.2479,  0.0507,  1.4377,  1.348 ,  0.2354]),
        'specific_finger_names' : ['r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 0,0, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 31 
ring = {
        'qpos' : np.array([0.1585, -1.3338,  0.5634,
        0.4974, -0.1385,  0.7018,  0.7721,  0.7062,  0.0104,  0.3008,  0.1836,  0.328 ,  0.0616,
        0.2346,  0.43  ,  0.3295,  0.224 ,  0.2286,  0.4096,  0.288]),
        'specific_finger_names' : ['r_link1', 'r_link2', ],
        'specific_sensor_names' : ['r_link1', 'r_link2'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 32
ventral = {
        'qpos' : np.array([0.1554, -0.8631,  1.0498,
        0.3295, -0.3155,  0.5995,  0.3295,  0.2495, -0.1105,  0.8702,  0.9557,  1.2695, -0.0812,
        1.0707,  1.0341,  1.1126, -0.0177,  1.0046,  1.3495,  1.1126]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2', 'r_link3', 'r_link4', 'r_link5'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2', 'h_joint3', 'h_joint4', 'h_joint5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 33 
inferior_pincer = {
        'qpos' : np.array([0.145938,
       -1.409169,  0.418985,  0.128677, -0.092154,  1.439658,  0.029815,  0.329538,  0.075066,
        0.246627,  0.199292,  0.229108,  0.129992,  0.274698,  0.205569,  0.235385,  0.275241,
        0.292744,  0.279323,  0.23852]),
        'specific_finger_names' : ['base', 'r_link1', 'r_link2'],
        'specific_sensor_names' : ['base', 'r_link1', 'r_link2'], 
        'specific_ctrl_names' : ['h_joint1', 'h_joint2'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# taxonomy_name_list = [
#         'large_diameter', # 1
#         'small_diameter', # 2
#         'medium_diameter', # 3
#         'adducted_thumb', # 4 
#         'light_tool', # 5
#         'prismatic_4_finger', # 6
#         'prismatic_3_finger', # 7
#         'prismatic_2_finger', # 8
#         'palmar_pinch', # 9
#         'power_disk', # 10
#         'power_sphere', # 11
#         'precision_disk', # 12
#         'precision_sphere', # 13
#         'tripod', # 14
#         'fixed_hook', # 15
#         'lateral', # 16
#         'index_finger_extension', # 17
#         'extension_type', # 18
#         'writing_tripod', # 20 21 
#         'parallel_extension', # 22
#         # 'adduction_grip', # 23
#         'tip_pinch', # 24
#         'lateral_tripod', # 25
#         'shpere_4_finger', # 26
#         'quadpod', # 27
#         'shpere_3_finger', # 28
#         'stick', # 29
#         'palmar', # 30
#         'ring', # 31
#         'ventral', # 32
#         'inferior_pincer', # 33
# ]

taxonomy_name_list = [
        'large_diameter', # 1
        'small_diameter', # 2
        'medium_diameter', # 3
        'adducted_thumb', # 4 
        'light_tool', # 5
        'prismatic_4_finger', # 6
        'prismatic_3_finger', # 7
        'prismatic_2_finger', # 8
        'palmar_pinch', # 9
        'power_disk', # 10
        'power_sphere', # 11
        'precision_disk', # 12
        'precision_sphere', # 13
        'tripod', # 14
        'fixed_hook', # 15
        'lateral', # 16
        'index_finger_extension', # 17
        'extension_type', # 18
        'writing_tripod', # 20 21 
        'parallel_extension', # 22
        'tip_pinch', # 24
        'lateral_tripod', # 25
        'shpere_4_finger', # 26
        'quadpod', # 27
        'shpere_3_finger', # 28
        'stick', # 29
        'palmar', # 30
        'ring', # 31
        'ventral', # 32
        'inferior_pincer', # 33
]

finger_name_pair = {
        0 : [0],
        1 : [1],
        2 : [2,3],
        3 : [4],
        4 : [5,6], 
        5 : [7],
        6 : [8,9],
        7 : [10],
        8 : [11,12],
        9 : [13],
        10 : [14,15],
        11 : [16]
}