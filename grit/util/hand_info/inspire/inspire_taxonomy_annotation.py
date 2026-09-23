# import jax.numpy as np 
import numpy as np 


######################
# 1 
large_diameter = {
        'qpos' : np.array([1.1641  ,
        0.139692,  0.174093,  0.183506,  0.693107,  0.639221,  0.646745,  0.596464,  0.679199,
        0.626394,  0.776558,  0.716184]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : False,
}

# 2 
small_diameter = {
        'qpos' : np.array([0.54  , 0.36  , 0.4487, 0.4729, 1.14  , 1.0514, 1.1   , 1.0145, 1.14  , 1.0514, 1.2   ,
       1.1067]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : False,
}

# 3
medium_diameter = {
        'qpos' : np.array([0.7747, 0.2967, 0.3695, 0.3893, 0.9988, 0.9209, 0.98  , 0.9038, 1.06  , 0.9776, 1.02  ,
       0.9407]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : False,
}


# 5
light_tool = {
        'qpos' : np.array([0.      ,
        0.247147,  0.30801 ,  0.324665,  1.200767,  1.107413,  1.200767,  1.107413,  1.31899 ,
        1.216444,  1.31899 ,  1.216444]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : False,
}

# 6
prismatic_4_finger = {
        'qpos' : np.array([1.14  , 0.12  , 0.1496, 0.1576, 0.9   , 0.83  , 0.9   , 0.83  , 0.96  , 0.8854, 0.96  ,
       0.8854]),
        'specific_finger_names' : ['right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : True,
}

# 7 
prismatic_3_finger = {
        'qpos' : np.array([1.14  , 0.12  , 0.1496, 0.1576, 0.9   , 0.83  , 0.9   , 0.83  , 1.02  , 0.9407, 1.56  ,
       1.4387]),
        'specific_finger_names' : ['right_thumb', 'right_index', 'right_middle', 'right_ring'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : True,
}

# 8
prismatic_2_finger = {
        'qpos' : np.array([1.14  , 0.24  , 0.1496, 0.1576, 0.9   , 0.83  , 0.96  , 0.8854, 1.56  , 1.4387, 1.56  ,
       1.4387]),
        'specific_finger_names' : ['right_thumb', 'right_index', 'right_middle'],
        'specific_sensor_names' : ['thumb', 'index', 'middle'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : True,
}

# 9
palmar_pinch = {
        'qpos' : np.array([1.139678,
        0.128946,  0.160701,  0.169391,  0.732515,  0.675564,  0.315259,  0.290749,  0.419573,
        0.386953,  0.359303,  0.331369]),
        'specific_finger_names' : ['right_thumb', 'right_index'],
        'specific_sensor_names' : ['thumb', 'index'], 
        'specific_ctrl_names' : ['thumb', 'index'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : True,
}

# 10
power_disk = {
        'qpos' : np.array([1.139678,
        0.263862,  0.328842,  0.346623,  0.718606,  0.662737,  0.695425,  0.641359,  0.709334,
        0.654186,  0.741787,  0.684116]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'],
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : False,
}

# 11 
power_sphere = {
        'qpos' : np.array([1.1641  ,
        0.169541,  0.211292,  0.222717,  0.850737,  0.784595,  0.829874,  0.765355,  0.955051,
        0.880799,  1.189177,  1.096723]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : False,
}

# 12
precision_disk = {
        'qpos' : np.array([0.601238,
        0.319978,  0.398776,  0.42034 ,  0.99214 ,  0.915005,  1.024593,  0.944935,  1.077909,
        0.994106,  1.165996,  1.075345]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : True,
}

# 13 
precision_sphere = {
        'qpos' : np.array([ 1.139678,
        0.119395,  0.148797,  0.156843,  0.843783,  0.778182,  0.809011,  0.746114,  0.982868,
        0.906454,  1.182223,  1.09031]),
        'specific_finger_names' : ['right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : True,
}

# 14  # defualt
tripod = {
        'qpos' : np.array([1.139678,
        0.201777,  0.251467,  0.265065,  0.748741,  0.690529,  0.869281,  0.801698,  1.4998  ,
        1.383197,  1.56007 ,  1.438781]),
        'specific_finger_names' : ['thumb', 'index', 'middle'] ,
        'specific_sensor_names' : ['thumb', 'index', 'middle'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : True,

}

# 15
fixed_hook = {
        'qpos' : np.array([0.    , 0.    , 0.    , 0.    , 1.2   , 1.1067, 1.14  , 1.0514, 1.14  , 1.0514, 1.2   ,
       1.1067]),
        'specific_finger_names' : ['palm', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0, ],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1,-1,-1,], 
        'do_finger_tip' : False,
}

# 16
lateral = {
        'qpos' : np.array([0.36  , 0.3   , 0.3739, 0.3941, 1.32  , 1.2174, 1.44  , 1.328 , 1.52  , 1.4018, 1.56  ,
       1.4387]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index'] ,
        'specific_sensor_names' : ['thumb', 'index'], 
        'specific_ctrl_names' : ['thumb', 'index'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 2,2, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 2,2, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : False,
}

# 17
index_finger_extension = {
        'qpos' : np.array([0.66  , 0.3   , 0.3739, 0.3941, 0.66  , 0.6087, 1.1   , 1.0145, 1.26  , 1.162 , 1.32  ,
       1.2174]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : False,
}

# 18
extension_type = {
        'qpos' : np.array([1.14  , 0.24  , 0.2991, 0.3153, 0.42  , 0.3873, 0.44  , 0.4058, 0.42  , 0.3873, 0.36  ,
       0.332 ]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1, -1,-1], 
        'do_finger_tip' : False,
}

# 19 (scissors) is skipped

# 20 # 21 
writing_tripod = {
        'qpos' : np.array([0.36  , 0.3   , 0.3739, 0.3941, 0.96  , 0.8854, 1.32  , 1.2174, 1.44  , 1.328 , 1.56  ,
       1.4387]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle'],
        'specific_sensor_names' : ['thumb', 'index', 'middle'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 2,2, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, 1,1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 2,2, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, 1,1, -1,-1, -1,-1], 
        'do_finger_tip' : True,
}

# 22 # X 
parallel_extension = {
        'qpos' : np.array([1.14  , 0.18  , 0.2243, 0.2365, 0.96  , 0.8854, 0.98  , 0.9038, 1.08  , 0.996 , 1.08  ,
       0.996 ]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0, ],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1,-1,-1,], 
        'do_finger_tip' : False,
}

# # 23  # X
# adduction_grip = {
#         'qpos' : np.array([-0.698 ,  0.3   ,  0.86  ,  0.624 ,  0.0164,  1.1455,  1.02  ,  0.5944,  0.0364,  1.2055,
#         0.96  ,  0.5478,  0.0364,  1.2655,  0.92  ,  0.5175,  0.0164,  1.2255,  1.08  ,  0.6425]),
#         'specific_finger_names' : ['palm', 'right_index', 'right_middle'],
#         'specific_sensor_names' : ['index', 'middle'], 
#         'specific_ctrl_names' : ['index', 'middle'], 
#         'hand_face_dir_idx_in_mat' : [0, 1,1,1, 2,2, 2,2, 0,0, 0,0],
#         'hand_face_dir_sign' : [1, 1,1,1, -1,-1, 1,1, -1,-1, -1,-1],  
#         'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0, ],
#         'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1,-1,-1,], 
#         'do_finger_tip' : False,
# }

# 24 
tip_pinch= {
        'qpos' : np.array([1.139678,
        0.230432,  0.287178,  0.302707,  0.873918,  0.805974,  0.428846,  0.395504,  0.438118,
        0.404056,  0.405665,  0.374126]),
        'specific_finger_names' : ['right_thumb', 'right_index',],
        'specific_sensor_names' : ['thumb', 'index'], 
        'specific_ctrl_names' : ['thumb', 'index'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0, ],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1,-1,-1,], 
        'do_finger_tip' : True,
}

# 25
lateral_tripod = {
        'qpos' : np.array([0.8012, 0.2914, 0.3572, 0.3733, 1.0169, 0.9383, 1.26  , 1.162 , 1.56  , 1.4387, 1.56  ,
       1.4387]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle'] ,
        'specific_sensor_names' : ['thumb', 'index', 'middle'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0, ],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1,-1,-1,], 
        'do_finger_tip' : False,
}

# 26
shpere_4_finger = {
        'qpos' : np.array([0.737302,
        0.280578,  0.349673,  0.368581,  1.200767,  1.107413,  1.054728,  0.972727,  1.193813,
        1.100999,  1.6     ,  1.58843]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring'] ,
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0, ],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1,-1,-1,], 
        'do_finger_tip' : False,
}

# 27 precision_quadpod
quadpod = {
        'qpos' : np.array([1.14  , 0.06  , 0.0748, 0.0788, 0.82  , 0.7562, 0.9   , 0.83  , 1.08  , 0.996 , 1.56  ,
       1.4387]),
        'specific_finger_names' : ['right_thumb', 'right_index', 'right_middle', 'right_ring', ],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', ], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', ], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0, ],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1,-1,-1,], 
        'do_finger_tip' : True,
}

# 28 precision_quadpod
shpere_3_finger = {
        'qpos' : np.array([1.14  , 0.06  , 0.0748, 0.0788, 0.82  , 0.7562, 0.9   , 0.83  , 1.4999, 1.3833, 1.56  ,
       1.4387]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle',],
        'specific_sensor_names' : ['thumb', 'index', 'middle',], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle',], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0, ],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1,-1,-1,], 
        'do_finger_tip' : False,
}

# 29
stick = {
        'qpos' : np.array([0.540766,
        0.316396,  0.394312,  0.415634,  0.899417,  0.829491,  1.080227,  0.996244,  1.478938,
        1.363956,  1.56007 ,  1.438781]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0, ],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1,-1,-1,], 
        'do_finger_tip' : False,
}

# 30
palmar = {
        'qpos' : np.array([0.      ,
        0.247147,  0.30801 ,  0.324665,  1.274946,  1.175824,  1.295809,  1.195065,  1.328262,
        1.224995,  1.286537,  1.186513]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0,  0,0, ],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1,-1,-1,], 
        'do_finger_tip' : False,
}

# 31 
ring = {
        'qpos' : np.array([ 1.139678,
        0.183868,  0.229148,  0.241538,  0.848419,  0.782458,  0.299033,  0.275784,  0.299033,
        0.275784,  0.      ,  0.      ]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', ],
        'specific_sensor_names' : ['thumb', 'index'], 
        'specific_ctrl_names' : ['thumb', 'index'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0, ],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1,-1,-1,], 
        'do_finger_tip' : False,
}

# 32
ventral = {
        'qpos' : np.array([0.486108,
        0.336693,  0.419608,  0.442298,  0.799739,  0.737562,  1.140497,  1.051828,  1.237857,
        1.141618,  1.395487,  1.28699]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'],
        'specific_sensor_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'specific_ctrl_names' : ['thumb', 'index', 'middle', 'ring', 'little'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0, ],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1,-1,-1,], 
        'do_finger_tip' : False,
}

# 33 
inferior_pincer = {
        'qpos' : np.array([1.1641  ,
        0.241177,  0.30057 ,  0.316823,  0.621246,  0.572947,  0.      ,  0.      ,  0.      ,
        0.      ,  0.      ,  0.   ]),
        'specific_finger_names' : ['palm', 'right_thumb', 'right_index'],
        'specific_sensor_names' : ['thumb', 'index'], 
        'specific_ctrl_names' : ['thumb', 'index'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0, 0,0, 0,0, 0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1, -1,-1, -1,-1, -1,-1],  
        'hand_af_dir_idx_in_mat' : [0, 1,1, 0,0, 0,0, 0,0,  0,0, ],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, -1,-1, -1,-1,-1,-1,], 
        'do_finger_tip' : False,
}

# taxonomy_name_list = [
#         'large_diameter', # 1
#         'small_diameter', # 2
#         'medium_diameter', # 3
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
#         'writing_tripod', # 20,21 
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
        # 'fixed_hook', # 15
        'lateral', # 16
        'index_finger_extension', # 17
        'extension_type', # 18
        'writing_tripod', # 20,21 
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
        4 : [5], 
        5 : [6],
        6 : [7],
        7 : [8],
        8 : [9],
        9 : [10],
        10 : [11],
        11 : [12],
}