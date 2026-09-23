# import jax.numpy as jp 
import numpy as np 


#############################
## Power
# 1
large_diameter = {
        'qpos' : np.array([0.073716,
        1.18421 ,  1.05249 ,  0.919444,  0.161988,  1.243748,  0.845566,  0.913914,  0.172318,
        1.303287,  1.08635 ,  0.43285 ,  0.341099,  0.029274,  0.34387 ,  0.24201]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 2
small_diameter = {
        'qpos' : np.array([0.09 , 1.064, 1.146, 
        1.273, 0.01 , 1.064, 1.386, 0.973, 0.09 , 1.064, 1.226, 1.033, 1.383,
        0.215, 0.551, 0.678]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 3
medium_diameter = {
        'qpos' : np.array([0.07 , 1.064, 0.586, 
        1.273, 0.13 , 1.064, 0.846, 0.913, 0.07 , 1.064, 0.906, 0.613, 1.303,
        0.215, 0.551, 0.678]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 4
adducted_thumb = {
        'qpos' : np.array([0.082168,
        1.18421 ,  1.08635 ,  1.27333 ,  0.130999,  1.243748,  0.845566,  0.913914,  0.073716,
        1.303287,  1.08635 ,  0.43285 ,  0.283374,  0.264886,  0.345701,  0.345363]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 5
light_tool = {
        'qpos' : np.array([-0.2296  ,
        0.942448,  1.586727,  1.072426, -0.11034 ,  0.965902,  1.686427,  0.913914, -0.11034 ,
        1.075958,  1.626231,  1.192231,  0.435044,  0.115412,  0.68813 ,  0.877154]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}


### precision ###
# 6 
prismatic_3_finger = {
        'qpos' : np.array([-0.25026 ,
        0.54372 ,  1.526531,  0.672461, -0.11034 ,  0.54372 ,  1.366636,  0.913914, -0.11034 ,
        0.855846,  1.026154,  1.192231,  1.383549,  0.39536 ,  0.710104,  0.937286]),
        'specific_finger_names' : ['r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : True,
}

# 7 # 8
prismatic_2_finger = {
        'qpos' : np.array([ 0.01 ,  0.904,  1.146,  0.693, -0.11 ,  1.004,  0.846,  0.913, -0.29 ,  1.604,  1.506,
        1.613,  1.383,  0.295,  0.451,  0.918]),
        'specific_finger_names' : ['r_ff', 'r_mf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : True,
}

# 9
palmar_pinch = {
        'qpos' : np.array([0.082168,
        1.359217,  0.268063,  0.014454, -0.17044 ,  0.004266,  0.316972,  0.366497, -0.11034 ,
        0.074629,  0.350832,  0.414419,  1.194527,  0.082477,  0.353026, -0.022945]),
        'specific_finger_names' : ['r_ff', 'r_th'], 
        'specific_sensor_names' : ['ff', 'th'], 
        'specific_ctrl_names' : ['ffa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : True,
}

# 10
power_disk = {
        'qpos' : np.array([-0.00986 ,
        0.054783,  1.496434,  1.289918, -0.00986 ,  0.227986,  1.38921 ,  1.393135,  0.02958 ,
        0.274895,  1.503958,  1.328624,  0.839121,  0.898253,  0.942662,  1.27740]),
        'specific_finger_names' : ['r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 11 
power_sphere = {
        'qpos' : np.array([-0.00986 ,
        0.597846,  1.332776,  1.112975, -0.00986 ,  0.455315,  1.291392,  1.160897,  0.02958 ,
        0.651972,  1.364755,  0.871521,  1.316769,  0.58917 ,  0.616714,  1.113923]),
        'specific_finger_names' : ['r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}


# 12 
precision_disk = {
        'qpos' : np.array([-0.00986 ,
        0.839608,  1.065657,  0.655872, -0.00986 ,  0.628517,  1.250007,  0.768305, -0.016434,
        0.958685,  1.127734,  0.864149,  0.691978,  0.84885 ,  0.76687 ,  0.803868]),
        'specific_finger_names' : ['r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : True,
}

# 13
precision_sphere = {
        'qpos' : np.array([-0.02958 ,
        0.843217,  1.184168,  0.335163, -0.02958 ,  0.783678,  1.144664,  0.349908,  0.032398,
        1.05972 ,  1.225552,  0.206142,  1.383549,  0.554968,  0.543468,  0.426165]),
        'specific_finger_names' : ['r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : True,
}


# defualt
# 14
tripod = {
        'qpos' : np.array([-0.261528,
        0.962294,  0.91893 ,  0.399673, -0.211758,  1.045287,  0.888832,  0.442066, -0.20988 ,
        1.483706,  1.487028,  1.350742,  1.396   ,  0.504299,  0.503182,  0.46186]),
        'specific_finger_names' : ['r_ff', 'r_mf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : True,
}


# 15
fixed_hook = {
        'qpos' : np.array([0.14978 ,
        0.944252,  1.426832,  0.773834,  0.08968 ,  1.00379 ,  1.306441,  0.794109,  0.13006 ,
        1.00379 ,  1.146545,  0.991327,  0.283374,  0.134413,  0.191883,  0.21758]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_rf'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 16
lateral={
        'qpos' : np.array([0.15 , 1.424, 1.406, 1.613, 0.19 , 1.364, 1.546, 1.333, 0.13 , 1.544, 1.606, 1.333, 0.263,
       0.435, 1.071, 0.818]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_th'], 
        'specific_sensor_names' : ['ff', 'th'], 
        'specific_ctrl_names' : ['ffa', 'rfa'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 1,1,1, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, -1,-1,-1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 1,1, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, -1,-1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 17
index_finger_extension = {
        'qpos' : np.array([0.45028 ,
        1.18421 ,  0.365881,  0.4734  ,  0.37046 ,  1.283441,  0.845566,  0.652186,  0.31036 ,
        1.303287,  1.278224,  0.491831,  0.722538,  0.548634,  0.728416,  0.319055]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 18
extension_type = {
        'qpos' : np.array([0.08968 ,
        0.063804,  0.537063,  0.716696, -0.02958 ,  0.224378,  0.315091,  0.681676,  0.02958 ,
        0.004266,  0.493797,  0.971052,  1.028143,  0.475164,  0.644182,  1.30183]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}


# 20
writing_tripod = {
        'qpos' : np.array([-0.4099  ,
        0.944252,  1.342182,  1.033719, -0.2296  ,  1.564895,  1.345944,  0.733285, -0.2897  ,
        1.604587,  1.40614 ,  1.472391,  1.093791,  0.567635,  0.602065,  1.09889]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_th'], 
        'specific_sensor_names' : ['palm', 'ff', 'mf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : True,
}

# 22
parallel_extension = {
        'qpos' : np.array([-0.17044 ,
        1.238336,  0.480629,  0.202456, -0.10001 ,  1.216685,  0.567161,  0.209828, -0.02958 ,
        1.292462,  0.493797,  0.019983,  1.383549,  0.254752,  0.371338,  0.057857]),
        'specific_finger_names' : ['r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}


# 23
adduction_grip = {
        'qpos' : np.array([-0.1931,  0.4918,  0.3709,  0.9151,  0.1367,  0.5308,  0.6393,  0.2108, -0.19  ,  1.524 ,
        1.446 ,  1.413 ,  0.643 ,  0.455 ,  0.711 ,  0.538 ]),
        'specific_finger_names' : ['r_ff', 'r_mf',], 
        'specific_sensor_names' : ['ff', 'mf'], 
        'specific_ctrl_names' : ['ffa', 'mfa', ], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 1,1,1, 1,1,1, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, -1,-1,-1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 1,1, 1,1, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, -1,-1, 1,1],  
        'do_finger_tip' : False,
}

# 24
tip_pinch= {
        'qpos' : np.array([-0.05024 ,
        0.989357,  0.91893 ,  0.759089, -0.20988 ,  0.114322,  0.292517,  0.561871, -0.20988 ,
        0.197315,  0.300042,  0.519479,  1.291868,  0.180015,  0.314571,  0.914736]),
        'specific_finger_names' : ['r_ff', 'r_th'], 
        'specific_sensor_names' : ['ff', 'th'], 
        'specific_ctrl_names' : ['ffa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : True,
}




# 25
lateral_tripod = {
        'qpos' : np.array([-0.14978 ,
        1.124671,  1.206741,  1.153524,  0.02958 ,  1.243748,  1.306441,  0.85309 ,  0.02958 ,
        1.564895,  1.526531,  1.393135,  1.02022 ,  0.799448,  0.495857,  1.245462]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 26
shpere_4_finger = {
        'qpos' : np.array([ -0.13  ,  0.764 ,  1.206 ,
        0.913 , -0.01  ,  0.624 ,  1.066 ,  0.773 ,  0.03  ,  1.024 ,  
        0.926 ,  0.873 ,  1.383 , 0.395 ,  0.091 ,  1.218 ]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 27 precision_quadpod
quadpod = {
        'qpos' : np.array([-0.13006 ,
        1.258182,  0.529538,  0.195083, -0.00986 ,  0.996573,  1.005462,  0.15269 ,  0.02958 ,
        1.12828 ,  0.999818,  0.252221,  1.383549,  0.39536 ,  0.376831,  0.405495]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : True,
}

# 28
shpere_3_finger = {
        'qpos' : np.array([-0.05024 ,
        1.01642 ,  1.018629,  0.624538,  0.05024 ,  1.065133,  1.048727,  0.666931, -0.20988 ,
        1.61    ,  1.464455,  1.166427,  1.383549,  0.315555,  0.314571,  0.899703]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 29 
stick = {
        'qpos' : np.array([-0.17983 ,
        0.762028,  1.464455,  0.906541, -0.08968 ,  1.364629,  1.421189,  0.766462, -0.08968 ,
        1.472881,  1.586727,  0.399673,  0.362604,  0.456163,  1.014078,  0.721187]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 30
palmar = {
        'qpos' : np.array([0.06996 ,
        1.279832,  1.449406,  0.624538,  0.097193,  1.258182,  1.515245,  0.43285 ,  0.14978 ,
        1.182406,  1.55851 ,  0.493674,  0.436176,  0.15088 ,  0.691792,  1.174055]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 1,1,1, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 1,1, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 31 - 2 finger
ring = {
        'qpos' : np.array([-0.13006 ,
        1.045287,  0.905762,  0.873365, -0.06996 ,  0.184685,  0.066783, -0.026096, -0.11034 ,
       -0.075119,  0.166483, -0.046371,  1.383549,  0.069809,  0.331052,  0.677967]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_th'], 
        'specific_sensor_names' : ['ff', 'th'], 
        'specific_ctrl_names' : ['ffa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}


# 31 - 3 finger
power_ring_3f = {
        'qpos' : np.array([-0.15 ,  0.884,  1.146,  
        0.853, -0.09 ,  1.124,  1.306,  0.493, -0.03 ,  0.064,  0.206,
        0.113,  1.383,  0.295,  0.311,  1.218]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 32 
ventral = {
        'qpos' : np.array([0.45028 ,
        0.596042,  0.46558 ,  0.349908,  0.129121,  1.285245,  1.515245,  0.801482,  0.085924,
        1.604587,  1.586727,  0.469713,  0.733857,  0.08881 ,  0.805325,  0.79823]),
        'specific_finger_names' : ['r_palm', 'r_ff', 'r_mf', 'r_rf', 'r_th'], 
        'specific_sensor_names' : ['ff', 'mf', 'rf', 'th'], 
        'specific_ctrl_names' : ['ffa', 'mfa', 'rfa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

# 33
inferior_pincer= {
        'qpos' : np.array([-0.14978 ,
        1.18421 ,  0.486273,  0.252221, -0.158232,  0.361497,  0.177769,  0.112141, -0.17044 ,
        0.623105, -0.008462, -0.015037,  1.282813,  0.254752,  0.591078,  0.05785]),
        'specific_finger_names' : ['r_ff', 'r_th'], 
        'specific_sensor_names' : ['ff', 'th'], 
        'specific_ctrl_names' : ['ffa', 'tha'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  
        'hand_af_dir_idx_in_mat' : [2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1],  
        'do_finger_tip' : False,
}

########## intermediate
# TODO 

# taxonomy_name_list = [
#         'large_diameter', # 1
#         'small_diameter', # 2
#         'medium_diameter', # 3    
#         'adducted_thumb', # 4
#         'light_tool', # 5
#         'prismatic_4_finger', # 6
#         'prismatic_3_finger', # 7 # 8
#         'palmar_pinch', # 9
#         'power_disk',   # 10
#         'power_sphere', # 11
#         'precision_disk', # 12
#         'precision_sphere', # 13
#         'tripod', # 14  
#         'fixed_hook', #  15
#         'lateral', # 16
#         'index_finger_extension', # 17
#         'extension_type', # 18
#         'writing_tripod', # 20
#         'parallel_extension', # 22 
#         # 'adduction_grip', # 23
#         'tip_pinch', # 24
#         'lateral_tripod', # 25
#         'shpere_4_finger', # 26
#         'quadpod', # 27
#         'shpere_3_finger', # 28
#         'stick', # 29
#         'palmar', # 30
#         'power_ring_2f', # 31
#         'power_ring_3f', # 31.5
#         'ventral', # 32
#         'inferior_pincer', # 33
# ]

taxonomy_name_list = [
        'large_diameter', # 1
        'small_diameter', # 2
        'medium_diameter', # 3    
        'adducted_thumb', # 4
        'light_tool', # 5
        'prismatic_3_finger', # 6
        'prismatic_2_finger', # 7 # 8
        'palmar_pinch', # 9
        'power_disk',   # 10
        'power_sphere', # 11
        'precision_disk', # 12
        'precision_sphere', # 13
        'tripod', # 14  
        'fixed_hook', #  15
        'lateral', # 16
        'index_finger_extension', # 17
        'extension_type', # 18
        'writing_tripod', # 20
        'parallel_extension', # 22 
        'tip_pinch', # 24
        'lateral_tripod', # 25
        'shpere_4_finger', # 26
        'quadpod', # 27
        'shpere_3_finger', # 28
        'stick', # 29
        'palmar', # 30
        'ring', # 31
        'power_ring_3f', # 31.5
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
}