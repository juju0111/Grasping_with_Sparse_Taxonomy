# import jax.numpy as jp 
import numpy as np 

######################
# 0 
none = {
        # Dummy qpos
        'qpos' : np.array([-1.798 ,  0.    ,  0.6   ,  0.4217, -0.2636,  0.8455,  0.6   ,  0.2994, -0.2036,  0.6455,
        0.8   ,  0.4305,  0.0964,  0.7055,  0.68  ,  0.3498,  0.0764,  0.9055,  0.62  ,  0.3117]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        # TODO
        # 'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        # 'hand_face_dir_sign' : [1, -1,-1,-1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        # 'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        # 'hand_af_dir_sign' : [1, -1,-1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 1 
large_diameter = {
        'qpos' : np.array([-0.083972 , -1.76159  ,  0.5292   ,  0.2892   , -0.298879 ,
                        0.78     ,  0.4892   ,  0.7092   , -0.130865 ,  0.6      ,
                        0.6492   ,  0.5892   , -0.010865 ,  0.66     ,  0.5292   ,
                        0.6492   ,  0.2425467,  0.3681121,  0.6492   ,  0.9092   ]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 2 
small_diameter = {
        'qpos' : np.array([0.121336,
       -1.541204,  0.655702,  0.844298, -0.298747,  0.619584,  1.009934,  1.206729, -0.010613,
        0.599515,  1.126371,  1.295287,  0.061106,  0.600561,  1.267747,  1.21856 , -0.017453,
        0.137189,  1.121773,  1.570]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 3
medium_diameter = {
        'qpos' : np.array([-0.083587,
       -1.762101,  0.649142,  0.529425, -0.298747,  0.779994,  0.709821,  1.17721 , -0.191169,
        0.660052,  0.909896,  1.090292, -0.010613,  0.600561,  0.915507,  1.26616 , -0.017453,
        0.127943,  0.869493,  1.28996]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 4
adducted_thumb = {
        'qpos' : np.array([ 0.216028 , -0.58159  ,  0.2292   ,  0.1692   , -0.078879 ,
                        0.72     ,  0.9692   ,  1.5692   , -0.110865 ,  0.72     ,
                        1.0892   ,  1.5692   , -0.070865 ,  0.48     ,  1.0292   ,
                        1.4492   , -0.0174533, -0.018879 ,  0.6692   ,  1.5692   ]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 5
light_tool = {
        'qpos' : np.array([ 0.155702,
       -0.24121 ,  0.542545,  1.095212, -0.098689,  0.918347,  0.968935,  1.549481, -0.110642,
        1.052568,  1.088652,  1.549481, -0.071008,  0.813418,  1.274093,  1.448627, -0.017453,
       -0.019065,  1.115427,  1.54858]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 6
prismatic_4_finger = {
        'qpos' : np.array([0.32626 ,
       -1.523431,  0.7623  ,  0.004638, -0.298747,  0.779994,  0.709821,  0.709821, -0.191169,
        0.660052,  1.106691,  0.191594, -0.010613,  0.725995,  0.88536 ,  0.201507, -0.017453,
        0.127943,  0.869493,  0.92978]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 7 
prismatic_3_finger = {
        'qpos' : np.array([-0.083587,
       -1.782414,  0.644223,  0.155514, -0.298747,  0.858193,  0.882017,  0.098116, -0.110013,
        0.736212,  0.962375,  0.060397, -0.010613,  0.798214,  1.001187,  0.134867, -0.017453,
        0.127943,  1.569213,  1.56921]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 8
prismatic_2_finger = {
        'qpos' : np.array([0.116245,
       -1.322846,  0.7377  ,  0.129275, -0.298747,  0.896291,  0.709821,  0.042357, -0.191169,
        0.843617,  0.909896,  0.135835, -0.010613,  0.600561,  1.569213,  1.569213, -0.017453,
        0.127943,  1.569213,  1.56921]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 9
palmar_pinch = {
        'qpos' : np.array([-0.0034  ,
       -1.401556,  0.542545,  0.173554, -0.186765,  1.056701,  0.644223,  0.022678, -0.081074,
        0.660052,  0.129275,  0.160434, -0.010613,  0.600561,  0.32844 ,  0.098373, -0.017453,
        0.127943,  0.529947,  0.24276]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 10
power_disk = {
        'qpos' : np.array([0.686467,
       -1.41679 ,  0.824618,  0.813138, -0.098689,  0.040102,  1.349405,  1.449443, -0.010613,
        0.119122,  1.388764,  1.369085,  0.049153,  0.079821,  1.569213,  1.369293, -0.017453,
        0.207919,  1.28996 ,  1.56921]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 11 
power_sphere = {
        'qpos' : np.array([0.471361,
       -1.581829,  0.399869,  0.850858, -0.318879,  1.056701,  0.806579,  0.962375, -0.191169,
        0.902202,  0.888577,  1.000094,  0.075576,  0.790612,  0.898053,  1.01388 , -0.017453,
        0.276801,  1.224907,  1.377227]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 12
precision_disk = {
        'qpos' : np.array([0.521001,
       -0.891207,  0.191594,  0.949255, -0.25408 ,  0.421076,  0.944335,  0.806579, -0.010613,
        0.119122,  1.313326,  0.668822,  0.241033,  0.383903,  1.110667,  1.074173, -0.017453,
        0.410865,  1.044027,  1.32169]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 13 
precision_sphere = {
        'qpos' : np.array([0.666102,
       -1.327924,  0.273592,  0.524505, -0.318879,  0.71984 ,  1.026333,  0.242432, -0.010613,
        0.745976,  0.882017,  0.129275,  0.236629,  0.399107,  1.41848 ,  0.303053, -0.017453,
        0.388213,  1.358187,  0.77112]),
        'specific_finger_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# defualt
#14
tripod = {
        'qpos' : np.array([ 0.243527,
       -1.386322,  0.599944,  0.235873, -0.280503,  0.681742,  1.188689,  0.073516, -0.198718,
        0.888532,  0.944335,  0.186674,  0.185671,  0.900841,  1.569213,  1.569213, -0.017453,
        0.241204,  1.569213,  1.569213]),
        'specific_finger_names' : ['dg_1', 'dg_2', 'dg_3'] ,
        'specific_sensor_names' : ['dg_1', 'dg_2', 'dg_3'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,

}

# 15
fixed_hook = {
        'qpos' : np.array([ 0.216028 , -0.58159  ,  0.2292   ,  0.1692   , -0.078879 ,
                        0.72     ,  0.9692   ,  1.5692   , -0.110865 ,  0.72     ,
                        1.0892   ,  1.5692   , -0.070865 ,  0.48     ,  1.0292   ,
                        1.4492   , -0.0174533, -0.018879 ,  0.6692   ,  1.5692   ]),
        'specific_finger_names' : ['palm', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 16
lateral = {
        'qpos' : np.array([ 0.216028 , -0.60159  ,  0.4092   ,  1.0292   , -0.218879 ,
                        0.72     ,  1.2092   ,  1.5492   , -0.110865 ,  0.96     ,
                        1.5692   ,  1.5492   , -0.070865 ,  0.84     ,  1.5692   ,
                        1.5692   , -0.0174533, -0.018879 ,  1.4692   ,  1.5492   ]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2'] ,
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 1,1,1, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 1,1, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 17
index_finger_extension = {
        'qpos' : np.array([ 0.216028  , -0.60159   ,  0.6492    ,  0.7292    , -0.318879  ,
                        0.24      ,  0.2492    ,  0.1692    , -0.170865  ,  0.72      ,
                        1.0892    ,  1.5492    , -0.070865  ,  0.83999645,  1.18919994,
                        1.4492    , -0.0174533 , -0.018879  ,  1.1292    ,  1.5492    ]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 18
extension_type = {
        'qpos' : np.array([ 2.36028000e-01, -1.18159000e+00,  6.69200000e-01,  1.14920000e+00,
                        -2.98879000e-01,  3.71399312e-11,  2.29200000e-01,  5.29200000e-01,
                        -1.30865000e-01,  5.99999994e-02,  3.49200000e-01,  4.09200000e-01,
                        1.09135000e-01,  3.98880169e-11,  2.29200000e-01,  6.49200000e-01,
                        -1.74533000e-02,  3.21120994e-01,  2.89199749e-01,  5.49178368e-01]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 19 (scissors) is skipped

# 20 # 21 
writing_tripod = {
        'qpos' : np.array([ 0.545185,
       -0.639841,  0.122715,  0.982054, -0.102463,  0.972486,  0.919736,  0.824618, -0.010613,
        1.298623,  1.344486,  0.699981, -0.010613,  1.320854,  1.548587,  1.074173, -0.017453,
        0.241204,  1.548587,  1.54858]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 1,1,1, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 1,1, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 22 
parallel_extension = {
        'qpos' : np.array([ 0.156028 , -1.76159  ,  0.5292   ,  0.2692   , -0.278879 ,
                        0.9      ,  0.7292   , -0.0308   , -0.190865 ,  0.72     ,
                        0.9092   ,  0.1692   , -0.010865 ,  0.78     ,  0.7292   ,
                        0.2292   ,  0.0425467,  0.081121 ,  0.6692   ,  1.0892   ]),
        'specific_finger_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
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
        'specific_finger_names' : ['dg_2', 'dg_3'],
        'specific_sensor_names' : ['dg_2', 'dg_3'], 
        'specific_ctrl_names' : ['dg_2', 'dg_3'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 1,1,1, 1,1,1, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, -1,-1,-1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 1,0, 1,1, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, -1,-1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# Remaining taxonomies
# 24 
tip_pinch= {
        'qpos' : np.array([-0.064495,
       -1.500579,  0.468747,  0.688501, -0.11945 ,  0.900301,  0.629463,  0.649142,  0.056073,
        0.394469,  0.166994,  0.399869,  0.134713,  0.458023,  0.3094  ,  0.268147,  0.06446 ,
        0.200523,  0.43792 ,  0.898053]),
        'specific_finger_names' : ['dg_1', 'dg_2',],
        'specific_sensor_names' : ['dg_1', 'dg_2'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 25
lateral_tripod = {
        'qpos' : np.array([0.461179,
       -1.322846,  0.068597,  1.308406, -0.098689,  0.960455,  0.909896,  1.093572, -0.030744,
        1.320104,  1.129651,  0.688501,  0.023989,  1.463392,  1.44228 ,  1.101147, -0.017453,
        0.065996,  1.569213,  1.5708 ]),
        'specific_finger_names' : ['dg_1', 'dg_2', 'dg_3'] ,
        'specific_sensor_names' : ['dg_1', 'dg_2', 'dg_3'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,

}

# 26
shpere_4_finger = {
        'qpos' : np.array([0.078061,
       -1.571673,  0.280151,  0.531065, -0.298747,  1.120865,  0.649142,  0.286711, -0.150905,
        0.900249,  0.621263,  0.688501,  0.089417,  0.900841,  0.669573,  0.588653, -0.017453,
        0.248139,  1.569213,  1.5708]),
        'specific_finger_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4'] ,
        'specific_sensor_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,

}

# 27 precision_quadpod
quadpod = {
        'qpos' : np.array([ 0.09588 ,
       -1.609758,  0.537625,  0.117795, -0.266033,  1.120865,  0.686862,  0.286711, -0.08359 ,
        0.947116,  0.788539,  0.293271,  0.089417,  1.016773,  0.669573,  0.382387, -0.017453,
        0.248139,  1.569213,  1.56921]),
        'specific_finger_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', ],
        'specific_sensor_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', ], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', ], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : True,
}

# 28 precision_quadpod
shpere_3_finger = {
        'qpos' : np.array([0.056423,
       -1.533587,  0.599944,  0.668822, -0.298747,  0.842152,  0.73114 ,  0.7377  ,  0.087529,
        0.900249,  0.788539,  0.688501,  0.075576,  1.592627,  1.236013,  1.483533, -0.017453,
        0.248139,  1.569213,  1.56921]),
        'specific_finger_names' : ['dg_1', 'dg_2', 'dg_3',],
        'specific_sensor_names' : ['dg_1', 'dg_2', 'dg_3',], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3',], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 29
stick = {
        'qpos' : np.array([0.228253,
       -0.639841,  0.76886 ,  0.32935 , -0.098689,  0.16041 ,  1.329726,  1.288727, -0.032632,
        0.72059 ,  1.068972,  1.469122,  0.012665,  0.674681,  1.20904 ,  1.448627, -0.017453,
        0.144123,  1.348667,  1.52954]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 30
palmar = {
        'qpos' : np.array([-0.383972,
       -0.002539,  0.809859,  0.668822, -0.09051 ,  1.43968 ,  1.55276 ,  0.166994, -0.035148,
        1.417745,  1.5708  ,  0.217833,  0.005744,  1.292346,  1.5708  ,  0.515667, -0.017453,
        0.068308,  1.569213,  1.424827]),
        'specific_finger_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 31 
ring = {
        'qpos' : np.array([.034785,
       -1.434564,  0.431028,  0.688501, -0.138952,  0.980506,  0.709821,  0.32935 , -0.008725,
        0.300734,  0.168634,  0.449067,  0.041604,  0.060816,  0.429987,  0.32844 ,  0.026876,
        0.22826 ,  0.40936 ,  0.28877]),
        'specific_finger_names' : ['dg_1', 'dg_2', ],
        'specific_sensor_names' : ['dg_1', 'dg_2'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 32
ventral = {
        'qpos' : np.array([ 0.156028 , -0.86159  ,  1.0492   ,  0.3292   , -0.098879 ,
                        0.6      ,  0.3292   ,  0.2492   , -0.110865 ,  0.72     ,
                        0.6692   ,  1.5692   , -0.070865 ,  0.48     ,  1.2092   ,
                        1.4492   , -0.0174533, -0.0318879,  1.3492   ,  1.5092   ]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2', 'dg_3', 'dg_4', 'dg_5'], 
        'hand_face_dir_idx_in_mat' : [0, 2,2,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'hand_face_dir_sign' : [1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        'hand_af_dir_idx_in_mat' : [0, 2,2, 0,0, 0,0, 0,0, 0,0],
        'hand_af_dir_sign' : [1, 1,1, 1,1, 1,1, 1,1, 1,1],
        'do_finger_tip' : False,
}

# 33 
inferior_pincer = {
        'qpos' : np.array([0.029694,
       -1.401556,  0.468747,  0.129275,  0.000711,  1.43968 ,  0.029238,  0.32935 ,  0.07306 ,
        0.298781,  0.229313,  0.180114,  0.130309,  0.29838 ,  0.236413,  0.15232 ,  0.057714,
        0.386826,  0.231653,  0.255453]),
        'specific_finger_names' : ['palm', 'dg_1', 'dg_2'],
        'specific_sensor_names' : ['palm', 'dg_1', 'dg_2'], 
        'specific_ctrl_names' : ['dg_1', 'dg_2'], 
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
