import numpy as np 

rh_mocap_name = "inspire_rh:mocap"
rh_wrist_base_name = "inspire_rh_palm"
rh_palm_site_name = "inspire_rh_palm_touch"
# Keypoint names used to locate the hand center
rh_thumb_link1_name = "inspire_right_thumb_3" 
rh_th_tip_name = ["inspire_right_thumb_tip", "inspire_right_thumb_face_tip", "inspire_rh_thumb_tip_touch"]
rh_ff_tip_name = ["inspire_right_index_tip", "inspire_right_index_face_tip", "inspire_rh_index_tip_touch"]  
rh_mf_tip_name = ["inspire_right_middle_tip", "inspire_right_middle_face_tip", "inspire_rh_middle_tip_touch"]   
rh_rf_tip_name = ["inspire_right_ring_tip", "inspire_right_ring_face_tip", "inspire_rh_ring_tip_touch"]    
rh_lf_tip_name = ["inspire_right_little_tip", "inspire_right_little_face_tip", "inspire_rh_little_tip_touch"]

rh_hand_center = np.array([[ 0.801466,  0.      , -0.59804 ,  0.045786],
                            [-0.000046,  1.      , -0.000062,  0.02054 ],
                            [ 0.59804 ,  0.000077,  0.801466,  0.142958],
                            [ 0.      ,  0.      ,  0.      ,  1.      ]])


# rh_body_names = ['arm_part', 'inspire_rh_palm',  
#                 'inspire_right_thumb_1', 'inspire_right_thumb_2', 'inspire_right_thumb_3', 'inspire_right_thumb_4', 
#                 'inspire_right_index_1', 'inspire_right_index_2', 'inspire_right_middle_1', 'inspire_right_middle_2',
#                 'inspire_right_ring_1', 'inspire_right_ring_2', 'inspire_right_little_1', 'inspire_right_little_2',
#                 ] # 13 
rh_body_parts = ['inspire_rh_palm',  
                'inspire_right_thumb_1', 'inspire_right_thumb_2', 'inspire_right_thumb_3', 'inspire_right_thumb_4', 
                'inspire_right_thumb_face_tip', 'inspire_right_thumb_tip',
                'inspire_right_index_1', 'inspire_right_index_2', 'inspire_right_index_face_tip', 'inspire_right_index_tip',
                'inspire_right_middle_1', 'inspire_right_middle_2', 'inspire_right_middle_face_tip', 'inspire_right_middle_tip',
                'inspire_right_ring_1', 'inspire_right_ring_2', 'inspire_right_ring_face_tip', 'inspire_right_ring_tip',
                'inspire_right_little_1', 'inspire_right_little_2', 'inspire_right_little_face_tip', 'inspire_right_little_tip',
                ] # 13 + 10 
rh_af_parts   = ['inspire_rh_palm',  
                'inspire_right_thumb_2',  'inspire_right_thumb_3', 'inspire_right_thumb_face_tip',  
                'inspire_right_index_1',  'inspire_right_index_2', 'inspire_right_index_face_tip',  
                'inspire_right_middle_1', 'inspire_right_middle_2', 'inspire_right_middle_face_tip', 
                'inspire_right_ring_1',   'inspire_right_ring_2', 'inspire_right_ring_face_tip',   
                'inspire_right_little_1', 'inspire_right_little_2', 'inspire_right_little_face_tip', 
                ] # 16 

# Touch sensor names added
fore_arm_touch_sensor = "inspire_r_arm_part_touch"
hand_touch_sensors = [
    "inspire_rh_palm_body_touch",      # palm
    "inspire_rh_thumb_2_body_touch",
    "inspire_rh_thumb_3_body_touch",
    "inspire_rh_thumb_tip_body_touch",
    "inspire_rh_index_1_body_touch",
    "inspire_rh_index_tip_body_touch",
    "inspire_rh_middle_1_body_touch",
    "inspire_rh_middle_tip_body_touch",
    "inspire_rh_ring_1_body_touch",
    "inspire_rh_ring_tip_body_touch",
    "inspire_rh_little_1_body_touch",
    "inspire_rh_little_tip_body_touch"
]

hand_contact_sensors_for_object = [
    "inspire_rh_palm_body_obj_contact",
    "inspire_rh_thumb_2_body_obj_contact",
    "inspire_rh_thumb_3_body_obj_contact",
    "inspire_rh_thumb_tip_body_obj_contact",
    "inspire_rh_index_1_body_obj_contact",
    "inspire_rh_index_tip_body_obj_contact",
    "inspire_rh_middle_1_body_obj_contact",
    "inspire_rh_middle_tip_body_obj_contact",
    "inspire_rh_ring_1_body_obj_contact",
    "inspire_rh_ring_tip_body_obj_contact",
    "inspire_rh_little_1_body_obj_contact",
    "inspire_rh_little_tip_body_obj_contact"
]

hand_contact_sensors_for_table = [
    "inspire_rh_palm_table_contact",
    "inspire_rh_thumb_2_table_contact",
    "inspire_rh_thumb_3_table_contact",
    "inspire_rh_thumb_tip_table_contact",
    "inspire_rh_index_1_table_contact",
    "inspire_rh_index_tip_table_contact",
    "inspire_rh_middle_1_table_contact",
    "inspire_rh_middle_tip_table_contact",
    "inspire_rh_ring_1_table_contact",
    "inspire_rh_ring_tip_table_contact",
    "inspire_rh_little_1_table_contact",
    "inspire_rh_little_tip_table_contact"
]

hand_contact_sensors = [
    "inspire_rh_palm_self_coll",
    "inspire_rh_thumb_2_self_coll",
    "inspire_rh_thumb_3_self_coll",
    "inspire_rh_thumb_tip_self_coll",
    "inspire_rh_index_1_self_coll",
    "inspire_rh_index_tip_self_coll",
    "inspire_rh_middle_1_self_coll",
    "inspire_rh_middle_tip_self_coll",
    "inspire_rh_ring_1_self_coll",
    "inspire_rh_ring_tip_self_coll",
    "inspire_rh_little_1_self_coll",
    "inspire_rh_little_tip_self_coll"
]

anchor_body_name = "inspire_right_middle_1"
palm_sensor_idx = 0  
fore_arm_sensor_idx = 1 
finger_motor_num = 6 # ok 
finger_link_num = 12 # ok 
hand_qvel_num = 18 # ok 

ctrl_joint_idx = [0,1,4,6,8,10] # coupled mechanism 
tip_body_idx = [4,6,8,10,12] # For contact in sensor_body_name_list
tip_end_name = "_tip" 

# Action scale settings
wrist_transl_scale = 80.0 
wrist_rot_scale = 40.0 # Damping issue 
finger_action_scale = 5.0 # 5.0  
