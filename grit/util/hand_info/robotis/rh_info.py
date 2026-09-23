import numpy as np  

rh_mocap_name = "sh5_rh:mocap"
rh_wrist_base_name = "hx5_r_base"
rh_palm_site_name = "sh5_rh_0_touch" 
# Keypoint names used to locate the hand center
rh_thumb_link1_name = "finger_r_link1_4" 
rh_th_tip_name = ["sh5_rh_thumb_end", "sh5_rh_3_touch"]
rh_ff_tip_name = ["sh5_rh_index_end", "sh5_rh_6_touch"]
rh_mf_tip_name = ["sh5_rh_middle_end", "sh5_rh_9_touch"]
rh_rf_tip_name = ["sh5_rh_ring_end", "sh5_rh_12_touch"]
rh_lf_tip_name = ["sh5_rh_pinky_end", "sh5_rh_15_touch"]

rh_hand_center = np.array([[ 0.85236, -0.     , -0.52295,  0.0635 ],
                            [-0.00017,  1.     , -0.00028,  0.01434],
                            [ 0.52295,  0.00032,  0.85236,  0.10376],
                            [ 0.     ,  0.     ,  0.     ,  1.     ]])

rh_body_parts = ['hx5_r_base', 
                'finger_r_link1_1', 'finger_r_link1_2', 'finger_r_link1_3', 'finger_r_link1_4',  'sh5_rh_thumb_end',
                'finger_r_link2_1', 'finger_r_link2_2', 'finger_r_link2_3', 'finger_r_link2_4', 'sh5_rh_index_end',
                'finger_r_link3_1', 'finger_r_link3_2', 'finger_r_link3_3', 'finger_r_link3_4', 'sh5_rh_middle_end',
                'finger_r_link4_1', 'finger_r_link4_2', 'finger_r_link4_3', 'finger_r_link4_4', 'sh5_rh_ring_end',
                'finger_r_link5_1', 'finger_r_link5_2', 'finger_r_link5_3', 'finger_r_link5_4', 'sh5_rh_pinky_end',
                ] # 21
rh_af_parts = ['hx5_r_base', 
                'finger_r_link1_2', 'finger_r_link1_4', 'sh5_rh_thumb_end',
                'finger_r_link2_2', 'finger_r_link2_3', 'sh5_rh_index_end',
                'finger_r_link3_2', 'finger_r_link3_3', 'sh5_rh_middle_end',
                'finger_r_link4_2', 'finger_r_link4_3', 'sh5_rh_ring_end',
                'finger_r_link5_2', 'finger_r_link5_3', 'sh5_rh_pinky_end',
                ] # 11

# Touch sensor names added
# NOTE: the SENSOR is "ffw_sh5_r_arm_part_touch" (the SITE is "sh5_r_arm_part_touch").
# Using the site name here left the forearm touch sensor unindexed → forearm
# never read / never colourable. Use the actual sensor name.
fore_arm_touch_sensor = "ffw_sh5_r_arm_part_touch"
hand_touch_sensors = [
    "ffw_sh5_rh_0_touch",       # palm (sh5_rh_0_touch, per XML)
    "ffw_sh5_rh_1_touch",
    "ffw_sh5_rh_2_touch",
    "ffw_sh5_rh_3_touch",
    "ffw_sh5_rh_4_touch",
    "ffw_sh5_rh_5_touch",
    "ffw_sh5_rh_6_touch",
    "ffw_sh5_rh_7_touch",
    "ffw_sh5_rh_8_touch",
    "ffw_sh5_rh_9_touch",
    "ffw_sh5_rh_10_touch",
    "ffw_sh5_rh_11_touch",
    "ffw_sh5_rh_12_touch",
    "ffw_sh5_rh_13_touch",
    "ffw_sh5_rh_14_touch",
    "ffw_sh5_rh_15_touch"
]

hand_contact_sensors_for_object = [
    "ffw_sh5_rh_0_body_obj_contact",
    "ffw_sh5_rh_1_body_obj_contact",
    "ffw_sh5_rh_2_body_obj_contact",
    "ffw_sh5_rh_3_body_obj_contact",
    "ffw_sh5_rh_4_body_obj_contact",
    "ffw_sh5_rh_5_body_obj_contact",
    "ffw_sh5_rh_6_body_obj_contact",
    "ffw_sh5_rh_7_body_obj_contact",
    "ffw_sh5_rh_8_body_obj_contact",
    "ffw_sh5_rh_9_body_obj_contact",
    "ffw_sh5_rh_10_body_obj_contact",
    "ffw_sh5_rh_11_body_obj_contact",
    "ffw_sh5_rh_12_body_obj_contact",
    "ffw_sh5_rh_13_body_obj_contact",
    "ffw_sh5_rh_14_body_obj_contact",
    "ffw_sh5_rh_15_body_obj_contact",
]

hand_contact_sensors_for_table = [
    "ffw_sh5_rh_0_table_contact",
    "ffw_sh5_rh_1_table_contact",
    "ffw_sh5_rh_2_table_contact",
    "ffw_sh5_rh_3_table_contact",
    "ffw_sh5_rh_4_table_contact",
    "ffw_sh5_rh_5_table_contact",
    "ffw_sh5_rh_6_table_contact",
    "ffw_sh5_rh_7_table_contact",
    "ffw_sh5_rh_8_table_contact",
    "ffw_sh5_rh_9_table_contact",
    "ffw_sh5_rh_10_table_contact",
    "ffw_sh5_rh_11_table_contact",
    "ffw_sh5_rh_12_table_contact",
    "ffw_sh5_rh_13_table_contact",
    "ffw_sh5_rh_14_table_contact",
    "ffw_sh5_rh_15_table_contact",
]


hand_contact_sensors = [
    "ffw_sh5_rh_0_self_coll",
    "ffw_sh5_rh_1_self_coll",
    "ffw_sh5_rh_2_self_coll",
    "ffw_sh5_rh_3_self_coll",
    "ffw_sh5_rh_4_self_coll",
    "ffw_sh5_rh_5_self_coll",
    "ffw_sh5_rh_6_self_coll",
    "ffw_sh5_rh_7_self_coll",
    "ffw_sh5_rh_8_self_coll",
    "ffw_sh5_rh_9_self_coll",
    "ffw_sh5_rh_10_self_coll",
    "ffw_sh5_rh_11_self_coll",
    "ffw_sh5_rh_12_self_coll",
    "ffw_sh5_rh_13_self_coll",
    "ffw_sh5_rh_14_self_coll",
    "ffw_sh5_rh_15_self_coll",
]

anchor_body_name = "finger_r_link3_2"
palm_sensor_idx = 0  
fore_arm_sensor_idx = 1 
finger_motor_num = 20 
finger_link_num = 20
hand_qvel_num = 26 

ctrl_joint_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# coupled mechanism 
tip_body_idx = [4,7,10,13,16] # For contact in sensor_body_name_list
tip_end_name = "_end" 


# Action scale settings
wrist_transl_scale = 80.0 
wrist_rot_scale = 40.0 # Damping issue 
finger_action_scale = 5.0 # 5.0  