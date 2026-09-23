import numpy as np 

rh_mocap_name = "tesollo_rh:mocap"
rh_wrist_base_name  = "tesollo_rh_palm"
rh_palm_site_name   = "rh_dg_0_touch"
# Keypoint names used to locate the hand center
rh_thumb_link1_name = "rl_dg_1_4" 
rh_th_tip_name      = ["rl_dg_1_Tip", "rl_dg_1_Tip_Face", "rh_dg_1_4_touch"]
rh_ff_tip_name      = ["rl_dg_2_Tip", "rl_dg_2_Tip_Face", "rh_dg_2_4_touch"]  
rh_mf_tip_name      = ["rl_dg_3_Tip", "rl_dg_3_Tip_Face", "rh_dg_3_4_touch"]   
rh_rf_tip_name      = ["rl_dg_4_Tip", "rl_dg_4_Tip_Face", "rh_dg_4_4_touch"]    
rh_lf_tip_name      = ["rl_dg_5_Tip", "rl_dg_5_Tip_Face", "rh_dg_5_4_touch"]

rh_hand_center = np.array([[ 0.860527,  0.      , -0.509405,  0.066034],
                                        [ 0.00019 ,  1.      ,  0.000322,  0.006132],
                                        [ 0.509405, -0.000374,  0.860527,  0.162842],
                                        [ 0.      ,  0.      ,  0.      ,  1.      ]])
                                        
# rh_body_names = [
#                 'arm_part', 'tesollo_rh_palm',
#                 'rl_dg_1_1', 'rl_dg_1_2', 'rl_dg_1_3', 'rl_dg_1_4', 
#                 'rl_dg_2_1', 'rl_dg_2_2', 'rl_dg_2_3', 'rl_dg_2_4',  
#                 'rl_dg_3_1', 'rl_dg_3_2', 'rl_dg_3_3', 'rl_dg_3_4',  
#                 'rl_dg_4_1', 'rl_dg_4_2', 'rl_dg_4_3', 'rl_dg_4_4',  
#                 'rl_dg_5_1', 'rl_dg_5_2', 'rl_dg_5_3', 'rl_dg_5_4',   
#                 ] # 21 for contact 
rh_body_parts = ['tesollo_rh_palm',
                'rl_dg_1_1', 'rl_dg_1_2', 'rl_dg_1_3', 'rl_dg_1_4', 'rl_dg_1_Tip', 'rl_dg_1_Tip_Face',
                'rl_dg_2_1', 'rl_dg_2_2', 'rl_dg_2_3', 'rl_dg_2_4', 'rl_dg_2_Tip', 'rl_dg_2_Tip_Face',  
                'rl_dg_3_1', 'rl_dg_3_2', 'rl_dg_3_3', 'rl_dg_3_4', 'rl_dg_3_Tip', 'rl_dg_3_Tip_Face',   
                'rl_dg_4_1', 'rl_dg_4_2', 'rl_dg_4_3', 'rl_dg_4_4', 'rl_dg_4_Tip', 'rl_dg_4_Tip_Face',  
                'rl_dg_5_1', 'rl_dg_5_2', 'rl_dg_5_3', 'rl_dg_5_4', 'rl_dg_5_Tip', 'rl_dg_5_Tip_Face',   
                ] # 31
rh_af_parts   = ['tesollo_rh_palm',
                'rl_dg_1_2', 'rl_dg_1_4', 'rl_dg_1_Tip',
                'rl_dg_2_1', 'rl_dg_2_3', 'rl_dg_2_Tip',  
                'rl_dg_3_1', 'rl_dg_3_3', 'rl_dg_3_Tip',   
                'rl_dg_4_1', 'rl_dg_4_3', 'rl_dg_4_Tip',  
                'rl_dg_5_2', 'rl_dg_5_4', 'rl_dg_5_Tip',   
                ] # 16     

# Touch sensor names added
fore_arm_touch_sensor = "tesollo_r_arm_part_touch"
hand_touch_sensors = [
    "tesollo_rh_dg_palm_body_touch",     # palm
    "tesollo_rh_dg_1_2_body_touch",
    "tesollo_rh_dg_1_3_body_touch",
    "tesollo_rh_dg_1_4_body_touch",
    "tesollo_rh_dg_2_2_body_touch",
    "tesollo_rh_dg_2_3_body_touch",
    "tesollo_rh_dg_2_4_body_touch",
    "tesollo_rh_dg_3_2_body_touch",
    "tesollo_rh_dg_3_3_body_touch",
    "tesollo_rh_dg_3_4_body_touch",
    "tesollo_rh_dg_4_2_body_touch",
    "tesollo_rh_dg_4_3_body_touch",
    "tesollo_rh_dg_4_4_body_touch",
    "tesollo_rh_dg_5_1_body_touch",
    "tesollo_rh_dg_5_3_body_touch",
    "tesollo_rh_dg_5_4_body_touch"
]
hand_contact_sensors_for_object = [
    "tesollo_rh_dg_palm_body_obj_contact",
    "tesollo_rh_dg_1_2_body_obj_contact",
    "tesollo_rh_dg_1_3_body_obj_contact",
    "tesollo_rh_dg_1_4_body_obj_contact",
    "tesollo_rh_dg_2_2_body_obj_contact",
    "tesollo_rh_dg_2_3_body_obj_contact",
    "tesollo_rh_dg_2_4_body_obj_contact",
    "tesollo_rh_dg_3_2_body_obj_contact",
    "tesollo_rh_dg_3_3_body_obj_contact",
    "tesollo_rh_dg_3_4_body_obj_contact",
    "tesollo_rh_dg_4_2_body_obj_contact",
    "tesollo_rh_dg_4_3_body_obj_contact",
    "tesollo_rh_dg_4_4_body_obj_contact",
    "tesollo_rh_dg_5_1_body_obj_contact",
    "tesollo_rh_dg_5_3_body_obj_contact",
    "tesollo_rh_dg_5_4_body_obj_contact"
]
hand_contact_sensors_for_table = [
    "tesollo_rh_dg_palm_table_contact",
    "tesollo_rh_dg_1_2_table_contact",
    "tesollo_rh_dg_1_3_table_contact",
    "tesollo_rh_dg_1_4_table_contact",
    "tesollo_rh_dg_2_2_table_contact",
    "tesollo_rh_dg_2_3_table_contact",
    "tesollo_rh_dg_2_4_table_contact",
    "tesollo_rh_dg_3_2_table_contact",
    "tesollo_rh_dg_3_3_table_contact",
    "tesollo_rh_dg_3_4_table_contact",
    "tesollo_rh_dg_4_2_table_contact",
    "tesollo_rh_dg_4_3_table_contact",
    "tesollo_rh_dg_4_4_table_contact",
    "tesollo_rh_dg_5_1_table_contact",
    "tesollo_rh_dg_5_3_table_contact",
    "tesollo_rh_dg_5_4_table_contact"
]
hand_contact_sensors = [
    "tesollo_rh_dg_palm_self_coll",
    "tesollo_rh_dg_1_2_self_coll",
    "tesollo_rh_dg_1_3_self_coll",
    "tesollo_rh_dg_1_4_self_coll",
    "tesollo_rh_dg_2_2_self_coll",
    "tesollo_rh_dg_2_3_self_coll",
    "tesollo_rh_dg_2_4_self_coll",
    "tesollo_rh_dg_3_2_self_coll",
    "tesollo_rh_dg_3_3_self_coll",
    "tesollo_rh_dg_3_4_self_coll",
    "tesollo_rh_dg_4_2_self_coll",
    "tesollo_rh_dg_4_3_self_coll",
    "tesollo_rh_dg_4_4_self_coll",
    "tesollo_rh_dg_5_1_self_coll",
    "tesollo_rh_dg_5_3_self_coll",
    "tesollo_rh_dg_5_4_self_coll"
]

anchor_body_name = "rl_dg_3_1"
palm_sensor_idx = 0  
fore_arm_sensor_idx = 1 
finger_motor_num = 20 
finger_link_num = 20 
hand_qvel_num = 26 

ctrl_joint_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
tip_body_idx = [4, 7, 10, 13, 16] # For contact in sensor_body_name_list
tip_end_name = "_Tip" 

# Action scale settings
wrist_transl_scale = 80.0 
wrist_rot_scale = 40.0 # Damping issue 
finger_action_scale = 5.0 # 5.0  