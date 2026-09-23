import numpy as np 

rh_mocap_name = "alle_r_hand:mocap"
rh_wrist_base_name = "alle_r_palm"
rh_palm_site_name = "allegro_rh_palm_touch"
# Keypoint names used to locate the hand center
rh_thumb_link1_name = "alle_r_th_distal" 
rh_th_tip_name = ["alle_r_th_end", "alle_r_th_face_end", "allegro_rh_th_3_touch"]
rh_ff_tip_name = ["alle_r_ff_end", "alle_r_ff_face_end", "allegro_rh_ff_3_touch"]  
rh_mf_tip_name = ["alle_r_mf_end", "alle_r_mf_face_end", "allegro_rh_mf_3_touch"]   
rh_rf_tip_name = ["alle_r_rf_end", "alle_r_rf_face_end", "allegro_rh_rf_3_touch"] 

rh_hand_center = np.array([[ 0.83, -0.  , -0.56,  0.06],
                            [ 0.01,  1.  ,  0.01,  0.02],
                            [ 0.56, -0.02,  0.83,  0.01],
                            [ 0.  ,  0.  ,  0.  ,  1.  ]])

# rh_body_names = ['arm_part', 'alle_r_palm', 
#                 'alle_r_ff_base', 'alle_r_ff_proximal', 'alle_r_ff_medial', 
#                 'alle_r_ff_distal', 'alle_r_ff_tip', 'alle_r_mf_base', 
#                 'alle_r_mf_proximal', 'alle_r_mf_medial', 'alle_r_mf_distal', 
#                 'alle_r_mf_tip', 'alle_r_rf_base', 'alle_r_rf_proximal', 
#                 'alle_r_rf_medial', 'alle_r_rf_distal', 'alle_r_rf_tip',
#                 'alle_r_th_base','alle_r_th_proximal','alle_r_th_medial','alle_r_th_distal','alle_r_th_tip',
#                 ] # 21 for contact 
rh_body_parts = ['alle_r_palm',
                'alle_r_ff_proximal', 'alle_r_ff_medial', 'alle_r_ff_distal', 'alle_r_ff_tip', 'alle_r_ff_end', 'alle_r_ff_face_end',
                'alle_r_mf_proximal', 'alle_r_mf_medial', 'alle_r_mf_distal', 'alle_r_mf_tip', 'alle_r_mf_end', 'alle_r_mf_face_end',
                'alle_r_rf_proximal', 'alle_r_rf_medial', 'alle_r_rf_distal', 'alle_r_rf_tip', 'alle_r_rf_end', 'alle_r_rf_face_end',
                'alle_r_th_proximal','alle_r_th_medial','alle_r_th_distal','alle_r_th_tip', 'alle_r_th_end',   'alle_r_th_face_end' 
                ] # 25 
rh_af_parts   = ['alle_r_palm',
                'alle_r_ff_proximal', 'alle_r_ff_medial', 'alle_r_ff_face_end',
                'alle_r_mf_proximal', 'alle_r_mf_medial', 'alle_r_mf_face_end',
                'alle_r_rf_proximal', 'alle_r_rf_medial', 'alle_r_rf_face_end',
                'alle_r_th_proximal', 'alle_r_th_distal', 'alle_r_th_face_end' 
                ] # 13 

# Touch sensor names as defined in allegro_rh_w_fore_arm_touch_contact.xml
fore_arm_touch_sensor = "allegro_r_arm_part_touch"
hand_touch_sensors = [
    "allegro_rh_palm_body_touch",      # palm
    "allegro_rh_ff_1_body_touch",      # index finger - proximal
    "allegro_rh_ff_2_body_touch",      # index finger - middle
    "allegro_rh_ff_3_body_touch",      # index finger - distal
    "allegro_rh_mf_1_body_touch",      # middle finger - proximal
    "allegro_rh_mf_2_body_touch",      # middle finger - middle
    "allegro_rh_mf_3_body_touch",      # middle finger - distal
    "allegro_rh_rf_1_body_touch",      # ring finger - proximal
    "allegro_rh_rf_2_body_touch",      # ring finger - middle
    "allegro_rh_rf_3_body_touch",      # ring finger - distal
    "allegro_rh_thumb_1_body_touch",   # thumb - proximal
    "allegro_rh_thumb_2_body_touch",   # thumb - middle
    "allegro_rh_thumb_3_body_touch"    # thumb - distal
    # Note: allegro_rh_thumb_4_body_touch is not present in the XML (commented out)
]

# Contact with the object
hand_contact_sensors_for_object = [
    "allegro_rh_palm_body_obj_contact",
    "allegro_rh_ff_1_body_obj_contact",
    "allegro_rh_ff_2_body_obj_contact",
    "allegro_rh_ff_3_body_obj_contact",
    "allegro_rh_mf_1_body_obj_contact",
    "allegro_rh_mf_2_body_obj_contact",
    "allegro_rh_mf_3_body_obj_contact",
    "allegro_rh_rf_1_body_obj_contact",
    "allegro_rh_rf_2_body_obj_contact",
    "allegro_rh_rf_3_body_obj_contact",
    "allegro_rh_thumb_1_body_obj_contact",
    "allegro_rh_thumb_2_body_obj_contact",
    "allegro_rh_thumb_3_body_obj_contact"
]

hand_contact_sensors_for_table = [
    "allegro_rh_palm_table_contact",
    "allegro_rh_ff_1_table_contact",
    "allegro_rh_ff_2_table_contact",
    "allegro_rh_ff_3_table_contact",
    "allegro_rh_mf_1_table_contact",
    "allegro_rh_mf_2_table_contact",
    "allegro_rh_mf_3_table_contact",
    "allegro_rh_rf_1_table_contact",
    "allegro_rh_rf_2_table_contact",
    "allegro_rh_rf_3_table_contact",
    "allegro_rh_thumb_1_table_contact",
    "allegro_rh_thumb_2_table_contact",
    "allegro_rh_thumb_3_table_contact"
]

# Self-collision 
hand_contact_sensors = [
    "allegro_rh_palm_self_coll",
    "allegro_rh_ff_1_self_coll",
    "allegro_rh_ff_2_self_coll",
    "allegro_rh_ff_3_self_coll",
    "allegro_rh_mf_1_self_coll",
    "allegro_rh_mf_2_self_coll",
    "allegro_rh_mf_3_self_coll",
    "allegro_rh_rf_1_self_coll",
    "allegro_rh_rf_2_self_coll",
    "allegro_rh_rf_3_self_coll",
    "allegro_rh_thumb_1_self_coll",
    "allegro_rh_thumb_2_self_coll",
    "allegro_rh_thumb_3_self_coll"
]

anchor_body_name = "alle_r_mf_proximal"
palm_sensor_idx = 0  
fore_arm_sensor_idx = 1 
finger_motor_num = 16 # ok 
finger_link_num = 16 # ok 
hand_qvel_num = 22 # ok 

ctrl_joint_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # coupled mechanism 
tip_body_idx = [4,7,10,13] # For contact in sensor_body_name_list
tip_end_name = "_end" 

# Action scale settings
wrist_transl_scale = 80.0 
wrist_rot_scale = 40.0 # Damping issue 
finger_action_scale = 5.0 # 5.0                  