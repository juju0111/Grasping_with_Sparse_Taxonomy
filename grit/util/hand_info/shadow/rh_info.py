import numpy as np  

rh_mocap_name = "shadow_rh:mocap"
rh_wrist_base_name = "shadow_rh_palm"
rh_palm_site_name = "shadow_rh_palm_touch"
# Keypoint names used to locate the hand center
rh_thumb_link1_name = "shadow_rh_thdistal" 
rh_th_tip_name = ["shadow_rh_th_end", "shadow_rh_thdistal_touch"]
rh_ff_tip_name = ["shadow_rh_ff_end", "shadow_rh_ffdistal_touch"]
rh_mf_tip_name = ["shadow_rh_mf_end", "shadow_rh_mfdistal_touch"]   
rh_rf_tip_name = ["shadow_rh_rf_end", "shadow_rh_rfdistal_touch"]    
rh_lf_tip_name = ["shadow_rh_lf_end", "shadow_rh_lfdistal_touch"]

rh_hand_center = np.array([[ 0.772008, -0.      , -0.635612,  0.050503],
                            [-0.000202,  1.      , -0.000246,  0.030462],
                            [ 0.635612,  0.000319,  0.772008,  0.096707],
                            [ 0.      ,  0.      ,  0.      ,  1.      ]])

rh_body_parts = ['shadow_rh_palm',
        'shadow_rh_ffknuckle', 'shadow_rh_ffproximal', 'shadow_rh_ffmiddle', 'shadow_rh_ffdistal', 'shadow_rh_ff_end',
        'shadow_rh_mfknuckle', 'shadow_rh_mfproximal', 'shadow_rh_mfmiddle', 'shadow_rh_mfdistal', 'shadow_rh_mf_end',
        'shadow_rh_rfknuckle', 'shadow_rh_rfproximal', 'shadow_rh_rfmiddle', 'shadow_rh_rfdistal', 'shadow_rh_rf_end',   
        'shadow_rh_lfmetacarpal', 'shadow_rh_lfknuckle', 'shadow_rh_lfproximal', 'shadow_rh_lfmiddle', 'shadow_rh_lfdistal', 'shadow_rh_lf_end',  
        'shadow_rh_thbase', 'shadow_rh_thproximal', 'shadow_rh_thhub', 'shadow_rh_thmiddle', 'shadow_rh_thdistal', 'shadow_rh_th_end',   
        ]
rh_af_parts   = ['shadow_rh_palm',
                'shadow_rh_ffproximal', 'shadow_rh_ffmiddle', 'shadow_rh_ff_end',
                'shadow_rh_mfproximal', 'shadow_rh_mfmiddle', 'shadow_rh_mf_end',  
                'shadow_rh_rfproximal', 'shadow_rh_rfmiddle', 'shadow_rh_rf_end',   
                'shadow_rh_lfproximal', 'shadow_rh_lfmiddle', 'shadow_rh_lf_end',  
                'shadow_rh_thproximal', 'shadow_rh_thmiddle', 'shadow_rh_th_end',   
                ]

# Touch sensor names added
fore_arm_touch_sensor = "shadow_r_arm_part_touch"
hand_touch_sensors = [
    "shadow_rh_palm_body_touch",     # palm

    "shadow_rh_ffproximal_body_touch",
    "shadow_rh_ffmiddle_body_touch",
    "shadow_rh_ffdistal_body_touch",

    "shadow_rh_mfproximal_body_touch",
    "shadow_rh_mfmiddle_body_touch",
    "shadow_rh_mfdistal_body_touch",

    "shadow_rh_rfproximal_body_touch",
    "shadow_rh_rfmiddle_body_touch",
    "shadow_rh_rfdistal_body_touch",

    "shadow_rh_lfproximal_body_touch",
    "shadow_rh_lfmiddle_body_touch",
    "shadow_rh_lfdistal_body_touch",

    "shadow_rh_thproximal_body_touch",
    "shadow_rh_thmiddle_body_touch",
    "shadow_rh_thdistal_body_touch",
]

hand_contact_sensors_for_object = [
    "shadow_rh_palm_body_obj_contact",
    "shadow_rh_ffproximal_body_obj_contact",
    "shadow_rh_ffmiddle_body_obj_contact",
    "shadow_rh_ffdistal_body_obj_contact",
    "shadow_rh_mfproximal_body_obj_contact",
    "shadow_rh_mfmiddle_body_obj_contact",
    "shadow_rh_mfdistal_body_obj_contact",
    "shadow_rh_rfproximal_body_obj_contact",
    "shadow_rh_rfmiddle_body_obj_contact",
    "shadow_rh_rfdistal_body_obj_contact",
    "shadow_rh_lfproximal_body_obj_contact",
    "shadow_rh_lfmiddle_body_obj_contact",
    "shadow_rh_lfdistal_body_obj_contact",
    "shadow_rh_thproximal_body_obj_contact",
    "shadow_rh_thmiddle_body_obj_contact",
    "shadow_rh_thdistal_body_obj_contact",
]

hand_contact_sensors_for_table = [
    "shadow_rh_palm_table_contact",
    "shadow_rh_ffproximal_table_contact",
    "shadow_rh_ffmiddle_table_contact",
    "shadow_rh_ffdistal_table_contact",
    "shadow_rh_mfproximal_table_contact",
    "shadow_rh_mfmiddle_table_contact",
    "shadow_rh_mfdistal_table_contact",
    "shadow_rh_rfproximal_table_contact",
    "shadow_rh_rfmiddle_table_contact",
    "shadow_rh_rfdistal_table_contact",
    "shadow_rh_lfproximal_table_contact",
    "shadow_rh_lfmiddle_table_contact",
    "shadow_rh_lfdistal_table_contact",
    "shadow_rh_thproximal_table_contact",
    "shadow_rh_thmiddle_table_contact",
    "shadow_rh_thdistal_table_contact",
]

hand_contact_sensors = [
    "shadow_rh_palm_self_coll",
    "shadow_rh_ffproximal_self_coll",
    "shadow_rh_ffmiddle_self_coll",
    "shadow_rh_ffdistal_self_coll",
    "shadow_rh_mfproximal_self_coll",
    "shadow_rh_mfmiddle_self_coll",
    "shadow_rh_mfdistal_self_coll",
    "shadow_rh_rfproximal_self_coll",
    "shadow_rh_rfmiddle_self_coll",
    "shadow_rh_rfdistal_self_coll",
    "shadow_rh_lfproximal_self_coll",
    "shadow_rh_lfmiddle_self_coll",
    "shadow_rh_lfdistal_self_coll",
    "shadow_rh_thproximal_self_coll",
    "shadow_rh_thmiddle_self_coll",
    "shadow_rh_thdistal_self_coll",
]

anchor_body_name = "shadow_rh_mfproximal"
palm_sensor_idx = 0  
fore_arm_sensor_idx = 1 
finger_motor_num = 18 
finger_link_num = 22 
hand_qvel_num = 24 

# coupled mechanism — passive *J1 joints sit at qpos-block 3/7/11/16, so the
# actuator→joint index is NOT identity. This is the model-driven mapping
# (jnt_qposadr−7 per actuator); the runtime override in SingleHandSubEnv
# (_apply_model_driven_ctrl_joint_idx) recomputes/enforces this from the model.
ctrl_joint_idx = [0,1,2,4,5,6,8,9,10,12,13,14,15,17,18,19,20,21]
tip_body_idx = [4,7,10,13,16] # For contact in sensor_body_name_list
tip_end_name = "_end" 

# Action scale settings
wrist_transl_scale = 80.0 
wrist_rot_scale = 40.0 # Damping issue 
finger_action_scale = 5.0 # 5.0  