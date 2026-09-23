# The grasping environment

Three modules, one level of inheritance:

| file | class | what it adds |
|---|---|---|
| [`grasping_core.py`](../grit/training/rl_envs/grasping_core.py) | `GraspingCore(SubEnvHandler)` | the task: obs blocks, reward terms + mixer, curriculum, stage clock, termination, re-grasp retry / early success, BPS shape tail |
| [`grasping_teacher.py`](../grit/training/rl_envs/grasping_teacher.py) | `GraspingTeacher(GraspingCore)` → **`grasping_teacher`** | privileged critic obs (+20 GT floats), Lagrangian dual ascent on the penalty weights |
| [`grasping_student.py`](../grit/training/rl_envs/grasping_student.py) | `GraspingStudent(GraspingCore)` → **`grasping_student`** | frozen partial-view BPS, contact / clock / object masking, obs-driven stage machine, history block, blind-after-grasp, re-grasp label & prediction |
| [`grasping_kernels/`](../grit/training/rl_envs/grasping_kernels) | — | the Warp kernels, one module per topic (reward terms, mixer, done, contact, obs, fd_vel, retry, student view, …) |

Teacher and student override a handful of named hooks of the core
(`_setup_obs_buffers`, `_before_core_obs`, `_collect_obs_tail`, `_apply_ep_norm_horizon`,
`_collect_obs_kernel`, `_collect_success_kernel`, `_post_cond_update_hook`, `step`,
`on_train_progress`), each with one `super()` call, so opening either file shows that
policy's whole per-step flow. Everything below the Python surface is per-world GPU code.

```
  cfg.handler ─▶ GraspingCore        obs blocks · reward → mixer · curriculum · stage clock
                  │                   termination · re-grasp retry / early success · BPS tail
        ┌─────────┴──────────┐
  GraspingTeacher      GraspingStudent
  + privileged critic  + frozen partial-view BPS · masking · history
  + Lagrangian λ       + obs-driven stage machine · blind-after-grasp · re-grasp label
  = "grasping_teacher" = "grasping_student"
```

The presets are two registered env names (plus the aliases
`grasping_teacher` / `grasping_student` that the
bundled checkpoints' config snapshots use). Select with `env_name:` in the yaml or
`--env` on the CLI.

## Episode

| phase | steps (default) | what happens |
|---|---|---|
| approach | `0 … LIFT_STEP` (90) | wrist Δpose + finger targets are free; target point on the object + taxonomy finger pose shape the reward |
| lift | `LIFT_STEP … HOLD_STEP` (140) | scripted wrist-z lift of the mocap target until the object is `LIFT_TARGET_M` up (`STAGE_ACTION_JAX_MODE`: wrist xy / rotation frozen, fingers slowed); `LIFT_WRIST_RISE_CAP_M` (opt-in) also stops it once the wrist itself rose that far |
| hold | `HOLD_STEP … MAX_EPISODE_STEPS` (270) | wrist target frozen; the fingers keep the object at `LIFT_TARGET_M` (0.1 m) above its spawn height |

* **Re-grasp retry** (`retry.*`): if the object is dropped in lift/hold (no contact & not lifted
  for `RETRY_DROP_CONTACT_STEPS`) or never left the table by `HOLD_STEP`, the world's stage clock
  is rewound to `LIFT_STEP − RETRY_APPROACH_STEPS` so the policy approaches again — at most
  `RETRY_MAX` times and only if the remaining budget fits a full cycle. Wall time is tracked
  separately (`ep_elapsed`) and is what triggers the timeout.
* **Early success** (train only): strict (`bonus_streak ≥ SUCCESS_STREAK_MIN`, fingers on the
  target band with contact) or lift-hold (`SUCCESS_EARLY_LIFT_HOLD_STEPS` consecutive lifted
  steps in hold) ends the episode as a success (`done_reason = 1`, bootstrapped like a
  time limit). Evaluation always plays the whole window.
* **Termination** (`termination.*`): object below the table (`DONE_OBJ_Z_MIN`), xy drift, wrist
  flipped, crushing force (`DONE_OBJ_TOUCH`), object thrown, hand far from object, hold lost,
  table crush, timeout. `SubEnvHandler.REASON_NAME` maps the codes.

## Observation (actor)

Built block by block in `_collect_obs_core` (the order is the layout; `obs_term_layout`
returns `(name, start, end)`):

| term | width | meaning |
|---|---|---|
| `site_pcd_err` | 3·(n_sites−1) | each fingertip site → nearest object surface point (wrist frame) |
| `hand_center_pcd_err` | 3 | hand centre → nearest object point |
| `rot6d` | 6 | hand-centre rotation vs the sampled target rotation |
| `site_z_above_table` | n_sites | fingertip heights above the support |
| `obj_pose_diff` | 6 | object xyz + rpy relative to its spawn pose |
| `joint_qpos`, `qpos_tax_err` | n_ctrl each | finger angles; taxonomy target − angle |
| `finger_link_mask` | n_sites | which fingers the sampled grasp taxonomy uses |
| `per_slot_contact` | 4·n_slots | object / self / table contact flags + impulses |
| `obj_touch_force`, `obj_ground_force` | 1 + 1 | summed object touch force; object↔table force |
| `torque_proxy` | n_ctrl | commanded − measured joint angle (stall proxy) |
| `fd_vel` | 12 | finite-difference wrist + object velocities |
| `proj_gravity`, `lift_target_vec` | 3 + 3 | gravity in the wrist frame; vector to the lift goal |
| `ep_stage` | 4 | episode clock (single-attempt horizon) + stage one-hot |
| `student_hist` | K·(2·n_ctrl[+act]) | **student only**: history of joint_qpos / torque_proxy (/ action) |
| `bps_feature` / `partial_bps_feature` | 1024 | exp(−γ·dist) from 1024 hand-centre basis points to the object surface; the student sees a frozen partial view (HPR from a random camera) |

The **teacher critic** additionally gets 20 privileged floats (true object rotation, relative
velocities, vector to the target point, gravity, lift height) and optionally the curriculum
pressure. The **student** masks contact / clock / object channels according to the
`student_*` keys in `cfg.Training` (see `docs/DISTILL.md`).

## Action

`[wrist Δxyz (3) | wrist Δrot (3) | (3 reserved) | finger position targets (n_ctrl)]`, all in
`[−1, 1]`, scaled by `rl_env.xyz_scale / rot_scale / finger_scale` per control step
(`sim_nstep` physics substeps of `sim_dt`). Applied by `WarpActionApplier`
(`grit/util/warp_action.py`), which also hosts the stage-gated wrist rules.

## Reward

Every term is computed by its own weight-free kernel and combined in the mixer
(`_object_grasping_reward_mixer_kernel`): `W_*` weights and `K_*` kernel widths are plain
class attributes, so they can be set from yaml (`handler:`), ramped by the curriculum, or
driven by the Lagrangian duals without recompiling. Groups (`CONFIG_KEYS_GROUPS`):

* `reward_tracking` — approach (hand centre → target point), finger (fingertips → object),
  face direction, taxonomy mimic, wrist direction, lift, velocity damping, object motion.
* `reward_contact` — contact ratio / direction / positive-negative contact, force closure,
  semantic-contact recall / precision / F1 (which finger parts touch), coverage bonus.
* `reward_penalty` — self-collision, hand↔table, object↔table force, finger/wrist height,
  joint limits, action cost & smoothness, gentle-touch.
* `success` — streak bonus, lift-success bonus, coverage coupling.
* `reward_post` — `REWARD_DT` scaling and clipping.
* `reward_lagrangian` — `LAG_CONSTRAINTS: {W_x: [metric, target, W_max]}`: each listed
  penalty weight becomes `base + λ`, with `λ` raised while the per-step metric exceeds its
  target and lowered otherwise (dual ascent, EMA-smoothed, warm-up gated).
* `curriculum` — lift assist decay, anti-gravity decay, action-scale decay, reward gate
  (weights ramp only once `success_rate_lift` passes a threshold).

## Extending

Add a new observation / reward stage to `GraspingCore` if both policies need it, or
override one hook in the teacher / student file. Keep it one level deep — the previous
nine-layer inheritance chain of this task is what this layout replaced.
