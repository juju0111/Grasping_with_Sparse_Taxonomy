# Teacher → student distillation

The teacher sees things a real robot cannot: contact sensors, the object pose at every
step, the episode clock, the full object shape. `scripts/distill.py` trains a **deployable
student** with the `grasping_student` env preset:

| teacher (`grasping`) | student (`grasping_student`) |
|---|---|
| contact flags + impulses, object touch force | zeroed (`student_drop_contact`) |
| episode clock `ep_step_norm` | zeroed (`student_drop_ep_step`); stage from an obs-driven state machine (`student_stage_from_obs`) |
| live object pose / velocity | tracked only during approach (`student_track_approach_only`); blanked after the grasp (`student_blind_after_grasp`), object velocity always blanked |
| full live BPS shape feature | one **frozen partial view** captured at episode start from a random camera pose (hidden-point removal), with depth-sensor noise / dropout |
| — | history of joint angles + torque proxy (+ last action) × K (`student_hist_k`), with update-skip / noise / 1-tick action-delay domain randomisation |
| privileged critic | critic (hybrid RL) = teacher obs + velocities (`collect_critic_obs`) |

All of these are `cfg.Training.student_*` / `pv_*` keys — the yaml `student.training_overrides`
block. The observation width follows automatically (`obs_dim` = core + history + BPS).

## Pipeline

```bash
python scripts/distill.py -c distill_teacher_to_student
```

1. **Teacher resolve** — `teacher.save_dir` (or `teacher.{hand,task,env_name,name_suffix}`
   under `output/checkpoints`) → config snapshot + latest checkpoint. The critic input width is
   read from the checkpoint.
2. **Student env** — the teacher snapshot with `env_name := student.env_name` and the
   `student.training_overrides` injected into `cfg.Training`, rebuilt at `student.nworld`.
   `stage_autocal` runs one teacher episode to calibrate the obs-driven stage thresholds.
3. **DAgger** (`distill.n_iters`) — per-episode driver mix (teacher / student / blend, `beta_*`
   annealing), GPU ring buffer, loss = MSE + Gaussian KL to the teacher action distribution,
   plus auxiliary heads: object-pose reconstruction (`aux_objpd_coef`) and the **re-grasp
   head** (`aux_regrasp_coef`, BCE on the env's "lost the object" label).
4. **Hybrid RL fine-tune** (`rl_finetune.enabled`) — PPO on the true reward with a KL anchor to
   the teacher (`bc_kl_start → bc_kl_end`), asymmetric critic, regrasp head frozen.
5. **Save** — `output/checkpoints/<hand>_grasping_student_<run_tag>/` (DAgger) and
   `…_<run_tag>_rl/` (hybrid), each with `config.yaml`, the policy `.pt` and `regrasp_head.pt`.
   Both load in `evaluate.py` / `eval_success.py` as usual.

## Re-grasp head at deployment

The student cannot see the object once it is grasped, so a small head predicts "the object
slipped" from proprioception (`joint_qpos`, `torque_proxy`, history). Inside the env,
`regrasp_pred_torch` (written by the policy wrapper, see `RegraspPredActor` in `distill.py`)
rewinds the obs-driven stage machine to *approach* after `REGRASP_PRED_CONSEC` consecutive
positive predictions (`REGRASP_STAGE_RESET_SOURCE: pred`), so the student re-approaches
without any external signal. During training the ground-truth retry (`gt`) drives it.

## Result to expect

On the Mimic P0.50 hand (not part of this release) the recipe gave, on 1024 worlds:
teacher 0.866 strict / 0.925 lift, DAgger student 0.850 / 0.899, DAgger + hybrid RL student
0.863 / 0.909 — i.e. the deployable student loses ~1 pp against its privileged teacher. No
student checkpoint is bundled; `distill_teacher_to_student.yaml` reproduces the recipe from
the bundled Tesollo teacher (≈ 3 h on an RTX 5090).

## Snapshot keys that must travel with a student checkpoint

The student's stage machine and blind gate are configured from `cfg.Training`, so the
snapshot must carry them: `stage_hc_dist_m` / `stage_near_ticks` / `stage_lift_frac`
(written by `distill.py` after `stage_autocal`) and `student_blind_near_m` (0 = blind once
the obs-driven stage leaves *approach*; > 0 = latch when the hand centre comes within that
distance). A checkpoint evaluated with different values scores very differently — always
evaluate from the run's own `config.yaml`.
