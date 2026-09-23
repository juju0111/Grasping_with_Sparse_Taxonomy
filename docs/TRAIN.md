# Training the teacher

```bash
python scripts/train.py -c grasping_policy_teacher                     # bundled recipe
python scripts/train.py -c grasping_policy_async_best_w_ground_force \
    --overrides 'using_hand_name_list=[tesollo]' output.run_tag=my_run    # any hand
```

`scripts/train.py` is a single-file PPO driver:

1. `build_orchestrator` composes `config/training/<name>.yaml` (Hydra defaults chain:
   `base` → `grasping_policy_async_best` → `…_w_ground_force` → `grasping_policy_teacher`)
   and builds the scene: `n_sub_env` object variants × `nworld` parallel worlds on the GPU.
2. `build_rl_env_and_handlers` instantiates the registered env and applies every `handler:`
   knob to the handler (`apply_handler_knobs`; unknown keys warn, missing keys keep defaults).
3. `build_policies_and_optimizers` — `policy.name: bps_encoder_mlp` (separate encoder for the
   BPS tail, asymmetric critic when `training.async_ppo: true`), observation normaliser.
4. The loop: rollout `unroll_length` steps → GAE → `n_ppo_epochs` × minibatches (adaptive LR
   on `desired_kl`), eval every `total_new_steps / n_evals` steps, checkpoint after each eval,
   curriculum hooks (`on_train_progress` / `on_eval_metrics`).

Outputs land in `output/checkpoints/<hand>_<env>[_<run_tag>]/`:
`config.yaml` (fully resolved snapshot), `*_<seen_steps>.pt`, `*_metrics.jsonl`,
`*_train_log.txt`.

## Knobs you will actually touch

| key | default | note |
|---|---|---|
| `nworld` | 2048 | parallel worlds (≈ 6–8 GB VRAM); keep `training.batch_size × num_minibatch ≤ nworld × unroll_length` |
| `n_sub_env`, `obj_idxs` | 30, list | object variants; `obj_idxs: null` samples randomly from the 83 objects and `training.resample_objects_every` re-draws them |
| `training.total_new_steps` | 1.5e8 | ≈ 4 h on an RTX 5090 |
| `training.reward_norm.enabled` | true | return-based reward normalisation — keep on, the λ targets are tuned with it |
| `policy.obs_norm` | true | keep on (large drop without) |
| `handler.termination.MAX_EPISODE_STEPS` | 270 | must cover `retry_budget_steps()` (printed at start) |
| `handler.curriculum.reward` | | `{W_x: [final, start_step, end_step]}` linear ramps |
| `handler.reward_lagrangian.LAG_CONSTRAINTS` | | `{W_x: [metric, target, W_max]}` |
| `wandb.enabled` | false | `--overrides wandb.enabled=true wandb.project=… wandb.entity=…` |
| `resume.enabled` | false | continue the same run (forks into `_resume`) |
| `--load-from DIR` | | weight-only warm start from another run |

## Reading the log

Each iteration prints per-term episode returns (`reward_term/*`), the λ updates
(`[lagrangian] W_TABLE_CONTACT=…(λ+…)`), and every eval prints

```
EVAL @ seen=  60,096,512  mean_return=+1104  success_rate= 78.5%  lift= 88.1%  strict= 78.5%  contact_f1=0.71 …
```

`success_rate` = strict success (fingers on the taxonomy band + object in the lift band at
the end of the window), `lift` = object ≥ `LIFT_SUCCESS_THRESH` above spawn at the end.
Compare checkpoints with `scripts/eval_success.py` (deterministic, 1024 worlds).

## Time budget (RTX 5090, Tesollo, 2048 worlds)

| | |
|---|---|
| env build (30 variants) | ~1 min |
| control steps / s | ~20 k (≈ 160 k physics steps / s) |
| 150 M steps | 4.1 h |
| first useful lifts | ~30 M steps; success > 80 % from ~90 M |
