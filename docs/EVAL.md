# Evaluating and visualising

## Interactive viewer (`scripts/evaluate.py`)

```bash
python scripts/evaluate.py --save-dir pretrained/<run> --hand <hand> --env grasping_teacher --nworld 16
# a run you trained (looked up under output/checkpoints/<hand>_<env><suffix>)
python scripts/evaluate.py --hand tesollo --env grasping_teacher --name-suffix _my_run --nworld 32
```

Opens a GLFW window with every world side by side (grid spacing 1.5 m), rolls the policy
out deterministically, recycles each world at the end of its episode window and shows a
green ball above worlds that are currently in a success state. `--env` must be the name
stored in the checkpoint file name (`grasping_teacher` / `grasping_student`). `--max-ticks N` quits after N steps (for scripts / CI).

Keys: `SPACE` pause · `A` policy↔random · `D` deterministic↔stochastic · `R` reset ·
`E` auto-reset · `H` ghost target hand · `V` contact colouring · `B`/`N`/`O` object / table /
self contact markers · `K` contact-force arrows · `J` task markers · `P` point cloud · `M`
overlay · `T` transparency · `0`–`4` geom groups · `Q`/`ESC` quit. Double-click a body to
select it; drag the gizmo (Ctrl+LMB rotate, Ctrl+RMB translate, Alt+LMB grasp interpolation).

Headless machines: `xvfb-run -a python scripts/evaluate.py …` (software GL).

## Success rate (`scripts/eval_success.py`)

```bash
python scripts/eval_success.py --save-dir pretrained/<run> --hand <hand> --env grasping_teacher --nworld 1024 --json out.json
```

Deterministic rollout of one full episode window on `--nworld` worlds (the run's `obj_idxs`
objects spread over the worlds, `EVAL_RESET_SEED` fixed). Prints `success_rate`,
`success_rate_lift`, `success_rate_strict`, mean return and the first-failure reasons.
Numbers are reproducible to about ±0.3 pp (contact solver nondeterminism on the GPU).

## Videos (`scripts/render_rollout.py`, `scripts/make_hero_gif.py`)

```bash
MUJOCO_GL=glfw python scripts/render_rollout.py --save-dir pretrained/<run> --hand <hand> --env grasping_teacher \
    --nworld 12 --pick 6 --cols 3 --out docs/media/<hand>.mp4 --gif --gif-width 720 --save-frames <hand>_hero.npy
python scripts/make_hero_gif.py --frames tesollo=tesollo_hero.npy shadow=shadow_hero.npy … --cols 3 --out docs/media/hero_palm_view.gif   # hero tiles: --hero-taxonomy <name> on render_rollout.py picks the taxonomy per hand
```

Offscreen rendering (invisible GLFW window on the current display, or `MUJOCO_GL=egl`),
one tile per world with *hand · grasp taxonomy · object* burned in, green frame = success,
camera follows the object. Defaults: 480×360 tiles, the scene exactly as trained (`--polish`
adds brighter lighting, shadows and an `--object-color`), the evaluation protocol (no
early-success termination, strict success judged at the end of the window), a **palm-side
camera** (`--palm-view`: a first pass with the same eval seed plays the whole window without
rendering, recording each world's outcome and its wrist orientation in the hold phase; the
render pass then puts the camera squarely in front of the palm (index to little finger in a
row across the frame), slightly tilted towards the fingertips by `--tip-tilt`, at hand height
(elevation clamped to `--el-min`/`--el-max`, never looking up from the table), and searches
`--az-search` azimuths around that direction for the best trade-off between the palm facing the
camera and the fingertips staying between the camera and the object; per taxonomy only the
worlds whose palm faces the camera best are rendered and picked, and bulky objects listed in
`--avoid-objects` rank last) and a **wrist-rise cap**
(`--wrist-rise-cap 0.12`): the stage rule that lifts the wrist after `LIFT_STEP` stops and
holds once the wrist has risen 12 cm, so a slipped object never sends the wrist to the top
of the lift window (handler knob `LIFT_WRIST_RISE_CAP_M`, off during training). `--pick N`
rolls out more worlds than it shows and prefers successful ones; `--taxonomy-out DIR`
additionally stores one clip per grasp taxonomy seen in the rollout.

```bash
# taxonomy × hand grid (rows = taxonomies common to the listed hands) and one hand's taxonomies
python scripts/make_taxonomy_grid.py --clips clips --hands tesollo shadow allegro --out docs/media/taxonomy_grid.gif
python scripts/make_taxonomy_grid.py --clips clips --per-hand tesollo --cols 6 --out docs/media/taxonomy_tesollo.mp4
# per-clip mp4/gif + docs/gallery/<hand>.md (collapsible per taxonomy) + docs/gallery/index.html (drop-down picker)
python scripts/make_gallery.py --clips clips --out docs/media/clips --pages docs/gallery
```
