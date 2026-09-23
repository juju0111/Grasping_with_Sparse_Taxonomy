# INSTALL

Tested on Ubuntu 22.04 / kernel 6.17, NVIDIA RTX 5090 (driver 570), Python 3.12.
An NVIDIA GPU with CUDA 12.x is **required** — every environment step runs as
MuJoCo Warp GPU kernels. CPU-only machines can import the package but cannot
train or evaluate.

## 1. Conda environment

```bash
conda create -n grit_share python=3.12 -y
conda activate grit_share
pip install --upgrade pip setuptools wheel
```

## 2. PyTorch (CUDA build first, so pip does not pull a CPU wheel later)

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # → 2.10.0+cu128 True
```

Older GPUs (RTX 30/40) also work with the `cu121`/`cu124` index; only
`torch.cuda.is_available() == True` matters.

## 3. Simulation stack + everything else

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the exact combination the bundled checkpoints were
produced with: `mujoco==3.11.0`, `warp-lang==1.16.0` and `mujoco-warp` at git
commit `dbc52e3e` (installed straight from GitHub — needs `git`). Other
`mujoco`/`mujoco-warp` pairs will usually work but are not validated; keep the
two at the same major.minor.

Verify:

```bash
python -c "import warp as wp; wp.init(); import mujoco, mujoco_warp; print('warp', wp.__version__, '| mujoco', mujoco.__version__)"
python -c "import sys; sys.path.insert(0, '.'); import grit.training; from grit.training.rl_env_base import list_rl_envs; print(list_rl_envs())"
```

The first `import warp` / first env build JIT-compiles CUDA kernels; expect
30–90 s of compile time once (cached under `~/.cache/warp`).

## 4. Running

All entry points live in `scripts/` and find the project root themselves, so
they can be launched from any working directory:

```bash
python scripts/evaluate.py --save-dir pretrained/tesollo_grasping_teacher --hand tesollo --env grasping_teacher --nworld 16
python scripts/train.py   -c grasping_policy_teacher
python scripts/distill.py -c distill_teacher_to_student
```

### Headless servers

Training and distillation never open a window. `evaluate.py` needs an X
display with OpenGL (GLFW). On a machine without one, either forward X, or run
it under Xvfb (software GL, slow but functional):

```bash
xvfb-run -a -s "-screen 0 1600x1200x24" python scripts/evaluate.py ...
```

`GRIT_FORCE_HEADLESS=1` forces the GUI stubs on even if `DISPLAY` is set
(useful when `DISPLAY` points at a dead X server).

### Weights & Biases

Logging is **off** in the shipped configs (`wandb.enabled: false`). Turn it on
with `--overrides wandb.enabled=true wandb.project=<proj> wandb.entity=<you>`
after `wandb login`.

## 5. Common problems

| Symptom | Fix |
|---|---|
| `KeyError: 'DISPLAY'` / GLFW init failure during training | set `GRIT_FORCE_HEADLESS=1` |
| `batch_size*num_minibatch=… cannot exceed NWORLD*unroll_length` | you lowered `nworld`; lower `training.batch_size` / `training.num_minibatch` too |
| CUDA OOM at env build | lower `nworld` (2048 default ≈ 6–8 GB) or `n_sub_env` |
| `SAVE_DIR not found` in evaluate.py | pass `--save-dir <dir>` or the exact `--name-suffix` the training run printed |
| `mujoco_warp` API errors | you are not on the pinned commit — reinstall from `requirements.txt` |
