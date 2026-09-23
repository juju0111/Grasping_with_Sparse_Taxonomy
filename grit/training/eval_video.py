"""Eval rollout video — saved as mp4/gif in the ckpt folder right after each eval during training.

Motivation: numeric metrics alone (success_rate / ratio / pinch) do not show *what* went
wrong. Keeping the scene of a few worlds at every eval period lets you skim the training
progress visually (rendered headless via EGL, without launching the viewer).

Design
  * **Uses the training env as is** (no separate env build). Only the qpos of the first
    ``n_worlds`` worlds is copied to a CPU MjData for rendering, so the cost scales with
    the number of rendered worlds regardless of nworld.
  * Calls the handler's ``snapshot_train_state`` / ``restore_train_state`` around the
    capture so **the training state is untouched** (same contract as eval).
  * Render backend is EGL (headless). On failure it silently disables itself — training
    continues.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


class EvalVideoRecorder:
    """Renders and saves a short rollout right after eval."""

    def __init__(self, handler, out_dir, *, n_worlds: int = 4, width: int = 320,
                 height: int = 240, n_frames: int = 150, fps: int = 30,
                 every: int = 1, camera: Optional[str] = None,
                 distance: float = 1.2, elevation: float = -20.0,
                 azimuth: float = 150.0):
        self.h = handler
        self.out_dir = Path(out_dir)
        self.n_worlds = max(1, min(int(n_worlds), int(handler.NWORLD)))
        self.w, self.hgt = int(width), int(height)
        self.n_frames, self.fps = int(n_frames), int(fps)
        self.every = max(1, int(every))
        self._calls = 0
        self._renderer = None
        self._data = None
        self._model = None
        self._cam = None
        self._cam_cfg = (camera, distance, elevation, azimuth)
        self.enabled = True

    # ── lazy setup (renderer created on first call; disabled on failure) ─
    def _setup(self) -> bool:
        if self._renderer is not None:
            return True
        if not self.enabled:
            return False
        try:
            os.environ.setdefault("MUJOCO_GL", "egl")
            import mujoco
            self._model = self.h.mjm
            self._data = mujoco.MjData(self._model)
            self._renderer = mujoco.Renderer(self._model, height=self.hgt, width=self.w)
            cam_name, dist, elev, azim = self._cam_cfg
            self._cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self._cam)
            if cam_name:
                cid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                if cid >= 0:
                    self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self._cam.fixedcamid = cid
                    return True
            # free camera — looking between the two hands
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._cam.distance, self._cam.elevation, self._cam.azimuth = dist, elev, azim
            env = self.h.sampled_env
            self._cam.lookat[:] = [0.0, float(getattr(env, "y_offset", 0.0)) / 2, 1.0]
            return True
        except Exception as e:                       # environment cannot render → silently off
            print(f"[eval-video] renderer init failed → disabled ({type(e).__name__}: {e})")
            self.enabled = False
            return False

    # ── capture ─────────────────────────────────────────────────────────
    def capture(self, policy, step: int, tag: str = "") -> Optional[str]:
        """Render a short rollout and save it as ``<out_dir>/rollout_<step>.mp4``.

        Returns the saved path (``None`` when skipped or failed).
        """
        self._calls += 1
        if not self.enabled or (self._calls - 1) % self.every:
            return None
        if not self._setup():
            return None
        import mujoco
        import torch

        h = self.h
        h.snapshot_train_state()
        frames = []
        try:
            obs = h.reset()
            qadr = None
            for _ in range(self.n_frames):
                with torch.no_grad():
                    act = policy(obs)               # deterministic (tanh(loc))
                obs = h.step(act)[0]
                qpos = h.d.qpos.numpy()
                if qadr is None:
                    qadr = min(qpos.shape[1], self._model.nq)
                tiles = []
                for w in range(self.n_worlds):
                    self._data.qpos[:qadr] = qpos[w, :qadr]
                    mujoco.mj_forward(self._model, self._data)
                    self._renderer.update_scene(self._data, self._cam)
                    tiles.append(self._renderer.render())
                frames.append(_grid(tiles))
        except Exception as e:
            print(f"[eval-video] capture failed ({type(e).__name__}: {e}) → skipping this round")
            frames = []
        finally:
            h.restore_train_state()

        if not frames:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"rollout_{tag+'_' if tag else ''}{step}"
        try:
            import numpy as _np
            import imageio.v2 as imageio
            path = self.out_dir / f"{stem}.mp4"
            errs = []
            # pyav first: **in-process** encoding. imageio's ffmpeg backend forks a
            #    subprocess, and with a large training process the fork dies with ENOMEM
            #    ("Cannot allocate memory"). pyav is a libav binding and writes from the
            #    same process without forking.
            for how in ("pyav", "ffmpeg"):
                try:
                    if how == "pyav":
                        import imageio.v3 as iio3
                        iio3.imwrite(path, _np.asarray(frames),
                                     plugin="pyav", codec="libx264", fps=self.fps)
                    else:
                        imageio.mimsave(path, frames, fps=self.fps,
                                        macro_block_size=1)
                    break
                except Exception as e_mp4:
                    errs.append(f"{how}: {type(e_mp4).__name__} "
                                f"{str(e_mp4)[:80]}")
            else:                                    # both failed → gif fallback
                print(f"[eval-video] mp4 failed → falling back to gif ({' | '.join(errs)})")
                path = self.out_dir / f"{stem}.gif"
                imageio.mimsave(path, frames, duration=1.0 / self.fps)
            print(f"[eval-video] {path.name}  ({len(frames)} frame, "
                  f"{self.n_worlds} world)")
            return str(path)
        except Exception as e:
            print(f"[eval-video] save failed ({type(e).__name__}: {e})")
            return None


def _grid(tiles):
    """Tile the world frames into a near-square grid."""
    n = len(tiles)
    if n == 1:
        return tiles[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    hgt, wid, c = tiles[0].shape
    canvas = np.zeros((rows * hgt, cols * wid, c), dtype=tiles[0].dtype)
    for i, t in enumerate(tiles):
        r, q = divmod(i, cols)
        canvas[r * hgt:(r + 1) * hgt, q * wid:(q + 1) * wid] = t
    return canvas
