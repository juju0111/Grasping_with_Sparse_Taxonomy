"""Per-world condition meta-dictionary for warp parallel RL envs.

Holds per-world buffers (shape ``(NWORLD, *field_shape)``) used to drive
RL conditioning — e.g. per-world target wrist pose for hand-pose-tracking
tasks. Each registered field is a GPU-resident ``wp.array`` (float) plus
an optional sampler that fills it with random values.

Typical use:

    cond = WarpPerWorldCondition(NWORLD=env.NWORLD,
                                 device=env.d.qpos.device)
    cond.register('target_wrist_pos', shape=(3,))
    cond.register_uniform('target_wrist_pos',
                          low=[-0.2, -0.2, 0.1],
                          high=[ 0.2,  0.2, 0.5])

    cond.reset()                                  # call inside env.reset()
    cond.sample('target_wrist_pos')               # any-time re-sample
    cond.set('target_wrist_pos', np_value)        # explicit override
    cond.set_world_mask('target_wrist_pos',
                        done_mask, sampled_value) # only the done worlds

    target = cond.get('target_wrist_pos')         # wp.array (NWORLD, 3)

For the common hand-pose-tracking setup, see ``register_hand_pose_target``
which wires up ``target_wrist_pos / target_wrist_quat / target_qpos`` from
an ``SingleHandSubEnv`` + ``HandUtils`` pair in one call.

Designed for CUDA-graph compatibility: all writes go through pre-allocated
CPU staging via ``wp.copy()`` — no fresh ``wp.array`` allocation after the
captured graph exists.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import warp as wp


# ──────────────────────────────────────────────────────────────────────────
# GPU sampler kernels (sync-free per-world write, optionally masked)
# ──────────────────────────────────────────────────────────────────────────


@wp.kernel
def _cond_uniform_kernel(
    seed_arr:     wp.array(dtype=int),                  # type: ignore  (1,)
    seed_offset:  int,
    n_dim:        int,
    low:          wp.array(dtype=float, ndim=1),        # type: ignore  (n_dim,)
    high:         wp.array(dtype=float, ndim=1),        # type: ignore  (n_dim,)
    field:        wp.array(dtype=float, ndim=2),        # type: ignore  (NWORLD, n_dim)
    apply_mask:   wp.array(dtype=int, ndim=1),          # type: ignore  (NWORLD,) 1=apply
    apply_to_all: int,
):
    """Per-world uniform sample in-place into ``field``. When ``apply_to_all
    == 0`` the kernel only writes worlds with ``apply_mask[w] == 1`` —
    used for masked re-sampling on per-world reset without any CPU sync."""
    w = wp.tid()
    if apply_to_all == 0 and apply_mask[w] == 0:
        return
    state = wp.rand_init(seed_arr[0] + seed_offset, w)
    for i in range(n_dim):
        field[w, i] = wp.randf(state) * (high[i] - low[i]) + low[i]


@wp.kernel
def _cond_so3_kernel(
    seed_arr:     wp.array(dtype=int),                  # type: ignore  (1,)
    seed_offset:  int,
    field:        wp.array(dtype=float, ndim=2),        # type: ignore  (NWORLD, 4)  wxyz
    apply_mask:   wp.array(dtype=int, ndim=1),          # type: ignore  (NWORLD,)
    apply_to_all: int,
):
    """Sample uniform on SO(3) per world (4D Gaussian → normalize, w >= 0)."""
    w = wp.tid()
    if apply_to_all == 0 and apply_mask[w] == 0:
        return
    state = wp.rand_init(seed_arr[0] + seed_offset, w)
    qw = wp.randn(state)
    qx = wp.randn(state)
    qy = wp.randn(state)
    qz = wp.randn(state)
    n  = wp.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1.0e-9:
        qw = 1.0; qx = 0.0; qy = 0.0; qz = 0.0
    else:
        inv = 1.0 / n
        qw  = qw * inv; qx = qx * inv; qy = qy * inv; qz = qz * inv
    # Enforce w >= 0 (q and -q encode the same rotation).
    if qw < 0.0:
        qw = -qw; qx = -qx; qy = -qy; qz = -qz
    field[w, 0] = qw
    field[w, 1] = qx
    field[w, 2] = qy
    field[w, 3] = qz


@wp.kernel
def _cond_yaw_kernel(
    seed_arr:     wp.array(dtype=int),                  # type: ignore  (1,)
    seed_offset:  int,
    field:        wp.array(dtype=float, ndim=2),        # type: ignore  (NWORLD, 4)
    apply_mask:   wp.array(dtype=int, ndim=1),          # type: ignore
    apply_to_all: int,
):
    """Random yaw rotation around world-z (palm-down friendly init)."""
    w = wp.tid()
    if apply_to_all == 0 and apply_mask[w] == 0:
        return
    state = wp.rand_init(seed_arr[0] + seed_offset, w)
    yaw = wp.randf(state) * 6.283185307179586 - 3.141592653589793
    field[w, 0] = wp.cos(yaw * 0.5)
    field[w, 1] = 0.0
    field[w, 2] = 0.0
    field[w, 3] = wp.sin(yaw * 0.5)


@wp.kernel
def _cond_constant_kernel(
    n_dim:        int,
    value:        wp.array(dtype=float, ndim=1),        # type: ignore  (n_dim,)
    field:        wp.array(dtype=float, ndim=2),        # type: ignore  (NWORLD, n_dim)
    apply_mask:   wp.array(dtype=int, ndim=1),          # type: ignore
    apply_to_all: int,
):
    """Write a constant per-world (used for ``quat_mode='identity'``)."""
    w = wp.tid()
    if apply_to_all == 0 and apply_mask[w] == 0:
        return
    for i in range(n_dim):
        field[w, i] = value[i]


class WarpPerWorldCondition:
    """Registry of per-world condition buffers (GPU-resident).

    Each registered field is a ``wp.array`` of shape ``(NWORLD, *shape)``
    with float dtype. ``get(name)`` returns the array directly (no copy)
    so it can be consumed inside other warp kernels and inside captured
    CUDA graphs.

    Args:
        NWORLD:  number of parallel warp worlds.
        device:  warp device for buffer allocation. Defaults to ``'cuda'``.
        dtype:   numpy dtype used for the host staging buffer. The wp.array
                 itself is allocated as ``float`` regardless. Default
                 ``np.float32``.
        seed:    optional seed for the internal numpy RNG used by samplers.
    """

    def __init__(
        self,
        NWORLD: int,
        device='cuda',
        dtype = np.float32,
        seed:   int | None = None,
    ):
        self.NWORLD = int(NWORLD)
        self.device = device
        self.dtype  = dtype
        self._rng   = np.random.default_rng(seed)

        # name -> {'shape': tuple, 'gpu': wp.array, 'cpu': wp.array,
        #          'sampler': Optional[Callable[[np.random.Generator], np.ndarray]],
        #          'gpu_sampler': Optional[dict] — GPU-side spec consumed by sample_gpu/sample_all_gpu}
        self._fields: dict[str, dict] = {}

        # Shared GPU seed buffer for the GPU sampler path. Bumped every
        # ``sample_gpu`` / ``sample_all_gpu`` call so successive launches
        # produce independent samples without a CPU sync.
        self._gpu_seed_value = int(seed) if seed is not None else 0
        self._gpu_seed_wp    = wp.array(
            np.array([self._gpu_seed_value], dtype=np.int32),
            dtype=int, device=self.device,
        )

        # Reusable dummy mask passed when ``apply_to_all=1`` (kernel never
        # reads it but warp requires a valid wp.array argument).
        self._dummy_mask_wp  = wp.zeros((self.NWORLD,), dtype=int, device=self.device)

    # ──────────────────────────────────────────────────────────────────
    # Registration
    # ──────────────────────────────────────────────────────────────────

    def register(self, name: str, shape=(), init=None) -> None:
        """Register a per-world field of shape ``(NWORLD, *shape)``.

        Args:
            name:  unique field name. Re-registering raises ``ValueError``.
            shape: per-world shape. ``()`` → ``(NWORLD,)``;
                   ``(D,)`` → ``(NWORLD, D)``; higher dims allowed.
            init:  optional initial value (broadcastable to full shape).
                   Default zeros.
        """
        if name in self._fields:
            raise ValueError(f"field '{name}' is already registered")
        full_shape = (self.NWORLD,) + tuple(shape)

        cpu_buf = np.zeros(full_shape, dtype=self.dtype)
        if init is not None:
            cpu_buf[...] = np.broadcast_to(
                np.asarray(init, dtype=self.dtype), full_shape
            )

        # GPU-resident buffer + persistent CPU staging buffer
        gpu = wp.array(cpu_buf,        dtype=float, device=self.device)
        cpu = wp.array(cpu_buf.copy(), dtype=float, device='cpu')

        self._fields[name] = dict(
            shape       = tuple(shape),
            gpu         = gpu,
            cpu         = cpu,
            sampler     = None,
            gpu_sampler = None,
        )

    def register_uniform(self, name: str, low, high) -> None:
        """Register a ``Uniform[low, high)`` sampler for ``name``.

        ``low`` / ``high`` may be scalar or any value broadcastable to
        the per-world shape.
        """
        f          = self._require(name)
        full_shape = (self.NWORLD,) + f['shape']
        low_arr    = np.asarray(low,  dtype=self.dtype)
        high_arr   = np.asarray(high, dtype=self.dtype)

        def _sampler(rng: np.random.Generator) -> np.ndarray:
            return rng.uniform(low_arr, high_arr, size=full_shape).astype(
                self.dtype, copy=False
            )
        f['sampler'] = _sampler

    def register_normal(self, name: str, mu = 0.0, sigma = 1.0) -> None:
        """Register a ``Normal(mu, sigma)`` sampler for ``name``."""
        f          = self._require(name)
        full_shape = (self.NWORLD,) + f['shape']
        mu_arr     = np.asarray(mu,    dtype=self.dtype)
        sigma_arr  = np.asarray(sigma, dtype=self.dtype)

        def _sampler(rng: np.random.Generator) -> np.ndarray:
            return (mu_arr + sigma_arr *
                    rng.standard_normal(size=full_shape)).astype(
                self.dtype, copy=False
            )
        f['sampler'] = _sampler

    def register_sampler(
        self,
        name: str,
        fn: Callable[[np.random.Generator], np.ndarray],
    ) -> None:
        """Register a custom sampler.

        ``fn(rng) -> np.ndarray`` must return an array broadcastable to
        ``(NWORLD, *shape)``. Use this for task-specific distributions
        (e.g. yaw-only quat sampling, mixture of modes, etc.).
        """
        self._require(name)['sampler'] = fn

    # ──────────────────────────────────────────────────────────────────
    # GPU sampler registration (sync-free path)
    # ──────────────────────────────────────────────────────────────────
    # GPU samplers run inside warp kernels and consume an optional
    # GPU-resident mask (``mask_wp``) for masked re-sampling — eliminates
    # the CPU sync that ``cond.sample(name, world_mask=...)`` requires.

    def register_uniform_gpu(self, name: str, low, high) -> None:
        """Install a GPU-side ``Uniform[low, high)`` sampler for ``name``.

        ``low`` / ``high`` are broadcast to ``shape`` and uploaded as
        ``wp.array``. ``sample_gpu(name)`` then runs entirely on GPU.
        """
        f      = self._require(name)
        n_dim  = int(np.prod(f['shape']) if f['shape'] else 1)
        low_a  = np.broadcast_to(np.asarray(low,  dtype=np.float32),
                                  f['shape'] or (1,)).flatten().astype(np.float32, copy=True)
        high_a = np.broadcast_to(np.asarray(high, dtype=np.float32),
                                  f['shape'] or (1,)).flatten().astype(np.float32, copy=True)
        if low_a.size != n_dim or high_a.size != n_dim:
            raise ValueError(
                f"low/high broadcast mismatch for '{name}': "
                f"got {low_a.size}/{high_a.size}, expected {n_dim}"
            )
        f['gpu_sampler'] = dict(
            kind   = 'uniform',
            n_dim  = n_dim,
            low_wp = wp.array(low_a,  dtype=float, device=self.device),
            high_wp= wp.array(high_a, dtype=float, device=self.device),
        )

    def register_so3_gpu(self, name: str) -> None:
        """Install a GPU-side uniform SO(3) sampler for a 4-vector field.

        Field shape must be ``(4,)`` (wxyz). Use for ``target_wrist_quat``
        with ``mode='so3'`` semantics.
        """
        f = self._require(name)
        if f['shape'] != (4,):
            raise ValueError(
                f"register_so3_gpu requires shape=(4,); '{name}' has {f['shape']}"
            )
        f['gpu_sampler'] = dict(kind='so3')

    def register_yaw_gpu(self, name: str) -> None:
        """Install a GPU-side random-yaw quaternion sampler (around world-z).

        Field shape must be ``(4,)`` (wxyz).
        """
        f = self._require(name)
        if f['shape'] != (4,):
            raise ValueError(
                f"register_yaw_gpu requires shape=(4,); '{name}' has {f['shape']}"
            )
        f['gpu_sampler'] = dict(kind='yaw')

    def register_constant_gpu(self, name: str, value) -> None:
        """Install a GPU sampler that writes a constant (per-world identical)
        each call. Useful for ``quat_mode='identity'``."""
        f      = self._require(name)
        n_dim  = int(np.prod(f['shape']) if f['shape'] else 1)
        val_a  = np.broadcast_to(np.asarray(value, dtype=np.float32),
                                  f['shape'] or (1,)).flatten().astype(np.float32, copy=True)
        if val_a.size != n_dim:
            raise ValueError(
                f"constant value broadcast mismatch for '{name}'"
            )
        f['gpu_sampler'] = dict(
            kind  = 'constant',
            n_dim = n_dim,
            value_wp = wp.array(val_a, dtype=float, device=self.device),
        )

    def unregister(self, name: str) -> None:
        """Drop ``name`` from the registry. wp.array buffers are released."""
        if name not in self._fields:
            return
        del self._fields[name]

    # ──────────────────────────────────────────────────────────────────
    # Read / write
    # ──────────────────────────────────────────────────────────────────

    def get(self, name: str):
        """Return the GPU ``wp.array`` for ``name`` (no copy).

        Safe to consume inside captured CUDA graphs — the underlying
        buffer is never re-allocated after registration.
        """
        return self._require(name)['gpu']

    def get_numpy(self, name: str) -> np.ndarray:
        """Read the field back to CPU as numpy (incurs GPU→CPU sync)."""
        f = self._require(name)
        wp.copy(f['cpu'], f['gpu'])
        return f['cpu'].numpy().copy()

    def set(self, name: str, value) -> None:
        """Write ``value`` to the GPU buffer (full-array write).

        ``value`` may be:
          * ``np.ndarray`` broadcastable to ``(NWORLD, *shape)``
          * ``wp.array`` of exact shape ``(NWORLD, *shape)`` — copied via
            ``wp.copy`` (host roundtrip skipped).

        Capture-graph compatible: re-uses the field's CPU staging buffer.
        """
        f          = self._require(name)
        full_shape = (self.NWORLD,) + f['shape']

        if isinstance(value, wp.array):
            if tuple(value.shape) != full_shape:
                raise ValueError(
                    f"wp.array shape {tuple(value.shape)} != {full_shape}"
                )
            wp.copy(f['gpu'], value)
            return

        np_buf = np.broadcast_to(
            np.asarray(value, dtype=self.dtype), full_shape
        ).astype(self.dtype, copy=False)
        f['cpu'].numpy()[...] = np_buf
        wp.copy(f['gpu'], f['cpu'])

    def set_per_world(self, name: str, world_indices, value) -> None:
        """Write ``value`` only to the listed world rows.

        Args:
            world_indices: 1-D int sequence (length K).
            value:         broadcastable to ``(K, *shape)``.

        Pulls the current GPU state into CPU staging, edits the listed
        rows, then uploads. One GPU→CPU sync + one CPU→GPU sync per call.
        """
        f       = self._require(name)
        idxs    = np.asarray(world_indices, dtype=np.int64)
        rows_sh = (len(idxs),) + f['shape']
        val_np  = np.broadcast_to(
            np.asarray(value, dtype=self.dtype), rows_sh
        ).astype(self.dtype, copy=False)

        wp.copy(f['cpu'], f['gpu'])
        f['cpu'].numpy()[idxs] = val_np
        wp.copy(f['gpu'], f['cpu'])

    def set_world_mask(self, name: str, world_mask, value) -> None:
        """Same as ``set_per_world`` but takes a bool mask of length NWORLD.

        ``value`` may be either ``(K, *shape)`` (only the masked rows) or
        ``(NWORLD, *shape)`` (full array — only the masked rows are used).
        """
        mask = np.asarray(world_mask, dtype=bool)
        if mask.shape != (self.NWORLD,):
            raise ValueError(
                f"world_mask shape {mask.shape} != ({self.NWORLD},)"
            )
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return

        f          = self._require(name)
        full_shape = (self.NWORLD,) + f['shape']
        val_np     = np.asarray(value, dtype=self.dtype)
        if val_np.shape == full_shape:
            val_np = val_np[idxs]
        self.set_per_world(name, idxs, val_np)

    # ──────────────────────────────────────────────────────────────────
    # Sampling / reset
    # ──────────────────────────────────────────────────────────────────

    def seed(self, seed: int) -> None:
        """Re-seed the internal numpy RNG."""
        self._rng = np.random.default_rng(int(seed))

    def sample(self, name: str, world_mask=None) -> None:
        """Re-sample ``name`` using its registered sampler.

        Args:
            world_mask: optional bool array of length NWORLD. If given,
                        only those worlds are re-sampled (other rows
                        preserved). If None, the full buffer is overwritten.
        """
        f = self._require(name)
        if f['sampler'] is None:
            raise RuntimeError(
                f"field '{name}' has no sampler — "
                "call register_uniform / register_normal / register_sampler first."
            )
        sampled = f['sampler'](self._rng)

        if world_mask is None:
            f['cpu'].numpy()[...] = sampled
            wp.copy(f['gpu'], f['cpu'])
            return

        mask = np.asarray(world_mask, dtype=bool)
        if mask.shape != (self.NWORLD,):
            raise ValueError(
                f"world_mask shape {mask.shape} != ({self.NWORLD},)"
            )
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return

        wp.copy(f['cpu'], f['gpu'])
        f['cpu'].numpy()[idxs] = sampled[idxs]
        wp.copy(f['gpu'], f['cpu'])

    def sample_all(self, world_mask=None, skip_no_sampler: bool = True) -> None:
        """Re-sample every field that has a registered sampler.

        ``skip_no_sampler=False`` raises if any field is missing a sampler.
        """
        for name, f in self._fields.items():
            if f['sampler'] is None:
                if skip_no_sampler:
                    continue
                raise RuntimeError(f"field '{name}' has no sampler.")
            self.sample(name, world_mask=world_mask)

    def has_cpu_samplers(self) -> bool:
        """True iff any registered field has a CPU sampler (e.g. taxonomy
        injection on top of a GPU uniform). Used by :meth:`reset` to
        decide whether the CPU pass + its sync are needed."""
        return any(f['sampler'] is not None for f in self._fields.values())

    def reset(self, mask_wp=None) -> None:
        """Re-sample every field. **GPU samplers run first** (sync-free),
        then CPU samplers run on top for fields that registered them
        (e.g. taxonomy injection over a uniform ``target_qpos``).

        Replaces the legacy ``reset = sample_all`` alias which only ran
        CPU samplers — that silently no-op'd for fields registered with
        only a GPU sampler (``target_wrist_pos`` / ``target_wrist_quat``
        / ``target_qpos`` from :func:`register_hand_pose_target`).

        Args:
            mask_wp: optional ``wp.array`` bool mask of length NWORLD —
                     re-sample only those worlds. ``None`` → all worlds.

        CPU sync: only paid when ``has_cpu_samplers()`` is True AND
        ``mask_wp`` is provided (we need a CPU mirror of the mask for
        the CPU sampler pass). With ``mask_wp=None`` the CPU pass runs
        without any mask sync.
        """
        # GPU pass — skips fields without a GPU sampler (e.g. host-filled
        # ``target_af_xpos``). Sync-free.
        self.sample_all_gpu(mask_wp=mask_wp)

        # CPU pass — only fields whose ``f['sampler']`` is set. When
        # ``mask_wp`` is given we sync once to derive the np mirror; the
        # masked CPU sampler then overwrites the GPU result for those
        # worlds (intended for taxonomy injection — wasted GPU work for
        # the overwritten rows is cheap on per-world reset).
        if not self.has_cpu_samplers():
            return
        if mask_wp is None:
            self.sample_all(world_mask=None)
        else:
            mask_np = wp.to_torch(mask_wp).cpu().numpy().astype(bool)
            self.sample_all(world_mask=mask_np)

    # ──────────────────────────────────────────────────────────────────
    # GPU-side sampling (sync-free; used by per-world reset path)
    # ──────────────────────────────────────────────────────────────────

    def _bump_gpu_seed(self) -> None:
        """Advance the GPU seed by one and re-upload (cheap host→device copy)."""
        self._gpu_seed_value += 1
        self._gpu_seed_wp.assign(
            np.array([self._gpu_seed_value], dtype=np.int32),
        )

    def sample_gpu(self, name: str, mask_wp=None) -> None:
        """Re-sample ``name`` via its GPU sampler.

        Args:
            mask_wp: optional ``wp.array(int, NWORLD)`` GPU mask. ``1`` = re-sample
                     that world; ``0`` = leave existing value untouched.
                     ``None`` = re-sample every world.

        Raises:
            RuntimeError: if no GPU sampler was registered for ``name``.
        """
        f       = self._require(name)
        spec    = f.get('gpu_sampler')
        if spec is None:
            raise RuntimeError(
                f"field '{name}' has no GPU sampler "
                "(call register_uniform_gpu / register_so3_gpu / "
                "register_yaw_gpu / register_constant_gpu first)"
            )
        self._bump_gpu_seed()
        seed_offset = (hash(name) & 0x7FFFFFFF)   # deterministic per-field offset
        kind = spec['kind']
        # When mask_wp is None we run with apply_to_all=1 (kernel ignores
        # the mask buffer); pass _dummy_mask_wp as the placeholder.
        mask_arg     = mask_wp if mask_wp is not None else self._dummy_mask_wp
        apply_to_all = 0 if mask_wp is not None else 1

        if kind == 'uniform':
            wp.launch(
                _cond_uniform_kernel, dim=self.NWORLD,
                inputs=[
                    self._gpu_seed_wp, int(seed_offset), int(spec['n_dim']),
                    spec['low_wp'], spec['high_wp'],
                    f['gpu'], mask_arg, int(apply_to_all),
                ],
            )
        elif kind == 'so3':
            wp.launch(
                _cond_so3_kernel, dim=self.NWORLD,
                inputs=[
                    self._gpu_seed_wp, int(seed_offset),
                    f['gpu'], mask_arg, int(apply_to_all),
                ],
            )
        elif kind == 'yaw':
            wp.launch(
                _cond_yaw_kernel, dim=self.NWORLD,
                inputs=[
                    self._gpu_seed_wp, int(seed_offset),
                    f['gpu'], mask_arg, int(apply_to_all),
                ],
            )
        elif kind == 'constant':
            wp.launch(
                _cond_constant_kernel, dim=self.NWORLD,
                inputs=[
                    int(spec['n_dim']), spec['value_wp'],
                    f['gpu'], mask_arg, int(apply_to_all),
                ],
            )
        else:
            raise RuntimeError(f"unknown gpu_sampler kind '{kind}' on field '{name}'")

    def sample_all_gpu(self, mask_wp=None, skip_no_sampler: bool = True) -> None:
        """GPU-only counterpart of :meth:`sample_all`. Sync-free — every
        registered ``gpu_sampler`` runs in a single warp kernel each.

        Fields without a GPU sampler are skipped (or raise if
        ``skip_no_sampler=False``). Use this in the per-world reset path
        to avoid CPU sync on every control step.
        """
        for name, f in self._fields.items():
            if f.get('gpu_sampler') is None:
                if skip_no_sampler:
                    continue
                raise RuntimeError(f"field '{name}' has no GPU sampler.")
            self.sample_gpu(name, mask_wp=mask_wp)

    def has_gpu_samplers(self) -> bool:
        """True iff every registered field has a GPU sampler available
        (so :meth:`sample_all_gpu` covers them all without skipping)."""
        return bool(self._fields) and all(
            f.get('gpu_sampler') is not None for f in self._fields.values()
        )

    # ──────────────────────────────────────────────────────────────────
    # Introspection / dunders
    # ──────────────────────────────────────────────────────────────────

    def names(self) -> list[str]:
        return list(self._fields.keys())

    def shape(self, name: str) -> tuple:
        return (self.NWORLD,) + self._require(name)['shape']

    def has_sampler(self, name: str) -> bool:
        return self._require(name)['sampler'] is not None

    def __contains__(self, name: str) -> bool:
        return name in self._fields

    def __getitem__(self, name: str):
        return self.get(name)

    def __repr__(self) -> str:
        items = ", ".join(
            f"{n}{self.shape(n)}" for n in self._fields
        )
        return f"WarpPerWorldCondition(NWORLD={self.NWORLD}, fields=[{items}])"

    # ──────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────

    def _require(self, name: str) -> dict:
        if name not in self._fields:
            raise KeyError(
                f"field '{name}' not registered. "
                f"registered: {list(self._fields.keys())}"
            )
        return self._fields[name]


# ──────────────────────────────────────────────────────────────────────────
# Convenience: hand-pose-tracking preset
# ──────────────────────────────────────────────────────────────────────────

def make_quat_sampler(
    mode: str,
    n_world: int,
    identity_quat = (1.0, 0.0, 0.0, 0.0),
):
    """Return a ``WarpPerWorldCondition``-compatible sampler that fills a
    ``(NWORLD, 4)`` wxyz quaternion buffer per the requested mode.

    The returned function has signature ``fn(rng) -> np.ndarray`` with
    output shape ``(n_world, 4)`` — the full per-world buffer
    :meth:`WarpPerWorldCondition.sample` expects.

    Modes:
      ``'identity'`` — constant ``identity_quat`` (default ``(1, 0, 0, 0)``)
                       for every world.
      ``'yaw_only'`` — random yaw rotation around world-z (good for
                       palm-down init / lifting tasks).
      ``'so3'``      — uniform on SO(3) (normalized 4D Gaussian, ``w >= 0``
                       enforced for the double-cover canonicalisation).
    """
    n = int(n_world)
    if mode == "identity":
        ident = np.tile(np.asarray(identity_quat, dtype=np.float32)[None, :],
                         (n, 1))
        def _identity_sampler(rng: np.random.Generator) -> np.ndarray:
            del rng  # constant buffer
            return ident.copy()
        return _identity_sampler

    if mode == "yaw_only":
        def _yaw_sampler(rng: np.random.Generator) -> np.ndarray:
            yaw = rng.uniform(-np.pi, np.pi, size=n).astype(np.float32)
            q   = np.zeros((n, 4), dtype=np.float32)
            q[:, 0] = np.cos(yaw / 2.0)
            q[:, 3] = np.sin(yaw / 2.0)
            return q
        return _yaw_sampler

    if mode == "so3":
        def _so3_sampler(rng: np.random.Generator) -> np.ndarray:
            q = rng.standard_normal(size=(n, 4)).astype(np.float32)
            q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
            # Enforce w >= 0 (q and -q encode the same rotation).
            flip = q[:, 0] < 0
            q[flip] *= -1.0
            return q
        return _so3_sampler

    raise ValueError(
        f"unknown quat_mode='{mode}' (use identity / yaw_only / so3)"
    )


def register_hand_pose_target(
    cond:               WarpPerWorldCondition,
    env,
    hand_util,
    pos_low             = (-0.20, -0.20,  0.10),
    pos_high            = ( 0.20,  0.20,  0.50),
    quat_init           = (1.0, 0.0, 0.0, 0.0),   # wxyz identity
    quat_mode:    str   = 'identity',
    sample_qpos:  bool  = True,
    qpos_low            = None,
    qpos_high           = None,
    *,
    pos_name:  str      = 'target_wrist_pos',
    quat_name: str      = 'target_wrist_quat',
    qpos_name: str      = 'target_qpos',
) -> None:
    """Register the standard hand-pose-tracking target fields on ``cond``.

    Three fields are registered:

      * ``target_wrist_pos``  shape ``(3,)``      uniform[pos_low, pos_high)
      * ``target_wrist_quat`` shape ``(4,)`` wxyz with sampler picked by
                              ``quat_mode``:

                              =============  ========================================
                              ``quat_mode``  sampler
                              =============  ========================================
                              ``'identity'`` constant ``quat_init`` (default)
                              ``'yaw_only'`` random yaw around world-z
                              ``'so3'``      uniform on SO(3)
                              =============  ========================================

                              For task-specific distributions outside this
                              menu, leave at ``'identity'`` and override via
                              ``cond.register_sampler(quat_name, fn)`` after
                              this call.
      * ``target_qpos``       shape ``(n_ctrl,)`` uniform[qpos_low, qpos_high)
                              — only registered when ``sample_qpos=True``.
                              ``qpos_low / qpos_high`` default to the env's
                              actuator ctrlrange (per-actuator).

    Args:
        cond:        a ``WarpPerWorldCondition`` whose NWORLD matches ``env``.
        env:         ``SingleHandSubEnv`` instance (provides ``mjm.actuator_ctrlrange``).
        hand_util:   ``HandUtils`` instance (currently unused but kept in
                     the signature so callers can pass the same trio they
                     pass to ``WarpActionApplier``; future variants of this
                     helper may use mocap / wrist info from it).
        pos_low/high:        per-axis bounds for wrist xyz (length 3).
        quat_init:           constant wxyz quat — also used as the initial
                             buffer fill when ``quat_mode='identity'``.
        quat_mode:           ``'identity'`` / ``'yaw_only'`` / ``'so3'`` —
                             which per-world sampler to install for
                             ``target_wrist_quat``.
        sample_qpos:         if False, ``target_qpos`` is not registered.
        qpos_low/high:       per-actuator bounds. ``None`` → use ctrlrange.
        pos_name/quat_name/qpos_name: field-name overrides.
    """
    del hand_util  # currently unused; reserved for future symmetry

    cond.register(pos_name, shape=(3,))
    # GPU sampler — sync-free per-world re-sampling on per-world reset.
    cond.register_uniform_gpu(pos_name, low=pos_low, high=pos_high)

    cond.register(quat_name, shape=(4,),
                  init=np.asarray(quat_init, dtype=np.float32))
    if quat_mode == 'identity':
        cond.register_constant_gpu(quat_name, value=np.asarray(quat_init, dtype=np.float32))
    elif quat_mode == 'yaw_only':
        cond.register_yaw_gpu(quat_name)
    elif quat_mode == 'so3':
        cond.register_so3_gpu(quat_name)
    else:
        raise ValueError(
            f"unknown quat_mode='{quat_mode}' (use identity / yaw_only / so3)"
        )

    if not sample_qpos:
        return

    n_ctrl = int(env.mjm.nu)
    if qpos_low is None or qpos_high is None:
        ctrl_range = np.asarray(env.mjm.actuator_ctrlrange, dtype=np.float32)
        lo = ctrl_range[:, 0].copy()
        hi = ctrl_range[:, 1].copy()
        # Unbounded actuators (range = 0,0): collapse to 0.
        unbounded     = (lo == 0) & (hi == 0)
        lo[unbounded] = 0.0
        hi[unbounded] = 0.0
        if qpos_low  is None: qpos_low  = lo
        if qpos_high is None: qpos_high = hi

    cond.register(qpos_name, shape=(n_ctrl,))
    cond.register_uniform_gpu(qpos_name, low=qpos_low, high=qpos_high)
