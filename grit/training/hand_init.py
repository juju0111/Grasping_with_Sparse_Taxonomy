"""Per-hand finger qpos / ctrl randomization for warp parallel envs.

Translates the legacy JAX ``_finger_init_fn`` (
``thirdParty/taxonomy_guided_RL/.../base_env_0.py``) to a single,
data-driven warp kernel that is parameterized per hand.

Why this lives in its own module
--------------------------------
The original JAX implementation branches on ``hand_name`` and applies a
mix of *(uniform sample) + (selective re-sample) + (per-index multiplier)
+ (tendon coupling)*. Each branch reads/writes specific slice indices
(e.g. ``finger_init.at[1:4].mul(0.05)``).

Each hand is authored as ONE declarative op list in :data:`HAND_INIT_OPS`
(``scale`` / ``override`` / ``couple`` ops over the uniform base sample) — no
imperative per-hand slicing. :func:`get_hand_init_config` compiles that list
into three small per-hand "patch" tables so a **single** warp kernel covers
every hand:

* ``multiplier[i]`` — multiply the i-th sampled finger qpos by this
  scalar (``1.0`` = identity, ``0.0`` = clamp to zero).
* ``override (idx, low, high)`` — re-sample finger qpos at ``idx``
  uniformly in ``[low, high]`` (overrides the multiplier path). Used for
  the JAX ``actions_f = uniform(...); finger_init.at[idx].set(actions_f)``
  pattern.
* ``couple (src, dst, weight)`` — set ``f[dst] = f[src] * weight``
  *after* multipliers/overrides. Used for the inspire tendon coupling
  ``finger_init.at[5::2].set(finger_init[4::2] * weight)``.

A separate ``ctrl_qpos_idx[a]`` array maps each ctrl index to the qpos
DOF index that ctrl should mirror (so position-controlled actuators do
not immediately drag the random qpos back to zero on the next step).

Usage
-----

    from grit.training.hand_init import HandFingerInitializer

    finger_init = HandFingerInitializer(sampled_env, hand_util,
                                        seed=42, handler_idx=0)
    # full reset:
    finger_init.apply_all()
    # per-world reset (only ``done_mask == 1`` worlds):
    finger_init.apply_masked(done_mask_wp)

The kernel writes to ``d.qpos`` (finger DOFs) and ``d.ctrl`` (matching
actuator setpoints) — both pre-existing buffers, no GPU allocation per
call.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np
import warp as wp


# ──────────────────────────────────────────────────────────────────────────
# Generic finger init kernel (data-driven)
# ──────────────────────────────────────────────────────────────────────────


@wp.kernel
def _finger_init_kernel(
    seed_arr:        wp.array(dtype=int),                      # type: ignore
    n_finger:        int,
    finger_qpa:      wp.array(dtype=int,   ndim=1),            # type: ignore  (n_finger,)
    jnt_low:         wp.array(dtype=float, ndim=1),            # type: ignore  (n_finger,)
    jnt_high:        wp.array(dtype=float, ndim=1),            # type: ignore  (n_finger,)
    multiplier:      wp.array(dtype=float, ndim=1),            # type: ignore  (n_finger,)
    n_overrides:     int,
    override_idx:    wp.array(dtype=int,   ndim=1),            # type: ignore  (k,)
    override_low:    wp.array(dtype=float, ndim=1),            # type: ignore  (k,)
    override_high:   wp.array(dtype=float, ndim=1),            # type: ignore  (k,)
    n_couples:       int,
    couple_src:      wp.array(dtype=int,   ndim=1),            # type: ignore  (m,)
    couple_dst:      wp.array(dtype=int,   ndim=1),            # type: ignore  (m,)
    couple_weight:   wp.array(dtype=float, ndim=1),            # type: ignore  (m,)
    n_ctrl:          int,
    ctrl_qpos_idx:   wp.array(dtype=int,   ndim=1),            # type: ignore  (n_ctrl,) qpa-idx (-1 to skip)
    workspace:       wp.array(dtype=float, ndim=2),            # type: ignore  (NWORLD, n_finger)
    qpos:            wp.array(dtype=float, ndim=2),            # type: ignore
    ctrl:            wp.array(dtype=float, ndim=2),            # type: ignore
    apply_mask:      wp.array(dtype=int,   ndim=1),            # type: ignore  (NWORLD,)
    apply_to_all:    int,
):
    w = wp.tid()
    if apply_to_all == 0 and apply_mask[w] == 0:
        return

    state = wp.rand_init(seed_arr[0] + 31337, w)

    # 1) Uniform sample × multiplier  →  workspace[w, :]
    for i in range(n_finger):
        v = wp.randf(state) * (jnt_high[i] - jnt_low[i]) + jnt_low[i]
        workspace[w, i] = v * multiplier[i]

    # 2) Per-index overrides (re-sample within a different range)
    for k in range(n_overrides):
        idx = override_idx[k]
        v   = wp.randf(state) * (override_high[k] - override_low[k]) + override_low[k]
        workspace[w, idx] = v

    # 3) Tendon coupling f[dst] = f[src] * weight
    for k in range(n_couples):
        src = couple_src[k]
        dst = couple_dst[k]
        workspace[w, dst] = workspace[w, src] * couple_weight[k]

    # 4) Write to qpos at finger DOF addresses
    for i in range(n_finger):
        qpos[w, finger_qpa[i]] = workspace[w, i]

    # 5) Mirror to ctrl so position-controlled actuators do not immediately
    #    drag qpos back to its previous setpoint after the next sim step.
    for a in range(n_ctrl):
        idx = ctrl_qpos_idx[a]
        if idx >= 0:
            ctrl[w, a] = workspace[w, idx]


@wp.kernel
def _gather_collision_free_init_kernel(
    seed_arr:      wp.array(dtype=int),                     # type: ignore [seed]
    table:         wp.array(dtype=float, ndim=2),          # type: ignore (K, n_ctrl) rows
    n_rows:        int,
    n_ctrl:        int,
    ctrl_qpos_idx: wp.array(dtype=int,   ndim=1),          # type: ignore (n_ctrl,) qpos addr / -1
    qpos:          wp.array(dtype=float, ndim=2),          # type: ignore
    apply_mask:    wp.array(dtype=int,   ndim=1),          # type: ignore (NWORLD,)
    apply_to_all:  int,
):
    """reset init: per world, draw one row from a **self-collision-free** qpos
    table (taxonomy|pinch|bank) and set it as the finger qpos. This removes the
    problem of uniform sampling producing self-colliding poses. ctrl is
    qpos-relative (``ctrl=qpos+Δ``), so no mirroring is needed — the next step
    uses this qpos as its reference. Table columns follow actuator order (same
    mapping as ``ctrl_qpos_idx`` in finger_init)."""
    w = wp.tid()
    if apply_to_all == 0 and apply_mask[w] == 0:
        return
    state = wp.rand_init(seed_arr[0] + 91193, w)
    row = int(wp.randf(state) * float(n_rows))
    if row >= n_rows:
        row = n_rows - 1
    for a in range(n_ctrl):
        qa = ctrl_qpos_idx[a]
        if qa >= 0:
            qpos[w, qa] = table[row, a]


# ──────────────────────────────────────────────────────────────────────────
# Per-hand finger-init spec — DECLARATIVE op table
# (one ordered op list per hand; translated from legacy JAX ``_finger_init_fn``)
# ──────────────────────────────────────────────────────────────────────────
#
# Every hand starts from a uniform base sample over each finger DOF's
# ``[jnt_low, jnt_high]`` range, then applies its op list IN ORDER. There is no
# imperative per-hand slicing scattered across functions any more — adding a
# hand == adding one entry below. Indices are finger-DOF indices
# ``0 .. n_finger-1``; negative indices and ``None`` stops use numpy slice
# semantics, so the same spec is length-agnostic across model variants.
#
# Op grammar:
#   ("scale",    start, stop, step, factor)
#       finger[start:stop:step] *= factor      (JAX ``.at[...].mul(factor)``;
#                                               composes, e.g. 0.33 then 0.1)
#   ("override", idx, low, high)
#       finger[idx] = uniform(low, high)        (JAX ``actions_f = uniform(...);
#                                               .at[idx].set(actions_f)`` — REPLACES)
#   ("couple",   start, stop, step, dst_offset, weight)
#       finger[k + dst_offset] = finger[k] * weight  for k in [start:stop:step]
#       (applied AFTER scale/override, reading the post-scale value — JAX
#        inspire tendon equality ``.at[5::2].set(finger[4::2] * weight)``)
#
# NOTE: the kernel applies the compiled tables in 3 passes (all scales →
# overrides → couples). That matches the JAX per-op order for every hand here
# because no hand's override index overlaps a scaled index, and couples always
# read post-scale sources. Keep that invariant when adding a hand.

HAND_INIT_OPS: Dict[str, List[tuple]] = {
    # Inspire: thumb (DOF 0) free over its full range; proximal cluster damped;
    # every-other finger DOF scaled then tendon-coupled to its sibling.
    "inspire": [
        ("scale",   1, 4,    1, 0.05),
        ("scale",   4, None, 2, 0.4),
        ("couple",  4, None, 2, 1, 1.0843),    # f[k+1] = f[k]*1.0843, k = 4,6,8,…
    ],
    # Allex: thumb base (DOF 0) overridden to a closed range; finger DOFs damped.
    "allex": [
        ("override", 0, -1.75, -1.0),
        ("scale",    1, 2,    1, 0.1),
        ("scale",    2, 4,    1, 0.0),
        ("scale",    4, None, 1, 0.4),
        ("scale",    4, None, 4, 0.1),          # → 0.33*0.1 on each finger's base DOF
    ],
    # Allegro: thumb base = DOF ``n-4`` (last finger group), overridden; rest damped.
    "allegro": [
        ("override", -4, 0.5, 1.35),
        ("scale",    -3, -2,   1, 0.2),
        ("scale",    -2, None, 1, 0.2),
        ("scale",     0, -4,   1, 0.4),
        ("scale",     0, -4,   4, 0.4),        # → 0.33*0.33 on each finger's base DOF
    ],
    # Tesollo: thumb base = DOF 1, overridden; rest damped.
    "tesollo": [
        ("override", 1, -1.8, -1.0),
        ("scale",    2, 4,    1, 0.1),
        ("scale",    4, None, 1, 0.4),
        ("scale",    4, None, 4, 0.1),
    ],
    # shadow: no special branch in legacy JAX (TODO) → uniform default.
    "shadow": [
        ("override", 1, -1.8, -1.0),
        ("scale",    0, 15, 1, 0.4),
        ("scale",    0, 8,  3, 0.1),
        ("scale",    9, 10, 1, 0.1),
        ("scale",    -3, -2, 1, 0.1),
        ("scale",    -2, None, 1, 0.4),
    ],
    # robotis_sh5: no legacy branch → uniform default.
    "robotis_sh5": [
        ("override", 1, -1.8, -1.0),
        ("scale",    2, 4,    1, 0.1),
        ("scale",    4, None, 1, 0.4),
        ("scale",    4, None, 4, 0.1),
    ],
    # wuji_hand2: the thumb is the **first** group (DOF 0..3) — same layout as tesollo.
    "wuji_hand2": [
        ("override", 0, -0.4, 0.5),       # thumb_cmc_flex — opposition angle
        ("override", 1, -1.4, -0.7),       # thumb_cmc_abd
        ("scale",    2, 4,    1, 0.1),    # thumb mcp/ip → nearly extended
        ("scale",    4, None, 1, 0.3),    # all four fingers → 30%
        ("scale",    5, None, 4, 0.1),    # abduction of each finger (2nd DOF) → cumulative 0.03
    ],
}


def _build_config_from_ops(ops: List[tuple], n_finger: int) -> dict:
    """Compile a declarative op list into the kernel's
    ``{multiplier, overrides, couples}`` patch arrays.

    Length-agnostic: negative indices / ``None`` stops are resolved with numpy
    slice semantics against ``n_finger``. An empty op list yields a plain
    uniform sample (all-ones multiplier, no overrides / couples).
    """
    multiplier = np.ones(int(n_finger), dtype=np.float32)
    overrides: List[tuple] = []     # (idx, low, high)   — idx resolved >= 0
    couples:   List[tuple] = []     # (src, dst, weight)
    idx_space = np.arange(int(n_finger))

    for op in ops:
        kind = op[0]
        if kind == "scale":
            _, start, stop, step, factor = op
            multiplier[start:stop:step] *= np.float32(factor)
        elif kind == "override":
            _, idx, low, high = op
            ridx = int(idx_space[idx])                    # resolve negative index
            overrides.append((ridx, float(low), float(high)))
        elif kind == "couple":
            _, start, stop, step, dst_off, weight = op
            for s in idx_space[start:stop:step]:
                dst = int(s) + int(dst_off)
                if 0 <= dst < int(n_finger):              # skip out-of-range siblings
                    couples.append((int(s), dst, float(weight)))
        else:
            raise ValueError(f"unknown finger-init op kind: {kind!r}")

    return dict(multiplier=multiplier, overrides=overrides, couples=couples)


def get_hand_init_config(hand_name: str, n_finger: int) -> dict:
    """Return a ``{multiplier, overrides, couples}`` patch table for ``hand_name``.

    Hands without a legacy branch (``shadow`` / ``robotis_sh5`` / anything not
    in :data:`HAND_INIT_OPS`) fall back to a plain uniform sample.
    """
    ops = HAND_INIT_OPS.get(hand_name, [])
    return _build_config_from_ops(ops, int(n_finger))


# ──────────────────────────────────────────────────────────────────────────
# HandFingerInitializer — owns GPU buffers + dispatches the kernel
# ──────────────────────────────────────────────────────────────────────────


class HandFingerInitializer:
    """Build per-hand finger init data and run the warp kernel.

    Args:
        sampled_env: ``SingleHandSubEnv`` instance (provides ``mjm`` / ``d`` /
            ``NWORLD``).
        hand_util:   :class:`HandUtils` for this sub-env.
        seed:        base RNG seed.
        handler_idx: per-handler offset added to the seed (multi-hand).
        config:      optional explicit ``{multiplier, overrides, couples}``
            override; if ``None``, derived from
            :func:`get_hand_init_config(hand_util.hand_name, ...)`.

    Attributes:
        n_finger:    number of finger DOFs (== ``hand_util.finger_link_num``).
        n_ctrl:      number of actuators on the model.
        finger_qpa:  ``(n_finger,)`` np.int32 — qpos addresses of finger DOFs.
        ctrl_qpos_idx: ``(n_ctrl,)`` np.int32 — qpa-idx (or ``-1``) per ctrl.
    """

    def __init__(
        self,
        sampled_env,
        hand_util,
        seed: int = 42,
        handler_idx: int = 0,
        config: Optional[dict] = None,
    ):
        self.sampled_env = sampled_env
        self.hand_util   = hand_util
        self.handler_idx = int(handler_idx)
        self.NWORLD      = int(sampled_env.NWORLD)
        self.device      = sampled_env.d.qpos.device

        self.n_finger = int(hand_util.finger_link_num)
        self.n_ctrl   = int(sampled_env.mjm.nu)

        # Resolve finger-joint qpos addresses + per-DOF ranges from the
        # compiled skeleton model.
        self.finger_qpa, self.jnt_low_np, self.jnt_high_np = \
            self._resolve_finger_qpa_and_range(sampled_env, hand_util)

        # Build the ctrl→qpa mirror table so randomized qpos persists
        # through the next forward step (else position actuators would
        # immediately pull qpos back to their previous setpoint).
        self.ctrl_qpos_idx = self._build_ctrl_qpos_idx(
            sampled_env.mjm, self.finger_qpa, hand_util,
        )

        # Per-hand patch table (translated from legacy JAX ``_finger_init_fn``).
        cfg = config or get_hand_init_config(hand_util.hand_name, self.n_finger)
        multiplier = np.asarray(cfg["multiplier"], dtype=np.float32).copy()
        if multiplier.shape != (self.n_finger,):
            raise ValueError(
                f"multiplier shape {multiplier.shape} != (n_finger={self.n_finger},)"
            )

        overrides = list(cfg.get("overrides", []))   # [(idx, low, high), ...]
        couples   = list(cfg.get("couples",   []))   # [(src, dst, weight), ...]

        # Unpack override / couple tables (kernel needs separate arrays).
        if overrides:
            ov_idx  = np.asarray([t[0] for t in overrides], dtype=np.int32)
            ov_low  = np.asarray([t[1] for t in overrides], dtype=np.float32)
            ov_high = np.asarray([t[2] for t in overrides], dtype=np.float32)
        else:
            ov_idx  = np.zeros((1,), dtype=np.int32)
            ov_low  = np.zeros((1,), dtype=np.float32)
            ov_high = np.zeros((1,), dtype=np.float32)
        if couples:
            cp_src = np.asarray([t[0] for t in couples], dtype=np.int32)
            cp_dst = np.asarray([t[1] for t in couples], dtype=np.int32)
            cp_wgt = np.asarray([t[2] for t in couples], dtype=np.float32)
        else:
            cp_src = np.zeros((1,), dtype=np.int32)
            cp_dst = np.zeros((1,), dtype=np.int32)
            cp_wgt = np.zeros((1,), dtype=np.float32)
        self._n_overrides = int(len(overrides))
        self._n_couples   = int(len(couples))

        # GPU upload (immutable for the lifetime of the env).
        d = self.device
        self._finger_qpa_wp    = wp.array(self.finger_qpa,    dtype=int,   device=d)
        self._jnt_low_wp       = wp.array(self.jnt_low_np,    dtype=float, device=d)
        self._jnt_high_wp      = wp.array(self.jnt_high_np,   dtype=float, device=d)
        self._multiplier_wp    = wp.array(multiplier,         dtype=float, device=d)
        self._override_idx_wp  = wp.array(ov_idx,             dtype=int,   device=d)
        self._override_low_wp  = wp.array(ov_low,             dtype=float, device=d)
        self._override_high_wp = wp.array(ov_high,            dtype=float, device=d)
        self._couple_src_wp    = wp.array(cp_src,             dtype=int,   device=d)
        self._couple_dst_wp    = wp.array(cp_dst,             dtype=int,   device=d)
        self._couple_weight_wp = wp.array(cp_wgt,             dtype=float, device=d)
        self._ctrl_qpos_idx_wp = wp.array(self.ctrl_qpos_idx, dtype=int,   device=d)

        # Per-launch scratch + seed.
        self._workspace_wp = wp.zeros((self.NWORLD, self.n_finger),
                                       dtype=float, device=d)
        self._seed_value   = int(seed) + 1000 * self.handler_idx
        self._seed_wp      = wp.array(np.array([self._seed_value], dtype=np.int32),
                                       dtype=int, device=d)

        # Optional collision-free init: when attached, ``apply_*`` gathers a row
        # from this ``(K, n_ctrl)`` table (self-collision-free taxonomy/pinch/bank)
        # instead of the uniform sampler — see ``attach_collision_free_bank``.
        self._cf_table_wp = None
        self._cf_nrows    = 0
        self._cf_qpa_wp   = None

    # ── Public API ──────────────────────────────────────────────────────

    def attach_collision_free_bank(self, table_wp, n_rows: int, ctrl_qpa_wp) -> None:
        """Route init through a self-collision-free qpos bank (gather) instead of
        the uniform sampler. ``table_wp`` is a ``wp.array((K, n_ctrl))`` in
        actuator order; ``ctrl_qpa_wp`` is the ``(n_ctrl,)`` qpos-address per
        column that the table was BUILT with (e.g. ``applier._ctrl_qpa``) — the
        gather writes ``qpos[ctrl_qpa[a]] = table[row, a]``. Pass ``table_wp=None``
        to revert to uniform. Re-call after the bank is rebuilt (resample)."""
        ok = table_wp is not None and int(n_rows) > 0 and ctrl_qpa_wp is not None
        self._cf_table_wp = table_wp if ok else None
        self._cf_nrows    = int(n_rows) if ok else 0
        self._cf_qpa_wp   = ctrl_qpa_wp if ok else None

    def apply_all(self) -> None:
        """Randomize finger qpos / ctrl on every parallel world."""
        self._launch(apply_to_all=1, mask=None)

    def apply_masked(self, mask_wp) -> None:
        """Randomize only the worlds with ``mask_wp[w] == 1``.

        ``mask_wp`` must be a ``wp.array(dtype=int, ndim=1)`` of length
        :attr:`NWORLD` (typically the handler's ``done_mask_wp``).
        """
        self._launch(apply_to_all=0, mask=mask_wp)

    def reseed(self, seed: int) -> None:
        """Update the GPU seed buffer (also bumps every later launch)."""
        self._seed_value = int(seed) + 1000 * self.handler_idx
        self._seed_wp.assign(np.array([self._seed_value], dtype=np.int32))

    # ── Internal ────────────────────────────────────────────────────────

    def _launch(self, apply_to_all: int, mask) -> None:
        # advance the seed so consecutive resets sample different values.
        self._seed_value += 1
        self._seed_wp.assign(np.array([self._seed_value], dtype=np.int32))

        # When apply_to_all=1 the kernel ignores mask, but we still need to
        # supply *some* int wp.array for the parameter. Reuse the workspace
        # as a sentinel zero array of the right shape.
        if mask is None:
            mask = self._dummy_mask_wp()

        d = self.sampled_env.d

        # Collision-free bank attached → gather a self-collision-free row per
        # world instead of the uniform op-list sample.
        if self._cf_table_wp is not None and self._cf_nrows > 0:
            wp.launch(
                _gather_collision_free_init_kernel, dim=self.NWORLD,
                inputs=[
                    self._seed_wp, self._cf_table_wp, int(self._cf_nrows),
                    int(self.n_ctrl), self._cf_qpa_wp,
                    d.qpos, mask, int(apply_to_all),
                ],
            )
            return

        wp.launch(
            _finger_init_kernel, dim=self.NWORLD,
            inputs=[
                self._seed_wp,
                int(self.n_finger),
                self._finger_qpa_wp,
                self._jnt_low_wp,  self._jnt_high_wp,
                self._multiplier_wp,
                int(self._n_overrides),
                self._override_idx_wp, self._override_low_wp, self._override_high_wp,
                int(self._n_couples),
                self._couple_src_wp, self._couple_dst_wp, self._couple_weight_wp,
                int(self.n_ctrl),
                self._ctrl_qpos_idx_wp,
                self._workspace_wp,
                d.qpos, d.ctrl,
                mask,
                int(apply_to_all),
            ],
        )

    def _dummy_mask_wp(self):
        if not hasattr(self, "_dummy_mask_cache"):
            self._dummy_mask_cache = wp.zeros((self.NWORLD,), dtype=int,
                                              device=self.device)
        return self._dummy_mask_cache

    # ── Resolution helpers ──────────────────────────────────────────────

    @staticmethod
    def _resolve_finger_qpa_and_range(
        sampled_env, hand_util,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve the qpos addrs + (low, high) for every finger DOF.

        Strategy: walk the joints declared on the wrist body's children
        in body-id order (matching the way the original JAX code lays out
        ``q_init = [wrist_p, wrist_R, finger_init, obj_pose]``). Skip the
        wrist's own freejoint (the first 7 qpos slots).

        For coupled-tendon hands we still expose every joint DOF
        individually — ``HandFingerInitializer`` writes raw qpos values,
        and the post-init forward pass enforces the tendon equality
        constraint.
        """
        mjm = sampled_env.mjm
        n_finger = int(hand_util.finger_link_num)

        # Find the wrist body id; finger joints are everything in its
        # subtree except the wrist's own freejoint.
        wrist_bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY,    # type: ignore
                                       hand_util.rh_wrist_base_name)
        if wrist_bid < 0:
            raise ValueError(
                f"wrist body '{hand_util.rh_wrist_base_name}' not in model"
            )

        finger_qpa: List[int] = []
        for jid in range(int(mjm.njnt)):
            jt   = int(mjm.jnt_type[jid])
            jbid = int(mjm.jnt_bodyid[jid])
            # is jbid a descendant of wrist_bid (excluding wrist itself)?
            cur = jbid
            in_subtree = False
            while cur > 0:
                if cur == wrist_bid and cur != jbid:
                    in_subtree = True
                    break
                cur = int(mjm.body_parentid[cur])
            if not in_subtree:
                continue
            # skip free joints (mjJNT_FREE = 0); we only want hinge / slide.
            if jt == int(mujoco.mjtJoint.mjJNT_FREE):                  # type: ignore
                continue
            finger_qpa.append(int(mjm.jnt_qposadr[jid]))

        if len(finger_qpa) != n_finger:
            # Fall back to the first ``n_finger`` non-wrist hinge/slide
            # joints if subtree walk under-counts (some hands attach
            # fingers in unusual hierarchies).
            fallback: List[int] = []
            for jid in range(int(mjm.njnt)):
                jt = int(mjm.jnt_type[jid])
                if jt == int(mujoco.mjtJoint.mjJNT_FREE):              # type: ignore
                    continue
                fallback.append(int(mjm.jnt_qposadr[jid]))
            if len(fallback) >= n_finger:
                finger_qpa = fallback[:n_finger]

        finger_qpa_np = np.asarray(finger_qpa, dtype=np.int32)

        # Build per-DOF range from joint range; default to ±π for unlimited
        # joints (the legacy JAX clamp-and-scale logic narrows these via
        # the ``multiplier`` table anyway).
        jnt_range = np.asarray(mjm.jnt_range, dtype=np.float32)        # (njnt, 2)
        jnt_limited = np.asarray(mjm.jnt_limited, dtype=np.int32)       # (njnt,)

        # Walk the same joint order to fetch ranges.
        # Re-walk: collect joint ids in the same order as finger_qpa.
        # (Could be optimised; n_finger is small.)
        ordered_jids: List[int] = []
        for jid in range(int(mjm.njnt)):
            qpa = int(mjm.jnt_qposadr[jid])
            if qpa in finger_qpa_np:
                ordered_jids.append(jid)
                if len(ordered_jids) == len(finger_qpa_np):
                    break
        # Map qpa → jid for proper ordering.
        qpa_to_jid = {int(mjm.jnt_qposadr[j]): j for j in ordered_jids}

        jnt_low  = np.zeros(n_finger, dtype=np.float32)
        jnt_high = np.zeros(n_finger, dtype=np.float32)
        for i, qpa in enumerate(finger_qpa_np):
            jid = qpa_to_jid.get(int(qpa))
            if jid is None or jnt_limited[jid] == 0:
                jnt_low[i]  = -np.pi
                jnt_high[i] =  np.pi
            else:
                jnt_low[i]  = float(jnt_range[jid, 0])
                jnt_high[i] = float(jnt_range[jid, 1])
        return finger_qpa_np, jnt_low, jnt_high

    @staticmethod
    def _build_ctrl_qpos_idx(
        mjm, finger_qpa: np.ndarray, hand_util,
    ) -> np.ndarray:
        """Build ``ctrl_qpos_idx[a]`` = index *into ``finger_qpa``* that
        ``ctrl[a]`` should mirror.

        Resolution strategy:
            1. If ``mjm.actuator_trntype[a] == mjTRN_JOINT`` use the joint's
               qpos addr directly.
            2. Otherwise (tendon / site actuator) consult
               ``hand_util.ctrl_joint_idx_array`` which lists the *primary*
               finger index driven by each ctrl. Return ``-1`` when the
               actuator has no scalar qpos counterpart (then ctrl is left
               at zero — matches the legacy JAX path which only set qpos).
        """
        n_ctrl = int(mjm.nu)
        ctrl_qpos_idx = np.full(n_ctrl, -1, dtype=np.int32)

        qpa_set = {int(q): i for i, q in enumerate(finger_qpa)}
        ctrl_joint_idx_array = getattr(hand_util, "ctrl_joint_idx_array", None)

        for a in range(n_ctrl):
            tt = int(mjm.actuator_trntype[a])
            if tt == int(mujoco.mjtTrn.mjTRN_JOINT):                    # type: ignore
                jid = int(mjm.actuator_trnid[a, 0])
                if 0 <= jid < int(mjm.njnt):
                    qpa = int(mjm.jnt_qposadr[jid])
                    if qpa in qpa_set:
                        ctrl_qpos_idx[a] = int(qpa_set[qpa])
                continue

            # Non-joint actuator → fall back to ``ctrl_joint_idx_array``.
            if ctrl_joint_idx_array is None:
                continue
            if a >= len(ctrl_joint_idx_array):
                continue
            finger_idx = int(ctrl_joint_idx_array[a])
            if 0 <= finger_idx < len(finger_qpa):
                ctrl_qpos_idx[a] = int(finger_idx)

        return ctrl_qpos_idx
