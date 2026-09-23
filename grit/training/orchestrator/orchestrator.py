import os
import random
from typing import Callable, Optional, Sequence, Tuple
import numpy as np
from hydra import initialize, compose
from omegaconf import OmegaConf, ListConfig
from pathlib import Path
from etils import epath

from grit.util.utils import print_red
from grit.util import hand_utils
from grit.training.orchestrator.base import SingleHandSubEnv

_grit_dir = Path(__file__).resolve().parent  # walk up from THIS module, not cwd

def resolve_project_root() -> Path:
    for base in (_grit_dir, *_grit_dir.parents):
        if (
            (base / "README.md").is_file()
            and (base / "grit").is_dir()
            and (base / "config").is_dir()
        ):
            return base
    raise RuntimeError(
        f"Could not find project root from {_grit_dir}. "
        "cd into the repository and restart the kernel."
    )

PROJECT_ROOT = resolve_project_root()
home_dir = str(PROJECT_ROOT)

class Env_Orchestrator:
    def __init__(
        self,
        config_name: Optional[str] = None,
        test_mode:   bool = False,
        verbose:     bool = False,
        *,
        overall_cfg: Optional[OmegaConf] = None,
        overrides:   Optional[Sequence[str]] = None,
        obj_idxs:    Optional[Sequence[int]] = None,
    ):
        """Build an environment orchestrator from either a Hydra config name
        OR a pre-loaded ``OmegaConf`` config.

        ``config_name``  — name of a YAML under ``config/training/`` (no
        extension). Loaded via Hydra ``compose(...)``.
        ``overall_cfg``  — already-resolved ``DictConfig``. Use this for
        inference-time rebuild from a snapshot (``SAVE_DIR/config.yaml``)
        so the env always matches the checkpoint that produced it, even
        if the named YAML was edited after training.
        ``overrides``    — Hydra-style override strings (e.g.
        ``["nworld=32", "Training.n_obj=3"]``). Only used when loading
        via ``config_name``; ignored on the ``overall_cfg`` path since
        that one is already resolved.
        ``obj_idxs``     — **candidate pool** of indices into
        ``obj_xml_path_lst`` to materialise as variants (fixed dataset
        slice). ``None`` (default) → legacy random sampling from the full
        dataset. Precedence at build time: constructor arg > YAML key
        ``obj_idxs`` (top-level list of ints) > random sample.

        Pool size vs. ``n_sub_env`` (variant count from YAML — always
        authoritative):
            * ``len(pool) <  n_sub_env``  → ``AssertionError`` at build
              time (not enough candidates to fill the requested variants).
            * ``len(pool) == n_sub_env``  → use the whole pool verbatim.
            * ``len(pool) >  n_sub_env``  → random subsample of size
              ``n_sub_env`` from the pool (without replacement).

        Useful for reproducible debugging (``obj_idxs=[7]`` pins one
        object), curated regression sweeps (a small whitelist of objects
        the policy should master), and per-object evaluation runs.

        Exactly one of ``config_name`` / ``overall_cfg`` must be provided.
        """
        if (config_name is None) == (overall_cfg is None):
            raise ValueError(
                "Env_Orchestrator: provide exactly one of "
                "`config_name` (named YAML) or `overall_cfg` (pre-loaded "
                "DictConfig)."
            )

        if overall_cfg is not None:
            # Pre-loaded path — typically a snapshot from SAVE_DIR/config.yaml.
            if overrides:
                raise ValueError(
                    "Env_Orchestrator: `overrides` only applies to the "
                    "`config_name` path. Strip or apply them before "
                    "passing `overall_cfg`."
                )
            self.config_name = config_name or "<provided overall_cfg>"
            self.overall_cfg = overall_cfg 
        else:
            self.config_name = config_name
            # Hydra ``initialize`` requires a path relative to THIS file.
            # Moved from ``grit/util/`` → ``grit/training/orchestrator/`` so
            # the project-root climb is now three ``..`` levels, not two.
            config_path = f"../../../config/training"
            with initialize(version_base='1.3', config_path=config_path):
                self.overall_cfg = compose( 
                    config_name=config_name,
                    overrides=list(overrides) if overrides else [],
                )
        
        self.home_dir = home_dir 
        self.ri_package_asset_dir = epath.Path(home_dir) / "asset"
        self.ri_package_hand_asset_dir = self.ri_package_asset_dir / "dextrous_hand"

        # hand name, type list  
        self.using_hand_name_list:list = self.overall_cfg.using_hand_name_list              # type: ignore
        self.leader_hand_name:list = self.overall_cfg.get("leader_hand_name", [])           # type: ignore
        self.follower_hand_name:list = self.overall_cfg.get("follower_hand_name", [])       # type: ignore
        self.do_leader_follower:bool = self.overall_cfg.get("do_leader_follower", False)    # type: ignore
        
        self.hand_type_list:list = self.overall_cfg.hand_type_list                          # type: ignore
        # self.task_type:list = self.overall_cfg.task_type                                    # type: ignore

        # Create a hand_util for each using_hand_name and attach it as an attribute.
        if isinstance(self.using_hand_name_list, str):
            self.hand_names = [self.using_hand_name_list]
        else:
            self.hand_names = list(self.using_hand_name_list)
        
        ############# Build per hand type and hand name. ###############
        self.hand_xml_dict = {} 
        for hand_name in self.hand_names:
            for hand_type in self.hand_type_list:
                util: hand_utils.HandUtils = hand_utils.HandUtils(hand_name, hand_type, home_dir)
                attr_name: str = f"{hand_name}_{hand_type}_util"
                setattr(self, attr_name, util)    
                self.hand_xml_dict[f"{hand_name}_{hand_type}"] = getattr(self, f"{hand_name}_{hand_type}_util").hand_xml_path
        self.hand_type_name_list = list(self.hand_xml_dict.keys()) 

        ################## Object Dataset Loads ##################
        # Pass project_root so dataset paths resolve correctly regardless
        # of cwd — no chdir(notebook/hand) trick needed in callers.
        self.obj_path_dir_lst, self.obj_name_lst, self.non_contact_obj_name_lst, self.obj_xml_path_lst = hand_utils.get_obj_path_dir_lst(
            self.overall_cfg.Dataset, project_root=home_dir,                # type: ignore
        )
        self.n_obj = self.overall_cfg.Training.n_obj                        # type: ignore

        ############ BPS settings ##############
        self.bps_data_path = self.overall_cfg.Training.bps_data_path        # type: ignore
        self.n_bps = self.overall_cfg.Training.n_bps                        # type: ignore
        self.use_bps = self.overall_cfg.Training.use_bps                    # type: ignore
        self.use_bps_2x = self.overall_cfg.Training.use_bps_2x              # type: ignore

        self.bps_data_path = os.path.join(home_dir + "/data", self.bps_data_path) 

        try:
            self.bps_data = np.load(self.bps_data_path)['bps_data'] 
        except:
            print_red(f"BPS data file not found: {self.bps_data_path}")
            self.bps_data = np.random.uniform(-0.075, 0.15, size=(1024, 3))
        ########################################       
        
        # RL env parameters
        ########################################       
        self.CON_PER_ENV = self.overall_cfg.con_per_env                     # type: ignore
        self.NJMAX_MUILTPLY = self.overall_cfg.njmax_muiltiply              # type: ignore
        self.N_STEPS = self.overall_cfg.n_steps                             # type: ignore
        self.ACTION_REPEAT = self.overall_cfg.action_repeat                 # type: ignore    

        # ── Pinned object-index pool (optional) ──────────────────────────────
        # Precedence (highest first):
        #   1) constructor kwarg ``obj_idxs``
        #   2) YAML top-level ``obj_idxs`` (a list of ints in the config file)
        #   3) None → fall back to random sampling from the full
        #      ``obj_xml_path_lst`` at build time (legacy).
        #
        # ``obj_idxs`` is treated as a **candidate pool**, not the exact
        # variant list. ``N_SUB_ENV`` (= yaml ``n_sub_env``) stays
        # authoritative as the variant count; ``_resolve_obj_indices`` then:
        #   * ``len(pool) <  N_SUB_ENV`` → AssertionError (not enough cands)
        #   * ``len(pool) == N_SUB_ENV`` → use the whole pool verbatim
        #   * ``len(pool) >  N_SUB_ENV`` → random subsample without
        #                                  replacement
        _yaml_idxs = self.overall_cfg.get("obj_idxs", None)                 # type: ignore
        if obj_idxs is not None:
            self.obj_idxs: Optional[list[int]] = [int(i) for i in obj_idxs]
        elif _yaml_idxs is not None:
            self.obj_idxs = [int(i) for i in _yaml_idxs]
        else:
            self.obj_idxs = None

        if self.obj_idxs is not None:
            if not self.obj_idxs:
                raise ValueError(
                    "Env_Orchestrator: obj_idxs is set but empty. "
                    "Use ``None`` for random sampling."
                )
            _total = len(self.obj_xml_path_lst)
            _bad   = [i for i in self.obj_idxs if not (0 <= i < _total)]
            if _bad:
                raise ValueError(
                    f"Env_Orchestrator: obj_idxs contains out-of-range entries "
                    f"{_bad} (valid range: [0, {_total})). Check the active "
                    f"Dataset section of your YAML — total objects depends on "
                    f"which datasets have ``use: True``."
                )

        # ── did-you-mean-obj_idxs guard ────────────────────────────────────
        # ``obj_idx`` (singular scalar) is a LEGACY key NOT used by the current
        # training/orchestrator pipeline — object selection is driven SOLELY by
        # ``obj_idxs`` (plural pool). A frequent mistake is ``obj_idx=[...]``
        # intending the pool; that silently leaves obj_idxs=None → random
        # sampling from the whole dataset. Catch the list case loudly.
        _obj_idx_singular = self.overall_cfg.get("obj_idx", None)           # type: ignore
        if isinstance(_obj_idx_singular, (list, ListConfig)):
            raise ValueError(
                f"Env_Orchestrator: `obj_idx` (singular) is a scalar LEGACY key not "
                f"used by the current pipeline, but you passed a list "
                f"({list(_obj_idx_singular)}). Did you mean `obj_idxs` (plural)? "
                f"Object selection uses `obj_idxs` only → rename the override to "
                f"`obj_idxs=[...]` (with n_sub_env deciding how many are drawn)."
            )

        # ``N_SUB_ENV`` always sourced from the yaml (or test_mode short
        # circuit); the pool only affects WHICH indices are picked at
        # build time, not HOW MANY variants are built.
        if test_mode:
            self.N_SUB_ENV = 1
            self.NWORLD    = 1
        else:
            self.N_SUB_ENV = int(self.overall_cfg.n_sub_env)                # type: ignore
            self.NWORLD    = int(self.overall_cfg.nworld)                   # type: ignore

        # ``Training.object_source: mixed`` draws only ``mesh_frac`` of the
        # variants from the mesh pool (the rest are procedural), so the pool
        # only has to cover that share — see primitive_object.make_mixed_obj_spec_provider.
        self._object_source = str(
            self.overall_cfg.Training.get("object_source", "dataset") or "dataset").lower()  # type: ignore
        if self.obj_idxs is not None and self._object_source == "mixed":
            _po   = self.overall_cfg.Training.get("primitive_object", None)              # type: ignore
            _mf   = float((_po.get("mesh_frac", 0.5) if _po is not None else 0.5))
            _need = int(round(self.N_SUB_ENV * _mf))
            assert len(self.obj_idxs) >= _need, (
                f"Env_Orchestrator(mixed): obj_idxs pool has {len(self.obj_idxs)} entries "
                f"but n_sub_env={self.N_SUB_ENV} × mesh_frac={_mf} = {_need} mesh variants "
                f"are requested.")
        elif self.obj_idxs is not None and len(self.obj_idxs) < self.N_SUB_ENV:
            # Fail fast — surfacing this at build time gives a less
            # informative stack so we assert here while we still know the
            # caller's intent.
            assert len(self.obj_idxs) >= self.N_SUB_ENV, (
                f"Env_Orchestrator: obj_idxs pool has {len(self.obj_idxs)} "
                f"entries but n_sub_env={self.N_SUB_ENV} variants are "
                f"requested. Pool must contain ≥ n_sub_env candidates. "
                f"obj_idxs={self.obj_idxs}"
            )

        self.verbose = verbose

        # Set on the first ``build_sub_env`` / ``build_leader_follower_env``
        # call so ``resample_objects`` can re-dispatch to the SAME builder
        # (same hand / n_obj / for_inference / collision type) with a fresh
        # object draw. ``None`` until the env is built once.
        self._resample_spec: Optional[Tuple[str, dict]] = None

    # ──────────────────────────────────────────────────────────────────────
    # Object resampling gate
    # ──────────────────────────────────────────────────────────────────────
    @property
    def objects_fully_pinned(self) -> bool:
        """True when the variant set is fully determined by ``obj_idxs``.

        A rebuild would reproduce the exact same variants, so per-iteration
        object resampling is a no-op and is skipped:

          * ``obj_idxs is None``            → NOT pinned (random from dataset).
          * ``len(obj_idxs) == n_sub_env``  → PINNED (variants deterministic).
          * ``len(obj_idxs) >  n_sub_env``  → NOT pinned (a fresh ``n_sub_env``
                                              subset is drawn from the pool each
                                              rebuild).
        """
        if getattr(self, "_object_source", "dataset") in ("primitive", "mixed"):
            return False          # procedural objects are re-drawn on every rebuild
        return self.obj_idxs is not None and len(self.obj_idxs) == self.N_SUB_ENV

    def resample_objects(self) -> bool:
        """Re-sample objects and rebuild every sub-env's variants.

        Gives each world a freshly-drawn object set WITHOUT changing hand /
        ``n_obj`` / ``obs_dim`` / ``action_dim`` — so the policy, optimizer and
        rollout buffer built over the previous env stay valid (the caller must
        still re-wire any handler-derived state, e.g. ``prev_obs``).

        No-op returning ``False`` when :attr:`objects_fully_pinned` (the
        variants are already deterministic). Otherwise re-dispatches to the
        SAME builder used at construction with ``obj_idxs`` omitted, so
        ``_resolve_obj_indices`` re-samples fresh:

          * ``obj_idxs is None``          → uniform sample of ``n_sub_env``
                                            indices over the whole dataset.
          * ``len(obj_idxs) > n_sub_env`` → fresh subsample of ``n_sub_env``
                                            from the pinned candidate pool.

        EXPENSIVE — a full MjModel recompile + ``mjwarp.put_model`` /
        ``make_data`` + CUDA-graph re-capture per hand per call. Returns
        ``True`` when a rebuild happened.
        """
        if self.objects_fully_pinned:
            return False
        if self._resample_spec is None:
            raise RuntimeError(
                "resample_objects() called before build_sub_env() / "
                "build_sub_env(). Build the env once first."
            )
        # Release the PREVIOUS sub-envs (warp Model/Data + captured CUDA graphs)
        # BEFORE the rebuild allocates the new ones. They form reference cycles
        # (sub-env ↔ capture graph), so plain refcounting never frees them — an
        # explicit gc pass is required. Without this they stay resident through
        # the heavy put_model / make_data / graph-capture below → transient 2×
        # GPU → warp async-pool fragmentation → collision OOM after a handful of
        # rebuilds. The caller (train_loop) must have already dropped its own
        # env / handler references, or these objects won't actually be freed.
        import gc
        self.env_list = []
        gc.collect()
        try:
            import warp as wp
            wp.synchronize_device()
        except Exception:
            pass
        kind, kw = self._resample_spec
        if kind in ("dual_hand", "leader_follower"):
            raise NotImplementedError("dual-hand / leader-follower not included in grit_share")
        else:
            self.build_sub_env(**kw)
        return True

    # ──────────────────────────────────────────────────────────────────────
    # Object-index resolution (pinned vs. random)
    # ──────────────────────────────────────────────────────────────────────
    def _resolve_obj_indices(
        self,
        n_sub_env: int,
        obj_idxs_override: Optional[Sequence[int]] = None,
    ) -> list[int]:
        """Resolve a length-``n_sub_env`` list of ``obj_xml_path_lst`` indices
        to materialise as variants.

        Precedence:
            1) ``obj_idxs_override`` — per-call candidate pool (e.g. caller
               wants a different slice for evaluation than training).
            2) ``self.obj_idxs``     — pool from constructor / YAML.
            3) None pool             → uniform random sample of size
               ``n_sub_env`` over the full dataset (legacy behaviour).

        Pool sizing rules when a pool is supplied:
            * ``len(pool) <  n_sub_env`` → ``AssertionError`` (not enough
              candidates for the requested variant count).
            * ``len(pool) == n_sub_env`` → use the whole pool verbatim
              (deterministic).
            * ``len(pool) >  n_sub_env`` → random subsample of size
              ``n_sub_env`` without replacement.

        Out-of-range pool entries raise ``ValueError`` with the valid
        range so callers immediately see the active-dataset size.
        """
        if obj_idxs_override is not None:
            pool   = [int(i) for i in obj_idxs_override]
            source = "per-call override"
        elif self.obj_idxs is not None:
            pool   = list(self.obj_idxs)
            source = "Env_Orchestrator.obj_idxs (constructor / YAML)"
        else:
            return hand_utils.get_sampled_indices(n_sub_env, len(self.obj_xml_path_lst))

        if not pool:
            raise ValueError(
                f"Env_Orchestrator: obj_idxs ({source}) is empty. "
                f"Use ``None`` for random sampling from the full dataset."
            )
        total = len(self.obj_xml_path_lst)
        bad = [i for i in pool if not (0 <= i < total)]
        if bad:
            raise ValueError(
                f"Env_Orchestrator: obj_idxs ({source}) contains "
                f"out-of-range entries {bad} (valid range: [0, {total})). "
                f"Check the active Dataset section of your YAML."
            )

        assert len(pool) >= n_sub_env, (
            f"Env_Orchestrator: obj_idxs pool ({source}) has {len(pool)} "
            f"entries but n_sub_env={n_sub_env} variants are requested. "
            f"Pool must contain ≥ n_sub_env candidates. pool={pool}"
        )

        if len(pool) == n_sub_env:
            idxs = pool
            mode = "use-all"
        else:
            # Larger pool than n_sub_env → random subsample without
            # replacement. ``random.sample`` honours the module-level
            # ``random.seed`` set in client code (e.g. the notebook
            # seeds it to 42 right before constructing the orchestrator),
            # so reproducibility is preserved when the caller wants it.
            idxs = random.sample(pool, n_sub_env)
            mode = f"subsample {n_sub_env} from {len(pool)}"

        if self.verbose:
            print_red(
                f"[Env_Orchestrator] obj_idxs pool ({source}) "
                f"{mode} → sampled={idxs}  (total dataset={total})"
            )
        return idxs

    # Single-hand & heterogeneous-object setup.
    def build_sub_env(
        self,
        with_mjwarp:bool = True,
        for_inference:bool = False,
        geom_collision_type:str = "mesh",
        obj_idxs: Optional[Sequence[int]] = None,
        dual_hand: Optional[bool] = None,
        obj_spec_provider: Optional[Callable[[int], tuple]] = None,
    ):
        """Build the sub-env(s).

        ``obj_spec_provider`` — optional ``f(n_sub_env) -> (obj_spec_lst,
        per_spec_ngeom, meta)`` replacing the **dataset** object source with
        caller-supplied ``MjSpec`` objects (e.g. the procedurally generated
        primitives in :mod:`grit.util.primitive_object`). When given, no object
        XML is sampled from ``obj_xml_path_lst`` and ``obj_idxs`` is ignored;
        everything downstream (skeleton / variants / warp per-world override /
        PCD cache / pose-init kernels) is unchanged. The provider is stored in
        ``_resample_spec``, so ``resample_objects()`` calls it again for a fresh
        object set. The returned ``meta`` is kept on ``self.obj_spec_meta``.
        """
        # Remember how this env was built so ``resample_objects`` can rebuild
        # it with a fresh object draw (``obj_idxs`` deliberately excluded — a
        # resample re-samples via ``_resolve_obj_indices``).
        self._resample_spec = ("single_hand", dict(
            with_mjwarp=with_mjwarp, for_inference=for_inference,
            geom_collision_type=geom_collision_type,
            obj_spec_provider=obj_spec_provider,
        ))
        if self.overall_cfg.Training.with_table:                            # type: ignore
            self.floor_xml_path = [
                self.ri_package_asset_dir / "floor/floor_simple_white.xml",
                self.ri_package_asset_dir / "object/base_table.xml",
            ]
            self.table_height = self.overall_cfg.Training.table_height      # type: ignore
        else:
            self.floor_xml_path = [self.ri_package_asset_dir / "floor/floor_simple_white.xml"]
            self.table_height = 0.0
        self.use_simple = False if self.overall_cfg.Dataset.Objaverse.use else True # type: ignore

        self.env_list = []

        self.geom_collision_type = geom_collision_type
        if obj_spec_provider is not None:
            # ── Caller-supplied objects (procedural primitives, …) ──────────
            # Same contract as ``build_obj_spec_lst``; the dataset is not
            # touched at all, so ``sampled_obj_indices`` is empty.
            self.obj_spec_lst, self.per_spec_ngeom, self.obj_spec_meta = \
                obj_spec_provider(self.N_SUB_ENV)
            self.sampled_obj_indices = []
            assert len(self.obj_spec_lst) == self.N_SUB_ENV, (
                f"obj_spec_provider returned {len(self.obj_spec_lst)} specs but "
                f"n_sub_env={self.N_SUB_ENV} variants are requested.")
            print_red(f"[build_sub_env] obj_spec_provider → {len(self.obj_spec_lst)} "
                      f"procedural object spec(s), per_spec_ngeom={self.per_spec_ngeom}")
        else:
            # obj_spec_lst is built and managed by the Orchestrator.
            # ``_resolve_obj_indices`` returns the pinned indices when
            # ``self.obj_idxs`` (or per-call ``obj_idxs``) is set, otherwise it
            # falls back to ``hand_utils.get_sampled_indices`` (random).
            sampled_indices = self._resolve_obj_indices(self.N_SUB_ENV, obj_idxs_override=obj_idxs)
            self.sampled_obj_indices = sampled_indices            # cached for snapshot / introspection
            self.obj_spec_meta       = None
            obj_names        = [self.obj_name_lst[i] for i in sampled_indices]
            obj_xml_path_set = [self.obj_xml_path_lst[i] for i in sampled_indices]
            print_red(f"obj_xml_path_set : {obj_xml_path_set}")
            print_red(f"obj_names : {obj_names}")
            print_red(f"sampled_obj_indices : {sampled_indices}")

            print_red(f"[build_sub_env] geom_collision_type={geom_collision_type!r}")
            self.obj_spec_lst, self.per_spec_ngeom, _ = hand_utils.build_obj_spec_lst(
                obj_xml_path_set, obj_names,
                friction=self.overall_cfg.Training.friction,
                use_simple=self.use_simple,
                geom_collision_type=geom_collision_type,
                rename_body_name=self.overall_cfg.Training.obj_name,
                sim_dt=self.overall_cfg.sim_dt,   # solref timeconst = max(0.01, 2×dt)
                verbose=self.verbose or geom_collision_type == "sdf",
            )

        # ── Dual-hand default: when BOTH hand types of an embodiment are requested
        # (``hand_type_list: [right, left]``) build ONE shared-object scene per hand
        # (:class:`DualHandSubEnv`) and expose its per-hand views as env_list — the
        # bimanual counterpart of the single-hand path. ``dual_hand=False`` (arg or
        # yaml ``dual_hand: false``) forces the legacy one-scene-per-hand build.
        if dual_hand is None:
            dual_hand = bool(self.overall_cfg.get("dual_hand", True))
        if dual_hand and len(self.hand_type_list) >= 2:
            raise NotImplementedError(
                "grit_share ships the single-hand grasping stack only — dual-hand / "
                "leader-follower scenes are not included. Use hand_type_list: [right].")

        # Create an env for every hand in hand_type_name_list; obj_spec_lst is shared.
        self.scene_list = []
        for hand_key in self.hand_type_name_list:
            hand_util_ = getattr(self, f"{hand_key}_util")
            env_class  = SingleHandSubEnv(self.overall_cfg, hand_util_, self.verbose, for_inference=for_inference)
            env_class.reset(self.obj_spec_lst, self.per_spec_ngeom, self.n_obj, with_mjwarp=with_mjwarp)
            self.env_list.append(env_class)
            self.scene_list.append(env_class)
            print_red(f"[build_sub_env] {hand_key} env built. variants={len(env_class.variants)}")
