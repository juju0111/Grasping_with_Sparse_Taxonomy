"""grit.util.sim_core — self-contained simulation helpers.

This package replaces the lab-internal ``ri_motion_v5_package`` that the
original research codebase depended on. Only the surface actually used by
the grasping stack is provided:

* :mod:`transforms`  — small SE(3) helpers (``pr2t``, ``t2pr``, ``rpy2r``, ``r2quat`` …)
* :mod:`utils`       — ``get_idxs``, ``trim_scale``, ``get_monitor_size``,
                       ``TicToc``, ``SimpleTimer``, ``farthest_point_sampling``
* :mod:`viz`         — colour / print helpers, ``meters2xyz``
* :mod:`mjcf`        — ``merge_mjcfs`` (write a scene XML that ``<include>``s several MJCFs)
* :mod:`viewer`      — ``MuJoCoMinimalViewer`` (GLFW window + MjvScene marker/overlay queue)
* :mod:`parser`      — ``MuJoCoParser`` (CPU MjModel/MjData wrapper: step / forward /
                       pose getters / viewer lifecycle / marker plotting)

Everything is plain ``mujoco`` + ``glfw`` + ``numpy``.
"""
from .transforms import (  # noqa: F401
    t2p, t2r, t2pr, pr2t, p2t, r2t, rpy2r, r2quat, quat2r,
    get_R_from_twopoints, np_uv,
)
from .utils import (  # noqa: F401
    get_idxs, trim_scale, get_monitor_size, TicToc, SimpleTimer,
    farthest_point_sampling, indent_xml,
)
from .viz import (  # noqa: F401
    get_colors, print_red, print_yellow, print_green, print_blue, meters2xyz,
)
from .mjcf import merge_mjcfs  # noqa: F401
from .viewer import MuJoCoMinimalViewer, MinimalCallbacks  # noqa: F401
from .parser import MuJoCoParser  # noqa: F401
