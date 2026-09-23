"""Colour / print helpers and a depth → xyz unprojection."""
from __future__ import annotations

import numpy as np

from grit.util.utils import (  # noqa: F401  (re-exported)
    print_red, print_yellow, print_green, print_blue,
)


def get_colors(n_color=10, cmap_name="gist_rainbow", alpha=1.0):
    """``n_color`` RGBA tuples spread over a matplotlib colormap."""
    import matplotlib.pyplot as plt   # lazy: matplotlib is only needed for viewer colours
    rgba = plt.get_cmap(cmap_name)(np.linspace(0, 1, n_color))
    rgba[:, 3] = alpha
    return [tuple(r) for r in rgba]


def meters2xyz(depth_img, cam_matrix):
    """Metric depth image → ``(H, W, 3)`` points in the viewer camera frame.

    Camera frame convention: +X forward (depth), +Y left, +Z up — the same
    frame :meth:`MuJoCoParser.get_T_viewer` returns.
    """
    fx, cx = cam_matrix[0][0], cam_matrix[0][2]
    fy, cy = cam_matrix[1][1], cam_matrix[1][2]
    h, w = depth_img.shape[:2]
    idx = np.indices((h, w), dtype=np.float32).transpose(1, 2, 0)
    z = depth_img
    x = (idx[..., 1] - cx) * z / fx
    y = (idx[..., 0] - cy) * z / fy
    return np.stack([z, -x, -y], axis=-1)
