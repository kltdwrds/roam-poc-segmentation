"""PLY export helpers for the three-pane viewer."""

from __future__ import annotations
import numpy as np
from plyfile import PlyData, PlyElement


def write_colored_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write an XYZ + RGB point cloud. rgb expected uint8 in [0,255]."""
    assert xyz.shape[1] == 3 and rgb.shape[1] == 3 and len(xyz) == len(rgb)
    rgb = rgb.astype(np.uint8)
    verts = np.empty(
        len(xyz),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(path)


def subsample_to_budget(xyz: np.ndarray, *arrays, max_points: int = 1_500_000):
    """Random subsample so the resulting PLY stays well under 50 MB.

    A colored PLY costs ~15 bytes/point; 1.5M points ≈ 22 MB on disk.
    """
    if len(xyz) <= max_points:
        return (xyz,) + arrays
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xyz), size=max_points, replace=False)
    return (xyz[idx],) + tuple(a[idx] for a in arrays)
