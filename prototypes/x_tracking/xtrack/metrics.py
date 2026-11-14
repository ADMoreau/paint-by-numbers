"""How good is an estimated pose?"""

from __future__ import annotations

import numpy as np

from .camera import project


def overlay_error_mm(H_est: np.ndarray, H_gt: np.ndarray, width: int, height: int, nx: int = 9, ny: int = 7) -> tuple[float, float]:
    """How far the overlay lands from where it should, in canvas mm, over a grid
    across the whole camera view. Returns (mean, max)."""
    us, vs = np.meshgrid(np.linspace(0.05 * width, 0.95 * width, nx), np.linspace(0.05 * height, 0.95 * height, ny))
    pix = np.column_stack([us.ravel(), vs.ravel()])
    err = np.hypot(*(project(np.linalg.inv(H_est), pix) - project(np.linalg.inv(H_gt), pix)).T)
    return float(err.mean()), float(err.max())


def true_ids(dets, canvas, H_gt) -> list:
    """For each detection, the id of the real mark it sits on, or None (a false detection)."""
    proj = project(H_gt, np.array([m.center for m in canvas.marks]))
    out = []
    for d in dets:
        dist = np.hypot(*(proj - d.center).T)
        j = int(np.argmin(dist))
        out.append(canvas.marks[j].id if dist[j] < 6 + 0.3 * d.size else None)
    return out


def identity_ok(matches: dict, truth: list) -> bool:
    """Every matched detection was given its real mark's id."""
    return all(truth[i] == mid for i, mid in matches.items())
