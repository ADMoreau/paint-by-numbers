"""Pictures of an alignment."""

from __future__ import annotations

import cv2
import numpy as np

from . import align as al


def warp_target(target: np.ndarray, model: al.EdgeModel, fit, size) -> tuple[np.ndarray, np.ndarray]:
    """The target resampled into work pixels through ``fit`` (work -> model), and where it lands (mask)."""
    w, h = size
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    q = np.column_stack([xs.ravel(), ys.ravel()])
    p = (fit(q) + 0.5) / model.f - 0.5  # target px
    mx = p[:, 0].reshape(h, w).astype(np.float32)
    my = p[:, 1].reshape(h, w).astype(np.float32)
    out = cv2.remap(target, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    th, tw = target.shape[:2]
    mask = (mx >= 0) & (my >= 0) & (mx <= tw - 1) & (my <= th - 1)
    return out, mask


def edges_in_work(model: al.EdgeModel, fit, size) -> np.ndarray:
    """Target edge pixels drawn in work pixels (by inverting ``fit`` with a few fixed-point steps)."""
    Ginv = np.linalg.inv(fit.G)
    m = model.edges.pts
    x = al.apply_h(Ginv, m)
    for _ in range(8):
        x = x - al.apply_h(Ginv, fit(x)) + al.apply_h(Ginv, m)
    w, h = size
    img = np.zeros((h, w), bool)
    xi = np.round(x).astype(int)
    ok = (xi[:, 0] >= 0) & (xi[:, 1] >= 0) & (xi[:, 0] < w) & (xi[:, 1] < h)
    img[xi[ok, 1], xi[ok, 0]] = True
    return img


def panels(work: np.ndarray, lines, target: np.ndarray, model: al.EdgeModel, fit, width: int = 640) -> np.ndarray:
    """Three panels side by side: the straightened photo; the sketch lines found
    (yellow) with the target's edges where the alignment put them (green); the
    target laid onto the canvas at half opacity."""
    h, w = work.shape[:2]
    a = work.copy()
    b = (work * 0.6).astype(np.uint8)
    xi = lines.pts.astype(int)
    b[xi[:, 1], xi[:, 0]] = (0, 230, 255)
    e = cv2.dilate(edges_in_work(model, fit, (w, h)).astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    b[e] = (60, 220, 60)
    warped, mask = warp_target(target, model, fit, (w, h))
    c = work.copy()
    c[mask] = (0.5 * work[mask] + 0.5 * warped[mask]).astype(np.uint8)
    f = width / w
    row = [cv2.resize(x, (width, round(h * f)), interpolation=cv2.INTER_AREA) for x in (a, b, c)]
    gap = np.full((row[0].shape[0], 8, 3), 255, np.uint8)
    return np.hstack([row[0], gap, row[1], gap, row[2]])
