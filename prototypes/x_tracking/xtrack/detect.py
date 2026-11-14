"""Find hand-drawn X marks in a camera frame.

1. Background: a median over a wide window (computed at quarter size) gives
   the bare surface. Marks are what differs from it: darker (pencil on a light
   ground) or lighter (white pencil on a dark ground).
2. Mark pixels: contrast above the image's noise level, with hysteresis,
   grouped into connected blobs.
3. A blob counts as an X only if it is two roughly straight strokes that
   cross, with arms on both sides of the crossing. The crossing of straight
   lines fitted to the two strokes is the mark's position, to a fraction of a
   pixel; the two line angles are its stroke directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_THETAS = np.radians(np.arange(0.0, 180.0, 2.0))  # coarse; line fits refine the angle
_COS, _SIN = np.cos(_THETAS), np.sin(_THETAS)


@dataclass
class Detection:
    center: np.ndarray  # (2,) pixels
    directions: np.ndarray  # (2,) stroke angles, radians in [0, pi)
    size: float  # mean stroke length, pixels
    strength: float  # mean contrast, in noise units


def luminance(img_bgr: np.ndarray) -> np.ndarray:
    f = img_bgr.astype(np.float32) / 255
    return 0.0722 * f[..., 0] + 0.7152 * f[..., 1] + 0.2126 * f[..., 2]


def _background(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    small = cv2.resize((gray * 255).clip(0, 255).astype(np.uint8), (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    med = cv2.medianBlur(small, 31)  # ~120 px window at full size: wider than any stroke
    bg = cv2.resize(med, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    return cv2.GaussianBlur(bg, (0, 0), 3)


def detect_x(img_bgr: np.ndarray, polarity: str = "auto", low: float = 3.0, high: float = 6.0,
             min_extent: int = 8, max_extent_frac: float = 0.6, max_pixels: int = 2500) -> list[Detection]:  # fmt: skip
    """All X marks in the frame. ``polarity`` is "dark", "light" or "auto" (both)."""
    gray = luminance(img_bgr)
    diff = gray - _background(gray)
    sigma = 1.4826 * float(np.median(np.abs(diff - np.median(diff)))) + 1e-4
    signs = {"dark": (-1,), "light": (1,), "auto": (-1, 1)}[polarity]
    found = []
    for sign in signs:
        snr = cv2.GaussianBlur(sign * diff / sigma, (0, 0), 1.0)
        found += _find(snr, low, high, min_extent, max_extent_frac * min(gray.shape), max_pixels)
    return _dedupe(found)


def _find(snr, low, high, min_extent, max_extent, max_pixels) -> list[Detection]:
    weak = (snr > low).astype(np.uint8)
    weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    strong = snr > high
    out = []
    for k in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[k])
        extent = max(w, h)
        if extent < min_extent or extent > max_extent or area < 15 or area > 0.65 * w * h:
            continue
        sub = labels[y : y + h, x : x + w] == k
        if not strong[y : y + h, x : x + w][sub].any():
            continue
        ys, xs = np.nonzero(sub)
        wts = np.clip(snr[y : y + h, x : x + w][sub], 0.05, None)
        det = _two_strokes(xs.astype(float), ys.astype(float), wts, area, extent, max_pixels)
        if det is not None:
            det.center = det.center + (x, y)
            det.strength = float(wts.mean())
            out.append(det)
    return out


def _strongest_line(xs, ys, w, width, avoid=None, min_sep=np.radians(25)):
    """Weighted Hough vote: the (angle, offset) of the line covering the most ink."""
    rho = np.outer(-_SIN, xs) + np.outer(_COS, ys)  # points on a line at angle t: -x sin t + y cos t = rho
    bin_w = max(1.0, width)
    rmin = rho.min()
    idx = ((rho - rmin) / bin_w).astype(np.int64)
    nb = int(idx.max()) + 1
    flat = (np.arange(len(_THETAS))[:, None] * nb + idx).ravel()
    acc = np.bincount(flat, weights=np.broadcast_to(w, rho.shape).ravel(), minlength=len(_THETAS) * nb).reshape(len(_THETAS), nb)
    if avoid is not None:
        acc[np.abs((_THETAS - avoid + np.pi / 2) % np.pi - np.pi / 2) < min_sep] = 0
    ti, bi = divmod(int(np.argmax(acc)), nb)
    if acc[ti, bi] <= 0:
        return None
    t = _THETAS[ti]
    r = rmin + (bi + 0.5) * bin_w
    return np.array([-np.sin(t), np.cos(t)]) * r, np.array([np.cos(t), np.sin(t)])  # point, direction


def _fit_line(xs, ys, w):
    c = np.array([np.average(xs, weights=w), np.average(ys, weights=w)])
    dx, dy = xs - c[0], ys - c[1]
    cov = np.array([[np.sum(w * dx * dx), np.sum(w * dx * dy)], [np.sum(w * dx * dy), np.sum(w * dy * dy)]])
    vals, vecs = np.linalg.eigh(cov)
    return c, vecs[:, np.argmax(vals)]


def _two_strokes(xs, ys, w, area, extent, max_pixels) -> Detection | None:
    if len(xs) > max_pixels:
        step = int(np.ceil(len(xs) / max_pixels))
        xs, ys, w = xs[::step], ys[::step], w[::step]
    width = max(1.5, area / (2.83 * extent))  # stroke width estimate for an X in this bounding box
    tol = max(1.5, 0.75 * width + 0.75)
    first = _strongest_line(xs, ys, w, width)
    if first is None:
        return None
    c1, d1 = first
    rest = np.abs((xs - c1[0]) * d1[1] - (ys - c1[1]) * d1[0]) >= tol
    if rest.sum() < 5:
        return None
    second = _strongest_line(xs[rest], ys[rest], w[rest], width, avoid=np.arctan2(d1[1], d1[0]) % np.pi)
    if second is None:
        return None
    lines = [first, second]
    for _ in range(3):  # alternate: assign pixels to the nearer line, refit each line
        dist = np.stack([np.abs((xs - c[0]) * d[1] - (ys - c[1]) * d[0]) for c, d in lines])
        nearest = dist.argmin(0)
        near = dist.min(0) < tol
        new = []
        for k in (0, 1):
            sel = near & (nearest == k)
            if sel.sum() < 4:
                return None
            new.append(_fit_line(xs[sel], ys[sel], w[sel]))
        lines = new
    if near.mean() < 0.75:  # an X is just two strokes
        return None
    (c1, d1), (c2, d2) = lines
    if abs(d1[0] * d2[1] - d1[1] * d2[0]) < 0.5:  # strokes must cross at 30 degrees or more
        return None
    a, _ = np.linalg.solve(np.column_stack([d1, -d2]), c2 - c1)
    center = c1 + a * d1
    if np.min(np.hypot(xs - center[0], ys - center[1])) > tol + 1.5:  # ink at the crossing
        return None
    lengths = []
    for k, (c, d) in enumerate(lines):
        sel = near & (nearest == k)
        s = (xs[sel] - center[0]) * d[0] + (ys[sel] - center[1]) * d[1]
        lo, hi = s.min(), s.max()
        L = hi - lo
        arm = max(2.0, 0.18 * L)
        if L < 6 or lo > -arm or hi < arm:  # arms on both sides of the crossing
            return None
        lengths.append(L)
    if not 0.4 <= lengths[0] / lengths[1] <= 2.5:
        return None
    dirs = np.array([np.arctan2(d[1], d[0]) % np.pi for _, d in lines])
    return Detection(center, dirs, float(np.mean(lengths)), 0.0)


def _dedupe(dets: list[Detection], radius: float = 4.0) -> list[Detection]:
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.strength):
        if all(np.hypot(*(d.center - k.center)) > radius for k in kept):
            kept.append(d)
    return kept
