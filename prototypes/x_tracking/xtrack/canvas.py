"""A virtual canvas with hand-drawn X marks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CANVAS_24x18_MM = (609.6, 457.2)


@dataclass
class XMark:
    id: int
    center: np.ndarray  # (2,) mm: where straight lines fitted to the two strokes cross
    strokes: list  # two (k, 2) polylines, mm
    width_mm: float
    darkness: float  # 0..1
    directions: np.ndarray  # (2,) stroke angles, radians in [0, pi)
    size_mm: float  # mean stroke length


@dataclass
class Canvas:
    width: float
    height: float
    marks: list
    grain: np.ndarray  # float32 smooth noise, unit std, for surface texture
    grain_res: float  # grain samples per mm

    @property
    def size(self) -> tuple[float, float]:
        return self.width, self.height


def fit_line(pts: np.ndarray, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """(point on line, unit direction) by total least squares."""
    w = np.ones(len(pts)) if weights is None else np.asarray(weights, float)
    c = (pts * w[:, None]).sum(0) / w.sum()
    d = pts - c
    cov = (d * w[:, None]).T @ d
    vals, vecs = np.linalg.eigh(cov)
    return c, vecs[:, np.argmax(vals)]


def intersect(c1, d1, c2, d2) -> np.ndarray:
    a, _ = np.linalg.solve(np.column_stack([d1, -d2]), c2 - c1)
    return c1 + a * d1


def hand_drawn_x(rng, mark_id: int, center, size_mm: float = 15.0) -> XMark:
    """Two wobbly, slightly bowed strokes of uneven length, roughly diagonal and
    roughly perpendicular, crossing off-centre, like an X drawn by hand."""
    first = np.radians(45 + rng.uniform(-20, 20))
    angles = (first, first + np.radians(90 + rng.uniform(-12, 12)))
    strokes, fits, lengths = [], [], []
    for ang in angles:
        L = size_mm * rng.uniform(0.8, 1.2)
        u = np.linspace(-1, 1, 11)
        d = np.array([np.cos(ang), np.sin(ang)])
        n = np.array([-d[1], d[0]])
        shift = rng.uniform(-0.15, 0.15) * L
        bow = rng.uniform(-0.04, 0.04) * L * (1 - u**2)
        ph = rng.uniform(0, 2 * np.pi, 2)
        wobble = 0.012 * L * (np.sin(2.3 * np.pi * u + ph[0]) + 0.5 * np.sin(5.1 * np.pi * u + ph[1]))
        pts = np.asarray(center, float) + np.outer(u * L / 2 + shift, d) + np.outer(bow + wobble, n)
        strokes.append(pts)
        fits.append(fit_line(pts))
        lengths.append(L)
    cross = intersect(*fits[0], *fits[1])
    dirs = np.array([np.arctan2(f[1][1], f[1][0]) % np.pi for f in fits])
    return XMark(mark_id, cross, strokes, float(rng.uniform(0.7, 1.4)), float(rng.uniform(0.65, 0.95)), dirs, float(np.mean(lengths)))


def jittered_grid(rng, size, spacing: float, jitter: float = 0.25) -> np.ndarray:
    """Marks roughly every ``spacing`` mm, each nudged randomly so no two groups look alike."""
    W, H = size
    nx = max(1, int(round((W - spacing) / spacing)) + 1)
    ny = max(1, int(round((H - spacing) / spacing)) + 1)
    xs = np.linspace(spacing / 2, W - spacing / 2, nx)
    ys = np.linspace(spacing / 2, H - spacing / 2, ny)
    pts = np.array([(x, y) for y in ys for x in xs]) + rng.uniform(-jitter, jitter, (nx * ny, 2)) * spacing
    return np.clip(pts, 12, (W - 12, H - 12))


def corner_points(rng, size, inset: float = 35.0) -> np.ndarray:
    W, H = size
    pts = np.array([(inset, inset), (W - inset, inset), (W - inset, H - inset), (inset, H - inset)])
    return pts + rng.uniform(-6, 6, pts.shape)


def make_canvas(seed: int, layout: str = "grid", spacing: float = 100.0, size=CANVAS_24x18_MM) -> Canvas:
    """``layout`` is "grid" (a scattered constellation every ``spacing`` mm) or "corners"."""
    rng = np.random.default_rng(seed)
    centers = jittered_grid(rng, size, spacing) if layout == "grid" else corner_points(rng, size)
    marks = [hand_drawn_x(rng, i, c) for i, c in enumerate(centers)]
    res = 2.0
    noise = rng.standard_normal((int(size[1] * res) + 2, int(size[0] * res) + 2)).astype(np.float32)
    grain = cv2.GaussianBlur(noise, (0, 0), 2.0)
    grain /= grain.std()
    return Canvas(size[0], size[1], marks, grain, res)
