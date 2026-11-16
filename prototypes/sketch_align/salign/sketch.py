"""A simulated hand-drawn sketch of a target image, on a canvas.

The sketch is what a painter would draw before painting: the main outlines of
the image, drawn freehand, so proportions drift, strokes are broken, doubled or
missing, and detail is simplified. Extra lines that are not in the image are
added too: X marks for tracking, construction lines, shading hatches and,
optionally, a transfer grid.

The outlines come from a colour segmentation of the image (mean shift, then
k-means), not from the edge detector the aligner uses, so the aligner is not
handed its own edges back.

Coordinates: target pixels for the image; canvas millimetres (x right, y down,
origin top-left) for the sketch. ``Sketch.S`` maps target pixels to canvas mm
(the true placement) and ``Sketch.warp`` is the freehand distortion applied in
target pixels before placing: an outline point p is drawn at S(warp(p)).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from xtrack.canvas import hand_drawn_x, jittered_grid


@dataclass(frozen=True)
class Style:
    name: str
    warp_frac: float  # RMS of the smooth freehand drift, as a share of the image diagonal
    affine_frac: float  # proportion errors (stretch, shear), std
    stroke_frac: float  # extra offset of each stroke on its own, share of the diagonal
    simplify_frac: float  # detail left out (polyline simplification), share of the diagonal
    keep: float  # share of outline strokes actually drawn


CAREFUL = Style("careful", 0.004, 0.01, 0.0015, 0.002, 0.8)
TYPICAL = Style("typical", 0.010, 0.025, 0.003, 0.004, 0.65)
LOOSE = Style("loose", 0.020, 0.05, 0.005, 0.007, 0.5)
STYLES = {s.name: s for s in (CAREFUL, TYPICAL, LOOSE)}


@dataclass(frozen=True)
class Clutter:
    xs: bool = True  # tracking X's every ~10 cm over the canvas
    construction: int = 2  # long straight guide lines through the image
    hatching: int = 3  # patches of short parallel shading strokes
    grid: int = 0  # transfer grid: this many squares across the image (0 = none)


NONE = Clutter(False, 0, 0, 0)
USUAL = Clutter()
GRID = Clutter(grid=4)
CLUTTERS = {"none": NONE, "usual": USUAL, "grid": GRID}


class Warp:
    """Smooth freehand drift of target pixels: p -> p + A(p - c) + d(p), with d
    a bicubic field over a coarse lattice."""

    def __init__(self, size, A: np.ndarray, field_xy: np.ndarray):
        self.size = size
        self.A = A  # 2x2, (stretch/shear) minus identity
        self.field = field_xy  # (2, gy, gx) displacement at lattice nodes, target px

    def disp(self, pts) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 2)
        w, h = self.size
        c = np.array([w / 2, h / 2])
        gy, gx = self.field.shape[1:]
        u = pts[:, 0] / w * (gx - 1)
        v = pts[:, 1] / h * (gy - 1)
        d = np.stack([ndimage.map_coordinates(self.field[k], [v, u], order=3, mode="nearest") for k in range(2)], 1)
        return (pts - c) @ self.A.T + d

    def __call__(self, pts) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 2)
        return pts + self.disp(pts)

    def inverse(self, q) -> np.ndarray:
        q = np.asarray(q, float).reshape(-1, 2)
        p = q.copy()
        for _ in range(30):
            p = q - self.disp(p)
        return p


def random_warp(rng, size, style: Style) -> Warp:
    w, h = size
    diag = np.hypot(w, h)
    A = rng.normal(0, style.affine_frac, (2, 2))
    A = (A + A.T) / 2  # stretch and shear; a rotation is just part of the placement
    f = rng.normal(0, 1, (2, 4, 5))
    probe = Warp(size, np.zeros((2, 2)), f)
    grid = np.stack(np.meshgrid(np.linspace(0, w, 20), np.linspace(0, h, 15)), -1).reshape(-1, 2)
    rms = np.sqrt((probe.disp(grid) ** 2).sum(1).mean())
    return Warp(size, A, f * (style.warp_frac * diag / max(rms, 1e-9)))


@dataclass
class Sketch:
    strokes: list  # (k, 2) polylines in canvas mm
    kinds: list  # "outline" | "x" | "construction" | "hatch" | "grid"
    widths: list  # mm
    alphas: list  # 0..1 (pressure)
    S: np.ndarray  # 3x3: target px -> canvas mm (true placement)
    warp: Warp
    image_size: tuple
    canvas_size: tuple
    outlines_px: list = field(default_factory=list)  # the outline strokes before drift, target px

    def truth(self, p) -> np.ndarray:
        """Where target pixel p was drawn, in canvas mm (smooth part of the drift only)."""
        return apply_h(self.S, self.warp(p))


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    p = pts @ H[:, :2].T + H[:, 2]
    return p[:, :2] / p[:, 2:3]


def similarity(scale: float, angle: float, tx: float, ty: float) -> np.ndarray:
    c, s = scale * np.cos(angle), scale * np.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1.0]])


# ---------------------------------------------------------------- outlines a painter would draw


def outline_polylines(img: np.ndarray, rng, long_side: int = 600, n_colors: int = 8):
    """Closed boundaries of the image's main colour regions, in target px, with
    the colour contrast across each point (for deciding what gets drawn)."""
    h, w = img.shape[:2]
    f = long_side / max(h, w)
    small = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    ms = cv2.pyrMeanShiftFiltering(small, 8, 18)
    lab = cv2.cvtColor(ms, cv2.COLOR_BGR2LAB)
    data = lab.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(int(rng.integers(1 << 30)))
    _, labels, _ = cv2.kmeans(data, n_colors, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    lab_img = cv2.medianBlur(labels.reshape(lab.shape[:2]).astype(np.uint8), 5)
    sm = cv2.GaussianBlur(lab.astype(np.float32), (0, 0), 2.0)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0) / 8
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1) / 8
    contrast = np.sqrt((gx**2 + gy**2).sum(2))
    sh, sw = lab_img.shape
    min_area = 0.002 * sh * sw
    k3 = np.ones((3, 3), np.uint8)
    polys = []
    for k in range(n_colors):
        mask = cv2.morphologyEx((lab_img == k).astype(np.uint8), cv2.MORPH_OPEN, k3)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            conts, _ = cv2.findContours((cc == i).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for c in conts:
                c = c[:, 0, :].astype(float)
                if len(c) < 20:
                    continue
                q = np.clip(np.round(c).astype(int), 0, (sw - 1, sh - 1))
                on_border = (c[:, 0] <= 2) | (c[:, 1] <= 2) | (c[:, 0] >= sw - 3) | (c[:, 1] >= sh - 3)
                polys.append(((c + 0.5) / f - 0.5, contrast[q[:, 1], q[:, 0]], on_border))
    return polys


def _resample(poly: np.ndarray, step: float) -> np.ndarray:
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
    if d[-1] < step:
        return poly
    s = np.linspace(0, d[-1], max(2, int(d[-1] / step) + 1))
    return np.column_stack([np.interp(s, d, poly[:, 0]), np.interp(s, d, poly[:, 1])])


def _chaikin(poly: np.ndarray, n: int = 2) -> np.ndarray:
    for _ in range(n):
        if len(poly) < 3:
            return poly
        q = 0.75 * poly[:-1] + 0.25 * poly[1:]
        r = 0.25 * poly[:-1] + 0.75 * poly[1:]
        poly = np.vstack([poly[:1], np.column_stack([q, r]).reshape(-1, 2), poly[-1:]])
    return poly


def outline_strokes(polys, rng, style: Style, size) -> list:
    """Break region boundaries into pen strokes, leave some out (more of the
    low-contrast ones), simplify. Target px, before drift."""
    diag = np.hypot(*size)
    med = np.median(np.concatenate([c for _, c, _ in polys])) if polys else 1.0
    out = []
    for poly, contrast, border in polys:
        # split where the boundary runs along the image frame: nobody draws that
        runs, cur = [], []
        for i in range(len(poly)):
            if border[i]:
                if len(cur) > 2:
                    runs.append(np.array(cur))
                cur = []
            else:
                cur.append(i)
        if len(cur) > 2:
            runs.append(np.array(cur))
        for idx in runs:
            seg, con = poly[idx], contrast[idx]
            d = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
            pos = rng.uniform(0, 0.05) * diag
            while pos < d[-1]:
                length = rng.uniform(0.03, 0.12) * diag
                sel = (d >= pos) & (d <= pos + length)
                pos += length + rng.uniform(-0.01, 0.012) * diag
                if sel.sum() < 3 or d[sel][-1] - d[sel][0] < 0.008 * diag:
                    continue
                c_mean = np.mean(con[sel])
                if c_mean < 1.0:  # nobody outlines a smooth gradient (Lab units per pixel at 600 px)
                    continue
                p_keep = style.keep * np.clip(c_mean / med, 0.3, 1.6)
                if rng.uniform() > p_keep:
                    continue
                simp = cv2.approxPolyDP(seg[sel].astype(np.float32), style.simplify_frac * diag, False)[:, 0, :]
                out.append(simp.astype(float))
    return out


def _wobbly(rng, poly: np.ndarray, size: float) -> np.ndarray:
    """Densify, round off corners, overshoot the ends a little, add a slight tremor."""
    poly = _chaikin(_resample(poly, 0.004 * size), 2)
    if len(poly) >= 2:
        for end, nxt in ((0, 1), (-1, -2)):
            t = poly[end] - poly[nxt]
            t /= max(np.linalg.norm(t), 1e-9)
            ext = rng.uniform(-0.004, 0.012) * size
            if ext > 0:
                p = poly[end] + t * ext
                poly = np.vstack([p, poly]) if end == 0 else np.vstack([poly, p])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
    if len(poly) >= 3 and d[-1] > 0:
        tang = np.gradient(poly, axis=0)
        nrm = np.column_stack([-tang[:, 1], tang[:, 0]]) / np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
        ph = rng.uniform(0, 2 * np.pi, 2)
        k = 2 * np.pi / (0.05 * size)
        poly = poly + nrm * (0.0006 * size * (np.sin(k * d + ph[0]) + 0.5 * np.sin(2.7 * k * d + ph[1])))[:, None]
    return poly


def make_sketch(img: np.ndarray, rng, style: Style = TYPICAL, clutter: Clutter = USUAL, canvas_size=(609.6, 457.2), fill: float = 0.85) -> Sketch:
    """Sketch ``img`` on a canvas: placed upright, centred, filling ``fill`` of
    the canvas (give or take), then drawn freehand."""
    h, w = img.shape[:2]
    cw, ch = canvas_size
    scale = fill * min(cw / w, ch / h) * rng.uniform(0.9, 1.0)
    ang = np.radians(rng.normal(0, 1.5))
    cx, cy = cw / 2 + rng.normal(0, 8), ch / 2 + rng.normal(0, 8)
    S = similarity(scale, ang, 0, 0)
    S[:2, 2] = [cx, cy] - apply_h(S, [(w / 2, h / 2)])[0]
    warp = random_warp(rng, (w, h), style)
    diag_px = np.hypot(w, h)
    diag_mm = diag_px * scale

    strokes, kinds, widths, alphas = [], [], [], []

    def add(poly_mm, kind, width, alpha):
        strokes.append(poly_mm)
        kinds.append(kind)
        widths.append(width)
        alphas.append(alpha)

    outlines = outline_strokes(outline_polylines(img, rng), rng, style, (w, h))
    for s in outlines:
        p = warp(_wobbly(rng, s, diag_px))
        c = p.mean(0)
        a = np.radians(rng.normal(0, 1.0))
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        p = (p - c) @ R.T + c + rng.normal(0, style.stroke_frac * diag_px, 2)
        add(apply_h(S, p), "outline", rng.uniform(0.8, 1.8), rng.uniform(0.55, 1.0))
        if rng.uniform() < 0.15:  # gone over twice
            add(apply_h(S, p + rng.normal(0, 0.002 * diag_px, 2)), "outline", rng.uniform(0.8, 1.5), rng.uniform(0.4, 0.8))

    corners = apply_h(S, [(0, 0), (w, 0), (w, h), (0, h)])
    lo, hi = corners.min(0), corners.max(0)
    if clutter.xs:
        for i, c in enumerate(jittered_grid(rng, canvas_size, 100.0)):
            m = hand_drawn_x(rng, i, c)
            for st in m.strokes:
                add(st, "x", m.width_mm, m.darkness)
    for _ in range(clutter.construction):
        if rng.uniform() < 0.6:  # a centre line or a horizon
            v = rng.uniform() < 0.5
            t = rng.uniform(0.3, 0.7)
            a = lo + (hi - lo) * ([t, 0] if v else [0, t])
            b = a + (hi - lo) * ([0, 1] if v else [1, 0])
        else:
            a, b = rng.uniform(lo, hi), rng.uniform(lo, hi)
            if np.linalg.norm(b - a) < 0.3 * diag_mm:
                continue
        line = _wobbly(rng, np.array([a, b]), diag_mm)
        add(line, "construction", rng.uniform(0.6, 1.2), rng.uniform(0.3, 0.6))
    for _ in range(clutter.hatching):
        c = rng.uniform(lo, hi)
        a = rng.uniform(0, np.pi)
        d = np.array([np.cos(a), np.sin(a)])
        nrm = np.array([-d[1], d[0]])
        n, length, gap = int(rng.integers(6, 16)), rng.uniform(10, 30), rng.uniform(2.5, 5)
        for k in range(n):
            o = c + nrm * (k - n / 2) * gap + d * rng.normal(0, 2)
            L = length * rng.uniform(0.7, 1.2)
            add(_wobbly(rng, np.array([o - d * L / 2, o + d * L / 2]), diag_mm), "hatch", rng.uniform(0.6, 1.2), rng.uniform(0.4, 0.8))
    if clutter.grid:
        n = clutter.grid
        cell = w / n
        for k in range(1, int(w / cell) + 1):
            if k * cell < w:
                add(_wobbly(rng, apply_h(S, [(k * cell, 0), (k * cell, h)]), diag_mm), "grid", 0.8, 0.6)
        for k in range(1, int(h / cell) + 1):
            if k * cell < h:
                add(_wobbly(rng, apply_h(S, [(0, k * cell), (w, k * cell)]), diag_mm), "grid", 0.8, 0.6)
    return Sketch(strokes, kinds, widths, alphas, S, warp, (w, h), tuple(canvas_size), outlines)
