"""One simulated attempt: sketch, photograph, align, measure."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree

from xtrack.camera import MAIN, Camera, look_at

from . import align as al
from . import photo as ph
from . import sketch as sk
from .lines import sketch_lines


@dataclass
class Setup:
    style: sk.Style = sk.TYPICAL
    clutter: sk.Clutter = sk.USUAL
    look: ph.Look = ph.MAUVE_CHALK
    cam: Camera = MAIN
    mode: str = "tracked"  # "tracked": straighten with the tracker's pose; "free": align in the raw photo
    track_err_mm: float = 1.5
    distance: tuple = (600, 720)  # mm, whole canvas in view
    tilt_deg: tuple = (0, 20)
    search: str = "snap"  # "snap" from a rough placement, "upright" (+-15 deg), "any" rotation
    rough: tuple = (0.10, 10.0, 0.15)  # snap start: offset (share of image size), degrees, scale (log)
    rect_res: float = 2.0  # px per mm when straightened
    bend: bool = True


@dataclass
class Result:
    found: bool  # the placement is the right one (error near what the sketch's own drift allows)
    err: dict  # mm, mean over the whole image, per stage: "sim", "rigid" (affine / perspective), "bent"
    near: dict  # mm, mean over the parts of the image near something sketched
    p90: dict  # mm, 90th percentile over the whole image
    floor: dict  # best any similarity / affine / perspective map could do: the sketch's freehand drift
    score: float  # truncated chamfer of the result, 0 = perfect
    second: float  # the same for the best clearly different placement
    n_points: int
    seconds: dict = field(default_factory=dict)


def _fit_h(src, dst, kind):
    if kind == "similarity":
        M, _ = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32), method=cv2.LMEDS)
        return np.vstack([M, [0, 0, 1]])
    if kind == "affine":
        M, _ = cv2.estimateAffine2D(src.astype(np.float32), dst.astype(np.float32), method=cv2.LMEDS)
        return np.vstack([M, [0, 0, 1]])
    H, _ = cv2.findHomography(src, dst, 0)
    return H


def run(target: np.ndarray, seed: int, setup: Setup, model: al.EdgeModel | None = None, keep: bool = False):
    rng = np.random.default_rng(seed)
    secs = {}
    sketch = sk.make_sketch(target, rng, setup.style, setup.clutter)
    plane = ph.surface(sketch, setup.look, rng)
    cw, ch = sketch.canvas_size
    pose = look_at((cw / 2 + rng.normal(0, 15), ch / 2 + rng.normal(0, 15)), rng.uniform(*setup.distance), rng.uniform(*setup.tilt_deg), rng.uniform(0, 360), rng.normal(0, 3))
    photo, H = ph.photograph(plane, 4.0, setup.cam, pose, setup.look, rng)
    if setup.mode == "tracked":
        Hest = ph.tracking_error(rng, H, sketch.canvas_size, setup.track_err_mm)
        r = setup.rect_res
        work = ph.rectify(photo, Hest, sketch.canvas_size, r)
        to_work = lambda mm: al.apply_h(np.linalg.inv(Hest) @ H, mm) * r - 0.5  # noqa: E731
        line_px = 1.4 * r
    else:
        work = photo
        to_work = lambda mm: al.apply_h(H, mm)  # noqa: E731
        line_px = 1.4 * setup.cam.f / np.mean(setup.distance)
    wh, ww = work.shape[:2]

    t = time.perf_counter()
    if model is None:
        model = al.EdgeModel(target)
    secs["model"] = time.perf_counter() - t
    f = model.f
    w, h = sketch.image_size
    grid = np.stack(np.meshgrid(np.linspace(0, w - 1, 33), np.linspace(0, h - 1, 25)), -1).reshape(-1, 2)
    q_true = to_work(sketch.truth(grid))  # where the sketch put each target point, in work px
    m_true = (grid + 0.5) * f - 0.5  # model px
    mm_per_px = np.sqrt(abs(np.linalg.det(sketch.S[:2, :2])))
    drawn = np.vstack(sketch.outlines_px) if sketch.outlines_px else np.zeros((1, 2))
    near = cKDTree(drawn).query(grid)[0] < 0.04 * np.hypot(w, h)

    def err(fit) -> np.ndarray:
        p = (fit(q_true) + 0.5) / f - 0.5
        return np.hypot(*(p - grid).T) * mm_per_px

    floors = {k: float(np.mean(err(lambda q, G=_fit_h(q_true, m_true, k): al.apply_h(G, q)))) for k in ("similarity", "affine", "homography")}

    t = time.perf_counter()
    start, roi = None, None
    if setup.search == "snap":
        # a rough placement, as a person would drag it: the true one shifted, turned and scaled
        G_sim = _fit_h(q_true, m_true, "similarity")
        c = q_true.mean(0)
        size = np.ptp(q_true, 0).max()
        P = sk.similarity(np.exp(rng.uniform(-1, 1) * setup.rough[2]), np.radians(rng.uniform(-1, 1) * setup.rough[1]), 0, 0)
        P[:2, 2] = c + rng.uniform(-1, 1, 2) * setup.rough[0] * size - P[:2, :2] @ c
        start = G_sim @ np.linalg.inv(P)
        # only look at sketch lines around it
        mw, mh = model.size
        foot = al.apply_h(np.linalg.inv(start), [(0, 0), (mw, 0), (mw, mh), (0, mh)])
        ctr = foot.mean(0)
        roi = np.zeros((wh, ww), np.uint8)
        cv2.fillPoly(roi, [np.round(ctr + (foot - ctr) * 1.4).astype(np.int32)], 1)
    lines = sketch_lines(work, setup.look.polarity, line_px, roi=roi)
    secs["lines"] = time.perf_counter() - t
    final = "affine" if setup.mode == "tracked" else "homography"
    extra = {} if setup.bend else {"bend_levels": ()}
    a = al.align(model, lines, (ww, wh), start, "any" if setup.search == "any" else "upright", final, rng=rng, **extra)
    secs.update(a.seconds)
    stages = {"rigid": a.rigid, "bent": a.fit}
    e = {k: err(v) for k, v in stages.items()}
    floor_rigid = floors[final]
    found = bool(np.mean(e["rigid"]) < 2 * floor_rigid + 3.0)
    res = Result(
        found=found,
        err={k: float(np.mean(v)) for k, v in e.items()},
        near={k: float(np.mean(v[near])) for k, v in e.items()},
        p90={k: float(np.percentile(v, 90)) for k, v in e.items()},
        floor={**floors, "near_" + final: float(np.mean(err(lambda q, G=_fit_h(q_true, m_true, final): al.apply_h(G, q))[near]))},
        score=a.fit.score, second=a.second_score, n_points=len(lines.pts), seconds=secs,
    )
    if keep:
        return res, dict(sketch=sketch, photo=photo, work=work, lines=lines, align=a, model=model, q_true=q_true, grid=grid, near=near)
    return res
