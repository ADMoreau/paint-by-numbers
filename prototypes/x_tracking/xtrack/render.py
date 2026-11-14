"""What the phone camera sees: the canvas and its X marks, lighting, a hand
holding a brush, blur, sensor noise and video compression."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .camera import Camera, Pose, homography, in_front, local_scale, project
from .canvas import Canvas


@dataclass(frozen=True)
class Look:
    name: str
    ground_rgb: tuple
    mark_rgb: tuple
    wall_rgb: tuple = (118, 108, 96)
    weave_period_mm: float = 0.8  # canvas threads
    weave_amp: float = 0.025
    grain_amp: float = 0.015


WHITE_GRAPHITE = Look("white canvas, graphite X", (236, 233, 226), (72, 72, 80))
MAUVE_GRAPHITE = Look("mauve ground, graphite X", (145, 95, 109), (58, 56, 62))
MAUVE_CHALK = Look("mauve ground, white pencil X", (145, 95, 109), (232, 230, 222))
DARK_CHALK = Look("near-black ground, white pencil X", (42, 36, 34), (232, 230, 222))


@dataclass(frozen=True)
class Effects:
    hand_prob: float = 0.5
    defocus_sigma: tuple = (0.0, 1.2)
    motion_prob: float = 0.5
    motion_px: tuple = (0.0, 6.0)
    noise: float = 1.0
    jpeg_quality: tuple = (80, 93)
    exposure: tuple = (0.85, 1.12)
    light_gradient: float = 0.15
    vignette: float = 0.25


@dataclass
class Frame:
    image: np.ndarray  # BGR uint8
    H: np.ndarray  # ground truth: canvas mm -> pixels
    hand_mask: np.ndarray  # bool
    visible_ids: list  # marks fully in view and not covered by the hand


SKIN_RGB = [(224, 172, 138), (198, 134, 98), (141, 85, 36), (245, 205, 168), (110, 70, 45)]
HANDLE_RGB = [(160, 30, 30), (30, 30, 30), (170, 120, 70), (40, 70, 140)]


@lru_cache(maxsize=4)
def _pixel_grid(w: int, h: int):
    return np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))


# ---------------------------------------------------------------- the hand


def random_hand(rng, w: int, h: int, tip=None, side=None, arm_frac=None, brush_frac=None) -> dict:
    """Geometry of a forearm and hand reaching in from one edge, holding a brush
    whose tip is at ``tip``. Pass fixed values to animate a hand across frames."""
    m = min(w, h)
    side = int(rng.choice(4, p=[0.45, 0.3, 0.2, 0.05])) if side is None else side  # bottom, right, left, top
    tip = np.array(rng.uniform((0.15 * w, 0.15 * h), (0.85 * w, 0.85 * h)) if tip is None else tip, float)
    along = rng.uniform(0.25, 0.75)
    entry = np.array([(along * w, h + 0.1 * m), (w + 0.1 * m, along * h), (-0.1 * m, along * h), (along * w, -0.1 * m)][side])
    d = tip - entry
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    aw = (arm_frac if arm_frac is not None else rng.uniform(0.16, 0.28)) * m
    lb = (brush_frac if brush_frac is not None else rng.uniform(0.12, 0.25)) * m
    grip = tip - d * lb
    palm = grip - d * 0.8 * aw + n * 0.12 * aw
    return {
        "tip": tip, "grip": grip, "palm": palm, "wrist": palm - d * 0.55 * aw, "arm_end": entry - d * aw,
        "d": d, "n": n, "aw": aw, "lb": lb,
        "skin": SKIN_RGB[int(rng.integers(len(SKIN_RGB)))], "handle": HANDLE_RGB[int(rng.integers(len(HANDLE_RGB)))],
        "blur": float(rng.uniform(2.0, 5.0)), "shadow": float(rng.uniform(0.15, 0.35)),
        "shadow_offset": rng.normal(0, 0.05 * m, 2),
    }  # fmt: skip


def _pt(p) -> tuple[int, int]:
    return int(round(p[0] * 16)), int(round(p[1] * 16))


def _draw_hand(hand: dict, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Colour (BGR uint8) and coverage mask (uint8) of the hand and brush."""
    color = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    d, n, aw, lb = hand["d"], hand["n"], hand["aw"], hand["lb"]
    skin = tuple(int(c) for c in hand["skin"][::-1])
    sh = 4  # fixed-point shift for sub-pixel drawing

    def line(a, b, thick, col):
        for img, c in ((color, col), (mask, 255)):
            cv2.line(img, _pt(a), _pt(b), c, max(1, int(thick)), cv2.LINE_AA, sh)

    def poly(pts, col):
        p = np.array([_pt(q) for q in pts], np.int32)
        for img, c in ((color, col), (mask, 255)):
            cv2.fillPoly(img, [p], c, cv2.LINE_AA, sh)

    grip, palm, wrist, tip = hand["grip"], hand["palm"], hand["wrist"], hand["tip"]
    line(grip - d * (0.5 * aw + 0.25 * min(w, h)), grip, 0.07 * aw, tuple(int(c) for c in hand["handle"][::-1]))
    poly([wrist + n * 0.45 * aw, hand["arm_end"] + n * 0.55 * aw, hand["arm_end"] - n * 0.55 * aw, wrist - n * 0.45 * aw], skin)
    ang = float(np.degrees(np.arctan2(d[1], d[0])))
    for img, c in ((color, skin), (mask, 255)):
        cv2.ellipse(img, _pt(palm), (int(0.62 * aw * 16), int(0.48 * aw * 16)), ang, 0, 360, c, -1, cv2.LINE_AA, sh)
    for k, off in enumerate((-0.22, 0.0, 0.22)):
        line(palm + n * off * aw, grip + n * off * 0.35 * aw - d * 0.06 * aw * k, 0.2 * aw, skin)
    line(palm + n * 0.4 * aw, grip + n * 0.12 * aw, 0.22 * aw, skin)
    ferrule_end = grip + d * 0.35 * lb
    line(grip, ferrule_end, 0.08 * aw, (175, 178, 180))
    poly([ferrule_end + n * 0.035 * aw, tip + n * 0.01 * aw, tip - n * 0.01 * aw, ferrule_end - n * 0.035 * aw], (35, 40, 45))
    # soft shading across the arm
    u, v = _pixel_grid(w, h)
    across = ((u - palm[0]) * n[0] + (v - palm[1]) * n[1]) / aw
    shade = np.clip(0.9 + 0.12 * np.tanh(across), 0.75, 1.05)
    return np.clip(color * shade[..., None], 0, 255).astype(np.uint8), mask


def hand_layers(hand: dict, w: int, h: int):
    """(colour 0..1 BGR, soft alpha, sharp bool mask, shadow alpha) for compositing."""
    color8, mask8 = _draw_hand(hand, w, h)
    m = mask8.astype(np.float32) / 255
    s = hand["blur"]  # the hand is nearer than the canvas, so out of focus
    alpha = cv2.GaussianBlur(m, (0, 0), s)
    color = cv2.GaussianBlur(color8.astype(np.float32) / 255 * m[..., None], (0, 0), s) / np.maximum(alpha, 1e-3)[..., None]
    dx, dy = hand["shadow_offset"]
    shifted = cv2.warpAffine(m, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))
    shadow = cv2.GaussianBlur(shifted, (0, 0), 0.05 * min(w, h)) * hand["shadow"]
    return color, alpha, mask8 > 127, shadow


def hand_mask(hand: dict, w: int, h: int) -> np.ndarray:
    return _draw_hand(hand, w, h)[1] > 127


# ---------------------------------------------------------------- which marks are in view


def visible_marks(canvas: Canvas, H: np.ndarray, w: int, h: int, mask=None, margin: float = 2.0) -> list:
    """Marks completely inside the frame and not covered by ``mask`` (the hand)."""
    if not canvas.marks:
        return []
    pts = np.vstack([np.vstack(m.strokes) for m in canvas.marks])
    owner = np.concatenate([np.full(sum(len(s) for s in m.strokes), i) for i, m in enumerate(canvas.marks)])
    p = project(H, pts)
    inside = in_front(H, pts) & (p[:, 0] >= margin) & (p[:, 0] <= w - 1 - margin) & (p[:, 1] >= margin) & (p[:, 1] <= h - 1 - margin)
    n_pts = np.bincount(owner, minlength=len(canvas.marks))
    ok = np.bincount(owner, weights=inside, minlength=len(canvas.marks)) == n_pts
    if mask is not None and ok.any():
        q = np.clip(np.round(p).astype(int), 0, (w - 1, h - 1))
        covered = mask[q[:, 1], q[:, 0]] & inside
        frac = np.bincount(owner, weights=covered, minlength=len(canvas.marks)) / n_pts
        centers = np.clip(np.round(project(H, np.array([m.center for m in canvas.marks]))).astype(int), 0, (w - 1, h - 1))
        ok &= (frac <= 0.25) & ~mask[centers[:, 1], centers[:, 0]]
    return [canvas.marks[i].id for i in np.flatnonzero(ok)]


# ---------------------------------------------------------------- the frame


def render(canvas: Canvas, cam: Camera, pose: Pose, look: Look, effects: Effects, rng, hand="random", motion=None, ss: int = 2) -> Frame:
    """One camera frame. ``hand`` is "random", None, or geometry from random_hand.
    ``motion`` is an optional (dx, dy) blur vector in pixels; otherwise random."""
    w, h = cam.width, cam.height
    H = homography(cam, pose)
    A = np.array([[ss, 0, (ss - 1) / 2], [0, ss, (ss - 1) / 2], [0, 0, 1.0]])
    Hs = A @ H
    Hi = np.linalg.inv(Hs).astype(np.float32)
    u, v = _pixel_grid(w * ss, h * ss)
    den = Hi[2, 0] * u + Hi[2, 1] * v + Hi[2, 2]
    cx = (Hi[0, 0] * u + Hi[0, 1] * v + Hi[0, 2]) / den
    cy = (Hi[1, 0] * u + Hi[1, 1] * v + Hi[1, 2]) / den
    front = den * den[h * ss // 2, w * ss // 2] > 0  # rays that meet the canvas in front of the camera
    inside = (front & (cx >= 0) & (cx <= canvas.width) & (cy >= 0) & (cy <= canvas.height)).astype(np.float32)
    k = np.float32(2 * np.pi / look.weave_period_mm)
    weave = np.sin(k * cx) * np.sin(k * cy)
    grain = cv2.remap(canvas.grain, cx * np.float32(canvas.grain_res), cy * np.float32(canvas.grain_res), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    ground = np.array(look.ground_rgb[::-1], np.float32) / 255
    wall = np.array(look.wall_rgb[::-1], np.float32) / 255
    shade = 1 + np.float32(look.weave_amp) * weave + np.float32(look.grain_amp) * grain
    img = inside[..., None] * ground * shade[..., None] + (1 - inside)[..., None] * wall * (1 + 0.06 * grain)[..., None]

    # the X marks, drawn in sub-pixel precision at supersampled resolution
    cov = np.zeros((h * ss, w * ss), np.uint8)
    for m in canvas.marks:
        if not in_front(Hs, np.vstack(m.strokes)).all():
            continue
        c = project(Hs, m.center)[0]
        s = local_scale(Hs, m.center)
        reach = m.size_mm * s
        if c[0] < -reach or c[1] < -reach or c[0] > w * ss + reach or c[1] > h * ss + reach:
            continue
        th = m.width_mm * s
        value = int(round(255 * m.darkness * min(1.0, th)))
        for stroke in m.strokes:
            p = np.round(project(Hs, stroke) * 16).astype(np.int32)
            cv2.polylines(cov, [p], False, value, int(np.clip(round(th), 1, 200)), cv2.LINE_AA, 4)
    tooth = np.clip(0.62 + 0.45 * weave + 0.35 * grain, 0, 1)  # graphite catches on the weave
    a = cov.astype(np.float32) / 255 * tooth * inside
    mark = np.array(look.mark_rgb[::-1], np.float32) / 255
    img = img * (1 - a[..., None]) + mark * a[..., None]
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    # light: exposure, a gradient across the frame, vignetting
    u1, v1 = _pixel_grid(w, h)
    gx, gy = rng.uniform(-1, 1, 2) * effects.light_gradient
    r2 = ((u1 - w / 2) ** 2 + (v1 - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
    light = rng.uniform(*effects.exposure) * (1 + gx * (u1 / w - 0.5) * 2 + gy * (v1 / h - 0.5) * 2) * (1 - effects.vignette * r2)
    img *= light[..., None]

    mask = np.zeros((h, w), bool)
    if hand == "random":
        hand = random_hand(rng, w, h) if rng.uniform() < effects.hand_prob else None
    if hand is not None:
        color, alpha, mask, shadow = hand_layers(hand, w, h)
        img *= 1 - shadow[..., None]
        img = img * (1 - alpha[..., None]) + color * light[..., None] * alpha[..., None]

    # camera: focus, motion, sensor noise, compression
    sigma = rng.uniform(*effects.defocus_sigma)
    if sigma > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    if motion is None and rng.uniform() < effects.motion_prob:
        ang = rng.uniform(0, np.pi)
        motion = rng.uniform(*effects.motion_px) * np.array([np.cos(ang), np.sin(ang)])
    if motion is not None and np.hypot(*motion) >= 1.0:
        img = motion_blur(img, motion)
    img = img + rng.standard_normal(img.shape).astype(np.float32) * (0.004 + 0.012 * np.sqrt(np.clip(img, 0, 1))) * effects.noise
    img8 = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img8, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(*effects.jpeg_quality))])
    img8 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return Frame(img8, H, mask, visible_marks(canvas, H, w, h, mask))


def motion_blur(img: np.ndarray, vec) -> np.ndarray:
    length = float(np.hypot(*vec))
    size = int(np.ceil(length)) | 1
    kernel = np.zeros((size + 2, size + 2), np.float32)
    c = (size + 1) / 2
    d = np.asarray(vec, float) / length * (length / 2)
    cv2.line(kernel, _pt((c - d[0], c - d[1])), _pt((c + d[0], c + d[1])), 1.0, 1, cv2.LINE_AA, 4)
    return cv2.filter2D(img, -1, kernel / kernel.sum())
