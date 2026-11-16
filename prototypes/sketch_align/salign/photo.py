"""Photographing the sketched canvas with a phone, and straightening the photo.

The canvas surface (ground colour, grain, the sketch in chalk or pencil
catching on the canvas tooth) is drawn flat at ``res`` pixels per mm, then
seen through the X-tracking prototype's pinhole camera with its lighting,
defocus, motion blur, noise and JPEG.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from xtrack.camera import Camera, Pose, homography
from xtrack.render import Effects, motion_blur

from .sketch import Sketch


@dataclass(frozen=True)
class Look:
    name: str
    ground_rgb: tuple
    line_rgb: tuple
    wall_rgb: tuple = (118, 108, 96)
    polarity: str = "light"  # lines lighter or darker than the ground


MAUVE_CHALK = Look("mauve ground, white pencil", (145, 95, 109), (232, 230, 222), polarity="light")
WHITE_GRAPHITE = Look("white canvas, graphite", (236, 233, 226), (72, 72, 80), polarity="dark")
LOOKS = {"mauve_chalk": MAUVE_CHALK, "white_graphite": WHITE_GRAPHITE}

STILL = Effects(hand_prob=0.0, defocus_sigma=(0.0, 1.0), motion_prob=0.2, motion_px=(0.0, 2.5))


def surface(sketch: Sketch, look: Look, rng, res: float = 4.0) -> np.ndarray:
    """The canvas seen flat: BGR float32 0..1, ``res`` px per mm, pixel j covering [j, j+1) / res mm."""
    cw, ch = sketch.canvas_size
    W, H = int(round(cw * res)), int(round(ch * res))
    grain = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 2.0 * res / 2)
    grain /= grain.std()
    fine = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.6)
    fine /= fine.std()
    ground = np.array(look.ground_rgb[::-1], np.float32) / 255
    img = ground * (1 + 0.02 * grain + 0.02 * fine)[..., None]
    cov = np.zeros((H, W), np.uint8)
    for poly, wmm, a in zip(sketch.strokes, sketch.widths, sketch.alphas):
        p = np.round((poly * res - 0.5) * 16).astype(np.int32)
        th = wmm * res
        cv2.polylines(cov, [p], False, int(round(255 * a * min(1.0, th))), int(np.clip(round(th), 1, 50)), cv2.LINE_AA, 4)
    tooth = np.clip(0.6 + 0.35 * fine + 0.2 * grain, 0, 1)  # the pencil skips over the weave
    a = cov.astype(np.float32) / 255 * tooth
    line = np.array(look.line_rgb[::-1], np.float32) / 255
    return img * (1 - a[..., None]) + line * a[..., None]


def plane_to_mm(res: float) -> np.ndarray:
    return np.array([[1 / res, 0, 0.5 / res], [0, 1 / res, 0.5 / res], [0, 0, 1.0]])


def photograph(plane: np.ndarray, res: float, cam: Camera, pose: Pose, look: Look, rng, effects: Effects = STILL, ss: int = 2):
    """(BGR uint8 photo, H: canvas mm -> photo px)."""
    w, h = cam.width, cam.height
    H = homography(cam, pose)
    A = np.array([[ss, 0, (ss - 1) / 2], [0, ss, (ss - 1) / 2], [0, 0, 1.0]])
    M = A @ H @ plane_to_mm(res)
    wall = np.array(look.wall_rgb[::-1], np.float32) / 255
    big = cv2.warpPerspective(plane, M, (w * ss, h * ss), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(v) for v in wall))
    img = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    gx, gy = rng.uniform(-1, 1, 2) * effects.light_gradient
    r2 = ((u - w / 2) ** 2 + (v - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
    light = rng.uniform(*effects.exposure) * (1 + gx * (u / w - 0.5) * 2 + gy * (v / h - 0.5) * 2) * (1 - effects.vignette * r2)
    img = img * light[..., None]
    sigma = rng.uniform(*effects.defocus_sigma)
    if sigma > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    if rng.uniform() < effects.motion_prob:
        ang = rng.uniform(0, np.pi)
        mv = rng.uniform(*effects.motion_px) * np.array([np.cos(ang), np.sin(ang)])
        if np.hypot(*mv) >= 1.0:
            img = motion_blur(img, mv)
    img = img + rng.standard_normal(img.shape).astype(np.float32) * (0.004 + 0.012 * np.sqrt(np.clip(img, 0, 1))) * effects.noise
    img8 = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    _, buf = cv2.imencode(".jpg", img8, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(*effects.jpeg_quality))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), H


def tracking_error(rng, H: np.ndarray, canvas_size, mm: float) -> np.ndarray:
    """H as a tracker would report it: the canvas corners land about ``mm`` off."""
    cw, ch = canvas_size
    corners = np.array([(0, 0), (cw, 0), (cw, ch), (0, ch)], np.float64)
    noisy = corners + rng.normal(0, mm / np.sqrt(2), corners.shape)
    E, _ = cv2.findHomography(corners, noisy)  # true mm -> where the tracker thinks it is
    return H @ np.linalg.inv(E)


def rectify(photo: np.ndarray, H: np.ndarray, canvas_size, res: float = 2.0) -> np.ndarray:
    """The photo resampled onto the canvas plane (as the app could do with its
    tracked pose): ``res`` px per mm, same pixel convention as ``surface``."""
    cw, ch = canvas_size
    M = H @ plane_to_mm(res)
    return cv2.warpPerspective(photo, M, (int(round(cw * res)), int(round(ch * res))), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
