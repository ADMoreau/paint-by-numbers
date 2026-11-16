"""Line points: the sketch's pencil lines in a photo, and the target image's edges.

Both come out as thin sets of points, each with a unit normal (across the
line). The aligner matches sketch points to target edges with similar
direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LinePoints:
    pts: np.ndarray  # (n, 2) float, px
    normals: np.ndarray  # (n, 2) unit
    strength: np.ndarray  # (n,)


def _hessian(g: np.ndarray):
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    return dxx, dyy, dxy


def sketch_lines(img: np.ndarray, polarity: str = "light", width_px: float = 3.0, k_high: float = 5.0, k_low: float = 2.5, min_pixels: int = 12, roi: np.ndarray | None = None) -> LinePoints:
    """Ridge points of pencil/chalk lines about ``width_px`` wide.

    A pixel is on a line if, across the line, the image is curved like a ridge
    (light lines) or valley (dark lines), it is the strongest point across, and
    it is lighter (darker) than the ground on *both* sides, which rules out the
    edge of a painted area. Thresholds are in units of the image's own noise.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    if polarity == "dark":
        g = 255 - g
    sigma = max(0.8, width_px / 2.5)
    s = cv2.GaussianBlur(g, (0, 0), sigma)
    dxx, dyy, dxy = _hessian(s)
    # eigen-decomposition of [[dxx, dxy], [dxy, dyy]]: the most negative eigenvalue is across a light ridge
    tr = (dxx + dyy) / 2
    disc = np.sqrt(((dxx - dyy) / 2) ** 2 + dxy**2)
    lam = tr - disc  # most negative
    other = tr + disc
    strength = np.where((lam < 0) & (np.abs(lam) > np.abs(other)), -lam, 0).astype(np.float32)
    # normal = eigenvector of lam
    nx = dxy
    ny = lam - dxx
    alt = np.abs(nx) + np.abs(ny) < 1e-6
    nx = np.where(alt, lam - dyy, nx)
    ny = np.where(alt, dxy, ny)
    nn = np.sqrt(nx**2 + ny**2) + 1e-12
    nx, ny = nx / nn, ny / nn
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def sample(a, dx, dy):
        return cv2.remap(a, (xx + dx).astype(np.float32), (yy + dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    nms = (strength >= sample(strength, nx, ny)) & (strength >= sample(strength, -nx, -ny)) & (strength > 0)
    # lighter than the ground on both sides
    off = 2.5 * sigma + 1.0
    side = np.minimum(s - sample(s, nx * off, ny * off), s - sample(s, -nx * off, -ny * off))
    noise_img = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    noise = 1.4826 * np.median(np.abs(noise_img - np.median(noise_img))) + 0.5
    st = strength[strength > 0]
    s_scale = 1.4826 * np.median(st) if st.size else 1.0
    resp = np.where(nms & (side > 2.0 * noise), strength / max(s_scale, 1e-6), 0)
    if roi is not None:
        resp = resp * roi
    low = (resp > k_low).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    high_max = np.zeros(n)
    np.maximum.at(high_max, cc.ravel(), resp.ravel())
    keep = (stats[:, cv2.CC_STAT_AREA] >= min_pixels) & (high_max > k_high)
    keep[0] = False
    m = keep[cc] & (low > 0)
    ys, xs = np.nonzero(m)
    return LinePoints(np.column_stack([xs, ys]).astype(float), np.column_stack([nx[ys, xs], ny[ys, xs]]), resp[ys, xs])


def image_edges(img: np.ndarray, sigmas=(1.5, 3.0), high_pct: float = 85.0, min_chain: float = 0.0, min_strength: float = 0.0, min_grad: float = 20.0) -> LinePoints:
    """Colour edges of the target at a couple of scales (Canny on the Lab
    channel that changes most), with their gradient direction as normal."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = img.shape[:2]
    edges = np.zeros((h, w), bool)
    nx = np.zeros((h, w), np.float32)
    ny = np.zeros((h, w), np.float32)
    mag_all = np.zeros((h, w), np.float32)
    for sg in sigmas:
        b = cv2.GaussianBlur(lab, (0, 0), sg)
        gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
        mag = gx**2 + gy**2
        k = np.argmax(mag, axis=2)
        gxk = np.take_along_axis(gx, k[..., None], 2)[..., 0]
        gyk = np.take_along_axis(gy, k[..., None], 2)[..., 0]
        m = np.sqrt(gxk**2 + gyk**2)
        hi = max(np.percentile(m, high_pct), min_grad)  # absolute floor: smooth gradients and noise are not edges
        scale = 2000.0 / max(m.max(), 1e-6)
        e = cv2.Canny((gxk * scale).astype(np.int16), (gyk * scale).astype(np.int16), 0.4 * hi * scale, hi * scale, L2gradient=True) > 0
        new = e & ~edges
        nx[new] = gxk[new] / np.maximum(m[new], 1e-6)
        ny[new] = gyk[new] / np.maximum(m[new], 1e-6)
        mag_all[new] = m[new]
        edges |= e
    if min_chain > 0 or min_strength > 0:
        # keep long, strong edge chains: what a painter would outline, not texture
        n, cc, stats, _ = cv2.connectedComponentsWithStats(edges.astype(np.uint8), connectivity=8)
        mean_mag = np.bincount(cc.ravel(), weights=mag_all.ravel(), minlength=n) / np.maximum(stats[:, cv2.CC_STAT_AREA], 1)
        ref = np.median(mag_all[edges]) if edges.any() else 1.0
        keep = (stats[:, cv2.CC_STAT_AREA] >= min_chain * max(h, w)) & (mean_mag >= min_strength * ref)
        keep[0] = False
        edges = keep[cc]
    ys, xs = np.nonzero(edges)
    return LinePoints(np.column_stack([xs, ys]).astype(float), np.column_stack([nx[ys, xs], ny[ys, xs]]), mag_all[ys, xs])
