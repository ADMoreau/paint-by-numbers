"""Synthetic test images."""

import numpy as np
from PIL import Image


def landscape(width: int = 96, height: int = 72) -> Image.Image:
    """Sky gradient, green hill, red barn, dark tree and a yellow sun."""
    y, x = np.mgrid[0:height, 0:width] / np.array([height, width]).reshape(2, 1, 1)
    img = np.zeros((height, width, 3))
    img[:] = np.stack([0.45 + 0.5 * y, 0.65 + 0.33 * y, 0.95 + 0.0 * y], -1)
    hill = y > 0.55 + 0.12 * np.sin(6 * x)
    img[hill] = np.stack([0.25 + 0.2 * x, 0.5 - 0.15 * y, 0.15 + 0 * x], -1)[hill]
    img[(x > 0.6) & (x < 0.8) & (y > 0.45) & (y < 0.7)] = [0.7, 0.15, 0.12]
    img[((x - 0.25) ** 2 / 0.006 + (y - 0.5) ** 2 / 0.03) < 1] = [0.12, 0.2, 0.1]
    img[((x - 0.15) ** 2 + (y - 0.15) ** 2) < 0.006] = [1.0, 0.85, 0.3]
    return Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8))
