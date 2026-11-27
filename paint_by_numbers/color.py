"""Color conversions between sRGB, linear RGB, CIE XYZ and CIE L*a*b*.

Everything is written in torch so these functions can sit inside the
optimizer's loss and pass gradients back to the thickness maps.

Two spaces appear besides XYZ and Lab:

- linear sRGB (``linear_*``): for images coming in and going out.
- the working space (``working_*``): spectral reflectance in CHANNELS bands
  across the visible spectrum, where the Kubelka-Munk model runs. See
  WORKING_SPACE.

Conventions:

- Color channels (3, or CHANNELS for the working space) sit on axis ``dim``
  (default: the last axis), so both ``(..., C)`` and ``(C, H, W)`` layouts work.
- sRGB and linear RGB are nominally in [0, 1]. Nothing is clipped: values
  outside that range pass through, so out-of-gamut Lab colors survive a round
  trip. Clip only when writing images.
- XYZ is scaled so the white point has Y = 1.
- Illuminant D65, 2 degree observer (an open question in the spec for Golden's
  published values). The white point is the row sum of the sRGB matrix, so
  linear RGB (1, 1, 1) maps to exactly L* = 100, a* = b* = 0.
- Non-tensor inputs (lists, NumPy arrays) become float64 tensors. Float
  tensors keep their dtype and device.
- Every piecewise branch clamps its input before any power, so gradients are
  finite everywhere, including at 0 and at negative values.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

__all__ = [
    "srgb_to_linear",
    "linear_to_srgb",
    "linear_to_xyz",
    "xyz_to_linear",
    "xyz_to_lab",
    "lab_to_xyz",
    "linear_to_lab",
    "lab_to_linear",
    "srgb_to_lab",
    "lab_to_srgb",
    "WORKING_SPACE",
    "CHANNELS",
    "BAND_CENTRES_NM",
    "working_to_xyz",
    "xyz_to_working",
    "working_to_lab",
    "lab_to_working",
    "srgb_to_working",
    "working_to_srgb",
    "delta_e",
    "delta_e_squared",
]

# IEC 61966-2-1 sRGB primaries with a D65 white.
_RGB_TO_XYZ = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=torch.float64,
)
_XYZ_TO_RGB = torch.linalg.inv(_RGB_TO_XYZ)
_WHITE_XYZ = _RGB_TO_XYZ.sum(dim=1)  # (0.95047, 1.0, 1.08883)


def _rgb_to_xyz_matrix(primaries_xy, white_xyz: torch.Tensor) -> torch.Tensor:
    """RGB -> XYZ matrix from primary chromaticities, scaled so RGB (1, 1, 1) is ``white_xyz``."""
    p = torch.tensor([[x / y, 1.0, (1 - x - y) / y] for x, y in primaries_xy], dtype=torch.float64).T
    return p * torch.linalg.solve(p, white_xyz)


# Working space for paint reflectance: spectral reflectance in bands across the visible spectrum
# (data/spectral_bands.json, from scripts/make_spectral_bands.py). Three channels were not enough:
# a paint like Dioxazine Purple passes a little deep-violet light even when thick, so a heavy coat
# over white stays violet, but averaged over a broad "blue" channel that light dies out, and the
# model turned it gray from a 1:1 glaze up. Bands are summed CIE 1931 2 degree observer times D65,
# scaled so reflectance 1 in every band is exactly Lab's white.
_BANDS = json.loads((Path(__file__).parent / "data" / "spectral_bands.json").read_text())
BAND_CENTRES_NM = tuple(_BANDS["centres_nm"])
CHANNELS = len(BAND_CENTRES_NM)
WORKING_SPACE = f"spectral reflectance, {CHANNELS} bands ({BAND_CENTRES_NM[0]}-{BAND_CENTRES_NM[-1]} nm)"
_WORKING_TO_XYZ = torch.tensor(_BANDS["to_xyz"], dtype=torch.float64)  # (3, CHANNELS)
_WORKING_TO_XYZ = _WORKING_TO_XYZ * (_WHITE_XYZ / _WORKING_TO_XYZ.sum(dim=1)).unsqueeze(1)  # white exact, past rounding
_XYZ_TO_SMOOTH = torch.tensor(_BANDS["from_xyz"], dtype=torch.float64)  # (CHANNELS, 3): smoothest spectrum
_XYZ_TO_SMOOTH = _XYZ_TO_SMOOTH @ torch.linalg.inv(_WORKING_TO_XYZ @ _XYZ_TO_SMOOTH)  # an exact inverse of the above
# Fallback spectra for colors whose smoothest spectrum leaves [0, 1] (saturated ones): a sigmoid of
# a quadratic in wavelength (Jakob and Hanika 2019), always inside (0, 1).
_X = (torch.tensor(BAND_CENTRES_NM, dtype=torch.float64) - 550.0) / 150.0
_SIGMOID_BASIS = torch.stack([torch.ones_like(_X), _X, _X * _X], dim=1)  # (CHANNELS, 3)

# sRGB transfer function.
_SRGB_KNEE = 0.04045  # in encoded sRGB
# The standard quotes 0.0031308, which is not exactly 0.04045 / 12.92 and makes
# the round trip jump by ~3e-8 at the knee. Derive it so the inverse is exact.
_LINEAR_KNEE = _SRGB_KNEE / 12.92  # in linear RGB, ~0.00313080495
_SRGB_GAMMA = 2.4

# CIE L*a*b* companding.
_LAB_DELTA = 6.0 / 29.0
_LAB_DELTA_CUBED = _LAB_DELTA**3
_LAB_SLOPE = 1.0 / (3.0 * _LAB_DELTA**2)
_LAB_OFFSET = 4.0 / 29.0


def _prep(x, dim: int, channels: int = 3) -> torch.Tensor:
    """Return ``x`` as a floating tensor and check it has ``channels`` channels on ``dim``."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float64)
    elif not torch.is_floating_point(x):
        x = x.to(torch.float64)
    if x.ndim == 0 or x.shape[dim] != channels:
        raise ValueError(
            f"expected {channels} color channels on dim {dim}, got shape {tuple(x.shape)}"
        )
    return x


def _channel_shape(x: torch.Tensor, dim: int) -> list[int]:
    """Shape that broadcasts a length-3 vector along ``dim`` of ``x``."""
    shape = [1] * x.ndim
    shape[dim] = 3
    return shape


def _apply_matrix(matrix: torch.Tensor, x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x.movedim(dim, -1)
    y = x @ matrix.to(x).T
    return y.movedim(-1, dim)


# ---------------------------------------------------------------- sRGB <-> linear


def srgb_to_linear(srgb, dim: int = -1) -> torch.Tensor:
    """Decode gamma-encoded sRGB in [0, 1] to linear RGB."""
    x = _prep(srgb, dim)
    low = x / 12.92
    high = ((x.clamp(min=_SRGB_KNEE) + 0.055) / 1.055) ** _SRGB_GAMMA
    return torch.where(x <= _SRGB_KNEE, low, high)


def linear_to_srgb(linear, dim: int = -1) -> torch.Tensor:
    """Encode linear RGB to gamma-encoded sRGB. Does not clip."""
    x = _prep(linear, dim)
    low = x * 12.92
    high = 1.055 * x.clamp(min=_LINEAR_KNEE) ** (1.0 / _SRGB_GAMMA) - 0.055
    return torch.where(x <= _LINEAR_KNEE, low, high)


# ---------------------------------------------------------------- linear <-> XYZ


def linear_to_xyz(linear, dim: int = -1) -> torch.Tensor:
    """Linear sRGB to CIE XYZ (D65, white Y = 1)."""
    x = _prep(linear, dim)
    return _apply_matrix(_RGB_TO_XYZ, x, dim)


def xyz_to_linear(xyz, dim: int = -1) -> torch.Tensor:
    """CIE XYZ (D65, white Y = 1) to linear sRGB."""
    x = _prep(xyz, dim)
    return _apply_matrix(_XYZ_TO_RGB, x, dim)


# ---------------------------------------------------------------- XYZ <-> Lab


def _lab_f(t: torch.Tensor) -> torch.Tensor:
    cube_root = t.clamp(min=_LAB_DELTA_CUBED) ** (1.0 / 3.0)
    linear = t * _LAB_SLOPE + _LAB_OFFSET
    return torch.where(t > _LAB_DELTA_CUBED, cube_root, linear)


def _lab_f_inv(f: torch.Tensor) -> torch.Tensor:
    cube = f**3
    linear = (f - _LAB_OFFSET) / _LAB_SLOPE
    return torch.where(f > _LAB_DELTA, cube, linear)


def xyz_to_lab(xyz, dim: int = -1) -> torch.Tensor:
    """CIE XYZ (D65, white Y = 1) to CIE L*a*b*."""
    x = _prep(xyz, dim)
    white = _WHITE_XYZ.to(x).reshape(_channel_shape(x, dim))
    fx, fy, fz = _lab_f(x / white).unbind(dim)
    lightness = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack((lightness, a, b), dim)


def lab_to_xyz(lab, dim: int = -1) -> torch.Tensor:
    """CIE L*a*b* to CIE XYZ (D65, white Y = 1)."""
    x = _prep(lab, dim)
    lightness, a, b = x.unbind(dim)
    fy = (lightness + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    white = _WHITE_XYZ.to(x).reshape(_channel_shape(x, dim))
    return _lab_f_inv(torch.stack((fx, fy, fz), dim)) * white


# ---------------------------------------------------------------- composites


def linear_to_lab(linear, dim: int = -1) -> torch.Tensor:
    """Linear RGB (e.g. a rendered reflectance) to CIE L*a*b*."""
    return xyz_to_lab(linear_to_xyz(linear, dim), dim)


def lab_to_linear(lab, dim: int = -1) -> torch.Tensor:
    """CIE L*a*b* to linear RGB."""
    return xyz_to_linear(lab_to_xyz(lab, dim), dim)


def srgb_to_lab(srgb, dim: int = -1) -> torch.Tensor:
    """Gamma-encoded sRGB in [0, 1] to CIE L*a*b*."""
    return linear_to_lab(srgb_to_linear(srgb, dim), dim)


def lab_to_srgb(lab, dim: int = -1) -> torch.Tensor:
    """CIE L*a*b* to gamma-encoded sRGB. Does not clip."""
    return linear_to_srgb(lab_to_linear(lab, dim), dim)


# ---------------------------------------------------------------- working space (spectral bands)


def working_to_xyz(r, dim: int = -1) -> torch.Tensor:
    """Working-space (spectral) reflectance to CIE XYZ."""
    return _apply_matrix(_WORKING_TO_XYZ, _prep(r, dim, CHANNELS), dim)


def _sigmoid_spectra(xyz: torch.Tensor, iterations: int = 60) -> torch.Tensor:
    """Spectra sigmoid(c0 + c1 x + c2 x^2) matching ``xyz`` (..., 3), by damped Gauss-Newton.

    Colors no reflectance in [0, 1] can reach (outside the object-color solid) come back as close
    as the sigmoid family gets."""
    M, B = _WORKING_TO_XYZ.to(xyz), _SIGMOID_BASIS.to(xyz)
    y = xyz[..., 1].clamp(1e-4, 1 - 1e-4)
    c = torch.stack([torch.log(y / (1 - y)), torch.zeros_like(y), torch.zeros_like(y)], dim=-1)
    eye = torch.eye(3, dtype=xyz.dtype, device=xyz.device)
    for _ in range(iterations):
        r = torch.sigmoid(c @ B.T)
        res = r @ M.T - xyz
        J = M @ ((r * (1 - r)).unsqueeze(-1) * B)  # (..., 3, 3): d xyz / d c
        step = torch.linalg.solve(J.transpose(-1, -2) @ J + 1e-10 * eye, (J.transpose(-1, -2) @ res.unsqueeze(-1))).squeeze(-1)
        c = c - step.clamp(-4.0, 4.0)
    return torch.sigmoid(c @ B.T)


def xyz_to_working(xyz, dim: int = -1) -> torch.Tensor:
    """CIE XYZ to a working-space (spectral) reflectance with that color.

    Many spectra share a color; this picks the smoothest (least squared second differences, so a
    neutral is flat), or, where that leaves [0, 1], a smooth sigmoid-shaped one. Either way the
    result converts back to the same XYZ (to rounding) whenever any reflectance in [0, 1] can."""
    x = _prep(xyz, dim).movedim(dim, -1)
    with torch.no_grad():
        smooth = x @ _XYZ_TO_SMOOTH.to(x).T
        bad = ((smooth < 0) | (smooth > 1)).any(dim=-1)
        if bad.any():
            smooth[bad] = _sigmoid_spectra(x[bad].double()).to(smooth)
    return smooth.movedim(-1, dim)


def working_to_lab(r, dim: int = -1) -> torch.Tensor:
    """Working-space reflectance to CIE L*a*b*."""
    return xyz_to_lab(working_to_xyz(r, dim), dim)


def lab_to_working(lab, dim: int = -1) -> torch.Tensor:
    """CIE L*a*b* to a working-space reflectance with that color (see xyz_to_working)."""
    return xyz_to_working(lab_to_xyz(lab, dim), dim)


def srgb_to_working(srgb, dim: int = -1) -> torch.Tensor:
    """Gamma-encoded sRGB to a working-space reflectance with that color."""
    return xyz_to_working(linear_to_xyz(srgb_to_linear(srgb, dim), dim), dim)


def working_to_srgb(r, dim: int = -1) -> torch.Tensor:
    """Working-space reflectance to gamma-encoded sRGB. Does not clip: colors
    outside sRGB come back outside [0, 1]."""
    return linear_to_srgb(xyz_to_linear(working_to_xyz(r, dim), dim), dim)


# ---------------------------------------------------------------- color error


def delta_e_squared(lab1, lab2, dim: int = -1) -> torch.Tensor:
    """Squared Euclidean Lab distance. Smooth everywhere, so use it in losses."""
    diff = _prep(lab1, dim) - _prep(lab2, dim)
    return (diff**2).sum(dim)


def delta_e(lab1, lab2, dim: int = -1) -> torch.Tensor:
    """CIE76 color difference (Euclidean Lab distance), for reporting."""
    diff = _prep(lab1, dim) - _prep(lab2, dim)
    return torch.linalg.vector_norm(diff, dim=dim)
