"""Color conversions between sRGB, linear RGB, CIE XYZ and CIE L*a*b*.

Everything is written in torch so these functions can sit inside the
optimizer's loss and pass gradients back to the thickness maps.

Two linear RGB spaces appear:

- linear sRGB (``linear_*``): for images coming in and going out.
- the working space (``working_*``), linear Rec.2020: where paint
  reflectance lives and the Kubelka-Munk model runs. See WORKING_SPACE.

Conventions:

- The three color channels sit on axis ``dim`` (default: the last axis), so
  both ``(..., 3)`` and ``(3, H, W)`` layouts work.
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


# Working space for paint reflectance: linear RGB with ITU-R BT.2020 primaries.
# Its gamut holds nearly all real surface colors, so saturated paints (cadmium
# yellows, cobalt teals) keep every channel in [0, 1]; in linear sRGB 20 of
# Golden's Heavy Body colors fall outside and were clipped by up to 15 ΔE. It
# shares Lab's white point, so neutral grays are identical in both spaces.
WORKING_SPACE = "linear Rec.2020"
_REC2020_PRIMARIES = ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))
_WORKING_TO_XYZ = _rgb_to_xyz_matrix(_REC2020_PRIMARIES, _WHITE_XYZ)
_XYZ_TO_WORKING = torch.linalg.inv(_WORKING_TO_XYZ)

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


def _prep(x, dim: int) -> torch.Tensor:
    """Return ``x`` as a floating tensor and check it has 3 channels on ``dim``."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float64)
    elif not torch.is_floating_point(x):
        x = x.to(torch.float64)
    if x.ndim == 0 or x.shape[dim] != 3:
        raise ValueError(
            f"expected 3 color channels on dim {dim}, got shape {tuple(x.shape)}"
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


# ---------------------------------------------------------------- working space (linear Rec.2020)


def working_to_xyz(rgb, dim: int = -1) -> torch.Tensor:
    """Working-space reflectance (linear Rec.2020) to CIE XYZ."""
    return _apply_matrix(_WORKING_TO_XYZ, _prep(rgb, dim), dim)


def xyz_to_working(xyz, dim: int = -1) -> torch.Tensor:
    """CIE XYZ to working-space reflectance (linear Rec.2020)."""
    return _apply_matrix(_XYZ_TO_WORKING, _prep(xyz, dim), dim)


def working_to_lab(rgb, dim: int = -1) -> torch.Tensor:
    """Working-space reflectance to CIE L*a*b*."""
    return xyz_to_lab(working_to_xyz(rgb, dim), dim)


def lab_to_working(lab, dim: int = -1) -> torch.Tensor:
    """CIE L*a*b* to working-space reflectance."""
    return xyz_to_working(lab_to_xyz(lab, dim), dim)


def srgb_to_working(srgb, dim: int = -1) -> torch.Tensor:
    """Gamma-encoded sRGB to working-space reflectance."""
    return xyz_to_working(linear_to_xyz(srgb_to_linear(srgb, dim), dim), dim)


def working_to_srgb(rgb, dim: int = -1) -> torch.Tensor:
    """Working-space reflectance to gamma-encoded sRGB. Does not clip: colors
    outside sRGB come back outside [0, 1]."""
    return linear_to_srgb(xyz_to_linear(working_to_xyz(rgb, dim), dim), dim)


# ---------------------------------------------------------------- color error


def delta_e_squared(lab1, lab2, dim: int = -1) -> torch.Tensor:
    """Squared Euclidean Lab distance. Smooth everywhere, so use it in losses."""
    diff = _prep(lab1, dim) - _prep(lab2, dim)
    return (diff**2).sum(dim)


def delta_e(lab1, lab2, dim: int = -1) -> torch.Tensor:
    """CIE76 color difference (Euclidean Lab distance), for reporting."""
    diff = _prep(lab1, dim) - _prep(lab2, dim)
    return torch.linalg.vector_norm(diff, dim=dim)
