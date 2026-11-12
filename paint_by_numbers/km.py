"""Kubelka-Munk paint layers: single layer, layer stack, and mixing.

All reflectances are in the working space (linear Rec.2020, see
color.WORKING_SPACE), in [0, 1], one value per channel. Thickness
``t`` is in coats; 1.0 is one full brushed coat.

The spec gives the single-layer formula as

    a  = 1 + K/S,  b = sqrt(a^2 - 1),  th = tanh(b S t)
    R  = ((1 - a Rg) th + b Rg) / ((a - Rg) th + b)

This module computes the same quantity in an equivalent form:

    R_inf = a - b = 1 / (a + b)                      (the masstone)
    w     = th (a + b - Rg) / ((a - Rg) th + b)       (0 at t = 0, 1 as t -> inf)
    R     = Rg + (R_inf - Rg) w

which follows from 1 - 2 a Rg + Rg^2 = (R_inf - Rg)(a + b - Rg). The
rearranged form:

- returns Rg bit-for-bit at t = 0 (w is exactly 0),
- keeps R between Rg and R_inf, so the output stays in [0, 1],
- avoids cancellation in ``a - b`` for dark paints (large K/S) and in
  ``a^2 - 1`` for light paints (small K/S), which matters in float32.

Only K/S and S*t enter the formula. So a glaze of paint thinned with clear
medium at concentration c, painted t coats thick, renders the same as c*t
coats of full-strength paint. A 0.25-coat level can be painted as one coat of
1 part paint to 3 parts medium.
"""

from __future__ import annotations

import math

import torch

__all__ = ["MIN_KS", "layer", "masstone", "hiding_thickness", "render_stack", "mix"]

# K and S are clamped to at least this before use (spec value). Side effect:
# for a paint with small S, the fit can give K below this on a very light
# channel, and the clamp then darkens that channel's masstone slightly. With
# S = 0.15 it happens above ~0.964 linear reflectance; with S = 3, above ~0.992.
MIN_KS = 1e-4


def _as_float(x, like: torch.Tensor | None = None) -> torch.Tensor:
    """Tensor-ify ``x``. With ``like``, match its dtype and device (differentiably)."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float64)
    if like is not None:
        return x.to(dtype=like.dtype, device=like.device)
    if not torch.is_floating_point(x):
        return x.to(torch.float64)
    return x


def _coefficients(K: torch.Tensor, S: torch.Tensor):
    """Per-paint constants: a, b, masstone R_inf, and b*S (hiding rate per coat)."""
    K = K.clamp(min=MIN_KS)
    S = S.clamp(min=MIN_KS)
    k = K / S
    a = 1.0 + k
    b = torch.sqrt(k * (k + 2.0))  # sqrt(a^2 - 1) without cancellation
    r_inf = 1.0 / (a + b)  # a - b without cancellation
    return a, b, r_inf, b * S


def _apply(rg, a, b, r_inf, bs, t):
    th = torch.tanh(bs * t)
    w = th * (a + b - rg) / ((a - rg) * th + b)
    return rg + (r_inf - rg) * w


def layer(rg, K, S, t) -> torch.Tensor:
    """Reflectance of one paint layer (K, S, thickness t) over reflectance ``rg``.

    All arguments broadcast elementwise; the result takes ``t``'s dtype.
    """
    t = _as_float(t)
    rg, K, S = (_as_float(x, like=t) for x in (rg, K, S))
    return _apply(rg, *_coefficients(K, S), t)


def masstone(K, S) -> torch.Tensor:
    """Reflectance of an infinitely thick layer: a - b."""
    K = _as_float(K)
    return _coefficients(K, _as_float(S, like=K))[2]


def hiding_thickness(K, S, tol: float = 1e-3) -> torch.Tensor:
    """Coats needed before the paint hides any background to within ``tol``.

    For any background in [0, 1], a layer at least this thick is within
    ``tol`` (in reflectance) of the masstone on every channel. Uses the bound
    |R - R_inf| <= 1 - tanh(b S t). K and S have shape (..., 3); the result
    has shape (...), the worst channel.

    Light, transparent paints need many coats: a channel at 0.95 reflectance
    with S = 0.15 needs about 500. Use this instead of a fixed "thick" value.
    """
    K = _as_float(K)
    _, _, _, bs = _coefficients(K, _as_float(S, like=K))
    return (math.atanh(1.0 - tol) / bs).amax(dim=-1)


def render_stack(background, K, S, t) -> torch.Tensor:
    """Render a stack of paint layers over a background, bottom layer first.

    Args:
        background: working-space reflectance, shape (3,) for a flat color or
            (3, *spatial) for a per-pixel background (any shape that
            broadcasts to (3, *spatial)).
        K, S: shape (n, 3), one paint per layer. Paints may repeat. May also
            be (n, 3, *spatial) for per-pixel paint constants, which lets
            many candidate stacks render side by side as one image.
        t: thickness in coats, shape (n, 1, *spatial). Any n >= 0.

    Returns:
        Working-space reflectance, shape (3, *spatial), in t's dtype and device.
    """
    t = _as_float(t)
    if t.ndim < 2 or t.shape[1] != 1:
        raise ValueError(f"t must have shape (n, 1, *spatial), got {tuple(t.shape)}")
    n, spatial = t.shape[0], t.shape[2:]
    ones = (1,) * len(spatial)

    K, S, background = (_as_float(x, like=t) for x in (K, S, background))
    for name, x in (("K", K), ("S", S)):
        if x.ndim < 2 or x.shape[0] != n or x.shape[1] != 3:
            raise ValueError(f"{name} must have shape (n={n}, 3, ...), got {tuple(x.shape)}")
    if K.ndim == 2:
        K = K.reshape(n, 3, *ones)
    if S.ndim == 2:
        S = S.reshape(n, 3, *ones)
    if background.ndim == 1:
        background = background.reshape(3, *ones)

    a, b, r_inf, bs = _coefficients(K, S)  # each (n, 3, ...), computed once
    r = background
    for i in range(n):
        r = _apply(r, a[i], b[i], r_inf[i], bs[i], t[i])
    return r.expand(torch.broadcast_shapes(r.shape, (3, *spatial)))


def mix(c, K, S) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix paints by concentration: K_mix = sum_i c_i K_i, and the same for S.

    Args:
        c: concentrations, shape (..., m). They need not sum to 1: clear
            medium has K = S = 0, so 1 part paint to 3 parts medium is
            c = 0.25 of that paint.
        K, S: shape (m, 3).

    Returns:
        (K_mix, S_mix), each shape (..., 3).
    """
    K = _as_float(K)
    S = _as_float(S, like=K)
    c = _as_float(c, like=K)
    return c @ K, c @ S
