"""Thickness-map optimizer and quantization.

Given a target image (Lab), a background and a layer stack (one paint per
layer, bottom first, any number of layers, paints may repeat), find a
thickness map per layer so the rendered stack matches the target.

- Parameters ``u`` of shape (n, 1, H, W); thickness ``t = t_max * sigmoid(u)``,
  so 0 <= t <= t_max.
- Loss: mean squared Lab distance between rendered stack and target, plus
  ``lam_smooth`` * total variation of t, plus ``lam_paint`` * mean paint.
  Both penalties use each pixel's total over all layers, averaged over
  pixels, so their weight does not shrink as layers are added.
- Adam, then quantize: snap t to the nearest of ``levels``, then refine the
  levels by coordinate descent, and re-render. Errors are reported before
  quantizing, after nearest snapping, and after refinement.

Where this departs from the spec's first-draft settings, and why (measured
with all 9 database paints as the stack):

- ``init_u = -10`` (t ~ 1e-4 coats), not -3 (t ~ 0.095). The spec's start
  puts 0.095 coats on every layer, which grows with layer count, and it often
  settled on a paint mixture where no paint was the answer: a gray target on
  a gray background kept 1.2 coats.
- Adam ``beta2 = 0.99``, not 0.999. The large early gradients otherwise
  shrink later steps ~1000x and the solve stalls.
- Refinement after snapping. The continuous optimum often uses traces of
  high-S paints (a few thousandths of a coat of Titanium White) that nearest
  snapping zeroes, which cost up to 30 ΔE. Coordinate descent from several
  starts (nearest, floor, ceiling, zero) recovers it. It minimizes ΔE plus a
  small cost per coat (``refine_paint_cost``) so hidden paint, such as a coat
  under an opaque layer, is dropped. ``refine_smooth`` charges for each
  neighbor on a different level so regions stay in one piece, and
  ``max_thicknesses`` limits how many levels each layer may use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from . import color, km

__all__ = [
    "SolveConfig",
    "Solution",
    "solve",
    "snap",
    "refine_levels",
    "choose_layer_levels",
    "total_variation",
    "total_paint",
]


@dataclass(frozen=True)
class SolveConfig:
    t_max: float = 2.0  # coats, per layer
    lam_smooth: float = 0.05
    lam_paint: float = 0.01
    lr: float = 0.05
    beta2: float = 0.99  # Adam second-moment decay
    steps: int = 2000
    init_u: float = -10.0
    levels: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)  # coats
    refine_sweeps: int = 6  # max coordinate-descent sweeps per start; 0 = nearest snap only
    refine_paint_cost: float = 0.1  # ΔE charged per coat during refinement, so hidden paint is dropped
    max_thicknesses: int | None = None  # different nonzero thicknesses per layer; None: all levels
    refine_smooth: float = 2.0  # ΔE per neighbor on a different level, favoring regions over speckle; 0 = off


@dataclass
class Solution:
    t: torch.Tensor  # (n, 1, *spatial) continuous thickness, coats
    t_snapped: torch.Tensor  # (n, 1, *spatial) quantized to levels (after refinement)
    preview: torch.Tensor  # (3, *spatial) working-space reflectance of the quantized stack
    error: torch.Tensor  # (*spatial) ΔE76 of the quantized stack vs target
    mean_error: float  # quantized
    max_error: float  # quantized
    mean_error_continuous: float
    max_error_continuous: float
    mean_error_nearest: float  # nearest snapping without refinement
    loss_history: list[float] = field(default_factory=list)  # every 10 steps


def total_variation(t: torch.Tensor) -> torch.Tensor:
    """Mean over pixels of sum over layers of |dt/dx| + |dt/dy|. t: (n, 1, *spatial)."""
    tv = t.new_zeros(())
    for dim in range(2, t.ndim):
        if t.shape[dim] > 1:
            diff = torch.diff(t, dim=dim).abs()
            tv = tv + diff.sum(dim=0).mean()
    return tv


def total_paint(t: torch.Tensor) -> torch.Tensor:
    """Mean over pixels of total coats across all layers. t: (n, 1, *spatial)."""
    return t.sum(dim=0).mean() if t.shape[0] else t.new_zeros(())


def snap(t: torch.Tensor, levels) -> torch.Tensor:
    """Snap every thickness to the nearest level."""
    levels = torch.as_tensor(levels, dtype=t.dtype, device=t.device)
    return levels[(t.unsqueeze(-1) - levels).abs().argmin(dim=-1)]


def _snap_down(t: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    return levels[(torch.searchsorted(levels, t.contiguous(), right=True) - 1).clamp(0, len(levels) - 1)]


def _snap_up(t: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    return levels[torch.searchsorted(levels, t.contiguous(), right=False).clamp(0, len(levels) - 1)]


def _render_error(target_lab, background, K, S, t):
    out = km.render_stack(background, K, S, t)
    return out, color.delta_e(color.working_to_lab(out, dim=0), target_lab, dim=0)


def choose_layer_levels(t: torch.Tensor, levels, max_thicknesses: int | None) -> list[torch.Tensor]:
    """The levels each layer may use: 0 plus at most ``max_thicknesses`` others.

    For each layer, picks the subset of nonzero levels that best fits its
    continuous thickness (least squared distance to the nearest allowed level,
    0 always allowed), trying every subset. ``None`` allows every level.
    """
    from itertools import combinations

    levels = torch.as_tensor(levels, dtype=t.dtype, device=t.device).sort().values
    nonzero = levels[levels > 0]
    zero = levels.new_zeros(1)
    if max_thicknesses is None or max_thicknesses >= len(nonzero):
        return [torch.cat([zero, nonzero])] * t.shape[0]
    if max_thicknesses < 1:
        raise ValueError("max_thicknesses must be at least 1")
    subsets = [torch.cat([zero, nonzero[list(c)]]) for c in combinations(range(len(nonzero)), max_thicknesses)]
    chosen = []
    for i in range(t.shape[0]):
        values = t[i].reshape(-1)
        costs = [(values.unsqueeze(1) - s).abs().min(dim=1).values.pow(2).sum().item() for s in subsets]
        chosen.append(subsets[min(range(len(subsets)), key=costs.__getitem__)])
    return chosen


def _snap_each(t: torch.Tensor, layer_levels: list[torch.Tensor], how) -> torch.Tensor:
    return torch.stack([how(t[i], layer_levels[i]) for i in range(t.shape[0])])


def _neighbor_mismatch(layer: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """For each candidate level, how many of each pixel's 4 neighbors differ from it.

    layer: (H, W); levels: (L,). Returns (L, H, W).
    """
    h, w = layer.shape
    counts = layer.new_zeros(len(levels), h, w)
    lv = levels.view(-1, 1, 1)
    for shift, dim in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        neighbor = torch.roll(layer, shifts=shift, dims=dim)
        valid = torch.ones_like(layer, dtype=torch.bool)
        edge = 0 if shift == 1 else -1  # the row/column that wrapped around
        if dim == 0:
            valid[edge, :] = False
        else:
            valid[:, edge] = False
        counts += (valid & (neighbor != lv)).to(layer.dtype)
    return counts


def _coordinate_descent(target_lab, background, K, S, t, layer_levels, sweeps, paint_cost, smooth=0.0):
    """Per layer, give every pixel its best allowed level, sweeping until stable.

    Objective per pixel: ΔE + paint_cost * total coats, plus ``smooth`` for each
    of its 4 neighbors on a different level in that layer (Potts smoothing,
    needs a 2-D image). With smoothing, pixels update in a checkerboard so
    neighbors don't flip together. Returns t and the per-pixel objective
    without the smoothing term.
    """
    t = t.clone()
    use_smooth = smooth > 0 and t.ndim == 4
    if use_smooth:
        yy, xx = torch.meshgrid(torch.arange(t.shape[2]), torch.arange(t.shape[3]), indexing="ij")
        parities = [(yy + xx) % 2 == p for p in (0, 1)]

    def data_term(tt):
        return _render_error(target_lab, background, K, S, tt)[1] + paint_cost * tt.sum(dim=(0, 1))

    for _ in range(sweeps):
        changed = False
        for i in range(t.shape[0]):
            levels = layer_levels[i]
            costs = []
            for level in levels:
                trial = t.clone()
                trial[i] = level
                costs.append(data_term(trial))
            costs = torch.stack(costs)  # (L, *spatial); other layers fixed, so valid for every pixel
            for region in parities if use_smooth else [None]:
                current = t[i, 0]
                total = costs + smooth * _neighbor_mismatch(current, levels) if use_smooth else costs.clone()
                total -= 1e-4 * (levels.view(-1, *[1] * current.ndim) == current)  # keep current unless clearly better
                choice = levels[total.argmin(dim=0)]
                update = choice != current
                if region is not None:
                    update &= region
                if update.any():
                    changed = True
                    t[i, 0] = torch.where(update, choice, current)
        if not changed:
            break
    return t, data_term(t)


def refine_levels(
    target_lab,
    background,
    K,
    S,
    t,
    levels,
    sweeps: int = 6,
    paint_cost: float = 0.1,
    max_thicknesses: int | None = None,
    smooth: float = 0.0,
) -> torch.Tensor:
    """Quantize continuous thickness ``t`` to ``levels``.

    Minimizes per-pixel ΔE + ``paint_cost`` * total coats, so paint that is
    hidden or barely helps is dropped. Runs coordinate descent from four starts
    (nearest, floor, ceiling and all-zero levels) and keeps the best result per
    pixel.

    ``max_thicknesses`` limits each layer to that many different nonzero
    thicknesses (see choose_layer_levels). ``smooth`` (ΔE per differing
    neighbor) favors regions over speckle; it applies in a final pass over the
    combined result. With both off, the result never scores worse than nearest
    snapping.
    """
    levels = torch.as_tensor(levels, dtype=t.dtype, device=t.device).sort().values
    layer_levels = choose_layer_levels(t, levels, max_thicknesses)
    starts = [
        _snap_each(t, layer_levels, snap),
        _snap_each(t, layer_levels, _snap_down),
        _snap_each(t, layer_levels, _snap_up),
        torch.zeros_like(t),
    ]
    best_t, best_err = None, None
    for start in starts:
        cand_t, cand_err = _coordinate_descent(target_lab, background, K, S, start, layer_levels, sweeps, paint_cost)
        if best_t is None:
            best_t, best_err = cand_t, cand_err
        else:
            better = cand_err < best_err
            best_t = torch.where(better, cand_t, best_t)  # (spatial) broadcasts over (n, 1)
            best_err = torch.where(better, cand_err, best_err)
    if smooth > 0:
        best_t, _ = _coordinate_descent(target_lab, background, K, S, best_t, layer_levels, sweeps, paint_cost, smooth)
    return best_t


def solve(
    target_lab: torch.Tensor,
    background,
    K,
    S,
    config: SolveConfig = SolveConfig(),
    progress: Callable[[int, float], None] | None = None,
) -> Solution:
    """Optimize per-layer thickness maps.

    Args:
        target_lab: target image in L*a*b*, shape (3, *spatial). Its dtype
            (float32 if not floating) sets the working precision.
        background: working-space reflectance, (3,) or broadcastable to (3, *spatial).
        K, S: layer paints, (n, 3) bottom first, or (n, 3, *spatial).
        config: solver settings.
        progress: called as progress(step, loss) every 100 steps.
    """
    target_lab = torch.as_tensor(target_lab)
    if not torch.is_floating_point(target_lab):
        target_lab = target_lab.to(torch.float32)
    if target_lab.ndim < 2 or target_lab.shape[0] != 3:
        raise ValueError(f"target_lab must have shape (3, *spatial), got {tuple(target_lab.shape)}")
    K = torch.as_tensor(K)
    S = torch.as_tensor(S)
    n, spatial = K.shape[0], target_lab.shape[1:]
    dtype = target_lab.dtype

    u = torch.full((n, 1, *spatial), config.init_u, dtype=dtype, requires_grad=True)
    history: list[float] = []
    if n > 0 and config.steps > 0:
        opt = torch.optim.Adam([u], lr=config.lr, betas=(0.9, config.beta2))
        for step in range(config.steps):
            opt.zero_grad(set_to_none=True)
            t = config.t_max * torch.sigmoid(u)
            out = km.render_stack(background, K, S, t)
            color_loss = color.delta_e_squared(color.working_to_lab(out, dim=0), target_lab, dim=0).mean()
            loss = color_loss + config.lam_smooth * total_variation(t) + config.lam_paint * total_paint(t)
            loss.backward()
            opt.step()
            if step % 10 == 0 or step == config.steps - 1:
                history.append(loss.item())
            if progress and (step % 100 == 0 or step == config.steps - 1):
                progress(step, loss.item())

    with torch.no_grad():
        t = config.t_max * torch.sigmoid(u)
        _, err_cont = _render_error(target_lab, background, K, S, t)
        t_nearest = snap(t, config.levels)
        _, err_nearest = _render_error(target_lab, background, K, S, t_nearest)
        if n > 0 and config.refine_sweeps > 0:
            t_snapped = refine_levels(
                target_lab, background, K, S, t, config.levels, config.refine_sweeps, config.refine_paint_cost,
                config.max_thicknesses, config.refine_smooth,
            )  # fmt: skip
        else:
            t_snapped = t_nearest
        preview, err = _render_error(target_lab, background, K, S, t_snapped)

    return Solution(
        t=t.detach(),
        t_snapped=t_snapped,
        preview=preview,
        error=err,
        mean_error=err.mean().item(),
        max_error=err.max().item(),
        mean_error_continuous=err_cont.mean().item(),
        max_error_continuous=err_cont.max().item(),
        mean_error_nearest=err_nearest.mean().item(),
        loss_history=history,
    )
