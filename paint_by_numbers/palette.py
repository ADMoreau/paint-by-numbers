"""Paint selection and layer ordering.

Builds the layer stack greedily (spec, "Palette and order"):

1. K-means the target into ``n_colors`` colors in Lab.
2. If the background is darker (L*) than the image's median, start the stack
   with the lightest opaque paint available.
3. Each round, try every allowed paint at every insertion position in the
   current stack. Paints may repeat, but never directly on or under a layer of
   the same paint (two adjacent layers of one paint render as one thicker
   layer). All candidates of a round are solved together as one batched image
   (n_colors x candidates) with per-pixel K and S, and ranked by their error
   after quantizing to the solver's levels, since that is what gets painted.
4. Keep the best (paint, position). Stop when it improves the error by less
   than ``min_gain`` Lab units, or at ``max_layers``. With ``max_paints``, once
   that many different paints are in the stack, only those may be added again.

The error is the RMS of ΔE over the clusters, not the plain mean the spec's
first draft used. A small vivid feature (a sun, one cluster of 64) that is
30 ΔE off moves the mean by under 0.5 and was never painted; RMS weighs large
misses more, matching the solver's squared loss.

Ties (within TIE_DELTA_E) go to the stack with fewer ordering violations:
opaque below transparent, and among equal ratings, lighter below darker.

With a per-pixel background, each cluster uses the mean background of its
pixels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from sklearn.cluster import KMeans

from . import color, km
from .paints import TRANSPARENCY_RATINGS, Paint, ks_tensors
from .solve import SolveConfig, solve

__all__ = ["PaletteConfig", "Selection", "cluster_colors", "candidates", "order_violations", "select_stack"]

TIE_DELTA_E = 0.05


@dataclass(frozen=True)
class PaletteConfig:
    n_colors: int = 64
    min_gain: float = 0.5  # Lab units of RMS error a new layer must save
    max_layers: int | None = None  # None: no limit
    max_paints: int | None = None  # most different paints in the stack (repeats allowed); None: no limit
    steps: int = 400  # optimizer steps per selection round
    seed: int = 0  # k-means


@dataclass
class Selection:
    stack: list[Paint]  # bottom first
    forced_first: Paint | None  # lightest opaque, placed because the background is dark
    baseline_error: float  # RMS error of the bare background over the clusters
    rounds: list[dict] = field(default_factory=list)  # one per added layer
    cluster_lab: torch.Tensor | None = None  # (k, 3)


def cluster_colors(target_lab: torch.Tensor, n_colors: int = 64, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """K-means the image's pixels in Lab.

    Args:
        target_lab: (3, H, W).
    Returns:
        centers (k, 3) and per-pixel labels (H*W,), k <= n_colors.
    """
    pixels = target_lab.reshape(3, -1).T.double().numpy()
    k = min(n_colors, len(np.unique(pixels.round(3), axis=0)))
    model = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(pixels)
    return torch.from_numpy(model.cluster_centers_), torch.from_numpy(model.labels_)


def candidates(stack: Sequence[int], allowed: Sequence[int]) -> list[tuple[int, int]]:
    """Every (paint, insertion position) that does not sit next to the same paint."""
    out = []
    for pos in range(len(stack) + 1):
        below = stack[pos - 1] if pos > 0 else None
        above = stack[pos] if pos < len(stack) else None
        out.extend((p, pos) for p in allowed if p != below and p != above)
    return out


def order_violations(stack: Sequence[Paint]) -> int:
    """Pairs out of the preferred order: opaque below transparent; lighter below darker."""
    rank = {r: i for i, r in enumerate(TRANSPARENCY_RATINGS)}
    count = 0
    for i, lower in enumerate(stack):
        for upper in stack[i + 1 :]:
            if rank[lower.transparency] > rank[upper.transparency]:
                count += 1
            elif rank[lower.transparency] == rank[upper.transparency] and lower.lab[0] < upper.lab[0]:
                count += 1
    return count


def _background_per_cluster(background: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    """(3,) stays as is; (3, H, W) becomes the per-cluster mean, shape (3, k, 1)."""
    if background.ndim == 1:
        return background
    flat = background.reshape(3, -1).double()
    sums = torch.zeros(3, k, dtype=torch.float64).index_add_(1, labels, flat)
    counts = torch.bincount(labels, minlength=k).clamp(min=1)
    return (sums / counts).unsqueeze(-1)


def _rms(err: torch.Tensor) -> torch.Tensor:
    return err.pow(2).mean(dim=0).sqrt()


def _solve_stacks(stacks: list[list[int]], K_all, S_all, target, background, solve_config) -> torch.Tensor:
    """RMS quantized error over the clusters of each stack (all the same length)."""
    n, c = len(stacks[0]), len(stacks)
    if n == 0:
        out = background.expand(3, target.shape[1], 1) if background.ndim > 1 else background.reshape(3, 1, 1)
        err = color.delta_e(color.working_to_lab(out.to(target.dtype), dim=0), target[:, :, :1], dim=0)
        return _rms(err).expand(c)
    idx = torch.tensor(stacks).T  # (n, c)
    K = K_all[idx].permute(0, 2, 1).unsqueeze(2)  # (n, 3, 1, c)
    S = S_all[idx].permute(0, 2, 1).unsqueeze(2)
    sol = solve(target, background, K, S, solve_config)
    return _rms(sol.error)  # (c,)


def select_stack(
    target_lab: torch.Tensor,
    background: torch.Tensor,
    paints: Sequence[Paint],
    config: PaletteConfig = PaletteConfig(),
    solve_config: SolveConfig = SolveConfig(),
    progress: Callable[[str], None] | None = None,
) -> Selection:
    """Choose the layer stack (bottom first) for ``target_lab`` over ``background``.

    Args:
        target_lab: (3, H, W) target image in Lab.
        background: working-space reflectance, (3,) or (3, H, W).
        paints: the paints that may be used (each may appear in many layers).
        solve_config: solver settings; selection overrides steps and turns
            smoothing off (neighbors in the batched image are unrelated).
    """
    if not paints:
        raise ValueError("no paints to choose from")
    background = torch.as_tensor(background, dtype=torch.float64)
    centers, labels = cluster_colors(target_lab, config.n_colors, config.seed)
    k = centers.shape[0]
    bg = _background_per_cluster(background, labels, k).float()
    K_all, S_all = ks_tensors(paints)
    # Neighbors in the batched image are unrelated colors/candidates, and the thickness
    # subset is chosen per layer across the whole image: both apply to the final solve only.
    search = replace(solve_config, steps=config.steps, lam_smooth=0.0, refine_smooth=0.0, max_thicknesses=None)

    def batch_target(c: int) -> torch.Tensor:
        return centers.T.float().unsqueeze(-1).expand(3, k, c).contiguous()

    stack: list[int] = []
    forced = None
    image_median_l = target_lab[0].median().item()
    bg_l = color.working_to_lab(background.reshape(3, -1), dim=0)[0].median().item()
    opaque = [i for i, p in enumerate(paints) if p.transparency == "opaque"]
    if bg_l < image_median_l and opaque:
        lightest = max(opaque, key=lambda i: paints[i].lab[0])
        stack, forced = [lightest], paints[lightest]

    baseline = _solve_stacks([[]], K_all, S_all, batch_target(1), bg, search)[0].item()
    current = _solve_stacks([stack], K_all, S_all, batch_target(1), bg, search)[0].item() if stack else baseline
    selection = Selection(stack=[], forced_first=forced, baseline_error=baseline, cluster_lab=centers)
    if forced is not None:
        selection.rounds.append({"paint": forced.name, "position": 0, "rms_error": current, "forced": True})
    if progress:
        progress(f"background alone: RMS error {baseline:.2f}" + (f"; with {forced.name} first: {current:.2f}" if forced else ""))

    if config.max_paints is not None and config.max_paints < 1:
        raise ValueError("max_paints must be at least 1")
    allowed = list(range(len(paints)))
    while config.max_layers is None or len(stack) < config.max_layers:
        if config.max_paints is not None and len(set(stack)) >= config.max_paints:
            allowed = sorted(set(stack))  # palette is full: only more layers of paints already used
        cands = candidates(stack, allowed)
        if not cands:
            if progress:
                progress("stop: no more places to add a paint")
            break
        stacks = [stack[:pos] + [p] + stack[pos:] for p, pos in cands]
        errors = _solve_stacks(stacks, K_all, S_all, batch_target(len(stacks)), bg, search)
        best_err = errors.min().item()
        tied = [i for i in range(len(stacks)) if errors[i].item() <= best_err + TIE_DELTA_E]
        best = min(tied, key=lambda i: (order_violations([paints[j] for j in stacks[i]]), errors[i].item()))
        gain = current - errors[best].item()
        if gain < config.min_gain:
            if progress:
                progress(f"stop: best candidate saves {gain:.2f} < {config.min_gain}")
            break
        paint_idx, pos = cands[best]
        stack, current = stacks[best], errors[best].item()
        selection.rounds.append({"paint": paints[paint_idx].name, "position": pos, "rms_error": current, "forced": False})
        if progress:
            progress(f"layer {len(stack)}: {paints[paint_idx].name} at position {pos}, RMS error {current:.2f}")

    selection.stack = [paints[i] for i in stack]
    return selection
