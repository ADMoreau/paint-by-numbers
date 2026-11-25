"""One weight per layer: a painting planned as N layers, each one paint at one weight over one region.

The default planner gives each layer a thickness map with several levels. At the easel that means
painting patches of different dilutions side by side, and closing the gaps between them without
overlapping is hard. Here every layer is a single mix (one paint at one weight, such as 1/4 coat)
filled over one region, and gradations come from stacking layers instead: a pale wash over a
large area, then a stronger coat of the same or another paint over part of it. Where two layers
meet they overlap, so nothing has to butt up against anything.

    config = UniformConfig(layers=10)
    result = plan_uniform(image, paints, background, config)
    write_outputs(result, "out/")

Choosing the layers (on the image's k-means colors, as palette.select_stack does):

1. Each round, try every paint at every weight at every position in the stack. For each
   candidate, decide which colors get each layer (on or off) by coordinate descent: per layer,
   per color, keep whichever of on or off gives the lower ΔE plus ``paint_cost`` per coat, and
   sweep until nothing changes. Keep the candidate with the lowest RMS ΔE over the colors.
   The same paint may sit on top of itself: two layers of one paint with different regions
   build up a gradation.
2. Stop at ``layers`` layers, or earlier if no layer saves ``min_gain``.
3. Improve: for each layer in turn, try every other paint and weight in its place; keep a
   change if it lowers the error. Repeat ``improve_passes`` times.

Then the regions, at full resolution: every pixel starts with its color's on/off pattern, and the
same coordinate descent runs per pixel, except that each layer's cost difference (on minus off)
is smoothed before choosing, so regions come out as connected shapes rather than speckle. The
smoothing is a guided filter steered by the image: it averages over flat areas but stops at the
image's own edges, so outlines and thin lines keep their shape. Patches and holes smaller than
``min_region`` are then removed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import ndimage

from . import color, km
from .paints import Paint, ks_tensors
from .palette import TIE_DELTA_E, Selection, _background_per_cluster, cluster_colors, order_violations
from .solve import Solution

__all__ = ["UniformConfig", "UniformSelection", "select_uniform_stack", "solve_uniform", "clean_regions"]


@dataclass(frozen=True)
class UniformConfig:
    layers: int = 8  # how many layers (fewer only if extra layers would not help)
    weights: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)  # coats a layer may be painted at
    max_paints: int | None = None  # most different paints; None: no limit
    n_colors: int = 64  # k-means colors the layers are chosen on
    seed: int = 0
    paint_cost: float = 0.1  # ΔE charged per coat, so paint that is hidden or barely helps is left off
    min_gain: float = 0.05  # RMS ΔE a layer must save to be added
    improve_passes: int = 2
    sweeps: int = 10  # most coordinate-descent sweeps
    smooth: float = 0.008  # region smoothing radius, as a fraction of the long side; 0 = off
    edge: float = 5.0  # ΔE: color changes in the image bigger than this are edges regions may follow
    min_region: float = 0.0003  # smallest patch or hole kept, as a fraction of the image area


@dataclass
class UniformSelection(Selection):
    weights: list[float] = field(default_factory=list)  # coats per layer, bottom first
    assignment: torch.Tensor | None = None  # (n, 1, k) bool: which colors each layer covers
    labels: torch.Tensor | None = None  # (H*W,) each pixel's color


_TO_XYZ_OVER_WHITE = (color._WORKING_TO_XYZ / color._WHITE_XYZ.reshape(3, 1)).tolist()


def _lab_f(t):
    return torch.where(t > color._LAB_DELTA_CUBED, t.clamp(min=color._LAB_DELTA_CUBED).pow(1.0 / 3.0), t * color._LAB_SLOPE + color._LAB_OFFSET)


def _delta_e(rgb, target_lab):
    """ΔE76 of working-space reflectance ``rgb`` (3, ...) from ``target_lab`` (3, ...): the same as
    color.delta_e(color.working_to_lab(rgb, dim=0), target_lab, dim=0), written out per channel,
    which is several times faster on large batches."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = _TO_XYZ_OVER_WHITE
    fx = _lab_f(m00 * r + m01 * g + m02 * b)
    fy = _lab_f(m10 * r + m11 * g + m12 * b)
    fz = _lab_f(m20 * r + m21 * g + m22 * b)
    dl = 116.0 * fy - 16.0 - target_lab[0]
    da = 500.0 * (fx - fy) - target_lab[1]
    db = 200.0 * (fy - fz) - target_lab[2]
    return (dl * dl + da * da + db * db).sqrt()


def _cost(target, background, K, S, W, X, paint_cost):
    """Per-pixel ΔE of the stack with layer i painted where X[i], plus paint_cost per coat."""
    t = W * X.to(W.dtype)
    out = km.render_stack(background, K, S, t)
    err = color.delta_e(color.working_to_lab(out, dim=0), target, dim=0)
    return err, err + paint_cost * t.sum(dim=(0, 1))


def guided_filter(guide: np.ndarray, radius: int, eps: float) -> Callable[[np.ndarray], np.ndarray]:
    """Edge-aware smoothing that follows the edges of ``guide`` (H, W, C) (He et al. 2010).

    Returns a function smoothing an (H, W) array. Within each (2r+1)-square window the output is a
    linear function of the guide's channels, so it is smooth where the guide is flat and changes
    where the guide does: thin lines in the guide survive. ``eps`` (guide units squared) sets how
    big a change in the guide counts as an edge. The guide's statistics are computed once.
    """
    size = 2 * radius + 1
    box = lambda x: ndimage.uniform_filter(x, size=(size, size) + (1,) * (x.ndim - 2), mode="nearest")  # noqa: E731
    h, w, c = guide.shape
    mean_i = box(guide)
    var_i = box(np.einsum("hwi,hwj->hwij", guide, guide).reshape(h, w, c * c)).reshape(h, w, c, c)
    var_i -= np.einsum("hwi,hwj->hwij", mean_i, mean_i)
    inv = np.linalg.inv(var_i + eps * np.eye(c))

    def apply(p: np.ndarray) -> np.ndarray:
        mean_p = box(p)
        cov_ip = box(guide * p[..., None]) - mean_i * mean_p[..., None]
        a = np.einsum("hwij,hwj->hwi", inv, cov_ip)
        b = mean_p - (a * mean_i).sum(-1)
        return (box(a) * guide).sum(-1) + box(b)

    return apply


def _layer_maps(K, S, W):
    """Each layer, painted at its weight, as a map of the reflectance R beneath it.

    A Kubelka-Munk layer over R gives (m00 R + m01) / (m10 R + 1): a linear fractional (Möbius)
    map of R. Such maps compose like 2x2 matrices, so any run of layers is again one map of this
    form. Returns m00, m01, m10, each broadcast like K * W, shape (n, 3, ...).
    """
    a, b, r_inf, bs = km._coefficients(K, S)
    th = torch.tanh(bs * W)
    d = a * th + b
    return (b - th * (r_inf + b)) / d, th / d, -th / d


def _descend(target, background, K, S, W, X, sweeps, paint_cost, smooth=None):
    """Coordinate descent on which pixels each layer covers. X: (n, 1, *spatial) bool, changed in place.

    Per layer (others fixed), each pixel takes whichever of on/off costs less (ΔE plus
    ``paint_cost`` per coat). With ``smooth`` (a function of the (H, W) cost difference, on minus
    off) the difference is smoothed first, so a pixel follows its neighborhood. Returns X and the
    per-pixel ΔE.

    Each sweep goes top layer first. The reflectance beneath every layer is rendered once per
    sweep, and the layers above the current one are carried as a single composed map (see
    _layer_maps), so trying a layer on and off costs one layer and two map applications instead
    of two renders of the whole stack.
    """
    n, spatial = X.shape[0], X.shape[2:]
    ones = (1,) * len(spatial)
    if K.ndim == 2:
        K, S = K.reshape(n, 3, *ones), S.reshape(n, 3, *ones)
    m00, m01, m10 = _layer_maps(K, S, W)
    bg = background.reshape(3, *ones) if background.ndim == 1 else background
    bg = bg.to(m00.dtype).expand(3, *spatial)

    def paint(i, r):
        return (m00[i] * r + m01[i]) / (m10[i] * r + 1)

    for _ in range(sweeps):
        below, r = [], bg
        for i in range(n):
            below.append(r)
            r = torch.where(X[i], paint(i, r), r)
        a00 = torch.ones_like(bg)  # the layers above, composed: R -> (a00 R + a01) / (a10 R + 1)
        a01 = torch.zeros_like(bg)
        a10 = torch.zeros_like(bg)
        coats = (W * X.to(W.dtype)).sum(dim=0)  # (1, *spatial)
        changed = False
        for i in reversed(range(n)):
            rb = below[i]
            others = coats - W[i] * X[i].to(W.dtype)
            costs = []
            for r_i, extra in ((rb, 0.0), (paint(i, rb), W[i])):
                out = (a00 * r_i + a01) / (a10 * r_i + 1)
                costs.append(_delta_e(out, target) + paint_cost * (others + extra)[0])
            d = costs[1] - costs[0]
            if smooth is not None:
                d = torch.from_numpy(smooth(d.double().numpy())).to(d.dtype)
            new = (d < 0).unsqueeze(0)
            changed |= bool((new != X[i]).any())
            X[i] = new
            coats = others + W[i] * new.to(W.dtype)
            # carry layer i into the composed map where it is painted: A <- A . M_i
            p11 = a10 * m01[i] + 1
            n00 = (a00 * m00[i] + a01 * m10[i]) / p11
            n01 = (a00 * m01[i] + a01) / p11
            n10 = (a10 * m00[i] + m10[i]) / p11
            a00, a01, a10 = (torch.where(new, nx, ax) for nx, ax in ((n00, a00), (n01, a01), (n10, a10)))
        if not changed:
            break
    err, _ = _cost(target, background, K, S, W, X, paint_cost)
    return X, err


def _rms(err: torch.Tensor) -> torch.Tensor:
    return err.pow(2).mean(dim=0).sqrt()


def _evaluate(stacks, starts, K_all, S_all, target_k, bg, config):
    """Best assignment and RMS ΔE over the colors for each candidate stack (all the same length).

    stacks: list of c lists of (paint index, weight). starts: list of (n, 1, k, c) bool starting
    assignments; each color keeps the best result over the starts.
    """
    c, n = len(stacks), len(stacks[0])
    idx = torch.tensor([[p for p, _ in s] for s in stacks]).T  # (n, c)
    K = K_all[idx].permute(0, 2, 1).unsqueeze(2)  # (n, 3, 1, c)
    S = S_all[idx].permute(0, 2, 1).unsqueeze(2)
    W = torch.tensor([[w for _, w in s] for s in stacks], dtype=target_k.dtype).T.reshape(n, 1, 1, c)
    target = target_k.expand(3, target_k.shape[1], c)
    best_X, best_err, best_cost = None, None, None
    for X0 in starts:
        X, err = _descend(target, bg, K, S, W, X0.clone(), config.sweeps, config.paint_cost)
        cost = err + config.paint_cost * (W * X.to(W.dtype)).sum(dim=(0, 1))
        if best_X is None:
            best_X, best_err, best_cost = X, err, cost
        else:
            better = cost < best_cost
            best_X = torch.where(better, X, best_X)
            best_err = torch.where(better, err, best_err)
            best_cost = torch.where(better, cost, best_cost)
    return best_X, _rms(best_err)


def _pick(errors, stacks, paints):
    """Lowest error; near-ties go to the stack with fewer ordering violations (opaque below transparent)."""
    best_err = errors.min().item()
    tied = [i for i in range(len(stacks)) if errors[i].item() <= best_err + TIE_DELTA_E]
    return min(tied, key=lambda i: (order_violations([paints[p] for p, _ in stacks[i]]), errors[i].item()))


def select_uniform_stack(
    target_lab: torch.Tensor,
    background: torch.Tensor,
    paints: Sequence[Paint],
    config: UniformConfig = UniformConfig(),
    progress: Callable[[str], None] | None = None,
) -> UniformSelection:
    """Choose up to ``config.layers`` (paint, weight) layers, bottom first, and which colors each covers."""
    if not paints:
        raise ValueError("no paints to choose from")
    if config.layers < 1:
        raise ValueError("layers must be at least 1")
    weights = sorted({float(w) for w in config.weights})
    if not weights or weights[0] <= 0:
        raise ValueError("weights must be positive")
    if config.max_paints is not None and config.max_paints < 1:
        raise ValueError("max_paints must be at least 1")
    say = progress or (lambda msg: None)
    background = torch.as_tensor(background, dtype=torch.float64)
    centers, labels = cluster_colors(target_lab, config.n_colors, config.seed)
    k = centers.shape[0]
    target_k = centers.T.float().unsqueeze(-1)  # (3, k, 1)
    bg = _background_per_cluster(background, labels, k).float()
    K_all, S_all = (x.float() for x in ks_tensors(paints))

    bare = km.render_stack(bg, torch.empty(0, 3), torch.empty(0, 3), torch.zeros(0, 1, k, 1))
    baseline = _rms(color.delta_e(color.working_to_lab(bare, dim=0), target_k, dim=0))[0].item()
    say(f"background alone: RMS error {baseline:.2f}")
    sel = UniformSelection(stack=[], forced_first=None, baseline_error=baseline, cluster_lab=centers)

    def allowed(stack):
        used = {p for p, _ in stack}
        ps = sorted(used) if config.max_paints is not None and len(used) >= config.max_paints else range(len(paints))
        return [(p, w) for p in ps for w in weights]

    stack: list[tuple[int, float]] = []
    X = torch.zeros(0, 1, k, dtype=torch.bool)
    current = baseline
    while len(stack) < config.layers:
        cands = [(pw, pos) for pos in range(len(stack) + 1) for pw in allowed(stack)]
        stacks = [stack[:pos] + [pw] + stack[pos:] for pw, pos in cands]
        c = len(stacks)
        starts = []
        for new_on in (False, True):
            X0 = torch.zeros(len(stack) + 1, 1, k, c, dtype=torch.bool)
            for j, (_, pos) in enumerate(cands):
                X0[:pos, :, :, j] = X[:pos]
                X0[pos, :, :, j] = new_on
                X0[pos + 1 :, :, :, j] = X[pos:]
            starts.append(X0)
        Xs, errors = _evaluate(stacks, starts, K_all, S_all, target_k, bg, config)
        best = _pick(errors, stacks, paints)
        gain = current - errors[best].item()
        if gain < config.min_gain:
            say(f"stop at {len(stack)} layers: another layer would save only {gain:.2f}")
            break
        (p, w), pos = cands[best]
        stack, X, current = stacks[best], Xs[..., best], errors[best].item()
        sel.rounds.append({"paint": paints[p].name, "coats": w, "position": pos, "rms_error": current})
        say(f"layer {len(stack)}: {paints[p].name} at {w:g} coats, position {pos}, RMS error {current:.2f}")

    for pass_no in range(config.improve_passes):
        improved = False
        for i in range(len(stack)):
            options = [pw for pw in allowed(stack[:i] + stack[i + 1 :]) if pw != stack[i]]
            if not options:
                continue
            stacks = [stack[:i] + [pw] + stack[i + 1 :] for pw in options]
            starts = []
            for on in (False, True):
                X0 = X.unsqueeze(-1).expand(*X.shape, len(stacks)).clone()
                X0[i] = on
                starts.append(X0)
            Xs, errors = _evaluate(stacks, starts, K_all, S_all, target_k, bg, config)
            best = int(errors.argmin())
            if errors[best].item() < current - 1e-3:
                old = stack[i]
                stack, X, current = stacks[best], Xs[..., best], errors[best].item()
                improved = True
                say(f"improve: layer {i + 1} {paints[old[0]].name} {old[1]:g} -> {paints[stack[i][0]].name} "
                    f"{stack[i][1]:g} coats, RMS error {current:.2f}")  # fmt: skip
        if not improved:
            break

    sel.stack = [paints[p] for p, _ in stack]
    sel.weights = [w for _, w in stack]
    sel.assignment = X
    sel.labels = labels
    return sel


def clean_regions(X: torch.Tensor, min_area: int) -> torch.Tensor:
    """Remove patches and fill holes smaller than ``min_area`` pixels in each layer's region.

    X: (n, 1, H, W) bool. Patches are 8-connected, holes 4-connected, so thin diagonal strokes
    survive and do not count as gaps.
    """
    if min_area <= 1:
        return X
    out = X.clone()
    eight = np.ones((3, 3), bool)
    for i in range(X.shape[0]):
        m = out[i, 0].numpy().copy()
        for value, structure in ((True, eight), (False, None)):
            lab, n = ndimage.label(m == value, structure=structure)
            if n == 0:
                continue
            sizes = np.bincount(lab.ravel())
            small = sizes < min_area
            small[0] = False
            m[small[lab]] = not value
        out[i, 0] = torch.from_numpy(m)
    return out


def solve_uniform(
    target_lab: torch.Tensor,
    background: torch.Tensor,
    selection: UniformSelection,
    config: UniformConfig = UniformConfig(),
    progress: Callable[[str], None] | None = None,
) -> Solution:
    """Each layer's region at full resolution, starting from its colors' assignment.

    Layers left with no region are removed from ``selection``.
    """
    say = progress or (lambda msg: None)
    h, w = target_lab.shape[1:]
    n = len(selection.stack)
    K, S = (x.float() for x in ks_tensors(selection.stack))
    W = torch.tensor(selection.weights, dtype=torch.float32).reshape(n, 1, 1, 1)
    target = target_lab.float()
    bg = background.float()
    X = selection.assignment[:, :, selection.labels].reshape(n, 1, h, w).clone()
    _, err0 = _cost(target, bg, K, S, W, X, config.paint_cost)
    radius = round(config.smooth * max(h, w))
    smooth = guided_filter(target.permute(1, 2, 0).double().numpy(), radius, config.edge**2) if radius > 0 else None
    min_area = round(config.min_region * h * w)
    for _ in range(2):
        X, err = _descend(target, bg, K, S, W, X, config.sweeps, config.paint_cost, smooth)
        X = clean_regions(X, min_area)
    keep = [i for i in range(n) if X[i].any()]
    if len(keep) < n:
        say(f"dropping {n - len(keep)} layer(s) left with no region after smoothing")
        selection.stack = [selection.stack[i] for i in keep]
        selection.weights = [selection.weights[i] for i in keep]
        selection.assignment = selection.assignment[keep]
        X, K, S, W = X[keep], K[keep], S[keep], W[keep]
    t = W * X.float()
    preview = km.render_stack(bg, K, S, t)
    err = color.delta_e(color.working_to_lab(preview, dim=0), target, dim=0)
    say(f"regions: mean ΔE {err0.mean():.2f} from the colors' assignment, {err.mean():.2f} after smoothing")
    return Solution(
        t=t, t_snapped=t, preview=preview, error=err, mean_error=err.mean().item(), max_error=err.max().item(),
        mean_error_continuous=err.mean().item(), max_error_continuous=err.max().item(), mean_error_nearest=err0.mean().item(),
    )  # fmt: skip
