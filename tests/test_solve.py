"""Milestone 4: thickness-map solver."""

import pytest
import torch

from core_paints import core_paints
from paint_by_numbers import color, km, paints, solve

PAINTS = core_paints()
BY_NAME = {p.name: p for p in PAINTS}
K_ALL, S_ALL = paints.ks_tensors(PAINTS)

WHITE = [1.0, 1.0, 1.0]
BLACK = [0.0, 0.0, 0.0]
GRAY = [0.2158605] * 3  # sRGB #808080
UMBER = [0.19, 0.14, 0.10]


def flat(lab, h=4, w=4):
    return torch.as_tensor(lab, dtype=torch.float32).reshape(3, 1, 1).expand(3, h, w).contiguous()


def masstone_lab(paint):
    return color.working_to_lab(km.masstone(torch.tensor(paint.K), torch.tensor(paint.S)))


# ---------------------------------------------------------------- acceptance checks


@pytest.mark.parametrize("background", [WHITE, GRAY, BLACK, UMBER], ids=["white", "gray", "black", "umber"])
def test_target_equal_to_background_uses_no_paint(background):
    """Spec check: every layer below 0.05 coats. Uses all 9 paints as the stack."""
    bg = torch.tensor(background)
    sol = solve.solve(flat(color.working_to_lab(bg)), bg, K_ALL, S_ALL)
    assert sol.t.max().item() < 0.05
    assert sol.t_snapped.max().item() == 0.0


OPAQUE = [p for p in PAINTS if p.transparency == "opaque"]


@pytest.mark.parametrize("paint", OPAQUE, ids=lambda p: p.slug)
@pytest.mark.parametrize("background", [WHITE, BLACK], ids=["white", "black"])
def test_flat_masstone_target_solves_under_3(paint, background):
    """Spec check: a flat target equal to one palette paint's masstone, mean
    error under 3 Lab units, before and after quantizing. Stack is all 9 paints."""
    sol = solve.solve(flat(masstone_lab(paint)), torch.tensor(background), K_ALL, S_ALL)
    assert sol.mean_error_continuous < 3.0
    assert sol.mean_error < 3.0


# ---------------------------------------------------------------- behavior


def test_two_region_target_puts_each_paint_where_it_belongs():
    stack = [BY_NAME["Titanium White"], BY_NAME["Burnt Umber"]]
    K, S = paints.ks_tensors(stack)
    target = torch.cat([flat(masstone_lab(stack[0]), 6, 4), flat(masstone_lab(stack[1]), 6, 4)], dim=2)
    sol = solve.solve(target, torch.tensor(GRAY), K, S)
    white_t, umber_t = sol.t_snapped[0, 0], sol.t_snapped[1, 0]
    assert sol.mean_error < 3.0
    assert white_t[:, :4].min() >= 1.0 and umber_t[:, :4].max() == 0.0  # left: white only
    assert umber_t[:, 4:].min() >= 1.0  # right: umber on top hides the white


def test_reachable_image_is_recovered():
    """A target rendered from known snapped thickness maps solves to near zero."""
    gen = torch.Generator().manual_seed(0)
    stack = [BY_NAME[n] for n in ("Titanium White", "Ultramarine Blue", "Cadmium Yellow Primrose", "Quinacridone Magenta")]
    K, S = paints.ks_tensors(stack)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 12), torch.linspace(0, 1, 12), indexing="ij")
    t_true = torch.stack([1 + torch.sin(3 * xx + p) * torch.cos(2 * yy + q) for p, q in torch.rand(4, 2, generator=gen) * 6])
    t_true = solve.snap(t_true.unsqueeze(1).clamp(0, 2), solve.SolveConfig().levels)
    bg = torch.tensor(UMBER)
    target = color.working_to_lab(km.render_stack(bg, K, S, t_true), dim=0)
    sol = solve.solve(target, bg, K, S)
    assert sol.mean_error_continuous < 0.5
    assert sol.mean_error < 1.0


def test_solution_shapes_and_ranges():
    K, S = paints.ks_tensors(PAINTS[:3])
    target = torch.rand(3, 5, 7) * torch.tensor([60.0, 20.0, 20.0]).reshape(3, 1, 1) + torch.tensor([20.0, -10.0, -10.0]).reshape(3, 1, 1)
    cfg = solve.SolveConfig(steps=50)
    sol = solve.solve(target, torch.tensor(WHITE), K, S, cfg)
    assert sol.t.shape == sol.t_snapped.shape == (3, 1, 5, 7)
    assert sol.preview.shape == (3, 5, 7)
    assert sol.error.shape == (5, 7)
    assert sol.t.min() >= 0 and sol.t.max() <= cfg.t_max
    assert set(sol.t_snapped.unique().tolist()) <= set(cfg.levels)
    assert sol.mean_error == pytest.approx(sol.error.mean().item())
    assert sol.max_error == pytest.approx(sol.error.max().item())
    assert len(sol.loss_history) > 0 and sol.loss_history[-1] < sol.loss_history[0]


def test_preview_and_error_match_snapped_stack():
    K, S = paints.ks_tensors(PAINTS[:2])
    target = flat([60.0, 10.0, 10.0])
    sol = solve.solve(target, torch.tensor(WHITE), K, S, solve.SolveConfig(steps=100))
    rendered = km.render_stack(torch.tensor(WHITE), K, S, sol.t_snapped)
    assert torch.allclose(sol.preview, rendered)
    assert torch.allclose(sol.error, color.delta_e(color.working_to_lab(rendered, dim=0), target, dim=0))


def test_per_pixel_background():
    stack = [BY_NAME["Titanium White"]]
    K, S = paints.ks_tensors(stack)
    bg = torch.stack([torch.full((4, 4), v) for v in UMBER])
    bg[:, :, 2:] = 1.0  # right half already white
    sol = solve.solve(flat(masstone_lab(stack[0])), bg, K, S)
    assert sol.t_snapped[0, 0, :, :2].min() >= 1.0  # white needed over umber
    assert sol.mean_error < 3.0


def test_no_layers():
    bg = torch.tensor(GRAY)
    target = flat([50.0, 0.0, 0.0])
    sol = solve.solve(target, bg, torch.empty(0, 3), torch.empty(0, 3))
    assert sol.t.shape == (0, 1, 4, 4)
    expected = color.delta_e(color.working_to_lab(bg), [50.0, 0.0, 0.0]).item()
    assert sol.mean_error == pytest.approx(expected, abs=1e-4)


def test_target_must_be_three_channel_image():
    with pytest.raises(ValueError):
        solve.solve(torch.zeros(4, 4), torch.tensor(WHITE), K_ALL, S_ALL)


# ---------------------------------------------------------------- quantization


def test_refinement_beats_nearest_snapping_on_trace_solutions():
    """Burnt Umber's masstone over black: the continuous optimum can lean on
    traces of high-S paints that nearest snapping zeroes."""
    sol = solve.solve(flat(masstone_lab(BY_NAME["Burnt Umber"])), torch.tensor(BLACK), K_ALL, S_ALL)
    assert sol.mean_error <= sol.mean_error_nearest
    assert sol.mean_error < 3.0


@pytest.mark.parametrize("paint_cost", [0.0, 0.1])
def test_refine_levels_never_worse_than_nearest_and_stays_on_levels(paint_cost):
    gen = torch.Generator().manual_seed(1)
    levels = solve.SolveConfig().levels
    target = torch.stack([30 + 40 * torch.rand(6, 6, generator=gen), 20 * torch.rand(6, 6, generator=gen) - 10, 20 * torch.rand(6, 6, generator=gen) - 10])
    bg = torch.tensor(WHITE)
    t = 2 * torch.rand(len(PAINTS), 1, 6, 6, generator=gen)
    refined = solve.refine_levels(target, bg, K_ALL, S_ALL, t, levels, paint_cost=paint_cost)

    def objective(tt):
        err = color.delta_e(color.working_to_lab(km.render_stack(bg, K_ALL, S_ALL, tt), dim=0), target, dim=0)
        return err + paint_cost * tt.sum(dim=(0, 1))

    assert (objective(refined) <= objective(solve.snap(t, levels)) + 1e-4).all()
    assert set(refined.unique().tolist()) <= set(levels)


def test_refinement_drops_paint_hidden_under_an_opaque_layer():
    """Red under two coats of black is invisible; refinement must not keep it."""
    stack = [BY_NAME["Pyrrole Red"], BY_NAME["Carbon Black"]]
    K, S = paints.ks_tensors(stack)
    target = flat(masstone_lab(BY_NAME["Carbon Black"]))
    t = torch.full((2, 1, 4, 4), 2.0)  # start with red at full thickness too
    refined = solve.refine_levels(target, torch.tensor(WHITE), K, S, t, solve.SolveConfig().levels)
    assert refined[0].max().item() == 0.0
    assert refined[1].min().item() >= 1.0


def test_zero_refine_sweeps_is_nearest_snapping():
    K, S = paints.ks_tensors(PAINTS[:3])
    cfg = solve.SolveConfig(steps=100, refine_sweeps=0)
    sol = solve.solve(flat([60.0, 10.0, 10.0]), torch.tensor(WHITE), K, S, cfg)
    assert torch.equal(sol.t_snapped, solve.snap(sol.t, cfg.levels))
    assert sol.mean_error == pytest.approx(sol.mean_error_nearest)


def test_custom_levels():
    K, S = paints.ks_tensors(PAINTS[:3])
    cfg = solve.SolveConfig(steps=100, levels=(0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0))
    sol = solve.solve(flat([60.0, 10.0, 10.0]), torch.tensor(WHITE), K, S, cfg)
    assert set(sol.t_snapped.unique().tolist()) <= set(cfg.levels)


# ---------------------------------------------------------------- helpers


def test_snap_to_nearest_level():
    t = torch.tensor([0.0, 0.1, 0.13, 0.3, 0.74, 0.76, 1.4, 1.6, 2.0])
    assert solve.snap(t, (0.0, 0.25, 0.5, 1.0, 2.0)).tolist() == [0.0, 0.0, 0.25, 0.25, 0.5, 1.0, 1.0, 2.0, 2.0]


def test_total_variation():
    t = torch.zeros(2, 1, 3, 3)
    assert solve.total_variation(t).item() == 0.0
    t[0, 0, :, 1:] = 1.0  # one vertical edge in layer 0: 3 of 6 horizontal diffs are 1
    assert solve.total_variation(t).item() == pytest.approx(0.5)
    t[1] = t[0]  # same edge in layer 1: totals over layers, so it doubles
    assert solve.total_variation(t).item() == pytest.approx(1.0)
    assert solve.total_variation(torch.ones(3, 1, 1, 1)).item() == 0.0  # no neighbors


def test_total_paint_sums_layers():
    t = torch.full((3, 1, 2, 2), 0.5)
    assert solve.total_paint(t).item() == pytest.approx(1.5)
    assert solve.total_paint(torch.empty(0, 1, 2, 2)).item() == 0.0


def test_ks_tensors_keep_order_and_repeats():
    white, blue = BY_NAME["Titanium White"], BY_NAME["Ultramarine Blue"]
    K, S = paints.ks_tensors([white, blue, white])
    assert K.shape == S.shape == (3, 3)
    assert torch.equal(K[0], K[2]) and not torch.equal(K[0], K[1])
    assert paints.ks_tensors([])[0].shape == (0, 3)


# ---------------------------------------------------------------- thicknesses per layer and smoothing


def test_choose_layer_levels_fits_each_layer():
    levels = solve.SolveConfig().levels
    t = torch.zeros(2, 1, 4, 4)
    t[0, 0, :2] = 0.5
    t[0, 0, 2:] = 1.9  # layer 0: about 0.5 and 2
    t[1, 0, :, :2] = 0.13  # layer 1: about 1/8 only
    chosen = solve.choose_layer_levels(t, levels, 2)
    assert chosen[0].tolist() == [0.0, 0.5, 2.0]
    assert 0.125 in chosen[1].tolist() and len(chosen[1]) == 3
    assert all(len(c) == len(levels) for c in solve.choose_layer_levels(t, levels, None))
    with pytest.raises(ValueError):
        solve.choose_layer_levels(t, levels, 0)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_max_thicknesses_limits_each_layer(k):
    gen = torch.Generator().manual_seed(4)
    target = torch.stack([30 + 40 * torch.rand(8, 8, generator=gen), 20 * torch.rand(8, 8, generator=gen) - 10, 20 * torch.rand(8, 8, generator=gen) - 10])
    t = 2 * torch.rand(len(PAINTS), 1, 8, 8, generator=gen)
    refined = solve.refine_levels(target, torch.tensor(WHITE), K_ALL, S_ALL, t, solve.SolveConfig().levels, max_thicknesses=k)
    for i in range(len(PAINTS)):
        nonzero = refined[i, 0].unique()
        assert len(nonzero[nonzero > 0]) <= k


def test_neighbor_mismatch_counts():
    layer = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    counts = solve._neighbor_mismatch(layer, torch.tensor([0.0, 1.0]))
    # level 0: top-left has 2 neighbors (both 1) -> 2; top-right neighbors: 0 and 1 -> 1 differ
    assert counts[0].tolist() == [[2.0, 1.0], [1.0, 2.0]]
    assert counts[1].tolist() == [[0.0, 1.0], [1.0, 0.0]]


def _speckle(t):
    """Neighbor pairs on different levels, summed over layers."""
    return sum(int((torch.diff(t[i, 0], dim=d) != 0).sum()) for i in range(t.shape[0]) for d in (0, 1))


def test_smoothing_turns_speckle_into_regions():
    """A noisy target makes per-pixel levels speckle; smoothing merges them."""
    stack = [BY_NAME["Titanium White"], BY_NAME["Burnt Umber"]]
    K, S = paints.ks_tensors(stack)
    gen = torch.Generator().manual_seed(5)
    base = masstone_lab(BY_NAME["Burnt Umber"]).float().reshape(3, 1, 1).expand(3, 16, 16)
    target = base + torch.randn(3, 16, 16, generator=gen) * torch.tensor([6.0, 3.0, 3.0]).reshape(3, 1, 1)
    sol = solve.solve(target, torch.tensor(GRAY), K, S, solve.SolveConfig(steps=300))
    levels = solve.SolveConfig().levels
    rough = solve.refine_levels(target, torch.tensor(GRAY), K, S, sol.t, levels, smooth=0.0)
    smooth = solve.refine_levels(target, torch.tensor(GRAY), K, S, sol.t, levels, smooth=2.0)
    assert _speckle(smooth) < _speckle(rough) / 2
    err = lambda tt: color.delta_e(color.working_to_lab(km.render_stack(torch.tensor(GRAY), K, S, tt), dim=0), target, dim=0).mean()
    assert err(smooth) < err(rough) + 3.0  # costs a little accuracy, not a lot


def test_solve_config_passes_thickness_limit_and_smoothing():
    K, S = paints.ks_tensors(PAINTS[:3])
    cfg = solve.SolveConfig(steps=150, max_thicknesses=1, refine_smooth=1.0)
    sol = solve.solve(flat([60.0, 10.0, 10.0], 6, 6), torch.tensor(WHITE), K, S, cfg)
    for i in range(3):
        nonzero = sol.t_snapped[i, 0].unique()
        assert len(nonzero[nonzero > 0]) <= 1
