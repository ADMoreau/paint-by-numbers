"""Milestone 5 (part): paint selection and layer ordering."""

import pytest
import torch

from core_paints import core_paints
from paint_by_numbers import color, km, palette, paints, solve

PAINTS = core_paints()
BY_NAME = {p.name: p for p in PAINTS}
FAST = palette.PaletteConfig(n_colors=16, steps=150)
FAST_SOLVE = solve.SolveConfig(steps=150, refine_sweeps=3)


def flat_image(colors_lab, h=8, w=8):
    """Vertical stripes of the given Lab colors, shape (3, h, w * len)."""
    cols = [torch.tensor(c, dtype=torch.float32).reshape(3, 1, 1).expand(3, h, w) for c in colors_lab]
    return torch.cat(cols, dim=2).contiguous()


def masstone_lab(paint):
    return color.working_to_lab(km.masstone(torch.tensor(paint.K), torch.tensor(paint.S))).tolist()


# ---------------------------------------------------------------- candidates and order


def test_candidates_allow_repeats_but_not_adjacent():
    # stack: paint 0 then paint 1; allowed paints 0, 1, 2
    got = palette.candidates([0, 1], [0, 1, 2])
    assert sorted(got) == sorted([(1, 0), (2, 0), (2, 1), (0, 2), (2, 2)])
    assert (0, 2) in got  # paint 0 again, on top: a repeat, not adjacent to itself


def test_candidates_for_empty_stack():
    assert palette.candidates([], [3, 5]) == [(3, 0), (5, 0)]


def test_order_violations():
    white, umber = BY_NAME["Titanium White"], BY_NAME["Burnt Umber"]
    glaze = BY_NAME["Green Gold"]
    assert palette.order_violations([white, glaze]) == 0  # opaque below transparent
    assert palette.order_violations([glaze, white]) == 1
    assert palette.order_violations([white, umber]) == 0  # lighter below darker
    assert palette.order_violations([umber, white]) == 1


def test_cluster_colors_finds_flat_colors():
    target = flat_image([[30.0, 10.0, 10.0], [60.0, -20.0, 30.0], [85.0, 0.0, -10.0]])
    centers, labels = palette.cluster_colors(target, n_colors=64)
    assert centers.shape == (3, 3)  # only 3 distinct colors
    assert labels.shape == (8 * 24,)
    for c in ([30.0, 10.0, 10.0], [60.0, -20.0, 30.0], [85.0, 0.0, -10.0]):
        assert (centers - torch.tensor(c, dtype=torch.float64)).norm(dim=1).min() < 1e-3


# ---------------------------------------------------------------- selection


def test_target_equal_to_background_selects_no_layers():
    bg = torch.tensor([0.19, 0.14, 0.10])
    target = flat_image([color.working_to_lab(bg).tolist()])
    sel = palette.select_stack(target, bg, PAINTS, FAST, FAST_SOLVE)
    assert sel.stack == []
    assert sel.baseline_error < 0.01


def test_single_paint_target_selects_that_paint():
    umber = BY_NAME["Burnt Umber"]
    sel = palette.select_stack(flat_image([masstone_lab(umber)]), torch.ones(3), PAINTS, FAST, FAST_SOLVE)
    assert sel.stack[0].name == "Burnt Umber"
    assert sel.rounds[0]["rms_error"] < 3.0


def test_dark_background_forces_lightest_opaque_first():
    target = flat_image([[80.0, 0.0, 20.0], [70.0, 10.0, -10.0]])
    sel = palette.select_stack(target, torch.zeros(3), PAINTS, FAST, FAST_SOLVE)
    assert sel.forced_first is not None and sel.forced_first.name == "Titanium White"
    assert sel.rounds[0]["forced"] is True


def test_light_background_forces_nothing():
    target = flat_image([[40.0, 20.0, 10.0]])
    sel = palette.select_stack(target, torch.ones(3), PAINTS, FAST, FAST_SOLVE)
    assert sel.forced_first is None


def test_max_layers_caps_the_stack():
    target = flat_image([masstone_lab(BY_NAME[n]) for n in ("Burnt Umber", "Pyrrole Red", "Cadmium Yellow Primrose")])
    sel = palette.select_stack(target, torch.ones(3), PAINTS, palette.PaletteConfig(n_colors=16, steps=150, max_layers=1), FAST_SOLVE)
    assert len(sel.stack) == 1


def test_only_allowed_paints_are_used():
    owned = [BY_NAME["Titanium White"], BY_NAME["Ultramarine Blue"], BY_NAME["Burnt Umber"]]
    target = flat_image([[50.0, 30.0, 40.0], [30.0, 5.0, -20.0], [70.0, 0.0, 0.0]])
    sel = palette.select_stack(target, torch.ones(3), owned, FAST, FAST_SOLVE)
    assert sel.stack and {p.name for p in sel.stack} <= {p.name for p in owned}


def test_no_paint_twice_in_a_row():
    target = flat_image([masstone_lab(BY_NAME[n]) for n in ("Burnt Umber", "Pyrrole Red", "Titanium White", "Ultramarine Blue")])
    sel = palette.select_stack(target, torch.full((3,), 0.2), PAINTS, FAST, FAST_SOLVE)
    names = [p.name for p in sel.stack]
    assert all(a != b for a, b in zip(names, names[1:]))


def test_per_pixel_background_uses_cluster_means():
    bg = torch.ones(3, 8, 16)
    bg[:, :, 8:] = 0.0  # right half black
    labels = torch.tensor([0] * 8 + [1] * 8).repeat(8)  # left cluster 0, right cluster 1
    per_cluster = palette._background_per_cluster(bg, labels, 2)
    assert per_cluster.shape == (3, 2, 1)
    assert torch.allclose(per_cluster[:, 0, 0], torch.ones(3, dtype=torch.float64))
    assert torch.allclose(per_cluster[:, 1, 0], torch.zeros(3, dtype=torch.float64))


def test_rounds_record_falling_error():
    target = flat_image([masstone_lab(BY_NAME[n]) for n in ("Burnt Umber", "Pyrrole Red", "Ultramarine Blue")])
    sel = palette.select_stack(target, torch.ones(3), PAINTS, FAST, FAST_SOLVE)
    errors = [sel.baseline_error] + [r["rms_error"] for r in sel.rounds]
    assert all(b < a for a, b in zip(errors, errors[1:]))
    assert len(sel.rounds) == len(sel.stack)


def test_no_paints_raises():
    with pytest.raises(ValueError):
        palette.select_stack(flat_image([[50.0, 0.0, 0.0]]), torch.ones(3), [], FAST, FAST_SOLVE)


def test_max_paints_limits_different_paints():
    target = flat_image([masstone_lab(BY_NAME[n]) for n in ("Burnt Umber", "Pyrrole Red", "Cadmium Yellow Primrose", "Ultramarine Blue")])
    config = palette.PaletteConfig(n_colors=16, steps=150, max_paints=2)
    sel = palette.select_stack(target, torch.ones(3), PAINTS, config, FAST_SOLVE)
    assert 1 <= len({p.name for p in sel.stack}) <= 2


def test_max_paints_must_be_positive():
    with pytest.raises(ValueError, match="max_paints"):
        palette.select_stack(flat_image([[50.0, 0.0, 0.0]]), torch.ones(3), PAINTS,
                             palette.PaletteConfig(n_colors=4, steps=10, max_paints=0), FAST_SOLVE)  # fmt: skip
