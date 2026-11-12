"""Milestone 1: color conversions."""

import numpy as np
import pytest
import torch

from paint_by_numbers import color

DTYPES = [torch.float32, torch.float64]


def random_srgb(n: int, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, generator=gen, dtype=dtype)


def edge_srgb(dtype: torch.dtype) -> torch.Tensor:
    """Cube corners, every 8-bit gray level, and points around the sRGB knee."""
    corners = torch.tensor(
        [[r, g, b] for r in (0, 1) for g in (0, 1) for b in (0, 1)], dtype=dtype
    )
    grays = (torch.arange(256, dtype=dtype) / 255).unsqueeze(1).expand(-1, 3)
    knee = torch.tensor([0.04045 - 1e-6, 0.04045, 0.04045 + 1e-6], dtype=dtype)
    knee = knee.unsqueeze(1).expand(-1, 3)
    return torch.cat([corners, grays, knee])


# ---------------------------------------------------------------- acceptance check


@pytest.mark.parametrize("dtype", DTYPES)
def test_srgb_lab_round_trip_within_one_level(dtype):
    """Spec check: sRGB -> Lab -> sRGB within 1/255 on random colors."""
    srgb = torch.cat([random_srgb(100_000, dtype), edge_srgb(dtype)])
    back = color.lab_to_srgb(color.srgb_to_lab(srgb))
    assert (back - srgb).abs().max().item() < 1 / 255


@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-4), (torch.float64, 1e-10)])
def test_srgb_lab_round_trip_is_numerically_tight(dtype, tol):
    """Much tighter than the spec requires, to catch formula slips early."""
    srgb = torch.cat([random_srgb(100_000, dtype, seed=1), edge_srgb(dtype)])
    back = color.lab_to_srgb(color.srgb_to_lab(srgb))
    assert (back - srgb).abs().max().item() < tol


def test_8bit_colors_round_trip_to_same_integers():
    gen = torch.Generator().manual_seed(2)
    srgb8 = torch.randint(0, 256, (50_000, 3), generator=gen)
    back8 = torch.round(color.lab_to_srgb(color.srgb_to_lab(srgb8 / 255.0)) * 255)
    assert torch.equal(back8.long(), srgb8)


# ---------------------------------------------------------------- other round trips


@pytest.mark.parametrize("dtype", DTYPES)
def test_linear_round_trip(dtype):
    srgb = random_srgb(10_000, dtype, seed=3)
    back = color.linear_to_srgb(color.srgb_to_linear(srgb))
    assert torch.allclose(back, srgb, atol=1e-5 if dtype == torch.float32 else 1e-12)


@pytest.mark.parametrize("dtype", DTYPES)
def test_xyz_round_trip(dtype):
    linear = random_srgb(10_000, dtype, seed=4)
    back = color.xyz_to_linear(color.linear_to_xyz(linear))
    assert torch.allclose(back, linear, atol=1e-5 if dtype == torch.float32 else 1e-12)


def test_out_of_gamut_lab_survives_round_trip():
    """No clipping inside the conversions: saturated Lab colors come back intact."""
    lab = torch.tensor(
        [[50.0, 120.0, 0.0], [60.0, -110.0, 90.0], [30.0, 40.0, -130.0]],
        dtype=torch.float64,
    )
    srgb = color.lab_to_srgb(lab)
    assert (srgb < 0).any() or (srgb > 1).any()  # really is out of gamut
    assert torch.allclose(color.srgb_to_lab(srgb), lab, atol=1e-9)


# ---------------------------------------------------------------- reference values


@pytest.mark.parametrize(
    "srgb, lab",
    [
        ([1.0, 1.0, 1.0], [100.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([1.0, 0.0, 0.0], [53.2408, 80.0925, 67.2032]),
        ([0.0, 1.0, 0.0], [87.7347, -86.1827, 83.1793]),
        ([0.0, 0.0, 1.0], [32.2970, 79.1875, -107.8602]),
        ([128 / 255] * 3, [53.5850, 0.0, 0.0]),
    ],
)
def test_reference_values(srgb, lab):
    """Standard sRGB / D65 values (Lindbloom)."""
    got = color.srgb_to_lab(srgb)
    assert torch.allclose(got, torch.tensor(lab, dtype=torch.float64), atol=1e-3)


def test_white_point():
    xyz = color.linear_to_xyz([1.0, 1.0, 1.0])
    assert torch.allclose(xyz, torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64), atol=1e-6)


def test_golden_published_srgb_and_lab_agree_under_d65():
    """Golden's own sRGB and Lab for Green Gold (item 1170) agree under D65.

    Evidence for the spec's D65 / 2 degree assumption. Under D50 the same pair
    would disagree by several units.
    """
    lab = color.srgb_to_lab(torch.tensor([106, 110, 54]) / 255.0)
    assert color.delta_e(lab, [45.05, -10.69, 30.52]).item() < 0.5


# ---------------------------------------------------------------- gradients


def test_gradients_finite_at_zero_knees_and_negatives():
    values = torch.tensor(
        [0.0, 1e-12, 0.0031308, 0.04045, 0.5, 1.0, -0.01, 1.2], dtype=torch.float64
    )
    rgb = torch.cartesian_prod(values, values, values).requires_grad_(True)
    for fn in (color.srgb_to_lab, color.linear_to_lab):
        rgb.grad = None
        fn(rgb).sum().backward()
        assert torch.isfinite(rgb.grad).all(), fn.__name__


def test_gradients_finite_from_lab():
    lab = torch.tensor(
        [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [50.0, 20.0, -30.0], [100.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    color.lab_to_srgb(lab).sum().backward()
    assert torch.isfinite(lab.grad).all()


def test_gradients_match_finite_differences():
    gen = torch.Generator().manual_seed(5)
    srgb = (0.1 + 0.8 * torch.rand(6, 3, generator=gen, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(color.srgb_to_lab, (srgb,))
    lab = color.srgb_to_lab(srgb).detach().requires_grad_(True)
    assert torch.autograd.gradcheck(color.lab_to_srgb, (lab,))


# ---------------------------------------------------------------- layout and inputs


def test_channel_first_layout_matches_channel_last():
    image = random_srgb(4 * 5, torch.float64, seed=6).reshape(4, 5, 3)
    last = color.srgb_to_lab(image)
    first = color.srgb_to_lab(image.permute(2, 0, 1), dim=0)
    assert torch.allclose(first.permute(1, 2, 0), last)
    assert torch.allclose(color.lab_to_srgb(first, dim=0).permute(1, 2, 0), image)


def test_batched_layer_stack_layout():
    """(n, 3, H, W) as the stack renderer will produce for any number of layers."""
    stack = random_srgb(7 * 2 * 3, torch.float32, seed=7).reshape(7, 2, 3, 3).permute(0, 3, 1, 2)
    lab = color.srgb_to_lab(stack, dim=1)
    assert lab.shape == (7, 3, 2, 3)
    assert torch.allclose(color.lab_to_srgb(lab, dim=1), stack, atol=1e-4)


def test_accepts_lists_and_numpy():
    from_list = color.srgb_to_lab([0.2, 0.4, 0.6])
    from_numpy = color.srgb_to_lab(np.array([0.2, 0.4, 0.6]))
    assert from_list.dtype == torch.float64
    assert torch.equal(from_list, from_numpy)


def test_float32_stays_float32():
    assert color.srgb_to_lab(torch.rand(3)).dtype == torch.float32


def test_wrong_channel_count_raises():
    with pytest.raises(ValueError):
        color.srgb_to_lab(torch.rand(10, 4))


# ---------------------------------------------------------------- color error


def test_delta_e():
    a = torch.tensor([[50.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    b = torch.tensor([[53.0, 4.0, 0.0], [50.0, 0.0, 0.0]])
    assert torch.allclose(color.delta_e(a, b), torch.tensor([5.0, 0.0]))
    assert torch.allclose(color.delta_e_squared(a, b), torch.tensor([25.0, 0.0]))


# ---------------------------------------------------------------- working space (linear Rec.2020)


def test_working_space_matches_published_rec2020_matrix():
    """Derived from BT.2020 primaries with Lab's white; the published matrix
    uses a D65 white that differs in the 5th decimal."""
    published = torch.tensor(
        [[0.6369580, 0.1446169, 0.1688810], [0.2627002, 0.6779981, 0.0593017], [0.0, 0.0280727, 1.0609851]],
        dtype=torch.float64,
    )
    derived = color.working_to_xyz(torch.eye(3, dtype=torch.float64)).T  # columns = primaries
    assert torch.allclose(derived, published, atol=5e-4)


def test_working_white_is_lab_white():
    assert torch.allclose(color.working_to_lab([1.0, 1.0, 1.0]), torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64), atol=1e-9)


def test_neutral_grays_are_the_same_in_both_spaces():
    grays = torch.linspace(0, 1, 11, dtype=torch.float64).unsqueeze(1).expand(-1, 3)
    assert torch.allclose(color.srgb_to_working(grays), color.srgb_to_linear(grays), atol=1e-12)


def test_srgb_gamut_fits_inside_working_space():
    srgb = torch.cat([random_srgb(50_000, torch.float64, seed=8), edge_srgb(torch.float64)])
    working = color.srgb_to_working(srgb)
    assert working.min() >= -1e-9 and working.max() <= 1 + 1e-9


def test_saturated_paint_color_outside_srgb_fits_working_space():
    """Golden's Cadmium Yellow Medium: outside sRGB, inside Rec.2020."""
    lab = torch.tensor([84.13, 12.86, 94.59], dtype=torch.float64)
    srgb_linear = color.lab_to_linear(lab)
    working = color.lab_to_working(lab)
    assert srgb_linear.max() > 1.0 or srgb_linear.min() < 0.0
    assert working.min() > 0.01 and working.max() < 0.99


@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-4), (torch.float64, 1e-10)])
def test_working_round_trips(dtype, tol):
    srgb = random_srgb(20_000, dtype, seed=9)
    assert (color.working_to_srgb(color.srgb_to_working(srgb)) - srgb).abs().max() < tol
    lab = color.srgb_to_lab(srgb)
    assert (color.working_to_lab(color.lab_to_working(lab)) - lab).abs().max() < tol * 1000


def test_working_gradients_finite_at_zero():
    rgb = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
    color.working_to_lab(rgb).sum().backward()
    assert torch.isfinite(rgb.grad).all()
