"""Milestone 2: Kubelka-Munk renderer."""

import pytest
import torch

from paint_by_numbers import color, km

F64 = torch.float64

# Representative S per coat (the spec's original first guesses). The renderer
# tests use these fixed values so they do not depend on fit.py's calibration.
S_PER_COAT = {"opaque": 3.0, "semi-opaque": 1.5, "semi-transparent": 0.6, "transparent": 0.15}

# Spec's seed paints (Lab, transparency).
SEED_PAINTS = {
    "Green Gold": ([45.05, -10.69, 30.52], "transparent"),
    "Ultramarine Blue": ([24.11, 14.01, -27.81], "semi-transparent"),
    "Burnt Umber": ([27.33, 4.10, 4.19], "opaque"),
    "Cadmium Yellow Primrose": ([90.16, -10.32, 87.73], "opaque"),
}


def spec_ks(lab, rating):
    """K, S from the spec's original fixed-S-per-rating estimate."""
    r = color.lab_to_working(torch.tensor(lab, dtype=F64)).clamp(0.01, 0.99)
    s = torch.full((3,), S_PER_COAT[rating], dtype=F64)
    return (1 - r) ** 2 / (2 * r) * s, s, r


def spec_formula(rg, K, S, t):
    """The single-layer formula exactly as written in the spec."""
    K, S = K.clamp(min=1e-4), S.clamp(min=1e-4)
    a = 1 + K / S
    b = torch.sqrt(a**2 - 1)
    th = torch.tanh(b * S * t)
    return ((1 - a * rg) * th + b * rg) / ((a - rg) * th + b)


def random_paints(n, gen, dtype=F64, ks_range=(1e-3, 1e2), s_range=(0.05, 5.0)):
    """n paints with log-uniform K/S and S per channel. Returns K, S of shape (n, 3)."""

    def log_uniform(lo, hi):
        u = torch.rand(n, 3, generator=gen, dtype=F64)
        return torch.exp(torch.log(torch.tensor(lo)) + u * torch.log(torch.tensor(hi / lo)))

    ks = log_uniform(*ks_range)
    s = log_uniform(*s_range)
    return (ks * s).to(dtype), s.to(dtype)


BACKGROUNDS = {
    "black": [0.0, 0.0, 0.0],
    "white": [1.0, 1.0, 1.0],
    "gray": [0.2158605, 0.2158605, 0.2158605],  # sRGB #808080
    "umber": [0.19, 0.14, 0.10],
}


# ---------------------------------------------------------------- formula


def test_matches_spec_formula():
    gen = torch.Generator().manual_seed(0)
    K, S = random_paints(20_000, gen)
    rg = torch.rand(20_000, 3, generator=gen, dtype=F64)
    t = 5 * torch.rand(20_000, 1, generator=gen, dtype=F64)
    assert torch.allclose(km.layer(rg, K, S, t), spec_formula(rg, K, S, t), rtol=1e-10, atol=1e-12)


def test_masstone_is_a_minus_b():
    gen = torch.Generator().manual_seed(1)
    K, S = random_paints(1000, gen)
    a = 1 + K.clamp(min=km.MIN_KS) / S.clamp(min=km.MIN_KS)
    assert torch.allclose(km.masstone(K, S), a - torch.sqrt(a**2 - 1), rtol=1e-9)


@pytest.mark.parametrize("rating", S_PER_COAT)
def test_masstone_inverts_the_spec_ks_estimate(rating):
    """K/S = (1 - R)^2 / 2R puts the masstone back at R, so fitting starts exact,
    wherever the K clamp does not apply."""
    r = torch.linspace(0.01, 0.99, 981, dtype=F64).unsqueeze(1).expand(-1, 3)
    S = torch.full_like(r, S_PER_COAT[rating])
    K = (1 - r) ** 2 / (2 * r) * S
    unclamped = K >= km.MIN_KS
    assert r[unclamped].max() > 0.96
    assert torch.allclose(km.masstone(K, S)[unclamped], r[unclamped], atol=1e-12)


def test_k_clamp_darkens_very_light_transparent_channels():
    """Spec interaction: at S = 0.15, K < 1e-4 once R > ~0.964, and the clamp
    raises K, so the masstone lands darker than the published color."""
    r = torch.tensor([[0.95, 0.97, 0.99]], dtype=F64)
    S = torch.full_like(r, S_PER_COAT["transparent"])
    got = km.masstone((1 - r) ** 2 / (2 * r) * S, S).flatten()
    assert abs(got[0] - 0.95) < 1e-12  # K above the clamp: exact
    assert got[1] < 0.97 - 1e-4 and got[2] < 0.99 - 0.01  # clamped: darker


# ---------------------------------------------------------------- spec checks


@pytest.mark.parametrize("dtype", [torch.float32, F64])
def test_zero_thickness_returns_background_exactly(dtype):
    gen = torch.Generator().manual_seed(2)
    K, S = random_paints(10, gen, dtype=dtype)
    background = torch.rand(3, 8, 8, generator=gen, dtype=F64).to(dtype)
    t = torch.zeros(10, 1, 8, 8, dtype=dtype)
    assert torch.equal(km.render_stack(background, K, S, t), background)
    rg = background.permute(1, 2, 0)  # (8, 8, 3)
    assert torch.equal(km.layer(rg, K[0], S[0], torch.zeros((), dtype=dtype)), rg)


@pytest.mark.parametrize("name", SEED_PAINTS)
@pytest.mark.parametrize("bg", BACKGROUNDS)
def test_thickness_50_returns_masstone_for_seed_paints(name, bg):
    K, S, r = spec_ks(*SEED_PAINTS[name])
    out = km.layer(torch.tensor(BACKGROUNDS[bg], dtype=F64), K, S, torch.tensor(50.0, dtype=F64))
    assert (out - km.masstone(K, S)).abs().max() < 1e-3
    assert (out - r).abs().max() < 1e-3  # masstone is the paint's published color


def test_large_thickness_returns_masstone_for_any_paint():
    """Includes very transparent, very light channels, which need far more than 50 coats."""
    gen = torch.Generator().manual_seed(3)
    K, S = random_paints(5000, gen)
    for bg in BACKGROUNDS.values():
        out = km.layer(torch.tensor(bg, dtype=F64), K, S, torch.tensor(1e4, dtype=F64))
        assert (out - km.masstone(K, S)).abs().max() < 1e-9


@pytest.mark.parametrize("dtype", [torch.float32, F64])
def test_output_stays_in_unit_interval(dtype):
    gen = torch.Generator().manual_seed(4)
    n = 50
    K, S = random_paints(n, gen, dtype=dtype, ks_range=(1e-4, 1e4), s_range=(1e-4, 50.0))
    K[0], S[1] = 0.0, 0.0  # exercise the clamps
    t = 20 * torch.rand(n, 1, 32, 32, generator=gen, dtype=F64).to(dtype)
    t[:, :, :4] = 0.0
    t[:, :, -4:] = 500.0
    for bg in (torch.zeros(3), torch.ones(3), torch.rand(3, 32, 32, generator=gen)):
        out = km.render_stack(bg.to(dtype), K, S, t)
        assert out.min() >= 0 and out.max() <= 1
        # every intermediate layer too
        r = bg.to(dtype)
        if r.ndim == 1:
            r = r.reshape(3, 1, 1)
        for i in range(n):
            r = km.layer(r, K[i].reshape(3, 1, 1), S[i].reshape(3, 1, 1), t[i])
            assert r.min() >= 0 and r.max() <= 1


@pytest.mark.parametrize("bg", ["black", "white", "umber"])
def test_gradients_finite_at_zero_thickness(bg):
    gen = torch.Generator().manual_seed(5)
    K, S = random_paints(8, gen)
    K.requires_grad_(True)
    S.requires_grad_(True)
    t = torch.zeros(8, 1, 4, 4, dtype=F64, requires_grad=True)
    target = color.srgb_to_lab(torch.rand(3, 4, 4, generator=gen, dtype=F64), dim=0)

    out = km.render_stack(torch.tensor(BACKGROUNDS[bg], dtype=F64), K, S, t)
    loss = color.delta_e_squared(color.working_to_lab(out, dim=0), target, dim=0).mean()
    loss.backward()

    for name, x in (("t", t), ("K", K), ("S", S)):
        assert torch.isfinite(x.grad).all(), name
    # Zero-thickness layers pass changes beneath them straight through, so
    # every layer gets a nonzero gradient even from an all-zero start.
    assert (t.grad.abs().amax(dim=(1, 2, 3)) > 0).all()


def test_gradients_match_finite_differences():
    gen = torch.Generator().manual_seed(6)
    K, S = random_paints(4, gen, ks_range=(0.05, 5.0), s_range=(0.2, 3.0))
    t = (0.1 + 2 * torch.rand(4, 1, 3, 2, generator=gen, dtype=F64)).requires_grad_(True)
    bg = torch.rand(3, 3, 2, generator=gen, dtype=F64).requires_grad_(True)
    K.requires_grad_(True)
    S.requires_grad_(True)
    assert torch.autograd.gradcheck(km.render_stack, (bg, K, S, t))


# ---------------------------------------------------------------- layer stack


def test_stack_matches_layer_by_layer():
    gen = torch.Generator().manual_seed(7)
    K, S = random_paints(5, gen)
    t = 2 * torch.rand(5, 1, 6, 6, generator=gen, dtype=F64)
    bg = torch.tensor(BACKGROUNDS["umber"], dtype=F64)
    r = bg.reshape(3, 1, 1)
    for i in range(5):  # bottom first
        r = km.layer(r, K[i].reshape(3, 1, 1), S[i].reshape(3, 1, 1), t[i])
    assert torch.allclose(km.render_stack(bg, K, S, t), r, atol=1e-14)


def test_no_layers_returns_background_at_full_size():
    bg = torch.tensor([0.3, 0.4, 0.5], dtype=F64)
    out = km.render_stack(bg, torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 1, 5, 7, dtype=F64))
    assert out.shape == (3, 5, 7)
    assert torch.equal(out, bg.reshape(3, 1, 1).expand(3, 5, 7))


def test_hundreds_of_layers():
    """No fixed layer limit."""
    gen = torch.Generator().manual_seed(8)
    n = 300
    K, S = random_paints(n, gen, dtype=torch.float32)
    t = (0.05 * torch.rand(n, 1, 16, 16, generator=gen)).requires_grad_(True)
    out = km.render_stack(torch.ones(3), K, S, t)
    out.sum().backward()
    assert out.shape == (3, 16, 16)
    assert torch.isfinite(out).all() and torch.isfinite(t.grad).all()
    assert out.min() >= 0 and out.max() <= 1


def test_adjacent_layers_of_one_paint_equal_one_thicker_layer():
    """Why palette search skips placing a paint next to itself."""
    gen = torch.Generator().manual_seed(9)
    K, S = random_paints(200, gen)
    rg = torch.rand(200, 3, generator=gen, dtype=F64)
    t1 = 2 * torch.rand(200, 1, generator=gen, dtype=F64)
    t2 = 2 * torch.rand(200, 1, generator=gen, dtype=F64)
    two = km.layer(km.layer(rg, K, S, t1), K, S, t2)
    assert torch.allclose(two, km.layer(rg, K, S, t1 + t2), atol=1e-12)


def test_repeated_paints_in_a_stack():
    """The same paint in several non-adjacent layers is a valid stack."""
    white_k, white_s = torch.tensor([[0.02, 0.02, 0.025]]), torch.tensor([[3.0, 3.0, 3.0]])
    blue_k, blue_s = torch.tensor([[1.2, 0.6, 0.05]]), torch.tensor([[0.6, 0.6, 0.6]])
    K = torch.cat([white_k, blue_k, white_k])
    S = torch.cat([white_s, blue_s, white_s])
    t = torch.tensor([1.0, 1.0, 0.25]).reshape(3, 1, 1, 1)
    out = km.render_stack(torch.tensor([0.19, 0.14, 0.10]), K, S, t)
    without_top = km.render_stack(torch.tensor([0.19, 0.14, 0.10]), K[:2], S[:2], t[:2])
    assert (out.mean() > without_top.mean()).item()  # the white glaze on top lightens


def test_layer_order_matters():
    opaque_white = (torch.tensor([0.01, 0.01, 0.01]), torch.tensor([3.0, 3.0, 3.0]))
    glaze_blue = (torch.tensor([0.9, 0.5, 0.03]), torch.tensor([0.15, 0.15, 0.15]))
    t = torch.full((2, 1, 1, 1), 2.0)
    bg = torch.tensor([0.19, 0.14, 0.10])

    def stack(*paints):
        return km.render_stack(bg, torch.stack([p[0] for p in paints]), torch.stack([p[1] for p in paints]), t)

    blue_over_white = stack(opaque_white, glaze_blue).flatten()
    white_over_blue = stack(glaze_blue, opaque_white).flatten()
    assert (blue_over_white - white_over_blue).abs().max() > 0.05
    assert blue_over_white[2] > blue_over_white[0]  # blue glaze on top reads blue


def test_per_pixel_background():
    gen = torch.Generator().manual_seed(10)
    K, S = random_paints(3, gen)
    bg = torch.rand(3, 4, 5, generator=gen, dtype=F64)
    t = torch.rand(3, 1, 4, 5, generator=gen, dtype=F64)
    out = km.render_stack(bg, K, S, t)
    for y in range(4):
        for x in range(5):
            single = km.render_stack(bg[:, y, x], K, S, t[:, :, y : y + 1, x : x + 1])
            assert torch.allclose(out[:, y, x], single.flatten(), atol=1e-14)


def test_per_pixel_paints_render_candidate_stacks_side_by_side():
    """Batched palette search: column j uses its own paints."""
    gen = torch.Generator().manual_seed(11)
    n, cols = 4, 6
    K, S = random_paints(n * cols, gen)
    K = K.reshape(n, cols, 3).permute(0, 2, 1).unsqueeze(2)  # (n, 3, 1, cols)
    S = S.reshape(n, cols, 3).permute(0, 2, 1).unsqueeze(2)
    t = torch.rand(n, 1, 7, cols, generator=gen, dtype=F64)
    bg = torch.tensor(BACKGROUNDS["gray"], dtype=F64)
    out = km.render_stack(bg, K, S, t)
    for j in range(cols):
        single = km.render_stack(bg, K[:, :, 0, j], S[:, :, 0, j], t[..., j : j + 1])
        assert torch.allclose(out[..., j : j + 1], single, atol=1e-14)


# ---------------------------------------------------------------- hiding thickness


def test_hiding_thickness_guarantee():
    gen = torch.Generator().manual_seed(12)
    K, S = random_paints(5000, gen, s_range=(0.05, 5.0))
    t = km.hiding_thickness(K, S, tol=1e-3).unsqueeze(1)
    for bg in BACKGROUNDS.values():
        out = km.layer(torch.tensor(bg, dtype=F64), K, S, t)
        assert (out - km.masstone(K, S)).abs().max() <= 1e-3


def test_light_transparent_paint_needs_hundreds_of_coats():
    r = torch.tensor([[0.95, 0.5, 0.1]], dtype=F64)
    s = torch.full_like(r, S_PER_COAT["transparent"])
    assert km.hiding_thickness((1 - r) ** 2 / (2 * r) * s, s).item() > 400


# ---------------------------------------------------------------- mixing


def test_mix():
    K = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    S = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    k_mix, s_mix = km.mix(torch.tensor([0.25, 0.75]), K, S)
    assert torch.allclose(k_mix, torch.tensor([2.5, 2.0, 1.5]))
    assert torch.allclose(s_mix, torch.tensor([1.75, 1.75, 1.75]))
    k_batch, _ = km.mix(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), K, S)
    assert torch.equal(k_batch, K)


def test_glaze_with_medium_equals_thinner_layer():
    """1 part paint to 3 parts clear medium, 1 coat == 0.25 coat of paint."""
    gen = torch.Generator().manual_seed(13)
    K, S = random_paints(1, gen)
    medium = torch.zeros(1, 3, dtype=F64)
    k_glaze, s_glaze = km.mix(torch.tensor([0.25, 0.75]), torch.cat([K, medium]), torch.cat([S, medium]))
    rg = torch.rand(100, 3, generator=gen, dtype=F64)
    glaze = km.layer(rg, k_glaze, s_glaze, torch.tensor(1.0, dtype=F64))
    thin = km.layer(rg, K, S, torch.tensor(0.25, dtype=F64))
    assert torch.allclose(glaze, thin, atol=1e-12)


# ---------------------------------------------------------------- inputs


def test_zero_k_and_s_are_clamped():
    out = km.layer(torch.tensor([0.5, 0.5, 0.5]), torch.zeros(3), torch.zeros(3), torch.tensor(1.0))
    assert torch.isfinite(out).all()


def test_dtype_follows_thickness():
    out = km.render_stack([1.0, 1.0, 1.0], [[0.1, 0.2, 0.3]], [[1.0, 1.0, 1.0]], torch.ones(1, 1, 2, 2))
    assert out.dtype == torch.float32


def test_shape_errors():
    with pytest.raises(ValueError):
        km.render_stack(torch.ones(3), torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 4, 4))
    with pytest.raises(ValueError):
        km.render_stack(torch.ones(3), torch.ones(3, 3), torch.ones(2, 3), torch.ones(2, 1, 4, 4))
