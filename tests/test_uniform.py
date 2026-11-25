"""One weight per layer: layer choice, regions and outputs."""

import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from core_paints import CORE_NAMES, core_paints
from images import landscape
from paint_by_numbers import color, km, plan, uniform

PAINTS = core_paints()
FAST = uniform.UniformConfig(layers=5, n_colors=16, improve_passes=1)


@pytest.fixture(scope="module")
def image():
    return torch.from_numpy(np.asarray(landscape(64, 48), dtype=np.float32) / 255).permute(2, 0, 1).contiguous()


@pytest.fixture(scope="module")
def run(image, tmp_path_factory):
    result = plan.plan_uniform(image, PAINTS, plan.background_from_color("#808080"), FAST)
    out = tmp_path_factory.mktemp("uniform")
    plan.write_outputs(result, out)
    return result, out


def test_every_layer_has_one_weight(run):
    result, _ = run
    t = result.solution.t_snapped
    assert 0 < t.shape[0] <= FAST.layers
    assert len(result.selection.stack) == len(result.selection.weights) == t.shape[0]
    for layer, w in zip(t, result.selection.weights):
        assert set(layer.unique().tolist()) <= {0.0, w}
        assert w in FAST.weights
        assert (layer > 0).any()


def test_plan_beats_the_bare_background(run):
    result, _ = run
    assert result.solution.mean_error < result.selection.baseline_error / 2


def test_preview_is_the_stack_rendered(run):
    result, _ = run
    K = torch.tensor([p.K for p in result.selection.stack], dtype=torch.float32)
    S = torch.tensor([p.S for p in result.selection.stack], dtype=torch.float32)
    out = km.render_stack(result.background.linear.float(), K, S, result.solution.t_snapped)
    assert torch.allclose(out, result.solution.preview, atol=1e-5)
    err = color.delta_e(color.working_to_lab(out, dim=0), result.target_lab, dim=0)
    assert result.solution.mean_error == pytest.approx(err.mean().item(), abs=1e-4)


def test_layer_maps_and_recipe(run):
    result, out = run
    recipe = json.loads((out / "recipe.json").read_text())
    assert recipe["settings"]["mode"] == "one weight per layer"
    assert recipe["settings"]["uniform"]["layers"] == FAST.layers
    for i, (entry, w) in enumerate(zip(recipe["layers"], result.selection.weights)):
        assert entry["coats"] == w
        assert entry["how"] == plan.level_instructions(w)
        assert list(entry["coats_by_area"]) == [f"{w:g}"]
        assert len(entry["thicknesses"]) == 1
        pixels = np.asarray(Image.open(out / entry["file"]))
        assert set(np.unique(pixels).tolist()) == {0, round(w / result.solve_config.t_max * 255)}


def test_descend_never_raises_the_cost():
    """Without smoothing, each pixel's choice is exact given the other layers, so the cost only falls."""
    torch.manual_seed(0)
    target = color.srgb_to_lab(torch.rand(3, 6, 7), dim=0)
    K, S = (x.float() for x in (torch.tensor([p.K for p in PAINTS[:4]]), torch.tensor([p.S for p in PAINTS[:4]])))
    W = torch.tensor([0.25, 1.0, 0.5, 2.0]).reshape(4, 1, 1, 1)
    bg = torch.full((3,), 0.5)
    X = torch.rand(4, 1, 6, 7) > 0.5
    _, before = uniform._cost(target, bg, K, S, W, X, 0.1)
    X, _ = uniform._descend(target, bg, K, S, W, X, 10, 0.1)
    _, after = uniform._cost(target, bg, K, S, W, X, 0.1)
    assert (after <= before + 1e-5).all()


def test_layer_maps_match_the_paint_model():
    """A layer over reflectance R, as (m00 R + m01) / (m10 R + 1), equals km.layer; and two layers composed as matrices equal painting both."""
    K = torch.tensor([p.K for p in PAINTS], dtype=torch.float64)
    S = torch.tensor([p.S for p in PAINTS], dtype=torch.float64)
    W = torch.tensor([0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 0.3, 1.2], dtype=torch.float64).reshape(-1, 1)
    m00, m01, m10 = uniform._layer_maps(K, S, W)
    r = torch.linspace(0, 1, 11, dtype=torch.float64).reshape(-1, 1, 1)
    direct = km.layer(r, K, S, W)
    assert torch.allclose((m00 * r + m01) / (m10 * r + 1), direct, atol=1e-12)
    # layer 1 over layer 0, as one map
    a = torch.tensor([[m00[1, 0], m01[1, 0]], [m10[1, 0], 1.0]])
    b = torch.tensor([[m00[0, 0], m01[0, 0]], [m10[0, 0], 1.0]])
    c = a @ b
    x = torch.tensor(0.37, dtype=torch.float64)
    both = km.layer(km.layer(x, K[0, 0], S[0, 0], W[0, 0]), K[1, 0], S[1, 0], W[1, 0])
    assert torch.isclose((c[0, 0] * x + c[0, 1]) / (c[1, 0] * x + c[1, 1]), both, atol=1e-12)


def test_descend_ends_where_no_single_change_helps():
    """At the end, turning any one layer on or off at any pixel (rendered the slow, direct way) does not lower that pixel's cost."""
    torch.manual_seed(1)
    target = color.srgb_to_lab(torch.rand(3, 5, 6), dim=0)
    n = 5
    K = torch.tensor([p.K for p in PAINTS[:n]], dtype=torch.float32)
    S = torch.tensor([p.S for p in PAINTS[:n]], dtype=torch.float32)
    W = torch.tensor([0.25, 1.0, 0.5, 2.0, 0.125]).reshape(n, 1, 1, 1)
    bg = torch.tensor([0.8, 0.6, 0.4])
    X, _ = uniform._descend(target, bg, K, S, W, torch.rand(n, 1, 5, 6) > 0.5, 50, 0.1)
    _, cost = uniform._cost(target, bg, K, S, W, X, 0.1)
    for i in range(n):
        flipped = X.clone()
        flipped[i] = ~flipped[i]
        _, other = uniform._cost(target, bg, K, S, W, flipped, 0.1)
        assert (other >= cost - 1e-4).all()


def test_clean_regions_removes_specks_and_fills_pinholes():
    m = torch.zeros(1, 1, 20, 20, dtype=torch.bool)
    m[0, 0, 2:12, 2:12] = True  # a block...
    m[0, 0, 6, 6] = False  # ...with a pinhole
    m[0, 0, 17, 17] = True  # a speck
    m[0, 0, 15, 2:19] = True  # a thin line: long enough to stay
    out = uniform.clean_regions(m, min_area=5)[0, 0]
    assert out[6, 6] and not out[17, 17]
    assert out[15, 2:19].all() and out[2:12, 2:12].all()


def test_guided_filter_keeps_edges_and_flat_areas():
    guide = np.zeros((20, 20, 1))
    guide[:, 10:] = 50.0  # a strong edge down the middle
    p = np.where(np.arange(20) < 10, -1.0, 1.0)[None, :].repeat(20, 0)
    p_noisy = p + np.random.default_rng(0).normal(0, 0.3, p.shape)
    q = uniform.guided_filter(guide, 2, 25.0)(p_noisy)
    assert np.array_equal(np.sign(q), p)  # the noise is gone and the edge stays put
    assert np.abs(q - p).mean() < np.abs(p_noisy - p).mean() / 2


def test_weights_must_be_positive(image):
    with pytest.raises(ValueError, match="positive"):
        uniform.select_uniform_stack(color.srgb_to_lab(image, dim=0), torch.ones(3), PAINTS, uniform.UniformConfig(weights=(0.0, 1.0)))


def test_max_paints_limits_the_palette(image):
    cfg = uniform.UniformConfig(layers=4, n_colors=12, improve_passes=1, max_paints=2)
    sel = uniform.select_uniform_stack(color.srgb_to_lab(image, dim=0), torch.ones(3, dtype=torch.float64), PAINTS, cfg)
    assert len({p.name for p in sel.stack}) <= 2


def test_cli_layers(tmp_path):
    src = tmp_path / "img.png"
    landscape(96, 72).save(src)
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "paint_by_numbers.cli", str(src), "--out", str(out), "--size", "40", "--layers", "4",
         "--paints", ",".join(CORE_NAMES), "--levels", "0.25,0.5,1", "--min-region", "0.5"],
        capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    recipe = json.loads((out / "recipe.json").read_text())
    assert 1 <= len(recipe["layers"]) <= 4
    assert all(layer["coats"] in (0.25, 0.5, 1.0) for layer in recipe["layers"])
    assert recipe["settings"]["uniform"]["min_region"] == pytest.approx(0.005)
    assert "of the image" in proc.stdout
    assert (out / "project.pbn").is_file()


@pytest.mark.parametrize(
    "extra, message",
    [(["--layers", "0"], "at least 1"), (["--layers", "3", "--max-layers", "3"], "not both"), (["--layers", "3", "--min-region", "-1"], "negative")],
)
def test_cli_layers_rejects(tmp_path, capsys, extra, message):
    src = tmp_path / "img.png"
    landscape(16, 12).save(src)
    from paint_by_numbers import cli

    with pytest.raises(SystemExit):
        cli.main([str(src), "--out", str(tmp_path / "o"), *extra])
    assert message in capsys.readouterr().err
