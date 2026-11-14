"""Milestone 5: background swap and output files."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from images import landscape
from core_paints import core_paints
from paint_by_numbers import color, km, palette, paints, plan, solve

PAINTS = core_paints()
BY_NAME = {p.name: p for p in PAINTS}
FAST_PALETTE = palette.PaletteConfig(n_colors=32, steps=200)
FAST_SOLVE = solve.SolveConfig(steps=500, refine_sweeps=3)
BACKGROUNDS = {"white": "#ffffff", "gray": "#808080", "black": "#000000"}


@pytest.fixture(scope="module")
def image_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("img") / "landscape.png"
    landscape(96, 72).save(path)
    return path


@pytest.fixture(scope="module")
def runs(image_path, tmp_path_factory):
    """The same image planned over white, mid-gray and black."""
    image = plan.load_image(image_path, size=40)
    out = {}
    for name, hex_color in BACKGROUNDS.items():
        result = plan.plan(image, PAINTS, plan.background_from_color(hex_color), FAST_PALETTE, FAST_SOLVE)
        out_dir = tmp_path_factory.mktemp(name)
        out[name] = (result, out_dir, plan.write_outputs(result, out_dir, breakdown="both"))
    return out


# ---------------------------------------------------------------- acceptance check


def test_background_swap_changes_the_layer_stack(runs):
    """Spec check: white, #808080 and black give different layer stacks."""
    stacks = {name: tuple(p.name for p in r.selection.stack) for name, (r, _, _) in runs.items()}
    assert len(set(stacks.values())) == 3, stacks


@pytest.mark.parametrize("name", BACKGROUNDS)
def test_each_run_writes_all_outputs(runs, name):
    """Spec check: layers, preview, recipe and error map from every run."""
    result, out_dir, written = runs[name]
    stack = result.selection.stack
    layer_files = sorted((out_dir / "layers").glob("*.png"))
    assert len(stack) > 0 and len(layer_files) == len(stack)
    for i, (f, paint) in enumerate(zip(layer_files, stack), start=1):
        assert f.name == f"{i:02d}_{paint.slug}.png"
    for f in ("preview.png", "error.png", "recipe.json"):
        assert (out_dir / f).is_file()
    w, h = result.target_lab.shape[2], result.target_lab.shape[1]
    for f in [out_dir / "preview.png", out_dir / "error.png", *layer_files]:
        assert Image.open(f).size == (w, h)


@pytest.mark.parametrize("name", BACKGROUNDS)
def test_recipe_contents(runs, name):
    result, out_dir, _ = runs[name]
    recipe = json.loads((out_dir / "recipe.json").read_text())
    assert recipe["background"]["kind"] == "color"
    assert recipe["background"]["srgb_hex"] == BACKGROUNDS[name]
    assert [layer["paint"] for layer in recipe["layers"]] == [p.name for p in result.selection.stack]
    assert recipe["error"]["mean"] == pytest.approx(result.solution.mean_error, abs=1e-3)
    assert recipe["error"]["max"] == pytest.approx(result.solution.max_error, abs=1e-3)
    assert [lv["coats"] for lv in recipe["thickness_levels"]] == list(FAST_SOLVE.levels)
    for layer in recipe["layers"]:
        assert 0 < layer["coverage"] <= 1
        assert set(layer["coats_by_area"]) <= {f"{lv:g}" for lv in FAST_SOLVE.levels}


def test_layer_png_values_encode_thickness(runs):
    result, out_dir, _ = runs["white"]
    first = sorted((out_dir / "layers").glob("*.png"))[0]
    pixels = np.asarray(Image.open(first), dtype=np.float64)
    expected = (result.solution.t_snapped[0, 0] / FAST_SOLVE.t_max * 255).round().numpy()
    assert np.array_equal(pixels, expected)


def test_preview_matches_solution(runs):
    result, out_dir, _ = runs["gray"]
    preview = np.asarray(Image.open(out_dir / "preview.png"), dtype=np.float64) / 255
    expected = color.working_to_srgb(result.solution.preview, dim=0).clamp(0, 1).permute(1, 2, 0).numpy()
    assert np.abs(preview - expected).max() <= 0.5 / 255 + 1e-6


def test_mixed_sheens_get_a_varnish_note(runs):
    for result, out_dir, _ in runs.values():
        recipe = json.loads((out_dir / "recipe.json").read_text())
        mixed = len({p.sheen for p in result.selection.stack}) > 1
        assert any("varnish" in n for n in recipe["notes"]) == mixed


# ---------------------------------------------------------------- backgrounds


@pytest.mark.parametrize(
    "text, rgb",
    [("#7A6A58", (122, 106, 88)), ("7a6a58", (122, 106, 88)), ("#fff", (255, 255, 255)), ("122, 106, 88", (122, 106, 88))],
)
def test_parse_color(text, rgb):
    assert torch.allclose(plan.parse_color(text), torch.tensor(rgb, dtype=torch.float64) / 255)


@pytest.mark.parametrize("text", ["#12345", "red", "300,0,0", ""])
def test_parse_color_rejects(text):
    with pytest.raises(ValueError):
        plan.parse_color(text)


def test_background_from_color_is_linear():
    bg = plan.background_from_color("#808080")
    assert torch.allclose(bg.linear, torch.full((3,), 0.2158605, dtype=torch.float64), atol=1e-6)
    assert bg.description == "#808080"


def test_background_paint_is_masstone_and_layer_zero(image_path, tmp_path):
    umber = BY_NAME["Burnt Umber"]
    bg = plan.background_from_paint(umber)
    expected = km.masstone(torch.tensor(umber.K, dtype=torch.float64), torch.tensor(umber.S, dtype=torch.float64))
    assert torch.allclose(bg.linear, expected)
    image = plan.load_image(image_path, size=16)
    result = plan.plan(image, PAINTS, bg, palette.PaletteConfig(n_colors=8, steps=100, max_layers=2), solve.SolveConfig(steps=100, refine_sweeps=1))
    plan.write_outputs(result, tmp_path)
    recipe = json.loads((tmp_path / "recipe.json").read_text())
    assert recipe["background"] == {"kind": "paint", "value": "Burnt Umber", "srgb_hex": recipe["background"]["srgb_hex"], "lab": recipe["background"]["lab"]}
    assert recipe["layers"][0]["layer"] == 0 and recipe["layers"][0]["paint"] == "Burnt Umber"
    assert recipe["layers"][0]["role"] == "ground"
    assert [layer["layer"] for layer in recipe["layers"]] == list(range(len(result.selection.stack) + 1))


def test_background_image_is_resized_per_pixel(tmp_path, image_path):
    ground = tmp_path / "ground.png"
    Image.new("RGB", (10, 10), (128, 128, 128)).save(ground)
    bg = plan.background_from_image(ground, height=12, width=16)
    assert bg.kind == "image" and bg.linear.shape == (3, 12, 16)
    image = plan.load_image(image_path, size=16)
    result = plan.plan(image, PAINTS, plan.background_from_image(ground, *image.shape[1:]),
                       palette.PaletteConfig(n_colors=8, steps=100, max_layers=1), solve.SolveConfig(steps=100, refine_sweeps=1))  # fmt: skip
    plan.write_outputs(result, tmp_path / "out")
    assert json.loads((tmp_path / "out" / "recipe.json").read_text())["background"]["kind"] == "image"


def test_background_image_size_mismatch_raises(tmp_path, image_path):
    image = plan.load_image(image_path, size=16)
    bg = plan.Background("image", torch.ones(3, 5, 5, dtype=torch.float64), "x.png")
    with pytest.raises(ValueError, match="background image"):
        plan.plan(image, PAINTS, bg)


# ---------------------------------------------------------------- inputs and outputs


def test_load_image_resizes_long_side(image_path):
    img = plan.load_image(image_path, size=48)
    assert img.shape == (3, 36, 48)
    assert img.min() >= 0 and img.max() <= 1


def test_layer_filename_padding():
    p = BY_NAME["Titanium White"]
    assert plan.layer_filename(3, 9, p) == "03_titanium-white.png"
    assert plan.layer_filename(7, 120, p) == "007_titanium-white.png"


@pytest.mark.parametrize(
    "coats, text",
    [
        (0.0, "no paint"),
        (0.125, "1 coat of 1 part paint to 7 parts medium"),
        (0.25, "1 coat of 1 part paint to 3 parts medium"),
        (0.75, "1 coat of 3 parts paint to 1 part medium"),
        (1.0, "1 full coat"),
        (1.5, "1 full coat + 1 coat of 1 part paint to 1 part medium"),
        (2.0, "2 full coats"),
    ],
)
def test_level_instructions(coats, text):
    assert plan.level_instructions(coats) == text


def test_rerun_removes_stale_layer_files(runs, tmp_path):
    result, _, _ = runs["white"]
    stale = tmp_path / "layers" / "99_old-paint.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"")
    plan.write_outputs(result, tmp_path)
    assert not stale.exists()


# ---------------------------------------------------------------- color renderings


def _png(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255


def _srgb(reflectance):
    return color.working_to_srgb(reflectance.float(), dim=0).clamp(0, 1).permute(1, 2, 0).numpy()


@pytest.mark.parametrize("name", BACKGROUNDS)
def test_color_layer_and_step_files(runs, name):
    result, out_dir, _ = runs[name]
    stack = result.selection.stack
    color_files = sorted((out_dir / "layers_color").glob("*.png"))
    step_files = sorted((out_dir / "steps").glob("*.png"))
    assert [f.name for f in color_files] == [f.name for f in step_files] == sorted(f.name for f in (out_dir / "layers").glob("*.png"))
    recipe = json.loads((out_dir / "recipe.json").read_text())
    for layer, f in zip(recipe["layers"], color_files):
        assert layer["color_file"] == f"layers_color/{f.name}" and layer["step_file"] == f"steps/{f.name}"
    assert np.array_equal(_png(step_files[-1]), _png(out_dir / "preview.png"))  # last step is the finished painting
    assert len(color_files) == len(stack)


def test_color_layer_is_that_paint_alone_over_its_ground(runs):
    result, out_dir, _ = runs["black"]
    K, S = paints.ks_tensors(result.selection.stack)
    t = result.solution.t_snapped.double()
    layers = json.loads((out_dir / "recipe.json").read_text())["layers"]
    for i, (f, layer) in enumerate(zip(sorted((out_dir / "layers_color").glob("*.png")), layers)):
        ground = {"white": 1.0, "black": 0.0}[layer["color_file_ground"]]
        bg = torch.full((3,), ground, dtype=torch.float64)
        expected = _srgb(km.render_stack(bg, K[i : i + 1], S[i : i + 1], t[i : i + 1]))
        assert np.abs(_png(f) - expected).max() <= 0.5 / 255 + 1e-6


def test_white_paint_alone_is_shown_over_black():
    """White over white would be invisible, so the contrast pick chooses black."""
    white = BY_NAME["Titanium White"]
    K, S = paints.ks_tensors([white])
    t = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    t[..., :2] = 1.0
    _, ground = plan._layer_alone(K, S, t)
    assert ground == "black"
    _, ground = plan._layer_alone(*paints.ks_tensors([BY_NAME["Burnt Umber"]]), t)
    assert ground == "white"


@pytest.mark.parametrize("name", BACKGROUNDS)
def test_overview_sheet(runs, name):
    result, out_dir, _ = runs[name]
    w, h = result.target_lab.shape[2], result.target_lab.shape[1]
    sheet = Image.open(out_dir / "overview.png")
    n = len(result.selection.stack)
    assert sheet.width >= min(n, 6) * w
    assert sheet.height >= 3 * h  # layers alone, steps, target/preview/error


def test_steps_build_up_layer_by_layer(runs):
    result, out_dir, _ = runs["gray"]
    K, S = paints.ks_tensors(result.selection.stack)
    t, bg = result.solution.t_snapped.double(), result.background.linear.double()
    for i, f in enumerate(sorted((out_dir / "steps").glob("*.png")), start=1):
        expected = _srgb(km.render_stack(bg, K[:i], S[:i], t[:i]))
        assert np.abs(_png(f) - expected).max() <= 0.5 / 255 + 1e-6


# ---------------------------------------------------------------- passes (one layer broken into coats)


def test_layer_passes_add_back_up_exactly_and_nest():
    levels = solve.SolveConfig().levels
    t = torch.tensor([[0.0, 0.125, 0.25], [0.5, 1.0, 2.0], [0.75, 1.5, 0.125]])
    passes = plan.layer_passes(t, levels)
    total = sum(coats * mask.double() for coats, mask in passes)
    assert torch.equal(total.float(), t)
    for (_, outer), (_, inner) in zip(passes, passes[1:]):
        assert not (inner & ~outer).any()  # each pass lies inside the previous
    assert [c for c, _ in passes] == [0.125, 0.125, 0.25, 0.25, 0.25, 0.5, 0.5]


def test_layer_passes_skip_unused_levels():
    t = torch.tensor([[0.0, 0.5], [0.5, 2.0]])
    assert [(c, m.sum().item()) for c, m in plan.layer_passes(t, solve.SolveConfig().levels)] == [(0.5, 3), (1.5, 1)]
    assert plan.layer_passes(torch.zeros(2, 2), solve.SolveConfig().levels) == []


def test_passes_render_exactly_like_the_layer():
    """Stacked coats of one paint equal one layer of the summed thickness."""
    paint = BY_NAME["Quinacridone Magenta"]
    K, S = paints.ks_tensors([paint])
    gen = torch.Generator().manual_seed(3)
    levels = torch.tensor(solve.SolveConfig().levels, dtype=torch.float64)
    t = levels[torch.randint(0, len(levels), (6, 6), generator=gen)]
    bg = torch.tensor([0.4, 0.3, 0.35], dtype=torch.float64)
    one_layer = km.render_stack(bg, K, S, t.reshape(1, 1, 6, 6))
    passes = plan.layer_passes(t, solve.SolveConfig().levels)
    Kp, Sp = K.expand(len(passes), 3), S.expand(len(passes), 3)
    tp = torch.stack([(c * m.double()).reshape(1, 6, 6) for c, m in passes])
    assert torch.allclose(km.render_stack(bg, Kp, Sp, tp), one_layer, atol=1e-12)


@pytest.mark.parametrize("name", BACKGROUNDS)
def test_pass_files_and_recipe(runs, name):
    result, out_dir, _ = runs[name]
    recipe = json.loads((out_dir / "recipe.json").read_text())
    listed = []
    for i, layer in enumerate(recipe["layers"]):
        t = result.solution.t_snapped[i, 0]
        assert sum(p["coats"] for p in layer["passes"]) == pytest.approx(t.max().item())
        for p in layer["passes"]:
            mask = np.asarray(Image.open(out_dir / p["file"])) > 127
            assert mask.mean() == pytest.approx(p["area"], abs=1e-4)
            assert p["how"] == plan.level_instructions(p["coats"])
            listed.append(Path(p["file"]).name)
    assert sorted(listed) == sorted(f.name for f in (out_dir / "passes").glob("*.png"))
    assert (out_dir / "passes_overview.png").is_file()


def test_pass_filename():
    assert plan.pass_filename(3, 12, BY_NAME["Titanium White"], 2) == "03_titanium-white_pass2.png"


# ---------------------------------------------------------------- each thickness as its own layer


def test_layer_thicknesses_are_disjoint_and_cover_the_layer():
    t = torch.tensor([[0.0, 0.125, 0.25], [0.5, 1.0, 2.0], [0.75, 1.5, 0.125]])
    regions = plan.layer_thicknesses(t, solve.SolveConfig().levels)
    assert [c for c, _ in regions] == [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    union = torch.zeros_like(t, dtype=torch.bool)
    for _, mask in regions:
        assert not (mask & union).any()  # no overlap
        union |= mask
    assert torch.equal(union, t > 0)
    assert torch.equal(sum(c * m.float() for c, m in regions), t)


@pytest.mark.parametrize("name", BACKGROUNDS)
def test_thickness_files_and_recipe(runs, name):
    result, out_dir, _ = runs[name]
    layers = json.loads((out_dir / "recipe.json").read_text())["layers"]
    listed = []
    for i, layer in enumerate(layers):
        t = result.solution.t_snapped[i, 0]
        assert [e["coats"] for e in layer["thicknesses"]] == sorted(t[t > 0].unique().tolist())
        for e in layer["thicknesses"]:
            mask = np.asarray(Image.open(out_dir / e["mask_file"])) > 127
            assert np.array_equal(mask, (t == e["coats"]).numpy())
            assert e["how"] == plan.level_instructions(e["coats"])
            listed += [Path(e[k]).name for k in ("file", "mask_file", "after_file")]
        # after the thickest region the layer is complete: same as that layer's step image
        last_after = _png(out_dir / layer["thicknesses"][-1]["after_file"])
        assert np.array_equal(last_after, _png(out_dir / layer["step_file"]))
    assert sorted(listed) == sorted(f.name for f in (out_dir / "thickness").glob("*.png"))
    assert (out_dir / "thickness_overview.png").is_file()


def test_thickness_region_alone_renders_only_that_region(runs):
    result, out_dir, _ = runs["white"]
    layer = json.loads((out_dir / "recipe.json").read_text())["layers"][0]
    K, S = paints.ks_tensors(result.selection.stack[:1])
    ground = torch.full((3,), {"white": 1.0, "black": 0.0}[layer["color_file_ground"]], dtype=torch.float64)
    t = result.solution.t_snapped[0, 0].double()
    for e in layer["thicknesses"]:
        t_region = (e["coats"] * (t == e["coats"]).double()).reshape(1, 1, *t.shape)
        expected = _srgb(km.render_stack(ground, K, S, t_region))
        assert np.abs(_png(out_dir / e["file"]) - expected).max() <= 0.5 / 255 + 1e-6


@pytest.mark.parametrize("breakdown, present, absent", [
    ("thickness", ["thickness", "thickness_overview.png"], ["passes", "passes_overview.png"]),
    ("passes", ["passes", "passes_overview.png"], ["thickness", "thickness_overview.png"]),
    ("none", [], ["thickness", "passes", "thickness_overview.png", "passes_overview.png"]),
])  # fmt: skip
def test_breakdown_option(runs, tmp_path, breakdown, present, absent):
    result, _, _ = runs["white"]
    plan.write_outputs(result, tmp_path, breakdown="both")  # leave both behind, then switch
    plan.write_outputs(result, tmp_path, breakdown=breakdown)
    for name in present:
        assert (tmp_path / name).exists()
    for name in absent:
        target = tmp_path / name
        assert not target.exists() or (target.is_dir() and not any(target.glob("*.png")))
    layers = json.loads((tmp_path / "recipe.json").read_text())["layers"]
    assert ("thicknesses" in layers[0]) == (breakdown in ("thickness", "both"))
    assert ("passes" in layers[0]) == (breakdown in ("passes", "both"))


def test_unknown_breakdown_raises(runs, tmp_path):
    with pytest.raises(ValueError, match="breakdown"):
        plan.write_outputs(runs["white"][0], tmp_path, breakdown="slices")


def test_recipe_lists_paints_used(runs):
    result, out_dir, _ = runs["black"]
    recipe = json.loads((out_dir / "recipe.json").read_text())
    assert recipe["selection"]["paints_used"] == sorted({p.name for p in result.selection.stack})


@pytest.mark.parametrize("coats, label", [(0.125, "1/8 coat"), (0.5, "1/2 coat"), (1.0, "1 coat"), (1.5, "1 1/2 coats"), (2.0, "2 coats")])
def test_coats_label(coats, label):
    assert plan._coats_label(coats) == label


# ---------------------------------------------------------------- paint swatches


def _hex(rgb):
    return "#" + "".join(f"{v:02x}" for v in rgb)


@pytest.mark.parametrize("name", BACKGROUNDS)
def test_swatch_files_and_recipe(runs, name):
    result, out_dir, _ = runs[name]
    layers = json.loads((out_dir / "recipe.json").read_text())["layers"]
    swatch_files = sorted((out_dir / "swatches").glob("*.png"))
    assert [f.name for f in swatch_files] == sorted(f.name for f in (out_dir / "layers").glob("*.png"))
    for i, (layer, paint) in enumerate(zip(layers, result.selection.stack)):
        assert layer["swatch_file"] == f"swatches/{swatch_files[i].name}"
        t = result.solution.t_snapped[i, 0]
        used = sorted(t[t > 0].unique().tolist())
        swatch = layer["swatch"]
        assert list(swatch["on_white"]) == [f"{c:g}" for c in used] == list(swatch["on_black"])
        # a flat background other than white or black gets its own row
        assert ("on_ground" in swatch) == (name == "gray")
        K, S = torch.tensor(paint.K, dtype=torch.float64), torch.tensor(paint.S, dtype=torch.float64)
        assert swatch["full_strength"] == _hex(plan._srgb255(km.masstone(K, S)))
        for c in used:
            on_white = km.layer(torch.ones(3, dtype=torch.float64), K, S, torch.tensor(c, dtype=torch.float64))
            assert swatch["on_white"][f"{c:g}"] == _hex(plan._srgb255(on_white))


def test_swatch_colors_get_lighter_or_darker_toward_full_strength():
    """Thicker paint over white moves toward the masstone: a dark paint gets darker."""
    umber = BY_NAME["Burnt Umber"]
    colors = plan.swatch_colors(umber, [0.125, 0.5, 2.0])
    lightness = [sum(colors["on_white"][c]) for c in (0.125, 0.5, 2.0)]
    assert lightness == sorted(lightness, reverse=True)
    assert sum(colors["full_strength"]) <= lightness[-1] + 3
    assert "on_ground" not in colors
    ground = torch.tensor([0.4, 0.2, 0.3], dtype=torch.float64)
    assert set(plan.swatch_colors(umber, [1.0], ground)["on_ground"]) == {1.0}


def test_swatch_card_size():
    colors = plan.swatch_colors(BY_NAME["Titanium White"], [0.125, 0.25, 1.5], torch.tensor([0.4, 0.2, 0.3], dtype=torch.float64))
    for size in [(256, 192), (520, 260), (40, 40)]:
        assert plan._swatch_card(BY_NAME["Titanium White"], colors, size).size == size


def test_swatch_title_always_fits():
    """Long names shrink, then wrap, instead of running off the card."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    title = "12. Quinacridone / Nickel Azo Gold (semi-transparent)"
    for width, size in [(700, 16), (256, 12), (150, 12), (90, 12)]:
        font, lines = plan._fit_title(draw, title, width, size)
        assert " ".join(lines).replace(" (", "(").replace("(", " (") == title or len(lines) > 1
        assert all(draw.textlength(line, font=font) <= width or " " not in line for line in lines)
    font, lines = plan._fit_title(draw, title, 700, 16)
    assert lines == [title] and font.size == 16  # roomy: unchanged
    font, lines = plan._fit_title(draw, title, 90, 12)
    assert len(lines) >= 2 and font.size == 8  # narrow: smallest font, wrapped
