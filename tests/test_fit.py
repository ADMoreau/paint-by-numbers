"""Milestone 3: paint database and K/S fitting."""

import importlib.util
from pathlib import Path

import pytest
import torch

from paint_by_numbers import color, fit, km, paints

ROOT = Path(__file__).resolve().parent.parent
WHITE, BLACK = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]

PAINTS = paints.load_paints()
# Fluorescent and phosphorescent paints emit light, so a reflectance model cannot reproduce them.
MODELED = [p for p in PAINTS if p.special is None]


def in_srgb_gamut(lab) -> bool:
    srgb = color.lab_to_srgb(torch.tensor(lab, dtype=torch.float64))
    return bool(((srgb >= -1e-6) & (srgb <= 1 + 1e-6)).all())


# ---------------------------------------------------------------- acceptance check


@pytest.mark.parametrize("paint", MODELED, ids=lambda p: p.slug)
@pytest.mark.parametrize("background", [WHITE, BLACK], ids=["white", "black"])
def test_thick_layer_renders_published_color(paint, background):
    """Spec check: a thick layer over white and over black is within 2 Lab units."""
    assert fit.thick_layer_error(paint, background) < 2.0


# ---------------------------------------------------------------- database contents


def test_database_has_required_paints():
    names = {p.name for p in PAINTS}
    assert "Titanium White" in names
    assert "Carbon Black" in names  # a black
    assert "Pyrrole Red" in names  # a red
    assert {"Ultramarine Blue", "Phthalo Blue (Green Shade)"} <= names  # blues
    assert {"Green Gold", "Burnt Umber", "Cadmium Yellow Primrose"} <= names  # seeds


def test_seed_values_match_spec():
    spec = {
        "Green Gold": ((45.05, -10.69, 30.52), "transparent"),
        "Ultramarine Blue": ((24.11, 14.01, -27.81), "semi-transparent"),
        "Burnt Umber": ((27.33, 4.10, 4.19), "opaque"),
        "Cadmium Yellow Primrose": ((90.16, -10.32, 87.73), "opaque"),
    }
    by_name = {p.name: p for p in PAINTS}
    for name, (lab, rating) in spec.items():
        assert by_name[name].lab == pytest.approx(lab)
        assert by_name[name].transparency == rating


def test_records_are_valid():
    records = paints.load_records()
    assert len({r["name"] for r in records}) == len(records)
    assert len({r["item"] for r in records}) == len(records)
    for r in records:
        assert r["transparency"] in paints.TRANSPARENCY_RATINGS, r["name"]
        assert len(r["lab"]) == 3 and len(r["srgb"]) == 3
        assert all(0 <= v <= 255 for v in r["srgb"])
        assert r["pigments"] or r.get("special"), r["name"]
        assert r["sheen"] is None or r["sheen"] in {"gloss", "semi-gloss", "satin", "matte"}, r["name"]
        assert isinstance(r.get("discontinued", False), bool)
        assert r["source"].startswith("https://goldenartistcolors.com/products/")


@pytest.mark.parametrize("paint", [p for p in PAINTS if in_srgb_gamut(p.lab)], ids=lambda p: p.slug)
def test_published_srgb_and_lab_agree_under_d65(paint):
    """Golden's own sRGB and Lab agree under D65 for every paint inside the sRGB
    gamut (spec open question). Outside it, Golden's sRGB is clipped."""
    lab_from_srgb = color.srgb_to_lab(torch.tensor(paint.srgb) / 255.0)
    assert color.delta_e(lab_from_srgb, paint.lab).item() < 1.0


def test_stored_ks_are_what_fit_produces():
    """K and S come from fit.py, not by hand, and are current with COATS_TO_HIDE."""
    records = paints.load_records()
    for stored, fresh in zip(records, fit.fit_records(records)):
        assert stored["K"] == fresh["K"], stored["name"]
        assert stored["S"] == fresh["S"], stored["name"]


# ---------------------------------------------------------------- estimate_ks


def test_estimate_ks_mid_gray():
    K, S = fit.estimate_ks([50.0, 0.0, 0.0], "opaque")
    r = 0.18418652  # linear reflectance at L* = 50
    assert torch.all(S == S[0])  # same S on every channel
    assert torch.allclose(K / S, torch.full((3,), (1 - r) ** 2 / (2 * r), dtype=torch.float64), rtol=1e-6)


@pytest.mark.parametrize("paint", PAINTS, ids=lambda p: p.slug)
def test_coverage_matches_rating(paint):
    """Each paint hides black and white to 2 ΔE in exactly its rating's coat count."""
    coats = fit.COATS_TO_HIDE[paint.transparency]
    assert fit.hiding_error(paint.K, paint.S, coats) <= fit.HIDING_DELTA_E + 1e-3
    assert fit.hiding_error(paint.K, paint.S, 0.95 * coats) > fit.HIDING_DELTA_E


@pytest.mark.parametrize("rating", list(fit.COATS_TO_HIDE))
def test_more_transparent_ratings_scatter_less(rating):
    """Same color, different rating: S scales inversely with coats to hide."""
    _, s_opaque = fit.estimate_ks([40.0, 10.0, -10.0], "opaque")
    _, s = fit.estimate_ks([40.0, 10.0, -10.0], rating)
    expected = s_opaque[0] / fit.COATS_TO_HIDE[rating]
    assert s[0].item() == pytest.approx(expected.item(), rel=1e-6)


def test_light_and_dark_opaque_paints_hide_equally():
    """The point of calibrating: one coat of white over black hides as well as
    one coat of black over white."""
    by_name = {p.name: p for p in PAINTS}
    white, black = by_name["Titanium White"], by_name["Carbon Black"]
    assert fit.hiding_error(white.K, white.S, 1.0) == pytest.approx(fit.hiding_error(black.K, black.S, 1.0), abs=1e-3)


def test_unknown_rating_raises():
    with pytest.raises(ValueError):
        fit.estimate_ks([50.0, 0.0, 0.0], "translucent")


def test_reflectance_is_clipped():
    assert torch.all(fit.published_reflectance([1.0, 0.0, 0.0]) == 0.01)
    assert torch.all(fit.published_reflectance([100.0, 0.0, 0.0]) == 0.99)


def test_masstone_of_fit_equals_published_reflectance():
    for p in MODELED:
        r = fit.published_reflectance(p.lab)
        masstone = km.masstone(torch.tensor(p.K, dtype=torch.float64), torch.tensor(p.S, dtype=torch.float64))
        assert torch.allclose(masstone, r, atol=1e-5), p.name


# ---------------------------------------------------------------- load / save


def _record(name, item, lab, K=None, S=None):
    return {
        "name": name, "item": item, "line": "Heavy Body", "lab": lab, "srgb": [0, 0, 0],
        "transparency": "opaque", "sheen": "gloss", "pigments": ["PW6"], "K": K, "S": S,
    }  # fmt: skip


def test_interference_colors_are_skipped(tmp_path):
    db = tmp_path / "db.json"
    records = [_record("Interference Gold", 2401, [0, 0, 0]), _record("Gray", 1, [50.0, 0.0, 0.0])]
    paints.save_records(fit.fit_records(records), db)
    saved = {r["name"]: r for r in paints.load_records(db)}
    assert saved["Interference Gold"]["K"] is None
    assert [p.name for p in paints.load_paints(db)] == ["Gray"]


def test_unfitted_paint_raises(tmp_path):
    db = tmp_path / "db.json"
    paints.save_records([_record("Gray", 1, [50.0, 0.0, 0.0])], db)
    with pytest.raises(ValueError, match="paint_by_numbers.fit"):
        paints.load_paints(db)


def test_save_load_round_trip(tmp_path):
    db = tmp_path / "db.json"
    records = paints.load_records()
    paints.save_records(records, db)
    assert paints.load_records(db) == sorted(records, key=lambda r: r["name"].lower())


def test_slugify():
    assert paints.slugify("Phthalo Blue (Green Shade)") == "phthalo-blue-green-shade"
    assert paints.slugify("Cadmium Yellow Primrose") == "cadmium-yellow-primrose"


# ---------------------------------------------------------------- fetch script parser

PAGE = """<html><head><title>Heavy Body Acrylic Color - Test Red | Golden Artist Colors</title></head>
<body><h1 class="title"> Test   Red </h1>
<div>Pigment / Technical Data</div><table><tr><th>CI Name (Number)</th><th>Classification</th></tr>
<tr><td>PR254 (56110)</td><td>Synthetic Organic</td></tr>
<tr><td>PB15:3 (74160)</td><td>Synthetic Organic</td></tr></table>
<dl><dt>Transparency</dt><dd>Semi-Opaque</dd><dt>Lightfastness</dt><dd>Excellent</dd>
<dt>Sheen</dt><dd>Gloss</dd></dl>
<dl><dt>srgb</dt><dd><span>200</span>, <span>40</span>, <span>38</span></dd>
<dt>lab</dt><dd><span>43.54</span>, <span>54.93</span>, <span>-3.2</span></dd></dl>
<table><tr><th>Size</th><th>Item</th><th>UPC</th></tr>
<tr><td>2 oz tube</td><td>0001277-2</td><td>738797127722</td></tr></table>
<img src="https://images.goldenartistcolors.com/i/abc/productGalleryLarge.v1/1277-test-red-tint-glaze-400.webp">
</body></html>"""


def _fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_golden", ROOT / "scripts" / "fetch_golden.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetch_script_parses_product_page():
    record = _fetch_module().parse_product_page(PAGE, "https://example.test/x")
    assert record["name"] == "Test Red"
    assert record["item"] == 1277
    assert record["lab"] == [43.54, 54.93, -3.2]
    assert record["srgb"] == [200, 40, 38]
    assert record["transparency"] == "semi-opaque"
    assert record["sheen"] == "gloss"
    assert record["pigments"] == ["PR254", "PB15:3"]
    assert record["K"] is None and record["S"] is None
    assert record["tint_glaze_image"].endswith("1277-test-red-tint-glaze-400.webp")


def test_fetch_script_rejects_page_without_color_data():
    with pytest.raises(ValueError, match=r"no L\*a\*b\* published"):
        _fetch_module().parse_product_page("<h1>Nothing</h1>", "https://example.test/x")


def test_default_palette_leaves_out_discontinued_and_special():
    default = paints.default_palette(PAINTS)
    assert all(not p.discontinued and p.special is None for p in default)
    assert len(default) == len([p for p in PAINTS if not p.discontinued and p.special is None])
    assert "Titanium White" in {p.name for p in default}
