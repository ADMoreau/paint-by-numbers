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
RECORDS = {r["name"]: r for r in paints.load_records()}
# Paints whose S comes from the transparency rating alone (no usable tint and glaze photo).
RATED = [p for p in PAINTS if not fit._photo_usable(RECORDS[p.name])]


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
    """K and S come from fit.py, not by hand, and are current with COATS_TO_HIDE and the photos."""
    records = paints.load_records()
    for stored, fresh in zip(records, fit.fit_records(records)):
        if stored["K"] is None:
            assert fresh["K"] is None, stored["name"]
            continue
        # the photo refit is an optimizer run, so allow for floating-point differences between machines
        assert stored["K"] == pytest.approx(fresh["K"], rel=1e-4), stored["name"]
        assert stored["S"] == pytest.approx(fresh["S"], rel=1e-4), stored["name"]


# ---------------------------------------------------------------- estimate_ks


def test_estimate_ks_mid_gray():
    K, S = fit.estimate_ks([50.0, 0.0, 0.0], "opaque")
    r = 0.18418652  # linear reflectance at L* = 50
    assert torch.all(S == S[0])  # same S on every channel
    assert torch.allclose(K / S, torch.full((3,), (1 - r) ** 2 / (2 * r), dtype=torch.float64), rtol=1e-6)


@pytest.mark.parametrize("paint", RATED, ids=lambda p: p.slug)
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
    white = fit.estimate_ks(RECORDS["Titanium White"]["lab"], "opaque")
    black = fit.estimate_ks(RECORDS["Carbon Black"]["lab"], "opaque")
    assert fit.hiding_error(*white, 1.0) == pytest.approx(fit.hiding_error(*black, 1.0), abs=1e-3)


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


# ---------------------------------------------------------------- tint and glaze photos


def _photo_module():
    spec = importlib.util.spec_from_file_location("fetch_tint_glaze", ROOT / "scripts" / "fetch_tint_glaze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _swatch_photo(background, bands):
    """A 400 px photo in Golden's layout: ``bands`` maps band -> (color over white, color over the stripe)."""
    import numpy as np
    from PIL import Image

    mod = _photo_module()
    a = np.full((400, 400, 3), background, dtype=np.uint8)
    a[:, 255:316] = (55, 60, 53)  # the black stripe
    for band, (y0, y1) in mod.BANDS.items():
        if band in bands:
            over_white, over_black = bands[band]
            a[y0 - 6 : y1 + 6, 125:] = over_white
            a[y0 - 6 : y1 + 6, 255:316] = over_black
    return Image.fromarray(a)


def test_photo_measure_reads_each_band():
    mod = _photo_module()
    bands = {
        "glaze_1_10": ((200, 150, 230), (60, 50, 70)),
        "glaze_1_1": ((120, 60, 170), (50, 40, 60)),
        "masstone": ((40, 30, 50), (35, 30, 40)),
        "tint_1_1": ((140, 110, 200), (140, 110, 200)),
        "tint_1_10": ((210, 200, 240), (210, 200, 240)),
    }
    m = mod.measure(_swatch_photo((255, 255, 255), bands))
    assert m["ground"] is None  # cut-out white: the chart itself
    assert m["stripe"] == [55, 60, 53]
    for band in ("glaze_1_10", "glaze_1_1", "masstone"):
        assert m[band] == {"white": list(bands[band][0]), "black": list(bands[band][1])}
    assert m["tint_1_1"] == [140, 110, 200] and m["tint_1_10"] == [210, 200, 240]


def test_photo_measure_whites_on_gray_card():
    """Whites are shot on a gray card: it is the ground, and their tint bands are empty."""
    mod = _photo_module()
    m = mod.measure(_swatch_photo((95, 100, 105), {"masstone": ((240, 240, 240), (230, 230, 230))}))
    assert m["ground"] == [95, 100, 105]
    assert m["tint_1_1"] is None and m["tint_1_10"] is None


def test_photo_measure_rejects_other_layouts():
    from PIL import Image

    mod = _photo_module()
    with pytest.raises(mod.Unreadable):
        mod.measure(Image.new("RGB", (400, 400), "white"))


def test_refine_s_recovers_s_from_a_rendered_photo():
    """Render a photo from known K and S; refitting S from the rating's starting point finds them again."""
    ks = (1 - fit.published_reflectance([30.0, 20.0, -25.0])) ** 2 / (2 * fit.published_reflectance([30.0, 20.0, -25.0]))
    s_true = torch.tensor([3.0, 6.0, 0.8], dtype=torch.float64)
    K = ks * s_true
    white = fit.estimate_ks([96.0, 0.0, 0.0], "opaque")
    to255 = lambda R: (color.lab_to_srgb(color.working_to_lab(R)) * 255).tolist()
    ground = color.lab_to_working(color.srgb_to_lab(torch.tensor(fit.PHOTO_CHART_WHITE, dtype=torch.float64) / 255))
    stripe = color.lab_to_working(color.srgb_to_lab(torch.tensor([55.0, 60.0, 53.0], dtype=torch.float64) / 255))
    photo = {"ground": None, "stripe": [55.0, 60.0, 53.0]}
    for band, t in fit.PHOTO_COATS.items():
        photo[band] = {side: to255(km.layer(under, K, s_true, torch.tensor(t, dtype=torch.float64)))
                       for side, under in (("white", ground), ("black", stripe))}
    for band, c in fit.PHOTO_TINTS.items():
        photo[band] = to255(km.layer(ground, c * K + (1 - c) * white[0], c * s_true + (1 - c) * white[1], torch.tensor(1.0, dtype=torch.float64)))
    _, s0 = fit.estimate_ks([30.0, 20.0, -25.0], "transparent")
    S = fit.refine_s(ks[None], s0[None], [photo], white)[0]
    assert torch.allclose(S, s_true, rtol=0.02)
    assert fit.photo_error(ks * S, S, photo, white).max() < 0.1


def test_photo_fit_matches_photos_better_than_the_rating():
    """For every paint with a photo, the stored fit renders the photo's bands closer than the rating's S does."""
    white = (RECORDS["Titanium White"]["K"], RECORDS["Titanium White"]["S"])
    stored, rated = [], []
    for r in RECORDS.values():
        if not fit._photo_usable(r):
            continue
        photo = r["tint_glaze_photo"]
        e_fit = fit.photo_error(r["K"], r["S"], photo, white)
        e_rating = fit.photo_error(*fit.estimate_ks(r["lab"], r["transparency"]), photo, white)
        assert (e_fit**2).sum() <= (e_rating**2).sum() + 1e-6, r["name"]
        stored.append(e_fit.mean().item())
        rated.append(e_rating.mean().item())
    assert len(stored) > 100
    assert torch.tensor(stored).median() < 0.7 * torch.tensor(rated).median()


def test_dioxazine_purple_glazes_violet():
    """The bug the photos fixed: a thin glaze of Dioxazine Purple over white was nearly white,
    so plans used it as a gentle gray and real paint turned everything purple."""
    p = next(p for p in PAINTS if p.name == "Dioxazine Purple")
    lab = color.working_to_lab(km.layer(torch.ones(3, dtype=torch.float64), torch.tensor(p.K), torch.tensor(p.S), torch.tensor(0.125)))
    L, a, b = lab.tolist()
    assert L < 60 and a > 15 and b < -30  # a strong violet, as on Golden's swatch
