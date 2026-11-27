"""Golden's other acrylic lines (Fluid, High Flow, OPEN, SoFlat), from their Pigment Detail Charts."""

import importlib.util
from pathlib import Path

import pytest

from paint_by_numbers import fit, paints

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paint_by_numbers" / "data"
LINES = {"Fluid": "paints_fluid.json", "High Flow": "paints_high_flow.json", "OPEN": "paints_open.json", "SoFlat": "paints_soflat.json"}


def _charts():
    spec = importlib.util.spec_from_file_location("fetch_golden_charts", ROOT / "scripts" / "fetch_golden_charts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("line", LINES)
def test_line_records_are_valid_and_fitted(line):
    db = DATA / LINES[line]
    records = paints.load_records(db)
    assert len(records) > 40
    assert len({r["name"] for r in records}) == len(records)
    assert len({r["item"] for r in records}) == len(records)
    for r in records:
        assert r["line"] == line
        assert r["transparency"] in paints.TRANSPARENCY_RATINGS, r["name"]
        assert len(r["lab"]) == 3 and not all(v == 0 for v in r["lab"]), r["name"]
        assert "discontinued" not in r["name"].lower()
        assert r["source"].startswith("https://goldenartistcolors.com/products/golden-artist-acrylics/")
    # K and S come from fit.py, current with COATS_TO_HIDE, the photos and the other lines' photos
    for stored, fresh in zip(records, fit.fit_records(records, fit.photo_donors(db))):
        assert stored["K"] == pytest.approx(fresh["K"], rel=1e-4), stored["name"]
        assert stored["S"] == pytest.approx(fresh["S"], rel=1e-4), stored["name"]
    loaded = [p for p in paints.load_paints(db) if p.special is None]  # fluorescents lie outside the working gamut
    for p in loaded[::7]:
        assert fit.thick_layer_error(p, 1.0) < (0.5 if p.fit_source == "rating" else 2.5), p.name


CHART = """<html><table><thead><tr><th>Color</th><th>Color Name (ItemNumber)</th><th>Series</th><th>Pigment (CI Name)</th>
<th>Opacity / Transparency</th><th>Lightfastness rating</th><th>Munsell Notation Listing</th><th>Gloss Average</th>
<th>Viscosity Range</th><th>CIE L*a*b* Values</th><th>Tint Strength</th></tr></thead><tbody>
<tr><td><img src="x.webp"/></td><td><a class="underline" href="/products/fluid-acrylic-color-primary-yellow">Primary Yellow</a> <!-- -->(<!-- -->2422<!-- -->)</td>
<td>2</td><td>PY175, PY74, PW6</td><td><div><img alt="semi-transparent"/> </div>semi-transparent</td><td><div>Excellent</div></td>
<td><div>Hue:5 Y Value:8.5 Chroma:12.5</div></td><td><div>44.27</div></td><td><div>400-1000 CPS</div></td>
<td><div> L*84.52 a*4.79 b*88.39</div></td><td><div>95.17</div></td></tr>
<tr><td></td><td><a href="/products/x">Quinacridone Crimson (Discontinued)</a> ( X-0135 )</td><td>7</td><td>PR206</td><td>transparent</td>
<td>Excellent</td><td>Hue:0 Value:0 Chroma:0</td><td>0</td><td></td><td>L*28.5 a*30 b*10</td><td>0</td></tr>
<tr><td></td><td><a href="/products/y">Iridescent Gold (Fine)</a> (2450)</td><td>7</td><td>N/A</td><td>opaque</td>
<td>Excellent</td><td>Hue:0 Value:0 Chroma:0</td><td>0</td><td></td><td>L*0 a*0 b*0</td><td>0</td></tr>
</tbody></table></html>"""


def test_chart_parser():
    m = _charts()
    rows = m.parse_chart(CHART)
    assert len(rows) == 3
    r = m.record(rows[0], "Fluid", "https://example.test/chart")
    assert (r["name"], r["item"], r["line"]) == ("Primary Yellow", 2422, "Fluid")
    assert r["lab"] == [84.52, 4.79, 88.39]
    assert r["transparency"] == "semi-transparent"
    assert r["pigments"] == ["PY175", "PY74", "PW6"]
    assert r["gloss_average"] == 44.27 and r["tint_strength"] == 95.17
    assert r["product_page"] == "https://goldenartistcolors.com/products/fluid-acrylic-color-primary-yellow"
    assert r["K"] is None and r["sheen"] is None and not r["discontinued"]
    coded = m.record(rows[1], "Fluid", "https://example.test/chart")
    assert (coded["name"], coded["item"], coded["item_code"], coded["discontinued"]) == ("Quinacridone Crimson", 135, "X-0135", True)
    assert coded["gloss_average"] is None  # 0 means not measured
    with pytest.raises(m.Skip, match="all zero"):
        m.record(rows[2], "Fluid", "https://example.test/chart")


# ---------------------------------------------------------------- Liquitex (from the Japan catalog's Munsell data)

LIQUITEX = DATA / "paints_liquitex.json"


def _liquitex():
    spec = importlib.util.spec_from_file_location("fetch_liquitex_japan", ROOT / "scripts" / "fetch_liquitex_japan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_liquitex_records_are_valid_and_fitted():
    records = paints.load_records(LIQUITEX)
    assert len(records) > 90
    assert len({r["name"] for r in records}) == len(records)
    assert len({r["item"] for r in records}) == len(records)
    for r in records:
        assert r["line"] == "Liquitex Heavy Body"
        assert r["transparency"] in paints.TRANSPARENCY_RATINGS, r["name"]
        assert r["us_match"] in {"exact", "probable", "none"}
        assert (r["product_page"] is None) == (r["us_match"] == "none"), r["name"]
        assert r["jp_number"] and r["jp_name"] and r["munsell"]
    for stored, fresh in zip(records, fit.fit_records(records)):
        assert (stored["K"], stored["S"]) == (fresh["K"], fresh["S"]), stored["name"]
    by_name = {p.name: p for p in paints.load_paints(LIQUITEX)}
    for name in ("Titanium White", "Mars Black", "Phthalocyanine Blue (Green Shade)", "Cadmium Red Medium"):
        assert fit.thick_layer_error(by_name[name], 1.0) < 0.5, name


def test_liquitex_names_cover_the_catalog():
    """Every English name is unique, and every US product page slug is used once."""
    names = _liquitex().NAMES
    assert len({n for n, _, _ in names.values()}) == len(names)
    slugs = [s for _, s, _ in names.values() if s]
    assert len(set(slugs)) == len(slugs)


def test_munsell_spec():
    m = _liquitex()
    assert m.munsell_spec("7.63R", "8.06", "5.54") == "7.63R 8.06/5.54"
    assert m.munsell_spec("0.00YR", "2.05", "5.72") == "10R 2.05/5.72"  # hue 0 is 10 of the previous family
    assert m.munsell_spec("0.00R", "3", "4") == "10RP 3/4"
    assert m.munsell_spec("6.11Y", "9.60", "0.13") == "N9.6"  # near-neutral


def test_munsell_to_lab_and_surface_reflection():
    pytest.importorskip("colour")
    m = _liquitex()
    gray = m.munsell_to_lab("N5")
    assert gray[0] == pytest.approx(51, abs=1.5) and abs(gray[1]) < 1 and abs(gray[2]) < 1
    black = m.with_surface_reflection([16.5, 0.0, 0.0])
    assert 23 < black[0] < 25  # a surface reflection lifts a matte-measured black toward Golden's
    assert m.with_surface_reflection([100.0, 0.0, 0.0])[0] == pytest.approx(100.0, abs=0.01)
