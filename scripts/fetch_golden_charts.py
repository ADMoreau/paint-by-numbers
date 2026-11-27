"""Add Golden acrylic paint lines to the database from their Pigment Detail Charts.

Build time only: the planner never touches the network. Golden publishes one chart per line at
goldenartistcolors.com/products/golden-artist-acrylics/<line>/pigment-data, listing every color's
item number, pigments, opacity rating, lightfastness, Munsell notation, gloss and CIE L*a*b*:
everything python -m paint_by_numbers.fit needs (L*a*b* and opacity) in one request per line.

    python scripts/fetch_golden_charts.py                    # every line below
    python scripts/fetch_golden_charts.py fluid high-flow    # some lines
    python scripts/fetch_golden_charts.py --cache pages/     # save/reuse the HTML

Each line goes to its own database, paint_by_numbers/data/paints_<line>.json (Heavy Body keeps
its per-product-page database, paints_heavy_body.json, and its chart is only compared against
it). Records have the same fields as the Heavy Body ones, plus munsell, gloss_average and
tint_strength from the chart. sRGB is computed from L*a*b* (clipped). Sheen is left empty: the
chart has a gloss reading instead of Golden's sheen name. K and S are null: run the fit on each database next, e.g.

    python -m paint_by_numbers.fit --db paint_by_numbers/data/paints_fluid.json
"""

from __future__ import annotations

import argparse
import datetime
import html
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paint_by_numbers import color  # noqa: E402
from paint_by_numbers.paints import DEFAULT_DB, TRANSPARENCY_RATINGS, load_records, save_records  # noqa: E402

CHART = "https://goldenartistcolors.com/products/golden-artist-acrylics/{line}/pigment-data"
SITE = "https://goldenartistcolors.com"
USER_AGENT = "Mozilla/5.0 (compatible; paint-by-numbers-fetch/0.1)"
DELAY_S = 1.0
DATA = ROOT / "paint_by_numbers" / "data"

# chart path -> (line name in the records, database file)
LINES = {
    "heavy-body": ("Heavy Body", DEFAULT_DB),
    "fluid": ("Fluid", DATA / "paints_fluid.json"),
    "high-flow": ("High Flow", DATA / "paints_high_flow.json"),
    "open": ("OPEN", DATA / "paints_open.json"),
    "soflat": ("SoFlat", DATA / "paints_soflat.json"),
}
# Not used: the Mineral Spirit Acrylic chart (mineral-spirit-acrylic-color) has no L*a*b* for half
# its colors, numbers instead of opacity ratings, and values that don't match the names (Pyrrole
# Red at L*41 a*-14 b*-32, a blue), as checked 2026-11-26.


SPECIAL = ("fluorescent", "phosphorescent", "iridescent", "interference")


def fetch(url: str, hops: int = 3) -> str:
    """The page at ``url``, following the site's meta-refresh redirects (some charts moved)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    m = re.search(r'<meta http-equiv="refresh" content="\d+;url=([^"]+)"', raw)
    if m and hops > 0 and "<table" not in raw:
        time.sleep(DELAY_S)
        return fetch(urllib.parse.urljoin(url, html.unescape(m.group(1))), hops - 1)
    return raw


def _text(cell: str) -> str:
    """A cell's visible text."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<!--.*?-->|<[^>]+>", " ", cell, flags=re.S))).strip()


def parse_chart(raw: str) -> list[dict]:
    """The chart's rows as {column header: cell HTML}."""
    table = re.search(r"<table.*?</table>", raw, flags=re.S)
    if not table:
        raise ValueError("no table on the page")
    heads = [_text(h) for h in re.findall(r"<th[^>]*>(.*?)</th>", table.group(0), flags=re.S)]
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", table.group(0), flags=re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)
        if len(cells) == len(heads):
            rows.append(dict(zip(heads, cells)))
    return rows


def _number(text: str) -> float | None:
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", text.strip())
    return float(text) if m and float(text) != 0 else None  # the chart writes 0 for "not measured"


def lab_to_srgb255(lab: list[float]) -> list[int]:
    srgb = color.lab_to_srgb(torch.tensor(lab, dtype=torch.float64)).clamp(0, 1)
    return [int(round(v * 255)) for v in srgb.tolist()]


class Skip(ValueError):
    """A row without the color data the planner needs."""


def record(row: dict, line: str, chart_url: str) -> dict:
    name_cell = row["Color Name (ItemNumber)"]
    link = re.search(r'href="([^"]+)"', name_cell)
    # item numbers are plain (2422) or, for Mineral Spirit Acrylic, coded (X-0135)
    m = re.fullmatch(r"(.*?)\s*\(\s*(([A-Z]+-)?(\d+))\s*\)", _text(name_cell))
    if not m:
        raise ValueError(f"unexpected name cell: {_text(name_cell)!r}")
    name, item, item_code = m.group(1).strip(), int(m.group(4)), m.group(2)
    discontinued = bool(re.search(r"\(discontinued\)", name, flags=re.I))
    name = re.sub(r"\s*\(discontinued\)", "", name, flags=re.I).strip()
    lab_text = _text(row.get("CIE L*a*b* Values", ""))
    lm = re.search(r"L\*\s*(-?[\d.]+)\s*a\*\s*(-?[\d.]+)\s*b\*\s*(-?[\d.]+)", lab_text)
    if not lm:
        raise Skip(f"{name}: no L*a*b*")
    lab = [float(v) for v in lm.groups()]
    if all(v == 0 for v in lab):
        raise Skip(f"{name}: L*a*b* all zero")
    transparency = _text(row.get("Opacity / Transparency", "")).lower().replace(" ", "-")
    if transparency not in TRANSPARENCY_RATINGS:
        raise Skip(f"{name}: opacity {transparency or 'missing'!r}")
    pigments = re.findall(r"\b(P(?:Bk|Br|[A-Z])\d+(?::\d+)?)", _text(row.get("Pigment (CI Name)", "")))
    gloss = _number(_text(row.get("Gloss Average", "")))
    lightfastness = _text(row.get("Lightfastness rating", "")) or None
    lower = name.lower()
    return {
        "name": name,
        "item": item,
        **({"item_code": item_code} if item_code != str(item) else {}),
        "line": line,
        "lab": lab,
        "srgb": lab_to_srgb255(lab),
        "transparency": transparency,
        "sheen": None,  # the chart gives a gloss reading (gloss_average), not Golden's sheen name
        "lightfastness": None if lightfastness in (None, "N/A") else lightfastness,
        "pigments": pigments,
        "discontinued": discontinued,
        "special": next((k for k in SPECIAL if k in lower), None),
        "K": None,
        "S": None,
        "munsell": _text(row.get("Munsell Notation Listing", "")) or None,
        "gloss_average": gloss,
        "tint_strength": _number(_text(row.get("Tint Strength", ""))),
        "source": chart_url,
        "product_page": SITE + link.group(1) if link else None,
        "retrieved": datetime.date.today().isoformat(),
        "tint_glaze_image": None,
    }


def check_heavy_body(records: list[dict]) -> None:
    """Compare the Heavy Body chart with the per-page database: L*a*b* and opacity."""
    pages = {r["item"]: r for r in load_records(DEFAULT_DB)}
    lab_diff, opacity, missing = [], [], []
    for r in records:
        p = pages.get(r["item"])
        if p is None:
            missing.append(r["name"])
            continue
        lab_diff.append(max(abs(a - b) for a, b in zip(r["lab"], p["lab"])))
        if r["transparency"] != p["transparency"]:
            opacity.append(f"{r['name']}: chart {r['transparency']}, page {p['transparency']}")
    print(f"heavy body check: {len(lab_diff)} colors on both; largest L*a*b* difference {max(lab_diff):.2f}")
    print(f"  opacity differs on {len(opacity)}" + "".join(f"\n    {o}" for o in opacity))
    if missing:
        print(f"  on the chart but not in the page database: {', '.join(missing)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lines", nargs="*", choices=[[], *LINES], help="chart paths (default: all)")
    ap.add_argument("--cache", type=Path, help="directory to save fetched HTML and reuse it")
    args = ap.parse_args(argv)
    fetched = False
    for path in args.lines or list(LINES):
        line, db = LINES[path]
        url = CHART.format(line=path)
        cached = args.cache / f"{path}.html" if args.cache else None
        if cached and cached.exists():
            raw = cached.read_text(encoding="utf-8")
        else:
            if fetched:
                time.sleep(DELAY_S)
            fetched = True
            raw = fetch(url)
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(raw, encoding="utf-8")
        try:
            rows = parse_chart(raw)
        except ValueError as e:
            print(f"{line}: can't read the chart ({e}); the page may now be drawn in the browser")
            if cached:
                cached.unlink(missing_ok=True)
            continue
        records, skipped = {}, []
        for row in rows:
            try:
                r = record(row, line, url)
            except Skip as e:
                skipped.append(str(e))
                continue
            records[r["item"]] = r
        print(f"{line}: {len(records)} colors" + (f", skipped {len(skipped)}: {'; '.join(skipped)}" if skipped else ""))
        if path == "heavy-body":
            check_heavy_body(list(records.values()))
            continue
        old = {r["item"]: r for r in load_records(db)} if db.exists() else {}
        for item, r in records.items():  # keep a previous fit if the color data is unchanged
            o = old.get(item)
            if o and o.get("K") and o["lab"] == r["lab"] and o["transparency"] == r["transparency"]:
                r["K"], r["S"] = o["K"], o["S"]
            for key in ("tint_glaze_image", "tint_glaze_photo"):  # from scripts/fetch_tint_glaze.py
                if o and o.get(key):
                    r[key] = o[key]
        save_records(list(records.values()), db)
        print(f"  wrote {db.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
