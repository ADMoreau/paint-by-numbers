"""Add Liquitex Heavy Body colors to the database from the Liquitex Japan catalog's Munsell data.

Build time only. Liquitex doesn't publish L*a*b* values, but Liquitex Japan's catalog lists each
color's Munsell notation (re-measured in Japan, March 2026), pigments and opacity, for its Regular
Type (heavy body) and Soft Type colors. This script reads that table, converts the Munsell
notation to L*a*b* (colour-science's Munsell renotation), and gives each color its US English
name, matched by translating the Japanese name (see NAMES).

    python scripts/fetch_liquitex_japan.py --cache pages/
    python -m paint_by_numbers.fit --db paint_by_numbers/data/paints_liquitex.json

Needs pdftotext (poppler-utils) and colour-science (pip install colour-science).

How good is it? Munsell notation to L*a*b* loses accuracy: run on Golden's charts, which give
both, the conversion lands a median 3.4 ΔE (90th percentile about 10) from Golden's measured
L*a*b*. Treat these colors as less certain than Golden's. Other details:

- Opacity is three levels in the catalog. 不透明 (opaque) -> opaque, 半透明 (semi-transparent)
  -> semi-opaque, 透明 (transparent) -> transparent. Compared with Golden's ratings for paints of
  the same single pigment, Liquitex's opaque spans Golden's opaque and semi-opaque, and its
  semi-transparent colors sit at Golden's semi-opaque.
- Titanium White has no Munsell value in the catalog. Its value comes from Liquitex's US
  technical sheet (W/B 9.6, a neutral at value 9.6).
- The Cadmium-Free colors' notations are round numbers (5.0R 3.90/10.00), probably nominal.
- Fluorescent, iridescent and metallic colors have no Munsell value and are skipped.
- Measured differently from Golden. For the 23 colors with the same name and pigments as a
  Golden Heavy Body color, Liquitex's values are darker and more saturated, most for dark and
  transparent colors (Mars Black L* 16.5 against Golden's 25), as when a measurement leaves out
  the light reflected off the paint's surface. Adding SURFACE_REFLECTION of white to every
  color best matches Golden's convention, which the paint model is fitted to: it cuts the median
  difference from 8.1 to 5.3 ΔE. The remaining gap is the Munsell conversion plus real
  differences between the paints. "lab" has the correction; "lab_from_munsell" is without it.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
import urllib.request
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from paint_by_numbers import color  # noqa: E402
from paint_by_numbers.paints import load_records, save_records  # noqa: E402

CATALOG = "https://www.jp.liquitex.com/wp-content/uploads/2026/03/LQ-Catalog-202603_ALL.pdf"
US_SHEET = "https://coolcolors.lbl.gov/LBNL-Pigment-Database/assets/manufacturer-TDS/Liquitex/PaintTechInfo.pdf"
US_PAGE = "https://www.liquitex.com/products/professional-heavy-body-acrylic-{slug}"
USER_AGENT = "Mozilla/5.0 (compatible; paint-by-numbers-fetch/0.1)"
DB = ROOT / "paint_by_numbers" / "data" / "paints_liquitex.json"
LINE = "Liquitex Heavy Body"

OPACITY = {"不透明": "opaque", "半透明": "semi-opaque", "透明": "transparent"}

SURFACE_REFLECTION = 0.02  # share of white added in XYZ: Golden's measurement convention (see above)

# Japanese color number -> (English name, US Heavy Body product page slug or None, match).
# "exact": the US line has a color of that name (or the catalog says it was renamed, e.g. Acra
# Magenta -> Quinacridone Magenta). "probable": the names differ and the match rests on pigments
# and the older US sheet (noted). None: no US Heavy Body color by that name; kept under the
# translated name.
NAMES: dict[str, tuple[str, str | None, str]] = {
    "0001": ("Light Portrait Pink", None, "none"),
    "0003": ("Medium Magenta", "medium-magenta", "exact"),
    "0004": ("Quinacridone Magenta", "quinacridone-magenta", "exact"),
    "0006": ("Deep Magenta", "deep-magenta", "exact"),
    "0007": ("Quinacridone Red", "quinacridone-red", "exact"),
    "0008": ("Naphthol Red Light", "naphthol-red-light", "exact"),
    "0009": ("Cadmium Red Medium", "cadmium-red-medium", "exact"),
    "0010": ("Naphthol Crimson", "naphthol-crimson", "exact"),
    "0012": ("Cadmium Red Light", "cadmium-red-light", "exact"),
    "0013": ("Scarlet Red", None, "none"),
    "0014": ("Yellow Orange Azo", "yellow-orange-azo", "exact"),
    "0016": ("Cadmium Orange", "cadmium-orange", "exact"),
    "0017": ("Vivid Red Orange", "vivid-red-orange", "exact"),
    "0019": ("Cadmium Yellow Medium", "cadmium-yellow-medium", "exact"),
    "0020": ("Yellow Medium Azo", "yellow-medium-azo", "exact"),
    "0021": ("Yellow Light Hansa", "yellow-light-hansa", "exact"),
    "0022": ("Cadmium Yellow Light", "cadmium-yellow-light", "exact"),
    "0023": ("Brilliant Yellow Green", "brilliant-yellow-green", "exact"),
    "0024": ("Vivid Lime Green", "vivid-lime-green", "exact"),
    "0025": ("Light Green Oxide", None, "none"),
    "0026": ("Light Emerald Green", "light-emerald-green", "exact"),
    "0027": ("Light Green Permanent", "light-green-permanent", "exact"),  # パーマネント グリーン ライト
    "0028": ("Emerald Green", "emerald-green", "exact"),
    "0029": ("Green Deep Permanent", "green-deep-permanent", "exact"),  # パーマネント グリーン ディープ
    "0030": ("Phthalocyanine Green (Blue Shade)", "phthalocyanine-green-blue-shade", "exact"),  # PG7; PG36 is 0134
    "0031": ("Chromium Oxide Green", "chromium-oxide-green", "exact"),
    "0032": ("Hooker's Green Hue Permanent", "hooker-s-green-hue-permanent", "exact"),
    "0033": ("Bright Aqua Green", "bright-aqua-green", "exact"),
    "0035": ("Light Blue Permanent", "light-blue-permanent", "exact"),  # パーマネント ライト ブルー
    "0036": ("Brilliant Blue", "brilliant-blue", "exact"),
    "0037": ("Cerulean Blue", "cerulean-blue", "exact"),
    "0038": ("Cerulean Blue Hue", "cerulean-blue-hue", "exact"),
    "0039": ("Light Blue Violet", "light-blue-violet", "exact"),
    "0040": ("Cobalt Blue", "cobalt-blue", "exact"),
    # ブリリアント ブルー パープル: the US sheet lists "Cobalt Blue Hue/Brilliant Blue Purple" (PB29, PW6)
    "0041": ("Cobalt Blue Hue", "cobalt-blue-hue", "probable"),
    "0042": ("Phthalocyanine Blue (Green Shade)", "phthalocyanine-blue-green-shade", "exact"),  # red shade is 0157
    # two ultramarines in each line; French Ultramarine is the traditional red shade
    "0043": ("Ultramarine Blue (Green Shade)", "ultramarine-blue-green-shade", "probable"),
    "0044": ("Prism Violet", "prism-violet", "exact"),
    "0045": ("Dioxazine Purple", "dioxazine-purple", "exact"),
    "0046": ("Brilliant Purple", "brilliant-purple", "exact"),
    "0047": ("Permanent Light Violet", None, "none"),
    "0048": ("Raw Sienna", "raw-sienna", "exact"),
    "0049": ("Red Oxide", "red-oxide", "exact"),
    "0050": ("Burnt Sienna", "burnt-sienna", "exact"),
    "0051": ("Liquitex Yellow", None, "none"),
    "0052": ("Yellow Oxide", "yellow-oxide", "exact"),
    "0053": ("Bronze Yellow", "bronze-yellow", "exact"),
    "0054": ("Raw Umber", "raw-umber", "exact"),
    "0055": ("Burnt Umber", "burnt-umber", "exact"),
    "0056": ("Ivory Black", "ivory-black", "exact"),
    "0057": ("Mars Black", "mars-black", "exact"),
    "0058": ("Payne's Gray", "payne-s-gray", "exact"),
    "0060": ("Neutral Gray 4", None, "none"),
    "0061": ("Neutral Gray 5", "neutral-gray-5", "exact"),
    "0063": ("Neutral Gray 7", None, "none"),
    "0065": ("Titanium White", "titanium-white", "exact"),
    "0072": ("Alizarin Crimson Hue Permanent", "alizarin-crimson-hue-permanent", "exact"),
    "0074": ("Unbleached Titanium", "unbleached-titanium", "exact"),
    "0075": ("Parchment", "parchment", "exact"),
    "0077": ("Sap Green Permanent", "sap-green-permanent", "exact"),
    "0078": ("Viridian Hue Permanent", "viridian-hue-permanent", "exact"),
    "0083": ("Cobalt Green", "cobalt-green", "exact"),
    "0084": ("Turquoise Deep", "turquoise-deep", "exact"),
    "0086": ("Naples Yellow Hue", "naples-yellow-hue", "exact"),
    "0125": ("Deep Violet", "deep-violet", "exact"),
    "0126": ("Hooker's Green Deep Hue Permanent", "hooker-s-green-deep-hue-permanent", "exact"),
    "0127": ("Manganese Blue Hue", "manganese-blue-hue", "exact"),
    "0128": ("Van Dyke Red", "van-dyke-red", "exact"),  # バンダイク レッド ヒュー
    "0129": ("Quinacridone Crimson", "quinacridone-crimson", "exact"),
    "0130": ("Quinacridone Red Orange", "quinacridone-red-orange", "exact"),
    "0134": ("Phthalocyanine Green (Yellow Shade)", "phthalocyanine-green-yellow-shade", "exact"),
    "0135": ("Cobalt Turquoise", "cobalt-turquoise", "exact"),
    "0137": ("Ultramarine Blue (Red Shade)", "ultramarine-blue-red-shade", "probable"),  # フレンチ ウルトラマリン ブルー
    # ジンク ホワイト: the US sheet lists "Transparent Mixing White/Zinc White" (PW4)
    "0140": ("Transparent Mixing White", "transparent-mixing-white", "probable"),
    "0150": ("Pyrrole Crimson", "pyrrole-crimson", "exact"),
    "0151": ("Pyrrole Red", "pyrrole-red", "exact"),
    "0152": ("Pyrrole Orange", "pyrrole-orange", "exact"),
    "0153": ("Indian Yellow", "indian-yellow", "exact"),
    "0154": ("Transparent Viridian Hue", "transparent-viridian-hue", "exact"),
    "0155": ("Green Gold", "green-gold", "exact"),
    "0156": ("Indanthrene Blue", "indantherene-blue", "exact"),
    "0157": ("Phthalocyanine Blue (Red Shade)", "phthalocyanine-blue-red-shade", "exact"),
    "0158": ("Transparent Raw Sienna", "transparent-raw-sienna", "exact"),
    "0159": ("Transparent Raw Umber", "transparent-raw-umber", "exact"),
    "0160": ("Transparent Burnt Sienna", "transparent-burnt-sienna", "exact"),
    "0161": ("Transparent Burnt Umber", "transparent-burnt-umber", "exact"),
    "0162": ("Prussian Blue Hue", "prussian-blue-hue", "exact"),
    "0171": ("Cadmium-Free Red Deep", "cadmium-free-red-deep", "exact"),
    "0172": ("Cadmium-Free Red Medium", "cadmium-free-red-medium", "exact"),
    "0173": ("Cadmium-Free Red Light", "cadmium-free-red-light", "exact"),
    "0174": ("Cadmium-Free Orange", "cadmium-free-orange", "exact"),
    "0175": ("Cadmium-Free Yellow Deep", "cadmium-free-yellow-deep", "exact"),
    "0176": ("Cadmium-Free Yellow Medium", "cadmium-free-yellow-medium", "exact"),
    "0177": ("Cadmium-Free Yellow Light", "cadmium-free-yellow-light", "exact"),
    "0181": ("Muted Pink", "muted-pink", "exact"),
    "0182": ("Muted Violet", "muted-violet", "exact"),
    "0183": ("Muted Green", "muted-green", "exact"),
    "0184": ("Muted Turquoise", "muted-turquoise", "exact"),
    "0185": ("Muted Grey", "muted-grey", "exact"),
    "0236": ("Light Bismuth Yellow", "light-bismuth-yellow", "probable"),  # ペール (pale) ビスマス イエロー, PY184 + PW6
    "0237": ("Light Phthalocyanine Green", "light-phthalocyanine-green", "exact"),
    "0238": ("Indigo", "indigo", "exact"),
    "0239": ("Transparent Orange", "transparent-orange", "exact"),
    "6005": ("Deep Brilliant Red", None, "none"),
    "6136": ("Cobalt Teal", "cobalt-teal", "exact"),
}

# Colors the catalog gives no Munsell value for, from Liquitex's US technical sheet.
US_SHEET_MUNSELL = {"0065": "N9.6"}  # Titanium White: "W/B 9.6"

# Opacity ratings missing from the catalog's color chart, assumed like their neighbors.
OPACITY_FALLBACK = {"0174": "不透明", "0175": "不透明"}  # Cadmium-Free Orange / Yellow Deep (the rest of the set is opaque)


def catalog_text(cache: Path | None) -> str:
    pdf = cache / "liquitex-japan-catalog-202603.pdf" if cache else Path("/tmp/liquitex-japan-catalog.pdf")
    if not pdf.exists():
        pdf.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(CATALOG, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            pdf.write_bytes(resp.read())
    return subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True, text=True).stdout


_ENTRY = re.compile(
    r"(?:^|\s)(\d{3,4})\s{2,}(\S(?:.*?\S)?)\s{2,}(-|\d+(?:\.\d+)?[A-Z]+)\s+(-|[\d.]+)\s+(-|[\d.]+)\s+(\S(?:.*?\S)?)"
    r"(?=\s{3,}\d{3,4}\s{2,}|\s*$)"
)


def parse_catalog(text: str) -> tuple[dict, dict]:
    """The Munsell table (number -> fields) and the color chart's opacity and series (number -> (rating, series))."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if "マンセルデータ／カラーインデックス" in line)
    end = next(i for i in range(start, len(lines)) if "LIQUITEX PRIME" in lines[i])
    rows = {}
    for line in lines[start:end]:
        for m in _ENTRY.finditer(line):
            num = m.group(1).zfill(4)
            pigments = re.split(r"\s{3,}", m.group(6))[0]  # drop text spilling over from the next column
            rows[num] = {"jp_name": re.sub(r"\s+", " ", m.group(2)), "hue": m.group(3), "value": m.group(4),
                         "chroma": m.group(5), "pigments": pigments}  # fmt: skip
    opacity = {m.group(1): (m.group(2), int(m.group(3))) for m in re.finditer(r"(\d{4}) (不透明|半透明|透明) G-(\d)", text)}
    return rows, opacity


def munsell_spec(hue: str, value: str, chroma: str) -> str:
    """'7.63R', '8.06', '5.54' -> '7.63R 8.06/5.54'. Near-neutral colors become N<value>; hue 0 becomes 10 of the previous family."""
    if float(chroma) < 0.3:
        return f"N{float(value):g}"
    m = re.fullmatch(r"([\d.]+)([A-Z]+)", hue)
    h, family = float(m.group(1)), m.group(2)
    if h == 0:
        order = ["R", "YR", "Y", "GY", "G", "BG", "B", "PB", "P", "RP"]
        h, family = 10.0, order[order.index(family) - 1]
    return f"{h:g}{family} {float(value):g}/{float(chroma):g}"


def munsell_to_lab(spec: str) -> list[float]:
    """Munsell notation -> CIE L*a*b* through the Munsell renotation (illuminant C), as the
    conversion that best reproduces Golden's published L*a*b* from its own Munsell notation."""
    import colour

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xyY = colour.munsell_colour_to_xyY(spec)
        illuminant_c = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["C"]
        lab = colour.XYZ_to_Lab(colour.xyY_to_XYZ(xyY), illuminant_c)
    return [round(float(v), 2) for v in lab]


def with_surface_reflection(lab: list[float], k: float = SURFACE_REFLECTION) -> list[float]:
    """``lab`` with share ``k`` of white light added (in XYZ), as reflected off the paint's surface."""
    xyz = color.lab_to_xyz(torch.tensor(lab, dtype=torch.float64))
    out = color.xyz_to_lab(xyz * (1 - k) + k * color._WHITE_XYZ.double())
    return [round(float(v), 2) for v in out]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, help="directory to save the catalog PDF and reuse it")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args(argv)

    rows, opacity = parse_catalog(catalog_text(args.cache))
    today = datetime.date.today().isoformat()
    old = {r["item"]: r for r in load_records(args.db)} if args.db.exists() else {}
    records, skipped = [], []
    for num, row in sorted(rows.items()):
        if row["hue"] == "-" and num not in US_SHEET_MUNSELL:
            skipped.append(f"{num} {row['jp_name']}: no Munsell value (metallic, fluorescent or iridescent)")
            continue
        if num not in NAMES:
            skipped.append(f"{num} {row['jp_name']}: no English name")
            continue
        name, slug, match = NAMES[num]
        if num in US_SHEET_MUNSELL:
            spec, source = US_SHEET_MUNSELL[num], US_SHEET
        else:
            spec, source = munsell_spec(row["hue"], row["value"], row["chroma"]), CATALOG
        try:
            lab_munsell = munsell_to_lab(spec)
        except Exception as e:  # outside the renotation data
            skipped.append(f"{num} {name}: {spec} can't be converted ({e})")
            continue
        lab = with_surface_reflection(lab_munsell)
        rating = opacity.get(num, (OPACITY_FALLBACK.get(num), None))
        if rating[0] is None:
            skipped.append(f"{num} {name}: no opacity rating")
            continue
        srgb = color.lab_to_srgb(torch.tensor(lab, dtype=torch.float64)).clamp(0, 1)
        record = {
            "name": name,
            "item": int(num),
            "line": LINE,
            "lab": lab,
            "srgb": [int(round(v * 255)) for v in srgb.tolist()],
            "transparency": OPACITY[rating[0]],
            "sheen": None,
            "lightfastness": None,
            "pigments": re.findall(r"\b(P(?:Bk|Br|[A-Z])\d+(?::\d+)?)", row["pigments"].replace(" ", "")),
            "discontinued": False,
            "special": None,
            "K": None,
            "S": None,
            "munsell": spec,
            "lab_from_munsell": lab_munsell,
            "series": rating[1],
            "jp_number": num,
            "jp_name": row["jp_name"],
            "us_match": match,
            "source": source,
            "product_page": US_PAGE.format(slug=slug) if slug else None,
            "retrieved": today,
            "tint_glaze_image": None,
        }
        o = old.get(record["item"])  # keep a previous fit if the color data is unchanged
        if o and o.get("K") and o["lab"] == lab and o["transparency"] == record["transparency"]:
            record["K"], record["S"] = o["K"], o["S"]
        records.append(record)
    save_records(records, args.db)
    counts = {k: sum(r["us_match"] == k for r in records) for k in ("exact", "probable", "none")}
    print(f"{len(records)} colors written to {args.db.relative_to(ROOT)}: {counts['exact']} matched to a US Heavy Body name, "
          f"{counts['probable']} probably matched, {counts['none']} with no US Heavy Body name")
    for s in skipped:
        print(f"skip   {s}")
    print(json.dumps(sorted({r["name"] for r in records if r["us_match"] != "exact"})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
