"""Measure Golden's tint and glaze swatch photos into a paint database.

Build time only: the planner never touches the network. Each Golden acrylic product page has a photo
(its URL is stored as ``tint_glaze_image``; for records without one, the product page named in
``product_page`` is fetched to find it) of five brushed bands over a white chart with a black
stripe down it, top to bottom:

    glaze 1:10   1 part paint to 10 parts medium, one coat
    glaze 1:1    1 part paint to 1 part medium, one coat
    masstone     paint straight from the tube, one coat
    tint 1:1     1 part paint to 1 part Titanium White
    tint 1:10    1 part paint to 10 parts Titanium White

Golden's published L*a*b* is the masstone at full hiding, which for dark, strong, transparent paints
(Dioxazine Purple, the phthalos, quinacridones) is close to black and says nothing about how strong
or what hue a thin layer is. The photo shows both. This script stores each band's median color
(sRGB 0-255, over the white chart and over the black stripe) as ``tint_glaze_photo``; then
python -m paint_by_numbers.fit uses them to set each paint's S per channel.

    python scripts/fetch_tint_glaze.py                  # Heavy Body: every paint with a photo
    python scripts/fetch_tint_glaze.py --cache photos/  # save/reuse the downloaded photos
    python scripts/fetch_tint_glaze.py --db paint_by_numbers/data/paints_fluid.json --cache photos/fluid/

The photos are Golden's, so only the measured numbers go in the database. Whites are shot on a gray
card instead of the white chart; the card's color is stored as ``ground``, and their tint bands are
empty (white mixed with white), stored as null.
"""

from __future__ import annotations

import argparse
import datetime
import io
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paint_by_numbers.paints import DEFAULT_DB, load_records, save_records  # noqa: E402

USER_AGENT = "Mozilla/5.0 (compatible; paint-by-numbers-fetch/0.1)"
DELAY_S = 1.0
SIZE = 400  # the photos are 400x400; others are resized to it

# Rows sampled in each band (well inside the brushed edges) on the 400 px photo.
BANDS = {"glaze_1_10": (30, 52), "glaze_1_1": (95, 120), "masstone": (172, 210), "tint_1_1": (258, 285), "tint_1_10": (340, 365)}
TINTS = ("tint_1_1", "tint_1_10")
SWATCH_LEFT = 125  # bands start right of the labels
WHITE_SPAN, STRIPE_MARGIN = 110, 10  # columns sampled left of the stripe; inset from its edges
STRIPE_WIDTH = (45, 95)  # plausible stripe widths, px
CUT_OUT = 250  # a background this light everywhere is the cut-out white around a colored paint's chart


class Unreadable(ValueError):
    """A photo that doesn't have the expected layout."""


def _median(pixels: np.ndarray) -> list[float]:
    return [round(float(v), 1) for v in np.median(pixels.reshape(-1, 3), axis=0)]


def find_stripe(a: np.ndarray) -> tuple[int, int]:
    """Left and right columns of the black stripe, from the top and bottom rows where only it shows."""
    edge = np.concatenate([a[0:6], a[SIZE - 6 : SIZE]]).mean(axis=(0, 2))
    dark = np.where(edge[150:380] < min(110.0, 0.6 * float(np.median(edge))))[0] + 150
    if len(dark) == 0 or not STRIPE_WIDTH[0] <= dark.max() - dark.min() <= STRIPE_WIDTH[1]:
        raise Unreadable("no black stripe found")
    return int(dark.min()), int(dark.max())


def measure(image: Image.Image) -> dict:
    """Median band colors from a tint and glaze photo."""
    a = np.asarray(image.convert("RGB").resize((SIZE, SIZE)), dtype=np.float64)
    left, right = find_stripe(a)
    white_cols = slice(max(left - WHITE_SPAN, SWATCH_LEFT), left - 12)
    black_cols = slice(left + STRIPE_MARGIN, right - STRIPE_MARGIN + 1)
    background = _median(a[0:6, SWATCH_LEFT : left - 15])
    cut_out = min(background) > CUT_OUT
    out = {
        "ground": None if cut_out else background,  # None: the white chart (not visible in the photo)
        "stripe": _median(np.concatenate([a[0:6, left + 6 : right - 5], a[SIZE - 6 : SIZE, left + 6 : right - 5]])),
    }
    for band, (y0, y1) in BANDS.items():
        if band in TINTS:
            out[band] = _median(a[y0:y1, white_cols]) if cut_out else None
        else:
            out[band] = {"white": _median(a[y0:y1, white_cols]), "black": _median(a[y0:y1, black_cols])}
    return out


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


PHOTO_ON_PAGE = re.compile(r'https://images\.goldenartistcolors\.com/[^"\s]+?tint-glaze[^"\s/]*\.webp')


def photo_url(page_html: str) -> str | None:
    """The tint and glaze photo's URL on a product page."""
    m = PHOTO_ON_PAGE.search(page_html)
    return m.group(0) if m else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--cache", type=Path, help="directory to save downloaded photos and reuse them")
    args = ap.parse_args(argv)
    records = load_records(args.db)
    fetched, measured, failed = False, 0, []
    for r in records:
        url = r.get("tint_glaze_image")
        source = r.get("source") or ""
        page = r.get("product_page") or (source if "/products/" in source and "pigment-data" not in source else None)
        if not url and page and not r.get("special"):
            if fetched:
                time.sleep(DELAY_S)
            fetched = True
            try:
                url = photo_url(fetch(page).decode("utf-8", errors="replace"))
            except OSError as e:
                failed.append(f"{r['name']}: product page: {e}")
                continue
            if url:
                r["tint_glaze_image"] = url
        if not url:
            continue
        cached = args.cache / f"{r['item']}{Path(url).suffix}" if args.cache else None
        if cached and cached.exists():
            raw = cached.read_bytes()
        else:
            if fetched:
                time.sleep(DELAY_S)
            fetched = True
            try:
                raw = fetch(url)
            except OSError as e:
                failed.append(f"{r['name']}: {e}")
                continue
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(raw)
        try:
            photo = measure(Image.open(io.BytesIO(raw)))
        except (Unreadable, OSError) as e:
            failed.append(f"{r['name']}: {e}")
            r.pop("tint_glaze_photo", None)
            continue
        photo["retrieved"] = datetime.date.today().isoformat()
        r["tint_glaze_photo"] = photo
        measured += 1
    save_records(records, args.db)
    print(f"measured {measured} photos" + "".join(f"\n  skipped {f}" for f in failed))
    print(f"wrote {args.db}; now run: python -m paint_by_numbers.fit --db {args.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
