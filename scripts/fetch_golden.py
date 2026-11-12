"""Add Golden Heavy Body paints to the database from their product pages.

Build time only: the planner never touches the network. Each paint is one
request to goldenartistcolors.com/products/heavy-body-acrylic-color-<slug>.
Results merge into paint_by_numbers/data/paints_heavy_body.json (matched by item
number), and any change to an existing record's color data is printed.

    python scripts/fetch_golden.py "Titanium White" "Carbon Black"
    python scripts/fetch_golden.py --cache pages/ "Green Gold"   # save/reuse HTML
    python scripts/fetch_golden.py --all --cache pages/          # whole line (~156 requests)

A page that cannot be fetched or parsed is reported at the end; the rest are
still saved.

Fetched records have K and S set to null. Run `python -m paint_by_numbers.fit` next.
"""

from __future__ import annotations

import argparse
import datetime
import html
import re
import sys
import time
import urllib.request
from pathlib import Path

from paint_by_numbers.paints import DEFAULT_DB, TRANSPARENCY_RATINGS, is_interference, load_records, save_records, slugify

URL = "https://goldenartistcolors.com/products/heavy-body-acrylic-color-{slug}"
SITEMAP_INDEX = "https://goldenartistcolors.com/products/sitemap-index.xml"
USER_AGENT = "Mozilla/5.0 (compatible; paint-by-numbers-fetch/0.1)"
DELAY_S = 1.0  # between network requests

_NUM = r"(-?\d+(?:\.\d+)?)"


def page_text(raw: str) -> str:
    """Visible text with tag boundaries as ' | ' separators."""
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " | ", text))
    text = re.sub(r"(\s*\|\s*)+", " | ", text)
    return re.sub(r"\s+", " ", text)


class NoColorData(ValueError):
    """The page publishes no usable color data (special-effect or discontinued product)."""


_TRANSPARENCY_LABEL = r"\| (?:Opacity / )?Transparency \|"


def parse_product_page(raw: str, url: str) -> dict:
    text = page_text(raw)

    def find(pattern: str, source: str = text) -> tuple[str, ...]:
        m = re.search(pattern, source, flags=re.S)
        if not m:
            raise ValueError(f"{url}: pattern not found: {pattern}")
        return m.groups()

    def optional(pattern: str) -> str | None:
        m = re.search(pattern, text, flags=re.S)
        return m.group(1).strip() if m else None

    (name,) = find(r"<h1[^>]*>(.*?)</h1>", raw)
    name = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", name))).strip()
    lab_match = re.search(rf"\| lab \| {_NUM} \| , \| {_NUM} \| , \| {_NUM} \|", text)
    if not lab_match:
        raise NoColorData(f"{name}: no L*a*b* published")
    lab = [float(v) for v in lab_match.groups()]
    srgb = [int(v) for v in find(r"\| srgb \| (\d+) \| , \| (\d+) \| , \| (\d+) \|")]
    transparency = optional(_TRANSPARENCY_LABEL + r" ([^|]+?) \|")
    transparency = transparency.lower().replace(" ", "-") if transparency else None
    if transparency in (None, "n/a"):
        raise NoColorData(f"{name}: no transparency rating")
    if transparency not in TRANSPARENCY_RATINGS:
        raise ValueError(f"{url}: unknown transparency {transparency!r}")
    sheen = optional(r"\| Sheen \| ([^|]+?) \|")
    (pigment_block,) = find(r"CI Name \(Number\)(.*?)" + _TRANSPARENCY_LABEL)
    pigments = re.findall(r"\b(P(?:Bk|Br|[A-Z])\d+(?::\d+)?)\s*\(", pigment_block)
    item = optional(r"\| Item \| UPC \| [^|]+ \| 0*(\d+)-\d+ \|") or optional(r"\| Item: (\d+) \|")
    if item is None:
        raise ValueError(f"{url}: no item number")
    (title,) = find(r"<title>(.*?)</title>", raw)
    swatch = re.search(r'https://images\.goldenartistcolors\.com/[^"\s]+?tint-glaze-400\.webp', raw)
    special = next((kind for kind in ("fluorescent", "phosphorescent") if kind in name.lower()), None)

    return {
        "name": name,
        "item": int(item),
        "line": "Heavy Body",
        "lab": lab,
        "srgb": srgb,
        "transparency": transparency,
        "sheen": sheen.lower() if sheen else None,
        "lightfastness": optional(r"\| Lightfastness(?: rating)? \| ([^|]+?) \|"),
        "pigments": pigments,
        "discontinued": "(Discontinued)" in title,
        "special": special,
        "K": None,
        "S": None,
        "source": url,
        "retrieved": datetime.date.today().isoformat(),
        "tint_glaze_image": swatch.group(0) if swatch else None,
    }


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def all_heavy_body_slugs() -> list[str]:
    """Every Heavy Body color page listed in Golden's products sitemap."""
    index = fetch(SITEMAP_INDEX)
    slugs: set[str] = set()
    for sitemap in re.findall(r"<loc>([^<]+)</loc>", index):
        time.sleep(DELAY_S)
        slugs.update(re.findall(r"/products/heavy-body-acrylic-color-([a-z0-9-]+)</loc>", fetch(sitemap)))
    return sorted(slugs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="paint names as on Golden's site, e.g. 'Titanium White'")
    ap.add_argument("--all", action="store_true", help="every Heavy Body color in Golden's products sitemap")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--cache", type=Path, help="directory to save fetched HTML and reuse it")
    args = ap.parse_args(argv)

    records = load_records(args.db) if args.db.exists() else []
    by_item = {r["item"]: r for r in records}
    fetched_any = False
    failures: list[tuple[str, str]] = []
    skipped: list[str] = []

    names = list(args.names)
    if args.all:
        names += all_heavy_body_slugs()
        print(f"sitemap lists {len(names)} Heavy Body colors")

    try:
        for name in names:
            slug = slugify(name)
            url = URL.format(slug=slug)
            cached = args.cache / f"{slug}.html" if args.cache else None
            try:
                if cached and cached.exists():
                    raw = cached.read_text(encoding="utf-8")
                else:
                    if fetched_any:
                        time.sleep(DELAY_S)
                    fetched_any = True
                    raw = fetch(url)
                    if cached:
                        cached.parent.mkdir(parents=True, exist_ok=True)
                        cached.write_text(raw, encoding="utf-8")
                record = parse_product_page(raw, url)
            except NoColorData as e:
                skipped.append(slug)
                print(f"skip   {e}")
                continue
            except (OSError, ValueError) as e:  # network errors are OSErrors
                failures.append((slug, str(e)))
                print(f"FAIL   {slug}: {e}")
                continue

            if is_interference(record):
                print(f"skip   {record['name']}: interference color (Lab all zeros)")
                continue
            old = by_item.get(record["item"])
            if old:
                changes = [
                    f"{k} {old.get(k)} -> {record[k]}"
                    for k in ("name", "lab", "srgb", "transparency", "sheen", "pigments", "discontinued")
                    if old.get(k) != record[k]
                ]
                print(f"update {record['name']}: " + ("; ".join(changes) if changes else "no changes"))
            else:
                print(f"add    {record['name']} ({record['item']}) {record['transparency']} lab={record['lab']}")
            by_item[record["item"]] = record
    finally:  # keep what was fetched even if interrupted
        save_records(list(by_item.values()), args.db)
        print(f"wrote {len(by_item)} paints to {args.db}" + (f"; skipped {len(skipped)} without color data" if skipped else ""))

    if failures:
        print(f"\n{len(failures)} failed:")
        for slug, err in failures:
            print(f"  {slug}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
