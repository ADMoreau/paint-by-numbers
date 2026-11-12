"""Paint database: load and save Golden paint records.

The database is a JSON list of records like::

    {"name": "Green Gold", "item": 1170, "line": "Heavy Body",
     "lab": [45.05, -10.69, 30.52], "srgb": [106, 110, 54],
     "transparency": "transparent", "sheen": "gloss",
     "pigments": ["PY150", "PG36", "PY175"], "K": [...], "S": [...],
     "source": "https://...", "retrieved": "2026-11-11",
     "tint_glaze_image": "https://..."}

``K`` and ``S`` are filled in by ``python -m paint_by_numbers.fit``, never by hand.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

DEFAULT_DB = Path(__file__).parent / "data" / "paints_heavy_body.json"

# Most opaque first.
TRANSPARENCY_RATINGS = ("opaque", "semi-opaque", "semi-transparent", "transparent")


@dataclass(frozen=True)
class Paint:
    name: str
    item: int
    line: str
    lab: tuple[float, float, float]
    srgb: tuple[int, int, int]
    transparency: str
    sheen: str | None
    pigments: tuple[str, ...]
    K: tuple[float, float, float]
    S: tuple[float, float, float]
    lightfastness: str | None = None  # Golden's rating: "Excellent", "Very Good", "Fair", "Very Poor"
    discontinued: bool = False
    special: str | None = None  # "fluorescent" or "phosphorescent": not modeled as plain paint

    @property
    def slug(self) -> str:
        return slugify(self.name)


def default_palette(paints: Sequence[Paint]) -> list[Paint]:
    """Paints used when the user names none: not discontinued, not fluorescent/phosphorescent."""
    return [p for p in paints if not p.discontinued and p.special is None]


def ks_tensors(stack: Sequence[Paint], dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """K and S for a layer stack (bottom first; paints may repeat), each shape (n, 3)."""
    if not stack:
        return torch.empty(0, 3, dtype=dtype), torch.empty(0, 3, dtype=dtype)
    return (
        torch.tensor([p.K for p in stack], dtype=dtype),
        torch.tensor([p.S for p in stack], dtype=dtype),
    )


def find_paint(paints: Sequence[Paint], name: str) -> Paint:
    """Look up a paint by name, ignoring case and punctuation ('phthalo blue green shade' works)."""
    by_slug = {p.slug: p for p in paints}
    slug = slugify(name)
    if slug in by_slug:
        return by_slug[slug]
    import difflib

    close = difflib.get_close_matches(slug, by_slug, n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(by_slug[c].name for c in close)}?" if close else ""
    raise ValueError(f"no paint named {name!r} in the database.{hint}")


def select_paints(paints: Sequence[Paint], names: str | Sequence[str]) -> list[Paint]:
    """Paint names (a comma-separated string or a list) -> paints, in database
    order, without duplicates."""
    if isinstance(names, str):
        names = names.split(",")
    wanted = {find_paint(paints, n).slug for n in names if n.strip()}
    if not wanted:
        raise ValueError("no paints named")
    return [p for p in paints if p.slug in wanted]


def read_paint_names(path: str | Path) -> list[str]:
    """Paint names from a palette file: one or more per line, comma-separated;
    blank lines and '#' comments are ignored."""
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]
        names.extend(n.strip() for n in line.split(",") if n.strip())
    return names


def slugify(name: str) -> str:
    """'Phthalo Blue (Green Shade)' -> 'phthalo-blue-green-shade'."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def is_interference(record: dict) -> bool:
    """Interference colors publish L*a*b* as all zeros; they are skipped."""
    return all(v == 0 for v in record["lab"])


def load_records(path: str | Path = DEFAULT_DB) -> list[dict]:
    """Raw records, including interference colors and unfitted K/S."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_records(records: list[dict], path: str | Path = DEFAULT_DB) -> None:
    """Write records sorted by name, with short number lists kept on one line."""
    records = sorted(records, key=lambda r: r["name"].lower())
    text = json.dumps(records, indent=2, ensure_ascii=False)
    text = re.sub(
        r"\[\s+([^\[\]{}]*?)\s+\]",
        lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]",
        text,
    )
    Path(path).write_text(text + "\n", encoding="utf-8")


def load_paints(path: str | Path = DEFAULT_DB) -> list[Paint]:
    """Fitted paints ready for rendering. Interference colors are skipped."""
    paints = []
    for r in load_records(path):
        if is_interference(r):
            continue
        if r.get("K") is None or r.get("S") is None:
            raise ValueError(f"{r['name']} has no K/S yet; run `python -m paint_by_numbers.fit`")
        paints.append(
            Paint(
                name=r["name"],
                item=r["item"],
                line=r["line"],
                lab=tuple(r["lab"]),
                srgb=tuple(r["srgb"]),
                transparency=r["transparency"],
                sheen=r["sheen"],
                pigments=tuple(r["pigments"]),
                K=tuple(r["K"]),
                S=tuple(r["S"]),
                lightfastness=r.get("lightfastness"),
                discontinued=bool(r.get("discontinued", False)),
                special=r.get("special"),
            )
        )
    return paints
