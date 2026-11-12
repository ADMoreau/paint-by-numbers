"""Estimate Kubelka-Munk K and S per paint from Golden's published color.

    python -m paint_by_numbers.fit [--db PATH]

fills in K and S for every paint in the database and prints a summary.

Method:

1. Published L*a*b* -> working-space (linear Rec.2020) reflectance R, clipped to
   REFLECTANCE_CLIP.
2. K/S = (1 - R)^2 / (2R) per channel.
3. S per coat, calibrated per paint by coverage: the transparency rating sets
   how many coats hide a black-and-white chart (COATS_TO_HIDE, the tuning
   knob), meaning the layer is within HIDING_DELTA_E of the paint's masstone
   over both black and white. Bisection finds the S that meets it. S is the
   same on all three channels.
4. K = (K/S) * S.

Step 2 inverts the masstone formula, so a thick layer renders back to R. The
only errors come from the clip (very dark or very light channels) and from
km.MIN_KS on very light channels.

Why step 3 calibrates: one fixed S per rating (the spec's first guess) made
dark paints hide almost at once and light ones poorly, because hiding depends
on b*S and b is small for light colors. Two coats of Titanium White over black
reached only L* 94 of its 98.

Not built yet: refining K and S per paint against the tint and glaze swatch
on each product page (its URL is stored as ``tint_glaze_image``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from . import color, km
from .paints import DEFAULT_DB, Paint, is_interference, load_paints, load_records, save_records

# Coats needed to hide a black-and-white chart, by transparency rating. First
# guesses; tune here. Ratios follow the spec's original S table
# (3.0 : 1.5 : 0.6 : 0.15).
COATS_TO_HIDE = {
    "opaque": 1.0,
    "semi-opaque": 2.0,
    "semi-transparent": 5.0,
    "transparent": 20.0,
}
HIDING_DELTA_E = 2.0  # "hidden" means within this many Lab units of the masstone

REFLECTANCE_CLIP = (0.01, 0.99)
SIGNIFICANT_DIGITS = 6  # stored precision of K and S

_S_SEARCH = (1e-4, 1e4)  # bisection bracket for S per coat
_BLACK_WHITE = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)


def published_reflectance(lab) -> torch.Tensor:
    """Working-space reflectance of the published color, clipped to REFLECTANCE_CLIP."""
    lab = torch.as_tensor(lab, dtype=torch.float64)
    return color.lab_to_working(lab).clamp(*REFLECTANCE_CLIP)


def hiding_error(K, S, coats: float) -> float:
    """Worst ΔE, over black and over white, between ``coats`` of paint and its masstone."""
    K = torch.as_tensor(K, dtype=torch.float64)
    S = torch.as_tensor(S, dtype=torch.float64)
    out = km.layer(_BLACK_WHITE, K, S, torch.tensor(coats, dtype=torch.float64))
    target = color.working_to_lab(km.masstone(K, S))
    return color.delta_e(color.working_to_lab(out), target).max().item()


def calibrate_s(ks: torch.Tensor, coats: float, delta_e: float = HIDING_DELTA_E) -> float:
    """S per coat at which ``coats`` coats hide black and white to within ``delta_e``.

    ``ks`` is the per-channel K/S ratio, which stays fixed.
    """
    lo, hi = _S_SEARCH

    def hides(s: float) -> bool:
        return hiding_error(ks * s, torch.full_like(ks, s), coats) <= delta_e

    if not hides(hi):
        raise ValueError(f"no S up to {hi} hides within {delta_e} ΔE in {coats} coats")
    if hides(lo):
        return lo
    for _ in range(60):  # geometric bisection: ~1e-8 relative precision
        mid = (lo * hi) ** 0.5
        lo, hi = (lo, mid) if hides(mid) else (mid, hi)
    return hi


def estimate_ks(lab, transparency: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(K, S) per channel, each shape (3,), from published Lab and transparency rating."""
    if transparency not in COATS_TO_HIDE:
        raise ValueError(f"unknown transparency {transparency!r}; expected one of {list(COATS_TO_HIDE)}")
    r = published_reflectance(lab)
    ks = (1 - r) ** 2 / (2 * r)
    s = torch.full_like(r, calibrate_s(ks, COATS_TO_HIDE[transparency]))
    return ks * s, s


def _rounded(x: torch.Tensor) -> list[float]:
    return [float(f"{v:.{SIGNIFICANT_DIGITS}g}") for v in x.tolist()]


def fit_records(records: list[dict]) -> list[dict]:
    """Copies of ``records`` with K and S filled in. Interference colors get null."""
    fitted = []
    for record in records:
        record = dict(record)
        if is_interference(record):
            record["K"] = record["S"] = None
        else:
            K, S = estimate_ks(record["lab"], record["transparency"])
            record["K"], record["S"] = _rounded(K), _rounded(S)
        fitted.append(record)
    return fitted


def thick_layer_error(paint: Paint, background) -> float:
    """ΔE between a fully hiding layer of ``paint`` over ``background`` and its published color.

    "Fully hiding" means km.hiding_thickness at a 1e-5 reflectance tolerance,
    rather than a fixed coat count that light transparent paints may not reach.
    """
    K = torch.tensor(paint.K, dtype=torch.float64)
    S = torch.tensor(paint.S, dtype=torch.float64)
    t = km.hiding_thickness(K, S, tol=1e-5)
    out = km.layer(torch.as_tensor(background, dtype=torch.float64), K, S, t)
    return color.delta_e(color.working_to_lab(out), paint.lab).item()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fill in K and S for every paint in the database.")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args(argv)

    save_records(fit_records(load_records(args.db)), args.db)

    print(
        f"{'paint':28s} {'rating':17s} {'S/coat':>7s} {'dE 1 coat':>9s} {'dE 2 coats':>10s} "
        f"{'thick dE white':>14s} {'thick dE black':>14s}"
    )
    for p in load_paints(args.db):
        print(
            f"{p.name:28s} {p.transparency:17s} {p.S[0]:7.3f} "
            f"{hiding_error(p.K, p.S, 1.0):9.1f} {hiding_error(p.K, p.S, 2.0):10.1f} "
            f"{thick_layer_error(p, [1.0, 1.0, 1.0]):14.2f} {thick_layer_error(p, [0.0, 0.0, 0.0]):14.2f}"
        )
    print("dE n coats: worst of black/white vs masstone. thick dE: vs published color.")
    print(f"wrote K and S to {args.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
