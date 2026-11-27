"""Estimate Kubelka-Munk K and S per paint from Golden's published color and swatch photos.

    python -m paint_by_numbers.fit [--db PATH]

fills in K and S for every paint in the database and prints a summary.

Method:

1. Published L*a*b* -> working-space (linear Rec.2020) reflectance R, clipped to
   REFLECTANCE_CLIP.
2. K/S = (1 - R)^2 / (2R) per channel. This is fixed: a thick layer always
   renders back to R.
3. S per coat, first by coverage: the transparency rating sets how many coats
   hide a black-and-white chart (COATS_TO_HIDE), meaning the layer is within
   HIDING_DELTA_E of the paint's masstone over both black and white. Bisection
   finds the S that meets it. S is the same on all three channels.
4. Then, for paints with measurements of Golden's tint and glaze photo
   (``tint_glaze_photo``, from scripts/fetch_tint_glaze.py), S per channel is
   refit to the photo: one coat at full strength, 1:1 and 1:10 glazes over the
   white chart and the black stripe, and 1:1 and 1:10 tints with Titanium
   White (refine_s). K/S stays as in step 2.
5. K = (K/S) * S.

Why step 4: the published color is the masstone at full hiding. For dark,
strong, transparent paints it is close to black (Dioxazine Purple is
L*25 a*2.6 b*1.3), so step 3 made them weak, gray glazes: one full coat of
Dioxazine Purple over white came out nearly white, where Golden's photo shows
even a 1:10 glaze as a strong violet. The photo fixes both the strength and
the hue of thin layers. Photo colors are only roughly right (opaque paints'
photographed masstones are a median 8 ΔE from the published values), which is
why the published color stays exact and the photo only sets S.

Why step 3 calibrates (paints without a photo): one fixed S per rating made
dark paints hide almost at once and light ones poorly, because hiding depends
on b*S and b is small for light colors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from . import color, km
from .paints import DEFAULT_DB, Paint, is_interference, load_paints, load_records, save_records

# Refitting S to Golden's tint and glaze photos (step 4). One brushed coat of a glaze of 1 part
# paint to n parts medium counts as 1/(n+1) coats, as in the plans' mixing instructions.
PHOTO_COATS = {"masstone": 1.0, "glaze_1_1": 0.5, "glaze_1_10": 1 / 11}
PHOTO_TINTS = {"tint_1_1": 0.5, "tint_1_10": 1 / 11}  # fraction of paint in a mix with TINT_WHITE
TINT_WHITE = "Titanium White"
PHOTO_CHART_WHITE = (245, 245, 245)  # sRGB of the white chart under the bands (the photos cut it out)
REFINE_STEPS = 1500
REFINE_LR = 0.05
REFINE_S_RANGE = (1e-3, 1e3)

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


def _photo_lab(srgb255) -> torch.Tensor:
    return color.srgb_to_lab(torch.as_tensor(srgb255, dtype=torch.float64) / 255.0)


def _photo_ground(photo: dict) -> torch.Tensor:
    ground = photo.get("ground") or PHOTO_CHART_WHITE
    return color.lab_to_working(_photo_lab(ground))


def photo_error(K, S, photo: dict, white: tuple | None = None) -> torch.Tensor:
    """ΔE of each band in a tint and glaze photo, as the model renders it with (K, S).

    Layers come first (each band over the chart, then over the stripe), then
    the tints (mixed with ``white`` = (K, S) of TINT_WHITE), if the photo has
    them. K and S have shape (..., 3); the result has shape (..., bands).
    """
    K = torch.as_tensor(K, dtype=torch.float64)
    S = torch.as_tensor(S, dtype=torch.float64)
    ground, stripe = _photo_ground(photo), color.lab_to_working(_photo_lab(photo["stripe"]))
    out, target = [], []
    for band, t in PHOTO_COATS.items():
        for under, side in ((ground, "white"), (stripe, "black")):
            out.append(km.layer(under, K, S, torch.tensor(t, dtype=torch.float64)))
            target.append(_photo_lab(photo[band][side]))
    if white is not None:
        for band, c in PHOTO_TINTS.items():
            if photo.get(band) is not None:
                Kw, Sw = (torch.as_tensor(x, dtype=torch.float64) for x in white)
                out.append(km.layer(ground, c * K + (1 - c) * Kw, c * S + (1 - c) * Sw, torch.tensor(1.0, dtype=torch.float64)))
                target.append(_photo_lab(photo[band]))
    lab = color.working_to_lab(torch.stack(out, dim=-2))
    return color.delta_e(lab, torch.stack(target))


def refine_s(ks: torch.Tensor, s0: torch.Tensor, photos: list[dict], white: tuple | None) -> torch.Tensor:
    """S per channel, shape (P, 3), fitting each paint's photo with K = ks * S.

    ``ks`` (P, 3) is the fixed K/S ratio; ``s0`` (P, 3) the starting S. All
    paints are fit at once (they are independent), by least squares on ΔE.
    """
    ground = torch.stack([_photo_ground(p) for p in photos])
    stripe = torch.stack([color.lab_to_working(_photo_lab(p["stripe"])) for p in photos])
    unders, coats, targets = [], [], []
    for band, t in PHOTO_COATS.items():
        for under, side in ((ground, "white"), (stripe, "black")):
            unders.append(under)
            coats.append(t)
            targets.append(torch.stack([_photo_lab(p[band][side]) for p in photos]))
    under, target = torch.stack(unders, 1), torch.stack(targets, 1)  # (P, bands, 3)
    coats = torch.tensor(coats, dtype=torch.float64)[None, :, None]
    conc = torch.tensor(list(PHOTO_TINTS.values()), dtype=torch.float64)[None, :, None]
    has_tint = torch.tensor([[p.get(b) is not None and white is not None for b in PHOTO_TINTS] for p in photos])
    tint_target = torch.stack([
        torch.stack([_photo_lab(p[b]) if p.get(b) is not None else torch.zeros(3, dtype=torch.float64) for b in PHOTO_TINTS])
        for p in photos
    ])
    if white is not None:
        Kw, Sw = (torch.as_tensor(x, dtype=torch.float64) for x in white)
    lo, hi = (torch.log(torch.tensor(v, dtype=torch.float64)) for v in REFINE_S_RANGE)
    log_s = torch.log(s0.clamp(*REFINE_S_RANGE)).clone().requires_grad_()
    opt = torch.optim.Adam([log_s], lr=REFINE_LR)
    for _ in range(REFINE_STEPS):
        opt.zero_grad()
        S = log_s.clamp(lo, hi).exp()
        K = ks * S
        out = km.layer(under, K[:, None], S[:, None], coats)
        loss = (color.delta_e(color.working_to_lab(out), target) ** 2).sum()
        if white is not None and has_tint.any():
            mix = km.layer(ground[:, None], conc * K[:, None] + (1 - conc) * Kw, conc * S[:, None] + (1 - conc) * Sw,
                           torch.tensor(1.0, dtype=torch.float64))
            loss = loss + (color.delta_e(color.working_to_lab(mix), tint_target) ** 2 * has_tint).sum()
        loss.backward()
        opt.step()
    return log_s.detach().clamp(lo, hi).exp()


def _photo_usable(record: dict) -> bool:
    return bool(record.get("tint_glaze_photo")) and not record.get("special") and not is_interference(record)


def fit_records(records: list[dict]) -> list[dict]:
    """Copies of ``records`` with K and S filled in. Interference colors get null."""
    fitted, ks = [], {}
    for i, record in enumerate(records):
        record = dict(record)
        if is_interference(record):
            record["K"] = record["S"] = None
        else:
            K, S = estimate_ks(record["lab"], record["transparency"])
            record["K"], record["S"] = _rounded(K), _rounded(S)
            ks[i] = K / S
        fitted.append(record)

    def refine(indices: list[int], white: tuple | None) -> None:
        if not indices:
            return
        S = refine_s(
            torch.stack([ks[i] for i in indices]),
            torch.stack([torch.tensor(fitted[i]["S"], dtype=torch.float64) for i in indices]),
            [fitted[i]["tint_glaze_photo"] for i in indices],
            white,
        )
        for i, s in zip(indices, S):
            fitted[i]["K"], fitted[i]["S"] = _rounded(ks[i] * s), _rounded(s)

    photo = [i for i, r in enumerate(fitted) if _photo_usable(r)]
    # the white the tints are mixed with goes first, so the others' tints use its refit values
    white = [i for i in photo if fitted[i]["name"] == TINT_WHITE]
    refine(white, None)
    white_ks = next(((fitted[i]["K"], fitted[i]["S"]) for i, r in enumerate(fitted) if r["name"] == TINT_WHITE and r["K"]), None)
    refine([i for i in photo if i not in white], white_ks)
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
        f"{'paint':28s} {'rating':17s} {'S/coat':>7s} {'dE 1 coat':>9s} {'dE 2 coats':>10s} {'photo dE':>8s} "
        f"{'thick dE white':>14s} {'thick dE black':>14s}"
    )
    records = {r["name"]: r for r in load_records(args.db)}
    white = records.get(TINT_WHITE)
    white_ks = (white["K"], white["S"]) if white and white.get("K") else None
    for p in load_paints(args.db):
        photo = records[p.name].get("tint_glaze_photo")
        photo_de = f"{photo_error(p.K, p.S, photo, white_ks).mean().item():8.1f}" if photo and _photo_usable(records[p.name]) else f"{'-':>8s}"
        print(
            f"{p.name:28s} {p.transparency:17s} {sum(p.S) / 3:7.3f} "
            f"{hiding_error(p.K, p.S, 1.0):9.1f} {hiding_error(p.K, p.S, 2.0):10.1f} {photo_de} "
            f"{thick_layer_error(p, [1.0, 1.0, 1.0]):14.2f} {thick_layer_error(p, [0.0, 0.0, 0.0]):14.2f}"
        )
    print("dE n coats: worst of black/white vs masstone. photo dE: mean over the tint and glaze photo's bands.")
    print("thick dE: vs published color. S/coat: mean of the channels.")
    print(f"wrote K and S to {args.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
