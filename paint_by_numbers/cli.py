"""Command-line entry point.

    python -m paint_by_numbers.cli input.jpg --out out/ \\
        --background "#7A6A58" --size 256 \\
        [--max-layers N] [--min-gain 0.5] \\
        [--paints "Titanium White,Ultramarine Blue,..."] \\
        [--background-paint NAME | --background-image PATH]

Writes layers/NN_<paint>.png, preview.png, error.png and recipe.json to --out.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

from . import paints as paintdb
from .palette import PaletteConfig
from .plan import (
    BREAKDOWNS,
    background_from_color,
    background_from_image,
    background_from_paint,
    layer_thicknesses,
    load_image,
    swatch_colors,
    plan,
    write_outputs,
)
from .solve import SolveConfig


def parse_levels(text: str | None, t_max: float) -> tuple[float, ...]:
    """Thickness levels in coats. Default: SolveConfig's, cut to t_max (which is always a level)."""
    if text is None:
        levels = {lv for lv in SolveConfig().levels if lv <= t_max} | {0.0, t_max}
    else:
        try:
            levels = {float(v) for v in text.split(",") if v.strip()} | {0.0}
        except ValueError:
            raise ValueError(f"--levels must be comma-separated numbers, got {text!r}") from None
        if max(levels) > t_max:
            raise ValueError(f"--levels go up to {max(levels)} but --t-max is {t_max}")
        if min(levels) < 0:
            raise ValueError("--levels must not be negative")
    return tuple(sorted(levels))


def swatch_blocks(colors: dict) -> str:
    """Colored terminal blocks: full strength, then each thickness over white (24-bit ANSI)."""
    block = lambda rgb: f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m\u2588\u2588\x1b[0m"  # noqa: E731
    return block(colors["full_strength"]) + " " + "".join(block(rgb) for rgb in colors["on_white"].values())


def _use_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m paint_by_numbers.cli",
        description="Plan layered Golden acrylic paint stacks from an image: one grayscale thickness map per layer.",
    )
    ap.add_argument("input", type=Path, help="image to paint")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--size", type=int, default=256, help="working size, long side in px (default 256)")

    bg = ap.add_mutually_exclusive_group()
    bg.add_argument("--background", metavar="COLOR", help="background color, '#RRGGBB' or 'R,G,B' (default white)")
    bg.add_argument("--background-paint", metavar="NAME", help="an opaque ground coat of this database paint")
    bg.add_argument("--background-image", metavar="PATH", type=Path, help="per-pixel background image")

    ap.add_argument("--paints", metavar="NAMES",
                    help="comma-separated paints you own (default: every current, non-fluorescent paint in the database)")  # fmt: skip
    ap.add_argument("--paints-file", metavar="PATH", type=Path,
                    help="text file of paints you own, one per line ('#' comments); combined with --paints")
    ap.add_argument("--max-paints", type=int,
                    help="most different paints to use; the planner picks the best set (default: no limit)")
    ap.add_argument("--max-layers", type=int, help="cap on the number of layers (default: no limit)")
    ap.add_argument("--min-gain", type=float, default=PaletteConfig.min_gain,
                    help="RMS Lab error a new layer must save to be added (default %(default)s)")  # fmt: skip

    tune = ap.add_argument_group("solver")
    tune.add_argument("--steps", type=int, default=SolveConfig.steps, help="optimizer steps (default %(default)s)")
    tune.add_argument("--lam-smooth", type=float, default=SolveConfig.lam_smooth, help="smoothness weight (default %(default)s)")
    tune.add_argument("--lam-paint", type=float, default=SolveConfig.lam_paint, help="paint-amount weight (default %(default)s)")
    tune.add_argument("--t-max", type=float, default=SolveConfig.t_max, help="max coats per layer (default %(default)s)")
    tune.add_argument("--levels", help="thickness levels in coats, comma-separated (default 0,0.125,0.25,0.5,0.75,1,1.5,2)")
    tune.add_argument("--max-thicknesses", type=int, default=SolveConfig.max_thicknesses,
                      help="most different thicknesses per layer, 0 for no limit (default %(default)s)")
    tune.add_argument("--smoothing", type=float, default=SolveConfig.refine_smooth,
                      help="how strongly thickness regions are kept in one piece rather than speckled; "
                           "0 is off (default %(default)s)")

    ap.add_argument("--breakdown", choices=BREAKDOWNS, default="thickness",
                    help="how to split each layer further: each thickness as its own layer (default), "
                         "nested coats ('passes'), both, or none")
    ap.add_argument("--db", type=Path, default=paintdb.DEFAULT_DB, help="paint database JSON")
    ap.add_argument("-q", "--quiet", action="store_true", help="only print the summary")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    say = (lambda msg: None) if args.quiet else (lambda msg: print(msg, flush=True))
    start = time.perf_counter()

    try:
        if not args.input.is_file():
            raise ValueError(f"input image not found: {args.input}")
        if args.size < 8:
            raise ValueError("--size must be at least 8")
        if args.max_layers is not None and args.max_layers < 0:
            raise ValueError("--max-layers must not be negative")
        if args.max_paints is not None and args.max_paints < 1:
            raise ValueError("--max-paints must be at least 1")
        if args.t_max <= 0:
            raise ValueError("--t-max must be positive")
        if args.max_thicknesses is not None and args.max_thicknesses < 0:
            raise ValueError("--max-thicknesses must not be negative")
        if args.smoothing < 0:
            raise ValueError("--smoothing must not be negative")
        levels = parse_levels(args.levels, args.t_max)

        all_paints = paintdb.load_paints(args.db)
        if args.paints_file and not args.paints_file.is_file():
            raise ValueError(f"paints file not found: {args.paints_file}")
        named = (args.paints.split(",") if args.paints else []) + (
            paintdb.read_paint_names(args.paints_file) if args.paints_file else []
        )
        available = paintdb.select_paints(all_paints, named) if named else paintdb.default_palette(all_paints)
        image = load_image(args.input, args.size)
        if args.background_paint:
            background = background_from_paint(paintdb.find_paint(all_paints, args.background_paint))
        elif args.background_image:
            if not args.background_image.is_file():
                raise ValueError(f"background image not found: {args.background_image}")
            background = background_from_image(args.background_image, image.shape[1], image.shape[2])
        else:
            background = background_from_color(args.background or "#ffffff")
    except ValueError as e:
        ap.error(str(e))

    palette_config = PaletteConfig(min_gain=args.min_gain, max_layers=args.max_layers, max_paints=args.max_paints)
    solve_config = replace(
        SolveConfig(), steps=args.steps, lam_smooth=args.lam_smooth, lam_paint=args.lam_paint, t_max=args.t_max, levels=levels,
        max_thicknesses=args.max_thicknesses or None, refine_smooth=args.smoothing,
    )
    left_out = "" if named else (
        f" (leaving out {len(all_paints) - len(available)} discontinued or fluorescent/phosphorescent;"
        " name them in --paints to use them)"
    )  # fmt: skip
    say(f"{args.input.name}: {image.shape[2]}x{image.shape[1]} px, background {background.description}, "
        f"{len(available)} paints{left_out}")  # fmt: skip

    result = plan(image, available, background, palette_config, solve_config, progress=say)
    written = write_outputs(result, args.out, source=str(args.input), breakdown=args.breakdown)

    sol, stack = result.solution, result.selection.stack
    distinct = len({p.name for p in stack})
    print(f"\n{len(stack)} layers using {distinct} different paint{'s' if distinct != 1 else ''}, bottom first:")
    if background.kind == "paint":
        print(f"   0  {background.paint.name} (ground coat)")
    color_blocks = _use_color()
    for i, p in enumerate(stack, start=1):
        regions = layer_thicknesses(sol.t_snapped[i - 1, 0], solve_config.levels)
        n = len(regions)
        blocks = swatch_blocks(swatch_colors(p, [c for c, _ in regions])) + "  " if color_blocks else ""
        print(f"  {i:2d}  {blocks}{p.name}  [{p.transparency}]  {n} thickness{'es' if n != 1 else ''}")
    print(f"color error (ΔE): mean {sol.mean_error:.2f}, max {sol.max_error:.1f}"
          f"  (before quantizing: mean {sol.mean_error_continuous:.2f})")  # fmt: skip
    print(f"wrote layer maps, overview.png, preview.png, error.png, recipe.json"
          + {"thickness": ", thickness/ and thickness_overview.png", "passes": ", passes/ and passes_overview.png",
             "both": ", thickness/, passes/ and their overviews", "none": ""}[args.breakdown]
          + f" to {args.out}/")  # fmt: skip
    print(f"done in {time.perf_counter() - start:.0f}s")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
