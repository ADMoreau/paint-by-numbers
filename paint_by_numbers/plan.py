"""End-to-end planning: image + paints + background -> layer stack and output files.

    image = load_image("input.jpg", size=256)
    bg = background_from_color("#7A6A58")          # or background_from_paint / _image
    result = plan(image, paints, bg)
    write_outputs(result, "out/")

Outputs (spec, "What to build"):

- ``layers/NN_<paint-slug>.png``: one per layer, bottom first. Pixel value is
  thickness: black is none, white is ``t_max`` coats. NN starts at 01 and is
  zero-padded to at least two digits.
- ``preview.png``: the simulated finished stack over the background.
- ``recipe.json``: background, layer order, paint per layer, thickness levels
  (with how to paint each), mean and max color error, and notes.
- ``error.png``: per-pixel ΔE between preview and target; black is 0, white is
  ERROR_PNG_SCALE or more.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from . import color, km
from .paints import Paint
from .palette import PaletteConfig, Selection, select_stack
from .solve import Solution, SolveConfig, solve

__all__ = [
    "Background",
    "PlanResult",
    "load_image",
    "parse_color",
    "background_from_color",
    "background_from_paint",
    "background_from_image",
    "plan",
    "write_outputs",
    "layer_filename",
    "layer_passes",
    "pass_filename",
    "layer_thicknesses",
    "swatch_colors",
    "thickness_basename",
    "BREAKDOWNS",
    "level_instructions",
]

ERROR_PNG_SCALE = 20.0  # ΔE shown as white in error.png


# ---------------------------------------------------------------- inputs


def load_image(path: str | Path, size: int = 256) -> torch.Tensor:
    """sRGB image in [0, 1], shape (3, H, W), long side resized to ``size``."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        scale = size / max(img.size)
        if scale != 1:
            new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            img = img.resize(new, Image.Resampling.LANCZOS)
        return _pil_to_tensor(img)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


def parse_color(text: str) -> torch.Tensor:
    """'#7A6A58', '7A6A58', '#7A6', or '122,106,88' -> sRGB in [0, 1], shape (3,)."""
    s = text.strip()
    m = re.fullmatch(r"#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})", s)
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return torch.tensor([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=torch.float64) / 255.0
    parts = re.split(r"[,\s]+", s)
    if len(parts) == 3 and all(p.isdigit() and int(p) <= 255 for p in parts):
        return torch.tensor([int(p) for p in parts], dtype=torch.float64) / 255.0
    raise ValueError(f"not a color: {text!r} (use '#RRGGBB' or 'R,G,B' with 0-255 values)")


@dataclass
class Background:
    kind: str  # "color", "paint" or "image"
    linear: torch.Tensor  # (3,) or (3, H, W) working-space (linear Rec.2020) reflectance
    description: str  # what the user gave: "#7a6a58", "Burnt Umber", "ground.png"
    paint: Paint | None = None


def _hex(srgb: torch.Tensor) -> str:
    return "#" + "".join(f"{round(v * 255):02x}" for v in srgb.clamp(0, 1).tolist())


def background_from_color(text: str = "#ffffff") -> Background:
    srgb = parse_color(text)
    return Background("color", color.srgb_to_working(srgb), _hex(srgb))


def background_from_paint(paint: Paint) -> Background:
    """An opaque ground coat of ``paint``: its masstone. Recorded as layer 0."""
    linear = km.masstone(torch.tensor(paint.K, dtype=torch.float64), torch.tensor(paint.S, dtype=torch.float64))
    return Background("paint", linear, paint.name, paint)


def background_from_image(path: str | Path, height: int, width: int) -> Background:
    """A per-pixel background, resized to the working resolution."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        srgb = _pil_to_tensor(img).double()
    return Background("image", color.srgb_to_working(srgb, dim=0), str(path))


# ---------------------------------------------------------------- planning


@dataclass
class PlanResult:
    image_srgb: torch.Tensor  # (3, H, W)
    target_lab: torch.Tensor  # (3, H, W)
    background: Background
    selection: Selection
    solution: Solution
    solve_config: SolveConfig
    palette_config: PaletteConfig


def plan(
    image_srgb: torch.Tensor,
    paints: Sequence[Paint],
    background: Background,
    palette_config: PaletteConfig = PaletteConfig(),
    solve_config: SolveConfig = SolveConfig(),
    progress: Callable[[str], None] | None = None,
) -> PlanResult:
    """Choose the layer stack, then solve its thickness maps at full resolution."""
    target = color.srgb_to_lab(image_srgb.float(), dim=0)
    if background.linear.ndim > 1 and background.linear.shape[1:] != target.shape[1:]:
        raise ValueError(f"background image is {tuple(background.linear.shape[1:])}, image is {tuple(target.shape[1:])}")
    selection = select_stack(target, background.linear, paints, palette_config, solve_config, progress)
    K, S = (torch.empty(0, 3), torch.empty(0, 3))
    if selection.stack:
        K = torch.tensor([p.K for p in selection.stack], dtype=torch.float64)
        S = torch.tensor([p.S for p in selection.stack], dtype=torch.float64)
    if progress:
        progress(f"solving {len(selection.stack)} layers at {target.shape[2]}x{target.shape[1]}")

    def report(step: int, loss: float) -> None:
        if progress and step % 500 == 0:
            progress(f"  step {step}: loss {loss:.3f}")

    solution = solve(target, background.linear.float(), K, S, solve_config, report)
    return PlanResult(image_srgb, target, background, selection, solution, solve_config, palette_config)


# ---------------------------------------------------------------- outputs


def layer_filename(index: int, total: int, paint: Paint) -> str:
    """'03_titanium-white.png'; at least two digits, more when total >= 100."""
    width = max(2, len(str(total)))
    return f"{index:0{width}d}_{paint.slug}.png"


def level_instructions(coats: float) -> str:
    """How to paint a thickness level, treating fractions of a coat as glazes."""
    if coats <= 0:
        return "no paint"
    whole, part = int(coats), coats - int(coats)
    pieces = []
    if whole:
        pieces.append(f"{whole} full coat" + ("s" if whole > 1 else ""))
    if part > 1e-9:
        f = Fraction(part).limit_denominator(16)
        paint_parts, medium_parts = f.numerator, f.denominator - f.numerator
        pieces.append(f"1 coat of {paint_parts} part{'s' if paint_parts > 1 else ''} paint to {medium_parts} part{'s' if medium_parts > 1 else ''} medium")
    return " + ".join(pieces)


def layer_passes(t: torch.Tensor, levels) -> list[tuple[float, torch.Tensor]]:
    """Split one layer's thickness map into stacked, uniform coats of the same paint.

    Returns (coats, mask) per pass, thinnest first: pass j adds
    ``level_j - level_(j-1)`` coats wherever ``t >= level_j``. Each mask lies
    inside the previous one, and because adjacent layers of one paint add up,
    the passes together render exactly like the layer.
    """
    used = sorted({float(lv) for lv in levels if lv > 0 and (t == lv).any()})
    passes, previous = [], 0.0
    for level in used:
        passes.append((level - previous, t >= level))
        previous = level
    return passes


def pass_filename(index: int, total: int, paint: Paint, pass_number: int) -> str:
    """'03_titanium-white_pass2.png'."""
    return layer_filename(index, total, paint)[: -len(".png")] + f"_pass{pass_number}.png"


BREAKDOWNS = ("thickness", "passes", "both", "none")


def layer_thicknesses(t: torch.Tensor, levels) -> list[tuple[float, torch.Tensor]]:
    """Split one layer into its thickness regions: (coats, mask) per level used,
    thinnest first. The masks do not overlap and together cover the layer."""
    return [(float(lv), t == lv) for lv in sorted(levels) if lv > 0 and (t == lv).any()]


def thickness_basename(index: int, total: int, paint: Paint, coats: float) -> str:
    """'03_titanium-white_0.125coats'."""
    return layer_filename(index, total, paint)[: -len(".png")] + f"_{coats:g}coats"


def _coats_label(coats: float) -> str:
    """0.125 -> '1/8 coat', 1.5 -> '1 1/2 coats', 2 -> '2 coats'."""
    f = Fraction(coats).limit_denominator(16)
    whole, part = divmod(f, 1)
    text = " ".join(str(x) for x in (whole, part) if x)
    return f"{text} coat" + ("s" if f > 1 else "")


def _save_gray(values: torch.Tensor, path: Path) -> None:
    arr = (values.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    Image.fromarray(arr, mode="L").save(path)


def _to_pil(reflectance: torch.Tensor) -> Image.Image:
    """Working-space reflectance (3, H, W) -> sRGB image, clipped to the sRGB gamut."""
    srgb = color.working_to_srgb(reflectance.float(), dim=0).clamp(0, 1)
    return Image.fromarray((srgb.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy(), mode="RGB")


def _save_rgb(reflectance: torch.Tensor, path: Path) -> None:
    _to_pil(reflectance).save(path)


def _layer_alone(K, S, t) -> tuple[torch.Tensor, str]:
    """One layer rendered over white or black, whichever shows it with more contrast."""
    painted = t[0, 0] > 0
    best = None
    for ground, value in (("white", 1.0), ("black", 0.0)):
        bg = torch.full((3,), value, dtype=torch.float64)
        out = km.render_stack(bg, K, S, t)
        bg_lab = color.working_to_lab(bg).reshape(3, *([1] * (out.ndim - 1)))
        contrast = color.delta_e(color.working_to_lab(out, dim=0), bg_lab, dim=0)
        score = contrast[painted].mean().item() if painted.any() else 0.0
        if best is None or score > best[0]:
            best = (score, out, ground)
    return best[1], best[2]


def _srgb255(reflectance: torch.Tensor) -> tuple[int, int, int]:
    v = color.working_to_srgb(reflectance).clamp(0, 1)
    return tuple(int(round(x * 255)) for x in v.tolist())


def swatch_colors(paint: Paint, thicknesses: Sequence[float], ground: torch.Tensor | None = None) -> dict:
    """What ``paint`` looks like: sRGB (0-255) at full strength, and at each
    thickness over white, over black and, if given, over ``ground`` (a flat
    working-space color such as the background).

    Returns {"full_strength": rgb, "on_white": {coats: rgb}, "on_black": {...},
    "on_ground": {...}}.
    """
    K = torch.tensor(paint.K, dtype=torch.float64)
    S = torch.tensor(paint.S, dtype=torch.float64)
    grounds = {"white": torch.ones(3, dtype=torch.float64), "black": torch.zeros(3, dtype=torch.float64)}
    if ground is not None:
        grounds["ground"] = ground.double()
    colors: dict = {"full_strength": _srgb255(km.masstone(K, S))}
    for name, g in grounds.items():
        colors[f"on_{name}"] = {
            float(c): _srgb255(km.layer(g, K, S, torch.tensor(float(c), dtype=torch.float64))) for c in thicknesses
        }
    return colors


def _short_coats(coats: float) -> str:
    return str(Fraction(coats).limit_denominator(16)) if coats < 1 else f"{coats:g}"


def _fit_title(draw, title: str, max_width: int, size: int, min_size: int = 8):
    """Font and lines for a title that must fit ``max_width``: shrink the font,
    then wrap before the parenthesized part (and at spaces) if still too wide."""
    from PIL import ImageFont

    for s in range(size, min_size - 1, -1):
        font = ImageFont.load_default(size=s)
        if draw.textlength(title, font=font) <= max_width:
            return font, [title]
    font = ImageFont.load_default(size=min_size)
    lines, line = [], ""
    for word in title.replace(" (", "\n(").replace("\n", " \n ").split(" "):
        if word == "\n":
            if line:
                lines.append(line)
            line = ""
            continue
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=font) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return font, lines


def _swatch_card(
    paint: Paint, colors: dict, size: tuple[int, int], title: str | None = None, font_size: int = 10
) -> Image.Image:
    """A paint chip: one column per thickness plus full strength, one row per ground.

    The title shrinks, then wraps, to stay inside the card.
    """
    from PIL import ImageDraw, ImageFont

    w, h = size
    rows = [k for k in ("on_white", "on_black", "on_ground") if k in colors]
    coats = list(colors["on_white"])
    cols = len(coats) + 1
    card = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(card)
    title_font, title_lines = _fit_title(draw, title or f"{paint.name} ({paint.transparency})", w - 8, font_size + 2)
    line_h = title_font.size + 3
    label_font = ImageFont.load_default(size=font_size)
    header = 4 + line_h * len(title_lines)
    label_h = font_size + 4
    left = min(int(font_size * 6), w // 4)
    cell_w = max(4, (w - left - 2) // cols)
    cell_h = max(4, (h - header - label_h - 2) // len(rows))
    top = header + label_h
    for k, line in enumerate(title_lines):
        draw.text((4, 3 + k * line_h), line, fill="black", font=title_font)
    long_labels = draw.textlength("3/4 coat", font=label_font) < cell_w - 4
    for c, label in enumerate([f"{_short_coats(x)} coat" if long_labels else _short_coats(x) for x in coats] + ["full"]):
        draw.text((left + c * cell_w + 2, header + 1), label, fill="black", font=label_font)
    full_names = all(draw.textlength(k.replace("_", " "), font=label_font) <= left - 4 for k in rows)
    for r, key in enumerate(rows):
        y = top + r * cell_h
        row_label = key.replace("_", " ") if full_names else key.removeprefix("on_")
        draw.text((2, y + cell_h // 2 - font_size // 2), row_label, fill="black", font=label_font)
        for c, x in enumerate(coats):
            x0 = left + c * cell_w
            draw.rectangle((x0 + 1, y + 1, x0 + cell_w - 2, y + cell_h - 2), fill=colors[key][x], outline=(200, 200, 200))
    x0 = left + len(coats) * cell_w  # full strength: one tall chip, the same on any ground
    draw.rectangle((x0 + 1, top + 1, x0 + cell_w - 2, top + len(rows) * cell_h - 2), fill=colors["full_strength"], outline=(200, 200, 200))
    return card


def _swatch_entry(colors: dict) -> dict:
    """Swatch colors as hex for the recipe."""
    hexes = lambda rgb: "#" + "".join(f"{v:02x}" for v in rgb)  # noqa: E731
    entry = {"full_strength": hexes(colors["full_strength"])}
    for key in ("on_white", "on_black", "on_ground"):
        if key in colors:
            entry[key] = {f"{c:g}": hexes(rgb) for c, rgb in colors[key].items()}
    return entry


def _overview(rows: list[tuple[str, list[tuple[str, Image.Image]]]], path: Path, columns: int = 6) -> None:
    """Labeled contact sheet: one titled section per row of (label, image) tiles."""
    from PIL import ImageDraw

    w, h = rows[0][1][0][1].size
    gap, title = 8, 16
    cols = min(columns, max(len(tiles) for _, tiles in rows))
    section_rows = [(len(tiles) + cols - 1) // cols for _, tiles in rows]
    height = sum(title + r * (h + gap) for r in section_rows)
    sheet = Image.new("RGB", (cols * (w + gap) - gap, height), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for (heading, tiles), nrows in zip(rows, section_rows):
        draw.text((0, y + 2), heading, fill="black")
        y += title
        for k, (label, img) in enumerate(tiles):
            x, yy = (k % cols) * (w + gap), y + (k // cols) * (h + gap)
            sheet.paste(img, (x, yy))
            if label:  # tiles that carry their own title (swatch cards) pass ""
                draw.rectangle((x, yy, x + 6 * len(label) + 6, yy + 13), fill="white")
                draw.text((x + 3, yy + 1), label, fill="black")
        y += nrows * (h + gap)
    sheet.save(path)


def write_outputs(
    result: PlanResult, out_dir: str | Path, source: str | None = None, breakdown: str = "thickness"
) -> dict[str, Path]:
    """Write layer maps, color renderings, preview, error map and recipe.

    Returns the paths written. ``source`` (the input image path) is recorded in
    the recipe if given.

    Besides the spec's outputs, each layer gets two color renderings, and
    ``overview.png`` shows them all on one sheet:

    - ``layers_color/NN_<paint>.png``: that layer alone, at its thicknesses,
      over white or black, whichever shows it more clearly (white paint over
      white would be invisible). The recipe records which.
    - ``steps/NN_<paint>.png``: the painting after layers 1..NN over the real
      background. The last step equals ``preview.png``.
    - ``swatches/NN_<paint>.png``: what the layer's paint should look like, at
      full strength and at each thickness the layer uses, over white, over
      black and over the background (if it is a flat color other than white or
      black). The recipe lists the same colors as hex.

    ``breakdown`` chooses how each layer is split further ("thickness",
    "passes", "both" or "none"):

    - thickness (default): ``thickness/NN_<paint>_<coats>coats.png`` is each
      thickness of the layer as its own layer: the region painted at exactly
      that thickness, in color, alone over the layer's ground. ``..._mask.png``
      marks the region (white) and ``..._after.png`` is the painting after it
      is applied. Regions do not overlap, so each spot gets one treatment.
      ``thickness_overview.png`` shows them all.
    - passes: ``passes/NN_<paint>_passP.png``, the layer as nested uniform
      coats of its paint (see layer_passes); ``passes_overview.png``.
    """
    if breakdown not in BREAKDOWNS:
        raise ValueError(f"breakdown must be one of {BREAKDOWNS}, got {breakdown!r}")
    out = Path(out_dir)
    layers_dir, color_dir, steps_dir = out / "layers", out / "layers_color", out / "steps"
    thickness_dir, passes_dir, swatch_dir = out / "thickness", out / "passes", out / "swatches"
    for d in (layers_dir, color_dir, steps_dir, thickness_dir, passes_dir, swatch_dir):
        for old in d.glob("*.png"):  # don't leave files from an earlier run
            old.unlink()
    for d in (layers_dir, color_dir, steps_dir, swatch_dir):
        d.mkdir(parents=True, exist_ok=True)
    if breakdown in ("thickness", "both"):
        thickness_dir.mkdir(parents=True, exist_ok=True)
    if breakdown in ("passes", "both"):
        passes_dir.mkdir(parents=True, exist_ok=True)
    for old in ("overview.png", "passes_overview.png", "thickness_overview.png"):
        (out / old).unlink(missing_ok=True)

    sol, cfg, stack = result.solution, result.solve_config, result.selection.stack
    written: dict[str, Path] = {}
    if stack:
        K = torch.tensor([p.K for p in stack], dtype=torch.float64)
        S = torch.tensor([p.S for p in stack], dtype=torch.float64)
        t_all = sol.t_snapped.double()
        bg_linear = result.background.linear.double()
    overview_alone: list[tuple[str, Image.Image]] = []
    overview_steps: list[tuple[str, Image.Image]] = []
    overview_swatches: list[tuple[str, Image.Image]] = []
    tile_size = (result.target_lab.shape[2], result.target_lab.shape[1])
    flat_ground = None  # a flat background that isn't plain white or black gets its own swatch row
    if result.background.linear.ndim == 1:
        g = result.background.linear.double()
        if not (torch.allclose(g, torch.ones(3, dtype=g.dtype), atol=1e-3) or torch.allclose(g, torch.zeros(3, dtype=g.dtype), atol=1e-3)):
            flat_ground = g
    pass_sections: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    thickness_sections: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    layer_entries = []
    if result.background.kind == "paint":
        p = result.background.paint
        layer_entries.append(
            {"layer": 0, "paint": p.name, "item": p.item, "transparency": p.transparency, "sheen": p.sheen,
             "role": "ground", "how": "opaque coat over the whole surface; let dry before layer 1"}
        )  # fmt: skip

    pixels = sol.t_snapped[0, 0].numel() if stack else 1
    for i, paint in enumerate(stack, start=1):
        t = sol.t_snapped[i - 1, 0]
        name = layer_filename(i, len(stack), paint)
        _save_gray(t / cfg.t_max, layers_dir / name)
        alone, ground = _layer_alone(K[i - 1 : i], S[i - 1 : i], t_all[i - 1 : i])
        alone_img, step_img = _to_pil(alone), _to_pil(km.render_stack(bg_linear, K[:i], S[:i], t_all[:i]))
        alone_img.save(color_dir / name)
        step_img.save(steps_dir / name)
        overview_alone.append((f"{i}. {paint.name}", alone_img))
        overview_steps.append((f"after {i}", step_img))
        written[f"layer_{i}"] = layers_dir / name
        written[f"layer_color_{i}"] = color_dir / name
        written[f"step_{i}"] = steps_dir / name
        usage = {f"{lv:g}": round((t == lv).sum().item() / pixels, 4) for lv in cfg.levels if lv > 0 and (t == lv).any()}

        ground_bg = torch.full((3,), 1.0 if ground == "white" else 0.0, dtype=torch.float64)
        K_i, S_i = K[i - 1 : i], S[i - 1 : i]

        used = [float(lv) for lv in cfg.levels if lv > 0 and (t == lv).any()]
        colors = swatch_colors(paint, used, flat_ground)
        rows = 2 + (flat_ground is not None)
        title = f"{i}. {paint.name} ({paint.transparency})"
        card_size = (max(400, 90 + 76 * (len(used) + 1)), 50 + 60 * rows)
        _swatch_card(paint, colors, card_size, title, font_size=14).save(swatch_dir / name)
        swatch_tile = _swatch_card(paint, colors, tile_size, title)
        overview_swatches.append(("", swatch_tile))
        written[f"swatch_{i}"] = swatch_dir / name

        entry = {
            "layer": i, "file": f"layers/{name}", "color_file": f"layers_color/{name}", "color_file_ground": ground,
            "step_file": f"steps/{name}", "swatch_file": f"swatches/{name}", "paint": paint.name, "item": paint.item,
            "transparency": paint.transparency, "sheen": paint.sheen, "lightfastness": paint.lightfastness,
            "coverage": round((t > 0).sum().item() / pixels, 4), "coats_by_area": usage,
            "swatch": _swatch_entry(colors),
        }  # fmt: skip

        if breakdown in ("thickness", "both"):
            before_img = _to_pil(km.render_stack(bg_linear, K[: i - 1], S[: i - 1], t_all[: i - 1]))
            entries, alone_tiles, after_tiles = [], [("", swatch_tile)], [("before this layer", before_img)]
            for coats, mask in layer_thicknesses(t, cfg.levels):
                base = thickness_basename(i, len(stack), paint, coats)
                t_region = (coats * mask.double()).reshape(1, 1, *mask.shape)
                layer_so_far = t_all[i - 1] * (t_all[i - 1] <= coats)  # this layer's regions up to this thickness
                t_after = torch.cat([t_all[: i - 1], layer_so_far.unsqueeze(0)])
                alone_img = _to_pil(km.render_stack(ground_bg, K_i, S_i, t_region))
                after_img = _to_pil(km.render_stack(bg_linear, K[:i], S[:i], t_after))
                alone_img.save(thickness_dir / f"{base}.png")
                after_img.save(thickness_dir / f"{base}_after.png")
                _save_gray(mask.float(), thickness_dir / f"{base}_mask.png")
                alone_tiles.append((_coats_label(coats), alone_img))
                after_tiles.append((f"after {_coats_label(coats)}", after_img))
                entries.append(
                    {"coats": coats, "how": level_instructions(coats), "area": round(mask.sum().item() / pixels, 4),
                     "file": f"thickness/{base}.png", "mask_file": f"thickness/{base}_mask.png",
                     "after_file": f"thickness/{base}_after.png"}
                )  # fmt: skip
                written[f"thickness_{i}_{coats:g}"] = thickness_dir / f"{base}.png"
            if len(alone_tiles) > 1:
                thickness_sections.append((f"{i}. {paint.name}: swatch, then each thickness on its own (over {ground})", alone_tiles))
                thickness_sections.append((f"{i}. {paint.name}: the painting after each thickness is applied", after_tiles))
            entry["thicknesses"] = entries

        if breakdown in ("passes", "both"):
            pass_entries, pass_tiles = [], []
            for p, (coats, mask) in enumerate(layer_passes(t, cfg.levels), start=1):
                fname = pass_filename(i, len(stack), paint, p)
                _save_gray(mask.float(), passes_dir / fname)
                t_pass = (coats * mask.double()).reshape(1, 1, *mask.shape)
                pass_tiles.append((f"pass {p}: {_coats_label(coats)}", _to_pil(km.render_stack(ground_bg, K_i, S_i, t_pass))))
                pass_entries.append(
                    {"pass": p, "coats": coats, "how": level_instructions(coats),
                     "area": round(mask.sum().item() / pixels, 4), "file": f"passes/{fname}"}
                )  # fmt: skip
                written[f"pass_{i}_{p}"] = passes_dir / fname
            if pass_tiles:
                pass_sections.append((f"{i}. {paint.name}: {len(pass_tiles)} passes, each inside the last (over {ground})", pass_tiles))
            entry["passes"] = pass_entries

        layer_entries.append(entry)

    _save_rgb(sol.preview, out / "preview.png")
    if stack:
        target_img = Image.fromarray((result.image_srgb.clamp(0, 1).permute(1, 2, 0) * 255).round().to(torch.uint8).numpy(), mode="RGB")
        _overview(
            [
                ("Paint swatch for each layer: full strength, and each thickness used over white, black and the ground", overview_swatches),
                ("Each layer alone (over white or black, whichever shows it)", overview_alone),
                (f"The painting after each layer, over the background ({result.background.description})", overview_steps),
                ("Target, finished preview, color error (white = 20 dE or more)",
                 [("target", target_img), ("preview", Image.open(out / "preview.png").convert("RGB")),
                  ("error", Image.fromarray((sol.error / ERROR_PNG_SCALE).clamp(0, 1).mul(255).round().to(torch.uint8).numpy(), mode="L").convert("RGB"))]),
            ],
            out / "overview.png",
        )  # fmt: skip
        written["overview"] = out / "overview.png"
    if pass_sections:
        _overview(pass_sections, out / "passes_overview.png", columns=max(len(tiles) for _, tiles in pass_sections))
        written["passes_overview"] = out / "passes_overview.png"
    if thickness_sections:
        cols = max(len(tiles) for _, tiles in thickness_sections)
        _overview(thickness_sections, out / "thickness_overview.png", columns=cols)
        written["thickness_overview"] = out / "thickness_overview.png"
    _save_gray(sol.error / ERROR_PNG_SCALE, out / "error.png")
    written["preview"], written["error"] = out / "preview.png", out / "error.png"

    bg = result.background
    bg_entry = {"kind": bg.kind, "value": bg.description}
    if bg.linear.ndim == 1:
        bg_entry["srgb_hex"] = _hex(color.working_to_srgb(bg.linear))
        bg_entry["lab"] = [round(v, 2) for v in color.working_to_lab(bg.linear).tolist()]

    notes = []
    surface = list(stack) + ([bg.paint] if bg.paint else [])
    sheens = sorted({p.sheen for p in surface if p.sheen})
    if len(sheens) > 1:
        notes.append(f"Paints with different sheens ({', '.join(sheens)}): finish with a varnish for an even surface.")
    weak = sorted({f"{p.name} ({p.lightfastness})" for p in surface if p.lightfastness in ("Fair", "Poor", "Very Poor")})
    if weak:
        notes.append(f"Low lightfastness, may fade: {', '.join(weak)}.")
    if result.selection.forced_first is not None:
        notes.append(f"{result.selection.forced_first.name} was placed first because the background is darker than the image.")

    recipe = {
        "image": {"width": result.target_lab.shape[2], "height": result.target_lab.shape[1], **({"source": source} if source else {})},
        "background": bg_entry,
        "layers": layer_entries,
        "thickness_levels": [
            {"coats": lv, "pixel_value": round(lv / cfg.t_max * 255), "how": level_instructions(lv)} for lv in cfg.levels
        ],
        "error": {
            "mean": round(sol.mean_error, 3),
            "max": round(sol.max_error, 3),
            "mean_before_quantizing": round(sol.mean_error_continuous, 3),
            "max_before_quantizing": round(sol.max_error_continuous, 3),
            "error_png_white_at": ERROR_PNG_SCALE,
            "metric": "CIE76 ΔE",
        },
        "selection": {
            "colors_clustered": int(result.selection.cluster_lab.shape[0]),
            "rms_error_background_only": round(result.selection.baseline_error, 3),
            "rounds": [{**r, "rms_error": round(r["rms_error"], 3)} for r in result.selection.rounds],
            "paints_used": sorted({p.name for p in stack}),
        },
        "settings": {"solve": asdict(cfg), "palette": asdict(result.palette_config)},
        "notes": notes,
    }
    (out / "recipe.json").write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written["recipe"] = out / "recipe.json"
    return written
