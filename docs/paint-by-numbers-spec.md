# Paint by Numbers: build spec

This file is a handoff to a coding session. Build the program it describes, milestone by milestone, and run the acceptance checks at each one.

## What to build

A command-line program that takes an image, a set of Golden acrylic paints and a background color, and writes one grayscale image per paint layer. Each layer image says where that paint goes and how thickly, the way one channel of an RGB image says how much red goes where.

Outputs per run:

- `layers/NN_<paint-slug>.png`: one per layer, bottom layer first. Pixel value is thickness: black is no paint, white is the maximum number of coats. `NN` is zero-padded to at least two digits, and to more when there are 100 or more layers.
- `preview.png`: the simulated finished stack over the chosen background.
- `recipe.json`: background, layer order, paint per layer, thickness levels, mean and max color error.
- `error.png`: per-pixel color error between preview and target.
- Added for viewing: `layers_color/NN_<paint-slug>.png` (each layer alone in its paint color, over white or black, whichever shows it with more contrast), `steps/NN_<paint-slug>.png` (the stack after layers 1..NN over the background), and `overview.png` (all of these plus target, preview and error on one sheet). `swatches/NN_<paint-slug>.png` shows what each layer's paint should look like: full strength, and each thickness the layer uses over white, over black and over a flat background; `recipe.json` lists the same colors as hex. Each layer is broken down further with `--breakdown`: by default `thickness/NN_<paint-slug>_<coats>coats.png` shows each thickness as its own layer (the area at exactly that thickness, in color, plus a mask and the painting after it is applied; the areas don't overlap), with `thickness_overview.png`. With `--breakdown passes`, `passes/NN_<paint-slug>_passP.png` breaks each layer into uniform coats of its paint: pass P adds the step from the previous level to level P wherever the layer is at least that thick, so each pass lies inside the previous one, and because adjacent coats of one paint add up, the passes reproduce the layer exactly. `passes_overview.png` shows them in color.

## Scope

In scope: color planning with layered single-paint layers over a user-chosen background.

Out of scope, do not build:

- Brush strokes or stroke direction.
- Any workflow that needs physical paint measurement.
- Exact agreement with a hand-painted result. Approximate is accepted.

## Layer count

There is no fixed limit on the number of layers. The stack renderer, solver, palette selection and outputs must all take any n ≥ 1 and must not assume a small or fixed n. Layers keep being added while each one still improves the match (see Palette and order). Per-layer thickness is a separate limit (`t_max`, in coats).

Runtime grows roughly linearly with n. There is no time limit: working results matter more than speed.

## Environment

- Python 3.10+, PyTorch (CPU build), NumPy, Pillow, scikit-learn (k-means only), pytest.
- Must run on a laptop CPU with no discrete GPU. Solve at 256 px on the long side by default.
- No network access at run time. Paint data lives in the repo as JSON.

## Repo layout

```
paint_by_numbers/
  data/paints_heavy_body.json   # paint database
  color.py                      # sRGB <-> linear RGB <-> XYZ <-> Lab (torch, differentiable)
  km.py                         # Kubelka-Munk layer, stack renderer, mixing
  fit.py                        # estimate K and S per paint from published data
  palette.py                    # palette selection and layer ordering
  solve.py                      # thickness-map optimizer, quantization
  cli.py                        # entry point
tests/
```

## Paint database

One record per paint. K and S are filled in by `fit.py`, not by hand.

```json
{
  "name": "Green Gold",
  "item": 1170,
  "line": "Heavy Body",
  "lab": [45.05, -10.69, 30.52],
  "srgb": [106, 110, 54],
  "transparency": "transparent",
  "sheen": "gloss",
  "pigments": ["PY150", "PG36", "PY175"],
  "K": null,
  "S": null
}
```

`transparency` is one of `opaque`, `semi-opaque`, `semi-transparent`, `transparent`.

Seed values. Green Gold was read from Golden's product page. The other three came from search listings: re-check each against its product page before relying on it.

| Paint | L\* | a\* | b\* | Transparency |
| --- | --- | --- | --- | --- |
| Green Gold | 45.05 | -10.69 | 30.52 | transparent |
| Ultramarine Blue | 24.11 | 14.01 | -27.81 | semi-transparent |
| Burnt Umber | 27.33 | 4.10 | 4.19 | opaque |
| Cadmium Yellow Primrose | 90.16 | -10.32 | 87.73 | opaque |

Titanium White is required and is not in the seed table. Add it and at least a black, a red and a blue, taking values from each color's page at `goldenartistcolors.com/products/heavy-body-acrylic-color-<name>`. Enter them by hand or with a small one-off script; keep requests few and cache the results in the JSON.

Skip any paint whose published L\*a\*b\* is all zeros (interference colors).

Status: the database holds all 129 Heavy Body colors that publish color data. 27 products without it (interference, iridescent, mica flakes, micaceous iron oxide, coarse alumina, and discontinued Hansa Yellow Medium) are skipped. Records also carry `lightfastness`, `discontinued` and `special` (`fluorescent` or `phosphorescent`). The default palette leaves out discontinued and special paints; they are used only when named in `--paints`.

## Color model

Work in three channels of linear RGB with Rec.2020 (ITU-R BT.2020) primaries, sharing Lab's D65 white. Thickness `t` is in coats; 1.0 is one full brushed coat. Images come in and go out as sRGB; only paint reflectance lives in Rec.2020. Linear sRGB was too narrow: 20 of Golden's Heavy Body colors (cadmium and other strong yellows, cobalt teals and turquoises) fall outside it and were clipped by up to 15 Lab units. In Rec.2020 every non-fluorescent paint fits, and a thick layer renders to its published color.

One layer of paint (K, S, thickness t) over a surface with reflectance `Rg`, per channel:

```
a  = 1 + K / S
b  = sqrt(a^2 - 1)
th = tanh(b * S * t)
R  = ((1 - a*Rg) * th + b*Rg) / ((a - Rg) * th + b)
```

Clamp K and S to at least 1e-4 before use. Properties that must hold:

- `t = 0` returns `Rg` exactly.
- Large `t` returns `a - b`, the paint's masstone.

The stack: `R0 = background`, then `Rk = layer(R(k-1), Kk, Sk, tk)` for k = 1..n, bottom first.

Mixtures (used only when fitting tints, and for optional pre-mixed layers): `K_mix = sum(c_i * K_i)`, `S_mix = sum(c_i * S_i)`.

## Background

The background is a required user setting with a default of white. It is the reflectance under the first layer and enters the renderer in exactly one place. Support three forms:

1. `--background "#7A6A58"`: a hex or sRGB color.
2. `--background-paint "Burnt Umber"`: an opaque coat of a database paint. Use its masstone color and record it as layer 0 in `recipe.json`.
3. `--background-image ground.png`: a per-pixel background, resized to the working resolution.

Palette selection and layer ordering must run with the background in place, because the best paints and order depend on it.

## Estimating K and S (`fit.py`)

1. Convert the paint's published color to linear RGB reflectance `R`, clipped to [0.01, 0.99].
2. `K/S = (1 - R)^2 / (2R)` per channel.
3. S per coat, calibrated per paint by coverage. The transparency rating sets how many coats hide a black-and-white chart: after that many coats, the layer is within 2 Lab units of the paint's masstone over both black and white. Find the S that meets it by bisection, with K/S held fixed so the masstone does not move. S is the same on all three channels. These are first guesses; keep them in one config dict so they are easy to tune:

   | Rating | Coats to hide |
   | --- | --- |
   | opaque | 1 |
   | semi-opaque | 2 |
   | semi-transparent | 5 |
   | transparent | 20 |

   The ratios follow the original first-guess table of one S per rating (3.0, 1.5, 0.6, 0.15). That table made dark paints hide almost at once and light ones poorly, because hiding depends on `b * S` and `b` is small for light colors. Two coats of Titanium White over black reached only L\* 94 of its 98.

4. `K = (K/S) * S`.

Later refinement, not needed for the first working version: adjust K and S per paint so the renderer also reproduces the tint and glaze swatch shown on each product page. Fit Titanium White first.

## Palette and order (`palette.py`)

1. K-means the target image into 64 colors in Lab.
2. Greedy selection builds the layer stack directly. Each round, try every candidate paint at every insertion position in the current stack. Solve the 64 colors as a 64 by 1 image over the background, quantize to the solver's levels, and keep the (paint, position) pair that most reduces the RMS color error over the 64 colors. With a per-pixel background, each color uses the mean background of its pixels.
   - RMS, not the plain mean of the first draft: a small vivid feature (a sun, one color of 64) that is 30 Lab units off moves the mean by under 0.5, so it never earned a layer.
   - Paints may repeat: one paint can appear in any number of layers.
   - Skip a candidate that would sit directly on or under a layer of the same paint. Two adjacent layers of one paint render the same as one thicker layer.
   - Solve all candidates in a round as one batched image, so a round costs one solve. The renderer accepts per-pixel K and S for this.
   - Stop when the best candidate reduces RMS error by less than `--min-gain` (default 0.5 Lab units), or at `--max-layers` if given (default: no limit).
3. Order comes from the position search. Break ties between equal-error positions by keeping opaque paints below transparent ones and, among equal ratings, lighter paints lower.
4. If the background's L\* is below the image's median L\*, start the stack with the lightest opaque paint available before the first round.

## Solver (`solve.py`)

- Parameters: `u` of shape (n, 1, H, W); thickness `t = t_max * sigmoid(u)`, `t_max` default 2.0. Initialize `u = -10` (about zero paint). The first draft's `u = -3` puts 0.095 coats on every layer, which grows with layer count, and it often settled on a paint mixture where no paint was the answer.
- Loss: mean squared Lab distance between rendered stack and target, plus `lam_smooth` times total variation of `t`, plus `lam_paint` times mean paint. Defaults 0.05 and 0.01; expose both as flags. Both penalties use each pixel's total across all layers, averaged over pixels, so their weight does not shrink as layers are added.
- Optimizer: Adam, lr 0.05, betas (0.9, 0.99), 2000 steps default. With the default 0.999, early large gradients shrink later steps about 1000 times and the solve stalls.
- Quantize: snap `t` to the levels 0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2 coats, then refine by per-pixel coordinate descent over the levels from four starts (nearest, floor, ceiling, all zero), minimizing ΔE plus 0.1 per coat so paint hidden under an opaque layer is dropped, plus `--smoothing` (default 2) Lab units for each neighboring pixel on a different thickness in the same layer, so regions come out in one piece instead of speckled (updated in a checkerboard). `--max-thicknesses N` limits each layer to the N nonzero levels that best fit its continuous thickness. On the test render, smoothing 2 cut separate regions by about 80% for 0.3 Lab units of mean error; re-render, and report error before quantizing, after nearest snapping and after refinement. Nearest snapping alone can lose up to 30 Lab units, because the continuous optimum often uses traces of high-S paints that round to zero. Levels are configurable. The first draft had five (0, 0.25, 0.5, 1, 2); adding 0.125, 0.75 and 1.5 roughly halved quantization error on a smooth gradient. Levels below one coat can be painted as one coat of glaze: 0.125 is 1 part paint to 7 parts medium.

## CLI

```
python -m paint_by_numbers.cli input.jpg --out out/ \
  --background "#7A6A58" --size 256 \
  [--max-layers N] [--min-gain 0.5] \
  [--paints "Titanium White,Ultramarine Blue,..."] \
  [--background-paint NAME | --background-image PATH]
```

`--paints` restricts selection to paints the user owns; `--paints-file` reads the same list from a text file, and `--max-paints N` caps how many different paints the plan uses (once N are in the stack, only those can be added again). Exactly one background option may be given; with none, use white.

## Milestones and acceptance checks

1. **Color conversions.** Round trip sRGB to Lab to sRGB within 1/255 on random colors.
2. **KM renderer.** Tests: `t = 0` returns the background; `t = 50` returns the masstone within 1e-3; output stays in [0, 1]; gradients are finite at `t = 0`.
3. **Paint fitting.** For every database paint, a thick layer over white and over black both render to its published color within 2 Lab units.
4. **Solver.** A target equal to the background solves to all layers below 0.05 coats. A flat target equal to one palette paint's masstone solves to mean error under 3 Lab units. Use opaque paints for the masstone check: a transparent paint needs about 20 coats to reach its masstone, far past `t_max`.
5. **Background swap.** The same test image solved over white, mid-gray `#808080` and black produces different layer stacks, and each run writes all outputs.
6. **CLI end to end.** One command on a sample image writes layers, preview, recipe and error map on CPU at 256 px. The run time is printed for information; there is no time limit.

## Known limits to note in the README

- Three-channel KM is an approximation; some mixtures will be off in hue.
- K and S are estimated from published data, and the transparency-to-S table is the least certain part.
- Some saturated screen colors are unreachable with paint; `error.png` shows where.
- Fluorescent and phosphorescent paints emit light (published L\* up to 106), so a reflectance model cannot reproduce them; they are used only when named.
- Sheen is not modeled. If the palette mixes matte and gloss paints, `recipe.json` should carry a note suggesting a final varnish.

## Open questions, do not block on these

- Illuminant and observer of Golden's published L\*a\*b\* values. Assume D65, 2 degree until checked.
- Film thickness and surface behind Golden's masstone values.
- Paint-to-white and paint-to-medium ratios in the tint and glaze swatches.