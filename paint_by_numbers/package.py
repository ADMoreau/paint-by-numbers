"""Project package for the phone app: one file holding everything needed to paint along.

``project.pbn`` is a zip of:

- ``package.json``: format name and version, and which file is the target image;
- ``recipe.json``: the plan (layers, paints, thickness levels, swatch colors);
- ``target.png``: the source image (longer side at most 2048 px), shown when placing the image;
- ``preview.png``: the predicted finished painting;
- per layer, its thickness map (``layers/``: gray value = thickness, see the recipe's
  ``thickness_levels``), the painting after it (``steps/``) and its swatch card (``swatches/``).

The app derives each thickness region from the layer's map, so the per-thickness images are
not included.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from PIL import Image

PACKAGE_NAME = "project.pbn"
FORMAT = "paint-by-numbers project"
VERSION = 1
LAYER_FILES = ("file", "step_file", "swatch_file")


def write_package(out_dir: str | Path, source: str | Path | None = None, max_side: int = 2048) -> Path:
    """Bundle the outputs in ``out_dir`` (after ``write_outputs``) into ``out_dir/project.pbn``."""
    out = Path(out_dir)
    recipe = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
    path = out / PACKAGE_NAME
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out / "recipe.json", "recipe.json")
        z.write(out / "preview.png", "preview.png")
        for layer in recipe["layers"]:
            for key in LAYER_FILES:
                name = layer.get(key)
                if name and (out / name).is_file():
                    z.write(out / name, name)
        target = "preview.png"
        if source is not None and Path(source).is_file():
            with Image.open(source) as im:
                im = im.convert("RGB")
                im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format="PNG")
            z.writestr("target.png", buf.getvalue())
            target = "target.png"
        z.writestr("package.json", json.dumps({"format": FORMAT, "version": VERSION, "target_file": target}, indent=2))
    return path
