"""Write a sketch-alignment test case for the app's on-device tests (SketchAlignTest).

    python export_test_case.py OUT_DIR NAME                     # simulated sketch of the synthetic landscape
    python export_test_case.py OUT_DIR NAME --photo P --target T # a real photo of a sketch

Writes NAME_photo.png (grayscale, long side 1280 like the app's camera frames), NAME_target.png
and NAME.json: the line width and polarity used, and this prototype's placement (work px -> model
px, perspective, no bending) for the Kotlin port to be compared with. For a simulated case the
JSON also holds ground truth: where the sketch put target points (``q``, work px) and the same
points in model px (``m``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from salign import align as al  # noqa: E402
from salign import sketch as sk  # noqa: E402
from salign import trial  # noqa: E402
from salign.lines import sketch_lines  # noqa: E402

LONG = 1280


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("name")
    ap.add_argument("--photo")
    ap.add_argument("--target")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--line", type=float, default=2.5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    truth = {}
    if args.photo:
        from PIL import Image, ImageOps

        photo = cv2.cvtColor(np.asarray(ImageOps.exif_transpose(Image.open(args.photo)).convert("RGB")), cv2.COLOR_RGB2BGR)
        target = cv2.imread(args.target)
        line = args.line
        s = LONG / max(photo.shape[:2])
    else:
        from images import landscape

        target = cv2.cvtColor(np.asarray(landscape(1200, 900)), cv2.COLOR_RGB2BGR)
        setup = replace(trial.Setup(), style=sk.STYLES["typical"], mode="free", search="upright", bend=False)
        _, d = trial.run(target, args.seed, setup, keep=True)
        photo = d["photo"]
        s = LONG / max(photo.shape[:2])
        line = 1.4 * setup.cam.f / np.mean(setup.distance) * s
        model = d["model"]
        near = d["near"]
        truth = {"q": (d["q_true"][near] * s).tolist(), "m": ((d["grid"][near] + 0.5) * model.f - 0.5).tolist()}
    gray = cv2.cvtColor(cv2.resize(photo, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    model = al.EdgeModel(target)
    lines = sketch_lines(gray, "light", line)
    a = al.align(model, lines, (w, h), None, "upright", "homography", bend_levels=(), scale_range=(0.3, 1.6))
    cv2.imwrite(str(out / f"{args.name}_photo.png"), gray)
    cv2.imwrite(str(out / f"{args.name}_target.png"), target)
    info = {"line_px": float(line), "light": True, "G_python": a.fit.G.ravel().tolist(), "score_python": a.fit.score, **truth}
    (out / f"{args.name}.json").write_text(json.dumps(info))
    print(f"{args.name}: {w}x{h}, {len(lines.pts)} line points, score {a.fit.score:.3f} (next {a.second_score:.3f})")
    if truth:
        p = al.apply_h(a.fit.G, np.array(truth["q"]))
        err = np.hypot(*(p - np.array(truth["m"])).T) / model.f  # target px
        print(f"  python error vs truth: median {np.median(err):.1f} target px")


if __name__ == "__main__":
    main()
