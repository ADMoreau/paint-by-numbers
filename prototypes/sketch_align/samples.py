"""Example pictures: a simulated sketch, what was found, and the image laid onto it.

    python samples.py --out results/samples.jpg [--image path.png] [--style typical] [--seed 1]

Without --image, the planner's synthetic landscape is used.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from salign import align as al  # noqa: E402
from salign import sketch as sk  # noqa: E402
from salign import trial, viz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--out", default="results/samples.jpg")
    ap.add_argument("--style", default="typical")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--search", default="snap")
    args = ap.parse_args()
    if args.image:
        target = cv2.imread(args.image)
    else:
        from images import landscape

        target = cv2.cvtColor(np.asarray(landscape(1200, 900)), cv2.COLOR_RGB2BGR)
    model = al.EdgeModel(target)
    setup = replace(trial.Setup(), style=sk.STYLES[args.style], search=args.search)
    rows = []
    for seed in args.seeds:
        r, d = trial.run(target, seed, setup, model=model, keep=True)
        print(f"seed {seed}: found={r.found} near-content error placed {r.near['rigid']:.1f} mm, bent {r.near['bent']:.1f} mm (sketch drift {r.floor['near_affine']:.1f} mm)")
        rows.append(viz.panels(d["work"], d["lines"], target, model, d["align"].fit))
    gap = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.vstack([out, gap, r])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
