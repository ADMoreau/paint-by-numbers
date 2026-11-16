"""Simulation experiments for aligning a target image to a sketch on the canvas.

    python run_experiments.py --images print2=path.png [name=path ...] --out results/ [--n 10]

The synthetic landscape from the planner's tests is always included.

E1  Sketch style: careful / typical / loose freehand (snap from a rough placement).
E2  Finding the image with no help: snap vs. upright search vs. any rotation.
E3  Extra lines: none / X's + guides + hatching / a transfer grid.
E4  Raw photo (no tracking: the phone on a stand, perspective fitted directly).
E5  Graphite on a white canvas instead of white pencil on mauve.

Writes summary.md and results.json to --out.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per worker process

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from salign import align as al  # noqa: E402
from salign import photo as ph  # noqa: E402
from salign import sketch as sk  # noqa: E402
from salign import trial  # noqa: E402

IMAGES: dict = {}


def _init(images):
    cv2.setNumThreads(1)
    IMAGES.update(images)


@lru_cache(maxsize=8)
def _target(name: str) -> np.ndarray:
    if name == "landscape":
        from images import landscape

        return cv2.cvtColor(np.asarray(landscape(1200, 900)), cv2.COLOR_RGB2BGR)
    return cv2.imread(IMAGES[name])


@lru_cache(maxsize=8)
def _model(name: str) -> al.EdgeModel:
    return al.EdgeModel(_target(name))


def _one(job):
    exp, cond, image, setup, seed = job
    t = time.perf_counter()
    r = trial.run(_target(image), seed, setup, model=_model(image))
    d = asdict(r)
    d.update(exp=exp, cond=cond, image=image, seed=seed, wall=time.perf_counter() - t)
    return d


BASE = trial.Setup()


def experiments(images: list, n: int) -> list:
    jobs = []

    def add(exp, cond, setup, imgs=images, count=n):
        for im in imgs:
            for s in range(count):
                jobs.append((exp, cond, im, setup, 1000 * len(jobs) + s))

    for name, style in sk.STYLES.items():
        add("E1", name, replace(BASE, style=style))
    for mode in ("upright", "any"):
        add("E2", mode, replace(BASE, search=mode))
    for name, cl in sk.CLUTTERS.items():
        if name != "usual":  # "usual" is E1 typical
            add("E3", name, replace(BASE, clutter=cl))
    add("E4", "raw photo", replace(BASE, mode="free"))
    add("E5", "white canvas, graphite", replace(BASE, look=ph.WHITE_GRAPHITE))
    return jobs


def summarize(rows: list) -> str:
    def med(v):
        return float(np.median(v)) if len(v) else float("nan")

    out = ["# Sketch alignment: simulation results", ""]
    out.append("Errors are in mm on a 24 x 18 in canvas, averaged over the parts of the image within 4% of the diagonal of something sketched; the median over trials is shown. "
               "**Found** = the right placement (error within twice the sketch's own drift + 3 mm). "
               "**Drift** = the best any single affine (or, for a raw photo, perspective) map could do: how far the freehand sketch strays from the image. "
               "**Placed** = the fitted placement alone; **Bent** = with the smooth bending on top.")  # fmt: skip
    out.append("")
    titles = {
        "E1": "E1: sketch style (snap from a rough placement)",
        "E2": "E2: finding the image with no help (typical sketch)",
        "E3": "E3: extra lines on the canvas (typical sketch, snap)",
        "E4": "E4: raw photo, no tracking (typical sketch, snap)",
        "E5": "E5: graphite on white canvas (typical sketch, snap)",
    }
    for exp in sorted({r["exp"] for r in rows}):
        out += [f"## {titles[exp]}", "", "| condition | image | found | drift | placed | bent | bent, 90th pct (whole image) | time (s) |", "|---|---|---|---|---|---|---|---|"]
        sel = [r for r in rows if r["exp"] == exp]
        for cond in dict.fromkeys(r["cond"] for r in sel):
            for im in dict.fromkeys(r["image"] for r in sel):
                rr = [r for r in sel if r["cond"] == cond and r["image"] == im]
                if not rr:
                    continue
                ok = [r for r in rr if r["found"]]
                fk = "near_homography" if any("near_homography" in r["floor"] for r in rr) else "near_affine"
                secs = med([sum(v for k, v in r["seconds"].items() if k != "model") for r in rr])
                out.append(
                    f"| {cond} | {im} | {len(ok)}/{len(rr)} | {med([r['floor'][fk] for r in rr]):.1f} | "
                    f"{med([r['near']['rigid'] for r in ok]):.1f} | {med([r['near']['bent'] for r in ok]):.1f} | {med([r['p90']['bent'] for r in ok]):.1f} | {secs:.1f} |"
                )
        out.append("")
    # does the score margin flag the misses?
    found = [r["second"] - r["score"] for r in rows if r["found"]]
    missed = [r["second"] - r["score"] for r in rows if not r["found"]]
    out += ["## Telling a good result from a bad one", ""]
    out.append(f"Score margin (best clearly different placement minus the chosen one; bigger = more certain): found median {med(found):.3f} (5th pct {np.percentile(found, 5) if found else float('nan'):.3f}), missed median {med(missed):.3f} (95th pct {np.percentile(missed, 95) if missed else float('nan'):.3f}).")
    for thr in (0.05, 0.08, 0.1):
        acc = sum(1 for r in rows if r["second"] - r["score"] >= thr)
        bad = sum(1 for r in rows if r["second"] - r["score"] >= thr and not r["found"])
        out.append(f"- accept when margin >= {thr}: {acc}/{len(rows)} accepted, {bad} of them wrong")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=[], help="name=path target images (the landscape is always included)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    images = dict(a.split("=", 1) for a in args.images)
    names = ["landscape", *images]
    jobs = experiments(names, args.n)
    print(f"{len(jobs)} trials", flush=True)
    rows = []
    t = time.time()
    with ProcessPoolExecutor(args.workers or None, initializer=_init, initargs=(images,)) as ex:
        for i, r in enumerate(ex.map(_one, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t:.0f}s", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=1, default=float))
    md = summarize(rows)
    (out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
