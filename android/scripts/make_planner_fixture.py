"""Write the fixture the phone planner's unit tests compare against (app/src/test/resources/planner_fixture.json).

Run from the repository root with the desktop planner's environment:

    python android/scripts/make_planner_fixture.py

It records the desktop planner's answers for small cases: colour conversions, single paint
layers, the guided filter, region cleanup, and a whole one-weight-per-layer plan of the synthetic
landscape (with the k-means colours, so the phone code can start from the same clusters).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from core_paints import CORE_NAMES, core_paints  # noqa: E402
from images import landscape  # noqa: E402
from paint_by_numbers import color, km, plan, uniform  # noqa: E402
from paint_by_numbers.palette import cluster_colors  # noqa: E402

OUT = ROOT / "android/app/src/test/resources/planner_fixture.json"


def bits(mask) -> str:
    return "".join("1" if v else "0" for v in np.asarray(mask).ravel())


def main():
    rng = np.random.default_rng(0)
    paints = core_paints()
    fx = {}

    srgb = rng.random((8, 3))
    fx["color"] = {
        "srgb": srgb.tolist(),
        "srgb_to_lab": color.srgb_to_lab(torch.from_numpy(srgb)).tolist(),
        "srgb_to_working": color.srgb_to_working(torch.from_numpy(srgb)).tolist(),
        "working_to_lab": color.working_to_lab(torch.from_numpy(srgb)).tolist(),
        "working_to_srgb": color.working_to_srgb(torch.from_numpy(srgb)).tolist(),
    }

    rg = rng.random(3)
    fx["km"] = {
        "rg": rg.tolist(),
        "cases": [
            {"paint": p.name, "t": t,
             "layer": km.layer(torch.from_numpy(rg), torch.tensor(p.K, dtype=torch.float64), torch.tensor(p.S, dtype=torch.float64),
                               torch.tensor(t, dtype=torch.float64)).tolist(),
             "masstone": km.masstone(torch.tensor(p.K, dtype=torch.float64), torch.tensor(p.S, dtype=torch.float64)).tolist()}
            for p in paints[:4] for t in (0.125, 1.0, 2.0)
        ],
    }  # fmt: skip

    h, w = 12, 10
    guide = rng.random((h, w, 3)) * 50
    guide[:, w // 2 :] += 30
    p = rng.standard_normal((h, w))
    fx["guided_filter"] = {
        "w": w, "h": h, "radius": 2, "eps": 25.0, "guide": guide.ravel().tolist(), "p": p.ravel().tolist(),
        "out": uniform.guided_filter(guide, 2, 25.0)(p).ravel().tolist(),
    }  # fmt: skip

    h, w = 20, 16
    mask = rng.random((h, w)) < 0.45
    mask[3:5, 3:5] = True
    cleaned = uniform.clean_regions(torch.from_numpy(mask).reshape(1, 1, h, w), 5)[0, 0].numpy()
    fx["clean"] = {"w": w, "h": h, "min_area": 5, "mask": bits(mask), "out": bits(cleaned)}

    img = landscape(64, 48)
    image = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1).contiguous()
    config = uniform.UniformConfig(layers=5, n_colors=16, improve_passes=1)
    bg = plan.background_from_color("#808080")
    target = color.srgb_to_lab(image.float(), dim=0)
    centers, labels = cluster_colors(target, config.n_colors, config.seed)
    log = []
    sel = uniform.select_uniform_stack(target, bg.linear, paints, config, log.append)
    sol = uniform.solve_uniform(target, bg.linear, sel, config)
    regions = (sol.t > 0)[:, 0]
    arr = np.asarray(img, dtype=np.uint8)
    fx["plan"] = {
        "w": 64, "h": 48,
        "pixels": [int(r) << 16 | int(g) << 8 | int(b) for r, g, b in arr.reshape(-1, 3)],
        "paints": list(CORE_NAMES),
        "background_hex": "#808080",
        "config": {"layers": config.layers, "n_colors": config.n_colors, "improve_passes": config.improve_passes},
        "centers": centers.tolist(),
        "labels": labels.tolist(),
        "baseline": sel.baseline_error,
        "rounds": [{"paint": r["paint"], "coats": r["coats"], "position": r["position"], "rms_error": r["rms_error"]} for r in sel.rounds],
        "log": log,
        "stack": [{"paint": p.name, "coats": wt} for p, wt in zip(sel.stack, sel.weights)],
        "assignment": [bits(a[0]) for a in sel.assignment],
        "regions": [bits(r) for r in regions],
        "mean_error": sol.mean_error,
        "mean_error_from_colors": sol.mean_error_nearest,
    }  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fx) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: stack {[(s['paint'], s['coats']) for s in fx['plan']['stack']]}, mean ΔE {sol.mean_error:.3f}")


if __name__ == "__main__":
    main()
