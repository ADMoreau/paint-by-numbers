import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

from salign import align as al  # noqa: E402
from salign import sketch as sk  # noqa: E402
from salign import trial  # noqa: E402
from salign.lines import sketch_lines  # noqa: E402


def _landscape():
    from images import landscape

    return cv2.cvtColor(np.asarray(landscape(600, 450)), cv2.COLOR_RGB2BGR)


def test_finds_light_lines_but_not_the_edge_of_a_painted_area():
    img = np.full((200, 300), 90, np.uint8)
    img[:, 200:] = 200  # a light area (e.g. bare canvas beyond the paint)
    cv2.line(img, (20, 100), (180, 100), 190, 3, cv2.LINE_AA)  # a chalk line
    img = cv2.add(img, np.random.default_rng(0).normal(0, 2, img.shape).clip(0, 255).astype(np.uint8))
    lp = sketch_lines(img, "light", 3.0)
    on_line = np.abs(lp.pts[:, 1] - 100) <= 2
    assert on_line.sum() > 120
    assert not ((np.abs(lp.pts[:, 0] - 200) <= 3) & ~on_line).any()  # nothing along the step edge


def test_refine_recovers_a_placement():
    target = _landscape()
    model = al.EdgeModel(target, long_side=600)
    true = sk.similarity(1.3, np.radians(4), 40, 25)  # work -> model
    # sketch points: the target's own edges, mapped into work px
    e = model.edges
    q = al.apply_h(np.linalg.inv(true), e.pts)
    lines = al.LinePoints(q, e.normals @ true[:2, :2] / 1.3, e.strength)  # normals rotate with the map
    start = sk.similarity(1.2, np.radians(0), 55, 10)
    fit = al.refine(model, lines, start, "similarity")
    grid = np.stack(np.meshgrid(np.linspace(0, 400, 9), np.linspace(0, 300, 7)), -1).reshape(-1, 2)
    assert np.abs(al.apply_h(fit.G, grid) - al.apply_h(true, grid)).max() < 1.0


def test_simulated_sketch_is_found_and_placed_close_to_its_drift():
    target = _landscape()
    r = trial.run(target, 3, trial.Setup(style=sk.CAREFUL))
    assert r.found
    assert r.near["bent"] < r.floor["near_affine"] + 2.0
