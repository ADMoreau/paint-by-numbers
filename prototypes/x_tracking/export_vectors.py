"""Export simulated frames (detections + ground truth + the Python locator's answers)
as JSON test vectors for the Android app's Kotlin port (TrackingVectorsTest).

    python prototypes/x_tracking/export_vectors.py android/app/src/androidTest/assets/tracking_vectors.json

240 single-frame lock-on cases (white paper with graphite X's and dark ground with chalk X's, at
250, 350 and 500 mm) and the 30-second painting session, all at 75 mm X spacing. Takes ~10 min.
"""

import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

cv2.setNumThreads(1)  # before any OpenCV call: its thread pool deadlocks forked workers
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as rx  # noqa: E402
from xtrack import camera, detect, locate, metrics  # noqa: E402
from xtrack import render as rd  # noqa: E402

SPACING = 75.0


def det_rows(dets):
    return [[float(d.center[0]), float(d.center[1]), float(d.directions[0]), float(d.directions[1]), float(d.size), float(d.strength)] for d in dets]


def cold_case(args):
    look, distance, seed = args
    can, cam = rx.get_canvas("grid", SPACING), camera.MAIN
    rng = np.random.default_rng(seed)
    fr = rd.render(can, cam, camera.random_pose(rng, can.size, distance), rx.LOOKS[look], rd.Effects(), rng)
    dets = detect.detect_x(fr.image, rx.POLARITY[look])
    est = rx.get_locator(SPACING, min_cold=4).cold_start(dets)
    truth = metrics.true_ids(dets, can, fr.H)
    py = {"located": est is not None}
    if est is not None:
        mean, mx = metrics.overlay_error_mm(est.H, fr.H, cam.width, cam.height)
        py.update(n=len(est.matches), id_ok=bool(metrics.identity_ok(est.matches, truth)), err_mean=mean, err_max=mx, uncertainty=est.uncertainty_mm)
    return {"look": look, "distance": distance, "seed": seed, "dets": det_rows(dets), "H": fr.H.tolist(), "truth": truth, "py": py}


def main():
    out_path = sys.argv[1]
    can, cam = rx.get_canvas("grid", SPACING), camera.MAIN
    marks = locate.build_map(can, np.random.default_rng(rx.MAP_SEED))
    cases = [(look, d, 5000 + 97 * i + int(d)) for look in ("white", "dark_chalk") for d in (250.0, 350.0, 500.0) for i in range(40)]
    n_frames = int(rx.FPS * rx.DURATION)
    with ProcessPoolExecutor(initializer=rx._init_worker) as pool:
        cold = list(pool.map(cold_case, cases, chunksize=4))
        frames = sorted(pool.map(rx.e3_task, [(k, SPACING, "white") for k in range(n_frames)], chunksize=4), key=lambda r: r[0])
    tracker = locate.Tracker(rx.get_locator(SPACING, min_cold=4))
    session = []
    for k, dets, H, n_vis, blur in frames:
        est = tracker.update(dets)
        py = {"mode": est.mode if est else "lost"}
        if est is not None:
            truth = metrics.true_ids(dets, can, H)
            py.update(n=len(est.matches), id_ok=bool(metrics.identity_ok(est.matches, truth)))
            py["err_mean"], py["err_max"] = metrics.overlay_error_mm(est.H, H, cam.width, cam.height)
        session.append({"t": k / rx.FPS, "dets": det_rows(dets), "H": np.asarray(H).tolist(), "truth": metrics.true_ids(dets, can, H), "py": py})
    data = {
        "camera": {"K": cam.K.tolist(), "width": cam.width, "height": cam.height},
        "spacing": SPACING,
        "map": [{"id": int(m.id), "x": float(m.xy[0]), "y": float(m.xy[1]), "d1": float(m.directions[0]), "d2": float(m.directions[1]), "size": float(m.size_mm)} for m in marks],
        "cold": cold,
        "session": session,
    }
    with open(out_path, "w") as f:
        json.dump(data, f)
    located = [c for c in cold if c["py"]["located"]]
    print(f"cold: {len(located)}/{len(cold)} located, wrong ids {sum(not c['py']['id_ok'] for c in located)}")
    tracked = [s for s in session if s["py"]["mode"] != "lost"]
    print(f"session: {len(tracked)}/{len(session)} frames with a pose")


if __name__ == "__main__":
    main()
