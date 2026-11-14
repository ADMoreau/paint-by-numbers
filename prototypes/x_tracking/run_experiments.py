"""Simulation experiments for tracking a canvas from hand-drawn X marks.

    python run_experiments.py --out results/ [--quick]

E1  How often are enough X's in view? (geometry only, many random camera poses)
E2  Locking on from a single frame: render, detect, identify, measure.
E3  A 30-second session: wide view, move in, paint close-up with a hand in
    the way, then a fast pan. Tracking frame to frame.

Writes summary.md, results.json and annotated sample frames to --out.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from xtrack import camera
from xtrack import canvas as cv
from xtrack import detect, locate, metrics
from xtrack import render as rd

LOOKS = {"white": rd.WHITE_GRAPHITE, "mauve": rd.MAUVE_GRAPHITE, "mauve_chalk": rd.MAUVE_CHALK, "dark_chalk": rd.DARK_CHALK}
CAMERAS = {"main": camera.MAIN, "ultrawide": camera.ULTRAWIDE}
POLARITY = {"white": "dark", "mauve": "dark", "mauve_chalk": "light", "dark_chalk": "light"}  # the app knows the mark colour
CANVAS_SEED, MAP_SEED = 1, 1234


def _init_worker():
    cv2.setNumThreads(1)


@lru_cache(maxsize=16)
def get_canvas(layout: str, spacing: float):
    return cv.make_canvas(CANVAS_SEED, layout, spacing)


@lru_cache(maxsize=16)
def get_locator(spacing: float, cam_name: str = "main", min_cold: int = 3):
    cam = CAMERAS[cam_name]
    marks = locate.build_map(get_canvas("grid", spacing), np.random.default_rng(MAP_SEED))
    return locate.Locator(marks, cam.K, cam.width, cam.height, min_cold=min_cold)


# ---------------------------------------------------------------- E1: visibility


def e1_task(args):
    layout, spacing, cam_name, distance, n, seed = args
    can, cam = get_canvas(layout, spacing), CAMERAS[cam_name]
    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(n):
        H = camera.homography(cam, camera.random_pose(rng, can.size, distance))
        mask = rd.hand_mask(rd.random_hand(rng, cam.width, cam.height), cam.width, cam.height) if rng.uniform() < 0.6 else None
        counts.append(len(rd.visible_marks(can, H, cam.width, cam.height, mask)))
    c = np.array(counts)
    return {"layout": layout, "spacing": spacing, "camera": cam_name, "distance": distance,
            "p3": float((c >= 3).mean()), "p4": float((c >= 4).mean()), "mean": float(c.mean())}  # fmt: skip


# ---------------------------------------------------------------- E2: cold start


def e2_task(args):
    look, spacing, distance, seed = args
    can, cam = get_canvas("grid", spacing), camera.MAIN
    rng = np.random.default_rng(seed)
    fr = rd.render(can, cam, camera.random_pose(rng, can.size, distance), LOOKS[look], rd.Effects(), rng)
    t0 = time.perf_counter()
    dets = detect.detect_x(fr.image, POLARITY[look])
    t1 = time.perf_counter()
    est = get_locator(spacing, min_cold=4).cold_start(dets)
    t2 = time.perf_counter()
    truth = metrics.true_ids(dets, can, fr.H)
    vis = set(fr.visible_ids)
    row = {"look": look, "spacing": spacing, "distance": distance, "seed": seed, "visible": len(vis),
           "found": len(vis & {t for t in truth if t is not None}), "false_dets": sum(t is None for t in truth),
           "ms_detect": 1000 * (t1 - t0), "ms_locate": 1000 * (t2 - t1), "located": est is not None}  # fmt: skip
    if est is not None:
        mean, mx = metrics.overlay_error_mm(est.H, fr.H, cam.width, cam.height)
        row.update(n=len(est.matches), id_ok=metrics.identity_ok(est.matches, truth), err_mean=mean, err_max=mx,
                   uncertainty=est.uncertainty_mm)
    return row


# ---------------------------------------------------------------- E3: a painting session

FPS, DURATION = 30, 30.0
HAND_EPISODES = [(7.0, 12.0, 0, 0.22, 0.18, 11), (15.0, 22.0, 1, 0.25, 0.2, 12), (27.5, 30.0, 0, 0.2, 0.16, 13)]


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def trajectory(t: float, size) -> tuple:
    """(target, distance, tilt, tilt direction, roll) of the phone at time t."""
    W, H = size
    center, A, B = np.array([W / 2, H / 2]), np.array([0.33 * W, 0.42 * H]), np.array([0.78 * W, 0.7 * H])

    def offset(tau):
        return np.array([110 * np.sin(0.31 * tau + 0.5) + 50 * np.sin(0.83 * tau), 80 * np.sin(0.27 * tau + 1.2) + 35 * np.sin(0.71 * tau)])

    def wander(tt):
        tau = tt - 5
        # starts exactly where the move-in ended
        return A + offset(tau) - offset(0.0), 200 + 50 * np.sin(0.4 * tau), 12 + 8 * np.sin(0.35 * tau), 30 + 25 * tau, 10 + 15 * np.sin(0.2 * tau)

    if t < 2:
        return center, 650.0, 4.0, 30.0, 0.0
    if t < 5:
        s = _smooth((t - 2) / 3)
        return center + s * (A - center), 650 + s * (200 - 650), 4 + 8 * s, 30.0, 10 * s
    if t < 25:
        return wander(t)
    end = wander(25.0)
    if t < 27:  # fast pan across the canvas
        s = _smooth((t - 25) / 2)
        return end[0] + s * (B - end[0]), end[1] + s * (180 - end[1]), end[2], end[3], end[4]
    return B, 180.0, end[2], end[3], end[4]


def e3_pose(t, size):
    target, dist, tilt, tdir, roll = trajectory(t, size)
    return camera.look_at(target, dist, tilt, tdir, roll)


def e3_hand(t, w, h):
    for start, end, side, arm, brush, seed in HAND_EPISODES:
        if start <= t < end:
            tip = (w * (0.5 + 0.22 * np.sin(0.9 * t + seed)), h * (0.5 + 0.18 * np.sin(1.3 * t + 2 * seed)))
            return rd.random_hand(np.random.default_rng(seed), w, h, tip=tip, side=side, arm_frac=arm, brush_frac=brush)
    return None


def e3_task(args):
    k, spacing, look = args
    can, cam = get_canvas("grid", spacing), camera.MAIN
    t = k / FPS
    pose = e3_pose(t, can.size)
    H0, H1 = camera.homography(cam, pose), camera.homography(cam, e3_pose(t + 0.5 / FPS, can.size))
    c = camera.project(np.linalg.inv(H0), [cam.K[:2, 2]])
    motion = camera.project(H1, c)[0] - cam.K[:2, 2]  # blur from motion during the exposure
    rng = np.random.default_rng(1000 + k)
    fr = rd.render(can, cam, pose, LOOKS[look], rd.Effects(defocus_sigma=(0.0, 0.8)), rng, hand=e3_hand(t, cam.width, cam.height), motion=motion)
    dets = detect.detect_x(fr.image, POLARITY[look])
    return k, dets, fr.H, len(fr.visible_ids), float(np.hypot(*motion))


def run_e3(pool, spacing: float, look: str, out: Path, n_frames: int):
    results = sorted(pool.map(e3_task, [(k, spacing, look) for k in range(n_frames)], chunksize=4), key=lambda r: r[0])
    tracker = locate.Tracker(get_locator(spacing, min_cold=4))
    can, cam = get_canvas("grid", spacing), camera.MAIN
    rows = []
    for k, dets, H, n_vis, blur in results:
        est = tracker.update(dets)
        row = {"t": k / FPS, "visible": n_vis, "detected": len(dets), "blur_px": blur, "mode": est.mode if est else "lost"}
        if est is not None:
            truth = metrics.true_ids(dets, can, H)
            row.update(n=len(est.matches), id_ok=metrics.identity_ok(est.matches, truth), H_est=est.H, matches=est.matches,
                       uncertainty=est.uncertainty_mm)
            row["err_mean"], row["err_max"] = metrics.overlay_error_mm(est.H, H, cam.width, cam.height)
        rows.append(row)
    return rows


def e3_strip(rows, spacing, look, path, times=(1, 4, 8, 10, 16, 20, 26, 29)):
    """Annotated frames from the session at chosen times."""
    cam = camera.MAIN
    tiles = []
    for t in times:
        k = min(int(round(t * FPS)), len(rows) - 1)
        _, dets, H, _, _ = e3_task((k, spacing, look))
        r = rows[k]
        est = SimpleNamespace(H=r["H_est"], matches=r["matches"]) if "H_est" in r else None
        status = {"cold": "locked on", "track": "tracking", "hold": "holding last position", "lost": "lost"}[r["mode"]]
        label = f"t = {t:.0f} s: {status}" + (f", {r['err_mean']:.1f} mm off" if est is not None else "")
        tiles.append(annotate(_frame_image(k, spacing, look), dets, est, H, cam, label))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]))


def _frame_image(k, spacing, look):
    can, cam = get_canvas("grid", spacing), camera.MAIN
    t = k / FPS
    pose = e3_pose(t, can.size)
    H0, H1 = camera.homography(cam, pose), camera.homography(cam, e3_pose(t + 0.5 / FPS, can.size))
    c = camera.project(np.linalg.inv(H0), [cam.K[:2, 2]])
    motion = camera.project(H1, c)[0] - cam.K[:2, 2]
    rng = np.random.default_rng(1000 + k)
    return rd.render(can, cam, pose, LOOKS[look], rd.Effects(defocus_sigma=(0.0, 0.8)), rng, hand=e3_hand(t, cam.width, cam.height), motion=motion).image


# ---------------------------------------------------------------- pictures


def annotate(img, dets, est, H_gt, cam, label):
    """Detections (green real / red false), matched X's (magenta), and the
    overlay a 25 mm canvas grid would get from the estimated pose (cyan)."""
    out = img.copy()
    if est is not None:
        Hi = np.linalg.inv(est.H)
        c = camera.project(Hi, [(0, 0), (cam.width, 0), (cam.width, cam.height), (0, cam.height)])
        lo, hi = np.floor(c.min(0) / 25) * 25, np.ceil(c.max(0) / 25) * 25
        for x in np.arange(lo[0], hi[0] + 1, 25):
            p = camera.project(est.H, np.column_stack([np.full(50, x), np.linspace(lo[1], hi[1], 50)]))
            cv2.polylines(out, [np.round(p).astype(np.int32)], False, (255, 220, 0), 1, cv2.LINE_AA)
        for y in np.arange(lo[1], hi[1] + 1, 25):
            p = camera.project(est.H, np.column_stack([np.linspace(lo[0], hi[0], 50), np.full(50, y)]))
            cv2.polylines(out, [np.round(p).astype(np.int32)], False, (255, 220, 0), 1, cv2.LINE_AA)
    for i, d in enumerate(dets):
        c = tuple(int(v) for v in d.center)
        cv2.circle(out, c, int(d.size / 2) + 5, (0, 200, 0), 2, cv2.LINE_AA)
        if est is not None and i in est.matches:
            cv2.circle(out, c, 6, (255, 0, 255), -1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (cam.width, 40), (255, 255, 255), -1)
    cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return cv2.resize(out, (640, 480), interpolation=cv2.INTER_AREA)


def sample_sheet(cases, path):
    tiles = []
    for look, spacing, distance, seed in cases:
        can, cam = get_canvas("grid", spacing), camera.MAIN
        rng = np.random.default_rng(seed)
        fr = rd.render(can, cam, camera.random_pose(rng, can.size, distance), LOOKS[look], rd.Effects(), rng)
        dets = detect.detect_x(fr.image, POLARITY[look])
        est = get_locator(spacing, min_cold=4).cold_start(dets)
        if est is not None and len(est.matches) < 4:
            est = None
        err = f"{metrics.overlay_error_mm(est.H, fr.H, cam.width, cam.height)[0]:.1f} mm off" if est else "no lock"
        tiles.append(annotate(fr.image, dets, est, fr.H, cam, f"{look}, X every {spacing:.0f} mm, {distance:.0f} mm away: {err}"))
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]))


# ---------------------------------------------------------------- summaries


def pct(x):
    return f"{100 * x:.0f}%"


def summarize(e1, e2, e3, spacing3) -> str:
    lines = ["# X-mark tracking: simulation results", ""]
    lines += ["## E1: how often are enough X's in view?", "",
              "Random camera poses aimed anywhere on a 24 x 18 in canvas, tilted up to 25 degrees, any rotation; a hand covers part of the view 60% of the time. Cells: share of frames with at least 3 / at least 4 X's fully visible.", ""]  # fmt: skip
    for cam_name, layout in (("main", "grid"), ("ultrawide", "grid"), ("main", "corners")):
        rows = [r for r in e1 if r["camera"] == cam_name and r["layout"] == layout]
        if not rows:
            continue
        spacings = sorted({r["spacing"] for r in rows})
        dists = sorted({r["distance"] for r in rows})
        title = f"{cam_name} camera" + (", 4 X's near the corners" if layout == "corners" else "")
        head = "| distance | " + " | ".join(("corners" if layout == "corners" else f"X every {s:.0f} mm") for s in spacings) + " |"
        lines += [f"**{title}**", "", head, "|" + " --- |" * (len(spacings) + 1)]
        for d in dists:
            cells = []
            for s in spacings:
                r = next(r for r in rows if r["distance"] == d and r["spacing"] == s)
                cells.append(f"{pct(r['p3'])} / {pct(r['p4'])}")
            lines.append(f"| {d:.0f} mm | " + " | ".join(cells) + " |")
        lines.append("")
    if e2:
        lines += ["## E2: locking on from a single frame", "",
                  "Rendered frames (blur, noise, compression, a hand half the time). Lock = four or more X's identified. Wrong = locked with a wrong identity or more than 10 mm off. Error = how far the overlay lands from where it belongs, averaged over the whole view (median / 95th percentile, in mm on the canvas).", ""]  # fmt: skip
        lines += ["| ground | X every | distance | X's in view | found | false/frame | lock | wrong | error mm |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
        for look in LOOKS:
            for s in sorted({r["spacing"] for r in e2}):
                for d in sorted({r["distance"] for r in e2}):
                    rs = [r for r in e2 if r["look"] == look and r["spacing"] == s and r["distance"] == d]
                    if not rs:
                        continue
                    vis = sum(r["visible"] for r in rs)
                    found = sum(r["found"] for r in rs)
                    lock4 = [r for r in rs if r["located"] and r["n"] >= 4]
                    good = [r for r in lock4 if r["id_ok"] and r["err_mean"] <= 10]
                    wrong = len(lock4) - len(good)
                    errs = [r["err_mean"] for r in good]
                    err = f"{np.median(errs):.1f} / {np.percentile(errs, 95):.1f}" if errs else "-"
                    lines.append(f"| {look} | {s:.0f} mm | {d:.0f} mm | {vis / len(rs):.1f} | {pct(found / max(vis, 1))} | "
                                 f"{np.mean([r['false_dets'] for r in rs]):.1f} | {pct(len(good) / len(rs))} | {wrong} | {err} |")  # fmt: skip
        t_det = np.median([r["ms_detect"] for r in e2])
        t_loc = np.median([r["ms_locate"] for r in e2])
        lines += ["", f"Median time per frame on this laptop (one core, Python): detect {t_det:.0f} ms, lock on {t_loc:.0f} ms.", ""]
    if e3:
        lines += [f"## E3: a 30-second painting session (X every {spacing3:.0f} mm, white canvas)", "",
                  "0-2 s wide view from 650 mm; 2-5 s move in to 200 mm; 5-25 s wander close-up (140-250 mm, tilt up to 20 degrees) with a hand in view 7-12 s and 15-22 s; 25-27 s fast pan across the canvas; 27-30 s paint at 180 mm with a hand.", ""]  # fmt: skip
        phases = [("wide view", 0, 2), ("moving in", 2, 5), ("close-up, no hand", 5, 25), ("close-up, hand", 5, 25), ("fast pan", 25, 27), ("painting at 180 mm", 27, 30)]
        hand_t = lambda t: any(a <= t < b for a, b, *_ in HAND_EPISODES)  # noqa: E731
        lines += ["| phase | frames | tracked | holding still (no X matched) | lost | wrong | error mm (median / 95% / max) |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, a, b in phases:
            rs = [r for r in e3 if a <= r["t"] < b and (name not in ("close-up, no hand", "close-up, hand") or hand_t(r["t"]) == (name == "close-up, hand"))]
            if not rs:
                continue
            tracked = [r for r in rs if r["mode"] in ("cold", "track")]
            held = [r for r in rs if r["mode"] == "hold"]
            lost = [r for r in rs if r["mode"] == "lost"]
            wrong = [r for r in rs if r["mode"] != "lost" and not r.get("id_ok", True)]
            errs = [r["err_mean"] for r in rs if r["mode"] != "lost"]
            err = f"{np.median(errs):.1f} / {np.percentile(errs, 95):.1f} / {np.max(errs):.1f}" if errs else "-"
            lines.append(f"| {name} | {len(rs)} | {pct(len(tracked) / len(rs))} | {pct(len(held) / len(rs))} | {pct(len(lost) / len(rs))} | {len(wrong)} | {err} |")
        gaps, run = [], 0
        for r in e3:
            run = run + 1 if r["mode"] == "lost" else 0
            gaps.append(run)
        lines += ["", f"Longest stretch without a lock: {max(gaps) / FPS:.1f} s.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quick", action="store_true", help="fewer samples, for a fast check")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--experiments", default="e1,e2,e3", help="comma-separated subset of e1,e2,e3")
    ap.add_argument("--looks", default="white,mauve,mauve_chalk", help=f"grounds for E2, from {','.join(LOOKS)}")
    ap.add_argument("--session-spacing", type=float, default=100.0, help="X spacing for E3, mm")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n1, n2, n3 = (200, 8, 150) if args.quick else (1500, 40, int(FPS * DURATION))
    todo = set(args.experiments.split(","))
    looks = args.looks.split(",")
    e1, e2, e3 = [], [], []
    t_start = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
        e1_cfg = [("grid", s, "main", d, n1, 100 + i) for i, (s, d) in enumerate((s, d) for s in (50, 75, 100, 150) for d in (100, 150, 200, 300))]
        e1_cfg += [("grid", s, "ultrawide", d, n1, 200 + i) for i, (s, d) in enumerate((s, d) for s in (100, 150) for d in (100, 150, 200, 300))]
        e1_cfg += [("corners", 0.0, "main", d, n1, 300 + i) for i, d in enumerate((100, 150, 200, 300, 450, 650))]
        e1 = list(pool.map(e1_task, e1_cfg)) if "e1" in todo else []
        print(f"E1 done ({time.perf_counter() - t_start:.0f}s)", flush=True)
        e2_cfg = [(look, s, d, 10_000 + k + 100 * j)
                  for j, (look, s, d) in enumerate((l, s, d) for l in looks for s in (50, 75, 100, 150) for d in (100, 150, 200, 300))
                  for k in range(n2)]  # fmt: skip
        e2 = list(pool.map(e2_task, e2_cfg, chunksize=2)) if "e2" in todo else []
        print(f"E2 done ({time.perf_counter() - t_start:.0f}s)", flush=True)
        e3 = run_e3(pool, args.session_spacing, "white", args.out, n3) if "e3" in todo else []
        print(f"E3 done ({time.perf_counter() - t_start:.0f}s)", flush=True)
    if e3:
        e3_strip(e3, args.session_spacing, "white", args.out / "session.png")
    e3_json = [{k: v for k, v in r.items() if k not in ("H_est", "matches")} for r in e3]
    (args.out / "results.json").write_text(json.dumps({"e1": e1, "e2": e2, "e3": e3_json}, indent=1, default=float))
    (args.out / "summary.md").write_text(summarize(e1, e2, e3, args.session_spacing))
    if "e2" in todo:
        sample_sheet([("white", 75, 150, 20_001), ("white", 100, 200, 20_002), ("white", 150, 300, 20_003),
                      (looks[-1], 100, 200, 20_004), (looks[-1], 75, 150, 20_005), ("white", 100, 100, 20_006)], args.out / "samples.png")  # fmt: skip
    print((args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
