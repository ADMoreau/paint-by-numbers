"""Tests for the X-mark tracking prototype. Run: python -m pytest prototypes/x_tracking/tests"""

import numpy as np
import pytest

from xtrack import camera, canvas as cv, detect, locate, metrics, render as rd

CAM = camera.MAIN


@pytest.fixture(scope="module")
def can():
    return cv.make_canvas(7, "grid", 100)


@pytest.fixture(scope="module")
def loc(can):
    marks = locate.build_map(can, np.random.default_rng(1), pos_noise_mm=0.3)
    return locate.Locator(marks, CAM.K, CAM.width, CAM.height)


def fake_detections(can, H, ids=None, noise_px=0.3, seed=0):
    """Detections exactly where the marks project, as a perfect detector would report."""
    rng = np.random.default_rng(seed)
    out = []
    for m in can.marks:
        if ids is not None and m.id not in ids:
            continue
        c = camera.project(H, m.center)[0] + rng.normal(0, noise_px, 2)
        p0, px, py = camera.project(H, [m.center, m.center + (1, 0), m.center + (0, 1)])
        a, b = px - p0, py - p0
        scale = np.sqrt(abs(a[0] * b[1] - a[1] * b[0]))
        dirs = []
        for a in m.directions:
            q = camera.project(H, [m.center, m.center + 5 * np.array([np.cos(a), np.sin(a)])])
            dirs.append(np.arctan2(*(q[1] - q[0])[::-1]) % np.pi)
        out.append(detect.Detection(c, np.array(dirs), m.size_mm * scale, 20.0))
    return out


# ---------------------------------------------------------------- camera


def test_look_at_puts_target_at_image_center():
    pose = camera.look_at((200, 150), 250, tilt_deg=20, tilt_dir_deg=70, roll_deg=33)
    H = camera.homography(CAM, pose)
    assert np.allclose(camera.project(H, [(200, 150)])[0], CAM.K[:2, 2], atol=1e-6)
    assert camera.in_front(H, [(200, 150)])[0]


def test_head_on_scale_matches_focal_length():
    H = camera.homography(CAM, camera.look_at((300, 200), 200))
    assert camera.local_scale(H, (300, 200)) == pytest.approx(CAM.f / 200, rel=1e-6)


# ---------------------------------------------------------------- detection


def test_detector_finds_visible_marks_precisely(can):
    rng = np.random.default_rng(3)
    pose = camera.look_at(can.marks[8].center + (40, 30), 220, tilt_deg=12, tilt_dir_deg=20, roll_deg=15)
    fr = rd.render(can, CAM, pose, rd.WHITE_GRAPHITE, rd.Effects(hand_prob=0.0, motion_prob=0.0), rng)
    dets = detect.detect_x(fr.image)
    truth = metrics.true_ids(dets, can, fr.H)
    assert set(fr.visible_ids) <= {t for t in truth if t is not None}
    proj = {m.id: camera.project(fr.H, m.center)[0] for m in can.marks}
    errs = [np.hypot(*(d.center - proj[t])) for d, t in zip(dets, truth) if t is not None]
    assert np.median(errs) < 0.6


def test_detector_ignores_blank_canvas():
    blank = cv.make_canvas(7, "grid", 100)
    blank.marks = []
    fr = rd.render(blank, CAM, camera.look_at((300, 200), 200), rd.WHITE_GRAPHITE, rd.Effects(hand_prob=0.0), np.random.default_rng(4))
    assert len(detect.detect_x(fr.image)) <= 1


# ---------------------------------------------------------------- localization


def test_cold_start_identifies_marks_and_pose(can, loc):
    pose = camera.look_at((320, 220), 300, tilt_deg=18, tilt_dir_deg=110, roll_deg=-40)
    H = camera.homography(CAM, pose)
    vis = rd.visible_marks(can, H, CAM.width, CAM.height)
    dets = fake_detections(can, H, set(vis))
    est = loc.cold_start(dets)
    assert est is not None and len(est.matches) >= 4
    assert metrics.identity_ok(est.matches, metrics.true_ids(dets, can, H))
    assert metrics.overlay_error_mm(est.H, H, CAM.width, CAM.height)[0] < 1.5


def test_cold_start_refuses_three_marks(can, loc):
    H = camera.homography(CAM, camera.look_at((300, 220), 300))
    vis = rd.visible_marks(can, H, CAM.width, CAM.height)
    assert loc.cold_start(fake_detections(can, H, set(vis[:3]))) is None  # three can't be verified


def test_size_check_rejects_specks(can, loc):
    """Real X's plus tiny specks must not lock onto a pose that treats the specks as X's."""
    H = camera.homography(CAM, camera.look_at((300, 220), 250))
    dets = fake_detections(can, H, set(rd.visible_marks(can, H, CAM.width, CAM.height)))
    rng = np.random.default_rng(9)
    dets += [detect.Detection(rng.uniform((0, 0), (CAM.width, CAM.height)), rng.uniform(0, np.pi, 2), 12.0, 5.0) for _ in range(25)]
    est = loc.cold_start(dets)
    assert est is None or metrics.identity_ok(est.matches, metrics.true_ids(dets, can, H))


def test_tracking_follows_motion_and_holds_without_marks(can, loc):
    H0 = camera.homography(CAM, camera.look_at((300, 220), 250, 10, 30, 5))
    est = loc.cold_start(fake_detections(can, H0, set(rd.visible_marks(can, H0, CAM.width, CAM.height))))
    assert est is not None
    H1 = camera.homography(CAM, camera.look_at((306, 223), 245, 11, 32, 6))  # a small move
    vis = rd.visible_marks(can, H1, CAM.width, CAM.height)
    nxt = loc.track(fake_detections(can, H1, set(vis[:2]), seed=1), est)  # only two X's in view
    assert nxt is not None and nxt.mode == "track"
    assert metrics.overlay_error_mm(nxt.H, H1, CAM.width, CAM.height)[0] < 3.0
    held = loc.track([], nxt)
    assert held.mode == "hold" and np.allclose(held.H, nxt.H)


# ---------------------------------------------------------------- metrics


def test_overlay_error_zero_for_same_pose():
    H = camera.homography(CAM, camera.look_at((300, 200), 200, 10, 0, 0))
    assert metrics.overlay_error_mm(H, H, CAM.width, CAM.height) == (pytest.approx(0, abs=1e-9), pytest.approx(0, abs=1e-9))


def test_tracker_recovers_after_a_jump(can, loc):
    """If the view jumps somewhere else, the tracker re-locks instead of holding a stale pose."""
    tracker = locate.Tracker(loc)
    HA = camera.homography(CAM, camera.look_at((200, 150), 280))
    assert tracker.update(fake_detections(can, HA, set(rd.visible_marks(can, HA, CAM.width, CAM.height)))) is not None
    HB = camera.homography(CAM, camera.look_at((420, 300), 280, 10, 40, 20))
    est = tracker.update(fake_detections(can, HB, set(rd.visible_marks(can, HB, CAM.width, CAM.height)), seed=2))
    assert est is not None
    assert metrics.overlay_error_mm(est.H, HB, CAM.width, CAM.height)[0] < 1.5
