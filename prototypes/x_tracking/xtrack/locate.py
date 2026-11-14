"""Work out where the camera is from the X marks it sees.

The app keeps a map of the canvas: each X's position and stroke angles,
learned once from a wide view. Each frame:

- cold start (no idea where we are): try every pair of detected X's against
  every pair of map X's. Each pairing implies a camera position (a similarity
  transform for a camera facing the canvas); keep it if other detections land
  on map X's with matching stroke angles. The best few are refined with the
  full camera model, and the result is accepted only if it is unambiguous.
- tracking (we knew where we were last frame): predict where map X's should
  appear, match detections to the nearest prediction, and refine the camera
  pose. The previous pose acts as a gentle prior, so even one or two visible
  X's keep the overlay in place for a while.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class MapMark:
    id: int
    xy: tuple
    directions: tuple  # stroke angles in the canvas, radians in [0, pi)
    size_mm: float = 15.0  # mean stroke length


@dataclass
class PoseEstimate:
    rvec: np.ndarray  # world -> camera rotation vector
    tvec: np.ndarray  # world -> camera translation, mm
    H: np.ndarray  # canvas mm -> pixels
    matches: dict = field(default_factory=dict)  # detection index -> map id
    rmse: float = 0.0  # reprojection error of the matched X's, pixels
    mode: str = ""  # "cold", "track" or "hold"
    uncertainty_mm: float = float("inf")  # how far the overlay could plausibly be off, worst place in view (1 sigma)


def build_map(canvas, rng, pos_noise_mm: float = 0.5, dir_noise_deg: float = 3.0) -> list[MapMark]:
    """The map a wide scan would produce: true positions plus measurement noise."""
    out = []
    for m in canvas.marks:
        xy = m.center + rng.normal(0, pos_noise_mm, 2)
        dirs = (m.directions + np.radians(rng.normal(0, dir_noise_deg, 2))) % np.pi
        out.append(MapMark(m.id, tuple(xy), tuple(dirs), float(m.size_mm * (1 + rng.normal(0, 0.05)))))
    return out


def pose_to_H(K, rvec, tvec) -> np.ndarray:
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    H = K @ np.column_stack([R[:, 0], R[:, 1], tvec])
    return H / np.linalg.norm(H)


def project_xy(K, rvec, tvec, xy) -> np.ndarray:
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    P = np.asarray(xy, float) @ R[:, :2].T + tvec
    uv = P @ K.T
    out = uv[:, :2] / uv[:, 2:3]
    out[P[:, 2] <= 1e-6] = np.nan  # behind the camera
    return out


def _project_finite(K, rvec, tvec, xy) -> np.ndarray:
    """Like project_xy, but points behind the camera get a huge finite error
    instead of NaN, so the optimizer can move away from them."""
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    P = np.asarray(xy, float) @ R[:, :2].T + tvec
    z = np.maximum(P[:, 2], 1.0)
    uv = P @ K.T
    return uv[:, :2] / z[:, None] + np.where(P[:, 2:3] < 1.0, 1e5, 0.0)


def _camera_height(rvec, tvec) -> float:
    """Distance of the camera in front of the canvas plane (positive = in front)."""
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    return float(-(-R.T @ np.asarray(tvec, float))[2])


def _angle_dev(a, b):
    """Difference between line angles (mod pi), in [0, pi/2]."""
    return np.abs((a - b + np.pi / 2) % np.pi - np.pi / 2)


def _dir_dev(da, db):
    """Worst stroke-angle disagreement between two X's under the better pairing of strokes. (..., 2) arrays."""
    straight = np.maximum(_angle_dev(da[..., 0], db[..., 0]), _angle_dev(da[..., 1], db[..., 1]))
    swapped = np.maximum(_angle_dev(da[..., 0], db[..., 1]), _angle_dev(da[..., 1], db[..., 0]))
    return np.minimum(straight, swapped)


class Locator:
    def __init__(self, marks: list[MapMark], K: np.ndarray, width: int, height: int, x_size_mm: float = 15.0,
                 dir_tol_deg: float = 15.0, residual_mm: float = 1.5, min_cold: int = 4):  # fmt: skip
        self.ids = np.array([m.id for m in marks])
        self.xy = np.array([m.xy for m in marks], float)
        self.dirs = np.array([m.directions for m in marks], float)
        self.sizes = np.array([m.size_mm for m in marks], float)
        self.size_tol = np.log(1.6)  # a detection may look up to 1.6x bigger or smaller than predicted
        self.min_coverage = 0.4  # cold start: at least this share of the X's predicted in view must be found
        self.max_cold_uncertainty_mm = 10.0  # cold start: refuse locks shakier than this (1 sigma, worst place in view); good locks estimate under ~7
        self.K = K
        self.f = float(K[0, 0])
        self.pp = K[:2, 2].copy()
        self.size = (width, height)
        self.x_size_mm = x_size_mm
        self.dir_tol = np.radians(dir_tol_deg)
        self.residual_mm = residual_mm
        self.min_cold = min_cold
        self.tree = cKDTree(self.xy)
        nn = self.tree.query(self.xy, k=min(2, len(self.xy)))[0]
        self.spacing = float(np.median(nn[:, 1])) if len(self.xy) > 1 else 100.0
        a, b = np.nonzero(~np.eye(len(self.xy), dtype=bool))
        vec = self.xy[b] - self.xy[a]
        order = np.argsort(np.hypot(vec[:, 0], vec[:, 1]))
        self.pair_a, self.pair_b = a[order], b[order]
        self.pair_len = np.hypot(vec[order, 0], vec[order, 1])
        self.pair_ang = np.arctan2(vec[order, 1], vec[order, 0])

    # ---------------------------------------------------------------- helpers

    def _px_per_mm(self, rvec, tvec) -> float:
        """Scale near the image centre for this pose."""
        H = pose_to_H(self.K, rvec, tvec)
        c = np.linalg.solve(H, np.r_[self.pp, 1.0])
        c = c[:2] / c[2]
        p = project_xy(self.K, rvec, tvec, [c, c + (1, 0), c + (0, 1)])
        if not np.isfinite(p).all():
            return 1.0
        return float(np.sqrt(abs(np.linalg.det(np.column_stack([p[1] - p[0], p[2] - p[0]])))))

    def _mark_scales(self, rvec, tvec, idx) -> np.ndarray:
        """Pixels per mm at each map X in ``idx``."""
        xy = self.xy[idx]
        p0, px, py = (project_xy(self.K, rvec, tvec, xy + o) for o in ((0, 0), (1, 0), (0, 1)))
        a, b = px - p0, py - p0
        return np.sqrt(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))

    def _predicted_in_view(self, rvec, tvec) -> int:
        w, h = self.size
        p = project_xy(self.K, rvec, tvec, self.xy)
        mx, my = 0.03 * w, 0.03 * h
        return int(np.sum(np.isfinite(p[:, 0]) & (p[:, 0] > mx) & (p[:, 0] < w - mx) & (p[:, 1] > my) & (p[:, 1] < h - my)))

    def _pred_dirs(self, rvec, tvec, idx) -> np.ndarray:
        """Where map X's stroke angles appear in the image, (len(idx), 2)."""
        xy = self.xy[idx]
        out = []
        for k in (0, 1):
            d = np.column_stack([np.cos(self.dirs[idx, k]), np.sin(self.dirs[idx, k])]) * 5.0
            a, b = project_xy(self.K, rvec, tvec, xy), project_xy(self.K, rvec, tvec, xy + d)
            out.append(np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0]) % np.pi)
        return np.stack(out, -1)

    def _solve(self, x0, img_pts, map_xy, prior=None, w_prior: float = 0.0):
        x_prior = None if prior is None else np.r_[prior.rvec, prior.tvec]

        def resid(x):
            r = (_project_finite(self.K, x[:3], x[3:], map_xy) - img_pts).ravel()
            if x_prior is not None and w_prior > 0:
                z = max(_camera_height(x_prior[:3], x_prior[3:]), 10.0)
                r = np.r_[r, w_prior * (x[:3] - x_prior[:3]) * self.f, w_prior * (x[3:] - x_prior[3:]) * self.f / z]
            return r

        sol = least_squares(resid, x0, method="trf", x_scale=np.r_[0.1, 0.1, 0.1, 10, 10, 10], max_nfev=60,
                            loss="soft_l1", f_scale=2.0)  # one bad match should not drag the pose
        rvec, tvec = sol.x[:3], sol.x[3:]
        res = _project_finite(self.K, rvec, tvec, map_xy) - img_pts
        self._last_jac = sol.jac
        return rvec, tvec, float(np.sqrt(np.mean(np.sum(res**2, 1))))

    def _uncertainty(self, rvec, tvec, map_noise_mm: float = 0.5, detect_noise_px: float = 1.0) -> float:
        """1-sigma overlay uncertainty in canvas mm at the worst of nine points across the
        view, propagated from the last solve's Jacobian (detection noise plus map noise)."""
        J = getattr(self, "_last_jac", None)
        if J is None:
            return float("inf")
        sigma = float(np.hypot(detect_noise_px, map_noise_mm * self._px_per_mm(rvec, tvec)))
        try:
            cov = np.linalg.inv(J.T @ J) * sigma**2
        except np.linalg.LinAlgError:
            return float("inf")
        w, h = self.size
        grid = np.array([(x, y) for x in (0.1 * w, 0.5 * w, 0.9 * w) for y in (0.1 * h, 0.5 * h, 0.9 * h)])

        def canvas_points(x):
            Hi = np.linalg.inv(pose_to_H(self.K, x[:3], x[3:]))
            q = grid @ Hi[:, :2].T + Hi[:, 2]
            return q[:, :2] / q[:, 2:3]

        x0 = np.r_[rvec, tvec]
        base = canvas_points(x0)
        G = np.zeros((len(grid), 2, 6))
        for k, eps in enumerate((1e-5, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3)):
            x = x0.copy()
            x[k] += eps
            G[:, :, k] = (canvas_points(x) - base) / eps
        var = np.einsum("pik,kl,pil->p", G, cov, G)  # trace of each point's 2x2 covariance
        return float(np.sqrt(np.max(var))) if np.all(np.isfinite(var)) else float("inf")

    def _associate(self, dets, rvec, tvec, gate_px):
        """Detections matched one-to-one to the nearest predicted map X with agreeing strokes."""
        w, h = self.size
        pred = project_xy(self.K, rvec, tvec, self.xy)
        view = np.flatnonzero(np.isfinite(pred[:, 0]) & (pred[:, 0] > -gate_px) & (pred[:, 0] < w + gate_px) & (pred[:, 1] > -gate_px) & (pred[:, 1] < h + gate_px))
        if len(view) == 0 or not dets:
            return {}
        centers = np.array([d.center for d in dets])
        dist, j = cKDTree(pred[view]).query(centers, distance_upper_bound=gate_px)
        cands = sorted((dd, i, view[jj]) for i, (dd, jj) in enumerate(zip(dist, j)) if np.isfinite(dd))
        if not cands:
            return {}
        cidx = np.array([c[2] for c in cands])
        pdirs = self._pred_dirs(rvec, tvec, cidx)
        psize = self.sizes[cidx] * self._mark_scales(rvec, tvec, cidx)
        matches, used = {}, set()
        for (dd, i, m), pd, ps in zip(cands, pdirs, psize):
            if i in matches or m in used:
                continue
            if _dir_dev(np.asarray(dets[i].directions), pd) > self.dir_tol:
                continue
            if abs(np.log(dets[i].size / ps)) > self.size_tol:
                continue
            matches[i] = int(m)
            used.add(int(m))
        return matches

    def _refit(self, dets, rvec, tvec, prior=None, w_prior=0.0, rounds: int = 2, w_few=None):
        """Associate, solve, re-associate with a tighter gate, solve again.
        With fewer than four matches, the prior is weighted ``w_few`` instead."""
        matches = {}
        rmse = np.inf
        for r in range(rounds):
            s = self._px_per_mm(rvec, tvec)
            gate = (0.3 if r == 0 else 0.15) * self.spacing * s
            gate = float(np.clip(gate, 6 + 3 * self.residual_mm * s, 250))
            matches = self._associate(dets, rvec, tvec, gate)
            if len(matches) < (1 if prior is not None else 3):
                return None
            img = np.array([dets[i].center for i in matches])
            mxy = self.xy[list(matches.values())]
            w = w_prior if (w_few is None or len(matches) >= 4) else w_few
            rvec, tvec, rmse = self._solve(np.r_[rvec, tvec], img, mxy, prior, w)
        # every match must fit on its own, not just on average
        res = np.hypot(*(_project_finite(self.K, rvec, tvec, mxy) - img).T)
        keep = res <= self._tolerance_px(rvec, tvec)
        if not keep.all():
            matches = {i: m for (i, m), k in zip(matches.items(), keep) if k}
            if len(matches) < (1 if prior is not None else 3):
                return None
            img = np.array([dets[i].center for i in matches])
            mxy = self.xy[list(matches.values())]
            w = w_prior if (w_few is None or len(matches) >= 4) else w_few
            rvec, tvec, rmse = self._solve(np.r_[rvec, tvec], img, mxy, prior, w)
        return rvec, tvec, matches, rmse

    def _tolerance_px(self, rvec, tvec) -> float:
        return 2.0 + self.residual_mm * self._px_per_mm(rvec, tvec)

    # ---------------------------------------------------------------- cold start

    def _initial_poses(self, s, alpha, p_i, m_a, img_pts, map_pts) -> list:
        """Starting points for refining a hypothesis: a camera facing the canvas,
        plus tilt-aware solutions from the planar (4+ points) or three-point solver."""
        Rwc = cv2.Rodrigues(np.array([0.0, 0.0, alpha]))[0]
        Cxy = s * np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]]) @ (self.pp - p_i) + m_a
        R = Rwc.T
        inits = [np.r_[cv2.Rodrigues(R)[0].ravel(), -R @ np.r_[Cxy, -s * self.f]]]
        obj = np.column_stack([map_pts, np.zeros(len(map_pts))]).astype(np.float64)
        img = np.ascontiguousarray(img_pts, np.float64)
        try:
            if len(obj) >= 4:
                _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, self.K, None, flags=cv2.SOLVEPNP_IPPE)
            else:
                _, rvecs, tvecs = cv2.solveP3P(obj, img, self.K, None, flags=cv2.SOLVEPNP_P3P)
            inits += [np.r_[r.ravel(), t.ravel()] for r, t in zip(rvecs, tvecs) if _camera_height(r.ravel(), t.ravel()) > 0]
        except cv2.error:
            pass
        return inits

    def cold_start(self, dets, max_hyp_dets: int = 8, top_k: int = 6) -> PoseEstimate | None:
        if len(dets) < 3:
            return None
        P = np.array([d.center for d in dets])
        ddirs = np.array([d.directions for d in dets])
        dsize = np.array([d.size for d in dets])
        mm_per_px = self.x_size_mm / float(np.median([d.size for d in dets]))
        smin, smax = 0.55 * mm_per_px, 1.8 * mm_per_px
        tol_mm = max(6.0, 0.3 * self.spacing)
        hyp_idx = np.argsort([-d.strength for d in dets])[:max_hyp_dets]
        hyps = []
        for i, j in combinations(hyp_idx, 2):
            dv = P[j] - P[i]
            L = float(np.hypot(*dv))
            if L < 5:
                continue
            lo, hi = np.searchsorted(self.pair_len, [smin * L, smax * L])
            if hi <= lo:
                continue
            a = self.pair_a[lo:hi]
            s = self.pair_len[lo:hi] / L
            alpha = self.pair_ang[lo:hi] - np.arctan2(dv[1], dv[0])
            c, sn = np.cos(alpha)[:, None], np.sin(alpha)[:, None]
            rel = P - P[i]
            X = s[:, None] * (c * rel[None, :, 0] - sn * rel[None, :, 1]) + self.xy[a, 0][:, None]
            Y = s[:, None] * (sn * rel[None, :, 0] + c * rel[None, :, 1]) + self.xy[a, 1][:, None]
            dist, idx = self.tree.query(np.stack([X, Y], -1).reshape(-1, 2))
            dist, idx = dist.reshape(X.shape), idx.reshape(X.shape)
            rot = (ddirs[None] + alpha[:, None, None]) % np.pi
            # coarse checks: a similarity transform ignores perspective, so be lenient here
            ok = (dist < tol_mm) & (_dir_dev(rot, self.dirs[idx]) < 1.5 * self.dir_tol)
            ok &= np.abs(np.log(dsize[None, :] * s[:, None] / self.sizes[idx])) < np.log(2.0)
            score = ok.sum(1)
            for h in np.flatnonzero(score >= 3):
                hyps.append((int(score[h]), float(s[h]), float(alpha[h]), int(i), int(a[h]), tuple(np.flatnonzero(ok[h])), tuple(idx[h][ok[h]])))
        if not hyps:
            return None
        # refine the strongest distinct hypotheses with the full camera model
        hyps.sort(key=lambda t: -t[0])
        results, seen = [], set()
        for score, s, alpha, i, a, det_ix, map_ix in hyps:
            key = tuple(sorted(zip(det_ix, map_ix)))
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > top_k * 4:
                break
            best = None
            for x0 in self._initial_poses(s, alpha, P[i], self.xy[a], P[list(det_ix)], self.xy[list(map_ix)]):
                rvec, tvec, _ = self._solve(x0, P[list(det_ix)], self.xy[list(map_ix)])
                out = self._refit(dets, rvec, tvec)
                if out is None:
                    continue
                rvec, tvec, matches, rmse = out
                if _camera_height(rvec, tvec) <= 0 or rmse > self._tolerance_px(rvec, tvec):
                    continue
                if len(matches) < self.min_coverage * self._predicted_in_view(rvec, tvec):
                    continue  # the pose predicts X's where there are none
                if best is None or (len(matches), -rmse) > (best[0], -best[1]):
                    best = (len(matches), rmse, rvec, tvec, matches)
            if best is not None:
                results.append(best)
            if len(results) >= top_k:
                break
        if not results:
            return None
        results.sort(key=lambda r: (-r[0], r[1]))
        n, rmse, rvec, tvec, matches = results[0]
        best_ids = set(matches.values())
        for n2, rmse2, _, _, m2 in results[1:]:
            if n2 == n and set(m2.values()) != best_ids and rmse2 < 1.5 * rmse + 0.5:
                return None  # two different answers fit equally well
        need = self.min_cold + (1 if len(dets) - n > 10 else 0)  # cluttered scene: want one more X
        if n < need or not self._well_spread(np.array([dets[i].center for i in matches])):
            return None
        img = np.array([dets[i].center for i in matches])
        rvec, tvec, rmse = self._solve(np.r_[rvec, tvec], img, self.xy[list(matches.values())])
        unc = self._uncertainty(rvec, tvec)
        if unc > self.max_cold_uncertainty_mm:
            return None  # the X's in view don't pin the camera down well enough
        return PoseEstimate(rvec, tvec, pose_to_H(self.K, rvec, tvec), matches, rmse, "cold", unc)

    def _well_spread(self, pts) -> bool:
        """Matched X's must span the view in both directions, or the tilt is poorly pinned down."""
        if len(pts) < 3:
            return False
        smallest = np.sqrt(max(np.linalg.eigvalsh(np.cov(pts.T)).min(), 0.0))
        return smallest >= 0.04 * float(np.hypot(*self.size))

    # ---------------------------------------------------------------- tracking

    def track(self, dets, prior: PoseEstimate, anchor: PoseEstimate | None = None, w_prior: float = 0.02,
              w_few: float = 0.3) -> PoseEstimate | None:  # fmt: skip
        """Update from ``prior`` (the predicted pose). The solve is gently anchored to
        ``anchor`` (the last measured pose; defaults to ``prior``), firmly so when fewer
        than four X's match. Returns the prior unchanged ("hold") when no X matches."""
        out = self._refit(dets, prior.rvec.copy(), prior.tvec.copy(), anchor or prior, w_prior, w_few=w_few)
        if out is None:
            return PoseEstimate(prior.rvec, prior.tvec, prior.H, {}, prior.rmse, "hold")
        rvec, tvec, matches, rmse = out
        if rmse > self._tolerance_px(rvec, tvec):
            return None
        if len(matches) == 1:  # one X can only nudge the pose if it is clearly the X we expected
            (i, m), = matches.items()
            pred = project_xy(self.K, prior.rvec, prior.tvec, self.xy[[m]])[0]
            if np.hypot(*(dets[i].center - pred)) > 0.2 * self.spacing * self._px_per_mm(prior.rvec, prior.tvec):
                return PoseEstimate(prior.rvec, prior.tvec, prior.H, {}, prior.rmse, "hold")
        return PoseEstimate(rvec, tvec, pose_to_H(self.K, rvec, tvec), matches, rmse, "track", self._uncertainty(rvec, tvec))


class Tracker:
    """Frame-to-frame tracking with recovery, as the app would run it.

    - predicts the next pose from the current motion (constant velocity);
    - holds the last pose briefly when no X can be matched, trying a fresh
      lock-on every frame meanwhile, and gives up after ``max_hold`` frames;
    - every ``verify_every`` frames, checks the tracked identities against a
      fresh lock-on and switches if they conflict.
    """

    def __init__(self, locator: Locator, max_hold: int = 10, verify_every: int = 10):
        self.loc = locator
        self.max_hold = max_hold
        self.verify_every = verify_every
        self.state: PoseEstimate | None = None
        self.prev: PoseEstimate | None = None
        self.hold = 0
        self.frame = 0

    def _predicted(self) -> PoseEstimate:
        s, p = self.state, self.prev
        if p is None or s.mode == "hold" or p.mode == "hold":
            return s
        tvec = s.tvec + 0.5 * (s.tvec - p.tvec)  # damped; rotation is not extrapolated
        return PoseEstimate(s.rvec, tvec, pose_to_H(self.loc.K, s.rvec, tvec), s.matches, s.rmse, s.mode)

    def update(self, dets) -> PoseEstimate | None:
        self.frame += 1
        est = None
        if self.state is not None:
            est = self.loc.track(dets, self._predicted(), anchor=self.state)
            if est is not None and est.mode == "hold":
                self.hold += 1
                if self.hold > self.max_hold:
                    est = None
            elif est is not None:
                self.hold = 0
        recheck = est is not None and est.mode == "track" and self.frame % self.verify_every == 0
        if (est is None or est.mode == "hold" or recheck) and len(dets) >= 4:
            cold = self.loc.cold_start(dets)
            if cold is not None and len(cold.matches) >= 4:
                conflict = est is not None and any(est.matches.get(i, m) != m for i, m in cold.matches.items())
                if est is None or est.mode == "hold" or conflict:
                    est = cold
                    self.hold = 0
        self.prev, self.state = (self.state if est is not None else None), est
        return est
