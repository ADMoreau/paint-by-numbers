"""Finding the target image in a photo of its sketch.

1. **Model** the target by its colour edges, binned by direction, each bin
   with a distance transform (nearest edge of that direction, anywhere).
2. **Search** (when there is no starting placement): for each scale and
   rotation, slide a blurred copy of the edges over the sketch lines (FFT
   correlation) and keep the best few placements.
3. **Refine**: oriented ICP. Each sketch point is matched to the nearest target
   edge running the same way (within ~30 degrees); the placement is solved by
   robust point-to-line least squares while the match radius shrinks.
   Similarity, affine or full perspective.
4. **Bend** (optional): a smooth displacement grid on top of the placement, so
   the plan follows where the painter actually drew, not just where the image
   fits best overall.

All maps go *from* work pixels (the straightened canvas photo, or the raw
photo) *to* model pixels (the target, downscaled to ``long_side``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .lines import LinePoints, image_edges

K_BINS = 8


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    p = pts @ H[:, :2].T + H[:, 2]
    return p[:, :2] / p[:, 2:3]


def local_jacobian(H: np.ndarray, pts) -> np.ndarray:
    """(n, 2, 2) derivative of the homography at each point."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    s = pts @ H[2, :2] + H[2, 2]
    p = apply_h(H, pts)
    J = np.empty((len(pts), 2, 2))
    for r in range(2):
        J[:, r, 0] = (H[r, 0] - p[:, r] * H[2, 0]) / s
        J[:, r, 1] = (H[r, 1] - p[:, r] * H[2, 1]) / s
    return J


def normal_angles(H: np.ndarray, pts, normals) -> np.ndarray:
    """Direction of each line normal after mapping through H (normals map by J^-T), mod pi."""
    J = local_jacobian(H, pts)
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    # J^-T n, up to a positive factor when det > 0 (sign does not matter mod pi)
    nx = (J[:, 1, 1] * normals[:, 0] - J[:, 1, 0] * normals[:, 1]) * np.sign(det)
    ny = (-J[:, 0, 1] * normals[:, 0] + J[:, 0, 0] * normals[:, 1]) * np.sign(det)
    return np.arctan2(ny, nx) % np.pi


class EdgeModel:
    def __init__(self, target: np.ndarray, long_side: int = 800, **edge_args):
        h, w = target.shape[:2]
        self.f = long_side / max(h, w)  # model px per target px
        small = cv2.resize(target, (round(w * self.f), round(h * self.f)), interpolation=cv2.INTER_AREA)
        self.size = (small.shape[1], small.shape[0])
        self.long = max(self.size)
        e = image_edges(small, **edge_args)
        self.edges = e
        sw, sh = self.size
        ang = np.arctan2(e.normals[:, 1], e.normals[:, 0]) % np.pi
        bins = np.floor(ang / (np.pi / K_BINS)).astype(int) % K_BINS
        self.dt = np.empty((K_BINS, sh, sw), np.float32)
        self.lab = np.empty((K_BINS, sh, sw), np.int64)
        tables_xy, tables_n, offset = [], [], 0
        xs, ys = e.pts[:, 0].astype(int), e.pts[:, 1].astype(int)
        for b in range(K_BINS):
            sel = bins == b
            src = np.full((sh, sw), 255, np.uint8)
            src[ys[sel], xs[sel]] = 0
            if not sel.any():
                self.dt[b] = 1e6
                self.lab[b] = 0
                tables_xy.append(np.zeros((1, 2)))
                tables_n.append(np.zeros((1, 2)))
                offset += 1
                continue
            dt, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            n_lab = int(lab.max()) + 1
            txy = np.zeros((n_lab, 2))
            tn = np.zeros((n_lab, 2))
            L = lab[ys[sel], xs[sel]]
            txy[L] = e.pts[sel]
            tn[L] = e.normals[sel]
            self.dt[b] = dt
            self.lab[b] = lab.astype(np.int64) + offset
            tables_xy.append(txy)
            tables_n.append(tn)
            offset += n_lab
        self.table_xy = np.vstack(tables_xy)
        self.table_n = np.vstack(tables_n)
        src = np.full((sh, sw), 255, np.uint8)
        src[ys, xs] = 0
        self.dt_all = cv2.distanceTransform(src, cv2.DIST_L2, 5)

    def nearest(self, p: np.ndarray, ang: np.ndarray):
        """For model points ``p`` with normal angle ``ang``: distance to the nearest
        edge within one direction bin either side, that edge point and its normal.
        Points outside the model get distance inf."""
        sw, sh = self.size
        inside = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] <= sw - 1) & (p[:, 1] <= sh - 1)
        xi = np.clip(np.round(p[:, 0]).astype(int), 0, sw - 1)
        yi = np.clip(np.round(p[:, 1]).astype(int), 0, sh - 1)
        b = np.floor(ang / (np.pi / K_BINS)).astype(int) % K_BINS
        best = np.full(len(p), np.inf)
        lab = np.zeros(len(p), np.int64)
        for off in (-1, 0, 1):
            bb = (b + off) % K_BINS
            d = self.dt[bb, yi, xi]
            better = d < best
            best = np.where(better, d, best)
            lab = np.where(better, self.lab[bb, yi, xi], lab)
        e = self.table_xy[lab]
        best = np.where(inside, np.hypot(*(p - e).T), np.inf)  # exact (sub-pixel) distance
        return best, e, self.table_n[lab]


# ---------------------------------------------------------------- transforms


KINDS = {"similarity": 4, "affine": 6, "homography": 8}


def _to_params(kind: str, G: np.ndarray) -> np.ndarray:
    G = G / G[2, 2]
    if kind == "similarity":
        return np.array([(G[0, 0] + G[1, 1]) / 2, (G[1, 0] - G[0, 1]) / 2, G[0, 2], G[1, 2]])
    if kind == "affine":
        return G[:2].ravel()
    return G.ravel()[:8]


def _to_matrix(kind: str, t: np.ndarray) -> np.ndarray:
    if kind == "similarity":
        a, b, tx, ty = t
        return np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.0]])
    if kind == "affine":
        return np.vstack([t.reshape(2, 3), [0, 0, 1.0]])
    return np.append(t, 1.0).reshape(3, 3)


def _jac(kind: str, t: np.ndarray, q: np.ndarray):
    """Mapped points and d(x)/dt, d(y)/dt for normalised points q."""
    n = len(q)
    qx, qy = q[:, 0], q[:, 1]
    one, zero = np.ones(n), np.zeros(n)
    if kind == "similarity":
        a, b, tx, ty = t
        x, y = a * qx - b * qy + tx, b * qx + a * qy + ty
        return x, y, np.column_stack([qx, -qy, one, zero]), np.column_stack([qy, qx, zero, one])
    if kind == "affine":
        x = t[0] * qx + t[1] * qy + t[2]
        y = t[3] * qx + t[4] * qy + t[5]
        return x, y, np.column_stack([qx, qy, one, zero, zero, zero]), np.column_stack([zero, zero, zero, qx, qy, one])
    u = t[0] * qx + t[1] * qy + t[2]
    v = t[3] * qx + t[4] * qy + t[5]
    s = t[6] * qx + t[7] * qy + 1
    x, y = u / s, v / s
    Jx = np.column_stack([qx / s, qy / s, one / s, zero, zero, zero, -x * qx / s, -x * qy / s])
    Jy = np.column_stack([zero, zero, zero, qx / s, qy / s, one / s, -y * qx / s, -y * qy / s])
    return x, y, Jx, Jy


@dataclass
class Fit:
    G: np.ndarray  # work px -> model px
    kind: str
    matched: float = 0.0  # share of sketch points with a matching edge
    rms: float = 0.0  # point-to-line, work px
    score: float = 1.0  # truncated chamfer, 0 (perfect) .. 1
    bend: "Bend | None" = None

    def __call__(self, q) -> np.ndarray:
        p = apply_h(self.G, q)
        return p if self.bend is None else p + self.bend.disp(p)


def _norm(pts: np.ndarray, size) -> tuple[np.ndarray, np.ndarray]:
    """Similarity normalisations for work points and model points."""
    c = pts.mean(0)
    s = pts.std() + 1e-9
    Tq = np.array([[1 / s, 0, -c[0] / s], [0, 1 / s, -c[1] / s], [0, 0, 1.0]])
    m = max(size) / 2
    Tp = np.array([[1 / m, 0, -size[0] / 2 / m], [0, 1 / m, -size[1] / 2 / m], [0, 0, 1.0]])
    return Tq, Tp


def model_px_per_work_px(G: np.ndarray, pts) -> np.ndarray:
    J = local_jacobian(G, pts)
    return np.sqrt(np.abs(J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]))


def chamfer(model: EdgeModel, lines: LinePoints, fit, tau_work: float) -> float:
    """Mean truncated squared distance (in work px, normalised to 0..1) from sketch points to matching target edges."""
    G = fit if isinstance(fit, np.ndarray) else fit.G
    p = apply_h(G, lines.pts) if isinstance(fit, np.ndarray) else fit(lines.pts)
    d, _, _ = model.nearest(p, normal_angles(G, lines.pts, lines.normals))
    d_work = d / model_px_per_work_px(G, lines.pts)
    return float(np.mean(np.minimum(d_work, tau_work) ** 2) / tau_work**2)


def refine(model: EdgeModel, lines: LinePoints, G0: np.ndarray, kind: str = "similarity", taus=None, iters: int = 4) -> Fit:
    """Oriented point-to-line ICP. ``taus`` are match radii in model px, coarse to fine."""
    if taus is None:
        taus = [r * model.long for r in (0.06, 0.035, 0.02, 0.012, 0.008)]
    q = lines.pts
    Tq, Tp = _norm(q, model.size)
    qn = apply_h(Tq, q)
    m_scale = 1 / Tp[0, 0]  # model px per normalised unit
    Gn = Tp @ G0 @ np.linalg.inv(Tq)
    if kind == "similarity":
        Gn = _to_matrix("similarity", _to_params("similarity", Gn))
    elif kind == "affine":
        Gn = Gn / Gn[2, 2]
        Gn[2] = [0, 0, 1]
    t = _to_params(kind, Gn)
    w = np.zeros(len(q))
    for tau in taus:
        for _ in range(iters):
            G = np.linalg.inv(Tp) @ _to_matrix(kind, t) @ Tq
            p = apply_h(G, q)
            d, e, ne = model.nearest(p, normal_angles(G, q, lines.normals))
            en = apply_h(Tp, e)
            x, y, Jx, Jy = _jac(kind, t, qn)
            r = ne[:, 0] * (x - en[:, 0]) + ne[:, 1] * (y - en[:, 1])
            r_px = r * m_scale
            ok = np.isfinite(d) & (d < tau)
            z = np.where(ok, r_px / tau, 1.0)
            w = np.where(ok & (np.abs(z) < 1), (1 - z**2) ** 2, 0.0)
            if w.sum() < 10:
                break
            J = ne[:, :1] * Jx + ne[:, 1:] * Jy
            A = (J * w[:, None]).T @ J
            g = (J * w[:, None]).T @ r
            A += np.eye(len(t)) * 1e-9 * np.trace(A)
            t = t - np.linalg.solve(A, g)
    G = np.linalg.inv(Tp) @ _to_matrix(kind, t) @ Tq
    G /= G[2, 2]
    fit = Fit(G, kind)
    sel = w > 0
    if sel.any():
        scale = model_px_per_work_px(G, q[sel])
        p = apply_h(G, q[sel])
        d, e, ne = model.nearest(p, normal_angles(G, q[sel], lines.normals[sel]))
        rr = np.abs(((p - e) * ne).sum(1)) / scale
        fit.rms = float(np.sqrt(np.mean(rr**2)))
    fit.matched = float(sel.mean())
    return fit


# ---------------------------------------------------------------- global search


@dataclass
class Candidate:
    score: float
    G: np.ndarray  # work px -> model px
    scale: float  # image long side in work px
    angle: float


def search(model: EdgeModel, lines: LinePoints, work_size, scales, angles, work_long: int = 256, pad: float = 0.6, per_pose: int = 3) -> list:
    """Brute-force scale x rotation, FFT over translation, on a small copy of the sketch."""
    W, H = work_size
    f = work_long / max(W, H)
    sw, sh = round(W * f), round(H * f)
    sk = np.zeros((sh, sw), np.float32)
    xy = np.round(lines.pts * f).astype(int)
    ok = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < sw) & (xy[:, 1] < sh)
    np.add.at(sk, (xy[ok, 1], xy[ok, 0]), 1.0)
    sk = np.minimum(sk, 2.0)
    cands = []
    mw, mh = model.size
    for s in scales:
        k = s * f / model.long  # small px per model px
        sigma = 1.2  # small px
        for a in angles:
            M = np.array([[k * np.cos(a), -k * np.sin(a), 0], [k * np.sin(a), k * np.cos(a), 0]])
            box = (M[:, :2] @ np.array([[0, mw, mw, 0], [0, 0, mh, mh]])).T
            lo = box.min(0)
            M[:, 2] = -lo
            tw, th = np.ceil(box.max(0) - lo).astype(int) + 1
            if tw < 4 or th < 4 or tw > 3 * sw or th > 3 * sh:
                continue
            dtw = cv2.warpAffine(model.dt_all, M, (tw, th), flags=cv2.INTER_LINEAR, borderValue=1e4) * k
            foot = cv2.warpAffine(np.ones((mh, mw), np.float32), M, (tw, th), flags=cv2.INTER_NEAREST) > 0.5
            att = np.exp(-(dtw**2) / (2 * sigma**2))
            T = np.where(foot, att - att[foot].mean(), 0).astype(np.float32)
            # the image may hang off the frame, but by less than ``pad`` of its size (under half in view is rejected below)
            px, py = int(np.ceil(pad * tw)), int(np.ceil(pad * th))
            big = cv2.copyMakeBorder(sk, py, py, px, px, cv2.BORDER_CONSTANT, value=0)
            R = cv2.matchTemplate(big, T, cv2.TM_CCORR)  # R[y, x]: template's top-left at big (x, y)
            Rd = cv2.dilate(R, np.ones((5, 5), np.uint8))
            peaks = np.argwhere((R >= Rd) & (R > 0))
            if not len(peaks):
                continue
            vals = R[peaks[:, 0], peaks[:, 1]]
            for i in np.argsort(-vals)[:per_pose]:
                ry, rx = peaks[i]
                # template top-left in sk coordinates
                ox, oy = rx - px, ry - py
                # model -> small:  M[:, :2] p + M[:, 2] + (ox, oy); small -> work: / f
                A = np.vstack([M, [0, 0, 1]])
                A[:2, 2] += (ox, oy)
                A = np.diag([1 / f, 1 / f, 1.0]) @ A  # model -> work
                inside_frac = _overlap(A, model.size, work_size)
                if inside_frac < 0.5:
                    continue
                cands.append(Candidate(float(vals[i]), np.linalg.inv(A), s, a))
    cands.sort(key=lambda c: -c.score)
    return cands


def _overlap(A: np.ndarray, msize, wsize) -> float:
    mw, mh = msize
    pts = np.stack(np.meshgrid(np.linspace(0, mw, 9), np.linspace(0, mh, 9)), -1).reshape(-1, 2)
    p = apply_h(A, pts)
    return float(np.mean((p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] <= wsize[0]) & (p[:, 1] <= wsize[1])))


def distinct(cands: list, n: int, min_shift: float = 0.08) -> list:
    """The best ``n`` candidates that are not near-duplicates of a better one."""
    out = []
    for c in cands:
        Ginv = np.linalg.inv(c.G)
        ctr = apply_h(Ginv, [(0.5, 0.5)])[0]
        dup = False
        for o in out:
            oc = apply_h(np.linalg.inv(o.G), [(0.5, 0.5)])[0]
            if np.hypot(*(ctr - oc)) < min_shift * max(c.scale, o.scale) and abs(np.log(c.scale / o.scale)) < 0.1 and abs((c.angle - o.angle + np.pi) % (2 * np.pi) - np.pi) < np.radians(8):
                dup = True
                break
        if not dup:
            out.append(c)
            if len(out) == n:
                break
    return out


# ---------------------------------------------------------------- bending


@dataclass
class Bend:
    """A smooth displacement field on a grid over the model: p -> p + d(p)."""

    size: tuple
    grid: tuple  # (gx, gy) nodes
    c: np.ndarray = field(default=None)  # (gy*gx, 2)

    def weights(self, p: np.ndarray):
        gx, gy = self.grid
        w, h = self.size
        u = np.clip(p[:, 0] / w * (gx - 1), 0, gx - 1 - 1e-9)
        v = np.clip(p[:, 1] / h * (gy - 1), 0, gy - 1 - 1e-9)
        i, j = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = u - i, v - j
        idx = np.stack([j * gx + i, j * gx + i + 1, (j + 1) * gx + i, (j + 1) * gx + i + 1], 1)
        wts = np.stack([(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv], 1)
        return idx, wts

    def disp(self, p) -> np.ndarray:
        p = np.asarray(p, float).reshape(-1, 2)
        idx, wts = self.weights(p)
        return (self.c[idx] * wts[..., None]).sum(1)


def _bending_matrix(gx: int, gy: int) -> np.ndarray:
    rows = []
    K = gx * gy
    for j in range(gy):
        for i in range(gx):
            if 0 < i < gx - 1:
                r = np.zeros(K)
                r[j * gx + i - 1], r[j * gx + i], r[j * gx + i + 1] = 1, -2, 1
                rows.append(r)
            if 0 < j < gy - 1:
                r = np.zeros(K)
                r[(j - 1) * gx + i], r[j * gx + i], r[(j + 1) * gx + i] = 1, -2, 1
                rows.append(r)
            if i < gx - 1 and j < gy - 1:
                r = np.zeros(K)
                r[j * gx + i], r[j * gx + i + 1], r[(j + 1) * gx + i], r[(j + 1) * gx + i + 1] = 1, -1, -1, 1
                rows.append(r * np.sqrt(2))
    D = np.array(rows)
    return D.T @ D


def bend(model: EdgeModel, lines: LinePoints, fit: Fit, cells: int = 7, lam: float = 0.3, taus=None, iters: int = 3, reverse: float = 0.0) -> Fit:
    """Add a smooth displacement grid (``cells`` across the long side) to a fitted
    placement. If ``fit`` is already bent, start from its displacement."""
    if taus is None:
        taus = [r * model.long for r in (0.02, 0.012, 0.008, 0.006)]
    mw, mh = model.size
    gx = cells + 1 if mw >= mh else max(2, round(cells * mw / mh)) + 1
    gy = cells + 1 if mh >= mw else max(2, round(cells * mh / mw)) + 1
    b = Bend((mw, mh), (gx, gy), np.zeros((gx * gy, 2)))
    if fit.bend is not None:
        nodes = np.stack(np.meshgrid(np.linspace(0, mw, gx), np.linspace(0, mh, gy)), -1).reshape(-1, 2)
        b.c = fit.bend.disp(nodes)
    q = lines.pts
    p0 = apply_h(fit.G, q)
    ang = normal_angles(fit.G, q, lines.normals)
    idx, wts = b.weights(p0)
    K = gx * gy
    R = _bending_matrix(gx, gy)
    if reverse > 0:
        # the target's strongest edges: a painter outlines these, so each should have a sketch line nearby
        e_all = model.edges
        strong = e_all.strength >= np.percentile(e_all.strength, 60)
        rev_pts, rev_n = e_all.pts[strong], e_all.normals[strong]
        rev_ang = np.arctan2(rev_n[:, 1], rev_n[:, 0]) % np.pi
    for tau in taus:
        for _ in range(iters):
            p = p0 + b.disp(p0)
            d, e, ne = model.nearest(p, ang)
            ok = np.isfinite(d) & (d < tau)
            j = np.flatnonzero(ok)  # pairs: sketch point j <-> target edge point e (normal ne)
            pe, pn = e[ok], ne[ok]
            pw = np.ones(len(j))
            if reverse > 0:
                from scipy.spatial import cKDTree

                tree = cKDTree(p)
                dd, nn = tree.query(rev_pts, k=6, distance_upper_bound=tau)
                good = np.isfinite(dd)
                # nearest sketch point running the same way (within ~30 degrees)
                diff = np.abs((ang[np.minimum(nn, len(p) - 1)] - rev_ang[:, None] + np.pi / 2) % np.pi - np.pi / 2)
                good &= diff < np.radians(30)
                first = np.where(good.any(1), np.argmax(good, 1), -1)
                has = first >= 0
                jr = nn[has, first[has]]
                j = np.concatenate([j, jr])
                pe = np.vstack([pe, rev_pts[has]])
                pn = np.vstack([pn, rev_n[has]])
                pw = np.concatenate([pw, np.full(len(jr), reverse)])
            r0 = ((p0[j] - pe) * pn).sum(1)  # residual with no bending
            r = ((p[j] - pe) * pn).sum(1)
            z = r / tau
            w = np.where(np.abs(z) < 1, (1 - z**2) ** 2, 0.0) * pw
            keep_ = w > 0
            if keep_.sum() < 20:
                break
            j, pn, r0, ws = j[keep_], pn[keep_], r0[keep_], w[keep_]
            # rows: [ne_x * phi, ne_y * phi] over 2K unknowns
            n = len(j)
            Arow = np.zeros((n, 2 * K))
            ii = np.repeat(np.arange(n), 4)
            Arow[ii, idx[j].ravel()] = (wts[j] * pn[:, :1]).ravel()
            Arow[ii, K + idx[j].ravel()] = (wts[j] * pn[:, 1:]).ravel()
            A = (Arow * ws[:, None]).T @ Arow
            g = (Arow * ws[:, None]).T @ r0
            reg = lam * ws.sum() / K * (np.kron(np.eye(2), R) + 1e-4 * np.eye(2 * K))
            z = np.linalg.solve(A + reg, -g)
            b.c = np.column_stack([z[:K], z[K:]])
    out = Fit(fit.G, fit.kind, fit.matched, fit.rms, fit.score, b)
    return out


# ---------------------------------------------------------------- the whole thing


@dataclass
class Alignment:
    fit: Fit  # final: placement plus bending
    rigid: Fit  # placement only (similarity, or perspective for a raw photo)
    second_score: float  # score of the best clearly different placement (ambiguity check)
    seconds: dict


def snap_candidates(model: EdgeModel, lines: LinePoints, work_size, G0: np.ndarray, scale_range: float = 0.25, angle_range: float = 12.0, shift: float = 0.25) -> list:
    """Search near a rough placement ``G0``: scales and turns around it, any
    shift, keeping only placements whose centre is within ``shift`` of the
    rough one's (as a share of the image size)."""
    A0 = np.linalg.inv(G0)
    s0 = np.hypot(A0[0, 0], A0[1, 0]) / A0[2, 2] * model.long
    a0 = np.arctan2(A0[1, 0], A0[0, 0])
    scales = s0 * np.exp(np.arange(-scale_range, scale_range + 1e-9, np.log(1.06)))
    angles = a0 + np.radians(np.arange(-angle_range, angle_range + 1e-9, 3.0))
    c0 = apply_h(A0, [(model.size[0] / 2, model.size[1] / 2)])[0]
    out = []
    for c in search(model, lines, work_size, scales, angles):
        cc = apply_h(np.linalg.inv(c.G), [(model.size[0] / 2, model.size[1] / 2)])[0]
        if np.hypot(*(cc - c0)) < shift * s0:
            out.append(c)
    return out


def align(model: EdgeModel, lines: LinePoints, work_size, start: np.ndarray | None = None, rotation: str = "upright", final: str = "affine", n_starts: int = 6, bend_levels=((2, (0.04, 0.03, 0.02)), (4, (0.025, 0.018, 0.012)), (8, (0.012, 0.008, 0.006))), lam: float = 0.03, reverse: float = 0.5, max_points: int = 8000, rng=None, scale_range=(0.3, 1.15)) -> Alignment:
    """Find the target in the sketch. ``start``: a rough placement (work px ->
    model px) to snap from; otherwise search everywhere, upright (+-15 deg) or
    at any rotation. ``final``: "affine" for a straightened photo, "homography"
    for a raw one. ``bend_levels``: (cells, match radii as a share of the model
    size), coarse to fine; empty for no bending."""
    import time

    secs = {}
    rng = np.random.default_rng(0) if rng is None else rng
    pts = lines
    if len(lines.pts) > max_points:
        sel = rng.choice(len(lines.pts), max_points, replace=False)
        pts = LinePoints(lines.pts[sel], lines.normals[sel], lines.strength[sel])
    W, H = work_size
    L = model.long
    t = time.perf_counter()
    if start is not None:
        cands = snap_candidates(model, lines, work_size, start)
    else:
        lo, hi = scale_range[0] * max(W, H), scale_range[1] * max(W, H)  # image long side vs photo long side
        scales = np.exp(np.arange(np.log(lo), np.log(hi), np.log(1.07)))
        angles = np.radians(np.arange(-15, 15.1, 3)) if rotation == "upright" else np.radians(np.arange(0, 360, 4))
        cands = search(model, lines, work_size, scales, angles)
    cands = distinct(cands, n_starts)
    secs["search"] = time.perf_counter() - t
    t = time.perf_counter()
    tau = 0.01 * max(W, H)
    fits = [refine(model, pts, c.G, "similarity") for c in cands]
    if start is not None:
        fits.append(refine(model, pts, start, "similarity", [r * L for r in (0.12, 0.07, 0.04, 0.025, 0.015, 0.01, 0.008)]))
    for f_ in fits:
        f_.score = chamfer(model, pts, f_, tau)
    fits.sort(key=lambda f_: f_.score)
    best = fits[0]
    # the best placement that is clearly different (centre moved by over 5% of the image)
    second = 1.0
    c_best = apply_h(np.linalg.inv(best.G), [(model.size[0] / 2, model.size[1] / 2)])[0]
    size = L / np.sqrt(abs(np.linalg.det(best.G[:2, :2])))
    for f_ in fits[1:]:
        c = apply_h(np.linalg.inv(f_.G), [(model.size[0] / 2, model.size[1] / 2)])[0]
        if np.hypot(*(c - c_best)) > 0.05 * size:
            second = f_.score
            break
    rigid = best
    if final == "homography":  # through affine first: a tilted photo can be far from any similarity
        rigid = refine(model, pts, rigid.G, "affine", [r * L for r in (0.06, 0.04, 0.025)])
    if final != "similarity":
        rigid = refine(model, pts, rigid.G, final, [r * L for r in (0.04, 0.025, 0.015, 0.01)])
        rigid.score = chamfer(model, pts, rigid, tau)
    secs["rigid"] = time.perf_counter() - t
    t = time.perf_counter()
    fit = rigid
    for cells, radii in bend_levels:
        fit = bend(model, pts, fit, cells, lam, [r * L for r in radii], reverse=reverse)
    if bend_levels:
        fit.score = chamfer(model, pts, fit, tau)
    secs["bend"] = time.perf_counter() - t
    return Alignment(fit, rigid, second, secs)
