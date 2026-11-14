package io.github.admoreau.paintbynumbers.tracking

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Turns scan keyframes into a map of the canvas: X positions on the true (flat, undistorted) canvas
 * plane, found together with the camera pose of every keyframe by bundle adjustment.
 *
 * With the camera's intrinsics known, views from different positions pin down the plane's real
 * shape; no canvas size or corners are needed. The map's origin and orientation are arbitrary and
 * its scale is set so the X's come out [xSizeMm] across, so map units are roughly millimetres.
 */
object BundleAdjust {
    class Result(
        val marks: List<MapMark>,
        val rmsePx: Double, // reprojection error over all observations
        val viewSpreadDeg: Double, // widest angle between two keyframes' views of the canvas: low = shape poorly determined
        val keyframes: Int,
    )

    private class Obs(val kf: Int, val point: Int, val u: Double, val v: Double, val size: Double, val dir1: Double, val dir2: Double)

    /**
     * [keep] limits the map to these X ids (the scan's reliable ones). X's that still do not fit the
     * flat-canvas solution (stray detections that happened to track) are dropped and the rest re-solved.
     */
    fun run(keyframes: List<Keyframe>, refPositions: List<DoubleArray>, cam: Intrinsics, xSizeMm: Double = 15.0,
            keep: Set<Int>? = null, pnp: PnpSolver = OpenCvPnp): Result? {
        var allowed = keep
        repeat(3) {
            val first = solve(keyframes, refPositions, cam, xSizeMm, allowed, pnp) ?: return null
            val bad = first.second
            if (bad.isEmpty()) return first.first
            allowed = (allowed ?: refPositions.indices.toSet()) - bad
        }
        return solve(keyframes, refPositions, cam, xSizeMm, allowed, pnp)?.first
    }

    /** One solve: the result, and the ids of X's that fit badly. */
    private fun solve(keyframes: List<Keyframe>, refPositions: List<DoubleArray>, cam: Intrinsics, xSizeMm: Double,
                      keep: Set<Int>?, pnp: PnpSolver): Pair<Result, Set<Int>>? {
        if (keyframes.size < 2) return null
        val counts = HashMap<Int, Int>()
        for (kf in keyframes) for (o in kf.obs) if (keep == null || o.id in keep) counts[o.id] = (counts[o.id] ?: 0) + 1
        val ids = counts.filter { it.value >= 2 }.keys.sorted()
        if (ids.size < 4) return null
        val index = ids.withIndex().associate { (i, id) -> id to i }
        val obs = ArrayList<Obs>()
        for ((k, kf) in keyframes.withIndex()) for (o in kf.obs) index[o.id]?.let { obs += Obs(k, it, o.x, o.y, o.size, o.dir1, o.dir2) }
        val nK = keyframes.size
        val nP = ids.size

        // start: the first keyframe faces the canvas, at the distance that makes X's xSizeMm across
        val sizes0 = keyframes[0].obs.map { it.size }.sorted()
        val depth = cam.fx * xSizeMm / sizes0[sizes0.size / 2]
        val pts = DoubleArray(2 * nP)
        for ((i, id) in ids.withIndex()) {
            val r = refPositions[id]
            pts[2 * i] = (r[0] - cam.cx) * depth / cam.fx
            pts[2 * i + 1] = (r[1] - cam.cy) * depth / cam.fy
        }
        val poses = Array(nK) { doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, depth) }
        for (k in 1 until nK) poses[k] = initialPose(obs.filter { it.kf == k }, pts, cam, pnp) ?: return null

        // gauge: hold two well-observed, far-apart X's fixed (position, rotation and scale of the map)
        val perPoint = IntArray(nP).also { c -> for (o in obs) c[o.point]++ }
        val a = perPoint.indices.maxBy { perPoint[it] }
        val good = perPoint.indices.filter { it != a && perPoint[it] >= max(2, perPoint[a] / 2) }.ifEmpty { perPoint.indices.filter { it != a } }
        val b = good.maxBy { hypot(pts[2 * it] - pts[2 * a], pts[2 * it + 1] - pts[2 * a + 1]) }
        val pointCol = IntArray(nP) { -1 }
        var nParams = 6 * nK
        for (i in 0 until nP) if (i != a && i != b) { pointCol[i] = nParams; nParams += 2 }

        optimize(obs, poses, pts, pointCol, nParams, cam)

        // scale so X's are xSizeMm across
        val est = ArrayList<Double>()
        for (o in obs) est += o.size / pxPerUnit(poses[o.kf], pts[2 * o.point], pts[2 * o.point + 1], cam)
        val scale = xSizeMm / est.sorted()[est.size / 2]
        for (i in pts.indices) pts[i] *= scale
        for (p in poses) for (k in 3 until 6) p[k] *= scale

        // per-X stroke angles (averaged over views) and sizes
        val marks = ArrayList<MapMark>()
        for (i in 0 until nP) {
            val mine = obs.filter { it.point == i }
            val angles = mine.mapNotNull { o -> planeAngles(o, poses[o.kf], pts[2 * i], pts[2 * i + 1], cam) }
            if (angles.isEmpty()) continue
            val ref = angles[0]
            var s1 = 0.0; var c1 = 0.0; var s2 = 0.0; var c2 = 0.0
            for (an in angles) {
                val straight = angleDev(an[0], ref[0]) + angleDev(an[1], ref[1])
                val swapped = angleDev(an[1], ref[0]) + angleDev(an[0], ref[1])
                val (p, q) = if (straight <= swapped) an[0] to an[1] else an[1] to an[0]
                s1 += sin(2 * p); c1 += cos(2 * p); s2 += sin(2 * q); c2 += cos(2 * q)
            }
            val d1 = mod(atan2(s1, c1) / 2, PI)
            val d2 = mod(atan2(s2, c2) / 2, PI)
            val size = mine.map { it.size / pxPerUnit(poses[it.kf], pts[2 * i], pts[2 * i + 1], cam) }.sorted().let { it[it.size / 2] }
            marks += MapMark(ids[i], pts[2 * i], pts[2 * i + 1], d1, d2, size) // keeps the scan's X id
        }

        var ss = 0.0
        val out = DoubleArray(2)
        val perPointSs = DoubleArray(nP)
        for (o in obs) {
            PoseProjector(cam, poses[o.kf]).project(pts[2 * o.point], pts[2 * o.point + 1], out)
            val e = (out[0] - o.u) * (out[0] - o.u) + (out[1] - o.v) * (out[1] - o.v)
            ss += e
            perPointSs[o.point] += e
        }
        val pointRms = DoubleArray(nP) { sqrt(perPointSs[it] / perPoint[it]) }
        val typical = pointRms.sorted()[nP / 2]
        val bad = (0 until nP).filter { pointRms[it] > max(2.5, 3 * typical) }.map { ids[it] }.toSet()
        return Result(marks, sqrt(ss / obs.size), viewSpread(poses, pts), nK) to bad
    }

    private fun mod(a: Double, m: Double) = ((a % m) + m) % m
    private fun angleDev(a: Double, b: Double) = abs(mod(a - b + PI / 2, PI) - PI / 2)

    private fun initialPose(obs: List<Obs>, pts: DoubleArray, cam: Intrinsics, pnp: PnpSolver): DoubleArray? {
        if (obs.size < 4) return null
        val map = DoubleArray(2 * obs.size)
        val img = DoubleArray(2 * obs.size)
        for ((k, o) in obs.withIndex()) {
            map[2 * k] = pts[2 * o.point]; map[2 * k + 1] = pts[2 * o.point + 1]
            img[2 * k] = o.u; img[2 * k + 1] = o.v
        }
        val out = DoubleArray(2)
        return pnp.solve(map, img, cam).filter { PoseProjector(cam, it).height() > 0 }.minByOrNull { x ->
            val p = PoseProjector(cam, x)
            obs.withIndex().sumOf { (k, o) ->
                p.project(map[2 * k], map[2 * k + 1], out)
                if (out[0].isFinite()) hypot(out[0] - o.u, out[1] - o.v) else 1e9
            }
        }
    }

    private fun pxPerUnit(x: DoubleArray, px: Double, py: Double, cam: Intrinsics): Double {
        val p = PoseProjector(cam, x)
        val a = p.project(px, py, DoubleArray(2))
        val b = p.project(px + 1, py, DoubleArray(2))
        val c = p.project(px, py + 1, DoubleArray(2))
        return sqrt(abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))
    }

    /** The detection's stroke angles carried onto the canvas plane through this keyframe's pose. */
    private fun planeAngles(o: Obs, pose: DoubleArray, px: Double, py: Double, cam: Intrinsics): DoubleArray? {
        val hi = M3.inv(PoseProjector(cam, pose).homography()) ?: return null
        val c = M3.apply(hi, o.u, o.v)
        return doubleArrayOf(o.dir1, o.dir2).map { ang ->
            val q = M3.apply(hi, o.u + 5 * cos(ang), o.v + 5 * sin(ang))
            mod(atan2(q[1] - c[1], q[0] - c[0]), PI)
        }.toDoubleArray().takeIf { it.all { v -> v.isFinite() } && px.isFinite() && py.isFinite() }
    }

    /** Widest angle between the directions from which two keyframes looked at the map's centre. */
    private fun viewSpread(poses: Array<DoubleArray>, pts: DoubleArray): Double {
        val n = pts.size / 2
        val cx = (0 until n).sumOf { pts[2 * it] } / n
        val cy = (0 until n).sumOf { pts[2 * it + 1] } / n
        val dirs = poses.map { x ->
            val r = rodrigues(x)
            // camera centre C = -R^T t
            val c0 = -(r[0] * x[3] + r[3] * x[4] + r[6] * x[5])
            val c1 = -(r[1] * x[3] + r[4] * x[4] + r[7] * x[5])
            val c2 = -(r[2] * x[3] + r[5] * x[4] + r[8] * x[5])
            val d = doubleArrayOf(cx - c0, cy - c1, -c2)
            val len = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            DoubleArray(3) { d[it] / len }
        }
        var worst = 0.0
        for (i in dirs.indices) for (j in i + 1 until dirs.size) {
            val dot = (dirs[i][0] * dirs[j][0] + dirs[i][1] * dirs[j][1] + dirs[i][2] * dirs[j][2]).coerceIn(-1.0, 1.0)
            worst = max(worst, Math.toDegrees(acos(dot)))
        }
        return worst
    }

    /** Levenberg-Marquardt over all poses and free X positions, robust (soft L1, 2 px) to stray matches. */
    private fun optimize(obs: List<Obs>, poses: Array<DoubleArray>, pts: DoubleArray, pointCol: IntArray, n: Int, cam: Intrinsics) {
        val f2 = 4.0
        fun residuals(): DoubleArray {
            val r = DoubleArray(2 * obs.size)
            val out = DoubleArray(2)
            val proj = poses.map { PoseProjector(cam, it) }
            for ((k, o) in obs.withIndex()) {
                proj[o.kf].projectFinite(pts[2 * o.point], pts[2 * o.point + 1], out)
                r[2 * k] = out[0] - o.u; r[2 * k + 1] = out[1] - o.v
            }
            return r
        }
        fun cost(r: DoubleArray) = r.sumOf { f2 * (sqrt(1 + it * it / f2) - 1) }

        var r = residuals()
        var c = cost(r)
        var lambda = 1e-3
        val a = DoubleArray(n * n)
        val g = DoubleArray(n)
        val jPose = Array(2) { DoubleArray(6) }
        val jPt = Array(2) { DoubleArray(2) }
        val out = DoubleArray(2)
        val base = DoubleArray(2)
        repeat(60) {
            a.fill(0.0); g.fill(0.0)
            for ((k, o) in obs.withIndex()) {
                val pose = poses[o.kf]
                val px = pts[2 * o.point]; val py = pts[2 * o.point + 1]
                PoseProjector(cam, pose).projectFinite(px, py, base)
                for (q in 0 until 6) {
                    val h = if (q < 3) 1e-6 else 1e-3
                    val pp = pose.copyOf().also { it[q] += h }
                    PoseProjector(cam, pp).projectFinite(px, py, out)
                    jPose[0][q] = (out[0] - base[0]) / h; jPose[1][q] = (out[1] - base[1]) / h
                }
                val proj = PoseProjector(cam, pose)
                proj.projectFinite(px + 1e-3, py, out)
                jPt[0][0] = (out[0] - base[0]) / 1e-3; jPt[1][0] = (out[1] - base[1]) / 1e-3
                proj.projectFinite(px, py + 1e-3, out)
                jPt[0][1] = (out[0] - base[0]) / 1e-3; jPt[1][1] = (out[1] - base[1]) / 1e-3
                val pc = 6 * o.kf
                val qc = pointCol[o.point]
                for (row in 0 until 2) {
                    val res = r[2 * k + row]
                    val w = 1.0 / sqrt(1 + res * res / f2)
                    // columns touched by this residual: 6 pose columns, then 2 point columns if free
                    val cols = IntArray(if (qc >= 0) 8 else 6) { if (it < 6) pc + it else qc + it - 6 }
                    val vals = DoubleArray(cols.size) { if (it < 6) jPose[row][it] else jPt[row][it - 6] }
                    for (i in cols.indices) {
                        g[cols[i]] += w * vals[i] * res
                        for (j in cols.indices) a[cols[i] * n + cols[j]] += w * vals[i] * vals[j]
                    }
                }
            }
            var improved = false
            for (attempt in 0 until 8) {
                val damped = a.copyOf()
                for (i in 0 until n) damped[i * n + i] += lambda * max(a[i * n + i], 1e-9)
                val step = solveLinear(damped, DoubleArray(n) { -g[it] }, n)
                if (step == null) { lambda *= 10; continue }
                val oldPoses = poses.map { it.copyOf() }
                val oldPts = pts.copyOf()
                for (kf in poses.indices) for (q in 0 until 6) poses[kf][q] += step[6 * kf + q]
                for (i in pointCol.indices) if (pointCol[i] >= 0) { pts[2 * i] += step[pointCol[i]]; pts[2 * i + 1] += step[pointCol[i] + 1] }
                val rn = residuals()
                val cn = cost(rn)
                if (cn.isFinite() && cn < c) {
                    val drop = (c - cn) / max(c, 1e-12)
                    r = rn; c = cn
                    lambda = max(lambda / 3, 1e-9)
                    improved = drop > 1e-9
                    break
                }
                for (kf in poses.indices) oldPoses[kf].copyInto(poses[kf])
                oldPts.copyInto(pts)
                lambda *= 10
            }
            if (!improved) return
        }
    }
}
