package io.github.admoreau.paintbynumbers.tracking

import org.opencv.calib3d.Calib3d
import org.opencv.core.CvException
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/** One X in the canvas map: position and stroke angles in canvas mm. */
data class MapMark(val id: Int, val x: Double, val y: Double, val d1: Double, val d2: Double, val sizeMm: Double)

enum class TrackMode { COLD, TRACK, HOLD }

/** Where the camera is relative to the canvas. */
class PoseEstimate(
    val x: DoubleArray, // world -> camera rotation vector (3) and translation in mm (3)
    val H: DoubleArray, // canvas mm -> image buffer pixels
    val matches: Map<Int, Int>, // detection index -> map index
    val rmse: Double, // reprojection error of the matched X's, px
    val mode: TrackMode,
    val uncertaintyMm: Double = Double.POSITIVE_INFINITY, // how far the overlay could plausibly be off (1 sigma, worst place in view)
)

/** Camera poses consistent with point correspondences, used as starting points. */
fun interface PnpSolver {
    /** [map] and [img] are interleaved (x, y) pairs; returns poses (rvec, tvec). */
    fun solve(map: DoubleArray, img: DoubleArray, cam: Intrinsics): List<DoubleArray>
}

/** OpenCV's planar solver (IPPE) for 4+ points, the three-point solver otherwise. */
object OpenCvPnp : PnpSolver {
    override fun solve(map: DoubleArray, img: DoubleArray, cam: Intrinsics): List<DoubleArray> {
        val n = map.size / 2
        val obj = MatOfPoint3f(*Array(n) { Point3(map[2 * it], map[2 * it + 1], 0.0) })
        val pts = MatOfPoint2f(*Array(n) { Point(img[2 * it], img[2 * it + 1]) })
        val k = Mat(3, 3, CvType.CV_64F).apply { put(0, 0, *cam.K) }
        val dist = MatOfDouble()
        val rvecs = ArrayList<Mat>()
        val tvecs = ArrayList<Mat>()
        try {
            if (n >= 4) Calib3d.solvePnPGeneric(obj, pts, k, dist, rvecs, tvecs, false, Calib3d.SOLVEPNP_IPPE)
            else Calib3d.solveP3P(obj, pts, k, dist, rvecs, tvecs, Calib3d.SOLVEPNP_P3P)
        } catch (e: CvException) {
            return emptyList()
        } finally {
            obj.release(); pts.release(); k.release(); dist.release()
        }
        val out = rvecs.indices.map { i ->
            val r = DoubleArray(3).also { rvecs[i].get(0, 0, it) }
            val t = DoubleArray(3).also { tvecs[i].get(0, 0, it) }
            r + t
        }
        (rvecs + tvecs).forEach { it.release() }
        return out
    }
}

/** Fixed-radius nearest-neighbour lookups among map marks. */
private class PointGrid(private val xs: DoubleArray, private val ys: DoubleArray, private val cell: Double) {
    private val x0 = xs.minOrNull() ?: 0.0
    private val y0 = ys.minOrNull() ?: 0.0
    private val nx = (((xs.maxOrNull() ?: 0.0) - x0) / cell).toInt() + 1
    private val ny = (((ys.maxOrNull() ?: 0.0) - y0) / cell).toInt() + 1
    private val start = IntArray(nx * ny + 1)
    private val items: IntArray

    init {
        val cellOf = IntArray(xs.size) { cellIndex(xs[it], ys[it]) }
        for (c in cellOf) start[c + 1]++
        for (c in 0 until nx * ny) start[c + 1] += start[c]
        val fill = start.copyOf()
        items = IntArray(xs.size)
        for (i in xs.indices) items[fill[cellOf[i]]++] = i
    }

    private fun cellIndex(x: Double, y: Double): Int {
        val cx = ((x - x0) / cell).toInt().coerceIn(0, nx - 1)
        val cy = ((y - y0) / cell).toInt().coerceIn(0, ny - 1)
        return cy * nx + cx
    }

    /** Index of the nearest point if it is closer than [cell], else -1; distance in [dist][0]. */
    fun nearestWithin(x: Double, y: Double, dist: DoubleArray): Int {
        val fx = floor((x - x0) / cell).toInt()
        val fy = floor((y - y0) / cell).toInt()
        var best = -1
        var bestD2 = cell * cell
        for (cy in fy - 1..fy + 1) {
            if (cy < 0 || cy >= ny) continue
            for (cx in fx - 1..fx + 1) {
                if (cx < 0 || cx >= nx) continue
                val c = cy * nx + cx
                for (k in start[c] until start[c + 1]) {
                    val i = items[k]
                    val dx = xs[i] - x
                    val dy = ys[i] - y
                    val d2 = dx * dx + dy * dy
                    if (d2 < bestD2) { bestD2 = d2; best = i }
                }
            }
        }
        dist[0] = sqrt(bestD2)
        return best
    }
}

private fun mod(a: Double, m: Double): Double = ((a % m) + m) % m

/** Difference between line angles (mod pi), in [0, pi/2]. */
private fun angleDev(a: Double, b: Double): Double = abs(mod(a - b + PI / 2, PI) - PI / 2)

/** Worst stroke-angle disagreement between two X's under the better pairing of strokes. */
private fun dirDev(a1: Double, a2: Double, b1: Double, b2: Double): Double =
    min(max(angleDev(a1, b1), angleDev(a2, b2)), max(angleDev(a1, b2), angleDev(a2, b1)))

private fun median(v: DoubleArray): Double {
    val s = v.sorted()
    val n = s.size
    return if (n % 2 == 1) s[n / 2] else 0.5 * (s[n / 2 - 1] + s[n / 2])
}

/**
 * Works out where the camera is from the X marks it sees. Port of
 * prototypes/x_tracking/xtrack/locate.py (see that file for the reasoning):
 *
 * - [coldStart]: no idea where we are. Pairs of detected X's against pairs of map X's give
 *   candidate camera positions; the best few are refined with the full camera model and the
 *   answer is accepted only if it is unambiguous, well spread and certain enough.
 * - [track]: we knew where we were. Match detections to predicted map X's and refine, gently
 *   anchored to the previous pose so even one or two visible X's hold the overlay in place.
 *
 * Coordinates: detections and [cam] are in pixels of the analysed image buffer ([width] x [height]).
 */
class Locator(
    marks: List<MapMark>,
    val cam: Intrinsics,
    val width: Int,
    val height: Int,
    val xSizeMm: Double = 15.0,
    dirTolDeg: Double = 15.0,
    val residualMm: Double = 1.5,
    val minCold: Int = 4,
    private val pnp: PnpSolver = OpenCvPnp,
) {
    val marks: List<MapMark> = marks.toList()
    private val m = marks.size
    private val mx = DoubleArray(m) { marks[it].x }
    private val my = DoubleArray(m) { marks[it].y }
    private val md1 = DoubleArray(m) { marks[it].d1 }
    private val md2 = DoubleArray(m) { marks[it].d2 }
    private val msize = DoubleArray(m) { marks[it].sizeMm }
    private val sizeTol = ln(1.6) // a detection may look up to 1.6x bigger or smaller than predicted
    var minCoverage = 0.4 // cold start: at least this share of the X's predicted in view must be found
    var maxColdUncertaintyMm = 10.0 // cold start: refuse locks shakier than this
    private val dirTol = Math.toRadians(dirTolDeg)
    private val f = cam.fx

    /** Median distance from each X to its nearest neighbour, mm. */
    val spacing: Double = if (m < 2) 100.0 else median(DoubleArray(m) { i ->
        (0 until m).filter { it != i }.minOf { hypot(mx[it] - mx[i], my[it] - my[i]) }
    })

    private val pairA: IntArray
    private val pairB: IntArray
    private val pairLen: DoubleArray
    private val pairAng: DoubleArray

    init {
        val all = ArrayList<IntArray>()
        for (a in 0 until m) for (b in 0 until m) if (a != b) all += intArrayOf(a, b)
        val len = all.map { hypot(mx[it[1]] - mx[it[0]], my[it[1]] - my[it[0]]) }
        val order = all.indices.sortedBy { len[it] }
        pairA = IntArray(order.size) { all[order[it]][0] }
        pairB = IntArray(order.size) { all[order[it]][1] }
        pairLen = DoubleArray(order.size) { len[order[it]] }
        pairAng = DoubleArray(order.size) { atan2(my[pairB[it]] - my[pairA[it]], mx[pairB[it]] - mx[pairA[it]]) }
    }

    private val tolMm = max(6.0, 0.3 * spacing)
    private val grid = PointGrid(mx, my, tolMm)
    private val ls = LeastSquares(6, doubleArrayOf(0.1, 0.1, 0.1, 10.0, 10.0, 10.0), fScale = 2.0, maxIter = 60)
    private var lastJac: Array<DoubleArray>? = null
    private val tmp = DoubleArray(2)
    private val tmp2 = DoubleArray(2)
    private val tmp3 = DoubleArray(2)

    // ---------------------------------------------------------------- helpers

    fun homography(x: DoubleArray): DoubleArray = PoseProjector(cam, x).homography()

    fun cameraHeight(x: DoubleArray): Double = PoseProjector(cam, x).height()

    /** Pixels per mm near the image centre for this pose. */
    private fun pxPerMm(x: DoubleArray): Double {
        val p = PoseProjector(cam, x)
        val hi = M3.inv(p.homography()) ?: return 1.0
        val c = M3.apply(hi, cam.cx, cam.cy)
        val a = p.project(c[0], c[1], DoubleArray(2))
        val b = p.project(c[0] + 1, c[1], DoubleArray(2))
        val d = p.project(c[0], c[1] + 1, DoubleArray(2))
        val v = sqrt(abs((b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0])))
        return if (v.isFinite()) v else 1.0
    }

    /** Pixels per mm at map X [i]. */
    private fun markScale(p: PoseProjector, i: Int): Double {
        val a = p.project(mx[i], my[i], tmp)
        val ax = a[0]; val ay = a[1]
        val b = p.project(mx[i] + 1, my[i], tmp2)
        val d = p.project(mx[i], my[i] + 1, tmp3)
        return sqrt(abs((b[0] - ax) * (d[1] - ay) - (b[1] - ay) * (d[0] - ax)))
    }

    private fun predictedInView(x: DoubleArray): Int {
        val p = PoseProjector(cam, x)
        val ex = 0.03 * width
        val ey = 0.03 * height
        var n = 0
        for (i in 0 until m) {
            val q = p.project(mx[i], my[i], tmp)
            if (q[0].isFinite() && q[0] > ex && q[0] < width - ex && q[1] > ey && q[1] < height - ey) n++
        }
        return n
    }

    /** The image angle of map X [i]'s stroke [k] (0 or 1). */
    private fun predDir(p: PoseProjector, i: Int, k: Int): Double {
        val ang = if (k == 0) md1[i] else md2[i]
        val a = p.project(mx[i], my[i], tmp)
        val ax = a[0]; val ay = a[1]
        val b = p.project(mx[i] + 5 * cos(ang), my[i] + 5 * sin(ang), tmp2)
        return mod(atan2(b[1] - ay, b[0] - ax), PI)
    }

    private class Solved(val x: DoubleArray, val rmse: Double)

    /** Refine pose [x0] so map points [mIdx] land on image points [img] (interleaved x, y). */
    private fun solve(x0: DoubleArray, img: DoubleArray, mIdx: IntArray, prior: DoubleArray? = null, wPrior: Double = 0.0): Solved {
        val n = mIdx.size
        val usePrior = prior != null && wPrior > 0
        val z = if (usePrior) max(PoseProjector(cam, prior!!).height(), 10.0) else 1.0
        val out = DoubleArray(2)
        val fn = { x: DoubleArray ->
            val p = PoseProjector(cam, x)
            val r = DoubleArray(2 * n + if (usePrior) 6 else 0)
            for (k in 0 until n) {
                p.projectFinite(mx[mIdx[k]], my[mIdx[k]], out)
                r[2 * k] = out[0] - img[2 * k]
                r[2 * k + 1] = out[1] - img[2 * k + 1]
            }
            if (usePrior) {
                for (k in 0 until 3) r[2 * n + k] = wPrior * (x[k] - prior!![k]) * f
                for (k in 0 until 3) r[2 * n + 3 + k] = wPrior * (x[3 + k] - prior!![3 + k]) * f / z
            }
            r
        }
        val res = ls.solve(x0, fn)
        lastJac = res.jac
        val p = PoseProjector(cam, res.x)
        var ss = 0.0
        for (k in 0 until n) {
            p.projectFinite(mx[mIdx[k]], my[mIdx[k]], out)
            val dx = out[0] - img[2 * k]
            val dy = out[1] - img[2 * k + 1]
            ss += dx * dx + dy * dy
        }
        return Solved(res.x, sqrt(ss / n))
    }

    /** 1-sigma overlay uncertainty in canvas mm at the worst of nine points across the view,
     *  propagated from the last solve's Jacobian (detection noise plus map noise). */
    private fun uncertainty(x: DoubleArray, mapNoiseMm: Double = 0.5, detectNoisePx: Double = 1.0): Double {
        val jac = lastJac ?: return Double.POSITIVE_INFINITY
        val sigma = hypot(detectNoisePx, mapNoiseMm * pxPerMm(x))
        val jtj = DoubleArray(36)
        for (row in jac) for (p in 0 until 6) for (q in 0 until 6) jtj[p * 6 + q] += row[p] * row[q]
        val cov = invert(jtj, 6) ?: return Double.POSITIVE_INFINITY
        for (i in cov.indices) cov[i] *= sigma * sigma
        val gx = doubleArrayOf(0.1 * width, 0.5 * width, 0.9 * width)
        val gy = doubleArrayOf(0.1 * height, 0.5 * height, 0.9 * height)
        fun canvasPoints(xx: DoubleArray): DoubleArray? {
            val hi = M3.inv(homography(xx)) ?: return null
            val o = DoubleArray(18)
            var k = 0
            for (u in gx) for (v in gy) {
                val q = M3.apply(hi, u, v)
                o[k++] = q[0]; o[k++] = q[1]
            }
            return o
        }
        val base = canvasPoints(x) ?: return Double.POSITIVE_INFINITY
        val g = Array(6) { DoubleArray(18) }
        val eps = doubleArrayOf(1e-5, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3)
        for (k in 0 until 6) {
            val xx = x.copyOf().also { it[k] += eps[k] }
            val q = canvasPoints(xx) ?: return Double.POSITIVE_INFINITY
            for (i in 0 until 18) g[k][i] = (q[i] - base[i]) / eps[k]
        }
        var worst = 0.0
        for (pt in 0 until 9) {
            var v = 0.0
            for (c in 0 until 2) {
                val i = 2 * pt + c
                for (p in 0 until 6) for (q in 0 until 6) v += g[p][i] * cov[p * 6 + q] * g[q][i]
            }
            if (!v.isFinite()) return Double.POSITIVE_INFINITY
            worst = max(worst, v)
        }
        return sqrt(worst)
    }

    /** Detections matched one-to-one to the nearest predicted map X with agreeing strokes and size. */
    private fun associate(dets: List<XDetection>, x: DoubleArray, gate: Double): LinkedHashMap<Int, Int> {
        val p = PoseProjector(cam, x)
        val px = DoubleArray(m)
        val py = DoubleArray(m)
        val inView = BooleanArray(m)
        var any = false
        for (i in 0 until m) {
            val q = p.project(mx[i], my[i], tmp)
            px[i] = q[0]; py[i] = q[1]
            inView[i] = q[0].isFinite() && q[0] > -gate && q[0] < width + gate && q[1] > -gate && q[1] < height + gate
            any = any || inView[i]
        }
        val matches = LinkedHashMap<Int, Int>()
        if (!any || dets.isEmpty()) return matches
        data class Cand(val d: Double, val det: Int, val mark: Int)
        val cands = ArrayList<Cand>()
        for ((i, det) in dets.withIndex()) {
            var best = -1
            var bestD = Double.POSITIVE_INFINITY
            for (j in 0 until m) {
                if (!inView[j]) continue
                val d = hypot(px[j] - det.x, py[j] - det.y)
                if (d < bestD) { bestD = d; best = j }
            }
            if (best >= 0 && bestD < gate) cands += Cand(bestD, i, best)
        }
        cands.sortWith(compareBy<Cand>({ it.d }, { it.det }, { it.mark }))
        val used = HashSet<Int>()
        for (c in cands) {
            if (c.det in matches || c.mark in used) continue
            val det = dets[c.det]
            if (dirDev(det.dir1, det.dir2, predDir(p, c.mark, 0), predDir(p, c.mark, 1)) > dirTol) continue
            val ps = msize[c.mark] * markScale(p, c.mark)
            if (abs(ln(det.size / ps)) > sizeTol) continue
            matches[c.det] = c.mark
            used += c.mark
        }
        return matches
    }

    private class Refit(val x: DoubleArray, val matches: Map<Int, Int>, val rmse: Double)

    private fun imagePoints(dets: List<XDetection>, idx: Collection<Int>): DoubleArray {
        val out = DoubleArray(2 * idx.size)
        for ((k, i) in idx.withIndex()) { out[2 * k] = dets[i].x; out[2 * k + 1] = dets[i].y }
        return out
    }

    /** Associate, solve, re-associate with a tighter gate, solve again; then drop matches that do not
     *  fit on their own. With fewer than four matches, the prior is weighted [wFew] instead. */
    private fun refit(dets: List<XDetection>, x0: DoubleArray, prior: DoubleArray?, wPrior: Double, rounds: Int = 2, wFew: Double? = null): Refit? {
        val need = if (prior != null) 1 else 3
        var x = x0
        var matches: Map<Int, Int> = emptyMap()
        var rmse = Double.POSITIVE_INFINITY
        for (r in 0 until rounds) {
            val s = pxPerMm(x)
            val gate = min(max((if (r == 0) 0.3 else 0.15) * spacing * s, 6 + 3 * residualMm * s), 250.0)
            matches = associate(dets, x, gate)
            if (matches.size < need) return null
            val w = if (wFew == null || matches.size >= 4) wPrior else wFew
            val sol = solve(x, imagePoints(dets, matches.keys), matches.values.toIntArray(), prior, w)
            x = sol.x
            rmse = sol.rmse
        }
        val p = PoseProjector(cam, x)
        val tol = tolerancePx(x)
        val keep = matches.filter { (i, mk) ->
            val q = p.projectFinite(mx[mk], my[mk], tmp)
            hypot(q[0] - dets[i].x, q[1] - dets[i].y) <= tol
        }
        if (keep.size != matches.size) {
            matches = keep
            if (matches.size < need) return null
            val w = if (wFew == null || matches.size >= 4) wPrior else wFew
            val sol = solve(x, imagePoints(dets, matches.keys), matches.values.toIntArray(), prior, w)
            x = sol.x
            rmse = sol.rmse
        }
        return Refit(x, matches, rmse)
    }

    private fun tolerancePx(x: DoubleArray): Double = 2.0 + residualMm * pxPerMm(x)

    // ---------------------------------------------------------------- cold start

    /** Starting points for refining a hypothesis: a camera facing the canvas, plus tilt-aware
     *  solutions from the planar (4+ points) or three-point solver. */
    private fun initialPoses(s: Double, alpha0: Double, pix: Double, piy: Double, a: Int, img: DoubleArray, map: DoubleArray): List<DoubleArray> {
        val alpha = mod(alpha0 + PI, 2 * PI) - PI
        val ca = cos(alpha)
        val sa = sin(alpha)
        val dx = cam.cx - pix
        val dy = cam.cy - piy
        val cx = s * (ca * dx - sa * dy) + mx[a]
        val cy = s * (sa * dx + ca * dy) + my[a]
        val cz = -s * f
        // R = Rz(alpha)^T; t = -R C
        val facing = doubleArrayOf(0.0, 0.0, -alpha, -(ca * cx + sa * cy), -(-sa * cx + ca * cy), -cz)
        val inits = arrayListOf(facing)
        for (x in pnp.solve(map, img, cam)) if (x.all { it.isFinite() } && cameraHeight(x) > 0) inits += x
        return inits
    }

    private class Hyp(val score: Int, val s: Double, val alpha: Double, val i: Int, val a: Int, val detIx: IntArray, val mapIx: IntArray)
    private class Candidate(val n: Int, val rmse: Double, val x: DoubleArray, val matches: Map<Int, Int>)

    private fun lowerBound(v: DoubleArray, key: Double): Int {
        var lo = 0
        var hi = v.size
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (v[mid] < key) lo = mid + 1 else hi = mid
        }
        return lo
    }

    /** Cold start gives up after this long (ms) and answers from what it has tried; a camera frame must not stall. */
    var coldBudgetMs = Long.MAX_VALUE

    fun coldStart(dets: List<XDetection>, maxHypDets: Int = 8, topK: Int = 6, maxScoreDets: Int = 16): PoseEstimate? {
        val deadline = if (coldBudgetMs == Long.MAX_VALUE) Long.MAX_VALUE else System.nanoTime() + coldBudgetMs * 1_000_000
        val n = dets.size
        if (n < 3 || m < 3) return null
        val mmPerPx = xSizeMm / median(DoubleArray(n) { dets[it].size })
        val smin = 0.55 * mmPerPx
        val smax = 1.8 * mmPerPx
        val byStrength = dets.indices.sortedByDescending { dets[it].strength }
        val hypIdx = byStrength.take(maxHypDets)
        val scoreIdx = byStrength.take(maxScoreDets).sorted() // hypotheses are scored on the strongest detections; refits use all
        val hyps = ArrayList<Hyp>()
        val dist = DoubleArray(1)
        val okDet = IntArray(n)
        val okMap = IntArray(n)
        for (ii in hypIdx.indices) for (jj in ii + 1 until hypIdx.size) {
            if (System.nanoTime() > deadline) break
            val i = hypIdx[ii]
            val j = hypIdx[jj]
            val dvx = dets[j].x - dets[i].x
            val dvy = dets[j].y - dets[i].y
            val len = hypot(dvx, dvy)
            if (len < 5) continue
            val lo = lowerBound(pairLen, smin * len)
            val hi = lowerBound(pairLen, smax * len)
            if (hi <= lo) continue
            val angDv = atan2(dvy, dvx)
            val di = dets[i]
            val dj = dets[j]
            for (h in lo until hi) {
                val a = pairA[h]
                val b = pairB[h]
                val s = pairLen[h] / len
                val alpha = pairAng[h] - angDv
                // the pair itself must fit: the two detections look like their map X's (cheap; prunes most candidates)
                if (dirDev(mod(di.dir1 + alpha, PI), mod(di.dir2 + alpha, PI), md1[a], md2[a]) >= 1.5 * dirTol) continue
                if (dirDev(mod(dj.dir1 + alpha, PI), mod(dj.dir2 + alpha, PI), md1[b], md2[b]) >= 1.5 * dirTol) continue
                if (abs(ln(di.size * s / msize[a])) >= ln(2.0) || abs(ln(dj.size * s / msize[b])) >= ln(2.0)) continue
                val c = cos(alpha)
                val sn = sin(alpha)
                var score = 0
                for (k in scoreIdx) {
                    val rx = dets[k].x - dets[i].x
                    val ry = dets[k].y - dets[i].y
                    val X = s * (c * rx - sn * ry) + mx[a]
                    val Y = s * (sn * rx + c * ry) + my[a]
                    val idx = grid.nearestWithin(X, Y, dist)
                    if (idx < 0 || dist[0] >= tolMm) continue
                    // coarse checks: a similarity transform ignores perspective, so be lenient here
                    val d = dets[k]
                    if (dirDev(mod(d.dir1 + alpha, PI), mod(d.dir2 + alpha, PI), md1[idx], md2[idx]) >= 1.5 * dirTol) continue
                    if (abs(ln(d.size * s / msize[idx])) >= ln(2.0)) continue
                    okDet[score] = k
                    okMap[score] = idx
                    score++
                }
                if (score >= 3) hyps += Hyp(score, s, alpha, i, a, okDet.copyOf(score), okMap.copyOf(score))
            }
        }
        if (hyps.isEmpty()) return null

        // refine the strongest distinct hypotheses with the full camera model
        val sorted = hyps.sortedByDescending { it.score }
        val results = ArrayList<Candidate>()
        val seen = HashSet<List<Int>>()
        for (hyp in sorted) {
            if (System.nanoTime() > deadline) break
            val key = ArrayList<Int>(2 * hyp.detIx.size).apply { for (k in hyp.detIx.indices) { add(hyp.detIx[k]); add(hyp.mapIx[k]) } }
            if (!seen.add(key)) continue
            if (seen.size > topK * 4) break
            val img = imagePoints(dets, hyp.detIx.toList())
            val map = DoubleArray(2 * hyp.mapIx.size).also { for ((k, mk) in hyp.mapIx.withIndex()) { it[2 * k] = mx[mk]; it[2 * k + 1] = my[mk] } }
            var best: Candidate? = null
            for (x0 in initialPoses(hyp.s, hyp.alpha, dets[hyp.i].x, dets[hyp.i].y, hyp.a, img, map)) {
                val sol = solve(x0, img, hyp.mapIx)
                val out = refit(dets, sol.x, null, 0.0) ?: continue
                if (cameraHeight(out.x) <= 0 || out.rmse > tolerancePx(out.x)) continue
                if (out.matches.size < minCoverage * predictedInView(out.x)) continue // predicts X's where there are none
                val b = best
                if (b == null || out.matches.size > b.n || (out.matches.size == b.n && out.rmse < b.rmse)) {
                    best = Candidate(out.matches.size, out.rmse, out.x, out.matches)
                }
            }
            best?.let { results += it }
            if (results.size >= topK) break
        }
        if (results.isEmpty()) return null
        results.sortWith(compareBy<Candidate>({ -it.n }, { it.rmse }))
        val top = results[0]
        val topIds = top.matches.values.toSet()
        for (other in results.drop(1)) {
            if (other.n == top.n && other.matches.values.toSet() != topIds && other.rmse < 1.5 * top.rmse + 0.5) {
                return null // two different answers fit equally well
            }
        }
        val need = minCold + if (n - top.n > 10) 1 else 0 // cluttered scene: want one more X
        if (top.n < need || !wellSpread(imagePoints(dets, top.matches.keys))) return null
        val sol = solve(top.x, imagePoints(dets, top.matches.keys), top.matches.values.toIntArray())
        val unc = uncertainty(sol.x)
        if (unc > maxColdUncertaintyMm) return null // the X's in view don't pin the camera down well enough
        return PoseEstimate(sol.x, homography(sol.x), top.matches, sol.rmse, TrackMode.COLD, unc)
    }

    /** Matched X's must span the view in both directions, or the tilt is poorly pinned down. */
    private fun wellSpread(pts: DoubleArray): Boolean {
        val n = pts.size / 2
        if (n < 3) return false
        var sx = 0.0; var sy = 0.0
        for (k in 0 until n) { sx += pts[2 * k]; sy += pts[2 * k + 1] }
        sx /= n; sy /= n
        var a = 0.0; var b = 0.0; var c = 0.0
        for (k in 0 until n) {
            val dx = pts[2 * k] - sx
            val dy = pts[2 * k + 1] - sy
            a += dx * dx; b += dx * dy; c += dy * dy
        }
        a /= n - 1; b /= n - 1; c /= n - 1
        val smallest = (a + c) / 2 - sqrt(((a - c) / 2) * ((a - c) / 2) + b * b)
        return sqrt(max(smallest, 0.0)) >= 0.04 * hypot(width.toDouble(), height.toDouble())
    }

    // ---------------------------------------------------------------- tracking

    /** Update from [prior] (the predicted pose), gently anchored to [anchor] (the last measured pose),
     *  firmly so when fewer than four X's match. Returns the prior unchanged (HOLD) when no X matches. */
    fun track(dets: List<XDetection>, prior: PoseEstimate, anchor: PoseEstimate? = null, wPrior: Double = 0.02, wFew: Double = 0.3): PoseEstimate? {
        val hold = PoseEstimate(prior.x, prior.H, emptyMap(), prior.rmse, TrackMode.HOLD)
        val out = refit(dets, prior.x.copyOf(), (anchor ?: prior).x, wPrior, wFew = wFew) ?: return hold
        if (out.rmse > tolerancePx(out.x)) return null
        if (out.matches.size == 1) { // one X can only nudge the pose if it is clearly the X we expected
            val (i, mk) = out.matches.entries.first()
            val q = PoseProjector(cam, prior.x).project(mx[mk], my[mk], DoubleArray(2))
            if (!(hypot(dets[i].x - q[0], dets[i].y - q[1]) <= 0.2 * spacing * pxPerMm(prior.x))) return hold
        }
        return PoseEstimate(out.x, homography(out.x), out.matches, out.rmse, TrackMode.TRACK, uncertainty(out.x))
    }
}

/**
 * Frame-to-frame tracking with recovery, as the app runs it:
 * - predicts the next pose from the current motion (damped constant velocity);
 * - holds the last pose briefly when no X can be matched, trying a fresh lock-on every frame
 *   meanwhile, and gives up after [maxHold] frames;
 * - every [verifyEvery] frames, checks the tracked identities against a fresh lock-on and
 *   switches if they conflict.
 */
class Tracker(val locator: Locator, private val maxHold: Int = 10, private val verifyEvery: Int = 10) {
    var state: PoseEstimate? = null
        private set
    private var prev: PoseEstimate? = null
    private var hold = 0
    private var frame = 0

    fun reset() {
        state = null; prev = null; hold = 0; frame = 0
    }

    private fun predicted(): PoseEstimate {
        val s = state!!
        val p = prev
        if (p == null || s.mode == TrackMode.HOLD || p.mode == TrackMode.HOLD) return s
        val x = DoubleArray(6) { if (it < 3) s.x[it] else s.x[it] + 0.5 * (s.x[it] - p.x[it]) } // rotation not extrapolated
        return PoseEstimate(x, locator.homography(x), s.matches, s.rmse, s.mode)
    }

    fun update(dets: List<XDetection>): PoseEstimate? {
        frame++
        var est: PoseEstimate? = null
        if (state != null) {
            est = locator.track(dets, predicted(), anchor = state)
            if (est != null && est.mode == TrackMode.HOLD) {
                hold++
                if (hold > maxHold) est = null
            } else if (est != null) {
                hold = 0
            }
        }
        val recheck = est != null && est.mode == TrackMode.TRACK && frame % verifyEvery == 0
        if ((est == null || est.mode == TrackMode.HOLD || recheck) && dets.size >= 4) {
            val cold = locator.coldStart(dets)
            if (cold != null && cold.matches.size >= 4) {
                val conflict = est != null && cold.matches.any { (i, mk) -> (est!!.matches[i] ?: mk) != mk }
                if (est == null || est.mode == TrackMode.HOLD || conflict) {
                    est = cold
                    hold = 0
                }
            }
        }
        prev = if (est != null) state else null
        state = est
        return est
    }
}
