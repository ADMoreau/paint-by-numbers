package io.github.admoreau.paintbynumbers.align

import io.github.admoreau.paintbynumbers.tracking.M3
import io.github.admoreau.paintbynumbers.tracking.solveLinear
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.stream.Collectors
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Finds the target image in a camera frame of its chalk or pencil sketch: "Snap to drawing".
 * Ported from `prototypes/sketch_align/salign/align.py` (without the bending), where the method is
 * worked out and tested in simulation; see that folder's README.
 *
 * All maps go *from* work pixels (the camera frame) *to* model pixels (the target, downscaled to
 * [EdgeModel.long]). Homographies are row-major 3x3 DoubleArrays.
 */
object SketchAligner {
    enum class Kind(val n: Int) { SIMILARITY(4), AFFINE(6), HOMOGRAPHY(8) }

    class Fit(val G: DoubleArray, val kind: Kind, val matched: Double, val rms: Double, var score: Double = 1.0)

    class Candidate(val score: Double, val G: DoubleArray, val scale: Double, val angle: Double)

    class Result(
        val G: DoubleArray, // work px -> model px (perspective)
        val score: Double, // truncated chamfer, 0 (perfect) .. 1
        val secondScore: Double, // best clearly different placement's score: close to [score] = ambiguous
        val matched: Double, // share of line points with a matching image edge
        val linePoints: Int,
        val millis: Map<String, Long>,
    )

    // ------------------------------------------------------------ geometry helpers

    private fun apply(G: DoubleArray, x: Double, y: Double, out: DoubleArray): DoubleArray = M3.apply(G, x, y, out)

    /** The direction (mod pi) of a line normal at work point (x, y) once mapped through [G]: normals map by J^-T. */
    private fun normalAngle(G: DoubleArray, x: Double, y: Double, nx: Double, ny: Double): Double {
        val s = G[6] * x + G[7] * y + G[8]
        val px = (G[0] * x + G[1] * y + G[2]) / s
        val py = (G[3] * x + G[4] * y + G[5]) / s
        val j00 = (G[0] - px * G[6]) / s
        val j01 = (G[1] - px * G[7]) / s
        val j10 = (G[3] - py * G[6]) / s
        val j11 = (G[4] - py * G[7]) / s
        val sg = sign(j00 * j11 - j01 * j10)
        return angleMod(atan2((-j01 * nx + j00 * ny) * sg, (j11 * nx - j10 * ny) * sg))
    }

    /** Model pixels per work pixel at (x, y) under [G]. */
    private fun localScale(G: DoubleArray, x: Double, y: Double): Double {
        val s = G[6] * x + G[7] * y + G[8]
        val px = (G[0] * x + G[1] * y + G[2]) / s
        val py = (G[3] * x + G[4] * y + G[5]) / s
        val j00 = (G[0] - px * G[6]) / s
        val j01 = (G[1] - px * G[7]) / s
        val j10 = (G[3] - py * G[6]) / s
        val j11 = (G[4] - py * G[7]) / s
        return sqrt(abs(j00 * j11 - j01 * j10))
    }

    private fun inv(a: DoubleArray): DoubleArray = M3.inv(a) ?: throw IllegalStateException("singular placement")

    private fun toParams(kind: Kind, g: DoubleArray): DoubleArray {
        val G = DoubleArray(9) { g[it] / g[8] }
        return when (kind) {
            Kind.SIMILARITY -> doubleArrayOf((G[0] + G[4]) / 2, (G[3] - G[1]) / 2, G[2], G[5])
            Kind.AFFINE -> G.copyOfRange(0, 6)
            Kind.HOMOGRAPHY -> G.copyOfRange(0, 8)
        }
    }

    private fun toMatrix(kind: Kind, t: DoubleArray): DoubleArray = when (kind) {
        Kind.SIMILARITY -> doubleArrayOf(t[0], -t[1], t[2], t[1], t[0], t[3], 0.0, 0.0, 1.0)
        Kind.AFFINE -> doubleArrayOf(t[0], t[1], t[2], t[3], t[4], t[5], 0.0, 0.0, 1.0)
        Kind.HOMOGRAPHY -> DoubleArray(9) { if (it < 8) t[it] else 1.0 }
    }

    /** Mapped normalised point (qx, qy) into [xy], and d(x)/dt, d(y)/dt into [jx], [jy]. */
    private fun jac(kind: Kind, t: DoubleArray, qx: Double, qy: Double, xy: DoubleArray, jx: DoubleArray, jy: DoubleArray) {
        when (kind) {
            Kind.SIMILARITY -> {
                xy[0] = t[0] * qx - t[1] * qy + t[2]; xy[1] = t[1] * qx + t[0] * qy + t[3]
                jx[0] = qx; jx[1] = -qy; jx[2] = 1.0; jx[3] = 0.0
                jy[0] = qy; jy[1] = qx; jy[2] = 0.0; jy[3] = 1.0
            }
            Kind.AFFINE -> {
                xy[0] = t[0] * qx + t[1] * qy + t[2]; xy[1] = t[3] * qx + t[4] * qy + t[5]
                jx[0] = qx; jx[1] = qy; jx[2] = 1.0; jx[3] = 0.0; jx[4] = 0.0; jx[5] = 0.0
                jy[0] = 0.0; jy[1] = 0.0; jy[2] = 0.0; jy[3] = qx; jy[4] = qy; jy[5] = 1.0
            }
            Kind.HOMOGRAPHY -> {
                val u = t[0] * qx + t[1] * qy + t[2]
                val v = t[3] * qx + t[4] * qy + t[5]
                val s = t[6] * qx + t[7] * qy + 1
                val x = u / s
                val y = v / s
                xy[0] = x; xy[1] = y
                jx[0] = qx / s; jx[1] = qy / s; jx[2] = 1 / s; jx[3] = 0.0; jx[4] = 0.0; jx[5] = 0.0; jx[6] = -x * qx / s; jx[7] = -x * qy / s
                jy[0] = 0.0; jy[1] = 0.0; jy[2] = 0.0; jy[3] = qx / s; jy[4] = qy / s; jy[5] = 1 / s; jy[6] = -y * qx / s; jy[7] = -y * qy / s
            }
        }
    }

    // ------------------------------------------------------------ scoring and refinement

    /** Mean truncated squared distance (work px, normalised to 0..1) from line points to matching image edges. */
    fun chamfer(model: EdgeModel, pts: LinePoints, G: DoubleArray, tauWork: Double): Double {
        val p = DoubleArray(2)
        val e = DoubleArray(4)
        var sum = 0.0
        for (i in 0 until pts.size) {
            val x = pts.x[i].toDouble()
            val y = pts.y[i].toDouble()
            apply(G, x, y, p)
            val d = model.nearest(p[0], p[1], normalAngle(G, x, y, pts.nx[i].toDouble(), pts.ny[i].toDouble()), e)
            val dw = min(d / localScale(G, x, y), tauWork)
            sum += dw * dw
        }
        return sum / max(1, pts.size) / (tauWork * tauWork)
    }

    /**
     * Oriented point-to-line ICP: each line point is matched to the nearest image edge running the
     * same way, and the placement is solved by robust (Tukey) least squares while the match radius
     * [taus] (model px) shrinks.
     */
    fun refine(model: EdgeModel, pts: LinePoints, G0: DoubleArray, kind: Kind, taus: DoubleArray? = null, iters: Int = 4): Fit {
        val tauList = taus ?: doubleArrayOf(0.06, 0.035, 0.02, 0.012, 0.008).map { it * model.long }.toDoubleArray()
        val n = pts.size
        // normalise both sides so the parameters are well scaled
        var sx = 0.0
        var sy = 0.0
        for (i in 0 until n) { sx += pts.x[i]; sy += pts.y[i] }
        val cx = sx / n
        val cy = sy / n
        val allMean = (sx + sy) / (2 * n)
        var v = 0.0
        for (i in 0 until n) { val a = pts.x[i] - allMean; val b = pts.y[i] - allMean; v += a * a + b * b }
        val s = sqrt(v / (2 * n)) + 1e-9
        val Tq = doubleArrayOf(1 / s, 0.0, -cx / s, 0.0, 1 / s, -cy / s, 0.0, 0.0, 1.0)
        val m = max(model.w, model.h) / 2.0
        val Tp = doubleArrayOf(1 / m, 0.0, -model.w / 2.0 / m, 0.0, 1 / m, -model.h / 2.0 / m, 0.0, 0.0, 1.0)
        val TpInv = inv(Tp)
        val qnx = DoubleArray(n) { (pts.x[it] - cx) / s }
        val qny = DoubleArray(n) { (pts.y[it] - cy) / s }
        var Gn = M3.mul(M3.mul(Tp, G0), inv(Tq))
        if (kind == Kind.SIMILARITY) Gn = toMatrix(kind, toParams(kind, Gn))
        else if (kind == Kind.AFFINE) Gn = DoubleArray(9) { if (it < 6) Gn[it] / Gn[8] else if (it == 8) 1.0 else 0.0 }
        var t = toParams(kind, Gn)
        val k = kind.n
        val w = DoubleArray(n)
        val p = DoubleArray(2)
        val e = DoubleArray(4)
        val xy = DoubleArray(2)
        val jx = DoubleArray(k)
        val jy = DoubleArray(k)
        val J = DoubleArray(k)
        for (tau in tauList) {
            for (it in 0 until iters) {
                val G = M3.mul(M3.mul(TpInv, toMatrix(kind, t)), Tq)
                val A = DoubleArray(k * k)
                val g = DoubleArray(k)
                var wSum = 0.0
                for (i in 0 until n) {
                    val x = pts.x[i].toDouble()
                    val y = pts.y[i].toDouble()
                    apply(G, x, y, p)
                    val d = model.nearest(p[0], p[1], normalAngle(G, x, y, pts.nx[i].toDouble(), pts.ny[i].toDouble()), e)
                    if (!(d < tau)) { w[i] = 0.0; continue }
                    val enx = (e[0] - model.w / 2.0) / m
                    val eny = (e[1] - model.h / 2.0) / m
                    jac(kind, t, qnx[i], qny[i], xy, jx, jy)
                    val r = e[2] * (xy[0] - enx) + e[3] * (xy[1] - eny)
                    val z = r * m / tau
                    if (abs(z) >= 1) { w[i] = 0.0; continue }
                    val wi = (1 - z * z) * (1 - z * z)
                    w[i] = wi
                    wSum += wi
                    for (a in 0 until k) J[a] = e[2] * jx[a] + e[3] * jy[a]
                    for (a in 0 until k) {
                        g[a] += wi * J[a] * r
                        for (b in 0 until k) A[a * k + b] += wi * J[a] * J[b]
                    }
                }
                if (wSum < 10) break
                var tr = 0.0
                for (a in 0 until k) tr += A[a * k + a]
                for (a in 0 until k) A[a * k + a] += 1e-9 * tr
                val step = solveLinear(A, g, k) ?: break
                t = DoubleArray(k) { t[it] - step[it] }
            }
        }
        val G = M3.mul(M3.mul(TpInv, toMatrix(kind, t)), Tq).let { g -> DoubleArray(9) { g[it] / g[8] } }
        var sel = 0
        var rr = 0.0
        var nr = 0
        for (i in 0 until n) {
            if (w[i] <= 0) continue
            sel++
            val x = pts.x[i].toDouble()
            val y = pts.y[i].toDouble()
            apply(G, x, y, p)
            val d = model.nearest(p[0], p[1], normalAngle(G, x, y, pts.nx[i].toDouble(), pts.ny[i].toDouble()), e)
            if (!d.isFinite()) continue
            val r = abs((p[0] - e[0]) * e[2] + (p[1] - e[1]) * e[3]) / localScale(G, x, y)
            rr += r * r
            nr++
        }
        return Fit(G, kind, sel.toDouble() / max(1, n), if (nr > 0) sqrt(rr / nr) else 0.0)
    }

    // ------------------------------------------------------------ global search

    /** Share of a 9 x 9 grid over the model that lands inside the work image through [A] (model -> work). */
    private fun overlap(A: DoubleArray, mw: Int, mh: Int, W: Int, H: Int): Double {
        val p = DoubleArray(2)
        var inside = 0
        for (j in 0..8) for (i in 0..8) {
            apply(A, mw * i / 8.0, mh * j / 8.0, p)
            if (p[0] >= 0 && p[1] >= 0 && p[0] <= W && p[1] <= H) inside++
        }
        return inside / 81.0
    }

    /**
     * Brute force over [scales] (the image's long side in work px) x [angles] (model -> work
     * rotation), with translation by correlation on a small copy of the sketch: a blurred copy of the
     * image's edges slides over the sketch lines and the best few positions per pose are kept.
     */
    fun search(model: EdgeModel, lines: LinePoints, W: Int, H: Int, scales: DoubleArray, angles: DoubleArray, workLong: Int = 256, pad: Double = 0.6, perPose: Int = 3): List<Candidate> {
        val f = workLong.toDouble() / max(W, H)
        val sw = Math.rint(W * f).toInt()
        val sh = Math.rint(H * f).toInt()
        val sk = FloatArray(sw * sh)
        for (i in 0 until lines.size) {
            val xi = Math.rint(lines.x[i] * f).toInt()
            val yi = Math.rint(lines.y[i] * f).toInt()
            if (xi in 0 until sw && yi in 0 until sh) sk[yi * sw + xi] = min(sk[yi * sw + xi] + 1f, 2f)
        }
        val skM = Mat(sh, sw, CvType.CV_32F).also { it.put(0, 0, sk) }
        val mw = model.w
        val mh = model.h
        val ones = Mat(mh, mw, CvType.CV_8U, Scalar(1.0))
        val kernel = Mat.ones(5, 5, CvType.CV_8U)
        val sigma = 1.2 // small px

        /** One scale and turn: the best few positions. Runs in parallel with the others (inputs are only read). */
        fun pose(s: Double, a: Double): List<Candidate> {
            val k = s * f / model.long // small px per model px
            val c = k * cos(a)
            val sn = k * sin(a)
            val bx = doubleArrayOf(0.0, c * mw, c * mw - sn * mh, -sn * mh)
            val by = doubleArrayOf(0.0, sn * mw, sn * mw + c * mh, c * mh)
            val lox = bx.min()
            val loy = by.min()
            val tw = ceil(bx.max() - lox).toInt() + 1
            val th = ceil(by.max() - loy).toInt() + 1
            if (tw < 4 || th < 4 || tw > 3 * sw || th > 3 * sh) return emptyList()
            val M = Mat(2, 3, CvType.CV_64F).also { it.put(0, 0, c, -sn, -lox, sn, c, -loy) }
            val dtw = Mat()
            val foot = Mat()
            val notFoot = Mat()
            val tM = Mat()
            val big = Mat()
            val R = Mat()
            val Rd = Mat()
            try {
                val size = Size(tw.toDouble(), th.toDouble())
                Imgproc.warpAffine(model.dtAll, dtw, M, size, Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(1e4))
                Imgproc.warpAffine(ones, foot, M, size, Imgproc.INTER_NEAREST)
                if (Core.countNonZero(foot) == 0) return emptyList()
                // template: how close each pixel is to an image edge, exp(-d^2 / 2 sigma^2), zero-mean over the image's footprint
                Core.multiply(dtw, dtw, tM, -k * k / (2 * sigma * sigma))
                Core.exp(tM, tM)
                Core.subtract(tM, Core.mean(tM, foot), tM)
                Core.compare(foot, Scalar(0.0), notFoot, Core.CMP_EQ)
                tM.setTo(Scalar(0.0), notFoot)
                // the image may hang off the frame, but by less than [pad] of its size (under half in view is rejected below)
                val px = ceil(pad * tw).toInt()
                val py = ceil(pad * th).toInt()
                Core.copyMakeBorder(skM, big, py, py, px, px, Core.BORDER_CONSTANT, Scalar(0.0))
                Imgproc.matchTemplate(big, tM, R, Imgproc.TM_CCORR) // R[y, x]: template's top-left at big (x, y)
                Imgproc.dilate(R, Rd, kernel)
                val rw = R.cols()
                val rv = R.floats()
                val rdv = Rd.floats()
                // the best few local maxima
                val bestI = IntArray(perPose) { -1 }
                val bestV = FloatArray(perPose) { Float.NEGATIVE_INFINITY }
                for (i in rv.indices) {
                    val v = rv[i]
                    if (v <= 0 || v < rdv[i] || v <= bestV[perPose - 1]) continue
                    var j = perPose - 1
                    while (j > 0 && bestV[j - 1] < v) { bestV[j] = bestV[j - 1]; bestI[j] = bestI[j - 1]; j-- }
                    bestV[j] = v; bestI[j] = i
                }
                val out = ArrayList<Candidate>()
                for (j in 0 until perPose) {
                    val i = bestI[j]
                    if (i < 0) continue
                    val ox = i % rw - px
                    val oy = i / rw - py
                    // model -> small, then small -> work
                    val A = doubleArrayOf(c / f, -sn / f, (-lox + ox) / f, sn / f, c / f, (-loy + oy) / f, 0.0, 0.0, 1.0)
                    if (overlap(A, mw, mh, W, H) < 0.5) continue
                    out += Candidate(bestV[j].toDouble(), inv(A), s, a)
                }
                return out
            } finally {
                for (mm in listOf(M, dtw, foot, notFoot, tM, big, R, Rd)) mm.release()
            }
        }

        val poses = scales.flatMap { s -> angles.map { a -> s to a } }
        val out = poses.parallelStream().map { pose(it.first, it.second) }.collect(Collectors.toList()).flatten().toMutableList()
        for (mm in listOf(skM, ones, kernel)) mm.release()
        out.sortByDescending { it.score }
        return out
    }

    private fun centre(G: DoubleArray, model: EdgeModel, out: DoubleArray = DoubleArray(2)) = apply(inv(G), model.w / 2.0, model.h / 2.0, out)

    /** The best [n] candidates that are not near-duplicates of a better one. */
    fun distinct(cands: List<Candidate>, model: EdgeModel, n: Int, minShift: Double = 0.08): List<Candidate> {
        val out = ArrayList<Candidate>()
        val centres = ArrayList<DoubleArray>()
        for (c in cands) {
            val ctr = centre(c.G, model)
            var dup = false
            for ((o, oc) in out.zip(centres)) {
                val dAng = abs(((c.angle - o.angle + PI) % (2 * PI) + 2 * PI) % (2 * PI) - PI)
                if (hypot(ctr[0] - oc[0], ctr[1] - oc[1]) < minShift * max(c.scale, o.scale) && abs(ln(c.scale / o.scale)) < 0.1 && dAng < Math.toRadians(8.0)) { dup = true; break }
            }
            if (!dup) {
                out += c; centres += ctr
                if (out.size == n) break
            }
        }
        return out
    }

    private fun range(from: Double, to: Double, step: Double): DoubleArray {
        val out = ArrayList<Double>()
        var v = from
        while (v <= to + 1e-9) { out += v; v += step }
        return out.toDoubleArray()
    }

    /** Search near a rough placement [G0]: scales and turns around it, centre within [shift] of the image size. */
    fun snapCandidates(model: EdgeModel, lines: LinePoints, W: Int, H: Int, G0: DoubleArray, scaleRange: Double = 0.25, angleRange: Double = 12.0, shift: Double = 0.25): List<Candidate> {
        val A0 = inv(G0)
        val s0 = hypot(A0[0], A0[3]) / A0[8] * model.long
        val a0 = atan2(A0[3], A0[0])
        val scales = range(-scaleRange, scaleRange, ln(1.06)).map { s0 * exp(it) }.toDoubleArray()
        val angles = range(-angleRange, angleRange, 3.0).map { a0 + Math.toRadians(it) }.toDoubleArray()
        val c0 = apply(A0, model.w / 2.0, model.h / 2.0, DoubleArray(2))
        val cc = DoubleArray(2)
        return search(model, lines, W, H, scales, angles).filter { c ->
            centre(c.G, model, cc)
            hypot(cc[0] - c0[0], cc[1] - c0[1]) < shift * s0
        }
    }

    /**
     * Finds the target in the frame's [lines] ([W] x [H] work px). [start]: where the image is drawn
     * now (work px -> model px), searched around first; [upAngle]: the model -> work rotation that
     * shows the image upright on screen, around which the whole frame is also searched (+-15 deg).
     * Returns a perspective placement (the frame is a raw camera image), or null if nothing fits.
     */
    fun align(model: EdgeModel, lines: LinePoints, W: Int, H: Int, start: DoubleArray?, upAngle: Double, scaleRange: Pair<Double, Double> = 0.3 to 1.6, nStarts: Int = 6, maxPoints: Int = 8000): Result? {
        if (lines.size < 50) return null
        val millis = LinkedHashMap<String, Long>()
        var t0 = System.nanoTime()
        fun lap(name: String) { val t = System.nanoTime(); millis[name] = (t - t0) / 1_000_000; t0 = t }
        val pts = if (lines.size <= maxPoints) lines else {
            val idx = (0 until lines.size).shuffled(java.util.Random(0)).take(maxPoints).sorted().toIntArray()
            lines.subset(idx)
        }
        val cands = ArrayList<Candidate>()
        if (start != null) runCatching { cands += snapCandidates(model, lines, W, H, start) }
        val lo = scaleRange.first * max(W, H)
        val hi = scaleRange.second * max(W, H)
        val scales = range(ln(lo), ln(hi) - 1e-9, ln(1.07)).map { exp(it) }.toDoubleArray()
        val angles = range(-15.0, 15.0, 3.0).map { upAngle + Math.toRadians(it) }.toDoubleArray()
        cands += search(model, lines, W, H, scales, angles)
        cands.sortByDescending { it.score }
        val starts = distinct(cands, model, nStarts)
        lap("search")
        val L = model.long.toDouble()
        val tau = 0.01 * max(W, H)
        val fits = ArrayList<Fit>()
        fits += starts.parallelStream().map { c -> runCatching { refine(model, pts, c.G, Kind.SIMILARITY) }.getOrNull() }.collect(Collectors.toList()).filterNotNull()
        if (start != null) runCatching {
            fits += refine(model, pts, start, Kind.SIMILARITY, doubleArrayOf(0.12, 0.07, 0.04, 0.025, 0.015, 0.01, 0.008).map { it * L }.toDoubleArray())
        }
        if (fits.isEmpty()) return null
        fits.parallelStream().forEach { f -> f.score = chamfer(model, pts, f.G, tau) }
        fits.sortBy { it.score }
        val best = fits[0]
        // the best placement that is clearly different (centre moved by over 5% of the image)
        var second = 1.0
        val cBest = centre(best.G, model)
        val size = L / sqrt(abs(best.G[0] * best.G[4] - best.G[1] * best.G[3]))
        val c = DoubleArray(2)
        for (f in fits.drop(1)) {
            centre(f.G, model, c)
            if (hypot(c[0] - cBest[0], c[1] - cBest[1]) > 0.05 * size) { second = f.score; break }
        }
        lap("refine")
        // a tilted camera can be far from any similarity: through affine to full perspective
        var rigid = refine(model, pts, best.G, Kind.AFFINE, doubleArrayOf(0.06, 0.04, 0.025).map { it * L }.toDoubleArray())
        rigid = refine(model, pts, rigid.G, Kind.HOMOGRAPHY, doubleArrayOf(0.04, 0.025, 0.015, 0.01).map { it * L }.toDoubleArray())
        rigid.score = chamfer(model, pts, rigid.G, tau)
        lap("perspective")
        return Result(rigid.G, rigid.score, second, rigid.matched, lines.size, millis)
    }
}
