package io.github.admoreau.paintbynumbers.tracking

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/** A hand-drawn X found in a frame, in pixel coordinates of the analysed image buffer. */
data class XDetection(
    val x: Double,
    val y: Double,
    val dir1: Double, // stroke angles, radians in [0, pi)
    val dir2: Double,
    val size: Double, // mean stroke length, px
    val strength: Double, // mean contrast in noise units
    val slender: Double = 99.0, // stroke length / stroke width: hand-drawn X's are long thin lines, clutter is blobby
)

/**
 * Finds hand-drawn X marks. Port of prototypes/x_tracking/xtrack/detect.py:
 *
 * 1. Background: a median over a wide window (at quarter size) gives the bare surface;
 *    marks are what differs from it, darker or lighter depending on [polarity].
 * 2. Mark pixels: contrast above the frame's noise level, with hysteresis, as connected blobs.
 * 3. A blob counts as an X only if it is two roughly straight strokes that cross with arms on
 *    both sides. The crossing of lines fitted to the strokes is the mark's position.
 *
 * Not thread-safe: reuses buffers between calls.
 */
class XDetector {
    enum class Polarity { DARK, LIGHT }

    var polarity = Polarity.DARK
    var low = 3.0
    var high = 6.0
    var minExtent = 8
    var maxExtentFrac = 0.6
    var maxPixels = 700
    /** Noise floor in gray levels: camera denoising can make dark or flat frames nearly noiseless. */
    var minSigma = 2.0
    /** A mark must differ from its surroundings by at least this many gray levels somewhere. */
    var minContrast = 12.0

    private val small = Mat()
    private val med = Mat()
    private val med16 = Mat()
    private val bg16 = Mat()
    private val gray16 = Mat()
    private val diff = Mat() // 16-bit fixed point: (gray - background) * SCALE, then smoothed
    private val diffSmall = Mat()
    private val weak8 = Mat()
    private val labels = Mat()
    private val stats = Mat()
    private val centroids = Mat()
    private val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
    private var labelBuf = IntArray(0)
    private var diffBuf = ShortArray(0)

    /** Milliseconds spent in the last [detect]: image filters, buffer copies, blob gathering, stroke tests; and blobs tested. */
    val lastTimings = DoubleArray(5)

    /** A place to look: blobs whose box centre lies within [r] px of ([x], [y]). */
    class Region(val x: Double, val y: Double, val r: Double)

    /**
     * [gray] is an 8-bit single-channel image (the camera's luminance plane). With [regions], only
     * blobs near them get the (costly) stroke test: for tracking, where the X's are expected.
     */
    fun detect(gray: Mat, regions: List<Region>? = null): List<XDetection> {
        val tStart = System.nanoTime()
        val w = gray.cols()
        val h = gray.rows()
        // background: median over a wide window at quarter size (smoothed there too), scaled back up
        Imgproc.resize(gray, small, Size((w / 4).toDouble(), (h / 4).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.medianBlur(small, med, 31) // ~120 px window at full size: wider than any stroke
        // contrast against the background, in 16-bit fixed point (gray levels * SCALE) so the
        // smoothed, interpolated background keeps fractions of a gray level
        med.convertTo(med16, CvType.CV_16S, SCALE.toDouble())
        Imgproc.GaussianBlur(med16, med16, Size(0.0, 0.0), 0.75)
        Imgproc.resize(med16, bg16, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        gray.convertTo(gray16, CvType.CV_16S, SCALE.toDouble())
        Core.subtract(gray16, bg16, diff)
        val sigma = max(noiseSigma(w, h), minSigma)
        Imgproc.GaussianBlur(diff, diff, Size(0.0, 0.0), 1.0)
        val sign = if (polarity == Polarity.DARK) -1.0 else 1.0
        val lowLevel = low * sigma * SCALE
        if (sign > 0) Core.compare(diff, Scalar(lowLevel), weak8, Core.CMP_GT)
        else Core.compare(diff, Scalar(-lowLevel), weak8, Core.CMP_LT)
        Imgproc.morphologyEx(weak8, weak8, Imgproc.MORPH_CLOSE, kernel)
        val n = Imgproc.connectedComponentsWithStats(weak8, labels, stats, centroids, 8, CvType.CV_32S)
        val tFilters = System.nanoTime()

        if (labelBuf.size != w * h) {
            labelBuf = IntArray(w * h)
            diffBuf = ShortArray(w * h)
        }
        labels.get(0, 0, labelBuf)
        diff.get(0, 0, diffBuf)
        val st = IntArray(n * 5)
        stats.get(0, 0, st)
        val maxExtent = maxExtentFrac * min(w, h)
        val strongSnr = max(high, minContrast / sigma)
        val toSnr = sign / (sigma * SCALE)
        val tCopy = System.nanoTime()
        var strokeNs = 0L
        var tested = 0

        val out = ArrayList<XDetection>()
        for (k in 1 until n) {
            val x0 = st[k * 5]
            val y0 = st[k * 5 + 1]
            val bw = st[k * 5 + 2]
            val bh = st[k * 5 + 3]
            val area = st[k * 5 + 4]
            val extent = max(bw, bh)
            if (extent < minExtent || extent > maxExtent || area < 15 || area > 0.65 * bw * bh) continue
            if (regions != null) {
                val cx = x0 + bw / 2.0
                val cy = y0 + bh / 2.0
                if (regions.none { hypot(it.x - cx, it.y - cy) <= it.r }) continue
            }
            val xs = DoubleArray(area)
            val ys = DoubleArray(area)
            val ws = DoubleArray(area)
            var count = 0
            var strong = false
            var sumW = 0.0
            for (yy in y0 until y0 + bh) {
                val row = yy * w
                for (xx in x0 until x0 + bw) {
                    if (labelBuf[row + xx] != k || count >= area) continue
                    val s = diffBuf[row + xx] * toSnr
                    if (s > strongSnr) strong = true
                    xs[count] = (xx - x0).toDouble()
                    ys[count] = (yy - y0).toDouble()
                    ws[count] = max(s, 0.05)
                    sumW += ws[count]
                    count++
                }
            }
            if (!strong || count < 15) continue
            val ts = System.nanoTime()
            tested++
            val det = twoStrokes(xs, ys, ws, count, area, extent)
            strokeNs += System.nanoTime() - ts
            if (det == null) continue
            out += det.copy(x = det.x + x0, y = det.y + y0, strength = sumW / count)
        }
        val tEnd = System.nanoTime()
        lastTimings[0] = (tFilters - tStart) / 1e6
        lastTimings[1] = (tCopy - tFilters) / 1e6
        lastTimings[2] = (tEnd - tCopy - strokeNs) / 1e6
        lastTimings[3] = strokeNs / 1e6
        lastTimings[4] = tested.toDouble()
        return dedupe(out)
    }

    /** Noise level of the contrast image in gray levels, from the median absolute deviation. */
    private fun noiseSigma(w: Int, h: Int): Double {
        Imgproc.resize(diff, diffSmall, Size((w / 8).toDouble(), (h / 8).toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
        val v = ShortArray(diffSmall.rows() * diffSmall.cols())
        diffSmall.get(0, 0, v)
        val f = FloatArray(v.size) { v[it].toFloat() }
        f.sort()
        val median = f[f.size / 2]
        val dev = FloatArray(f.size) { abs(f[it] - median) }
        dev.sort()
        return 1.4826 * dev[dev.size / 2] / SCALE + 0.1
    }

    // ------------------------------------------------------------------ the two-stroke test

    private class Line(val px: Double, val py: Double, val dx: Double, val dy: Double) {
        fun dist(x: Double, y: Double) = abs((x - px) * dy - (y - py) * dx)
        val angle: Double get() = ((atan2(dy, dx) % PI) + PI) % PI
    }

    private fun twoStrokes(xs0: DoubleArray, ys0: DoubleArray, ws0: DoubleArray, n0: Int, area: Int, extent: Int): XDetection? {
        var n = n0
        var xs = xs0
        var ys = ys0
        var ws = ws0
        if (n > maxPixels) { // subsample big blobs
            val step = ceil(n.toDouble() / maxPixels).toInt()
            val m = (n + step - 1) / step
            xs = DoubleArray(m) { xs0[it * step] }
            ys = DoubleArray(m) { ys0[it * step] }
            ws = DoubleArray(m) { ws0[it * step] }
            n = m
        }
        val width = max(1.5, area / (2.83 * extent)) // stroke width for an X in this bounding box
        val tol = max(1.5, 0.75 * width + 0.75)
        val first = strongestLine(xs, ys, ws, n, width, null, null) ?: return null
        val rest = BooleanArray(n) { first.dist(xs[it], ys[it]) >= tol }
        if (rest.count { it } < 5) return null
        val second = strongestLine(xs, ys, ws, n, width, first.angle, rest) ?: return null

        var lines = arrayOf(first, second)
        val nearest = IntArray(n)
        val near = BooleanArray(n)
        repeat(3) { // assign pixels to the nearer line, refit each line
            for (i in 0 until n) {
                val d0 = lines[0].dist(xs[i], ys[i])
                val d1 = lines[1].dist(xs[i], ys[i])
                nearest[i] = if (d0 <= d1) 0 else 1
                near[i] = min(d0, d1) < tol
            }
            val a = fitLine(xs, ys, ws, n) { near[it] && nearest[it] == 0 } ?: return null
            val b = fitLine(xs, ys, ws, n) { near[it] && nearest[it] == 1 } ?: return null
            lines = arrayOf(a, b)
        }
        if (near.count { it }.toDouble() / n < 0.75) return null // an X is just two strokes
        val (l1, l2) = lines
        val cross = l1.dx * l2.dy - l1.dy * l2.dx
        if (abs(cross) < 0.5) return null // strokes must cross at 30 degrees or more
        val rx = l2.px - l1.px
        val ry = l2.py - l1.py
        val a = (-rx * l2.dy + l2.dx * ry) / -cross
        val cx = l1.px + a * l1.dx
        val cy = l1.py + a * l1.dy
        var minD = Double.MAX_VALUE
        for (i in 0 until n) minD = min(minD, hypot(xs[i] - cx, ys[i] - cy))
        if (minD > tol + 1.5) return null // ink at the crossing

        val lengths = DoubleArray(2)
        var widest = 1.0
        for (k in 0..1) {
            var lo = Double.MAX_VALUE
            var hi = -Double.MAX_VALUE
            var sp = 0.0
            var sp2 = 0.0
            var cnt = 0
            for (i in 0 until n) {
                if (!near[i] || nearest[i] != k) continue
                val s = (xs[i] - cx) * lines[k].dx + (ys[i] - cy) * lines[k].dy
                lo = min(lo, s)
                hi = max(hi, s)
                val p = (xs[i] - cx) * lines[k].dy - (ys[i] - cy) * lines[k].dx // signed distance across the stroke
                sp += p; sp2 += p * p; cnt++
            }
            if (cnt > 1) widest = max(widest, sqrt(12 * max(sp2 / cnt - (sp / cnt) * (sp / cnt), 0.0))) // uniform band of width w has std w / sqrt(12)
            val len = hi - lo
            val arm = max(2.0, 0.18 * len)
            if (len < 6 || lo > -arm || hi < arm) return null // arms on both sides of the crossing
            lengths[k] = len
        }
        val ratio = lengths[0] / lengths[1]
        if (ratio < 0.4 || ratio > 2.5) return null
        return XDetection(cx, cy, l1.angle, l2.angle, lengths.average(), 0.0, lengths.average() / widest)
    }

    /** Weighted Hough vote: the line covering the most ink, optionally avoiding an angle. */
    private fun strongestLine(xs: DoubleArray, ys: DoubleArray, ws: DoubleArray, n: Int, width: Double,
                              avoid: Double?, mask: BooleanArray?): Line? {
        val binW = max(1.0, width)
        var reach = 1.0
        for (i in 0 until n) if (mask == null || mask[i]) reach = max(reach, xs[i] + ys[i] + 1.0)
        val nb = (2 * reach / binW).toInt() + 2
        val acc = DoubleArray(THETAS.size * nb)
        for (t in THETAS.indices) {
            if (avoid != null && angleDev(THETAS[t], avoid) < MIN_SEP) continue
            val s = SIN[t]
            val c = COS[t]
            val base = t * nb
            for (i in 0 until n) {
                if (mask != null && !mask[i]) continue
                val r = -xs[i] * s + ys[i] * c // points on a line at angle t satisfy this = rho
                acc[base + ((r + reach) / binW).toInt()] += ws[i]
            }
        }
        var best = -1
        var bestV = 0.0
        for (j in acc.indices) if (acc[j] > bestV) { bestV = acc[j]; best = j }
        if (best < 0) return null
        val th = THETAS[best / nb]
        val rho = -reach + (best % nb + 0.5) * binW
        return Line(-sin(th) * rho, cos(th) * rho, cos(th), sin(th))
    }

    private inline fun fitLine(xs: DoubleArray, ys: DoubleArray, ws: DoubleArray, n: Int, sel: (Int) -> Boolean): Line? {
        var sw = 0.0
        var sx = 0.0
        var sy = 0.0
        var cnt = 0
        for (i in 0 until n) if (sel(i)) { sw += ws[i]; sx += ws[i] * xs[i]; sy += ws[i] * ys[i]; cnt++ }
        if (cnt < 4) return null
        val mx = sx / sw
        val my = sy / sw
        var sxx = 0.0
        var syy = 0.0
        var sxy = 0.0
        for (i in 0 until n) if (sel(i)) {
            val dx = xs[i] - mx
            val dy = ys[i] - my
            sxx += ws[i] * dx * dx
            syy += ws[i] * dy * dy
            sxy += ws[i] * dx * dy
        }
        val ang = 0.5 * atan2(2 * sxy, sxx - syy) // principal direction
        return Line(mx, my, cos(ang), sin(ang))
    }

    private fun dedupe(dets: List<XDetection>, radius: Double = 4.0): List<XDetection> {
        val kept = ArrayList<XDetection>()
        for (d in dets.sortedByDescending { it.strength }) {
            if (kept.all { hypot(it.x - d.x, it.y - d.y) > radius }) kept += d
        }
        return kept
    }

    companion object {
        private val THETAS = DoubleArray(60) { it * PI / 60 } // 3 degree steps; line fits refine
        private val COS = DoubleArray(THETAS.size) { cos(THETAS[it]) }
        private val SIN = DoubleArray(THETAS.size) { sin(THETAS[it]) }
        private const val MIN_SEP = 25 * PI / 180
        private const val SCALE = 8 // fixed-point steps per gray level

        fun angleDev(a: Double, b: Double): Double {
            val d = (((a - b + PI / 2) % PI) + PI) % PI - PI / 2
            return abs(d)
        }
    }
}
