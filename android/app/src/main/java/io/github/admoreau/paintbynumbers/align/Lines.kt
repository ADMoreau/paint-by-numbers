package io.github.admoreau.paintbynumbers.align

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.stream.IntStream
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Thin sets of line points, each with a unit normal (across the line) and a strength. The sketch's
 * chalk or pencil lines in a camera frame come out this way, and so do the target image's edges;
 * the aligner matches the one to the other. Ported from `prototypes/sketch_align/salign/lines.py`.
 */
class LinePoints(val x: FloatArray, val y: FloatArray, val nx: FloatArray, val ny: FloatArray, val strength: FloatArray) {
    val size: Int get() = x.size

    fun subset(idx: IntArray) = LinePoints(
        FloatArray(idx.size) { x[idx[it]] }, FloatArray(idx.size) { y[idx[it]] },
        FloatArray(idx.size) { nx[idx[it]] }, FloatArray(idx.size) { ny[idx[it]] },
        FloatArray(idx.size) { strength[idx[it]] },
    )
}

internal fun Mat.floats(): FloatArray = FloatArray((total() * channels()).toInt()).also { get(0, 0, it) }

/** Bilinear sample of a w x h float image at (x, y), clamping to the border. */
private fun sample(a: FloatArray, w: Int, h: Int, x: Float, y: Float): Float {
    val xc = x.coerceIn(0f, (w - 1).toFloat())
    val yc = y.coerceIn(0f, (h - 1).toFloat())
    val x0 = min(xc.toInt(), w - 2)
    val y0 = min(yc.toInt(), h - 2)
    val fx = xc - x0
    val fy = yc - y0
    val i = y0 * w + x0
    return (a[i] * (1 - fx) + a[i + 1] * fx) * (1 - fy) + (a[i + w] * (1 - fx) + a[i + w + 1] * fx) * fy
}

/** Median of [v] (approximate for big arrays: every k-th value). */
private fun median(v: FloatArray, n: Int = v.size, maxSamples: Int = 200_000): Float {
    if (n == 0) return 0f
    val step = max(1, n / maxSamples)
    val s = FloatArray((n + step - 1) / step) { v[it * step] }
    s.sort()
    return s[s.size / 2]
}

/** The p-th percentile (0..100) of [v], with linear interpolation as numpy does. */
internal fun percentile(v: FloatArray, p: Double): Double {
    if (v.isEmpty()) return 0.0
    val s = v.copyOf().also { it.sort() }
    val r = (s.size - 1) * p / 100.0
    val lo = r.toInt()
    val hi = min(lo + 1, s.size - 1)
    return s[lo] + (s[hi] - s[lo]) * (r - lo)
}

object Lines {
    /**
     * Ridge points of chalk or pencil lines about [widthPx] wide in a grayscale (CV_8UC1) image.
     *
     * A pixel is on a line if, across the line, the image is curved like a ridge ([light] lines) or a
     * valley (dark lines), it is the strongest point across, and it is lighter (darker) than the
     * ground on *both* sides, which rules out the edge of a painted area. Thresholds are in units of
     * the image's own noise, so lighting and exposure don't matter.
     */
    fun sketch(gray: Mat, light: Boolean, widthPx: Double = 2.5, kHigh: Double = 5.0, kLow: Double = 2.5, minPixels: Int = 12): LinePoints {
        val w = gray.cols()
        val h = gray.rows()
        val g = Mat()
        gray.convertTo(g, CvType.CV_32F)
        if (!light) Core.subtract(Mat(g.size(), CvType.CV_32F, Scalar(255.0)), g, g)
        val sigma = max(0.8, widthPx / 2.5)
        val s = Mat()
        Imgproc.GaussianBlur(g, s, Size(0.0, 0.0), sigma)
        val dxxM = Mat()
        val dyyM = Mat()
        val dxyM = Mat()
        Imgproc.Sobel(s, dxxM, CvType.CV_32F, 2, 0, 3)
        Imgproc.Sobel(s, dyyM, CvType.CV_32F, 0, 2, 3)
        Imgproc.Sobel(s, dxyM, CvType.CV_32F, 1, 1, 3)
        val b1 = Mat()
        Imgproc.GaussianBlur(g, b1, Size(0.0, 0.0), 1.0)
        val G = g.floats()
        val S = s.floats()
        val XX = dxxM.floats()
        val YY = dyyM.floats()
        val XY = dxyM.floats()
        val B1 = b1.floats()
        for (m in listOf(g, s, dxxM, dyyM, dxyM, b1)) m.release()

        // the Hessian's most negative eigenvalue is across a light ridge; its eigenvector is the normal
        val n = w * h
        val strength = FloatArray(n)
        val nxA = FloatArray(n)
        val nyA = FloatArray(n)
        IntStream.range(0, h).parallel().forEach { row -> for (i in row * w until (row + 1) * w) {
            val dxx = XX[i]
            val dyy = YY[i]
            val dxy = XY[i]
            val tr = (dxx + dyy) / 2
            val disc = sqrt(((dxx - dyy) / 2) * ((dxx - dyy) / 2) + dxy * dxy)
            val lam = tr - disc
            val other = tr + disc
            if (lam < 0 && abs(lam) > abs(other)) strength[i] = -lam
            var vx = dxy
            var vy = lam - dxx
            if (abs(vx) + abs(vy) < 1e-6f) { vx = lam - dyy; vy = dxy }
            val nn = sqrt(vx * vx + vy * vy) + 1e-12f
            nxA[i] = vx / nn
            nyA[i] = vy / nn
        } }
        var nPos = 0
        for (v in strength) if (v > 0) nPos++

        // noise: robust spread of the fine detail; strength scale: robust size of the ridge responses
        val noiseImg = FloatArray(n) { G[it] - B1[it] }
        val med = median(noiseImg)
        for (i in 0 until n) noiseImg[i] = abs(noiseImg[i] - med)
        val noise = 1.4826f * median(noiseImg) + 0.5f
        val pos = FloatArray(nPos)
        run { var k = 0; for (i in 0 until n) if (strength[i] > 0) pos[k++] = strength[i] }
        val sScale = max(if (nPos > 0) 1.4826f * median(pos) else 1f, 1e-6f)

        // strongest across (non-maximum suppression) and lighter than the ground on both sides
        val off = (2.5 * sigma + 1.0).toFloat()
        val resp = FloatArray(n)
        val low = ByteArray(n)
        IntStream.range(0, h).parallel().forEach { row -> for (i in row * w until (row + 1) * w) {
            val st = strength[i]
            if (st <= 0) continue
            val x = (i % w).toFloat()
            val y = (i / w).toFloat()
            val nx = nxA[i]
            val ny = nyA[i]
            if (st < sample(strength, w, h, x + nx, y + ny) || st < sample(strength, w, h, x - nx, y - ny)) continue
            val side = min(S[i] - sample(S, w, h, x + nx * off, y + ny * off), S[i] - sample(S, w, h, x - nx * off, y - ny * off))
            if (side <= 2 * noise) continue
            resp[i] = st / sScale
            if (resp[i] > kLow) low[i] = 1
        } }

        // hysteresis: keep connected runs that are long enough and reach the high threshold somewhere
        val lowM = Mat(h, w, CvType.CV_8UC1).also { it.put(0, 0, low) }
        val labels = Mat()
        val stats = Mat()
        val cents = Mat()
        val nLab = Imgproc.connectedComponentsWithStats(lowM, labels, stats, cents, 8, CvType.CV_32S)
        val lab = IntArray(n).also { labels.get(0, 0, it) }
        val st = IntArray(nLab * 5).also { stats.get(0, 0, it) }
        for (m in listOf(lowM, labels, stats, cents)) m.release()
        val highMax = FloatArray(nLab)
        for (i in 0 until n) if (low[i].toInt() != 0 && resp[i] > highMax[lab[i]]) highMax[lab[i]] = resp[i]
        val keep = BooleanArray(nLab) { it > 0 && st[it * 5 + Imgproc.CC_STAT_AREA] >= minPixels && highMax[it] > kHigh }
        var count = 0
        for (i in 0 until n) if (low[i].toInt() != 0 && keep[lab[i]]) count++
        val ox = FloatArray(count)
        val oy = FloatArray(count)
        val onx = FloatArray(count)
        val ony = FloatArray(count)
        val os = FloatArray(count)
        var k = 0
        for (i in 0 until n) {
            if (low[i].toInt() == 0 || !keep[lab[i]]) continue
            ox[k] = (i % w).toFloat(); oy[k] = (i / w).toFloat()
            onx[k] = nxA[i]; ony[k] = nyA[i]; os[k] = resp[i]
            k++
        }
        return LinePoints(ox, oy, onx, ony, os)
    }

    /**
     * Colour edges of the target image (CV_8UC3, RGB) at a couple of scales: Canny on whichever Lab
     * channel changes most at each pixel, with the gradient direction as normal. [minGrad] is an
     * absolute floor so smooth gradients and noise are not edges.
     */
    fun imageEdges(rgb: Mat, sigmas: DoubleArray = doubleArrayOf(1.5, 3.0), highPct: Double = 85.0, minGrad: Double = 20.0): LinePoints {
        val w = rgb.cols()
        val h = rgb.rows()
        val n = w * h
        val lab8 = Mat()
        Imgproc.cvtColor(rgb, lab8, Imgproc.COLOR_RGB2Lab)
        val lab = Mat()
        lab8.convertTo(lab, CvType.CV_32FC3)
        lab8.release()
        val edges = BooleanArray(n)
        val nxA = FloatArray(n)
        val nyA = FloatArray(n)
        val magA = FloatArray(n)
        val b = Mat()
        val gxM = Mat()
        val gyM = Mat()
        for (sg in sigmas) {
            Imgproc.GaussianBlur(lab, b, Size(0.0, 0.0), sg)
            Imgproc.Sobel(b, gxM, CvType.CV_32F, 1, 0, 3)
            Imgproc.Sobel(b, gyM, CvType.CV_32F, 0, 1, 3)
            val GX = gxM.floats()
            val GY = gyM.floats()
            val gxk = FloatArray(n)
            val gyk = FloatArray(n)
            val m = FloatArray(n)
            var mMax = 0f
            for (i in 0 until n) {
                var best = -1f
                for (c in 0 until 3) {
                    val ax = GX[3 * i + c]
                    val ay = GY[3 * i + c]
                    val q = ax * ax + ay * ay
                    if (q > best) { best = q; gxk[i] = ax; gyk[i] = ay }
                }
                m[i] = sqrt(best)
                if (m[i] > mMax) mMax = m[i]
            }
            val hi = max(percentile(m, highPct), minGrad)
            val scale = 2000.0 / max(mMax.toDouble(), 1e-6)
            val dx = ShortArray(n) { (gxk[it] * scale).toInt().toShort() }
            val dy = ShortArray(n) { (gyk[it] * scale).toInt().toShort() }
            val dxM = Mat(h, w, CvType.CV_16SC1).also { it.put(0, 0, dx) }
            val dyM = Mat(h, w, CvType.CV_16SC1).also { it.put(0, 0, dy) }
            val eM = Mat()
            Imgproc.Canny(dxM, dyM, eM, 0.4 * hi * scale, hi * scale, true)
            val e = ByteArray(n).also { eM.get(0, 0, it) }
            for (mm in listOf(dxM, dyM, eM)) mm.release()
            for (i in 0 until n) {
                if (e[i].toInt() == 0 || edges[i]) continue
                val mi = max(m[i], 1e-6f)
                nxA[i] = gxk[i] / mi
                nyA[i] = gyk[i] / mi
                magA[i] = m[i]
                edges[i] = true
            }
        }
        for (mm in listOf(lab, b, gxM, gyM)) mm.release()
        var count = 0
        for (i in 0 until n) if (edges[i]) count++
        val ox = FloatArray(count)
        val oy = FloatArray(count)
        val onx = FloatArray(count)
        val ony = FloatArray(count)
        val os = FloatArray(count)
        var k = 0
        for (i in 0 until n) {
            if (!edges[i]) continue
            ox[k] = (i % w).toFloat(); oy[k] = (i / w).toFloat()
            onx[k] = nxA[i]; ony[k] = nyA[i]; os[k] = magA[i]
            k++
        }
        return LinePoints(ox, oy, onx, ony, os)
    }
}
