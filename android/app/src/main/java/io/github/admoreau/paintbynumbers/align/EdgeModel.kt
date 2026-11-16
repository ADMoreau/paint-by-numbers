package io.github.admoreau.paintbynumbers.align

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt

private const val K_BINS = 8

internal fun angleMod(a: Double): Double = ((a % PI) + PI) % PI

/**
 * The target image as the aligner sees it: its colour edges, downscaled so the long side is
 * [longSide] "model pixels", sorted into 8 direction bins. Each bin has a distance map with labels,
 * so for any point and line direction the nearest edge running roughly the same way (within one
 * bin either side, about 30 degrees) is a table lookup, not a search.
 */
class EdgeModel(rgb: Mat, longSide: Int = 800) {
    /** Model pixels per target pixel. */
    val f: Double = longSide.toDouble() / max(rgb.cols(), rgb.rows())
    val w: Int = (rgb.cols() * f).roundToInt()
    val h: Int = (rgb.rows() * f).roundToInt()
    val long: Int = max(w, h)
    val targetWidth = rgb.cols()
    val targetHeight = rgb.rows()
    val edges: LinePoints

    private val dt: Array<FloatArray>
    private val lab: Array<IntArray> // index into the tables below, or -1
    private val tableX: FloatArray
    private val tableY: FloatArray
    private val tableNx: FloatArray
    private val tableNy: FloatArray

    /** Distance to the nearest edge of any direction (CV_32F, model size): what the search slides. */
    val dtAll: Mat

    init {
        val small = Mat()
        Imgproc.resize(rgb, small, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        edges = Lines.imageEdges(small)
        small.release()
        val n = w * h
        val bins = IntArray(edges.size) { floor(angleMod(atan2(edges.ny[it].toDouble(), edges.nx[it].toDouble())) / (PI / K_BINS)).toInt() % K_BINS }
        val tx = ArrayList<FloatArray>()
        val ty = ArrayList<FloatArray>()
        val tnx = ArrayList<FloatArray>()
        val tny = ArrayList<FloatArray>()
        var offset = 0
        val src = ByteArray(n)
        val srcM = Mat(h, w, CvType.CV_8UC1)
        val dtM = Mat()
        val labM = Mat()
        dt = Array(K_BINS) { FloatArray(0) }
        lab = Array(K_BINS) { IntArray(0) }
        for (b in 0 until K_BINS) {
            src.fill(-1) // 255: not an edge
            var any = false
            for (i in 0 until edges.size) if (bins[i] == b) { src[edges.y[i].toInt() * w + edges.x[i].toInt()] = 0; any = true }
            if (!any) {
                dt[b] = FloatArray(n) { 1e6f }
                lab[b] = IntArray(n) { -1 }
                continue
            }
            srcM.put(0, 0, src)
            Imgproc.distanceTransformWithLabels(srcM, dtM, labM, Imgproc.DIST_L2, 5, Imgproc.DIST_LABEL_PIXEL)
            dt[b] = dtM.floats()
            val l = IntArray(n).also { labM.get(0, 0, it) }
            var nLab = 0
            for (v in l) if (v + 1 > nLab) nLab = v + 1
            val bx = FloatArray(nLab)
            val by = FloatArray(nLab)
            val bnx = FloatArray(nLab)
            val bny = FloatArray(nLab)
            for (i in 0 until edges.size) {
                if (bins[i] != b) continue
                val L = l[edges.y[i].toInt() * w + edges.x[i].toInt()]
                bx[L] = edges.x[i]; by[L] = edges.y[i]; bnx[L] = edges.nx[i]; bny[L] = edges.ny[i]
            }
            for (i in 0 until n) l[i] += offset
            lab[b] = l
            tx += bx; ty += by; tnx += bnx; tny += bny
            offset += nLab
        }
        fun cat(parts: List<FloatArray>): FloatArray {
            val out = FloatArray(parts.sumOf { it.size })
            var k = 0
            for (p in parts) { p.copyInto(out, k); k += p.size }
            return out
        }
        tableX = cat(tx); tableY = cat(ty); tableNx = cat(tnx); tableNy = cat(tny)
        src.fill(-1)
        for (i in 0 until edges.size) src[edges.y[i].toInt() * w + edges.x[i].toInt()] = 0
        srcM.put(0, 0, src)
        dtAll = Mat()
        Imgproc.distanceTransform(srcM, dtAll, Imgproc.DIST_L2, 5)
        for (m in listOf(srcM, dtM, labM)) m.release()
    }

    /**
     * For model point ([px], [py]) on a line whose normal points at angle [ang] (mod pi): distance to
     * the nearest edge within one direction bin either side, with that edge point and its normal in
     * [out] (x, y, nx, ny). Infinite outside the model.
     */
    fun nearest(px: Double, py: Double, ang: Double, out: DoubleArray): Double {
        if (!(px >= 0 && py >= 0 && px <= w - 1 && py <= h - 1)) return Double.POSITIVE_INFINITY
        val idx = Math.rint(py).toInt() * w + Math.rint(px).toInt()
        val b = floor(ang / (PI / K_BINS)).toInt().mod(K_BINS)
        var best = Float.POSITIVE_INFINITY
        var l = -1
        for (o in -1..1) {
            val bb = (b + o).mod(K_BINS)
            val d = dt[bb][idx]
            if (d < best) { best = d; l = lab[bb][idx] }
        }
        if (l < 0) return Double.POSITIVE_INFINITY
        out[0] = tableX[l].toDouble(); out[1] = tableY[l].toDouble()
        out[2] = tableNx[l].toDouble(); out[3] = tableNy[l].toDouble()
        return hypot(px - out[0], py - out[1]) // exact (sub-pixel) distance
    }

    fun release() = dtAll.release()

    companion object {
        fun fromBitmap(bitmap: Bitmap, longSide: Int = 800): EdgeModel {
            val rgba = Mat()
            Utils.bitmapToMat(bitmap, rgba)
            val rgb = Mat()
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
            rgba.release()
            return EdgeModel(rgb, longSide).also { rgb.release() }
        }
    }
}
