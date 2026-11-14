package io.github.admoreau.paintbynumbers.tracking

import org.opencv.core.Mat
import org.opencv.objdetect.ArucoDetector
import org.opencv.objdetect.DetectorParameters
import org.opencv.objdetect.Objdetect
import kotlin.math.hypot
import kotlin.math.sqrt

/** A printed square marker found in a frame: its id and four corners (x, y pairs, buffer pixels, the marker's own clockwise order from its top-left). */
class MarkerDetection(val id: Int, val corners: DoubleArray) {
    /** Mean side length in pixels. */
    val side: Double get() = (0 until 4).sumOf { k ->
        val j = (k + 1) % 4
        hypot(corners[2 * j] - corners[2 * k], corners[2 * j + 1] - corners[2 * k + 1])
    } / 4
}

/** The standard 4x4 ArUco markers (ids 0-49), with sub-pixel corners. */
class MarkerFinder {
    private val detector = ArucoDetector(
        Objdetect.getPredefinedDictionary(Objdetect.DICT_4X4_50),
        DetectorParameters().apply { set_cornerRefinementMethod(Objdetect.CORNER_REFINE_SUBPIX) },
    )

    fun find(gray: Mat): List<MarkerDetection> {
        val corners = ArrayList<Mat>()
        val ids = Mat()
        detector.detectMarkers(gray, corners, ids)
        val out = ArrayList<MarkerDetection>()
        for (k in corners.indices) {
            val c = FloatArray(8).also { corners[k].get(0, 0, it) }
            val id = IntArray(1).also { ids.get(k, 0, it) }[0]
            out += MarkerDetection(id, DoubleArray(8) { c[it].toDouble() })
            corners[k].release()
        }
        ids.release()
        return out
    }

    companion object {
        const val DICTIONARY_SIZE = 50
    }
}

/**
 * Tracks the canvas from square markers of known size, mapping them as it goes: the first marker
 * seen defines the canvas coordinates (mm, its top-left corner at the origin), and any other marker
 * seen while the canvas is located is added to the map, its corners averaged over many views.
 * So any number of markers, placed anywhere, work without a separate scan.
 *
 * Map entries are per corner: id = marker id * 4 + corner.
 */
class MarkerTracker(initialMap: List<MapMark>, private val sideMm: Double) {
    private val map = HashMap<Int, DoubleArray>() // corner id -> (x, y) in mm
    private val pending = HashMap<Int, DoubleArray>() // corner id -> running sum (x, y, n) for markers not yet mapped
    private var prev: DoubleArray? = null // last pose
    private var hold = 0
    private val ls = LeastSquares(6, doubleArrayOf(0.1, 0.1, 0.1, 10.0, 10.0, 10.0), fScale = 2.0, maxIter = 30)

    init {
        for (m in initialMap) map[m.id] = doubleArrayOf(m.x, m.y)
    }

    /** The current map, for saving. */
    fun mapMarks(): List<MapMark> = map.entries.sortedBy { it.key }.map { (id, p) -> MapMark(id, p[0], p[1], 0.0, 0.0, sideMm) }

    val markerCount: Int get() = map.size / 4

    class Result(val x: DoubleArray, val H: DoubleArray, val used: Int, val rmse: Double, val holding: Boolean)

    fun update(markers: List<MarkerDetection>, cam: Intrinsics): Result? {
        if (map.isEmpty()) {
            // the first marker seen anchors the map: corners at (0,0), (s,0), (s,s), (0,s)
            val first = markers.maxByOrNull { it.side } ?: return null
            val s = sideMm
            val pts = listOf(0.0 to 0.0, s to 0.0, s to s, 0.0 to s)
            for (k in 0 until 4) map[first.id * 4 + k] = doubleArrayOf(pts[k].first, pts[k].second)
        }
        val mapPts = ArrayList<Double>()
        val imgPts = ArrayList<Double>()
        for (m in markers) for (k in 0 until 4) {
            val p = map[m.id * 4 + k] ?: continue
            mapPts += p[0]; mapPts += p[1]
            imgPts += m.corners[2 * k]; imgPts += m.corners[2 * k + 1]
        }
        val n = mapPts.size / 2
        if (n < 4) { // no mapped marker in view: hold the last pose briefly
            val p = prev ?: return null
            if (++hold > 15) { prev = null; return null }
            return Result(p, PoseProjector(cam, p).homography(), 0, 0.0, true)
        }
        hold = 0
        val map2 = mapPts.toDoubleArray()
        val img2 = imgPts.toDoubleArray()
        fun refine(start: DoubleArray): DoubleArray = ls.solve(start) { xx ->
            val p = PoseProjector(cam, xx)
            val out = DoubleArray(2)
            DoubleArray(2 * n) { i ->
                if (i % 2 == 0) p.projectFinite(map2[i], map2[i + 1], out)
                out[i % 2] - img2[i]
            }
        }.x
        fun fresh(): DoubleArray? = OpenCvPnp.solve(map2, img2, cam).filter { PoseProjector(cam, it).height() > 0 }
            .minByOrNull { reprojection(it, map2, img2, cam) }?.let(::refine)
        fun ok(x: DoubleArray) = sqrt(reprojection(x, map2, img2, cam) / n) <= 6.0 && PoseProjector(cam, x).height() > 0
        // continue from the last pose (smooth); start afresh if that fails (after a jump)
        val x = prev?.let(::refine)?.takeIf(::ok) ?: fresh()?.takeIf(::ok)
        if (x == null) { prev = null; return null }
        val rmse = sqrt(reprojection(x, map2, img2, cam) / n)
        prev = x
        learn(markers, x, cam)
        return Result(x, PoseProjector(cam, x).homography(), n / 4, rmse, false)
    }

    private fun reprojection(x: DoubleArray, map: DoubleArray, img: DoubleArray, cam: Intrinsics): Double {
        val p = PoseProjector(cam, x)
        val out = DoubleArray(2)
        var ss = 0.0
        for (k in 0 until map.size / 2) {
            p.project(map[2 * k], map[2 * k + 1], out)
            ss += (out[0] - img[2 * k]) * (out[0] - img[2 * k]) + (out[1] - img[2 * k + 1]) * (out[1] - img[2 * k + 1])
        }
        return if (ss.isFinite()) ss else 1e18
    }

    /** Markers not yet in the map: place their corners on the canvas through the pose, average, and adopt once steady. */
    private fun learn(markers: List<MarkerDetection>, x: DoubleArray, cam: Intrinsics) {
        val hi = M3.inv(PoseProjector(cam, x).homography()) ?: return
        for (m in markers) {
            if (map.containsKey(m.id * 4)) continue
            if (m.side < 20) continue // too small to place accurately
            for (k in 0 until 4) {
                val q = M3.apply(hi, m.corners[2 * k], m.corners[2 * k + 1])
                if (!q[0].isFinite()) return
                val acc = pending.getOrPut(m.id * 4 + k) { DoubleArray(3) }
                acc[0] += q[0]; acc[1] += q[1]; acc[2] += 1.0
            }
            if ((pending[m.id * 4]?.get(2) ?: 0.0) >= 10) {
                for (k in 0 until 4) {
                    val acc = pending.remove(m.id * 4 + k) ?: continue
                    map[m.id * 4 + k] = doubleArrayOf(acc[0] / acc[2], acc[1] / acc[2])
                }
            }
        }
    }
}
