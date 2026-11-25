package io.github.admoreau.paintbynumbers.tracking

import org.opencv.calib3d.Calib3d
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import kotlin.math.hypot
import kotlin.math.max

/** An X seen in a keyframe, with the identity the scan gave it. */
class ScanObservation(val id: Int, val x: Double, val y: Double, val dir1: Double, val dir2: Double, val size: Double)

class Keyframe(val obs: List<ScanObservation>)

/**
 * Collects views of the canvas while the user moves the phone, for [BundleAdjust] to turn into a
 * map. Runs on the analysis thread, one call per frame.
 *
 * X identities are followed in 2D: every X gets a position in the first frame's pixels, and each
 * frame a homography from that first frame (the canvas is flat, so one homography relates any two
 * views) predicts where every X should be. Detections are matched to predictions, the homography
 * is re-fitted robustly, and X's not seen before are added. A frame becomes a keyframe once the
 * view has changed enough since the last one.
 */
class ScanSession(private val width: Int, private val height: Int, private val minSlender: Double = 0.0) {
    /** [startCenter]: where the centre of the first view is in this frame (buffer pixels), showing how far the view has moved; null when lost. */
    class Status(val idsByDet: Map<Int, Int>, val mapped: Int, val keyframes: Int, val lost: Boolean, val startCenter: DoubleArray? = null)

    /** Each X's position in the first frame's pixels (a projective map of the canvas). */
    val refPositions = ArrayList<DoubleArray>()
    private val seen = ArrayList<Int>() // frames in which each X was matched
    private val inView = ArrayList<Int>() // frames in which each X should have been visible
    val keyframes = ArrayList<Keyframe>()
    private var hRefToCur: DoubleArray? = null
    private var lastKey: Map<Int, DoubleArray> = emptyMap()
    private var lostFrames = 0
    private val diag = hypot(width.toDouble(), height.toDouble())

    private fun medianNearest(dets: List<XDetection>): Double {
        if (dets.size < 2) return 60.0
        val d = dets.map { a -> dets.filter { it !== a }.minOf { hypot(it.x - a.x, it.y - a.y) } }.sorted()
        return d[d.size / 2]
    }

    /**
     * Only detections at least [minSlender] X-like may start the map or add an X to it; following
     * known X's accepts somewhat less (60%), since a prediction already says where to look.
     */
    fun update(allDets: List<XDetection>): Status {
        val keep = allDets.indices.filter { allDets[it].slender >= 0.6 * minSlender }
        val dets = keep.map { allDets[it] }
        val st = track(dets)
        return Status(st.idsByDet.mapKeys { keep[it.key] }, st.mapped, st.keyframes, st.lost, st.startCenter)
    }

    private fun track(dets: List<XDetection>): Status {
        val h = hRefToCur
        if (h == null) {
            val seeds = dets.indices.filter { dets[it].slender >= minSlender }
            if (seeds.size < 6) return Status(emptyMap(), 0, 0, false)
            val obs = seeds.mapIndexed { id, i ->
                val d = dets[i]
                refPositions += doubleArrayOf(d.x, d.y)
                seen += 1
                inView += 1
                ScanObservation(id, d.x, d.y, d.dir1, d.dir2, d.size)
            }
            hRefToCur = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            addKeyframe(obs)
            return Status(seeds.withIndex().associate { (id, i) -> i to id }, 0, keyframes.size, false, doubleArrayOf(width / 2.0, height / 2.0))
        }
        val spacing = medianNearest(dets)
        val pred = refPositions.map { M3.apply(h, it[0], it[1]) }
        val wide = (0.35 + 0.15 * minOf(lostFrames, 3)) * spacing
        var matches = mutualNearest(dets, pred, max(20.0, wide))
        if (matches.size < MIN_MATCHES) return lost()
        var hNew = fitHomography(matches, dets, ransac = true) ?: return lost()
        // re-associate against the fitted homography with a tighter gate, then refit on all of them
        val pred2 = refPositions.map { M3.apply(hNew, it[0], it[1]) }
        matches = mutualNearest(dets, pred2, max(8.0, 0.2 * spacing))
        if (matches.size < MIN_MATCHES) return lost()
        hNew = fitHomography(matches, dets, ransac = false) ?: return lost()
        hRefToCur = hNew
        lostFrames = 0

        val ids = HashMap(matches)
        for ((j, p) in refPositions.withIndex()) {
            val q = M3.apply(hNew, p[0], p[1])
            if (q[0] > 20 && q[0] < width - 20 && q[1] > 20 && q[1] < height - 20) inView[j]++
        }
        for (j in matches.values) seen[j]++
        val inv = M3.inv(hNew)
        if (inv != null) {
            val pred3 = refPositions.map { M3.apply(hNew, it[0], it[1]) }
            for ((i, d) in dets.withIndex()) {
                if (i in ids || d.slender < minSlender) continue
                val near = pred3.minOfOrNull { hypot(it[0] - d.x, it[1] - d.y) } ?: Double.POSITIVE_INFINITY
                if (near < 0.5 * spacing) continue // probably a known X, just poorly placed
                val r = M3.apply(inv, d.x, d.y)
                if (!r[0].isFinite()) continue
                ids[i] = refPositions.size
                refPositions += r
                seen += 1
                inView += 1
            }
        }
        val obs = ids.map { (i, id) -> val d = dets[i]; ScanObservation(id, d.x, d.y, d.dir1, d.dir2, d.size) }
        val common = obs.filter { it.id in lastKey }
        if (common.size >= 4) {
            val moved = common.sumOf { val p = lastKey[it.id]!!; hypot(it.x - p[0], it.y - p[1]) } / common.size
            if (moved > 0.06 * diag && keyframes.size < MAX_KEYFRAMES) addKeyframe(obs)
        }
        return Status(ids, reliableIds().size, keyframes.size, false, M3.apply(hNew, width / 2.0, height / 2.0))
    }

    /** X's found again and again as the view changed: real marks, not flickering clutter. */
    fun reliableIds(): Set<Int> = refPositions.indices.filter { seen[it] >= MIN_SEEN && seen[it] >= 0.6 * inView[it] }.toSet()

    private fun lost(): Status {
        lostFrames++
        return Status(emptyMap(), reliableIds().size, keyframes.size, true)
    }

    private fun addKeyframe(obs: List<ScanObservation>) {
        keyframes += Keyframe(obs)
        lastKey = obs.associate { it.id to doubleArrayOf(it.x, it.y) }
    }

    /** Detection index -> X id, for pairs that are each other's nearest within [gate] px. */
    private fun mutualNearest(dets: List<XDetection>, pred: List<DoubleArray>, gate: Double): Map<Int, Int> {
        val out = HashMap<Int, Int>()
        if (dets.isEmpty() || pred.isEmpty()) return out
        val detBest = IntArray(dets.size) { -1 }
        for ((i, d) in dets.withIndex()) {
            var best = -1
            var bd = gate
            for ((j, p) in pred.withIndex()) {
                val dd = hypot(p[0] - d.x, p[1] - d.y)
                if (dd < bd) { bd = dd; best = j }
            }
            detBest[i] = best
        }
        for ((i, j) in detBest.withIndex()) {
            if (j < 0) continue
            val p = pred[j]
            val back = dets.indices.minBy { hypot(dets[it].x - p[0], dets[it].y - p[1]) }
            if (back == i) out[i] = j
        }
        return out
    }

    private fun fitHomography(matches: Map<Int, Int>, dets: List<XDetection>, ransac: Boolean): DoubleArray? {
        val src = MatOfPoint2f(*matches.values.map { Point(refPositions[it][0], refPositions[it][1]) }.toTypedArray())
        val dst = MatOfPoint2f(*matches.keys.map { Point(dets[it].x, dets[it].y) }.toTypedArray())
        val mask = Mat()
        val hm = if (ransac) Calib3d.findHomography(src, dst, Calib3d.RANSAC, 4.0, mask) else Calib3d.findHomography(src, dst, 0, 0.0, mask)
        src.release(); dst.release(); mask.release()
        if (hm.empty()) return null
        val out = DoubleArray(9).also { hm.get(0, 0, it) }
        hm.release()
        return if (out.all { it.isFinite() }) out else null
    }

    companion object {
        const val MAX_KEYFRAMES = 30
        const val MIN_SEEN = 8
        const val MIN_MATCHES = 5

        /** The strictest X-likeness that still leaves [want] detections: picks out the user's X's from clutter. */
        fun chooseMinSlender(dets: List<XDetection>, want: Int = 6): Double {
            var t = 8.0
            while (t > 4.0 && dets.count { it.slender >= t } < want) t -= 0.5
            return t
        }
    }
}
