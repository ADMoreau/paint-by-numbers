package io.github.admoreau.paintbynumbers.tracking

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * A simulated scan: X's on a flat canvas seen by a moving camera (first view deliberately tilted),
 * fed through [ScanSession] and [BundleAdjust]. The map must come out the true shape: after the
 * best rotation, shift and scale onto the real X positions, errors are well under a millimetre.
 */
@RunWith(AndroidJUnit4::class)
class ScanTest {
    companion object {
        @BeforeClass
        @JvmStatic
        fun load() = check(OpenCVLoader.initLocal()).let { }
    }

    private val cam = Intrinsics(979.0, 979.0, 640.0, 480.0)
    private val w = 1280
    private val h = 960

    private class TrueX(val x: Double, val y: Double, val d1: Double, val d2: Double)

    private fun canvas(rng: Random): List<TrueX> {
        val out = ArrayList<TrueX>()
        var y = 40.0
        while (y < 450) {
            var x = 40.0
            while (x < 600) {
                val a = Math.toRadians(45 + rng.nextDouble(-20.0, 20.0))
                out += TrueX(x + rng.nextDouble(-15.0, 15.0), y + rng.nextDouble(-15.0, 15.0), a % PI, (a + Math.toRadians(90 + rng.nextDouble(-12.0, 12.0))) % PI)
                x += 75
            }
            y += 75
        }
        return out
    }

    /** Camera looking at [target] from [dist] mm, tilted [tilt] degrees towards direction [dir], rolled [roll]. */
    private fun pose(tx: Double, ty: Double, dist: Double, tilt: Double, dir: Double, roll: Double): DoubleArray {
        val a = Math.toRadians(dir)
        val axis = doubleArrayOf(cos(a), sin(a), 0.0)
        val rt = rodrigues(DoubleArray(3) { axis[it] * Math.toRadians(tilt) })
        val rr = rodrigues(doubleArrayOf(0.0, 0.0, Math.toRadians(roll)))
        val rwc = M3.mul(rt, rr) // camera axes in world coordinates (columns)
        val c = doubleArrayOf(tx - dist * rwc[2], ty - dist * rwc[5], -dist * rwc[8])
        val r = DoubleArray(9) { rwc[(it % 3) * 3 + it / 3] } // R = Rwc^T
        val t = DoubleArray(3) { i -> -(r[i * 3] * c[0] + r[i * 3 + 1] * c[1] + r[i * 3 + 2] * c[2]) }
        // rotation matrix -> vector
        val angle = Math.acos(((r[0] + r[4] + r[8] - 1) / 2).coerceIn(-1.0, 1.0))
        val rv = if (angle < 1e-9) doubleArrayOf(0.0, 0.0, 0.0) else {
            val k = angle / (2 * sin(angle))
            doubleArrayOf((r[7] - r[5]) * k, (r[2] - r[6]) * k, (r[3] - r[1]) * k)
        }
        return rv + t
    }

    private fun detect(marks: List<TrueX>, x: DoubleArray, rng: Random, truthOut: MutableList<Int>): List<XDetection> {
        val p = PoseProjector(cam, x)
        val out = ArrayList<XDetection>()
        val a = DoubleArray(2)
        val b = DoubleArray(2)
        for ((i, m) in marks.withIndex()) {
            p.project(m.x, m.y, a)
            if (!a[0].isFinite() || a[0] < 15 || a[0] > w - 15 || a[1] < 15 || a[1] > h - 15 || rng.nextDouble() < 0.05) continue
            val dirs = doubleArrayOf(m.d1, m.d2).map { d ->
                p.project(m.x + 7.5 * cos(d), m.y + 7.5 * sin(d), b)
                ((atan2(b[1] - a[1], b[0] - a[0]) + Math.toRadians(rng.nextDouble(-2.0, 2.0))) % PI + PI) % PI
            }
            p.project(m.x + 15, m.y, b)
            val size = hypot(b[0] - a[0], b[1] - a[1])
            out += XDetection(a[0] + rng.nextDouble(-0.7, 0.7), a[1] + rng.nextDouble(-0.7, 0.7), dirs[0], dirs[1], size, rng.nextDouble(5.0, 20.0))
            truthOut += i
        }
        if (rng.nextDouble() < 0.5) { // a stray false detection
            out += XDetection(rng.nextDouble(50.0, w - 50.0), rng.nextDouble(50.0, h - 50.0), 0.8, 2.3, 30.0, 6.0)
            truthOut += -1
        }
        return out
    }

    /** RMS distance after the best similarity transform of [est] onto [truth] (paired lists of points). */
    private fun alignedRms(est: List<DoubleArray>, truth: List<DoubleArray>): Pair<Double, Double> {
        val n = est.size
        val ex = est.sumOf { it[0] } / n; val ey = est.sumOf { it[1] } / n
        val tx = truth.sumOf { it[0] } / n; val ty = truth.sumOf { it[1] } / n
        var sxx = 0.0; var sxy = 0.0; var ee = 0.0
        for (i in 0 until n) {
            val ax = est[i][0] - ex; val ay = est[i][1] - ey
            val bx = truth[i][0] - tx; val by = truth[i][1] - ty
            sxx += ax * bx + ay * by
            sxy += ax * by - ay * bx
            ee += ax * ax + ay * ay
        }
        val s = hypot(sxx, sxy) / ee
        val th = atan2(sxy, sxx)
        var err = 0.0
        for (i in 0 until n) {
            val ax = est[i][0] - ex; val ay = est[i][1] - ey
            val px = s * (cos(th) * ax - sin(th) * ay) + tx
            val py = s * (sin(th) * ax + cos(th) * ay) + ty
            err += (px - truth[i][0]) * (px - truth[i][0]) + (py - truth[i][1]) * (py - truth[i][1])
        }
        return sqrt(err / n) to s
    }

    private fun simulate(moveScale: Double, seed: Int): Pair<BundleAdjust.Result, Double>? {
        val rng = Random(seed)
        val marks = canvas(rng)
        val session = ScanSession(w, h)
        val scanToTruth = HashMap<Int, Int>()
        for (k in 0 until 90) {
            val t = k / 89.0
            // start tilted 18 degrees, then circle around while keeping the canvas in view
            val x = pose(
                300 + moveScale * 120 * sin(2 * PI * t), 225 + moveScale * 90 * sin(4 * PI * t),
                720 + 60 * sin(2 * PI * t), 18 + moveScale * 8 * sin(2 * PI * t), 30 + 360 * t * moveScale, 5 * sin(2 * PI * t),
            )
            val truth = ArrayList<Int>()
            val dets = detect(marks, x, rng, truth)
            val st = session.update(dets)
            for ((di, id) in st.idsByDet) if (truth[di] >= 0) scanToTruth.putIfAbsent(id, truth[di])
        }
        val res = BundleAdjust.run(session.keyframes, session.refPositions, cam, keep = session.reliableIds())
        if (res == null) {
            println("scan (move $moveScale): no map from ${session.keyframes.size} keyframes")
            return null
        }
        val pairs = res.marks.mapNotNull { m -> scanToTruth[m.id]?.let { doubleArrayOf(m.x, m.y) to doubleArrayOf(marks[it].x, marks[it].y) } }
        val (rms, scale) = alignedRms(pairs.map { it.first }, pairs.map { it.second })
        val msg = "scan (move ${moveScale}): ${res.marks.size}/${marks.size} X's, ${res.keyframes} keyframes, reprojection %.2f px, view spread %.1f deg, map error %.2f mm, scale %.3f".format(res.rmsePx, res.viewSpreadDeg, rms, scale)
        Log.i("ScanTest", msg)
        println(msg)
        return res to rms
    }

    @Test
    fun movingScanRecoversTrueShape() {
        for (seed in 1..3) {
            val out = simulate(1.0, seed)
            assertNotNull("bundle adjustment failed", out)
            val (res, rms) = out!!
            assertTrue("map error $rms mm", rms < 1.0)
            assertTrue("too few X's mapped: ${res.marks.size}", res.marks.size >= 40)
            assertTrue("view spread ${res.viewSpreadDeg}", res.viewSpreadDeg > 10)
        }
    }

    @Test
    fun barelyMovingScanReportsLowSpread() {
        val res = simulate(0.1, 7)?.first ?: return // too few distinct views for a map at all: also fine
        assertTrue("view spread ${res.viewSpreadDeg} should be small", res.viewSpreadDeg < 10)
    }
}
