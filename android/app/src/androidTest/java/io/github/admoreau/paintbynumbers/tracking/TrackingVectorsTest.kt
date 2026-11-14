package io.github.admoreau.paintbynumbers.tracking

import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import kotlin.math.hypot

/**
 * Replays frames from the Python simulation (prototypes/x_tracking, exported to
 * tracking_vectors.json: detections, true camera homography and the Python locator's results)
 * through the Kotlin port, on the phone, and checks it does as well.
 */
@RunWith(AndroidJUnit4::class)
class TrackingVectorsTest {
    companion object {
        private const val TAG = "TrackingVectors"
        private lateinit var data: JSONObject

        @BeforeClass
        @JvmStatic
        fun load() {
            check(OpenCVLoader.initLocal())
            val ctx = InstrumentationRegistry.getInstrumentation().context
            data = JSONObject(ctx.assets.open("tracking_vectors.json").bufferedReader().readText())
        }
    }

    private fun locator(): Locator {
        val cam = data.getJSONObject("camera")
        val k = cam.getJSONArray("K")
        val intr = Intrinsics(k.getJSONArray(0).getDouble(0), k.getJSONArray(1).getDouble(1), k.getJSONArray(0).getDouble(2), k.getJSONArray(1).getDouble(2))
        val map = data.getJSONArray("map")
        val marks = (0 until map.length()).map {
            val o = map.getJSONObject(it)
            MapMark(o.getInt("id"), o.getDouble("x"), o.getDouble("y"), o.getDouble("d1"), o.getDouble("d2"), o.getDouble("size"))
        }
        return Locator(marks, intr, cam.getInt("width"), cam.getInt("height"), xSizeMm = 15.0, minCold = 4)
    }

    private fun dets(o: JSONObject): List<XDetection> {
        val a = o.getJSONArray("dets")
        return (0 until a.length()).map {
            val r = a.getJSONArray(it)
            XDetection(r.getDouble(0), r.getDouble(1), r.getDouble(2), r.getDouble(3), r.getDouble(4), r.getDouble(5))
        }
    }

    private fun matrix(a: JSONArray) = DoubleArray(9) { a.getJSONArray(it / 3).getDouble(it % 3) }

    /** Mean and max distance in canvas mm between where two homographies put a 9x7 grid across the view. */
    private fun overlayError(hEst: DoubleArray, hGt: DoubleArray, w: Int, h: Int): Pair<Double, Double> {
        val ie = M3.inv(hEst)!!
        val ig = M3.inv(hGt)!!
        var sum = 0.0
        var mx = 0.0
        for (iy in 0 until 7) for (ix in 0 until 9) {
            val u = w * (0.05 + 0.9 * ix / 8)
            val v = h * (0.05 + 0.9 * iy / 6)
            val a = M3.apply(ie, u, v)
            val b = M3.apply(ig, u, v)
            val e = hypot(a[0] - b[0], a[1] - b[1])
            sum += e
            mx = maxOf(mx, e)
        }
        return sum / 63 to mx
    }

    private fun idsOk(est: PoseEstimate, loc: Locator, truth: JSONArray): Boolean =
        est.matches.all { (i, m) -> !truth.isNull(i) && truth.getInt(i) == loc.marks[m].id }

    private fun median(v: List<Double>) = v.sorted().let { if (it.isEmpty()) Double.NaN else it[it.size / 2] }

    @Test
    fun coldStartMatchesPrototype() {
        val loc = locator()
        val w = data.getJSONObject("camera").getInt("width")
        val h = data.getJSONObject("camera").getInt("height")
        val cases = data.getJSONArray("cold")
        var located = 0
        var pyLocated = 0
        var wrong = 0
        val errs = ArrayList<Double>()
        val pyErrs = ArrayList<Double>()
        val ms = ArrayList<Double>()
        for (c in 0 until cases.length()) {
            val o = cases.getJSONObject(c)
            val py = o.getJSONObject("py")
            if (py.getBoolean("located")) { pyLocated++; pyErrs += py.getDouble("err_mean") }
            val d = dets(o)
            val t0 = SystemClock.elapsedRealtimeNanos()
            val est = loc.coldStart(d)
            ms += (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
            if (est == null) continue
            located++
            if (!idsOk(est, loc, o.getJSONArray("truth"))) wrong++
            errs += overlayError(est.H, matrix(o.getJSONArray("H")), w, h).first
        }
        val msg = "cold start: kotlin located $located/${cases.length()} (python $pyLocated), wrong ids $wrong, " +
            "median error %.2f mm (python %.2f), median %.0f ms, worst %.0f ms".format(median(errs), median(pyErrs), median(ms), ms.max())
        Log.i(TAG, msg)
        println(msg)
        // the app's setting: give up after 120 ms so a frame never stalls
        val quick = locator().apply { coldBudgetMs = 120 }
        var quickLocated = 0
        var quickWrong = 0
        val quickMs = ArrayList<Double>()
        for (c in 0 until cases.length()) {
            val o = cases.getJSONObject(c)
            val t0 = SystemClock.elapsedRealtimeNanos()
            val est = quick.coldStart(dets(o))
            quickMs += (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
            if (est == null) continue
            quickLocated++
            if (!idsOk(est, quick, o.getJSONArray("truth"))) quickWrong++
        }
        val quickMsg = "cold start with a 120 ms budget: located $quickLocated/${cases.length()}, wrong ids $quickWrong, median %.0f ms, worst %.0f ms".format(median(quickMs), quickMs.max())
        Log.i(TAG, quickMsg)
        println(quickMsg)
        assertEquals("wrong locks with budget: $quickMsg", 0, quickWrong)
        assertEquals("wrong locks: $msg", 0, wrong)
        assertTrue("too few locks: $msg", located >= 0.9 * pyLocated)
        assertTrue("less accurate: $msg", median(errs) <= 1.5 * median(pyErrs) + 0.2)
    }

    @Test
    fun sessionMatchesPrototype() {
        val loc = locator()
        val w = data.getJSONObject("camera").getInt("width")
        val h = data.getJSONObject("camera").getInt("height")
        val frames = data.getJSONArray("session")
        val tracker = Tracker(loc)
        var withPose = 0
        var pyWithPose = 0
        var wrong = 0
        var bigErr = 0
        val errs = ArrayList<Double>()
        val pyErrs = ArrayList<Double>()
        val ms = ArrayList<Double>()
        for (k in 0 until frames.length()) {
            val o = frames.getJSONObject(k)
            val py = o.getJSONObject("py")
            if (py.getString("mode") != "lost") { pyWithPose++; pyErrs += py.getDouble("err_mean") }
            val d = dets(o)
            val t0 = SystemClock.elapsedRealtimeNanos()
            val est = tracker.update(d)
            ms += (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
            if (est == null) continue
            withPose++
            if (est.matches.isNotEmpty() && !idsOk(est, loc, o.getJSONArray("truth"))) wrong++
            val e = overlayError(est.H, matrix(o.getJSONArray("H")), w, h).first
            errs += e
            if (e > 5.0) bigErr++
        }
        val msg = "session: kotlin $withPose/${frames.length()} frames with a pose (python $pyWithPose), wrong ids $wrong, " +
            "frames off by >5 mm $bigErr, median error %.2f mm (python %.2f), median %.1f ms, worst %.0f ms".format(
                median(errs), median(pyErrs), median(ms), ms.max(),
            )
        Log.i(TAG, msg)
        println(msg)
        assertTrue("wrong identities: $msg", wrong <= frames.length() / 100)
        assertTrue("lost too often: $msg", withPose >= 0.95 * pyWithPose)
        assertTrue("less accurate: $msg", median(errs) <= 1.5 * median(pyErrs) + 0.2)
    }
}
