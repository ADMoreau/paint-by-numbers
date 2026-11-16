package io.github.admoreau.paintbynumbers.align

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.github.admoreau.paintbynumbers.tracking.M3
import org.json.JSONObject
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot

/**
 * Snap to drawing: the Kotlin port must find the image where the Python prototype does
 * (`prototypes/sketch_align/export_test_case.py` writes the cases), on the same grayscale frame.
 *
 * - `sketch_align/synthetic*`: a simulated freehand sketch of the planner's synthetic landscape,
 *   photographed by a simulated phone, with ground truth.
 * - `local/real*` (not in git: a photo of the user's own artwork): a real chalk sketch on a
 *   painted canvas. Skipped when absent.
 */
@RunWith(AndroidJUnit4::class)
class SketchAlignTest {
    companion object {
        @BeforeClass
        @JvmStatic
        fun load() = check(OpenCVLoader.initLocal()).let { }
    }

    private val ctx = InstrumentationRegistry.getInstrumentation().context

    private fun has(path: String) = runCatching { ctx.assets.open(path).close() }.isSuccess
    private fun bitmap(path: String): Bitmap = ctx.assets.open(path).use { BitmapFactory.decodeStream(it) }
    private fun json(path: String) = JSONObject(ctx.assets.open(path).bufferedReader().readText())

    private fun gray(path: String): Mat {
        val rgba = Mat()
        Utils.bitmapToMat(bitmap(path), rgba)
        return Mat().also { Imgproc.cvtColor(rgba, it, Imgproc.COLOR_RGBA2GRAY); rgba.release() }
    }

    private class Run(val result: SketchAligner.Result, val model: EdgeModel, val info: JSONObject, val w: Int, val h: Int) {
        val gPython = DoubleArray(9) { info.getJSONArray("G_python").getDouble(it) }
    }

    private fun run(dir: String, name: String): Run {
        val info = json("$dir/$name.json")
        val g = gray("$dir/${name}_photo.png")
        var t = System.nanoTime()
        val model = EdgeModel.fromBitmap(bitmap("$dir/${name}_target.png"))
        val tModel = (System.nanoTime() - t) / 1_000_000
        t = System.nanoTime()
        val lines = Lines.sketch(g, info.getBoolean("light"), info.getDouble("line_px"))
        val tLines = (System.nanoTime() - t) / 1_000_000
        val r = SketchAligner.align(model, lines, g.cols(), g.rows(), null, 0.0)
        assertNotNull("no placement found", r)
        Log.i("SketchAlignTest", "$name: ${g.cols()}x${g.rows()}, ${lines.size} line points; model $tModel ms, lines $tLines ms, ${r!!.millis}; " +
            "score %.3f (python %.3f), next %.3f".format(r.score, info.getDouble("score_python"), r.secondScore))
        return Run(r, model, info, g.cols(), g.rows())
    }

    /** Median and 90th percentile of the distance (work px) between where two placements put the image's edges that land in the frame. */
    private fun disagreement(r: Run, ga: DoubleArray, gb: DoubleArray): Pair<Double, Double> {
        val ia = M3.inv(ga)!!
        val ib = M3.inv(gb)!!
        val pa = DoubleArray(2)
        val pb = DoubleArray(2)
        val d = ArrayList<Double>()
        val e = r.model.edges
        for (i in 0 until e.size) {
            M3.apply(ia, e.x[i].toDouble(), e.y[i].toDouble(), pa)
            M3.apply(ib, e.x[i].toDouble(), e.y[i].toDouble(), pb)
            if (pa[0] in 0.0..r.w.toDouble() && pa[1] in 0.0..r.h.toDouble() && pb[0] in 0.0..r.w.toDouble() && pb[1] in 0.0..r.h.toDouble()) d += hypot(pa[0] - pb[0], pa[1] - pb[1])
        }
        d.sort()
        return d[d.size / 2] to d[(d.size * 9) / 10]
    }

    /** Median error against ground truth, target px. */
    private fun truthError(r: Run, g: DoubleArray): Double {
        val q = r.info.getJSONArray("q")
        val m = r.info.getJSONArray("m")
        val p = DoubleArray(2)
        val err = (0 until q.length()).map { i ->
            M3.apply(g, q.getJSONArray(i).getDouble(0), q.getJSONArray(i).getDouble(1), p)
            hypot(p[0] - m.getJSONArray(i).getDouble(0), p[1] - m.getJSONArray(i).getDouble(1)) / r.model.f
        }.sorted()
        return err[err.size / 2]
    }

    @Test
    fun syntheticSketchIsFound() {
        val r = run("sketch_align", "synthetic")
        val mine = truthError(r, r.result.G)
        val theirs = truthError(r, r.gPython)
        val (med, p90) = disagreement(r, r.result.G, r.gPython)
        Log.i("SketchAlignTest", "synthetic: error vs truth %.1f target px (python %.1f); vs python median %.1f, p90 %.1f work px".format(mine, theirs, med, p90))
        assertTrue("error vs truth $mine (python $theirs)", mine <= theirs + 3.0)
        assertTrue("disagrees with the prototype: median $med px", med < 4.0)
    }

    @Test
    fun realSketchMatchesPrototype() {
        assumeTrue("no private test images", has("local/real.json"))
        val r = run("local", "real")
        val (med, p90) = disagreement(r, r.result.G, r.gPython)
        Log.i("SketchAlignTest", "real: vs python median %.1f, p90 %.1f work px".format(med, p90))
        assertTrue("disagrees with the prototype: median $med, p90 $p90 px", med < 4.0 && p90 < 10.0)
    }
}
