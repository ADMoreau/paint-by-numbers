package io.github.admoreau.paintbynumbers.tracking

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot

/**
 * Real camera frames of a painted canvas with chalk X's (plus paint texture, a ragged painted
 * edge and a laptop keyboard in view). The learned X-likeness cutoff must keep every X and almost nothing else.
 */
@RunWith(AndroidJUnit4::class)
class RealFramesTest {
    companion object {
        @BeforeClass
        @JvmStatic
        fun load() = check(OpenCVLoader.initLocal()).let { }

        // X centres found by hand-checking each frame
        val X_CENTRES = listOf(
            listOf(673.7 to 762.9, 1017.3 to 905.4, 563.3 to 132.8, 454.1 to 818.4, 947.0 to 394.1, 635.3 to 436.2, 382.4 to 376.2),
            listOf(665.9 to 750.7, 1008.5 to 895.7, 561.5 to 117.7, 445.3 to 805.1, 945.5 to 383.1, 630.9 to 423.1, 377.5 to 361.0),
            listOf(671.2 to 751.1, 1015.6 to 894.6, 560.7 to 114.3, 450.2 to 806.6, 948.6 to 380.5, 633.3 to 422.6, 377.4 to 361.0),
        )
    }

    private fun frame(k: Int): Mat {
        val ctx = InstrumentationRegistry.getInstrumentation().context
        val bmp = ctx.assets.open("real_frame_$k.png").use { BitmapFactory.decodeStream(it) }
        val rgba = Mat()
        Utils.bitmapToMat(bmp, rgba)
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        rgba.release()
        return gray
    }

    @Test
    fun timing() {
        val det = XDetector().apply { polarity = XDetector.Polarity.LIGHT }
        val f = frame(0)
        repeat(3) { det.detect(f) } // warm up
        val sums = DoubleArray(5)
        val runs = 10
        repeat(runs) {
            det.detect(f)
            for (i in sums.indices) sums[i] += det.lastTimings[i] / runs
        }
        println("timing: filters %.1f ms, copies %.1f ms, blobs %.1f ms, stroke tests %.1f ms over %.0f blobs".format(sums[0], sums[1], sums[2], sums[3], sums[4]))
    }

    @Test
    fun learnedCutoffKeepsTheXsAndDropsClutter() {
        val det = XDetector().apply { polarity = XDetector.Polarity.LIGHT }
        for (k in 0..2) {
            val dets = det.detect(frame(k))
            val t = ScanSession.chooseMinSlender(dets)
            val strict = dets.filter { it.slender >= t }
            val msg = "frame $k: ${dets.size} detections, cutoff $t, kept ${strict.size}: " +
                strict.joinToString { "(%.0f,%.0f s=%.1f)".format(it.x, it.y, it.slender) }
            println(msg)
            for ((x, y) in X_CENTRES[k]) assertTrue("missed X at ($x, $y): $msg", strict.any { hypot(it.x - x, it.y - y) < 10 })
            // an occasional texture streak can pass (the scan drops marks that are not seen again and again)
            assertTrue("extra detections kept: $msg", strict.size <= X_CENTRES[k].size + 1)
        }
    }
}
