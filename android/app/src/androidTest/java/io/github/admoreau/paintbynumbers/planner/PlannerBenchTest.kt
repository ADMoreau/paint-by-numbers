package io.github.admoreau.paintbynumbers.planner

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.github.admoreau.paintbynumbers.plan.PlanStore
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.io.File

/**
 * Plans a painting on the phone and times it: 14 one-weight layers at 512 px with a 13-paint
 * palette over mauve, the settings of the desktop run it is compared with.
 *
 * Plans `bench.png` from the app's external files folder if it is there
 * (adb push img.png /sdcard/Android/data/io.github.admoreau.paintbynumbers/files/bench.png),
 * otherwise a synthetic image. The plan is written, read back as the app reads plans, then deleted.
 */
@RunWith(AndroidJUnit4::class)
class PlannerBenchTest {
    companion object {
        @BeforeClass
        @JvmStatic
        fun load() = check(OpenCVLoader.initLocal()).let { }

        val PALETTE = listOf(
            "Light Magenta", "Zinc White", "Titan Violet Pale", "Van Dyke Brown Hue", "Titan Mars Pale", "Cadmium Orange",
            "Dioxazine Purple", "Raw Umber", "Primary Magenta", "Hooker's Green Hue", "Light Phthalo Green", "Mars Yellow", "Cobalt Blue",
        )
    }

    private fun synthetic(w: Int, h: Int): Bitmap {
        val px = IntArray(w * h) { p ->
            val x = (p % w).toDouble() / w
            val y = (p / w).toDouble() / h
            var r = 0.45 + 0.5 * y; var g = 0.65 + 0.33 * y; var b = 0.95
            if (y > 0.55 + 0.12 * kotlin.math.sin(6 * x)) { r = 0.25 + 0.2 * x; g = 0.5 - 0.15 * y; b = 0.15 }
            if (x > 0.6 && x < 0.8 && y > 0.45 && y < 0.7) { r = 0.7; g = 0.15; b = 0.12 }
            if ((x - 0.15) * (x - 0.15) + (y - 0.15) * (y - 0.15) < 0.006) { r = 1.0; g = 0.85; b = 0.3 }
            fun c(v: Double) = (v.coerceIn(0.0, 1.0) * 255).toInt()
            (0xFF shl 24) or (c(r) shl 16) or (c(g) shl 8) or c(b)
        }
        return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
    }

    @Test
    fun plansAndTimes() {
        val app = InstrumentationRegistry.getInstrumentation().targetContext
        val file = File(app.getExternalFilesDir(null), "bench.png")
        val image = if (file.exists()) BitmapFactory.decodeFile(file.path) else synthetic(1024, 768)
        println("planner bench: ${if (file.exists()) "bench.png" else "synthetic image"} ${image.width}x${image.height}, ${Parallel.threads} threads")
        val paints = PaintDb.select(PhonePlanner.paints(app), PALETTE)
        val start = SystemClock.elapsedRealtime()
        var last = start
        val dir = PhonePlanner.run(image, paints, "#915F6D", UniformConfig(layers = 14), app.cacheDir, progress = { msg ->
            val now = SystemClock.elapsedRealtime()
            println("planner bench: %6.1f s (+%5.1f) %s".format((now - start) / 1000.0, (now - last) / 1000.0, msg))
            last = now
        })
        val seconds = (SystemClock.elapsedRealtime() - start) / 1000.0
        try {
            val plan = PlanStore.load(dir)
            val recipe = org.json.JSONObject(File(dir, "recipe.json").readText())
            val err = recipe.getJSONObject("error").getDouble("mean")
            println("planner bench: done in %.1f s: %d layers, mean ΔE %.3f, %dx%d".format(seconds, plan.layers.size, err, plan.width, plan.height))
            for (l in plan.layers) println("planner bench:   ${l.number}. ${l.paint} ${l.thicknesses.joinToString { "${it.label} %.1f%%".format(100 * it.area) }}")
            assertEquals(512, maxOf(plan.width, plan.height))
            assertTrue(plan.layers.isNotEmpty())
            for (l in plan.layers) assertEquals("${l.paint} has one weight", 1, l.thicknesses.size)
        } finally {
            dir.deleteRecursively()
        }
    }
}
