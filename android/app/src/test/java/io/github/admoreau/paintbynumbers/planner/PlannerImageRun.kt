package io.github.admoreau.paintbynumbers.planner

import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * A whole plan of a real image on the computer, to compare with the desktop planner. Skipped unless
 * PLANNER_IMAGE names a binary PPM (P6) image already at the planning size, e.g.
 *
 *     PLANNER_IMAGE=/path/img512.ppm PLANNER_BG='#915F6D' PLANNER_PAINTS='Zinc White,Cobalt Blue,...' \
 *         ./gradlew testDebugUnitTest --tests '*PlannerImageRun*' -i
 */
class PlannerImageRun {
    @Test
    fun run() {
        val path = System.getenv("PLANNER_IMAGE")
        assumeTrue("PLANNER_IMAGE not set", path != null)
        val (w, h, px) = readPpm(File(path!!))
        val db = PaintDb.parse(File("../../paint_by_numbers/data/paints_heavy_body.json").readText())
        val paints = System.getenv("PLANNER_PAINTS")?.let { PaintDb.select(db, it.split(",").map(String::trim)) } ?: db
        val hex = (System.getenv("PLANNER_BG") ?: "#ffffff").removePrefix("#").toInt(16)
        val bg = DoubleArray(Bands.N).also { ColorMath.srgbToWorking((hex shr 16 and 255) / 255.0, (hex shr 8 and 255) / 255.0, (hex and 255) / 255.0, it) }
        val layers = System.getenv("PLANNER_LAYERS")?.toInt() ?: 14
        val t0 = System.nanoTime()
        val plan = UniformPlanner(paints, bg, UniformConfig(layers = layers), { println("%7.1f s  %s".format((System.nanoTime() - t0) / 1e9, it)) })
            .plan(Target.fromArgb(w, h, px))
        println("done in %.1f s on %d threads: %d layers, mean ΔE %.3f".format((System.nanoTime() - t0) / 1e9, Parallel.threads, plan.layers.size, plan.meanError))
        plan.layers.forEachIndexed { i, l -> println("  ${i + 1}. ${l.paint.name} ${l.coats} (%.1f%%)".format(100.0 * plan.regions[i].count { it } / (w * h))) }
    }

    /** Width, height and ARGB pixels of a binary PPM with 8-bit channels and no comments. */
    private fun readPpm(f: File): Triple<Int, Int, IntArray> {
        val bytes = f.readBytes()
        var pos = 0
        fun token(): String {
            while (bytes[pos].toInt().toChar().isWhitespace()) pos++
            val start = pos
            while (!bytes[pos].toInt().toChar().isWhitespace()) pos++
            return String(bytes, start, pos - start)
        }
        check(token() == "P6")
        val w = token().toInt()
        val h = token().toInt()
        check(token() == "255")
        pos++
        val px = IntArray(w * h) { i ->
            val o = pos + 3 * i
            (0xFF shl 24) or ((bytes[o].toInt() and 255) shl 16) or ((bytes[o + 1].toInt() and 255) shl 8) or (bytes[o + 2].toInt() and 255)
        }
        return Triple(w, h, px)
    }
}
