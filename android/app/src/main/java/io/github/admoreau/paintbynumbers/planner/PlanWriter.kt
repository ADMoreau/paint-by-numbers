package io.github.admoreau.paintbynumbers.planner

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import io.github.admoreau.paintbynumbers.plan.PlanStore
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Plans on the phone and saves the plan as a folder in the same layout as an unpacked
 * project.pbn from the desktop planner, so the rest of the app reads it with [PlanStore.load]:
 * recipe.json, package.json, target.png, preview.png, layers/NN_paint.png (gray = thickness) and
 * steps/NN_paint.png (the painting after each layer).
 */
object PhonePlanner {
    /** The paint database bundled with the app (shared with the desktop planner). */
    fun paints(context: Context): List<Paint> =
        PaintDb.parse(context.assets.open(PaintDb.ASSET).bufferedReader().use { it.readText() })

    /** Decodes [uri], longer side at most [maxSide] px. */
    fun decode(context: Context, uri: Uri, maxSide: Int = 2048): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample; inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val bmp = context.contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, opts) }
            ?: error("could not read the image")
        return fit(bmp, maxSide)
    }

    /** [bmp] shrunk (area-averaged, like the desktop's Lanczos resize) so its longer side is [side]; never enlarged. */
    fun fit(bmp: Bitmap, side: Int): Bitmap {
        val s = side.toDouble() / max(bmp.width, bmp.height)
        if (s >= 1.0) return bmp
        val w = max(1, (bmp.width * s).roundToInt())
        val h = max(1, (bmp.height * s).roundToInt())
        val src = Mat()
        Utils.bitmapToMat(bmp, src)
        val dst = Mat()
        Imgproc.resize(src, dst, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(dst, out)
        src.release()
        dst.release()
        return out
    }

    /**
     * Plans [image] (shrunk to [size] px on its longer side) with [paints] over a flat background
     * [backgroundHex], and writes the plan to a new folder under [parent]. Returns the folder.
     */
    fun run(
        image: Bitmap,
        paints: List<Paint>,
        backgroundHex: String,
        config: UniformConfig,
        parent: File,
        size: Int = 512,
        progress: (String) -> Unit = {},
        cancelled: () -> Boolean = { false },
    ): File {
        val small = fit(image, size)
        val w = small.width
        val h = small.height
        val px = IntArray(w * h).also { small.getPixels(it, 0, w, 0, 0, w, h) }
        val bg = parseHex(backgroundHex).let { (r, g, b) -> DoubleArray(3).also { ColorMath.srgbToWorking(r / 255.0, g / 255.0, b / 255.0, it) } }
        val plan = UniformPlanner(paints, bg, config, progress, cancelled).plan(Target.fromArgb(w, h, px))
        val dir = File(parent, "plan_${System.currentTimeMillis()}")
        try {
            write(plan, dir, image, backgroundHex, config)
            PlanStore.load(dir) // fails if the app could not read it
        } catch (e: Throwable) {
            dir.deleteRecursively()
            throw e
        }
        return dir
    }

    private fun parseHex(hex: String): Triple<Int, Int, Int> {
        val s = hex.trim().removePrefix("#")
        require(Regex("[0-9a-fA-F]{6}").matches(s)) { "not a colour: $hex" }
        val v = s.toInt(16)
        return Triple((v shr 16) and 255, (v shr 8) and 255, v and 255)
    }

    private fun png(bmp: Bitmap, file: File) {
        file.parentFile?.mkdirs()
        file.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }

    private fun hex(argb: Int) = "#%06x".format(argb and 0xFFFFFF)

    private fun argb(r: Double, g: Double, b: Double) = ColorMath.workingToArgb(r, g, b)

    /** "1 full coat", "1 coat of 1 part paint to 3 parts Gloss Medium", as level_instructions in plan.py. */
    fun howToPaint(coats: Double): String {
        if (coats <= 0) return "no paint"
        val whole = coats.toInt()
        val part = coats - whole
        val pieces = ArrayList<String>()
        if (whole > 0) pieces += "$whole full coat" + if (whole > 1) "s" else ""
        if (part > 1e-9) {
            val den = (1..16).first { d -> kotlin.math.abs(part * d - Math.rint(part * d)) < 1e-9 || d == 16 }
            val num = Math.rint(part * den).toInt()
            val medium = den - num
            pieces += "1 coat of $num part${if (num > 1) "s" else ""} paint to $medium part${if (medium > 1) "s" else ""} Gloss Medium"
        }
        return pieces.joinToString(" + ")
    }

    private fun write(plan: UniformPlan, dir: File, source: Bitmap, backgroundHex: String, config: UniformConfig) {
        dir.mkdirs()
        val w = plan.w
        val h = plan.h
        val np = w * h
        val weights = config.weights.toSortedSet().toList()
        val tMax = weights.last()
        val bg = plan.background
        val flatGround = !(bg.all { kotlin.math.abs(it - 1.0) < 1e-3 } || bg.all { kotlin.math.abs(it) < 1e-3 })
        val digits = max(2, plan.layers.size.toString().length)

        // the painting after each layer
        val r = FloatArray(3 * np) { bg[it % 3].toFloat() }
        val out = IntArray(np)
        val layers = JSONArray()
        for ((i, layer) in plan.layers.withIndex()) {
            val name = "%0${digits}d_%s.png".format(i + 1, layer.paint.slug)
            val region = plan.regions[i]
            val co = Km.Coefficients(layer.paint.K, layer.paint.S)
            val gray = Math.rint(layer.coats / tMax * 255).toInt()
            val map = IntArray(np) { if (region[it]) (0xFF shl 24) or (gray * 0x010101) else 0xFF000000.toInt() }
            png(Bitmap.createBitmap(map, w, h, Bitmap.Config.ARGB_8888), File(dir, "layers/$name"))
            for (p in 0 until np) {
                if (region[p]) for (c in 0 until 3) r[3 * p + c] = co.layer(r[3 * p + c].toDouble(), layer.coats, c).toFloat()
                out[p] = argb(r[3 * p].toDouble(), r[3 * p + 1].toDouble(), r[3 * p + 2].toDouble())
            }
            png(Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888), File(dir, "steps/$name"))

            val key = UniformPlanner.fmt(layer.coats)
            val area = region.count { it }.toDouble() / np
            fun over(ground: DoubleArray) = JSONObject().put(key, hex(argb(co.layer(ground[0], layer.coats, 0), co.layer(ground[1], layer.coats, 1), co.layer(ground[2], layer.coats, 2))))
            val swatch = JSONObject()
                .put("full_strength", hex(argb(co.rInf[0], co.rInf[1], co.rInf[2])))
                .put("on_white", over(doubleArrayOf(1.0, 1.0, 1.0)))
                .put("on_black", over(doubleArrayOf(0.0, 0.0, 0.0)))
            if (flatGround) swatch.put("on_ground", over(bg))
            layers.put(
                JSONObject()
                    .put("layer", i + 1)
                    .put("file", "layers/$name")
                    .put("step_file", "steps/$name")
                    .put("paint", layer.paint.name)
                    .put("item", layer.paint.item)
                    .put("transparency", layer.paint.transparency)
                    .put("sheen", layer.paint.sheen ?: JSONObject.NULL)
                    .put("lightfastness", layer.paint.lightfastness ?: JSONObject.NULL)
                    .put("coverage", Math.round(area * 1e4) / 1e4)
                    .put("coats_by_area", JSONObject().put(key, Math.round(area * 1e4) / 1e4))
                    .put("swatch", swatch)
                    .put("coats", layer.coats)
                    .put("how", howToPaint(layer.coats)),
            )
        }
        val preview = IntArray(np) { argb(plan.preview[3 * it].toDouble(), plan.preview[3 * it + 1].toDouble(), plan.preview[3 * it + 2].toDouble()) }
        png(Bitmap.createBitmap(preview, w, h, Bitmap.Config.ARGB_8888), File(dir, "preview.png"))
        png(fit(source, 2048), File(dir, "target.png"))

        val bgLab = DoubleArray(3).also { ColorMath.workingToLab(bg[0], bg[1], bg[2], it) }
        val levels = JSONArray()
        for (lv in listOf(0.0) + weights) {
            levels.put(JSONObject().put("coats", lv).put("pixel_value", Math.rint(lv / tMax * 255).toInt()).put("how", howToPaint(lv)))
        }
        val rounds = JSONArray()
        for (rd in plan.rounds) rounds.put(JSONObject().put("paint", rd.paint).put("coats", rd.coats).put("position", rd.position).put("rms_error", Math.round(rd.rmsError * 1000) / 1000.0))
        val recipe = JSONObject()
            .put("image", JSONObject().put("width", w).put("height", h))
            .put("background", JSONObject().put("kind", "color").put("value", backgroundHex.lowercase()).put("srgb_hex", hex(argb(bg[0], bg[1], bg[2])))
                .put("lab", JSONArray(bgLab.map { Math.round(it * 100) / 100.0 })))
            .put("layers", layers)
            .put("thickness_levels", levels)
            .put("error", JSONObject().put("mean", Math.round(plan.meanError * 1000) / 1000.0).put("max", Math.round(plan.maxError * 1000) / 1000.0).put("metric", "CIE76 ΔE"))
            .put("selection", JSONObject().put("colors_clustered", plan.clusterCount).put("rms_error_background_only", Math.round(plan.baselineError * 1000) / 1000.0)
                .put("rounds", rounds).put("paints_used", JSONArray(plan.layers.map { it.paint.name }.toSortedSet().toList())))
            .put("settings", JSONObject().put("mode", "one weight per layer").put("planned_on", "phone").put("uniform", JSONObject()
                .put("layers", config.layers).put("weights", JSONArray(weights)).put("n_colors", config.nColors).put("paint_cost", config.paintCost)
                .put("smooth", config.smooth).put("edge", config.edge).put("min_region", config.minRegion)))
            .put("notes", JSONArray())
        File(dir, "recipe.json").writeText(recipe.toString(2))
        File(dir, "package.json").writeText(JSONObject().put("format", "paint-by-numbers project").put("version", 1).put("target_file", "target.png").toString(2))
    }
}
