package io.github.admoreau.paintbynumbers.plan

import android.content.Context
import android.net.Uri
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.zip.ZipInputStream

/** One thickness a layer is painted at, and what that means at the easel. */
data class PlanThickness(
    val coats: Double,
    val label: String, // e.g. "1/8 coat"
    val how: String, // e.g. "1 coat of 1 part paint to 7 parts Gloss Medium"
    val area: Double, // share of the canvas
    val pixelValue: Int, // its gray value in the layer's thickness map
    val colorHex: String, // the paint at this thickness over the ground (or white)
    val onWhiteHex: String? = null, // the swatch: this thickness over white, over black, over a flat background
    val onBlackHex: String? = null,
    val onGroundHex: String? = null,
)

data class PlanLayer(
    val number: Int,
    val paint: String,
    val transparency: String,
    val fullStrengthHex: String,
    val mapFile: String?, // gray thickness map
    val stepFile: String?, // the painting after this layer
    val swatchFile: String?,
    val thicknesses: List<PlanThickness>,
)

/** A plan made by the desktop planner and imported from its project.pbn. */
data class Plan(
    val dir: File,
    val width: Int,
    val height: Int,
    val backgroundHex: String,
    val groundPaint: String?, // a ground coat to paint first, if the plan has one
    val layers: List<PlanLayer>,
    val targetFile: String,
    val previewFile: String,
) {
    fun file(name: String) = File(dir, name)
}

private val FRACTIONS = mapOf(0.125 to "1/8", 0.25 to "1/4", 0.5 to "1/2", 0.75 to "3/4")

/** "1/8 coat", "1 coat", "1 1/2 coats", "2 coats". */
fun coatsLabel(c: Double): String {
    val whole = c.toInt()
    val part = FRACTIONS[c - whole]
    val text = listOfNotNull(whole.takeIf { it > 0 }?.toString(), part).joinToString(" ")
    return "$text coat" + if (c > 1) "s" else ""
}

object PlanStore {
    private val SAFE_NAME = Regex("^[A-Za-z0-9_\\-.]+(/[A-Za-z0-9_\\-.]+)*$")
    private const val MAX_TOTAL = 300L * 1024 * 1024

    /** Unpacks a project.pbn into a fresh folder in private storage; returns the folder's name. */
    fun import(context: Context, uri: Uri): String {
        val name = "plan_${System.currentTimeMillis()}"
        val dir = File(context.filesDir, name)
        dir.mkdirs()
        try {
            var total = 0L
            context.contentResolver.openInputStream(uri)!!.use { raw ->
                ZipInputStream(raw.buffered()).use { zip ->
                    while (true) {
                        val e = zip.nextEntry ?: break
                        if (e.isDirectory) continue
                        if (!SAFE_NAME.matches(e.name) || e.name.split('/').any { it == ".." || it == "." }) throw IOException("unexpected file in package: ${e.name}")
                        val out = File(dir, e.name)
                        if (!out.canonicalPath.startsWith(dir.canonicalPath + File.separator)) throw IOException("unexpected file in package: ${e.name}")
                        out.parentFile?.mkdirs()
                        out.outputStream().use { o ->
                            val buf = ByteArray(64 * 1024)
                            while (true) {
                                val n = zip.read(buf)
                                if (n < 0) break
                                total += n
                                if (total > MAX_TOTAL) throw IOException("package is too large")
                                o.write(buf, 0, n)
                            }
                        }
                    }
                }
            }
            load(dir) // fails if it is not a plan
            return name
        } catch (e: Exception) {
            dir.deleteRecursively()
            throw e
        }
    }

    fun load(dir: File): Plan {
        val meta = File(dir, "package.json").takeIf { it.exists() }?.let { JSONObject(it.readText()) }
        val r = JSONObject(File(dir, "recipe.json").readText())
        val image = r.getJSONObject("image")
        val levels = r.getJSONArray("thickness_levels")
        val pixelOf = HashMap<Double, Int>()
        val howOf = HashMap<Double, String>()
        for (i in 0 until levels.length()) {
            val lv = levels.getJSONObject(i)
            pixelOf[lv.getDouble("coats")] = lv.getInt("pixel_value")
            howOf[lv.getDouble("coats")] = lv.getString("how")
        }
        val bg = r.getJSONObject("background")
        var groundPaint: String? = null
        val layers = ArrayList<PlanLayer>()
        val arr = r.getJSONArray("layers")
        for (i in 0 until arr.length()) {
            val l = arr.getJSONObject(i)
            if (l.optString("role") == "ground") {
                groundPaint = l.optString("paint")
                continue
            }
            val swatch = l.optJSONObject("swatch")
            val colors = swatch?.optJSONObject("on_ground") ?: swatch?.optJSONObject("on_white")
            fun swatchHex(ground: String, key: String) = swatch?.optJSONObject(ground)?.optString(key)?.takeIf { it.isNotEmpty() }
            val byArea = l.optJSONObject("coats_by_area") ?: JSONObject()
            val thick = byArea.keys().asSequence().map { key ->
                val c = key.toDouble()
                PlanThickness(
                    coats = c,
                    label = coatsLabel(c),
                    how = howOf[c] ?: "",
                    area = byArea.getDouble(key),
                    pixelValue = pixelOf[c] ?: Math.round(c / 2.0 * 255).toInt(),
                    colorHex = colors?.optString(key)?.takeIf { it.isNotEmpty() } ?: swatch?.optString("full_strength") ?: "#808080",
                    onWhiteHex = swatchHex("on_white", key),
                    onBlackHex = swatchHex("on_black", key),
                    onGroundHex = swatchHex("on_ground", key),
                )
            }.filter { it.area > 0 }.sortedBy { it.coats }.toList()
            layers += PlanLayer(
                number = l.getInt("layer"),
                paint = l.getString("paint"),
                transparency = l.optString("transparency"),
                fullStrengthHex = swatch?.optString("full_strength") ?: "#808080",
                mapFile = l.optString("file").ifEmpty { null },
                stepFile = l.optString("step_file").ifEmpty { null },
                swatchFile = l.optString("swatch_file").ifEmpty { null },
                thicknesses = thick,
            )
        }
        return Plan(
            dir = dir,
            width = image.getInt("width"),
            height = image.getInt("height"),
            backgroundHex = bg.optString("srgb_hex", "#ffffff"),
            groundPaint = groundPaint,
            layers = layers,
            targetFile = meta?.optString("target_file")?.ifEmpty { null } ?: "preview.png",
            previewFile = "preview.png",
        )
    }
}
