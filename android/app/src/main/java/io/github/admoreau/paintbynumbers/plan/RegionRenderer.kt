package io.github.admoreau.paintbynumbers.plan

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import kotlin.math.abs
import kotlin.math.max

/**
 * Draws a layer's thickness regions as an overlay: each region filled with its thickness's highlight
 * colour (the same colour for a thickness in every layer, bright so it shows on any canvas) and
 * outlined, so it can be traced. With [only] set, just that thickness is filled (the rest of the
 * layer is faintly shaded for context).
 */
object RegionRenderer {
    private val HIGHLIGHT = listOf(
        0.125 to 0xFFFFEB3B.toInt(), 0.25 to 0xFF76FF03.toInt(), 0.5 to 0xFF00E5FF.toInt(), 0.75 to 0xFF448AFF.toInt(),
        1.0 to 0xFFE040FB.toInt(), 1.5 to 0xFFFF4081.toInt(), 2.0 to 0xFFFF9100.toInt(),
    )

    /** The highlight colour for a thickness (nearest standard level). */
    fun highlight(coats: Double): Int = HIGHLIGHT.minBy { abs(it.first - coats) }.second

    fun render(plan: Plan, layer: PlanLayer, only: PlanThickness?, targetLongSide: Int = 1400): Bitmap? {
        val map = layer.mapFile?.let { BitmapFactory.decodeFile(plan.file(it).path) } ?: return null
        val w = map.width
        val h = map.height
        val px = IntArray(w * h)
        map.getPixels(px, 0, w, 0, 0, w, h)
        map.recycle()
        // each pixel's thickness index in layer.thicknesses, or -1 for no paint
        val levels = layer.thicknesses
        val index = IntArray(w * h) { i ->
            val v = Color.red(px[i])
            if (v < 4) -1 else levels.indices.minByOrNull { abs(levels[it].pixelValue - v) } ?: -1
        }
        val scale = max(1, targetLongSide / max(w, h))
        val sw = w * scale
        val sh = h * scale
        val fill = IntArray(levels.size) { k ->
            val c = highlight(levels[k].coats)
            Color.argb(if (only == null || levels[k] == only) 150 else 0, Color.red(c), Color.green(c), Color.blue(c))
        }
        val shade = Color.argb(60, 128, 128, 128)
        val out = IntArray(sw * sh)
        fun level(x: Int, y: Int): Int = index[(y / scale) * w + (x / scale)]
        for (y in 0 until sh) for (x in 0 until sw) {
            val k = level(x, y)
            if (k < 0) continue
            val selected = only == null || levels[k] == only
            out[y * sw + x] = if (selected) fill[k] else shade
        }
        // outlines: a 2-pixel line wherever the region changes, in black or white to stand out from the fill
        for (y in 0 until sh) for (x in 0 until sw) {
            val k = level(x, y)
            if (k < 0) continue
            val selected = only == null || levels[k] == only
            if (!selected) continue
            var edge = false
            loop@ for (dy in -2..2) for (dx in -2..2) {
                if (abs(dx) + abs(dy) > 2) continue
                val xx = x + dx
                val yy = y + dy
                if (xx < 0 || yy < 0 || xx >= sw || yy >= sh || level(xx, yy) != k) { edge = true; break@loop }
            }
            if (edge) {
                val c = fill[k]
                out[y * sw + x] = Color.argb(255, Color.red(c), Color.green(c), Color.blue(c)) // opaque edge in the same colour
            }
        }
        return Bitmap.createBitmap(out, sw, sh, Bitmap.Config.ARGB_8888)
    }
}
