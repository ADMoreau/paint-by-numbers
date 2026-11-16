package io.github.admoreau.paintbynumbers.plan

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Draws a layer's thickness regions as an overlay: each region filled with what it should look like
 * once the layer is painted (the plan's picture of the painting after this layer), and outlined in
 * white (boldly where the paint ends, thinly between its thicknesses), so it can be traced. With [only] set, just that thickness is filled (the rest of the layer
 * is faintly shaded for context).
 *
 * The plan's thickness map is small (often 256 px across), so each plan pixel covers a few mm of
 * canvas. Blown up pixel by pixel the regions look like stacks of boxes; instead each region's mask
 * is blurred slightly and upscaled smoothly, and every output pixel goes to the strongest region,
 * which rounds the staircase edges into curves without losing thin strokes.
 */
object RegionRenderer {
    /** Blur of each region's mask, in plan pixels, before upscaling: enough to round edges, little enough to keep 1 px strokes. */
    private const val SMOOTH_SIGMA = 0.7

    /** [fill]: the regions in their colours (drawn at the overlay's opacity); [lines]: their white outlines, same size (drawn solid). */
    class Regions(val fill: Bitmap, val lines: Bitmap)

    fun render(plan: Plan, layer: PlanLayer, only: PlanThickness?, longSide: Int = 1600): Regions? {
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
        val s = longSide.toDouble() / max(w, h)
        val sw = max(1, (w * s).roundToInt())
        val sh = max(1, (h * s).roundToInt())

        // smooth region map: each region's mask (and "no paint"), blurred a little and upscaled; the strongest wins
        val label = IntArray(sw * sh) { -1 }
        val best = FloatArray(sw * sh) { Float.NEGATIVE_INFINITY }
        val mask = Mat(h, w, CvType.CV_32F)
        val up = Mat()
        val maskArr = FloatArray(w * h)
        val upArr = FloatArray(sw * sh)
        for (k in -1 until levels.size) {
            if (k >= 0 && index.none { it == k }) continue
            for (i in maskArr.indices) maskArr[i] = if (index[i] == k) 1f else 0f
            mask.put(0, 0, maskArr)
            Imgproc.GaussianBlur(mask, mask, Size(0.0, 0.0), SMOOTH_SIGMA)
            Imgproc.resize(mask, up, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            up.get(0, 0, upArr)
            for (i in upArr.indices) if (upArr[i] > best[i]) { best[i] = upArr[i]; label[i] = k }
        }
        mask.release()
        up.release()

        // what each spot should look like once this layer is painted (or, without that picture, the thickness's swatch colour)
        val expected = layer.stepFile?.let { BitmapFactory.decodeFile(plan.file(it).path) }?.let { b ->
            val scaled = Bitmap.createScaledBitmap(b, sw, sh, true)
            IntArray(sw * sh).also { scaled.getPixels(it, 0, sw, 0, 0, sw, sh) }
        }
        val flat = IntArray(levels.size) { runCatching { Color.parseColor(levels[it].colorHex) }.getOrDefault(Color.GRAY) }
        val shade = Color.argb(60, 128, 128, 128)
        fun shown(k: Int) = k >= 0 && (only == null || levels[k] == only)
        val out = IntArray(sw * sh)
        for (i in out.indices) {
            val k = label[i]
            if (k < 0) continue
            out[i] = if (shown(k)) (expected?.get(i) ?: flat[k]) or 0xFF000000.toInt() else shade
        }
        // outlines, solid white: bold where the paint ends, thin between thicknesses of the same layer
        val lines = IntArray(sw * sh)
        val white = Color.WHITE
        val thin = Color.argb(200, 255, 255, 255)
        for (y in 0 until sh) for (x in 0 until sw) {
            val k = label[y * sw + x]
            if (!shown(k)) continue
            var outer = false
            var inner = false
            loop@ for (dy in -2..2) for (dx in -2..2) {
                val d = abs(dx) + abs(dy)
                if (d > 2) continue
                val xx = x + dx
                val yy = y + dy
                val kk = if (xx < 0 || yy < 0 || xx >= sw || yy >= sh) -1 else label[yy * sw + xx]
                if (kk == k) continue
                if (!shown(kk)) { outer = true; break@loop }
                if (d <= 1) inner = true
            }
            if (outer) lines[y * sw + x] = white else if (inner) lines[y * sw + x] = thin
        }
        return Regions(Bitmap.createBitmap(out, sw, sh, Bitmap.Config.ARGB_8888), Bitmap.createBitmap(lines, sw, sh, Bitmap.Config.ARGB_8888))
    }
}
