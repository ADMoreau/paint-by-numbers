package io.github.admoreau.paintbynumbers.project

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import io.github.admoreau.paintbynumbers.tracking.MapMark
import io.github.admoreau.paintbynumbers.tracking.XDetector
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.cos
import kotlin.math.sin

/**
 * Where the image sits on the canvas: its centre in map coordinates (roughly mm), its size as
 * map units per image pixel, and its rotation in radians (clockwise, as seen facing the canvas).
 */
data class Placement(val cx: Double, val cy: Double, val scale: Double, val angle: Double) {
    /** Image pixels -> map coordinates, row-major 3x3, for an image [w] x [h] pixels. */
    fun matrix(w: Int, h: Int): DoubleArray {
        val c = scale * cos(angle)
        val s = scale * sin(angle)
        return doubleArrayOf(c, -s, cx - c * w / 2 + s * h / 2, s, c, cy - s * w / 2 - c * h / 2, 0.0, 0.0, 1.0)
    }
}

/** How the app knows where the canvas is. */
enum class TrackingMode { X_MARKS, MARKERS, FREE }

/** User adjustments to the X finder. */
data class DetectorSettings(
    val contrast: Double = 3.0, // how far above the noise a mark must stand out (lower finds fainter marks, and more clutter)
    val minSizePx: Int = 8, // smallest mark, in camera pixels
    val shape: Double? = null, // how X-like a mark must be; null = learned from the user's X's when scanning
) {
    companion object {
        val DEFAULT = DetectorSettings()
    }
}

/** Everything the app remembers about the painting in progress. */
data class Project(
    val mode: TrackingMode = TrackingMode.X_MARKS,
    val polarity: XDetector.Polarity = XDetector.Polarity.DARK,
    val detector: DetectorSettings = DetectorSettings.DEFAULT,
    val marks: List<MapMark> = emptyList(), // X marks map
    val markerMap: List<MapMark> = emptyList(), // square markers map: one entry per marker corner (id = marker id * 4 + corner)
    val markerSizeMm: Double = 40.0, // printed side of the square markers
    val imageFile: String? = null,
    val planDir: String? = null, // an imported plan's folder, when painting from a plan
    val currentLayer: Int = 0,
    val placement: Placement? = null, // on the tracked canvas (X marks)
    val markerPlacement: Placement? = null, // on the tracked canvas (square markers)
    val freePlacement: Placement? = null, // on the screen, in fractions of its width (free mode)
    val minSlender: Double = 0.0, // how X-like a detection must be to count (learned from the user's X's at scan time)
) {
    /** The X-likeness cutoff in use: the user's setting, or what the scan learned. */
    val shapeCutoff: Double get() = detector.shape ?: minSlender
}

/** Saves the project as JSON (and the chosen image as a file) in the app's private storage. */
class ProjectStore(private val context: Context) {
    private val file = File(context.filesDir, "project.json")

    fun load(): Project {
        if (!file.exists()) return Project()
        return try {
            val o = JSONObject(file.readText())
            val det = o.optJSONObject("detector")
            Project(
                mode = runCatching { TrackingMode.valueOf(o.optString("mode", "X_MARKS")) }.getOrDefault(TrackingMode.X_MARKS),
                polarity = XDetector.Polarity.valueOf(o.optString("polarity", "DARK")),
                detector = if (det == null) DetectorSettings.DEFAULT else DetectorSettings(
                    det.optDouble("contrast", 3.0), det.optInt("min_size_px", 8), if (det.has("shape")) det.getDouble("shape") else null,
                ),
                marks = readMarks(o.optJSONArray("marks")),
                markerMap = readMarks(o.optJSONArray("marker_map")),
                markerSizeMm = o.optDouble("marker_size_mm", 40.0),
                imageFile = o.optString("image_file").ifEmpty { null },
                planDir = o.optString("plan_dir").ifEmpty { null },
                currentLayer = o.optInt("current_layer", 0),
                placement = readPlacement(o.optJSONObject("placement")),
                markerPlacement = readPlacement(o.optJSONObject("marker_placement")),
                freePlacement = readPlacement(o.optJSONObject("free_placement")),
                minSlender = o.optDouble("min_slender", 0.0),
            )
        } catch (e: Exception) {
            Project()
        }
    }

    private fun readMarks(a: JSONArray?): List<MapMark> = if (a == null) emptyList() else (0 until a.length()).map {
        val m = a.getJSONObject(it)
        MapMark(m.getInt("id"), m.getDouble("x"), m.getDouble("y"), m.getDouble("d1"), m.getDouble("d2"), m.getDouble("size"))
    }

    private fun writeMarks(ms: List<MapMark>) = JSONArray().also { a ->
        for (m in ms) a.put(JSONObject().put("id", m.id).put("x", m.x).put("y", m.y).put("d1", m.d1).put("d2", m.d2).put("size", m.sizeMm))
    }

    private fun readPlacement(o: JSONObject?) = o?.let { Placement(it.getDouble("cx"), it.getDouble("cy"), it.getDouble("scale"), it.getDouble("angle")) }

    private fun writePlacement(p: Placement) = JSONObject().put("cx", p.cx).put("cy", p.cy).put("scale", p.scale).put("angle", p.angle)

    fun save(p: Project) {
        val det = JSONObject().put("contrast", p.detector.contrast).put("min_size_px", p.detector.minSizePx)
        p.detector.shape?.let { det.put("shape", it) }
        val o = JSONObject()
            .put("mode", p.mode.name)
            .put("polarity", p.polarity.name)
            .put("detector", det)
            .put("marks", writeMarks(p.marks))
            .put("marker_map", writeMarks(p.markerMap))
            .put("marker_size_mm", p.markerSizeMm)
            .put("image_file", p.imageFile ?: "")
            .put("plan_dir", p.planDir ?: "")
            .put("current_layer", p.currentLayer)
            .put("min_slender", p.minSlender)
        p.placement?.let { o.put("placement", writePlacement(it)) }
        p.markerPlacement?.let { o.put("marker_placement", writePlacement(it)) }
        p.freePlacement?.let { o.put("free_placement", writePlacement(it)) }
        val tmp = File(context.filesDir, "project.json.tmp")
        tmp.writeText(o.toString(2))
        tmp.renameTo(file)
    }

    /** Copies a picked image into private storage; returns its file name. */
    fun importImage(uri: Uri): String {
        val name = "image_${System.currentTimeMillis()}"
        context.contentResolver.openInputStream(uri)!!.use { input ->
            File(context.filesDir, name).outputStream().use { input.copyTo(it) }
        }
        return name
    }

    /** Decodes a stored image, downsampled so its longer side is at most [maxSide] px. */
    fun loadImage(name: String, maxSide: Int = 2048): Bitmap? {
        val f = File(context.filesDir, name)
        if (!f.exists()) return null
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(f.path, bounds)
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > maxSide) sample *= 2
        return BitmapFactory.decodeFile(f.path, BitmapFactory.Options().apply { inSampleSize = sample })
    }
}
