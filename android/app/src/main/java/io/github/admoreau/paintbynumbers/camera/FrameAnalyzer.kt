package io.github.admoreau.paintbynumbers.camera

import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import io.github.admoreau.paintbynumbers.project.DetectorSettings
import io.github.admoreau.paintbynumbers.tracking.Intrinsics
import io.github.admoreau.paintbynumbers.tracking.MarkerDetection
import io.github.admoreau.paintbynumbers.tracking.MarkerFinder
import io.github.admoreau.paintbynumbers.tracking.XDetection
import io.github.admoreau.paintbynumbers.tracking.XDetector
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import java.io.File

/** One analysed frame. Detections are in buffer pixels; [bufferToView] maps them onto the preview. */
class Frame(
    val dets: List<XDetection>,
    val markers: List<MarkerDetection>,
    val width: Int,
    val height: Int,
    val bufferToView: Matrix?,
    val intrinsics: Intrinsics?,
    val detectMs: Double,
    val fps: Double,
) {
    fun toView(x: Double, y: Double, out: FloatArray = FloatArray(2)): FloatArray {
        out[0] = x.toFloat(); out[1] = y.toFloat()
        bufferToView?.mapPoints(out)
        return out
    }
}

/**
 * Development aid: creating `files/dump_request` in the app's storage (adb shell run-as ... touch)
 * makes the next frame be saved there as `frame_<n>.png`, the raw luminance the detector sees.
 */
object FrameDump {
    @Volatile var dir: File? = null
    private var counter = 0
    private var checks = 0

    fun maybeSave(gray: Mat) {
        val d = dir ?: return
        if (++checks % 10 != 0) return
        val req = File(d, "dump_request")
        if (!req.exists()) return
        req.delete()
        Imgcodecs.imwrite(File(d, "frame_${counter++}.png").path, gray)
    }
}

/** What to look for in each frame. */
enum class Finder { X_MARKS, MARKERS, NONE }

/**
 * Finds X marks (or square markers) in the camera's luminance plane and hands each frame to
 * [onFrame], on the analysis thread (so tracking can run there too). CameraX supplies the
 * transforms that line overlays up with the on-screen preview on any phone, resolution and
 * orientation.
 */
class FrameAnalyzer(private val onFrame: (Frame) -> Unit) : ImageAnalysis.Analyzer {
    @Volatile var polarity = XDetector.Polarity.DARK
    @Volatile var settings = DetectorSettings.DEFAULT
    @Volatile var finder = Finder.X_MARKS
    @Volatile var sensorIntrinsics: SensorIntrinsics? = null
    /** Where to look for X's in the next frame (null = everywhere); set by tracking. */
    @Volatile var regions: List<XDetector.Region>? = null

    private val detector = XDetector()
    private val markerFinder by lazy { MarkerFinder() }
    private var sensorToView: Matrix? = null
    private var lastFrameNs = 0L
    private var fps = 0.0

    override fun getTargetCoordinateSystem(): Int = ImageAnalysis.COORDINATE_SYSTEM_VIEW_REFERENCED

    override fun updateTransform(matrix: Matrix?) {
        sensorToView = matrix?.let { Matrix(it) }
    }

    override fun analyze(image: ImageProxy) {
        try {
            val plane = image.planes[0] // Y (luminance): all the detectors need
            val gray = Mat(image.height, image.width, CvType.CV_8UC1, plane.buffer, plane.rowStride.toLong())
            val t0 = SystemClock.elapsedRealtimeNanos()
            var dets = emptyList<XDetection>()
            var markers = emptyList<MarkerDetection>()
            when (finder) {
                Finder.X_MARKS -> {
                    val s = settings
                    detector.polarity = polarity
                    detector.low = s.contrast
                    detector.high = 2 * s.contrast
                    detector.minExtent = s.minSizePx
                    dets = detector.detect(gray, regions)
                }
                Finder.MARKERS -> markers = markerFinder.find(gray)
                Finder.NONE -> {}
            }
            val t1 = SystemClock.elapsedRealtimeNanos()
            FrameDump.maybeSave(gray)
            gray.release()

            if (lastFrameNs > 0) {
                val instant = 1e9 / (t1 - lastFrameNs)
                fps = if (fps == 0.0) instant else 0.9 * fps + 0.1 * instant
            }
            lastFrameNs = t1

            val s2b = image.imageInfo.sensorToBufferTransformMatrix
            val toView = sensorToView?.let { s2v ->
                Matrix().also { m -> s2b.invert(m); m.postConcat(s2v) } // buffer -> sensor -> view
            }
            val intr = sensorIntrinsics?.toBuffer(s2b)
            onFrame(Frame(dets, markers, image.width, image.height, toView, intr, (t1 - t0) / 1e6, fps))
        } finally {
            image.close()
        }
    }
}
