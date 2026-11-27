package io.github.admoreau.paintbynumbers.ui

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapShader
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Shader
import android.os.SystemClock
import android.util.Log
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.FilterChip
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import io.github.admoreau.paintbynumbers.align.EdgeModel
import io.github.admoreau.paintbynumbers.align.Lines
import io.github.admoreau.paintbynumbers.align.SketchAligner
import io.github.admoreau.paintbynumbers.camera.Finder
import io.github.admoreau.paintbynumbers.camera.Frame
import io.github.admoreau.paintbynumbers.camera.FrameAnalyzer
import io.github.admoreau.paintbynumbers.plan.Plan
import io.github.admoreau.paintbynumbers.plan.PlanThickness
import io.github.admoreau.paintbynumbers.plan.RegionRenderer
import io.github.admoreau.paintbynumbers.project.DetectorSettings
import io.github.admoreau.paintbynumbers.project.Placement
import io.github.admoreau.paintbynumbers.project.Project
import io.github.admoreau.paintbynumbers.project.TrackingMode
import io.github.admoreau.paintbynumbers.tracking.Locator
import io.github.admoreau.paintbynumbers.tracking.M3
import io.github.admoreau.paintbynumbers.tracking.MapMark
import io.github.admoreau.paintbynumbers.tracking.MarkerTracker
import io.github.admoreau.paintbynumbers.tracking.PoseEstimate
import io.github.admoreau.paintbynumbers.tracking.PoseProjector
import io.github.admoreau.paintbynumbers.tracking.TrackMode
import io.github.admoreau.paintbynumbers.tracking.Tracker
import io.github.admoreau.paintbynumbers.tracking.XDetector
import io.github.admoreau.paintbynumbers.tracking.solveLinear
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import org.opencv.core.Mat
import kotlinx.coroutines.withContext
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sin

/** Where the canvas is in one frame, however it was found. */
private class Anchor(
    val h: DoubleArray?, // canvas map -> buffer pixels; null when not located (or in free mode)
    val status: String,
    val matched: Set<Int> = emptySet(), // detection indices used (X marks)
    val pose: PoseEstimate? = null,
)

/** X marks: runs on the analysis thread, (re)building the tracker for the camera in use. */
private class XTracking(private val marks: List<MapMark>, private val shapeCutoff: Double) {
    private var tracker: Tracker? = null
    private var key: Any? = null
    private var frames = 0
    private val xSize = marks.map { it.sizeMm }.sorted().let { if (it.isEmpty()) 15.0 else it[it.size / 2] }

    fun update(f: Frame): Anchor {
        val intr = f.intrinsics ?: return Anchor(null, "Reading camera details…")
        val k = Triple(f.width, f.height, intr)
        if (tracker == null || key != k) {
            tracker = Tracker(Locator(marks, intr, f.width, f.height, xSizeMm = xSize).apply { coldBudgetMs = 120 })
            key = k
        }
        // only detections nearly as X-like as the user's X's
        val keep = f.dets.indices.filter { f.dets[it].slender >= 0.8 * shapeCutoff }
        val est = tracker!!.update(keep.map { f.dets[it] }) ?: return Anchor(null, "Looking for X's: get at least 4 in view")
        val matched = est.matches.keys.map { keep[it] }.toSet()
        val status = if (est.mode == TrackMode.HOLD) "Holding: no X's matched"
        else "Locked on ${est.matches.size} X's" + if (est.uncertaintyMm.isFinite()) "   ±%.1f mm".format(est.uncertaintyMm) else ""
        return Anchor(est.H, status, matched, est)
    }

    /** Where the next frame should look: around each mapped X in view; everywhere when lost, and every 10th frame. */
    fun regions(a: Anchor, f: Frame): List<XDetector.Region>? {
        val t = tracker ?: return null
        val pose = a.pose
        if (pose == null || pose.mode == TrackMode.HOLD || ++frames % 10 == 0) return null
        val p = PoseProjector(t.locator.cam, pose.x)
        val q = DoubleArray(2)
        val r = DoubleArray(2)
        val out = ArrayList<XDetector.Region>()
        for (m in marks) {
            p.project(m.x, m.y, q)
            if (!q[0].isFinite() || q[0] < -40 || q[0] > f.width + 40 || q[1] < -40 || q[1] > f.height + 40) continue
            p.project(m.x + t.locator.spacing, m.y, r)
            out += XDetector.Region(q[0], q[1], max(40.0, 0.45 * hypot(r[0] - q[0], r[1] - q[1])))
        }
        return out
    }
}

/** Square markers: tracks and maps them as it goes. */
private class MarkerTracking(map: List<MapMark>, sideMm: Double) {
    val tracker = MarkerTracker(map, sideMm)

    fun update(f: Frame): Anchor {
        val intr = f.intrinsics ?: return Anchor(null, "Reading camera details…")
        val r = tracker.update(f.markers, intr) ?: return Anchor(null, "Looking for square markers")
        val status = if (r.holding) "Holding: no markers in view" else "Locked on ${r.used} marker${if (r.used == 1) "" else "s"} (${tracker.markerCount} mapped)"
        return Anchor(r.H, status)
    }
}

private class PaintFrame(val frame: Frame, val anchor: Anchor, val trackMs: Double)

enum class OverlayKind { REGIONS, AFTER, TARGET }

private fun matrixValues(m: Matrix): DoubleArray = FloatArray(9).also { m.getValues(it) }.let { v -> DoubleArray(9) { v[it].toDouble() } }

private fun wrap(a: Double): Double = ((a + PI) % (2 * PI) + 2 * PI) % (2 * PI) - PI

/** The mean of several homographies of nearly the same view (each scaled to unit norm, signs aligned). */
private fun meanHomography(hs: Collection<DoubleArray>): DoubleArray? {
    if (hs.isEmpty()) return null
    val ref = M3.normalized(hs.first())
    val sum = DoubleArray(9)
    for (h in hs) {
        val n = M3.normalized(h)
        val sign = if ((0 until 9).sumOf { n[it] * ref[it] } < 0) -1.0 else 1.0
        for (k in 0 until 9) sum[k] += sign * n[k]
    }
    return M3.normalized(sum)
}

/** Chalk or pencil line width the drawing finder looks for, in camera-frame pixels (1280 x 960 frames, canvas filling the view). */
private const val SNAP_LINE_WIDTH_PX = 2.5

/**
 * The placement (a similarity) closest to homography [h] (image px -> placement space), fitted by
 * least squares over a 5 x 5 grid on a [w] x [hh] image.
 */
private fun fitPlacement(h: DoubleArray, w: Int, hh: Int): Placement? {
    val a = DoubleArray(16)
    val b = DoubleArray(4)
    val p = DoubleArray(2)
    for (j in 0..4) for (i in 0..4) {
        val x = w * i / 4.0
        val y = hh * j / 4.0
        M3.apply(h, x, y, p)
        if (!p[0].isFinite() || !p[1].isFinite()) continue
        // u = c x - s y + tx, v = s x + c y + ty
        val r1 = doubleArrayOf(x, -y, 1.0, 0.0)
        val r2 = doubleArrayOf(y, x, 0.0, 1.0)
        for (r in 0..3) {
            b[r] += r1[r] * p[0] + r2[r] * p[1]
            for (c in 0..3) a[r * 4 + c] += r1[r] * r1[c] + r2[r] * r2[c]
        }
    }
    val t = solveLinear(a, b, 4) ?: return null
    val (c, s, tx, ty) = t
    return Placement(tx + c * w / 2 - s * hh / 2, ty + s * w / 2 + c * hh / 2, hypot(c, s), atan2(s, c))
}

private tailrec fun Context.findActivity(): Activity? = when (this) {
    is Activity -> this
    is ContextWrapper -> baseContext.findActivity()
    else -> null
}

/** The painting view: the image (or the current layer of the plan) laid onto the canvas. */
@Composable
fun PaintScreen(
    project: Project,
    image: Bitmap?,
    plan: Plan?,
    onPlacement: (TrackingMode, Placement) -> Unit,
    onMarkerMap: (List<MapMark>) -> Unit,
    onLayer: (Int) -> Unit,
    onDetector: (DetectorSettings) -> Unit,
    onPolarity: (XDetector.Polarity) -> Unit,
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    val main = remember { ContextCompat.getMainExecutor(context) }
    val mode = project.mode
    var current by remember { mutableStateOf<PaintFrame?>(null) }
    val xTracking = remember { if (mode == TrackingMode.X_MARKS) XTracking(project.marks, project.shapeCutoff) else null }
    val markerTracking = remember { if (mode == TrackingMode.MARKERS) MarkerTracking(project.markerMap, project.markerSizeMm) else null }
    // Lock view (phone on a stand): the last second of canvas positions, averaged into one that then
    // stays put, so a hand or pencil covering X's can't move the overlay
    val recentH = remember { ArrayDeque<DoubleArray>() }
    var lockedH by remember { mutableStateOf<DoubleArray?>(null) } // canvas -> buffer pixels
    // Snap to drawing: where the image was found in the camera frame (image px -> buffer px); like a
    // locked view, it holds while the phone stays still
    var snapped by remember { mutableStateOf<DoubleArray?>(null) }
    var snapping by remember { mutableStateOf(false) }
    var snapNote by remember { mutableStateOf<String?>(null) }
    var undo by remember { mutableStateOf<Triple<Placement?, DoubleArray?, DoubleArray?>?>(null) } // placement, lockedH, snapped before
    val scope = rememberCoroutineScope()
    val analyzer = remember {
        lateinit var a: FrameAnalyzer
        a = FrameAnalyzer { f ->
            val t0 = SystemClock.elapsedRealtimeNanos()
            val anchor = when {
                xTracking != null -> xTracking.update(f).also { a.regions = xTracking.regions(it, f) }
                markerTracking != null -> markerTracking.update(f)
                else -> Anchor(null, "Free mode: the image stays where you put it on the screen")
            }
            val ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
            main.execute {
                current = PaintFrame(f, anchor, ms)
                val h = anchor.h
                when {
                    h == null -> recentH.clear()
                    anchor.pose?.mode != TrackMode.HOLD -> {
                        recentH.addLast(h)
                        while (recentH.size > 15) recentH.removeFirst()
                    }
                }
            }
        }
        a.finder = when (mode) {
            TrackingMode.X_MARKS -> Finder.X_MARKS
            TrackingMode.MARKERS -> Finder.MARKERS
            TrackingMode.FREE -> Finder.NONE
        }
        a
    }
    analyzer.polarity = project.polarity
    analyzer.settings = project.detector
    DisposableEffect(Unit) { onDispose { markerTracking?.let { onMarkerMap(it.tracker.mapMarks()) } } }

    var opacity by remember { mutableFloatStateOf(0.6f) }
    var showMarks by remember { mutableStateOf(mode != TrackingMode.FREE) }
    var moving by remember { mutableStateOf(false) }
    var tuning by remember { mutableStateOf(false) }
    var showLayers by remember { mutableStateOf(plan != null) }
    var hideUi by remember { mutableStateOf(false) } // just the camera and the image; a tap brings the controls back
    var hint by remember { mutableStateOf(false) }
    var placement by remember {
        mutableStateOf(when (mode) { TrackingMode.X_MARKS -> project.placement; TrackingMode.MARKERS -> project.markerPlacement; TrackingMode.FREE -> project.freePlacement })
    }
    var viewSize by remember { mutableStateOf(IntSize.Zero) }
    val latestPlacement by rememberUpdatedState(placement)
    BackHandler(onBack = onBack)
    BackHandler(enabled = hideUi) { hideUi = false }
    DisposableEffect(Unit) { onDispose { latestPlacement?.let { onPlacement(mode, it) } } }

    // painting hands are busy: don't let the screen time out while this view is open
    val view = LocalView.current
    DisposableEffect(view) {
        view.keepScreenOn = true
        onDispose { view.keepScreenOn = false }
    }

    // with the controls hidden, hide Android's status and navigation bars too (a swipe from the edge shows them briefly)
    val window = remember(context) { context.findActivity()?.window }
    DisposableEffect(hideUi) {
        val bars = window?.let { WindowCompat.getInsetsController(it, it.decorView) }
        if (hideUi) {
            bars?.systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            bars?.hide(WindowInsetsCompat.Type.systemBars())
        }
        onDispose { bars?.show(WindowInsetsCompat.Type.systemBars()) }
    }
    LaunchedEffect(hideUi) {
        hint = hideUi
        if (hideUi) { delay(2500); hint = false }
    }
    fun hideControls() {
        if (moving) { moving = false; placement?.let { onPlacement(mode, it) } }
        hideUi = true
    }

    // the plan's current layer and what to show of it
    var layerIndex by remember { mutableIntStateOf(project.currentLayer.coerceIn(0, max(0, (plan?.layers?.size ?: 1) - 1))) }
    var thickness by remember { mutableStateOf<PlanThickness?>(null) }
    var kind by remember { mutableStateOf(if (plan != null) OverlayKind.REGIONS else OverlayKind.TARGET) }
    val layer = plan?.layers?.getOrNull(layerIndex)
    // what to lay on the canvas, and (for regions) their outlines, drawn solid on top whatever the opacity
    val drawn by produceState<Pair<Bitmap?, Bitmap?>>(null to null, image, plan, layerIndex, thickness, kind) {
        value = withContext(Dispatchers.Default) {
            when {
                plan == null || layer == null || kind == OverlayKind.TARGET -> image to null
                kind == OverlayKind.AFTER -> (layer.stepFile?.let { BitmapFactory.decodeFile(plan.file(it).path) } ?: image) to null
                else -> RegionRenderer.render(plan, layer, thickness)?.let { it.fill to it.lines } ?: (image to null)
            }
        }
    }
    val overlay = drawn.first
    val outlines = drawn.second
    fun shaderOf(b: Bitmap) = Paint().apply { shader = BitmapShader(b, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP); isFilterBitmap = true; isAntiAlias = true }
    val shaderPaint = remember(overlay) { overlay?.let { shaderOf(it) } }
    val linePaint = remember(outlines) { outlines?.let { shaderOf(it) } }

    /** Placement space -> view pixels for this frame: the tracked (or locked) canvas, or the screen itself in free mode. */
    fun anchorToView(pf: PaintFrame?): DoubleArray? {
        if (mode == TrackingMode.FREE) return if (viewSize == IntSize.Zero) null else doubleArrayOf(viewSize.width.toDouble(), 0.0, 0.0, 0.0, viewSize.width.toDouble(), 0.0, 0.0, 0.0, 1.0)
        // locked: frozen in camera pixels, so it stays right when the preview is re-laid out (e.g. Hide)
        val h = lockedH ?: pf?.anchor?.h ?: return null
        val b2v = pf?.frame?.bufferToView ?: return null
        return M3.mul(matrixValues(b2v), h)
    }

    /** The placement-space point under a screen position (null above the canvas horizon). */
    fun toPlace(g: DoubleArray, x: Float, y: Float): DoubleArray? {
        val gi = M3.inv(g) ?: return null
        val q = M3.apply(gi, x.toDouble(), y.toDouble())
        if (!q[0].isFinite() || g[6] * q[0] + g[7] * q[1] + g[8] <= 0) return null
        return q
    }

    /** The image centred on screen, upright, 60% (tracked) or 80% (free) of the screen wide. */
    fun autoPlace(g: DoubleArray, img: Bitmap): Placement? {
        val cx = viewSize.width / 2f
        val cy = viewSize.height / 2f
        val half = (if (mode == TrackingMode.FREE) 0.4f else 0.3f) * viewSize.width
        val c = toPlace(g, cx, cy) ?: return null
        val l = toPlace(g, cx - half, cy) ?: return null
        val r = toPlace(g, cx + half, cy) ?: return null
        return Placement(c[0], c[1], hypot(r[0] - l[0], r[1] - l[1]) / img.width, atan2(r[1] - l[1], r[0] - l[0]))
    }

    // the image as the aligner sees it (its edges, by direction): built on the first snap, kept for the next
    val edgeModel = remember(image) { lazy { image?.let { EdgeModel.fromBitmap(it) } } }
    DisposableEffect(edgeModel) { onDispose { if (edgeModel.isInitialized()) edgeModel.value?.release() } }

    /** Find the image in the camera's view of the drawing, and hold it there. */
    fun snapToDrawing() {
        val pf = current ?: return
        val img = image ?: return
        val bv = pf.frame.bufferToView?.let { matrixValues(it) } ?: return
        val vb = M3.inv(bv) ?: return
        // the rough start: where the image is drawn now (image px -> buffer px)
        val drawn = snapped ?: run {
            val g = anchorToView(pf)
            val pl = placement
            if (g != null && pl != null) M3.mul(vb, M3.mul(g, pl.matrix(img.width, img.height))) else null
        }
        val upAngle = -atan2(bv[3], bv[0]) // the image upright on screen, as a turn in the camera frame
        val light = project.polarity == XDetector.Polarity.LIGHT
        snapping = true
        snapNote = "Looking for your drawing… keep hands out of view."
        scope.launch {
            val frame = CompletableDeferred<Mat>()
            analyzer.grab = { m -> if (!frame.complete(m)) m.release() }
            val gray = withTimeoutOrNull(3000) { frame.await() }
            if (gray == null) {
                analyzer.grab = null
                frame.cancel()
                snapping = false
                snapNote = "The camera isn't running: try again."
                return@launch
            }
            val found = withContext(Dispatchers.Default) {
                try {
                    val model = edgeModel.value ?: return@withContext null
                    val f = model.f
                    val toModel = doubleArrayOf(f, 0.0, 0.5 * f - 0.5, 0.0, f, 0.5 * f - 0.5, 0.0, 0.0, 1.0) // image px -> model px
                    val start = drawn?.let { M3.inv(it) }?.let { M3.mul(toModel, it) } // buffer px -> model px
                    val t0 = SystemClock.elapsedRealtime()
                    val lines = Lines.sketch(gray, light, SNAP_LINE_WIDTH_PX)
                    val r = SketchAligner.align(model, lines, gray.cols(), gray.rows(), start, upAngle) ?: return@withContext null
                    Log.i("Snap", "${lines.size} line points, ${r.millis}, total ${SystemClock.elapsedRealtime() - t0} ms, score %.3f, next %.3f".format(r.score, r.secondScore))
                    M3.inv(r.G)?.let { M3.mul(it, toModel) } // image px -> buffer px
                } catch (e: Exception) {
                    Log.w("Snap", "snap failed", e)
                    null
                } finally {
                    gray.release()
                }
            }
            snapping = false
            if (found == null) {
                snapNote = "Couldn't find your drawing. Check it's in view with nothing in front of it, and that Light/Dark lines matches it."
                return@launch
            }
            undo = Triple(placement, lockedH, snapped)
            snapped = found
            lockedH = null
            moving = false // the snap places the image; finger gestures would only move a placement that isn't shown
            // move the placement there too, so the image stays put after letting go (and tracking, if any, carries on from there)
            val now = current
            val g = anchorToView(now)
            val b2vNow = now?.frame?.bufferToView
            if (g != null && b2vNow != null) {
                M3.inv(g)?.let { gi -> fitPlacement(M3.mul(gi, M3.mul(matrixValues(b2vNow), found)), img.width, img.height) }
                    ?.let { placement = it; onPlacement(mode, it) }
            }
            snapNote = "Snapped to your drawing. Not lined up? Tap Undo, or move the image closer and snap again."
        }
    }

    fun undoSnap() {
        undo?.let { (pl, lh, sn) ->
            placement = pl
            pl?.let { onPlacement(mode, it) }
            lockedH = lh
            snapped = sn
        }
        undo = null
        snapNote = null
    }
    LaunchedEffect(snapNote, snapping) {
        if (snapNote != null && !snapping) { delay(15000); snapNote = null; undo = null }
    }

    // no placement yet: put the image where the phone points (or mid-screen), ready to adjust
    LaunchedEffect(current, viewSize) {
        if (placement != null || image == null || viewSize == IntSize.Zero) return@LaunchedEffect
        val g = anchorToView(current) ?: return@LaunchedEffect
        autoPlace(g, image)?.let { placement = it; moving = true }
    }

    Box(Modifier.fillMaxSize().onSizeChanged { viewSize = it }) {
        CameraView(analyzer, Modifier.fillMaxSize())
        val gestures = if (!moving) Modifier.pointerInput("tap") { detectTapGestures(onTap = { hideUi = !hideUi }) } else Modifier.pointerInput("move") {
            detectTransformGestures { centroid, pan, zoom, rotationDeg ->
                val g = anchorToView(current) ?: return@detectTransformGestures
                val pl = placement ?: return@detectTransformGestures
                val p0 = toPlace(g, centroid.x - pan.x, centroid.y - pan.y) ?: return@detectTransformGestures
                val p1 = toPlace(g, centroid.x, centroid.y) ?: return@detectTransformGestures
                // the screen rotation, measured in placement space (handles perspective and orientation)
                val r = Math.toRadians(rotationDeg.toDouble())
                val v = 80f
                val a0 = toPlace(g, centroid.x + v, centroid.y) ?: return@detectTransformGestures
                val a1 = toPlace(g, centroid.x + v * cos(r).toFloat(), centroid.y + v * sin(r).toFloat()) ?: return@detectTransformGestures
                val dTheta = wrap(atan2(a1[1] - p1[1], a1[0] - p1[0]) - atan2(a0[1] - p1[1], a0[0] - p1[0]))
                // move by the pan, then scale and turn about the fingers
                val dx = pl.cx - p0[0]
                val dy = pl.cy - p0[1]
                val c = cos(dTheta) * zoom
                val s = sin(dTheta) * zoom
                placement = Placement(p1[0] + c * dx - s * dy, p1[1] + s * dx + c * dy, pl.scale * zoom, pl.angle + dTheta)
            }
        }
        Canvas(Modifier.fillMaxSize().then(gestures)) {
            val pf = current
            val g = anchorToView(pf)
            val pl = placement
            val ov = overlay
            // target image pixels -> view: snapped to the drawing (through the camera frame), or via the placement
            val sn = snapped
            val b2v = pf?.frame?.bufferToView
            val imageToView = when {
                sn != null && b2v != null -> M3.mul(matrixValues(b2v), sn)
                g != null && pl != null && image != null -> M3.mul(g, pl.matrix(image.width, image.height))
                else -> null
            }
            if (imageToView != null && image != null && ov != null && shaderPaint != null) {
                shaderPaint.alpha = (opacity * 255).toInt()
                // overlay pixels -> target image pixels -> view
                val toBase = doubleArrayOf(image.width.toDouble() / ov.width, 0.0, 0.0, 0.0, image.height.toDouble() / ov.height, 0.0, 0.0, 0.0, 1.0)
                drawWarped(ov, shaderPaint, M3.mul(imageToView, toBase))
                val ol = outlines
                if (ol != null && linePaint != null && ol.width == ov.width && ol.height == ov.height) drawWarped(ol, linePaint, M3.mul(imageToView, toBase))
            }
            if (showMarks && !hideUi && pf != null) {
                if (mode == TrackingMode.X_MARKS) {
                    drawDetections(pf.frame) { i -> if (i in pf.anchor.matched) Color(0xFF00E676) else Color(0xFF9E9E9E) }
                    // where the overlay thinks the X's are: if the phone gets bumped while locked, these drift off the real X's
                    (lockedH ?: pf.anchor.h)?.let { drawMapMarks(project.marks, it, pf.frame) }
                } else if (mode == TrackingMode.MARKERS) {
                    drawMarkers(pf.frame)
                }
            }
        }

        if (hideUi) {
            if (hint) Text(
                "Tap anywhere to show the controls",
                color = Color.White,
                modifier = Modifier.align(Alignment.BottomCenter).safeDrawingPadding().padding(bottom = 32.dp)
                    .background(Color(0x99000000), RoundedCornerShape(8.dp)).padding(horizontal = 12.dp, vertical = 6.dp),
            )
        } else {
            // ---- top: status and tuning
            val pf = current
            Column(Modifier.align(Alignment.TopStart).fillMaxWidth().safeDrawingPadding()) {
                Column(Modifier.background(Color(0x99000000)).padding(10.dp)) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        // the X's and the drawing are usually the same chalk or pencil, so one setting serves both
                        MarkColorChips(project.polarity, onPolarity, if (mode == TrackingMode.X_MARKS) "marks" else "lines")
                        if (mode == TrackingMode.X_MARKS) FilterChip(selected = tuning, onClick = { tuning = !tuning }, label = { Text("Tune") })
                        FilterChip(selected = false, onClick = { hideControls() }, label = { Text("Hide") })
                    }
                    when {
                        snapped != null -> Text("Snapped to your drawing and locked: keep the phone still. If it moves, tap Snap again.", color = Color(0xFF80D8FF))
                        lockedH != null -> Text("View locked: keep the phone still. If it moves, tap Locked, then Lock view again.", color = Color(0xFF80D8FF))
                        else -> Text(pf?.anchor?.status ?: "Starting camera…", color = Color.White)
                    }
                    snapNote?.let { note ->
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(note, color = Color(0xFFFFD54F), modifier = Modifier.weight(1f, fill = false))
                            if (undo != null && !snapping) TextButton(onClick = { undoSnap() }) { Text("Undo") }
                        }
                    }
                    if (pf != null && mode != TrackingMode.FREE) {
                        Text("%.0f fps   detect %.0f ms   track %.0f ms".format(pf.frame.fps, pf.frame.detectMs, pf.trackMs), color = Color(0xFFBDBDBD))
                    }
                    if (moving) Text("Drag to move, pinch to resize, twist to turn. Tap Move image again to lock it.", color = Color(0xFFFFD54F))
                }
                if (tuning && mode == TrackingMode.X_MARKS) DetectorTuning(project.detector, project.minSlender, onDetector)
            }

            // ---- bottom: layers, overlay and placement controls
            Column(
                Modifier.align(Alignment.BottomCenter).fillMaxWidth().safeDrawingPadding().background(Color(0xAA000000)).padding(horizontal = 12.dp, vertical = 6.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                if (plan != null && layer != null && showLayers) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        TextButton(onClick = { if (layerIndex > 0) { layerIndex--; thickness = null; onLayer(layerIndex) } }, enabled = layerIndex > 0) { Text("◀") }
                        Box(Modifier.size(18.dp).clip(CircleShape).background(Color(android.graphics.Color.parseColor(layer.fullStrengthHex))))
                        Text(
                            "  Layer ${layerIndex + 1} of ${plan.layers.size}: ${layer.paint}",
                            color = Color.White,
                            modifier = Modifier.weight(1f),
                        )
                        TextButton(onClick = { if (layerIndex < plan.layers.size - 1) { layerIndex++; thickness = null; onLayer(layerIndex) } }, enabled = layerIndex < plan.layers.size - 1) { Text("▶") }
                    }
                    // a layer painted at one weight (planned with --layers) has nothing to pick between
                    val single = layer.thicknesses.singleOrNull()
                    if (single == null) Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        FilterChip(selected = thickness == null, onClick = { thickness = null }, label = { Text("All") })
                        for (t in layer.thicknesses) {
                            FilterChip(selected = thickness == t, onClick = { thickness = t; kind = OverlayKind.REGIONS }, label = { Text(t.label.removeSuffix(" coats").removeSuffix(" coat")) }, leadingIcon = {
                                Box(Modifier.size(14.dp).clip(CircleShape).background(Color(android.graphics.Color.parseColor(t.colorHex))).border(1.dp, Color(0x99FFFFFF), CircleShape))
                            })
                        }
                    }
                    val t = thickness ?: single
                    LayerSwatch(layer, t, plan.backgroundHex, chipHeight = 28.dp)
                    Text(
                        if (t != null) "${t.label}: ${t.how}. %.0f%% of the canvas.".format(100 * t.area) +
                            (if (single != null && layerIndex == 0 && plan.groundPaint != null) " Ground coat first: ${plan.groundPaint}." else "")
                        else "${layer.transparency}, ${layer.thicknesses.size} thickness${if (layer.thicknesses.size == 1) "" else "es"}" +
                            (if (layerIndex == 0 && plan.groundPaint != null) ". Ground coat first: ${plan.groundPaint}" else "") + ". Pick one to see where it goes.",
                        color = Color(0xFFE0E0E0),
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        FilterChip(selected = kind == OverlayKind.REGIONS, onClick = { kind = OverlayKind.REGIONS }, label = { Text("Regions") })
                        FilterChip(selected = kind == OverlayKind.AFTER, onClick = { kind = OverlayKind.AFTER }, label = { Text("After layer") })
                        FilterChip(selected = kind == OverlayKind.TARGET, onClick = { kind = OverlayKind.TARGET }, label = { Text("Final image") })
                    }
                }
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Opacity", color = Color.White, modifier = Modifier.width(60.dp))
                    Slider(value = opacity, onValueChange = { opacity = it }, modifier = Modifier.weight(1f))
                }
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                    FilterChip(
                        selected = snapping,
                        enabled = !snapping && image != null && pf?.frame?.bufferToView != null,
                        onClick = { snapToDrawing() },
                        label = { Text(if (snapping) "Snapping…" else "Snap to drawing") },
                    )
                    val locked = lockedH != null || snapped != null
                    if (mode != TrackingMode.FREE || locked) FilterChip(
                        selected = locked,
                        enabled = locked || pf?.anchor?.h != null,
                        onClick = {
                            if (locked) { lockedH = null; snapped = null } else lockedH = meanHomography(recentH) ?: pf?.anchor?.h
                        },
                        label = { Text(if (locked) "Locked" else "Lock view") },
                    )
                    FilterChip(selected = moving, enabled = image != null, onClick = {
                        moving = !moving
                        if (moving) snapped = null // hand placement takes over (the snap already moved the placement there)
                        if (!moving) placement?.let { onPlacement(mode, it) }
                    }, label = { Text("Move image") })
                    if (moving) {
                        OutlinedButton(onClick = {
                            val g = anchorToView(current)
                            if (g != null && image != null) autoPlace(g, image)?.let { placement = it }
                        }) { Text("Center") }
                    }
                    if (plan != null) FilterChip(selected = showLayers, onClick = { showLayers = !showLayers }, label = { Text("Layers") })
                    if (mode != TrackingMode.FREE) FilterChip(selected = showMarks, onClick = { showMarks = !showMarks }, label = { Text(if (mode == TrackingMode.MARKERS) "Show markers" else "Show X's") })
                }
                if (image == null) Text("No image: import a plan or choose an image on the setup screen.", color = Color.White)
            }
        }
    }
}

/**
 * Draws [bitmap] through [m] (bitmap pixels -> view pixels, w > 0 where in front of the camera). The
 * outline is clipped to the part in front of the camera, then filled through a perspective shader,
 * so extreme tilts never produce garbage.
 */
private fun DrawScope.drawWarped(bitmap: Bitmap, paint: Paint, m: DoubleArray) {
    fun w(x: Double, y: Double) = m[6] * x + m[7] * y + m[8]
    val bw = bitmap.width.toDouble()
    val bh = bitmap.height.toDouble()
    val wCenter = w(bw / 2, bh / 2)
    if (wCenter <= 0) return
    val eps = 0.02 * wCenter
    val rect = listOf(doubleArrayOf(0.0, 0.0), doubleArrayOf(bw, 0.0), doubleArrayOf(bw, bh), doubleArrayOf(0.0, bh))
    val poly = ArrayList<DoubleArray>()
    for (i in rect.indices) { // Sutherland-Hodgman against w > eps
        val a = rect[i]
        val b = rect[(i + 1) % rect.size]
        val wa = w(a[0], a[1]) - eps
        val wb = w(b[0], b[1]) - eps
        if (wa > 0) poly += a
        if ((wa > 0) != (wb > 0)) {
            val t = wa / (wa - wb)
            poly += doubleArrayOf(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        }
    }
    if (poly.size < 3) return
    val path = Path()
    for ((i, p) in poly.withIndex()) {
        val q = M3.apply(m, p[0], p[1])
        if (i == 0) path.moveTo(q[0].toFloat(), q[1].toFloat()) else path.lineTo(q[0].toFloat(), q[1].toFloat())
    }
    path.close()
    val mn = M3.normalized(m) // positive scale keeps w > 0 in front
    (paint.shader as BitmapShader).setLocalMatrix(Matrix().apply { setValues(FloatArray(9) { mn[it].toFloat() }) })
    drawIntoCanvas { it.nativeCanvas.drawPath(path, paint) }
}

/** Where the map says each X should be, as small crosses: shows how well the overlay is registered. */
private fun DrawScope.drawMapMarks(marks: List<MapMark>, h: DoubleArray, frame: Frame) {
    val p = FloatArray(2)
    for (mk in marks) {
        if (h[6] * mk.x + h[7] * mk.y + h[8] <= 0) continue // behind the camera
        val q = M3.apply(h, mk.x, mk.y)
        if (!q[0].isFinite() || q[0] < -50 || q[0] > frame.width + 50 || q[1] < -50 || q[1] > frame.height + 50) continue
        frame.toView(q[0], q[1], p)
        val c = Offset(p[0], p[1])
        drawLine(Color(0xFF18FFFF), c - Offset(12f, 0f), c + Offset(12f, 0f), strokeWidth = 3f)
        drawLine(Color(0xFF18FFFF), c - Offset(0f, 12f), c + Offset(0f, 12f), strokeWidth = 3f)
    }
}

/** Outlines of the square markers found in the frame. */
private fun DrawScope.drawMarkers(frame: Frame) {
    val a = FloatArray(2)
    val b = FloatArray(2)
    for (m in frame.markers) {
        for (k in 0 until 4) {
            val j = (k + 1) % 4
            frame.toView(m.corners[2 * k], m.corners[2 * k + 1], a)
            frame.toView(m.corners[2 * j], m.corners[2 * j + 1], b)
            drawLine(if (k == 0) Color(0xFFFFD600) else Color(0xFF00E676), Offset(a[0], a[1]), Offset(b[0], b[1]), strokeWidth = 4f)
        }
    }
}
