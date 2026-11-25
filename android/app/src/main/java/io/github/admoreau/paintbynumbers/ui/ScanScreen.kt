package io.github.admoreau.paintbynumbers.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import io.github.admoreau.paintbynumbers.camera.Frame
import io.github.admoreau.paintbynumbers.camera.FrameAnalyzer
import io.github.admoreau.paintbynumbers.project.DetectorSettings
import io.github.admoreau.paintbynumbers.project.Project
import io.github.admoreau.paintbynumbers.tracking.BundleAdjust
import io.github.admoreau.paintbynumbers.tracking.Intrinsics
import io.github.admoreau.paintbynumbers.tracking.MapMark
import io.github.admoreau.paintbynumbers.tracking.ScanSession
import io.github.admoreau.paintbynumbers.tracking.XDetection
import io.github.admoreau.paintbynumbers.tracking.XDetector
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.hypot
import kotlin.math.max

private val MARK_GREEN = Color(0xFF00E676)
private val MARK_GRAY = Color(0xFF9E9E9E)
private val MARK_PINK = Color(0xFFFF4081)
private val TEXT_DIM = Color(0xFFBDBDBD)
private val TEXT_WARN = Color(0xFFFFD54F)
private val TEXT_BAD = Color(0xFFFF8A80)

/** X's that must be in view to start: the scan seeds its map from at least this many. */
private const val NEED_IN_VIEW = 6
/** Views wanted before the scan finishes by itself (it also needs every side covered). */
private const val WANT_VIEWS = 10
/** How far the view must move toward a side to count it covered, as a fraction of the camera image's diagonal. */
private const val SIDE_REACH = 0.12f
/** Mean brightness (0-255) below which the camera is taken to be covered. */
private const val DARK_FRAME = 15.0

/** Draws each detection in [frame] as a circle on the preview. */
fun DrawScope.drawDetections(frame: Frame, color: (Int) -> Color = { MARK_GREEN }) {
    val p = FloatArray(2)
    val r = FloatArray(2)
    for ((i, d) in frame.dets.withIndex()) {
        frame.toView(d.x, d.y, p)
        r[0] = (d.size / 2 + 6).toFloat(); r[1] = 0f
        frame.bufferToView?.mapVectors(r)
        drawCircle(color(i), radius = hypot(r[0], r[1]), center = Offset(p[0], p[1]), style = Stroke(width = 4f))
        drawCircle(MARK_PINK, radius = 5f, center = Offset(p[0], p[1]))
    }
}

/** Dark / light mark switch, shown over the camera so the effect is visible at once. */
@Composable
fun MarkColorChips(polarity: XDetector.Polarity, onPolarity: (XDetector.Polarity) -> Unit, noun: String = "marks") {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        FilterChip(selected = polarity == XDetector.Polarity.DARK, onClick = { onPolarity(XDetector.Polarity.DARK) }, label = { Text("Dark $noun") })
        FilterChip(selected = polarity == XDetector.Polarity.LIGHT, onClick = { onPolarity(XDetector.Polarity.LIGHT) }, label = { Text("Light $noun") })
    }
}

/** Owns the scan, shared between the analysis thread (updates) and the UI (start, finish). */
private class ScanHolder {
    private var scanning = false
    private var session: ScanSession? = null
    private var intrinsics: Intrinsics? = null
    private var minSlender = 0.0
    private var polarity = XDetector.Polarity.DARK

    @Synchronized
    fun update(f: Frame): ScanSession.Status? {
        intrinsics = f.intrinsics ?: intrinsics
        if (!scanning || f.polarity != polarity) return null // a frame looked at with the other mark colour, from before the start
        val s = session ?: ScanSession(f.width, f.height, minSlender).also { session = it }
        return s.update(f.dets)
    }

    @Synchronized
    fun start(minSlender: Double, polarity: XDetector.Polarity) {
        this.minSlender = minSlender
        this.polarity = polarity
        session = null
        scanning = true
    }

    @Synchronized
    fun stop(): Pair<ScanSession?, Intrinsics?> {
        scanning = false
        return session to intrinsics
    }
}

private enum class Phase { FIND, MOVE, SOLVING, DONE }

/** The four ways to move the phone during the scan, in the order of [Coverage.reach]. */
private enum class Side { LEFT, RIGHT, UP, DOWN }

/** How far the view has moved toward each [Side] since the scan started, as fractions of the image diagonal. */
private class Coverage(val reach: List<Float> = List(4) { 0f }) {
    fun done(side: Side) = reach[side.ordinal] >= SIDE_REACH
    val sidesDone get() = Side.entries.count { done(it) }

    /** Folds in a frame: [center] is where the first view's centre now is (buffer pixels). */
    fun plus(frame: Frame, center: DoubleArray): Coverage {
        val c = frame.toView(center[0], center[1])
        val mid = frame.toView(frame.width / 2.0, frame.height / 2.0)
        val a = frame.toView(0.0, 0.0)
        val b = frame.toView(frame.width.toDouble(), frame.height.toDouble())
        val diag = hypot(b[0] - a[0], b[1] - a[1]).takeIf { it > 0 } ?: return this
        // the picture moves the opposite way to the phone: phone left -> canvas slides right in view
        val dx = (c[0] - mid[0]) / diag
        val dy = (c[1] - mid[1]) / diag
        val r = reach
        return Coverage(listOf(max(r[0], dx), max(r[1], -dx), max(r[2], dy), max(r[3], -dy)))
    }
}

private class ScanFrame(val frame: Frame, val status: ScanSession.Status?)

/** How clearly a frame shows hand-drawn X's: used to tell whether the canvas has dark or light marks. */
private fun xScore(dets: List<XDetection>): Double = dets.count { it.slender >= 5.0 } + 0.001 * dets.sumOf { it.slender.coerceAtMost(20.0) }

/**
 * Scan the canvas, in two steps:
 * 1. Find: point at the canvas until at least [NEED_IN_VIEW] X's have green circles, then tap Start.
 *    Dark or light marks are worked out automatically (the finder tries both while you aim).
 * 2. Move: slide the phone toward each arrow on screen; the scan finishes by itself once every
 *    side is covered and there are enough views, and the map is worked out.
 */
@Composable
fun ScanScreen(
    project: Project,
    onPolarity: (XDetector.Polarity) -> Unit,
    onDetector: (DetectorSettings) -> Unit,
    onSave: (List<MapMark>, Double) -> Unit,
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    val main = remember { ContextCompat.getMainExecutor(context) }
    val scope = rememberCoroutineScope()
    val holder = remember { ScanHolder() }
    var current by remember { mutableStateOf<ScanFrame?>(null) }
    var phase by remember { mutableStateOf(Phase.FIND) }
    var result by remember { mutableStateOf<BundleAdjust.Result?>(null) }
    var minSlender by remember { mutableStateOf(0.0) }
    var tuning by remember { mutableStateOf(false) }
    var autoColor by remember { mutableStateOf(true) }
    val colorScore = remember { DoubleArray(2) } // recent xScore per mark colour, while finding
    var bestColor by remember { mutableStateOf(project.polarity) }
    var scanColor by remember { mutableStateOf(project.polarity) }
    var coverage by remember { mutableStateOf(Coverage()) }
    val color = if (phase == Phase.FIND) (if (autoColor) bestColor else project.polarity) else scanColor

    // the X-likeness cutoff: the user's setting, else the strictest that still leaves 6 X's in view
    fun cutoff(dets: List<XDetection>) = project.detector.shape ?: ScanSession.chooseMinSlender(dets, NEED_IN_VIEW)

    fun finish() {
        if (phase != Phase.MOVE) return
        val (session, intr) = holder.stop()
        phase = Phase.SOLVING
        scope.launch {
            result = if (session == null || intr == null) null else withContext(Dispatchers.Default) {
                BundleAdjust.run(session.keyframes, session.refPositions, intr, keep = session.reliableIds())
            }
            phase = Phase.DONE
        }
    }

    fun onFrame(f: Frame, st: ScanSession.Status?) {
        if (phase == Phase.FIND && autoColor) {
            val i = f.polarity.ordinal
            colorScore[i] = 0.7 * colorScore[i] + 0.3 * xScore(f.dets)
            val other = 1 - i
            if (colorScore[i] > colorScore[other] + 0.5) bestColor = f.polarity // a margin, so the choice doesn't flicker
            if (f.polarity != bestColor) return // only show what the chosen colour finds
        }
        current = ScanFrame(f, st)
        if (phase == Phase.MOVE && st?.startCenter != null) {
            coverage = coverage.plus(f, st.startCenter)
            val views = st.keyframes
            if ((views >= WANT_VIEWS && coverage.sidesDone == 4) || views >= ScanSession.MAX_KEYFRAMES) finish()
        }
    }

    val analyzer = remember {
        FrameAnalyzer { f ->
            val st = holder.update(f)
            main.execute { onFrame(f, st) }
        }
    }
    analyzer.bothPolarities = phase == Phase.FIND && autoColor
    analyzer.polarity = color
    analyzer.settings = project.detector
    BackHandler {
        holder.stop()
        onBack()
    }

    fun start() {
        val dets = current?.frame?.dets ?: return
        if (autoColor) onPolarity(color) // remembered for painting too
        scanColor = color
        minSlender = cutoff(dets)
        coverage = Coverage()
        holder.start(minSlender, color)
        phase = Phase.MOVE
    }

    val frame = current?.frame
    val dets = frame?.dets ?: emptyList()
    val findCut = cutoff(dets)
    val green = dets.count { it.slender >= findCut }
    val gray = dets.size - green

    Box(Modifier.fillMaxSize()) {
        CameraView(analyzer, Modifier.fillMaxSize())
        Canvas(Modifier.fillMaxSize()) {
            val sf = current ?: return@Canvas
            if (phase == Phase.FIND || phase == Phase.MOVE) {
                val tracked = sf.status?.idsByDet?.keys
                val t = if (phase == Phase.FIND) cutoff(sf.frame.dets) else minSlender
                drawDetections(sf.frame) { i ->
                    val good = if (tracked != null) i in tracked else sf.frame.dets[i].slender >= t
                    if (good) MARK_GREEN else MARK_GRAY
                }
            }
            if (phase == Phase.MOVE) for (side in Side.entries) if (!coverage.done(side)) drawSideArrow(side)
        }

        Column(
            Modifier.align(Alignment.TopStart).fillMaxWidth().safeDrawingPadding().background(Color(0xAA000000)).padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            when (phase) {
                Phase.FIND -> {
                    Text("Step 1 of 2: find your X's", style = MaterialTheme.typography.titleMedium, color = Color.White)
                    Text("Point the camera at the canvas and step back until at least $NEED_IN_VIEW of your X's have green circles.", color = Color.White)
                    Text(
                        "${minOf(green, NEED_IN_VIEW)} of $NEED_IN_VIEW X's found" + if (green > NEED_IN_VIEW) " (+${green - NEED_IN_VIEW} more)" else "",
                        style = MaterialTheme.typography.titleMedium,
                        color = if (green >= NEED_IN_VIEW) MARK_GREEN else TEXT_WARN,
                    )
                    val hint = when {
                        frame == null -> "Starting the camera…"
                        frame.brightness < DARK_FRAME -> "The camera sees only darkness. Is the lens covered, or the phone face down?"
                        green >= NEED_IN_VIEW -> "Hold steady and tap Start."
                        gray >= 3 -> "Gray circles are marks that don't look enough like X's. If they're on your X's, tap Tune and lower X shape or Contrast."
                        else -> "Step back or turn so more X's are in view. X's need clear canvas around them."
                    }
                    Text(hint, color = TEXT_DIM)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        FilterChip(selected = autoColor, onClick = { autoColor = true }, label = { Text("Auto") })
                        FilterChip(selected = !autoColor && project.polarity == XDetector.Polarity.DARK, onClick = { autoColor = false; onPolarity(XDetector.Polarity.DARK) }, label = { Text("Dark X's") })
                        FilterChip(selected = !autoColor && project.polarity == XDetector.Polarity.LIGHT, onClick = { autoColor = false; onPolarity(XDetector.Polarity.LIGHT) }, label = { Text("Light X's") })
                        FilterChip(selected = tuning, onClick = { tuning = !tuning }, label = { Text("Tune") })
                    }
                    if (autoColor) Text("Looking for ${if (bestColor == XDetector.Polarity.DARK) "dark" else "light"} X's (worked out from what the camera sees).", color = TEXT_DIM)
                }
                Phase.MOVE -> {
                    val st = current?.status
                    val views = st?.keyframes ?: 0
                    Text("Step 2 of 2: move around the canvas", style = MaterialTheme.typography.titleMedium, color = Color.White)
                    Text("Slowly slide the phone toward each arrow, keeping the X's in view. An arrow goes away once that side is done.", color = Color.White)
                    val progress = (minOf(views, WANT_VIEWS) + coverage.sidesDone * WANT_VIEWS / 4f) / (2f * WANT_VIEWS)
                    LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                    Text("${coverage.sidesDone} of 4 sides, ${minOf(views, WANT_VIEWS)} of $WANT_VIEWS views", color = TEXT_DIM)
                    when {
                        st == null || views == 0 -> Text("Waiting for $NEED_IN_VIEW X's in view…", color = TEXT_WARN)
                        st.lost -> Text("Lost track: point back at the X's.", color = TEXT_BAD)
                        coverage.sidesDone == 4 && views < WANT_VIEWS -> Text("Keep moving around a little longer.", color = TEXT_DIM)
                    }
                }
                Phase.SOLVING -> Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(Modifier.padding(4.dp))
                    Text("Working out the map…", color = Color.White)
                }
                Phase.DONE -> {
                    val r = result
                    if (r == null || r.marks.size < 4) {
                        Text("That didn't work", style = MaterialTheme.typography.titleMedium, color = TEXT_BAD)
                        Text("Couldn't make a map from that scan. Try again with more X's in view, moving more slowly.", color = Color.White)
                    } else {
                        Text("Mapped ${r.marks.size} X's", style = MaterialTheme.typography.titleMedium, color = MARK_GREEN)
                        if (r.viewSpreadDeg < 12) {
                            Text(
                                "The phone didn't move enough to measure the canvas shape well (%.0f°). It will work, but tracking is steadier if you scan again moving further to each side.".format(r.viewSpreadDeg),
                                color = TEXT_WARN,
                            )
                        } else {
                            Text("Tap Use this map, then Paint.", color = Color.White)
                        }
                    }
                }
            }
        }
        if (tuning && phase == Phase.FIND) {
            Box(Modifier.align(Alignment.Center).padding(horizontal = 8.dp)) {
                DetectorTuning(project.detector, findCut, onDetector)
            }
        }
        Row(Modifier.align(Alignment.BottomCenter).safeDrawingPadding().padding(16.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            when (phase) {
                Phase.FIND -> {
                    OutlinedButton(onClick = { holder.stop(); onBack() }) { Text("Cancel") }
                    Button(onClick = ::start, enabled = green >= NEED_IN_VIEW) { Text("Start") }
                }
                Phase.MOVE -> {
                    OutlinedButton(onClick = { holder.stop(); current = null; phase = Phase.FIND }) { Text("Start over") }
                    Button(onClick = ::finish, enabled = (current?.status?.keyframes ?: 0) >= 4) { Text("Finish now") }
                }
                Phase.SOLVING -> {}
                Phase.DONE -> {
                    val r = result
                    val ok = r != null && r.marks.size >= 4
                    OutlinedButton(onClick = { result = null; current = null; phase = Phase.FIND }) { Text(if (ok) "Scan again" else "Try again") }
                    if (ok) Button(onClick = { onSave(r!!.marks, minSlender) }) { Text("Use this map") }
                }
            }
        }
    }
}

/** A large chevron at the middle of one edge of the screen, pointing outward: "move the phone this way". */
private fun DrawScope.drawSideArrow(side: Side) {
    val s = 34.dp.toPx() // half the chevron's width
    val inset = 40.dp.toPx()
    val w = size.width
    val h = size.height
    // tip and the two tail ends, for a chevron pointing right at the origin, then placed and turned
    val (cx, cy, ux, uy) = when (side) {
        Side.LEFT -> listOf(inset, h / 2, -1f, 0f)
        Side.RIGHT -> listOf(w - inset, h / 2, 1f, 0f)
        Side.UP -> listOf(w / 2, h * 0.36f, 0f, -1f) // below the instructions panel
        Side.DOWN -> listOf(w / 2, h * 0.80f, 0f, 1f) // above the buttons
    }
    val px = -uy
    val py = ux
    val path = Path().apply {
        moveTo(cx - ux * s * 0.6f + px * s, cy - uy * s * 0.6f + py * s)
        lineTo(cx + ux * s * 0.4f, cy + uy * s * 0.4f)
        lineTo(cx - ux * s * 0.6f - px * s, cy - uy * s * 0.6f - py * s)
    }
    drawPath(path, Color(0x99000000), style = Stroke(width = 16.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round))
    drawPath(path, Color.White, style = Stroke(width = 9.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round))
}
