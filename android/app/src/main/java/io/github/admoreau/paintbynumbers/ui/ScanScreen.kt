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
import androidx.compose.material3.LinearProgressIndicator
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
import io.github.admoreau.paintbynumbers.tracking.XDetector
import androidx.compose.material3.FilterChip
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.hypot

private val MARK_GREEN = Color(0xFF00E676)
private val MARK_GRAY = Color(0xFF9E9E9E)
private val MARK_PINK = Color(0xFFFF4081)

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

    @Synchronized
    fun update(f: Frame): ScanSession.Status? {
        intrinsics = f.intrinsics ?: intrinsics
        if (!scanning) return null
        val s = session ?: ScanSession(f.width, f.height, minSlender).also { session = it }
        return s.update(f.dets)
    }

    @Synchronized
    fun start(minSlender: Double) {
        this.minSlender = minSlender
        session = null
        scanning = true
    }

    @Synchronized
    fun stop(): Pair<ScanSession?, Intrinsics?> {
        scanning = false
        return session to intrinsics
    }
}

private enum class Phase { READY, SCANNING, SOLVING, DONE }

private class ScanFrame(val frame: Frame, val status: ScanSession.Status?)

/** Scan the canvas: follow the X's while the phone moves, then work out their map. */
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
    var phase by remember { mutableStateOf(Phase.READY) }
    var result by remember { mutableStateOf<BundleAdjust.Result?>(null) }
    var minSlender by remember { mutableStateOf(0.0) }
    var tuning by remember { mutableStateOf(false) }
    // the X-likeness cutoff: the user's setting, else the strictest that still leaves 6 X's in view
    fun cutoff(dets: List<io.github.admoreau.paintbynumbers.tracking.XDetection>) = project.detector.shape ?: ScanSession.chooseMinSlender(dets)
    val analyzer = remember {
        FrameAnalyzer { f ->
            val st = holder.update(f)
            main.execute { current = ScanFrame(f, st) }
        }
    }
    analyzer.polarity = project.polarity
    analyzer.settings = project.detector
    BackHandler {
        holder.stop()
        onBack()
    }

    fun finish() {
        val (session, intr) = holder.stop()
        phase = Phase.SOLVING
        scope.launch {
            result = if (session == null || intr == null) null else withContext(Dispatchers.Default) {
                BundleAdjust.run(session.keyframes, session.refPositions, intr, keep = session.reliableIds())
            }
            phase = Phase.DONE
        }
    }

    Box(Modifier.fillMaxSize()) {
        CameraView(analyzer, Modifier.fillMaxSize())
        Canvas(Modifier.fillMaxSize()) {
            val sf = current ?: return@Canvas
            val tracked = sf.status?.idsByDet?.keys
            val t = if (phase == Phase.READY) cutoff(sf.frame.dets) else minSlender
            drawDetections(sf.frame) { i ->
                val good = if (tracked != null) i in tracked else sf.frame.dets[i].slender >= t
                if (good) MARK_GREEN else MARK_GRAY
            }
        }
        val st = current?.status
        Column(
            Modifier.align(Alignment.TopStart).fillMaxWidth().safeDrawingPadding().background(Color(0xAA000000)).padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                MarkColorChips(project.polarity, onPolarity)
                FilterChip(selected = tuning, onClick = { tuning = !tuning }, label = { Text("Tune") })
            }
            when (phase) {
                Phase.READY -> {
                    Text("Stand back so most of your X's are in view, then tap Start. Green circles should be on your X's; gray ones are ignored.", color = Color.White)
                    val dets = current?.frame?.dets ?: emptyList()
                    val t = cutoff(dets)
                    Text("${dets.count { it.slender >= t }} X's in view", color = Color(0xFFBDBDBD))
                }
                Phase.SCANNING -> {
                    val views = st?.keyframes ?: 0
                    Text("Now move the phone slowly side to side and up and down, keeping the X's in view.", color = Color.White)
                    LinearProgressIndicator(progress = { (views / 10f).coerceAtMost(1f) }, modifier = Modifier.fillMaxWidth())
                    Text("${st?.mapped ?: 0} X's confirmed, $views views" + if (views >= 10) ". That's plenty: tap Finish." else "", color = Color(0xFFBDBDBD))
                    if (st == null || views == 0) Text("Waiting for at least 6 X's in view…", color = Color(0xFFFFD54F))
                    else if (st.lost) Text("Lost track: point back at the X's.", color = Color(0xFFFF8A80))
                }
                Phase.SOLVING -> Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(Modifier.padding(4.dp))
                    Text("Working out the map…", color = Color.White)
                }
                Phase.DONE -> {
                    val r = result
                    if (r == null || r.marks.size < 4) {
                        Text("Couldn't make a map from that scan. Try again with more X's in view, moving more slowly.", color = Color(0xFFFF8A80))
                    } else {
                        Text("Mapped ${r.marks.size} X's (fit within %.1f px).".format(r.rmsePx), color = Color.White)
                        if (r.viewSpreadDeg < 12) {
                            Text(
                                "The phone didn't move enough to measure the canvas shape reliably (%.0f°). Scan again, moving it further side to side and up and down.".format(r.viewSpreadDeg),
                                color = Color(0xFFFFD54F),
                            )
                        }
                    }
                }
            }
        }
        if (tuning) {
            Box(Modifier.align(Alignment.Center).padding(horizontal = 8.dp)) {
                DetectorTuning(project.detector, minSlender.takeIf { it > 0 } ?: cutoff(current?.frame?.dets ?: emptyList()), onDetector)
            }
        }
        Row(Modifier.align(Alignment.BottomCenter).safeDrawingPadding().padding(16.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            when (phase) {
                Phase.READY -> {
                    OutlinedButton(onClick = onBack) { Text("Cancel") }
                    val dets = current?.frame?.dets ?: emptyList()
                    Button(onClick = {
                        minSlender = cutoff(dets)
                        holder.start(minSlender)
                        phase = Phase.SCANNING
                    }, enabled = dets.size >= 6) { Text("Start") }
                }
                Phase.SCANNING -> {
                    OutlinedButton(onClick = { holder.start(minSlender) }) { Text("Restart") }
                    Button(onClick = ::finish, enabled = (st?.keyframes ?: 0) >= 4) { Text("Finish") }
                }
                Phase.SOLVING -> {}
                Phase.DONE -> {
                    OutlinedButton(onClick = { result = null; phase = Phase.READY }) { Text("Scan again") }
                    val r = result
                    Button(onClick = { onSave(r!!.marks, minSlender) }, enabled = r != null && r.marks.size >= 4) { Text("Save map") }
                }
            }
        }
    }
}
