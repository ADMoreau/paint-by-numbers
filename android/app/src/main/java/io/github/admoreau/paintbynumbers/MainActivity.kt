package io.github.admoreau.paintbynumbers

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import io.github.admoreau.paintbynumbers.camera.FrameDump
import io.github.admoreau.paintbynumbers.plan.Plan
import io.github.admoreau.paintbynumbers.plan.PlanStore
import io.github.admoreau.paintbynumbers.planner.Paint
import io.github.admoreau.paintbynumbers.planner.PaintDb
import io.github.admoreau.paintbynumbers.planner.PhonePlanner
import io.github.admoreau.paintbynumbers.planner.UniformConfig
import io.github.admoreau.paintbynumbers.planner.UniformPlanner
import io.github.admoreau.paintbynumbers.project.AppFolder
import io.github.admoreau.paintbynumbers.project.MarkerSheet
import io.github.admoreau.paintbynumbers.project.Placement
import io.github.admoreau.paintbynumbers.project.Project
import io.github.admoreau.paintbynumbers.project.SavedCanvas
import io.github.admoreau.paintbynumbers.project.ProjectStore
import io.github.admoreau.paintbynumbers.project.TrackingMode
import io.github.admoreau.paintbynumbers.ui.HomeScreen
import io.github.admoreau.paintbynumbers.ui.LayersScreen
import io.github.admoreau.paintbynumbers.ui.PaintScreen
import io.github.admoreau.paintbynumbers.ui.PlanScreen
import io.github.admoreau.paintbynumbers.ui.Planning
import io.github.admoreau.paintbynumbers.ui.ScanScreen
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        check(OpenCVLoader.initLocal()) { "OpenCV failed to load" }
        val store = ProjectStore(applicationContext)
        FrameDump.dir = filesDir
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                Surface(Modifier.fillMaxSize()) { App(store) }
            }
        }
    }
}

private enum class Screen { HOME, SCAN, PAINT, PLAN, LAYERS }

@Composable
private fun App(store: ProjectStore) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var project by remember { mutableStateOf(store.load()) }
    var screen by rememberSaveable { mutableStateOf(Screen.HOME) }
    var message by remember { mutableStateOf<String?>(null) }
    val image by produceState<Bitmap?>(null, project.imageFile) {
        value = project.imageFile?.let { name -> withContext(Dispatchers.IO) { store.loadImage(name) } }
    }
    val plan by produceState<Plan?>(null, project.planDir) {
        value = project.planDir?.let { dir -> withContext(Dispatchers.IO) { runCatching { PlanStore.load(File(context.filesDir, dir)) }.getOrNull() } }
    }
    var canvases by remember { mutableStateOf(store.loadCanvases()) }
    var scanPolarity by remember { mutableStateOf<io.github.admoreau.paintbynumbers.tracking.XDetector.Polarity?>(null) }
    var imageSize by remember { mutableStateOf(store.imageSize(project.imageFile)) }

    /** The project's maps and placements as its saved canvas (keeping the canvas's name). */
    fun snapshot(p: Project, base: SavedCanvas): SavedCanvas = base.copy(
        polarity = p.polarity, minSlender = p.minSlender, marks = p.marks, markerMap = p.markerMap,
        placement = p.placement, markerPlacement = p.markerPlacement,
        imageW = imageSize?.first ?: 0, imageH = imageSize?.second ?: 0,
    )

    fun saveCanvases(cs: List<SavedCanvas>) {
        canvases = cs
        store.saveCanvases(cs)
    }

    fun update(p: Project) {
        project = p
        store.save(p)
        // keep the active canvas's saved copy current
        val i = canvases.indexOfFirst { it.id == p.canvasId }
        if (i >= 0) {
            val next = snapshot(p, canvases[i])
            if (next != canvases[i]) saveCanvases(canvases.toMutableList().also { it[i] = next })
        }
    }

    // maps scanned before canvases were saved become the first saved canvas
    LaunchedEffect(Unit) {
        if (project.canvasId == null && project.marks.isNotEmpty()) {
            val c = snapshot(project, SavedCanvas("c${System.currentTimeMillis()}", "Canvas 1", System.currentTimeMillis(), project.polarity, project.minSlender, project.marks))
            saveCanvases(canvases + c)
            update(project.copy(canvasId = c.id))
        }
    }

    /** Switches to a saved canvas: its maps, and its placements if they suit the current image. */
    fun useCanvas(c: SavedCanvas) {
        val size = imageSize
        fun fit(pl: Placement?) = if (size == null) null else pl?.forImage(c.imageW, c.imageH, size.first, size.second)
        update(project.copy(
            canvasId = c.id, polarity = c.polarity, minSlender = c.minSlender, marks = c.marks, markerMap = c.markerMap,
            placement = fit(c.placement), markerPlacement = fit(c.markerPlacement),
        ))
    }

    fun deleteCanvas(c: SavedCanvas) {
        saveCanvases(canvases.filter { it.id != c.id })
        if (project.canvasId == c.id) update(project.copy(canvasId = null, marks = emptyList(), markerMap = emptyList(), placement = null, markerPlacement = null))
    }

    fun renameCanvas(c: SavedCanvas, name: String) {
        saveCanvases(canvases.map { if (it.id == c.id) it.copy(name = name) else it })
    }

    /**
     * A new image to paint; a replaced plan's or image's files are removed. If it has the same shape
     * as the old one (such as a plan of the same picture), it keeps its place on the canvas.
     */
    fun newImage(imageFile: String, planDir: String?) {
        val old = project
        val from = imageSize
        val to = store.imageSize(imageFile)
        fun keep(pl: Placement?) = if (from == null || to == null) null else pl?.forImage(from.first, from.second, to.first, to.second)
        imageSize = to
        update(project.copy(
            imageFile = imageFile, planDir = planDir, currentLayer = 0,
            placement = keep(old.placement), markerPlacement = keep(old.markerPlacement), freePlacement = keep(old.freePlacement),
        ))
        old.planDir?.takeIf { it != planDir }?.let { File(context.filesDir, it).deleteRecursively() }
        old.imageFile?.takeIf { old.planDir == null && it != imageFile }?.let { File(context.filesDir, it).delete() }
    }

    LaunchedEffect(Unit) { withContext(Dispatchers.IO) { runCatching { AppFolder.ensure(context) } } }
    val picker = rememberLauncherForActivityResult(AppFolder.Open()) { uri ->
        if (uri != null) newImage(store.importImage(uri), null)
    }
    val planPicker = rememberLauncherForActivityResult(AppFolder.Open()) { uri ->
        if (uri != null) scope.launch {
            message = try {
                val dir = withContext(Dispatchers.IO) { PlanStore.import(context, uri) }
                val p = withContext(Dispatchers.IO) { PlanStore.load(File(context.filesDir, dir)) }
                newImage("$dir/${p.targetFile}", dir)
                "Plan imported: ${p.layers.size} layers."
            } catch (e: Exception) {
                "That file isn't a plan from the desktop planner (project.pbn): ${e.message}"
            }
        }
    }
    var granted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }

    val paints by produceState<List<Paint>>(emptyList()) {
        value = withContext(Dispatchers.IO) { runCatching { PhonePlanner.paints(context) }.getOrDefault(emptyList()) }
    }
    val planning = remember { Planning() }

    /** Plans the current image with the project's settings; on success it becomes the project's plan. */
    fun startPlanning() {
        val source = image ?: return
        val p = project
        planning.running = true
        planning.cancel = false
        planning.error = null
        planning.log = emptyList()
        planning.startedAt = System.currentTimeMillis()
        scope.launch {
            try {
                val dir = withContext(Dispatchers.Default) {
                    PhonePlanner.run(
                        source, PaintDb.select(paints, p.palette), p.backgroundHex, UniformConfig(layers = p.planLayers), context.filesDir, p.planSize,
                        progress = { msg -> scope.launch { planning.log = planning.log + msg } },
                        cancelled = { planning.cancel },
                    )
                }
                val loaded = withContext(Dispatchers.IO) { PlanStore.load(dir) }
                newImage("${dir.name}/${loaded.targetFile}", dir.name)
                message = "Plan made: ${loaded.layers.size} layers, ${loaded.layers.map { it.paint }.distinct().size} paints."
                screen = Screen.LAYERS
            } catch (e: UniformPlanner.Cancelled) {
                planning.error = null
            } catch (e: Throwable) {
                planning.error = "Couldn't make the plan: ${e.message ?: e.javaClass.simpleName}"
            } finally {
                planning.running = false
            }
        }
    }

    if (screen in setOf(Screen.SCAN, Screen.PAINT) && !granted) {
        Column(Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
            Text("The camera is needed to see your canvas.")
            Button(onClick = { permission.launch(Manifest.permission.CAMERA) }, modifier = Modifier.padding(top = 16.dp)) { Text("Allow camera") }
            Button(onClick = { screen = Screen.HOME }, modifier = Modifier.padding(top = 8.dp)) { Text("Back") }
        }
        return
    }
    when (screen) {
        Screen.HOME -> HomeScreen(
            project = project,
            image = image,
            plan = plan,
            message = message,
            onChange = ::update,
            onImportPlan = { planPicker.launch(arrayOf("*/*")) },
            onPickImage = { picker.launch(arrayOf("image/*")) },
            onSaveMarkerSheet = {
                scope.launch {
                    val name = withContext(Dispatchers.IO) { runCatching { MarkerSheet.save(context, project.markerSizeMm) }.getOrNull() }
                    message = if (name != null) "Saved \"$name\" to Download/${AppFolder.NAME}: print it at 100% (no scaling)." else "Couldn't save the marker sheet on this phone."
                }
            },
            onScan = { message = null; scanPolarity = null; screen = Screen.SCAN },
            onPlan = { message = null; screen = Screen.PLAN },
            onLayers = { message = null; screen = Screen.LAYERS },
            canvases = canvases,
            onUseCanvas = ::useCanvas,
            onRenameCanvas = ::renameCanvas,
            onDeleteCanvas = ::deleteCanvas,
            onPaint = { message = null; screen = Screen.PAINT },
        )
        Screen.LAYERS -> {
            val p = plan
            if (p != null && p.dir.name == project.planDir && p.layers.isNotEmpty()) {
                LayersScreen(p, project.currentLayer, onLayer = { update(project.copy(currentLayer = it)) }, onBack = { screen = Screen.HOME })
            } else if (project.planDir == null) {
                LaunchedEffect(Unit) { screen = Screen.HOME }
            } else {
                Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) { Text("Loading the plan…") }
            }
        }
        Screen.PLAN -> PlanScreen(
            project = project,
            image = image,
            paints = paints,
            planning = planning,
            onChange = ::update,
            onStart = ::startPlanning,
            onBack = { screen = Screen.HOME },
        )
        Screen.SCAN -> ScanScreen(
            project = project.copy(polarity = scanPolarity ?: project.polarity),
            onPolarity = { scanPolarity = it }, // belongs to the new canvas, saved with it

            onDetector = { update(project.copy(detector = it)) },
            onSave = { marks, minSlender ->
                // each scan is a new saved canvas; a new map has new coordinates, so the image is placed afresh
                val now = System.currentTimeMillis()
                val name = "Canvas ${canvases.size + 1}, " + java.text.SimpleDateFormat("MMM d", java.util.Locale.getDefault()).format(java.util.Date(now))
                val polarity = scanPolarity ?: project.polarity
                val c = SavedCanvas("c$now", name, now, polarity, minSlender, marks)
                saveCanvases(canvases + c)
                update(project.copy(canvasId = c.id, polarity = polarity, marks = marks, minSlender = minSlender, markerMap = emptyList(), placement = null, markerPlacement = null))
                screen = Screen.HOME
            },
            onBack = { screen = Screen.HOME },
        )
        Screen.PAINT -> PaintScreen(
            project = project,
            image = image,
            plan = plan,
            onPlacement = { mode, pl ->
                update(
                    when (mode) {
                        TrackingMode.X_MARKS -> project.copy(placement = pl)
                        TrackingMode.MARKERS -> project.copy(markerPlacement = pl)
                        TrackingMode.FREE -> project.copy(freePlacement = pl)
                    },
                )
            },
            onMarkerMap = { update(project.copy(markerMap = it)) },
            onLayer = { update(project.copy(currentLayer = it)) },
            onDetector = { update(project.copy(detector = it)) },
            onPolarity = { update(project.copy(polarity = it)) },
            onBack = { screen = Screen.HOME },
        )
    }
}
