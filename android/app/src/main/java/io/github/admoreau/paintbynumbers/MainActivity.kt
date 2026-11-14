package io.github.admoreau.paintbynumbers

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
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
import io.github.admoreau.paintbynumbers.project.MarkerSheet
import io.github.admoreau.paintbynumbers.project.Project
import io.github.admoreau.paintbynumbers.project.ProjectStore
import io.github.admoreau.paintbynumbers.project.TrackingMode
import io.github.admoreau.paintbynumbers.ui.HomeScreen
import io.github.admoreau.paintbynumbers.ui.PaintScreen
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

private enum class Screen { HOME, SCAN, PAINT }

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
    fun update(p: Project) {
        project = p
        store.save(p)
    }

    /** A new image to paint: placements start afresh; a replaced plan's or image's files are removed. */
    fun newImage(imageFile: String, planDir: String?) {
        val old = project
        update(project.copy(imageFile = imageFile, planDir = planDir, currentLayer = 0, placement = null, markerPlacement = null, freePlacement = null))
        old.planDir?.takeIf { it != planDir }?.let { File(context.filesDir, it).deleteRecursively() }
        old.imageFile?.takeIf { old.planDir == null && it != imageFile }?.let { File(context.filesDir, it).delete() }
    }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) newImage(store.importImage(uri), null)
    }
    val planPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
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

    if (screen != Screen.HOME && !granted) {
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
            onPickImage = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
            onSaveMarkerSheet = {
                scope.launch {
                    val name = withContext(Dispatchers.IO) { runCatching { MarkerSheet.save(context, project.markerSizeMm) }.getOrNull() }
                    message = if (name != null) "Saved \"$name\" to Downloads: print it at 100% (no scaling)." else "Couldn't save the marker sheet on this phone."
                }
            },
            onScan = { message = null; screen = Screen.SCAN },
            onPaint = { message = null; screen = Screen.PAINT },
        )
        Screen.SCAN -> ScanScreen(
            project = project,
            onPolarity = { update(project.copy(polarity = it)) },
            onDetector = { update(project.copy(detector = it)) },
            onSave = { marks, minSlender ->
                update(project.copy(marks = marks, minSlender = minSlender, placement = null)) // a new map has new coordinates
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
