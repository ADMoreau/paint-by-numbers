package io.github.admoreau.paintbynumbers.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import io.github.admoreau.paintbynumbers.plan.Plan
import io.github.admoreau.paintbynumbers.project.Project
import io.github.admoreau.paintbynumbers.project.TrackingMode
import io.github.admoreau.paintbynumbers.tracking.XDetector

/** Project setup: how the canvas is found, what to paint, and the way into painting. */
@Composable
fun HomeScreen(
    project: Project,
    image: Bitmap?,
    plan: Plan?,
    message: String?,
    onChange: (Project) -> Unit,
    onImportPlan: () -> Unit,
    onPickImage: () -> Unit,
    onSaveMarkerSheet: () -> Unit,
    onScan: () -> Unit,
    onPaint: () -> Unit,
) {
    Column(
        Modifier.fillMaxSize().safeDrawingPadding().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("Paint by Numbers", style = MaterialTheme.typography.headlineMedium)
        if (message != null) Text(message, color = MaterialTheme.colorScheme.primary)

        Section("1. How to find the canvas") {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(selected = project.mode == TrackingMode.X_MARKS, onClick = { onChange(project.copy(mode = TrackingMode.X_MARKS)) }, label = { Text("X marks") })
                FilterChip(selected = project.mode == TrackingMode.MARKERS, onClick = { onChange(project.copy(mode = TrackingMode.MARKERS)) }, label = { Text("Square markers") })
                FilterChip(selected = project.mode == TrackingMode.FREE, onClick = { onChange(project.copy(mode = TrackingMode.FREE)) }, label = { Text("Free") })
            }
            when (project.mode) {
                TrackingMode.X_MARKS -> {
                    Text(
                        "Draw X's about 1.5 cm across, every 8-12 cm over the canvas, each with clear canvas around it, in a colour that stands out (dark or light). " +
                            "Then scan: the app walks you through it in two steps and takes about half a minute.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    val n = project.marks.size
                    Text(if (n == 0) "Not scanned yet." else "$n X's mapped (${if (project.polarity == XDetector.Polarity.DARK) "dark" else "light"} marks). New canvas or more X's? Scan again.")
                    Button(onClick = onScan) { Text(if (n == 0) "Scan canvas" else "Scan a new canvas") }
                }
                TrackingMode.MARKERS -> {
                    Text(
                        "Print the marker sheet, cut out a few markers (keep a white border) and stick them flat around or on the canvas, anywhere. " +
                            "No scan needed: the app maps them as it sees them. Keep at least one in view while painting; more markers keep it steadier.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    var text by remember(project.markerSizeMm) { mutableStateOf("%.0f".format(project.markerSizeMm)) }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(
                            value = text,
                            onValueChange = { t ->
                                text = t
                                t.toDoubleOrNull()?.takeIf { it in 10.0..300.0 }?.let { mm ->
                                    if (mm != project.markerSizeMm) onChange(project.copy(markerSizeMm = mm, markerMap = emptyList(), markerPlacement = null))
                                }
                            },
                            label = { Text("Printed size (mm)") },
                            singleLine = true,
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                            modifier = Modifier.width(170.dp),
                        )
                    }
                    Text("Measure a printed marker's black square and enter its side here.", style = MaterialTheme.typography.bodySmall)
                    OutlinedButton(onClick = onSaveMarkerSheet) { Text("Save marker sheet (PDF)") }
                    val n = project.markerMap.size / 4
                    if (n > 0) {
                        Text("$n marker${if (n == 1) "" else "s"} mapped.")
                        OutlinedButton(onClick = { onChange(project.copy(markerMap = emptyList(), markerPlacement = null)) }) { Text("Forget marker positions") }
                    }
                }
                TrackingMode.FREE -> Text(
                    "No tracking: the image is fixed to the screen where you place it. Keep the phone still (a stand or tripod helps) and trace through the camera view. If you've drawn the composition on the canvas, Snap to drawing lines the image up with it.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        Section("2. What to paint") {
            if (image != null) {
                Image(image.asImageBitmap(), contentDescription = "Image to paint", contentScale = ContentScale.Fit, modifier = Modifier.fillMaxWidth().heightIn(max = 220.dp))
            }
            if (plan != null) {
                Text("Plan: ${plan.layers.size} layers" + (plan.groundPaint?.let { " over a ground of $it" } ?: "") + ", ${plan.layers.map { it.paint }.distinct().size} paints.")
            } else if (image == null) {
                Text("Import a plan made by the desktop planner (project.pbn), or just choose an image.")
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = onImportPlan) { Text("Import plan") }
                OutlinedButton(onClick = onPickImage) { Text("Choose image") }
            }
            Text("You place, size and turn it on the canvas in the Paint view.", style = MaterialTheme.typography.bodySmall)
        }

        val ready = image != null && (project.mode != TrackingMode.X_MARKS || project.marks.size >= 4)
        Button(onClick = onPaint, enabled = ready, modifier = Modifier.fillMaxWidth()) { Text("Paint") }
        if (project.mode == TrackingMode.X_MARKS && project.marks.size < 4) Text("Scan the canvas first (the map needs at least 4 X's).", style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            content()
        }
    }
}
