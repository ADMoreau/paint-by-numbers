package io.github.admoreau.paintbynumbers.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import io.github.admoreau.paintbynumbers.planner.Paint
import io.github.admoreau.paintbynumbers.planner.slugify
import io.github.admoreau.paintbynumbers.project.Project
import kotlinx.coroutines.delay
import kotlin.math.roundToInt

/** A plan being made on the phone: what it last reported, and how it ended. */
class Planning {
    var running by mutableStateOf(false)
    var log by mutableStateOf(listOf<String>())
    var startedAt by mutableLongStateOf(0L)
    var error by mutableStateOf<String?>(null)
    @Volatile var cancel = false
}

private val PRESETS = listOf("#ffffff" to "White", "#f2ead8" to "Cream", "#808080" to "Gray", "#915f6d" to "Mauve", "#000000" to "Black")

/** Settings for planning the chosen image on the phone (paints, canvas colour, layers), and progress while it runs. */
@Composable
fun PlanScreen(
    project: Project,
    image: Bitmap?,
    paints: List<Paint>,
    planning: Planning,
    onChange: (Project) -> Unit,
    onStart: () -> Unit,
    onBack: () -> Unit,
) {
    if (planning.running) {
        val view = LocalView.current
        LaunchedEffect(Unit) { view.keepScreenOn = true }
        androidx.compose.runtime.DisposableEffect(Unit) { onDispose { view.keepScreenOn = false } }
        var now by remember { mutableLongStateOf(System.currentTimeMillis()) }
        LaunchedEffect(Unit) { while (true) { now = System.currentTimeMillis(); delay(500) } }
        Column(
            Modifier.fillMaxSize().safeDrawingPadding().padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("Planning ${project.planLayers} layers", style = MaterialTheme.typography.headlineSmall)
            LinearProgressIndicator(Modifier.fillMaxWidth())
            Text("%d s".format((now - planning.startedAt) / 1000))
            for (line in planning.log.takeLast(4)) Text(line, style = MaterialTheme.typography.bodySmall)
            Text("Keep the app open: this takes about a minute.", style = MaterialTheme.typography.bodySmall)
            OutlinedButton(onClick = { planning.cancel = true }) { Text("Cancel") }
        }
        return
    }

    val owned = project.palette.map { slugify(it) }.toSet()
    Column(
        Modifier.fillMaxSize().safeDrawingPadding().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = onBack) { Text("◀ Back") }
            Text("Make a plan", style = MaterialTheme.typography.headlineSmall)
        }
        if (image != null) {
            Image(image.asImageBitmap(), contentDescription = "Image to plan", contentScale = ContentScale.Fit, modifier = Modifier.fillMaxWidth().heightIn(max = 180.dp))
        }
        planning.error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
        Text(
            "The plan is a stack of layers. Each layer is one paint at one strength (such as 1/4 coat: 1 part paint to 3 parts Gloss Medium) over one area. " +
                "Light washes go down first and stronger coats build up the image on top.",
            style = MaterialTheme.typography.bodySmall,
        )

        PlanCard("Canvas colour") {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                for ((hex, name) in PRESETS) {
                    val selected = project.backgroundHex.equals(hex, ignoreCase = true)
                    Box(
                        Modifier.size(36.dp).background(hexColor(hex), CircleShape)
                            .border(if (selected) 3.dp else 1.dp, if (selected) MaterialTheme.colorScheme.primary else Color.Gray, CircleShape)
                            .clickable { onChange(project.copy(backgroundHex = hex)) },
                    )
                }
            }
            var text by remember(project.backgroundHex) { mutableStateOf(project.backgroundHex) }
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                OutlinedTextField(
                    value = text,
                    onValueChange = { t ->
                        text = t
                        val h = "#" + t.trim().removePrefix("#")
                        if (Regex("#[0-9a-fA-F]{6}").matches(h)) onChange(project.copy(backgroundHex = h.lowercase()))
                    },
                    label = { Text("Colour (#RRGGBB)") },
                    singleLine = true,
                    modifier = Modifier.width(180.dp),
                )
                Box(Modifier.size(36.dp).background(hexColor(project.backgroundHex)).border(1.dp, Color.Gray))
            }
            Text("The colour of the canvas before you start (a toned ground, or white).", style = MaterialTheme.typography.bodySmall)
        }

        PlanCard("Layers: ${project.planLayers}") {
            Slider(
                value = project.planLayers.toFloat(),
                onValueChange = { onChange(project.copy(planLayers = it.roundToInt())) },
                valueRange = 4f..20f,
                steps = 15,
            )
            Text("More layers match the image more closely but take longer to paint. Fewer are used if more wouldn't help.", style = MaterialTheme.typography.bodySmall)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("Detail")
                for ((size, label) in listOf(256 to "Quick", 512 to "Fine")) {
                    FilterChip(selected = project.planSize == size, onClick = { onChange(project.copy(planSize = size)) }, label = { Text(label) })
                }
            }
        }

        PlanCard("Your paints (${owned.size} chosen)") {
            Text("Tick the paints you have. The plan only uses these.", style = MaterialTheme.typography.bodySmall)
            var filter by remember { mutableStateOf("") }
            OutlinedTextField(value = filter, onValueChange = { filter = it }, label = { Text("Search") }, singleLine = true, modifier = Modifier.fillMaxWidth())
            if (owned.isNotEmpty()) TextButton(onClick = { onChange(project.copy(palette = emptyList())) }) { Text("Clear all") }
            val shown = paints.filter { filter.isBlank() || it.name.contains(filter.trim(), ignoreCase = true) }
                .sortedWith(compareBy({ it.slug !in owned }, { it.name }))
            for (p in shown) {
                val on = p.slug in owned
                Row(
                    Modifier.fillMaxWidth().clickable {
                        val next = if (on) project.palette.filter { slugify(it) != p.slug } else project.palette + p.name
                        onChange(project.copy(palette = next))
                    },
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Checkbox(checked = on, onCheckedChange = null)
                    Box(Modifier.padding(horizontal = 8.dp).size(20.dp).background(hexColor(masstoneHex(p)), CircleShape))
                    Column {
                        Text(p.name)
                        Text(p.transparency + if (p.approximate) " · approximate colour (no swatch data)" else "", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
                    }
                }
            }
        }

        val ready = image != null && owned.isNotEmpty()
        Button(onClick = onStart, enabled = ready, modifier = Modifier.fillMaxWidth()) { Text("Make plan") }
        if (image == null) Text("Choose an image first.", style = MaterialTheme.typography.bodySmall)
        else if (owned.isEmpty()) Text("Tick at least one paint.", style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun PlanCard(title: String, content: @Composable () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            content()
        }
    }
}

private fun hexColor(hex: String): Color = runCatching { Color(android.graphics.Color.parseColor(hex)) }.getOrDefault(Color.White)

private fun masstoneHex(p: Paint): String {
    val m = io.github.admoreau.paintbynumbers.planner.Km.masstone(p.K, p.S)
    return "#%06x".format(io.github.admoreau.paintbynumbers.planner.ColorMath.workingToArgb(m) and 0xFFFFFF)
}
