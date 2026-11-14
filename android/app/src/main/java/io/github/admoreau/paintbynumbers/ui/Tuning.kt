package io.github.admoreau.paintbynumbers.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.FilterChip
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import io.github.admoreau.paintbynumbers.project.DetectorSettings
import kotlin.math.roundToInt

/**
 * Knobs for the X finder, shown over the camera so their effect is visible at once:
 * - Contrast: how strongly a mark must stand out from the surface (lower finds fainter marks, and more clutter).
 * - Smallest X: ignore marks smaller than this in the camera image.
 * - X shape: how much like two thin crossing lines a mark must be (Auto = learned from your X's when scanning).
 */
@Composable
fun DetectorTuning(settings: DetectorSettings, learnedShape: Double, onChange: (DetectorSettings) -> Unit, modifier: Modifier = Modifier) {
    Column(modifier.fillMaxWidth().background(Color(0xCC000000)).padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text("X finder", color = Color.White)
        Knob("Contrast", "%.1f".format(settings.contrast), settings.contrast.toFloat(), 1.5f..8f) {
            onChange(settings.copy(contrast = (it * 10).roundToInt() / 10.0))
        }
        Knob("Smallest X", "${settings.minSizePx} px", settings.minSizePx.toFloat(), 6f..60f) {
            onChange(settings.copy(minSizePx = it.roundToInt()))
        }
        val auto = settings.shape == null
        val shape = settings.shape ?: learnedShape
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("X shape", color = Color.White, modifier = Modifier.width(84.dp))
            FilterChip(selected = auto, onClick = { onChange(settings.copy(shape = if (auto) learnedShape.coerceAtLeast(3.0) else null)) }, label = { Text("Auto") })
            Text(if (auto) "learned %.1f".format(learnedShape) else "%.1f".format(shape), color = Color(0xFFBDBDBD))
        }
        if (!auto) {
            Slider(value = shape.toFloat().coerceIn(2f, 14f), onValueChange = { onChange(settings.copy(shape = (it * 2).roundToInt() / 2.0)) }, valueRange = 2f..14f)
        }
        Text("Lower contrast or shape finds fainter or rougher X's but also more clutter; raise them if gray circles pile up on things that aren't X's.", color = Color(0xFFBDBDBD))
        OutlinedButton(onClick = { onChange(DetectorSettings.DEFAULT) }) { Text("Reset") }
    }
}

@Composable
private fun Knob(label: String, value: String, current: Float, range: ClosedFloatingPointRange<Float>, onValue: (Float) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(label, color = Color.White, modifier = Modifier.width(84.dp))
        Slider(value = current.coerceIn(range.start, range.endInclusive), onValueChange = onValue, valueRange = range, modifier = Modifier.weight(1f))
        Text(value, color = Color(0xFFBDBDBD), modifier = Modifier.width(52.dp))
    }
}
