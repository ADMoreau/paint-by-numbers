package io.github.admoreau.paintbynumbers.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import io.github.admoreau.paintbynumbers.plan.Plan
import io.github.admoreau.paintbynumbers.plan.PlanThickness
import io.github.admoreau.paintbynumbers.plan.RegionRenderer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.min
import kotlin.math.roundToInt

private enum class LayerView(val label: String) { WHERE("Where it goes"), AFTER("After this layer"), FINAL("Finished"), TARGET("Your image") }

/** What the screen draws: a base picture, optionally darkened, with the layer's areas and outlines on top. */
private class LayerPicture(val base: ImageBitmap?, val baseColor: Color?, val dim: Boolean, val fill: ImageBitmap?, val lines: ImageBitmap?, val aspect: Float)

/**
 * The plan layer by layer, without the camera: where each layer goes (over the painting so far),
 * what to mix, and how the painting looks after it. For looking the plan over before painting.
 */
@Composable
fun LayersScreen(plan: Plan, startLayer: Int, onLayer: (Int) -> Unit, onBack: () -> Unit) {
    var index by remember { mutableIntStateOf(startLayer.coerceIn(0, plan.layers.size - 1)) }
    var view by remember { mutableStateOf(LayerView.WHERE) }
    var thickness by remember(index) { mutableStateOf<PlanThickness?>(null) }
    val layer = plan.layers[index]
    val aspect = plan.width.toFloat() / plan.height
    val background = remember(plan) { Color(android.graphics.Color.parseColor(plan.backgroundHex)) }

    fun decode(name: String?): ImageBitmap? = name?.let { BitmapFactory.decodeFile(plan.file(it).path) }?.asImageBitmap()

    val picture by produceState<LayerPicture?>(null, plan, index, view, thickness) {
        value = withContext(Dispatchers.Default) {
            when (view) {
                LayerView.WHERE -> {
                    val before = if (index > 0) decode(plan.layers[index - 1].stepFile) else null
                    val r = RegionRenderer.render(plan, layer, thickness)
                    LayerPicture(before, if (before == null) background else null, true, r?.fill?.asImageBitmap(), r?.lines?.asImageBitmap(), aspect)
                }
                LayerView.AFTER -> LayerPicture(decode(layer.stepFile), null, false, null, null, aspect)
                LayerView.FINAL -> LayerPicture(decode(plan.previewFile), null, false, null, null, aspect)
                LayerView.TARGET -> LayerPicture(decode(plan.targetFile), null, false, null, null, aspect)
            }
        }
    }
    val thumbs by produceState<List<ImageBitmap?>>(emptyList(), plan) {
        value = withContext(Dispatchers.IO) {
            plan.layers.map { l ->
                l.stepFile?.let { f ->
                    val opts = BitmapFactory.Options().apply { inSampleSize = 4 }
                    BitmapFactory.decodeFile(plan.file(f).path, opts)?.asImageBitmap()
                }
            }
        }
    }

    // zoom and pan; double-tap resets
    var scale by remember { mutableFloatStateOf(1f) }
    var pan by remember { mutableStateOf(Offset.Zero) }

    fun go(i: Int) {
        index = i.coerceIn(0, plan.layers.size - 1)
        onLayer(index)
    }

    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = onBack) { Text("◀ Back") }
            Text("Layers", style = MaterialTheme.typography.titleLarge)
        }
        Box(
            Modifier.weight(1f).fillMaxWidth().clip(androidx.compose.ui.graphics.RectangleShape)
                .pointerInput(Unit) {
                    detectTransformGestures { _, p, z, _ ->
                        scale = (scale * z).coerceIn(1f, 8f)
                        pan = if (scale == 1f) Offset.Zero else pan + p
                    }
                }
                .pointerInput(Unit) { detectTapGestures(onDoubleTap = { scale = 1f; pan = Offset.Zero }) },
            contentAlignment = Alignment.Center,
        ) {
            val pic = picture
            Canvas(Modifier.fillMaxSize().graphicsLayer { scaleX = scale; scaleY = scale; translationX = pan.x; translationY = pan.y }) {
                if (pic != null) drawPicture(pic)
            }
            if (scale == 1f) Text(
                "Pinch to zoom",
                style = MaterialTheme.typography.labelSmall,
                color = Color(0x99FFFFFF),
                modifier = Modifier.align(Alignment.TopEnd).padding(8.dp),
            )
        }

        Column(Modifier.padding(horizontal = 12.dp, vertical = 6.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = { go(index - 1) }, enabled = index > 0) { Text("◀") }
                Box(Modifier.size(20.dp).clip(CircleShape).background(hexColor(layer.fullStrengthHex)).border(1.dp, Color(0x66FFFFFF), CircleShape))
                Text("  Layer ${index + 1} of ${plan.layers.size}: ${layer.paint}", modifier = Modifier.weight(1f), style = MaterialTheme.typography.titleMedium)
                TextButton(onClick = { go(index + 1) }, enabled = index < plan.layers.size - 1) { Text("▶") }
            }
            val single = layer.thicknesses.singleOrNull()
            if (single == null && layer.thicknesses.isNotEmpty()) Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                FilterChip(selected = thickness == null, onClick = { thickness = null }, label = { Text("All") })
                for (t in layer.thicknesses) FilterChip(
                    selected = thickness == t,
                    onClick = { thickness = t; view = LayerView.WHERE },
                    label = { Text(t.label.removeSuffix(" coats").removeSuffix(" coat")) },
                    leadingIcon = { Box(Modifier.size(14.dp).clip(CircleShape).background(hexColor(t.colorHex))) },
                )
            }
            val t = thickness ?: single
            Text(
                when {
                    t != null -> "${t.label}: ${t.how}. Covers %.0f%% of the canvas.".format(100 * t.area)
                    else -> "${layer.thicknesses.size} thicknesses, %.0f%% of the canvas. Pick one to see where it goes.".format(100 * layer.thicknesses.sumOf { it.area })
                } + (if (index == 0 && plan.groundPaint != null) " Paint a ground coat of ${plan.groundPaint} first." else ""),
                style = MaterialTheme.typography.bodyMedium,
            )
            if (t != null) Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(14.dp).clip(CircleShape).background(hexColor(t.colorHex)))
                Text("  how it looks on the canvas · ${layer.transparency}", style = MaterialTheme.typography.bodySmall, color = Color(0xFFBDBDBD))
            }
            Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                for (v in LayerView.entries) FilterChip(selected = view == v, onClick = { view = v }, label = { Text(v.label) })
            }
            if (view == LayerView.WHERE) Text(
                "Bright areas outlined in white get this layer, shown as they'll look once painted. The darkened rest is the painting so far.",
                style = MaterialTheme.typography.bodySmall,
                color = Color(0xFFBDBDBD),
            )
            // every layer at a glance: the painting after it; tap to jump
            val listState = rememberLazyListState()
            LaunchedEffect(index) { listState.animateScrollToItem(maxOf(0, index - 2)) }
            LazyRow(state = listState, horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth().height(76.dp)) {
                itemsIndexed(plan.layers) { i, l ->
                    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.clickable { go(i) }) {
                        val th = thumbs.getOrNull(i)
                        val mod = Modifier.height(56.dp).size((56 * aspect).dp, 56.dp)
                            .border(if (i == index) 2.dp else 0.dp, if (i == index) MaterialTheme.colorScheme.primary else Color.Transparent)
                        if (th != null) Image(th, contentDescription = "After layer ${i + 1}", contentScale = ContentScale.Fit, modifier = mod)
                        else Box(mod.background(Color.DarkGray))
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(Modifier.size(8.dp).clip(CircleShape).background(hexColor(l.fullStrengthHex)))
                            Text(" ${i + 1}", style = MaterialTheme.typography.labelSmall)
                        }
                    }
                }
            }
        }
    }
}

/** Draws [pic] fitted in the canvas, centred. */
private fun DrawScope.drawPicture(pic: LayerPicture) {
    val fit = min(size.width / pic.aspect, size.height)
    val w = fit * pic.aspect
    val h = fit
    val off = IntOffset(((size.width - w) / 2).roundToInt(), ((size.height - h) / 2).roundToInt())
    val dst = IntSize(w.roundToInt(), h.roundToInt())
    pic.baseColor?.let { drawRect(it, topLeft = Offset(off.x.toFloat(), off.y.toFloat()), size = androidx.compose.ui.geometry.Size(w, h)) }
    pic.base?.let { drawImage(it, dstOffset = off, dstSize = dst, filterQuality = FilterQuality.High) }
    if (pic.dim) drawRect(Color(0x99000000), topLeft = Offset(off.x.toFloat(), off.y.toFloat()), size = androidx.compose.ui.geometry.Size(w, h))
    pic.fill?.let { drawImage(it, dstOffset = off, dstSize = dst, filterQuality = FilterQuality.High) }
    pic.lines?.let { drawImage(it, dstOffset = off, dstSize = dst, filterQuality = FilterQuality.High) }
}

private fun hexColor(hex: String): Color = runCatching { Color(android.graphics.Color.parseColor(hex)) }.getOrDefault(Color.Gray)
