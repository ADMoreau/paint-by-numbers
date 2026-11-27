package io.github.admoreau.paintbynumbers.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import io.github.admoreau.paintbynumbers.plan.PlanLayer
import io.github.admoreau.paintbynumbers.plan.PlanThickness

/** One chip of a swatch: [paint] as it looks over [ground] (shown as a frame around it), or alone. */
private data class Chip(val label: String, val paint: String, val ground: String?)

/**
 * A layer's paint swatch: the paint straight from the tube, then the layer's mix as it looks over
 * white, over black and over the background, each chip framed by what it sits on. With no
 * thickness picked (a layer with several), one chip per thickness over the ground instead.
 */
@Composable
fun LayerSwatch(
    layer: PlanLayer,
    thickness: PlanThickness?,
    backgroundHex: String,
    modifier: Modifier = Modifier,
    chipHeight: Dp = 34.dp,
    labelColor: Color = Color(0xFFBDBDBD),
) {
    val chips = buildList {
        add(Chip(if (layer.transparency.isNotEmpty()) "Tube · ${layer.transparency}" else "Tube", layer.fullStrengthHex, null))
        if (thickness != null) {
            thickness.onWhiteHex?.let { add(Chip("On white", it, "#FFFFFF")) }
            thickness.onBlackHex?.let { add(Chip("On black", it, "#000000")) }
            thickness.onGroundHex?.let { add(Chip("On background", it, backgroundHex)) }
            if (size == 1) add(Chip(thickness.label, thickness.colorHex, null)) // a plan without the swatch colours
        } else {
            for (t in layer.thicknesses) add(Chip(t.label, t.colorHex, t.onGroundHex?.let { backgroundHex }))
        }
    }
    val shape = RoundedCornerShape(4.dp)
    Row(modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        for (c in chips) Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Box(
                Modifier.size(chipHeight * 1.7f, chipHeight).clip(shape)
                    .background(swatchColor(c.ground ?: c.paint)).border(1.dp, Color(0x66FFFFFF), shape),
            ) {
                // over a ground: the ground shows as a frame, so the chip reads like a painted test patch
                if (c.ground != null) Box(Modifier.fillMaxSize().padding(5.dp).clip(RoundedCornerShape(2.dp)).background(swatchColor(c.paint)))
            }
            Text(c.label, style = MaterialTheme.typography.labelSmall, color = labelColor)
        }
    }
}

fun swatchColor(hex: String): Color = runCatching { Color(android.graphics.Color.parseColor(hex)) }.getOrDefault(Color.Gray)
