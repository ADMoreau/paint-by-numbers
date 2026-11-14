package io.github.admoreau.paintbynumbers.plan

import android.graphics.Color
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** A small plan from the desktop planner (synthetic landscape, 3 layers over mauve): import, read, draw regions. */
@RunWith(AndroidJUnit4::class)
class PlanTest {
    @Test
    fun importsReadsAndDrawsRegions() {
        val inst = InstrumentationRegistry.getInstrumentation()
        val app = inst.targetContext
        val pkg = File(app.cacheDir, "test_plan.pbn")
        inst.context.assets.open("test_plan.pbn").use { input -> pkg.outputStream().use { input.copyTo(it) } }
        val dirName = PlanStore.import(app, Uri.fromFile(pkg))
        val dir = File(app.filesDir, dirName)
        try {
            val plan = PlanStore.load(dir)
            assertEquals(64, plan.width)
            assertEquals(3, plan.layers.size)
            assertEquals("target.png", plan.targetFile)
            assertTrue(plan.file(plan.targetFile).exists())
            for (layer in plan.layers) {
                assertTrue("${layer.paint} has no thicknesses", layer.thicknesses.isNotEmpty())
                val all = RegionRenderer.render(plan, layer, null)
                assertNotNull(all)
                val px = IntArray(all!!.width * all.height).also { all.getPixels(it, 0, all.width, 0, 0, all.width, all.height) }
                val painted = px.count { Color.alpha(it) > 0 }.toDouble() / px.size
                val expected = layer.thicknesses.sumOf { it.area }
                println("${layer.paint}: painted share %.3f (recipe %.3f), %d thicknesses".format(painted, expected, layer.thicknesses.size))
                assertEquals("painted area of ${layer.paint}", expected, painted, 0.05)
                for (t in layer.thicknesses) assertNotNull(RegionRenderer.render(plan, layer, t))
            }
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun rejectsFilesThatAreNotPlans() {
        val inst = InstrumentationRegistry.getInstrumentation()
        val app = inst.targetContext
        val junk = File(app.cacheDir, "junk.pbn").apply { writeText("not a zip") }
        val before = app.filesDir.list()!!.toSet()
        assertTrue(runCatching { PlanStore.import(app, Uri.fromFile(junk)) }.isFailure)
        assertEquals("a failed import must not leave files behind", before, app.filesDir.list()!!.toSet())
    }
}
