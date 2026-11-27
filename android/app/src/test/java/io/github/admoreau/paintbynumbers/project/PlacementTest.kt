package io.github.admoreau.paintbynumbers.project

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class PlacementTest {
    private fun corner(m: DoubleArray, x: Double, y: Double) = doubleArrayOf(m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])

    @Test
    fun samePictureAtAnotherSizeStaysPut() {
        val pl = Placement(cx = 120.0, cy = 80.0, scale = 0.25, angle = 0.3)
        val moved = pl.forImage(1800, 1350, 512, 384)!!
        val a = pl.matrix(1800, 1350)
        val b = moved.matrix(512, 384)
        for ((x0, y0) in listOf(0.0 to 0.0, 1.0 to 0.0, 0.0 to 1.0, 1.0 to 1.0)) {
            val p = corner(a, x0 * 1800, y0 * 1350)
            val q = corner(b, x0 * 512, y0 * 384)
            assertEquals(p[0], q[0], 1e-9)
            assertEquals(p[1], q[1], 1e-9)
        }
    }

    @Test
    fun differentShapeIsPlacedAfresh() {
        assertNull(Placement(0.0, 0.0, 1.0, 0.0).forImage(1800, 1350, 1000, 1000))
        assertNull(Placement(0.0, 0.0, 1.0, 0.0).forImage(0, 0, 512, 384))
    }
}
