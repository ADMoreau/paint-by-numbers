package io.github.admoreau.paintbynumbers.planner

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The phone planner against the desktop planner's answers on small cases
 * (planner_fixture.json, written by android/scripts/make_planner_fixture.py).
 */
class PlannerParityTest {
    private val fx = JSONObject(javaClass.classLoader!!.getResource("planner_fixture.json")!!.readText())
    private val db = PaintDb.parse(File("../../paint_by_numbers/data/paints_heavy_body.json").readText())

    private fun JSONArray.doubles() = DoubleArray(length()) { getDouble(it) }
    private fun bits(s: String) = BooleanArray(s.length) { s[it] == '1' }

    @Test
    fun colorConversions() {
        val c = fx.getJSONObject("color")
        val srgb = c.getJSONArray("srgb")
        val out = DoubleArray(3)
        for (i in 0 until srgb.length()) {
            val (r, g, b) = srgb.getJSONArray(i).doubles()
            fun check(key: String, tol: Double, f: () -> Unit) {
                f()
                val want = c.getJSONArray(key).getJSONArray(i).doubles()
                for (q in 0 until 3) assertEquals("$key[$i][$q]", want[q], out[q], tol)
            }
            check("srgb_to_lab", 1e-9) { ColorMath.srgbToLab(r, g, b, out) }
            check("srgb_to_working", 1e-12) { ColorMath.srgbToWorking(r, g, b, out) }
            check("working_to_lab", 1e-9) { ColorMath.workingToLab(r, g, b, out) }
            val want = c.getJSONArray("working_to_srgb").getJSONArray(i).doubles()
            ColorMath.workingToSrgb(r, g, b, out)
            for (q in 0 until 3) assertEquals(want[q].coerceIn(0.0, 1.0), out[q], 1e-9)
            val lab = c.getJSONArray("working_to_lab").getJSONArray(i).doubles()
            assertEquals(0.0, ColorMath.deltaE(r, g, b, lab[0], lab[1], lab[2]), 1e-9)
        }
    }

    @Test
    fun paintLayers() {
        val k = fx.getJSONObject("km")
        val rg = k.getJSONArray("rg").doubles()
        val cases = k.getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val cs = cases.getJSONObject(i)
            val co = Km.Coefficients(cs.getJSONArray("K").doubles(), cs.getJSONArray("S").doubles())
            val t = cs.getDouble("t")
            val want = cs.getJSONArray("layer").doubles()
            val m = Km.layerMap(co, t)
            for (q in 0 until 3) {
                assertEquals(want[q], co.layer(rg[q], t, q), 1e-12)
                assertEquals("Möbius form", want[q], (m[q] * rg[q] + m[3 + q]) / (m[6 + q] * rg[q] + 1), 1e-12)
                assertEquals(cs.getJSONArray("masstone").getDouble(q), co.rInf[q], 1e-12)
            }
        }
    }

    @Test
    fun guidedFilter() {
        val g = fx.getJSONObject("guided_filter")
        val w = g.getInt("w")
        val h = g.getInt("h")
        val guide = g.getJSONArray("guide").doubles().let { d -> FloatArray(d.size) { d[it].toFloat() } }
        val p = g.getJSONArray("p").doubles()
        val out = DoubleArray(w * h)
        GuidedFilter(guide, w, h, g.getInt("radius"), g.getDouble("eps")).apply(p, out)
        val want = g.getJSONArray("out").doubles()
        for (i in want.indices) assertEquals("pixel $i", want[i], out[i], 1e-4) // guide stored as float here
    }

    @Test
    fun cleanRegions() {
        val c = fx.getJSONObject("clean")
        val mask = bits(c.getString("mask"))
        Regions.clean(mask, c.getInt("w"), c.getInt("h"), c.getInt("min_area"))
        assertEquals(c.getString("out"), mask.joinToString("") { if (it) "1" else "0" })
    }

    private fun planInputs(): Triple<Target, List<Paint>, DoubleArray> {
        val p = fx.getJSONObject("plan")
        val w = p.getInt("w")
        val h = p.getInt("h")
        val px = p.getJSONArray("pixels").let { a -> IntArray(a.length()) { a.getInt(it) or (0xFF shl 24) } }
        val names = p.getJSONArray("paints").let { a -> List(a.length()) { a.getString(it) } }
        // the fixture's order, with the fixture's K and S (the tests' paints use the rating-only fit)
        val ks = fx.getJSONArray("paints").let { a -> List(a.length()) { a.getJSONObject(it) } }.associateBy { it.getString("name") }
        val paints = names.map { n ->
            db.first { it.name == n }.copy(K = ks.getValue(n).getJSONArray("K").doubles(), S = ks.getValue(n).getJSONArray("S").doubles())
        }
        val bg = DoubleArray(3).also { ColorMath.srgbToWorking(128 / 255.0, 128 / 255.0, 128 / 255.0, it) }
        return Triple(Target.fromArgb(w, h, px), paints, bg)
    }

    private fun config(): UniformConfig {
        val c = fx.getJSONObject("plan").getJSONObject("config")
        return UniformConfig(layers = c.getInt("layers"), nColors = c.getInt("n_colors"), improvePasses = c.getInt("improve_passes"))
    }

    @Test
    fun selectionMatchesDesktop() {
        val p = fx.getJSONObject("plan")
        val (_, paints, bg) = planInputs()
        val centers = p.getJSONArray("centers").let { a -> DoubleArray(3 * a.length()) { a.getJSONArray(it / 3).getDouble(it % 3) } }
        val k = centers.size / 3
        val log = ArrayList<String>()
        val sel = UniformPlanner(paints, bg, config(), { log += it }).select(centers, k)
        log.forEach(::println)
        assertEquals(p.getDouble("baseline"), sel.baseline, 1e-4)
        val stack = p.getJSONArray("stack")
        assertEquals(stack.length(), sel.layers.size)
        for (i in 0 until stack.length()) {
            assertEquals("layer $i paint", stack.getJSONObject(i).getString("paint"), paints[sel.layers[i].first].name)
            assertEquals("layer $i coats", stack.getJSONObject(i).getDouble("coats"), sel.layers[i].second, 0.0)
            assertEquals("layer $i colours", p.getJSONArray("assignment").getString(i), sel.assignment[i].joinToString("") { if (it) "1" else "0" })
        }
        val rounds = p.getJSONArray("rounds")
        for (i in 0 until rounds.length()) assertEquals(rounds.getJSONObject(i).getDouble("rms_error"), sel.rounds[i].rmsError, 1e-3)
    }

    @Test
    fun regionsMatchDesktop() {
        val p = fx.getJSONObject("plan")
        val (target, paints, bg) = planInputs()
        val centers = p.getJSONArray("centers").let { a -> DoubleArray(3 * a.length()) { a.getJSONArray(it / 3).getDouble(it % 3) } }
        val labels = p.getJSONArray("labels").let { a -> IntArray(a.length()) { a.getInt(it) } }
        val k = centers.size / 3
        val planner = UniformPlanner(paints, bg, config())
        val plan = planner.plan(target, KMeans.Result(centers, k, labels))
        println("mean ΔE %.3f (desktop %.3f), from colours %.3f (desktop %.3f)".format(
            plan.meanError, p.getDouble("mean_error"), plan.meanErrorFromColors, p.getDouble("mean_error_from_colors")))
        assertEquals(p.getDouble("mean_error_from_colors"), plan.meanErrorFromColors, 1e-3)
        assertEquals(p.getDouble("mean_error"), plan.meanError, 0.05)
        val regions = p.getJSONArray("regions")
        assertEquals(regions.length(), plan.layers.size)
        for (i in 0 until regions.length()) {
            val want = bits(regions.getString(i))
            val same = want.indices.count { want[it] == plan.regions[i][it] }
            println("layer $i: ${want.size - same} of ${want.size} pixels differ")
            assertTrue("layer $i: ${want.size - same} pixels differ", same >= 0.99 * want.size)
        }
    }

    @Test
    fun ownKMeansPlanIsAsGood() {
        val p = fx.getJSONObject("plan")
        val (target, paints, bg) = planInputs()
        val plan = UniformPlanner(paints, bg, config()).plan(target)
        println("own k-means: ${plan.clusterCount} colours, mean ΔE %.3f (desktop %.3f)".format(plan.meanError, p.getDouble("mean_error")))
        assertTrue(plan.meanError < p.getDouble("mean_error") * 1.15)
    }
}

class CubeRootTest {
    @Test
    fun matchesMathCbrt() {
        var worst = 0.0
        var x = 0.008
        while (x < 2.0) {
            worst = maxOf(worst, kotlin.math.abs(ColorMath.cubeRoot(x) / Math.cbrt(x) - 1))
            x *= 1.0001
        }
        assertTrue("worst relative error $worst", worst < 1e-13)
    }
}
