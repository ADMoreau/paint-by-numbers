package io.github.admoreau.paintbynumbers.planner

import org.json.JSONArray
import kotlin.math.sqrt
import kotlin.math.tanh

/** A paint from the desktop planner's database (paint_by_numbers/data/paints_heavy_body.json). */
data class Paint(
    val name: String,
    val item: Int,
    val transparency: String,
    val lab: DoubleArray,
    val K: DoubleArray,
    val S: DoubleArray,
    val sheen: String? = null,
    val lightfastness: String? = null,
    val discontinued: Boolean = false,
    val special: String? = null,
) {
    val slug: String get() = slugify(name)
}

/** 'Phthalo Blue (Green Shade)' -> 'phthalo-blue-green-shade'. */
fun slugify(name: String) = name.lowercase().replace(Regex("[^a-z0-9]+"), "-").trim('-')

object PaintDb {
    const val ASSET = "paints_heavy_body.json"

    /** Fitted paints from the database's JSON text; interference colours (Lab all zero) are skipped. */
    fun parse(json: String): List<Paint> {
        val a = JSONArray(json)
        val out = ArrayList<Paint>()
        for (i in 0 until a.length()) {
            val r = a.getJSONObject(i)
            val lab = r.getJSONArray("lab").let { l -> DoubleArray(3) { l.getDouble(it) } }
            if (lab.all { it == 0.0 }) continue
            val k = r.optJSONArray("K") ?: continue
            val s = r.optJSONArray("S") ?: continue
            out += Paint(
                name = r.getString("name"),
                item = r.optInt("item"),
                transparency = r.getString("transparency"),
                lab = lab,
                K = DoubleArray(3) { k.getDouble(it) },
                S = DoubleArray(3) { s.getDouble(it) },
                sheen = r.optString("sheen").takeIf { r.has("sheen") && !r.isNull("sheen") },
                lightfastness = r.optString("lightfastness").takeIf { r.has("lightfastness") && !r.isNull("lightfastness") },
                discontinued = r.optBoolean("discontinued", false),
                special = r.optString("special").takeIf { r.has("special") && !r.isNull("special") },
            )
        }
        return out
    }

    /** Paints by name, ignoring case and punctuation, in database order. */
    fun select(paints: List<Paint>, names: Collection<String>): List<Paint> {
        val wanted = names.map { slugify(it) }.toSet()
        val known = paints.map { it.slug }.toSet()
        val missing = wanted - known
        require(missing.isEmpty()) { "no paint named ${missing.joinToString()}" }
        return paints.filter { it.slug in wanted }
    }
}

/** Most opaque first. */
val TRANSPARENCY_RATINGS = listOf("opaque", "semi-opaque", "semi-transparent", "transparent")

/** Pairs out of the preferred order: opaque below transparent; lighter below darker. */
fun orderViolations(stack: List<Paint>): Int {
    var count = 0
    for (i in stack.indices) for (j in i + 1 until stack.size) {
        val lo = TRANSPARENCY_RATINGS.indexOf(stack[i].transparency)
        val hi = TRANSPARENCY_RATINGS.indexOf(stack[j].transparency)
        if (lo > hi || (lo == hi && stack[i].lab[0] < stack[j].lab[0])) count++
    }
    return count
}

/**
 * Kubelka-Munk paint layers (km.py). Reflectances are working-space (linear Rec.2020), one value
 * per channel; thickness is in coats.
 */
object Km {
    const val MIN_KS = 1e-4

    /** Per-paint constants per channel: a, b, masstone R_inf, b*S. */
    class Coefficients(K: DoubleArray, S: DoubleArray) {
        val a = DoubleArray(3)
        val b = DoubleArray(3)
        val rInf = DoubleArray(3)
        val bs = DoubleArray(3)

        init {
            for (c in 0 until 3) {
                val k0 = maxOf(K[c], MIN_KS)
                val s0 = maxOf(S[c], MIN_KS)
                val ks = k0 / s0
                a[c] = 1.0 + ks
                b[c] = sqrt(ks * (ks + 2.0))
                rInf[c] = 1.0 / (a[c] + b[c])
                bs[c] = b[c] * s0
            }
        }

        /** One layer [t] coats thick over reflectance [rg], channel [c] (the cancellation-free form of km.py). */
        fun layer(rg: Double, t: Double, c: Int): Double {
            val th = tanh(bs[c] * t)
            val w = th * (a[c] + b[c] - rg) / ((a[c] - rg) * th + b[c])
            return rg + (rInf[c] - rg) * w
        }
    }

    /**
     * A layer [t] coats thick as a map of the reflectance R beneath it: (m00 R + m01) / (m10 R + 1),
     * per channel. Returned as [m00 r, m00 g, m00 b, m01 r, ..., m10 b]. Such maps compose like 2x2
     * matrices, so a run of layers is again one map of this form.
     */
    fun layerMap(co: Coefficients, t: Double): DoubleArray {
        val m = DoubleArray(9)
        for (c in 0 until 3) {
            val th = tanh(co.bs[c] * t)
            val d = co.a[c] * th + co.b[c]
            m[c] = (co.b[c] - th * (co.rInf[c] + co.b[c])) / d
            m[3 + c] = th / d
            m[6 + c] = -th / d
        }
        return m
    }

    fun masstone(K: DoubleArray, S: DoubleArray) = Coefficients(K, S).rInf.copyOf()
}
