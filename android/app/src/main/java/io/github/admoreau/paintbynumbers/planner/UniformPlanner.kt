package io.github.admoreau.paintbynumbers.planner

import kotlin.math.max
import kotlin.math.sqrt

/**
 * One weight per layer, ported from the desktop planner's uniform.py: a painting planned as N
 * layers, each one paint at one weight (such as 1/4 coat) over one region. Gradations come from
 * stacking layers rather than from painting several dilutions side by side.
 *
 * 1. Cluster the image's colours (k-means, [UniformConfig.nColors]).
 * 2. Greedily add the (paint, weight, position) that lowers the RMS ΔE over the colours most,
 *    deciding which colours each layer covers by coordinate descent; then try swapping each layer.
 * 3. At full resolution, run the same descent per pixel, with each layer's on-minus-off cost
 *    smoothed by a guided filter so regions come out as shapes that follow the image's edges;
 *    then drop specks and fill pinholes.
 */
data class UniformConfig(
    val layers: Int = 8,
    val weights: List<Double> = listOf(0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    val maxPaints: Int? = null,
    val nColors: Int = 64,
    val seed: Int = 0,
    val paintCost: Double = 0.1, // ΔE charged per coat
    val minGain: Double = 0.05, // RMS ΔE a layer must save to be added
    val improvePasses: Int = 2,
    val sweeps: Int = 10,
    val smooth: Double = 0.008, // region smoothing radius, fraction of the long side; 0 = off
    val edge: Double = 5.0, // ΔE: image colour changes bigger than this are edges regions may follow
    val minRegion: Double = 0.0003, // smallest patch or hole kept, fraction of the image area
)

/** The image to plan: Lab per pixel (3 per pixel, row-major) and a key per pixel equal for equal colours. */
class Target(val w: Int, val h: Int, val lab: FloatArray, val colorKey: IntArray) {
    companion object {
        /** From packed ARGB pixels (as Bitmap.getPixels gives). */
        fun fromArgb(w: Int, h: Int, argb: IntArray): Target {
            val lab = FloatArray(3 * w * h)
            val cache = HashMap<Int, FloatArray>()
            val out = DoubleArray(3)
            val key = IntArray(w * h)
            for (p in 0 until w * h) {
                val rgb = argb[p] and 0xFFFFFF
                key[p] = rgb
                val v = cache.getOrPut(rgb) {
                    ColorMath.srgbToLab(((rgb shr 16) and 255) / 255.0, ((rgb shr 8) and 255) / 255.0, (rgb and 255) / 255.0, out)
                    floatArrayOf(out[0].toFloat(), out[1].toFloat(), out[2].toFloat())
                }
                v.copyInto(lab, 3 * p)
            }
            return Target(w, h, lab, key)
        }
    }
}

/** A chosen layer: a paint at a weight. */
data class Layer(val paint: Paint, val coats: Double)

class UniformPlan(
    val w: Int,
    val h: Int,
    val layers: List<Layer>, // bottom first
    val regions: List<BooleanArray>, // per layer, w*h
    val background: DoubleArray, // working-space reflectance
    val preview: FloatArray, // working-space reflectance, 3 per pixel
    val error: FloatArray, // ΔE per pixel
    val meanError: Double,
    val maxError: Double,
    val meanErrorFromColors: Double, // before smoothing the regions
    val baselineError: Double, // RMS over the colours, bare background
    val clusterCount: Int,
    val rounds: List<Round>,
) {
    class Round(val paint: String, val coats: Double, val position: Int, val rmsError: Double)
}

/** The layer choice on the image's colours (see [UniformPlanner.select]). */
class Selection(
    val layers: List<Pair<Int, Double>>, // (paint index, weight), bottom first
    val assignment: List<BooleanArray>, // per layer, which colours it covers
    val rmsError: Double,
    val baseline: Double,
    val rounds: List<UniformPlan.Round>,
)

class UniformPlanner(
    private val paints: List<Paint>,
    private val background: DoubleArray,
    private val config: UniformConfig = UniformConfig(),
    private val progress: (String) -> Unit = {},
    private val cancelled: () -> Boolean = { false },
) {
    class Cancelled : RuntimeException("planning cancelled")

    private val weights = config.weights.map { it }.toSortedSet().toList()
    private val coefs = paints.map { Km.Coefficients(it.K, it.S) }
    private val maps = coefs.map { co -> weights.map { Km.layerMap(co, it) } } // [paint][weight index]

    init {
        require(paints.isNotEmpty()) { "no paints to choose from" }
        require(config.layers >= 1) { "layers must be at least 1" }
        require(weights.isNotEmpty() && weights[0] > 0) { "weights must be positive" }
        require(config.maxPaints == null || config.maxPaints >= 1) { "maxPaints must be at least 1" }
    }

    private fun check() {
        if (cancelled()) throw Cancelled()
    }

    fun plan(target: Target, clusters: KMeans.Result? = null): UniformPlan {
        val km = clusters ?: run {
            progress("finding the image's main colours")
            KMeans.cluster(target.lab, target.colorKey, config.nColors, config.seed)
        }
        check()
        val sel = select(km.centers, km.k)
        progress("drawing regions for ${sel.layers.size} layers at ${target.w}x${target.h}")
        return solve(target, km.labels, sel, km.k)
    }

    // ------------------------------------------------------------ choosing the layers

    /** A candidate stack as parallel arrays, bottom first. */
    private class Stack(val paint: IntArray, val weight: IntArray)

    /**
     * Coordinate descent for one colour over a stack: x (on/off per layer) is changed in place.
     * Returns the ΔE of the result and fills [costOut] (ΔE plus paint cost).
     */
    private fun descendColor(st: Stack, x: BooleanArray, l: Double, a: Double, b: Double, buf: DoubleArray): Double {
        val n = x.size
        val below = buf // 3 * n
        for (sweep in 0 until config.sweeps) {
            var r0 = background[0]
            var r1 = background[1]
            var r2 = background[2]
            for (i in 0 until n) {
                below[3 * i] = r0
                below[3 * i + 1] = r1
                below[3 * i + 2] = r2
                if (x[i]) {
                    val m = maps[st.paint[i]][st.weight[i]]
                    r0 = (m[0] * r0 + m[3]) / (m[6] * r0 + 1)
                    r1 = (m[1] * r1 + m[4]) / (m[7] * r1 + 1)
                    r2 = (m[2] * r2 + m[5]) / (m[8] * r2 + 1)
                }
            }
            // the layers above, composed: R -> (a00 R + a01) / (a10 R + 1), per channel
            var a00r = 1.0; var a00g = 1.0; var a00b = 1.0
            var a01r = 0.0; var a01g = 0.0; var a01b = 0.0
            var a10r = 0.0; var a10g = 0.0; var a10b = 0.0
            var coats = 0.0
            for (i in 0 until n) if (x[i]) coats += weights[st.weight[i]]
            var changed = false
            for (i in n - 1 downTo 0) {
                val m = maps[st.paint[i]][st.weight[i]]
                val wi = weights[st.weight[i]]
                val rb0 = below[3 * i]
                val rb1 = below[3 * i + 1]
                val rb2 = below[3 * i + 2]
                val others = if (x[i]) coats - wi else coats
                val off = ColorMath.deltaE(
                    (a00r * rb0 + a01r) / (a10r * rb0 + 1), (a00g * rb1 + a01g) / (a10g * rb1 + 1), (a00b * rb2 + a01b) / (a10b * rb2 + 1), l, a, b,
                ) + config.paintCost * others
                val p0 = (m[0] * rb0 + m[3]) / (m[6] * rb0 + 1)
                val p1 = (m[1] * rb1 + m[4]) / (m[7] * rb1 + 1)
                val p2 = (m[2] * rb2 + m[5]) / (m[8] * rb2 + 1)
                val on = ColorMath.deltaE(
                    (a00r * p0 + a01r) / (a10r * p0 + 1), (a00g * p1 + a01g) / (a10g * p1 + 1), (a00b * p2 + a01b) / (a10b * p2 + 1), l, a, b,
                ) + config.paintCost * (others + wi)
                val new = on - off < 0
                if (new != x[i]) changed = true
                x[i] = new
                coats = if (new) others + wi else others
                if (new) {
                    // A <- A . M_i
                    var p11 = a10r * m[3] + 1
                    val n00r = (a00r * m[0] + a01r * m[6]) / p11; val n01r = (a00r * m[3] + a01r) / p11; val n10r = (a10r * m[0] + m[6]) / p11
                    p11 = a10g * m[4] + 1
                    val n00g = (a00g * m[1] + a01g * m[7]) / p11; val n01g = (a00g * m[4] + a01g) / p11; val n10g = (a10g * m[1] + m[7]) / p11
                    p11 = a10b * m[5] + 1
                    val n00b = (a00b * m[2] + a01b * m[8]) / p11; val n01b = (a00b * m[5] + a01b) / p11; val n10b = (a10b * m[2] + m[8]) / p11
                    a00r = n00r; a01r = n01r; a10r = n10r
                    a00g = n00g; a01g = n01g; a10g = n10g
                    a00b = n00b; a01b = n01b; a10b = n10b
                }
            }
            if (!changed) break
        }
        return renderError(st, x, l, a, b)
    }

    /** ΔE of the stack rendered with the stable single-layer formula (as km.render_stack). */
    private fun renderError(st: Stack, x: BooleanArray, l: Double, a: Double, b: Double): Double {
        var r0 = background[0]
        var r1 = background[1]
        var r2 = background[2]
        for (i in x.indices) if (x[i]) {
            val co = coefs[st.paint[i]]
            val t = weights[st.weight[i]]
            r0 = co.layer(r0, t, 0)
            r1 = co.layer(r1, t, 1)
            r2 = co.layer(r2, t, 2)
        }
        return ColorMath.deltaE(r0, r1, r2, l, a, b)
    }

    /**
     * Best assignment and RMS ΔE over the colours for each candidate stack. Each colour starts
     * from every assignment [starts] gives (stack index, start index -> colour-major n*k array)
     * and keeps the lowest-cost result.
     */
    private fun evaluate(stacks: List<Stack>, centers: DoubleArray, k: Int, nStarts: Int, start: (Int, Int, BooleanArray) -> Unit): Pair<List<BooleanArray>, DoubleArray> {
        val n = stacks[0].paint.size
        val results = arrayOfNulls<BooleanArray>(stacks.size)
        val errors = DoubleArray(stacks.size)
        Parallel.forEach(stacks.size) { s ->
            if (s % 64 == 0) check()
            val st = stacks[s]
            val best = BooleanArray(n * k)
            val buf = DoubleArray(3 * n)
            val x0 = BooleanArray(n * k)
            val x = BooleanArray(n)
            val bestErr = DoubleArray(k)
            val bestCost = DoubleArray(k) { Double.POSITIVE_INFINITY }
            for (si in 0 until nStarts) {
                start(s, si, x0)
                for (c in 0 until k) {
                    for (i in 0 until n) x[i] = x0[i * k + c]
                    val err = descendColor(st, x, centers[3 * c], centers[3 * c + 1], centers[3 * c + 2], buf)
                    var coats = 0.0
                    for (i in 0 until n) if (x[i]) coats += weights[st.weight[i]]
                    val cost = err + config.paintCost * coats
                    if (cost < bestCost[c]) {
                        bestCost[c] = cost
                        bestErr[c] = err
                        for (i in 0 until n) best[i * k + c] = x[i]
                    }
                }
            }
            var ss = 0.0
            for (c in 0 until k) ss += bestErr[c] * bestErr[c]
            errors[s] = sqrt(ss / k)
            results[s] = best
        }
        return results.map { it!! } to errors
    }

    private fun allowed(stack: List<Pair<Int, Int>>): List<Pair<Int, Int>> {
        val used = stack.map { it.first }.toSet()
        val ps = if (config.maxPaints != null && used.size >= config.maxPaints) used.sorted() else paints.indices.toList()
        return ps.flatMap { p -> weights.indices.map { p to it } }
    }

    private fun toStack(s: List<Pair<Int, Int>>) = Stack(IntArray(s.size) { s[it].first }, IntArray(s.size) { s[it].second })

    /** Lowest error; near-ties go to the stack with fewer ordering violations (opaque below transparent). */
    private fun pick(errors: DoubleArray, stacks: List<List<Pair<Int, Int>>>): Int {
        val best = errors.min()
        return errors.indices.filter { errors[it] <= best + TIE_DELTA_E }
            .minWith(compareBy<Int>({ orderViolations(stacks[it].map { p -> paints[p.first] }) }, { errors[it] }))
    }

    /** Chooses up to [UniformConfig.layers] layers on the colours [centers] (k x 3 Lab), bottom first. */
    fun select(centers: DoubleArray, k: Int): Selection {
        var ss = 0.0
        for (c in 0 until k) {
            val e = ColorMath.deltaE(background[0], background[1], background[2], centers[3 * c], centers[3 * c + 1], centers[3 * c + 2])
            ss += e * e
        }
        val baseline = sqrt(ss / k)
        progress("background alone: RMS error %.2f".format(baseline))
        val rounds = ArrayList<UniformPlan.Round>()
        var stack = listOf<Pair<Int, Int>>()
        var x = listOf<BooleanArray>() // per layer, per colour
        var current = baseline

        fun flat(layers: List<BooleanArray>) = BooleanArray(layers.size * k).also { f -> layers.forEachIndexed { i, l -> l.copyInto(f, i * k) } }
        fun unflat(f: BooleanArray, n: Int) = List(n) { i -> f.copyOfRange(i * k, (i + 1) * k) }

        while (stack.size < config.layers) {
            check()
            val cands = (0..stack.size).flatMap { pos -> allowed(stack).map { it to pos } }
            val stacks = cands.map { (pw, pos) -> stack.subList(0, pos) + pw + stack.subList(pos, stack.size) }
            val (xs, errors) = evaluate(stacks.map { toStack(it) }, centers, k, 2) { s, si, out ->
                val pos = cands[s].second
                for (i in 0..stack.size) {
                    for (c in 0 until k) out[i * k + c] = when {
                        i < pos -> x[i][c]
                        i == pos -> si == 1
                        else -> x[i - 1][c]
                    }
                }
            }
            val best = pick(errors, stacks)
            val gain = current - errors[best]
            if (gain < config.minGain) {
                progress("stop at ${stack.size} layers: another layer would save only %.2f".format(gain))
                break
            }
            val (pw, pos) = cands[best]
            stack = stacks[best]
            x = unflat(xs[best], stack.size)
            current = errors[best]
            rounds += UniformPlan.Round(paints[pw.first].name, weights[pw.second], pos, current)
            progress("layer ${stack.size}: ${paints[pw.first].name} at ${fmt(weights[pw.second])} coats, RMS error %.2f".format(current))
        }

        for (pass in 0 until config.improvePasses) {
            var improved = false
            for (i in stack.indices) {
                check()
                val options = allowed(stack.subList(0, i) + stack.subList(i + 1, stack.size)).filter { it != stack[i] }
                if (options.isEmpty()) continue
                val stacks = options.map { stack.subList(0, i) + it + stack.subList(i + 1, stack.size) }
                val cur = flat(x)
                val (xs, errors) = evaluate(stacks.map { toStack(it) }, centers, k, 2) { _, si, out ->
                    cur.copyInto(out)
                    for (c in 0 until k) out[i * k + c] = si == 1
                }
                var best = 0
                for (j in errors.indices) if (errors[j] < errors[best]) best = j
                if (errors[best] < current - 1e-3) {
                    val old = stack[i]
                    stack = stacks[best]
                    x = unflat(xs[best], stack.size)
                    current = errors[best]
                    improved = true
                    progress("improve: layer ${i + 1} ${paints[old.first].name} ${fmt(weights[old.second])} -> " +
                        "${paints[stack[i].first].name} ${fmt(weights[stack[i].second])} coats, RMS error %.2f".format(current))
                }
            }
            if (!improved) break
        }
        return Selection(stack.map { it.first to weights[it.second] }, x, current, baseline, rounds)
    }

    // ------------------------------------------------------------ regions at full resolution

    /** Each layer's region per pixel, starting from its colours' assignment; layers left empty are dropped. */
    fun solve(target: Target, labels: IntArray, sel: Selection, k: Int): UniformPlan {
        val w = target.w
        val h = target.h
        val np = w * h
        var layerIdx = sel.layers.map { (p, wt) -> p to weights.indexOf(wt) }
        var n = layerIdx.size
        var x = List(n) { i -> BooleanArray(np) { sel.assignment[i][labels[it]] } }
        val err0 = pixelErrors(target, layerIdx, x)
        val radius = Math.rint(config.smooth * max(w, h)).toInt()
        val filter = if (radius > 0) GuidedFilter(target.lab, w, h, radius, config.edge * config.edge) else null
        val minArea = Math.rint(config.minRegion * w * h).toInt()
        repeat(2) {
            descendPixels(target, layerIdx, x, filter)
            for (m in x) Regions.clean(m, w, h, minArea)
        }
        val keep = x.indices.filter { i -> x[i].any { it } }
        if (keep.size < n) {
            progress("dropping ${n - keep.size} layer(s) left with no region after smoothing")
            layerIdx = keep.map { layerIdx[it] }
            x = keep.map { x[it] }
            n = keep.size
        }
        val keptRounds = sel.rounds
        val preview = FloatArray(3 * np)
        val err = FloatArray(np)
        render(target, layerIdx, x, preview, err)
        val mean = err.sumOf { it.toDouble() } / np
        val mean0 = err0.sumOf { it.toDouble() } / np
        progress("regions: mean ΔE %.2f from the colours' assignment, %.2f after smoothing".format(mean0, mean))
        return UniformPlan(
            w, h, layerIdx.map { (p, wi) -> Layer(paints[p], weights[wi]) }, x, background.copyOf(), preview, err,
            mean, err.maxOrNull()?.toDouble() ?: 0.0, mean0, sel.baseline, k, keptRounds,
        )
    }

    private fun render(target: Target, layers: List<Pair<Int, Int>>, x: List<BooleanArray>, preview: FloatArray?, err: FloatArray) {
        Parallel.forChunks(target.w * target.h) { from, to ->
            for (p in from until to) {
                var r0 = background[0]
                var r1 = background[1]
                var r2 = background[2]
                for (i in layers.indices) if (x[i][p]) {
                    val co = coefs[layers[i].first]
                    val t = weights[layers[i].second]
                    r0 = co.layer(r0, t, 0)
                    r1 = co.layer(r1, t, 1)
                    r2 = co.layer(r2, t, 2)
                }
                if (preview != null) {
                    preview[3 * p] = r0.toFloat()
                    preview[3 * p + 1] = r1.toFloat()
                    preview[3 * p + 2] = r2.toFloat()
                }
                err[p] = ColorMath.deltaE(r0, r1, r2, target.lab[3 * p].toDouble(), target.lab[3 * p + 1].toDouble(), target.lab[3 * p + 2].toDouble()).toFloat()
            }
        }
    }

    private fun pixelErrors(target: Target, layers: List<Pair<Int, Int>>, x: List<BooleanArray>) =
        FloatArray(target.w * target.h).also { render(target, layers, x, null, it) }

    /**
     * The per-pixel coordinate descent (uniform._descend with smoothing). Each sweep goes top layer
     * first; the layers above the current one are carried per pixel as one composed map.
     */
    private fun descendPixels(target: Target, layers: List<Pair<Int, Int>>, x: List<BooleanArray>, filter: GuidedFilter?) {
        val n = layers.size
        val np = target.w * target.h
        val lab = target.lab
        val below = Array(n) { FloatArray(3 * np) }
        val a = Array(9) { FloatArray(np) } // a00 rgb, a01 rgb, a10 rgb
        val coats = FloatArray(np)
        val d = DoubleArray(np)
        for (sweep in 0 until config.sweeps) {
            check()
            Parallel.forChunks(np) { from, to ->
                for (p in from until to) {
                    var r0 = background[0]
                    var r1 = background[1]
                    var r2 = background[2]
                    var c = 0.0
                    for (i in 0 until n) {
                        val b = below[i]
                        b[3 * p] = r0.toFloat()
                        b[3 * p + 1] = r1.toFloat()
                        b[3 * p + 2] = r2.toFloat()
                        if (x[i][p]) {
                            val m = maps[layers[i].first][layers[i].second]
                            r0 = (m[0] * r0 + m[3]) / (m[6] * r0 + 1)
                            r1 = (m[1] * r1 + m[4]) / (m[7] * r1 + 1)
                            r2 = (m[2] * r2 + m[5]) / (m[8] * r2 + 1)
                            c += weights[layers[i].second]
                        }
                    }
                    coats[p] = c.toFloat()
                    for (q in 0 until 3) {
                        a[q][p] = 1f
                        a[3 + q][p] = 0f
                        a[6 + q][p] = 0f
                    }
                }
            }
            var changed = false
            for (i in n - 1 downTo 0) {
                val m = maps[layers[i].first][layers[i].second]
                val wi = weights[layers[i].second]
                val xi = x[i]
                val b = below[i]
                Parallel.forChunks(np) { from, to ->
                    for (p in from until to) {
                        val others = if (xi[p]) coats[p] - wi else coats[p].toDouble()
                        var offRgb0 = 0.0; var offRgb1 = 0.0; var offRgb2 = 0.0
                        var onRgb0 = 0.0; var onRgb1 = 0.0; var onRgb2 = 0.0
                        for (q in 0 until 3) {
                            val rb = b[3 * p + q].toDouble()
                            val pr = (m[q] * rb + m[3 + q]) / (m[6 + q] * rb + 1)
                            val a00 = a[q][p]
                            val a01 = a[3 + q][p]
                            val a10 = a[6 + q][p]
                            val off = (a00 * rb + a01) / (a10 * rb + 1)
                            val on = (a00 * pr + a01) / (a10 * pr + 1)
                            when (q) {
                                0 -> { offRgb0 = off; onRgb0 = on }
                                1 -> { offRgb1 = off; onRgb1 = on }
                                else -> { offRgb2 = off; onRgb2 = on }
                            }
                        }
                        val l = lab[3 * p].toDouble()
                        val la = lab[3 * p + 1].toDouble()
                        val lb = lab[3 * p + 2].toDouble()
                        val costOff = ColorMath.deltaE(offRgb0, offRgb1, offRgb2, l, la, lb) + config.paintCost * others
                        val costOn = ColorMath.deltaE(onRgb0, onRgb1, onRgb2, l, la, lb) + config.paintCost * (others + wi)
                        d[p] = costOn - costOff
                    }
                }
                filter?.apply(d, d)
                val chunkChanged = java.util.concurrent.atomic.AtomicBoolean(false)
                Parallel.forChunks(np) { from, to ->
                    var ch = false
                    for (p in from until to) {
                        val new = d[p] < 0
                        if (new != xi[p]) ch = true
                        val others = if (xi[p]) coats[p] - wi else coats[p].toDouble()
                        xi[p] = new
                        coats[p] = (if (new) others + wi else others).toFloat()
                        if (new) for (q in 0 until 3) {
                            val a00 = a[q][p].toDouble()
                            val a01 = a[3 + q][p].toDouble()
                            val a10 = a[6 + q][p].toDouble()
                            val p11 = a10 * m[3 + q] + 1
                            a[q][p] = ((a00 * m[q] + a01 * m[6 + q]) / p11).toFloat()
                            a[3 + q][p] = ((a00 * m[3 + q] + a01) / p11).toFloat()
                            a[6 + q][p] = ((a10 * m[q] + m[6 + q]) / p11).toFloat()
                        }
                    }
                    if (ch) chunkChanged.set(true)
                }
                if (chunkChanged.get()) changed = true
            }
            if (!changed) break
        }
    }

    companion object {
        const val TIE_DELTA_E = 0.05

        /** 0.125 -> "0.125", 1.0 -> "1", 1.5 -> "1.5" (Python's "{:g}" for these values). */
        fun fmt(v: Double): String = if (v == Math.rint(v)) v.toLong().toString() else v.toBigDecimal().stripTrailingZeros().toPlainString()
    }
}
