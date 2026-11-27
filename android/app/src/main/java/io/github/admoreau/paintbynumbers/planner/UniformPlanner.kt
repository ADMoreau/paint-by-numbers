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
    val background: DoubleArray, // working-space reflectance, Bands.N values
    val preview: FloatArray, // working-space reflectance, Bands.N per pixel
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

    /** Per-thread work arrays for [descendColor], sized for [n] layers of [Bands.N] bands. */
    private class Scratch(n: Int) {
        val below = DoubleArray(N * n) // the reflectance under each layer
        val r = DoubleArray(N)
        val a00 = DoubleArray(N) // the layers above, composed: R -> (a00 R + a01) / (a10 R + 1)
        val a01 = DoubleArray(N)
        val a10 = DoubleArray(N)
        val off = DoubleArray(N)
        val on = DoubleArray(N)
    }

    /**
     * Coordinate descent for one colour over a stack: x (on/off per layer) is changed in place.
     * Returns the ΔE of the result.
     */
    private fun descendColor(st: Stack, x: BooleanArray, l: Double, a: Double, b: Double, sc: Scratch): Double {
        val n = x.size
        val below = sc.below
        val r = sc.r
        val a00 = sc.a00
        val a01 = sc.a01
        val a10 = sc.a10
        val off = sc.off
        val on = sc.on
        for (sweep in 0 until config.sweeps) {
            background.copyInto(r)
            for (i in 0 until n) {
                r.copyInto(below, i * N)
                if (x[i]) {
                    val m = maps[st.paint[i]][st.weight[i]]
                    for (c in 0 until N) r[c] = (m[c] * r[c] + m[N + c]) / (m[2 * N + c] * r[c] + 1)
                }
            }
            a00.fill(1.0)
            a01.fill(0.0)
            a10.fill(0.0)
            var coats = 0.0
            for (i in 0 until n) if (x[i]) coats += weights[st.weight[i]]
            var changed = false
            for (i in n - 1 downTo 0) {
                val m = maps[st.paint[i]][st.weight[i]]
                val wi = weights[st.weight[i]]
                val base = i * N
                for (c in 0 until N) {
                    val rb = below[base + c]
                    off[c] = (a00[c] * rb + a01[c]) / (a10[c] * rb + 1)
                    val pr = (m[c] * rb + m[N + c]) / (m[2 * N + c] * rb + 1)
                    on[c] = (a00[c] * pr + a01[c]) / (a10[c] * pr + 1)
                }
                val others = if (x[i]) coats - wi else coats
                val costOff = ColorMath.deltaE(off, 0, l, a, b) + config.paintCost * others
                val costOn = ColorMath.deltaE(on, 0, l, a, b) + config.paintCost * (others + wi)
                val new = costOn - costOff < 0
                if (new != x[i]) changed = true
                x[i] = new
                coats = if (new) others + wi else others
                if (new) for (c in 0 until N) { // A <- A . M_i
                    val p11 = a10[c] * m[N + c] + 1
                    val n00 = (a00[c] * m[c] + a01[c] * m[2 * N + c]) / p11
                    val n01 = (a00[c] * m[N + c] + a01[c]) / p11
                    val n10 = (a10[c] * m[c] + m[2 * N + c]) / p11
                    a00[c] = n00; a01[c] = n01; a10[c] = n10
                }
            }
            if (!changed) break
        }
        return renderError(st, x, l, a, b, r)
    }

    /** ΔE of the stack rendered with the stable single-layer formula (as km.render_stack); [r] is scratch. */
    private fun renderError(st: Stack, x: BooleanArray, l: Double, a: Double, b: Double, r: DoubleArray): Double {
        background.copyInto(r)
        for (i in x.indices) if (x[i]) {
            val co = coefs[st.paint[i]]
            val t = weights[st.weight[i]]
            for (c in 0 until N) r[c] = co.layer(r[c], t, c)
        }
        return ColorMath.deltaE(r, 0, l, a, b)
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
            val buf = Scratch(n)
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
            val e = ColorMath.deltaE(background, 0, centers[3 * c], centers[3 * c + 1], centers[3 * c + 2])
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
        val preview = FloatArray(N * np)
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
            val r = DoubleArray(N)
            for (p in from until to) {
                background.copyInto(r)
                for (i in layers.indices) if (x[i][p]) {
                    val co = coefs[layers[i].first]
                    val t = weights[layers[i].second]
                    for (c in 0 until N) r[c] = co.layer(r[c], t, c)
                }
                if (preview != null) for (c in 0 until N) preview[N * p + c] = r[c].toFloat()
                err[p] = ColorMath.deltaE(r, 0, target.lab[3 * p].toDouble(), target.lab[3 * p + 1].toDouble(), target.lab[3 * p + 2].toDouble()).toFloat()
            }
        }
    }

    private fun pixelErrors(target: Target, layers: List<Pair<Int, Int>>, x: List<BooleanArray>) =
        FloatArray(target.w * target.h).also { render(target, layers, x, null, it) }

    /**
     * The per-pixel coordinate descent (uniform._descend with smoothing). Each sweep goes top layer
     * first; the layers above the current one are carried per pixel as one composed map.
     *
     * The reflectance under each layer is kept only under every [CHECKPOINT]th layer (with
     * [Bands.N] values per pixel, every layer would be over 100 MB for a large plan) and rebuilt
     * from the checkpoint below as each layer comes up; the layers below the current one don't
     * change until the sweep reaches them, so the rebuilt values are the same.
     */
    private fun descendPixels(target: Target, layers: List<Pair<Int, Int>>, x: List<BooleanArray>, filter: GuidedFilter?) {
        val n = layers.size
        val np = target.w * target.h
        val lab = target.lab
        val checkpoints = Array((n + CHECKPOINT - 1) / CHECKPOINT) { FloatArray(N * np) }
        val a = Array(3 * N) { FloatArray(np) } // a00 per band, a01 per band, a10 per band
        val coats = FloatArray(np)
        val d = DoubleArray(np)
        for (sweep in 0 until config.sweeps) {
            check()
            Parallel.forChunks(np) { from, to ->
                val r = DoubleArray(N)
                for (p in from until to) {
                    background.copyInto(r)
                    var c = 0.0
                    for (i in 0 until n) {
                        if (i % CHECKPOINT == 0) {
                            val ck = checkpoints[i / CHECKPOINT]
                            for (q in 0 until N) ck[N * p + q] = r[q].toFloat()
                        }
                        if (x[i][p]) {
                            val m = maps[layers[i].first][layers[i].second]
                            for (q in 0 until N) r[q] = (m[q] * r[q] + m[N + q]) / (m[2 * N + q] * r[q] + 1)
                            c += weights[layers[i].second]
                        }
                    }
                    coats[p] = c.toFloat()
                    for (q in 0 until N) {
                        a[q][p] = 1f
                        a[N + q][p] = 0f
                        a[2 * N + q][p] = 0f
                    }
                }
            }
            var changed = false
            for (i in n - 1 downTo 0) {
                val m = maps[layers[i].first][layers[i].second]
                val wi = weights[layers[i].second]
                val xi = x[i]
                val ck = checkpoints[i / CHECKPOINT]
                val from0 = (i / CHECKPOINT) * CHECKPOINT
                Parallel.forChunks(np) { from, to ->
                    val rb = DoubleArray(N)
                    val off = DoubleArray(N)
                    val on = DoubleArray(N)
                    for (p in from until to) {
                        for (q in 0 until N) rb[q] = ck[N * p + q].toDouble()
                        for (j in from0 until i) if (x[j][p]) { // rebuild the reflectance under layer i
                            val mj = maps[layers[j].first][layers[j].second]
                            for (q in 0 until N) rb[q] = (mj[q] * rb[q] + mj[N + q]) / (mj[2 * N + q] * rb[q] + 1)
                        }
                        for (q in 0 until N) {
                            val r0 = rb[q]
                            val pr = (m[q] * r0 + m[N + q]) / (m[2 * N + q] * r0 + 1)
                            val a00 = a[q][p]
                            val a01 = a[N + q][p]
                            val a10 = a[2 * N + q][p]
                            off[q] = (a00 * r0 + a01) / (a10 * r0 + 1)
                            on[q] = (a00 * pr + a01) / (a10 * pr + 1)
                        }
                        val others = if (xi[p]) coats[p] - wi else coats[p].toDouble()
                        val l = lab[3 * p].toDouble()
                        val la = lab[3 * p + 1].toDouble()
                        val lb = lab[3 * p + 2].toDouble()
                        val costOff = ColorMath.deltaE(off, 0, l, la, lb) + config.paintCost * others
                        val costOn = ColorMath.deltaE(on, 0, l, la, lb) + config.paintCost * (others + wi)
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
                        if (new) for (q in 0 until N) {
                            val a00 = a[q][p].toDouble()
                            val a01 = a[N + q][p].toDouble()
                            val a10 = a[2 * N + q][p].toDouble()
                            val p11 = a10 * m[N + q] + 1
                            a[q][p] = ((a00 * m[q] + a01 * m[2 * N + q]) / p11).toFloat()
                            a[N + q][p] = ((a00 * m[N + q] + a01) / p11).toFloat()
                            a[2 * N + q][p] = ((a10 * m[q] + m[2 * N + q]) / p11).toFloat()
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
        private const val N = Bands.N
        private const val CHECKPOINT = 4 // keep the reflectance under every 4th layer (see descendPixels)

        /** 0.125 -> "0.125", 1.0 -> "1", 1.5 -> "1.5" (Python's "{:g}" for these values). */
        fun fmt(v: Double): String = if (v == Math.rint(v)) v.toLong().toString() else v.toBigDecimal().stripTrailingZeros().toPlainString()
    }
}
