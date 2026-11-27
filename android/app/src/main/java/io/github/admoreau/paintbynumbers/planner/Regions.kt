package io.github.admoreau.paintbynumbers.planner

/**
 * Edge-aware smoothing that follows the edges of a guide image (He et al. 2010), as in uniform.py.
 *
 * Within each (2r+1)-square window the output is a linear function of the guide's channels, so it
 * is smooth where the guide is flat and changes where the guide does: thin lines survive. [eps]
 * (guide units squared) sets how big a change in the guide counts as an edge. The guide's
 * statistics are computed once; [apply] smooths one (H, W) array.
 *
 * [guide]: 3 channels per pixel, row-major. Windows at the border repeat the edge pixels (scipy's
 * uniform_filter, mode="nearest"). Window sums are kept in double; the per-pixel buffers between
 * passes are float, which halves the memory traffic that dominates on a phone.
 */
class GuidedFilter(private val guide: FloatArray, private val w: Int, private val h: Int, private val radius: Int, eps: Double) {
    private val n = w * h
    private val meanI = Array(3) { FloatArray(n) }
    private val inv = Array(6) { FloatArray(n) } // symmetric 3x3 inverse: 00 01 02 11 12 22
    private val rows = Array(4) { FloatArray(n) } // row-pass sums
    private val coef = Array(4) { FloatArray(n) } // a0, a1, a2, b
    private val norm = 1.0 / ((2 * radius + 1) * (2 * radius + 1))

    init {
        // per pixel: the guide's channels and their pairwise products, box-summed
        val pairs = arrayOf(0 to 0, 0 to 1, 0 to 2, 1 to 1, 1 to 2, 2 to 2)
        val sums = Array(9) { DoubleArray(n) } // 3 means, 6 second moments
        val src = DoubleArray(n)
        val buf = DoubleArray(n)
        for (q in 0 until 9) {
            for (p in 0 until n) src[p] = if (q < 3) guide[3 * p + q].toDouble() else pairs[q - 3].let { (i, j) -> guide[3 * p + i].toDouble() * guide[3 * p + j] }
            boxDouble(src, buf, sums[q])
        }
        for (c in 0 until 3) for (p in 0 until n) meanI[c][p] = sums[c][p].toFloat()
        Parallel.forChunks(n) { from, to ->
            for (p in from until to) {
                val m0 = sums[0][p]
                val m1 = sums[1][p]
                val m2 = sums[2][p]
                val a = sums[3][p] - m0 * m0 + eps
                val b = sums[4][p] - m0 * m1
                val c = sums[5][p] - m0 * m2
                val d = sums[6][p] - m1 * m1 + eps
                val e = sums[7][p] - m1 * m2
                val f = sums[8][p] - m2 * m2 + eps
                val c00 = d * f - e * e
                val c01 = c * e - b * f
                val c02 = b * e - c * d
                val det = a * c00 + b * c01 + c * c02
                inv[0][p] = (c00 / det).toFloat()
                inv[1][p] = (c01 / det).toFloat()
                inv[2][p] = (c02 / det).toFloat()
                inv[3][p] = ((a * f - c * c) / det).toFloat()
                inv[4][p] = ((b * c - a * e) / det).toFloat()
                inv[5][p] = ((a * d - b * b) / det).toFloat()
            }
        }
    }

    /** Smoothed copy of [p] (H*W), written to [out] (may be [p]). Not thread-safe: one call at a time. */
    fun apply(p: DoubleArray, out: DoubleArray) {
        val r = radius
        val g = guide
        // 1. row sums of p and of each guide channel times p
        Parallel.forChunks(h) { y0, y1 ->
            for (y in y0 until y1) {
                val o = y * w
                var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0
                for (k in -r..r) {
                    val q = o + k.coerceIn(0, w - 1)
                    val v = p[q]
                    s0 += v; s1 += g[3 * q] * v; s2 += g[3 * q + 1] * v; s3 += g[3 * q + 2] * v
                }
                for (x in 0 until w) {
                    rows[0][o + x] = s0.toFloat(); rows[1][o + x] = s1.toFloat(); rows[2][o + x] = s2.toFloat(); rows[3][o + x] = s3.toFloat()
                    val qa = o + (x + r + 1).coerceAtMost(w - 1)
                    val qs = o + (x - r).coerceAtLeast(0)
                    val va = p[qa]
                    val vs = p[qs]
                    s0 += va - vs
                    s1 += g[3 * qa] * va - g[3 * qs] * vs
                    s2 += g[3 * qa + 1] * va - g[3 * qs + 1] * vs
                    s3 += g[3 * qa + 2] * va - g[3 * qs + 2] * vs
                }
            }
        }
        // 2. column sums give the window means; solve each pixel's linear coefficients
        columns(rows) { q, m ->
            val mp = m[0] * norm
            val x = m[1] * norm - meanI[0][q] * mp
            val y = m[2] * norm - meanI[1][q] * mp
            val z = m[3] * norm - meanI[2][q] * mp
            val a0 = inv[0][q] * x + inv[1][q] * y + inv[2][q] * z
            val a1 = inv[1][q] * x + inv[3][q] * y + inv[4][q] * z
            val a2 = inv[2][q] * x + inv[4][q] * y + inv[5][q] * z
            coef[0][q] = a0.toFloat()
            coef[1][q] = a1.toFloat()
            coef[2][q] = a2.toFloat()
            coef[3][q] = (mp - (a0 * meanI[0][q] + a1 * meanI[1][q] + a2 * meanI[2][q])).toFloat()
        }
        // 3. row sums of the coefficients
        Parallel.forChunks(h) { y0, y1 ->
            for (y in y0 until y1) {
                val o = y * w
                for (c in 0 until 4) {
                    val src = coef[c]
                    val dst = rows[c]
                    var s = 0.0
                    for (k in -r..r) s += src[o + k.coerceIn(0, w - 1)]
                    for (x in 0 until w) {
                        dst[o + x] = s.toFloat()
                        s += src[o + (x + r + 1).coerceAtMost(w - 1)] - src[o + (x - r).coerceAtLeast(0)]
                    }
                }
            }
        }
        // 4. column sums: the mean coefficients, applied to the guide
        columns(rows) { q, m ->
            out[q] = (m[0] * g[3 * q] + m[1] * g[3 * q + 1] + m[2] * g[3 * q + 2] + m[3]) * norm
        }
    }

    /**
     * Sliding column sums of the 4 row-summed arrays [src]: calls [each] with every pixel and its 4
     * window sums. A band of columns per task, walking down the rows, so reads are contiguous.
     */
    private inline fun columns(src: Array<FloatArray>, crossinline each: (Int, DoubleArray) -> Unit) {
        val r = radius
        val bands = (w + 31) / 32
        Parallel.forChunks(bands) { b0, b1 ->
            val x0 = b0 * 32
            val x1 = minOf(w, b1 * 32)
            val span = x1 - x0
            val col = Array(4) { DoubleArray(span) }
            val m = DoubleArray(4)
            for (k in -r..r) {
                val o = k.coerceIn(0, h - 1) * w + x0
                for (c in 0 until 4) for (i in 0 until span) col[c][i] += src[c][o + i]
            }
            for (y in 0 until h) {
                val o = y * w + x0
                for (i in 0 until span) {
                    for (c in 0 until 4) m[c] = col[c][i]
                    each(o + i, m)
                }
                val add = (y + r + 1).coerceAtMost(h - 1) * w + x0
                val sub = (y - r).coerceAtLeast(0) * w + x0
                for (c in 0 until 4) {
                    val cc = col[c]
                    val sc = src[c]
                    for (i in 0 until span) cc[i] += sc[add + i] - sc[sub + i]
                }
            }
        }
    }

    /** Window mean of [src] into [out] in double, using [buf] (setup only). */
    private fun boxDouble(src: DoubleArray, buf: DoubleArray, out: DoubleArray) {
        val r = radius
        for (y in 0 until h) {
            val o = y * w
            var s = 0.0
            for (k in -r..r) s += src[o + k.coerceIn(0, w - 1)]
            for (x in 0 until w) {
                buf[o + x] = s
                s += src[o + (x + r + 1).coerceAtMost(w - 1)] - src[o + (x - r).coerceAtLeast(0)]
            }
        }
        for (x in 0 until w) {
            var s = 0.0
            for (k in -r..r) s += buf[k.coerceIn(0, h - 1) * w + x]
            for (y in 0 until h) {
                out[y * w + x] = s * norm
                s += buf[(y + r + 1).coerceAtMost(h - 1) * w + x] - buf[(y - r).coerceAtLeast(0) * w + x]
            }
        }
    }
}

object Regions {
    /**
     * Removes patches and fills holes smaller than [minArea] pixels in a mask (w x h), in place.
     * Patches are 8-connected and holes 4-connected, so thin diagonal strokes survive and do not
     * count as gaps.
     */
    fun clean(mask: BooleanArray, w: Int, h: Int, minArea: Int) {
        if (minArea <= 1) return
        for (value in booleanArrayOf(true, false)) {
            val label = IntArray(w * h) { -1 }
            val stack = IntArray(w * h)
            val members = IntArray(w * h)
            for (start in 0 until w * h) {
                if (mask[start] != value || label[start] >= 0) continue
                var top = 0
                var count = 0
                stack[top++] = start
                label[start] = start
                while (top > 0) {
                    val p = stack[--top]
                    members[count++] = p
                    val x = p % w
                    val y = p / w
                    for (dy in -1..1) for (dx in -1..1) {
                        if (dx == 0 && dy == 0) continue
                        if (!value && dx != 0 && dy != 0) continue // holes: 4-connected
                        val nx = x + dx
                        val ny = y + dy
                        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
                        val q = ny * w + nx
                        if (mask[q] == value && label[q] < 0) {
                            label[q] = start
                            stack[top++] = q
                        }
                    }
                }
                if (count < minArea) for (i in 0 until count) mask[members[i]] = !value
            }
        }
    }
}
