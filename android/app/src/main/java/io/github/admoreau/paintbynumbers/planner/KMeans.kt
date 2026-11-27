package io.github.admoreau.paintbynumbers.planner

import kotlin.math.ln
import kotlin.random.Random

/**
 * K-means of an image's colours in Lab, like the desktop planner's cluster_colors (scikit-learn's
 * KMeans: k-means++ starts, Lloyd iterations, best of [nInit] runs).
 *
 * Works on the image's distinct colours, each weighted by how many pixels have it, which gives
 * the same clusters as running on every pixel but is much faster on flat artwork.
 */
object KMeans {
    class Result(val centers: DoubleArray, val k: Int, val labels: IntArray) // centers: k x 3

    /**
     * [lab]: per pixel L, a, b (3 per pixel); [key]: per pixel an id that is equal for equal colours
     * (such as its packed 8-bit sRGB). Returns centres and each pixel's cluster.
     */
    fun cluster(lab: FloatArray, key: IntArray, nColors: Int, seed: Int = 0, nInit: Int = 3, maxIter: Int = 300, tol: Double = 1e-4): Result {
        val n = key.size
        // distinct colours and their pixel counts
        val index = HashMap<Int, Int>()
        val pixelColor = IntArray(n)
        val firstPixel = ArrayList<Int>()
        for (p in 0 until n) {
            pixelColor[p] = index.getOrPut(key[p]) { firstPixel.add(p); firstPixel.size - 1 }
        }
        val m = firstPixel.size
        val x = DoubleArray(3 * m)
        val w = DoubleArray(m)
        for (j in 0 until m) for (c in 0 until 3) x[3 * j + c] = lab[3 * firstPixel[j] + c].toDouble()
        for (p in 0 until n) w[pixelColor[p]] += 1.0
        val k = minOf(nColors, m)

        // tolerance relative to the data's spread, as scikit-learn does
        var tolAbs = 0.0
        for (c in 0 until 3) {
            var mean = 0.0
            for (j in 0 until m) mean += w[j] * x[3 * j + c]
            mean /= n
            var v = 0.0
            for (j in 0 until m) v += w[j] * (x[3 * j + c] - mean).let { it * it }
            tolAbs += v / n
        }
        tolAbs = tolAbs / 3 * tol

        val rng = Random(seed)
        var best: DoubleArray? = null
        var bestLabels: IntArray? = null
        var bestInertia = Double.POSITIVE_INFINITY
        repeat(nInit) {
            val centers = plusPlus(x, w, k, rng)
            val labels = IntArray(m)
            val inertia = lloyd(x, w, centers, k, labels, maxIter, tolAbs)
            if (inertia < bestInertia) {
                bestInertia = inertia
                best = centers
                bestLabels = labels
            }
        }
        val colorLabels = bestLabels!!
        return Result(best!!, k, IntArray(n) { colorLabels[pixelColor[it]] })
    }

    private fun dist2(x: DoubleArray, i: Int, c: DoubleArray, j: Int): Double {
        val d0 = x[3 * i] - c[3 * j]
        val d1 = x[3 * i + 1] - c[3 * j + 1]
        val d2 = x[3 * i + 2] - c[3 * j + 2]
        return d0 * d0 + d1 * d1 + d2 * d2
    }

    /** Greedy k-means++: each new centre is the best of a few candidates drawn in proportion to weight x distance². */
    private fun plusPlus(x: DoubleArray, w: DoubleArray, k: Int, rng: Random): DoubleArray {
        val m = w.size
        val centers = DoubleArray(3 * k)
        val trials = 2 + ln(k.toDouble()).toInt()
        val total = w.sum()
        fun draw(weights: DoubleArray, sum: Double): Int {
            var r = rng.nextDouble() * sum
            for (j in 0 until m) {
                r -= weights[j]
                if (r <= 0) return j
            }
            return m - 1
        }
        val first = draw(w, total)
        for (c in 0 until 3) centers[c] = x[3 * first + c]
        val closest = DoubleArray(m) { dist2(x, it, centers, 0) }
        val potential = DoubleArray(m)
        val cand = DoubleArray(m)
        for (ci in 1 until k) {
            for (j in 0 until m) potential[j] = w[j] * closest[j]
            val pot = potential.sum()
            var bestSum = Double.POSITIVE_INFINITY
            var bestIdx = -1
            var bestD: DoubleArray? = null
            repeat(trials) {
                val idx = if (pot > 0) draw(potential, pot) else rng.nextInt(m)
                val one = doubleArrayOf(x[3 * idx], x[3 * idx + 1], x[3 * idx + 2])
                var s = 0.0
                for (j in 0 until m) {
                    cand[j] = minOf(closest[j], dist2(x, j, one, 0))
                    s += w[j] * cand[j]
                }
                if (s < bestSum) {
                    bestSum = s
                    bestIdx = idx
                    bestD = cand.copyOf()
                }
            }
            for (c in 0 until 3) centers[3 * ci + c] = x[3 * bestIdx + c]
            bestD!!.copyInto(closest)
        }
        return centers
    }

    /** Lloyd iterations until the centres move less than [tolAbs] in total; returns the inertia. */
    private fun lloyd(x: DoubleArray, w: DoubleArray, centers: DoubleArray, k: Int, labels: IntArray, maxIter: Int, tolAbs: Double): Double {
        val m = w.size
        val dist = DoubleArray(m)
        var inertia = 0.0
        for (iter in 0 until maxIter) {
            Parallel.forChunks(m) { from, to ->
                for (j in from until to) {
                    var bd = Double.POSITIVE_INFINITY
                    var bl = 0
                    for (c in 0 until k) {
                        val d = dist2(x, j, centers, c)
                        if (d < bd) {
                            bd = d
                            bl = c
                        }
                    }
                    labels[j] = bl
                    dist[j] = bd
                }
            }
            val sums = DoubleArray(3 * k)
            val counts = DoubleArray(k)
            inertia = 0.0
            for (j in 0 until m) {
                val l = labels[j]
                counts[l] += w[j]
                for (c in 0 until 3) sums[3 * l + c] += w[j] * x[3 * j + c]
                inertia += w[j] * dist[j]
            }
            // an empty cluster takes the colour farthest from its centre
            for (c in 0 until k) if (counts[c] == 0.0) {
                var far = 0
                for (j in 1 until m) if (dist[j] > dist[far]) far = j
                for (q in 0 until 3) sums[3 * c + q] = x[3 * far + q]
                counts[c] = 1.0
                dist[far] = 0.0
            }
            var shift = 0.0
            for (c in 0 until k) for (q in 0 until 3) {
                val v = sums[3 * c + q] / counts[c]
                shift += (v - centers[3 * c + q]).let { it * it }
                centers[3 * c + q] = v
            }
            if (shift <= tolAbs) break
        }
        // final labels for the final centres
        inertia = 0.0
        for (j in 0 until m) {
            var bd = Double.POSITIVE_INFINITY
            for (c in 0 until k) {
                val d = dist2(x, j, centers, c)
                if (d < bd) {
                    bd = d
                    labels[j] = c
                }
            }
            inertia += w[j] * bd
        }
        return inertia
    }
}
