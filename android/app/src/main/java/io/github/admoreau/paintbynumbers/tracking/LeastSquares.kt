package io.github.admoreau.paintbynumbers.tracking

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Robust nonlinear least squares (Levenberg-Marquardt) for small problems, standing in for
 * scipy.optimize.least_squares(loss="soft_l1") in the prototype.
 *
 * Minimises 0.5 * sum f^2 * rho((r_i / f)^2) with rho(z) = 2 (sqrt(1 + z) - 1): quadratic for
 * small residuals, linear for large ones, so one bad match cannot drag the answer far.
 */
class LeastSquares(
    private val n: Int,
    private val xScale: DoubleArray,
    private val fScale: Double = 2.0,
    private val maxIter: Int = 60,
) {
    class Result(val x: DoubleArray, val jac: Array<DoubleArray>, val residuals: DoubleArray)

    private fun cost(r: DoubleArray): Double {
        var c = 0.0
        val f2 = fScale * fScale
        for (v in r) c += f2 * (sqrt(1 + v * v / f2) - 1)
        return c
    }

    private fun jacobian(x: DoubleArray, r0: DoubleArray, fn: (DoubleArray) -> DoubleArray): Array<DoubleArray> {
        val jac = Array(r0.size) { DoubleArray(n) }
        val xp = x.copyOf()
        for (k in 0 until n) {
            val h = 1.49e-8 * max(1.0, abs(x[k]))
            xp[k] = x[k] + h
            val r1 = fn(xp)
            for (i in r0.indices) jac[i][k] = (r1[i] - r0[i]) / h
            xp[k] = x[k]
        }
        return jac
    }

    fun solve(x0: DoubleArray, fn: (DoubleArray) -> DoubleArray): Result {
        var x = x0.copyOf()
        var r = fn(x)
        var c = cost(r)
        var lambda = -1.0
        val f2 = fScale * fScale
        val a = DoubleArray(n * n)
        val g = DoubleArray(n)
        var jac = jacobian(x, r, fn)
        for (iter in 0 until maxIter) {
            // normal equations in scaled variables (x = scale * y), IRLS weights rho'(z)
            a.fill(0.0)
            g.fill(0.0)
            for (i in r.indices) {
                val w = 1.0 / sqrt(1 + r[i] * r[i] / f2)
                val row = jac[i]
                for (p in 0 until n) {
                    val jp = row[p] * xScale[p]
                    g[p] += w * jp * r[i]
                    for (q in p until n) a[p * n + q] += w * jp * row[q] * xScale[q]
                }
            }
            for (p in 0 until n) for (q in 0 until p) a[p * n + q] = a[q * n + p]
            if (lambda < 0) lambda = 1e-3 * (0 until n).maxOf { a[it * n + it] }.coerceAtLeast(1e-12)
            var improved = false
            var smallStep = false
            for (attempt in 0 until 10) {
                val damped = a.copyOf()
                for (p in 0 until n) damped[p * n + p] += lambda
                val dy = solveLinear(damped, DoubleArray(n) { -g[it] }, n)
                if (dy == null) { lambda *= 4; continue }
                val xn = DoubleArray(n) { x[it] + dy[it] * xScale[it] }
                val rn = fn(xn)
                val cn = cost(rn)
                val stepNorm = sqrt(dy.sumOf { it * it })
                val yNorm = sqrt((0 until n).sumOf { (x[it] / xScale[it]).pow(2) })
                if (cn.isFinite() && cn < c) {
                    smallStep = stepNorm < 1e-8 * (1 + yNorm)
                    val relDrop = (c - cn) / max(c, 1e-300)
                    x = xn
                    r = rn
                    c = cn
                    lambda = max(lambda / 3, 1e-12)
                    improved = true
                    if (relDrop < 1e-10) smallStep = true
                    break
                }
                lambda *= 4
                if (stepNorm < 1e-10 * (1 + yNorm)) { smallStep = true; break }
            }
            if (!improved) break
            jac = jacobian(x, r, fn)
            if (smallStep) break
        }
        // Jacobian scaled like scipy's result.jac for this loss: J^T J approximates the Hessian
        val scaled = Array(r.size) { i ->
            val s = (1 + r[i] * r[i] / f2).pow(-0.75)
            DoubleArray(n) { jac[i][it] * s }
        }
        return Result(x, scaled, r)
    }
}
