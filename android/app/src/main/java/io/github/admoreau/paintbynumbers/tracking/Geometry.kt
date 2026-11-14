package io.github.admoreau.paintbynumbers.tracking

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Small fixed-size linear algebra for the tracker. 3x3 matrices are row-major DoubleArray(9).
 *
 * Canvas coordinates are millimetres on the canvas plane z = 0: x to the right, y down, origin at
 * the top-left corner. A camera pose is x = (rvec, tvec): the world-to-camera rotation (as a
 * rotation vector) and translation in mm, as in OpenCV.
 */
object M3 {
    fun mul(a: DoubleArray, b: DoubleArray): DoubleArray {
        val c = DoubleArray(9)
        for (i in 0 until 3) for (j in 0 until 3) {
            c[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]
        }
        return c
    }

    fun det(a: DoubleArray): Double =
        a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6])

    /** Inverse, or null when (nearly) singular. */
    fun inv(a: DoubleArray): DoubleArray? {
        val d = det(a)
        val scale = a.maxOf { abs(it) }
        if (!d.isFinite() || abs(d) <= 1e-12 * scale * scale * scale) return null
        val r = 1.0 / d
        return doubleArrayOf(
            (a[4] * a[8] - a[5] * a[7]) * r, (a[2] * a[7] - a[1] * a[8]) * r, (a[1] * a[5] - a[2] * a[4]) * r,
            (a[5] * a[6] - a[3] * a[8]) * r, (a[0] * a[8] - a[2] * a[6]) * r, (a[2] * a[3] - a[0] * a[5]) * r,
            (a[3] * a[7] - a[4] * a[6]) * r, (a[1] * a[6] - a[0] * a[7]) * r, (a[0] * a[4] - a[1] * a[3]) * r,
        )
    }

    /** Apply homography [h] to (x, y). Returns NaNs when the point maps to infinity. */
    fun apply(h: DoubleArray, x: Double, y: Double, out: DoubleArray = DoubleArray(2)): DoubleArray {
        val w = h[6] * x + h[7] * y + h[8]
        out[0] = (h[0] * x + h[1] * y + h[2]) / w
        out[1] = (h[3] * x + h[4] * y + h[5]) / w
        return out
    }

    /** Scale to unit Frobenius norm, keeping the sign (w stays positive in front of the camera). */
    fun normalized(h: DoubleArray): DoubleArray {
        val n = sqrt(h.sumOf { it * it })
        return DoubleArray(9) { h[it] / n }
    }
}

/** Rotation matrix for rotation vector (r[off], r[off+1], r[off+2]). */
fun rodrigues(r: DoubleArray, off: Int = 0): DoubleArray {
    val rx = r[off]
    val ry = r[off + 1]
    val rz = r[off + 2]
    val theta = sqrt(rx * rx + ry * ry + rz * rz)
    if (theta < 1e-12) return doubleArrayOf(1.0, -rz, ry, rz, 1.0, -rx, -ry, rx, 1.0)
    val x = rx / theta
    val y = ry / theta
    val z = rz / theta
    val c = cos(theta)
    val s = sin(theta)
    val k = 1 - c
    return doubleArrayOf(
        c + x * x * k, x * y * k - z * s, x * z * k + y * s,
        y * x * k + z * s, c + y * y * k, y * z * k - x * s,
        z * x * k - y * s, z * y * k + x * s, c + z * z * k,
    )
}

/** Pinhole camera intrinsics, in pixels of the analysed image buffer. */
data class Intrinsics(val fx: Double, val fy: Double, val cx: Double, val cy: Double) {
    val K: DoubleArray get() = doubleArrayOf(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
}

/** Projection of canvas points for one pose, with the rotation matrix computed once. */
class PoseProjector(private val cam: Intrinsics, x: DoubleArray) {
    private val r = rodrigues(x, 0)
    private val tx = x[3]
    private val ty = x[4]
    private val tz = x[5]

    /** Pixel position of canvas point (X, Y), or NaNs behind the camera. */
    fun project(X: Double, Y: Double, out: DoubleArray): DoubleArray {
        val px = r[0] * X + r[1] * Y + tx
        val py = r[3] * X + r[4] * Y + ty
        val pz = r[6] * X + r[7] * Y + tz
        if (pz <= 1e-6) {
            out[0] = Double.NaN
            out[1] = Double.NaN
        } else {
            out[0] = cam.fx * px / pz + cam.cx
            out[1] = cam.fy * py / pz + cam.cy
        }
        return out
    }

    /** Like [project], but points behind the camera get a huge finite error so a solver can back away. */
    fun projectFinite(X: Double, Y: Double, out: DoubleArray): DoubleArray {
        val px = r[0] * X + r[1] * Y + tx
        val py = r[3] * X + r[4] * Y + ty
        val pzRaw = r[6] * X + r[7] * Y + tz
        val pz = if (pzRaw < 1.0) 1.0 else pzRaw
        val penalty = if (pzRaw < 1.0) 1e5 else 0.0
        out[0] = cam.fx * px / pz + cam.cx + penalty
        out[1] = cam.fy * py / pz + cam.cy + penalty
        return out
    }

    /** Canvas mm -> pixels, unit Frobenius norm. */
    fun homography(): DoubleArray = M3.normalized(
        M3.mul(cam.K, doubleArrayOf(r[0], r[1], tx, r[3], r[4], ty, r[6], r[7], tz)),
    )

    /** Distance of the camera in front of the canvas plane (positive = in front). */
    fun height(): Double = r[2] * tx + r[5] * ty + r[8] * tz
}

/** Solve the n x n system [a] x = [b] in place by Gaussian elimination with partial pivoting; null if singular. */
fun solveLinear(a: DoubleArray, b: DoubleArray, n: Int): DoubleArray? {
    val m = a.copyOf()
    val x = b.copyOf()
    for (col in 0 until n) {
        var piv = col
        for (r in col + 1 until n) if (abs(m[r * n + col]) > abs(m[piv * n + col])) piv = r
        if (abs(m[piv * n + col]) < 1e-300) return null
        if (piv != col) {
            for (c in 0 until n) { val t = m[col * n + c]; m[col * n + c] = m[piv * n + c]; m[piv * n + c] = t }
            val t = x[col]; x[col] = x[piv]; x[piv] = t
        }
        val d = m[col * n + col]
        for (r in col + 1 until n) {
            val f = m[r * n + col] / d
            if (f == 0.0) continue
            for (c in col until n) m[r * n + c] -= f * m[col * n + c]
            x[r] -= f * x[col]
        }
    }
    for (r in n - 1 downTo 0) {
        var s = x[r]
        for (c in r + 1 until n) s -= m[r * n + c] * x[c]
        x[r] = s / m[r * n + r]
        if (!x[r].isFinite()) return null
    }
    return x
}

/** Inverse of an n x n matrix, or null if singular. */
fun invert(a: DoubleArray, n: Int): DoubleArray? {
    val out = DoubleArray(n * n)
    for (c in 0 until n) {
        val e = DoubleArray(n).also { it[c] = 1.0 }
        val col = solveLinear(a, e, n) ?: return null
        for (r in 0 until n) out[r * n + c] = col[r]
    }
    return out
}
