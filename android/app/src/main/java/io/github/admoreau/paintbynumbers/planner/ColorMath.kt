package io.github.admoreau.paintbynumbers.planner

import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Colour conversions, as in the desktop planner's color.py: sRGB, CIE XYZ (D65, white Y = 1),
 * CIE L*a*b*, and the working space where paint reflectance lives: spectral reflectance in
 * [Bands.N] wavelength bands ([N] values per colour).
 *
 * Matrices are row-major. Nothing is clipped except where noted.
 */
object ColorMath {
    private val RGB_TO_XYZ = doubleArrayOf(
        0.4124564, 0.3575761, 0.1804375,
        0.2126729, 0.7151522, 0.0721750,
        0.0193339, 0.1191920, 0.9503041,
    )
    val WHITE = DoubleArray(3) { r -> RGB_TO_XYZ[3 * r] + RGB_TO_XYZ[3 * r + 1] + RGB_TO_XYZ[3 * r + 2] }
    const val N = Bands.N

    /** Bands to XYZ, rescaled so reflectance 1 in every band is exactly the white (as color.py). */
    val WORKING_TO_XYZ = DoubleArray(3 * N).also { m ->
        for (r in 0 until 3) {
            var sum = 0.0
            for (j in 0 until N) sum += Bands.TO_XYZ[r * N + j]
            for (j in 0 until N) m[r * N + j] = Bands.TO_XYZ[r * N + j] * WHITE[r] / sum
        }
    }

    /** XYZ to the smoothest spectrum, made an exact inverse of [WORKING_TO_XYZ] (as color.py). */
    private val XYZ_TO_SMOOTH = run {
        val a = Bands.FROM_XYZ
        val ma = DoubleArray(9) { k -> (0 until N).sumOf { j -> WORKING_TO_XYZ[(k / 3) * N + j] * a[j * 3 + k % 3] } }
        val inv = inverse3(ma)
        DoubleArray(N * 3) { k -> (0 until 3).sumOf { q -> a[(k / 3) * 3 + q] * inv[q * 3 + k % 3] } }
    }
    private val XYZ_TO_RGB = inverse3(RGB_TO_XYZ)

    /** Working space straight to XYZ / white, the input of the Lab companding. */
    private val W2N = DoubleArray(3 * N) { WORKING_TO_XYZ[it] / WHITE[it / N] }
    private val L2N = DoubleArray(9) { RGB_TO_XYZ[it] / WHITE[it / 3] }

    /** Band centres scaled to about [-1, 1], for the sigmoid fallback spectra. */
    private val X = DoubleArray(N) { (Bands.CENTRES_NM[it] - 550.0) / 150.0 }

    private const val SRGB_KNEE = 0.04045
    private const val LINEAR_KNEE = SRGB_KNEE / 12.92
    private const val DELTA = 6.0 / 29.0
    private const val DELTA_CUBED = DELTA * DELTA * DELTA
    private const val SLOPE = 1.0 / (3.0 * DELTA * DELTA)
    private const val OFFSET = 4.0 / 29.0

    fun srgbToLinear(x: Double): Double = if (x <= SRGB_KNEE) x / 12.92 else ((x + 0.055) / 1.055).pow(2.4)

    fun linearToSrgb(x: Double): Double = if (x <= LINEAR_KNEE) x * 12.92 else 1.055 * x.pow(1.0 / 2.4) - 0.055

    private fun f(t: Double): Double = if (t > DELTA_CUBED) cubeRoot(t) else t * SLOPE + OFFSET

    /**
     * Cube root of x > 0, to within about 1e-15 relative: a bit-trick first guess (a few percent
     * off) and two Halley steps, each cubing the error. Several times faster than Math.cbrt on
     * phones, and the colour error is the planner's innermost operation.
     */
    fun cubeRoot(x: Double): Double {
        var y = java.lang.Float.intBitsToFloat(java.lang.Float.floatToRawIntBits(x.toFloat()) / 3 + 709921077).toDouble()
        var y3 = y * y * y
        y *= (y3 + 2 * x) / (2 * y3 + x)
        y3 = y * y * y
        y *= (y3 + 2 * x) / (2 * y3 + x)
        return y
    }

    private fun labOf(fx: Double, fy: Double, fz: Double, out: DoubleArray, o: Int) {
        out[o] = 116.0 * fy - 16.0
        out[o + 1] = 500.0 * (fx - fy)
        out[o + 2] = 200.0 * (fy - fz)
    }

    /** Working-space reflectance r[ro until ro + N] to Lab, written to out[o..o+2]. */
    fun workingToLab(r: DoubleArray, ro: Int, out: DoubleArray, o: Int = 0) {
        var x = 0.0; var y = 0.0; var z = 0.0
        for (j in 0 until N) {
            val v = r[ro + j]
            x += W2N[j] * v; y += W2N[N + j] * v; z += W2N[2 * N + j] * v
        }
        labOf(f(x), f(y), f(z), out, o)
    }

    /** Gamma-encoded sRGB in [0, 1] to Lab. */
    fun srgbToLab(r: Double, g: Double, b: Double, out: DoubleArray, o: Int = 0) {
        val m = L2N
        val lr = srgbToLinear(r); val lg = srgbToLinear(g); val lb = srgbToLinear(b)
        labOf(f(m[0] * lr + m[1] * lg + m[2] * lb), f(m[3] * lr + m[4] * lg + m[5] * lb), f(m[6] * lr + m[7] * lg + m[8] * lb), out, o)
    }

    /** ΔE76 of working-space reflectance r[ro until ro + N] from Lab (l, a, bb). */
    fun deltaE(r: DoubleArray, ro: Int, l: Double, a: Double, bb: Double): Double {
        var x = 0.0; var y = 0.0; var z = 0.0
        for (j in 0 until N) {
            val v = r[ro + j]
            x += W2N[j] * v; y += W2N[N + j] * v; z += W2N[2 * N + j] * v
        }
        val fx = f(x)
        val fy = f(y)
        val fz = f(z)
        val dl = 116.0 * fy - 16.0 - l
        val da = 500.0 * (fx - fy) - a
        val db = 200.0 * (fy - fz) - bb
        return sqrt(dl * dl + da * da + db * db)
    }

    /**
     * CIE XYZ to a working-space reflectance with that colour, written to out[o until o + N]: the
     * smoothest spectrum, or where that leaves [0, 1] a sigmoid-shaped one (as color.xyz_to_working).
     */
    fun xyzToWorking(x: Double, y: Double, z: Double, out: DoubleArray, o: Int = 0) {
        var ok = true
        for (j in 0 until N) {
            val v = XYZ_TO_SMOOTH[3 * j] * x + XYZ_TO_SMOOTH[3 * j + 1] * y + XYZ_TO_SMOOTH[3 * j + 2] * z
            out[o + j] = v
            if (v < 0 || v > 1) ok = false
        }
        if (!ok) sigmoidSpectrum(x, y, z, out, o)
    }

    /** sigmoid(c0 + c1 x + c2 x^2) per band, matching XYZ by damped Gauss-Newton (as color._sigmoid_spectra). */
    private fun sigmoidSpectrum(x: Double, y: Double, z: Double, out: DoubleArray, o: Int) {
        val yc = y.coerceIn(1e-4, 1 - 1e-4)
        val c = doubleArrayOf(kotlin.math.ln(yc / (1 - yc)), 0.0, 0.0)
        val target = doubleArrayOf(x, y, z)
        val r = DoubleArray(N)
        val jac = DoubleArray(9)
        repeat(60) {
            for (j in 0 until N) r[j] = 1.0 / (1.0 + kotlin.math.exp(-(c[0] + c[1] * X[j] + c[2] * X[j] * X[j])))
            jac.fill(0.0)
            val res = DoubleArray(3)
            for (row in 0 until 3) {
                var acc = 0.0
                for (j in 0 until N) {
                    val m = WORKING_TO_XYZ[row * N + j]
                    acc += m * r[j]
                    val d = m * r[j] * (1 - r[j])
                    jac[row * 3] += d; jac[row * 3 + 1] += d * X[j]; jac[row * 3 + 2] += d * X[j] * X[j]
                }
                res[row] = acc - target[row]
            }
            // (J'J + 1e-10 I) step = J' res
            val jtj = DoubleArray(9) { k -> (0 until 3).sumOf { q -> jac[q * 3 + k / 3] * jac[q * 3 + k % 3] } + if (k % 4 == 0) 1e-10 else 0.0 }
            val jtr = DoubleArray(3) { k -> (0 until 3).sumOf { q -> jac[q * 3 + k] * res[q] } }
            val inv = inverse3(jtj)
            for (k in 0 until 3) {
                val step = inv[3 * k] * jtr[0] + inv[3 * k + 1] * jtr[1] + inv[3 * k + 2] * jtr[2]
                c[k] -= step.coerceIn(-4.0, 4.0)
            }
        }
        for (j in 0 until N) out[o + j] = 1.0 / (1.0 + kotlin.math.exp(-(c[0] + c[1] * X[j] + c[2] * X[j] * X[j])))
    }

    /** Gamma-encoded sRGB to working-space reflectance, written to out[o until o + N]. */
    fun srgbToWorking(r: Double, g: Double, b: Double, out: DoubleArray, o: Int = 0) {
        val lr = srgbToLinear(r); val lg = srgbToLinear(g); val lb = srgbToLinear(b)
        val m = RGB_TO_XYZ
        xyzToWorking(m[0] * lr + m[1] * lg + m[2] * lb, m[3] * lr + m[4] * lg + m[5] * lb, m[6] * lr + m[7] * lg + m[8] * lb, out, o)
    }

    /** Working-space reflectance r[ro until ro + N] to gamma-encoded sRGB, clipped to [0, 1], in out[o..o+2]. */
    fun workingToSrgb(r: DoubleArray, ro: Int, out: DoubleArray, o: Int = 0) {
        var x = 0.0; var y = 0.0; var z = 0.0
        for (j in 0 until N) {
            val v = r[ro + j]
            x += WORKING_TO_XYZ[j] * v; y += WORKING_TO_XYZ[N + j] * v; z += WORKING_TO_XYZ[2 * N + j] * v
        }
        val m = XYZ_TO_RGB
        for (c in 0 until 3) out[o + c] = linearToSrgb(m[3 * c] * x + m[3 * c + 1] * y + m[3 * c + 2] * z).coerceIn(0.0, 1.0)
    }

    /** Working-space reflectance r[ro until ro + N] to an opaque ARGB colour int. */
    fun workingToArgb(r: DoubleArray, ro: Int = 0): Int {
        val s = DoubleArray(3)
        workingToSrgb(r, ro, s)
        return (0xFF shl 24) or (round255(s[0]) shl 16) or (round255(s[1]) shl 8) or round255(s[2])
    }

    fun round255(v: Double): Int = Math.rint(v * 255.0).toInt().coerceIn(0, 255)

    fun inverse3(m: DoubleArray): DoubleArray {
        val (a, b, c) = Triple(m[0], m[1], m[2])
        val (d, e, f) = Triple(m[3], m[4], m[5])
        val (g, h, i) = Triple(m[6], m[7], m[8])
        val A = e * i - f * h
        val B = f * g - d * i
        val C = d * h - e * g
        val det = a * A + b * B + c * C
        return doubleArrayOf(
            A / det, (c * h - b * i) / det, (b * f - c * e) / det,
            B / det, (a * i - c * g) / det, (c * d - a * f) / det,
            C / det, (b * g - a * h) / det, (a * e - b * d) / det,
        )
    }

}
