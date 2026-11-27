package io.github.admoreau.paintbynumbers.planner

import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Colour conversions, as in the desktop planner's color.py: sRGB, CIE XYZ (D65, white Y = 1),
 * CIE L*a*b*, and the working space where paint reflectance lives (linear Rec.2020).
 *
 * Matrices are 3x3, row-major. Nothing is clipped except where noted.
 */
object ColorMath {
    private val RGB_TO_XYZ = doubleArrayOf(
        0.4124564, 0.3575761, 0.1804375,
        0.2126729, 0.7151522, 0.0721750,
        0.0193339, 0.1191920, 0.9503041,
    )
    val WHITE = DoubleArray(3) { r -> RGB_TO_XYZ[3 * r] + RGB_TO_XYZ[3 * r + 1] + RGB_TO_XYZ[3 * r + 2] }
    private val REC2020 = arrayOf(doubleArrayOf(0.708, 0.292), doubleArrayOf(0.170, 0.797), doubleArrayOf(0.131, 0.046))
    val WORKING_TO_XYZ = rgbToXyz(REC2020, WHITE)
    private val XYZ_TO_RGB = inverse3(RGB_TO_XYZ)
    private val WORKING_TO_LINEAR_SRGB = mul3(XYZ_TO_RGB, WORKING_TO_XYZ)
    private val LINEAR_SRGB_TO_WORKING = inverse3(WORKING_TO_LINEAR_SRGB)

    /** Working space straight to XYZ / white, the input of the Lab companding. */
    private val W2N = DoubleArray(9) { WORKING_TO_XYZ[it] / WHITE[it / 3] }
    private val L2N = DoubleArray(9) { RGB_TO_XYZ[it] / WHITE[it / 3] }

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

    private fun lab(m: DoubleArray, r: Double, g: Double, b: Double, out: DoubleArray, o: Int) {
        val fx = f(m[0] * r + m[1] * g + m[2] * b)
        val fy = f(m[3] * r + m[4] * g + m[5] * b)
        val fz = f(m[6] * r + m[7] * g + m[8] * b)
        out[o] = 116.0 * fy - 16.0
        out[o + 1] = 500.0 * (fx - fy)
        out[o + 2] = 200.0 * (fy - fz)
    }

    /** Working-space reflectance to Lab, written to out[o..o+2]. */
    fun workingToLab(r: Double, g: Double, b: Double, out: DoubleArray, o: Int = 0) = lab(W2N, r, g, b, out, o)

    /** Gamma-encoded sRGB in [0, 1] to Lab. */
    fun srgbToLab(r: Double, g: Double, b: Double, out: DoubleArray, o: Int = 0) =
        lab(L2N, srgbToLinear(r), srgbToLinear(g), srgbToLinear(b), out, o)

    /** ΔE76 of working-space reflectance (r, g, b) from Lab (l, a, bb). */
    fun deltaE(r: Double, g: Double, b: Double, l: Double, a: Double, bb: Double): Double {
        val m = W2N
        val fx = f(m[0] * r + m[1] * g + m[2] * b)
        val fy = f(m[3] * r + m[4] * g + m[5] * b)
        val fz = f(m[6] * r + m[7] * g + m[8] * b)
        val dl = 116.0 * fy - 16.0 - l
        val da = 500.0 * (fx - fy) - a
        val db = 200.0 * (fy - fz) - bb
        return sqrt(dl * dl + da * da + db * db)
    }

    /** Gamma-encoded sRGB to working-space reflectance, written to out[o..o+2]. */
    fun srgbToWorking(r: Double, g: Double, b: Double, out: DoubleArray, o: Int = 0) =
        apply(LINEAR_SRGB_TO_WORKING, srgbToLinear(r), srgbToLinear(g), srgbToLinear(b), out, o)

    /** Working-space reflectance to gamma-encoded sRGB, clipped to [0, 1]. */
    fun workingToSrgb(r: Double, g: Double, b: Double, out: DoubleArray, o: Int = 0) {
        apply(WORKING_TO_LINEAR_SRGB, r, g, b, out, o)
        for (c in 0 until 3) out[o + c] = linearToSrgb(out[o + c]).coerceIn(0.0, 1.0)
    }

    /** Working-space reflectance to an opaque ARGB colour int. */
    fun workingToArgb(r: Double, g: Double, b: Double): Int {
        val s = DoubleArray(3)
        workingToSrgb(r, g, b, s)
        return (0xFF shl 24) or (round255(s[0]) shl 16) or (round255(s[1]) shl 8) or round255(s[2])
    }

    fun round255(v: Double): Int = Math.rint(v * 255.0).toInt().coerceIn(0, 255)

    private fun apply(m: DoubleArray, x: Double, y: Double, z: Double, out: DoubleArray, o: Int) {
        out[o] = m[0] * x + m[1] * y + m[2] * z
        out[o + 1] = m[3] * x + m[4] * y + m[5] * z
        out[o + 2] = m[6] * x + m[7] * y + m[8] * z
    }

    private fun rgbToXyz(primaries: Array<DoubleArray>, white: DoubleArray): DoubleArray {
        val p = DoubleArray(9)
        for ((j, xy) in primaries.withIndex()) {
            val (x, y) = xy[0] to xy[1]
            p[j] = x / y
            p[3 + j] = 1.0
            p[6 + j] = (1 - x - y) / y
        }
        val inv = inverse3(p)
        val s = DoubleArray(3) { r -> inv[3 * r] * white[0] + inv[3 * r + 1] * white[1] + inv[3 * r + 2] * white[2] }
        return DoubleArray(9) { p[it] * s[it % 3] }
    }

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

    private fun mul3(x: DoubleArray, y: DoubleArray) = DoubleArray(9) { k ->
        val r = k / 3
        val c = k % 3
        x[3 * r] * y[c] + x[3 * r + 1] * y[3 + c] + x[3 * r + 2] * y[6 + c]
    }
}
