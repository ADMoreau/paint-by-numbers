package io.github.admoreau.paintbynumbers.camera

import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.os.Build
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
import io.github.admoreau.paintbynumbers.tracking.Intrinsics
import kotlin.math.hypot
import kotlin.math.tan

/** Focal length and principal point in sensor (pixel array) coordinates, and where they came from. */
data class SensorIntrinsics(val fx: Double, val fy: Double, val cx: Double, val cy: Double, val source: String) {
    /** The same camera in pixels of an image buffer, given CameraX's sensor-to-buffer transform. */
    fun toBuffer(sensorToBuffer: Matrix): Intrinsics {
        val c = floatArrayOf(cx.toFloat(), cy.toFloat())
        sensorToBuffer.mapPoints(c)
        val vx = floatArrayOf(fx.toFloat(), 0f)
        val vy = floatArrayOf(0f, fy.toFloat())
        sensorToBuffer.mapVectors(vx)
        sensorToBuffer.mapVectors(vy)
        return Intrinsics(hypot(vx[0], vx[1]).toDouble(), hypot(vy[0], vy[1]).toDouble(), c[0].toDouble(), c[1].toDouble())
    }
}

/**
 * Reads the camera's intrinsics so tracking works on any phone without a calibration step:
 * the lens calibration the phone reports if it is plausible, else focal length and sensor size,
 * else a typical phone field of view.
 */
object CameraIntrinsics {
    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    fun read(info: CameraInfo): SensorIntrinsics? {
        val c2 = Camera2CameraInfo.from(info)
        val active = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE) ?: return null
        val w = active.width().toDouble()
        val h = active.height().toDouble()
        if (Build.VERSION.SDK_INT >= 28) {
            val cal = c2.getCameraCharacteristic(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
            val pre = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE) ?: active
            if (cal != null && cal.size >= 4) {
                val fx = cal[0].toDouble()
                val fy = cal[1].toDouble()
                val cx = cal[2].toDouble() + pre.left
                val cy = cal[3].toDouble() + pre.top
                val plausible = fx in 0.3 * w..5.0 * w && fy in 0.3 * w..5.0 * w &&
                    cx in active.left + 0.3 * w..active.left + 0.7 * w && cy in active.top + 0.3 * h..active.top + 0.7 * h
                if (plausible) return SensorIntrinsics(fx, fy, cx, cy, "lens calibration")
            }
        }
        val cx = active.exactCenterX().toDouble()
        val cy = active.exactCenterY().toDouble()
        val focal = c2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
        val phys = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        val pixels = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
        if (focal != null && focal > 0 && phys != null && phys.width > 0 && pixels != null && pixels.width > 0) {
            val f = focal / (phys.width / pixels.width)
            if (f in 0.3 * w..5.0 * w) return SensorIntrinsics(f.toDouble(), f.toDouble(), cx, cy, "focal length")
        }
        val f = (w / 2) / tan(Math.toRadians(33.0)) // a typical 66 degree main camera
        return SensorIntrinsics(f, f, cx, cy, "assumed")
    }
}
