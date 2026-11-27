package io.github.admoreau.paintbynumbers.project

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.pdf.PdfDocument
import android.os.Build
import android.provider.MediaStore
import io.github.admoreau.paintbynumbers.tracking.MarkerFinder
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.Objdetect

/**
 * A printable sheet of square markers (4x4 ArUco, ids from 0) at a chosen size, with a scale bar to
 * check the printer did not resize it. Saved to Downloads.
 */
object MarkerSheet {
    private const val PT_PER_MM = 72 / 25.4

    /** Writes the PDF; returns its display name, or null when this Android version cannot save to Downloads. */
    fun save(context: Context, sideMm: Double): String? {
        if (Build.VERSION.SDK_INT < 29) return null
        val pageW = 612 // US Letter in points; fits A4 too with the margins used
        val pageH = 792
        val margin = 15 * PT_PER_MM
        val side = sideMm * PT_PER_MM
        val gap = maxOf(0.5 * side, 8 * PT_PER_MM)
        val cols = ((pageW - 2 * margin + gap) / (side + gap)).toInt().coerceAtLeast(1)
        val rows = ((pageH - 2 * margin - 20 + gap) / (side + gap + 10)).toInt().coerceAtLeast(1)
        val count = minOf(cols * rows, MarkerFinder.DICTIONARY_SIZE)

        val doc = PdfDocument()
        val page = doc.startPage(PdfDocument.PageInfo.Builder(pageW, pageH, 1).create())
        val c = page.canvas
        val text = Paint().apply { textSize = 8f; isAntiAlias = true }
        val crisp = Paint().apply { isFilterBitmap = false; isAntiAlias = false }
        val dict = Objdetect.getPredefinedDictionary(Objdetect.DICT_4X4_50)
        for (i in 0 until count) {
            val col = i % cols
            val row = i / cols
            val x = margin + col * (side + gap)
            val y = margin + 20 + row * (side + gap + 10)
            val mat = Mat()
            Objdetect.generateImageMarker(dict, i, 6 * 20, mat, 1) // 6 cells (4 data + border), 20 px each
            val rgba = Mat()
            Imgproc.cvtColor(mat, rgba, Imgproc.COLOR_GRAY2RGBA)
            val bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(rgba, bmp)
            mat.release(); rgba.release()
            c.drawBitmap(bmp, Rect(0, 0, bmp.width, bmp.height), RectF(x.toFloat(), y.toFloat(), (x + side).toFloat(), (y + side).toFloat()), crisp)
            bmp.recycle()
            c.drawText("$i", x.toFloat(), (y + side + 9).toFloat(), text)
        }
        // scale check
        val bar = 50 * PT_PER_MM
        val stroke = Paint().apply { strokeWidth = 1f }
        c.drawLine(margin.toFloat(), (margin + 6).toFloat(), (margin + bar).toFloat(), (margin + 6).toFloat(), stroke)
        c.drawText("This line should measure 50 mm. Markers: %.0f mm. Cut them out with a white border and stick them around the canvas, flat.".format(sideMm), margin.toFloat(), (margin + 16).toFloat(), text)
        doc.finishPage(page)

        val name = "paint-by-numbers markers %.0fmm.pdf".format(sideMm)
        val values = AppFolder.values(name, "application/pdf")
        val uri = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: return null
        context.contentResolver.openOutputStream(uri)!!.use { doc.writeTo(it) }
        doc.close()
        return name
    }
}
