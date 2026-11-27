package io.github.admoreau.paintbynumbers.project

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.DocumentsContract
import android.provider.MediaStore
import androidx.activity.result.contract.ActivityResultContracts

/**
 * The app's own folder in the phone's shared storage, Download/Paint by Numbers: plans
 * (project.pbn) and images go there, the marker sheet is saved there, and the app's file pickers
 * open there, so files for the app are easy to find.
 */
object AppFolder {
    const val NAME = "Paint by Numbers"
    const val RELATIVE_PATH = "Download/$NAME/"
    private const val README = "About this folder.txt"

    /** The folder as a document, for a picker's starting place. */
    val uri: Uri = DocumentsContract.buildDocumentUri("com.android.externalstorage.documents", "primary:Download/$NAME")

    /** Creates the folder (with a short note in it, as a folder can't exist empty in shared storage) if it isn't there. */
    fun ensure(context: Context) {
        if (Build.VERSION.SDK_INT < 29) return
        val resolver = context.contentResolver
        val found = resolver.query(
            MediaStore.Downloads.EXTERNAL_CONTENT_URI, arrayOf(MediaStore.Downloads._ID),
            "${MediaStore.Downloads.RELATIVE_PATH} = ? AND ${MediaStore.Downloads.DISPLAY_NAME} = ?", arrayOf(RELATIVE_PATH, README), null,
        )?.use { it.count > 0 } ?: false
        if (found) return
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, README)
            put(MediaStore.Downloads.MIME_TYPE, "text/plain")
            put(MediaStore.Downloads.RELATIVE_PATH, RELATIVE_PATH)
        }
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: return
        resolver.openOutputStream(uri)?.use {
            it.write(
                ("Files for the Paint by Numbers app.\n\n" +
                    "Put plans from the desktop planner (project.pbn) and images to paint here: " +
                    "the app's Import plan and Choose image open this folder first. The marker sheet is saved here too.\n").toByteArray(),
            )
        }
    }

    /** A save into the folder, for MediaStore.Downloads. */
    fun values(name: String, mime: String) = ContentValues().apply {
        put(MediaStore.Downloads.DISPLAY_NAME, name)
        put(MediaStore.Downloads.MIME_TYPE, mime)
        put(MediaStore.Downloads.RELATIVE_PATH, RELATIVE_PATH)
    }

    /** The system file picker, opening in the app's folder (where the phone allows choosing the start). */
    class Open : ActivityResultContracts.OpenDocument() {
        override fun createIntent(context: Context, input: Array<String>): Intent =
            super.createIntent(context, input).apply {
                if (Build.VERSION.SDK_INT >= 26) putExtra(DocumentsContract.EXTRA_INITIAL_URI, uri)
            }
    }
}
