package io.github.admoreau.paintbynumbers.planner

import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** A fixed pool of background threads for splitting loops across the phone's cores. */
object Parallel {
    val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 8)
    private val pool: ExecutorService = Executors.newFixedThreadPool(threads) { r ->
        Thread(r, "planner").apply { isDaemon = true }
    }

    /**
     * Runs [body] over [0, n) split into contiguous chunks; returns when all are done. There are
     * several chunks per thread, handed out as threads come free, because phones mix fast and slow
     * cores and an even split would leave the fast ones waiting.
     */
    fun forChunks(n: Int, body: (from: Int, to: Int) -> Unit) {
        if (n <= 0) return
        val parts = minOf(threads * 4, n)
        if (threads == 1 || parts == 1) return body(0, n)
        val next = java.util.concurrent.atomic.AtomicInteger()
        val tasks = (0 until minOf(threads, parts)).map {
            Callable {
                while (true) {
                    val p = next.getAndIncrement()
                    if (p >= parts) break
                    body((n.toLong() * p / parts).toInt(), (n.toLong() * (p + 1) / parts).toInt())
                }
            }
        }
        for (f in pool.invokeAll(tasks)) f.get() // rethrows a worker's exception
    }

    /** Runs [body] for each i in [0, n), handing out indices one at a time (for uneven work). */
    fun forEach(n: Int, body: (Int) -> Unit) {
        if (n <= 0) return
        val next = java.util.concurrent.atomic.AtomicInteger()
        val parts = minOf(threads, n)
        val tasks = (0 until parts).map {
            Callable {
                while (true) {
                    val i = next.getAndIncrement()
                    if (i >= n) break
                    body(i)
                }
            }
        }
        for (f in pool.invokeAll(tasks)) f.get()
    }
}
