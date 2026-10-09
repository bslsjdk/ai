package bslsjdk.ornithnpu;

import java.util.ArrayDeque;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.LongSupplier;

/**
 * Memory-aware FIFO scheduler for expensive compute jobs.
 *
 * Jobs remain queued while current RSS + their estimated incremental allocation
 * + the configured safety reserve would exceed the hard runtime memory limit.
 * It deliberately serializes jobs so GPU/NPU buffers and CPU scratch arrays do
 * not multiply peak memory through concurrent execution.
 */
public final class MemoryAwareComputeScheduler implements AutoCloseable {
    public static final long HARD_LIMIT_BYTES = 4L * 1024L * 1024L * 1024L;
    public static final long DEFAULT_RESERVE_BYTES = 512L * 1024L * 1024L;
    private static final int MAX_QUEUED_JOBS = 64;
    private static final long POLL_INTERVAL_MS = 500L;

    public static final class Snapshot {
        public final int queuedJobs;
        public final String activeJob;
        public final long currentRssBytes;
        public final long hardLimitBytes;
        public final long reserveBytes;
        public final long deferredJobs;
        public final String lastDecision;

        Snapshot(int queuedJobs, String activeJob, long currentRssBytes,
                 long hardLimitBytes, long reserveBytes, long deferredJobs,
                 String lastDecision) {
            this.queuedJobs = queuedJobs;
            this.activeJob = activeJob;
            this.currentRssBytes = currentRssBytes;
            this.hardLimitBytes = hardLimitBytes;
            this.reserveBytes = reserveBytes;
            this.deferredJobs = deferredJobs;
            this.lastDecision = lastDecision;
        }

        public String toReport() {
            return "queued_jobs=" + queuedJobs
                    + "\nactive_job=" + activeJob
                    + "\nrss_mib=" + mib(currentRssBytes)
                    + "\nhard_limit_mib=" + mib(hardLimitBytes)
                    + "\nreserve_mib=" + mib(reserveBytes)
                    + "\ndeferred_jobs=" + deferredJobs
                    + "\nlast_decision=" + lastDecision;
        }
    }

    private static final class Job<T> {
        final String name;
        final long estimatedBytes;
        final Callable<T> callable;
        final FutureTask<T> future;

        Job(String name, long estimatedBytes, Callable<T> callable) {
            this.name = name;
            this.estimatedBytes = estimatedBytes;
            this.callable = callable;
            this.future = new FutureTask<>(callable);
        }
    }

    private final LongSupplier rssBytesSupplier;
    private final long hardLimitBytes;
    private final long reserveBytes;
    private final ArrayDeque<Job<?>> queue = new ArrayDeque<>();
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final Thread dispatcher;
    private String activeJob = "none";
    private String lastDecision = "idle";
    private long deferredJobs;

    public MemoryAwareComputeScheduler(LongSupplier rssBytesSupplier) {
        this(rssBytesSupplier, HARD_LIMIT_BYTES, DEFAULT_RESERVE_BYTES);
    }

    public MemoryAwareComputeScheduler(LongSupplier rssBytesSupplier,
                                       long hardLimitBytes, long reserveBytes) {
        if (rssBytesSupplier == null) throw new IllegalArgumentException("RSS supplier required");
        if (hardLimitBytes < 1 || reserveBytes < 0 || reserveBytes >= hardLimitBytes)
            throw new IllegalArgumentException("invalid memory budget");
        this.rssBytesSupplier = rssBytesSupplier;
        this.hardLimitBytes = hardLimitBytes;
        this.reserveBytes = reserveBytes;
        dispatcher = new Thread(this::dispatchLoop, "aimeng-memory-aware-dispatch");
        dispatcher.setDaemon(true);
        dispatcher.start();
    }

    public <T> Future<T> submit(String name, long estimatedAdditionalBytes,
                                Callable<T> callable) {
        if (name == null || name.trim().isEmpty()) throw new IllegalArgumentException("job name required");
        if (estimatedAdditionalBytes < 0) throw new IllegalArgumentException("negative memory estimate");
        if (callable == null) throw new IllegalArgumentException("callable required");
        Job<T> job = new Job<>(name, estimatedAdditionalBytes, callable);
        synchronized (queue) {
            if (closed.get()) throw new IllegalStateException("scheduler is closed");
            if (queue.size() >= MAX_QUEUED_JOBS) throw new IllegalStateException("compute queue is full");
            queue.addLast(job);
            lastDecision = "QUEUED " + name;
            queue.notifyAll();
        }
        return job.future;
    }

    public Snapshot snapshot() {
        long rss = safeRssBytes();
        synchronized (queue) {
            return new Snapshot(queue.size(), activeJob, rss, hardLimitBytes,
                    reserveBytes, deferredJobs, lastDecision);
        }
    }

    private void dispatchLoop() {
        while (!closed.get()) {
            Job<?> job;
            synchronized (queue) {
                while (queue.isEmpty() && !closed.get()) {
                    try { queue.wait(); } catch (InterruptedException interrupted) {
                        if (closed.get()) return;
                    }
                }
                if (closed.get()) return;
                job = queue.peekFirst();
            }
            if (job == null) continue;
            if (job.future.isCancelled()) {
                synchronized (queue) {
                    queue.removeFirstOccurrence(job);
                    lastDecision = "CANCELLED " + job.name;
                    queue.notifyAll();
                }
                continue;
            }

            long rss = safeRssBytes();
            if (rss < 0) {
                synchronized (queue) {
                    lastDecision = "DEFERRED " + job.name + " because RSS is unavailable";
                    deferredJobs++;
                }
                waitForMemoryChange();
                continue;
            }
            long projected;
            try {
                projected = Math.addExact(Math.addExact(rss, job.estimatedBytes), reserveBytes);
            } catch (ArithmeticException overflow) {
                projected = Long.MAX_VALUE;
            }
            if (projected > hardLimitBytes) {
                synchronized (queue) {
                    lastDecision = "DEFERRED " + job.name + " projected_mib=" + mib(projected)
                            + " hard_limit_mib=" + mib(hardLimitBytes);
                    deferredJobs++;
                }
                waitForMemoryChange();
                continue;
            }

            synchronized (queue) {
                if (queue.peekFirst() != job) continue;
                queue.removeFirst();
                activeJob = job.name;
                lastDecision = "RUNNING " + job.name;
            }
            try {
                if (!job.future.isCancelled()) job.future.run();
            } finally {
                synchronized (queue) {
                    activeJob = "none";
                    lastDecision = "FINISHED " + job.name;
                    queue.notifyAll();
                }
            }
        }
    }

    private long safeRssBytes() {
        try { return rssBytesSupplier.getAsLong(); } catch (Throwable ignored) { return -1L; }
    }

    private void waitForMemoryChange() {
        synchronized (queue) {
            if (closed.get()) return;
            try { queue.wait(POLL_INTERVAL_MS); } catch (InterruptedException ignored) {
                if (closed.get()) return;
            }
        }
    }

    private static String mib(long bytes) {
        return bytes < 0 ? "unknown" : String.format(java.util.Locale.US, "%.1f", bytes / (1024.0 * 1024.0));
    }

    @Override public void close() {
        if (!closed.compareAndSet(false, true)) return;
        synchronized (queue) {
            for (Job<?> job : queue) job.future.cancel(false);
            queue.clear();
            queue.notifyAll();
        }
        dispatcher.interrupt();
    }
}
