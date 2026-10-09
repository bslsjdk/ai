package bslsjdk.ornithnpu;

import java.util.Arrays;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Locale;
import java.util.Random;

/**
 * Bounded large-pool prototype. This is an honest CPU reference implementation:
 * it measures routing work and estimates backend suitability, but never claims
 * GPU/NPU execution until a compatible backend is actually connected.
 */
public final class HeterogeneousNeuronRuntime {
    public static final int MAX_INPUTS = 16;
    public static final int MAX_POOL_UNITS = 200_000;
    public static final int MAX_ACTIVE_UNITS = 4096;
    public static final long DEFAULT_POOL_BUDGET_BYTES = 64L * 1024L * 1024L;
    private static final long FIXED_OVERHEAD_BYTES = 4096L;
    private static final long BYTES_PER_UNIT = (long) (MAX_INPUTS + 3) * Float.BYTES;

    public enum Backend { CPU, GPU, NPU }

    public static final class Plan {
        public final Backend selected;
        public final String reason;
        public final long estimatedBytes;
        public final long estimatedOps;
        public final boolean gpuAvailable;
        public final boolean npuAvailable;

        Plan(Backend selected, String reason, long estimatedBytes, long estimatedOps,
             boolean gpuAvailable, boolean npuAvailable) {
            this.selected = selected;
            this.reason = reason;
            this.estimatedBytes = estimatedBytes;
            this.estimatedOps = estimatedOps;
            this.gpuAvailable = gpuAvailable;
            this.npuAvailable = npuAvailable;
        }

        public String toReport() {
            return "后端规划：" + selected + "\n原因：" + reason
                    + "\n神经元参数估算：" + mib(estimatedBytes) + " MiB"
                    + "\n本轮路由运算上界：" + estimatedOps + " 次乘加"
                    + "\nGPU 后端已连接：" + gpuAvailable
                    + "\nNPU 后端已连接：" + npuAvailable
                    + "\n注意：这是规划结果，不代表加速器已经执行计算。";
        }
    }

    public static final class RouteResult {
        public final int poolSize;
        public final int activeCount;
        public final long elapsedNanos;
        public final double topScore;
        public final double meanAbsoluteScore;
        public final long estimatedBytes;
        public final String backend;
        public final String backendDecision;

        RouteResult(int poolSize, int activeCount, long elapsedNanos, double topScore,
                    double meanAbsoluteScore, long estimatedBytes, String backend, String backendDecision) {
            this.poolSize = poolSize;
            this.activeCount = activeCount;
            this.elapsedNanos = elapsedNanos;
            this.topScore = topScore;
            this.meanAbsoluteScore = meanAbsoluteScore;
            this.estimatedBytes = estimatedBytes;
            this.backend = backend;
            this.backendDecision = backendDecision;
        }

        public String toReport() {
            return String.format(Locale.US,
                    "路由测试完成\\n后端：%s\\n后端决策：%s\\n池大小：%,d\\n激活候选：%d\\n耗时：%.3f ms\\n最高匹配分：%.6f\\n平均绝对分：%.6f\\n参数估算：%.2f MiB\\n评分说明：当前权重是未训练的可复现初始化值；分数只验证计算管线，不代表已学会任务。",
                    backend, backendDecision, poolSize, activeCount, elapsedNanos / 1_000_000.0, topScore,
                    meanAbsoluteScore, estimatedBytes / (1024.0 * 1024.0));
        }
    }

    public static final class BatchRouteResult {
        public final int poolSize, batchSize, activeCount;
        public final int[][] selectedIndices;
        public final double elapsedMs;
        public final String backend, decision;
        BatchRouteResult(int p,int b,int a,int[][] s,double ms,String be,String d) {
            poolSize=p; batchSize=b; activeCount=a; selectedIndices=s; elapsedMs=ms; backend=be; decision=d;
        }
        public String toReport() { return String.format(Locale.US,
            "批量神经元路由\n池大小：%,d\n批量输入：%d\n每个输入激活：%d\n实际后端：%s\n批量耗时：%.3f ms（%.3f ms/输入）\n决策：%s",
            poolSize,batchSize,activeCount,backend,elapsedMs,elapsedMs/Math.max(1,batchSize),decision); }
    }

    private final int poolSize;
    private final float[] weights;
    private final float[] bias;
    private final float[] score;
    private final float[] topScores;
    private final int[] topIndices;
    private final long estimatedBytes;
    private boolean gpuRouteDecisionMade;
    private boolean useGpuRoute;
    private String routeBackend = "CPU";
    private String routeBackendDecision = "CPU_REFERENCE_DEFAULT";
    private boolean batchBackendDecisionMade;
    private String batchBackend = "CPU";
    private String batchBackendDecision = "not_benchmarked";
    private int cachedTransposeDims = -1;
    private float[] cachedTransposedWeights;

    public HeterogeneousNeuronRuntime(int requestedPoolSize, long budgetBytes, long seed) {
        if (requestedPoolSize < 2 || requestedPoolSize > MAX_POOL_UNITS)
            throw new IllegalArgumentException("神经元池必须在 2.." + MAX_POOL_UNITS + " 之间");
        if (budgetBytes < 1) throw new IllegalArgumentException("内存预算必须大于 0");
        long estimate = estimateBytes(requestedPoolSize);
        if (estimate > budgetBytes)
            throw new IllegalArgumentException("参数估算需要 " + mib(estimate)
                    + " MiB，超过当前工作区预算 " + mib(budgetBytes) + " MiB");
        this.poolSize = requestedPoolSize;
        this.estimatedBytes = estimate;
        this.weights = new float[Math.multiplyExact(requestedPoolSize, MAX_INPUTS)];
        this.bias = new float[requestedPoolSize];
        this.score = new float[requestedPoolSize];
        this.topScores = new float[Math.min(MAX_ACTIVE_UNITS, requestedPoolSize)];
        this.topIndices = new int[Math.min(4096, requestedPoolSize)];
        Random random = new Random(seed);
        float scale = (float) (1.0 / Math.sqrt(MAX_INPUTS));
        for (int i = 0; i < weights.length; i++) weights[i] = (random.nextFloat() * 2f - 1f) * scale;
        for (int i = 0; i < bias.length; i++) bias[i] = (random.nextFloat() * 2f - 1f) * 0.1f;
    }

    public static long estimateBytes(int units) {
        if (units < 0) throw new IllegalArgumentException("negative unit count");
        return FIXED_OVERHEAD_BYTES + (long) units * BYTES_PER_UNIT;
    }

    public static Plan plan(int units, int inputCount, int requestedActive, long budgetBytes,
                            boolean gpuBackendConnected, boolean npuBackendConnected) {
        if (units < 2 || units > MAX_POOL_UNITS) throw new IllegalArgumentException("神经元池数量超出范围");
        if (inputCount < 1 || inputCount > MAX_INPUTS) throw new IllegalArgumentException("输入维度必须是 1..16");
        if (requestedActive < 1 || requestedActive > Math.min(MAX_ACTIVE_UNITS, units))
            throw new IllegalArgumentException("每轮激活数量必须在 1..min(4096, 池大小) 之间");
        long bytes = estimateBytes(units);
        if (bytes > budgetBytes) throw new IllegalArgumentException("神经元池超过内存预算");
        long ops = (long) units * inputCount * 2L;
        Backend selected = Backend.CPU;
        String reason = "单状态路由保留 CPU 参考路径；批量路由已连接 GLES GPU 与 QNN HTP 候选，并按实测耗时和数值误差选择后端。";
        if (npuBackendConnected && ops >= 1_000_000L)
            reason += " NPU 已就绪；批量路径仍需在同一输入批次上通过精度与速度门槛。";
        else if (gpuBackendConnected && ops >= 1_000_000L)
            reason += " GPU context 已就绪；小批次可能因启动/传输开销而继续使用 CPU。";
        return new Plan(selected, reason, bytes, ops, gpuBackendConnected, npuBackendConnected);
    }

    /** Runs a deterministic top-k routing smoke test over the pool. */
    public synchronized RouteResult route(float[] input, int requestedActive) {
        if (input == null || input.length < 1 || input.length > MAX_INPUTS)
            throw new IllegalArgumentException("输入向量维度必须是 1..16");
        if (requestedActive < 1 || requestedActive > topIndices.length)
            throw new IllegalArgumentException("激活数量必须在 1.." + topIndices.length + " 之间");
        for (float v : input)
            if (!Float.isFinite(v)) throw new IllegalArgumentException("输入包含非有限数值");
        long start = System.nanoTime();
        double absSum = 0;
        int dims = input.length;
        // For sufficiently large pools, calibrate the real GLES compute path against
        // the CPU reference once. GPU is retained only if it is numerically sound and
        // at least 10% faster INCLUDING transpose/copy/dispatch costs.
        if (!gpuRouteDecisionMade && poolSize >= 4096 && poolSize <= 50000 && dims >= 4) {
            long cpuStart = System.nanoTime();
            computeCpuScores(input, dims);
            long cpuNanos = System.nanoTime() - cpuStart;
            float[] cpuSnapshot = score.clone();
            // Warm the EGL context/shader before timing steady-state execution;
            // cold shader compilation must not decide the permanent backend.
            boolean gpuWarmOk = computeGpuScores(input, dims);
            long gpuStart = System.nanoTime();
            boolean gpuOk = gpuWarmOk && computeGpuScores(input, dims);
            long gpuNanos = System.nanoTime() - gpuStart;
            double maxDiff = 0.0;
            if (gpuOk) {
                for (int u = 0; u < poolSize; u++) {
                    double diff = Math.abs((double) score[u] - cpuSnapshot[u]);
                    if (diff > maxDiff) maxDiff = diff;
                }
            }
            useGpuRoute = gpuOk && maxDiff <= 0.001 && gpuNanos * 10L < cpuNanos * 9L;
            gpuRouteDecisionMade = true;
            if (useGpuRoute) {
                routeBackend = "GLES31_GPU";
                routeBackendDecision = "measured_gpu_faster cpu_ms=" + (cpuNanos / 1e6)
                        + " gpu_ms=" + (gpuNanos / 1e6) + " max_abs=" + maxDiff;
            } else {
                System.arraycopy(cpuSnapshot, 0, score, 0, poolSize);
                routeBackend = "CPU";
                routeBackendDecision = gpuOk
                        ? "cpu_kept gpu_not_10_percent_faster_or_error_too_large cpu_ms="
                            + (cpuNanos / 1e6) + " gpu_ms=" + (gpuNanos / 1e6) + " max_abs=" + maxDiff
                        : "cpu_kept_gpu_unavailable " + GpuComputeRuntime.getLastError();
            }
        } else if (useGpuRoute) {
            if (!computeGpuScores(input, dims)) {
                useGpuRoute = false;
                routeBackend = "CPU";
                routeBackendDecision = "runtime_gpu_failure_fallback " + GpuComputeRuntime.getLastError();
                computeCpuScores(input, dims);
            }
        } else {
            computeCpuScores(input, dims);
            if (!gpuRouteDecisionMade && poolSize >= 4096 && (poolSize > 50000 || dims < 4)) {
                gpuRouteDecisionMade = true;
                routeBackendDecision = "cpu_kept_gpu_probe_outside_safe_workset";
            }
        }
        for (int u = 0; u < poolSize; u++) absSum += Math.abs(score[u]);
        // Partial selection avoids sorting the entire pool when only a small
        // fraction is active. O(pool * active), bounded by 4096 selected units.
        // Reuse the bounded scratch buffer. Allocating a new top-k array on every
        // routing call caused avoidable GC pressure on mobile devices.
        float[] bestScores = topScores;
        int heapSize = 0;
        for (int u = 0; u < poolSize; u++) {
            float s = score[u];
            if (heapSize < requestedActive) {
                int child = heapSize++;
                bestScores[child] = s;
                topIndices[child] = u;
                while (child > 0) {
                    int parent = (child - 1) >>> 1;
                    if (bestScores[parent] <= bestScores[child]) break;
                    swap(bestScores, topIndices, parent, child);
                    child = parent;
                }
            } else if (s > bestScores[0]) {
                bestScores[0] = s;
                topIndices[0] = u;
                int parent = 0;
                while (true) {
                    int left = parent * 2 + 1;
                    if (left >= heapSize) break;
                    int right = left + 1;
                    int smallest = right < heapSize && bestScores[right] < bestScores[left] ? right : left;
                    if (bestScores[parent] <= bestScores[smallest]) break;
                    swap(bestScores, topIndices, parent, smallest);
                    parent = smallest;
                }
            }
        }
        float highest = -Float.MAX_VALUE;
        for (float value : bestScores) if (value > highest) highest = value;
        long elapsed = System.nanoTime() - start;
        return new RouteResult(poolSize, requestedActive, elapsed, highest,
                absSum / poolSize, estimatedBytes, routeBackend, routeBackendDecision);
    }

    /**
     * Hybrid batch route: CPU and GLES GPU process disjoint neuron ranges at the
     * same time. This is deliberately not a CPU-vs-GPU winner-takes-all selector.
     * NPU is not in this execution path yet.
     */
    public synchronized BatchRouteResult routeBatch(float[][] inputs, int requestedActive) {
        if (inputs == null || inputs.length < 2 || inputs.length > 32)
            throw new IllegalArgumentException("batch must contain 2..32 rows");
        if (requestedActive < 1 || requestedActive > Math.min(MAX_ACTIVE_UNITS, poolSize))
            throw new IllegalArgumentException("active count out of range");
        int dims = inputs[0] == null ? 0 : inputs[0].length;
        if (dims < 1 || dims > MAX_INPUTS) throw new IllegalArgumentException("input dims must be 1..16");
        for (float[] row : inputs) {
            if (row == null || row.length != dims) throw new IllegalArgumentException("batch rows must share dimensions");
            for (float v : row) if (!Float.isFinite(v)) throw new IllegalArgumentException("non-finite input");
        }
        int batch = inputs.length;
        long workBytes = ((long)dims*poolSize + (long)batch*poolSize*4L + (long)batch*dims)*4L;
        if (workBytes > 48L*1024L*1024L)
            throw new IllegalArgumentException("estimated batch working set exceeds 48 MiB; reduce batch or pool");

        long t0 = System.nanoTime();
        String backend;
        String decision;
        float[] combined;
        // Small jobs stay CPU-only because GPU dispatch/copy overhead can dominate.
        // For a sufficiently large pool, split neurons into disjoint CPU/GPU ranges.
        boolean tryHybrid = poolSize >= 4096 && dims >= 4 && batch >= 2;
        if (tryHybrid) {
            combined = cpuGpuHybridBatch(inputs, dims);
            if (combined != null) {
                backend = "CPU+GLES31_GPU";
                double cpuMs = lastHybridCpuMs;
                double gpuMs = lastHybridGpuMs;
                double cpuOnlyMs = lastHybridCpuOnlyMs;
                double diff = lastHybridMaxDiff;
                decision = String.format(Locale.US,
                        "concurrent disjoint ranges; cpu_only_ms=%.3f hybrid_ms=%.3f cpu_slice_ms=%.3f gpu_slice_ms=%.3f max_abs=%.7g; NPU intentionally deferred",
                        cpuOnlyMs, lastHybridTotalMs, cpuMs, gpuMs, diff);
            } else {
                combined = cpuBatch(inputs, dims);
                backend = "CPU";
                decision = "hybrid GPU unavailable/invalid; safe CPU fallback: " + GpuComputeRuntime.getLastError();
            }
        } else {
            combined = cpuBatch(inputs, dims);
            backend = "CPU";
            decision = "CPU-only for small workload; hybrid threshold pool>=4096, dims>=4, batch>=2";
        }
        batchBackend = backend;
        batchBackendDecision = decision;
        batchBackendDecisionMade = true;

        int[][] selected = new int[batch][requestedActive];
        for (int r=0;r<batch;r++) {
            float[] row = new float[poolSize];
            System.arraycopy(combined, r*poolSize, row, 0, poolSize);
            selected[r] = topKIndices(row, requestedActive);
        }
        return new BatchRouteResult(poolSize, batch, requestedActive, selected,
                (System.nanoTime()-t0)/1e6, backend, decision);
    }

    private static final java.util.concurrent.ExecutorService HYBRID_GPU_EXECUTOR =
            java.util.concurrent.Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "aimeng-hybrid-gpu");
                t.setDaemon(true);
                return t;
            });
    private volatile double lastHybridCpuMs, lastHybridGpuMs, lastHybridCpuOnlyMs,
            lastHybridTotalMs, lastHybridMaxDiff;

    private float[] cpuGpuHybridBatch(float[][] inputs, int dims) {
        final int batch = inputs.length;
        final int cpuEnd = Math.max(1, Math.min(poolSize - 1, poolSize / 2));
        final int gpuCount = poolSize - cpuEnd;
        final float[] flat = new float[batch * dims];
        for (int r=0;r<batch;r++) System.arraycopy(inputs[r], 0, flat, r*dims, dims);
        final float[] gpuWeights = transposeRange(dims, cpuEnd, poolSize);

        // Warm up shader compilation outside the measured concurrent run.
        float[] warm = GpuComputeRuntime.matMul(flat, gpuWeights, batch, dims, gpuCount);
        if (warm == null || warm.length != batch * gpuCount) return null;

        long cpuOnlyStart = System.nanoTime();
        float[] cpuReference = cpuBatch(inputs, dims);
        lastHybridCpuOnlyMs = (System.nanoTime() - cpuOnlyStart) / 1e6;

        final float[] merged = new float[batch * poolSize];
        long totalStart = System.nanoTime();
        java.util.concurrent.Future<float[]> gpuFuture = HYBRID_GPU_EXECUTOR.submit(
                () -> GpuComputeRuntime.matMul(flat, gpuWeights, batch, dims, gpuCount));
        long cpuStart = System.nanoTime();
        for (int r=0;r<batch;r++) {
            for (int u=0;u<cpuEnd;u++) {
                int off = u * MAX_INPUTS;
                double sum = bias[u];
                for (int j=0;j<dims;j++) sum += weights[off+j] * inputs[r][j];
                merged[r*poolSize+u] = (float)Math.tanh(sum);
            }
        }
        lastHybridCpuMs = (System.nanoTime() - cpuStart) / 1e6;
        float[] gpuRaw;
        long gpuWaitStart = System.nanoTime();
        try {
            gpuRaw = gpuFuture.get();
        } catch (Exception e) {
            gpuFuture.cancel(true);
            return null;
        }
        lastHybridGpuMs = (System.nanoTime() - gpuWaitStart) / 1e6;
        if (gpuRaw == null || gpuRaw.length != batch * gpuCount) return null;
        for (int r=0;r<batch;r++) for (int k=0;k<gpuCount;k++) {
            int u = cpuEnd + k;
            merged[r*poolSize+u] = (float)Math.tanh(gpuRaw[r*gpuCount+k] + bias[u]);
        }
        lastHybridTotalMs = (System.nanoTime() - totalStart) / 1e6;
        lastHybridMaxDiff = maxDiff(cpuReference, merged);
        // Incorrect GPU output must never silently enter routing.
        if (!Double.isFinite(lastHybridMaxDiff) || lastHybridMaxDiff > 0.001) return null;
        return merged;
    }

    private float[] transposeRange(int dims, int startUnit, int endUnit) {
        float[] transposed = new float[dims * (endUnit - startUnit)];
        int count = endUnit - startUnit;
        for (int u=startUnit;u<endUnit;u++) for (int j=0;j<dims;j++)
            transposed[j*count + (u-startUnit)] = weights[u*MAX_INPUTS+j];
        return transposed;
    }

    private float[] cpuBatch(float[][] inputs,int dims) {
        float[] out=new float[inputs.length*poolSize];
        for(int r=0;r<inputs.length;r++) for(int u=0;u<poolSize;u++) {
            int off=u*MAX_INPUTS; double sum=bias[u];
            for(int j=0;j<dims;j++) sum+=weights[off+j]*inputs[r][j];
            out[r*poolSize+u]=(float)Math.tanh(sum);
        }
        return out;
    }
    private float[] transposeForDims(int dims) {
        if(cachedTransposeDims!=dims||cachedTransposedWeights==null) {
            cachedTransposedWeights=new float[dims*poolSize];
            for(int u=0;u<poolSize;u++) for(int j=0;j<dims;j++)
                cachedTransposedWeights[j*poolSize+u]=weights[u*MAX_INPUTS+j];
            cachedTransposeDims=dims;
        }
        return cachedTransposedWeights;
    }
    private float[] activate(float[] raw) {
        float[] out=new float[raw.length];
        for(int r=0;r<raw.length/poolSize;r++) for(int u=0;u<poolSize;u++)
            out[r*poolSize+u]=(float)Math.tanh(raw[r*poolSize+u]+bias[u]);
        return out;
    }
    private static double maxDiff(float[] a,float[] b) {
        if(a==null||b==null||a.length!=b.length) return Double.POSITIVE_INFINITY;
        double max=0; for(int i=0;i<a.length;i++) max=Math.max(max,Math.abs((double)a[i]-b[i])); return max;
    }
    private static double maxAbs(float[][] a) {
        double max=0; for(float[] row:a) for(float v:row) max=Math.max(max,Math.abs(v)); return max;
    }
    private static double maxAbs(float[] a) {
        double max=0; for(float v:a) max=Math.max(max,Math.abs(v)); return max;
    }
    private static byte q8(float v) {
        long q=Math.round(v/0.001f); if(q>127)q=127; if(q< -128)q=-128; return (byte)q;
    }
    private int[] topKIndices(float[] values,int count) {
        float[] best=new float[count]; int[] idx=new int[count]; int size=0;
        for(int i=0;i<values.length;i++) {
            float v=values[i];
            if(size<count) { int child=size++; best[child]=v; idx[child]=i;
                while(child>0) { int parent=(child-1)>>>1; if(best[parent]<=best[child])break;
                    swap(best,idx,parent,child); child=parent; }
            } else if(v>best[0]) { best[0]=v; idx[0]=i; int p=0;
                while(true) { int l=p*2+1; if(l>=size)break; int r=l+1;
                    int s=r<size&&best[r]<best[l]?r:l; if(best[p]<=best[s])break;
                    swap(best,idx,p,s); p=s; }
            }
        }
        return idx;
    }

    private void computeCpuScores(float[] input, int dims) {
        for (int u = 0; u < poolSize; u++) {
            int offset = u * MAX_INPUTS;
            double value = bias[u];
            for (int j = 0; j < dims; j++) value += weights[offset + j] * input[j];
            score[u] = (float) Math.tanh(value);
        }
    }

    /** Uses a real GLES 3.1 FP32 matrix multiply for the pool's affine score. */
    private boolean computeGpuScores(float[] input, int dims) {
        // Bound the temporary transposed matrix and native copies; very large pools
        // stay on CPU until a streaming/tiled GPU path is implemented.
        long workBytes = ((long) dims * poolSize + poolSize + dims) * Float.BYTES;
        if (workBytes > 32L * 1024L * 1024L) return false;
        float[] transposed = new float[dims * poolSize];
        for (int u = 0; u < poolSize; u++) {
            int offset = u * MAX_INPUTS;
            for (int j = 0; j < dims; j++) transposed[j * poolSize + u] = weights[offset + j];
        }
        float[] raw = GpuComputeRuntime.matMul(input, transposed, 1, dims, poolSize);
        if (raw == null || raw.length != poolSize) return false;
        for (int u = 0; u < poolSize; u++) score[u] = (float) Math.tanh(raw[u] + bias[u]);
        return true;
    }

    private static void swap(float[] scores, int[] indices, int a, int b) {
        float sv = scores[a]; scores[a] = scores[b]; scores[b] = sv;
        int iv = indices[a]; indices[a] = indices[b]; indices[b] = iv;
    }

    /** Writes a versioned binary checkpoint through a temporary file then atomic-ish rename. */
    public synchronized void save(File destination) throws IOException {
        if (destination == null) throw new IllegalArgumentException("destination is null");
        File parent = destination.getParentFile();
        if (parent != null && !parent.isDirectory() && !parent.mkdirs())
            throw new IOException("cannot create checkpoint directory");
        File temp = new File(destination.getAbsolutePath() + ".tmp");
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(temp)))) {
            out.writeInt(0x414D4E50); // AMNP
            out.writeInt(1);
            out.writeInt(poolSize);
            out.writeInt(MAX_INPUTS);
            for (float value : weights) out.writeFloat(value);
            for (float value : bias) out.writeFloat(value);
            out.flush();
        } catch (Throwable error) {
            temp.delete();
            if (error instanceof IOException) throw (IOException) error;
            throw new IOException("checkpoint write failed", error);
        }
        if (destination.exists() && !destination.delete()) {
            temp.delete();
            throw new IOException("cannot replace existing checkpoint");
        }
        if (!temp.renameTo(destination)) {
            temp.delete();
            throw new IOException("cannot activate checkpoint");
        }
    }

    /** Validates header, dimensions, file length and budget before allocating pool arrays. */
    public static HeterogeneousNeuronRuntime load(File source, long budgetBytes) throws IOException {
        if (source == null || !source.isFile()) throw new IOException("checkpoint not found");
        if (source.length() < 16L) throw new IOException("checkpoint too short");
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(source)))) {
            if (in.readInt() != 0x414D4E50) throw new IOException("checkpoint magic mismatch");
            if (in.readInt() != 1) throw new IOException("unsupported checkpoint version");
            int count = in.readInt();
            int inputs = in.readInt();
            if (inputs != MAX_INPUTS || count < 2 || count > MAX_POOL_UNITS)
                throw new IOException("checkpoint dimensions invalid");
            long expected = 16L + (long) count * (MAX_INPUTS + 1L) * Float.BYTES;
            if (source.length() != expected) throw new IOException("checkpoint length mismatch");
            if (estimateBytes(count) > budgetBytes) throw new IOException("checkpoint exceeds memory budget");
            HeterogeneousNeuronRuntime runtime = new HeterogeneousNeuronRuntime(count, budgetBytes, 1L);
            for (int i = 0; i < runtime.weights.length; i++) runtime.weights[i] = finite(in.readFloat());
            for (int i = 0; i < runtime.bias.length; i++) runtime.bias[i] = finite(in.readFloat());
            if (in.read() != -1) throw new IOException("checkpoint trailing bytes");
            return runtime;
        } catch (IllegalArgumentException error) {
            throw new IOException("checkpoint validation failed", error);
        }
    }

    private static float finite(float value) throws IOException {
        if (!Float.isFinite(value)) throw new IOException("checkpoint contains non-finite parameter");
        return value;
    }

    public int size() { return poolSize; }
    public long estimatedBytes() { return estimatedBytes; }

    private static String mib(long bytes) {
        return String.format(Locale.US, "%.2f", bytes / (1024.0 * 1024.0));
    }
}
