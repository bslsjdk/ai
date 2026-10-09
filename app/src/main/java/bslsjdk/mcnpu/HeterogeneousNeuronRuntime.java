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

        RouteResult(int poolSize, int activeCount, long elapsedNanos, double topScore,
                    double meanAbsoluteScore, long estimatedBytes) {
            this.poolSize = poolSize;
            this.activeCount = activeCount;
            this.elapsedNanos = elapsedNanos;
            this.topScore = topScore;
            this.meanAbsoluteScore = meanAbsoluteScore;
            this.estimatedBytes = estimatedBytes;
        }

        public String toReport() {
            return String.format(Locale.US,
                    "路由测试完成（CPU 参考实现）\n池大小：%,d\n激活候选：%d\n耗时：%.3f ms\n最高匹配分：%.6f\n平均绝对分：%.6f\n参数估算：%.2f MiB\n评分说明：当前权重是未训练的可复现初始化值；分数只验证计算管线，不代表已学会任务。",
                    poolSize, activeCount, elapsedNanos / 1_000_000.0, topScore,
                    meanAbsoluteScore, estimatedBytes / (1024.0 * 1024.0));
        }
    }

    private final int poolSize;
    private final float[] weights;
    private final float[] bias;
    private final float[] score;
    private final float[] topScores;
    private final int[] topIndices;
    private final long estimatedBytes;

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
        String reason = "默认 CPU 参考路径；当前版本未把大池路由接到 GPU/NPU 张量后端。";
        // Deliberately do not infer backend availability from device presence or pool size.
        // Future adapters must supply measured break-even data and compatible operators.
        if (npuBackendConnected && ops >= 1_000_000L)
            reason = "NPU 后端已声明连接，但必须通过算子/形状校验及基准测试后才能正式选择。";
        else if (gpuBackendConnected && ops >= 1_000_000L)
            reason = "GPU 后端已声明连接，但必须通过基准测试证明收益后才能正式选择。";
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
        Arrays.fill(score, 0f);
        long start = System.nanoTime();
        double absSum = 0;
        int dims = input.length;
        for (int u = 0; u < poolSize; u++) {
            int offset = u * MAX_INPUTS;
            double value = bias[u];
            for (int j = 0; j < dims; j++) value += weights[offset + j] * input[j];
            float s = (float) Math.tanh(value);
            score[u] = s;
            absSum += Math.abs(s);
        }
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
                absSum / poolSize, estimatedBytes);
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
