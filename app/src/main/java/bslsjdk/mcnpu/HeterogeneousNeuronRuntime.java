package bslsjdk.ornithnpu;

import java.util.Arrays;
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
        if (requestedActive < 1 || requestedActive > units) throw new IllegalArgumentException("激活数量超出池大小");
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
        float[] bestScores = new float[requestedActive];
        Arrays.fill(bestScores, -Float.MAX_VALUE);
        for (int u = 0; u < poolSize; u++) {
            float s = score[u];
            for (int k = 0; k < requestedActive; k++) {
                if (s > bestScores[k]) {
                    for (int shift = requestedActive - 1; shift > k; shift--) {
                        bestScores[shift] = bestScores[shift - 1];
                        topIndices[shift] = topIndices[shift - 1];
                    }
                    bestScores[k] = s;
                    topIndices[k] = u;
                    break;
                }
            }
        }
        long elapsed = System.nanoTime() - start;
        return new RouteResult(poolSize, requestedActive, elapsed, bestScores[0],
                absSum / poolSize, estimatedBytes);
    }

    public int size() { return poolSize; }
    public long estimatedBytes() { return estimatedBytes; }

    private static String mib(long bytes) {
        return String.format(Locale.US, "%.2f", bytes / (1024.0 * 1024.0));
    }
}
