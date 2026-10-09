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

    /** Batched route amortizes accelerator overhead across independent states. */
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
        // Conservative transient estimate includes the CPU reference, transposed
        // weights, Java/native copies and GPU staging buffers, not just tensor payloads.
        long workBytes = ((long)dims*poolSize + (long)batch*poolSize*4L + (long)batch*dims)*4L;
        if (workBytes > 48L*1024L*1024L)
            throw new IllegalArgumentException("estimated batch working set exceeds 48 MiB; reduce batch or pool");
        long t0 = System.nanoTime();
        float[] chosen;
        if (!batchBackendDecisionMade) {
            float[] cpu = cpuBatch(inputs, dims);
            long cpuNs = System.nanoTime()-t0;
            chosen = cpu;
            String backend = "CPU";
            String decision = "CPU_REFERENCE cpu_ms=" + cpuNs/1e6;
            float[] wt = transposeForDims(dims);
            float[] flat = new float[batch*dims];
            for (int r=0;r<batch;r++) System.arraycopy(inputs[r],0,flat,r*dims,dims);
            // Compile/warm the shader before comparing steady-state latency.
            GpuComputeRuntime.matMul(flat,wt,batch,dims,poolSize);
            long g0=System.nanoTime();
            float[] raw=GpuComputeRuntime.matMul(flat,wt,batch,dims,poolSize);
            long gpuNs=System.nanoTime()-g0;
            float[] gpu=null;
            double gpuDiff=Double.POSITIVE_INFINITY;
            if(raw!=null && raw.length==batch*poolSize) {
                gpu=activate(raw);
                gpuDiff=maxDiff(cpu,gpu);
            }
            boolean gpuGood=gpu!=null && gpuDiff<=0.001 && gpuNs*10L<cpuNs*9L;

            float[] npu=null;
            long npuNs=Long.MAX_VALUE;
            double npuDiff=Double.POSITIVE_INFINITY;
            String npuInfo="NPU skipped: not ready or batch/pool too small";
            if(NpuRuntime.isReady() && batch>=8 && poolSize>=4096
                    && maxAbs(inputs)<=1.20 && maxAbs(weights)<=1.20) {
                try {
                    NpuRuntime.prewarmMatMulInt8(batch,dims,poolSize);
                    byte[] a=new byte[batch*dims], b=new byte[dims*poolSize];
                    for(int r=0;r<batch;r++) for(int j=0;j<dims;j++)
                        a[r*dims+j]=q8(inputs[r][j]*0.1f);
                    for(int j=0;j<dims;j++) for(int u=0;u<poolSize;u++)
                        b[j*poolSize+u]=q8(weights[u*MAX_INPUTS+j]*0.1f);
                    long n0=System.nanoTime();
                    byte[] packed=NpuRuntime.matMulInt8Buf(a,b,batch,dims,poolSize);
                    npuNs=System.nanoTime()-n0;
                    if(packed!=null && packed.length>=4+batch*poolSize) {
                        float scaleC=java.nio.ByteBuffer.wrap(packed,0,4)
                            .order(java.nio.ByteOrder.LITTLE_ENDIAN).getFloat();
                        if(Float.isFinite(scaleC)&&scaleC>0) {
                            npu=new float[batch*poolSize];
                            for(int r=0;r<batch;r++) for(int u=0;u<poolSize;u++)
                                npu[r*poolSize+u]=(float)Math.tanh(
                                    packed[4+r*poolSize+u]*scaleC/0.01f+bias[u]);
                            npuDiff=maxDiff(cpu,npu);
                            npuInfo="NPU candidate ms="+npuNs/1e6+" max_abs="+npuDiff;
                        } else npuInfo="NPU invalid output scale";
                    } else npuInfo="NPU failed: "+NpuRuntime.getLastNativeError();
                } catch(Throwable e) { npuInfo="NPU exception: "+e.getClass().getSimpleName()+":"+e.getMessage(); }
            }
            boolean npuGood=npu!=null && npuDiff<=0.02 && npuNs*10L<cpuNs*9L;
            if(npuGood && (!gpuGood || npuNs<gpuNs)) {
                chosen=npu; backend="QNN_HTP_V73_INT8";
                decision="measured NPU faster; cpu_ms="+cpuNs/1e6+" npu_ms="+npuNs/1e6+" max_abs="+npuDiff;
            } else if(gpuGood) {
                chosen=gpu; backend="GLES31_GPU";
                decision="measured GPU faster; cpu_ms="+cpuNs/1e6+" gpu_ms="+gpuNs/1e6+" max_abs="+gpuDiff+"; "+npuInfo;
            } else {
                decision="CPU kept; cpu_ms="+cpuNs/1e6+" gpu_ms="+gpuNs/1e6+" gpu_max_abs="+gpuDiff+"; "+npuInfo;
            }
            batchBackendDecisionMade=true;
            batchBackend=backend;
            batchBackendDecision=decision;
        } else if("GLES31_GPU".equals(batchBackend)) {
            float[] flat=new float[batch*dims];
            for(int r=0;r<batch;r++) System.arraycopy(inputs[r],0,flat,r*dims,dims);
            float[] raw=GpuComputeRuntime.matMul(flat,transposeForDims(dims),batch,dims,poolSize);
            if(raw!=null && raw.length==batch*poolSize) chosen=activate(raw);
            else { chosen=cpuBatch(inputs,dims); batchBackend="CPU"; batchBackendDecision="GPU runtime failure fallback: "+GpuComputeRuntime.getLastError(); }
        } else if("QNN_HTP_V73_INT8".equals(batchBackend)) {
            try {
                byte[] a=new byte[batch*dims],b=new byte[dims*poolSize];
                for(int r=0;r<batch;r++) for(int j=0;j<dims;j++) a[r*dims+j]=q8(inputs[r][j]*0.1f);
                for(int j=0;j<dims;j++) for(int u=0;u<poolSize;u++) b[j*poolSize+u]=q8(weights[u*MAX_INPUTS+j]*0.1f);
                byte[] packed=NpuRuntime.matMulInt8Buf(a,b,batch,dims,poolSize);
                if(packed==null||packed.length<4+batch*poolSize) throw new IllegalStateException(NpuRuntime.getLastNativeError());
                float scaleC=java.nio.ByteBuffer.wrap(packed,0,4).order(java.nio.ByteOrder.LITTLE_ENDIAN).getFloat();
                chosen=new float[batch*poolSize];
                for(int r=0;r<batch;r++) for(int u=0;u<poolSize;u++)
                    chosen[r*poolSize+u]=(float)Math.tanh(packed[4+r*poolSize+u]*scaleC/0.01f+bias[u]);
            } catch(Throwable e) { chosen=cpuBatch(inputs,dims); batchBackend="CPU"; batchBackendDecision="NPU runtime failure fallback: "+e.getMessage(); }
        } else {
            chosen=cpuBatch(inputs,dims);
        }
        int[][] selected=new int[batch][requestedActive];
        for(int r=0;r<batch;r++) {
            float[] row=new float[poolSize];
            System.arraycopy(chosen,r*poolSize,row,0,poolSize);
            selected[r]=topKIndices(row,requestedActive);
        }
        return new BatchRouteResult(poolSize,batch,requestedActive,selected,
                (System.nanoTime()-t0)/1e6,batchBackend,batchBackendDecision);
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
