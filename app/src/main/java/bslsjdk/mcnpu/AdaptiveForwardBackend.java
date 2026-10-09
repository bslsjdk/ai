package bslsjdk.ornithnpu;

import java.util.List;

/**
 * Conservative heterogeneous router. Tiny batches stay on CPU; only batches
 * large enough to amortize QNN dispatch and 32x32x32 padding are sent to HTP.
 * Failed or unsupported NPU execution falls back to CPU.
 *
 * Routes a whole subgraph, not individual neurons, avoiding repeated tensor
 * copies and synchronization between CPU and NPU.
 */
public final class AdaptiveForwardBackend implements NeuronLabEngine.ForwardBackend {
    private static final int MIN_SAMPLES_FOR_NPU = 16;
    private static final int MIN_ACTIVE_UNITS_FOR_NPU = 8;

    private final NeuronLabEngine.ForwardBackend npu;
    private volatile String lastBackend = "CPU";
    private volatile String lastReason = "not_run";

    public AdaptiveForwardBackend(NeuronLabEngine.ForwardBackend npu) {
        if (npu == null) throw new IllegalArgumentException("NPU backend is required");
        this.npu = npu;
    }

    @Override public String name() {
        return lastBackend.equals("CPU") ? "CPU_ROUTED reason=" + lastReason : lastBackend;
    }

    @Override public double[][] predict(List<NeuronLabEngine.Unit> active,
                                        List<NeuronLabEngine.Sample> samples) throws Exception {
        if (active == null || samples == null || active.isEmpty() || samples.isEmpty())
            throw new IllegalArgumentException("empty active units or samples");

        if (samples.size() < MIN_SAMPLES_FOR_NPU || active.size() < MIN_ACTIVE_UNITS_FOR_NPU) {
            lastBackend = "CPU";
            lastReason = "workload_below_npu_break_even samples=" + samples.size()
                    + " active_units=" + active.size()
                    + " thresholds=" + MIN_SAMPLES_FOR_NPU + "x" + MIN_ACTIVE_UNITS_FOR_NPU;
            return cpuPredict(active, samples);
        }

        try {
            double[][] out = npu.predict(active, samples);
            validateShape(out, samples.size(), active.size());
            lastBackend = npu.name();
            lastReason = "npu_selected";
            return out;
        } catch (Throwable failure) {
            lastBackend = "CPU";
            lastReason = "npu_fallback_" + failure.getClass().getSimpleName();
            return cpuPredict(active, samples);
        }
    }

    private static double[][] cpuPredict(List<NeuronLabEngine.Unit> active,
                                         List<NeuronLabEngine.Sample> samples) {
        double[][] out = new double[samples.size()][active.size()];
        for (int r = 0; r < samples.size(); r++)
            for (int c = 0; c < active.size(); c++)
                out[r][c] = active.get(c).predict(samples.get(r).x);
        return out;
    }

    private static void validateShape(double[][] out, int rows, int columns) {
        if (out == null || out.length != rows) throw new IllegalStateException("NPU_BAD_ROW_COUNT");
        for (double[] row : out) {
            if (row == null || row.length != columns) throw new IllegalStateException("NPU_BAD_COLUMN_COUNT");
            for (double value : row)
                if (Double.isNaN(value) || Double.isInfinite(value))
                    throw new IllegalStateException("NPU_NON_FINITE_OUTPUT");
        }
    }
}
