package bslsjdk.ornithnpu;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

/**
 * Executes the affine neuron forward pass through the existing QNN/HTP V73
 * INT8 MatMul graph. The learning rule and score controller remain on CPU.
 */
public final class NpuNeuronForward implements NeuronLabEngine.ForwardBackend {
    private static final double FEATURE_SCALE = 0.05;
    private static final double PRODUCT_SCALE = FEATURE_SCALE * FEATURE_SCALE;

    @Override public String name() { return "QNN_HTP_V73_INT8_MATMUL"; }

    @Override public double[][] predict(List<NeuronLabEngine.Unit> active,
                                        List<NeuronLabEngine.Sample> samples) throws Exception {
        if (!NpuRuntime.isReady()) throw new IllegalStateException("NPU_NOT_READY " + NpuRuntime.getLastError());
        if (active == null || active.isEmpty() || samples == null || samples.isEmpty())
            throw new IllegalArgumentException("empty active units or samples");
        // HTP accepts only whitelisted matrix dimensions. Pad all three axes to
        // the known-good 32 bucket rather than sending tiny 4x2x4 shapes that
        // native mmSizeAllowed() will reject.
        final int m = 32, k = 32, n = 32;
        if (samples.size() > m || active.size() > n)
            throw new IllegalArgumentException("NPU lab bucket supports at most 32 samples and units");
        byte[] a = new byte[m * k];
        byte[] b = new byte[k * n];
        for (int r = 0; r < samples.size(); r++) {
            a[r * k] = quantize(samples.get(r).x * FEATURE_SCALE);
            a[r * k + 1] = quantize(FEATURE_SCALE);
        }
        for (int c = 0; c < active.size(); c++) {
            b[c] = quantize(active.get(c).weight * FEATURE_SCALE);
            b[n + c] = quantize(active.get(c).bias * FEATURE_SCALE);
        }
        byte[] raw = NpuRuntime.matMulInt8Buf(a, b, m, k, n);
        if (raw == null || raw.length < 4 + m * n) {
            throw new IllegalStateException("HTP_MATMUL_FAILED " + NpuRuntime.getLastNativeError());
        }
        ByteBuffer buffer = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        float scaleC = buffer.getFloat();
        if (!Float.isFinite(scaleC) || scaleC <= 0.0f)
            throw new IllegalStateException("HTP_INVALID_OUTPUT_SCALE " + scaleC);
        double[][] out = new double[samples.size()][active.size()];
        for (int r = 0; r < samples.size(); r++) {
            for (int c = 0; c < active.size(); c++) {
                int q = raw[4 + r * n + c];
                out[r][c] = ((double) q * scaleC) / PRODUCT_SCALE;
            }
        }
        return out;
    }

    private static byte quantize(double value) {
        long q = Math.round(value / 0.001);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        return (byte) q;
    }
}
