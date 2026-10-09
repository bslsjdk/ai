package bslsjdk.ornithnpu;

/**
 * Runtime GLES 3.1 compute backend. This is a real GPU compute path, not a
 * device-name guess. It performs FP32 matrix multiplication and returns null
 * with lastError set when the device/driver cannot execute the shader.
 */
public final class GpuComputeRuntime {
    private static volatile String lastError = "not initialized";
    static {
        try { System.loadLibrary("ornithnpu"); }
        catch (Throwable t) { lastError = "ERR_LOAD_LIBRARY " + t.getClass().getSimpleName() + ": " + t.getMessage(); }
    }
    private GpuComputeRuntime() {}

    public static String getLastError() { return lastError; }

    public static float[] matMul(float[] a, float[] b, int m, int k, int n) {
        if (a == null || b == null || m < 1 || k < 1 || n < 1
                || (long) m * k != a.length || (long) k * n != b.length) {
            lastError = "ERR_BAD_ARGUMENTS";
            return null;
        }
        long bytes = ((long) a.length + b.length + (long) m * n) * 4L;
        if (bytes > 64L * 1024L * 1024L) {
            lastError = "ERR_GPU_WORKSET_LIMIT bytes=" + bytes + " cap=67108864";
            return null;
        }
        try {
            float[] out = nativeMatMul(a, b, m, k, n);
            if (out == null) {
                lastError = nativeLastError();
                if (lastError == null || lastError.isEmpty()) lastError = "ERR_GPU_NULL_RESULT";
            } else {
                lastError = "OK";
            }
            return out;
        } catch (Throwable t) {
            lastError = "ERR_GPU_EXCEPTION " + t.getClass().getSimpleName() + ": " + t.getMessage();
            return null;
        }
    }

    public static String status() {
        try {
            String nativeStatus = nativeStatus();
            return nativeStatus == null ? "GPU_STATUS_UNKNOWN" : nativeStatus;
        } catch (Throwable t) {
            return "GPU_UNAVAILABLE " + t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    private static native float[] nativeMatMul(float[] a, float[] b, int m, int k, int n);
    private static native String nativeLastError();
    private static native String nativeStatus();
}
