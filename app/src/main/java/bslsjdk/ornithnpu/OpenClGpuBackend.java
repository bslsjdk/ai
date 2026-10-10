package bslsjdk.ornithnpu;

/**
 * Optional Android GPU compute backend using the device OpenCL runtime.
 * No OpenCL SDK is linked into the APK: the vendor runtime is discovered at
 * runtime. If the phone has no usable GPU OpenCL implementation, callers fall
 * back to the Java CPU path.
 */
public final class OpenClGpuBackend {
    private static boolean libraryLoaded;
    private static boolean initialized;
    private static boolean failed;
    private static String device = "GPU 未初始化";
    private static long gpuOperations;

    static {
        try {
            System.loadLibrary("ornithnpu");
            libraryLoaded = true;
        } catch (Throwable ignored) {
            libraryLoaded = false;
            failed = true;
            device = "GPU 不可用：本机计算库未加载";
        }
    }

    private OpenClGpuBackend() {}

    public static synchronized boolean initialize() {
        if (initialized) return true;
        if (failed || !libraryLoaded) return false;
        try {
            initialized = nativeInit();
            if (initialized) {
                String name = nativeDeviceName();
                device = (name == null || name.isEmpty()) ? "OpenCL GPU" : name;
            } else {
                failed = true;
                device = "GPU 不可用：没有可用的 OpenCL 计算设备";
            }
        } catch (Throwable error) {
            failed = true;
            device = "GPU 初始化失败：" + error.getClass().getSimpleName();
        }
        return initialized;
    }

    public static synchronized boolean isAvailable() {
        return initialized && !failed;
    }

    public static synchronized String deviceName() { return device; }

    public static synchronized String statusText() {
        if (!isAvailable()) return "CPU 兜底（未检测到可用 OpenCL GPU）";
        return gpuOperations == 0
                ? "GPU/OpenCL：" + device + "（已就绪，尚未执行达到阈值的大矩阵）"
                : "GPU/OpenCL：" + device + "（已执行 " + gpuOperations + " 次 GPU 矩阵运算；小矩阵走 CPU）";
    }

    /**
     * Returns null if the GPU operation failed; callers must then run the CPU
     * implementation. The GPU path is deliberately used only for sufficiently
     * large matrices to avoid paying driver and transfer overhead on tiny ops.
     */
    public static synchronized float[] linear(float[] matrix, float[] input, float[] bias,
                                               int out, int in) {
        if (!isAvailable() || (long) out * in < 4096L) return null;
        try {
            float[] result = nativeLinear(matrix, input, bias, out, in);
            if (result == null || result.length != out) {
                failed = true;
                device = "GPU 执行失败，已切换 CPU";
                try { nativeShutdown(); } catch (Throwable ignored) {}
                return null;
            }
            gpuOperations++;
            return result;
        } catch (Throwable error) {
            failed = true;
            device = "GPU 执行失败，已切换 CPU：" + error.getClass().getSimpleName();
            try { nativeShutdown(); } catch (Throwable ignored) {}
            return null;
        }
    }

    public static synchronized void shutdown() {
        if (!initialized) return;
        try { nativeShutdown(); } catch (Throwable ignored) {}
        initialized = false;
        device = "GPU 已停止";
    }

    private static native boolean nativeInit();
    private static native String nativeDeviceName();
    private static native float[] nativeLinear(float[] matrix, float[] input, float[] bias,
                                                int out, int in);
    private static native void nativeShutdown();
}
