package bslsjdk.mcnpu;

import java.io.File;

/**
 * Runtime boundary for the single local Ornith-1.5-9B-MLX-4bit model.
 *
 * The native side owns the real Safetensors/MLX executor and QNN HTP path.
 * There is no network endpoint, provider registry, or cloud-model fallback.
 */
public final class Ornith15Runtime {
    private static volatile boolean loaded;
    private static volatile String lastInfo = "NOT_LOADED";

    private Ornith15Runtime() {}

    public static synchronized String load(String modelPath, long contextTokens) {
        if (modelPath == null || modelPath.isEmpty())
            return "ERR ORNITH15_RUNTIME null_model";
        // First long-context stage: real 64K resident attention KV. The model
        // itself advertises 262K, but larger resident KV would violate the phone
        // memory ceiling with the current FP16 cache design.
        if (contextTokens <= 0 || contextTokens > 65536)
            return "ERR ORNITH15_RUNTIME context=" + contextTokens + " first_stage_max=65536";

        File f = new File(modelPath);
        if (!f.isFile() || f.length() <= 0)
            return "ERR ORNITH15_RUNTIME model_missing=" + modelPath;

        try {
            System.loadLibrary("mcnpu");
            String r = nativeLoad(modelPath, contextTokens);
            lastInfo = r == null ? "ERR ORNITH15_RUNTIME null_native_reply" : r;
            loaded = r != null && r.startsWith("OK ORNITH15_RUNTIME/1") && !r.contains("inference=MLX4BIT_PROBE_ONLY");
            return lastInfo;
        } catch (Throwable t) {
            loaded = false;
            lastInfo = "ERR ORNITH15_RUNTIME " + t.getClass().getSimpleName() + ": " + t.getMessage();
            return lastInfo;
        }
    }

    /**
     * Runs the loaded local model synchronously through the native inference path.
     * No network transport or remote provider is involved.
     */
    public static synchronized String generate(String prompt, int maxTokens) {
        if (!loaded) return "ERR ORNITH15_RUNTIME not_loaded";
        if (prompt == null || prompt.isEmpty()) return "ERR ORNITH15_RUNTIME empty_prompt";
        try {
            String r = nativeGenerate(prompt, maxTokens);
            return r == null ? "ERR ORNITH15_RUNTIME null_generate" : r;
        } catch (Throwable t) {
            return "ERR ORNITH15_RUNTIME generate_" + t.getClass().getSimpleName();
        }
    }

    public static synchronized String info() {
        try {
            String r = nativeInfo();
            return r == null ? lastInfo : r;
        } catch (Throwable t) {
            return lastInfo;
        }
    }

    public static synchronized void unload() {
        try { nativeUnload(); } catch (Throwable ignored) {}
        loaded = false;
        lastInfo = "NOT_LOADED";
    }

    public static boolean isLoaded() {
        return loaded;
    }

    private static native String nativeLoad(String modelPath, long contextTokens);
    private static native String nativeGenerate(String prompt, int maxTokens);
    private static native String nativeInfo();
    private static native void nativeUnload();
}
