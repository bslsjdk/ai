package bslsjdk.ornithnpu;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * CPU-only Android inference for aimeng-mobile-diffusion-json-v1.
 * Kept separate from Ornith15Runtime: this is a different model architecture.
 * The portable bundle is produced by aimeng/scripts/export_mobile_diffusion.py.
 */
public final class SparseDiffusionMobileModel {
    private final Map<String, float[]> weights = new HashMap<>();
    private final Map<String, int[]> shapes = new HashMap<>();
    private final Map<String, Integer> stoi = new HashMap<>();
    private String[] itos;
    private int neurons, width, activeK, fanout, maxSteps, contextLength;
    private boolean loaded;

    public synchronized void load(File file) throws Exception {
        if (file == null || !file.isFile() || file.length() <= 0) {
            throw new IllegalArgumentException("portable model file missing");
        }
        JSONObject root = new JSONObject(Files.readString(file.toPath(), StandardCharsets.UTF_8));
        if (!"aimeng-mobile-diffusion-json-v1".equals(root.optString("format"))) {
            throw new IllegalArgumentException("unsupported mobile model format");
        }
        JSONObject cfg = root.getJSONObject("config");
        neurons = cfg.getInt("neurons");
        width = cfg.getInt("width");
        activeK = cfg.getInt("active_k");
        fanout = cfg.getInt("fanout");
        maxSteps = cfg.getInt("max_steps");
        contextLength = cfg.optInt("context", 64);
        if (neurons < 8 || width < 4 || activeK < 1 || activeK > neurons
                || fanout < 1 || fanout >= neurons || maxSteps < 1 || maxSteps > 32) {
            throw new IllegalArgumentException("invalid model dimensions");
        }
        weights.clear();
        shapes.clear();
        JSONObject ts = root.getJSONObject("tensors");
        for (String key : ts.keySet()) {
            JSONObject t = ts.getJSONObject(key);
            JSONArray shapeJson = t.getJSONArray("shape");
            int[] shape = new int[shapeJson.length()];
            int expected = 1;
            for (int i = 0; i < shape.length; i++) {
                shape[i] = shapeJson.getInt(i);
                if (shape[i] < 0 || (shape[i] > 0 && expected > Integer.MAX_VALUE / shape[i])) {
                    throw new IllegalArgumentException("invalid tensor shape: " + key);
                }
                expected *= shape[i];
            }
            JSONArray values = t.getJSONArray("values");
            if (values.length() != expected) {
                throw new IllegalArgumentException("tensor size mismatch: " + key);
            }
            float[] data = new float[expected];
            for (int i = 0; i < expected; i++) data[i] = (float) values.getDouble(i);
            weights.put(key, data);
            shapes.put(key, shape);
        }
        stoi.clear();
        JSONObject vocab = root.getJSONObject("stoi");
        for (String ch : vocab.keySet()) stoi.put(ch, vocab.getInt(ch));
        JSONArray tokens = root.getJSONArray("itos");
        itos = new String[tokens.length()];
        for (int i = 0; i < itos.length; i++) itos[i] = tokens.getString(i);
        require("embedding.weight"); require("node_embedding"); require("neighbors");
        require("edge_logits"); require("context_proj.weight"); require("context_proj.bias");
        require("self_proj.weight"); require("message_proj.weight"); require("gate.weight");
        require("gate.bias"); require("decoder.weight"); require("decoder.bias");
        require("halt_head.weight"); require("halt_head.bias");
        loaded = true;
    }

    public synchronized boolean isLoaded() { return loaded; }

    public synchronized String generate(String prompt, int count) {
        if (!loaded) throw new IllegalStateException("mobile diffusion model not loaded");
        if (prompt == null || prompt.isEmpty()) throw new IllegalArgumentException("empty prompt");
        if (count < 0 || count > 200) throw new IllegalArgumentException("count must be 0..200");
        StringBuilder out = new StringBuilder(prompt);
        Random rng = new Random();
        for (int i = 0; i < count; i++) {
            int from = Math.max(0, out.length() - contextLength);
            String context = out.substring(from);
            int[] ids = new int[context.length()];
            for (int j = 0; j < context.length(); j++) {
                Integer id = stoi.get(context.substring(j, j + 1));
                ids[j] = id == null ? 0 : id;
            }
            float[] logits = forward(ids);
            int next = sample(logits, 0.8f, rng);
            if (next >= 0 && next < itos.length) out.append(itos[next]);
        }
        return out.toString();
    }

    private float[] forward(int[] tokenIds) {
        int vocab = itos.length;
        float[] embedding = w("embedding.weight");
        float[] context = new float[width];
        for (int id : tokenIds) {
            int safe = id >= 0 && id < vocab ? id : 0;
            int base = safe * width;
            for (int j = 0; j < width; j++) context[j] += embedding[base + j];
        }
        float denom = Math.max(1, tokenIds.length);
        for (int j = 0; j < width; j++) context[j] /= denom;
        context = tanh(linear(context, "context_proj.weight", "context_proj.bias"));

        float[] node = w("node_embedding");
        float[] route = new float[neurons];
        for (int n = 0; n < neurons; n++) {
            float dot = 0;
            for (int j = 0; j < width; j++) dot += context[j] * node[n * width + j];
            route[n] = (float) (dot / Math.sqrt(width));
        }
        int[] seeds = topK(route, activeK);
        float[][] state = new float[neurons][width];
        float[] energy = new float[neurons];
        for (int id : seeds) {
            energy[id] = 1f;
            for (int j = 0; j < width; j++) state[id][j] = (float) Math.tanh(context[j] + node[id * width + j]);
        }

        float[] edgeLogits = w("edge_logits");
        float[] edgeWeights = new float[neurons * fanout];
        for (int n = 0; n < neurons; n++) {
            float max = -Float.MAX_VALUE;
            for (int e = 0; e < fanout; e++) max = Math.max(max, edgeLogits[n * fanout + e]);
            float sum = 0;
            for (int e = 0; e < fanout; e++) {
                float v = (float) Math.exp(edgeLogits[n * fanout + e] - max);
                edgeWeights[n * fanout + e] = v; sum += v;
            }
            for (int e = 0; e < fanout; e++) edgeWeights[n * fanout + e] /= Math.max(sum, 1e-20f);
        }
        float[] lastLogits = null;
        float[] survivalMix = new float[vocab];
        float survival = 1f;
        float[] neighbors = w("neighbors");
        for (int step = 0; step < maxSteps; step++) {
            int[] active = topK(energy, activeK);
            float[][] messages = new float[neurons][width];
            float[] energyIncoming = new float[neurons];
            for (int src : active) {
                for (int e = 0; e < fanout; e++) {
                    int dst = (int) neighbors[src * fanout + e];
                    if (dst < 0 || dst >= neurons) continue;
                    float ew = edgeWeights[src * fanout + e];
                    float absMean = 0;
                    for (int j = 0; j < width; j++) {
                        float msg = state[src][j] * ew;
                        messages[dst][j] += msg;
                        absMean += Math.abs(msg);
                    }
                    energyIncoming[dst] += absMean / width;
                }
            }
            for (int n = 0; n < neurons; n++) energy[n] = energy[n] * 0.9f + energyIncoming[n];
            int[] candidates = topK(energy, activeK);
            for (int id : candidates) {
                float[] old = state[id].clone();
                float[] incoming = messages[id];
                float[] self = linear(old, "self_proj.weight", null);
                float[] msg = linear(incoming, "message_proj.weight", null);
                float[] joined = new float[2 * width];
                System.arraycopy(old, 0, joined, 0, width);
                System.arraycopy(incoming, 0, joined, width, width);
                float[] gate = sigmoid(linear(joined, "gate.weight", "gate.bias"));
                for (int j = 0; j < width; j++) {
                    float proposal = (float) Math.tanh(self[j] + msg[j]);
                    state[id][j] = gate[j] * proposal + (1f - gate[j]) * old[j];
                }
            }
            float energySum = 0;
            for (float e : energy) energySum += e;
            float[] pooled = new float[width];
            if (energySum > 1e-20f) {
                for (int n = 0; n < neurons; n++)
                    for (int j = 0; j < width; j++) pooled[j] += state[n][j] * energy[n] / energySum;
            }
            lastLogits = linear(pooled, "decoder.weight", "decoder.bias");
            float halt = sigmoidScalar(linear(pooled, "halt_head.weight", "halt_head.bias")[0]);
            if (step == 0) halt = 0f;
            for (int v = 0; v < vocab; v++) survivalMix[v] += survival * halt * lastLogits[v];
            survival *= (1f - halt);
        }
        if (lastLogits == null) throw new IllegalStateException("model produced no diffusion steps");
        for (int v = 0; v < vocab; v++) survivalMix[v] += survival * lastLogits[v];
        return survivalMix;
    }

    private float[] linear(float[] input, String weightName, String biasName) {
        int[] shape = shapes.get(weightName);
        if (shape == null || shape.length != 2 || shape[1] != input.length)
            throw new IllegalStateException("linear shape mismatch: " + weightName);
        int out = shape[0], in = shape[1];
        float[] matrix = w(weightName);
        float[] bias = biasName == null ? null : w(biasName);
        float[] result = new float[out];
        for (int o = 0; o < out; o++) {
            float sum = bias == null ? 0f : bias[o];
            int base = o * in;
            for (int i = 0; i < in; i++) sum += matrix[base + i] * input[i];
            result[o] = sum;
        }
        return result;
    }

    private static float[] tanh(float[] x) {
        for (int i = 0; i < x.length; i++) x[i] = (float) Math.tanh(x[i]);
        return x;
    }
    private static float[] sigmoid(float[] x) {
        for (int i = 0; i < x.length; i++) x[i] = sigmoidScalar(x[i]);
        return x;
    }
    private static float sigmoidScalar(float x) {
        if (x >= 0) { float z = (float) Math.exp(-x); return 1f / (1f + z); }
        float z = (float) Math.exp(x); return z / (1f + z);
    }
    private static int[] topK(float[] values, int k) {
        int[] ids = new int[k];
        boolean[] chosen = new boolean[values.length];
        Arrays.fill(ids, -1);
        for (int slot = 0; slot < k; slot++) {
            int best = -1; float score = -Float.MAX_VALUE;
            for (int i = 0; i < values.length; i++) {
                if (!chosen[i] && values[i] > score) { score = values[i]; best = i; }
            }
            ids[slot] = best;
            if (best >= 0) chosen[best] = true;
        }
        return ids;
    }
    private static int sample(float[] logits, float temperature, Random rng) {
        float max = -Float.MAX_VALUE;
        for (float x : logits) max = Math.max(max, x);
        double sum = 0;
        double[] probs = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = Math.exp((logits[i] - max) / Math.max(0.05f, temperature));
            sum += probs[i];
        }
        if (!(sum > 0) || Double.isInfinite(sum) || Double.isNaN(sum)) {
            int best = 0;
            for (int i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
            return best;
        }
        double target = rng.nextDouble() * sum;
        for (int i = 0; i < probs.length; i++) if ((target -= probs[i]) <= 0) return i;
        return probs.length - 1;
    }
    private float[] w(String key) {
        float[] result = weights.get(key);
        if (result == null) throw new IllegalStateException("missing tensor: " + key);
        return result;
    }
    private void require(String key) {
        if (!weights.containsKey(key)) throw new IllegalArgumentException("portable bundle missing tensor: " + key);
    }
}
