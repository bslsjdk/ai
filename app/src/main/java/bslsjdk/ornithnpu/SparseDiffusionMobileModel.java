package bslsjdk.ornithnpu;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.io.FileOutputStream;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.StandardCopyOption;
import java.util.Iterator;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Android inference for aimeng-mobile-diffusion-json-v1 with optional OpenCL GPU acceleration and CPU fallback.
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
    private float[] lastPooled;
    // Persistent residual state carried across separate prompts and generations.
    private float[] residentState;
    private float[] residentEnergy;
    private float[] lastLogits;
    private int lastChosen = -1;
    private java.util.List<LearningTrace> lastTrace = new java.util.ArrayList<>();
    private java.util.List<String> diffusionTrace = new java.util.ArrayList<>();
    private long learningUpdates = 0;
    private String baseModelFingerprint = "";
    private float[] baseDecoderWeight;
    private float[] baseDecoderBias;
    private static final float MAX_DECODER_DRIFT = 0.25f;

    private static final class LearningTrace {
        final float[] pooled;
        final float[] probabilities;
        final int chosen;
        LearningTrace(float[] pooled, float[] probabilities, int chosen) {
            this.pooled = pooled; this.probabilities = probabilities; this.chosen = chosen;
        }
    }

    public synchronized void load(File file) throws Exception {
        loaded = false;
        if (file == null || !file.isFile() || file.length() <= 0 || file.length() > 32L * 1024 * 1024) {
            throw new IllegalArgumentException("portable model file missing or exceeds 32 MiB");
        }
        JSONObject root = new JSONObject(new String(AimengModelFormat.readJsonBytes(file), StandardCharsets.UTF_8));
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
        if (neurons < 8 || neurons > 2048 || width < 4 || width > 128
                || activeK < 1 || activeK > Math.min(neurons, 256)
                || fanout < 1 || fanout > 32 || fanout >= neurons || maxSteps < 1 || maxSteps > 12
                || contextLength < 1 || contextLength > 512) {
            throw new IllegalArgumentException("invalid model dimensions");
        }
        weights.clear();
        shapes.clear();
        JSONObject ts = root.getJSONObject("tensors");
        Iterator<String> tensorKeys = ts.keys();
        while (tensorKeys.hasNext()) {
            String key = tensorKeys.next();
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
        Iterator<String> vocabKeys = vocab.keys();
        while (vocabKeys.hasNext()) { String ch = vocabKeys.next(); stoi.put(ch, vocab.getInt(ch)); }
        JSONArray tokens = root.getJSONArray("itos");
        if (tokens.length() < 2 || tokens.length() > 2048) throw new IllegalArgumentException("vocabulary outside mobile safety limits");
        itos = new String[tokens.length()];
        for (int i = 0; i < itos.length; i++) itos[i] = tokens.getString(i);
        require("embedding.weight"); require("node_embedding"); require("neighbors");
        require("edge_logits"); require("context_proj.weight"); require("context_proj.bias");
        require("self_proj.weight"); require("message_proj.weight"); require("gate.weight");
        require("gate.bias"); require("decoder.weight"); require("decoder.bias");
        require("halt_head.weight"); require("halt_head.bias");
        baseDecoderWeight = w("decoder.weight").clone();
        baseDecoderBias = w("decoder.bias").clone();
        baseModelFingerprint = modelFingerprint();
        residentState = new float[neurons * width];
        residentEnergy = new float[neurons];
        loaded = true;
        lastTrace.clear();
        diffusionTrace.clear();
        // Probe the phone GPU once after a valid model is loaded. Failure is normal
        // on devices without an exposed OpenCL GPU runtime; the CPU path remains valid.
        OpenClGpuBackend.initialize();
    }

    public synchronized boolean isLoaded() { return loaded; }

    public synchronized String backendStatus() {
        return OpenClGpuBackend.statusText();
    }

    public synchronized String generate(String prompt, int count) {
        if (!loaded) throw new IllegalStateException("mobile diffusion model not loaded");
        if (prompt == null || prompt.isEmpty()) throw new IllegalArgumentException("empty prompt");
        if (count < 0 || count > 200) throw new IllegalArgumentException("count must be 0..200");
        StringBuilder out = new StringBuilder(prompt);
        lastTrace = new java.util.ArrayList<>();
        Random rng = new Random();
        for (int i = 0; i < count; i++) {
            int[] codePoints = out.codePoints().toArray();
            int from = Math.max(0, codePoints.length - contextLength);
            int[] ids = new int[codePoints.length - from];
            for (int j = from; j < codePoints.length; j++) {
                String token = new String(Character.toChars(codePoints[j]));
                Integer id = stoi.get(token);
                ids[j - from] = id == null ? 0 : id;
            }
            float[] logits = forward(ids);
            int next = sample(logits, 0.8f, rng);
            if (next >= 0 && next < itos.length) {
                lastTrace.add(new LearningTrace(lastPooled.clone(), probabilities(logits, 0.8f), next));
                lastChosen = next;
                out.append(itos[next]);
            }
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
        float[] energy = residentEnergy == null || residentEnergy.length != neurons
                ? new float[neurons] : residentEnergy.clone();
        for (int n = 0; n < neurons; n++) {
            int base = n * width;
            if (residentState != null && residentState.length == neurons * width)
                System.arraycopy(residentState, base, state[n], 0, width);
            energy[n] = Math.max(0f, Math.min(100f, energy[n] * 0.85f));
        }
        for (int id : seeds) {
            energy[id] = Math.max(1f, energy[id]);
            for (int j = 0; j < width; j++) {
                float fresh = (float) Math.tanh(context[j] + node[id * width + j]);
                state[id][j] = 0.65f * state[id][j] + 0.35f * fresh;
            }
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
        float[] pooledForLearning = new float[width];
        float[] survivalMix = new float[vocab];
        float survival = 1f;
        diffusionTrace.clear();
        float[] neighbors = w("neighbors");
        for (int step = 0; step < maxSteps; step++) {
            int[] active = topK(energy, activeK);
            float[][] messages = new float[neurons][width];
            float[] energyIncoming = new float[neurons];
            int propagatedMessages = 0;
            StringBuilder edgeTrace = new StringBuilder();
            for (int src : active) {
                for (int e = 0; e < fanout; e++) {
                    int dst = (int) neighbors[src * fanout + e];
                    if (dst < 0 || dst >= neurons) continue;
                    float ew = edgeWeights[src * fanout + e];
                    if (propagatedMessages < 12) {
                        if (edgeTrace.length() > 0) edgeTrace.append(", ");
                        edgeTrace.append(src).append("→").append(dst).append("(")
                                .append(String.format(java.util.Locale.ROOT, "%.3f", ew)).append(")");
                    }
                    propagatedMessages++;
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
            float deltaSum = 0f;
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
                    float updated = gate[j] * proposal + (1f - gate[j]) * old[j];
                    deltaSum += Math.abs(updated - old[j]);
                    state[id][j] = updated;
                }
            }
            float deltaMean = deltaSum / Math.max(1, candidates.length * width);
            float energySum = 0;
            for (float e : energy) energySum += e;
            float[] pooled = new float[width];
            if (energySum > 1e-20f) {
                for (int n = 0; n < neurons; n++)
                    for (int j = 0; j < width; j++) pooled[j] += state[n][j] * energy[n] / energySum;
            }
            lastLogits = linear(pooled, "decoder.weight", "decoder.bias");
            pooledForLearning = pooled.clone();
            float halt = sigmoidScalar(linear(pooled, "halt_head.weight", "halt_head.bias")[0]);
            if (step == 0) halt = 0f;
            StringBuilder trace = new StringBuilder();
            trace.append("扩散步 ").append(step + 1)
                    .append(" / ").append(maxSteps)
                    .append("\n活跃更新节点：");
            int shown = 0;
            for (int id : candidates) {
                if (shown++ >= Math.min(8, candidates.length)) break;
                if (shown > 1) trace.append(", ");
                trace.append(id).append("(").append(String.format(java.util.Locale.ROOT, "%.3f", energy[id])).append(")");
            }
            trace.append("\n实际消息传播：").append(propagatedMessages).append(" 条");
            if (edgeTrace.length() > 0) trace.append(" · ").append(edgeTrace);
            trace.append("\n能量总和：").append(String.format(java.util.Locale.ROOT, "%.5f", energySum))
                    .append(" · 状态平均变化：").append(String.format(java.util.Locale.ROOT, "%.6f", deltaMean))
                    .append("\n停止概率：").append(String.format(java.util.Locale.ROOT, "%.2f%%", halt * 100f));
            boolean willStop = step + 1 >= 2 && halt >= 0.80f && deltaMean <= 0.025f;
            trace.append("\n本步决策：").append(willStop ? "满足停止条件" : "继续扩散");
            diffusionTrace.add(trace.toString());
            for (int v = 0; v < vocab; v++) survivalMix[v] += survival * halt * lastLogits[v];
            survival *= (1f - halt);
            if (willStop) break;
        }
        if (lastLogits == null) throw new IllegalStateException("model produced no diffusion steps");
        for (int v = 0; v < vocab; v++) survivalMix[v] += survival * lastLogits[v];
        // Retain bounded state residue across calls. This is actual model state,
        // not a UI-only trace; it is checkpointed by the runtime service.
        if (residentState == null || residentState.length != neurons * width)
            residentState = new float[neurons * width];
        if (residentEnergy == null || residentEnergy.length != neurons)
            residentEnergy = new float[neurons];
        for (int n = 0; n < neurons; n++) {
            System.arraycopy(state[n], 0, residentState, n * width, width);
            residentEnergy[n] = Math.max(0f, Math.min(100f, energy[n]));
        }
        lastPooled = pooledForLearning;
        this.lastLogits = survivalMix;
        return survivalMix;
    }



    public synchronized float[] debugLogits(String prompt) {
        if (!loaded) throw new IllegalStateException("model not loaded");
        if (prompt == null || prompt.isEmpty()) throw new IllegalArgumentException("empty prompt");
        int[] codePoints = prompt.codePoints().toArray();
        int from = Math.max(0, codePoints.length - contextLength);
        int[] ids = new int[codePoints.length - from];
        for (int i = from; i < codePoints.length; i++) {
            Integer id = stoi.get(new String(Character.toChars(codePoints[i])));
            ids[i - from] = id == null ? 0 : id;
        }
        return forward(ids).clone();
    }

    public synchronized int learnFromText(String text, File stateFile) throws Exception {
        if (!loaded) throw new IllegalStateException("model not loaded");
        if (text == null || text.length() < 2) throw new IllegalArgumentException("need at least two characters");
        int[] codePoints = text.codePoints().toArray();
        if (codePoints.length > 512) codePoints = Arrays.copyOf(codePoints, 512);
        int examples = 0;
        float[] matrix = w("decoder.weight");
        float[] bias = w("decoder.bias");
        float lr = 0.01f / Math.max(1, Math.min(64, text.length() - 1));
        for (int pos = 1; pos < codePoints.length; pos++) {
            Integer targetValue = stoi.get(new String(Character.toChars(codePoints[pos])));
            if (targetValue == null || targetValue < 0 || targetValue >= itos.length) continue;
            int start = Math.max(0, pos - contextLength);
            int[] ids = new int[pos - start];
            for (int j = start; j < pos; j++) {
                String token = new String(Character.toChars(codePoints[j]));
                Integer id = stoi.get(token);
                ids[j - start] = id == null ? 0 : id;
            }
            float[] logits = forward(ids);
            float[] p = probabilities(logits, 1f);
            int target = targetValue;
            for (int o = 0; o < itos.length; o++) {
                float grad = (o == target ? 1f : 0f) - p[o];
                grad = Math.max(-0.05f, Math.min(0.05f, grad));
                bias[o] = clamp(bias[o] + lr * grad, baseDecoderBias[o] - MAX_DECODER_DRIFT, baseDecoderBias[o] + MAX_DECODER_DRIFT);
                int base = o * width;
                for (int j = 0; j < width; j++) {
                    float delta = lr * grad * lastPooled[j];
                    matrix[base + j] = clamp(matrix[base + j] + Math.max(-0.002f, Math.min(0.002f, delta)), baseDecoderWeight[base + j] - MAX_DECODER_DRIFT, baseDecoderWeight[base + j] + MAX_DECODER_DRIFT);
                }
            }
            examples++;
        }
        if (examples > 0) {
            learningUpdates++;
            if (stateFile != null) saveLearningState(stateFile);
        }
        lastTrace.clear();
        return examples;
    }

    /** Applies a bounded REINFORCE-style update to the decoder from explicit user feedback.
     * Positive reward reinforces sampled characters; negative reward suppresses them.
     * The first on-device learning stage intentionally updates only the decoder head.
     */
    public synchronized int applyFeedback(float reward, File stateFile) throws Exception {
        if (!loaded) throw new IllegalStateException("model not loaded");
        if (lastTrace.isEmpty()) throw new IllegalStateException("generate text before giving feedback");
        reward = Math.max(-1f, Math.min(1f, reward));
        if (Math.abs(reward) < 0.01f) return 0;
        float[] matrix = w("decoder.weight");
        float[] bias = w("decoder.bias");
        int vocab = itos.length;
        int[] shape = shapes.get("decoder.weight");
        if (shape == null || shape.length != 2 || shape[0] != vocab || shape[1] != width)
            throw new IllegalStateException("decoder shape mismatch");
        float lr = 0.02f / Math.max(1, lastTrace.size());
        int changed = 0;
        for (LearningTrace trace : lastTrace) {
            for (int o = 0; o < vocab; o++) {
                float grad = ((o == trace.chosen ? 1f : 0f) - trace.probabilities[o]) * reward;
                grad = Math.max(-0.05f, Math.min(0.05f, grad));
                if (Math.abs(grad) < 1e-8f) continue;
                bias[o] = clamp(bias[o] + lr * grad, baseDecoderBias[o] - MAX_DECODER_DRIFT, baseDecoderBias[o] + MAX_DECODER_DRIFT);
                int base = o * width;
                for (int j = 0; j < width; j++) {
                    float delta = lr * grad * trace.pooled[j];
                    delta = Math.max(-0.005f, Math.min(0.005f, delta));
                    matrix[base + j] = clamp(matrix[base + j] + delta, baseDecoderWeight[base + j] - MAX_DECODER_DRIFT, baseDecoderWeight[base + j] + MAX_DECODER_DRIFT);
                }
                changed++;
            }
        }
        learningUpdates++;
        if (stateFile != null) saveLearningState(stateFile);
        lastTrace.clear();
        return changed;
    }

    public synchronized void resetLearningState(File stateFile) throws Exception {
        if (!loaded || baseDecoderWeight == null || baseDecoderBias == null)
            throw new IllegalStateException("model not loaded");
        System.arraycopy(baseDecoderWeight, 0, w("decoder.weight"), 0, baseDecoderWeight.length);
        System.arraycopy(baseDecoderBias, 0, w("decoder.bias"), 0, baseDecoderBias.length);
        lastTrace.clear();
        learningUpdates = 0;
        if (stateFile != null && stateFile.exists() && !stateFile.delete())
            throw new java.io.IOException("cannot remove saved learning state");
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    /** Save the resident neural state using a compact binary checkpoint. */
    public synchronized void saveRuntimeState(File stateFile) throws Exception {
        if (!loaded || stateFile == null) return;
        File parent = stateFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs())
            throw new java.io.IOException("cannot create runtime-state directory");
        File temp = new File(parent, stateFile.getName() + ".tmp");
        try (java.io.DataOutputStream out = new java.io.DataOutputStream(
                new java.io.BufferedOutputStream(new FileOutputStream(temp)))) {
            out.writeInt(0x41494D52); // AIMR
            out.writeInt(1);
            out.writeUTF(baseModelFingerprint);
            out.writeInt(neurons);
            out.writeInt(width);
            for (int i = 0; i < neurons * width; i++)
                out.writeFloat(residentState == null ? 0f : residentState[i]);
            for (int i = 0; i < neurons; i++)
                out.writeFloat(residentEnergy == null ? 0f : residentEnergy[i]);
            out.flush();
        }
        try (java.io.RandomAccessFile raf = new java.io.RandomAccessFile(temp, "rw")) {
            raf.getFD().sync();
        }
        try {
            Files.move(temp.toPath(), stateFile.toPath(), StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(temp.toPath(), stateFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /** Restore state only when its dimensions and base-model fingerprint match. */
    public synchronized boolean loadRuntimeState(File stateFile) throws Exception {
        if (!loaded || stateFile == null || !stateFile.isFile()) return false;
        long expectedBytes = 4L + 4L + 2L + baseModelFingerprint.length()
                + 4L + 4L + 4L * ((long) neurons * width + neurons);
        if (stateFile.length() < expectedBytes || stateFile.length() > expectedBytes + 16L)
            return false;
        try (java.io.DataInputStream in = new java.io.DataInputStream(
                new java.io.BufferedInputStream(new java.io.FileInputStream(stateFile)))) {
            if (in.readInt() != 0x41494D52 || in.readInt() != 1) return false;
            if (!baseModelFingerprint.equals(in.readUTF())) return false;
            if (in.readInt() != neurons || in.readInt() != width) return false;
            float[] state = new float[neurons * width];
            float[] energy = new float[neurons];
            for (int i = 0; i < state.length; i++) {
                float v = in.readFloat();
                if (!Float.isFinite(v) || v < -1.5f || v > 1.5f) return false;
                state[i] = v;
            }
            for (int i = 0; i < energy.length; i++) {
                float v = in.readFloat();
                if (!Float.isFinite(v) || v < 0f || v > 100f) return false;
                energy[i] = v;
            }
            residentState = state;
            residentEnergy = energy;
            return true;
        } catch (java.io.EOFException e) {
            return false;
        }
    }

    public synchronized void resetRuntimeState() {
        if (residentState != null) Arrays.fill(residentState, 0f);
        if (residentEnergy != null) Arrays.fill(residentEnergy, 0f);
        lastTrace.clear();
        diffusionTrace.clear();
    }

    public synchronized String getLastDiffusionTraceText() {
        if (diffusionTrace.isEmpty()) return "还没有扩散记录。输入提示词并执行前向计算后，这里会显示真实记录的每一步。";
        StringBuilder out = new StringBuilder();
        for (String step : diffusionTrace) {
            if (out.length() > 0) out.append("\n\n");
            out.append(step);
        }
        return out.toString();
    }

    public synchronized String getModelSummary() {
        return loaded ? "神经元 " + neurons + " · 宽度 " + width + " · 每步激活上限 " + activeK
                + " · 每节点出边 " + fanout + " · 最大扩散步数 " + maxSteps : "模型尚未加载";
    }

    public synchronized long getLearningUpdates() { return learningUpdates; }
    public synchronized boolean canGiveFeedback() { return loaded && !lastTrace.isEmpty(); }

    public synchronized void loadLearningState(File stateFile) throws Exception {
        if (!loaded || stateFile == null || !stateFile.isFile()) return;
        JSONObject state = new JSONObject(new String(Files.readAllBytes(stateFile.toPath()), StandardCharsets.UTF_8));
        if (!"aimeng-mobile-learning-v1".equals(state.optString("format"))
                || !baseModelFingerprint.equals(state.optString("base_model_fingerprint"))
                || state.optInt("vocab", -1) != itos.length
                || state.optInt("width", -1) != width) return;
        JSONArray savedW = state.getJSONArray("decoder_weight");
        JSONArray savedB = state.getJSONArray("decoder_bias");
        float[] matrix = w("decoder.weight");
        float[] bias = w("decoder.bias");
        if (savedW.length() != matrix.length || savedB.length() != bias.length) return;
        for (int i = 0; i < matrix.length; i++) {
            double value = savedW.getDouble(i);
            if (!Double.isFinite(value)) return;
            matrix[i] = clamp((float) value, baseDecoderWeight[i] - MAX_DECODER_DRIFT, baseDecoderWeight[i] + MAX_DECODER_DRIFT);
        }
        for (int i = 0; i < bias.length; i++) {
            double value = savedB.getDouble(i);
            if (!Double.isFinite(value)) return;
            bias[i] = clamp((float) value, baseDecoderBias[i] - MAX_DECODER_DRIFT, baseDecoderBias[i] + MAX_DECODER_DRIFT);
        }
        learningUpdates = state.optLong("updates", 0);
    }

    private void saveLearningState(File stateFile) throws Exception {
        JSONObject root = new JSONObject();
        root.put("format", "aimeng-mobile-learning-v1");
        root.put("base_model_fingerprint", baseModelFingerprint);
        root.put("vocab", itos.length);
        root.put("width", width);
        root.put("updates", learningUpdates);
        JSONArray savedW = new JSONArray();
        for (float x : w("decoder.weight")) savedW.put((double) x);
        JSONArray savedB = new JSONArray();
        for (float x : w("decoder.bias")) savedB.put((double) x);
        root.put("decoder_weight", savedW);
        root.put("decoder_bias", savedB);
        File parent = stateFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) throw new java.io.IOException("cannot create learning-state directory");
        File temp = new File(stateFile.getParentFile(), stateFile.getName() + ".tmp");
        try (FileOutputStream out = new FileOutputStream(temp)) {
            out.write(root.toString().getBytes(StandardCharsets.UTF_8));
            out.getFD().sync();
        }
        if (stateFile.exists() && !stateFile.delete()) throw new java.io.IOException("cannot replace learning state");
        try {
            Files.move(temp.toPath(), stateFile.toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(temp.toPath(), stateFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private String modelFingerprint() {
        String[] keys = weights.keySet().toArray(new String[0]);
        Arrays.sort(keys);
        int h = 1;
        for (String key : keys) {
            h = 31 * h + key.hashCode();
            h = 31 * h + Arrays.hashCode(weights.get(key));
            h = 31 * h + Arrays.hashCode(shapes.get(key));
        }
        return Integer.toHexString(h) + ":" + itos.length + ":" + neurons + ":" + width;
    }

    private static float[] probabilities(float[] logits, float temperature) {
        float max = -Float.MAX_VALUE;
        for (float x : logits) max = Math.max(max, x);
        float[] p = new float[logits.length];
        double sum = 0;
        for (int i = 0; i < logits.length; i++) {
            p[i] = (float) Math.exp((logits[i] - max) / Math.max(0.05f, temperature));
            sum += p[i];
        }
        if (!(sum > 0) || Double.isInfinite(sum) || Double.isNaN(sum)) {
            Arrays.fill(p, 1f / Math.max(1, p.length));
            return p;
        }
        for (int i = 0; i < p.length; i++) p[i] /= (float) sum;
        return p;
    }

    private float[] linear(float[] input, String weightName, String biasName) {
        int[] shape = shapes.get(weightName);
        if (shape == null || shape.length != 2 || shape[1] != input.length)
            throw new IllegalStateException("linear shape mismatch: " + weightName);
        int out = shape[0], in = shape[1];
        float[] matrix = w(weightName);
        float[] bias = biasName == null ? null : w(biasName);
        float[] gpuResult = OpenClGpuBackend.linear(matrix, input, bias, out, in);
        if (gpuResult != null) return gpuResult;
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
