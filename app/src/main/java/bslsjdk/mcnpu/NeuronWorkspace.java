package bslsjdk.ornithnpu;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Small, bounded, serializable multi-input / multi-output neural-network workspace.
 * The current model is one tanh hidden layer plus a linear output layer, trained on CPU.
 * It is intentionally bounded for mobile safety; it is not a language model.
 */
public final class NeuronWorkspace {
    public static final String FORMAT = "aimeng-neuron-workspace/v1";
    public static final String DATA_FORMAT = "aimeng-dataset/v1";
    public static final String NEURON_FORMAT = "aimeng-neuron/v1";
    public static final String NEURON_PACK_FORMAT = "aimeng-neuron-pack/v1";
    public static final String TASK_FORMAT = "aimeng-training-task/v1";
    public static final int MAX_INPUTS = 16;
    public static final int MAX_OUTPUTS = 16;
    public static final int MAX_NEURONS = 128;
    public static final int MAX_SAMPLES = 5000;
    public static final int MAX_EPOCHS = 5000;
    public static final int MAX_SAVED_NEURONS = 512;
    /** Separate workspace budget leaves headroom for Android, JNI and QNN allocations. */
    public static final long MAX_WORKSPACE_BUDGET_BYTES = 64L * 1024L * 1024L;
    private static final int MAX_HISTORY = 100;
    private static final int MAX_EVENTS = 300;
    private static final double ADAM_BETA1 = 0.9;
    private static final double ADAM_BETA2 = 0.999;

    public static final class Sample {
        public final double[] input;
        public final double[] output;

        public Sample(double[] input, double[] output) {
            if (input == null || output == null || input.length < 1 || output.length < 1)
                throw new IllegalArgumentException("sample input/output cannot be empty");
            this.input = input.clone();
            this.output = output.clone();
            for (double v : this.input) if (!Double.isFinite(v))
                throw new IllegalArgumentException("input contains a non-finite value");
            for (double v : this.output) if (!Double.isFinite(v))
                throw new IllegalArgumentException("output contains a non-finite value");
        }

        JSONObject toJson() throws JSONException {
            JSONObject o = new JSONObject();
            o.put("input", toJsonArray(input));
            o.put("output", toJsonArray(output));
            return o;
        }
    }

    public static final class Neuron {
        public String id;
        public double[] inputWeights;
        public double bias;
        public double[] outputWeights;
        public boolean enabled = true;
        public double score;
        public double resourceScore;
        public long estimatedBytes;
        public int revision = 1;
        public String libraryName = "";
        public long savedAt;

        Neuron(String id, int inputs, int outputs) {
            this.id = id;
            inputWeights = new double[inputs];
            outputWeights = new double[outputs];
        }

        Neuron copy() {
            Neuron n = new Neuron(id, inputWeights.length, outputWeights.length);
            n.inputWeights = inputWeights.clone();
            n.outputWeights = outputWeights.clone();
            n.bias = bias;
            n.enabled = enabled;
            n.score = score;
            n.resourceScore = resourceScore;
            n.estimatedBytes = estimatedBytes;
            n.revision = revision;
            n.libraryName = libraryName;
            n.savedAt = savedAt;
            return n;
        }

        JSONObject toJson() throws JSONException {
            JSONObject o = new JSONObject();
            o.put("id", id);
            o.put("inputWeights", toJsonArray(inputWeights));
            o.put("bias", bias);
            o.put("outputWeights", toJsonArray(outputWeights));
            o.put("enabled", enabled);
            o.put("score", score);
            o.put("resourceScore", resourceScore);
            o.put("estimatedBytes", estimatedBytes);
            o.put("revision", revision);
            if (!libraryName.isEmpty()) o.put("libraryName", libraryName);
            if (savedAt > 0) o.put("savedAt", savedAt);
            return o;
        }

        static Neuron fromJson(JSONObject o, int inputs, int outputs) throws JSONException {
            Neuron n = new Neuron(o.optString("id", "H-import"), inputs, outputs);
            n.inputWeights = readArray(o.optJSONArray("inputWeights"), inputs, "inputWeights");
            n.outputWeights = readArray(o.optJSONArray("outputWeights"), outputs, "outputWeights");
            n.bias = finiteOrThrow(o.optDouble("bias", 0), "bias");
            n.enabled = o.optBoolean("enabled", true);
            n.score = finiteOrThrow(o.optDouble("score", 0), "score");
            n.resourceScore = finiteOrThrow(o.optDouble("resourceScore", n.score), "resourceScore");
            n.estimatedBytes = Math.max(0, o.optLong("estimatedBytes", 0));
            n.revision = Math.max(1, o.optInt("revision", 1));
            n.libraryName = o.optString("libraryName", "");
            n.savedAt = Math.max(0, o.optLong("savedAt", 0));
            return n;
        }
    }

    public static final class ForwardResult {
        public final double[] input;
        public final double[] hidden;
        public final double[] output;
        public final double[][] hiddenToOutputContribution;

        ForwardResult(double[] input, double[] hidden, double[] output, double[][] contributions) {
            this.input = input;
            this.hidden = hidden;
            this.output = output;
            this.hiddenToOutputContribution = contributions;
        }
    }

    public interface ProgressListener {
        void onProgress(int epoch, int requestedEpochs, double trainingMse, double validationMse);
    }

    public interface CancelCheck {
        boolean isCancelled();
    }

    public static final class TrainingResult {
        public final int epochsRun;
        public final int requestedEpochs;
        public final double initialValidationMse;
        public final double finalValidationMse;
        public final double finalTrainingMse;
        public final long elapsedMs;
        public final boolean cancelled;
        public final boolean earlyStopped;
        public final String report;

        TrainingResult(int epochsRun, int requestedEpochs, double initialValidationMse,
                       double finalValidationMse, double finalTrainingMse, long elapsedMs,
                       boolean cancelled, boolean earlyStopped, String report) {
            this.epochsRun = epochsRun;
            this.requestedEpochs = requestedEpochs;
            this.initialValidationMse = initialValidationMse;
            this.finalValidationMse = finalValidationMse;
            this.finalTrainingMse = finalTrainingMse;
            this.elapsedMs = elapsedMs;
            this.cancelled = cancelled;
            this.earlyStopped = earlyStopped;
            this.report = report;
        }
    }

    private static final class AdamState {
        final double[][] mIn, vIn, mOut, vOut;
        final double[] mHiddenBias, vHiddenBias, mOutputBias, vOutputBias;

        AdamState(NeuronWorkspace w) {
            int h = w.neurons.size();
            mIn = new double[h][w.inputCount];
            vIn = new double[h][w.inputCount];
            mOut = new double[h][w.outputCount];
            vOut = new double[h][w.outputCount];
            mHiddenBias = new double[h];
            vHiddenBias = new double[h];
            mOutputBias = new double[w.outputCount];
            vOutputBias = new double[w.outputCount];
        }
    }

    public static final class ImportInfo {
        public final int inputs;
        public final int outputs;
        public final int samples;
        ImportInfo(int inputs, int outputs, int samples) {
            this.inputs = inputs;
            this.outputs = outputs;
            this.samples = samples;
        }
    }

    private static final class Dataset {
        final ArrayList<Sample> samples = new ArrayList<>();
        int inputs;
        int outputs;
        String[] inputConcepts;
        String[] outputConcepts;
    }

    public int inputCount;
    public int outputCount;
    public double learningRate = 0.01;
    public int preferredEpochs = 120;
    public String taskName = "示例回归任务";
    public String[] inputConcepts;
    public String[] outputConcepts;
    public long totalTrainingRuns;
    public String lastReport = "尚未训练。先运行内置示例，或导入自己的 CSV/JSON 数据。";
    public double lastValidationMse = Double.NaN;

    private final ArrayList<Neuron> neurons = new ArrayList<>();
    private final ArrayList<Sample> samples = new ArrayList<>();
    private final ArrayList<Neuron> savedNeurons = new ArrayList<>();
    private final ArrayList<JSONObject> taskHistory = new ArrayList<>();
    private final ArrayList<JSONObject> recentEvents = new ArrayList<>();
    private long idCounter = 1;
    private final Random random = new Random(20261009L);

    public NeuronWorkspace(int inputs, int hidden, int outputs) {
        validateArchitecture(inputs, hidden, outputs);
        inputCount = inputs;
        outputCount = outputs;
        outputBias = new double[outputs];
        inputConcepts = defaultConcepts("输入", inputs);
        outputConcepts = defaultConcepts("输出", outputs);
        initializeNeurons(hidden);
        createDemoSamples(48);
    }

    public static NeuronWorkspace createDefault() {
        return new NeuronWorkspace(1, 8, 1);
    }

    private static String[] defaultConcepts(String prefix, int count) {
        String[] names = new String[count];
        for (int i = 0; i < count; i++) names[i] = prefix + "_" + i;
        return names;
    }

    private static JSONArray toJsonStrings(String[] values) {
        JSONArray out = new JSONArray();
        if (values != null) for (String value : values) out.put(value == null ? "" : value);
        return out;
    }

    private static String[] readConcepts(JSONArray values, int count, String prefix) {
        String[] out = defaultConcepts(prefix, count);
        if (values == null) return out;
        if (values.length() != count) throw new IllegalArgumentException(prefix + "概念名称数量与网络维度不匹配");
        for (int i = 0; i < count; i++) {
            String value = values.optString(i, "").trim();
            if (!value.isEmpty()) out[i] = value;
        }
        return out;
    }

    public synchronized void setConceptNames(String inputNames, String outputNames) {
        String[] in = parseConceptNames(inputNames, inputCount, "输入");
        String[] out = parseConceptNames(outputNames, outputCount, "输出");
        inputConcepts = in;
        outputConcepts = out;
        recordEvent("concept_names_updated", safeObject("inputs", in.length, "outputs", out.length));
    }

    private static String[] parseConceptNames(String text, int count, String prefix) {
        if (text == null || text.trim().isEmpty()) return defaultConcepts(prefix, count);
        String[] parts = text.split("[,，]");
        if (parts.length != count)
            throw new IllegalArgumentException(prefix + "概念名称需要 " + count + " 个，使用逗号分隔");
        String[] out = new String[count];
        for (int i = 0; i < count; i++) {
            out[i] = parts[i].trim();
            if (out[i].isEmpty()) throw new IllegalArgumentException(prefix + "概念名称不能留空");
            if (out[i].length() > 48) throw new IllegalArgumentException(prefix + "概念名称不能超过 48 个字符");
        }
        return out;
    }

    public synchronized String[] inputConceptNames() { return inputConcepts.clone(); }
    public synchronized String[] outputConceptNames() { return outputConcepts.clone(); }

    public synchronized int hiddenCount() { return neurons.size(); }
    public synchronized int sampleCount() { return samples.size(); }
    public synchronized int savedNeuronCount() { return savedNeurons.size(); }
    public synchronized int taskCount() { return taskHistory.size(); }
    public synchronized Neuron neuronAt(int index) { return neurons.get(index); }
    public synchronized Neuron savedNeuronAt(int index) { return savedNeurons.get(index); }
    public synchronized List<Neuron> neuronsSnapshot() {
        ArrayList<Neuron> out = new ArrayList<>();
        for (Neuron n : neurons) out.add(n.copy());
        return out;
    }
    public synchronized List<Neuron> savedNeuronsSnapshot() {
        ArrayList<Neuron> out = new ArrayList<>();
        for (Neuron n : savedNeurons) out.add(n.copy());
        return out;
    }
    public synchronized List<Sample> samplesSnapshot() {
        ArrayList<Sample> out = new ArrayList<>();
        for (Sample s : samples) out.add(new Sample(s.input, s.output));
        return out;
    }
    public synchronized List<JSONObject> taskHistorySnapshot() {
        ArrayList<JSONObject> out = new ArrayList<>();
        for (JSONObject o : taskHistory) {
            try { out.add(new JSONObject(o.toString())); }
            catch (JSONException ignored) { }
        }
        return out;
    }
    public synchronized List<JSONObject> eventsSnapshot() {
        ArrayList<JSONObject> out = new ArrayList<>();
        for (JSONObject o : recentEvents) {
            try { out.add(new JSONObject(o.toString())); }
            catch (JSONException ignored) { }
        }
        return out;
    }

    public synchronized void createDemoSamples(int count) {
        int size = Math.max(2, Math.min(MAX_SAMPLES, count));
        samples.clear();
        for (int row = 0; row < size; row++) {
            double[] x = new double[inputCount];
            double[] y = new double[outputCount];
            if (inputCount == 1 && outputCount == 1) {
                x[0] = -1.0 + 2.0 * row / (size - 1.0);
                y[0] = 2.0 * x[0] + 1.0;
            } else {
                for (int i = 0; i < inputCount; i++)
                    x[i] = Math.sin((row + 1) * (i + 1) * 0.31) * 0.9;
                for (int o = 0; o < outputCount; o++) {
                    double v = (o + 1) * 0.2;
                    for (int i = 0; i < inputCount; i++)
                        v += ((o + 1) * 0.35 / (i + 1.0)) * x[i];
                    y[o] = v;
                }
            }
            samples.add(new Sample(x, y));
        }
        recordEvent("demo_dataset_created", safeObject("samples", size, "inputs", inputCount,
                "outputs", outputCount));
    }

    private void initializeNeurons(int count) {
        neurons.clear();
        for (int i = 0; i < count; i++) {
            Neuron n = new Neuron(nextId(), inputCount, outputCount);
            double inScale = Math.sqrt(6.0 / (inputCount + count));
            double outScale = Math.sqrt(6.0 / (count + outputCount));
            for (int j = 0; j < inputCount; j++)
                n.inputWeights[j] = (random.nextDouble() * 2 - 1) * inScale;
            n.bias = (random.nextDouble() * 2 - 1) * 0.08;
            for (int j = 0; j < outputCount; j++)
                n.outputWeights[j] = (random.nextDouble() * 2 - 1) * outScale;
            neurons.add(n);
        }
    }

    private String nextId() {
        return String.format(Locale.US, "H-%03d", idCounter++);
    }

    private static void validateArchitecture(int inputs, int hidden, int outputs) {
        if (inputs < 1 || inputs > MAX_INPUTS)
            throw new IllegalArgumentException("输入数量必须是 1 到 " + MAX_INPUTS);
        if (outputs < 1 || outputs > MAX_OUTPUTS)
            throw new IllegalArgumentException("输出数量必须是 1 到 " + MAX_OUTPUTS);
        if (hidden < 1 || hidden > MAX_NEURONS)
            throw new IllegalArgumentException("隐藏神经元数量必须是 1 到 " + MAX_NEURONS);
    }

    public synchronized void reconfigure(int inputs, int hidden, int outputs, boolean addDemoData) {
        validateArchitecture(inputs, hidden, outputs);
        boolean dimensionsChanged = inputs != inputCount || outputs != outputCount;
        // Reconfiguration explicitly resets trainable output bias along with hidden weights.
        outputBias = new double[outputs];
        inputCount = inputs;
        outputCount = outputs;
        if (inputConcepts == null || inputConcepts.length != inputs) inputConcepts = defaultConcepts("输入", inputs);
        if (outputConcepts == null || outputConcepts.length != outputs) outputConcepts = defaultConcepts("输出", outputs);
        initializeNeurons(hidden);
        if (dimensionsChanged || addDemoData) {
            samples.clear();
            if (addDemoData) createDemoSamples(48);
        }
        recordEvent("network_reconfigured", safeObject("inputs", inputs, "hidden", hidden,
                "outputs", outputs, "dataset_cleared", dimensionsChanged || addDemoData));
        lastReport = "网络结构已重建。旧权重已重置；确认数据集维度后再训练。";
    }

    public synchronized void addNeuron() {
        if (getEstimatedBytes() + estimatedNeuronBytes(inputCount, outputCount) > MAX_WORKSPACE_BUDGET_BYTES)
            throw new IllegalArgumentException("工作区已达到安全资源预算，先剪枝、减少数据或保存后导出");
        if (neurons.size() >= MAX_NEURONS) throw new IllegalArgumentException("当前版本最多支持 " + MAX_NEURONS + " 个隐藏神经元");
        Neuron n = new Neuron(nextId(), inputCount, outputCount);
        double inScale = Math.sqrt(6.0 / (inputCount + neurons.size() + 1));
        double outScale = Math.sqrt(6.0 / (neurons.size() + 1 + outputCount));
        for (int i = 0; i < inputCount; i++) n.inputWeights[i] = (random.nextDouble() * 2 - 1) * inScale;
        n.bias = (random.nextDouble() * 2 - 1) * 0.05;
        for (int o = 0; o < outputCount; o++) n.outputWeights[o] = (random.nextDouble() * 2 - 1) * outScale;
        neurons.add(n);
        recordEvent("neuron_added", safeObject("id", n.id, "hidden_count", neurons.size()));
    }

    public synchronized void removeNeuron(String id) {
        if (neurons.size() <= 1) throw new IllegalArgumentException("至少保留一个隐藏神经元");
        for (int i = 0; i < neurons.size(); i++) {
            if (neurons.get(i).id.equals(id)) {
                neurons.remove(i);
                recordEvent("neuron_removed", safeObject("id", id, "hidden_count", neurons.size()));
                return;
            }
        }
    }

    public synchronized void setNeuronEnabled(String id, boolean enabled) {
        for (Neuron n : neurons) if (n.id.equals(id)) {
            if (!enabled && n.enabled) {
                int active = 0;
                for (Neuron candidate : neurons) if (candidate.enabled) active++;
                if (active <= 1) throw new IllegalArgumentException("至少保留一个活动神经元");
            }
            n.enabled = enabled;
            n.revision++;
            recordEvent("neuron_state_changed", safeObject("id", id, "enabled", enabled));
            return;
        }
    }

    public synchronized void saveNeuronsToLibrary(List<String> ids) {
        if (ids == null || ids.isEmpty()) throw new IllegalArgumentException("请先勾选要保存的神经元");
        long now = System.currentTimeMillis();
        int added = 0;
        for (Neuron n : neurons) {
            if (!ids.contains(n.id)) continue;
            Neuron copy = n.copy();
            copy.libraryName = "保存 " + copy.id + " · " + (now + added);
            copy.savedAt = now + added;
            savedNeurons.add(copy);
            added++;
        }
        while (savedNeurons.size() > MAX_SAVED_NEURONS) savedNeurons.remove(0);
        if (added == 0) throw new IllegalArgumentException("所选神经元没有匹配到当前网络");
        recordEvent("neurons_saved_to_library", safeObject("count", added, "library_size", savedNeurons.size()));
    }

    public synchronized void saveSingleNeuronToLibrary(String id) {
        saveNeuronsToLibrary(Collections.singletonList(id));
    }

    public synchronized void appendSavedNeuron(int libraryIndex) {
        if (libraryIndex < 0 || libraryIndex >= savedNeurons.size())
            throw new IllegalArgumentException("保存库索引无效");
        if (neurons.size() >= MAX_NEURONS) throw new IllegalArgumentException("网络已达到神经元数量上限");
        Neuron copy = savedNeurons.get(libraryIndex).copy();
        if (copy.inputWeights.length != inputCount || copy.outputWeights.length != outputCount)
            throw new IllegalArgumentException("这个神经元的输入/输出维度与当前网络不匹配");
        copy.id = nextId();
        copy.libraryName = "";
        copy.savedAt = 0;
        copy.revision++;
        neurons.add(copy);
        recordEvent("saved_neuron_loaded", safeObject("id", copy.id, "library_index", libraryIndex));
    }

    public synchronized void deleteSavedNeuron(int index) {
        if (index >= 0 && index < savedNeurons.size()) {
            String id = savedNeurons.get(index).id;
            savedNeurons.remove(index);
            recordEvent("saved_neuron_deleted", safeObject("id", id));
        }
    }

    public synchronized ForwardResult predict(double[] input) {
        if (input == null || input.length != inputCount)
            throw new IllegalArgumentException("需要 " + inputCount + " 个输入值");
        for (double x : input) if (!Double.isFinite(x))
            throw new IllegalArgumentException("输入包含 NaN 或无穷大");
        return forward(input, -1);
    }

    private ForwardResult forward(double[] input, int skippedNeuron) {
        double[] hidden = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            Neuron n = neurons.get(i);
            if (!n.enabled || i == skippedNeuron) continue;
            double z = n.bias;
            for (int j = 0; j < inputCount; j++) z += n.inputWeights[j] * input[j];
            hidden[i] = Math.tanh(z);
        }
        double[] output = outputBias == null ? new double[outputCount] : outputBias.clone();
        double[][] contribution = new double[neurons.size()][outputCount];
        for (int i = 0; i < neurons.size(); i++) {
            if (hidden[i] == 0.0) continue;
            Neuron n = neurons.get(i);
            for (int o = 0; o < outputCount; o++) {
                contribution[i][o] = hidden[i] * n.outputWeights[o];
                output[o] += contribution[i][o];
            }
        }
        return new ForwardResult(input.clone(), hidden, output, contribution);
    }

    public synchronized double calculateMse() {
        if (samples.isEmpty()) return Double.NaN;
        return mse(samples, -1);
    }

    private double mse(List<Sample> data, int skippedNeuron) {
        if (data.isEmpty()) return Double.NaN;
        // Reuse the calibrated hybrid hidden-layer path for full-network train/validation
        // evaluation, but never initiate GPU calibration from a metric call. Calibration
        // belongs to the training epoch so tiny validation subsets cannot select a backend.
        double[][] hiddenBatch = skippedNeuron < 0 && trainingHybridDecisionMade && useHybridTraining
                && data.size() >= 64 ? buildEpochHiddenActivations(data) : null;
        double sum = 0;
        long count = 0;
        for (int i = 0; i < data.size(); i++) {
            Sample sample = data.get(i);
            double[] out = hiddenBatch == null
                    ? predictOutputOnly(sample.input, skippedNeuron)
                    : predictOutputOnlyFromHidden(hiddenBatch[i]);
            for (int o = 0; o < outputCount; o++) {
                double d = out[o] - sample.output[o];
                sum += d * d;
                count++;
            }
        }
        return count == 0 ? Double.NaN : sum / count;
    }

    private double[] predictOutputOnlyFromHidden(double[] hidden) {
        double[] output = outputBias.clone();
        for (int i = 0; i < neurons.size(); i++) {
            double activation = hidden[i];
            if (activation == 0.0) continue;
            double[] weights = neurons.get(i).outputWeights;
            for (int o = 0; o < outputCount; o++) output[o] += activation * weights[o];
        }
        return output;
    }

    /** Minimal-allocation inference path for loss/evaluation loops. */
    private double[] predictOutputOnly(double[] input, int skippedNeuron) {
        double[] output = outputBias.clone();
        for (int i = 0; i < neurons.size(); i++) {
            if (i == skippedNeuron) continue;
            Neuron n = neurons.get(i);
            if (!n.enabled) continue;
            double z = n.bias;
            for (int j = 0; j < inputCount; j++) z += n.inputWeights[j] * input[j];
            double activation = Math.tanh(z);
            if (activation == 0.0) continue;
            for (int o = 0; o < outputCount; o++) output[o] += activation * n.outputWeights[o];
        }
        return output;
    }

    public synchronized void scoreNeurons() {
        if (samples.isEmpty()) return;
        ArrayList<Sample> validation = selectScoreSamples();
        double baseline = mse(validation, -1);
        for (int i = 0; i < neurons.size(); i++) {
            Neuron n = neurons.get(i);
            double ablated = mse(validation, i);
            n.score = ablated - baseline;
            n.estimatedBytes = estimatedNeuronBytes(n);
            double memoryKiB = n.estimatedBytes / 1024.0;
            double computeKiloOps = validation.size()
                    * (double) (inputCount + outputCount + 2) / 1000.0;
            // Keep raw utility separate from a small, explicit resource penalty.
            n.resourceScore = n.score - 0.001 * memoryKiB - 0.00002 * computeKiloOps;
        }
        recordEvent("neuron_scores_updated", safeObject("count", neurons.size(),
                "baseline_validation_mse", baseline,
                "estimated_workspace_bytes", getEstimatedBytes(),
                "workspace_budget_bytes", MAX_WORKSPACE_BUDGET_BYTES));
    }

    private ArrayList<Sample> selectScoreSamples() {
        ArrayList<Sample> out = new ArrayList<>();
        if (samples.size() < 5) {
            out.addAll(samples);
            return out;
        }
        // Keep scoring separate from most training rows to reduce trivial memorization.
        for (int i = 0; i < samples.size(); i++) if (i % 5 == 0) out.add(samples.get(i));
        if (out.size() > 128) {
            ArrayList<Sample> bounded = new ArrayList<>();
            for (int i = 0; i < 128; i++) {
                int index = (int) (((long) i * (out.size() - 1)) / 127);
                bounded.add(out.get(index));
            }
            return bounded;
        }
        return out;
    }

    public synchronized long getEstimatedBytes() {
        long total = 4096L;
        for (Neuron n : neurons) total = saturatingAdd(total, estimatedNeuronBytes(n));
        for (Neuron n : savedNeurons) total = saturatingAdd(total, estimatedNeuronBytes(n));
        for (Sample sample : samples) {
            total = saturatingAdd(total, 64L + 24L
                    + 8L * (sample.input.length + sample.output.length));
        }
        for (JSONObject event : recentEvents) total = saturatingAdd(total, 128L + event.toString().length() * 2L);
        for (JSONObject task : taskHistory) total = saturatingAdd(total, 128L + task.toString().length() * 2L);
        // Conservative multiplier for references, ArrayList capacity and JVM object headers.
        return saturatingAdd(total, total / 2L);
    }

    private static long estimatedNeuronBytes(Neuron n) {
        if (n == null) return 0L;
        return estimatedNeuronBytes(n.inputWeights.length, n.outputWeights.length);
    }

    private static long estimatedNeuronBytes(int inputs, int outputs) {
        // Approximate Java arrays + object state + IDs; not an RSS substitute.
        return 192L + 24L + 8L * inputs + 24L + 8L * outputs;
    }

    private static long saturatingAdd(long a, long b) {
        if (b > Long.MAX_VALUE - a) return Long.MAX_VALUE;
        return a + b;
    }

    public synchronized String evolveOneGeneration() {
        if (samples.size() < 5) throw new IllegalArgumentException("安全进化至少需要 5 条样本，先导入更多数据");
        scoreNeurons();
        ArrayList<Sample> validation = selectScoreSamples();
        double baseline = mse(validation, -1);
        if (!Double.isFinite(baseline)) throw new IllegalStateException("验证集误差无效，拒绝进化");

        int elite = -1;
        for (int i = 0; i < neurons.size(); i++)
            if (neurons.get(i).enabled && (elite < 0
                    || neurons.get(i).resourceScore > neurons.get(elite).resourceScore)) elite = i;
        if (elite < 0) throw new IllegalStateException("至少需要一个活动神经元");

        // Remove at most one redundant non-elite unit, and only after counterfactual
        // validation shows that pruning does not materially damage the task.
        int prune = -1;
        double bestPrunedMse = baseline;
        if (neurons.size() > 2) {
            for (int i = 0; i < neurons.size(); i++) {
                if (i == elite) continue;
                Neuron candidate = neurons.get(i);
                if (candidate.resourceScore > 0.0) continue;
                boolean wasEnabled = candidate.enabled;
                candidate.enabled = false;
                double candidateMse = mse(validation, -1);
                candidate.enabled = wasEnabled;
                if (Double.isFinite(candidateMse) && candidateMse <= baseline * 1.005 + 1e-9
                        && (prune < 0 || candidateMse < bestPrunedMse)) {
                    prune = i;
                    bestPrunedMse = candidateMse;
                }
            }
        }
        StringBuilder report = new StringBuilder();
        if (prune >= 0) {
            String removed = neurons.remove(prune).id;
            recordEvent("resource_prune", safeObject("id", removed, "before_mse", baseline,
                    "predicted_pruned_mse", bestPrunedMse, "reason", "non_elite_counterfactual_within_0.5_percent"));
            report.append("剪除冗余候选 ").append(removed).append("；验证误差变化在 0.5% 安全容差内。\n");
        } else {
            report.append("本代未发现可安全剪除的低贡献单元。\n");
        }

        if (neurons.size() >= MAX_NEURONS
                || getEstimatedBytes() + estimatedNeuronBytes(inputCount, outputCount) > MAX_WORKSPACE_BUDGET_BYTES) {
            scoreNeurons();
            report.append("资源预算或数量上限阻止复制；未分配超预算内存。");
            recordEvent("evolution_blocked", safeObject("reason", "workspace_budget_or_neuron_cap",
                    "hidden", neurons.size(), "estimated_bytes", getEstimatedBytes()));
            return report.toString();
        }

        scoreNeurons();
        int parentIndex = -1;
        for (int i = 0; i < neurons.size(); i++)
            if (neurons.get(i).enabled && (parentIndex < 0
                    || neurons.get(i).resourceScore > neurons.get(parentIndex).resourceScore)) parentIndex = i;
        if (parentIndex < 0) throw new IllegalStateException("没有可复制的活动神经元");
        Neuron parent = neurons.get(parentIndex);
        double[] originalOutputWeights = parent.outputWeights.clone();
        Neuron child = parent.copy();
        child.id = nextId();
        child.revision = 1;
        child.libraryName = "";
        child.savedAt = 0;
        // Split the parent's output contribution before mutation to avoid immediately
        // doubling its influence just because a copy was created.
        for (int o = 0; o < outputCount; o++) {
            parent.outputWeights[o] = originalOutputWeights[o] * 0.5;
            child.outputWeights[o] = originalOutputWeights[o] * 0.5;
        }
        for (int i = 0; i < inputCount; i++) child.inputWeights[i] += (random.nextDouble() * 2.0 - 1.0) * 0.01;
        child.bias += (random.nextDouble() * 2.0 - 1.0) * 0.01;
        for (int o = 0; o < outputCount; o++) child.outputWeights[o] += (random.nextDouble() * 2.0 - 1.0) * 0.005;
        neurons.add(child);
        double childMse = mse(validation, -1);
        if (Double.isFinite(childMse) && childMse <= baseline * 1.005 + 1e-9) {
            child.revision++;
            recordEvent("evolution_clone_accepted", safeObject("parent", parent.id, "child", child.id,
                    "before_mse", baseline, "after_mse", childMse, "estimated_bytes", getEstimatedBytes()));
            report.append("复制 ").append(parent.id).append(" → ").append(child.id)
                    .append("，添加小变异；验证 MSE ").append(format(baseline)).append(" → ")
                    .append(format(childMse)).append("，未超过 0.5% 容差。");
        } else {
            neurons.remove(neurons.size() - 1);
            parent.outputWeights = originalOutputWeights;
            recordEvent("evolution_clone_rejected", safeObject("parent", parent.id, "child", child.id,
                    "before_mse", baseline, "candidate_mse", childMse, "reason", "validation_regression"));
            report.append("候选后代导致验证误差明显变差，已回滚变异与父代权重。");
        }
        scoreNeurons();
        lastValidationMse = mse(validation, -1);
        lastReport = "资源竞争与进化\n" + report + "\n原始分数衡量消融贡献，资源分数额外扣除估算存储和计算成本。";
        return report.toString();
    }

    public synchronized TrainingResult train(int requestedEpochs, ProgressListener listener,
                                              CancelCheck cancel) {
        if (requestedEpochs < 1 || requestedEpochs > MAX_EPOCHS)
            throw new IllegalArgumentException("训练轮数必须是 1 到 " + MAX_EPOCHS);
        if (samples.size() < 2) throw new IllegalArgumentException("至少需要两条训练样本");
        double estimatedPerEpoch = (double) samples.size() * neurons.size()
                * (inputCount + outputCount + 2);
        int safeLimit = (int) Math.max(1, Math.min(MAX_EPOCHS, 80000000.0 / Math.max(1.0, estimatedPerEpoch)));
        if (requestedEpochs > safeLimit)
            throw new IllegalArgumentException("为保护手机运行内存与响应，当前网络/数据估算最多运行 "
                    + safeLimit + " 轮。请减少数据、神经元数量或训练轮数。");

        long start = System.currentTimeMillis();
        ArrayList<Sample> training = new ArrayList<>();
        ArrayList<Sample> validation = new ArrayList<>();
        if (samples.size() >= 5) {
            for (int i = 0; i < samples.size(); i++) {
                if (i % 5 == 0) validation.add(samples.get(i));
                else training.add(samples.get(i));
            }
        } else {
            training.addAll(samples.subList(0, samples.size() - 1));
            validation.add(samples.get(samples.size() - 1));
        }
        if (training.isEmpty() || validation.isEmpty()) throw new IllegalArgumentException("无法建立训练/验证拆分");

        double initialValidation = mse(validation, -1);
        double bestValidation = initialValidation;
        double bestTraining = mse(training, -1);
        double[][] bestIn = copyInputWeights();
        double[] bestHiddenBias = hiddenBiases();
        double[][] bestOut = copyOutputWeights();
        double[] bestOutputBias = outputBias.clone();
        AdamState adam = new AdamState(this);
        int epochsRun = 0, patience = 0;
        boolean cancelled = false, earlyStopped = false;
        int patienceLimit = Math.max(40, Math.min(120, requestedEpochs / 4));
        int updateEvery = Math.max(1, requestedEpochs / 40);

        for (int epoch = 1; epoch <= requestedEpochs; epoch++) {
            if (cancel != null && cancel.isCancelled()) { cancelled = true; break; }
            applyAdamEpoch(training, adam, epoch);
            epochsRun = epoch;
            double trainMse = mse(training, -1);
            double valMse = mse(validation, -1);
            if (Double.isFinite(valMse) && valMse < bestValidation - 1e-10) {
                bestValidation = valMse;
                bestTraining = trainMse;
                bestIn = copyInputWeights();
                bestHiddenBias = hiddenBiases();
                bestOut = copyOutputWeights();
                bestOutputBias = outputBias.clone();
                patience = 0;
            } else {
                patience++;
            }
            if (listener != null && (epoch == 1 || epoch % updateEvery == 0 || epoch == requestedEpochs))
                listener.onProgress(epoch, requestedEpochs, trainMse, valMse);
            if (patience >= patienceLimit && epoch >= Math.min(50, requestedEpochs)) {
                earlyStopped = epoch < requestedEpochs;
                break;
            }
        }

        restoreWeights(bestIn, bestHiddenBias, bestOut, bestOutputBias);
        for (Neuron n : neurons) n.revision++;
        scoreNeurons();
        double finalValidation = mse(validation, -1);
        double finalTraining = mse(training, -1);
        long elapsed = Math.max(0, System.currentTimeMillis() - start);
        totalTrainingRuns++;
        lastValidationMse = finalValidation;
        String report = "任务：" + taskName + "\n"
                + "结构：输入 " + inputCount + " → 隐藏 " + neurons.size() + " → 输出 " + outputCount + "\n"
                + "样本：" + samples.size() + "（训练 " + training.size() + "，验证 " + validation.size() + "）\n"
                + "轮数：" + epochsRun + "/" + requestedEpochs
                + (earlyStopped ? "（验证集长期未改善，提前停止）" : "")
                + (cancelled ? "（用户停止，保留当前最佳权重）" : "") + "\n"
                + "验证集 MSE：" + format(initialValidation) + " → " + format(finalValidation) + "\n"
                + "训练集 MSE：" + format(finalTraining) + "\n"
                + "耗时：" + elapsed + " ms；后端：" + lastTrainingBackend + "；激活：tanh；输出：linear；优化器：Adam\n"
                + "评分含义：score = 遮蔽该隐藏神经元后验证集 MSE 的增加量；正值越大，当前验证集越依赖它。它不是通用能力证明。";
        lastReport = report;
        try {
            JSONObject task = new JSONObject();
            task.put("name", taskName);
            task.put("createdAt", System.currentTimeMillis());
            task.put("inputs", inputCount);
            task.put("hidden", neurons.size());
            task.put("outputs", outputCount);
            task.put("samples", samples.size());
            task.put("epochsRequested", requestedEpochs);
            task.put("epochsRun", epochsRun);
            task.put("learningRate", learningRate);
            task.put("initialValidationMse", initialValidation);
            task.put("finalValidationMse", finalValidation);
            task.put("finalTrainingMse", finalTraining);
            task.put("elapsedMs", elapsed);
            task.put("backend", lastTrainingBackend);
            task.put("cancelled", cancelled);
            task.put("earlyStopped", earlyStopped);
            task.put("status", cancelled ? "cancelled" : "completed");
            taskHistory.add(task);
            while (taskHistory.size() > MAX_HISTORY) taskHistory.remove(0);
            recordEvent("training_completed", new JSONObject(task.toString()));
        } catch (JSONException ignored) { }
        return new TrainingResult(epochsRun, requestedEpochs, initialValidation, finalValidation,
                finalTraining, elapsed, cancelled, earlyStopped, report);
    }

    private void applyAdamEpoch(List<Sample> training, AdamState state, int step) {
        int hCount = neurons.size();
        double[][] gIn = new double[hCount][inputCount];
        double[] gHiddenBias = new double[hCount];
        double[][] gOut = new double[hCount][outputCount];
        double[] gOutputBias = new double[outputCount];

        // Compute the hidden layer once per epoch. For sufficiently large workloads,
        // CPU and GLES GPU calculate disjoint neuron ranges concurrently. Parameters
        // are not updated until the full-batch gradient has been accumulated, so this
        // is mathematically equivalent to the original full-batch forward pass.
        double[][] hiddenBatch = buildEpochHiddenActivations(training);
        for (int sampleIndex = 0; sampleIndex < training.size(); sampleIndex++) {
            Sample sample = training.get(sampleIndex);
            ForwardResult f = forwardForTraining(sample.input, hiddenBatch[sampleIndex]);
            double[] dOut = new double[outputCount];
            for (int o = 0; o < outputCount; o++) {
                dOut[o] = 2.0 * (f.output[o] - sample.output[o]) / outputCount;
                gOutputBias[o] += dOut[o];
                for (int h = 0; h < hCount; h++)
                    if (neurons.get(h).enabled) gOut[h][o] += dOut[o] * f.hidden[h];
            }
            for (int h = 0; h < hCount; h++) {
                Neuron n = neurons.get(h);
                if (!n.enabled) continue;
                double upstream = 0;
                for (int o = 0; o < outputCount; o++) upstream += dOut[o] * n.outputWeights[o];
                double delta = upstream * (1.0 - f.hidden[h] * f.hidden[h]);
                gHiddenBias[h] += delta;
                for (int i = 0; i < inputCount; i++) gIn[h][i] += delta * sample.input[i];
            }
        }

        double divisor = 1.0 / training.size();
        double correction1 = 1.0 - Math.pow(ADAM_BETA1, step);
        double correction2 = 1.0 - Math.pow(ADAM_BETA2, step);
        learningRate = Math.max(0.00001, Math.min(0.1, learningRate));
        for (int h = 0; h < hCount; h++) {
            Neuron n = neurons.get(h);
            if (!n.enabled) continue;
            for (int i = 0; i < inputCount; i++) {
                double g = clamp(gIn[h][i] * divisor, -5, 5);
                state.mIn[h][i] = ADAM_BETA1 * state.mIn[h][i] + (1 - ADAM_BETA1) * g;
                state.vIn[h][i] = ADAM_BETA2 * state.vIn[h][i] + (1 - ADAM_BETA2) * g * g;
                n.inputWeights[i] -= learningRate * (state.mIn[h][i] / correction1)
                        / (Math.sqrt(state.vIn[h][i] / correction2) + 1e-8);
            }
            double gb = clamp(gHiddenBias[h] * divisor, -5, 5);
            state.mHiddenBias[h] = ADAM_BETA1 * state.mHiddenBias[h] + (1 - ADAM_BETA1) * gb;
            state.vHiddenBias[h] = ADAM_BETA2 * state.vHiddenBias[h] + (1 - ADAM_BETA2) * gb * gb;
            n.bias -= learningRate * (state.mHiddenBias[h] / correction1)
                    / (Math.sqrt(state.vHiddenBias[h] / correction2) + 1e-8);
            for (int o = 0; o < outputCount; o++) {
                double g = clamp(gOut[h][o] * divisor, -5, 5);
                state.mOut[h][o] = ADAM_BETA1 * state.mOut[h][o] + (1 - ADAM_BETA1) * g;
                state.vOut[h][o] = ADAM_BETA2 * state.vOut[h][o] + (1 - ADAM_BETA2) * g * g;
                n.outputWeights[o] -= learningRate * (state.mOut[h][o] / correction1)
                        / (Math.sqrt(state.vOut[h][o] / correction2) + 1e-8);
            }
        }
        for (int o = 0; o < outputCount; o++) {
            double g = clamp(gOutputBias[o] * divisor, -5, 5);
            state.mOutputBias[o] = ADAM_BETA1 * state.mOutputBias[o] + (1 - ADAM_BETA1) * g;
            state.vOutputBias[o] = ADAM_BETA2 * state.vOutputBias[o] + (1 - ADAM_BETA2) * g * g;
            outputBias[o] -= learningRate * (state.mOutputBias[o] / correction1)
                    / (Math.sqrt(state.vOutputBias[o] / correction2) + 1e-8);
        }
    }

    private static final java.util.concurrent.ExecutorService TRAINING_GPU_EXECUTOR =
            java.util.concurrent.Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "aimeng-training-gpu");
                t.setDaemon(true);
                return t;
            });
    private boolean trainingHybridDecisionMade;
    private boolean useHybridTraining;
    private boolean trainingHybridValidated;
    private int trainingHybridInputShape = -1;
    private int trainingHybridHiddenShape = -1;
    private boolean trainingGpuWarmed;
    private double[][] epochHiddenScratch;
    private int epochHiddenRows = -1;
    private int epochHiddenWidth = -1;
    private String lastTrainingBackend = "CPU";

    private double[][] buildEpochHiddenActivations(List<Sample> data) {
        final int rows = data.size();
        final int hiddenCount = neurons.size();
        final int dims = inputCount;
        if (epochHiddenScratch == null || epochHiddenRows != rows || epochHiddenWidth != hiddenCount) {
            epochHiddenScratch = new double[rows][hiddenCount];
            epochHiddenRows = rows;
            epochHiddenWidth = hiddenCount;
        }
        double[][] hidden = epochHiddenScratch;
        long bytes = ((long) rows * dims + (long) dims * (hiddenCount / 2)
                + (long) rows * (hiddenCount / 2)) * Float.BYTES;
        boolean eligible = rows >= 64 && hiddenCount >= 32 && dims >= 4
                && bytes <= 32L * 1024L * 1024L;
        if (!eligible) {
            computeCpuHiddenRange(data, hidden, 0, hiddenCount);
            lastTrainingBackend = "CPU";
            return hidden;
        }

        if (trainingHybridInputShape != dims || trainingHybridHiddenShape != hiddenCount) {
            trainingHybridInputShape = dims;
            trainingHybridHiddenShape = hiddenCount;
            trainingHybridDecisionMade = false;
            trainingHybridValidated = false;
            useHybridTraining = false;
        }
        if (trainingHybridDecisionMade && !useHybridTraining) {
            computeCpuHiddenRange(data, hidden, 0, hiddenCount);
            lastTrainingBackend = "CPU (GPU calibration rejected)";
            return hidden;
        }

        int cpuEnd = Math.max(1, hiddenCount / 2);
        int gpuCount = hiddenCount - cpuEnd;
        float[] flatInputs = new float[rows * dims];
        for (int r = 0; r < rows; r++) {
            double[] in = data.get(r).input;
            for (int j = 0; j < dims; j++) flatInputs[r * dims + j] = (float) in[j];
        }
        float[] transposed = new float[dims * gpuCount];
        for (int j = 0; j < dims; j++) {
            for (int h = cpuEnd; h < hiddenCount; h++) {
                Neuron neuron = neurons.get(h);
                transposed[j * gpuCount + h - cpuEnd] =
                        neuron.enabled ? (float) neuron.inputWeights[j] : 0f;
            }
        }

        if (!trainingGpuWarmed) {
            float[] warm = GpuComputeRuntime.matMul(flatInputs, transposed, rows, dims, gpuCount);
            if (warm == null || warm.length != rows * gpuCount) {
                trainingHybridDecisionMade = true;
                useHybridTraining = false;
                computeCpuHiddenRange(data, hidden, 0, hiddenCount);
                lastTrainingBackend = "CPU (GPU unavailable: " + GpuComputeRuntime.getLastError() + ")";
                return hidden;
            }
            trainingGpuWarmed = true;
        }

        double[][] cpuReference = null;
        double cpuOnlyMs = 0.0;
        if (!trainingHybridDecisionMade) {
            long cpuStart = System.nanoTime();
            cpuReference = new double[rows][hiddenCount];
            computeCpuHiddenRange(data, cpuReference, 0, hiddenCount);
            cpuOnlyMs = (System.nanoTime() - cpuStart) / 1_000_000.0;
        }

        final double[][] target = hidden;
        java.util.concurrent.Future<float[]> gpuFuture = TRAINING_GPU_EXECUTOR.submit(
                () -> GpuComputeRuntime.matMul(flatInputs, transposed, rows, dims, gpuCount));
        long hybridStart = System.nanoTime();
        computeCpuHiddenRange(data, target, 0, cpuEnd);
        float[] gpuRaw;
        try {
            gpuRaw = gpuFuture.get();
        } catch (Exception error) {
            gpuFuture.cancel(true);
            trainingHybridDecisionMade = true;
            useHybridTraining = false;
            computeCpuHiddenRange(data, hidden, 0, hiddenCount);
            lastTrainingBackend = "CPU (GPU task failed)";
            return hidden;
        }
        if (gpuRaw == null || gpuRaw.length != rows * gpuCount) {
            trainingHybridDecisionMade = true;
            useHybridTraining = false;
            computeCpuHiddenRange(data, hidden, 0, hiddenCount);
            lastTrainingBackend = "CPU (GPU output invalid)";
            return hidden;
        }
        for (int r = 0; r < rows; r++) {
            for (int k = 0; k < gpuCount; k++) {
                Neuron neuron = neurons.get(cpuEnd + k);
                hidden[r][cpuEnd + k] = neuron.enabled
                        ? Math.tanh((double) gpuRaw[r * gpuCount + k] + neuron.bias) : 0.0;
            }
        }

        double hybridMs = (System.nanoTime() - hybridStart) / 1_000_000.0;
        if (cpuReference != null) {
            double maxDiff = maxHiddenDifference(cpuReference, hidden);
            trainingHybridDecisionMade = true;
            trainingHybridValidated = Double.isFinite(maxDiff) && maxDiff <= 0.001;
            useHybridTraining = trainingHybridValidated && hybridMs < cpuOnlyMs * 0.90;
            if (!useHybridTraining) {
                hidden = cpuReference;
                lastTrainingBackend = "CPU (hybrid rejected; cpu_ms=" + cpuOnlyMs
                        + ", hybrid_ms=" + String.format(Locale.US, "%.3f", hybridMs)
                        + ", max_abs=" + maxDiff + ")";
                return hidden;
            }
        }
        if (!trainingHybridValidated && trainingHybridDecisionMade && !useHybridTraining) {
            computeCpuHiddenRange(data, hidden, 0, hiddenCount);
            lastTrainingBackend = "CPU (hybrid validation failed)";
            return hidden;
        }
        lastTrainingBackend = "CPU+GLES31_GPU (disjoint hidden-neuron ranges)";
        return hidden;
    }

    private void computeCpuHiddenRange(List<Sample> data, double[][] target, int startHidden, int endHidden) {
        for (int r = 0; r < data.size(); r++) {
            double[] input = data.get(r).input;
            for (int h = startHidden; h < endHidden; h++) {
                Neuron neuron = neurons.get(h);
                if (!neuron.enabled) { target[r][h] = 0.0; continue; }
                double sum = neuron.bias;
                for (int j = 0; j < inputCount; j++) sum += neuron.inputWeights[j] * input[j];
                target[r][h] = Math.tanh(sum);
            }
        }
    }

    private static double maxHiddenDifference(double[][] expected, double[][] actual) {
        double max = 0.0;
        for (int r = 0; r < expected.length; r++) {
            for (int h = 0; h < expected[r].length; h++) {
                double diff = Math.abs(expected[r][h] - actual[r][h]);
                if (!Double.isFinite(diff)) return Double.POSITIVE_INFINITY;
                if (diff > max) max = diff;
            }
        }
        return max;
    }

    private double[] outputBias;

    /** Allocation-light forward pass for backpropagation; no contribution matrix or input clone. */
    private ForwardResult forwardForTraining(double[] input) {
        double[] hidden = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            Neuron n = neurons.get(i);
            if (!n.enabled) continue;
            double z = n.bias;
            double[] weights = n.inputWeights;
            for (int j = 0; j < inputCount; j++) z += weights[j] * input[j];
            hidden[i] = Math.tanh(z);
        }
        return forwardForTraining(input, hidden);
    }

    private ForwardResult forwardForTraining(double[] input, double[] hidden) {
        double[] output = outputBias.clone();
        for (int i = 0; i < neurons.size(); i++) {
            double activation = hidden[i];
            if (activation == 0.0) continue;
            double[] weights = neurons.get(i).outputWeights;
            for (int o = 0; o < outputCount; o++) output[o] += activation * weights[o];
        }
        return new ForwardResult(input, hidden, output, null);
    }

    private double[][] copyInputWeights() {
        double[][] out = new double[neurons.size()][];
        for (int i = 0; i < neurons.size(); i++) out[i] = neurons.get(i).inputWeights.clone();
        return out;
    }

    private double[][] copyOutputWeights() {
        double[][] out = new double[neurons.size()][];
        for (int i = 0; i < neurons.size(); i++) out[i] = neurons.get(i).outputWeights.clone();
        return out;
    }

    private double[] hiddenBiases() {
        double[] out = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) out[i] = neurons.get(i).bias;
        return out;
    }

    private void restoreWeights(double[][] in, double[] hiddenBias, double[][] out, double[] outBias) {
        for (int i = 0; i < neurons.size(); i++) {
            neurons.get(i).inputWeights = in[i].clone();
            neurons.get(i).outputWeights = out[i].clone();
            neurons.get(i).bias = hiddenBias[i];
        }
        outputBias = outBias.clone();
    }

    private static double clamp(double value, double min, double max) {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    public synchronized String summary() {
        int active = 0;
        for (Neuron n : neurons) if (n.enabled) active++;
        return "任务：" + taskName + "\n"
                + "网络：输入 " + inputCount + " → 隐藏 " + neurons.size() + "（活动 " + active
                + "，禁用 " + (neurons.size() - active) + "）→ 输出 " + outputCount + "\n"
                + "数据：" + samples.size() + " 条；本地已保存神经元：" + savedNeurons.size()
                + "；训练任务记录：" + taskHistory.size() + "\n"
                + "估算工作区：" + formatMiB(getEstimatedBytes()) + " / " + formatMiB(MAX_WORKSPACE_BUDGET_BYTES) + " MiB\n"
                + "学习率：" + format(learningRate) + "；运行次数：" + totalTrainingRuns + "\n"
                + "当前验证 MSE：" + (Double.isFinite(lastValidationMse) ? format(lastValidationMse) : "尚未训练");
    }

    public synchronized JSONObject toJson() throws JSONException {
        JSONObject root = new JSONObject();
        root.put("format", FORMAT);
        root.put("schemaVersion", 1);
        root.put("createdAt", System.currentTimeMillis());
        root.put("inputs", inputCount);
        root.put("outputs", outputCount);
        root.put("inputConcepts", toJsonStrings(inputConcepts));
        root.put("outputConcepts", toJsonStrings(outputConcepts));
        root.put("learningRate", learningRate);
        root.put("preferredEpochs", preferredEpochs);
        root.put("taskName", taskName);
        root.put("totalTrainingRuns", totalTrainingRuns);
        root.put("lastReport", lastReport);
        if (Double.isFinite(lastValidationMse)) root.put("lastValidationMse", lastValidationMse);
        root.put("outputBias", toJsonArray(outputBias == null ? new double[outputCount] : outputBias));
        JSONArray ns = new JSONArray();
        for (Neuron n : neurons) ns.put(n.toJson());
        root.put("neurons", ns);
        JSONArray data = new JSONArray();
        for (Sample s : samples) data.put(s.toJson());
        root.put("samples", data);
        JSONArray library = new JSONArray();
        for (Neuron n : savedNeurons) library.put(n.toJson());
        root.put("savedNeurons", library);
        JSONArray tasks = new JSONArray();
        for (JSONObject t : taskHistory) tasks.put(new JSONObject(t.toString()));
        root.put("taskHistory", tasks);
        JSONArray events = new JSONArray();
        int start = Math.max(0, recentEvents.size() - MAX_EVENTS);
        for (int i = start; i < recentEvents.size(); i++) events.put(new JSONObject(recentEvents.get(i).toString()));
        root.put("recentEvents", events);
        root.put("idCounter", idCounter);
        return root;
    }

    public static NeuronWorkspace fromJson(JSONObject root) throws JSONException {
        String format = root.optString("format", "");
        if (!FORMAT.equals(format)) throw new IllegalArgumentException("不支持的工作区格式：" + format);
        int inputs = root.optInt("inputs", 1);
        int outputs = root.optInt("outputs", 1);
        JSONArray ns = root.optJSONArray("neurons");
        int hidden = ns == null ? 8 : Math.max(1, Math.min(MAX_NEURONS, ns.length()));
        NeuronWorkspace w = new NeuronWorkspace(inputs, hidden, outputs);
        w.neurons.clear();
        JSONArray outputBiasJson = root.optJSONArray("outputBias");
        w.outputBias = outputBiasJson == null ? new double[outputs]
                : readArray(outputBiasJson, outputs, "outputBias");
        if (ns != null) {
            for (int i = 0; i < ns.length() && i < MAX_NEURONS; i++)
                w.neurons.add(Neuron.fromJson(ns.getJSONObject(i), inputs, outputs));
        }
        if (w.neurons.isEmpty()) w.initializeNeurons(8);
        w.samples.clear();
        JSONArray data = root.optJSONArray("samples");
        if (data != null) {
            if (data.length() > MAX_SAMPLES) throw new IllegalArgumentException("工作区样本超过上限 " + MAX_SAMPLES);
            for (int i = 0; i < data.length(); i++) {
                JSONObject s = data.getJSONObject(i);
                double[] x = readArray(s.optJSONArray("input"), inputs, "sample.input");
                double[] y = readArray(s.optJSONArray("output"), outputs, "sample.output");
                w.samples.add(new Sample(x, y));
            }
        }
        if (w.samples.isEmpty()) w.createDemoSamples(48);
        JSONArray library = root.optJSONArray("savedNeurons");
        if (library != null) {
            for (int i = 0; i < library.length() && i < MAX_SAVED_NEURONS; i++) {
                JSONObject o = library.getJSONObject(i);
                int in = o.optJSONArray("inputWeights") == null ? inputs : o.optJSONArray("inputWeights").length();
                int out = o.optJSONArray("outputWeights") == null ? outputs : o.optJSONArray("outputWeights").length();
                w.savedNeurons.add(Neuron.fromJson(o, in, out));
            }
        }
        JSONArray tasks = root.optJSONArray("taskHistory");
        if (tasks != null) {
            for (int i = Math.max(0, tasks.length() - MAX_HISTORY); i < tasks.length(); i++)
                w.taskHistory.add(new JSONObject(tasks.getJSONObject(i).toString()));
        }
        JSONArray events = root.optJSONArray("recentEvents");
        if (events != null) {
            for (int i = Math.max(0, events.length() - MAX_EVENTS); i < events.length(); i++)
                w.recentEvents.add(new JSONObject(events.getJSONObject(i).toString()));
        }
        w.inputConcepts = readConcepts(root.optJSONArray("inputConcepts"), inputs, "输入");
        w.outputConcepts = readConcepts(root.optJSONArray("outputConcepts"), outputs, "输出");
        w.learningRate = clamp(root.optDouble("learningRate", 0.01), 0.00001, 0.1);
        w.preferredEpochs = Math.max(1, Math.min(MAX_EPOCHS, root.optInt("preferredEpochs", 120)));
        w.taskName = root.optString("taskName", "导入的训练任务");
        w.totalTrainingRuns = Math.max(0, root.optLong("totalTrainingRuns", 0));
        w.lastReport = root.optString("lastReport", "已加载工作区。");
        w.lastValidationMse = root.optDouble("lastValidationMse", Double.NaN);
        w.idCounter = Math.max(w.neurons.size() + 1L, root.optLong("idCounter", w.neurons.size() + 1L));
        return w;
    }

    public synchronized String exportWorkspaceJson() throws JSONException {
        return toJson().toString(2);
    }

    public synchronized String exportDatasetJson() throws JSONException {
        JSONObject root = new JSONObject();
        root.put("format", DATA_FORMAT);
        root.put("inputs", inputCount);
        root.put("outputs", outputCount);
        root.put("inputConcepts", toJsonStrings(inputConcepts));
        root.put("outputConcepts", toJsonStrings(outputConcepts));
        JSONArray data = new JSONArray();
        for (Sample s : samples) data.put(s.toJson());
        root.put("samples", data);
        return root.toString(2);
    }

    public synchronized String exportDatasetCsv() {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < inputCount; i++) {
            if (i > 0) out.append(',');
            out.append(csv(inputConcepts[i]));
        }
        for (int i = 0; i < outputCount; i++) out.append(',').append(csv(outputConcepts[i]));
        out.append('\n');
        for (Sample s : samples) {
            for (double v : s.input) out.append(format(v)).append(',');
            for (int i = 0; i < s.output.length; i++) {
                if (i > 0) out.append(',');
                out.append(format(s.output[i]));
            }
            out.append('\n');
        }
        return out.toString();
    }

    public synchronized String exportTrainingResultsCsv() {
        StringBuilder out = new StringBuilder("name,created_at,inputs,hidden,outputs,samples,epochs_requested,epochs_run,learning_rate,initial_validation_mse,final_validation_mse,final_training_mse,elapsed_ms,backend,status\n");
        for (JSONObject t : taskHistory) {
            out.append(csv(t.optString("name", ""))).append(',')
                    .append(t.optLong("createdAt", 0)).append(',')
                    .append(t.optInt("inputs", 0)).append(',')
                    .append(t.optInt("hidden", 0)).append(',')
                    .append(t.optInt("outputs", 0)).append(',')
                    .append(t.optInt("samples", 0)).append(',')
                    .append(t.optInt("epochsRequested", 0)).append(',')
                    .append(t.optInt("epochsRun", 0)).append(',')
                    .append(t.optDouble("learningRate", 0)).append(',')
                    .append(t.optDouble("initialValidationMse", Double.NaN)).append(',')
                    .append(t.optDouble("finalValidationMse", Double.NaN)).append(',')
                    .append(t.optDouble("finalTrainingMse", Double.NaN)).append(',')
                    .append(t.optLong("elapsedMs", 0)).append(',')
                    .append(csv(t.optString("backend", ""))).append(',')
                    .append(csv(t.optString("status", ""))).append('\n');
        }
        return out.toString();
    }

    public synchronized String exportNeuronJson(String id) throws JSONException {
        for (Neuron n : neurons) if (n.id.equals(id)) {
            JSONObject root = new JSONObject();
            root.put("format", NEURON_FORMAT);
            root.put("inputs", inputCount);
            root.put("outputs", outputCount);
            root.put("neuron", n.toJson());
            return root.toString(2);
        }
        throw new IllegalArgumentException("神经元已不存在：" + id);
    }

    public synchronized String exportNeuronPackJson(List<String> ids) throws JSONException {
        if (ids == null || ids.isEmpty()) throw new IllegalArgumentException("请先勾选要导出的神经元");
        JSONObject root = new JSONObject();
        root.put("format", NEURON_PACK_FORMAT);
        root.put("inputs", inputCount);
        root.put("outputs", outputCount);
        JSONArray pack = new JSONArray();
        for (Neuron n : neurons) if (ids.contains(n.id)) pack.put(n.toJson());
        if (pack.length() == 0) throw new IllegalArgumentException("没有可导出的已选神经元");
        root.put("neurons", pack);
        return root.toString(2);
    }

    public synchronized String exportTraceJsonl() {
        StringBuilder out = new StringBuilder();
        for (JSONObject event : recentEvents) out.append(event.toString()).append('\n');
        return out.toString();
    }

    public synchronized void saveInternal(File file) throws Exception {
        if (file == null) throw new IllegalArgumentException("save file is null");
        JSONObject json = toJson();
        File parent = file.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) throw new java.io.IOException("无法创建工作区目录");
        File temp = new File(file.getParentFile(), file.getName() + ".tmp");
        try (FileOutputStream fos = new FileOutputStream(temp);
             OutputStreamWriter writer = new OutputStreamWriter(fos, StandardCharsets.UTF_8)) {
            writer.write(json.toString());
            writer.flush();
            fos.getFD().sync();
        }
        try {
            java.nio.file.Files.move(temp.toPath(), file.toPath(),
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                    java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        } catch (java.nio.file.AtomicMoveNotSupportedException unsupported) {
            java.nio.file.Files.move(temp.toPath(), file.toPath(),
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
    }

    public synchronized void appendTraceFile(File file, JSONObject event) throws Exception {
        JSONObject row = new JSONObject(event.toString());
        row.put("timestamp", System.currentTimeMillis());
        synchronized (recentEvents) {
            recentEvents.add(row);
            while (recentEvents.size() > MAX_EVENTS) recentEvents.remove(0);
        }
        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) parent.mkdirs();
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(file, true), StandardCharsets.UTF_8))) {
            writer.write(row.toString());
            writer.newLine();
        }
    }

    public synchronized String importDatasetContent(String content, String fileName, boolean allowRebuild) {
        Dataset parsed = parseDataset(content, fileName, outputCount);
        if (parsed.samples.isEmpty()) throw new IllegalArgumentException("文件中没有可用的训练样本");
        if (parsed.inputs != inputCount || parsed.outputs != outputCount) {
            if (!allowRebuild)
                throw new IllegalArgumentException("DATA_DIMENSION_MISMATCH:" + parsed.inputs + ":" + parsed.outputs + ":" + parsed.samples.size());
            reconfigure(parsed.inputs, Math.max(1, neurons.size()), parsed.outputs, false);
        }
        samples.clear();
        samples.addAll(parsed.samples);
        if (parsed.inputConcepts != null && parsed.inputConcepts.length == inputCount)
            inputConcepts = parsed.inputConcepts;
        if (parsed.outputConcepts != null && parsed.outputConcepts.length == outputCount)
            outputConcepts = parsed.outputConcepts;
        recordEvent("dataset_imported", safeObject("source", fileName, "samples", samples.size(),
                "inputs", inputCount, "outputs", outputCount));
        lastReport = "已导入训练数据：" + samples.size() + " 条，输入 " + inputCount + " 维，输出 " + outputCount + " 维。";
        return lastReport;
    }

    public synchronized void importNeuronContent(String content, boolean allowRebuild) throws JSONException {
        JSONObject root = new JSONObject(content);
        String format = root.optString("format", "");
        int in = root.optInt("inputs", inputCount);
        int out = root.optInt("outputs", outputCount);
        JSONArray pack = root.optJSONArray("neurons");
        JSONObject single = root.optJSONObject("neuron");
        ArrayList<Neuron> imported = new ArrayList<>();
        if (single != null) imported.add(Neuron.fromJson(single, in, out));
        if (pack != null) {
            for (int i = 0; i < pack.length() && imported.size() < MAX_NEURONS; i++)
                imported.add(Neuron.fromJson(pack.getJSONObject(i), in, out));
        }
        if (imported.isEmpty()) throw new IllegalArgumentException("JSON 中没有可导入的神经元");
        if (in < 1 || in > MAX_INPUTS || out < 1 || out > MAX_OUTPUTS)
            throw new IllegalArgumentException("导入神经元的输入/输出维度不在支持范围内");
        if (in != inputCount || out != outputCount) {
            if (!allowRebuild) throw new IllegalArgumentException("DATA_DIMENSION_MISMATCH:" + in + ":" + out + ":0");
            reconfigure(in, Math.max(1, neurons.size()), out, false);
        }
        for (Neuron n : imported) {
            if (neurons.size() >= MAX_NEURONS) break;
            Neuron copy = n.copy();
            copy.id = nextId();
            copy.libraryName = "";
            copy.savedAt = 0;
            neurons.add(copy);
            Neuron bankCopy = copy.copy();
            bankCopy.libraryName = "导入 " + bankCopy.id;
            bankCopy.savedAt = System.currentTimeMillis();
            savedNeurons.add(bankCopy);
        }
        while (savedNeurons.size() > MAX_SAVED_NEURONS) savedNeurons.remove(0);
        recordEvent("neurons_imported", safeObject("format", format, "count", imported.size()));
    }

    public synchronized void importTaskContent(String content) throws JSONException {
        JSONObject root = new JSONObject(content);
        if (!TASK_FORMAT.equals(root.optString("format", "")))
            throw new IllegalArgumentException("不是 AIMENG 训练任务文件");
        taskName = root.optString("name", "导入的训练任务");
        preferredEpochs = Math.max(1, Math.min(MAX_EPOCHS, root.optInt("epochs", 120)));
        learningRate = clamp(root.optDouble("learningRate", learningRate), 0.00001, 0.1);
        recordEvent("training_task_imported", safeObject("name", taskName, "epochs", preferredEpochs));
    }

    public synchronized String exportTaskJson() throws JSONException {
        JSONObject root = new JSONObject();
        root.put("format", TASK_FORMAT);
        root.put("name", taskName);
        root.put("epochs", preferredEpochs);
        root.put("learningRate", learningRate);
        root.put("inputs", inputCount);
        root.put("hidden", neurons.size());
        root.put("outputs", outputCount);
        root.put("createdAt", System.currentTimeMillis());
        root.put("datasetSamples", samples.size());
        return root.toString(2);
    }

    public static boolean isWorkspaceContent(String content) {
        try { return FORMAT.equals(new JSONObject(content).optString("format", "")); }
        catch (Throwable ignored) { return false; }
    }

    public static String jsonFormat(String content) {
        try { return new JSONObject(content).optString("format", ""); }
        catch (Throwable ignored) { return ""; }
    }

    public static ImportInfo inspectImport(String content, String fileName, int expectedOutputs) {
        Dataset d = parseDataset(content, fileName, expectedOutputs);
        return new ImportInfo(d.inputs, d.outputs, d.samples.size());
    }

    private static Dataset parseDataset(String content, String fileName, int expectedOutputs) {
        if (content == null || content.trim().isEmpty()) throw new IllegalArgumentException("文件为空");
        String trimmed = content.trim();
        if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
            try {
                JSONObject root = trimmed.startsWith("[")
                        ? new JSONObject().put("samples", new JSONArray(trimmed))
                        : new JSONObject(trimmed);
                String format = root.optString("format", "");
                if (FORMAT.equals(format)) throw new IllegalArgumentException("这是完整工作区，请选择“导入工作区”路径");
                if (NEURON_FORMAT.equals(format) || NEURON_PACK_FORMAT.equals(format))
                    throw new IllegalArgumentException("这是神经元文件，请选择“导入神经元”路径");
                JSONArray rows = root.optJSONArray("samples");
                if (rows == null) {
                    if (root.has("input") && root.has("output")) {
                        rows = new JSONArray().put(new JSONObject().put("input", root.optJSONArray("input"))
                                .put("output", root.optJSONArray("output")));
                    } else throw new IllegalArgumentException("JSON 需要包含 samples 数组，每条样本包含 input 与 output 数组");
                }
                if (rows.length() < 1) throw new IllegalArgumentException("samples 数组为空");
                if (rows.length() > MAX_SAMPLES) throw new IllegalArgumentException("样本超过上限 " + MAX_SAMPLES);
                Dataset d = new Dataset();
                d.inputs = root.optInt("inputs", -1);
                d.outputs = root.optInt("outputs", -1);
                JSONArray inputNamesJson = root.optJSONArray("inputConcepts");
                JSONArray outputNamesJson = root.optJSONArray("outputConcepts");
                if (inputNamesJson != null && d.inputs > 0 && d.inputs <= MAX_INPUTS)
                    d.inputConcepts = readConcepts(inputNamesJson, d.inputs, "输入");
                if (outputNamesJson != null && d.outputs > 0 && d.outputs <= MAX_OUTPUTS)
                    d.outputConcepts = readConcepts(outputNamesJson, d.outputs, "输出");
                for (int i = 0; i < rows.length(); i++) {
                    JSONObject row = rows.getJSONObject(i);
                    double[] x = readAnyArray(row.optJSONArray("input"), "样本 input");
                    double[] y = readAnyArray(row.optJSONArray("output"), "样本 output");
                    if (i == 0) {
                        if (d.inputs < 1) d.inputs = x.length;
                        if (d.outputs < 1) d.outputs = y.length;
                    }
                    if (x.length != d.inputs || y.length != d.outputs)
                        throw new IllegalArgumentException("JSON 样本维度不一致，位置：" + i);
                    d.samples.add(new Sample(x, y));
                }
                validateArchitecture(d.inputs, Math.max(1, Math.min(MAX_NEURONS, 8)), d.outputs);
                return d;
            } catch (JSONException e) {
                throw new IllegalArgumentException("JSON 解析失败：" + e.getMessage());
            }
        }

        Dataset d = new Dataset();
        String[] lines = content.replace("\uFEFF", "").split("\\r?\\n");
        int expectedColumns = -1;
        int inputGuess = -1;
        int outputGuess = Math.max(1, Math.min(MAX_OUTPUTS, expectedOutputs));
        boolean firstMeaningful = true;
        for (String raw : lines) {
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] fields = line.split(",", -1);
            double[] values = new double[fields.length];
            boolean numeric = true;
            for (int i = 0; i < fields.length; i++) {
                try {
                    values[i] = Double.parseDouble(fields[i].trim());
                    if (!Double.isFinite(values[i])) numeric = false;
                } catch (NumberFormatException e) { numeric = false; }
            }
            if (firstMeaningful) {
                firstMeaningful = false;
                if (!numeric) continue; // a single optional CSV header row
            } else if (!numeric) {
                throw new IllegalArgumentException("CSV 第 " + (d.samples.size() + 1) + " 行包含非数字内容");
            }
            if (!numeric) throw new IllegalArgumentException("CSV 内容不是纯数值表格");
            if (expectedColumns < 0) {
                expectedColumns = values.length;
                inputGuess = expectedColumns - outputGuess;
                if (inputGuess < 1) {
                    outputGuess = 1;
                    inputGuess = expectedColumns - 1;
                }
                if (inputGuess < 1 || expectedColumns > MAX_INPUTS + MAX_OUTPUTS)
                    throw new IllegalArgumentException("CSV 列数超出支持范围，至少需要 2 列");
                d.inputs = inputGuess;
                d.outputs = outputGuess;
            }
            if (values.length != expectedColumns)
                throw new IllegalArgumentException("CSV 每一行必须有相同列数");
            double[] x = Arrays.copyOfRange(values, 0, d.inputs);
            double[] y = Arrays.copyOfRange(values, d.inputs, values.length);
            d.samples.add(new Sample(x, y));
            if (d.samples.size() > MAX_SAMPLES) throw new IllegalArgumentException("样本超过上限 " + MAX_SAMPLES);
        }
        if (d.samples.isEmpty()) throw new IllegalArgumentException("CSV 中没有数值样本");
        if (d.outputs < 1 || d.outputs > MAX_OUTPUTS || d.inputs < 1 || d.inputs > MAX_INPUTS)
            throw new IllegalArgumentException("CSV 推断出的输入/输出维度无效。CSV 格式为 input 列在前、output 列在后；请先设置输出维度。");
        return d;
    }

    private void recordEvent(String name, JSONObject values) {
        try {
            JSONObject row = new JSONObject();
            row.put("timestamp", System.currentTimeMillis());
            row.put("event", name);
            if (values != null) {
                JSONArray names = values.names();
                if (names != null) for (int i = 0; i < names.length(); i++) {
                    String key = names.optString(i);
                    row.put(key, values.opt(key));
                }
            }
            recentEvents.add(row);
            while (recentEvents.size() > MAX_EVENTS) recentEvents.remove(0);
        } catch (JSONException ignored) { }
    }

    private static JSONObject safeObject(Object... pairs) {
        JSONObject o = new JSONObject();
        try { for (int i = 0; i + 1 < pairs.length; i += 2) o.put(String.valueOf(pairs[i]), pairs[i + 1]); }
        catch (JSONException ignored) { }
        return o;
    }

    private static JSONArray toJsonArray(double[] values) {
        JSONArray arr = new JSONArray();
        if (values != null) {
            for (double v : values) {
                try {
                    arr.put(v);
                } catch (JSONException error) {
                    throw new IllegalStateException("无法序列化数值数组", error);
                }
            }
        }
        return arr;
    }

    private static double[] readAnyArray(JSONArray arr, String label) {
        if (arr == null || arr.length() < 1 || arr.length() > MAX_INPUTS + MAX_OUTPUTS)
            throw new IllegalArgumentException(label + " 必须是非空数值数组");
        double[] values = new double[arr.length()];
        for (int i = 0; i < arr.length(); i++) values[i] = finiteOrThrow(arr.optDouble(i, Double.NaN), label);
        return values;
    }

    private static double[] readArray(JSONArray arr, int expected, String label) {
        if (arr == null || arr.length() != expected)
            throw new IllegalArgumentException(label + " 维度不匹配，期望 " + expected);
        double[] values = new double[expected];
        for (int i = 0; i < expected; i++) values[i] = finiteOrThrow(arr.optDouble(i, Double.NaN), label);
        return values;
    }

    private static double finiteOrThrow(double value, String label) {
        if (!Double.isFinite(value)) throw new IllegalArgumentException(label + " 包含 NaN 或无穷大");
        return value;
    }

    private static String format(double value) {
        return String.format(Locale.US, "%.6f", value);
    }

    private static String formatMiB(long bytes) {
        return String.format(Locale.US, "%.2f", bytes / (1024.0 * 1024.0));
    }

    private static String csv(String value) {
        if (value == null) return "";
        return "\"" + value.replace("\"", "\"\"").replace("\n", " ").replace("\r", " ") + "\"";
    }
}