package bslsjdk.ornithnpu;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.Arrays;
import java.util.Random;

/**
 * Compact language-learning-oriented super-neuron prototype.
 *
 * Each instance accepts multiple scalar feature ports, fuses them through a
 * low-rank bottleneck with learned per-port gates, carries a small recurrent
 * state, and emits logits for multiple output classes (e.g. next-token IDs).
 *
 * This is an experimental CPU reference, not a production language model.
 * The input features are expected to be supplied by a tokenizer/embedding layer.
 * Training uses supervised softmax cross-entropy with truncated one-step BPTT.
 */
public final class SuperNeuronV2 {
    public static final String FORMAT = "aimeng-super-neuron-v2/v1";
    private static final double MAX_ABS_PARAMETER = 4.0;

    private final int inputCount;
    private final int hiddenSize;
    private final int outputCount;

    // Compact per-port projection into a shared low-rank representation.
    private final double[] inputProjection; // [inputCount * hiddenSize]
    private final double[] gateWeight;
    private final double[] gateBias;
    // Diagonal recurrent connection: O(hiddenSize), not O(hiddenSize^2).
    private final double[] recurrentScale;
    private final int outputRank;
    private final double[] hiddenToOutputRank; // [hiddenSize * outputRank]
    private final double[] outputEmbedding; // [outputCount * outputRank]
    private final double[] lastLatent;
    private final double[] latentGradient;
    private final double[] hiddenGradient;
    private final double[] gateGradientScratch;
    private final double[] lastInputGradient;
    private final double[] outputBias;
    private final double[] state;
    // Per-hidden-unit eligibility trace for delayed scalar reward; no sequence tape.
    private final double[] eligibilityTrace;

    // One-step cache used by trainClass(); deliberately bounded.
    private final double[] lastInput;
    private final double[] lastGate;
    private final double[] lastHidden;
    private final double[] lastPreviousState;
    private double[] lastProbabilities;
    private boolean hasForward;
    private long forwardSteps;
    private long trainSteps;

    public SuperNeuronV2(int inputCount, int hiddenSize, int outputCount, long seed) {
        if (inputCount < 1 || inputCount > 256)
            throw new IllegalArgumentException("inputCount must be 1..256");
        if (hiddenSize < 2 || hiddenSize > 256)
            throw new IllegalArgumentException("hiddenSize must be 2..256");
        if (outputCount < 2 || outputCount > 65536)
            throw new IllegalArgumentException("outputCount must be 2..65536");
        this.inputCount = inputCount;
        this.hiddenSize = hiddenSize;
        this.outputCount = outputCount;
        inputProjection = new double[inputCount * hiddenSize];
        gateWeight = new double[inputCount];
        gateBias = new double[inputCount];
        recurrentScale = new double[hiddenSize];
        outputRank = Math.min(32, hiddenSize);
        hiddenToOutputRank = new double[hiddenSize * outputRank];
        outputEmbedding = new double[outputCount * outputRank];
        lastLatent = new double[outputRank];
        latentGradient = new double[outputRank];
        hiddenGradient = new double[hiddenSize];
        gateGradientScratch = new double[inputCount];
        lastInputGradient = new double[inputCount];
        outputBias = new double[outputCount];
        state = new double[hiddenSize];
        eligibilityTrace = new double[hiddenSize];
        lastInput = new double[inputCount];
        lastGate = new double[inputCount];
        lastHidden = new double[hiddenSize];
        lastPreviousState = new double[hiddenSize];
        lastProbabilities = new double[outputCount];

        Random random = new Random(seed);
        double inputScale = 1.0 / Math.sqrt(inputCount);
        for (int i = 0; i < inputProjection.length; i++)
            inputProjection[i] = (random.nextDouble() * 2.0 - 1.0) * inputScale;
        double outputScale = 1.0 / Math.sqrt(hiddenSize);
        for (int i = 0; i < hiddenToOutputRank.length; i++)
            hiddenToOutputRank[i] = (random.nextDouble() * 2.0 - 1.0) * outputScale;
        double embeddingScale = 1.0 / Math.sqrt(outputRank);
        for (int i = 0; i < outputEmbedding.length; i++)
            outputEmbedding[i] = (random.nextDouble() * 2.0 - 1.0) * embeddingScale;
        for (int h = 0; h < hiddenSize; h++)
            recurrentScale[h] = (random.nextDouble() * 2.0 - 1.0) * 0.1;
        Arrays.fill(gateWeight, 1.0);
    }

    /** Run one step and return a defensive copy of next-token/class probabilities. */
    public double[] forward(double[] inputs) {
        return forwardInto(inputs, new double[outputCount]);
    }

    /** Allocation-conscious API for token loops: caller owns the output buffer. */
    public double[] forwardInto(double[] inputs, double[] outputBuffer) {
        validateInputs(inputs);
        if (outputBuffer == null || outputBuffer.length != outputCount)
            throw new IllegalArgumentException("output buffer length must equal outputCount");
        System.arraycopy(inputs, 0, lastInput, 0, inputCount);
        System.arraycopy(state, 0, lastPreviousState, 0, hiddenSize);

        for (int p = 0; p < inputCount; p++) {
            double z = clamp(gateBias[p] + gateWeight[p] * inputs[p], -12.0, 12.0);
            lastGate[p] = sigmoid(z);
        }

        for (int h = 0; h < hiddenSize; h++) {
            double sum = recurrentScale[h] * state[h];
            for (int p = 0; p < inputCount; p++)
                sum += inputs[p] * lastGate[p] * inputProjection[p * hiddenSize + h];
            lastHidden[h] = Math.tanh(clamp(sum, -12.0, 12.0));
        }
        System.arraycopy(lastHidden, 0, state, 0, hiddenSize);
        for (int h = 0; h < hiddenSize; h++)
            eligibilityTrace[h] = clamp(0.9 * eligibilityTrace[h] + lastHidden[h], -10.0, 10.0);

        // Factorized output head: hidden -> compact rank -> token/class scores.
        // This reduces output-head parameters from O(vocabulary * hidden) to
        // O(hidden * rank + vocabulary * rank), with rank <= 32.
        for (int r = 0; r < outputRank; r++) {
            double sum = 0.0;
            for (int h = 0; h < hiddenSize; h++)
                sum += lastHidden[h] * hiddenToOutputRank[h * outputRank + r];
            lastLatent[r] = sum;
        }
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (int o = 0; o < outputCount; o++) {
            double sum = outputBias[o];
            int base = o * outputRank;
            for (int r = 0; r < outputRank; r++)
                sum += outputEmbedding[base + r] * lastLatent[r];
            lastProbabilities[o] = sum;
            if (sum > maxLogit) maxLogit = sum;
        }
        double total = 0.0;
        for (int o = 0; o < outputCount; o++) {
            double p = Math.exp(clamp(lastProbabilities[o] - maxLogit, -80.0, 0.0));
            lastProbabilities[o] = p;
            total += p;
        }
        if (!(total > 0.0) || !Double.isFinite(total)) {
            Arrays.fill(lastProbabilities, 1.0 / outputCount);
        } else {
            for (int o = 0; o < outputCount; o++) lastProbabilities[o] /= total;
        }
        hasForward = true;
        forwardSteps++;
        System.arraycopy(lastProbabilities, 0, outputBuffer, 0, outputCount);
        return outputBuffer;
    }

    /**
     * Apply one supervised next-token/class target to the most recent forward pass.
     * Returns cross-entropy loss before the update. Gradients are clipped by
     * parameter bounds; this prototype intentionally keeps no unbounded history.
     */
    public double trainClass(int targetClass, double learningRate) {
        if (!hasForward) throw new IllegalStateException("forward must run before trainClass");
        if (targetClass < 0 || targetClass >= outputCount)
            throw new IllegalArgumentException("targetClass out of range");
        if (!Double.isFinite(learningRate) || learningRate <= 0.0 || learningRate > 0.1)
            throw new IllegalArgumentException("learningRate must be in (0, 0.1]");
        double loss = -Math.log(Math.max(1.0e-12, lastProbabilities[targetClass]));
        Arrays.fill(latentGradient, 0.0);
        Arrays.fill(hiddenGradient, 0.0);

        // First accumulate dL/d(latent) with the old output embeddings, then update
        // embeddings. Reusing scratch arrays avoids per-step gradient allocations.
        for (int o = 0; o < outputCount; o++) {
            double grad = lastProbabilities[o] - (o == targetClass ? 1.0 : 0.0);
            int base = o * outputRank;
            for (int r = 0; r < outputRank; r++) {
                latentGradient[r] += grad * outputEmbedding[base + r];
                outputEmbedding[base + r] = clamp(
                        outputEmbedding[base + r] - learningRate * grad * lastLatent[r],
                        -MAX_ABS_PARAMETER, MAX_ABS_PARAMETER);
            }
            outputBias[o] = clamp(outputBias[o] - learningRate * grad,
                    -MAX_ABS_PARAMETER, MAX_ABS_PARAMETER);
        }

        for (int h = 0; h < hiddenSize; h++) {
            for (int r = 0; r < outputRank; r++) {
                int index = h * outputRank + r;
                hiddenGradient[h] += latentGradient[r] * hiddenToOutputRank[index];
                hiddenToOutputRank[index] = clamp(hiddenToOutputRank[index]
                        - learningRate * latentGradient[r] * lastHidden[h],
                        -MAX_ABS_PARAMETER, MAX_ABS_PARAMETER);
            }
        }

        // Compute gate gradients before changing inputProjection, so this step
        // uses the same parameter snapshot as the forward pass.
        for (int p = 0; p < inputCount; p++) {
            double dGate = 0.0;
            double dDirectInput = 0.0;
            for (int h = 0; h < hiddenSize; h++) {
                double preGradient = hiddenGradient[h] * (1.0 - lastHidden[h] * lastHidden[h]);
                double projection = inputProjection[p * hiddenSize + h];
                dGate += preGradient * projection * lastInput[p];
                dDirectInput += preGradient * projection * lastGate[p];
            }
            gateGradientScratch[p] = dGate * lastGate[p] * (1.0 - lastGate[p]);
            // Cache dL/dinput before any parameters are updated. A trainable token
            // embedding table can use this gradient without retaining a sequence tape.
            lastInputGradient[p] = clamp(
                    dDirectInput + gateGradientScratch[p] * gateWeight[p], -10.0, 10.0);
        }

        for (int h = 0; h < hiddenSize; h++) {
            double preGradient = hiddenGradient[h] * (1.0 - lastHidden[h] * lastHidden[h]);
            recurrentScale[h] = clamp(recurrentScale[h]
                    - learningRate * preGradient * lastPreviousState[h], -1.0, 1.0);
            for (int p = 0; p < inputCount; p++) {
                int index = p * hiddenSize + h;
                double gatedInput = lastInput[p] * lastGate[p];
                inputProjection[index] = clamp(inputProjection[index]
                        - learningRate * preGradient * gatedInput,
                        -MAX_ABS_PARAMETER, MAX_ABS_PARAMETER);
            }
        }

        for (int p = 0; p < inputCount; p++) {
            double gateGradient = gateGradientScratch[p];
            gateWeight[p] = clamp(gateWeight[p] - learningRate * gateGradient * lastInput[p],
                    -MAX_ABS_PARAMETER, MAX_ABS_PARAMETER);
            gateBias[p] = clamp(gateBias[p] - learningRate * gateGradient,
                    -MAX_ABS_PARAMETER, MAX_ABS_PARAMETER);
        }
        trainSteps++;
        hasForward = false;
        return loss;
    }

    /** Reset only recurrent working state; learned parameters remain unchanged. */
    public void resetState() {
        Arrays.fill(eligibilityTrace, 0.0);
        Arrays.fill(state, 0.0);
        Arrays.fill(lastPreviousState, 0.0);
        Arrays.fill(lastHidden, 0.0);
        Arrays.fill(lastInput, 0.0);
        Arrays.fill(lastGate, 0.0);
        Arrays.fill(lastProbabilities, 1.0 / outputCount);
        hasForward = false;
    }

    /**
     * Apply delayed scalar reward through local eligibility traces. This update is
     * confined to this SuperNeuronV2 instance and does not traverse a neuron pool.
     */
    public void applyReward(double reward, double learningRate) {
        if (!Double.isFinite(reward) || reward < -1.0 || reward > 1.0)
            throw new IllegalArgumentException("reward must be finite and in [-1, 1]");
        if (!Double.isFinite(learningRate) || learningRate <= 0.0 || learningRate > 0.02)
            throw new IllegalArgumentException("learningRate must be in (0, 0.02]");
        for (int h = 0; h < hiddenSize; h++) {
            recurrentScale[h] = clamp(recurrentScale[h]
                    + learningRate * reward * eligibilityTrace[h], -1.0, 1.0);
            double trace = learningRate * reward * eligibilityTrace[h];
            for (int p = 0; p < inputCount; p++) {
                int index = p * hiddenSize + h;
                inputProjection[index] = clamp(inputProjection[index]
                        + trace * lastInput[p], -MAX_ABS_PARAMETER, MAX_ABS_PARAMETER);
            }
        }
    }

    /** Copy learned genes only; recurrent state and eligibility traces are reset. */
    public void copyLearnedParametersFrom(SuperNeuronV2 source) {
        requireSameShape(source);
        System.arraycopy(source.inputProjection, 0, inputProjection, 0, inputProjection.length);
        System.arraycopy(source.gateWeight, 0, gateWeight, 0, gateWeight.length);
        System.arraycopy(source.gateBias, 0, gateBias, 0, gateBias.length);
        System.arraycopy(source.recurrentScale, 0, recurrentScale, 0, recurrentScale.length);
        System.arraycopy(source.hiddenToOutputRank, 0, hiddenToOutputRank, 0, hiddenToOutputRank.length);
        System.arraycopy(source.outputEmbedding, 0, outputEmbedding, 0, outputEmbedding.length);
        System.arraycopy(source.outputBias, 0, outputBias, 0, outputBias.length);
        resetState();
    }

    /** Uniform gene crossover between two compatible individuals. */
    public void crossoverWith(SuperNeuronV2 other, Random random) {
        requireSameShape(other);
        if (random == null) throw new IllegalArgumentException("random cannot be null");
        crossoverArray(inputProjection, other.inputProjection, random);
        crossoverArray(gateWeight, other.gateWeight, random);
        crossoverArray(gateBias, other.gateBias, random);
        crossoverArray(recurrentScale, other.recurrentScale, random);
        crossoverArray(hiddenToOutputRank, other.hiddenToOutputRank, random);
        crossoverArray(outputEmbedding, other.outputEmbedding, random);
        crossoverArray(outputBias, other.outputBias, random);
        resetState();
    }

    /** Bounded Gaussian mutation used by the population-level evolutionary layer. */
    public void mutateParameters(Random random, double mutationRate, double sigma) {
        if (random == null) throw new IllegalArgumentException("random cannot be null");
        if (!Double.isFinite(mutationRate) || mutationRate < 0.0 || mutationRate > 1.0)
            throw new IllegalArgumentException("mutationRate must be in [0, 1]");
        if (!Double.isFinite(sigma) || sigma < 0.0 || sigma > 1.0)
            throw new IllegalArgumentException("sigma must be in [0, 1]");
        mutateArray(inputProjection, random, mutationRate, sigma, -MAX_ABS_PARAMETER, MAX_ABS_PARAMETER);
        mutateArray(gateWeight, random, mutationRate, sigma, -MAX_ABS_PARAMETER, MAX_ABS_PARAMETER);
        mutateArray(gateBias, random, mutationRate, sigma, -MAX_ABS_PARAMETER, MAX_ABS_PARAMETER);
        mutateArray(recurrentScale, random, mutationRate, sigma, -1.0, 1.0);
        mutateArray(hiddenToOutputRank, random, mutationRate, sigma, -MAX_ABS_PARAMETER, MAX_ABS_PARAMETER);
        mutateArray(outputEmbedding, random, mutationRate, sigma, -MAX_ABS_PARAMETER, MAX_ABS_PARAMETER);
        mutateArray(outputBias, random, mutationRate, sigma, -MAX_ABS_PARAMETER, MAX_ABS_PARAMETER);
        resetState();
    }

    private void requireSameShape(SuperNeuronV2 other) {
        if (other == null || inputCount != other.inputCount || hiddenSize != other.hiddenSize
                || outputCount != other.outputCount || outputRank != other.outputRank)
            throw new IllegalArgumentException("individual shape mismatch");
    }

    private static void crossoverArray(double[] target, double[] other, Random random) {
        for (int i = 0; i < target.length; i++) if (random.nextBoolean()) target[i] = other[i];
    }

    private static void mutateArray(double[] values, Random random, double rate, double sigma,
                                    double min, double max) {
        for (int i = 0; i < values.length; i++) {
            if (random.nextDouble() < rate)
                values[i] = clamp(values[i] + random.nextGaussian() * sigma, min, max);
        }
    }

    public int getInputCount() { return inputCount; }
    public int getHiddenSize() { return hiddenSize; }
    public int getOutputCount() { return outputCount; }
    /** Copies dL/dinput from the immediately preceding trainClass() call. */
    public void copyLastInputGradient(double[] destination) {
        if (destination == null || destination.length != inputCount)
            throw new IllegalArgumentException("gradient destination length must equal inputCount");
        System.arraycopy(lastInputGradient, 0, destination, 0, inputCount);
    }

    public long getForwardSteps() { return forwardSteps; }
    public long getTrainSteps() { return trainSteps; }
    public int getOutputRank() { return outputRank; }

    /** Number of learned scalar parameters, excluding runtime state and caches. */
    public long getParameterCount() {
        return (long) inputProjection.length + gateWeight.length + gateBias.length
                + recurrentScale.length + hiddenToOutputRank.length
                + outputEmbedding.length + outputBias.length;
    }

    /** Approximate parameter + recurrent-state bytes, excluding JVM object overhead. */
    public long estimatedStorageBytes() {
        long doubles = (long) inputProjection.length + gateWeight.length + gateBias.length
                + recurrentScale.length + hiddenToOutputRank.length + outputEmbedding.length + outputBias.length
                + state.length + lastInput.length + lastGate.length
                + lastHidden.length + lastPreviousState.length + lastProbabilities.length
                + lastLatent.length + latentGradient.length + hiddenGradient.length
                + gateGradientScratch.length + lastInputGradient.length + eligibilityTrace.length;
        return doubles * Double.BYTES;
    }

    public JSONObject toJson() throws JSONException {
        JSONObject root = new JSONObject();
        root.put("format", FORMAT);
        root.put("inputCount", inputCount);
        root.put("hiddenSize", hiddenSize);
        root.put("outputCount", outputCount);
        root.put("forwardSteps", forwardSteps);
        root.put("trainSteps", trainSteps);
        putArray(root, "inputProjection", inputProjection);
        putArray(root, "gateWeight", gateWeight);
        putArray(root, "gateBias", gateBias);
        putArray(root, "recurrentScale", recurrentScale);
        root.put("outputRank", outputRank);
        putArray(root, "hiddenToOutputRank", hiddenToOutputRank);
        putArray(root, "outputEmbedding", outputEmbedding);
        putArray(root, "outputBias", outputBias);
        putArray(root, "state", state);
        putArray(root, "eligibilityTrace", eligibilityTrace);
        return root;
    }

    public static SuperNeuronV2 fromJson(JSONObject root, long seed) throws JSONException {
        if (root == null || !FORMAT.equals(root.optString("format", "")))
            throw new IllegalArgumentException("unsupported SuperNeuronV2 checkpoint format");
        int inputs = root.getInt("inputCount");
        int hidden = root.getInt("hiddenSize");
        int outputs = root.getInt("outputCount");
        SuperNeuronV2 result = new SuperNeuronV2(inputs, hidden, outputs, seed);
        readArray(root, "inputProjection", result.inputProjection);
        readArray(root, "gateWeight", result.gateWeight);
        readArray(root, "gateBias", result.gateBias);
        readArray(root, "recurrentScale", result.recurrentScale);
        if (root.optInt("outputRank", result.outputRank) != result.outputRank)
            throw new IllegalArgumentException("checkpoint outputRank mismatch");
        readArray(root, "hiddenToOutputRank", result.hiddenToOutputRank);
        readArray(root, "outputEmbedding", result.outputEmbedding);
        readArray(root, "outputBias", result.outputBias);
        readArray(root, "state", result.state);
        if (root.has("eligibilityTrace")) readArray(root, "eligibilityTrace", result.eligibilityTrace);
        result.forwardSteps = Math.max(0L, root.optLong("forwardSteps", 0L));
        result.trainSteps = Math.max(0L, root.optLong("trainSteps", 0L));
        result.resetTransientCache();
        return result;
    }

    private void resetTransientCache() {
        Arrays.fill(lastProbabilities, 1.0 / outputCount);
        hasForward = false;
    }

    private void validateInputs(double[] inputs) {
        if (inputs == null || inputs.length != inputCount)
            throw new IllegalArgumentException("input length must equal inputCount");
        for (double value : inputs) if (!Double.isFinite(value))
            throw new IllegalArgumentException("input contains non-finite value");
    }

    private static double sigmoid(double value) {
        if (value >= 0) {
            double e = Math.exp(-value);
            return 1.0 / (1.0 + e);
        }
        double e = Math.exp(value);
        return e / (1.0 + e);
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void putArray(JSONObject root, String name, double[] values)
            throws JSONException {
        JSONArray array = new JSONArray();
        for (double value : values) array.put(value);
        root.put(name, array);
    }

    private static void readArray(JSONObject root, String name, double[] destination)
            throws JSONException {
        JSONArray array = root.getJSONArray(name);
        if (array.length() != destination.length)
            throw new IllegalArgumentException("checkpoint array length mismatch: " + name);
        for (int i = 0; i < destination.length; i++) {
            double value = array.getDouble(i);
            if (!Double.isFinite(value))
                throw new IllegalArgumentException("checkpoint contains non-finite value: " + name);
            destination[i] = value;
        }
    }
}
