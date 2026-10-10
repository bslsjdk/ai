package bslsjdk.ornithnpu;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.Random;

/**
 * Small next-token training baseline for SuperNeuronV2.
 *
 * The vocabulary size is the exact output dimension. Token embeddings are
 * trainable using the bounded input gradient exported by SuperNeuronV2.
 * Training is one-step/truncated: this class does not implement full BPTT.
 */
public final class SuperNeuronLanguageTrainer {
    private static final long MAX_TRAIN_STEPS_PER_CALL = 2_000_000L;
    private static final long MAX_TOTAL_ESTIMATED_BYTES = 16L * 1024L * 1024L;
    private static final double MAX_ABS_EMBEDDING = 4.0;

    private final int vocabularySize;
    private final int embeddingSize;
    private final double[] embeddings;
    private final double[] inputScratch;
    private final double[] outputScratch;
    private final double[] gradientScratch;
    private SuperNeuronV2 model;
    private long trainedTokenTargets;

    public SuperNeuronLanguageTrainer(
            int vocabularySize, int embeddingSize, int hiddenSize, long seed) {
        if (vocabularySize < 2 || vocabularySize > 65536)
            throw new IllegalArgumentException("vocabularySize must be 2..65536");
        if (embeddingSize < 4 || embeddingSize > 64)
            throw new IllegalArgumentException("embeddingSize must be 4..64");
        if (hiddenSize < 2 || hiddenSize > 256)
            throw new IllegalArgumentException("hiddenSize must be 2..256");

        this.vocabularySize = vocabularySize;
        this.embeddingSize = embeddingSize;
        // Estimate the full parameter + state + scratch footprint before allocating
        // the large vocabulary matrices. This prevents an over-budget configuration
        // from briefly allocating the model and only then rejecting it.
        long rank = Math.min(32, hiddenSize);
        long e = embeddingSize, h = hiddenSize, v = vocabularySize;
        long estimatedDoubles = v * e + e * h + 8L * e + 5L * h
                + h * rank + v * rank + 3L * v + 3L * rank;
        long estimated = estimatedDoubles * Double.BYTES;
        if (estimated > MAX_TOTAL_ESTIMATED_BYTES)
            throw new IllegalArgumentException("estimated model storage exceeds 16 MiB budget");
        this.model = new SuperNeuronV2(embeddingSize, hiddenSize, vocabularySize, seed ^ 0x5DEECE66DL);

        embeddings = new double[vocabularySize * embeddingSize];
        inputScratch = new double[embeddingSize];
        outputScratch = new double[vocabularySize];
        gradientScratch = new double[embeddingSize];
        Random random = new Random(seed);
        double scale = 1.0 / Math.sqrt(embeddingSize);
        for (int i = 0; i < embeddings.length; i++)
            embeddings[i] = (random.nextDouble() * 2.0 - 1.0) * scale;
    }

    /**
     * Trains on adjacent token pairs. Each epoch starts from a clean recurrent
     * state; weights and token embeddings persist across epochs.
     */
    public double trainSequence(int[] tokenIds, int epochs, double learningRate) {
        validateSequence(tokenIds);
        if (epochs < 1 || epochs > 10000)
            throw new IllegalArgumentException("epochs must be 1..10000");
        if (!Double.isFinite(learningRate) || learningRate <= 0.0 || learningRate > 0.05)
            throw new IllegalArgumentException("learningRate must be in (0, 0.05]");
        long steps = (long) (tokenIds.length - 1) * epochs;
        if (steps > MAX_TRAIN_STEPS_PER_CALL)
            throw new IllegalArgumentException("training call exceeds 2,000,000 token targets");

        double totalLoss = 0.0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            model.resetState();
            for (int i = 0; i < tokenIds.length - 1; i++) {
                int token = tokenIds[i];
                int target = tokenIds[i + 1];
                copyEmbedding(token, inputScratch);
                model.forwardInto(inputScratch, outputScratch);
                totalLoss += model.trainClass(target, learningRate);
                model.copyLastInputGradient(gradientScratch);
                int base = token * embeddingSize;
                for (int d = 0; d < embeddingSize; d++) {
                    double grad = Math.max(-5.0, Math.min(5.0, gradientScratch[d]));
                    embeddings[base + d] = clamp(
                            embeddings[base + d] - learningRate * grad,
                            -MAX_ABS_EMBEDDING, MAX_ABS_EMBEDDING);
                }
                trainedTokenTargets++;
            }
        }
        return totalLoss / steps;
    }

    public interface StepListener {
        /** Called after each target update. Return false to safely stop the run. */
        boolean onStep(long completedStepsThisCall, double loss);
    }

    /**
     * Run an exact number of next-character/token updates. Recurrent state and
     * eligibility traces reset at each pass through the supplied sequence.
     */
    public double trainSequenceSteps(int[] tokenIds, long targetSteps, double learningRate,
                                     StepListener listener) {
        validateSequence(tokenIds);
        if (targetSteps < 1 || targetSteps > MAX_TRAIN_STEPS_PER_CALL)
            throw new IllegalArgumentException("targetSteps must be 1..2,000,000");
        if (!Double.isFinite(learningRate) || learningRate <= 0.0 || learningRate > 0.05)
            throw new IllegalArgumentException("learningRate must be in (0, 0.05]");
        model.resetState();
        long completed = 0;
        double totalLoss = 0.0;
        int pairs = tokenIds.length - 1;
        while (completed < targetSteps) {
            int position = (int) (completed % pairs);
            if (position == 0) model.resetState();
            int token = tokenIds[position];
            int target = tokenIds[position + 1];
            copyEmbedding(token, inputScratch);
            model.forwardInto(inputScratch, outputScratch);
            int prediction = argmax(outputScratch);
            double loss = model.trainClass(target, learningRate);
            model.copyLastInputGradient(gradientScratch);
            int base = token * embeddingSize;
            for (int d = 0; d < embeddingSize; d++) {
                double grad = clamp(gradientScratch[d], -5.0, 5.0);
                embeddings[base + d] = clamp(embeddings[base + d] - learningRate * grad,
                        -MAX_ABS_EMBEDDING, MAX_ABS_EMBEDDING);
            }
            // A bounded local reward supplements the within-unit supervised update.
            // No gradient is propagated across the population or across time steps.
            model.applyReward(prediction == target ? 0.1 : -0.02,
                    Math.min(0.002, learningRate * 0.1));
            totalLoss += loss;
            completed++;
            trainedTokenTargets++;
            if (listener != null && !listener.onStep(completed, loss)) break;
        }
        return completed == 0 ? 0.0 : totalLoss / completed;
    }

    private static int argmax(double[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
        return best;
    }

    /** Full trainer checkpoint, including token embeddings and the local model. */
    public JSONObject toJson() throws JSONException {
        JSONObject root = new JSONObject();
        root.put("format", "aimeng-language-trainer/v1");
        root.put("vocabularySize", vocabularySize);
        root.put("embeddingSize", embeddingSize);
        root.put("hiddenSize", model.getHiddenSize());
        root.put("trainedTokenTargets", trainedTokenTargets);
        JSONArray values = new JSONArray();
        for (double value : embeddings) values.put(value);
        root.put("embeddings", values);
        root.put("model", model.toJson());
        return root;
    }

    public static SuperNeuronLanguageTrainer fromJson(JSONObject root, long seed) throws JSONException {
        if (root == null || !"aimeng-language-trainer/v1".equals(root.optString("format", "")))
            throw new IllegalArgumentException("unsupported trainer checkpoint format");
        int vocab = root.getInt("vocabularySize");
        int embedding = root.getInt("embeddingSize");
        int hidden = root.getInt("hiddenSize");
        SuperNeuronLanguageTrainer result = new SuperNeuronLanguageTrainer(vocab, embedding, hidden, seed);
        JSONArray values = root.getJSONArray("embeddings");
        if (values.length() != result.embeddings.length)
            throw new IllegalArgumentException("embedding checkpoint size mismatch");
        for (int i = 0; i < values.length(); i++) {
            double value = values.getDouble(i);
            if (!Double.isFinite(value) || Math.abs(value) > MAX_ABS_EMBEDDING)
                throw new IllegalArgumentException("invalid embedding checkpoint value");
            result.embeddings[i] = value;
        }
        SuperNeuronV2 restored = SuperNeuronV2.fromJson(root.getJSONObject("model"), seed ^ 0x5DEECE66DL);
        if (restored.getInputCount() != embedding || restored.getHiddenSize() != hidden
                || restored.getOutputCount() != vocab)
            throw new IllegalArgumentException("model checkpoint dimensions mismatch");
        result.model = restored;
        result.trainedTokenTargets = Math.max(0L, root.optLong("trainedTokenTargets", 0L));
        return result;
    }

    /** Produce an offspring from two elites using gene crossover plus bounded mutation. */
    public SuperNeuronLanguageTrainer reproduceWith(SuperNeuronLanguageTrainer other, long seed,
                                                      double mutationRate, double mutationSigma) {
        if (other == null || vocabularySize != other.vocabularySize
                || embeddingSize != other.embeddingSize || model.getHiddenSize() != other.model.getHiddenSize())
            throw new IllegalArgumentException("parent shape mismatch");
        if (!Double.isFinite(mutationRate) || mutationRate < 0.0 || mutationRate > 1.0
                || !Double.isFinite(mutationSigma) || mutationSigma < 0.0 || mutationSigma > 1.0)
            throw new IllegalArgumentException("invalid mutation settings");
        Random random = new Random(seed);
        SuperNeuronLanguageTrainer child = new SuperNeuronLanguageTrainer(
                vocabularySize, embeddingSize, model.getHiddenSize(), seed);
        for (int i = 0; i < embeddings.length; i++) {
            double gene = random.nextBoolean() ? embeddings[i] : other.embeddings[i];
            if (random.nextDouble() < mutationRate) gene += random.nextGaussian() * mutationSigma;
            child.embeddings[i] = clamp(gene, -MAX_ABS_EMBEDDING, MAX_ABS_EMBEDDING);
        }
        child.model.copyLearnedParametersFrom(model);
        child.model.crossoverWith(other.model, random);
        child.model.mutateParameters(random, mutationRate, mutationSigma);
        return child;
    }

    /** Mean next-token cross-entropy without updating any learned parameters. */
    public double evaluateSequence(int[] tokenIds) {
        validateSequence(tokenIds);
        model.resetState();
        double totalLoss = 0.0;
        for (int i = 0; i < tokenIds.length - 1; i++) {
            copyEmbedding(tokenIds[i], inputScratch);
            model.forwardInto(inputScratch, outputScratch);
            totalLoss -= Math.log(Math.max(1.0e-12, outputScratch[tokenIds[i + 1]]));
        }
        return totalLoss / (tokenIds.length - 1);
    }

    /** Predicts one next token and retains recurrent context for the next call. */
    public int predictNextToken(int tokenId) {
        checkToken(tokenId);
        copyEmbedding(tokenId, inputScratch);
        model.forwardInto(inputScratch, outputScratch);
        int best = 0;
        for (int i = 1; i < outputScratch.length; i++)
            if (outputScratch[i] > outputScratch[best]) best = i;
        return best;
    }

    public void resetContext() { model.resetState(); }
    public int getVocabularySize() { return vocabularySize; }
    public int getEmbeddingSize() { return embeddingSize; }
    public long getTrainedTokenTargets() { return trainedTokenTargets; }
    public long getParameterCount() {
        return model.getParameterCount() + (long) embeddings.length;
    }
    public long estimatedStorageBytes() {
        return (long) embeddings.length * Double.BYTES + model.estimatedStorageBytes()
                + (long) (inputScratch.length + outputScratch.length + gradientScratch.length)
                * Double.BYTES;
    }

    private void copyEmbedding(int tokenId, double[] destination) {
        int base = tokenId * embeddingSize;
        System.arraycopy(embeddings, base, destination, 0, embeddingSize);
    }

    private void validateSequence(int[] tokenIds) {
        if (tokenIds == null || tokenIds.length < 2)
            throw new IllegalArgumentException("sequence must contain at least two tokens");
        for (int token : tokenIds) checkToken(token);
    }

    private void checkToken(int tokenId) {
        if (tokenId < 0 || tokenId >= vocabularySize)
            throw new IllegalArgumentException("token ID out of vocabulary: " + tokenId);
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }
}
