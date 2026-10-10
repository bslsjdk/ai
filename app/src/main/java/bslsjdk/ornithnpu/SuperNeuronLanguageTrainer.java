package bslsjdk.ornithnpu;

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
    private static final long MAX_TOTAL_ESTIMATED_BYTES = 64L * 1024L * 1024L;
    private static final double MAX_ABS_EMBEDDING = 4.0;

    private final int vocabularySize;
    private final int embeddingSize;
    private final double[] embeddings;
    private final double[] inputScratch;
    private final double[] outputScratch;
    private final double[] gradientScratch;
    private final SuperNeuronV2 model;
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
        long embeddingBytes = (long) vocabularySize * embeddingSize * Double.BYTES;
        this.model = new SuperNeuronV2(embeddingSize, hiddenSize, vocabularySize, seed ^ 0x5DEECE66DL);
        long estimated = embeddingBytes + model.estimatedStorageBytes()
                + (long) (embeddingSize + vocabularySize) * Double.BYTES;
        if (estimated > MAX_TOTAL_ESTIMATED_BYTES)
            throw new IllegalArgumentException("estimated model storage exceeds 64 MiB budget");

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

    /**
     * Runs an exact number of online next-character updates over a cyclic sequence.
     * Each call is bounded in work; callers can run it in 1000-step chunks and
     * sample process memory between chunks. No sequence-wide gradient tape is kept.
     */
    public double trainSteps(int[] tokenIds, int steps, double learningRate) {
        validateSequence(tokenIds);
        if (steps < 1 || steps > 10000)
            throw new IllegalArgumentException("steps must be 1..10000 per call");
        if (!Double.isFinite(learningRate) || learningRate <= 0.0 || learningRate > 0.05)
            throw new IllegalArgumentException("learningRate must be in (0, 0.05]");
        model.resetState();
        double totalLoss = 0.0;
        int pairCount = tokenIds.length - 1;
        long startIndex = trainedTokenTargets % pairCount;
        for (int step = 0; step < steps; step++) {
            int index = (int) ((startIndex + step) % pairCount);
            if (index == 0 && step != 0) model.resetState();
            int token = tokenIds[index];
            int target = tokenIds[index + 1];
            copyEmbedding(token, inputScratch);
            model.forwardInto(inputScratch, outputScratch);
            double loss = model.trainClass(target, learningRate);
            totalLoss += loss;
            model.copyLastInputGradient(gradientScratch);
            int base = token * embeddingSize;
            for (int d = 0; d < embeddingSize; d++) {
                double grad = Math.max(-5.0, Math.min(5.0, gradientScratch[d]));
                embeddings[base + d] = clamp(
                        embeddings[base + d] - learningRate * grad,
                        -MAX_ABS_EMBEDDING, MAX_ABS_EMBEDDING);
            }
            double normalizedReward = clamp(1.0 - loss / Math.log(vocabularySize), -1.0, 1.0);
            model.applyReward(normalizedReward, learningRate * 0.05);
            trainedTokenTargets++;
        }
        return totalLoss / steps;
    }

    /**
     * One bounded distillation update for a context and a teacher character distribution.
     * The recurrent state is rebuilt from this sample's context; no sequence-wide tape is kept.
     */
    public double trainSoftTarget(int[] contextTokenIds, double[] teacherProbabilities, double learningRate) {
        if (contextTokenIds == null || contextTokenIds.length < 1 || contextTokenIds.length > 256)
            throw new IllegalArgumentException("context must contain 1..256 tokens");
        if (teacherProbabilities == null || teacherProbabilities.length != vocabularySize)
            throw new IllegalArgumentException("teacher distribution length must match vocabulary");
        validateSequenceForContext(contextTokenIds);
        double sum = 0.0;
        for (double p : teacherProbabilities) {
            if (!Double.isFinite(p) || p < 0.0 || p > 1.0)
                throw new IllegalArgumentException("teacher probabilities must be within [0,1]");
            sum += p;
        }
        if (Math.abs(sum - 1.0) > 1.0e-4)
            throw new IllegalArgumentException("teacher probabilities must sum to 1");
        if (!Double.isFinite(learningRate) || learningRate <= 0.0 || learningRate > 0.05)
            throw new IllegalArgumentException("learningRate must be in (0, 0.05]");
        model.resetState();
        for (int i = 0; i < contextTokenIds.length; i++) {
            copyEmbedding(contextTokenIds[i], inputScratch);
            model.forwardInto(inputScratch, outputScratch);
            if (i + 1 < contextTokenIds.length) {
                // Context tokens update recurrent state only; the final token receives the target gradient.
                continue;
            }
            double loss = model.trainDistribution(teacherProbabilities, learningRate);
            model.copyLastInputGradient(gradientScratch);
            int base = contextTokenIds[i] * embeddingSize;
            for (int d = 0; d < embeddingSize; d++) {
                double grad = Math.max(-5.0, Math.min(5.0, gradientScratch[d]));
                embeddings[base + d] = clamp(
                        embeddings[base + d] - learningRate * grad,
                        -MAX_ABS_EMBEDDING, MAX_ABS_EMBEDDING);
            }
            trainedTokenTargets++;
            return loss;
        }
        throw new IllegalStateException("distillation update did not run");
    }

    /** Returns next-character probabilities after rebuilding recurrent state from a context. */
    public double[] predictDistribution(int[] contextTokenIds) {
        if (contextTokenIds == null || contextTokenIds.length < 1 || contextTokenIds.length > 256)
            throw new IllegalArgumentException("context must contain 1..256 tokens");
        validateSequenceForContext(contextTokenIds);
        model.resetState();
        for (int token : contextTokenIds) {
            copyEmbedding(token, inputScratch);
            model.forwardInto(inputScratch, outputScratch);
        }
        return java.util.Arrays.copyOf(outputScratch, outputScratch.length);
    }

    private void validateSequenceForContext(int[] tokenIds) {
        for (int token : tokenIds) checkToken(token);
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

    /** Compact JSON checkpoint containing the model, embedding table and progress counter. */
    public org.json.JSONObject toJson() throws org.json.JSONException {
        org.json.JSONObject root = new org.json.JSONObject();
        root.put("format", "aimeng-character-trainer/v1");
        root.put("vocabularySize", vocabularySize);
        root.put("embeddingSize", embeddingSize);
        root.put("trainedTokenTargets", trainedTokenTargets);
        root.put("model", model.toJson());
        org.json.JSONArray values = new org.json.JSONArray();
        for (double value : embeddings) values.put(value);
        root.put("embeddings", values);
        return root;
    }

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
