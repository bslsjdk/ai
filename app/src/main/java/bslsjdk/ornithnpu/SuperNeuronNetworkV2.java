package bslsjdk.ornithnpu;

import java.util.Arrays;
import java.util.Random;

/** Sparse shared-core collaboration runtime for logical super-neurons. */
public final class SuperNeuronNetworkV2 {
    public static final int DEFAULT_NEURONS = 1000;
    public static final int DEFAULT_EDGES = 16000;
    private final int n, channels, maxEdges;
    private final double[] bias, state, activation, sharedWeight, sharedBias;
    private final int[] head, to, channel, next;
    private final double[] edgeWeight, edgeEligibility;
    private final double[] neuronEligibility;
    private final double[] sharedWeightEligibility, sharedBiasEligibility;
    // Reused per-active-neuron cache avoids recalculating the same channel tanh for every edge.
    private final double[] channelOutputScratch;
    private final int[] channelOutputStamp;
    private int channelOutputGeneration = 1;
    private int edges;
    private final double[] pending, current;
    private final int[] pendingStamp, currentStamp, pendingTouched, currentTouched;
    private int pendingCount, currentCount, generation = 1;
    private final int[] top, lastActive;
    private final double[] score;
    private int activeCount;
    private long steps;
    private double recurrentScale = 0.2;

    public SuperNeuronNetworkV2(int neurons, int edgeCapacity, int outputChannels, long seed) {
        if (neurons < 1 || neurons > 10000) throw new IllegalArgumentException("neurons must be 1..10000");
        if (edgeCapacity < 0 || edgeCapacity > 200000) throw new IllegalArgumentException("invalid edge capacity");
        if (outputChannels < 1 || outputChannels > 16) throw new IllegalArgumentException("channels must be 1..16");
        n = neurons; maxEdges = edgeCapacity; channels = outputChannels;
        bias = new double[n]; state = new double[n]; activation = new double[n];
        head = new int[n]; Arrays.fill(head, -1);
        to = new int[maxEdges]; channel = new int[maxEdges]; next = new int[maxEdges];
        edgeWeight = new double[maxEdges]; edgeEligibility = new double[maxEdges];
        neuronEligibility = new double[n];
        sharedWeight = new double[channels]; sharedBias = new double[channels];
        sharedWeightEligibility = new double[channels]; sharedBiasEligibility = new double[channels];
        channelOutputScratch = new double[channels];
        channelOutputStamp = new int[channels];
        pending = new double[n]; current = new double[n];
        pendingStamp = new int[n]; currentStamp = new int[n];
        pendingTouched = new int[n]; currentTouched = new int[n];
        top = new int[n]; lastActive = new int[n]; score = new double[n];
        Random random = new Random(seed);
        for (int c = 0; c < channels; c++) sharedWeight[c] = (random.nextDouble() - 0.5) * 0.8;
        for (int i = 0; i < n; i++) bias[i] = (random.nextDouble() - 0.5) * 0.04;
    }

    public int connect(int source, int target, int outputChannel, double weight) {
        check(source); check(target);
        if (outputChannel < 0 || outputChannel >= channels) throw new IllegalArgumentException("bad channel");
        if (!Double.isFinite(weight) || Math.abs(weight) > 4.0) throw new IllegalArgumentException("bad edge weight");
        if (edges >= maxEdges) throw new IllegalStateException("edge capacity exhausted");
        int e = edges++;
        to[e] = target; channel[e] = outputChannel; edgeWeight[e] = weight;
        next[e] = head[source]; head[source] = e;
        return e;
    }

    /** Sparse synchronous step. Messages emitted now are consumed on the next step. */
    public int step(int[] inputNodes, double[] inputValues, int inputCount, int activationBudget) {
        if (inputCount < 0 || inputNodes == null || inputValues == null
                || inputCount > inputNodes.length || inputCount > inputValues.length)
            throw new IllegalArgumentException("invalid sparse inputs");
        if (activationBudget < 1 || activationBudget > n) throw new IllegalArgumentException("bad activation budget");
        advance();
        currentCount = 0;
        for (int i = 0; i < pendingCount; i++) {
            int node = pendingTouched[i];
            if (pendingStamp[node] == generation - 1) {
                addCurrent(node, pending[node]);
                pending[node] = 0.0;
            }
        }
        pendingCount = 0;
        for (int i = 0; i < inputCount; i++) {
            check(inputNodes[i]);
            if (!Double.isFinite(inputValues[i])) throw new IllegalArgumentException("non-finite input");
            addCurrent(inputNodes[i], inputValues[i]);
        }
        activeCount = chooseTop(activationBudget);
        for (int i = 0; i < activeCount; i++) {
            int node = lastActive[i];
            double a = Math.tanh(current[node] + recurrentScale * state[node] + bias[node]);
            state[node] = activation[node] = a;
            neuronEligibility[node] = 0.9 * neuronEligibility[node] + current[node] * (1.0 - a * a);
            // Cache lazily: a channel is evaluated at most once per active node,
            // and unused channels cost nothing when a node has only a few outgoing edges.
            if (++channelOutputGeneration == 0) {
                Arrays.fill(channelOutputStamp, 0);
                channelOutputGeneration = 1;
            }
            int localChannelGeneration = channelOutputGeneration;
            for (int e = head[node]; e >= 0; e = next[e]) {
                edgeEligibility[e] *= 0.9;
                edgeEligibility[e] += a * state[to[e]];
                int c = channel[e];
                if (channelOutputStamp[c] != localChannelGeneration) {
                    channelOutputScratch[c] = Math.tanh(a * sharedWeight[c] + sharedBias[c]);
                    channelOutputStamp[c] = localChannelGeneration;
                }
                double channelTanh = channelOutputScratch[c];
                double channelDerivative = 1.0 - channelTanh * channelTanh;
                sharedWeightEligibility[c] = 0.9 * sharedWeightEligibility[c]
                        + edgeWeight[e] * a * channelDerivative;
                sharedBiasEligibility[c] = 0.9 * sharedBiasEligibility[c]
                        + edgeWeight[e] * channelDerivative;
                double message = channelTanh * edgeWeight[e];
                if (message != 0.0) addPending(to[e], message);
            }
        }
        steps++;
        return activeCount;
    }

    public double getActivation(int node) { check(node); return activation[node]; }
    public double getState(int node) { check(node); return state[node]; }
    public double getChannelOutput(int node, int c) {
        check(node);
        if (c < 0 || c >= channels) throw new IllegalArgumentException("bad channel");
        return Math.tanh(activation[node] * sharedWeight[c] + sharedBias[c]);
    }

    /**
     * Experimental reward-modulated local update. This is a coordination baseline,
     * not a substitute for end-to-end language-model backpropagation.
     */
    public void applyReward(double reward, double learningRate) {
        if (!Double.isFinite(reward) || Math.abs(reward) > 1.0)
            throw new IllegalArgumentException("reward must be finite and within [-1,1]");
        if (!Double.isFinite(learningRate) || learningRate <= 0.0 || learningRate > 0.02)
            throw new IllegalArgumentException("learningRate must be in (0,0.02]");
        for (int i = 0; i < activeCount; i++) {
            int node = lastActive[i];
            bias[node] = clamp(bias[node] + learningRate * reward * neuronEligibility[node], -2.0, 2.0);
            for (int e = head[node]; e >= 0; e = next[e]) {
                edgeWeight[e] = clamp(edgeWeight[e] + learningRate * reward * edgeEligibility[e], -4.0, 4.0);
            }
        }
        // Shared output-core parameters must also learn; otherwise the supposedly
        // shared computation remains permanently random and only edges can adapt.
        for (int c = 0; c < channels; c++) {
            sharedWeight[c] = clamp(sharedWeight[c]
                    + learningRate * reward * sharedWeightEligibility[c], -2.0, 2.0);
            sharedBias[c] = clamp(sharedBias[c]
                    + learningRate * reward * sharedBiasEligibility[c], -2.0, 2.0);
        }
    }

    public int getNeuronCount() { return n; }
    public int getEdgeCount() { return edges; }
    public int getChannelCount() { return channels; }
    public int getLastActiveCount() { return activeCount; }
    public long getStepCount() { return steps; }
    public long getParameterCount() { return (long) n + edges + channels * 2L; }
    public long estimatedStorageBytes() {
        long ints = (long) head.length + to.length + channel.length + next.length
                + pendingStamp.length + currentStamp.length + pendingTouched.length
                + currentTouched.length + top.length + lastActive.length;
        long doubles = (long) bias.length + state.length + activation.length
                + sharedWeight.length + sharedBias.length + edgeWeight.length
                + pending.length + current.length + score.length
                + edgeEligibility.length + neuronEligibility.length
                + sharedWeightEligibility.length + sharedBiasEligibility.length;
        return ints * 4L + doubles * 8L;
    }
    public void resetState() {
        Arrays.fill(state, 0); Arrays.fill(activation, 0);
        Arrays.fill(neuronEligibility, 0); Arrays.fill(edgeEligibility, 0);
        Arrays.fill(sharedWeightEligibility, 0); Arrays.fill(sharedBiasEligibility, 0);
        Arrays.fill(pending, 0); Arrays.fill(current, 0);
        Arrays.fill(pendingStamp, 0); Arrays.fill(currentStamp, 0);
        pendingCount = currentCount = activeCount = 0; generation = 1;
    }

    private int chooseTop(int budget) {
        int count = 0;
        for (int i = 0; i < currentCount; i++) {
            int node = currentTouched[i]; double s = Math.abs(current[node]);
            if (s == 0) continue;
            int pos = Math.min(count, budget - 1);
            while (pos > 0 && score[pos - 1] < s) pos--;
            if (count >= budget && pos == budget - 1 && score[pos] >= s) continue;
            int end = Math.min(count, budget - 1);
            for (int j = end; j > pos; j--) { score[j] = score[j - 1]; top[j] = top[j - 1]; }
            score[pos] = s; top[pos] = node;
            if (count < budget) count++;
        }
        System.arraycopy(top, 0, lastActive, 0, count);
        return count;
    }
    private void addCurrent(int node, double value) {
        if (currentStamp[node] != generation) {
            currentStamp[node] = generation; current[node] = value;
            currentTouched[currentCount++] = node;
        } else current[node] += value;
    }
    private void addPending(int node, double value) {
        if (pendingStamp[node] != generation) {
            pendingStamp[node] = generation; pending[node] = value;
            pendingTouched[pendingCount++] = node;
        } else pending[node] += value;
    }
    private void advance() {
        if (generation >= Integer.MAX_VALUE - 2) {
            Arrays.fill(pendingStamp, 0); Arrays.fill(currentStamp, 0); generation = 1;
        }
        generation++;
    }
    private static double clamp(double v, double min, double max) { return Math.max(min, Math.min(max, v)); }
    private void check(int node) {
        if (node < 0 || node >= n) throw new IllegalArgumentException("node out of range: " + node);
    }
}
