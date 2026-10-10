package bslsjdk.ornithnpu;

import java.util.Arrays;

/**
 * Bounded experimental VN wave field.
 *
 * This is an isolated concept-validation engine, not a replacement for the trained
 * NeuronWorkspace model. Connections are directed, state is persistent between calls,
 * phase coupling modulates propagation, and an active-source budget limits edge work.
 */
public final class VnWaveField {
    public static final int MAX_NODES = 2048;
    public static final int MAX_EDGES = 65536;
    public static final int MAX_TICKS_PER_CALL = 128;

    public static final class Snapshot {
        public final double[] energy;
        public final double[] phase;
        public final double[] activation;
        public final int activeSources;
        public final long ticks;
        public final double meanEnergy;

        private Snapshot(double[] energy, double[] phase, double[] activation,
                         int activeSources, long ticks) {
            this.energy = energy.clone();
            this.phase = phase.clone();
            this.activation = activation.clone();
            this.activeSources = activeSources;
            this.ticks = ticks;
            double sum = 0;
            for (double value : energy) sum += value;
            this.meanEnergy = energy.length == 0 ? 0 : sum / energy.length;
        }
    }

    private final int nodeCount;
    private final int activeBudget;
    private final double damping;
    private final double coupling;
    private final double phaseRate;
    private final int[] edgeFrom;
    private final int[] edgeTo;
    private final double[] edgeWeight;
    private int edgeCount;
    private final double[] energy;
    private final double[] phase;
    private final double[] activation;
    private final double[] nextEnergy;
    private final double[] nextPhase;
    private final double[] nextActivation;
    private final double[] incoming;
    private final double[] phaseTorque;
    private final boolean[] selected;
    private long ticks;
    private int lastActiveSources;

    public VnWaveField(int nodeCount, int activeBudget, int maxEdges,
                       double damping, double coupling, double phaseRate) {
        if (nodeCount < 1 || nodeCount > MAX_NODES)
            throw new IllegalArgumentException("nodeCount must be 1.." + MAX_NODES);
        if (activeBudget < 1 || activeBudget > nodeCount)
            throw new IllegalArgumentException("activeBudget must be 1..nodeCount");
        if (maxEdges < 0 || maxEdges > MAX_EDGES)
            throw new IllegalArgumentException("maxEdges must be 0.." + MAX_EDGES);
        if (!finiteRange(damping, 0.0, 1.0)
                || !finiteRange(coupling, 0.0, 1.0)
                || !finiteRange(phaseRate, 0.0, 1.0))
            throw new IllegalArgumentException("damping, coupling and phaseRate must be finite in [0,1]");
        this.nodeCount = nodeCount;
        this.activeBudget = activeBudget;
        this.damping = damping;
        this.coupling = coupling;
        this.phaseRate = phaseRate;
        edgeFrom = new int[maxEdges];
        edgeTo = new int[maxEdges];
        edgeWeight = new double[maxEdges];
        energy = new double[nodeCount];
        phase = new double[nodeCount];
        activation = new double[nodeCount];
        nextEnergy = new double[nodeCount];
        nextPhase = new double[nodeCount];
        nextActivation = new double[nodeCount];
        incoming = new double[nodeCount];
        phaseTorque = new double[nodeCount];
        selected = new boolean[nodeCount];
    }

    public synchronized int addDirectedEdge(int from, int to, double weight) {
        checkNode(from);
        checkNode(to);
        if (from == to) throw new IllegalArgumentException("self edges are not supported");
        if (!Double.isFinite(weight) || weight < -1.0 || weight > 1.0)
            throw new IllegalArgumentException("weight must be finite in [-1,1]");
        for (int e = 0; e < edgeCount; e++) {
            if (edgeFrom[e] == from && edgeTo[e] == to)
                throw new IllegalArgumentException("duplicate directed edge");
        }
        if (edgeCount >= edgeFrom.length) throw new IllegalStateException("edge budget exhausted");
        int id = edgeCount++;
        edgeFrom[id] = from;
        edgeTo[id] = to;
        edgeWeight[id] = weight;
        return id;
    }

    /** Inject one bounded disturbance, then evolve the field for 1..128 ticks. */
    public synchronized Snapshot step(double[] seed, int tickCount) {
        if (seed == null || seed.length != nodeCount)
            throw new IllegalArgumentException("seed length must equal nodeCount");
        if (tickCount < 1 || tickCount > MAX_TICKS_PER_CALL)
            throw new IllegalArgumentException("tickCount must be 1.." + MAX_TICKS_PER_CALL);
        for (double value : seed) if (!Double.isFinite(value) || value < -1.0 || value > 1.0)
            throw new IllegalArgumentException("seed values must be finite in [-1,1]");

        for (int t = 0; t < tickCount; t++) {
            Arrays.fill(incoming, 0.0);
            Arrays.fill(phaseTorque, 0.0);
            Arrays.fill(selected, false);
            int active = selectActiveSources(t == 0 ? seed : null);
            lastActiveSources = active;

            // Only selected sources emit waves. Direction is encoded by each edge.
            for (int e = 0; e < edgeCount; e++) {
                int from = edgeFrom[e];
                if (!selected[from]) continue;
                int to = edgeTo[e];
                double delta = wrap(phase[from] - phase[to]);
                double signal = edgeWeight[e] * energy[from];
                incoming[to] += signal * Math.cos(delta);
                phaseTorque[to] += signal * Math.sin(delta);
            }

            for (int i = 0; i < nodeCount; i++) {
                double disturbance = t == 0 ? seed[i] : 0.0;
                double wave = coupling * incoming[i];
                nextEnergy[i] = clamp((1.0 - damping) * energy[i]
                        + Math.abs(disturbance) + wave, 0.0, 1.0);
                nextActivation[i] = Math.tanh(disturbance + incoming[i]);
                nextPhase[i] = wrap(phase[i] + phaseRate * phaseTorque[i]
                        + disturbance * phaseRate);
            }
            System.arraycopy(nextEnergy, 0, energy, 0, nodeCount);
            System.arraycopy(nextActivation, 0, activation, 0, nodeCount);
            System.arraycopy(nextPhase, 0, phase, 0, nodeCount);
            ticks++;
        }
        return snapshot();
    }

    /** Explicitly decay the background field without injecting a new task. */
    public synchronized Snapshot backgroundTick() {
        return step(new double[nodeCount], 1);
    }

    public synchronized Snapshot snapshot() {
        return new Snapshot(energy, phase, activation, lastActiveSources, ticks);
    }

    public synchronized int getNodeCount() { return nodeCount; }
    public synchronized int getEdgeCount() { return edgeCount; }
    public synchronized int getActiveBudget() { return activeBudget; }
    public synchronized int getLastActiveSourceCount() { return lastActiveSources; }
    public synchronized long getTicks() { return ticks; }

    /** Conservative payload estimate; excludes Java object/array headers and alignment. */
    public synchronized long estimatedPayloadBytes() {
        return 8L * (energy.length + phase.length + activation.length
                + nextEnergy.length + nextPhase.length + nextActivation.length
                + incoming.length + phaseTorque.length)
                + selected.length
                + 16L * edgeCount;
    }

    private int selectActiveSources(double[] seed) {
        int count = 0;
        // Seeds are considered first by adding their magnitude to the current energy score.
        // Stable index-order tie breaking makes experiments reproducible.
        for (int slot = 0; slot < activeBudget; slot++) {
            int best = -1;
            double bestScore = -1.0;
            for (int i = 0; i < nodeCount; i++) {
                if (selected[i]) continue;
                double score = energy[i] + (seed == null ? 0.0 : Math.abs(seed[i]));
                if (score > bestScore) {
                    bestScore = score;
                    best = i;
                }
            }
            if (best < 0) break;
            selected[best] = true;
            count++;
        }
        return count;
    }

    private void checkNode(int node) {
        if (node < 0 || node >= nodeCount) throw new IllegalArgumentException("node index out of range");
    }

    private static boolean finiteRange(double value, double min, double max) {
        return Double.isFinite(value) && value >= min && value <= max;
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private static double wrap(double value) {
        double full = Math.PI * 2.0;
        value %= full;
        if (value < 0) value += full;
        return value;
    }
}
