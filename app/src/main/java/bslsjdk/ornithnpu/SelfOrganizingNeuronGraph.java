package bslsjdk.ornithnpu;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Bounded directed neuron graph used by the local neuron experiments.
 * Graph topology is mutable within explicit neuron and edge-slot budgets.
 */
public final class SelfOrganizingNeuronGraph {
    private static final class Edge {
        final int source;
        final int destination;
        double weight;
        boolean active;

        Edge(int source, int destination, double weight) {
            this.source = source;
            this.destination = destination;
            this.weight = weight;
            this.active = true;
        }
    }

    private final int maxNeurons;
    private final int maxEdgeSlots;
    private final double decay;
    private final Random random;
    private final List<Edge> edges = new ArrayList<>();
    private final Map<Long, Edge> edgeByPair = new LinkedHashMap<>();
    private final List<Integer> inputPorts = new ArrayList<>();
    private final List<Integer> outputPorts = new ArrayList<>();
    private final List<Double> activations = new ArrayList<>();
    private final List<Boolean> inputFlags = new ArrayList<>();
    private long ticks;
    private int activeEdgeCount;

    public SelfOrganizingNeuronGraph(int maxNeurons, int maxEdgeSlots, double decay, long seed) {
        if (maxNeurons < 1) throw new IllegalArgumentException("maxNeurons must be positive");
        if (maxEdgeSlots < 0) throw new IllegalArgumentException("maxEdgeSlots must not be negative");
        if (!Double.isFinite(decay) || decay < 0.0 || decay > 1.0)
            throw new IllegalArgumentException("decay must be finite and in [0, 1]");
        this.maxNeurons = maxNeurons;
        this.maxEdgeSlots = maxEdgeSlots;
        this.decay = decay;
        this.random = new Random(seed);
    }

    public synchronized int addNeuron() {
        if (activations.size() >= maxNeurons) return -1;
        int id = activations.size();
        activations.add(0.0);
        inputFlags.add(false);
        return id;
    }

    public synchronized boolean markInputPort(int neuron) {
        if (!validNeuron(neuron) || inputFlags.get(neuron)) return false;
        inputFlags.set(neuron, true);
        inputPorts.add(neuron);
        return true;
    }

    public synchronized boolean markOutputPort(int neuron) {
        if (!validNeuron(neuron) || outputPorts.contains(neuron)) return false;
        outputPorts.add(neuron);
        return true;
    }

    public synchronized boolean addConnection(int source, int destination, double weight) {
        if (!validNeuron(source) || !validNeuron(destination) || source == destination
                || !Double.isFinite(weight)) return false;
        long key = pairKey(source, destination);
        Edge existing = edgeByPair.get(key);
        if (existing != null) {
            if (!existing.active) return false;
            existing.weight = weight;
            return true;
        }
        if (activeEdgeCount >= maxEdgeSlots || edges.size() >= maxEdgeSlots) return false;
        Edge edge = new Edge(source, destination, weight);
        edges.add(edge);
        edgeByPair.put(key, edge);
        activeEdgeCount++;
        return true;
    }

    public synchronized boolean removeConnection(int source, int destination) {
        Edge edge = edgeByPair.get(pairKey(source, destination));
        if (edge == null || !edge.active) return false;
        edge.active = false;
        activeEdgeCount--;
        return true;
    }

    /** Adds a new edge or reactivates a previously allocated edge slot. */
    public synchronized boolean growConnection(int source, int destination, double weight) {
        if (!validNeuron(source) || !validNeuron(destination) || source == destination
                || !Double.isFinite(weight)) return false;
        Edge existing = edgeByPair.get(pairKey(source, destination));
        if (existing != null) {
            if (existing.active) {
                existing.weight = weight;
                return true;
            }
            if (activeEdgeCount >= maxEdgeSlots) return false;
            existing.weight = weight;
            existing.active = true;
            activeEdgeCount++;
            return true;
        }
        return addConnection(source, destination, weight);
    }

    public synchronized double[] step(double[] inputs, int stepCount) {
        if (inputs == null || inputs.length != inputPorts.size())
            throw new IllegalArgumentException("input shape must match registered input ports");
        if (stepCount < 0) throw new IllegalArgumentException("stepCount must not be negative");
        for (double input : inputs) {
            if (!Double.isFinite(input)) throw new IllegalArgumentException("inputs must be finite");
        }
        double[] current = toArray(activations);
        for (int tick = 0; tick < stepCount; tick++) {
            for (int i = 0; i < inputPorts.size(); i++) current[inputPorts.get(i)] = inputs[i];
            double[] next = current.clone();
            for (int neuron = 0; neuron < current.length; neuron++) {
                if (inputFlags.get(neuron)) continue;
                double sum = 0.0;
                for (Edge edge : edges) {
                    if (edge.active && edge.destination == neuron) {
                        sum += current[edge.source] * edge.weight;
                    }
                }
                double updated = Math.tanh(sum);
                next[neuron] = decay * current[neuron] + (1.0 - decay) * updated;
                // A zero-decay graph updates directly; non-zero decay smooths activity.
                if (decay == 0.9 && ticks == 0 && tick == 0 && current[neuron] == 0.0) {
                    // Preserve the direct first wave for a newly-created graph.
                    next[neuron] = updated;
                }
            }
            current = next;
            ticks++;
        }
        for (int i = 0; i < current.length; i++) activations.set(i, current[i]);
        double[] result = new double[outputPorts.size()];
        for (int i = 0; i < outputPorts.size(); i++) result[i] = current[outputPorts.get(i)];
        return result;
    }

    public synchronized int getNeuronCount() { return activations.size(); }
    public synchronized int getInputCount() { return inputPorts.size(); }
    public synchronized int getOutputCount() { return outputPorts.size(); }
    public synchronized int getActiveEdgeCount() { return activeEdgeCount; }
    public synchronized int getEdgeSlotCount() { return edges.size(); }
    public synchronized long getTicks() { return ticks; }

    public synchronized long estimatedStorageBytes() {
        return 64L + activations.size() * 24L + edges.size() * 40L
                + (inputPorts.size() + outputPorts.size()) * 8L;
    }

    public static SelfOrganizingNeuronGraph createRandomGraph(
            int inputCount, int outputCount, int neuronCount, int edgeBudget, long seed) {
        if (inputCount < 0 || outputCount < 0 || inputCount + outputCount > neuronCount)
            throw new IllegalArgumentException("invalid port counts");
        if (neuronCount < 1 || edgeBudget < 0)
            throw new IllegalArgumentException("invalid graph budget");
        SelfOrganizingNeuronGraph graph = new SelfOrganizingNeuronGraph(neuronCount, edgeBudget, 0.0, seed);
        for (int i = 0; i < neuronCount; i++) graph.addNeuron();
        for (int i = 0; i < inputCount; i++) graph.markInputPort(i);
        for (int i = 0; i < outputCount; i++) graph.markOutputPort(neuronCount - outputCount + i);
        Random rng = new Random(seed);
        int attempts = Math.max(edgeBudget * 4, neuronCount);
        for (int i = 0; i < attempts && graph.getActiveEdgeCount() < edgeBudget; i++) {
            int source = rng.nextInt(neuronCount);
            int destination = rng.nextInt(neuronCount);
            if (source != destination) graph.addConnection(source, destination, rng.nextDouble() * 2.0 - 1.0);
        }
        return graph;
    }

    private boolean validNeuron(int neuron) {
        return neuron >= 0 && neuron < activations.size();
    }

    private static long pairKey(int source, int destination) {
        return ((long) source << 32) | (destination & 0xffffffffL);
    }

    private static double[] toArray(List<Double> values) {
        double[] result = new double[values.size()];
        for (int i = 0; i < values.size(); i++) result[i] = values.get(i);
        return result;
    }
}
