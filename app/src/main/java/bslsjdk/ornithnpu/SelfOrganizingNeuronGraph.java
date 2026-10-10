package bslsjdk.ornithnpu;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.Arrays;
import java.util.Random;

/**
 * Bounded, self-organizing recurrent neuron graph.
 *
 * This is deliberately NOT a layered MLP: any neuron may connect to any other
 * neuron, cycles are allowed, and edges can be grown or pruned at runtime.
 * Inputs and outputs are ports attached to ordinary graph nodes.
 *
 * CPU reference implementation. It does not use GPU/NPU and makes no claim
 * that its local reward rule matches backpropagation sample efficiency.
 */
public final class SelfOrganizingNeuronGraph {
    public static final String FORMAT = "aimeng-self-organizing-neuron-graph/v1";
    public static final int DEFAULT_MAX_NEURONS = 256;
    public static final int DEFAULT_MAX_EDGES = 4096;
    private static final double MAX_ABS_WEIGHT = 4.0;

    private final int maxNeurons;
    private final int maxEdges;
    private final double traceDecay;
    private final Random random;

    private int neuronCount;
    private int edgeCount;
    private int inputCount;
    private int outputCount;
    private final int[] inputOrder;
    private int nextInputOrder;
    private final int[] outputOrder;
    private int nextOutputOrder;
    private final boolean[] enabled;
    private final boolean[] inputPort;
    private final boolean[] outputPort;
    private final double[] bias;
    private final double[] state;
    private final double[] previous;
    private final double[] activityMean;
    private final double[] scratchSum;
    private final int[] scratchIncomingCount;
    private final int[] edgeFrom;
    private final int[] edgeTo;
    private final int[] edgeNextOut;
    private final int[] edgeHeadOut;
    private int activeNeuronBudget = 256;
    private int lastActiveNeuronCount;
    private int[] activeNodes;
    private int[] nextActiveNodes;
    private boolean[] activeMask;
    private boolean[] nextActiveMask;
    private final double[] edgeWeight;
    private final double[] edgeTrace;
    private final double[] edgeUse;
    private final boolean[] edgeEnabled;
    private double rewardBaseline;
    private long ticks;
    private long topologyChanges;

    public SelfOrganizingNeuronGraph() {
        this(DEFAULT_MAX_MAX_NEURONS(), DEFAULT_MAX_EDGES, 0.90, 20261010L);
    }

    private static int DEFAULT_MAX_MAX_NEURONS() { return DEFAULT_MAX_NEURONS; }

    public SelfOrganizingNeuronGraph(int maxNeurons, int maxEdges,
                                     double traceDecay, long seed) {
        if (maxNeurons < 2 || maxNeurons > 1000)
            throw new IllegalArgumentException("maxNeurons must be 2..1000");
        if (maxEdges < 1 || maxEdges > 16000)
            throw new IllegalArgumentException("maxEdges must be 1..16000");
        if (!(traceDecay >= 0.0 && traceDecay < 1.0))
            throw new IllegalArgumentException("traceDecay must be in [0,1)");
        this.maxNeurons = maxNeurons;
        this.maxEdges = maxEdges;
        this.traceDecay = traceDecay;
        this.random = new Random(seed);
        enabled = new boolean[maxNeurons];
        inputPort = new boolean[maxNeurons];
        inputOrder = new int[maxNeurons];
        Arrays.fill(inputOrder, -1);
        outputOrder = new int[maxNeurons];
        Arrays.fill(outputOrder, -1);
        outputPort = new boolean[maxNeurons];
        bias = new double[maxNeurons];
        state = new double[maxNeurons];
        previous = new double[maxNeurons];
        activityMean = new double[maxNeurons];
        scratchSum = new double[maxNeurons];
        scratchIncomingCount = new int[maxNeurons];
        activeNodes = new int[maxNeurons];
        nextActiveNodes = new int[maxNeurons];
        activeMask = new boolean[maxNeurons];
        nextActiveMask = new boolean[maxNeurons];
        edgeFrom = new int[maxEdges];
        edgeTo = new int[maxEdges];
        edgeNextOut = new int[maxEdges];
        edgeHeadOut = new int[maxNeurons];
        Arrays.fill(edgeHeadOut, -1);
        edgeWeight = new double[maxEdges];
        edgeTrace = new double[maxEdges];
        edgeUse = new double[maxEdges];
        edgeEnabled = new boolean[maxEdges];
    }

    /**
     * Builds a reproducible starting graph. Output ports are readouts, not a
     * separate layer. Random edges can connect any non-input nodes and form cycles.
     */
    public static SelfOrganizingNeuronGraph createRandomGraph(
            int inputCount, int outputCount, int totalNeurons, int initialEdges, long seed) {
        if (inputCount < 1 || outputCount < 1 || totalNeurons < inputCount + outputCount
                || totalNeurons > 1000 || initialEdges < 0 || initialEdges > 16000)
            throw new IllegalArgumentException("invalid graph shape");
        int maxN = Math.min(1000, Math.max(totalNeurons, totalNeurons + Math.max(16, totalNeurons / 4)));
        int maxE = Math.min(16000, Math.max(1, Math.max(initialEdges + 128, totalNeurons * 8)));
        SelfOrganizingNeuronGraph g = new SelfOrganizingNeuronGraph(maxN, maxE, 0.90, seed);
        for (int i = 0; i < totalNeurons; i++) g.addNeuron();
        for (int i = 0; i < inputCount; i++) g.markInputPort(i);
        for (int i = totalNeurons - outputCount; i < totalNeurons; i++) g.markOutputPort(i);
        Random r = new Random(seed ^ 0x5DEECE66DL);
        int randomEdgeTarget = Math.max(0, initialEdges - outputCount);
        int added = 0, attempts = 0, maxAttempts = Math.max(100, initialEdges * 20);
        while (added < randomEdgeTarget && attempts++ < maxAttempts && g.edgeCount < g.maxEdges) {
            int from = r.nextInt(totalNeurons), to = r.nextInt(totalNeurons);
            if (g.inputPort[to] || from == to) continue;
            if (g.addConnection(from, to, (r.nextDouble() * 2.0 - 1.0) * 0.35)) added++;
        }
        // Guarantee every readout is reachable directly from a sensor. These edges
        // are inserted last, so the outgoing adjacency visits them early under a
        // small active-neuron budget instead of starving all action ports.
        for (int o = totalNeurons - outputCount; o < totalNeurons; o++) {
            boolean connected = g.findEdge(g.inputOrder[0], o) >= 0;
            if (!connected) g.addConnection(g.inputOrder[0], o,
                    (r.nextDouble() * 2.0 - 1.0) * 0.2);
        }
        return g;
    }

    /** Adds an ordinary graph node; it has no fixed layer or role. */
    public int addNeuron() {
        if (neuronCount >= maxNeurons) return -1;
        int id = neuronCount++;
        enabled[id] = true;
        bias[id] = (random.nextDouble() - 0.5) * 0.04;
        return id;
    }

    /** Marks an existing node as an input port. Ports remain normal graph nodes. */
    public boolean markInputPort(int neuronId) {
        if (!validNeuron(neuronId) || inputPort[neuronId]) return false;
        inputPort[neuronId] = true;
        inputOrder[nextInputOrder++] = neuronId;
        inputCount++;
        return true;
    }

    /** Marks an existing node as an output readout port. */
    public boolean markOutputPort(int neuronId) {
        if (!validNeuron(neuronId) || outputPort[neuronId]) return false;
        outputPort[neuronId] = true;
        outputOrder[nextOutputOrder++] = neuronId;
        outputCount++;
        return true;
    }

    /** Adds a directed synapse. Cycles and recurrent connections are permitted. */
    public boolean addConnection(int from, int to, double weight) {
        if (!validNeuron(from) || !validNeuron(to) || from == to
                || !Double.isFinite(weight) || edgeCount >= maxEdges
                || findEdge(from, to) >= 0) return false;
        int e = edgeCount++;
        edgeFrom[e] = from;
        edgeTo[e] = to;
        edgeNextOut[e] = edgeHeadOut[from];
        edgeHeadOut[from] = e;
        edgeWeight[e] = clamp(weight, -MAX_ABS_WEIGHT, MAX_ABS_WEIGHT);
        edgeTrace[e] = 0;
        edgeUse[e] = 0;
        edgeEnabled[e] = true;
        topologyChanges++;
        return true;
    }

    /** Disables a synapse without shifting arrays, preserving stable edge slots. */
    public boolean removeConnection(int from, int to) {
        int e = findEdge(from, to);
        if (e < 0) return false;
        edgeEnabled[e] = false;
        edgeTrace[e] = 0;
        edgeUse[e] = 0;
        topologyChanges++;
        return true;
    }

    /**
     * Reactivates a disabled edge or adds a new one. This is the explicit,
     * budget-checked topology plasticity hook for experiment controllers.
     */
    public boolean growConnection(int from, int to, double initialWeight) {
        int e = findEdge(from, to);
        if (e >= 0) {
            if (edgeEnabled[e]) return false;
            edgeEnabled[e] = true;
            edgeWeight[e] = clamp(initialWeight, -MAX_ABS_WEIGHT, MAX_ABS_WEIGHT);
            topologyChanges++;
            return true;
        }
        return addConnection(from, to, initialWeight);
    }

    /**
     * One bounded synchronous propagation tick. The input array is ordered by
     * markInputPort call order. Every propagation round reads a snapshot, so
     * results do not depend on Java loop ordering.
     */
    public double[] step(double[] inputs, int propagationRounds) {
        if (inputs == null || inputs.length != inputCount)
            throw new IllegalArgumentException("input length must equal input port count");
        if (propagationRounds < 1 || propagationRounds > 8)
            throw new IllegalArgumentException("propagationRounds must be 1..8");
        for (double v : inputs) if (!Double.isFinite(v))
            throw new IllegalArgumentException("input contains non-finite value");

        for (int p = 0; p < outputCount; p++) {
            int id = outputOrder[p];
            if (!inputPort[id]) state[id] = 0.0;
        }
        for (int p = 0; p < inputCount; p++) {
            int i = inputOrder[p];
            state[i] = enabled[i] ? clamp(inputs[p], -1.0, 1.0) : 0.0;
        }

        // Begin from sensor ports, then activate only nodes reached through outgoing
        // edges, capped by activeNeuronBudget. The graph remains dynamic and recurrent;
        // this is bounded sparse activation, not a fixed layer layout.
        int activeCount = 0;
        for (int p = 0; p < inputCount; p++) {
            int id = inputOrder[p];
            if (enabled[id] && !activeMask[id]) {
                activeMask[id] = true;
                activeNodes[activeCount++] = id;
            }
        }
        for (int round = 0; round < propagationRounds; round++) {
            int nextCount = 0;
            for (int p = 0; p < inputCount; p++) {
                int id = inputOrder[p];
                if (enabled[id] && !nextActiveMask[id] && nextCount < activeNeuronBudget) {
                    nextActiveMask[id] = true;
                    nextActiveNodes[nextCount++] = id;
                }
            }
            for (int a = 0; a < activeCount; a++) {
                int source = activeNodes[a];
                for (int e = edgeHeadOut[source]; e >= 0; e = edgeNextOut[e]) {
                    if (!edgeEnabled[e]) continue;
                    int target = edgeTo[e];
                    if (!enabled[target] || inputPort[target]) continue;
                    if (!nextActiveMask[target]) {
                        if (nextCount >= activeNeuronBudget) continue;
                        nextActiveMask[target] = true;
                        nextActiveNodes[nextCount++] = target;
                        scratchSum[target] = bias[target];
                    }
                    scratchSum[target] += state[source] * edgeWeight[e];
                }
            }
            for (int i = 0; i < nextCount; i++) {
                int id = nextActiveNodes[i];
                if (!inputPort[id])
                    state[id] = Math.tanh(clamp(scratchSum[id], -8.0, 8.0));
            }
            for (int i = 0; i < activeCount; i++) activeMask[activeNodes[i]] = false;
            int[] nodeSwap = activeNodes; activeNodes = nextActiveNodes; nextActiveNodes = nodeSwap;
            boolean[] maskSwap = activeMask; activeMask = nextActiveMask; nextActiveMask = maskSwap;
            activeCount = nextCount;
        }
        lastActiveNeuronCount = activeCount;
        for (int i = 0; i < activeCount; i++) {
            int id = activeNodes[i];
            activityMean[id] = activityMean[id] * 0.99 + Math.abs(state[id]) * 0.01;
        }
        // Inactive edges lose eligibility gradually and cannot learn from stale states.
        for (int e = 0; e < edgeCount; e++) {
            if (!edgeEnabled[e]) continue;
            if (activeMask[edgeFrom[e]] && activeMask[edgeTo[e]]) {
                double pre = state[edgeFrom[e]];
                double post = state[edgeTo[e]];
                edgeTrace[e] = clamp(traceDecay * edgeTrace[e] + pre * post, -10, 10);
                edgeUse[e] = edgeUse[e] * 0.995 + Math.abs(pre * post) * 0.005;
            } else {
                edgeTrace[e] *= traceDecay;
                edgeUse[e] *= 0.995;
            }
        }
        ticks++;
        for (int i = 0; i < activeCount; i++) activeMask[activeNodes[i]] = false;
        double[] outputs = new double[outputCount];
        int p = 0;
        for (int i = 0; i < outputCount; i++)
            outputs[p++] = state[outputOrder[i]];
        return outputs;
    }

    /**
     * Applies reward-modulated local plasticity. Call after step() with reward
     * from the environment. Baseline subtraction reduces raw reward bias.
     */
    public void applyReward(double reward, double learningRate) {
        applyRewardInternal(reward, learningRate, -1);
    }

    /** Reward update with explicit credit assigned to the action readout selected by the policy. */
    public void applyRewardForAction(int action, double reward, double learningRate) {
        if (action < 0 || action >= outputCount)
            throw new IllegalArgumentException("invalid action readout");
        applyRewardInternal(reward, learningRate, action);
    }

    private void applyRewardInternal(double reward, double learningRate, int selectedAction) {
        if (!Double.isFinite(reward) || !Double.isFinite(learningRate)
                || learningRate < 0 || learningRate > 0.1)
            throw new IllegalArgumentException("invalid reward or learning rate");
        double advantage = reward - rewardBaseline;
        rewardBaseline = rewardBaseline * 0.98 + reward * 0.02;
        for (int e = 0; e < edgeCount; e++) {
            if (!edgeEnabled[e]) continue;
            double update = learningRate * advantage * edgeTrace[e];
            if (selectedAction >= 0 && outputPort[edgeTo[e]]) {
                double actionScale = edgeTo[e] == outputOrder[selectedAction]
                        ? 1.0 : -1.0 / Math.max(1, outputCount - 1);
                update += learningRate * advantage * edgeTrace[e] * actionScale * 3.0;
            }
            edgeWeight[e] = clamp(edgeWeight[e] + update, -MAX_ABS_WEIGHT, MAX_ABS_WEIGHT);
        }
        // Reuse the per-neuron scratch buffer to aggregate incoming traces in O(V + E).
        // A nested neuron-by-edge scan would become prohibitively expensive as graphs grow.
        Arrays.fill(scratchSum, 0, neuronCount, 0.0);
        Arrays.fill(scratchIncomingCount, 0, neuronCount, 0);
        for (int e = 0; e < edgeCount; e++) {
            if (!edgeEnabled[e]) continue;
            int target = edgeTo[e];
            scratchSum[target] += edgeTrace[e];
            scratchIncomingCount[target]++;
        }
        for (int i = 0; i < neuronCount; i++) {
            if (enabled[i] && !inputPort[i] && scratchIncomingCount[i] > 0) {
                bias[i] = clamp(bias[i] + learningRate * advantage
                        * scratchSum[i] / scratchIncomingCount[i], -2, 2);
            }
        }
        if (selectedAction >= 0) {
            for (int a = 0; a < outputCount; a++) {
                int id = outputOrder[a];
                double scale = a == selectedAction ? 1.0 : -1.0 / Math.max(1, outputCount - 1);
                bias[id] = clamp(bias[id] + learningRate * advantage * scale * 0.5, -2, 2);
            }
        }
    }

    /** Decays traces between episodes without destroying learned weights. */
    public void resetEpisodeState() {
        Arrays.fill(state, 0, neuronCount, 0);
        Arrays.fill(previous, 0, neuronCount, 0);
        Arrays.fill(edgeTrace, 0, edgeCount, 0);
        Arrays.fill(activeMask, 0, neuronCount, false);
        Arrays.fill(nextActiveMask, 0, neuronCount, false);
        lastActiveNeuronCount = 0;
    }

    /**
     * Conservative topology adaptation: only adds a connection between two
     * repeatedly co-active nodes if the pair is not already connected.
     * The caller controls when this is invoked; budgets are always enforced.
     */
    public boolean tryGrowFromCoactivity(double minimumActivity, double initialWeight) {
        if (neuronCount < 2 || edgeCount >= maxEdges) return false;
        for (int attempt = 0; attempt < Math.min(64, neuronCount * 2); attempt++) {
            int a = random.nextInt(neuronCount);
            int b = random.nextInt(neuronCount);
            if (a == b || !enabled[a] || !enabled[b]) continue;
            if (inputPort[a] || outputPort[a] || inputPort[b] || outputPort[b]) continue;
            if (activityMean[a] < minimumActivity || activityMean[b] < minimumActivity) continue;
            if (Math.abs(state[a] * state[b]) < minimumActivity * minimumActivity) continue;
            if (findEdge(a, b) >= 0) continue;
            return addConnection(a, b, initialWeight);
        }
        return false;
    }

    /** Prunes the least-used active edge only when its use is below the threshold. */
    public boolean pruneLeastUsed(double maximumUse) {
        int candidate = -1;
        double least = Double.POSITIVE_INFINITY;
        for (int e = 0; e < edgeCount; e++) {
            if (edgeEnabled[e] && edgeUse[e] < least) {
                least = edgeUse[e];
                candidate = e;
            }
        }
        if (candidate < 0 || least > maximumUse) return false;
        edgeEnabled[candidate] = false;
        edgeTrace[candidate] = 0;
        topologyChanges++;
        return true;
    }

    public int getNeuronCount() { return neuronCount; }
    public int getEdgeSlotCount() { return edgeCount; }
    public int getActiveEdgeCount() {
        int n = 0;
        for (int e = 0; e < edgeCount; e++) if (edgeEnabled[e]) n++;
        return n;
    }
    public int getInputCount() { return inputCount; }
    public int getOutputCount() { return outputCount; }
    public long getTicks() { return ticks; }
    public long getTopologyChanges() { return topologyChanges; }
    public int getLastActiveNeuronCount() { return lastActiveNeuronCount; }
    public int getActiveNeuronBudget() { return activeNeuronBudget; }
    public void setActiveNeuronBudget(int budget) {
        if (budget < Math.max(1, inputCount) || budget > maxNeurons)
            throw new IllegalArgumentException("active neuron budget must be inputCount..maxNeurons");
        activeNeuronBudget = budget;
    }
    public double getRewardBaseline() { return rewardBaseline; }
    public double getNeuronActivity(int id) {
        if (!validNeuron(id)) throw new IllegalArgumentException("invalid neuron id");
        return state[id];
    }
    public long estimatedStorageBytes() {
        // Primitive-array payload estimate; excludes object headers and JSON snapshots.
        return (long) maxNeurons * (3 + 4 * 5 + 8 * 5 + 4 + 4 + 4 * 2 + 2)
                + (long) maxEdges * (4 * 3 + 8 * 3 + 1);
    }

    public JSONObject toJson() throws JSONException {
        JSONObject root = new JSONObject();
        root.put("format", FORMAT);
        root.put("maxNeurons", maxNeurons);
        root.put("maxEdges", maxEdges);
        root.put("activeNeuronBudget", activeNeuronBudget);
        root.put("traceDecay", traceDecay);
        root.put("neuronCount", neuronCount);
        root.put("edgeCount", edgeCount);
        root.put("inputCount", inputCount);
        root.put("outputCount", outputCount);
        JSONArray portOrder = new JSONArray();
        for (int p = 0; p < inputCount; p++) portOrder.put(inputOrder[p]);
        root.put("inputOrder", portOrder);
        JSONArray readoutOrder = new JSONArray();
        for (int p = 0; p < outputCount; p++) readoutOrder.put(outputOrder[p]);
        root.put("outputOrder", readoutOrder);
        root.put("rewardBaseline", rewardBaseline);
        root.put("ticks", ticks);
        root.put("topologyChanges", topologyChanges);
        JSONArray nodes = new JSONArray();
        for (int i = 0; i < neuronCount; i++) {
            JSONObject n = new JSONObject();
            n.put("id", i);
            n.put("enabled", enabled[i]);
            n.put("input", inputPort[i]);
            n.put("output", outputPort[i]);
            n.put("bias", bias[i]);
            n.put("activityMean", activityMean[i]);
            nodes.put(n);
        }
        root.put("neurons", nodes);
        JSONArray edges = new JSONArray();
        for (int e = 0; e < edgeCount; e++) {
            JSONObject edge = new JSONObject();
            edge.put("from", edgeFrom[e]);
            edge.put("to", edgeTo[e]);
            edge.put("weight", edgeWeight[e]);
            edge.put("trace", edgeTrace[e]);
            edge.put("use", edgeUse[e]);
            edge.put("enabled", edgeEnabled[e]);
            edges.put(edge);
        }
        root.put("edges", edges);
        return root;
    }

    /** Strict loader rejects malformed or over-budget checkpoints. */
    public static SelfOrganizingNeuronGraph fromJson(JSONObject root, long seed)
            throws JSONException {
        if (root == null || !FORMAT.equals(root.optString("format")))
            throw new JSONException("unsupported graph checkpoint format");
        int maxN = root.getInt("maxNeurons");
        int maxE = root.getInt("maxEdges");
        int nCount = root.getInt("neuronCount");
        int eCount = root.getInt("edgeCount");
        if (nCount < 0 || nCount > maxN || eCount < 0 || eCount > maxE)
            throw new JSONException("checkpoint exceeds declared graph budget");
        SelfOrganizingNeuronGraph g = new SelfOrganizingNeuronGraph(maxN, maxE,
                root.getDouble("traceDecay"), seed);
        JSONArray nodes = root.getJSONArray("neurons");
        JSONArray edges = root.getJSONArray("edges");
        JSONArray portOrder = root.optJSONArray("inputOrder");
        JSONArray readoutOrder = root.optJSONArray("outputOrder");
        if (nodes.length() != nCount || edges.length() != eCount)
            throw new JSONException("checkpoint count mismatch");
        for (int i = 0; i < nCount; i++) {
            JSONObject n = nodes.getJSONObject(i);
            if (n.getInt("id") != i) throw new JSONException("unstable neuron id");
            g.addNeuron();
            g.enabled[i] = n.optBoolean("enabled", true);
            g.bias[i] = finite(n.getDouble("bias"), "bias");
            g.activityMean[i] = finite(n.optDouble("activityMean", 0), "activityMean");
            if (n.optBoolean("input", false)) g.markInputPort(i);
            if (n.optBoolean("output", false)) g.markOutputPort(i);
        }
        if (portOrder != null) {
            if (portOrder.length() != g.inputCount) throw new JSONException("input port order mismatch");
            boolean[] seenInputs = new boolean[nCount];
            for (int p = 0; p < portOrder.length(); p++) {
                int id = portOrder.getInt(p);
                if (id < 0 || id >= nCount || !g.inputPort[id] || seenInputs[id])
                    throw new JSONException("invalid input port order");
                seenInputs[id] = true;
                g.inputOrder[p] = id;
            }
            g.nextInputOrder = g.inputCount;
        }
        if (readoutOrder != null) {
            if (readoutOrder.length() != g.outputCount) throw new JSONException("output port order mismatch");
            boolean[] seenOutputs = new boolean[nCount];
            for (int p = 0; p < readoutOrder.length(); p++) {
                int id = readoutOrder.getInt(p);
                if (id < 0 || id >= nCount || !g.outputPort[id] || seenOutputs[id])
                    throw new JSONException("invalid output port order");
                seenOutputs[id] = true;
                g.outputOrder[p] = id;
            }
            g.nextOutputOrder = g.outputCount;
        }
        for (int e = 0; e < eCount; e++) {
            JSONObject edge = edges.getJSONObject(e);
            int from = edge.getInt("from"), to = edge.getInt("to");
            double w = finite(edge.getDouble("weight"), "weight");
            if (!g.addConnection(from, to, w))
                throw new JSONException("invalid or duplicate connection");
            int id = g.edgeCount - 1;
            g.edgeEnabled[id] = edge.optBoolean("enabled", true);
            g.edgeTrace[id] = finite(edge.optDouble("trace", 0), "trace");
            g.edgeUse[id] = finite(edge.optDouble("use", 0), "use");
        }
        g.activeNeuronBudget = Math.max(g.inputCount, Math.min(maxN,
                root.optInt("activeNeuronBudget", Math.min(256, maxN))));
        g.rewardBaseline = finite(root.optDouble("rewardBaseline", 0), "rewardBaseline");
        g.ticks = Math.max(0, root.optLong("ticks", 0));
        g.topologyChanges = Math.max(0, root.optLong("topologyChanges", 0));
        return g;
    }

    private int findEdge(int from, int to) {
        for (int e = 0; e < edgeCount; e++)
            if (edgeFrom[e] == from && edgeTo[e] == to) return e;
        return -1;
    }
    private boolean validNeuron(int id) { return id >= 0 && id < neuronCount; }
    private static double finite(double v, String name) throws JSONException {
        if (!Double.isFinite(v)) throw new JSONException(name + " is not finite");
        return v;
    }
    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
