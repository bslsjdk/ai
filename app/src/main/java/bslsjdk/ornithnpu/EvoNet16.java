package bslsjdk.ornithnpu;

import org.json.JSONArray;
import org.json.JSONObject;
import java.util.Random;

/**
 * Compact 16-feature policy used only by the isolated neuroevolution experiment.
 * It intentionally does not share weights or replay data with the legacy 168-input QNet.
 */
final class EvoNet16 {
    static final int INPUTS = 16;
    static final int ACTIONS = 4;
    final int hidden;
    final double[][] w1;
    final double[][] w2;
    final double[] b1;
    final double[] b2;

    EvoNet16(long seed, int hidden) {
        if (hidden < 8 || hidden > 256) throw new IllegalArgumentException("hidden must be 8..256");
        this.hidden = hidden;
        w1 = new double[hidden][INPUTS];
        w2 = new double[ACTIONS][hidden];
        b1 = new double[hidden];
        b2 = new double[ACTIONS];
        Random r = new Random(seed);
        double scale = 0.5 / Math.sqrt(INPUTS);
        for (int j = 0; j < hidden; j++) {
            for (int i = 0; i < INPUTS; i++) w1[j][i] = (r.nextDouble() * 2 - 1) * scale;
        }
        for (int a = 0; a < ACTIONS; a++) {
            for (int j = 0; j < hidden; j++) w2[a][j] = (r.nextDouble() * 2 - 1) * 0.1;
        }
    }

    private final double[] hiddenScratch = new double[256];
    private final double[] qScratch = new double[ACTIONS];

    int choose(double[] x) {
        if (x == null || x.length != INPUTS) throw new IllegalArgumentException("EvoNet16 expects 16 inputs");
        for (int j = 0; j < hidden; j++) {
            double v = b1[j];
            for (int i = 0; i < INPUTS; i++) v += w1[j][i] * x[i];
            hiddenScratch[j] = Math.max(0.0, v);
        }
        int best = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < ACTIONS; a++) {
            double v = b2[a];
            for (int j = 0; j < hidden; j++) v += w2[a][j] * hiddenScratch[j];
            qScratch[a] = v;
            if (v > bestValue) { bestValue = v; best = a; }
        }
        return best;
    }

    double[] forward(double[] x) {
        if (x == null || x.length != INPUTS) throw new IllegalArgumentException("EvoNet16 expects 16 inputs");
        double[] h = new double[hidden];
        double[] q = new double[ACTIONS];
        for (int j = 0; j < hidden; j++) {
            double v = b1[j];
            for (int i = 0; i < INPUTS; i++) v += w1[j][i] * x[i];
            h[j] = Math.max(0.0, v);
        }
        for (int a = 0; a < ACTIONS; a++) {
            double v = b2[a];
            for (int j = 0; j < hidden; j++) v += w2[a][j] * h[j];
            q[a] = v;
        }
        return q;
    }

    EvoNet16 copy(long seed) {
        EvoNet16 n = new EvoNet16(seed, hidden);
        for (int j = 0; j < hidden; j++) {
            System.arraycopy(w1[j], 0, n.w1[j], 0, INPUTS);
            n.b1[j] = b1[j];
        }
        for (int a = 0; a < ACTIONS; a++) {
            System.arraycopy(w2[a], 0, n.w2[a], 0, hidden);
            n.b2[a] = b2[a];
        }
        return n;
    }

    void mutate(double sigma, Random r) {
        double inSigma = sigma * (0.5 / Math.sqrt(INPUTS));
        double outSigma = sigma * 0.10;
        double biasSigma = sigma * 0.02;

        // Local mutation: perturb a subset of hidden neurons instead of disturbing
        // every unit on every offspring. Output biases remain globally mutable.
        for (int j = 0; j < hidden; j++) {
            if (r.nextDouble() < 0.28) {
                for (int i = 0; i < INPUTS; i++) w1[j][i] += r.nextGaussian() * inSigma;
                b1[j] += r.nextGaussian() * biasSigma;
                for (int a = 0; a < ACTIONS; a++) w2[a][j] += r.nextGaussian() * outSigma;
            }
        }
        for (int a = 0; a < ACTIONS; a++) b2[a] += r.nextGaussian() * biasSigma;

        // Recycle one low-importance neuron slot from a high-importance neuron.
        // Importance is a cheap structural proxy (incoming magnitude × outgoing magnitude),
        // not a claim of causal neuron attribution; validation still decides survival.
        if (hidden > 1 && r.nextDouble() < 0.20) {
            int strongest = 0, weakest = 0;
            double high = Double.NEGATIVE_INFINITY, low = Double.POSITIVE_INFINITY;
            for (int j = 0; j < hidden; j++) {
                double incoming = 0.0, outgoing = 0.0;
                for (int i = 0; i < INPUTS; i++) incoming += Math.abs(w1[j][i]);
                for (int a = 0; a < ACTIONS; a++) outgoing += Math.abs(w2[a][j]);
                double score = incoming * outgoing;
                if (score > high) { high = score; strongest = j; }
                if (score < low) { low = score; weakest = j; }
            }
            if (strongest != weakest) {
                System.arraycopy(w1[strongest], 0, w1[weakest], 0, INPUTS);
                b1[weakest] = b1[strongest];
                for (int a = 0; a < ACTIONS; a++) {
                    w2[a][weakest] = w2[a][strongest] + r.nextGaussian() * outSigma;
                }
                for (int i = 0; i < INPUTS; i++) w1[weakest][i] += r.nextGaussian() * inSigma;
                b1[weakest] += r.nextGaussian() * biasSigma;
            }
        }
    }

    JSONObject toJson() throws Exception {
        JSONObject o = new JSONObject();
        o.put("format", "aimeng-evolution-policy-16/v1");
        o.put("inputSize", INPUTS);
        o.put("hiddenSize", hidden);
        o.put("outputSize", ACTIONS);
        o.put("features", "dx,dy,bfs_distance,local_3x3,previous_action_onehot");
        o.put("w1", matrix(w1));
        o.put("w2", matrix(w2));
        o.put("b1", vector(b1));
        o.put("b2", vector(b2));
        return o;
    }

    static EvoNet16 fromJson(JSONObject o) throws Exception {
        if (!"aimeng-evolution-policy-16/v1".equals(o.optString("format"))
                || o.optInt("inputSize", -1) != INPUTS || o.optInt("outputSize", -1) != ACTIONS) {
            throw new IllegalArgumentException("EvoNet16 checkpoint format mismatch");
        }
        int hidden = o.getInt("hiddenSize");
        EvoNet16 n = new EvoNet16(1L, hidden);
        readMatrix(o.getJSONArray("w1"), n.w1);
        readMatrix(o.getJSONArray("w2"), n.w2);
        readVector(o.getJSONArray("b1"), n.b1);
        readVector(o.getJSONArray("b2"), n.b2);
        return n;
    }

    private static JSONArray vector(double[] values) throws Exception {
        JSONArray a = new JSONArray();
        for (double v : values) a.put(v);
        return a;
    }

    private static JSONArray matrix(double[][] values) throws Exception {
        JSONArray a = new JSONArray();
        for (double[] row : values) a.put(vector(row));
        return a;
    }

    private static void readVector(JSONArray a, double[] out) throws Exception {
        if (a.length() != out.length) throw new IllegalArgumentException("checkpoint vector dimension mismatch");
        for (int i = 0; i < out.length; i++) {
            double v = a.getDouble(i);
            if (!Double.isFinite(v)) throw new IllegalArgumentException("non-finite checkpoint weight");
            out[i] = v;
        }
    }

    private static void readMatrix(JSONArray a, double[][] out) throws Exception {
        if (a.length() != out.length) throw new IllegalArgumentException("checkpoint matrix dimension mismatch");
        for (int i = 0; i < out.length; i++) readVector(a.getJSONArray(i), out[i]);
    }
}
