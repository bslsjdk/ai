package bslsjdk.ornithnpu;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Tiny on-device learning experiment. Parameter updates and score policy run on
 * CPU; forward evaluation can be delegated to the real QNN/HTP int8 matmul path.
 * This is a research toy, not a language model.
 */
public final class NeuronLabEngine {
    public static final class Unit {
        public final String id;
        public double weight;
        public double bias;
        public final double learningRate;
        public boolean enabled = true;
        public int revision = 1;
        public double score;
        public String state = "active";

        Unit(String id, double weight, double bias, double learningRate) {
            this.id = id;
            this.weight = weight;
            this.bias = bias;
            this.learningRate = learningRate;
        }

        double predict(double x) { return weight * x + bias; }

        double learn(List<Sample> data) {
            if (data.isEmpty()) throw new IllegalArgumentException("empty training data");
            double gw = 0.0, gb = 0.0, loss = 0.0;
            for (Sample s : data) {
                double error = predict(s.x) - s.target;
                loss += error * error;
                gw += 2.0 * error * s.x;
                gb += 2.0 * error;
            }
            double scale = 1.0 / data.size();
            weight -= learningRate * gw * scale;
            bias -= learningRate * gb * scale;
            return loss * scale;
        }
    }

    public static final class Sample {
        public final double x;
        public final double target;
        Sample(double x, double target) { this.x = x; this.target = target; }
    }

    public interface ForwardBackend {
        String name();
        double[][] predict(List<Unit> active, List<Sample> samples) throws Exception;
    }

    public static final class Result {
        public final String report;
        public final int activeCount;
        public final double initialMse;
        public final double finalMse;
        public final boolean npuUsed;
        Result(String report, int activeCount, double initialMse, double finalMse, boolean npuUsed) {
            this.report = report;
            this.activeCount = activeCount;
            this.initialMse = initialMse;
            this.finalMse = finalMse;
            this.npuUsed = npuUsed;
        }
    }

    private final ArrayList<Unit> units = new ArrayList<>();
    private final ArrayList<JSONObject> trace = new ArrayList<>();
    private final List<Sample> train = new ArrayList<>();
    private final List<Sample> validation = new ArrayList<>();
    private int epoch;

    public NeuronLabEngine(int unitCount) {
        if (unitCount < 2 || unitCount > 16) throw new IllegalArgumentException("unit count must be 2..16");
        train.add(new Sample(-1.0, -1.0));
        train.add(new Sample(-0.5, 0.0));
        train.add(new Sample(0.0, 1.0));
        train.add(new Sample(0.5, 2.0));
        train.add(new Sample(1.0, 3.0));
        validation.add(new Sample(-0.8, -0.6));
        validation.add(new Sample(-0.2, 0.6));
        validation.add(new Sample(0.3, 1.6));
        validation.add(new Sample(0.9, 2.8));
        for (int i = 0; i < unitCount; i++) {
            double[][] seeds = {{-0.8, 0.2}, {0.1, -0.5}, {1.8, 0.8}, {-1.5, 2.0},
                    {0.5, -1.0}, {2.4, -0.4}, {-0.2, 1.5}, {1.2, -1.4},
                    {-2.0, 0.1}, {0.9, 2.2}, {1.4, 0.0}, {-0.6, -0.8},
                    {0.2, 2.5}, {2.0, -1.2}, {-1.1, 1.1}, {0.7, 0.3}};
            double[] seed = seeds[i];
            units.add(new Unit("unit-" + i, seed[0], seed[1], 0.03));
        }
        trace("experiment_start", new JSONObject().put("units", unitCount).put("target", "y=2*x+1"));
    }

    public synchronized Result trainAndScore(int epochs, ForwardBackend backend, File traceFile) throws org.json.JSONException {
        if (epochs < 1 || epochs > 500) throw new IllegalArgumentException("epochs must be 1..500");
        double initial = evaluate(validation, null);
        int startEpoch = epoch;
        for (int i = 0; i < epochs; i++) {
            epoch++;
            for (Unit u : units) {
                double before = u.learn(train);
                trace("parameter_update", new JSONObject()
                        .put("epoch", epoch).put("unit_id", u.id).put("revision", u.revision)
                        .put("weight", u.weight).put("bias", u.bias).put("training_mse_before", before));
            }
            List<JSONObject> scores = scoreUnits(0.002);
            List<JSONObject> changes = applyScores(scores);
            trace("score_cycle", new JSONObject().put("epoch", epoch)
                    .put("scores", new JSONArray(scores)).put("state_changes", new JSONArray(changes))
                    .put("active_ids", activeIds()).put("validation_mse_cpu", evaluate(validation, null)));
        }
        double finalCpu = evaluate(validation, null);
        boolean npuUsed = false;
        String backendReport = "forward_backend=CPU";
        if (backend != null) {
            try {
                double npuMse = evaluate(validation, backend);
                npuUsed = true;
                double delta = Math.abs(npuMse - finalCpu);
                backendReport = "forward_backend=" + backend.name()
                        + " validation_mse=" + format(npuMse)
                        + " cpu_npu_abs_delta=" + format(delta)
                        + " npu_match_within_0.25=" + (delta <= 0.25);
                trace("backend_validation", new JSONObject().put("backend", backend.name())
                        .put("cpu_mse", finalCpu).put("backend_mse", npuMse)
                        .put("abs_delta", Math.abs(npuMse - finalCpu)));
            } catch (Throwable t) {
                backendReport = "forward_backend_fallback=CPU reason="
                        + t.getClass().getSimpleName() + ":" + String.valueOf(t.getMessage());
                trace("backend_error", new JSONObject().put("backend", backend.name())
                        .put("error", backendReport));
            }
        }
        try { flushTrace(traceFile); } catch (Throwable t) {
            backendReport += " trace_write_error=" + t.getClass().getSimpleName();
        }
        String report = "Mobile neuron lab\n"
                + "epoch=" + startEpoch + "->" + epoch + " units=" + units.size()
                + " active=" + activeCount() + "\n"
                + "target: y=2*x+1\n"
                + "held_out_mse: " + format(initial) + " -> " + format(finalCpu)
                + " improved=" + (finalCpu < initial) + "\n"
                + backendReport + "\n"
                + "trace=" + traceFile.getAbsolutePath() + "\n"
                + "NOTE: toy regression only; not a language model. Parameter learning/scoring run on CPU.";
        return new Result(report, activeCount(), initial, finalCpu, npuUsed);
    }

    public synchronized String summary() {
        StringBuilder out = new StringBuilder();
        out.append("units=").append(units.size()).append(" active=").append(activeCount())
                .append(" epoch=").append(epoch).append('\n');
        for (Unit u : units) out.append(u.id).append(" state=").append(u.state)
                .append(" score=").append(format(u.score))
                .append(" w=").append(format(u.weight)).append(" b=").append(format(u.bias))
                .append(" rev=").append(u.revision).append('\n');
        return out.toString();
    }

    private List<JSONObject> scoreUnits(double cost) throws org.json.JSONException {
        double baseline = evaluate(validation, null);
        ArrayList<JSONObject> records = new ArrayList<>();
        for (Unit u : units) {
            boolean wasEnabled = u.enabled;
            double counterfactual;
            double contribution;
            String comparison;
            if (wasEnabled) {
                u.enabled = false;
                counterfactual = evaluate(validation, null);
                u.enabled = true;
                contribution = counterfactual - baseline;
                comparison = "remove_active_unit";
            } else {
                u.enabled = true;
                counterfactual = evaluate(validation, null);
                u.enabled = false;
                contribution = baseline - counterfactual;
                comparison = "admit_sleeping_unit";
            }
            u.score = contribution - cost;
            records.add(new JSONObject().put("unit_id", u.id).put("revision", u.revision)
                    .put("enabled", wasEnabled).put("comparison", comparison)
                    .put("baseline_mse", baseline).put("counterfactual_mse", counterfactual)
                    .put("marginal_contribution", contribution).put("compute_cost", cost)
                    .put("score", u.score));
        }
        return records;
    }

    private List<JSONObject> applyScores(List<JSONObject> scores) throws org.json.JSONException {
        ArrayList<JSONObject> changes = new ArrayList<>();
        for (int i = 0; i < units.size(); i++) {
            Unit u = units.get(i);
            boolean wasEnabled = u.enabled;
            if (u.enabled && u.score < -0.0001) {
                u.enabled = false;
                u.state = "sleeping";
            } else if (!u.enabled && u.score > 0.0001) {
                u.enabled = true;
                u.state = "active";
            }
            if (wasEnabled != u.enabled) changes.add(new JSONObject()
                    .put("unit_id", u.id).put("revision", u.revision)
                    .put("from", wasEnabled ? "active" : "sleeping")
                    .put("to", u.state).put("score", u.score));
        }
        if (!units.isEmpty() && activeCount() == 0) {
            Unit best = Collections.max(units, (a, b) -> Double.compare(a.score, b.score));
            best.enabled = true;
            best.state = "active";
            changes.add(new JSONObject().put("unit_id", best.id).put("revision", best.revision)
                    .put("from", "sleeping").put("to", "active")
                    .put("score", best.score).put("reason", "minimum_active_unit_safety_floor"));
        }
        return changes;
    }

    private double evaluate(List<Sample> data, ForwardBackend backend) {
        ArrayList<Unit> active = new ArrayList<>();
        for (Unit u : units) if (u.enabled) active.add(u);
        if (active.isEmpty()) {
            // A real all-disabled counterfactual predicts zero. Do not silently
            // run a disabled unit just to avoid an empty ensemble.
            double zeroLoss = 0.0;
            for (Sample sample : data) zeroLoss += sample.target * sample.target;
            return zeroLoss / data.size();
        }
        double[][] predictions;
        try {
            if (backend == null) {
                predictions = new double[data.size()][active.size()];
                for (int r = 0; r < data.size(); r++)
                    for (int c = 0; c < active.size(); c++)
                        predictions[r][c] = active.get(c).predict(data.get(r).x);
            } else {
                predictions = backend.predict(active, data);
            }
        } catch (Exception e) {
            throw new IllegalStateException("forward backend failed", e);
        }
        double sum = 0.0;
        for (int r = 0; r < data.size(); r++) {
            double y = 0.0;
            for (int c = 0; c < active.size(); c++) y += predictions[r][c];
            y /= active.size();
            double err = y - data.get(r).target;
            sum += err * err;
        }
        return sum / data.size();
    }

    private JSONArray activeIds() throws org.json.JSONException {
        JSONArray ids = new JSONArray();
        for (Unit u : units) if (u.enabled) ids.put(u.id);
        return ids;
    }

    private int activeCount() {
        int count = 0;
        for (Unit u : units) if (u.enabled) count++;
        return count;
    }

    private void trace(String event, JSONObject values) {
        JSONObject row = new JSONObject();
        try {
            row.put("schema", "aimeng.neuron_trace.v1");
            row.put("event", event);
            row.put("time_ms", System.currentTimeMillis());
            row.put("epoch", epoch);
            row.put("data", values);
            trace.add(row);
        } catch (Exception ignored) { }
    }

    private void flushTrace(File file) throws Exception {
        if (file.length() > 2L * 1024L * 1024L) {
            File rotated = new File(file.getParentFile(), "neuron-lab-trace.previous.jsonl");
            if (rotated.exists()) rotated.delete();
            if (!file.renameTo(rotated)) throw new IllegalStateException("trace rotation failed");
        }
        try (FileOutputStream out = new FileOutputStream(file, true)) {
            for (JSONObject row : trace) {
                out.write(row.toString().getBytes(StandardCharsets.UTF_8));
                out.write('\n');
            }
            out.flush();
        }
        trace.clear();
    }

    private static String format(double v) { return String.format(java.util.Locale.US, "%.6f", v); }
}
