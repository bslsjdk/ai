package bslsjdk.ornithnpu;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import static org.junit.Assert.*;

public class SparseDiffusionMobileModelTest {
    @Rule public TemporaryFolder temp = new TemporaryFolder();

    private static JSONObject tensor(int[] shape, float[] values) throws Exception {
        JSONArray shapeJson = new JSONArray();
        for (int x : shape) shapeJson.put(x);
        JSONArray valueJson = new JSONArray();
        for (float x : values) valueJson.put((double) x);
        JSONObject result = new JSONObject();
        result.put("shape", shapeJson);
        result.put("values", valueJson);
        return result;
    }

    private static float[] values(int n, float scale, int phase) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) out[i] = (float) (Math.sin((i + 1 + phase) * 0.37) * scale);
        return out;
    }

    private File makeBundle() throws Exception {
        final int neurons = 8, width = 4, vocab = 4, fanout = 2;
        JSONObject root = new JSONObject();
        root.put("format", "aimeng-mobile-diffusion-json-v1");
        JSONObject config = new JSONObject();
        config.put("neurons", neurons);
        config.put("width", width);
        config.put("active_k", 2);
        config.put("fanout", fanout);
        config.put("max_steps", 3);
        config.put("context", 8);
        root.put("config", config);

        JSONObject stoi = new JSONObject();
        stoi.put("a", 1);
        stoi.put("b", 2);
        stoi.put("🙂", 3);
        root.put("stoi", stoi);
        JSONArray itos = new JSONArray();
        itos.put("<unk>"); itos.put("a"); itos.put("b"); itos.put("🙂");
        root.put("itos", itos);

        JSONObject ts = new JSONObject();
        ts.put("embedding.weight", tensor(new int[]{vocab, width}, values(vocab * width, 0.12f, 1)));
        ts.put("node_embedding", tensor(new int[]{neurons, width}, values(neurons * width, 0.08f, 2)));
        float[] neighbors = new float[neurons * fanout];
        for (int n = 0; n < neurons; n++) {
            neighbors[n * fanout] = (n + 1) % neurons;
            neighbors[n * fanout + 1] = (n + neurons - 1) % neurons;
        }
        ts.put("neighbors", tensor(new int[]{neurons, fanout}, neighbors));
        ts.put("edge_logits", tensor(new int[]{neurons, fanout}, new float[neurons * fanout]));
        ts.put("context_proj.weight", tensor(new int[]{width, width}, values(width * width, 0.08f, 3)));
        ts.put("context_proj.bias", tensor(new int[]{width}, values(width, 0.01f, 4)));
        ts.put("self_proj.weight", tensor(new int[]{width, width}, values(width * width, 0.08f, 5)));
        ts.put("message_proj.weight", tensor(new int[]{width, width}, values(width * width, 0.08f, 6)));
        ts.put("gate.weight", tensor(new int[]{width, width * 2}, values(width * width * 2, 0.05f, 7)));
        ts.put("gate.bias", tensor(new int[]{width}, values(width, 0.01f, 8)));
        ts.put("decoder.weight", tensor(new int[]{vocab, width}, values(vocab * width, 0.09f, 9)));
        ts.put("decoder.bias", tensor(new int[]{vocab}, values(vocab, 0.01f, 10)));
        ts.put("halt_head.weight", tensor(new int[]{1, width}, values(width, 0.02f, 11)));
        ts.put("halt_head.bias", tensor(new int[]{1}, new float[]{0f}));
        root.put("tensors", ts);

        File file = temp.newFile("mobile_diffusion.json");
        Files.write(file.toPath(), root.toString().getBytes(StandardCharsets.UTF_8));
        return file;
    }

    @Test public void learnsFromTextAndPersistsRewardAndPunishment() throws Exception {
        File bundle = makeBundle();
        File state = new File(temp.getRoot(), "learning-state.json");
        SparseDiffusionMobileModel model = new SparseDiffusionMobileModel();
        model.load(bundle);
        assertTrue(model.isLoaded());
        float[] unicodeLogits = model.debugLogits("a🙂");
        assertEquals(4, unicodeLogits.length);
        for (float value : unicodeLogits) assertTrue("logits must be finite", Float.isFinite(value));

        int examples = model.learnFromText("abab", state);
        assertEquals(3, examples);
        assertEquals(1L, model.getLearningUpdates());
        assertTrue(state.isFile());

        JSONObject base = new JSONObject(new String(Files.readAllBytes(bundle.toPath()), StandardCharsets.UTF_8));
        JSONArray originalWeights = base.getJSONObject("tensors").getJSONObject("decoder.weight").getJSONArray("values");
        JSONObject saved = new JSONObject(new String(Files.readAllBytes(state.toPath()), StandardCharsets.UTF_8));
        JSONArray learnedWeights = saved.getJSONArray("decoder_weight");
        boolean changed = false;
        for (int i = 0; i < originalWeights.length(); i++) {
            if (Math.abs(originalWeights.getDouble(i) - learnedWeights.getDouble(i)) > 1e-8) {
                changed = true;
                break;
            }
        }
        assertTrue("local text training must change decoder weights", changed);

        model.generate("a", 4);
        assertTrue(model.canGiveFeedback());
        model.applyFeedback(1f, state);
        assertEquals(2L, model.getLearningUpdates());
        model.generate("b", 4);
        model.applyFeedback(-1f, state);
        assertEquals(3L, model.getLearningUpdates());

        SparseDiffusionMobileModel restored = new SparseDiffusionMobileModel();
        restored.load(bundle);
        restored.loadLearningState(state);
        assertEquals("learned parameters must survive reload", 3L, restored.getLearningUpdates());
        restored.resetLearningState(state);
        assertEquals("reset must clear the learning counter", 0L, restored.getLearningUpdates());
        assertFalse("reset must remove the persisted adaptation", state.exists());
    }

    @Test public void diffusionTraceAndResidentStateSurviveCheckpoint() throws Exception {
        File bundle = makeBundle();
        File runtime = new File(temp.getRoot(), "neuron-residual.bin");
        SparseDiffusionMobileModel model = new SparseDiffusionMobileModel();
        model.load(bundle);
        model.debugLogits("a🙂");
        String trace = model.getLastDiffusionTraceText();
        assertTrue("trace should report actual diffusion steps", trace.contains("扩散步 1"));
        assertTrue("trace should include state-change measurement", trace.contains("状态平均变化"));
        model.saveRuntimeState(runtime);
        assertTrue("resident state checkpoint should exist", runtime.isFile());
        assertTrue("checkpoint should stay compact", runtime.length() < 2_000_000L);

        SparseDiffusionMobileModel restored = new SparseDiffusionMobileModel();
        restored.load(bundle);
        assertTrue("matching base model should restore residual state", restored.loadRuntimeState(runtime));
        restored.debugLogits("b");
        assertTrue(restored.getLastDiffusionTraceText().contains("停止概率"));
    }
}
