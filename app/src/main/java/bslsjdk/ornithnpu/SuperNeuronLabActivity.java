package bslsjdk.ornithnpu;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Debug;
import android.os.Looper;
import android.graphics.Color;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Locale;

/**
 * Isolated smoke-test UI for the super-neuron next-token trainer.
 * Input values are integer token IDs, not natural-language text; no tokenizer
 * is connected here yet. The normal ChatActivity remains the app launcher.
 */
public final class SuperNeuronLabActivity extends Activity {
    private final Handler main = new Handler(Looper.getMainLooper());
    private EditText vocabularyField, embeddingField, hiddenField, epochsField, learningRateField, sequenceField, storyField;
    private TextView resultView;
    private Button trainButton;
    private SuperNeuronLanguageTrainer trainer;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(16), dp(16), dp(20));
        root.setBackgroundColor(Color.rgb(246, 247, 249));
        scroll.addView(root);
        setContentView(scroll);

        TextView title = new TextView(this);
        title.setText("超级神经元 · 训练实验");
        title.setTextColor(Color.rgb(17, 24, 39));
        title.setTextSize(22);
        title.setTypeface(null, 1);
        root.addView(title);

        TextView warning = new TextView(this);
        warning.setText("实验包括字符级故事预测、单个模型内的局部梯度/资格迹，以及模型池精英选择与变异重组。每 1000 步记录 PSS/RSS；PSS 超过 2.5 GiB 时保存检查点并停止。模型估算存储预算 16 MiB，手机进程总内存仍必须低于 4 GiB。");
        warning.setTextColor(Color.rgb(153, 72, 16));
        warning.setTextSize(13);
        warning.setPadding(0, dp(8), 0, dp(12));
        root.addView(warning);

        vocabularyField = addField(root, "词表大小（输出数量）", "8");
        embeddingField = addField(root, "Token embedding 维度", "8");
        hiddenField = addField(root, "隐藏状态维度", "16");
        epochsField = addField(root, "训练轮数", "100");
        learningRateField = addField(root, "学习率（0.001 到 0.05）", "0.02");
        sequenceField = addField(root, "Token ID 序列（空格分隔）", "0 1 0 1 0 1 0 1 0 1 0 1");

        storyField = addMultilineField(root, "字符级儿童故事（原文）",
                "小兔子住在森林边的小木屋里。一天早晨，它发现门口有一颗闪闪发光的种子。小兔把种子种进土里，每天浇水，也和小鸟一起等它发芽。过了几天，嫩绿的小芽探出头来。大风来时，小兔用树枝挡风，下雨时又把积水轻轻排开。后来种子长成一棵小树，结出了甜甜的果子。小兔把果子分给小鸟、刺猬和路过的鹿。大家一起种下更多种子，让森林变得更加茂盛。小兔明白，耐心照料和分享，会让小小的善意慢慢长大。");

        Button storyButton = new Button(this);
        storyButton.setText("字符故事训练：10,000 步 + 内存探针 + 进化");
        storyButton.setAllCaps(false);
        LinearLayout.LayoutParams storyParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(54));
        storyParams.topMargin = dp(8);
        root.addView(storyButton, storyParams);
        storyButton.setOnClickListener(v -> trainCharacterStory(storyButton));

        trainButton = new Button(this);
        trainButton.setText("训练并验证下一 Token 预测");
        trainButton.setAllCaps(false);
        LinearLayout.LayoutParams buttonParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(52));
        buttonParams.topMargin = dp(12);
        root.addView(trainButton, buttonParams);

        Button resetButton = new Button(this);
        resetButton.setText("清空实验结果");
        resetButton.setAllCaps(false);
        root.addView(resetButton, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(48)));

        resultView = new TextView(this);
        resultView.setText("等待实验。输出维度将严格等于词表大小。");
        resultView.setTextColor(Color.rgb(31, 41, 55));
        resultView.setTextSize(14);
        resultView.setPadding(dp(12), dp(12), dp(12), dp(12));
        resultView.setBackgroundColor(Color.WHITE);
        LinearLayout.LayoutParams resultParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        resultParams.topMargin = dp(12);
        root.addView(resultView, resultParams);

        trainButton.setOnClickListener(v -> train());
        resetButton.setOnClickListener(v -> {
            trainer = null;
            resultView.setText("实验结果已清空。下一次训练会创建新的参数实例。");
        });
    }

    private EditText addField(LinearLayout root, String labelText, String defaultValue) {
        TextView label = new TextView(this);
        label.setText(labelText);
        label.setTextColor(Color.rgb(55, 65, 81));
        label.setTextSize(13);
        LinearLayout.LayoutParams labelParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        labelParams.topMargin = dp(8);
        root.addView(label, labelParams);

        EditText field = new EditText(this);
        field.setSingleLine(true);
        field.setText(defaultValue);
        field.setTextSize(15);
        field.setPadding(dp(10), 0, dp(10), 0);
        field.setBackgroundColor(Color.WHITE);
        root.addView(field, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(48)));
        return field;
    }

    private EditText addMultilineField(LinearLayout root, String labelText, String defaultValue) {
        TextView label = new TextView(this);
        label.setText(labelText);
        label.setTextColor(Color.rgb(55, 65, 81));
        label.setTextSize(13);
        root.addView(label, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        EditText field = new EditText(this);
        field.setText(defaultValue);
        field.setTextSize(14);
        field.setGravity(Gravity.TOP | Gravity.START);
        field.setMinLines(5);
        field.setMaxLines(10);
        field.setPadding(dp(10), dp(8), dp(10), dp(8));
        field.setBackgroundColor(Color.WHITE);
        root.addView(field, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        return field;
    }


    private void trainCharacterStory(Button storyButton) {
        final String story = storyField.getText().toString().trim();
        final CharacterTokenizer tokenizer;
        final int[] allTokens;
        try {
            int charCount = story.codePointCount(0, story.length());
            if (charCount < 40) throw new IllegalArgumentException("故事至少需要 40 个 Unicode 字符");
            if (charCount > 1000) throw new IllegalArgumentException("本次基准故事最多 1000 个字符");
            tokenizer = CharacterTokenizer.fit(story);
            allTokens = tokenizer.encode(story);
            if (tokenizer.getVocabularySize() > 256)
                throw new IllegalArgumentException("字符词表超过 256，请缩短或简化故事");
        } catch (Exception e) {
            Toast.makeText(this, e.getMessage() == null ? "故事配置无效" : e.getMessage(),
                    Toast.LENGTH_LONG).show();
            return;
        }

        trainButton.setEnabled(false);
        storyButton.setEnabled(false);
        resultView.setText("正在运行字符级故事基准：4 个个体、总计 10,000 次目标更新。每 1,000 步记录 PSS/RSS；触及 2.5 GiB PSS 红线会保存检查点并安全停止。");
        new Thread(() -> {
            JSONArray samples = new JSONArray();
            JSONObject report = new JSONObject();
            boolean[] safetyStopped = {false};
            String[] stopReason = {"completed"};
            try {
                int split = Math.max(2, Math.min(allTokens.length - 2, (int) (allTokens.length * 0.8)));
                int[] training = Arrays.copyOfRange(allTokens, 0, split);
                int[] validation = Arrays.copyOfRange(allTokens, split - 1, allTokens.length);
                SuperNeuronEvolutionPool pool = new SuperNeuronEvolutionPool(
                        4, tokenizer.getVocabularySize(), 8, 16, 20261010L);
                report.put("format", "aimeng-character-story-memory-report/v1");
                report.put("status", "running");
                report.put("storyCodePoints", story.codePointCount(0, story.length()));
                report.put("vocabularySizeAndOutputCount", tokenizer.getVocabularySize());
                report.put("targetTrainingSteps", 10000);
                report.put("populationSize", pool.getPopulationSize());
                report.put("memoryLimitPssKiB", 2621440L);
                report.put("baselinePssKiB", Debug.getPss());
                report.put("baselineRssKiB", readRssKb());
                report.put("samples", samples);
                writeJsonAtomically("super_neuron_memory_report.json", report);

                double[] lossWindow = {0.0};
                long[] windowCount = {0};
                double trainLoss = pool.trainAndEvolve(training, validation, 10000L, 0.015,
                        (completed, stepLoss) -> {
                            lossWindow[0] += stepLoss;
                            windowCount[0]++;
                            if (completed == 0 || completed % 1000L != 0L) return true;
                            double meanWindowLoss = lossWindow[0] / Math.max(1L, windowCount[0]);
                            lossWindow[0] = 0.0;
                            windowCount[0] = 0;
                            long pssKb = Debug.getPss();
                            long rssKb = readRssKb();
                            try {
                                JSONObject sample = new JSONObject();
                                sample.put("step", completed);
                                sample.put("windowMeanCrossEntropy", meanWindowLoss);
                                sample.put("pssKiB", pssKb);
                                sample.put("rssKiB", rssKb);
                                sample.put("timestampEpochMs", System.currentTimeMillis());
                                samples.put(sample);
                                report.put("lastCompletedStep", completed);
                                report.put("samples", samples);
                                writeJsonAtomically("super_neuron_memory_report.json", report);

                                SuperNeuronLanguageTrainer current = pool.getCurrentTrainer();
                                if (current != null) {
                                    JSONObject checkpoint = new JSONObject();
                                    checkpoint.put("format", "aimeng-super-neuron-safe-checkpoint/v1");
                                    checkpoint.put("completedGlobalSteps", completed);
                                    checkpoint.put("reason", pssKb > 2621440L ? "PSS_LIMIT_EXCEEDED" : "periodic");
                                    checkpoint.put("tokenizer", tokenizer.toJson());
                                    checkpoint.put("trainer", current.toJson());
                                    checkpoint.put("memoryReport", report);
                                    writeJsonAtomically("super_neuron_checkpoint.json", checkpoint);
                                }
                                if (pssKb > 2621440L) {
                                    safetyStopped[0] = true;
                                    stopReason[0] = "PSS 超过 2.5 GiB，已保存检查点并停止训练";
                                    report.put("status", "stopped_memory_limit");
                                    report.put("stopReason", stopReason[0]);
                                    writeJsonAtomically("super_neuron_memory_report.json", report);
                                    return false;
                                }
                                return true;
                            } catch (Exception e) {
                                safetyStopped[0] = true;
                                stopReason[0] = "报告或检查点写入失败，已请求停止：" + e.getClass().getSimpleName();
                                return false;
                            }
                        });

                SuperNeuronLanguageTrainer best = pool.getBestTrainer();
                double initialValidationLoss = pool.getInitialBestValidationLoss();
                double validationLoss = pool.wasStoppedEarly() ? Double.NaN : pool.getBestValidationLoss();
                int lastId = validation[validation.length - 1];
                best.resetContext();
                int prediction = best.predictNextToken(lastId);
                long finalPss = Debug.getPss();
                long finalRss = readRssKb();
                report.put("status", safetyStopped[0] || pool.wasStoppedEarly() ? "stopped" : "completed");
                report.put("stopReason", stopReason[0]);
                report.put("completedSteps", pool.getLastTrainingSteps());
                report.put("generationAfterEvolution", pool.getGeneration());
                report.put("meanTrainingCrossEntropy", trainLoss);
                report.put("initialHeldOutValidationCrossEntropy", initialValidationLoss);
                report.put("heldOutValidationCrossEntropy",
                        Double.isFinite(validationLoss) ? validationLoss : JSONObject.NULL);
                report.put("heldOutLossDelta", Double.isFinite(validationLoss)
                        ? validationLoss - initialValidationLoss : JSONObject.NULL);
                report.put("predictedTokenId", prediction);
                report.put("predictedCharacter", tokenizer.tokenAt(prediction));
                report.put("finalPssKiB", finalPss);
                report.put("finalRssKiB", finalRss);
                report.put("samples", samples);
                writeJsonAtomically("super_neuron_memory_report.json", report);

                JSONObject checkpoint = new JSONObject();
                checkpoint.put("format", "aimeng-super-neuron-safe-checkpoint/v1");
                checkpoint.put("completedGlobalSteps", pool.getLastTrainingSteps());
                checkpoint.put("reason", stopReason[0]);
                checkpoint.put("tokenizer", tokenizer.toJson());
                checkpoint.put("trainer", best.toJson());
                checkpoint.put("memoryReport", report);
                writeJsonAtomically("super_neuron_checkpoint.json", checkpoint);

                File reportDir = getExternalFilesDir(null);
                String reportDirectory = reportDir == null ? getFilesDir().getAbsolutePath() : reportDir.getAbsolutePath();
                StringBuilder curveText = new StringBuilder(
                        "\\n\\n每 1000 步的损失/内存曲线：\\n步数 | 平均交叉熵 | PSS MiB | RSS MiB\\n");
                for (int i = 0; i < samples.length(); i++) {
                    JSONObject sample = samples.getJSONObject(i);
                    curveText.append(String.format(Locale.US, "%d | %.5f | %.1f | %.1f\\n",
                            sample.getLong("step"),
                            sample.getDouble("windowMeanCrossEntropy"),
                            sample.getLong("pssKiB") / 1024.0,
                            sample.getLong("rssKiB") / 1024.0));
                }
                String result = String.format(Locale.US,
                        "字符级故事实验%s\\n\\n字符数：%d\\n字符词表 / 固定输出数量：%d\\n"
                                + "输出 ID：%d（%s）\n模型池：4 个个体\n实际训练步数：%d / 10,000\n"
                                + "训练平均交叉熵：%.5f\n独立尾段验证损失（前 -> 后）：%.5f -> %s\n"
                                + "进化代数：%d\n最终 PSS：%.1f MiB\n最终 RSS：%.1f MiB\n"
                                + "内存采样点：%d\n报告目录：%s\n"
                                + "文件：super_neuron_memory_report.json / super_neuron_checkpoint.json\n\n"
                                + "注意：单故事拟合不等于语言泛化；内存曲线来自本次 Android 运行。",
                        safetyStopped[0] || pool.wasStoppedEarly() ? "（安全停止）" : "完成",
                        allTokens.length, tokenizer.getVocabularySize(), prediction,
                        tokenizer.tokenAt(prediction), pool.getLastTrainingSteps(), trainLoss,
                        initialValidationLoss,
                        Double.isFinite(validationLoss) ? String.format(Locale.US, "%.5f", validationLoss) : "未完成",
                        pool.getGeneration(), finalPss / 1024.0, finalRss / 1024.0, samples.length(), reportDirectory);
                result += curveText.toString();
                main.post(() -> {
                    trainer = best;
                    resultView.setText(result);
                    trainButton.setEnabled(true);
                    storyButton.setEnabled(true);
                });
            } catch (Throwable e) {
                try {
                    report.put("status", "failed");
                    report.put("error", e.getClass().getSimpleName() + ": " + String.valueOf(e.getMessage()));
                    report.put("samples", samples);
                    writeJsonAtomically("super_neuron_memory_report.json", report);
                } catch (Exception ignored) { }
                main.post(() -> {
                    resultView.setText("字符故事训练失败：" + e.getClass().getSimpleName()
                            + ": " + String.valueOf(e.getMessage()));
                    trainButton.setEnabled(true);
                    storyButton.setEnabled(true);
                });
            }
        }, "super-neuron-character-story").start();
    }

    private long readRssKb() {
        try (BufferedReader reader = new BufferedReader(new FileReader("/proc/self/status"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("VmRSS:")) {
                    String[] parts = line.trim().split("\\s+");
                    return parts.length >= 2 ? Long.parseLong(parts[1]) : -1L;
                }
            }
        } catch (Exception ignored) { }
        return -1L;
    }

    private void writeJsonAtomically(String name, JSONObject json) throws IOException, JSONException {
        File directory = getExternalFilesDir(null);
        if (directory == null) directory = getFilesDir();
        if (!directory.exists() && !directory.mkdirs()) throw new IOException("cannot create report directory");
        File target = new File(directory, name);
        File temporary = new File(directory, name + ".tmp");
        byte[] bytes = json.toString(2).getBytes(StandardCharsets.UTF_8);
        try (FileOutputStream output = new FileOutputStream(temporary)) {
            output.write(bytes);
            output.flush();
            output.getFD().sync();
        }
        if (target.exists() && !target.delete()) throw new IOException("cannot replace " + name);
        if (!temporary.renameTo(target)) throw new IOException("cannot atomically rename " + name);
    }

    private void train() {
        final int vocab, embedding, hidden, epochs;
        final double lr;
        final int[] tokens;
        try {
            vocab = Integer.parseInt(vocabularyField.getText().toString().trim());
            embedding = Integer.parseInt(embeddingField.getText().toString().trim());
            hidden = Integer.parseInt(hiddenField.getText().toString().trim());
            epochs = Integer.parseInt(epochsField.getText().toString().trim());
            lr = Double.parseDouble(learningRateField.getText().toString().trim());
            tokens = parseTokens(sequenceField.getText().toString(), vocab);
            if ((long) (tokens.length - 1) * epochs > 100_000L)
                throw new IllegalArgumentException("实验页面每次最多训练 100,000 个 token 目标，避免长时间占用手机 CPU");
        } catch (Exception e) {
            Toast.makeText(this, e.getMessage() == null ? "配置无效" : e.getMessage(),
                    Toast.LENGTH_LONG).show();
            return;
        }

        trainButton.setEnabled(false);
        resultView.setText("训练中… 正在计算训练前损失，请勿连续点击。");
        new Thread(() -> {
            try {
                SuperNeuronLanguageTrainer candidate =
                        new SuperNeuronLanguageTrainer(vocab, embedding, hidden, 20261010L);
                double before = candidate.evaluateSequence(tokens);
                double trainLoss = candidate.trainSequence(tokens, epochs, lr);
                double after = candidate.evaluateSequence(tokens);
                candidate.resetContext();
                int prediction = candidate.predictNextToken(tokens[tokens.length - 1]);
                long bytes = candidate.estimatedStorageBytes();
                String report = String.format(Locale.US,
                        "实验完成\n\n词表大小 / 固定输出数量：%d\n实际预测 ID：%d\n"
                                + "序列长度：%d\n训练轮数：%d\n训练 token 目标数：%d\n"
                                + "训练时平均交叉熵：%.6f\n训练前序列损失：%.6f\n训练后序列损失：%.6f\n"
                                + "损失变化：%.6f\n参数标量数：%d\n估算模型与工作区：%.2f MiB\n\n"
                                + "注意：这是同一小序列上的拟合测试，不代表泛化或自然语言能力。"
                                + "当前没有 tokenizer，也没有独立验证集。",
                        candidate.getVocabularySize(), prediction, tokens.length, epochs,
                        candidate.getTrainedTokenTargets(), trainLoss, before, after, after - before,
                        candidate.getParameterCount(), bytes / (1024.0 * 1024.0));
                main.post(() -> {
                    trainer = candidate;
                    resultView.setText(report);
                    trainButton.setEnabled(true);
                });
            } catch (Throwable e) {
                main.post(() -> {
                    resultView.setText("训练失败：" + e.getClass().getSimpleName()
                            + ": " + String.valueOf(e.getMessage()));
                    trainButton.setEnabled(true);
                });
            }
        }, "super-neuron-training").start();
    }

    private static int[] parseTokens(String text, int vocabularySize) {
        if (text == null || text.trim().isEmpty())
            throw new IllegalArgumentException("Token ID 序列不能为空");
        String[] parts = text.trim().split("\\s+");
        if (parts.length < 2) throw new IllegalArgumentException("至少输入两个 token ID");
        if (parts.length > 20001) throw new IllegalArgumentException("序列长度不能超过 20001");
        int[] tokens = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                tokens[i] = Integer.parseInt(parts[i]);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Token ID 不是整数：" + parts[i]);
            }
            if (tokens[i] < 0 || tokens[i] >= vocabularySize)
                throw new IllegalArgumentException("Token ID 越界：" + tokens[i]
                        + "，有效范围是 0 到 " + (vocabularySize - 1));
        }
        return tokens;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
