package bslsjdk.ornithnpu;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Debug;
import android.view.View;
import android.view.WindowInsets;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.OutputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/** On-device recurrent character prediction experiment with bounded memory checks. */
public final class SuperNeuronLabActivity extends Activity {
    private static final int TOTAL_STEPS = 10000;
    private static final int REQUEST_EXPORT_CHECKPOINT = 5101;
    private static final int REQUEST_IMPORT_DISTILLATION = 5102;
    private static final int CHUNK_STEPS = 1000;
    private static final long STOP_PSS_BYTES = 2560L * 1024L * 1024L;
    private static final String STORY =
            "从前有一只小狐狸，名字叫团团。一天清晨，团团在森林边发现一颗发光的种子。" +
            "它没有把种子带回家，而是先去问老橡树。老橡树说，种子需要阳光、清水和耐心。" +
            "团团每天给种子浇一点水，还把附近的石头轻轻搬开。几天后，嫩芽钻出了泥土。" +
            "小兔子和小鸟都来帮忙，大家轮流照看它。下雨时，小动物们用树叶挡住积水；" +
            "刮风时，它们在旁边插上细树枝。嫩芽慢慢长成小树，春天开出许多金色的小花。" +
            "团团明白，分享与耐心能让森林更美好。";

    private TextView output;
    private Button runButton;
    private Button batchButton;
    private Button distillButton;
    private volatile boolean running;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(12), dp(16), dp(12));
        root.setBackgroundColor(0xFFF3F5F8);
        getWindow().setDecorFitsSystemWindows(false);
        root.setOnApplyWindowInsetsListener((view, insets) -> {
            android.graphics.Insets bars = insets.getInsets(
                    WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
            android.graphics.Insets ime = insets.getInsets(WindowInsets.Type.ime());
            view.setPadding(bars.left + dp(16), bars.top + dp(12), bars.right + dp(16),
                    Math.max(bars.bottom, ime.bottom) + dp(12));
            return insets;
        });

        TextView heading = new TextView(this);
        heading.setText("中文学习实验");
        heading.setTextSize(21);
        heading.setTextColor(0xFF172033);
        heading.setPadding(0, 0, 0, dp(8));
        root.addView(heading);

        TextView details = new TextView(this);
        details.setText("两种方式：直接训练，或导入电脑/Kaggle生成的老师知识文件做蒸馏。手机只运行小学生网络，训练结果可导出到下载文件夹。");
        details.setTextSize(14);
        details.setTextColor(0xFF374151);
        root.addView(details);

        runButton = new Button(this);
        runButton.setText("开始 10,000 步训练");
        root.addView(runButton);
        runButton.setOnClickListener(v -> startExperiment());

        batchButton = new Button(this);
        batchButton.setText("批量计算速度测试（不训练）");
        batchButton.setVisibility(View.GONE);
        root.addView(batchButton);
        batchButton.setOnClickListener(v -> startBatchBenchmark());
        Button advancedToggle = new Button(this);
        advancedToggle.setText("高级工具  ▾");
        root.addView(advancedToggle);
        advancedToggle.setOnClickListener(v -> {
            boolean show = batchButton.getVisibility() != View.VISIBLE;
            batchButton.setVisibility(show ? View.VISIBLE : View.GONE);
            advancedToggle.setText(show ? "收起高级工具  ▴" : "高级工具  ▾");
        });

        distillButton = new Button(this);
        distillButton.setText("知识蒸馏：导入老师知识文件");
        root.addView(distillButton);
        distillButton.setOnClickListener(v -> chooseDistillationFile());

        Button exportButton = new Button(this);
        exportButton.setText("导出训练检查点到手机文件夹");
        root.addView(exportButton);
        exportButton.setOnClickListener(v -> exportCheckpoint());

        ScrollView scroll = new ScrollView(this);
        output = new TextView(this);
        output.setTextSize(13);
        output.setTextColor(0xFF111827);
        output.setTextIsSelectable(true);
        output.setPadding(0, dp(12), 0, dp(12));
        output.setText("等待运行。训练在后台线程执行，界面每 1000 步更新一次。\n\n故事样本：\n" + STORY);
        scroll.addView(output);
        root.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));
        setContentView(root);
    }

    private void startExperiment() {
        if (running) return;
        running = true;
        runButton.setEnabled(false);
        batchButton.setEnabled(false);
        distillButton.setEnabled(false);
        output.setText("正在初始化 tokenizer 与模型……\n");
        Thread worker = new Thread(() -> runTraining(), "aimeng-character-trainer");
        worker.start();
    }

    private void runTraining() {
        SuperNeuronLanguageTrainer trainer = null;
        try {
            CharacterTokenizer tokenizer = new CharacterTokenizer(STORY);
            int[] tokens = tokenizer.encode(STORY);
            if (tokens.length < 100) throw new IllegalStateException("故事字符数不足 100");
            trainer = new SuperNeuronLanguageTrainer(CharacterTokenizer.VOCABULARY_SIZE, 12, 24, 20261010L);
            double initialLoss = trainer.evaluateSequence(tokens);
            StringBuilder report = new StringBuilder();
            report.append("训练路径：CPU Java 顺序参考实现；批量推理有独立 HTP 尝试路径，当前训练器尚未使用它。\n")
                    .append("故事字符数：").append(tokens.length)
                    .append("；已知字符：").append(tokenizer.getKnownCharacterCount())
                    .append("；输出词表：").append(tokenizer.getVocabularySize()).append("\n")
                    .append("参数量：").append(trainer.getParameterCount())
                    .append("；模型估算存储：").append(trainer.estimatedStorageBytes() / 1024).append(" KiB\n")
                    .append("初始验证交叉熵：").append(fmt(initialLoss)).append("\n")
                    .append("步数,平均训练损失,验证交叉熵,PSS_MiB,RSS_MiB\n");
            publish(report.toString());

            int completed = 0;
            for (int chunk = 0; chunk < TOTAL_STEPS / CHUNK_STEPS; chunk++) {
                long pss = pssBytes();
                if (pss >= STOP_PSS_BYTES) {
                    saveCheckpoint(trainer);
                    report.append("停止：PSS 已达到 2.5 GiB 阈值，已尝试保存检查点。\n");
                    break;
                }
                double chunkLoss = trainer.trainSteps(tokens, CHUNK_STEPS, 0.012);
                completed += CHUNK_STEPS;
                double validationLoss = trainer.evaluateSequence(tokens);
                pss = pssBytes();
                long rss = rssBytes();
                report.append(completed).append(',').append(fmt(chunkLoss)).append(',')
                        .append(fmt(validationLoss)).append(',')
                        .append(fmtMiB(pss)).append(',').append(fmtMiB(rss)).append('\n');
                saveCheckpoint(trainer);
                publish(report.toString());
                if (pss >= STOP_PSS_BYTES) {
                    report.append("安全停止：PSS 达到 2.5 GiB，检查点已保存。\n");
                    break;
                }
            }

            trainer.resetContext();
            int prefixLength = Math.min(24, tokens.length - 1);
            int generatedCount = Math.min(100, tokens.length / 2);
            int[] generated = new int[prefixLength + generatedCount];
            System.arraycopy(tokens, 0, generated, 0, prefixLength);
            int previous = generated[prefixLength - 1];
            for (int i = 0; i < prefixLength - 1; i++) trainer.predictNextToken(tokens[i]);
            for (int i = 0; i < generatedCount; i++) {
                int next = trainer.predictNextToken(previous);
                if (next < 0 || next >= CharacterTokenizer.VOCABULARY_SIZE)
                    throw new IllegalStateException("预测 Token ID 越界：" + next);
                generated[prefixLength + i] = next;
                previous = next;
            }
            report.append("\n最终验证交叉熵：").append(fmt(trainer.evaluateSequence(tokens)))
                    .append("\n训练目标步数：").append(trainer.getTrainedTokenTargets())
                    .append("\n生成文本（前 ").append(prefixLength).append(" 个字符为提示词）：\n")
                    .append(tokenizer.decode(generated)).append("\n")
                    .append("\n检查点已自动保存在应用内部空间。其他应用不能直接读取这个路径；点页面上方“导出训练检查点到手机文件夹”，选择“下载”或其他文件夹即可拿到 JSON 文件。")
                    .append("\n注意：只有验证交叉熵下降，且生成文本更接近可读中文，才算初步改善；单次短故事实验不证明通用语言能力。");
            publish(report.toString());
        } catch (Throwable error) {
            String message = "实验失败：" + error.getClass().getSimpleName() + ": " + error.getMessage();
            if (trainer != null) {
                try { saveCheckpoint(trainer); message += "\n已尝试保存当前检查点。"; }
                catch (Throwable ignored) { message += "\n检查点保存也失败。"; }
            }
            publish(message);
        } finally {
            runOnUiThread(() -> {
                running = false;
                runButton.setEnabled(true);
                batchButton.setEnabled(true);
                distillButton.setEnabled(true);
                runButton.setText("再次运行 10,000 步训练");
            });
        }
    }

    private void startBatchBenchmark() {
        if (running) return;
        running = true;
        runButton.setEnabled(false);
        batchButton.setEnabled(false);
        distillButton.setEnabled(false);
        output.setText("正在检查 NPU 并运行批量前向对照测试……\n");
        new Thread(this::runBatchBenchmark, "aimeng-batch-forward-benchmark").start();
    }

    private void runBatchBenchmark() {
        try {
            long pssBefore = pssBytes();
            if (pssBefore >= STOP_PSS_BYTES) {
                publish("测试取消：当前 PSS 已达到 2.5 GiB 安全阈值。\nPSS_MiB=" + fmtMiB(pssBefore));
                return;
            }

            boolean npuReady = NpuRuntime.init(getApplicationContext());
            String npuStatus = npuReady ? NpuRuntime.status() : NpuRuntime.getLastError();
            final int batchSize = 16;
            final int inputSize = 128;
            final int hiddenSize = 128;
            final int outputSize = 200;
            SuperNeuronV2 batchModel = new SuperNeuronV2(inputSize, hiddenSize, outputSize, 20261010L);
            SuperNeuronV2 cpuModel = new SuperNeuronV2(inputSize, hiddenSize, outputSize, 20261010L);
            double[][] inputs = new double[batchSize][inputSize];
            for (int row = 0; row < batchSize; row++)
                for (int p = 0; p < inputSize; p++)
                    inputs[row][p] = Math.sin((row + 1) * (p + 3) * 0.013);

            double[][] batchOutput = null;
            double[][] cpuOutput = null;
            double batchMs = 0.0;
            double cpuMs = 0.0;
            final int repeats = 3;
            for (int repeat = 0; repeat < repeats; repeat++) {
                double[][] states = new double[batchSize][hiddenSize];
                long start = System.nanoTime();
                batchOutput = batchModel.forwardBatch(inputs, states);
                batchMs += (System.nanoTime() - start) / 1_000_000.0;
            }
            for (int repeat = 0; repeat < repeats; repeat++) {
                cpuOutput = new double[batchSize][outputSize];
                long start = System.nanoTime();
                for (int row = 0; row < batchSize; row++) {
                    cpuModel.resetState();
                    cpuOutput[row] = cpuModel.forward(inputs[row]);
                }
                cpuMs += (System.nanoTime() - start) / 1_000_000.0;
            }

            double maxDelta = 0.0;
            for (int row = 0; row < batchSize; row++)
                for (int o = 0; o < outputSize; o++)
                    maxDelta = Math.max(maxDelta, Math.abs(batchOutput[row][o] - cpuOutput[row][o]));
            long pssAfter = pssBytes();
            long rssAfter = rssBytes();
            String report = "批量前向对照测试\n"
                    + "NPU 初始化：" + (npuReady ? "成功" : "失败，已回退 CPU") + "\n"
                    + "NPU 状态：" + npuStatus + "\n"
                    + "模型形状：batch=" + batchSize + ", input=" + inputSize
                    + ", hidden=" + hiddenSize + ", output=" + outputSize + "\n"
                    + "实际批量后端：" + batchModel.getLastBatchBackend() + "\n"
                    + "批量平均耗时：" + fmt(batchMs / repeats) + " ms\n"
                    + "CPU 逐条平均耗时：" + fmt(cpuMs / repeats) + " ms\n"
                    + "最大概率差：" + fmt(maxDelta) + "\n"
                    + "PSS： " + fmtMiB(pssBefore) + " -> " + fmtMiB(pssAfter) + " MiB\n"
                    + "RSS： " + fmtMiB(rssAfter) + " MiB\n"
                    + "注意：这是 16 路独立状态的合成前向基准，不代表语言训练加速；只有实际后端、误差和耗时数据同时合格，才考虑扩大接入范围。";
            publish(report);
        } catch (Throwable error) {
            publish("批量前向测试失败：" + error.getClass().getSimpleName() + ": " + error.getMessage());
        } finally {
            runOnUiThread(() -> {
                running = false;
                runButton.setEnabled(true);
                batchButton.setEnabled(true);
                distillButton.setEnabled(true);
            });
        }
    }

    private static final class DistillExample {
        final int[] context;
        final double[] probabilities;
        DistillExample(int[] context, double[] probabilities) {
            this.context = context;
            this.probabilities = probabilities;
        }
    }

    private void chooseDistillationFile() {
        if (running) return;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        try {
            startActivityForResult(intent, REQUEST_IMPORT_DISTILLATION);
        } catch (Throwable error) {
            Toast.makeText(this, "无法打开文件选择器：" + error.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void runDistillation(Uri uri) {
        running = true;
        runOnUiThread(() -> {
            runButton.setEnabled(false);
            batchButton.setEnabled(false);
            distillButton.setEnabled(false);
            output.setText("正在读取老师知识数据……\n");
        });
        new Thread(() -> {
            try {
                CharacterTokenizer tokenizer = new CharacterTokenizer(STORY);
                List<DistillExample> train = new ArrayList<>();
                List<DistillExample> eval = new ArrayList<>();
                try (java.io.InputStream stream = getContentResolver().openInputStream(uri);
                     BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
                    if (reader == null) throw new java.io.IOException("无法读取所选文件");
                    String first = reader.readLine();
                    if (first == null) throw new java.io.IOException("文件为空");
                    JSONObject meta = new JSONObject(first);
                    if (!"aimeng-character-distillation/v1".equals(meta.optString("format")))
                        throw new java.io.IOException("文件格式不正确。请使用 tools/generate_aimeng_distillation.py 生成的 JSONL。");
                    String line;
                    int lineNo = 1;
                    while ((line = reader.readLine()) != null) {
                        lineNo++;
                        if (line.trim().isEmpty()) continue;
                        JSONObject row = new JSONObject(line);
                        if (!"example".equals(row.optString("type"))) continue;
                        String contextText = row.optString("context", "");
                        int[] context = tokenizer.encode(contextText);
                        if (context.length == 0) continue;
                        double[] probs = new double[CharacterTokenizer.VOCABULARY_SIZE];
                        JSONObject source = row.optJSONObject("target_probs");
                        if (source == null) throw new java.io.IOException("第 " + lineNo + " 行缺少 target_probs");
                        java.util.Iterator<String> keys = source.keys();
                        while (keys.hasNext()) {
                            String character = keys.next();
                            int[] token = tokenizer.encode(character);
                            if (token.length != 1 || token[0] == 0) continue;
                            double p = source.optDouble(character, 0.0);
                            if (Double.isFinite(p) && p > 0.0) probs[token[0]] += p;
                        }
                        double sum = 0.0;
                        for (double p : probs) sum += p;
                        if (sum <= 1.0e-9) continue;
                        for (int i = 0; i < probs.length; i++) probs[i] /= sum;
                        DistillExample example = new DistillExample(context, probs);
                        if ("eval".equals(row.optString("split"))) eval.add(example);
                        else train.add(example);
                        if (train.size() + eval.size() > 50000)
                            throw new java.io.IOException("样本超过 50,000 条安全上限");
                    }
                }
                if (train.size() < 10 || eval.isEmpty())
                    throw new java.io.IOException("有效训练样本或验证样本不足。请生成同时包含 train 和 eval 的数据集。");
                SuperNeuronLanguageTrainer trainer = new SuperNeuronLanguageTrainer(
                        CharacterTokenizer.VOCABULARY_SIZE, 12, 24, 20261011L);
                double initialKl = evaluateDistillation(trainer, eval);
                StringBuilder report = new StringBuilder();
                report.append("知识蒸馏实验\n")
                        .append("老师：").append("离线生成的数据集").append("\n")
                        .append("训练样本：").append(train.size()).append("；独立验证样本：").append(eval.size()).append("\n")
                        .append("学生参数量：").append(trainer.getParameterCount()).append("；估算存储：")
                        .append(trainer.estimatedStorageBytes() / 1024).append(" KiB\n")
                        .append("初始验证 KL：").append(fmt(initialKl)).append("\n")
                        .append("更新数,平均训练 KL,验证 KL,老师 Top-3 覆盖率\n");
                publish(report.toString());
                double runningLoss = 0.0;
                int chunkCount = 0;
                int completed = 0;
                for (DistillExample example : train) {
                    double ce = trainer.trainSoftTarget(example.context, example.probabilities, 0.008);
                    double entropy = entropy(example.probabilities);
                    runningLoss += Math.max(0.0, ce - entropy);
                    chunkCount++;
                    completed++;
                    if (completed % 250 == 0 || completed == train.size()) {
                        double trainKl = runningLoss / Math.max(1, chunkCount);
                        double validationKl = evaluateDistillation(trainer, eval);
                        double top3 = evaluateTop3Coverage(trainer, eval);
                        report.append(completed).append(',')
                                .append(fmt(trainKl)).append(',')
                                .append(fmt(validationKl)).append(',')
                                .append(fmt(top3 * 100.0)).append("%\n");
                        saveCheckpoint(trainer);
                        publish(report.toString());
                        runningLoss = 0.0;
                        chunkCount = 0;
                        if (pssBytes() >= STOP_PSS_BYTES) {
                            report.append("安全停止：PSS 达到 2.5 GiB，已保存检查点。\n");
                            break;
                        }
                    }
                }
                report.append("\n解释：KL 越低越接近老师在本数据集上的字符分布；Top-3 覆盖率表示老师的前三候选中，有多少被学生自己的前三候选覆盖。")
                        .append("\n注意：本实验只蒸馏老师 tokenizer 中能精确映射到单个已知字符的 token，不能代表完整老师模型能力。")
                        .append("\n检查点已保存到应用内部空间。要拿到文件，请点“导出训练检查点到手机文件夹”。");
                publish(report.toString());
            } catch (Throwable error) {
                publish("知识蒸馏失败：" + error.getClass().getSimpleName() + ": " + error.getMessage());
            } finally {
                runOnUiThread(() -> {
                    running = false;
                    runButton.setEnabled(true);
                    batchButton.setEnabled(true);
                    distillButton.setEnabled(true);
                });
            }
        }, "aimeng-distillation-trainer").start();
    }

    private double evaluateDistillation(SuperNeuronLanguageTrainer trainer, List<DistillExample> examples) {
        double total = 0.0;
        int count = 0;
        for (DistillExample example : examples) {
            double[] predicted = trainer.predictDistribution(example.context);
            double kl = 0.0;
            for (int i = 0; i < predicted.length; i++) {
                double target = example.probabilities[i];
                if (target > 0.0) kl += target * Math.log(target / Math.max(1.0e-12, predicted[i]));
            }
            total += Math.max(0.0, kl);
            count++;
        }
        return total / Math.max(1, count);
    }

    private double evaluateTop3Coverage(SuperNeuronLanguageTrainer trainer, List<DistillExample> examples) {
        double matched = 0.0;
        double total = 0.0;
        for (DistillExample example : examples) {
            double[] predicted = trainer.predictDistribution(example.context);
            int[] studentTop = topIndices(predicted, 3);
            int[] teacherTop = topIndices(example.probabilities, 3);
            for (int teacherId : teacherTop) {
                for (int studentId : studentTop) {
                    if (teacherId == studentId) { matched++; break; }
                }
                total++;
            }
        }
        return matched / Math.max(1.0, total);
    }

    private static int[] topIndices(double[] values, int k) {
        int count = Math.min(k, values.length);
        int[] result = new int[count];
        boolean[] used = new boolean[values.length];
        for (int rank = 0; rank < count; rank++) {
            int best = -1;
            for (int i = 0; i < values.length; i++) {
                if (!used[i] && (best < 0 || values[i] > values[best])) best = i;
            }
            result[rank] = best;
            used[best] = true;
        }
        return result;
    }

    private static double entropy(double[] probabilities) {
        double result = 0.0;
        for (double p : probabilities) if (p > 0.0) result -= p * Math.log(p);
        return result;
    }

    private void exportCheckpoint() {
        File checkpoint = new File(getFilesDir(), "super-neuron-character-checkpoint.json");
        if (!checkpoint.isFile()) {
            Toast.makeText(this, "还没有训练检查点。先运行一次训练。", Toast.LENGTH_LONG).show();
            return;
        }
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/json");
        intent.putExtra(Intent.EXTRA_TITLE, "super-neuron-character-checkpoint.json");
        try {
            startActivityForResult(intent, REQUEST_EXPORT_CHECKPOINT);
        } catch (Throwable error) {
            Toast.makeText(this, "无法打开系统文件保存窗口：" + error.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
        if (requestCode == REQUEST_IMPORT_DISTILLATION) {
            runDistillation(data.getData());
            return;
        }
        if (requestCode != REQUEST_EXPORT_CHECKPOINT) return;
        Uri destination = data.getData();
        File checkpoint = new File(getFilesDir(), "super-neuron-character-checkpoint.json");
        try (FileInputStream input = new FileInputStream(checkpoint);
             OutputStream outputStream = getContentResolver().openOutputStream(destination, "wt")) {
            if (outputStream == null) throw new java.io.IOException("系统没有提供可写入的文件");
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) != -1) outputStream.write(buffer, 0, count);
            outputStream.flush();
            Toast.makeText(this, "检查点已导出。你可以在系统文件管理器中找到它。", Toast.LENGTH_LONG).show();
        } catch (Throwable error) {
            Toast.makeText(this, "导出失败：" + error.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void saveCheckpoint(SuperNeuronLanguageTrainer trainer) throws Exception {
        JSONObject json = trainer.toJson();
        json.put("story", STORY);
        json.put("savedAtStep", trainer.getTrainedTokenTargets());
        File file = new File(getFilesDir(), "super-neuron-character-checkpoint.json");
        File temp = new File(getFilesDir(), "super-neuron-character-checkpoint.tmp");
        try (FileOutputStream stream = new FileOutputStream(temp)) {
            stream.write(json.toString().getBytes(StandardCharsets.UTF_8));
            stream.getFD().sync();
        }
        if (file.exists() && !file.delete()) throw new IllegalStateException("无法替换旧检查点");
        if (!temp.renameTo(file)) throw new IllegalStateException("无法提交检查点文件");
    }

    private long pssBytes() { return Math.max(0L, (long) Debug.getPss() * 1024L); }

    private long rssBytes() {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream("/proc/self/status"), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("VmRSS:")) {
                    String[] parts = line.trim().split("\\s+");
                    return Long.parseLong(parts[1]) * 1024L;
                }
            }
        } catch (Exception ignored) { }
        return -1L;
    }

    private void publish(String text) {
        runOnUiThread(() -> output.setText(text));
    }

    private static String fmt(double value) { return String.format(Locale.US, "%.5f", value); }
    private static String fmtMiB(long bytes) {
        return bytes < 0 ? "unavailable" : String.format(Locale.US, "%.1f", bytes / (1024.0 * 1024.0));
    }
    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }
}
