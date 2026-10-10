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
        heading.setText("AIMENG 超级神经元 · 循环语言实验");
        heading.setTextSize(21);
        heading.setTextColor(0xFF172033);
        heading.setPadding(0, 0, 0, dp(8));
        root.addView(heading);

        TextView details = new TextView(this);
        details.setText("这个实验会用一小段中文故事训练字符预测。训练结果会自动保存；想把模型文件拿到手机其他位置，请使用下面的导出按钮。");
        details.setTextSize(14);
        details.setTextColor(0xFF374151);
        root.addView(details);

        runButton = new Button(this);
        runButton.setText("开始 10,000 步训练");
        root.addView(runButton);
        runButton.setOnClickListener(v -> startExperiment());

        batchButton = new Button(this);
        batchButton.setText("高级测试：批量计算速度（不训练）");
        root.addView(batchButton);
        batchButton.setOnClickListener(v -> startBatchBenchmark());

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
                runButton.setText("再次运行 10,000 步训练");
            });
        }
    }

    private void startBatchBenchmark() {
        if (running) return;
        running = true;
        runButton.setEnabled(false);
        batchButton.setEnabled(false);
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
            });
        }
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
        if (requestCode != REQUEST_EXPORT_CHECKPOINT || resultCode != RESULT_OK
                || data == null || data.getData() == null) return;
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
