package bslsjdk.ornithnpu;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.ComponentName;
import android.content.ServiceConnection;
import android.net.Uri;
import android.os.Bundle;
import android.os.Debug;
import android.graphics.Color;
import android.text.util.Linkify;
import android.view.View;
import android.widget.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * AIMENG's standalone phone-local neuron runtime.
 * This screen accepts the canonical AIMENG .aimg bundle and legacy mobile_diffusion.json bundles,
 * never Ornith safetensors, MLX packages, or GGUF models.
 */
public final class SparseDiffusionActivity extends Activity {
    private static final int PICK = 7712;
    private static final long MAX_BUNDLE_BYTES = 32L * 1024L * 1024L;
    private SparseDiffusionMobileModel model;
    private AimengNeuronService neuronService;
    private boolean serviceBound;
    private boolean residentRequested;
    private Button residentButton;
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private EditText prompt, count;
    private TextView status, result;
    private Button run, rewardButton, punishButton, learnTextButton, resetButton, pickButton;

    private final ServiceConnection neuronConnection = new ServiceConnection() {
        @Override public void onServiceConnected(ComponentName name, android.os.IBinder binder) {
            AimengNeuronService.LocalBinder local = (AimengNeuronService.LocalBinder) binder;
            neuronService = local.getService();
            model = neuronService.getModel();
            residentRequested = neuronService.isResidentEnabled();
            residentButton.setText(residentRequested ? "停止后台驻留服务" : "启动后台驻留服务");
            pickButton.setEnabled(true);
            restoreSavedModel();
        }
        @Override public void onServiceDisconnected(ComponentName name) {
            serviceBound = false;
            neuronService = null;
            model = null;
            pickButton.setEnabled(false);
            run.setEnabled(false);
        }
    };

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);

        ScrollView page = new ScrollView(this);
        page.setFillViewport(true);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(18);
        root.setPadding(pad, pad, pad, pad);
        page.addView(root);

        TextView title = new TextView(this);
        title.setText("AIMENG 神经元");
        title.setTextSize(26);
        title.setTextColor(Color.rgb(24, 34, 54));
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        root.addView(title);

        TextView subtitle = new TextView(this);
        subtitle.setText("手机本地运行 · 独立模型 · 可持续学习");
        subtitle.setTextSize(14);
        subtitle.setTextColor(Color.rgb(90, 103, 125));
        root.addView(subtitle, spaced());

        status = new TextView(this);
        status.setText("正在检查手机里已导入的 AIMENG 模型…");
        status.setTextSize(14);
        status.setPadding(dp(12), dp(12), dp(12), dp(12));
        status.setBackgroundColor(Color.rgb(239, 244, 250));
        root.addView(status, spaced());

        residentButton = new Button(this);
        residentButton.setText("启动后台驻留服务");
        residentButton.setEnabled(false);
        residentButton.setOnClickListener(v -> toggleResidentService());
        root.addView(residentButton, spaced());

        addSectionTitle(root, "1 · 模型");
        pickButton = new Button(this);
        pickButton.setText("导入 AIMENG 模型包（.aimg / 旧版 .json）");
        pickButton.setOnClickListener(v -> chooseModel());
        root.addView(pickButton, spaced());

        TextView guide = new TextView(this);
        guide.setText("这里不需要 Ornith-1.5-9B，也不接受 .pth / .safetensors。\n\n优先导入训练工作流生成的 mobile_diffusion.aimg。旧版 mobile_diffusion.json 仍兼容。若手里只有 JSON，在 AIMENG 仓库运行：python scripts/pack_mobile_aimg.py --input mobile_diffusion.json --output mobile_diffusion.aimg。\n\n注意：AIMG 只是统一交付格式，不会凭空提升模型能力；当前模型仍需检查生成样本并进行真机验证。");
        guide.setTextSize(13);
        guide.setTextColor(Color.rgb(75, 85, 99));
        guide.setAutoLinkMask(Linkify.WEB_URLS);
        guide.setLinkTextColor(Color.rgb(30, 100, 190));
        guide.setPadding(dp(4), dp(4), dp(4), dp(10));
        root.addView(guide, spaced());

        addSectionTitle(root, "2 · 生成");
        prompt = new EditText(this);
        prompt.setMinLines(2);
        prompt.setGravity(android.view.Gravity.TOP | android.view.Gravity.START);
        prompt.setHint("输入提示词，例如：你好");
        root.addView(prompt, spaced());

        count = new EditText(this);
        count.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        count.setText("40");
        count.setHint("生成字符数（1–200）");
        root.addView(count, spaced());

        run = new Button(this);
        run.setText("在手机本地生成");
        run.setEnabled(false);
        run.setOnClickListener(v -> generate());
        root.addView(run, spaced());

        result = new TextView(this);
        result.setText("生成结果会显示在这里。");
        result.setTextIsSelectable(true);
        result.setTextSize(16);
        result.setTextColor(Color.rgb(25, 35, 50));
        result.setPadding(dp(12), dp(12), dp(12), dp(12));
        result.setMinHeight(dp(100));
        result.setBackgroundColor(Color.rgb(247, 248, 250));
        root.addView(result, spaced());

        addSectionTitle(root, "3 · 学习");
        TextView learningHint = new TextView(this);
        learningHint.setText("可以从输入框中的文字学习，也可以对刚才的生成结果给出反馈。学习参数只保存在本机。");
        learningHint.setTextSize(13);
        learningHint.setTextColor(Color.rgb(90, 103, 125));
        root.addView(learningHint, spaced());

        learnTextButton = new Button(this);
        learnTextButton.setText("从输入文字学习");
        learnTextButton.setEnabled(false);
        learnTextButton.setOnClickListener(v -> learnFromInputText());
        root.addView(learnTextButton, spaced());

        LinearLayout feedback = new LinearLayout(this);
        feedback.setOrientation(LinearLayout.HORIZONTAL);
        rewardButton = new Button(this);
        rewardButton.setText("奖励 +1");
        rewardButton.setEnabled(false);
        rewardButton.setOnClickListener(v -> giveFeedback(1f));
        feedback.addView(rewardButton, new LinearLayout.LayoutParams(0, -2, 1));
        punishButton = new Button(this);
        punishButton.setText("惩罚 -1");
        punishButton.setEnabled(false);
        punishButton.setOnClickListener(v -> giveFeedback(-1f));
        feedback.addView(punishButton, new LinearLayout.LayoutParams(0, -2, 1));
        root.addView(feedback, spaced());

        resetButton = new Button(this);
        resetButton.setText("清除学习记录，恢复基础模型");
        resetButton.setEnabled(false);
        resetButton.setOnClickListener(v -> new AlertDialog.Builder(this)
                .setTitle("恢复基础模型")
                .setMessage("清除本机学习参数和神经元残留状态，但保留导入的模型文件。")
                .setNegativeButton("取消", (dialog, which) -> { })
                .setPositiveButton("恢复", (dialog, which) -> resetLearning())
                .show());
        root.addView(resetButton, spaced());

        setContentView(page);
        pickButton.setEnabled(false);
        Intent serviceIntent = new Intent(this, AimengNeuronService.class);
        serviceBound = bindService(serviceIntent, neuronConnection, BIND_AUTO_CREATE);
    }

    private int dp(int value) {
        return (int) (value * getResources().getDisplayMetrics().density + 0.5f);
    }

    private LinearLayout.LayoutParams spaced() {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, -2);
        p.bottomMargin = dp(8);
        return p;
    }

    private void addSectionTitle(LinearLayout root, String text) {
        TextView label = new TextView(this);
        label.setText(text);
        label.setTextSize(17);
        label.setTypeface(null, android.graphics.Typeface.BOLD);
        label.setTextColor(Color.rgb(40, 55, 78));
        LinearLayout.LayoutParams p = spaced();
        p.topMargin = dp(12);
        root.addView(label, p);
    }

    private File modelFile() {
        return new File(getFilesDir(), "aimeng-mobile-diffusion.aimg");
    }

    private File learningFile() {
        return new File(getFilesDir(), "aimeng-learning-state.json");
    }

    private File runtimeStateFile() {
        return new File(getFilesDir(), "aimeng-neuron-residual.bin");
    }

    private void saveRuntimeStateQuietly() {
        try { if (model != null && model.isLoaded()) model.saveRuntimeState(runtimeStateFile()); }
        catch (Throwable ignored) { }
    }

    private void toggleResidentService() {
        if (neuronService == null || model == null || !model.isLoaded()) {
            status.setText("先导入并加载 AIMENG 模型，再启动后台驻留。");
            return;
        }
        try {
            if (residentRequested || neuronService.isResidentEnabled()) {
                startService(new Intent(this, AimengNeuronService.class).setAction(AimengNeuronService.ACTION_STOP));
                residentRequested = false;
                residentButton.setText("启动后台驻留服务");
                status.setText("已请求停止后台驻留；当前神经元状态已尝试保存。");
            } else {
                startForegroundService(new Intent(this, AimengNeuronService.class).setAction(AimengNeuronService.ACTION_START));
                residentRequested = true;
                residentButton.setText("停止后台驻留服务");
                status.setText("后台驻留启动请求已发送。服务会显示常驻通知，空闲时等待输入，不会持续空转。");
            }
        } catch (Throwable e) {
            status.setText("无法切换后台驻留服务：" + errorText(e));
        }
    }

    private void chooseModel() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("*/*");
        startActivityForResult(i, PICK);
    }

    private void restoreSavedModel() {
        File file = modelFile();
        if (!file.isFile()) file = new File(getFilesDir(), "aimeng-mobile-diffusion.json");
        if (!file.isFile()) {
            status.setText("尚未导入模型。优先选择 AIMENG 导出的 .aimg；旧版 mobile_diffusion.json 也兼容。");
            return;
        }
        final File selectedFile = file;
        setBusy(true, "正在恢复本机 AIMENG 模型和学习记录…");
        worker.execute(() -> {
            try {
                model.load(selectedFile);
                model.loadLearningState(learningFile());
                boolean restoredResidual = model.loadRuntimeState(runtimeStateFile());
                long pssMiB = Debug.getPss() / 1024L;
                runOnUiThread(() -> {
                    status.setText("模型已自动恢复 · 神经元残留状态 " + (restoredResidual ? "已恢复" : "从初始状态启动") + " · 本地学习 " + model.getLearningUpdates()
                            + " 次 · 进程 PSS 约 " + pssMiB + " MiB · " + model.backendStatus());
                    setBusy(false, null);
                    setModelActionsEnabled(true);
                });
            } catch (Throwable e) {
                runOnUiThread(() -> {
                    status.setText("本机模型恢复失败：" + errorText(e) + "。请重新导入有效的 AIMENG .aimg 或旧版 .json 模型包。");
                    setBusy(false, null);
                    setModelActionsEnabled(false);
                });
            }
        });
    }

    @Override protected void onActivityResult(int request, int resultCode, Intent data) {
        super.onActivityResult(request, resultCode, data);
        if (request != PICK || resultCode != RESULT_OK || data == null) return;
        Uri uri = data.getData();
        if (uri == null) return;
        setBusy(true, "正在校验 AIMENG 二进制/旧版模型包…");
        worker.execute(() -> {
            File temp = new File(getFilesDir(), "aimeng-mobile-diffusion.aimg.tmp");
            try {
                try (InputStream in = getContentResolver().openInputStream(uri);
                     OutputStream out = new FileOutputStream(temp)) {
                    if (in == null) throw new IOException("无法打开所选文件");
                    byte[] b = new byte[8192];
                    int n;
                    long total = 0;
                    while ((n = in.read(b)) >= 0) {
                        total += n;
                        if (total > MAX_BUNDLE_BYTES) throw new IOException("模型包超过 32 MiB 手机安全限制");
                        out.write(b, 0, n);
                    }
                }
                // Validate in an isolated candidate first. A malformed import must
                // never unload or partially overwrite the currently working model.
                SparseDiffusionMobileModel candidate = new SparseDiffusionMobileModel();
                candidate.load(temp);
                model.load(temp);
                model.loadLearningState(learningFile());
                model.loadRuntimeState(runtimeStateFile());
                Files.move(temp.toPath(), modelFile().toPath(), StandardCopyOption.REPLACE_EXISTING);
                long pssMiB = Debug.getPss() / 1024L;
                runOnUiThread(() -> {
                    status.setText("AIMENG 模型导入成功 · 本地学习 " + model.getLearningUpdates()
                            + " 次 · 进程 PSS 约 " + pssMiB + " MiB · " + model.backendStatus());
                    setBusy(false, null);
                    setModelActionsEnabled(true);
                    result.setText("模型已就绪。输入提示词后点击“在手机本地生成”。");
                });
            } catch (Throwable e) {
                temp.delete();
                runOnUiThread(() -> {
                    status.setText("导入失败：" + errorText(e)
                            + "。请选 AIMENG 的 .aimg 或旧版 .json 模型包，而不是 .pth / .safetensors。");
                    setBusy(false, null);
                });
            }
        });
    }

    private String errorText(Throwable e) {
        String message = e.getMessage();
        return e.getClass().getSimpleName() + (message == null ? "" : ": " + message);
    }

    private void setBusy(boolean busy, String message) {
        if (message != null) status.setText(message);
        pickButton.setEnabled(!busy);
        boolean loaded = model != null && model.isLoaded();
        run.setEnabled(!busy && loaded);
        learnTextButton.setEnabled(!busy && loaded);
        resetButton.setEnabled(!busy && loaded);
        residentButton.setEnabled(!busy && loaded);
        if (busy) {
            rewardButton.setEnabled(false);
            punishButton.setEnabled(false);
        }
    }

    private void setModelActionsEnabled(boolean enabled) {
        run.setEnabled(enabled);
        learnTextButton.setEnabled(enabled);
        resetButton.setEnabled(enabled);
        residentButton.setEnabled(enabled);
        if (!enabled) {
            rewardButton.setEnabled(false);
            punishButton.setEnabled(false);
        }
    }

    private void generate() {
        final String p = prompt.getText().toString();
        final int n;
        try { n = Integer.parseInt(count.getText().toString().trim()); }
        catch (Exception e) { status.setText("生成字符数必须是 1 到 200"); return; }
        if (p.trim().isEmpty() || n < 1 || n > 200) {
            status.setText("请输入提示词，生成字符数必须是 1 到 200"); return;
        }
        setBusy(true, "正在执行 AIMENG 神经元推理… " + model.backendStatus());
        worker.execute(() -> {
            long t = System.nanoTime();
            try {
                String text = model.generate(p, n);
                saveRuntimeStateQuietly();
                long ms = (System.nanoTime() - t) / 1_000_000;
                long pssMiB = Debug.getPss() / 1024L;
                runOnUiThread(() -> {
                    result.setText(text);
                    status.setText("本地推理完成 · " + ms + " ms · " + model.backendStatus() + " · 进程 PSS 约 " + pssMiB
                            + " MiB · 当前仅测得进程 PSS，仍需实机压力测试");
                    setBusy(false, null);
                    setModelActionsEnabled(true);
                    rewardButton.setEnabled(model.canGiveFeedback());
                    punishButton.setEnabled(model.canGiveFeedback());
                });
            } catch (Throwable e) {
                runOnUiThread(() -> {
                    status.setText("推理失败：" + errorText(e));
                    setBusy(false, null);
                    setModelActionsEnabled(model.isLoaded());
                });
            }
        });
    }

    private void resetLearning() {
        setBusy(true, "正在清除本机学习参数…");
        worker.execute(() -> {
            try {
                model.resetLearningState(learningFile());
                model.resetRuntimeState();
                saveRuntimeStateQuietly();
                runOnUiThread(() -> {
                    status.setText("已恢复基础模型 · 本机学习记录已清除 · 神经元残留状态已重置");
                    setBusy(false, null);
                    setModelActionsEnabled(true);
                });
            } catch (Throwable e) {
                runOnUiThread(() -> {
                    status.setText("恢复失败：" + errorText(e));
                    setBusy(false, null);
                    setModelActionsEnabled(model.isLoaded());
                });
            }
        });
    }

    private void learnFromInputText() {
        final String text = prompt.getText().toString();
        if (text.codePointCount(0, text.length()) < 2) {
            status.setText("请在输入框放入至少两个字符的学习文本");
            return;
        }
        setBusy(true, "正在用输入文字进行本地学习…");
        worker.execute(() -> {
            try {
                int examples = model.learnFromText(text, learningFile());
                saveRuntimeStateQuietly();
                runOnUiThread(() -> {
                    status.setText("手机本地学习完成 · 样本 " + examples + " 个 · 累计学习 "
                            + model.getLearningUpdates() + " 次 · 已保存");
                    setBusy(false, null);
                    setModelActionsEnabled(true);
                });
            } catch (Throwable e) {
                runOnUiThread(() -> {
                    status.setText("本地学习失败：" + errorText(e));
                    setBusy(false, null);
                    setModelActionsEnabled(model.isLoaded());
                });
            }
        });
    }

    private void giveFeedback(float reward) {
        rewardButton.setEnabled(false);
        punishButton.setEnabled(false);
        worker.execute(() -> {
            try {
                int changed = model.applyFeedback(reward, learningFile());
                saveRuntimeStateQuietly();
                runOnUiThread(() -> {
                    status.setText((reward > 0 ? "已奖励" : "已惩罚") + " · 更新参数 "
                            + changed + " 项 · 累计学习 " + model.getLearningUpdates() + " 次 · 已保存到手机");
                    rewardButton.setEnabled(model.canGiveFeedback());
                    punishButton.setEnabled(model.canGiveFeedback());
                });
            } catch (Throwable e) {
                runOnUiThread(() -> status.setText("学习更新失败：" + errorText(e)));
            }
        });
    }

    @Override protected void onDestroy() {
        saveRuntimeStateQuietly();
        worker.shutdownNow();
        if (serviceBound) {
            try { unbindService(neuronConnection); } catch (Throwable ignored) { }
            serviceBound = false;
        }
        super.onDestroy();
    }
}
