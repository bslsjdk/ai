package bslsjdk.ornithnpu;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Separate CPU-only screen for AIMENG's portable sparse-diffusion model. */
public final class SparseDiffusionActivity extends Activity {
    private static final int PICK = 7712;
    private final SparseDiffusionMobileModel model = new SparseDiffusionMobileModel();
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private EditText prompt, count;
    private TextView status, result;
    private Button run, rewardButton, punishButton;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(16, 16, 16, 16);
        TextView title = new TextView(this);
        title.setText("AIMENG 神经元扩散 · 手机本地推理");
        title.setTextSize(20);
        root.addView(title);
        status = new TextView(this);
        status.setText("尚未导入模型。推理在本机 CPU 执行，不调用云端。");
        root.addView(status);
        Button pick = new Button(this);
        pick.setText("导入 mobile_diffusion.json");
        pick.setOnClickListener(v -> {
            Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            i.addCategory(Intent.CATEGORY_OPENABLE);
            i.setType("application/json");
            startActivityForResult(i, PICK);
        });
        root.addView(pick);
        prompt = new EditText(this);
        prompt.setHint("提示词，例如：你好");
        root.addView(prompt);
        count = new EditText(this);
        count.setInputType(2);
        count.setText("40");
        count.setHint("生成字符数，1 到 200");
        root.addView(count);
        run = new Button(this);
        run.setText("在手机上生成");
        run.setEnabled(false);
        run.setOnClickListener(v -> generate());
        root.addView(run);
        rewardButton = new Button(this);
        rewardButton.setText("奖励 +1（这次结果有帮助）");
        rewardButton.setEnabled(false);
        rewardButton.setOnClickListener(v -> giveFeedback(1f));
        root.addView(rewardButton);
        punishButton = new Button(this);
        punishButton.setText("惩罚 -1（这次结果不好）");
        punishButton.setEnabled(false);
        punishButton.setOnClickListener(v -> giveFeedback(-1f));
        root.addView(punishButton);
        ScrollView scroll = new ScrollView(this);
        result = new TextView(this);
        result.setTextIsSelectable(true);
        result.setTextSize(16);
        scroll.addView(result);
        root.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));
        setContentView(root);
    }

    @Override protected void onActivityResult(int request, int resultCode, Intent data) {
        super.onActivityResult(request, resultCode, data);
        if (request != PICK || resultCode != RESULT_OK || data == null) return;
        Uri uri = data.getData();
        if (uri == null) return;
        run.setEnabled(false);
        status.setText("正在复制并加载模型…");
        worker.execute(() -> {
            File file = new File(getFilesDir(), "aimeng-mobile-diffusion.json");
            try (InputStream in = getContentResolver().openInputStream(uri);
                 OutputStream out = new FileOutputStream(file)) {
                if (in == null) throw new IOException("无法打开文件");
                byte[] b = new byte[8192]; int n; long total = 0;
                while ((n = in.read(b)) >= 0) {
                    total += n;
                    if (total > 32L * 1024 * 1024) throw new IOException("模型包超过 32 MiB 限制");
                    out.write(b, 0, n);
                }
                model.load(file);
                model.loadLearningState(new File(getFilesDir(), "aimeng-learning-state.json"));
                runOnUiThread(() -> { status.setText("模型已加载 · 本地学习记录 " + model.getLearningUpdates() + " 次"); run.setEnabled(true); });
            } catch (Throwable e) {
                runOnUiThread(() -> status.setText("加载失败：" + e.getClass().getSimpleName() + ": " + e.getMessage()));
            }
        });
    }

    private void generate() {
        final String p = prompt.getText().toString();
        final int n;
        try { n = Integer.parseInt(count.getText().toString().trim()); }
        catch (Exception e) { status.setText("生成字符数必须是 1 到 200"); return; }
        if (p.trim().isEmpty() || n < 1 || n > 200) {
            status.setText("请输入提示词，生成字符数必须是 1 到 200"); return;
        }
        run.setEnabled(false);
        status.setText("手机 CPU 正在执行扩散…");
        worker.execute(() -> {
            long t = System.nanoTime();
            try {
                String text = model.generate(p, n);
                long ms = (System.nanoTime() - t) / 1_000_000;
                runOnUiThread(() -> {
                    result.setText(text);
                    status.setText("本地 CPU 推理完成 · " + ms + " ms · 可给本次结果奖励或惩罚");
                    run.setEnabled(true);
                    rewardButton.setEnabled(model.canGiveFeedback());
                    punishButton.setEnabled(model.canGiveFeedback());
                });
            } catch (Throwable e) {
                runOnUiThread(() -> { status.setText("推理失败：" + e.getClass().getSimpleName() + ": " + e.getMessage()); run.setEnabled(true); });
            }
        });
    }

    private void giveFeedback(float reward) {
        rewardButton.setEnabled(false);
        punishButton.setEnabled(false);
        worker.execute(() -> {
            try {
                int changed = model.applyFeedback(reward, new File(getFilesDir(), "aimeng-learning-state.json"));
                runOnUiThread(() -> status.setText((reward > 0 ? "已奖励" : "已惩罚")
                        + " · 更新参数 " + changed + " 项 · 累计学习 " + model.getLearningUpdates()
                        + " 次 · 已保存到手机"));
            } catch (Throwable e) {
                runOnUiThread(() -> status.setText("学习更新失败：" + e.getClass().getSimpleName() + ": " + e.getMessage()));
            }
        });
    }

    @Override protected void onDestroy() { worker.shutdownNow(); super.onDestroy(); }
}
