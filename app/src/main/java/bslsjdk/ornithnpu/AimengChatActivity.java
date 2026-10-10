package bslsjdk.ornithnpu;

import android.app.Activity;
import android.content.ComponentName;
import android.content.Intent;
import android.content.ServiceConnection;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Debug;
import android.os.IBinder;
import android.view.Gravity;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Simple chat surface over the same resident AIMENG model instance. */
public final class AimengChatActivity extends Activity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private SparseDiffusionMobileModel model;
    private AimengNeuronService service;
    private boolean bound;
    private LinearLayout messages;
    private ScrollView scroll;
    private EditText input;
    private Button send;
    private TextView status;

    private final ServiceConnection connection = new ServiceConnection() {
        @Override public void onServiceConnected(ComponentName name, IBinder binder) {
            service = ((AimengNeuronService.LocalBinder) binder).getService();
            model = service.getModel();
            bound = true;
            loadIfNeeded();
        }
        @Override public void onServiceDisconnected(ComponentName name) {
            bound = false; service = null; model = null;
            send.setEnabled(false);
            status.setText("本地神经元服务断开");
        }
    };

    private int dp(int x) { return (int)(x * getResources().getDisplayMetrics().density + 0.5f); }

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(12), dp(10), dp(12), dp(10));
        root.setBackgroundColor(Color.rgb(245, 247, 251));
        TextView title = new TextView(this);
        title.setText("AIMENG 本地聊天");
        title.setTextSize(21); title.setTextColor(Color.rgb(25, 40, 65));
        root.addView(title);
        status = new TextView(this);
        status.setText("连接本地神经元服务…");
        status.setTextColor(Color.rgb(75, 88, 105));
        root.addView(status);
        scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        messages = new LinearLayout(this);
        messages.setOrientation(LinearLayout.VERTICAL);
        messages.setPadding(dp(2), dp(10), dp(2), dp(10));
        scroll.addView(messages);
        root.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1f));
        input = new EditText(this);
        input.setSingleLine(false);
        input.setMinLines(1); input.setMaxLines(4);
        input.setHint("输入消息…");
        input.setImeOptions(EditorInfo.IME_ACTION_SEND);
        root.addView(input, new LinearLayout.LayoutParams(-1, -2));
        send = new Button(this);
        send.setText("发送到手机本地模型");
        send.setEnabled(false);
        root.addView(send);
        send.setOnClickListener(v -> sendMessage());
        input.setOnEditorActionListener((v, action, event) -> {
            if (action == EditorInfo.IME_ACTION_SEND) { sendMessage(); return true; }
            return false;
        });
        addMessage("系统", "这里运行的是当前已导入的 AIMENG 稀疏扩散模型，不会自动变成通用大语言模型。");
        setContentView(root);
        bindService(new Intent(this, AimengNeuronService.class), connection, BIND_AUTO_CREATE);
    }

    private void loadIfNeeded() {
        if (model == null || model.isLoaded()) {
            updateReadyStatus();
            return;
        }
        worker.execute(() -> {
            try {
                File bundle = new File(getFilesDir(), "aimeng-mobile-diffusion.aimg");
                if (!bundle.isFile()) bundle = new File(getFilesDir(), "aimeng-mobile-diffusion.json");
                if (!bundle.isFile()) {
                    runOnUiThread(() -> status.setText("尚未导入模型。先在“神经元运行台”导入 .aimg 或旧版 .json。"));
                    return;
                }
                model.load(bundle);
                model.loadLearningState(new File(getFilesDir(), "aimeng-learning-state.json"));
                model.loadRuntimeState(new File(getFilesDir(), "aimeng-neuron-residual.bin"));
                runOnUiThread(this::updateReadyStatus);
            } catch (Throwable e) {
                runOnUiThread(() -> status.setText("模型加载失败：" + e.getClass().getSimpleName() + ": " + e.getMessage()));
            }
        });
    }

    private void updateReadyStatus() {
        if (model != null && model.isLoaded()) {
            status.setText("已连接同一模型实例 · " + model.backendStatus()
                    + " · PSS " + (Debug.getPss() / 1024L) + " MiB");
            send.setEnabled(true);
        }
    }

    private void sendMessage() {
        if (model == null || !model.isLoaded()) {
            status.setText("模型尚未加载，请先导入模型。");
            return;
        }
        String prompt = input.getText().toString().trim();
        if (prompt.isEmpty()) return;
        input.setText("");
        addMessage("你", prompt);
        send.setEnabled(false);
        worker.execute(() -> {
            long start = android.os.SystemClock.elapsedRealtime();
            String answer;
            try {
                answer = model.generate(prompt, 160);
                model.saveRuntimeState(new File(getFilesDir(), "aimeng-neuron-residual.bin"));
            } catch (Throwable e) {
                answer = "本地推理失败：" + e.getClass().getSimpleName() + ": " + e.getMessage();
            }
            final String reply = answer;
            final long elapsed = android.os.SystemClock.elapsedRealtime() - start;
            runOnUiThread(() -> {
                addMessage("AIMENG", reply);
                status.setText("本地生成耗时 " + elapsed + " ms · PSS "
                        + (Debug.getPss() / 1024L) + " MiB · " + model.backendStatus());
                send.setEnabled(true);
            });
        });
    }

    private void addMessage(String who, String text) {
        TextView bubble = new TextView(this);
        bubble.setText(who + "\n" + text);
        bubble.setTextSize(15);
        bubble.setTextColor(Color.rgb(35, 48, 70));
        bubble.setPadding(dp(12), dp(10), dp(12), dp(10));
        bubble.setBackgroundColor(Color.WHITE);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, -2);
        p.bottomMargin = dp(8);
        p.gravity = who.equals("你") ? Gravity.END : Gravity.START;
        messages.addView(bubble, p);
        scroll.post(() -> scroll.fullScroll(ScrollView.FOCUS_DOWN));
    }

    @Override protected void onDestroy() {
        if (model != null && model.isLoaded()) {
            try { model.saveRuntimeState(new File(getFilesDir(), "aimeng-neuron-residual.bin")); }
            catch (Throwable ignored) { }
        }
        worker.shutdownNow();
        if (bound) { try { unbindService(connection); } catch (Throwable ignored) { } }
        super.onDestroy();
    }
}
