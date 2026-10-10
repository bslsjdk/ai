package bslsjdk.ornithnpu;

import android.app.Activity;
import android.content.ComponentName;
import android.content.Intent;
import android.content.ServiceConnection;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Debug;
import android.os.IBinder;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;

/** Real monitor for the service-owned model; it never invents neuron metrics. */
public final class DiffusionMonitorActivity extends Activity {
    private SparseDiffusionMobileModel model;
    private TextView status, summary, trace;
    private boolean bound;

    private int dp(int x) { return (int)(x * getResources().getDisplayMetrics().density + 0.5f); }
    private TextView card(String text) {
        TextView v = new TextView(this);
        v.setText(text); v.setTextSize(14); v.setTextColor(Color.rgb(35, 48, 70));
        v.setPadding(dp(14), dp(13), dp(14), dp(13)); v.setBackgroundColor(Color.WHITE);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, -2);
        p.bottomMargin = dp(10);
        return v;
    }
    private final ServiceConnection connection = new ServiceConnection() {
        @Override public void onServiceConnected(ComponentName name, IBinder binder) {
            model = ((AimengNeuronService.LocalBinder) binder).getService().getModel();
            refresh();
        }
        @Override public void onServiceDisconnected(ComponentName name) {
            model = null;
            status.setText("驻留服务已断开");
        }
    };

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        ScrollView scroll = new ScrollView(this);
        scroll.setFitsSystemWindows(true); scroll.setFillViewport(true);
        scroll.setBackgroundColor(Color.rgb(245, 247, 251));
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(18), dp(14), dp(18), dp(24));
        scroll.addView(root);
        root.addView(card("AIMENG · 神经元活动监视器"));
        status = card("正在连接本机模型…"); root.addView(status);
        summary = card("模型摘要尚未读取"); root.addView(summary);
        trace = card("扩散轨迹：尚未执行推理。先在聊天界面发送一条消息。");
        root.addView(trace);
        Button refresh = new Button(this);
        refresh.setText("刷新当前模型状态与扩散轨迹");
        refresh.setOnClickListener(v -> refresh());
        root.addView(refresh);
        Button chat = new Button(this);
        chat.setText("打开本地聊天并触发一次扩散");
        chat.setOnClickListener(v -> startActivity(new Intent(this, ChatActivity.class)));
        root.addView(chat);
        Button probe = new Button(this);
        probe.setText("打开逐步数值探针");
        probe.setOnClickListener(v -> startActivity(new Intent(this, DiffusionProbeActivity.class)));
        root.addView(probe);
        root.addView(card("这里展示的是模型实际记录的扩散步、活跃节点、路由、状态变化和停止条件，不把固定说明伪装成实时神经元活动。进程 PSS 是系统观测值，不等于总设备内存。"));
        setContentView(scroll);
        bound = bindService(new Intent(this, AimengNeuronService.class), connection, BIND_AUTO_CREATE);
    }

    private void refresh() {
        if (model == null) {
            status.setText("模型服务尚未连接。");
            return;
        }
        if (!model.isLoaded()) {
            File bundle = new File(getFilesDir(), "aimeng-mobile-diffusion.aimg");
            if (!bundle.isFile()) bundle = new File(getFilesDir(), "aimeng-mobile-diffusion.json");
            try {
                if (!bundle.isFile()) {
                    status.setText("尚未导入模型。请先打开神经元运行台导入 .aimg 或旧版 .json。");
                    return;
                }
                model.load(bundle);
                model.loadLearningState(new File(getFilesDir(), "aimeng-learning-state.json"));
                model.loadRuntimeState(new File(getFilesDir(), "aimeng-neuron-residual.bin"));
            } catch (Throwable e) {
                status.setText("模型加载失败：" + e.getClass().getSimpleName() + ": " + e.getMessage());
                return;
            }
        }
        status.setText("模型已加载 · " + model.backendStatus() + " · PSS "
                + (Debug.getPss() / 1024L) + " MiB");
        summary.setText(model.getModelSummary());
        String last = model.getLastDiffusionTraceText();
        trace.setText(last == null || last.trim().isEmpty()
                ? "扩散轨迹：当前模型实例还没有记录到一次前向推理。打开聊天发送消息后再刷新。"
                : "最近一次真实扩散轨迹\n" + last);
    }

    @Override protected void onDestroy() {
        if (bound) {
            try { unbindService(connection); } catch (Throwable ignored) { }
            bound = false;
        }
        super.onDestroy();
    }
}
