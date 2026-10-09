package bslsjdk.ornithnpu;

import android.app.Activity;
import android.app.AlertDialog;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.graphics.Color;
import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class NeuronLabActivity extends Activity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "aimeng-neuron-lab");
        t.setDaemon(true);
        return t;
    });
    private TextView npuState, report, unitsView;
    private EditText unitCount, epochCount;
    private Button runButton, npuButton;
    private volatile boolean busy;
    private NeuronLabEngine engine;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        buildUi();
        int count = getPreferences(MODE_PRIVATE).getInt("unit_count", 4);
        engine = new NeuronLabEngine(count);
        unitCount.setText(String.valueOf(count));
        epochCount.setText("60");
        refreshUnits();
        setNpuState("正在初始化 QNN / HTP V73…");
        worker.execute(() -> {
            boolean ok = NpuRuntime.init(getApplicationContext());
            String detail = ok ? NpuRuntime.status() : "NPU_OFFLINE " + NpuRuntime.getLastError();
            runOnUiThread(() -> {
                setNpuState(ok ? "NPU 已初始化：真实 QNN/HTP 路径可用，尚待前向实测" : detail);
                npuButton.setEnabled(ok);
                appendReport(ok
                        ? "运行时已初始化。训练/评分目前在 CPU 执行，NPU 按钮会执行真实 INT8 MatMul 前向并与 CPU 对照。"
                        : "NPU 初始化失败，仍可运行 CPU 学习实验。详情：" + detail);
            });
        });
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setBackgroundColor(Color.rgb(246, 247, 249));
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(18), dp(18), dp(18), dp(28));
        scroll.addView(root);
        setContentView(scroll);

        TextView title = text("AIMENG · 神经元实验", 26, true);
        root.addView(title);
        TextView subtitle = text("小规模可训练单元 / 评分进退场 / Qualcomm QNN HTP V73", 14, false);
        subtitle.setTextColor(Color.rgb(91, 101, 116));
        root.addView(subtitle);

        npuState = text("NPU：检测中", 14, true);
        npuState.setPadding(dp(12), dp(12), dp(12), dp(12));
        npuState.setBackgroundColor(Color.WHITE);
        LinearLayout.LayoutParams stateLp = params(-1, -2);
        stateLp.topMargin = dp(16);
        root.addView(npuState, stateLp);

        LinearLayout settings = new LinearLayout(this);
        settings.setOrientation(LinearLayout.HORIZONTAL);
        settings.setGravity(Gravity.CENTER_VERTICAL);
        settings.setPadding(0, dp(14), 0, dp(6));
        root.addView(settings);
        unitCount = new EditText(this);
        unitCount.setSingleLine(true);
        unitCount.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        unitCount.setHint("神经元数 2-16");
        unitCount.setText("4");
        settings.addView(unitCount, new LinearLayout.LayoutParams(0, dp(52), 1));
        epochCount = new EditText(this);
        epochCount.setSingleLine(true);
        epochCount.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        epochCount.setHint("训练轮数");
        epochCount.setText("60");
        LinearLayout.LayoutParams ep = new LinearLayout.LayoutParams(0, dp(52), 1);
        ep.leftMargin = dp(8);
        settings.addView(epochCount, ep);

        runButton = button("运行学习 + 评分 + 进退场");
        root.addView(runButton, params(-1, dp(52)));
        runButton.setOnClickListener(v -> startExperiment(false));

        npuButton = button("执行 NPU 前向并与 CPU 对照");
        npuButton.setEnabled(false);
        LinearLayout.LayoutParams npuLp = params(-1, dp(52));
        npuLp.topMargin = dp(8);
        root.addView(npuButton, npuLp);
        npuButton.setOnClickListener(v -> startExperiment(true));

        Button refresh = button("查看单元状态与评分");
        LinearLayout.LayoutParams refreshLp = params(-1, dp(48));
        refreshLp.topMargin = dp(8);
        root.addView(refresh, refreshLp);
        refresh.setOnClickListener(v -> appendReport(engine.summary()));

        Button diagnostics = button("打开 NPU 服务诊断");
        LinearLayout.LayoutParams diagLp = params(-1, dp(48));
        diagLp.topMargin = dp(8);
        root.addView(diagnostics, diagLp);
        diagnostics.setOnClickListener(v -> startActivity(new android.content.Intent(this, MainActivity.class)));

        unitsView = text("", 13, false);
        unitsView.setTextColor(Color.rgb(55, 65, 81));
        unitsView.setPadding(dp(12), dp(12), dp(12), dp(12));
        unitsView.setBackgroundColor(Color.WHITE);
        LinearLayout.LayoutParams unitsLp = params(-1, -2);
        unitsLp.topMargin = dp(12);
        root.addView(unitsView, unitsLp);

        report = text("准备就绪。先从 4 个单元开始，确认学习与 NPU 前向结果，再逐步增加数量。", 13, false);
        report.setTextIsSelectable(true);
        report.setTextColor(Color.rgb(31, 41, 55));
        report.setPadding(dp(12), dp(12), dp(12), dp(12));
        report.setBackgroundColor(Color.WHITE);
        LinearLayout.LayoutParams reportLp = params(-1, -2);
        reportLp.topMargin = dp(10);
        root.addView(report, reportLp);
    }

    private void startExperiment(boolean useNpu) {
        if (busy) return;
        final int count, epochs;
        try {
            count = Integer.parseInt(unitCount.getText().toString().trim());
            epochs = Integer.parseInt(epochCount.getText().toString().trim());
            if (count < 2 || count > 16) throw new IllegalArgumentException("神经元数量必须在 2 到 16 之间");
            if (epochs < 1 || epochs > 500) throw new IllegalArgumentException("训练轮数必须在 1 到 500 之间");
        } catch (Throwable t) {
            new AlertDialog.Builder(this).setTitle("参数不合法").setMessage(t.getMessage())
                    .setPositiveButton("知道了", null).show();
            return;
        }
        busy = true;
        runButton.setEnabled(false);
        npuButton.setEnabled(false);
        report.setText("正在运行实验…");
        worker.execute(() -> {
            try {
                if (engine == null || engineUnitCount() != count) {
                    engine = new NeuronLabEngine(count);
                    runOnUiThread(this::refreshUnits);
                }
                NeuronLabEngine.ForwardBackend backend = useNpu ? new NpuNeuronForward() : null;
                File trace = new File(getFilesDir(), "neuron-lab-trace.jsonl");
                NeuronLabEngine.Result result = engine.trainAndScore(epochs, backend, trace);
                String summary = engine.summary();
                runOnUiThread(() -> {
                    appendReport(result.report + "\n\n" + summary);
                    refreshUnits();
                    getPreferences(MODE_PRIVATE).edit().putInt("unit_count", count).apply();
                });
            } catch (Throwable t) {
                runOnUiThread(() -> appendReport("实验失败：" + t.getClass().getSimpleName() + ": " + t.getMessage()));
            } finally {
                runOnUiThread(() -> {
                    busy = false;
                    runButton.setEnabled(true);
                    npuButton.setEnabled(NpuRuntime.isReady());
                });
            }
        });
    }

    private int engineUnitCount() {
        try {
            String s = engine.summary();
            int at = s.indexOf("units=");
            int end = s.indexOf(' ', at);
            return Integer.parseInt(s.substring(at + 6, end));
        } catch (Throwable t) { return -1; }
    }

    private void refreshUnits() {
        if (unitsView != null && engine != null) unitsView.setText(engine.summary());
    }

    private void setNpuState(String value) { if (npuState != null) npuState.setText(value); }

    private void appendReport(String value) {
        if (report == null) return;
        report.setText(value);
    }

    private TextView text(String value, int size, boolean bold) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(size);
        view.setTextColor(Color.rgb(17, 24, 39));
        if (bold) view.setTypeface(null, android.graphics.Typeface.BOLD);
        return view;
    }

    private Button button(String value) {
        Button b = new Button(this);
        b.setText(value);
        b.setAllCaps(false);
        return b;
    }

    private LinearLayout.LayoutParams params(int width, int height) {
        return new LinearLayout.LayoutParams(width < 0 ? width : dp(width), height < 0 ? height : dp(height));
    }

    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }

    @Override protected void onDestroy() {
        worker.shutdownNow();
        super.onDestroy();
    }
}
