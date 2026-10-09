package bslsjdk.ornithnpu;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.PendingIntent;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import rikka.shizuku.Shizuku;

/**
 * General Android command console. Commands run as the app UID unless Shizuku
 * is connected and explicitly authorized, in which case they run as Android's
 * shell UID. This is not root and does not bundle Python/JDK runtimes.
 */
public final class CommandTerminalActivity extends Activity {
    private static final long COMMAND_TIMEOUT_SECONDS = 120;
    private static final int MAX_OUTPUT_CHARS = 500_000;
    private static final AtomicInteger TERMUX_REQUEST_IDS = new AtomicInteger(4100);
    private final Object processLock = new Object();
    private volatile Process currentProcess;
    private volatile boolean stopping;
    private EditText commandInput;
    private TextView status, output;
    private Button runButton, stopButton;
    private CheckBox termuxMode;
    private volatile boolean termuxRunning;
    private final StringBuilder transcript = new StringBuilder();

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        ShizukuHelper.init();
        TermuxCommandResultReceiver.resultHandler = this::onTermuxResult;
        buildUi();
        append("AIMENG 通用命令终端\n");
        append("默认目录：/sdcard\n");
        append("执行身份：" + executionMode() + "\n");
        append("提示：Android 系统通常没有 python、javac 或完整 Linux 包管理器。需要 Python/JDK 时，请先安装并配置兼容运行时（例如 Termux）；此终端不会假装它们已经存在。\n");
        append("命令有 120 秒默认时限，输出最多保留 500,000 字符。不要运行来源不明的脚本。\n\n");
        refreshStatus();
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(14), dp(12), dp(14), dp(20));
        root.setBackgroundColor(0xFFF5F7FA);
        scroll.addView(root);

        TextView title = label("通用命令终端", 23, true);
        root.addView(title);
        root.addView(label("运行 Android shell 命令、检查环境、执行脚本启动器。Shizuku 授权后使用 shell 身份；未授权时退回应用自身权限。", 13, false));

        status = label("", 12, true);
        root.addView(status, lp(0, 8));

        commandInput = new EditText(this);
        commandInput.setTextSize(14);
        commandInput.setGravity(Gravity.TOP | Gravity.START);
        commandInput.setMinLines(2);
        commandInput.setMaxLines(5);
        commandInput.setSingleLine(false);
        commandInput.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE
                | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
        commandInput.setHint("输入命令，例如：id；pwd；ls -la /sdcard");
        commandInput.setText("id");
        root.addView(commandInput, lp(0, 8));

        LinearLayout buttons = new LinearLayout(this);
        buttons.setOrientation(LinearLayout.HORIZONTAL);
        runButton = button("执行命令");
        stopButton = button("停止");
        stopButton.setEnabled(false);
        Button clearButton = button("清空输出");
        buttons.addView(runButton, new LinearLayout.LayoutParams(0, dp(48), 1));
        buttons.addView(stopButton, new LinearLayout.LayoutParams(0, dp(48), 1));
        buttons.addView(clearButton, new LinearLayout.LayoutParams(0, dp(48), 1));
        root.addView(buttons, lp(0, 8));

        LinearLayout quick = new LinearLayout(this);
        quick.setOrientation(LinearLayout.HORIZONTAL);
        Button environment = button("检查环境");
        Button auth = button("请求 Shizuku 授权");
        quick.addView(environment, new LinearLayout.LayoutParams(0, dp(46), 1));
        quick.addView(auth, new LinearLayout.LayoutParams(0, dp(46), 1));
        root.addView(quick, lp(0, 4));

        termuxMode = new CheckBox(this);
        termuxMode.setText("在 Termux 环境执行（需要安装 Termux 并单独授权）");
        termuxMode.setTextSize(12);
        root.addView(termuxMode, lp(0, 4));
        root.addView(label("Termux 模式可使用已安装的 Python、OpenJDK、Git 等包；需授权 RUN_COMMAND，并在 Termux 中启用 allow-external-apps。", 11, false), lp(0, 2));

        TextView outTitle = label("终端输出", 16, true);
        root.addView(outTitle, lp(0, 10));
        output = label("", 12, false);
        output.setTextIsSelectable(true);
        output.setTypeface(android.graphics.Typeface.MONOSPACE);
        output.setPadding(dp(10), dp(10), dp(10), dp(10));
        output.setBackgroundColor(0xFF101828);
        output.setTextColor(0xFFE4E7EC);
        root.addView(output, lp(0, 8));

        runButton.setOnClickListener(v -> runCommand());
        stopButton.setOnClickListener(v -> stopCommand());
        clearButton.setOnClickListener(v -> { synchronized (transcript) { transcript.setLength(0); } output.setText(""); });
        environment.setOnClickListener(v -> {
            commandInput.setText("id; echo '--- PATH ---'; echo \"$PATH\"; echo '--- runtimes ---'; command -v sh; command -v python; command -v python3; command -v java; command -v javac; command -v pkg; command -v git; echo '--- storage ---'; pwd; ls -ld /sdcard");
            runCommand();
        });
        auth.setOnClickListener(v -> {
            if (!ShizukuHelper.available()) {
                toast("Shizuku 未运行。可以先用应用权限执行基础命令。");
            } else if (ShizukuHelper.granted()) {
                toast("Shizuku 已授权。");
            } else {
                try { ShizukuHelper.requestPermission(); toast("已发起授权请求，请在 Shizuku 弹窗中确认。"); }
                catch (Throwable e) { toast("授权请求失败：" + e.getMessage()); }
            }
            refreshStatus();
        });
        setContentView(scroll);
    }

    private void runCommand() {
        final String command = commandInput.getText().toString().trim();
        if (command.isEmpty()) { toast("先输入一条命令。"); return; }
        synchronized (processLock) {
            if (currentProcess != null || termuxRunning) { toast("已有命令正在运行。"); return; }
            stopping = false;
        }
        append("\n$ " + command + "\n");
        runButton.setEnabled(false);
        stopButton.setEnabled(true);
        commandInput.setEnabled(false);
        refreshStatus();
        if (termuxMode != null && termuxMode.isChecked()) {
            executeInTermux(command);
            return;
        }
        Thread thread = new Thread(() -> execute(command), "aimeng-command-runner");
        thread.setDaemon(true);
        thread.start();
    }

    private void executeInTermux(String command) {
        try {
            getPackageManager().getPackageInfo("com.termux", 0);
        } catch (Throwable missing) {
            finishTermuxDispatchError("未检测到 Termux。系统 shell 本身不附带 Python/JDK；安装并初始化 Termux 后才能使用该模式。");
            return;
        }
        if (checkSelfPermission("com.termux.permission.RUN_COMMAND") != PackageManager.PERMISSION_GRANTED) {
            new AlertDialog.Builder(this)
                    .setTitle("需要授权 Termux 命令执行")
                    .setMessage("请在 Android 设置 → 应用 → AIMENG 神经元实验 → 权限 → 其他权限中，允许“在 Termux 环境中运行命令”。还需要在 Termux 的 ~/.termux/termux.properties 中设置 allow-external-apps=true，并重启 Termux。")
                    .setPositiveButton("知道了", (d, w) -> { })
                    .show();
            finishTermuxDispatchError("缺少 com.termux.permission.RUN_COMMAND 授权。");
            return;
        }
        try {
            Intent intent = new Intent();
            intent.setClassName("com.termux", "com.termux.app.RunCommandService");
            intent.setAction("com.termux.RUN_COMMAND");
            intent.putExtra("com.termux.RUN_COMMAND_PATH", "/data/data/com.termux/files/usr/bin/bash");
            String bounded = "if command -v timeout >/dev/null 2>&1; then timeout 120s bash -lc \"$1\"; else bash -lc \"$1\"; fi";
            intent.putExtra("com.termux.RUN_COMMAND_ARGUMENTS", new String[]{"-lc", bounded, "aimeng", command});
            intent.putExtra("com.termux.RUN_COMMAND_WORKDIR", "/data/data/com.termux/files/home");
            intent.putExtra("com.termux.RUN_COMMAND_BACKGROUND", true);
            intent.putExtra("com.termux.RUN_COMMAND_SESSION_ACTION", "0");
            intent.putExtra("com.termux.RUN_COMMAND_LABEL", "AIMENG command");
            intent.putExtra("com.termux.RUN_COMMAND_DESCRIPTION", "Command submitted by the user from AIMENG's terminal.");
            Intent resultIntent = new Intent(this, TermuxCommandResultReceiver.class);
            int requestId = TERMUX_REQUEST_IDS.incrementAndGet();
            int flags = PendingIntent.FLAG_ONE_SHOT;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) flags |= PendingIntent.FLAG_MUTABLE;
            PendingIntent pending = PendingIntent.getBroadcast(this, requestId, resultIntent, flags);
            intent.putExtra("com.termux.RUN_COMMAND_PENDING_INTENT", pending);
            termuxRunning = true;
            stopButton.setEnabled(false);
            status.setText("状态：Termux 后台命令运行中 · 最长 120 秒（依赖 Termux timeout 工具）");
            startService(intent);
        } catch (Throwable e) {
            termuxRunning = false;
            finishTermuxDispatchError("无法启动 Termux 命令：" + e.getClass().getSimpleName() + ": " + String.valueOf(e.getMessage()));
        }
    }

    private void onTermuxResult(String result) {
        runOnUiThread(() -> {
            termuxRunning = false;
            append(result);
            runButton.setEnabled(true);
            stopButton.setEnabled(false);
            commandInput.setEnabled(true);
            refreshStatus();
        });
    }

    private void finishTermuxDispatchError(String message) {
        runOnUiThread(() -> {
            append("[Termux 执行失败] " + message + "\n");
            runButton.setEnabled(true);
            stopButton.setEnabled(false);
            commandInput.setEnabled(true);
            refreshStatus();
        });
    }

    private void execute(String command) {
        Process process = null;
        StringBuilder result = new StringBuilder();
        boolean truncated = false;
        int exitCode = -1;
        try {
            String script = "cd /sdcard 2>/dev/null || cd /; eval \"$1\" 2>&1";
            if (ShizukuHelper.granted()) {
                process = Shizuku.newProcess(new String[]{"/system/bin/sh", "-c", script, "aimeng", command}, null, "/sdcard");
            } else {
                ProcessBuilder pb = new ProcessBuilder("/system/bin/sh", "-c", script, "aimeng", command);
                pb.redirectErrorStream(true);
                process = pb.start();
            }
            synchronized (processLock) { currentProcess = process; }
            final Process active = process;
            Thread reader = new Thread(() -> {
                try (BufferedReader br = new BufferedReader(new InputStreamReader(active.getInputStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        synchronized (result) {
                            if (result.length() < MAX_OUTPUT_CHARS) result.append(line).append('\n');
                        }
                        publishOutput(result.toString());
                    }
                } catch (Throwable e) {
                    synchronized (result) { result.append("\n[读取输出失败] ").append(e).append('\n'); }
                }
            }, "aimeng-command-output");
            reader.setDaemon(true);
            reader.start();
            boolean ended = process.waitFor(COMMAND_TIMEOUT_SECONDS, TimeUnit.SECONDS);
            if (!ended) {
                process.destroy();
                if (!process.waitFor(1500, TimeUnit.MILLISECONDS)) process.destroyForcibly();
                synchronized (result) { result.append("\n[超时] 命令超过 120 秒，已请求终止主进程。子进程是否全部退出取决于命令自身。\n"); }
            } else {
                exitCode = process.exitValue();
            }
            reader.join(1500);
            synchronized (result) {
                if (result.length() >= MAX_OUTPUT_CHARS) truncated = true;
                result.append("\n[结束] ");
                if (stopping) result.append("已请求停止");
                else if (!ended) result.append("超时");
                else result.append("退出码=").append(exitCode);
                result.append("；执行身份=").append(ShizukuHelper.granted() ? "Shizuku shell" : "应用 UID").append('\n');
                if (truncated) result.append("[输出达到上限，后续输出已截断]\n");
            }
        } catch (Throwable e) {
            synchronized (result) { result.append("\n[执行失败] ").append(e.getClass().getSimpleName()).append(": ").append(String.valueOf(e.getMessage())).append('\n'); }
        } finally {
            synchronized (processLock) { if (currentProcess == process) currentProcess = null; }
            final String finalResult;
            synchronized (result) { finalResult = result.toString(); }
            runOnUiThread(() -> {
                append(finalResult);
                runButton.setEnabled(true);
                stopButton.setEnabled(false);
                commandInput.setEnabled(true);
                refreshStatus();
            });
        }
    }

    private void stopCommand() {
        if (termuxRunning) {
            toast("Termux 命令由后台服务执行；此页不能直接杀掉它。命令会尝试在 120 秒后超时结束。");
            return;
        }
        stopping = true;
        Process p = currentProcess;
        if (p != null) {
            try { p.destroy(); } catch (Throwable ignored) {}
            try {
                Thread t = new Thread(() -> {
                    try { if (!p.waitFor(700, TimeUnit.MILLISECONDS)) p.destroyForcibly(); }
                    catch (Throwable ignored) { try { p.destroyForcibly(); } catch (Throwable ignored2) {} }
                }, "aimeng-command-stop");
                t.setDaemon(true);
                t.start();
            } catch (Throwable ignored) {}
            append("\n[停止请求已发送给当前 shell 进程。复杂脚本可能留下子进程，请检查进程列表。]\n");
        }
    }

    private void publishOutput(String text) {
        final String prefix;
        synchronized (transcript) { prefix = transcript.toString(); }
        runOnUiThread(() -> output.setText(prefix + text));
    }

    private void append(String text) {
        synchronized (transcript) {
            transcript.append(text);
            if (transcript.length() > MAX_OUTPUT_CHARS) transcript.delete(0, transcript.length() - MAX_OUTPUT_CHARS);
            output.setText(transcript.toString());
        }
    }

    private void refreshStatus() {
        if (status == null) return;
        status.setText("状态：" + executionMode() + (currentProcess == null ? " · 空闲" : " · 运行中"));
    }

    private String executionMode() {
        if (ShizukuHelper.granted()) return "Shizuku shell（非 root）";
        if (ShizukuHelper.available()) return "Shizuku 已连接但未授权，当前为应用 UID";
        return "普通应用权限（非 root）";
    }

    private TextView label(String s, int size, boolean bold) {
        TextView t = new TextView(this);
        t.setText(s); t.setTextSize(size); t.setTextColor(0xFF182230);
        if (bold) t.setTypeface(null, android.graphics.Typeface.BOLD);
        return t;
    }

    private Button button(String s) {
        Button b = new Button(this); b.setText(s); b.setAllCaps(false); b.setTextSize(12); return b;
    }

    private LinearLayout.LayoutParams lp(int ignored, int top) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, -2);
        p.topMargin = dp(top);
        return p;
    }

    private int dp(int v) { return (int)(v * getResources().getDisplayMetrics().density + 0.5f); }
    private void toast(String s) { Toast.makeText(this, s, Toast.LENGTH_LONG).show(); }

    @Override protected void onResume() { super.onResume(); refreshStatus(); }

    @Override protected void onDestroy() {
        TermuxCommandResultReceiver.resultHandler = null;
        stopping = true;
        Process p = currentProcess;
        if (p != null) try { p.destroy(); } catch (Throwable ignored) {}
        super.onDestroy();
    }
}
