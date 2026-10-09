package bslsjdk.ornithnpu;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Build;
import android.graphics.Color;
import android.graphics.Typeface;
import android.view.Gravity;
import android.view.View;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import android.net.Uri;
import android.content.Intent;
import android.content.ClipboardManager;
import android.content.ClipData;
import android.database.Cursor;
import android.provider.OpenableColumns;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import org.json.JSONArray;
import org.json.JSONObject;

public final class ChatActivity extends Activity {
    private static final String PREFS = "chat";
    private static final String HISTORY = "history";
    private static final String MODEL_PATH = "model_path";
    private static final String MODEL_FILENAME = "ornith-1.5-9b-mlx-4bit.safetensors";
    private static final int PICK_MODEL = 4201;
    private static final int SAVE_DIAGNOSTIC_REPORT = 4202;
    private static final String LAST_DIAGNOSTIC_ERROR = "last_diagnostic_error";
    private static final String LAST_DIAGNOSTIC_REPORT = "last_diagnostic_report";
    private static final int MAX_CONTEXT_MESSAGES = 32;
    // The full history remains persisted locally. AgentContext selects user-priority anchors,
    // relevant earlier turns and recent dialogue for the bounded native runtime.
    private static final int MAX_CONTEXT_CHARS = 150000;

    private LinearLayout messages;
    private ScrollView scroll;
    private EditText input;
    private TextView runtimeState;
    private TextView modelState;
    private TextView statusLine;
    private ProgressBar importProgress;
    private TextView importStage;
    private LinearLayout emptyState;
    private TextView emptyHint;
    private View diagnosticPanel;
    private TextView diagnosticPreview;
    private volatile String lastDiagnosticError = "";
    private volatile String lastDiagnosticReport = "";
    private final Handler main = new Handler(Looper.getMainLooper());

    private final AgentToolRegistry toolRegistry = new AgentToolRegistry();
    private final AgentExecutor agentExecutor =
            new AgentExecutor(toolRegistry, task -> { }, 8);
    private volatile boolean generating;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.activity_chat);

        messages = findViewById(R.id.messages);
        scroll = findViewById(R.id.messagesScroll);
        input = findViewById(R.id.input);
        runtimeState = findViewById(R.id.runtimeState);
        modelState = findViewById(R.id.modelState);
        statusLine = findViewById(R.id.statusLine);
        importProgress = findViewById(R.id.importProgress);
        importStage = findViewById(R.id.importStage);
        emptyState = findViewById(R.id.emptyState);
        emptyHint = findViewById(R.id.emptyHint);
        diagnosticPanel = findViewById(R.id.diagnosticPanel);
        diagnosticPreview = findViewById(R.id.diagnosticPreview);
        findViewById(R.id.copyDiagnostic).setOnClickListener(v -> copyDiagnosticError());
        findViewById(R.id.saveDiagnostic).setOnClickListener(v -> exportDiagnosticReport());

        findViewById(R.id.send).setOnClickListener(v -> sendMessage());
        findViewById(R.id.importModel).setOnClickListener(v -> importOrnithModel());
        findViewById(R.id.emptyImport).setOnClickListener(v -> importOrnithModel());
        input.setOnEditorActionListener((v, actionId, event) -> {
            if (event != null && event.getKeyCode() == android.view.KeyEvent.KEYCODE_ENTER
                    && event.isShiftPressed()) return false;
            if (event != null && event.getKeyCode() == android.view.KeyEvent.KEYCODE_ENTER) {
                sendMessage();
                return true;
            }
            return false;
        });

        loadHistory();
        String oldError = getSharedPreferences(PREFS, MODE_PRIVATE).getString(LAST_DIAGNOSTIC_ERROR, "");
        String oldReport = getSharedPreferences(PREFS, MODE_PRIVATE).getString(LAST_DIAGNOSTIC_REPORT, "");
        if (!oldError.isEmpty() && !oldReport.isEmpty()) showDiagnosticPanel(oldError, oldReport);
        updateRuntimeState();
        initLocalRuntime();
    }

    private void initLocalRuntime() {
        new Thread(() -> {
            boolean ok = NpuRuntime.init(getApplicationContext());
            String restoreError = null;
            if (ok) {
                String saved = getSharedPreferences(PREFS, MODE_PRIVATE)
                        .getString(MODEL_PATH, "");
                if (!saved.isEmpty()) {
                    if (!new File(saved).isFile()) {
                        restoreError = "ERR ORNITH15_RUNTIME model_missing (saved model file no longer exists)";
                    } else {
                        try {
                            String result = Ornith15Runtime.load(saved, 65536);
                            if (result == null || !result.startsWith("OK ORNITH15_RUNTIME/1"))
                                restoreError = result == null ? "ERR ORNITH15_RUNTIME null_native_reply" : result;
                        } catch (Throwable t) {
                            restoreError = "ERR ORNITH15_RUNTIME " + t.getClass().getSimpleName()
                                    + ": " + String.valueOf(t.getMessage());
                        }
                    }
                }
            } else {
                restoreError = "NPU initialization failed: " + NpuRuntime.getLastError();
            }
            final String failure = restoreError;
            main.post(() -> {
                updateRuntimeState();
                showEmptyStateIfNeeded();
                if (failure != null && !failure.isEmpty()) {
                    showDiagnosticError(failure);
                    addBubble("system", "本地运行环境/模型自动恢复失败：" + failure);
                    saveHistory();
                }
            });
        }, "mcnpu-init").start();
    }

    @Override protected void onResume() {
        super.onResume();
        updateRuntimeState();
        showEmptyStateIfNeeded();
    }

    private void showEmptyStateIfNeeded() {
        boolean empty = messages.getChildCount() == 0;
        emptyState.setVisibility(empty ? View.VISIBLE : View.GONE);
        scroll.setVisibility(empty ? View.GONE : View.VISIBLE);
        if (empty) {
            emptyHint.setText(NpuRuntime.isReady()
                    ? "本地 NPU 已在线，等待模型文件。"
                    : "本地 NPU 尚未启动，请先启动 NPU 服务。");
        }
    }

    private void updateRuntimeState() {
        boolean npu = NpuRuntime.isReady();
        boolean model = Ornith15Runtime.isLoaded();
        runtimeState.setText(npu ? "● NPU 在线" : "○ NPU 未就绪");
        runtimeState.setTextColor(npu ? Color.rgb(22, 120, 75) : Color.rgb(100, 116, 139));
        modelState.setText(model ? "Ornith-1.5-9B · 已加载" : "未加载模型");
        if (!npu) {
            statusLine.setText("Ornith-1.5-9B · 等待本地推理内核");
        } else if (!model) {
            statusLine.setText("Ornith-1.5-9B · 等待本地模型");
        } else {
            statusLine.setText(compactRuntimeInfo(Ornith15Runtime.info()));
        }
    }

    private static long longField(String s, String key) {
        if (s == null) return -1L;
        int i = s.indexOf(key);
        if (i < 0) return -1L;
        int j = i + key.length();
        int k = j;
        while (k < s.length()) {
            char ch = s.charAt(k);
            if ((ch >= '0' && ch <= '9') || ch == '-') k++;
            else break;
        }
        if (k == j) return -1L;
        try { return Long.parseLong(s.substring(j, k)); }
        catch (Throwable ignored) { return -1L; }
    }

    private static String compactRuntimeInfo(String info) {
        if (info == null || info.isEmpty()) return "Ornith-1.5-9B · NPU HTP V73";
        long prompt = longField(info, "last_gen_prompt_tokens=");
        long generated = longField(info, "last_gen_tokens=");
        long prefillUs = longField(info, "last_prefill_us=");
        long firstUs = longField(info, "last_first_token_us=");
        long decodeUs = longField(info, "last_decode_us=");
        long hwm = longField(info, "hwm_bytes=");
        if (hwm < 0) hwm = longField(info, "rss_bytes=");

        double hwmGiB = hwm > 0 ? hwm / 1073741824.0 : 0.0;
        if (generated > 0 && decodeUs > 0) {
            double prefillSec = prefillUs > 0 ? prefillUs / 1000000.0 : 0.0;
            double ttftSec = firstUs > 0 ? firstUs / 1000000.0 : 0.0;
            double tokPerSec = generated * 1000000.0 / decodeUs;
            return String.format(Locale.US,
                    "Ornith · %d prompt · %d tok · Prefill %.1fs · TTFT %.1fs · %.2f tok/s · HWM %.2fGiB",
                    prompt > 0 ? prompt : 0,
                    generated,
                    prefillSec,
                    ttftSec,
                    tokPerSec,
                    hwmGiB);
        }
        if (prompt > 0 || hwm > 0) {
            return String.format(Locale.US,
                    "Ornith · %d prompt · 64K · HWM %.2fGiB",
                    prompt > 0 ? prompt : 0,
                    hwmGiB);
        }
        return "Ornith-1.5-9B · MCNPU HTP V73 · 已加载";
    }

    private void showDiagnosticError(String error) {
        String safe = error == null || error.isEmpty() ? "未知错误（native 返回为空）" : error;
        lastDiagnosticError = safe;
        lastDiagnosticReport = buildDiagnosticReport(safe);
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                .putString(LAST_DIAGNOSTIC_ERROR, lastDiagnosticError)
                .putString(LAST_DIAGNOSTIC_REPORT, lastDiagnosticReport)
                .apply();
        showDiagnosticPanel(lastDiagnosticError, lastDiagnosticReport);
        statusLine.setText("运行失败 · 完整错误可复制，诊断可保存为 TXT");
    }

    private void showDiagnosticPanel(String error, String report) {
        lastDiagnosticError = error == null ? "" : error;
        lastDiagnosticReport = report == null ? "" : report;
        diagnosticPreview.setText(lastDiagnosticError);
        diagnosticPanel.setVisibility(View.VISIBLE);
    }

    private void clearDiagnosticError() {
        lastDiagnosticError = "";
        lastDiagnosticReport = "";
        diagnosticPanel.setVisibility(View.GONE);
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                .remove(LAST_DIAGNOSTIC_ERROR)
                .remove(LAST_DIAGNOSTIC_REPORT)
                .apply();
    }

    private static void appendReportField(StringBuilder out, String key, String value, int maxChars) {
        String text = value == null ? "<null>" : value;
        if (text.length() > maxChars) text = text.substring(0, maxChars) + "\n[该状态字段已截断]";
        out.append(key).append(": ").append(text).append("\n");
    }

    private String buildDiagnosticReport(String error) {
        StringBuilder out = new StringBuilder(8192);
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss Z", Locale.US);
        out.append("Ornith NPU diagnostic report\n");
        out.append("time: ").append(dateFormat.format(new Date())).append("\n");
        out.append("app_id: ").append(getPackageName()).append("\n");
        try {
            android.content.pm.PackageInfo info = getPackageManager().getPackageInfo(getPackageName(), 0);
            out.append("app_version: ").append(info.versionName).append("\n");
            out.append("version_code: ").append(info.getLongVersionCode()).append("\n");
        } catch (Throwable ignored) {
            out.append("app_version: <unavailable>\n");
        }
        out.append("device: ").append(Build.MANUFACTURER).append(" ").append(Build.MODEL)
                .append(" (").append(Build.DEVICE).append(")\n");
        out.append("android: ").append(Build.VERSION.RELEASE).append(" / API ")
                .append(Build.VERSION.SDK_INT).append("\n");
        out.append("runtime_rss_policy: hard_limit=3584MiB; report does not dump model weights\n");
        File model = new File(getFilesDir(), "models/" + MODEL_FILENAME);
        out.append("model_file: ").append(model.isFile() ? "present" : "missing").append("\n");
        if (model.isFile()) out.append("model_file_bytes: ").append(model.length()).append("\n");
        out.append("npu_ready: ").append(NpuRuntime.isReady()).append("\n");
        appendReportField(out, "npu_last_error", NpuRuntime.getLastError(), 4096);
        try {
            appendReportField(out, "npu_status", NpuRuntime.status(), 8192);
        } catch (Throwable t) {
            appendReportField(out, "npu_status_exception", t.toString(), 1024);
        }
        try {
            String nativeLogs = NpuRuntime.drainDiag();
            if (nativeLogs != null && !nativeLogs.isEmpty()) {
                final int maxNativeLogChars = 65536;
                if (nativeLogs.length() > maxNativeLogChars) {
                    nativeLogs = "[older native diagnostic lines omitted; keeping last 65536 characters]\n"
                            + nativeLogs.substring(nativeLogs.length() - maxNativeLogChars);
                }
                out.append("\n========== NATIVE DIAGNOSTIC LOG RING ==========\n");
                out.append(nativeLogs).append("\n");
                out.append("========== END NATIVE DIAGNOSTIC LOG RING ==========\n");
            }
        } catch (Throwable t) {
            appendReportField(out, "native_log_ring_exception", t.toString(), 1024);
        }
        out.append("model_loaded: ").append(Ornith15Runtime.isLoaded()).append("\n");
        try {
            appendReportField(out, "ornith_runtime_info", Ornith15Runtime.info(), 8192);
        } catch (Throwable t) {
            appendReportField(out, "ornith_runtime_info_exception", t.toString(), 1024);
        }
        out.append("\n========== FULL ERROR (verbatim) ==========\n");
        out.append(error == null ? "<null>" : error).append("\n");
        out.append("========== END FULL ERROR ==========\n");
        return out.toString();
    }

    private void copyDiagnosticError() {
        String text = lastDiagnosticError;
        if (text == null || text.isEmpty()) text = lastDiagnosticReport;
        if (text == null || text.isEmpty()) {
            Toast.makeText(this, "当前没有可复制的错误", Toast.LENGTH_SHORT).show();
            return;
        }
        ClipboardManager clipboard = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        if (clipboard == null) {
            Toast.makeText(this, "系统剪贴板不可用", Toast.LENGTH_SHORT).show();
            return;
        }
        clipboard.setPrimaryClip(ClipData.newPlainText("Ornith NPU 完整错误", text));
        Toast.makeText(this, "完整错误已复制", Toast.LENGTH_SHORT).show();
    }

    private void exportDiagnosticReport() {
        if (lastDiagnosticReport == null || lastDiagnosticReport.isEmpty()) {
            Toast.makeText(this, "当前没有诊断报告", Toast.LENGTH_SHORT).show();
            return;
        }
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("text/plain");
        intent.putExtra(Intent.EXTRA_TITLE, "ornith-npu-error-report.txt");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        try {
            startActivityForResult(intent, SAVE_DIAGNOSTIC_REPORT);
        } catch (Throwable t) {
            Toast.makeText(this, "无法创建报告文件：" + t.getClass().getSimpleName(), Toast.LENGTH_LONG).show();
        }
    }

    private void writeDiagnosticReport(Uri uri) {
        try (OutputStream out = getContentResolver().openOutputStream(uri, "w")) {
            if (out == null) throw new java.io.IOException("无法创建输出文件");
            out.write(lastDiagnosticReport.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            out.flush();
            Toast.makeText(this, "诊断报告已保存", Toast.LENGTH_LONG).show();
        } catch (Throwable t) {
            Toast.makeText(this, "保存报告失败：" + t.getClass().getSimpleName() + ": "
                    + String.valueOf(t.getMessage()), Toast.LENGTH_LONG).show();
        }
    }

    private void sendMessage() {
        if (generating) return;

        String text = input.getText().toString().trim();
        if (text.isEmpty()) return;

        addBubble("user", text);
        input.setText("");
        saveHistory();
        showEmptyStateIfNeeded();

        if (!NpuRuntime.isReady()) {
            addBubble("system",
                    "本地 NPU 尚未启动。先点右上角“导入模型”载入唯一的 Ornith-1.5-9B-MLX-4bit 模型文件。");
            saveHistory();
            return;
        }

        String modelPath = getSharedPreferences(PREFS, MODE_PRIVATE).getString(MODEL_PATH, "");
        if (modelPath.isEmpty() || !Ornith15Runtime.isLoaded()) {
            addBubble("system",
                    "MCNPU 已在线，但尚未加载 Ornith-1.5-9B。点右上角“导入模型”载入本地模型文件。");
            saveHistory();
            return;
        }

        final AgentTask task = new AgentTask(text);
        task.workspace = "chat";
        task.state = AgentTask.State.THINKING;
        final String prompt = buildAgentPrompt(text);

        generating = true;
        setSendEnabled(false);
        statusLine.setText("Ornith-1.5-9B · 生成中 · MCNPU");
        new Thread(() -> {
            String reply = null;
            try {
                reply = Ornith15Runtime.generate(prompt, 256);
                task.state = (reply != null && reply.startsWith("OK ORNITH15_GENERATE/1"))
                        ? AgentTask.State.FINAL : AgentTask.State.FAILED;
                task.steps++;
            } catch (Throwable t) {
                task.state = AgentTask.State.FAILED;
                reply = "ERR ORNITH15_RUNTIME " + t.getClass().getSimpleName();
            }
            final String result = reply;
            main.post(() -> {
                String shown = result != null && result.startsWith("OK ORNITH15_GENERATE/1 text=")
                        ? result.substring("OK ORNITH15_GENERATE/1 text=".length())
                        : result;
                addBubble("assistant", shown == null ? "推理失败" : shown);
                saveHistory();
                generating = false;
                setSendEnabled(true);
                updateRuntimeState();
            });
        }, "ornith-agent").start();
    }

    private void setSendEnabled(boolean on) {
        View send = findViewById(R.id.send);
        if (send == null) return;
        send.setEnabled(on);
        if (send instanceof android.widget.Button) {
            ((android.widget.Button) send).setText(on ? "发送" : "…");
        }
    }

    private String buildAgentPrompt(String current) {
        String raw = getSharedPreferences(PREFS, MODE_PRIVATE).getString(HISTORY, "[]");
        try {
            JSONArray history = new JSONArray(raw);
            String context = AgentContext.buildContext(history, current, MAX_CONTEXT_MESSAGES, MAX_CONTEXT_CHARS);
            if (context.isEmpty()) return current;
            // Keep the current USER turn explicit and last so prompt-budget trimming can never hide it.
            return "Conversation context:\n" + context + "\nCurrent USER message:\n" + current;
        } catch (Throwable ignored) {
            return current;
        }
    }

    private void importOrnithModel() {
        try {
            Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            i.addCategory(Intent.CATEGORY_OPENABLE);
            // Deliberately permissive: EXTRA_MIME_TYPES with "*/*" is illegal and makes some
            // pickers filter everything out. The real format check is the header probe below.
            i.setType("*/*");
            i.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivityForResult(i, PICK_MODEL);
        } catch (Throwable t) {
            Toast.makeText(this, "无法打开模型选择器: " + t.getClass().getSimpleName(), Toast.LENGTH_SHORT).show();
        }
    }

    /** SAF gives a document id in getLastPathSegment(), not a file name. Query the real one. */
    private String resolveDisplayName(Uri uri) {
        String name = null;
        try (Cursor c = getContentResolver().query(uri, null, null, null, null)) {
            if (c != null && c.moveToFirst()) {
                int idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (idx >= 0) name = c.getString(idx);
            }
        } catch (Throwable ignored) {}
        if (name == null) {
            String seg = uri.getLastPathSegment();
            if (seg != null) {
                int slash = seg.lastIndexOf('/');
                int colon = seg.lastIndexOf(':');
                int cut = Math.max(slash, colon);
                name = cut >= 0 ? seg.substring(cut + 1) : seg;
            }
        }
        return name == null ? "" : name;
    }

    private long querySize(Uri uri) {
        try (Cursor c = getContentResolver().query(uri, null, null, null, null)) {
            if (c != null && c.moveToFirst()) {
                int idx = c.getColumnIndex(OpenableColumns.SIZE);
                if (idx >= 0 && !c.isNull(idx)) return c.getLong(idx);
            }
        } catch (Throwable ignored) {}
        return -1L;
    }

    /**
     * Real safetensors check: 8-byte little-endian header length followed by the JSON '{'.
     * Extension names are not reliable on Android content URIs, so we never trust them.
     */
    private static boolean looksLikeSafetensors(InputStream in) throws java.io.IOException {
        byte[] head = new byte[9];
        int got = 0;
        while (got < head.length) {
            int r = in.read(head, got, head.length - got);
            if (r < 0) break;
            got += r;
        }
        if (got < 9) return false;
        long headerLen = 0;
        for (int i = 7; i >= 0; i--) headerLen = (headerLen << 8) | (head[i] & 0xFFL);
        return headerLen > 0 && headerLen < (200L * 1024 * 1024) && head[8] == '{';
    }

    private void stage(String text, int progress) {
        main.post(() -> {
            importStage.setVisibility(View.VISIBLE);
            importStage.setText(text);
            importProgress.setVisibility(View.VISIBLE);
            if (progress >= 0) {
                importProgress.setIndeterminate(false);
                importProgress.setProgress(progress);
            } else {
                importProgress.setIndeterminate(true);
            }
        });
    }

    private void clearStage() {
        main.post(() -> {
            importStage.setVisibility(View.GONE);
            importProgress.setVisibility(View.GONE);
        });
    }

    private static String humanBytes(long bytes) {
        if (bytes <= 0) return "未知大小";
        double gb = bytes / 1073741824.0;
        if (gb >= 1.0) return String.format(Locale.US, "%.2f GB", gb);
        double mb = bytes / 1048576.0;
        if (mb >= 1.0) return String.format(Locale.US, "%.1f MB", mb);
        return String.format(Locale.US, "%.0f KB", bytes / 1024.0);
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == SAVE_DIAGNOSTIC_REPORT) {
            if (resultCode == RESULT_OK && data != null && data.getData() != null) {
                writeDiagnosticReport(data.getData());
            }
            return;
        }
        if (requestCode != PICK_MODEL || resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        new Thread(() -> {
            try {
                String displayName = resolveDisplayName(uri);
                long sourceBytes = querySize(uri);

                File dir = new File(getFilesDir(), "models");
                if (!dir.exists() && !dir.mkdirs()) throw new java.io.IOException("无法创建模型目录");

                stage("正在校验 " + displayName + " (" + humanBytes(sourceBytes) + ")…", 0);

                // Header probe first: it is the only trustworthy format signal here.
                try (InputStream probe = getContentResolver().openInputStream(uri)) {
                    if (probe == null) throw new java.io.IOException("无法打开所选文件");
                    if (!looksLikeSafetensors(probe))
                        throw new java.io.IOException(
                                "这不是 safetensors 文件（文件头不符合格式）。请选择 Ornith-1.5-9B-MLX-4bit 的 .safetensors");
                }

                File dst = new File(dir, MODEL_FILENAME);
                File tokenizer = new File(dir, MODEL_FILENAME + ".tokenizer");
                try (InputStream asset = getAssets().open("ornith15.tokenizer");
                     FileOutputStream tokenOut = new FileOutputStream(tokenizer)) {
                    byte[] tokenBuf = new byte[64 * 1024];
                    int tokenN;
                    while ((tokenN = asset.read(tokenBuf)) != -1) tokenOut.write(tokenBuf, 0, tokenN);
                } catch (Throwable t) {
                    throw new java.io.IOException("词表资源缺失（ornith15.tokenizer）: " + t.getClass().getSimpleName());
                }

                long copied = 0;
                long lastPost = 0;
                try (InputStream in = getContentResolver().openInputStream(uri);
                     FileOutputStream out = new FileOutputStream(dst)) {
                    if (in == null) throw new java.io.IOException("无法打开模型文件");
                    byte[] buf = new byte[1024 * 1024];
                    int n;
                    while ((n = in.read(buf)) != -1) {
                        out.write(buf, 0, n);
                        copied += n;
                        if (sourceBytes > 0 && copied - lastPost > (8L * 1024 * 1024)) {
                            lastPost = copied;
                            final int pct = (int) Math.min(100, copied * 100 / sourceBytes);
                            final String done = humanBytes(copied);
                            final String total = humanBytes(sourceBytes);
                            stage("正在复制 " + done + " / " + total, pct);
                        }
                    }
                }
                stage("已复制 " + humanBytes(copied) + "，正在加载模型…", -1);

                String r = Ornith15Runtime.load(dst.getAbsolutePath(), 65536);
                if (!r.startsWith("OK ORNITH15_RUNTIME/1")) {
                    // Surface the native reason verbatim: it carries npu_probe / arch / memory detail.
                    throw new java.io.IOException(r);
                }
                getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                        .putString(MODEL_PATH, dst.getAbsolutePath()).apply();
                main.post(() -> {
                    clearStage();
                    clearDiagnosticError();
                    updateRuntimeState();
                    Toast.makeText(this, "Ornith-1.5-9B-MLX-4bit 已加载", Toast.LENGTH_LONG).show();
                });
            } catch (Throwable t) {
                final String msg = t.getMessage() == null ? t.getClass().getSimpleName() : t.getMessage();
                main.post(() -> {
                    clearStage();
                    updateRuntimeState();
                    showDiagnosticError(msg);
                    addBubble("system", "模型导入失败：" + msg);
                    saveHistory();
                    Toast.makeText(this, "模型导入失败：完整报错可复制或导出 TXT", Toast.LENGTH_LONG).show();
                });
            }
        }, "ornith-model-load").start();
    }

    private void addBubble(String role, String text) {
        TextView bubble = new TextView(this);
        bubble.setText(text);
        bubble.setTextSize(16);
        bubble.setTextColor(Color.rgb(17, 24, 39));
        bubble.setPadding(16, 12, 16, 12);
        bubble.setTypeface(Typeface.DEFAULT, Typeface.NORMAL);
        bubble.setTag(role);

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.setMargins(8, 6, 8, 6);

        if ("user".equals(role)) {
            bubble.setBackgroundColor(Color.rgb(225, 235, 255));
            lp.gravity = Gravity.END;
        } else if ("system".equals(role)) {
            bubble.setTextSize(13);
            bubble.setTextColor(Color.rgb(71, 85, 105));
            bubble.setBackgroundColor(Color.rgb(238, 241, 245));
            lp.gravity = Gravity.CENTER_HORIZONTAL;
        } else {
            bubble.setBackgroundColor(Color.WHITE);
            lp.gravity = Gravity.START;
        }

        messages.addView(bubble, lp);
        scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
    }

    private void loadHistory() {
        messages.removeAllViews();
        String raw = getSharedPreferences(PREFS, MODE_PRIVATE).getString(HISTORY, "[]");
        try {
            JSONArray arr = new JSONArray(raw);
            for (int i = 0; i < arr.length(); i++) {
                JSONObject m = arr.getJSONObject(i);
                addBubble(m.optString("role", "assistant"), m.optString("text", ""));
            }
        } catch (Throwable ignored) {
            getSharedPreferences(PREFS, MODE_PRIVATE).edit().remove(HISTORY).apply();
        }
    }

    private void saveHistory() {
        JSONArray arr = new JSONArray();
        for (int i = 0; i < messages.getChildCount(); i++) {
            View v = messages.getChildAt(i);
            if (!(v instanceof TextView)) continue;
            TextView t = (TextView) v;
            String role = t.getTag() instanceof String ? (String) t.getTag() : "assistant";
            try {
                JSONObject m = new JSONObject();
                m.put("role", role);
                m.put("text", t.getText().toString());
                arr.put(m);
            } catch (Throwable ignored) {}
        }
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(HISTORY, arr.toString()).apply();
    }
}
