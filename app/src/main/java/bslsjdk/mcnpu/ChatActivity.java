package bslsjdk.ornithnpu;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.graphics.Color;
import android.graphics.Typeface;
import android.view.Gravity;
import android.view.View;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import android.net.Uri;
import android.content.Intent;
import java.io.File;
import java.io.FileOutputStream;
import java.util.Locale;
import org.json.JSONArray;
import org.json.JSONObject;

public final class ChatActivity extends Activity {
    private static final String PREFS = "chat";
    private static final String HISTORY = "history";
    private static final String MODEL_PATH = "model_path";
    private static final String MODEL_FILENAME = "ornith-1.5-9b-mlx-4bit.safetensors";
    private static final int PICK_MODEL = 4201;
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

        findViewById(R.id.send).setOnClickListener(v -> sendMessage());
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
        updateRuntimeState();
        findViewById(R.id.importModel).setOnClickListener(v -> importOrnithModel());
        initLocalRuntime();
    }

    private void initLocalRuntime() {
        new Thread(() -> {
            boolean ok = NpuRuntime.init(getApplicationContext());
            if (ok) {
                String saved = getSharedPreferences(PREFS, MODE_PRIVATE)
                        .getString(MODEL_PATH, "");
                if (!saved.isEmpty() && new File(saved).isFile()) {
                    try { Ornith15Runtime.load(saved, 65536); } catch (Throwable ignored) {}
                }
            }
            main.post(() -> {
                updateRuntimeState();
                if (ok && Ornith15Runtime.isLoaded())
                    statusLine.setText("本地 NPU 在线 · Ornith-1.5-9B");
                else if (ok)
                    statusLine.setText("本地 NPU 在线 · 等待 Ornith-1.5-9B");
            });
        }, "mcnpu-init").start();
    }
    @Override protected void onResume() {
        super.onResume();
        updateRuntimeState();
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

    private void sendMessage() {
        if (generating) return;

        String text = input.getText().toString().trim();
        if (text.isEmpty()) return;

        addBubble("user", text);
        input.setText("");
        saveHistory();

        if (!NpuRuntime.isReady()) {
            addBubble("system",
                    "本地 NPU 尚未启动。先点顶部模型名称导入唯一的 Ornith-1.5-9B-MLX-4bit 模型文件。");
            saveHistory();
            return;
        }

        String modelPath = getSharedPreferences(PREFS, MODE_PRIVATE).getString(MODEL_PATH, "");
        if (modelPath.isEmpty() || !Ornith15Runtime.isLoaded()) {
            addBubble("system",
                    "MCNPU 已在线，但尚未加载 Ornith-1.5-9B。点击顶部“Ornith AI”导入唯一的本地模型文件。");
            saveHistory();
            return;
        }

        final AgentTask task = new AgentTask(text);
        task.workspace = "chat";
        task.state = AgentTask.State.THINKING;
        final String prompt = buildAgentPrompt(text);

        generating = true;
        statusLine.setText("Ornith-1.5-9B · 思考中 · MCNPU");
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
                updateRuntimeState();
            });
        }, "ornith-agent").start();
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
            i.setType("*/*");
            i.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{"application/octet-stream", "application/x-safetensors", "*/*"});
            startActivityForResult(i, PICK_MODEL);
        } catch (Throwable t) {
            Toast.makeText(this, "无法打开模型选择器: " + t.getClass().getSimpleName(), Toast.LENGTH_SHORT).show();
        }
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_MODEL || resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        new Thread(() -> {
            try {
                File dir = new File(getFilesDir(), "models");
                if (!dir.exists() && !dir.mkdirs()) throw new java.io.IOException("无法创建模型目录");
                String sourceName = uri.getLastPathSegment();
                String lower = sourceName == null ? "" : sourceName.toLowerCase(java.util.Locale.ROOT);
                if (!lower.endsWith(".safetensors"))
                    throw new java.io.IOException("这里只接受 Ornith-1.5-9B-MLX-4bit.safetensors");
                String dstName = MODEL_FILENAME;
                File dst = new File(dir, dstName);
                File tokenizer = new File(dir, MODEL_FILENAME + ".tokenizer");
                try (java.io.InputStream asset = getAssets().open("ornith15.tokenizer");
                     FileOutputStream tokenOut = new FileOutputStream(tokenizer)) {
                    byte[] tokenBuf = new byte[64 * 1024];
                    int tokenN;
                    while ((tokenN = asset.read(tokenBuf)) != -1) tokenOut.write(tokenBuf, 0, tokenN);
                }
                try (java.io.InputStream in = getContentResolver().openInputStream(uri);
                     FileOutputStream out = new FileOutputStream(dst)) {
                    if (in == null) throw new java.io.IOException("无法打开模型文件");
                    byte[] buf = new byte[1024 * 1024];
                    int n;
                    while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
                }
                String r = Ornith15Runtime.load(dst.getAbsolutePath(), 65536);
                if (!r.startsWith("OK ORNITH15_RUNTIME/1")) throw new java.io.IOException(r);
                getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                        .putString(MODEL_PATH, dst.getAbsolutePath()).apply();
                main.post(() -> {
                    statusLine.setText("Ornith-1.5-9B · 本地模型已识别 · MCNPU");
                    updateRuntimeState();
                    Toast.makeText(this, "Ornith-1.5-9B-MLX-4bit 已加载", Toast.LENGTH_SHORT).show();
                });
            } catch (Throwable t) {
                main.post(() -> Toast.makeText(
                        this, "模型导入失败: " + t.getMessage(), Toast.LENGTH_LONG).show());
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
