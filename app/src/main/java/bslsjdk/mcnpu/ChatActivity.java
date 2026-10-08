package bslsjdk.mcnpu;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.content.Context;
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
import java.io.FileInputStream;
import java.io.FileOutputStream;
import org.json.JSONArray;
import org.json.JSONObject;

public final class ChatActivity extends Activity {
    private static final String PREFS = "chat";
    private static final String HISTORY = "history";
    private static final String MODEL_PATH = "model_path";
    private static final int PICK_MODEL = 4201;
    private LinearLayout messages;
    private ScrollView scroll;
    private EditText input;
    private TextView runtimeState;
    private TextView statusLine;
    private final Handler main = new Handler(Looper.getMainLooper());

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.activity_chat);

        messages = findViewById(R.id.messages);
        scroll = findViewById(R.id.messagesScroll);
        input = findViewById(R.id.input);
        runtimeState = findViewById(R.id.runtimeState);
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
        findViewById(R.id.chatTitle).setOnClickListener(v -> chooseModel());
        initLocalRuntime();
    }

    private void initLocalRuntime() {
        new Thread(() -> {
            boolean ok = NpuRuntime.init(getApplicationContext());
            main.post(() -> {
                updateRuntimeState();
                if (ok) statusLine.setText("本地 NPU 在线 · Ornith-1.5-9B");
            });
        }, "mcnpu-init").start();
    }

    @Override protected void onResume() {
        super.onResume();
        updateRuntimeState();
    }

    private void updateRuntimeState() {
        boolean npu = NpuRuntime.isReady();
        runtimeState.setText(npu ? "NPU 在线" : "本地");
        runtimeState.setTextColor(npu ? Color.rgb(22, 120, 75) : Color.rgb(100, 116, 139));
        statusLine.setText(npu
                ? "Ornith-1.5-9B · MCNPU HTP V73"
                : "Ornith-1.5-9B · 等待本地推理内核");
    }

    private void sendMessage() {
        String text = input.getText().toString().trim();
        if (text.isEmpty()) return;

        addBubble("user", text);
        input.setText("");
        saveHistory();

        if (!NpuRuntime.isReady()) {
            addBubble("system",
                    "本地 NPU 尚未启动。先点顶部模型名称选择本地 Ornith-1.5-9B 模型文件。");
            saveHistory();
            return;
        }

        String modelPath = getSharedPreferences(PREFS, MODE_PRIVATE).getString(MODEL_PATH, "");
        if (modelPath.isEmpty() || !Ornith15Runtime.isLoaded()) {
            addBubble("system", "MCNPU 已在线，但尚未加载 Ornith-1.5-9B。点击顶部“Ornith AI”选择本地模型文件。");
            saveHistory();
            return;
        }
        final String prompt = text;
        new Thread(() -> {
            String reply = Ornith15Runtime.generate(prompt, 256);
            String shown = reply != null && reply.startsWith("OK ORNITH15_GENERATE/1 text=")
                    ? reply.substring("OK ORNITH15_GENERATE/1 text=".length())
                    : reply;
            main.post(() -> { addBubble("assistant", shown == null ? "推理失败" : shown); saveHistory(); });
        }, "ornith-generate").start();
    }

    private void chooseModel() {
        try {
            Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            i.addCategory(Intent.CATEGORY_OPENABLE);
            i.setType("*/*");
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
                String dstName = lower.endsWith(".safetensors") ? "ornith-1.5-9b-mlx-4bit.safetensors" : "ornith-1.5-9b.gguf";
                File dst = new File(dir, dstName);
                try (java.io.InputStream in = getContentResolver().openInputStream(uri);
                     FileOutputStream out = new FileOutputStream(dst)) {
                    if (in == null) throw new java.io.IOException("无法打开模型文件");
                    byte[] buf = new byte[1024 * 1024]; int n;
                    while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
                }
                String r = Ornith15Runtime.load(dst.getAbsolutePath(), 65536);
                if (!r.startsWith("OK ORNITH15_RUNTIME/1")) throw new java.io.IOException(r);
                getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(MODEL_PATH, dst.getAbsolutePath()).apply();
                main.post(() -> { statusLine.setText("Ornith-1.5-9B · 本地模型已识别 · MCNPU"); Toast.makeText(this, "模型文件已识别", Toast.LENGTH_SHORT).show(); });
            } catch (Throwable t) {
                main.post(() -> Toast.makeText(this, "模型加载失败: " + t.getMessage(), Toast.LENGTH_LONG).show());
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
