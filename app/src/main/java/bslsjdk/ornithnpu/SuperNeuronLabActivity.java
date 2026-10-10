package bslsjdk.ornithnpu;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Debug;
import android.os.Looper;
import android.graphics.Color;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Locale;

/**
 * Isolated smoke-test UI for the super-neuron next-token trainer.
 * Input values are integer token IDs, not natural-language text; no tokenizer
 * is connected here yet. The normal ChatActivity remains the app launcher.
 */
public final class SuperNeuronLabActivity extends Activity {
    private final Handler main = new Handler(Looper.getMainLooper());
    private EditText vocabularyField, embeddingField, hiddenField, epochsField, learningRateField, sequenceField, storyField;
    private TextView resultView;
    private Button trainButton;
    private SuperNeuronLanguageTrainer trainer;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(16), dp(16), dp(20));
        root.setBackgroundColor(Color.rgb(246, 247, 249));
        scroll.addView(root);
        setContentView(scroll);

        TextView title = new TextView(this);
        title.setText("超级神经元 · 训练实验");
        title.setTextColor(Color.rgb(17, 24, 39));
        title.setTextSize(22);
        title.setTypeface(null, 1);
        root.addView(title);

        TextView warning = new TextView(this);
        warning.setText("实验包括字符级故事预测、单个模型内的局部梯度/资格迹，以及模型池精英选择与变异重组。每 1000 步记录 PSS/RSS；PSS 超过 2.5 GiB 时保存检查点并停止。模型估算存储预算 16 MiB，手机进程总内存仍必须低于 4 GiB。");
        warning.setTextColor(Color.rgb(153, 72, 16));
        warning.setTextSize(13);
        warning.setPadding(0, dp(8), 0, dp(12));
        root.addView(warning);

        vocabularyField = addField(root, "词表大小（输出数量）", "8");
        embeddingField = addField(root, "Token embedding 维度", "8");
        hiddenField = addField(root, "隐藏状态维度", "16");
        epochsField = addField(root, "训练轮数", "100");
        learningRateField = addField(root, "学习率（0.001 到 0.05）", "0.02");
        sequenceField = addField(root, "Token ID 序列（空格分隔）", "0 1 0 1 0 1 0 1 0 1 0 1");

        storyField = addMultilineField(root, "字符级儿童故事（原文）",
                "小兔子住在森林边的小木屋里。一天早晨，它发现门口有一颗闪闪发光的种子。小兔把种子种进土里，每天浇水，也和小鸟一起等它发芽。过了几天，嫩绿的小芽探出头来。大风来时，小兔用树枝挡风，下雨时又把积水轻轻排开。后来种子长成一棵小树，结出了甜甜的果子。小兔把果子分给小鸟、刺猬和路过的鹿。大家一起种下更多种子，让森林变得更加茂盛。小兔明白，耐心照料和分享，会让小小的善意慢慢长大。");

        Button storyButton = new Button(this);
        storyButton.setText("字符故事训练：10,000 步 + 内存探针 + 进化");
        storyButton.setAllCaps(false);
        LinearLayout.LayoutParams storyParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(54));
        storyParams.topMargin = dp(8);
        root.addView(storyButton, storyParams);
        storyButton.setOnClickListener(v -> trainCharacterStory(storyButton));

        trainButton = new Button(this);
        trainButton.setText("训练并验证下一 Token 预测");
        trainButton.setAllCaps(false);
        LinearLayout.LayoutParams buttonParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(52));
        buttonParams.topMargin = dp(12);
        root.addView(trainButton, buttonParams);

        Button resetButton = new Button(this);
        resetButton.setText("清空实验结果");
        resetButton.setAllCaps(false);
        root.addView(resetButton, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(48)));

        resultView = new TextView(this);
        resultView.setText("等待实验。输出维度将严格等于词表大小。");
        resultView.setTextColor(Color.rgb(31, 41, 55));
        resultView.setTextSize(14);
        resultView.setPadding(dp(12), dp(12), dp(12), dp(12));
        resultView.setBackgroundColor(Color.WHITE);
        LinearLayout.LayoutParams resultParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        resultParams.topMargin = dp(12);
        root.addView(resultView, resultParams);

        trainButton.setOnClickListener(v -> train());
        resetButton.setOnClickListener(v -> {
            trainer = null;
            resultView.setText("实验结果已清空。下一次训练会创建新的参数实例。");
        });
    }

    private EditText addField(LinearLayout root, String labelText, String defaultValue) {
        TextView label = new TextView(this);
        label.setText(labelText);
        label.setTextColor(Color.rgb(55, 65, 81));
        label.setTextSize(13);
        LinearLayout.LayoutParams labelParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        labelParams.topMargin = dp(8);
        root.addView(label, labelParams);

        EditText field = new EditText(this);
        field.setSingleLine(true);
        field.setText(defaultValue);
        field.setTextSize(15);
        field.setPadding(dp(10), 0, dp(10), 0);
        field.setBackgroundColor(Color.WHITE);
        root.addView(field, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(48)));
        return field;
    }

    private EditText addMultilineField(LinearLayout root, String labelText, String defaultValue) {
        TextView label = new TextView(this);
        label.setText(labelText);
        label.setTextColor(Color.rgb(55, 65, 81));
        label.setTextSize(13);
        root.addView(label, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        EditText field = new EditText(this);
        field.setText(defaultValue);
        field.setTextSize(14);
        field.setGravity(Gravity.TOP | Gravity.START);
        field.setMinLines(5);
        field.setMaxLines(10);
        field.setPadding(dp(10), dp(8), dp(10), dp(8));
        field.setBackgroundColor(Color.WHITE);
        root.addView(field, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        return field;
    }

    private void train() {
        final int vocab, embedding, hidden, epochs;
        final double lr;
        final int[] tokens;
        try {
            vocab = Integer.parseInt(vocabularyField.getText().toString().trim());
            embedding = Integer.parseInt(embeddingField.getText().toString().trim());
            hidden = Integer.parseInt(hiddenField.getText().toString().trim());
            epochs = Integer.parseInt(epochsField.getText().toString().trim());
            lr = Double.parseDouble(learningRateField.getText().toString().trim());
            tokens = parseTokens(sequenceField.getText().toString(), vocab);
            if ((long) (tokens.length - 1) * epochs > 100_000L)
                throw new IllegalArgumentException("实验页面每次最多训练 100,000 个 token 目标，避免长时间占用手机 CPU");
        } catch (Exception e) {
            Toast.makeText(this, e.getMessage() == null ? "配置无效" : e.getMessage(),
                    Toast.LENGTH_LONG).show();
            return;
        }

        trainButton.setEnabled(false);
        resultView.setText("训练中… 正在计算训练前损失，请勿连续点击。");
        new Thread(() -> {
            try {
                SuperNeuronLanguageTrainer candidate =
                        new SuperNeuronLanguageTrainer(vocab, embedding, hidden, 20261010L);
                double before = candidate.evaluateSequence(tokens);
                double trainLoss = candidate.trainSequence(tokens, epochs, lr);
                double after = candidate.evaluateSequence(tokens);
                candidate.resetContext();
                int prediction = candidate.predictNextToken(tokens[tokens.length - 1]);
                long bytes = candidate.estimatedStorageBytes();
                String report = String.format(Locale.US,
                        "实验完成\n\n词表大小 / 固定输出数量：%d\n实际预测 ID：%d\n"
                                + "序列长度：%d\n训练轮数：%d\n训练 token 目标数：%d\n"
                                + "训练时平均交叉熵：%.6f\n训练前序列损失：%.6f\n训练后序列损失：%.6f\n"
                                + "损失变化：%.6f\n参数标量数：%d\n估算模型与工作区：%.2f MiB\n\n"
                                + "注意：这是同一小序列上的拟合测试，不代表泛化或自然语言能力。"
                                + "当前没有 tokenizer，也没有独立验证集。",
                        candidate.getVocabularySize(), prediction, tokens.length, epochs,
                        candidate.getTrainedTokenTargets(), trainLoss, before, after, after - before,
                        candidate.getParameterCount(), bytes / (1024.0 * 1024.0));
                main.post(() -> {
                    trainer = candidate;
                    resultView.setText(report);
                    trainButton.setEnabled(true);
                });
            } catch (Throwable e) {
                main.post(() -> {
                    resultView.setText("训练失败：" + e.getClass().getSimpleName()
                            + ": " + String.valueOf(e.getMessage()));
                    trainButton.setEnabled(true);
                });
            }
        }, "super-neuron-training").start();
    }

    private static int[] parseTokens(String text, int vocabularySize) {
        if (text == null || text.trim().isEmpty())
            throw new IllegalArgumentException("Token ID 序列不能为空");
        String[] parts = text.trim().split("\\s+");
        if (parts.length < 2) throw new IllegalArgumentException("至少输入两个 token ID");
        if (parts.length > 20001) throw new IllegalArgumentException("序列长度不能超过 20001");
        int[] tokens = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                tokens[i] = Integer.parseInt(parts[i]);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Token ID 不是整数：" + parts[i]);
            }
            if (tokens[i] < 0 || tokens[i] >= vocabularySize)
                throw new IllegalArgumentException("Token ID 越界：" + tokens[i]
                        + "，有效范围是 0 到 " + (vocabularySize - 1));
        }
        return tokens;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
