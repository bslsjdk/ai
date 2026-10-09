package bslsjdk.ornithnpu;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Debug;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Date;
import java.text.SimpleDateFormat;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Native Android workbench for the bounded multi-input/multi-output neuron network.
 * It intentionally does not load a language model. The launcher is declared in
 * AndroidManifest.xml and must point here.
 */
public final class NeuronLabActivity extends Activity {
    private static final int REQ_IMPORT = 6010;
    private static final int REQ_EXPORT = 6011;
    private static final int MAX_IMPORT_BYTES = 16 * 1024 * 1024;
    private static final int MAX_VISIBLE_NEURONS = 48;

    private final ExecutorService worker = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "aimeng-neuron-workspace");
        t.setDaemon(true);
        return t;
    });
    private final ExecutorService npuWorker = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "aimeng-npu-diagnostic");
        t.setDaemon(true);
        return t;
    });

    private LinearLayout shell;
    private FrameLayout pageHost;
    private LinearLayout navRow;
    private TextView globalStatus;
    private TextView selectedCountLabel;
    private ProgressBar activeTrainProgress;
    private TextView activeTrainLabel;
    private TextView pageTrainingReport;

    private volatile NeuronWorkspace workspace;
    private String currentPage = "home";
    private String npuStatus = "NPU 正在初始化；不影响 CPU 训练。";
    private String trainingStatus = "尚未开始训练。";
    private String lastPredictionReport = "输入一组数值，观察隐藏神经元激活与各输出的贡献。";
    private String lastInputText = "0";
    private volatile boolean training;
    private volatile boolean cancelTraining;
    private volatile boolean npuDiagnosticRunning;
    private final Set<String> selectedIds = new LinkedHashSet<>();

    private EditText taskNameField;
    private EditText epochsField;
    private EditText learningRateField;
    private EditText inputsField;
    private EditText hiddenField;
    private EditText outputsField;
    private EditText inputVectorField;
    private EditText neuronFilterField;
    private String currentNeuronFilter = "";
    private String pendingImportType = "dataset";
    private String pendingExportContent;
    private String pendingExportMime;
    private String pendingExportName;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        loadInternalWorkspace();
        buildShell();
        showPage("home");
        initializeNpuStatus();
    }

    private void loadInternalWorkspace() {
        File file = new File(getFilesDir(), "neuron-workspace.json");
        if (file.isFile()) {
            try {
                String content = readFile(file, MAX_IMPORT_BYTES);
                workspace = NeuronWorkspace.fromJson(new JSONObject(content));
                trainingStatus = "已恢复本机工作区。";
            } catch (Throwable error) {
                workspace = NeuronWorkspace.createDefault();
                trainingStatus = "旧工作区读取失败，已创建新工作区：" + shortError(error);
            }
        } else {
            workspace = NeuronWorkspace.createDefault();
            persistWorkspaceNow();
        }
    }

    private void buildShell() {
        shell = new LinearLayout(this);
        shell.setOrientation(LinearLayout.VERTICAL);
        shell.setBackgroundColor(0xFFF3F5F8);

        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.VERTICAL);
        header.setPadding(dp(18), dp(12), dp(18), dp(10));
        header.setBackgroundColor(Color.WHITE);
        header.addView(label("AIMENG · 神经元工作台", 23, true));
        TextView subtitle = label("多输入 / 多输出 · 本地训练 · 神经元独立保存", 12, false);
        subtitle.setTextColor(0xFF667085);
        header.addView(subtitle, params(-1, -2, 0, 3, 0, 0));
        globalStatus = label("工作区自动保存到应用内部存储。", 11, false);
        globalStatus.setTextColor(0xFF287D72);
        header.addView(globalStatus, params(-1, -2, 0, 5, 0, 0));
        shell.addView(header, params(-1, -2, 0, 0, 0, 0));

        pageHost = new FrameLayout(this);
        shell.addView(pageHost, new LinearLayout.LayoutParams(-1, 0, 1));

        HorizontalScrollView navScroll = new HorizontalScrollView(this);
        navScroll.setHorizontalScrollBarEnabled(false);
        navScroll.setBackgroundColor(Color.WHITE);
        navRow = new LinearLayout(this);
        navRow.setOrientation(LinearLayout.HORIZONTAL);
        String[][] pages = {
                {"home", "总览"}, {"neurons", "神经元"}, {"network", "网络图"},
                {"train", "训练任务"}, {"run", "运行"}, {"data", "数据/保存"}
        };
        for (String[] entry : pages) {
            Button b = new Button(this);
            b.setText(entry[1]);
            b.setAllCaps(false);
            b.setTextSize(12);
            b.setMinHeight(dp(48));
            b.setPadding(dp(8), 0, dp(8), 0);
            b.setOnClickListener(v -> showPage(entry[0]));
            navRow.addView(b, new LinearLayout.LayoutParams(dp(88), dp(54)));
        }
        navScroll.addView(navRow);
        shell.addView(navScroll, params(-1, 54, 0, 0, 0, 0));
        setContentView(shell);
    }

    private void showPage(String page) {
        currentPage = page;
        if (navRow != null) {
            for (int i = 0; i < navRow.getChildCount(); i++) {
                View v = navRow.getChildAt(i);
                if (v instanceof Button) {
                    Button b = (Button) v;
                    b.setEnabled(!training);
                    boolean active = page.equals(pageForNavIndex(i));
                    b.setTextColor(active ? 0xFF155E58 : 0xFF475467);
                    b.setBackground(tintedRound(active ? 0xFFDDF4F0 : 0xFFFFFFFF,
                            active ? 0xFFB9E4DC : 0xFFFFFFFF, 0));
                }
            }
        }

        pageHost.removeAllViews();
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setClipToPadding(false);
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(dp(14), dp(14), dp(14), dp(22));
        scroll.addView(content);
        pageHost.addView(scroll, new FrameLayout.LayoutParams(-1, -1));

        switch (page) {
            case "neurons": buildNeuronsPage(content); break;
            case "network": buildNetworkPage(content); break;
            case "train": buildTrainingPage(content); break;
            case "run": buildRunPage(content); break;
            case "data": buildDataPage(content); break;
            default: buildHomePage(content); break;
        }
    }

    private String pageForNavIndex(int index) {
        switch (index) {
            case 1: return "neurons";
            case 2: return "network";
            case 3: return "train";
            case 4: return "run";
            case 5: return "data";
            default: return "home";
        }
    }

    private void buildHomePage(LinearLayout content) {
        LinearLayout status = card(content, "工作区状态");
        addText(status, workspace.summary(), 14, false);
        addText(status, "工作区估算占用：" + formatMiB(workspace.getEstimatedBytes())
                + " / " + formatMiB(NeuronWorkspace.MAX_WORKSPACE_BUDGET_BYTES) + " MiB", 12, true);
        addText(status, "进程内存：" + memoryStatus(), 11, false);
        addText(status, "硬约束：整个应用运行时 RAM < 4096 MiB；发布目标 < 3800 MiB。4GB 是硬上限，不是分给神经元的全部配额。", 12, true);
        addText(status, "NPU： " + npuStatus, 11, false);

        LinearLayout quick = card(content, "快捷操作");
        addActionRow(quick, new String[]{"开始训练", "运行网络", "查看网络图"}, new Runnable[]{
                () -> showPage("train"), () -> showPage("run"), () -> showPage("network")
        });
        addActionRow(quick, new String[]{"管理神经元", "保存工作区", "NPU 诊断"}, new Runnable[]{
                () -> showPage("neurons"), this::persistWorkspaceWithToast, this::runNpuDiagnostic
        });
        Button evolve = primaryButton("资源竞争 · 安全进化一代");
        evolve.setOnClickListener(v -> runEvolution());
        quick.addView(evolve, params(-1, 48, 0, 8, 0, 0));

        LinearLayout how = card(content, "当前网络如何工作");
        addText(how, "输入层接收一组数值 → 活动隐藏神经元计算 tanh(加权输入 + 偏置) → 多个输出节点汇总各神经元的贡献。", 13, false);
        addText(how, "在“运行”页能看到每个隐藏单元的激活值，以及它对 y0、y1 等输出的贡献；“神经元”页可单独禁用、评分、保存、导出或批量操作。", 13, false);
        addText(how, "当前是可解释的数值任务网络，不是语言模型。它不能直接理解任意中文指令；复杂自然语言训练需要后续文本编码和任务数据管线。", 12, false);

        LinearLayout report = card(content, "最近一次训练 / 进化");
        addText(report, trainingStatus, 12, false);
        addText(report, workspace.lastReport, 12, false);
        LinearLayout saving = card(content, "保存和恢复");
        addText(saving, "模型、权重、训练数据、神经元库、训练历史和事件记录会自动保存到应用内部的 neuron-workspace.json。导出文件则由你通过系统文件选择器选择位置。", 12, false);
        addText(saving, "单神经元/批量神经元文件可以独立保存和重新导入。导入维度不兼容时会先询问，不会静默覆盖现有网络。", 12, false);
    }

    private void buildNeuronsPage(LinearLayout content) {
        LinearLayout overview = card(content, "单元管理");
        addText(overview, "原始评分 = 遮蔽该单元后验证集 MSE 的增加量；资源分数 = 原始贡献扣除少量存储和计算成本。高分只表示它在当前任务上有用，不代表会处理所有问题。", 12, false);
        addText(overview, "隐藏神经元上限：" + NeuronWorkspace.MAX_NEURONS
                + "；当前：" + workspace.hiddenCount() + "；估算工作区："
                + formatMiB(workspace.getEstimatedBytes()) + " MiB", 12, true);

        LinearLayout tools = card(content, "选择与批量操作");
        neuronFilterField = editText(currentNeuronFilter, "按 ID 筛选，例如 H-012", InputType.TYPE_CLASS_TEXT);
        tools.addView(neuronFilterField, params(-1, 48, 0, 0, 0, 0));
        Button applyFilter = secondaryButton("应用筛选");
        applyFilter.setOnClickListener(v -> {
            currentNeuronFilter = neuronFilterField.getText().toString().trim();
            showPage("neurons");
        });
        tools.addView(applyFilter, params(-1, 44, 0, 6, 0, 0));
        selectedCountLabel = label("已选 " + selectedIds.size() + " 个", 12, true);
        tools.addView(selectedCountLabel, params(-1, -2, 0, 6, 0, 0));
        addActionRow(tools, new String[]{"全选当前结果", "取消选择", "批量保存"}, new Runnable[]{
                this::selectVisibleNeurons,
                () -> { selectedIds.clear(); showPage("neurons"); },
                this::saveSelectedNeurons
        });
        addActionRow(tools, new String[]{"导出所选", "刷新评分", "新增神经元"}, new Runnable[]{
                this::exportSelectedNeurons,
                this::scoreNeuronsAsync,
                this::addNeuron
        });
        Button evolve = secondaryButton("运行一代资源竞争 / 复制 / 受保护剪枝");
        evolve.setOnClickListener(v -> runEvolution());
        tools.addView(evolve, params(-1, 46, 0, 8, 0, 0));

        LinearLayout list = card(content, "当前网络神经元");
        List<NeuronWorkspace.Neuron> units = workspace.neuronsSnapshot();
        int shown = 0;
        for (NeuronWorkspace.Neuron n : units) {
            if (!currentNeuronFilter.isEmpty()
                    && !n.id.toLowerCase(Locale.ROOT).contains(currentNeuronFilter.toLowerCase(Locale.ROOT))) continue;
            if (shown >= MAX_VISIBLE_NEURONS) break;
            addNeuronRow(list, n);
            shown++;
        }
        if (shown == 0) addText(list, "没有匹配的神经元。清除筛选条件即可查看。", 12, false);
        if (shown < units.size() && currentNeuronFilter.isEmpty())
            addText(list, "当前展示前 " + shown + " 个。总数 " + units.size()
                    + "；用上面的 ID 筛选可定位后面的单元，避免数百行控件拖慢手机。", 11, false);
    }

    private void addNeuronRow(LinearLayout parent, NeuronWorkspace.Neuron n) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.VERTICAL);
        row.setPadding(dp(10), dp(9), dp(10), dp(9));
        row.setBackground(tintedRound(Color.WHITE, 0xFFE4E7EC, 10));
        LinearLayout.LayoutParams rowLp = params(-1, -2, 0, 0, 0, 8);
        parent.addView(row, rowLp);

        LinearLayout top = new LinearLayout(this);
        top.setOrientation(LinearLayout.HORIZONTAL);
        top.setGravity(Gravity.CENTER_VERTICAL);
        CheckBox select = new CheckBox(this);
        select.setButtonTintList(android.content.res.ColorStateList.valueOf(0xFF167D72));
        select.setChecked(selectedIds.contains(n.id));
        select.setText(n.id);
        select.setTextSize(14);
        select.setTypeface(null, Typeface.BOLD);
        select.setOnCheckedChangeListener((button, checked) -> {
            if (checked) selectedIds.add(n.id); else selectedIds.remove(n.id);
            if (selectedCountLabel != null) selectedCountLabel.setText("已选 " + selectedIds.size() + " 个");
        });
        top.addView(select, new LinearLayout.LayoutParams(0, -2, 1));

        CheckBox enabled = new CheckBox(this);
        enabled.setText(n.enabled ? "参与计算" : "已禁用");
        enabled.setTextSize(11);
        enabled.setChecked(n.enabled);
        enabled.setOnCheckedChangeListener((button, checked) -> {
            try {
                workspace.setNeuronEnabled(n.id, checked);
                persistWorkspace();
                showPage("neurons");
            } catch (Throwable e) { toast("修改失败：" + shortError(e)); }
        });
        top.addView(enabled, new LinearLayout.LayoutParams(-2, -2));
        row.addView(top);

        addText(row, String.format(Locale.US, "贡献分 %.6f  ·  资源分 %.6f  ·  权重版本 r%d",
                n.score, n.resourceScore, n.revision), 11, false);
        addText(row, String.format(Locale.US, "偏置 %.4f  ·  输入权重 %d 个  ·  输出权重 %d 个",
                n.bias, n.inputWeights.length, n.outputWeights.length), 11, false);

        addActionRow(row, new String[]{"单独保存", "导出单元", "删除单元"}, new Runnable[]{
                () -> saveSingleNeuron(n.id),
                () -> exportSingleNeuron(n.id),
                () -> confirmRemoveNeuron(n.id)
        });
    }

    private void buildNetworkPage(LinearLayout content) {
        LinearLayout config = card(content, "网络结构");
        addText(config, "结构为输入 → tanh 隐藏层 → 线性输出。调整结构会重建网络并重置当前网络权重；已存入神经元库的副本不会删除。", 12, false);
        LinearLayout dimensions = new LinearLayout(this);
        dimensions.setOrientation(LinearLayout.HORIZONTAL);
        inputsField = editText(String.valueOf(workspace.inputCount), "输入数", InputType.TYPE_CLASS_NUMBER);
        hiddenField = editText(String.valueOf(workspace.hiddenCount()), "隐藏数", InputType.TYPE_CLASS_NUMBER);
        outputsField = editText(String.valueOf(workspace.outputCount), "输出数", InputType.TYPE_CLASS_NUMBER);
        dimensions.addView(inputsField, new LinearLayout.LayoutParams(0, dp(50), 1));
        dimensions.addView(hiddenField, params(82, 50, 6, 0, 0, 0));
        dimensions.addView(outputsField, params(82, 50, 6, 0, 0, 0));
        config.addView(dimensions);
        Button rebuild = primaryButton("确认重建网络");
        rebuild.setOnClickListener(v -> confirmReconfigure());
        config.addView(rebuild, params(-1, 48, 0, 8, 0, 0));
        Button add = secondaryButton("只新增一个隐藏神经元（保留现有权重）");
        add.setOnClickListener(v -> addNeuron());
        config.addView(add, params(-1, 46, 0, 6, 0, 0));

        LinearLayout graph = card(content, "连接示意图");
        NetworkDiagramView diagram = new NetworkDiagramView(this);
        diagram.setGraph(workspace.inputCount, workspace.outputCount, workspace.neuronsSnapshot());
        graph.addView(diagram, params(-1, 360, 0, 0, 0, 0));
        addText(graph, "线条表示输入权重/输出权重的相对大小；绿色是活动单元，灰色是已禁用单元。图形为当前隐藏层结构示意，过大的网络会只绘制部分节点。", 11, false);

        LinearLayout library = card(content, "从神经元库恢复");
        addText(library, "神经元库：" + workspace.savedNeuronCount() + " 个。导入前会检查输入/输出维度。", 12, false);
        Button goLibrary = secondaryButton("到数据/保存页查看神经元库");
        goLibrary.setOnClickListener(v -> showPage("data"));
        library.addView(goLibrary, params(-1, 44, 0, 6, 0, 0));
    }

    private void buildTrainingPage(LinearLayout content) {
        LinearLayout task = card(content, "训练任务设置");
        taskNameField = editText(workspace.taskName, "任务名称", InputType.TYPE_CLASS_TEXT);
        task.setTag("task");
        task.addView(taskNameField, params(-1, 50, 0, 0, 0, 0));
        epochsField = editText(String.valueOf(workspace.preferredEpochs), "训练轮数 1-5000", InputType.TYPE_CLASS_NUMBER);
        task.addView(epochsField, params(-1, 50, 0, 7, 0, 0));
        learningRateField = editText(String.format(Locale.US, "%.5f", workspace.learningRate), "学习率 0.00001-0.1", InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL);
        task.addView(learningRateField, params(-1, 50, 0, 7, 0, 0));
        addText(task, "训练器：CPU + Adam + tanh 隐藏层。按 5:1 切分训练/验证样本，保存验证集最佳权重并提前停止；训练过程能取消。每次最多受预算估算限制，不会直接占用 4GB。", 12, false);

        LinearLayout actions = card(content, "控制训练");
        Button start = primaryButton("开始训练并评分");
        start.setEnabled(!training);
        start.setOnClickListener(v -> startTraining());
        actions.addView(start, params(-1, 50, 0, 0, 0, 0));
        Button cancel = secondaryButton("停止训练（保留最佳权重）");
        cancel.setEnabled(training);
        cancel.setOnClickListener(v -> {
            cancelTraining = true;
            trainingStatus = "正在请求停止；会在当前 epoch 结束后保存最佳权重。";
            if (activeTrainLabel != null) activeTrainLabel.setText(trainingStatus);
        });
        actions.addView(cancel, params(-1, 46, 0, 6, 0, 0));

        activeTrainLabel = label(trainingStatus, 12, false);
        actions.addView(activeTrainLabel, params(-1, -2, 0, 8, 0, 0));
        activeTrainProgress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        activeTrainProgress.setMax(1000);
        activeTrainProgress.setProgress(0);
        actions.addView(activeTrainProgress, params(-1, 9, 0, 6, 0, 0));
        pageTrainingReport = label(workspace.lastReport, 12, false);
        pageTrainingReport.setTextIsSelectable(true);
        actions.addView(pageTrainingReport, params(-1, -2, 0, 10, 0, 0));

        LinearLayout evolution = card(content, "安全进化");
        addText(evolution, "高贡献单元有更高的复制优先级；资源分数同时考虑存储和计算成本。一次最多剪除一个非精英单元，且只有验证误差在 0.5% 安全容差内才允许剪除或接受后代。每代都能回滚。", 12, false);
        Button evolve = secondaryButton("运行一代（受保护复制与剪枝）");
        evolve.setOnClickListener(v -> runEvolution());
        evolution.addView(evolve, params(-1, 46, 0, 7, 0, 0));

        LinearLayout history = card(content, "已完成训练记录");
        List<JSONObject> tasks = workspace.taskHistorySnapshot();
        if (tasks.isEmpty()) addText(history, "还没有训练任务记录。", 12, false);
        for (int i = Math.max(0, tasks.size() - 8); i < tasks.size(); i++) {
            JSONObject t = tasks.get(i);
            addText(history, t.optString("name", "训练任务") + " · "
                    + t.optInt("epochsRun", 0) + "/" + t.optInt("epochsRequested", 0)
                    + " 轮 · 验证 MSE "
                    + String.format(Locale.US, "%.6f", t.optDouble("finalValidationMse", Double.NaN))
                    + " · " + t.optString("status", ""), 11, false);
        }
    }

    private void buildRunPage(LinearLayout content) {
        LinearLayout input = card(content, "输入与输出");
        addText(input, "输入维度：" + workspace.inputCount + "；输出维度：" + workspace.outputCount
                + "。按顺序输入逗号分隔的数值。这里运行的是当前网络，不是文本聊天模型。", 12, false);
        inputVectorField = editText(lastInputText, "例如：0.25 或 0.25,-0.1,0.8", InputType.TYPE_CLASS_TEXT);
        input.addView(inputVectorField, params(-1, 52, 0, 0, 0, 0));
        inputVectorField.setSingleLine(true);
        inputVectorField.setImeOptions(EditorInfo.IME_ACTION_DONE);
        Button run = primaryButton("运行当前网络");
        run.setOnClickListener(v -> runPrediction());
        input.addView(run, params(-1, 48, 0, 8, 0, 0));

        LinearLayout output = card(content, "结果与贡献解释");
        addText(output, lastPredictionReport, 12, false);
        Button refresh = secondaryButton("刷新计算结果");
        refresh.setOnClickListener(v -> runPrediction());
        output.addView(refresh, params(-1, 44, 0, 6, 0, 0));

        LinearLayout task = card(content, "给网络下达什么任务");
        addText(task, "当前任务形式是监督数值映射：每条样本包含 input 向量与 output 向量。训练时网络学习让预测输出接近标签。比如多输入、多输出的传感器数据、归一化指标或合成规则。", 12, false);
        addText(task, "自然语言问答、语义理解、图片输入和工具操作尚未接入这个小网络。不能仅靠把任意文本放进 CSV 就获得这些能力。", 12, true);
    }

    private void buildDataPage(LinearLayout content) {
        LinearLayout files = card(content, "工作区与文件");
        addText(files, "工作区文件保存在应用内部存储；导入/导出通过 Android 系统文件选择器完成。当前文件限制 " + (MAX_IMPORT_BYTES / (1024 * 1024)) + " MiB，以避免一次读入超大文件拖垮手机。", 12, false);
        addActionRow(files, new String[]{"导出完整工作区", "导入工作区", "保存本机工作区"}, new Runnable[]{
                () -> exportContent(safeWorkspaceJson(), "application/json", "aimeng-neuron-workspace.json"),
                () -> startImport("workspace"),
                this::persistWorkspaceWithToast
        });
        addActionRow(files, new String[]{"导出数据 JSON", "导出数据 CSV", "导入训练数据"}, new Runnable[]{
                () -> exportContent(safeDatasetJson(), "application/json", "aimeng-dataset.json"),
                () -> exportContent(workspace.exportDatasetCsv(), "text/csv", "aimeng-dataset.csv"),
                () -> startImport("dataset")
        });
        addActionRow(files, new String[]{"导出训练结果 CSV", "导出训练任务 JSON", "导入训练任务"}, new Runnable[]{
                () -> exportContent(workspace.exportTrainingResultsCsv(), "text/csv", "aimeng-training-results.csv"),
                () -> exportContent(safeTaskJson(), "application/json", "aimeng-training-task.json"),
                () -> startImport("task")
        });
        addActionRow(files, new String[]{"导出事件 JSONL", "导入神经元 JSON", "使用内置示例数据"}, new Runnable[]{
                () -> exportContent(workspace.exportTraceJsonl(), "application/x-ndjson", "aimeng-neuron-events.jsonl"),
                () -> startImport("neuron"),
                this::confirmDemoDataset
        });

        LinearLayout neurons = card(content, "本地神经元库");
        addText(neurons, "已保存 " + workspace.savedNeuronCount() + " / " + NeuronWorkspace.MAX_SAVED_NEURONS + " 个副本。保存副本独立于网络当前状态；训练或删掉网络中的单元不会改变已保存的版本。", 12, false);
        List<NeuronWorkspace.Neuron> saved = workspace.savedNeuronsSnapshot();
        int start = Math.max(0, saved.size() - 40);
        for (int i = start; i < saved.size(); i++) {
            final int index = i;
            NeuronWorkspace.Neuron n = saved.get(i);
            LinearLayout row = new LinearLayout(this);
            row.setGravity(Gravity.CENTER_VERTICAL);
            row.setOrientation(LinearLayout.HORIZONTAL);
            TextView name = label(n.libraryName.isEmpty() ? n.id : n.libraryName, 11, false);
            row.addView(name, new LinearLayout.LayoutParams(0, -2, 1));
            Button load = compactButton("载入");
            load.setOnClickListener(v -> appendSavedNeuron(index));
            row.addView(load, params(62, 40, 0, 0, 3, 0));
            Button del = compactButton("删除");
            del.setOnClickListener(v -> {
                new AlertDialog.Builder(this).setTitle("删除保存副本")
                        .setMessage("只删除神经元库中的副本，不修改当前网络。")
                        .setNegativeButton("取消", null)
                        .setPositiveButton("删除", (d, w) -> {
                            workspace.deleteSavedNeuron(index);
                            persistWorkspace();
                            showPage("data");
                        }).show();
            });
            row.addView(del, params(62, 40, 0, 0, 0, 0));
            neurons.addView(row, params(-1, -2, 0, 2, 0, 2));
        }
        if (saved.isEmpty()) addText(neurons, "还没有保存的神经元。到“神经元”页点击单独保存或批量保存。", 12, false);

        LinearLayout safety = card(content, "格式和安全边界");
        addText(safety, "工作区 JSON：模型 + 样本 + 神经元库 + 训练历史；数据 JSON/CSV：输入和输出样本；单神经元/神经元包 JSON：权重和维度；任务 JSON：训练参数；结果 CSV：训练记录；JSONL：最近事件。", 12, false);
        addText(safety, "文件导入会检查维度、非有限数值、行列一致性和上限；结构不匹配时先确认是否重建网络。当前工作区估算预算为 64 MiB，整个应用进程仍需低于 4 GiB。", 12, false);
    }

    private void startTraining() {
        if (training) { toast("训练已经在运行。"); return; }
        if (!runtimeMemoryAllowsWork()) return;
        final String taskName = taskNameField == null || taskNameField.getText().toString().trim().isEmpty()
                ? "数值任务" : taskNameField.getText().toString().trim();
        final int epochs;
        final double rate;
        try {
            epochs = Integer.parseInt(epochsField.getText().toString().trim());
            rate = Double.parseDouble(learningRateField.getText().toString().trim());
            if (epochs < 1 || epochs > NeuronWorkspace.MAX_EPOCHS)
                throw new IllegalArgumentException("训练轮数必须是 1 到 " + NeuronWorkspace.MAX_EPOCHS);
            if (!Double.isFinite(rate) || rate < 0.00001 || rate > 0.1)
                throw new IllegalArgumentException("学习率必须是 0.00001 到 0.1");
        } catch (Throwable e) {
            dialog("训练参数错误", shortError(e));
            return;
        }
        workspace.taskName = taskName;
        workspace.preferredEpochs = epochs;
        workspace.learningRate = rate;
        cancelTraining = false;
        training = true;
        trainingStatus = "正在训练：准备验证拆分与 CPU 优化器…";
        persistWorkspace();
        setNavigationEnabled(false);
        showPage("train");
        worker.execute(() -> {
            try {
                NeuronWorkspace.TrainingResult result = workspace.train(epochs,
                        (epoch, max, trainMse, validationMse) -> runOnUiThread(() -> {
                            trainingStatus = "训练中 " + epoch + "/" + max
                                    + " · train MSE " + String.format(Locale.US, "%.6f", trainMse)
                                    + " · validation MSE " + String.format(Locale.US, "%.6f", validationMse);
                            if (activeTrainLabel != null) activeTrainLabel.setText(trainingStatus);
                            if (activeTrainProgress != null)
                                activeTrainProgress.setProgress((int) (1000L * epoch / Math.max(1, max)));
                        }), () -> cancelTraining);
                trainingStatus = result.report;
                persistWorkspaceNow();
                runOnUiThread(() -> {
                    training = false;
                    setNavigationEnabled(true);
                    showPage("train");
                    toast(result.cancelled ? "训练已停止，已保存最佳权重" : "训练完成，工作区已保存");
                });
            } catch (Throwable error) {
                trainingStatus = "训练失败：" + shortError(error);
                persistWorkspaceNow();
                runOnUiThread(() -> {
                    training = false;
                    setNavigationEnabled(true);
                    showPage("train");
                    dialog("训练失败", trainingStatus);
                });
            }
        });
    }

    private void runEvolution() {
        if (training) { toast("训练期间不能并发修改网络结构。"); return; }
        if (!runtimeMemoryAllowsWork()) return;
        setBusyStatus("正在计算消融分数、资源成本和验证集剪枝候选…");
        worker.execute(() -> {
            try {
                String result = workspace.evolveOneGeneration();
                persistWorkspaceNow();
                trainingStatus = result;
                runOnUiThread(() -> {
                    globalStatus.setText("资源竞争完成 · 已保存");
                    showPage(currentPage);
                    dialog("单代资源竞争结果", result + "\n\n" + memoryStatus());
                });
            } catch (Throwable e) {
                runOnUiThread(() -> dialog("资源竞争失败", shortError(e)));
            }
        });
    }

    private void scoreNeuronsAsync() {
        if (training) { toast("训练期间请勿修改神经元。"); return; }
        setBusyStatus("正在计算验证集消融评分…");
        worker.execute(() -> {
            try {
                workspace.scoreNeurons();
                persistWorkspaceNow();
                runOnUiThread(() -> {
                    globalStatus.setText("评分完成 · 工作区已保存");
                    showPage("neurons");
                });
            } catch (Throwable e) { runOnUiThread(() -> dialog("评分失败", shortError(e))); }
        });
    }

    private void addNeuron() {
        if (training) { toast("训练期间不能改变网络结构。"); return; }
        if (!runtimeMemoryAllowsWork()) return;
        try {
            workspace.addNeuron();
            persistWorkspace();
            showPage(currentPage.equals("network") ? "network" : "neurons");
            toast("已新增一个随机初始化单元。建议训练并验证后再保留。");
        } catch (Throwable e) { dialog("新增神经元失败", shortError(e)); }
    }

    private void confirmRemoveNeuron(String id) {
        if (training) { toast("训练期间不能删减神经元。"); return; }
        new AlertDialog.Builder(this).setTitle("删除 " + id + "？")
                .setMessage("这会释放当前网络的单元与权重。已存入神经元库的副本不会被删除。建议先保存或导出。")
                .setNegativeButton("取消", null)
                .setNeutralButton("先保存副本", (d, w) -> saveSingleNeuron(id))
                .setPositiveButton("删除", (d, w) -> {
                    try {
                        workspace.removeNeuron(id);
                        selectedIds.remove(id);
                        persistWorkspace();
                        showPage("neurons");
                    } catch (Throwable e) { dialog("删除失败", shortError(e)); }
                }).show();
    }

    private void saveSingleNeuron(String id) {
        try {
            workspace.saveSingleNeuronToLibrary(id);
            persistWorkspace();
            toast("已将 " + id + " 的当前权重保存为独立副本。");
            showPage("neurons");
        } catch (Throwable e) { dialog("保存失败", shortError(e)); }
    }

    private void saveSelectedNeurons() {
        if (selectedIds.isEmpty()) { toast("请先勾选要保存的神经元。"); return; }
        try {
            ArrayList<String> ids = new ArrayList<>(selectedIds);
            workspace.saveNeuronsToLibrary(ids);
            persistWorkspace();
            toast("已保存 " + ids.size() + " 个独立副本。");
            showPage("neurons");
        } catch (Throwable e) { dialog("批量保存失败", shortError(e)); }
    }

    private void selectVisibleNeurons() {
        List<NeuronWorkspace.Neuron> units = workspace.neuronsSnapshot();
        int count = 0;
        for (NeuronWorkspace.Neuron n : units) {
            if (!currentNeuronFilter.isEmpty()
                    && !n.id.toLowerCase(Locale.ROOT).contains(currentNeuronFilter.toLowerCase(Locale.ROOT))) continue;
            if (count++ >= MAX_VISIBLE_NEURONS) break;
            selectedIds.add(n.id);
        }
        showPage("neurons");
        if (selectedCountLabel != null) selectedCountLabel.setText("已选 " + selectedIds.size() + " 个");
    }

    private void exportSingleNeuron(String id) {
        try {
            exportContent(workspace.exportNeuronJson(id), "application/json", id + ".aimeng-neuron.json");
        } catch (Throwable e) { dialog("导出失败", shortError(e)); }
    }

    private void exportSelectedNeurons() {
        if (selectedIds.isEmpty()) { toast("请先勾选需要导出的单元。"); return; }
        try {
            List<String> ids = new ArrayList<>(selectedIds);
            exportContent(workspace.exportNeuronPackJson(ids), "application/json", "aimeng-neuron-pack.json");
        } catch (Throwable e) { dialog("批量导出失败", shortError(e)); }
    }

    private void appendSavedNeuron(int index) {
        if (training) { toast("训练期间不能修改网络。"); return; }
        try {
            workspace.appendSavedNeuron(index);
            persistWorkspace();
            toast("神经元已复制到当前网络。");
            showPage("neurons");
        } catch (Throwable e) { dialog("载入失败", shortError(e)); }
    }

    private void confirmReconfigure() {
        if (training) { toast("训练期间不能重建网络。"); return; }
        final int in, hidden, out;
        try {
            in = Integer.parseInt(inputsField.getText().toString().trim());
            hidden = Integer.parseInt(hiddenField.getText().toString().trim());
            out = Integer.parseInt(outputsField.getText().toString().trim());
            if (in < 1 || in > NeuronWorkspace.MAX_INPUTS) throw new IllegalArgumentException("输入数必须为 1–16");
            if (hidden < 1 || hidden > NeuronWorkspace.MAX_NEURONS) throw new IllegalArgumentException("隐藏数必须为 1–128");
            if (out < 1 || out > NeuronWorkspace.MAX_OUTPUTS) throw new IllegalArgumentException("输出数必须为 1–16");
        } catch (Throwable e) { dialog("网络参数错误", shortError(e)); return; }
        new AlertDialog.Builder(this).setTitle("重建网络？")
                .setMessage("新结构：" + in + " → " + hidden + " → " + out
                        + "\n当前网络的训练权重会重置；神经元库、训练历史与已导出的文件不受影响。")
                .setNegativeButton("取消", null)
                .setPositiveButton("重建", (d, w) -> {
                    try {
                        workspace.reconfigure(in, hidden, out, false);
                        lastPredictionReport = "网络已重建。请加载/导入维度匹配的数据，再训练。";
                        persistWorkspace();
                        showPage("network");
                    } catch (Throwable e) { dialog("重建失败", shortError(e)); }
                }).show();
    }

    private void runPrediction() {
        try {
            String text = inputVectorField == null ? lastInputText : inputVectorField.getText().toString().trim();
            String[] fields = text.split("[,，\\s]+");
            if (fields.length != workspace.inputCount)
                throw new IllegalArgumentException("此网络需要 " + workspace.inputCount + " 个输入值，当前输入了 " + fields.length + " 个");
            double[] values = new double[fields.length];
            for (int i = 0; i < fields.length; i++) {
                values[i] = Double.parseDouble(fields[i]);
                if (!Double.isFinite(values[i])) throw new IllegalArgumentException("输入必须是有限数值");
            }
            NeuronWorkspace.ForwardResult result = workspace.predict(values);
            StringBuilder out = new StringBuilder("输入：").append(vector(values)).append("\n输出：");
            for (int i = 0; i < result.output.length; i++) {
                if (i > 0) out.append("  |  ");
                out.append("y").append(i).append(" = ").append(format(result.output[i]));
            }
            out.append("\n\n隐藏单元激活与输出贡献：\n");
            List<NeuronWorkspace.Neuron> units = workspace.neuronsSnapshot();
            for (int i = 0; i < units.size(); i++) {
                NeuronWorkspace.Neuron n = units.get(i);
                if (!n.enabled) continue;
                out.append(n.id).append("  激活 ").append(format(result.hidden[i]))
                        .append("  ·  ");
                for (int o = 0; o < workspace.outputCount; o++) {
                    if (o > 0) out.append(", ");
                    out.append("y").append(o).append(" += ")
                            .append(format(result.hiddenToOutputContribution[i][o]));
                }
                out.append("\n");
            }
            out.append("\n预测由 ").append(units.size()).append(" 个隐藏单元构成；绿色/活动单元参与求值。");
            lastInputText = text;
            lastPredictionReport = out.toString();
            showPage("run");
        } catch (Throwable e) { dialog("运行网络失败", shortError(e)); }
    }

    private void confirmDemoDataset() {
        new AlertDialog.Builder(this).setTitle("重置为内置示例数据？")
                .setMessage("当前训练样本将被替换为 48 条确定性数值样本；网络权重不会重置。若要保留现有数据，请先导出。")
                .setNegativeButton("取消", null)
                .setPositiveButton("替换数据", (d, w) -> {
                    workspace.createDemoSamples(48);
                    persistWorkspace();
                    showPage("data");
                }).show();
    }

    private void startImport(String type) {
        if (training) { toast("训练期间不能导入或替换工作区。"); return; }
        pendingImportType = type;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.putExtra(Intent.EXTRA_TITLE, "选择要导入的 AIMENG 文件");
        try { startActivityForResult(intent, REQ_IMPORT); }
        catch (Throwable e) { dialog("无法打开文件选择器", shortError(e)); }
    }

    private void exportContent(String content, String mime, String fileName) {
        if (content == null) { toast("没有可导出的内容。"); return; }
        if (content.getBytes(StandardCharsets.UTF_8).length > MAX_IMPORT_BYTES) {
            dialog("导出文件过大", "当前导出被限制在 16 MiB 内。可先减少数据样本或导出训练结果/单神经元文件。");
            return;
        }
        pendingExportContent = content;
        pendingExportMime = mime;
        pendingExportName = fileName;
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType(mime);
        intent.putExtra(Intent.EXTRA_TITLE, fileName);
        try { startActivityForResult(intent, REQ_EXPORT); }
        catch (Throwable e) { dialog("无法打开保存选择器", shortError(e)); }
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        if (requestCode == REQ_IMPORT) {
            final String importType = pendingImportType;
            worker.execute(() -> {
                try {
                    String fileName = displayName(uri);
                    String content = readUri(uri);
                    handleImport(content, fileName, importType);
                } catch (Throwable e) {
                    runOnUiThread(() -> dialog("文件导入失败", shortError(e)));
                }
            });
        } else if (requestCode == REQ_EXPORT) {
            final String exportText = pendingExportContent;
            final String exportName = pendingExportName;
            if (exportText == null) { toast("导出内容已失效，请重新操作。"); return; }
            worker.execute(() -> {
                try (OutputStream out = getContentResolver().openOutputStream(uri, "wt")) {
                    if (out == null) throw new java.io.IOException("无法打开目标文件");
                    out.write(exportText.getBytes(StandardCharsets.UTF_8));
                    out.flush();
                    runOnUiThread(() -> toast("文件已导出：" + exportName));
                } catch (Throwable e) {
                    runOnUiThread(() -> dialog("文件导出失败", shortError(e)));
                } finally {
                    pendingExportContent = null;
                    pendingExportMime = null;
                    pendingExportName = null;
                }
            });
        }
    }

    private void handleImport(String content, String fileName, String importType) {
        try {
            if (importType.equals("workspace")) {
                NeuronWorkspace imported = NeuronWorkspace.fromJson(new JSONObject(content));
                if (imported.getEstimatedBytes() > NeuronWorkspace.MAX_WORKSPACE_BUDGET_BYTES)
                    throw new IllegalArgumentException("导入工作区估算占用超过 64 MiB 安全预算，已拒绝加载");
                workspace = imported;
                selectedIds.clear();
                persistWorkspaceNow();
                runOnUiThread(() -> { showPage("home"); toast("完整工作区已导入并保存。"); });
                return;
            }
            if (importType.equals("dataset")) {
                try {
                    workspace.importDatasetContent(content, fileName, false);
                } catch (IllegalArgumentException mismatch) {
                    String message = mismatch.getMessage();
                    if (message != null && message.startsWith("DATA_DIMENSION_MISMATCH:")) {
                        String[] p = message.split(":");
                        int inputs = Integer.parseInt(p[1]), outputs = Integer.parseInt(p[2]);
                        runOnUiThread(() -> confirmImportDimensions(content, fileName, importType,
                                "训练数据是 " + inputs + " 输入 / " + outputs + " 输出；当前网络是 "
                                        + workspace.inputCount + " 输入 / " + workspace.outputCount
                                        + " 输出。是否重建网络结构？网络权重将重置，但导入前的工作区已自动保存。"));
                        return;
                    }
                    throw mismatch;
                }
            } else if (importType.equals("neuron")) {
                try {
                    workspace.importNeuronContent(content, false);
                } catch (IllegalArgumentException mismatch) {
                    String message = mismatch.getMessage();
                    if (message != null && message.startsWith("DATA_DIMENSION_MISMATCH:")) {
                        String[] p = message.split(":");
                        int inputs = Integer.parseInt(p[1]), outputs = Integer.parseInt(p[2]);
                        runOnUiThread(() -> confirmImportDimensions(content, fileName, importType,
                                "导入神经元需要 " + inputs + " 输入 / " + outputs + " 输出；当前网络是 "
                                        + workspace.inputCount + " / " + workspace.outputCount
                                        + "。重建后再导入，会重置当前网络权重，但不会删除已保存副本。"));
                        return;
                    }
                    throw mismatch;
                }
            } else if (importType.equals("task")) {
                workspace.importTaskContent(content);
            } else {
                throw new IllegalArgumentException("未知导入类型：" + importType);
            }
            persistWorkspaceNow();
            String report = workspace.lastReport;
            runOnUiThread(() -> {
                showPage(importType.equals("dataset") ? "train" : "data");
                toast(report == null || report.isEmpty() ? "导入完成并已保存。" : report);
            });
        } catch (Throwable e) {
            runOnUiThread(() -> dialog("文件格式或内容无效", shortError(e)));
        }
    }

    private void confirmImportDimensions(String content, String fileName, String importType, String message) {
        new AlertDialog.Builder(this).setTitle("输入/输出维度不匹配")
                .setMessage(message)
                .setNegativeButton("取消", (d, w) -> toast("已取消导入，没有修改网络。"))
                .setPositiveButton("保存当前后重建", (d, w) -> worker.execute(() -> {
                    try {
                        // Save the old state to a timestamped backup before a dimension reset.
                        File backup = new File(getFilesDir(), "neuron-workspace-backup-"
                                + System.currentTimeMillis() + ".json");
                        workspace.saveInternal(backup);
                        if (importType.equals("dataset"))
                            workspace.importDatasetContent(content, fileName, true);
                        else workspace.importNeuronContent(content, true);
                        persistWorkspaceNow();
                        runOnUiThread(() -> { showPage("data"); toast("已重建并导入；旧工作区有本地备份。"); });
                    } catch (Throwable e) { runOnUiThread(() -> dialog("维度重建/导入失败", shortError(e))); }
                })).show();
    }

    private void initializeNpuStatus() {
        npuWorker.execute(() -> {
            try {
                boolean ok = NpuRuntime.init(getApplicationContext());
                String detail = ok ? NpuRuntime.status()
                        : "离线（" + String.valueOf(NpuRuntime.getLastError()) + "）";
                npuStatus = ok ? "QNN/HTP 已初始化，等待独立矩阵诊断" : "不可用：" + detail;
            } catch (Throwable e) { npuStatus = "不可用：" + shortError(e); }
            runOnUiThread(() -> {
                if (globalStatus != null) globalStatus.setText("本机工作区已就绪 · " + npuStatus);
                if (currentPage.equals("home")) showPage("home");
            });
        });
    }

    private void runNpuDiagnostic() {
        if (npuDiagnosticRunning) { toast("NPU 诊断正在运行。"); return; }
        npuDiagnosticRunning = true;
        toast("正在初始化并执行小型 HTP 矩阵诊断…");
        npuWorker.execute(() -> {
            String report;
            try {
                if (!NpuRuntime.isReady() && !NpuRuntime.init(getApplicationContext()))
                    throw new IllegalStateException("NPU 初始化失败：" + NpuRuntime.getLastError());
                int m = 32, k = 32, n = 32;
                byte[] a = new byte[m * k];
                byte[] b = new byte[k * n];
                for (int i = 0; i < 8; i++) {
                    a[i * k] = 10;
                    b[i] = 10;
                    b[n + i] = 10;
                }
                long start = System.nanoTime();
                byte[] out = NpuRuntime.matMulInt8Buf(a, b, m, k, n);
                long elapsed = (System.nanoTime() - start) / 1_000_000L;
                if (out == null || out.length < 4 + m * n)
                    throw new IllegalStateException("HTP MatMul 返回长度无效：" + NpuRuntime.getLastNativeError());
                report = "QNN/HTP 矩阵诊断成功。\nbackend=" + NpuRuntime.status()
                        + "\nshape=32x32x32（为命中 HTP 白名单而填充）\nmatmul_ms=" + elapsed
                        + "\n注意：这只证明独立 INT8 MatMul 路径可执行，不证明它加速当前神经网络；小任务仍走 CPU。";
                npuStatus = "诊断成功，32³ INT8 MatMul " + elapsed + " ms；非网络自动加速证据";
            } catch (Throwable e) {
                report = "NPU 诊断失败，网络继续使用 CPU。\n" + shortError(e);
                npuStatus = "诊断失败/回退 CPU：" + shortError(e);
            } finally { npuDiagnosticRunning = false; }
            final String finalReport = report;
            runOnUiThread(() -> {
                if (globalStatus != null) globalStatus.setText(npuStatus);
                dialog("NPU 独立诊断", finalReport);
            });
        });
    }

    private void persistWorkspace() {
        if (workspace == null) return;
        worker.execute(this::persistWorkspaceNow);
        if (globalStatus != null) globalStatus.setText("有变更，正在保存本机工作区…");
    }

    private void persistWorkspaceWithToast() {
        if (workspace == null) return;
        if (training) { toast("训练结束后会自动保存。"); return; }
        setBusyStatus("正在保存工作区…");
        worker.execute(() -> {
            try {
                persistWorkspaceNow();
                runOnUiThread(() -> { globalStatus.setText("工作区已保存到应用内部存储。"); toast("本机工作区已保存。"); });
            } catch (Throwable e) { runOnUiThread(() -> dialog("保存失败", shortError(e))); }
        });
    }

    private void persistWorkspaceNow() {
        if (workspace == null) return;
        try { workspace.saveInternal(new File(getFilesDir(), "neuron-workspace.json")); }
        catch (Throwable e) {
            final String message = "自动保存失败：" + shortError(e);
            runOnUiThread(() -> { if (globalStatus != null) globalStatus.setText(message); });
        }
    }

    private String safeWorkspaceJson() {
        try { return workspace.exportWorkspaceJson(); }
        catch (Throwable e) { dialog("序列化失败", shortError(e)); return null; }
    }

    private String safeDatasetJson() {
        try { return workspace.exportDatasetJson(); }
        catch (Throwable e) { dialog("序列化失败", shortError(e)); return null; }
    }

    private String safeTaskJson() {
        try { return workspace.exportTaskJson(); }
        catch (Throwable e) { dialog("序列化失败", shortError(e)); return null; }
    }

    private void setBusyStatus(String value) {
        if (globalStatus != null) globalStatus.setText(value);
    }

    private void setNavigationEnabled(boolean enabled) {
        if (navRow != null) for (int i = 0; i < navRow.getChildCount(); i++)
            navRow.getChildAt(i).setEnabled(enabled);
    }

    private boolean runtimeMemoryAllowsWork() {
        double rss = currentRssMiB();
        if (rss > 3500.0) {
            dialog("内存保护已触发", String.format(Locale.US,
                    "当前进程 RSS %.1f MiB，超过 3500 MiB 保护线。为给 native runtime、QNN staging 和系统保留余量，暂不启动训练/进化。硬上限仍是 4096 MiB。", rss));
            return false;
        }
        return true;
    }

    private String memoryStatus() {
        double rss = currentRssMiB();
        long peak = readProcKb("VmHWM:");
        Debug.MemoryInfo info = new Debug.MemoryInfo();
        try { Debug.getMemoryInfo(info); } catch (Throwable ignored) { }
        return String.format(Locale.US, "RSS %.1f MiB · PSS %.1f MiB · 峰值 HWM %.1f MiB",
                rss, info.getTotalPss() / 1024.0, peak < 0 ? -1.0 : peak / 1024.0);
    }

    private double currentRssMiB() {
        long rss = readProcKb("VmRSS:");
        return rss < 0 ? 0 : rss / 1024.0;
    }

    private long readProcKb(String key) {
        try (java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.FileReader("/proc/self/status"))) {
            String line;
            while ((line = reader.readLine()) != null)
                if (line.startsWith(key)) {
                    String[] parts = line.trim().split("\\s+");
                    if (parts.length >= 2) return Long.parseLong(parts[1]);
                }
        } catch (Throwable ignored) { }
        return -1;
    }

    private String readUri(Uri uri) throws Exception {
        try (InputStream input = getContentResolver().openInputStream(uri);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            if (input == null) throw new java.io.IOException("无法读取所选文件");
            byte[] buffer = new byte[16 * 1024];
            int total = 0, count;
            while ((count = input.read(buffer)) != -1) {
                total += count;
                if (total > MAX_IMPORT_BYTES) throw new IllegalArgumentException("文件超过 16 MiB 导入上限");
                output.write(buffer, 0, count);
            }
            return new String(output.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    private String readFile(File file, int limit) throws Exception {
        try (FileInputStream in = new FileInputStream(file); ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buf = new byte[16 * 1024];
            int total = 0, n;
            while ((n = in.read(buf)) != -1) {
                total += n;
                if (total > limit) throw new IllegalArgumentException("内部工作区超过大小限制");
                out.write(buf, 0, n);
            }
            return new String(out.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    private String displayName(Uri uri) {
        try (android.database.Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int col = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME);
                if (col >= 0) return cursor.getString(col);
            }
        } catch (Throwable ignored) { }
        return "import.json";
    }

    private LinearLayout card(LinearLayout parent, String title) {
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(13), dp(12), dp(13), dp(12));
        panel.setBackground(tintedRound(Color.WHITE, 0xFFE4E7EC, 12));
        parent.addView(panel, params(-1, -2, 0, 0, 0, 11));
        TextView heading = label(title, 16, true);
        panel.addView(heading, params(-1, -2, 0, 0, 0, 8));
        return panel;
    }

    private void addText(LinearLayout parent, String text, int size, boolean bold) {
        TextView view = label(text, size, bold);
        view.setTextIsSelectable(true);
        parent.addView(view, params(-1, -2, 0, 2, 0, 4));
    }

    private void addActionRow(LinearLayout parent, String[] titles, Runnable[] actions) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        for (int i = 0; i < titles.length; i++) {
            final Runnable action = actions[i];
            Button b = compactButton(titles[i]);
            b.setOnClickListener(v -> action.run());
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, dp(46), 1);
            if (i > 0) lp.leftMargin = dp(5);
            row.addView(b, lp);
        }
        parent.addView(row, params(-1, -2, 0, 5, 0, 2));
    }

    private EditText editText(String value, String hint, int inputType) {
        EditText edit = new EditText(this);
        edit.setSingleLine(true);
        edit.setTextSize(14);
        edit.setInputType(inputType);
        edit.setHint(hint);
        edit.setText(value == null ? "" : value);
        edit.setPadding(dp(12), 0, dp(12), 0);
        edit.setBackground(tintedRound(Color.WHITE, 0xFFD0D5DD, 8));
        return edit;
    }

    private TextView label(String value, int size, boolean bold) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(size);
        view.setTextColor(0xFF172033);
        view.setLineSpacing(dp(2), 1.0f);
        if (bold) view.setTypeface(Typeface.create("sans-serif-medium", Typeface.BOLD));
        return view;
    }

    private Button primaryButton(String text) {
        Button b = buttonBase(text);
        b.setTextColor(Color.WHITE);
        b.setBackground(tintedRound(0xFF176B63, 0xFF176B63, 10));
        return b;
    }

    private Button secondaryButton(String text) {
        Button b = buttonBase(text);
        b.setTextColor(0xFF155E58);
        b.setBackground(tintedRound(0xFFE8F5F3, 0xFFC4E5E0, 9));
        return b;
    }

    private Button compactButton(String text) {
        Button b = buttonBase(text);
        b.setTextSize(11);
        b.setPadding(dp(4), 0, dp(4), 0);
        b.setTextColor(0xFF344054);
        b.setBackground(tintedRound(0xFFF2F4F7, 0xFFE4E7EC, 8));
        return b;
    }

    private Button buttonBase(String text) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        b.setMinHeight(dp(42));
        b.setTextSize(13);
        b.setPadding(dp(8), dp(4), dp(8), dp(4));
        return b;
    }

    private GradientDrawable tintedRound(int color, int stroke, int radius) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        d.setCornerRadius(dp(radius));
        if (stroke != color) d.setStroke(dp(1), stroke);
        return d;
    }

    private LinearLayout.LayoutParams params(int width, int height, int left, int top, int right, int bottom) {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(width < 0 ? width : dp(width),
                height < 0 ? height : dp(height));
        lp.setMargins(dp(left), dp(top), dp(right), dp(bottom));
        return lp;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private String format(double value) { return String.format(Locale.US, "%.6f", value); }
    private String formatMiB(long bytes) { return String.format(Locale.US, "%.2f", bytes / (1024.0 * 1024.0)); }
    private String vector(double[] values) {
        StringBuilder b = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {
            if (i > 0) b.append(", ");
            b.append(format(values[i]));
        }
        return b.append("]").toString();
    }

    private String shortError(Throwable t) {
        String message = t == null ? "未知错误" : t.getMessage();
        if (message == null || message.trim().isEmpty()) message = t.getClass().getSimpleName();
        if (message.length() > 900) message = message.substring(0, 900) + "…";
        return message;
    }

    private void toast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
    }

    private void dialog(String title, String message) {
        if (isFinishing() || isDestroyed()) return;
        new AlertDialog.Builder(this).setTitle(title).setMessage(message)
                .setPositiveButton("知道了", null).show();
    }

    @Override protected void onDestroy() {
        cancelTraining = true;
        if (!training) persistWorkspaceNow();
        worker.shutdown();
        npuWorker.shutdown();
        super.onDestroy();
    }
}