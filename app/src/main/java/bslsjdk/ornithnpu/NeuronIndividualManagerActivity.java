package bslsjdk.ornithnpu;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
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
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;

/**
 * Personal inventory for complete, independently identified neuron-network individuals.
 * Every checkpoint contains the full graph topology and all persisted learning state.
 */
public final class NeuronIndividualManagerActivity extends Activity {
    private static final int REQUEST_IMPORT = 4101;
    private static final int REQUEST_EXPORT = 4102;
    private static final String FILE_FORMAT = "aimeng-neuron-individual/v1";

    private final List<Individual> individuals = new ArrayList<>();
    private final Set<String> selectedIds = new HashSet<>();
    private File directory;
    private Individual current;
    private LinearLayout root;
    private LinearLayout listBox;
    private TextView status;
    private TextView currentInfo;
    private Button saveSelectedButton;
    private Button exportSelectedButton;
    private boolean experimentRunning;

    private static final class Individual {
        String id;
        String name;
        String fileName;
        SelfOrganizingNeuronGraph graph;
        String lastTrainingResult = "尚无训练记录";
        String lastValidationResult = "尚无独立验证记录";
        boolean checked;
        Individual(String id, String name, String fileName, SelfOrganizingNeuronGraph graph) {
            this.id = id; this.name = name; this.fileName = fileName; this.graph = graph;
        }
    }

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        directory = new File(getFilesDir(), "neuron_individuals");
        if (!directory.exists()) directory.mkdirs();
        buildUi();
        loadAll();
        refresh();
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(14), dp(16), dp(24));
        scroll.addView(root);
        setContentView(scroll);

        TextView title = new TextView(this);
        title.setText("神经网络个体管理");
        title.setTextSize(24);
        title.setTypeface(null, Typeface.BOLD);
        root.addView(title);

        TextView description = new TextView(this);
        description.setText("每个个体独立编号、独立保存。检查点保存完整神经元、连接、权重、偏置、拓扑和学习状态。选中多个个体可批量操作。");
        description.setTextSize(14);
        description.setPadding(0, dp(8), 0, dp(12));
        root.addView(description);

        LinearLayout row1 = makeRow();
        addButton(row1, "新建个体", v -> createIndividual());
        addButton(row1, "保存选中", v -> saveSelected());
        addButton(row1, "保存全部", v -> saveAll());
        root.addView(row1);

        LinearLayout row2 = makeRow();
        addButton(row2, "导入个体", v -> importIndividual());
        addButton(row2, "导出选中", v -> beginExport());
        addButton(row2, "重命名当前", v -> renameCurrent());
        root.addView(row2);

        LinearLayout row3 = makeRow();
        addButton(row3, "删除选中", v -> deleteSelected());
        addButton(row3, "刷新列表", v -> { loadAll(); refresh(); });
        root.addView(row3);

        LinearLayout row4 = makeRow();
        addButton(row4, "训练当前个体", v -> chooseTrainingEpisodes());
        addButton(row4, "独立验证当前", v -> runExperiment(false, 40));
        root.addView(row4);

        currentInfo = new TextView(this);
        currentInfo.setTextSize(14);
        currentInfo.setPadding(0, dp(12), 0, dp(8));
        root.addView(currentInfo);

        TextView listTitle = new TextView(this);
        listTitle.setText("个体列表（点名称切换当前个体，勾选框用于批量操作）");
        listTitle.setTypeface(null, Typeface.BOLD);
        listTitle.setPadding(0, dp(8), 0, dp(8));
        root.addView(listTitle);

        listBox = new LinearLayout(this);
        listBox.setOrientation(LinearLayout.VERTICAL);
        root.addView(listBox);

        status = new TextView(this);
        status.setTextSize(13);
        status.setPadding(0, dp(12), 0, dp(4));
        root.addView(status);
    }

    private LinearLayout makeRow() {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setPadding(0, dp(2), 0, dp(2));
        return row;
    }

    private void addButton(LinearLayout row, String label, View.OnClickListener listener) {
        Button b = new Button(this);
        b.setText(label);
        b.setTextSize(12);
        b.setOnClickListener(listener);
        row.addView(b, new LinearLayout.LayoutParams(0, dp(48), 1f));
    }

    private int dp(int value) {
        return (int) (value * getResources().getDisplayMetrics().density + 0.5f);
    }

    private void createIndividual() {
        final String[] sizes = {"32 个神经元", "64 个神经元", "128 个神经元",
                "256 个神经元", "512 个神经元", "768 个神经元"};
        final int[] neuronCounts = {32, 64, 128, 256, 512, 768};
        new AlertDialog.Builder(this).setTitle("选择新个体的神经元规模")
                .setItems(sizes, (dialog, which) -> promptIndividualName(neuronCounts[which]))
                .setNegativeButton("取消", null).show();
    }

    private void promptIndividualName(int neuronCount) {
        final EditText nameInput = new EditText(this);
        nameInput.setSingleLine(true);
        nameInput.setHint("个体名称，例如：迷宫探索-01");
        nameInput.setText("神经网络-" + (individuals.size() + 1));
        new AlertDialog.Builder(this).setTitle("命名 " + neuronCount + " 神经元个体")
                .setView(nameInput)
                .setNegativeButton("取消", null)
                .setPositiveButton("创建", (dialog, which) -> {
                    String id = UUID.randomUUID().toString();
                    String fileName = id + ".json";
                    String name = nameInput.getText().toString().trim();
                    if (name.isEmpty()) name = "神经网络-" + id.substring(0, 8);
                    int initialEdges = Math.min(8000, Math.max(80, neuronCount * 3));
                    SelfOrganizingNeuronGraph graph = SelfOrganizingNeuronGraph
                            .createRandomGraph(8, 4, neuronCount, initialEdges, System.nanoTime());
                    graph.setActiveNeuronBudget(Math.min(neuronCount, Math.max(12, neuronCount / 4)));
                    Individual individual = new Individual(id, name, fileName, graph);
                    individuals.add(individual);
                    current = individual;
                    selectedIds.clear();
                    selectedIds.add(id);
                    try {
                        writeIndividual(individual);
                        refresh();
                        toast("已创建并保存完整个体");
                    } catch (Exception e) {
                        individuals.remove(individual);
                        selectedIds.remove(id);
                        current = individuals.isEmpty() ? null : individuals.get(0);
                        showError("创建保存失败", e);
                    }
                }).show();
    }

    private void chooseActiveBudget() {
        if (current == null) { toast("请先创建或选择个体"); return; }
        final Individual target = current;
        java.util.ArrayList<Integer> budgets = new java.util.ArrayList<>();
        int[] candidates = {8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1000};
        for (int value : candidates) {
            if (value >= target.graph.getInputCount() && value <= target.graph.getNeuronCount()
                    && !budgets.contains(value)) budgets.add(value);
        }
        if (budgets.isEmpty()) budgets.add(target.graph.getNeuronCount());
        String[] labels = new String[budgets.size()];
        for (int i = 0; i < budgets.size(); i++)
            labels[i] = budgets.get(i) + " 个活动神经元 / tick"
                    + (budgets.get(i) == target.graph.getActiveNeuronBudget() ? "（当前）" : "");
        new AlertDialog.Builder(this).setTitle("设置每个传播 tick 的激活预算")
                .setItems(labels, (dialog, which) -> {
                    target.graph.setActiveNeuronBudget(budgets.get(which));
                    try { writeIndividual(target); refresh(); }
                    catch (Exception e) { showError("保存激活预算失败", e); }
                }).setNegativeButton("取消", null).show();
    }

    private JSONObject envelope(Individual item) throws JSONException {
        JSONObject root = new JSONObject();
        root.put("format", FILE_FORMAT);
        root.put("id", item.id);
        root.put("name", item.name);
        root.put("savedAt", System.currentTimeMillis());
        root.put("lastTrainingResult", item.lastTrainingResult);
        root.put("lastValidationResult", item.lastValidationResult);
        root.put("graph", item.graph.toJson());
        return root;
    }

    private void writeIndividual(Individual item) throws Exception {
        JSONObject json = envelope(item);
        File target = new File(directory, item.fileName);
        File temp = new File(directory, item.fileName + ".tmp");
        try (FileOutputStream out = new FileOutputStream(temp)) {
            out.write(json.toString().getBytes(StandardCharsets.UTF_8));
            out.getFD().sync();
        }
        if (target.exists() && !target.delete()) throw new java.io.IOException("无法替换旧检查点");
        if (!temp.renameTo(target)) {
            try (FileOutputStream out = new FileOutputStream(target)) {
                out.write(json.toString().getBytes(StandardCharsets.UTF_8));
                out.getFD().sync();
            }
            temp.delete();
        }
    }

    private Individual parseIndividual(String jsonText) throws Exception {
        JSONObject root = new JSONObject(jsonText);
        if (!FILE_FORMAT.equals(root.optString("format")))
            throw new JSONException("不是受支持的完整网络个体文件");
        String id = root.optString("id", UUID.randomUUID().toString());
        String name = root.optString("name", "导入个体-" + id.substring(0, Math.min(8, id.length())));
        SelfOrganizingNeuronGraph graph = SelfOrganizingNeuronGraph.fromJson(
                root.getJSONObject("graph"), System.nanoTime());
        // Avoid overwriting a different individual when importing the same file twice.
        boolean duplicate = false;
        for (Individual item : individuals) if (item.id.equals(id)) { duplicate = true; break; }
        if (duplicate) id = UUID.randomUUID().toString();
        Individual item = new Individual(id, name, id + ".json", graph);
        item.lastTrainingResult = root.optString("lastTrainingResult", "尚无训练记录");
        item.lastValidationResult = root.optString("lastValidationResult", "尚无独立验证记录");
        return item;
    }

    private void loadAll() {
        individuals.clear();
        if (directory == null || !directory.exists()) return;
        File[] files = directory.listFiles((dir, name) -> name.endsWith(".json"));
        if (files == null) return;
        java.util.Arrays.sort(files, Comparator.comparing(File::getName));
        for (File file : files) {
            try (FileInputStream in = new FileInputStream(file)) {
                Individual item = parseIndividual(readAll(in));
                item.fileName = file.getName();
                individuals.add(item);
            } catch (Exception ignored) {
                // A damaged checkpoint is skipped, never silently rewritten.
            }
        }
        if (current != null) {
            String oldId = current.id;
            current = null;
            for (Individual item : individuals) if (item.id.equals(oldId)) current = item;
        }
        if (current == null && !individuals.isEmpty()) current = individuals.get(0);
        selectedIds.retainAll(ids());
        if (selectedIds.isEmpty() && current != null) selectedIds.add(current.id);
    }

    private Set<String> ids() {
        Set<String> result = new HashSet<>();
        for (Individual item : individuals) result.add(item.id);
        return result;
    }

    private void refresh() {
        if (listBox == null) return;
        listBox.removeAllViews();
        for (Individual item : individuals) {
            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setPadding(dp(4), dp(5), dp(4), dp(5));
            row.setBackgroundColor(current == item ? 0xFFE4EDF8 : 0x00000000);
            CheckBox check = new CheckBox(this);
            check.setChecked(selectedIds.contains(item.id));
            check.setOnCheckedChangeListener((button, checked) -> {
                if (checked) selectedIds.add(item.id); else selectedIds.remove(item.id);
                updateButtons();
            });
            row.addView(check);
            LinearLayout details = new LinearLayout(this);
            details.setOrientation(LinearLayout.VERTICAL);
            TextView name = new TextView(this);
            name.setText(item.name);
            name.setTextSize(16);
            name.setTypeface(null, Typeface.BOLD);
            TextView meta = new TextView(this);
            meta.setText("ID: " + item.id + "\n神经元 " + item.graph.getNeuronCount()
                    + " · 活跃连接 " + item.graph.getActiveEdgeCount()
                    + " · 累计传播步 " + item.graph.getTicks());
            meta.setTextSize(12);
            details.addView(name);
            details.addView(meta);
            details.setOnClickListener(v -> {
                current = item;
                selectedIds.add(item.id);
                refresh();
            });
            row.addView(details, new LinearLayout.LayoutParams(0, -2, 1f));
            listBox.addView(row);
            View divider = new View(this);
            divider.setBackgroundColor(0xFFDDDDDD);
            listBox.addView(divider, new LinearLayout.LayoutParams(-1, dp(1)));
        }
        if (individuals.isEmpty()) {
            TextView empty = new TextView(this);
            empty.setText("还没有保存的个体。点击“新建个体”创建第一个完整网络。");
            empty.setPadding(0, dp(12), 0, dp(12));
            listBox.addView(empty);
        }
        currentInfo.setText(current == null ? "当前个体：无" :
                "当前个体：" + current.name + "\n唯一 ID：" + current.id
                + "\n完整结构：" + current.graph.getNeuronCount() + " 个神经元 / "
                + current.graph.getActiveEdgeCount() + " 条活跃连接 / "
                + current.graph.getEdgeSlotCount() + " 个连接槽"
                + "\n检查点估算数组内存：" + (current.graph.estimatedStorageBytes() / 1024) + " KiB"
                + "\n" + current.lastTrainingResult + "\n" + current.lastValidationResult);
        status.setText("本地个体数：" + individuals.size() + " · 已选中：" + selectedIds.size()
                + "\n文件目录：应用私有存储 / neuron_individuals");
        updateButtons();
    }

    private void updateButtons() {
        if (saveSelectedButton != null) saveSelectedButton.setEnabled(!selectedIds.isEmpty());
        if (exportSelectedButton != null) exportSelectedButton.setEnabled(!selectedIds.isEmpty());
    }

    private void chooseTrainingEpisodes() {
        if (current == null) { toast("请先创建或选择个体"); return; }
        final String[] labels = {"25 局（快速检查）", "100 局（默认）", "250 局（较长实验）"};
        final int[] values = {25, 100, 250};
        new AlertDialog.Builder(this).setTitle("训练当前个体")
                .setItems(labels, (dialog, which) -> runExperiment(true, values[which]))
                .setNegativeButton("取消", null).show();
    }

    private void runExperiment(boolean training, int episodes) {
        if (experimentRunning) { toast("已有训练/验证任务运行中，请稍候"); return; }
        if (current == null) { toast("请先创建或选择个体"); return; }
        final Individual target = current;
        experimentRunning = true;
        status.setText((training ? "正在训练 " : "正在独立验证 ") + target.name
                + "，地图数 " + episodes + "。任务在后台运行，界面仍可响应。");
        new Thread(() -> {
            SelfOrganizingMazeTrainer.Result result = null;
            Exception failure = null;
            try {
                long seed = training
                        ? (System.nanoTime() ^ target.id.hashCode())
                        : (0x5EED2026L ^ target.id.hashCode());
                SelfOrganizingNeuronGraph evaluatedGraph = target.graph;
                if (!training) {
                    // Validate a restored copy so evaluation does not change the saved
                    // individual's traces, activity statistics, tick counter or weights.
                    evaluatedGraph = SelfOrganizingNeuronGraph.fromJson(target.graph.toJson(), seed);
                }
                result = SelfOrganizingMazeTrainer.run(evaluatedGraph, episodes, seed, training);
            } catch (Exception e) { failure = e; }
            final SelfOrganizingMazeTrainer.Result completed = result;
            final Exception error = failure;
            runOnUiThread(() -> {
                experimentRunning = false;
                if (error != null) {
                    showError(training ? "训练失败" : "独立验证失败", error);
                    status.setText("任务失败；个体检查点未被标记为验证通过。");
                    return;
                }
                if (training) target.lastTrainingResult = completed.toDisplayString();
                else target.lastValidationResult = completed.toDisplayString();
                try { writeIndividual(target); }
                catch (Exception e) { showError("成绩保存失败", e); }
                current = target;
                refresh();
                new AlertDialog.Builder(this)
                        .setTitle(training ? "训练完成" : "独立验证完成")
                        .setMessage(completed.toDisplayString()
                                + (training ? "\\n注意：训练成绩不等于泛化能力，完成后请使用独立验证。" :
                                "\\n验证使用独立种子，且在网络副本上执行，不修改当前个体。"))
                        .setPositiveButton("确定", null).show();
            });
        }, training ? "NeuronMazeTrain" : "NeuronMazeValidate").start();
    }

    private void saveSelected() {
        int count = 0;
        try {
            for (Individual item : individuals) if (selectedIds.contains(item.id)) {
                writeIndividual(item); count++;
            }
            toast("已保存 " + count + " 个完整网络个体");
            refresh();
        } catch (Exception e) { showError("保存失败", e); }
    }

    private void saveAll() {
        try {
            for (Individual item : individuals) writeIndividual(item);
            toast("已保存全部 " + individuals.size() + " 个完整网络个体");
            refresh();
        } catch (Exception e) { showError("批量保存失败", e); }
    }

    private void renameCurrent() {
        if (current == null) { toast("请先创建或选择个体"); return; }
        EditText input = new EditText(this);
        input.setSingleLine(true);
        input.setText(current.name);
        new AlertDialog.Builder(this).setTitle("重命名个体").setView(input)
                .setNegativeButton("取消", null)
                .setPositiveButton("保存名称", (d, w) -> {
                    String name = input.getText().toString().trim();
                    if (!name.isEmpty()) {
                        current.name = name;
                        try { writeIndividual(current); refresh(); }
                        catch (Exception e) { showError("重命名保存失败", e); }
                    }
                }).show();
    }

    private void deleteSelected() {
        if (selectedIds.isEmpty()) { toast("请先勾选要删除的个体"); return; }
        new AlertDialog.Builder(this).setTitle("删除选中个体？")
                .setMessage("将删除 " + selectedIds.size() + " 个本地检查点。建议先导出备份。")
                .setNegativeButton("取消", null)
                .setPositiveButton("删除", (d, w) -> {
                    List<Individual> remove = new ArrayList<>();
                    for (Individual item : individuals) if (selectedIds.contains(item.id)) remove.add(item);
                    for (Individual item : remove) new File(directory, item.fileName).delete();
                    individuals.removeAll(remove);
                    selectedIds.clear();
                    current = individuals.isEmpty() ? null : individuals.get(0);
                    if (current != null) selectedIds.add(current.id);
                    refresh();
                }).show();
    }

    private void beginExport() {
        if (selectedIds.isEmpty()) { toast("请先勾选要导出的个体"); return; }
        List<Individual> chosen = new ArrayList<>();
        for (Individual item : individuals) if (selectedIds.contains(item.id)) chosen.add(item);
        if (chosen.size() > 1) {
            // Multiple individuals are exported as one bundle, preserving each complete graph.
            try {
                JSONObject bundle = new JSONObject();
                bundle.put("format", "aimeng-neuron-individual-bundle/v1");
                org.json.JSONArray array = new org.json.JSONArray();
                for (Individual item : chosen) array.put(envelope(item));
                bundle.put("individuals", array);
                Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType("application/json");
                intent.putExtra(Intent.EXTRA_TITLE, "aimeng-neuron-bundle.json");
                pendingExportText = bundle.toString(2);
                startActivityForResult(intent, REQUEST_EXPORT);
            } catch (Exception e) { showError("准备导出失败", e); }
            return;
        }
        try {
            pendingExportText = envelope(chosen.get(0)).toString(2);
            Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("application/json");
            intent.putExtra(Intent.EXTRA_TITLE, safeFileName(chosen.get(0).name) + ".neuron.json");
            startActivityForResult(intent, REQUEST_EXPORT);
        } catch (Exception e) { showError("准备导出失败", e); }
    }

    private String pendingExportText;

    private String safeFileName(String name) {
        return name.replaceAll("[^a-zA-Z0-9_\\-\\u4e00-\\u9fa5]", "_");
    }

    private void importIndividual() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/json");
        startActivityForResult(intent, REQUEST_IMPORT);
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        try {
            if (requestCode == REQUEST_EXPORT) {
                if (pendingExportText == null) throw new IllegalStateException("导出内容丢失");
                try (OutputStream out = getContentResolver().openOutputStream(uri, "wt")) {
                    if (out == null) throw new java.io.IOException("无法打开目标文件");
                    out.write(pendingExportText.getBytes(StandardCharsets.UTF_8));
                    out.flush();
                }
                pendingExportText = null;
                toast("完整网络文件已导出");
            } else if (requestCode == REQUEST_IMPORT) {
                String raw;
                try (InputStream in = getContentResolver().openInputStream(uri)) {
                    if (in == null) throw new java.io.IOException("无法读取文件");
                    raw = readAll(in);
                }
                JSONObject top = new JSONObject(raw);
                if ("aimeng-neuron-individual-bundle/v1".equals(top.optString("format"))) {
                    org.json.JSONArray array = top.getJSONArray("individuals");
                    int added = 0;
                    for (int i = 0; i < array.length(); i++) {
                        Individual item = parseIndividual(array.getJSONObject(i).toString());
                        individuals.add(item); writeIndividual(item); added++;
                    }
                    toast("已导入 " + added + " 个完整个体");
                } else {
                    Individual item = parseIndividual(raw);
                    individuals.add(item);
                    writeIndividual(item);
                    current = item;
                    selectedIds.clear(); selectedIds.add(item.id);
                    toast("已导入完整个体：" + item.name);
                }
                refresh();
            }
        } catch (Exception e) { showError("文件操作失败", e); }
    }

    private String readAll(InputStream in) throws Exception {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int n;
        while ((n = in.read(buffer)) != -1) out.write(buffer, 0, n);
        return new String(out.toByteArray(), StandardCharsets.UTF_8);
    }

    private void showError(String title, Exception e) {
        new AlertDialog.Builder(this).setTitle(title)
                .setMessage(e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage())
                .setPositiveButton("确定", null).show();
    }

    private void toast(String message) { Toast.makeText(this, message, Toast.LENGTH_LONG).show(); }
}
