package bslsjdk.ornithnpu;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Random-map neural Q-learning lab. Every generated map has a verified route.
 * A small pure-Java MLP is used; no external runtime is required.
 */
public final class GridWorldLabActivity extends Activity {
    private static final int SIZE = 12;
    private static final int CELLS = SIZE * SIZE;
    // Preserve the original 8 features, then append the complete wall map and the last 8 positions.
    private static final int BASE_FEATURES = 8;
    private static final int MAP_FEATURES = CELLS;
    private static final int HISTORY_LENGTH = 8;
    private static final int INPUT_SIZE = BASE_FEATURES + MAP_FEATURES + HISTORY_LENGTH * 2;
    private static final int DEFAULT_HIDDEN = 32;
    private static final int MIN_HIDDEN = 8;
    private static final int MAX_HIDDEN = 256;
    private static final int[] DX = {0, 1, 0, -1};
    private static final int[] DY = {-1, 0, 1, 0};
    private static final String[] ACTIONS = {"上", "右", "下", "左"};
    private static final int MAX_STEPS = 120;
    private static final int FAST_STREAK_REQUIRED = 100;
    private static final double FAST_STEP_FACTOR = 1.6;
    private static final int FAST_STEP_ALLOWANCE = 2;

    private final Handler main = new Handler(Looper.getMainLooper());
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Random rng = new Random(20261009L);
    private QNet net = new QNet(20261009L, DEFAULT_HIDDEN);
    private final List<Integer> path = new ArrayList<>();
    private TextView status, metrics, activationText, qText, log, hiddenActivationTitle;
    private EditText hiddenSizeInput;
    private String latestReportText = "";
    private Board board;
    private volatile boolean training;
    private volatile boolean cancelTraining;
    private boolean watching;
    private int start, goal, player, moves;
    private Set<Integer> walls = new HashSet<>();
    private double episodeReward;
    private long episodesDone;
    private int evalEpisodes, evalSuccesses;
    private volatile int fastWinStreak;
    private volatile int lastEpisodeSteps = -1;
    private volatile boolean stoppedByMastery;
    private double[] lastHidden = new double[DEFAULT_HIDDEN];
    private double[] lastQ = new double[4];

    private final Runnable watchTick = new Runnable() {
        @Override public void run() {
            if (!watching || isFinishing()) return;
            if (player == goal || moves >= MAX_STEPS) {
                watching = false;
                status.setText(player == goal ? "观察结束：到达目标。" : "观察结束：步数上限，未到达目标。");
                refreshReadout();
                return;
            }
            stepGame();
            if (watching) main.postDelayed(this, 260);
        }
    };

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        loadCheckpoint();
        buildUi();
        resetEpisode();
        status.setText("随机地图实验已就绪。每张地图都先构造通路并经 BFS 验证。");
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(14), dp(12), dp(14), dp(24));
        root.setBackgroundColor(0xFFF3F5F8);
        scroll.addView(root);
        root.addView(text("迷宫强化学习实验 · 随机地图", 22, true));
        root.addView(text("地图 12×12 · 网络 " + INPUT_SIZE + "→隐藏层→4 · 输入包含完整障碍地图和最近 8 个位置；短期路径记忆与长期网络权重分开保存。", 13, false));
        root.addView(text("隐藏层神经元数量（8～256）", 14, true), spaced());
        hiddenSizeInput = new EditText(this);
        hiddenSizeInput.setSingleLine(true);
        hiddenSizeInput.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        hiddenSizeInput.setText(Integer.toString(net.hiddenSize));
        hiddenSizeInput.setHint("例如 32、64、128、256");
        root.addView(hiddenSizeInput, spaced());
        addButton(root, "应用神经元数量（重建网络）", this::applyHiddenSize);
        board = new Board();
        LinearLayout.LayoutParams bp = new LinearLayout.LayoutParams(-1, dp(330));
        bp.topMargin = dp(10);
        root.addView(board, bp);
        LinearLayout controls = new LinearLayout(this);
        controls.setOrientation(LinearLayout.VERTICAL);
        root.addView(controls);
        addButton(controls, "训练 5,000 局", () -> startTraining(5000));
        addButton(controls, "训练 10,000 局", () -> startTraining(10000));
        addButton(controls, "训练 100,000 局", () -> startTraining(100000));
        addButton(controls, "训练 1,000,000 局", () -> startTraining(1000000));
        addButton(controls, "停止训练", this::stopTraining);
        addButton(controls, "观看当前随机地图策略", this::watchPolicy);
        addButton(controls, "生成新随机地图（保证有解）", () -> { watching = false; resetEpisode(); });
        addButton(controls, "单步执行", () -> { watching = false; stepGame(); });
        addButton(controls, "保存网络与报告", this::saveCheckpoint);
        addButton(controls, "复制训练报告（发给 ChatGPT）", this::copyTrainingReport);
        status = text("状态", 13, true);
        metrics = text("", 12, false);
        activationText = text("", 11, false);
        qText = text("", 12, false);
        log = text("日志：尚未开始训练。", 11, false);
        root.addView(status, spaced());
        root.addView(metrics, spaced());
        hiddenActivationTitle = text("隐藏神经元激活（" + net.hiddenSize + " 个）", 15, true);
        root.addView(hiddenActivationTitle, spaced());
        root.addView(activationText, spaced());
        root.addView(text("动作 Q 值", 15, true), spaced());
        root.addView(qText, spaced());
        root.addView(text("实验日志", 15, true), spaced());
        root.addView(log, spaced());
        setContentView(scroll);
        refreshReadout();
    }

    private void startTraining(int count) {
        if (training) { toast("训练已经在运行。"); return; }
        watching = false;
        cancelTraining = false;
        lastCompletedEpisode = 0;
        fastWinStreak = 0;
        lastEpisodeSteps = -1;
        stoppedByMastery = false;
        training = true;
        status.setText("开始随机地图训练：最多 " + count + " 局；连续 " + FAST_STREAK_REQUIRED + " 局快速通关即提前停止。");
        log.setText("每局更换随机地图；每局结束立即检查快速通关条件。");
        worker.execute(() -> {
            long wins = 0, totalSteps = 0;
            int reportEvery = count >= 100000 ? 5000 : 500;
            long started = System.currentTimeMillis();
            for (int ep = 1; ep <= count && !cancelTraining; ep++) {
                MapData map = generateMap(rng);
                int pos = map.start;
                List<Integer> history = new ArrayList<>();
                history.add(pos);
                int steps = 0;
                boolean reachedGoal = false;
                for (; steps < MAX_STEPS && !cancelTraining; steps++) {
                    double[] s = observe(pos, map.goal, map.walls, history);
                    double epsilon = Math.max(0.05, 1.0 - 0.95 * (ep / (double) Math.min(count, 10000)));
                    int action = net.choose(s, epsilon, rng);
                    Transition tr = transition(pos, action, map);
                    List<Integer> nextHistory = appendHistory(history, tr.next);
                    double reward = tr.reward;
                    if (!tr.done && history.contains(tr.next)) reward -= 0.08;
                    double[] next = observe(tr.next, map.goal, map.walls, nextHistory);
                    double target = tr.done ? reward : reward + 0.92 * max(net.forward(next).q);
                    net.update(s, action, target);
                    pos = tr.next;
                    history = nextHistory;
                    if (tr.done) { wins++; reachedGoal = true; break; }
                }
                int episodeSteps = Math.min(steps + 1, MAX_STEPS);
                totalSteps += episodeSteps;
                lastCompletedEpisode = ep;
                // Test the current greedy policy on a separate, newly generated map after every episode.
                MapData masteryMap = generateMap(rng);
                int masterySteps = greedyTestSteps(masteryMap);
                int shortestSteps = shortestDistance(masteryMap);
                lastEpisodeSteps = masterySteps;
                boolean fastWin = masterySteps > 0
                        && masterySteps <= shortestSteps * FAST_STEP_FACTOR + FAST_STEP_ALLOWANCE;
                fastWinStreak = fastWin ? fastWinStreak + 1 : 0;
                if (fastWinStreak >= FAST_STREAK_REQUIRED) {
                    stoppedByMastery = true;
                    break;
                }
                if (ep % reportEvery == 0 || ep == count) {
                    final int finished = ep;
                    final long winCount = wins;
                    final long stepCount = totalSteps;
                    final long elapsed = System.currentTimeMillis() - started;
                    main.post(() -> {
                        status.setText("训练中：" + finished + "/" + count + (cancelTraining ? "（正在停止）" : ""));
                        log.setText(String.format(Locale.US,
                                "本批训练局：%d\n训练期间到达目标：%d（累计，不是独立测试成功率）\n平均步数上限内步数：%.1f\n已用时：%.1f 秒",
                                finished, winCount, stepCount / (double) Math.max(1, finished), elapsed / 1000.0));
                        refreshReadout();
                    });
                }
            }
            training = false;
            // Count completed episodes, including a partial run stopped by the user.
            episodesDone += lastCompletedEpisode;
            main.post(() -> {
                evaluatePolicy(100);
                saveCheckpoint();
                status.setText(cancelTraining ? "训练已手动停止并保存。"
                        : stoppedByMastery ? "提前停止：已连续 " + FAST_STREAK_REQUIRED + " 局快速通关。"
                        : "本次训练局数上限已达到。");
                log.setText(String.format(Locale.US,
                        "本次训练局数：%d\n连续快速通关：%d/%d\n上一局步数：%s\n快速通关判定：成功且步数≤最短路×%.1f+%d\n独立随机地图测试：%d/100（%.1f%%）",
                        lastCompletedEpisode, fastWinStreak, FAST_STREAK_REQUIRED,
                        lastEpisodeSteps < 0 ? "未成功" : Integer.toString(lastEpisodeSteps),
                        FAST_STEP_FACTOR, FAST_STEP_ALLOWANCE, evalSuccesses,
                        100.0 * evalSuccesses / Math.max(1, evalEpisodes)));
                refreshReadout();
            });
        });
    }

    private volatile int lastCompletedEpisode;


    private void applyHiddenSize() {
        if (training) { toast("请先停止训练，再调整神经元数量。"); return; }
        String raw = hiddenSizeInput == null ? "" : hiddenSizeInput.getText().toString().trim();
        final int requested;
        try { requested = Integer.parseInt(raw); }
        catch (NumberFormatException e) { toast("请输入 8～256 的整数。"); return; }
        if (requested < MIN_HIDDEN || requested > MAX_HIDDEN) { toast("神经元数量范围是 8～256。"); return; }
        if (requested == net.hiddenSize) { toast("当前网络已经是 " + requested + " 个隐藏神经元。"); return; }
        net = new QNet(System.nanoTime(), requested);
        lastHidden = new double[requested];
        if (hiddenActivationTitle != null) hiddenActivationTitle.setText("隐藏神经元激活（" + requested + " 个）");
        lastQ = new double[4];
        fastWinStreak = 0;
        lastEpisodeSteps = -1;
        stoppedByMastery = false;
        evalEpisodes = 0;
        evalSuccesses = 0;
        resetEpisode();
        status.setText("网络已重建：8→" + requested + "→4。新网络权重随机初始化，旧网络文件保留到下次保存。请重新训练并比较独立测试成功率。");
        log.setText("已调整隐藏层大小。更换网络规模会重置当前权重，不会把旧网络的权重错误套到新结构上。");
        refreshReadout();
    }

    private void stopTraining() {
        if (!training) { toast("当前没有正在运行的训练。"); return; }
        cancelTraining = true;
        status.setText("已请求停止，将在当前训练步结束后保存。");
    }

    private void evaluatePolicy(int episodes) {
        evalEpisodes = episodes;
        evalSuccesses = 0;
        Random testRng = new Random(System.nanoTime() ^ episodesDone);
        for (int i = 0; i < episodes; i++) {
            MapData map = generateMap(testRng);
            int p = map.start;
            Set<Integer> seen = new HashSet<>();
            List<Integer> history = new ArrayList<>(); history.add(p);
            for (int t = 0; t < MAX_STEPS; t++) {
                int a = argmax(net.forward(observe(p, map.goal, map.walls, history)).q);
                Transition tr = transition(p, a, map);
                p = tr.next;
                history = appendHistory(history, p);
                if (tr.done) { evalSuccesses++; break; }
                if (!seen.add(p * 4 + a)) break;
            }
        }
    }

    private void watchPolicy() {
        if (training) { toast("训练还没结束。"); return; }
        watching = false;
        resetEpisode();
        watching = true;
        main.post(watchTick);
    }

    private void stepGame() {
        if (training || player == goal || moves >= MAX_STEPS) return;
        Forward f = net.forward(observe(player, goal, walls, path));
        lastHidden = f.h;
        lastQ = f.q;
        int action = argmax(f.q);
        Transition tr = transition(player, action, new MapData(start, goal, walls, null));
        player = tr.next;
        moves++;
        episodeReward += tr.reward;
        path.add(player);
        board.invalidate();
        refreshReadout();
        if (tr.done) {
            status.setText("成功：到达终点！步数 " + moves + "，奖励 " + fmt(episodeReward));
            watching = false;
        } else if (moves >= MAX_STEPS) {
            status.setText("本局结束：达到步数上限，未成功。");
            watching = false;
        } else {
            status.setText("动作：" + ACTIONS[action] + "，位置 (" + (player % SIZE) + "," + (player / SIZE) + ")");
        }
    }

    private void resetEpisode() {
        MapData map = generateMap(rng);
        start = map.start;
        goal = map.goal;
        walls = map.walls;
        player = start;
        moves = 0;
        episodeReward = 0;
        path.clear();
        path.add(player);
        Forward f = net.forward(observe(player, goal, walls, path));
        lastHidden = f.h;
        lastQ = f.q;
        if (board != null) board.invalidate();
        refreshReadout();
    }

    /**
     * Guarantee a path by first building a randomized monotone route from start to goal.
     * Walls are only placed outside that route. BFS is still run as a defensive verification.
     */
    private MapData generateMap(Random random) {
        int s = random.nextInt(CELLS);
        int g;
        do { g = random.nextInt(CELLS); } while (g == s);
        List<Integer> moves = new ArrayList<>();
        int sx = s % SIZE, sy = s / SIZE, gx = g % SIZE, gy = g / SIZE;
        for (int i = 0; i < Math.abs(gx - sx); i++) moves.add(gx > sx ? 1 : 3);
        for (int i = 0; i < Math.abs(gy - sy); i++) moves.add(gy > sy ? 2 : 0);
        Collections.shuffle(moves, random);
        Set<Integer> safeRoute = new HashSet<>();
        int p = s;
        safeRoute.add(p);
        for (int a : moves) { p = (p % SIZE + DX[a]) + (p / SIZE + DY[a]) * SIZE; safeRoute.add(p); }
        Set<Integer> mapWalls = new HashSet<>();
        for (int cell = 0; cell < CELLS; cell++) {
            if (cell != s && cell != g && !safeRoute.contains(cell) && random.nextDouble() < 0.27) mapWalls.add(cell);
        }
        MapData result = new MapData(s, g, mapWalls, safeRoute);
        if (!reachable(result)) return generateMap(random);
        return result;
    }

    /** Returns steps to goal for greedy policy, or -1 if it fails/loops. */
    private int greedyTestSteps(MapData map) {
        int p = map.start;
        Set<Integer> seen = new HashSet<>();
        for (int step = 1; step <= MAX_STEPS; step++) {
            int action = argmax(net.forward(observe(p, map.goal, map.walls, history)).q);
            Transition tr = transition(p, action, map);
            p = tr.next;
            history = appendHistory(history, p);
            if (tr.done) return step;
            if (!seen.add(p * 4 + action)) return -1;
        }
        return -1;
    }

    private int shortestDistance(MapData map) {
        int[] distance = new int[CELLS];
        java.util.Arrays.fill(distance, -1);
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        queue.add(map.start);
        distance[map.start] = 0;
        while (!queue.isEmpty()) {
            int p = queue.removeFirst();
            if (p == map.goal) return distance[p];
            for (int a = 0; a < 4; a++) {
                int x = p % SIZE + DX[a], y = p / SIZE + DY[a];
                if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) continue;
                int n = y * SIZE + x;
                if (!map.walls.contains(n) && distance[n] < 0) {
                    distance[n] = distance[p] + 1;
                    queue.addLast(n);
                }
            }
        }
        return MAX_STEPS;
    }

    private boolean reachable(MapData map) {
        boolean[] seen = new boolean[CELLS];
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        queue.add(map.start); seen[map.start] = true;
        while (!queue.isEmpty()) {
            int p = queue.removeFirst();
            if (p == map.goal) return true;
            for (int a = 0; a < 4; a++) {
                int x = p % SIZE + DX[a], y = p / SIZE + DY[a];
                if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) continue;
                int n = y * SIZE + x;
                if (!map.walls.contains(n) && !seen[n]) { seen[n] = true; queue.addLast(n); }
            }
        }
        return false;
    }

    private double[] observe(int pos, int target, Set<Integer> mapWalls) {
        return observe(pos, target, mapWalls, path);
    }

    private static List<Integer> appendHistory(List<Integer> old, int next) {
        ArrayList<Integer> result = new ArrayList<>(HISTORY_LENGTH);
        int start = Math.max(0, old.size() - HISTORY_LENGTH + 1);
        for (int i = start; i < old.size(); i++) result.add(old.get(i));
        result.add(next);
        if (result.size() > HISTORY_LENGTH) result.remove(0);
        return result;
    }

    private double[] observe(int pos, int target, Set<Integer> mapWalls, List<Integer> history) {
        int x = pos % SIZE, y = pos / SIZE, gx = target % SIZE, gy = target / SIZE;
        double[] s = new double[INPUT_SIZE];
        s[0] = x / (double)(SIZE - 1); s[1] = y / (double)(SIZE - 1);
        s[2] = gx / (double)(SIZE - 1); s[3] = gy / (double)(SIZE - 1);
        for (int a = 0; a < 4; a++) {
            int nx = x + DX[a], ny = y + DY[a];
            s[4 + a] = nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE
                    || mapWalls.contains(ny * SIZE + nx) ? 1.0 : 0.0;
        }
        for (int cell = 0; cell < CELLS; cell++) s[BASE_FEATURES + cell] = mapWalls.contains(cell) ? 1.0 : 0.0;
        int historyStart = Math.max(0, history.size() - HISTORY_LENGTH);
        int count = Math.min(HISTORY_LENGTH, history.size());
        for (int i = 0; i < count; i++) {
            int cell = history.get(historyStart + i);
            s[BASE_FEATURES + MAP_FEATURES + i * 2] = (cell % SIZE) / (double)(SIZE - 1);
            s[BASE_FEATURES + MAP_FEATURES + i * 2 + 1] = (cell / SIZE) / (double)(SIZE - 1);
        }
        return s;
    }

    private Transition transition(int pos, int action, MapData map) {
        int x = pos % SIZE + DX[action], y = pos / SIZE + DY[action];
        if (x < 0 || x >= SIZE || y < 0 || y >= SIZE || map.walls.contains(y * SIZE + x))
            return new Transition(pos, -0.12, false);
        int next = y * SIZE + x;
        if (next == map.goal) return new Transition(next, 1.0, true);
        return new Transition(next, -0.025, false);
    }

    private void refreshReadout() {
        if (board != null) board.invalidate();
        if (metrics != null) metrics.setText(String.format(Locale.US,
                "训练局数：%d\n地图：%d×%d · 隐藏层：%d · 当前步数：%d/%d\n连续快速通关：%d/%d\n独立随机地图测试：%d/%d（%.1f%%）\n本局奖励：%.3f",
                episodesDone, SIZE, SIZE, net.hiddenSize, moves, MAX_STEPS, fastWinStreak, FAST_STREAK_REQUIRED,
                evalSuccesses, evalEpisodes, 100.0 * evalSuccesses / Math.max(1, evalEpisodes), episodeReward));
        if (activationText != null) {
            StringBuilder b = new StringBuilder();
            for (int i = 0; i < lastHidden.length; i++) {
                b.append(String.format(Locale.US, "H%02d % .3f%s", i + 1, lastHidden[i], i % 2 == 1 ? "\n" : "     "));
            }
            activationText.setText(b.toString());
        }
        if (qText != null) qText.setText(String.format(Locale.US,
                "↑ 上：% .4f\n→ 右：% .4f\n↓ 下：% .4f\n← 左：% .4f",
                lastQ[0], lastQ[1], lastQ[2], lastQ[3]));
    }

    private void saveCheckpoint() {
        try {
            File dir = new File(getFilesDir(), "gridworld-lab");
            if (!dir.exists() && !dir.mkdirs()) throw new IllegalStateException("无法创建实验目录");
            JSONObject checkpoint = net.toJson();
            checkpoint.put("episodesTrained", episodesDone);
            checkpoint.put("savedAt", System.currentTimeMillis());
            write(new File(dir, "q_network.json"), checkpoint.toString(2));
            JSONObject report = new JSONObject();
            report.put("format", "aimeng-android-random-gridworld-report/v2");
            report.put("mapSize", SIZE);
            report.put("inputSize", INPUT_SIZE);
            report.put("inputFeatures", "8 legacy local features + 144 full wall-map cells + last 8 positions (x,y)");
            report.put("shortTermMemory", "last 8 positions per episode; repeated-visit penalty=-0.08 during training");
            report.put("longTermMemory", "trained Q-network weights saved in q_network.json");
            report.put("hiddenSize", net.hiddenSize);
            report.put("episodesTrained", episodesDone);
            report.put("evaluationType", "fresh_random_maps");
            report.put("evaluationEpisodes", evalEpisodes);
            report.put("evaluationSuccesses", evalSuccesses);
            report.put("successRate", evalSuccesses / (double)Math.max(1, evalEpisodes));
            report.put("fastWinStreak", fastWinStreak);
            report.put("fastWinStreakRequired", FAST_STREAK_REQUIRED);
            report.put("lastEpisodeSteps", lastEpisodeSteps);
            report.put("stoppedByMastery", stoppedByMastery);
            report.put("fastWinRule", "reaches goal and steps <= shortestPathSteps * 1.6 + 2");
            report.put("guaranteedPathGenerator", "randomized_route + off-route walls + BFS validation");
            report.put("note", "On-device randomized-map test; evaluation maps are generated independently.");
            latestReportText = report.toString(2);
            write(new File(dir, "training_report.json"), latestReportText);
            if (status != null) status.setText("网络与报告已保存。可点“复制训练报告”直接复制内容，不必进入安卓应用内部目录。");
        } catch (Exception e) { toast("保存失败：" + e.getMessage()); }
    }

    private void copyTrainingReport() {
        try {
            // Always snapshot the current state first, so the copied report matches the visible experiment.
            saveCheckpoint();
            if (latestReportText == null || latestReportText.trim().isEmpty()) {
                toast("报告尚未生成，请稍后再试。");
                return;
            }
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard == null) { toast("系统剪贴板不可用。"); return; }
            clipboard.setPrimaryClip(ClipData.newPlainText("AIMENG 随机地图训练报告", latestReportText));
            toast("训练报告已复制。现在可直接粘贴到 ChatGPT 对话里。");
        } catch (Exception e) {
            toast("复制报告失败：" + e.getMessage());
        }
    }

    private void loadCheckpoint() {
        try {
            File f = new File(new File(getFilesDir(), "gridworld-lab"), "q_network.json");
            if (!f.isFile()) return;
            byte[] bytes = new byte[(int)Math.min(f.length(), 256 * 1024)];
            try (java.io.FileInputStream in = new java.io.FileInputStream(f)) {
                int n = in.read(bytes);
                if (n > 0) {
                    JSONObject saved = new JSONObject(new String(bytes, 0, n, StandardCharsets.UTF_8));
                    int savedHidden = saved.optInt("hiddenSize", DEFAULT_HIDDEN);
                    int oldInputSize = saved.optInt("inputSize", -1);
                    if ((oldInputSize == 8 || oldInputSize == INPUT_SIZE) && savedHidden >= MIN_HIDDEN && savedHidden <= MAX_HIDDEN) {
                        net = new QNet(20261009L, savedHidden);
                        lastHidden = new double[savedHidden];
                        net.load(saved);
                    }
                }
            }
        } catch (Exception ignored) { }
    }

    private static void write(File f, String s) throws Exception {
        try (FileOutputStream out = new FileOutputStream(f)) { out.write(s.getBytes(StandardCharsets.UTF_8)); }
    }
    private static double max(double[] a) { double m = a[0]; for (double v : a) m = Math.max(m, v); return m; }
    private static int argmax(double[] a) { int b = 0; for (int i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }
    private static String fmt(double v) { return String.format(Locale.US, "%.3f", v); }
    private int dp(float n) { return (int)(n * getResources().getDisplayMetrics().density + 0.5f); }
    private TextView text(String s, int size, boolean bold) {
        TextView t = new TextView(this); t.setText(s); t.setTextSize(size); t.setTextColor(0xFF182230);
        if (bold) t.setTypeface(null, 1); return t;
    }
    private LinearLayout.LayoutParams spaced() { LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, -2); p.topMargin = dp(8); return p; }
    private void addButton(LinearLayout parent, String name, Runnable action) {
        Button b = new Button(this); b.setText(name); b.setAllCaps(false); b.setOnClickListener(v -> action.run());
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, dp(46)); p.topMargin = dp(5); parent.addView(b, p);
    }
    private void toast(String s) { Toast.makeText(this, s, Toast.LENGTH_SHORT).show(); }
    @Override protected void onDestroy() { watching = false; cancelTraining = true; main.removeCallbacks(watchTick); worker.shutdownNow(); super.onDestroy(); }

    private static final class MapData {
        final int start, goal; final Set<Integer> walls; final Set<Integer> route;
        MapData(int s, int g, Set<Integer> w, Set<Integer> r) { start=s; goal=g; walls=w; route=r; }
    }
    private static final class Transition {
        final int next; final double reward; final boolean done;
        Transition(int n, double r, boolean d) { next=n; reward=r; done=d; }
    }
    private final class Board extends View {
        private final Paint p = new Paint(3);
        Board() { super(GridWorldLabActivity.this); setBackgroundColor(0xFFFFFFFF); }
        @Override protected void onDraw(Canvas c) {
            super.onDraw(c);
            float cell = Math.min(getWidth() / (float)SIZE, getHeight() / (float)SIZE);
            float ox = (getWidth() - cell * SIZE) / 2f, oy = (getHeight() - cell * SIZE) / 2f;
            for (int y=0;y<SIZE;y++) for(int x=0;x<SIZE;x++) {
                int id=y*SIZE+x; float l=ox+x*cell, t=oy+y*cell;
                p.setColor(walls.contains(id)?0xFF344054:0xFFF9FAFB);
                c.drawRoundRect(new RectF(l+1,t+1,l+cell-1,t+cell-1),3,3,p);
                if (path.contains(id) && id!=player && id!=goal && !walls.contains(id)) { p.setColor(0xFFB2DDFF); c.drawCircle(l+cell/2,t+cell/2,cell*.12f,p); }
                if(id==start){p.setColor(0xFF7F56D9);c.drawCircle(l+cell/2,t+cell/2,cell*.22f,p);}
                if(id==goal){p.setColor(0xFF12B76A);c.drawCircle(l+cell/2,t+cell/2,cell*.25f,p);}
                if(id==player){p.setColor(0xFF1570EF);c.drawCircle(l+cell/2,t+cell/2,cell*.22f,p);}
            }
        }
    }
    private static final class QNet {
        final int hiddenSize;
        final int inputSize = INPUT_SIZE;
        final double[][] w1, w2;
        final double[] b1, b2;
        QNet(long seed, int hiddenSize) {
            this.hiddenSize = hiddenSize;
            w1 = new double[hiddenSize][INPUT_SIZE]; w2 = new double[4][hiddenSize];
            b1 = new double[hiddenSize]; b2 = new double[4];
            Random r=new Random(seed);
            for(int j=0;j<hiddenSize;j++)for(int i=0;i<INPUT_SIZE;i++)w1[j][i]=r.nextDouble()*0.5-0.25;
            for(int a=0;a<4;a++)for(int j=0;j<hiddenSize;j++)w2[a][j]=r.nextDouble()*0.5-0.25;
        }
        Forward forward(double[] x) {
            double[] h=new double[hiddenSize], pre=new double[hiddenSize], q=new double[4];
            for(int j=0;j<hiddenSize;j++){double v=b1[j];for(int i=0;i<inputSize;i++)v+=w1[j][i]*x[i];pre[j]=v;h[j]=Math.max(0,v);}
            for(int a=0;a<4;a++){q[a]=b2[a];for(int j=0;j<hiddenSize;j++)q[a]+=w2[a][j]*h[j];}
            return new Forward(q,h,pre);
        }
        int choose(double[] s,double eps,Random r){if(r.nextDouble()<eps)return r.nextInt(4);return argmax(forward(s).q);}
        void update(double[] x,int action,double target){
            Forward f=forward(x);double grad=Math.max(-1,Math.min(1,f.q[action]-target));double[] old=w2[action].clone();
            for(int j=0;j<hiddenSize;j++)w2[action][j]-=0.003*grad*f.h[j];b2[action]-=0.003*grad;
            for(int j=0;j<hiddenSize;j++)if(f.pre[j]>0){double back=grad*old[j];for(int i=0;i<inputSize;i++)w1[j][i]-=0.003*back*x[i];b1[j]-=0.003*back;}
        }
        JSONObject toJson() throws Exception {
            JSONObject o=new JSONObject();o.put("format","aimeng-android-random-gridworld-qnet/v4");o.put("inputSize",inputSize);o.put("hiddenSize",hiddenSize);o.put("outputSize",4);
            o.put("w1",matrix(w1));o.put("w2",matrix(w2));o.put("b1",array(b1));o.put("b2",array(b2));return o;
        }
        void load(JSONObject o)throws Exception{
            int savedInputs=o.optInt("inputSize",-1);
            if((savedInputs!=8&&savedInputs!=inputSize)||o.optInt("hiddenSize")!=hiddenSize||o.optInt("outputSize")!=4)return;
            JSONArray savedW1=o.getJSONArray("w1");
            if(savedW1.length()!=hiddenSize)throw new Exception("隐藏层维度不匹配");
            for(int j=0;j<hiddenSize;j++){
                JSONArray row=savedW1.getJSONArray(j);
                if(row.length()!=savedInputs)throw new Exception("输入维度不匹配");
                for(int i=0;i<savedInputs;i++)w1[j][i]=row.getDouble(i);
            }
            readMatrix(o.getJSONArray("w2"),w2);readArray(o.getJSONArray("b1"),b1);readArray(o.getJSONArray("b2"),b2);
        }
        static JSONArray array(double[] a)throws Exception{JSONArray j=new JSONArray();for(double v:a)j.put(v);return j;}
        static JSONArray matrix(double[][] a)throws Exception{JSONArray j=new JSONArray();for(double[] r:a)j.put(array(r));return j;}
        static void readArray(JSONArray j,double[] a)throws Exception{if(j.length()!=a.length)throw new Exception("维度不匹配");for(int i=0;i<a.length;i++)a[i]=j.getDouble(i);}
        static void readMatrix(JSONArray j,double[][] a)throws Exception{if(j.length()!=a.length)throw new Exception("维度不匹配");for(int i=0;i<a.length;i++)readArray(j.getJSONArray(i),a[i]);}
    }
    private static final class Forward { final double[] q,h,pre; Forward(double[] q,double[] h,double[] p){this.q=q;this.h=h;pre=p;} }
}
