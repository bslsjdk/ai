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
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
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
    // Each decision performs two shared-weight thought cycles. Cycle 2 receives
    // cycle 1 activations through a trainable hidden-to-hidden communication matrix.
    private static final int THOUGHT_CYCLES = 2;
    private static final int DEFAULT_HIDDEN = 64;
    private static final int MIN_HIDDEN = 8;
    private static final int MAX_HIDDEN = 256;
    private static final int[] DX = {0, 1, 0, -1};
    private static final int[] DY = {-1, 0, 1, 0};
    private static final String[] ACTIONS = {"上", "右", "下", "左"};
    private static final int MAX_STEPS = 120;
    // Bounded episodic replay: ~5.5 MiB of state vectors at capacity 2048.
    private static final int REPLAY_CAPACITY = 2048;
    private static final int REPLAY_WARMUP = 64;
    private static final int REPLAY_UPDATE_INTERVAL = 4;
    private static final int FAST_STREAK_REQUIRED = 5;
    private static final int MASTERY_CHECK_INTERVAL = 100;
    private static final int MASTERY_EVAL_EPISODES = 20;
    private static final double MASTERY_FAST_RATE_REQUIRED = 0.90;
    private static final double FAST_STEP_FACTOR = 1.6;
    private static final int FAST_STEP_ALLOWANCE = 2;
    // Reward schema v2 adds bounded BFS shortest-distance shaping.
    private static final int REWARD_VERSION = 2;
    private static final double DISTANCE_REWARD_PER_STEP = 0.10;

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
        loadReplayMemory();
        buildUi();
        resetEpisode();
        status.setText("随机地图实验已就绪。每张地图都经 BFS 验证；训练奖励 v2 已加入真实最短路距离反馈。");
        log.setText("奖励版本 v2：每靠近终点一步 +0.10，每远离一步 -0.10（按 BFS 可达距离）；原有到达奖励、碰墙惩罚和重复访问惩罚保留。旧奖励版本权重会自动备份，不会混合训练。");
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
        status.setText("开始循环思考训练：每次决策进行 " + THOUGHT_CYCLES + " 轮神经元信息传递；最多 " + count + " 局。");
        log.setText("隐藏神经元通过可训练的循环连接互相传递激活，并用 BPTT 学习连接权重。每 " + MASTERY_CHECK_INTERVAL + " 局评估 " + MASTERY_EVAL_EPISODES + " 张独立地图；连续 " + FAST_STREAK_REQUIRED + " 批达标才提前停止。");
        worker.execute(() -> {
            long wins = 0, totalSteps = 0, replayUpdates = 0;
            long globalEnvironmentSteps = 0;
            ReplayMemory replay = replayMemory;
            int reportEvery = count >= 100000 ? 5000 : 500;
            long started = System.currentTimeMillis();
            for (int ep = 1; ep <= count && !cancelTraining; ep++) {
                MapData map = generateMap(rng);
                int pos = map.start;
                int[] mapDistances = distanceMap(map);
                List<Integer> history = new ArrayList<>();
                history.add(pos);
                // Reuse the two state vectors throughout this episode; observations are
                // overwritten in place after the preceding update has consumed them.
                double[] stateBuffer = new double[INPUT_SIZE];
                double[] nextBuffer = new double[INPUT_SIZE];
                double[] wallFeatures = new double[CELLS];
                for (int cell = 0; cell < CELLS; cell++) wallFeatures[cell] = map.walls.contains(cell) ? 1.0 : 0.0;
                int steps = 0;
                boolean reachedGoal = false;
                for (; steps < MAX_STEPS && !cancelTraining; steps++) {
                    observeInto(stateBuffer, pos, map.goal, map.walls, history, wallFeatures);
                    double[] s = stateBuffer;
                    double epsilon = Math.max(0.05, 1.0 - 0.95 * (ep / (double) Math.min(count, 10000)));
                    // Avoid duplicate forwards: only compute Q values when greedy action selection needs them.
                    Forward currentForward = null;
                    int action;
                    if (rng.nextDouble() < epsilon) {
                        action = rng.nextInt(4);
                    } else {
                        currentForward = net.forward(s);
                        action = argmax(currentForward.q);
                    }
                    Transition tr = transition(pos, action, map);
                    List<Integer> nextHistory = appendHistory(history, tr.next);
                    double reward = tr.reward;
                    if (!tr.done && history.contains(tr.next)) reward -= 0.08;
                    // Dense feedback uses the actual shortest path through walls, not Manhattan distance.
                    int oldDistance = mapDistances[pos];
                    int nextDistance = mapDistances[tr.next];
                    if (oldDistance >= 0 && nextDistance >= 0) {
                        reward += DISTANCE_REWARD_PER_STEP * (oldDistance - nextDistance);
                    }
                    observeInto(nextBuffer, tr.next, map.goal, map.walls, nextHistory, wallFeatures);
                    // TD target needs only max(Q), not a retained Forward/BPTT cache.
                    double target = tr.done ? reward : reward + 0.92 * net.maxQ(nextBuffer);
                    net.update(s, currentForward, action, target);
                    replay.add(s, action, reward, nextBuffer, tr.done);
                    globalEnvironmentSteps++;
                    // Revisit a random past transition every four environment steps.
                    // This reuses experience without multiplying compute by a large factor.
                    if (replay.size >= REPLAY_WARMUP
                            && globalEnvironmentSteps % REPLAY_UPDATE_INTERVAL == 0) {
                        int ri = replay.sample(rng);
                        double replayTarget = replay.dones[ri] ? replay.rewards[ri]
                                : replay.rewards[ri] + 0.92 * net.maxQ(replay.nextStates[ri]);
                        Forward replayForward = net.forward(replay.states[ri]);
                        net.update(replay.states[ri], replayForward, replay.actions[ri], replayTarget);
                        replayUpdates++;
                    }
                    pos = tr.next;
                    history = nextHistory;
                    if (tr.done) { wins++; reachedGoal = true; break; }
                }
                int episodeSteps = Math.min(steps + 1, MAX_STEPS);
                totalSteps += episodeSteps;
                lastCompletedEpisode = ep;
                // Periodic held-out evaluation avoids paying for another full episode every update.
                if (ep % MASTERY_CHECK_INTERVAL == 0) {
                    int fastWins = 0;
                    int lastSteps = -1;
                    for (int test = 0; test < MASTERY_EVAL_EPISODES; test++) {
                        MapData masteryMap = generateMap(rng);
                        int masterySteps = greedyTestSteps(masteryMap);
                        int shortestSteps = shortestDistance(masteryMap);
                        lastSteps = masterySteps;
                        if (masterySteps > 0
                                && masterySteps <= shortestSteps * FAST_STEP_FACTOR + FAST_STEP_ALLOWANCE) fastWins++;
                    }
                    lastEpisodeSteps = lastSteps;
                    boolean masteryPass = fastWins >= Math.ceil(MASTERY_EVAL_EPISODES * MASTERY_FAST_RATE_REQUIRED);
                    fastWinStreak = masteryPass ? fastWinStreak + 1 : 0;
                    if (fastWinStreak >= FAST_STREAK_REQUIRED) {
                        stoppedByMastery = true;
                        break;
                    }
                }
                if (ep % reportEvery == 0 || ep == count) {
                    final int finished = ep;
                    final long winCount = wins;
                    final long stepCount = totalSteps;
                    final long elapsed = System.currentTimeMillis() - started;
                    saveTrainingCheckpoint(finished);
                    main.post(() -> {
                        status.setText("训练中：" + finished + "/" + count + (cancelTraining ? "（正在停止）" : ""));
                        log.setText(String.format(Locale.US,
                                "本批训练局：%d\n训练期间到达目标：%d（累计，不是独立测试成功率）\n平均步数上限内步数：%.1f\n已用时：%.1f 秒",
                                finished, winCount, stepCount / (double) Math.max(1, finished), elapsed / 1000.0));
                        refreshReadout();
                    });
                }
            }
            long elapsedMs = Math.max(1L, System.currentTimeMillis() - started);
            lastTrainingElapsedMs = elapsedMs;
            lastTrainingEnvironmentSteps = totalSteps;
            lastTrainingEpisodesPerSecond = lastCompletedEpisode * 1000.0 / elapsedMs;
            lastTrainingStepsPerSecond = totalSteps * 1000.0 / elapsedMs;
            lastReplayUpdates = replayUpdates;
            lastReplaySize = replay.size;
            training = false;
            // Count completed episodes, including a partial run stopped by the user.
            episodesDone += lastCompletedEpisode;
            main.post(() -> {
                evaluatePolicy(100);
                saveCheckpoint();
                status.setText(cancelTraining ? "训练已手动停止并保存。"
                        : stoppedByMastery ? "提前停止：已连续 " + FAST_STREAK_REQUIRED + " 局快速通关。"
                        : "本次训练局数上限已达到。");
                int[] finalGoalProbe = evaluateGoalAdjacent(100);
                log.setText(String.format(Locale.US,
                        "本次训练局数：%d\n连续达标评估批次：%d/%d（每批 %d 张地图，快速通关率≥%.0f%%）\n最近评估地图步数：%s\n快速通关判定：成功且步数≤最短路×%.1f+%d\n独立随机地图测试：%d/100（%.1f%%）\n终点相邻贪心决策：%d/%d（%.1f%%）",
                        lastCompletedEpisode, fastWinStreak, FAST_STREAK_REQUIRED, MASTERY_EVAL_EPISODES,
                        MASTERY_FAST_RATE_REQUIRED * 100.0,
                        lastEpisodeSteps < 0 ? "未成功" : Integer.toString(lastEpisodeSteps),
                        FAST_STEP_FACTOR, FAST_STEP_ALLOWANCE, evalSuccesses,
                        100.0 * evalSuccesses / Math.max(1, evalEpisodes), finalGoalProbe[0],
                        finalGoalProbe[1], 100.0 * finalGoalProbe[0] / Math.max(1, finalGoalProbe[1])));
                refreshReadout();
            });
        });
    }

    private volatile int lastCompletedEpisode;
    private volatile long lastTrainingElapsedMs;
    private volatile long lastTrainingEnvironmentSteps;
    private volatile double lastTrainingEpisodesPerSecond;
    private volatile double lastTrainingStepsPerSecond;
    private volatile long lastReplayUpdates;
    private volatile int lastReplaySize;
    private volatile String replayMemoryError = "";
    private final ReplayMemory replayMemory = new ReplayMemory();


    private void applyHiddenSize() {
        if (training) { toast("请先停止训练，再调整神经元数量。"); return; }
        String raw = hiddenSizeInput == null ? "" : hiddenSizeInput.getText().toString().trim();
        final int requested;
        try { requested = Integer.parseInt(raw); }
        catch (NumberFormatException e) { toast("请输入 8～256 的整数。"); return; }
        if (requested < MIN_HIDDEN || requested > MAX_HIDDEN) { toast("神经元数量范围是 8～256。"); return; }
        if (requested == net.hiddenSize) { toast("当前网络已经是 " + requested + " 个隐藏神经元。"); return; }
        net = new QNet(System.nanoTime(), requested);
        episodesDone = 0;
        lastCompletedEpisode = 0;
        lastHidden = new double[requested];
        if (hiddenActivationTitle != null) hiddenActivationTitle.setText("隐藏神经元激活（" + requested + " 个）");
        lastQ = new double[4];
        fastWinStreak = 0;
        lastEpisodeSteps = -1;
        stoppedByMastery = false;
        evalEpisodes = 0;
        evalSuccesses = 0;
        resetEpisode();
        status.setText("网络已重建：" + INPUT_SIZE + "→" + requested + "→4。新网络权重随机初始化，旧网络文件保留到下次保存。请重新训练并比较独立测试成功率。");
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
        List<Integer> history = new ArrayList<>(); history.add(p);
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

    private int[] distanceMap(MapData map) {
        int[] distance = new int[CELLS];
        java.util.Arrays.fill(distance, -1);
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        queue.add(map.goal);
        distance[map.goal] = 0;
        while (!queue.isEmpty()) {
            int p = queue.removeFirst();
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
        return distance;
    }

    private int[] evaluateGoalAdjacent(int mapCount) {
        int correct = 0, total = 0;
        Random probeRng = new Random(0x51A7E5L);
        for (int m = 0; m < mapCount; m++) {
            MapData map = generateMap(probeRng);
            for (int a = 0; a < 4; a++) {
                int x = map.goal % SIZE - DX[a];
                int y = map.goal / SIZE - DY[a];
                if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) continue;
                int pos = y * SIZE + x;
                if (map.walls.contains(pos)) continue;
                List<Integer> history = new ArrayList<>();
                history.add(pos);
                int selected = argmax(net.forward(observe(pos, map.goal, map.walls, history)).q);
                total++;
                if (selected == a) correct++;
            }
        }
        return new int[]{correct, total};
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
        double[] s = new double[INPUT_SIZE];
        double[] wallFeatures = new double[CELLS];
        for (int cell = 0; cell < CELLS; cell++) wallFeatures[cell] = mapWalls.contains(cell) ? 1.0 : 0.0;
        observeInto(s, pos, target, mapWalls, history, wallFeatures);
        return s;
    }

    private static void observeInto(double[] s, int pos, int target, Set<Integer> mapWalls,
                                    List<Integer> history, double[] wallFeatures) {
        java.util.Arrays.fill(s, 0.0);
        int x = pos % SIZE, y = pos / SIZE, gx = target % SIZE, gy = target / SIZE;
        s[0] = x / (double)(SIZE - 1); s[1] = y / (double)(SIZE - 1);
        s[2] = gx / (double)(SIZE - 1); s[3] = gy / (double)(SIZE - 1);
        for (int a = 0; a < 4; a++) {
            int nx = x + DX[a], ny = y + DY[a];
            s[4 + a] = nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE
                    || mapWalls.contains(ny * SIZE + nx) ? 1.0 : 0.0;
        }
        System.arraycopy(wallFeatures, 0, s, BASE_FEATURES, CELLS);
        int historyStart = Math.max(0, history.size() - HISTORY_LENGTH);
        int count = Math.min(HISTORY_LENGTH, history.size());
        for (int i = 0; i < count; i++) {
            int cell = history.get(historyStart + i);
            s[BASE_FEATURES + MAP_FEATURES + i * 2] = (cell % SIZE) / (double)(SIZE - 1);
            s[BASE_FEATURES + MAP_FEATURES + i * 2 + 1] = (cell / SIZE) / (double)(SIZE - 1);
        }
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
                "训练局数：%d\n地图：%d×%d · 隐藏层：%d · 内部思考：%d 轮\n当前步数：%d/%d\n连续达标评估：%d/%d\n独立随机地图测试：%d/%d（%.1f%%）\n本局奖励：%.3f",
                episodesDone, SIZE, SIZE, net.hiddenSize, THOUGHT_CYCLES, moves, MAX_STEPS, fastWinStreak, FAST_STREAK_REQUIRED,
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
        if (training) {
            toast("训练期间会自动保存检查点；训练结束后再生成最终报告。");
            return;
        }
        try {
            File dir = new File(getFilesDir(), "gridworld-lab");
            if (!dir.exists() && !dir.mkdirs()) throw new IllegalStateException("无法创建实验目录");
            JSONObject checkpoint = net.toJson();
            checkpoint.put("episodesTrained", episodesDone);
            checkpoint.put("rewardVersion", REWARD_VERSION);
            checkpoint.put("rewardShaping", "BFS shortest-distance delta: +0.10 closer, -0.10 farther; base rewards retained");
            checkpoint.put("savedAt", System.currentTimeMillis());
            writeAtomic(new File(dir, "q_network.json"), checkpoint.toString(2));
            saveReplayMemoryAtomic(dir);
            JSONObject report = new JSONObject();
            report.put("format", "aimeng-android-random-gridworld-report/v3");
            report.put("mapSize", SIZE);
            report.put("inputSize", INPUT_SIZE);
            report.put("inputFeatures", "8 local features + 144 wall-map cells + last 8 positions (x,y)");
            report.put("executionBackend", "CPU Java forward/backprop; NPU/GPU not yet wired into this QNet");
            report.put("shortTermMemory", "last 8 positions per episode; repeated-visit penalty=-0.08 during training");
            report.put("longTermMemory", "trained weights in q_network.json plus persistent sampled experience replay in replay_memory.bin");
            // Report configured per-episode history, not the unrelated current UI path length.
            report.put("shortTermMemoryLength", HISTORY_LENGTH);
            report.put("reportPathLength", Math.min(HISTORY_LENGTH, path.size()));
            JSONArray recentPath = new JSONArray();
            for (int i = Math.max(0, path.size() - HISTORY_LENGTH); i < path.size(); i++) recentPath.put(path.get(i));
            report.put("recentPathCellIds", recentPath);
            report.put("hiddenSize", net.hiddenSize);
            report.put("thoughtCycles", THOUGHT_CYCLES);
            report.put("neuronCommunication", "dense recurrent hidden-to-hidden messages");
            report.put("experienceReplay", "bounded random replay buffer; sampled one transition per four environment steps after warmup");
            report.put("experienceReplayCapacity", REPLAY_CAPACITY);
            report.put("experienceReplayWarmup", REPLAY_WARMUP);
            report.put("experienceReplayUpdates", lastReplayUpdates);
            report.put("experienceReplayFinalSize", lastReplaySize);
            report.put("memoryModel", "last 8 positions are episode-local short-term memory; replay_memory.bin persists up to 2048 transitions across launches; q_network.json stores persistent learned weights");
            report.put("replayMemoryPersistence", "atomic binary checkpoint alongside q_network.json");
            report.put("replayMemoryError", replayMemoryError);
            report.put("recurrentTraining", "truncated BPTT across internal thought cycles");
            report.put("executionBackend", "CPU Java recurrent forward/backprop; NPU/GPU not yet wired into this QNet");
            report.put("rewardVersion", REWARD_VERSION);
            report.put("rewardShaping", "BFS shortest-distance delta; base rewards retained; repeated-visit penalty remains -0.08");
            int[] goalProbe = evaluateGoalAdjacent(100);
            report.put("goalAdjacentGreedyCorrect", goalProbe[0]);
            report.put("goalAdjacentGreedyTests", goalProbe[1]);
            report.put("goalAdjacentGreedyAccuracy", goalProbe[0] / (double)Math.max(1, goalProbe[1]));
            report.put("episodesTrained", episodesDone);
            report.put("lastTrainingElapsedMs", lastTrainingElapsedMs);
            report.put("lastTrainingEnvironmentSteps", lastTrainingEnvironmentSteps);
            report.put("lastTrainingEpisodesPerSecond", lastTrainingEpisodesPerSecond);
            report.put("lastTrainingStepsPerSecond", lastTrainingStepsPerSecond);
            report.put("autosaveCheckpoint", "q_network.json, atomic replacement at each progress interval");
            report.put("autosaveError", lastAutosaveError);
            report.put("evaluationType", "fresh_random_maps");
            report.put("evaluationEpisodes", evalEpisodes);
            report.put("evaluationSuccesses", evalSuccesses);
            report.put("successRate", evalSuccesses / (double)Math.max(1, evalEpisodes));
            report.put("fastWinStreak", fastWinStreak);
            report.put("fastWinStreakRequired", FAST_STREAK_REQUIRED);
            report.put("masteryCheckIntervalEpisodes", MASTERY_CHECK_INTERVAL);
            report.put("masteryEvaluationEpisodes", MASTERY_EVAL_EPISODES);
            report.put("masteryFastRateRequired", MASTERY_FAST_RATE_REQUIRED);
            report.put("masteryStreakMeaning", "consecutive evaluation batches, not consecutive training episodes");
            report.put("lastEpisodeSteps", lastEpisodeSteps);
            report.put("stoppedByMastery", stoppedByMastery);
            report.put("fastWinRule", "reaches goal and steps <= shortestPathSteps * 1.6 + 2");
            report.put("guaranteedPathGenerator", "randomized_route + off-route walls + BFS validation");
            report.put("note", "On-device randomized-map test; evaluation maps are generated independently. Throughput values are from the most recent training run and include periodic held-out evaluation overhead.");
            latestReportText = report.toString(2);
            write(new File(dir, "training_report.json"), latestReportText);
            if (status != null) status.setText("网络与报告已保存。可点“复制训练报告”直接复制内容，不必进入安卓应用内部目录。");
        } catch (Exception e) { toast("保存失败：" + e.getMessage()); }
    }

    private void copyTrainingReport() {
        if (training) {
            toast("训练尚未结束。当前权重会周期性自动保存，请结束训练后复制最终报告。");
            return;
        }
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
            byte[] bytes = new byte[(int)Math.min(f.length(), 2 * 1024 * 1024)];
            try (java.io.FileInputStream in = new java.io.FileInputStream(f)) {
                int n = in.read(bytes);
                if (n > 0) {
                    JSONObject saved = new JSONObject(new String(bytes, 0, n, StandardCharsets.UTF_8));
                    int savedHidden = saved.optInt("hiddenSize", DEFAULT_HIDDEN);
                    int oldInputSize = saved.optInt("inputSize", -1);
                    int savedRewardVersion = saved.optInt("rewardVersion", 1);
                    if (savedHidden < MIN_HIDDEN || savedHidden > MAX_HIDDEN) savedHidden = DEFAULT_HIDDEN;
                    if (savedRewardVersion != REWARD_VERSION) {
                        File backup = new File(f.getParentFile(), "q_network_reward_v" + savedRewardVersion + "_backup_" + System.currentTimeMillis() + ".json");
                        if (!f.renameTo(backup)) return; // Never overwrite weights if backup failed.
                        net = new QNet(System.nanoTime(), savedHidden);
                        lastHidden = new double[savedHidden];
                        episodesDone = 0L;
                    } else if (oldInputSize == 8 || oldInputSize == INPUT_SIZE) {
                        episodesDone = Math.max(0L, saved.optLong("episodesTrained", 0L));
                        net = new QNet(20261009L, savedHidden);
                        lastHidden = new double[savedHidden];
                        net.load(saved);
                    }
                }
            }
        } catch (Exception ignored) { }
    }

    private void loadReplayMemory() {
        File dir = new File(getFilesDir(), "gridworld-lab");
        File file = new File(dir, "replay_memory.bin");
        if (!file.isFile()) return;
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            replayMemory.readFrom(in);
            lastReplaySize = replayMemory.size;
            replayMemoryError = "";
        } catch (Exception e) {
            replayMemory.clear();
            lastReplaySize = 0;
            replayMemoryError = e.getClass().getSimpleName() + ": " + String.valueOf(e.getMessage());
        }
    }

    private void saveReplayMemoryAtomic(File dir) throws Exception {
        File target = new File(dir, "replay_memory.bin");
        File temp = new File(dir, "replay_memory.bin.tmp");
        File backup = new File(dir, "replay_memory.bin.bak");
        try (FileOutputStream fos = new FileOutputStream(temp);
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(fos))) {
            replayMemory.writeTo(out);
            out.flush();
            fos.getFD().sync();
        }
        if (backup.exists() && !backup.delete()) throw new IllegalStateException("无法清理经验记忆备份");
        boolean hadTarget = target.exists();
        if (hadTarget && !target.renameTo(backup)) throw new IllegalStateException("无法备份旧经验记忆");
        if (!temp.renameTo(target)) {
            if (hadTarget) backup.renameTo(target);
            throw new IllegalStateException("无法提交经验记忆文件");
        }
        if (backup.exists()) backup.delete();
        replayMemoryError = "";
    }

    private void saveTrainingCheckpoint(int completedEpisodes) {
        try {
            File dir = new File(getFilesDir(), "gridworld-lab");
            if (!dir.exists() && !dir.mkdirs()) throw new IllegalStateException("无法创建实验目录");
            JSONObject checkpoint = net.toJson();
            checkpoint.put("episodesTrained", episodesDone + completedEpisodes);
            checkpoint.put("rewardVersion", REWARD_VERSION);
            checkpoint.put("savedAt", System.currentTimeMillis());
            checkpoint.put("checkpointKind", "periodic_training_autosave");
            writeAtomic(new File(dir, "q_network.json"), checkpoint.toString());
            saveReplayMemoryAtomic(dir);
            lastAutosaveError = "";
        } catch (Exception e) {
            lastAutosaveError = e.getClass().getSimpleName() + ": " + String.valueOf(e.getMessage());
        }
    }

    private volatile String lastAutosaveError = "";

    private static void write(File f, String s) throws Exception {
        try (FileOutputStream out = new FileOutputStream(f)) {
            out.write(s.getBytes(StandardCharsets.UTF_8));
            out.getFD().sync();
        }
    }

    /** Replace checkpoint atomically so a process death cannot leave half-written JSON. */
    private static void writeAtomic(File target, String content) throws Exception {
        File parent = target.getParentFile();
        File temp = new File(parent, target.getName() + ".tmp");
        File backup = new File(parent, target.getName() + ".bak");
        write(temp, content);
        if (backup.exists() && !backup.delete()) throw new IllegalStateException("无法清理旧检查点备份");
        boolean hadTarget = target.exists();
        if (hadTarget && !target.renameTo(backup)) throw new IllegalStateException("无法备份旧检查点");
        if (!temp.renameTo(target)) {
            if (hadTarget) backup.renameTo(target);
            throw new IllegalStateException("无法提交新检查点");
        }
        if (backup.exists()) backup.delete();
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
    private static final class ReplayMemory {
        final double[][] states = new double[REPLAY_CAPACITY][INPUT_SIZE];
        final double[][] nextStates = new double[REPLAY_CAPACITY][INPUT_SIZE];
        final int[] actions = new int[REPLAY_CAPACITY];
        final double[] rewards = new double[REPLAY_CAPACITY];
        final boolean[] dones = new boolean[REPLAY_CAPACITY];
        int size;
        int cursor;

        void add(double[] state, int action, double reward, double[] nextState, boolean done) {
            System.arraycopy(state, 0, states[cursor], 0, INPUT_SIZE);
            System.arraycopy(nextState, 0, nextStates[cursor], 0, INPUT_SIZE);
            actions[cursor] = action;
            rewards[cursor] = reward;
            dones[cursor] = done;
            cursor = (cursor + 1) % REPLAY_CAPACITY;
            if (size < REPLAY_CAPACITY) size++;
        }

        int sample(Random random) {
            return random.nextInt(size);
        }

        void clear() { size = 0; cursor = 0; }

        void writeTo(DataOutputStream out) throws Exception {
            out.writeInt(0x41494D52); // "AIMR"
            out.writeInt(1);
            out.writeInt(size);
            int start = (cursor - size + REPLAY_CAPACITY) % REPLAY_CAPACITY;
            for (int n = 0; n < size; n++) {
                int slot = (start + n) % REPLAY_CAPACITY;
                for (int i = 0; i < INPUT_SIZE; i++) out.writeDouble(states[slot][i]);
                for (int i = 0; i < INPUT_SIZE; i++) out.writeDouble(nextStates[slot][i]);
                out.writeInt(actions[slot]);
                out.writeDouble(rewards[slot]);
                out.writeBoolean(dones[slot]);
            }
        }

        void readFrom(DataInputStream in) throws Exception {
            clear();
            if (in.readInt() != 0x41494D52 || in.readInt() != 1) throw new IllegalStateException("经验记忆文件格式不匹配");
            int count = in.readInt();
            if (count < 0 || count > REPLAY_CAPACITY) throw new IllegalStateException("经验记忆数量越界");
            for (int n = 0; n < count; n++) {
                double[] state = new double[INPUT_SIZE];
                double[] next = new double[INPUT_SIZE];
                for (int i = 0; i < INPUT_SIZE; i++) state[i] = in.readDouble();
                for (int i = 0; i < INPUT_SIZE; i++) next[i] = in.readDouble();
                int action = in.readInt();
                double reward = in.readDouble();
                boolean done = in.readBoolean();
                if (action < 0 || action >= 4 || !Double.isFinite(reward)) throw new IllegalStateException("经验记录无效");
                add(state, action, reward, next, done);
            }
        }
    }

    private static final class QNet {
        final int hiddenSize;
        final int inputSize = INPUT_SIZE;
        final double[][] w1, w2, recurrent;
        final double[] b1, b2;
        // Reused scratch for allocation-free TD-target inference on the training worker.
        private final double[] maxQProjection, maxQFirst, maxQLast;

        QNet(long seed, int hiddenSize) {
            this.hiddenSize = hiddenSize;
            w1 = new double[hiddenSize][INPUT_SIZE];
            w2 = new double[4][hiddenSize];
            recurrent = new double[hiddenSize][hiddenSize];
            b1 = new double[hiddenSize];
            b2 = new double[4];
            maxQProjection = new double[hiddenSize];
            maxQFirst = new double[hiddenSize];
            maxQLast = new double[hiddenSize];
            Random r = new Random(seed);
            double inputScale = 0.5 / Math.sqrt(Math.max(1, INPUT_SIZE));
            double recurrentScale = 0.10 / Math.sqrt(Math.max(1, hiddenSize));
            for (int j = 0; j < hiddenSize; j++) {
                for (int i = 0; i < INPUT_SIZE; i++) w1[j][i] = (r.nextDouble() * 2.0 - 1.0) * inputScale;
                for (int k = 0; k < hiddenSize; k++) recurrent[j][k] = (r.nextDouble() * 2.0 - 1.0) * recurrentScale;
            }
            for (int a = 0; a < 4; a++) {
                for (int j = 0; j < hiddenSize; j++) w2[a][j] = (r.nextDouble() * 2.0 - 1.0) * 0.1;
            }
        }

        Forward forward(double[] x) {
            double[][] h = new double[THOUGHT_CYCLES][hiddenSize];
            double[][] pre = new double[THOUGHT_CYCLES][hiddenSize];
            double[] q = new double[4];
            // Store the shared input projection in pre[0], avoiding another vector allocation.
            for (int j = 0; j < hiddenSize; j++) {
                double v = b1[j];
                for (int i = 0; i < inputSize; i++) if (x[i] != 0.0) v += w1[j][i] * x[i];
                pre[0][j] = v;
                h[0][j] = Math.max(0.0, v);
            }
            for (int t = 1; t < THOUGHT_CYCLES; t++) {
                for (int j = 0; j < hiddenSize; j++) {
                    double v = pre[0][j];
                    for (int k = 0; k < hiddenSize; k++) v += recurrent[j][k] * h[t - 1][k];
                    pre[t][j] = v;
                    h[t][j] = Math.max(0.0, v);
                }
            }
            double[] last = h[THOUGHT_CYCLES - 1];
            for (int a = 0; a < 4; a++) {
                double v = b2[a];
                for (int j = 0; j < hiddenSize; j++) v += w2[a][j] * last[j];
                q[a] = v;
            }
            return new Forward(q, h[THOUGHT_CYCLES - 1], pre[THOUGHT_CYCLES - 1], h, pre);
        }

        /** Allocation-free inference for next-state TD targets; no gradient cache is needed. */
        double maxQ(double[] x) {
            for (int j = 0; j < hiddenSize; j++) {
                double v = b1[j];
                for (int i = 0; i < inputSize; i++) if (x[i] != 0.0) v += w1[j][i] * x[i];
                maxQProjection[j] = v;
                maxQFirst[j] = Math.max(0.0, v);
            }
            for (int j = 0; j < hiddenSize; j++) {
                double v = maxQProjection[j];
                for (int k = 0; k < hiddenSize; k++) v += recurrent[j][k] * maxQFirst[k];
                maxQLast[j] = Math.max(0.0, v);
            }
            double best = Double.NEGATIVE_INFINITY;
            for (int a = 0; a < 4; a++) {
                double v = b2[a];
                for (int j = 0; j < hiddenSize; j++) v += w2[a][j] * maxQLast[j];
                if (v > best) best = v;
            }
            return best;
        }

        int choose(double[] s, double eps, Random r) {
            if (r.nextDouble() < eps) return r.nextInt(4);
            return argmax(forward(s).q);
        }

        void update(double[] x, Forward f, int action, double target) {
            if (f == null) f = forward(x);
            double grad = Math.max(-1.0, Math.min(1.0, f.q[action] - target));
            double[][] h = f.thoughtH;
            double[][] pre = f.thoughtPre;
            // Only two thought cycles are currently configured. Keep backprop scratch
            // one-dimensional: allocating [hidden][input] gradient matrices per step
            // caused avoidable GC pressure on Android.
            double[] deltaLast = new double[hiddenSize];
            double[] deltaFirst = new double[hiddenSize];
            for (int j = 0; j < hiddenSize; j++) {
                double d = grad * w2[action][j];
                deltaLast[j] = pre[THOUGHT_CYCLES - 1][j] > 0.0 ? d : 0.0;
            }
            if (THOUGHT_CYCLES > 1) {
                for (int k = 0; k < hiddenSize; k++) {
                    double d = 0.0;
                    for (int j = 0; j < hiddenSize; j++) d += deltaLast[j] * recurrent[j][k];
                    deltaFirst[k] = pre[0][k] > 0.0 ? d : 0.0;
                }
            }

            final double lr = 0.0015;
            for (int j = 0; j < hiddenSize; j++) {
                double first = THOUGHT_CYCLES > 1 ? deltaFirst[j] : 0.0;
                double last = deltaLast[j];
                b1[j] -= lr * (first + last);
                double combined = first + last;
                if (combined != 0.0) {
                    for (int i = 0; i < inputSize; i++) if (x[i] != 0.0) w1[j][i] -= lr * combined * x[i];
                }
                if (THOUGHT_CYCLES > 1 && last != 0.0) {
                    for (int k = 0; k < hiddenSize; k++) recurrent[j][k] -= lr * last * h[0][k];
                }
            }
            for (int j = 0; j < hiddenSize; j++) w2[action][j] -= lr * grad * h[THOUGHT_CYCLES - 1][j];
            b2[action] -= lr * grad;
        }

        JSONObject toJson() throws Exception {
            JSONObject o = new JSONObject();
            o.put("format", "aimeng-android-random-gridworld-qnet/v5-recurrent");
            o.put("inputSize", inputSize);
            o.put("hiddenSize", hiddenSize);
            o.put("outputSize", 4);
            o.put("thoughtCycles", THOUGHT_CYCLES);
            o.put("communication", "dense_hidden_recurrent_bptt");
            o.put("w1", matrix(w1));
            o.put("w2", matrix(w2));
            o.put("recurrent", matrix(recurrent));
            o.put("b1", array(b1));
            o.put("b2", array(b2));
            return o;
        }

        void load(JSONObject o) throws Exception {
            int savedInputs = o.optInt("inputSize", -1);
            if ((savedInputs != 8 && savedInputs != inputSize)
                    || o.optInt("hiddenSize") != hiddenSize || o.optInt("outputSize") != 4) return;
            JSONArray savedW1 = o.getJSONArray("w1");
            if (savedW1.length() != hiddenSize) throw new Exception("隐藏层维度不匹配");
            for (int j = 0; j < hiddenSize; j++) {
                JSONArray row = savedW1.getJSONArray(j);
                if (row.length() != savedInputs) throw new Exception("输入维度不匹配");
                for (int i = 0; i < savedInputs; i++) w1[j][i] = row.getDouble(i);
                for (int i = savedInputs; i < inputSize; i++) w1[j][i] = 0.0;
            }
            readMatrix(o.getJSONArray("w2"), w2);
            readArray(o.getJSONArray("b1"), b1);
            readArray(o.getJSONArray("b2"), b2);
            JSONArray savedRecurrent = o.optJSONArray("recurrent");
            if (savedRecurrent != null) {
                readMatrix(savedRecurrent, recurrent);
            } else {
                // Legacy checkpoints keep their old feed-forward policy until recurrent
                // weights are learned during fine-tuning.
                for (double[] row : recurrent) java.util.Arrays.fill(row, 0.0);
            }
        }

        static JSONArray array(double[] a) throws Exception {
            JSONArray j = new JSONArray();
            for (double v : a) j.put(v);
            return j;
        }
        static JSONArray matrix(double[][] a) throws Exception {
            JSONArray j = new JSONArray();
            for (double[] r : a) j.put(array(r));
            return j;
        }
        static void readArray(JSONArray j, double[] a) throws Exception {
            if (j.length() != a.length) throw new Exception("维度不匹配");
            for (int i = 0; i < a.length; i++) a[i] = j.getDouble(i);
        }
        static void readMatrix(JSONArray j, double[][] a) throws Exception {
            if (j.length() != a.length) throw new Exception("维度不匹配");
            for (int i = 0; i < a.length; i++) readArray(j.getJSONArray(i), a[i]);
        }
    }
    private static final class Forward {
        final double[] q, h, pre;
        final double[][] thoughtH, thoughtPre;
        Forward(double[] q, double[] h, double[] pre, double[][] thoughtH, double[][] thoughtPre) {
            this.q = q; this.h = h; this.pre = pre;
            this.thoughtH = thoughtH; this.thoughtPre = thoughtPre;
        }
    }
}
