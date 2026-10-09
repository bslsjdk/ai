package bslsjdk.ornithnpu;

import android.app.Activity;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * A bounded, self-contained neural Q-learning grid-world experiment.
 * It deliberately uses a tiny MLP and no external Python runtime.
 */
public final class GridWorldLabActivity extends Activity {
    private static final int SIZE = 5;
    private static final int[] DX = {0, 1, 0, -1};
    private static final int[] DY = {-1, 0, 1, 0};
    private static final String[] ACTIONS = {"上", "右", "下", "左"};
    private static final Set<Integer> WALLS = new HashSet<>();
    private static final int GOAL = 24;
    static {
        int[] w = {1, 6, 8, 13, 16, 17};
        for (int p : w) WALLS.add(p);
    }

    private final Handler main = new Handler(Looper.getMainLooper());
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Random rng = new Random(20261009L);
    private final QNet net = new QNet(20261009L);
    private final List<Integer> path = new ArrayList<>();
    private TextView status, metrics, activationText, qText, log;
    private Board board;
    private volatile boolean training;
    private boolean watching;
    private int player = 0;
    private int moves = 0;
    private double episodeReward = 0;
    private int episodesDone = 0;
    private int successes = 0;
    private int evalEpisodes = 0;
    private int evalSuccesses = 0;
    private double[] lastHidden = new double[16];
    private double[] lastQ = new double[4];
    private final Runnable watchTick = new Runnable() {
        @Override public void run() {
            if (!watching || isFinishing()) return;
            if (player == GOAL || moves >= 40) {
                watching = false;
                status.setText(player == GOAL ? "观察结束：到达目标。" : "观察结束：步数上限，未到达目标。");
                refreshReadout();
                return;
            }
            stepGame();
            main.postDelayed(this, 420);
        }
    };

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        loadCheckpoint();
        buildUi();
        resetEpisode();
        status.setText("准备就绪。先训练，再点“观看策略”观察训练后的网络。");
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(14), dp(12), dp(14), dp(24));
        root.setBackgroundColor(0xFFF3F5F8);
        scroll.addView(root);

        TextView title = text("迷宫强化学习实验", 23, true);
        root.addView(title);
        root.addView(text("独立实验网络：8 输入 → 16 个 ReLU 神经元 → 4 个动作 Q 值。它与数值映射工作区分开保存，不会覆盖你的主网络。", 13, false));
        board = new Board();
        LinearLayout.LayoutParams bp = new LinearLayout.LayoutParams(-1, dp(300));
        bp.topMargin = dp(10);
        root.addView(board, bp);

        LinearLayout controls = new LinearLayout(this);
        controls.setOrientation(LinearLayout.VERTICAL);
        root.addView(controls);
        addButton(controls, "训练 1,000 局", () -> startTraining(1000));
        addButton(controls, "训练 5,000 局", () -> startTraining(5000));
        addButton(controls, "观看策略（自动走）", this::watchPolicy);
        addButton(controls, "单步执行", () -> { watching = false; stepGame(); });
        addButton(controls, "重置当前局", () -> { watching = false; resetEpisode(); });
        addButton(controls, "保存网络与报告", this::saveCheckpoint);

        status = text("状态", 13, true);
        metrics = text("", 12, false);
        activationText = text("", 12, false);
        qText = text("", 12, false);
        log = text("日志：实验尚未开始。", 11, false);
        root.addView(status, spaced());
        root.addView(metrics, spaced());
        root.addView(text("隐藏神经元激活（16 个）", 15, true), spaced());
        root.addView(activationText, spaced());
        root.addView(text("动作 Q 值（数值越高越倾向选择）", 15, true), spaced());
        root.addView(qText, spaced());
        root.addView(text("实验日志", 15, true), spaced());
        root.addView(log, spaced());
        setContentView(scroll);
        refreshReadout();
    }

    private void startTraining(int count) {
        if (training) return;
        watching = false;
        training = true;
        status.setText("正在训练 " + count + " 局。可以等待完成；训练在后台线程运行。");
        log.setText("日志：开始 Q-learning，探索率逐步下降。");
        worker.execute(() -> {
            int localWins = 0;
            double lastReward = 0;
            for (int ep = 1; ep <= count && training; ep++) {
                int start = randomStart();
                int pos = start;
                double rewardSum = 0;
                for (int t = 0; t < 45; t++) {
                    double[] s = observe(pos);
                    double epsilon = Math.max(0.04, 1.0 - 0.96 * ep / (double) count);
                    int action = net.choose(s, epsilon, rng);
                    Transition tr = transition(pos, action);
                    double[] next = observe(tr.next);
                    double[] nextQ = net.forward(next).q;
                    double target = tr.done ? tr.reward : tr.reward + 0.92 * max(nextQ);
                    net.update(s, action, target);
                    rewardSum += tr.reward;
                    pos = tr.next;
                    if (tr.done) { localWins++; break; }
                }
                lastReward = rewardSum;
                final int finished = ep;
                final int wins = localWins;
                final double reward = lastReward;
                if (ep % 25 == 0 || ep == count) {
                    main.post(() -> {
                        episodesDone += 25;
                        status.setText("训练中：" + finished + "/" + count + " 局");
                        log.setText(String.format(Locale.US,
                                "最近一局奖励 %.3f\n本批到达目标 %d/%d\n探索率正在下降；每 25 局刷新一次。", reward, wins, finished));
                        resetEpisode();
                        refreshReadout();
                    });
                }
            }
            training = false;
            main.post(() -> {
                evaluatePolicy();
                saveCheckpoint();
                status.setText("训练结束。可以观看策略，检查它是否真的学会寻路。");
                log.setText(String.format(Locale.US, "训练完成：累计训练局数约 %d；固定起点评估成功率 %.1f%%。", episodesDone, evalSuccesses * 100.0 / Math.max(1, evalEpisodes)));
                refreshReadout();
            });
        });
    }

    private void evaluatePolicy() {
        evalEpisodes = 0;
        evalSuccesses = 0;
        for (int start = 0; start < SIZE * SIZE; start++) {
            if (WALLS.contains(start) || start == GOAL) continue;
            evalEpisodes++;
            int p = start;
            Set<Integer> seen = new HashSet<>();
            for (int i = 0; i < 40; i++) {
                int a = argmax(net.forward(observe(p)).q);
                Transition tr = transition(p, a);
                p = tr.next;
                if (tr.done) { evalSuccesses++; break; }
                int marker = p * 4 + a;
                if (!seen.add(marker)) break;
            }
        }
    }

    private void watchPolicy() {
        if (training) { toast("训练还没结束，先等它完成。"); return; }
        watching = false;
        resetEpisode();
        watching = true;
        main.post(watchTick);
    }

    private void stepGame() {
        if (training) return;
        Forward f = net.forward(observe(player));
        lastHidden = f.h;
        lastQ = f.q;
        int action = argmax(f.q);
        Transition tr = transition(player, action);
        player = tr.next;
        moves++;
        episodeReward += tr.reward;
        path.add(player);
        board.invalidate();
        refreshReadout();
        if (tr.done) {
            status.setText("成功：网络到达终点！步数 " + moves + "，本局奖励 " + fmt(episodeReward));
            watching = false;
        } else if (moves >= 40) {
            status.setText("本局结束：达到 40 步上限，未成功。");
            watching = false;
        } else {
            status.setText("当前动作：" + ACTIONS[action] + "，位置 (" + (player % SIZE) + "," + (player / SIZE) + ")");
        }
    }

    private void resetEpisode() {
        player = randomStart();
        moves = 0;
        episodeReward = 0;
        path.clear();
        path.add(player);
        board.invalidate();
        Forward f = net.forward(observe(player));
        lastHidden = f.h;
        lastQ = f.q;
        refreshReadout();
    }

    private int randomStart() {
        int p;
        do { p = rng.nextInt(SIZE * SIZE); } while (WALLS.contains(p) || p == GOAL);
        return p;
    }

    private void refreshReadout() {
        if (board != null) board.invalidate();
        if (metrics != null) metrics.setText(String.format(Locale.US,
                "训练局数：%d   固定起点评估：%d/%d (%.1f%%)\n当前步数：%d/40   本局奖励：%.3f",
                episodesDone, evalSuccesses, evalEpisodes,
                evalSuccesses * 100.0 / Math.max(1, evalEpisodes), moves, episodeReward));
        if (activationText != null) {
            StringBuilder b = new StringBuilder();
            for (int i = 0; i < lastHidden.length; i++) {
                b.append(String.format(Locale.US, "H%02d  % .3f%s", i + 1, lastHidden[i], (i % 2 == 1 ? "\n" : "     ")));
            }
            activationText.setText(b.toString());
        }
        if (qText != null) {
            qText.setText(String.format(Locale.US, "↑ 上：% .4f\n→ 右：% .4f\n↓ 下：% .4f\n← 左：% .4f",
                    lastQ[0], lastQ[1], lastQ[2], lastQ[3]));
        }
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
            report.put("format", "aimeng-android-gridworld-report/v1");
            report.put("episodesTrained", episodesDone);
            report.put("evaluationEpisodes", evalEpisodes);
            report.put("evaluationSuccesses", evalSuccesses);
            report.put("successRate", evalSuccesses / (double) Math.max(1, evalEpisodes));
            report.put("note", "Real on-device Q-learning run; this report is separate from the Python benchmark.");
            write(new File(dir, "training_report.json"), report.toString(2));
            if (status != null) status.setText("已保存到应用内部 gridworld-lab 目录。");
        } catch (Exception e) { toast("保存失败：" + e.getMessage()); }
    }

    private void loadCheckpoint() {
        try {
            File f = new File(new File(getFilesDir(), "gridworld-lab"), "q_network.json");
            if (!f.isFile()) return;
            byte[] bytes = new byte[(int) Math.min(f.length(), 256 * 1024)];
            try (java.io.FileInputStream in = new java.io.FileInputStream(f)) {
                int n = in.read(bytes);
                if (n > 0) net.load(new JSONObject(new String(bytes, 0, n, StandardCharsets.UTF_8)));
            }
        } catch (Exception ignored) { }
    }

    private static void write(File f, String s) throws Exception {
        try (FileOutputStream out = new FileOutputStream(f)) { out.write(s.getBytes(StandardCharsets.UTF_8)); }
    }

    private double[] observe(int pos) {
        int x = pos % SIZE, y = pos / SIZE;
        double[] s = new double[8];
        s[0] = x / 4.0; s[1] = y / 4.0; s[2] = 1.0; s[3] = 1.0;
        for (int a = 0; a < 4; a++) {
            int nx = x + DX[a], ny = y + DY[a];
            int np = ny * SIZE + nx;
            s[4 + a] = nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE || WALLS.contains(np) ? 1.0 : 0.0;
        }
        return s;
    }

    private Transition transition(int pos, int action) {
        int x = pos % SIZE + DX[action], y = pos / SIZE + DY[action];
        if (x < 0 || x >= SIZE || y < 0 || y >= SIZE || WALLS.contains(y * SIZE + x))
            return new Transition(pos, -0.12, false);
        int next = y * SIZE + x;
        if (next == GOAL) return new Transition(next, 1.0, true);
        return new Transition(next, -0.025, false);
    }

    private static double max(double[] a) { double m = a[0]; for (double v : a) m = Math.max(m, v); return m; }
    private static int argmax(double[] a) { int best = 0; for (int i = 1; i < a.length; i++) if (a[i] > a[best]) best = i; return best; }
    private static String fmt(double v) { return String.format(Locale.US, "%.3f", v); }
    private int dp(float n) { return (int) (n * getResources().getDisplayMetrics().density + 0.5f); }
    private TextView text(String s, int size, boolean bold) {
        TextView t = new TextView(this); t.setText(s); t.setTextSize(size);
        t.setTextColor(0xFF182230); if (bold) t.setTypeface(null, 1); return t;
    }
    private LinearLayout.LayoutParams spaced() { LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, -2); p.topMargin = dp(8); return p; }
    private void addButton(LinearLayout parent, String name, Runnable action) {
        Button b = new Button(this); b.setText(name); b.setAllCaps(false);
        b.setOnClickListener(v -> action.run());
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, dp(46)); p.topMargin = dp(5); parent.addView(b, p);
    }
    private void toast(String s) { Toast.makeText(this, s, Toast.LENGTH_SHORT).show(); }

    @Override protected void onDestroy() {
        watching = false; training = false; main.removeCallbacks(watchTick); worker.shutdownNow(); super.onDestroy();
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
            float cell = Math.min(getWidth() / 5f, getHeight() / 5f);
            float ox = (getWidth() - cell * 5) / 2f, oy = (getHeight() - cell * 5) / 2f;
            for (int y = 0; y < 5; y++) for (int x = 0; x < 5; x++) {
                int id = y * 5 + x;
                float l = ox + x * cell, t = oy + y * cell;
                p.setColor(WALLS.contains(id) ? 0xFF344054 : 0xFFF9FAFB);
                c.drawRoundRect(new RectF(l+2,t+2,l+cell-2,t+cell-2), 7,7,p);
                if (id == GOAL) { p.setColor(0xFF12B76A); c.drawCircle(l+cell/2,t+cell/2,cell*.28f,p); }
                if (path.contains(id) && id != player && id != GOAL && !WALLS.contains(id)) {
                    p.setColor(0xFFB2DDFF); c.drawCircle(l+cell/2,t+cell/2,cell*.12f,p);
                }
                if (id == player) { p.setColor(0xFF1570EF); c.drawCircle(l+cell/2,t+cell/2,cell*.30f,p); }
            }
            p.setColor(0xFF475467); p.setTextSize(dp(11));
            c.drawText("蓝色：神经网络控制的小人    绿色：终点", dp(6), getHeight()-dp(7), p);
        }
    }

    private static final class QNet {
        final double[][] w1 = new double[16][8], w2 = new double[4][16];
        final double[] b1 = new double[16], b2 = new double[4];
        final Random random;
        QNet(long seed) { random = new Random(seed); for (int j=0;j<16;j++) for(int i=0;i<8;i++) w1[j][i]=(random.nextDouble()*0.5-0.25); for(int a=0;a<4;a++) for(int j=0;j<16;j++) w2[a][j]=(random.nextDouble()*0.5-0.25); }
        Forward forward(double[] x) {
            double[] h = new double[16], pre = new double[16], q = new double[4];
            for(int j=0;j<16;j++){ double v=b1[j]; for(int i=0;i<8;i++)v+=w1[j][i]*x[i]; pre[j]=v; h[j]=Math.max(0,v); }
            for(int a=0;a<4;a++){q[a]=b2[a];for(int j=0;j<16;j++)q[a]+=w2[a][j]*h[j];}
            return new Forward(q,h,pre);
        }
        int choose(double[] s,double eps,Random r){ if(r.nextDouble()<eps)return r.nextInt(4);return argmax(forward(s).q); }
        void update(double[] x,int action,double target){
            Forward f=forward(x); double grad=Math.max(-1,Math.min(1,f.q[action]-target)); double[] old=w2[action].clone();
            for(int j=0;j<16;j++)w2[action][j]-=0.003*grad*f.h[j]; b2[action]-=0.003*grad;
            for(int j=0;j<16;j++)if(f.pre[j]>0){double back=grad*old[j];for(int i=0;i<8;i++)w1[j][i]-=0.003*back*x[i];b1[j]-=0.003*back;}
        }
        JSONObject toJson() throws Exception {
            JSONObject o=new JSONObject();o.put("format","aimeng-android-gridworld-qnet/v1");o.put("inputSize",8);o.put("hiddenSize",16);o.put("outputSize",4);
            o.put("w1",matrix(w1));o.put("w2",matrix(w2));o.put("b1",array(b1));o.put("b2",array(b2));return o;
        }
        void load(JSONObject o) throws Exception { if(o.optInt("inputSize")!=8||o.optInt("hiddenSize")!=16||o.optInt("outputSize")!=4)return; readMatrix(o.getJSONArray("w1"),w1);readMatrix(o.getJSONArray("w2"),w2);readArray(o.getJSONArray("b1"),b1);readArray(o.getJSONArray("b2"),b2); }
        static JSONArray array(double[] a)throws Exception{JSONArray j=new JSONArray();for(double v:a)j.put(v);return j;}
        static JSONArray matrix(double[][] a)throws Exception{JSONArray j=new JSONArray();for(double[] r:a)j.put(array(r));return j;}
        static void readArray(JSONArray j,double[] a)throws Exception{if(j.length()!=a.length)throw new Exception("维度不匹配");for(int i=0;i<a.length;i++)a[i]=j.getDouble(i);}
        static void readMatrix(JSONArray j,double[][] a)throws Exception{if(j.length()!=a.length)throw new Exception("维度不匹配");for(int i=0;i<a.length;i++)readArray(j.getJSONArray(i),a[i]);}
    }
    private static final class Forward {
        final double[] q,h,pre;
        Forward(double[] q,double[] h,double[] p){this.q=q;this.h=h;pre=p;}
    }
}
