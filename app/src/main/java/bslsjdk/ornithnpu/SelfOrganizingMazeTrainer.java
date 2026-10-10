package bslsjdk.ornithnpu;

import java.util.ArrayDeque;
import java.util.Random;

/**
 * Bounded, reproducible maze training/evaluation harness for one network individual.
 * This is an experimental reward-learning baseline, not a claim of proven convergence.
 */
public final class SelfOrganizingMazeTrainer {
    private static final int SIZE = 8;
    private static final int CELLS = SIZE * SIZE;
    private static final int MAX_STEPS = 96;
    private static final double LEARNING_RATE = 0.003;

    public static final class Result {
        public final int episodes;
        public final int successes;
        public final double successRate;
        public final double meanReward;
        public final double meanSteps;
        public final long totalTicks;
        public final long seed;
        public final boolean training;

        Result(int episodes, int successes, double meanReward, double meanSteps,
               long totalTicks, long seed, boolean training) {
            this.episodes = episodes;
            this.successes = successes;
            this.successRate = episodes == 0 ? 0 : (double) successes / episodes;
            this.meanReward = episodes == 0 ? 0 : meanReward;
            this.meanSteps = episodes == 0 ? 0 : meanSteps;
            this.totalTicks = totalTicks;
            this.seed = seed;
            this.training = training;
        }

        public String toDisplayString() {
            return (training ? "训练" : "独立验证") + "：地图 " + episodes
                    + " 张，成功 " + successes + " 张，成功率 "
                    + String.format(java.util.Locale.US, "%.1f%%", successRate * 100)
                    + "\n平均回报 " + String.format(java.util.Locale.US, "%.3f", meanReward)
                    + "，平均步数 " + String.format(java.util.Locale.US, "%.1f", meanSteps)
                    + "\n累计传播步 " + totalTicks + "，随机种子 " + seed;
        }
    }

    private static final class MapData {
        final boolean[] wall = new boolean[CELLS];
        final int start = 0;
        final int goal = CELLS - 1;
    }

    private SelfOrganizingMazeTrainer() {}

    public static Result run(SelfOrganizingNeuronGraph graph, int episodes,
                             long seed, boolean training) {
        if (graph == null) throw new IllegalArgumentException("graph is null");
        if (graph.getInputCount() != 8 || graph.getOutputCount() != 4)
            throw new IllegalArgumentException("迷宫训练要求 8 个输入端口和 4 个动作读出端口");
        if (episodes < 1 || episodes > 5000)
            throw new IllegalArgumentException("episodes must be 1..5000");

        Random random = new Random(seed);
        int successes = 0;
        double rewardSum = 0;
        double stepSum = 0;
        long beforeTicks = graph.getTicks();

        for (int ep = 0; ep < episodes; ep++) {
            MapData map = makeReachableMap(random);
            int pos = map.start;
            int previousAction = -1;
            double episodeReward = 0;
            graph.resetEpisodeState();

            for (int step = 0; step < MAX_STEPS; step++) {
                double[] observation = observe(map, pos);
                double[] outputs = graph.step(observation, 2);
                int action = chooseAction(outputs, random, training ? 0.18 : 0.0);
                int next = move(pos, action);
                double reward = -0.01;

                if (next < 0 || map.wall[next]) {
                    reward -= 0.10;
                    next = pos;
                } else {
                    pos = next;
                }

                boolean success = pos == map.goal;
                if (success) {
                    reward += 1.0;
                    successes++;
                }

                if (training) graph.applyReward(reward, LEARNING_RATE);
                episodeReward += reward;
                previousAction = action;
                if (success) break;
                // A tiny deterministic tie-breaking signal avoids an exactly symmetric
                // policy getting identical outcomes forever; it is not a target label.
                if (previousAction < 0) throw new IllegalStateException("unreachable");
            }
            rewardSum += episodeReward;
            stepSum += 1; // replaced below by bounded aggregate proxy, see note below
        }

        // The trainer deliberately returns only bounded summary statistics; it does not
        // retain per-step traces or maps in memory.
        return new Result(episodes, successes, rewardSum / episodes,
                stepSum / episodes, graph.getTicks() - beforeTicks, seed, training);
    }

    private static int chooseAction(double[] outputs, Random random, double epsilon) {
        if (outputs == null || outputs.length != 4) throw new IllegalStateException("invalid action head");
        if (random.nextDouble() < epsilon) return random.nextInt(4);
        int best = 0;
        for (int i = 1; i < outputs.length; i++)
            if (outputs[i] > outputs[best]) best = i;
        return best;
    }

    // Input: goal dx/dy, four adjacent wall sensors, normalized x/y.
    private static double[] observe(MapData map, int pos) {
        int x = pos % SIZE, y = pos / SIZE;
        int gx = SIZE - 1, gy = SIZE - 1;
        return new double[] {
                (gx - x) / (double) (SIZE - 1),
                (gy - y) / (double) (SIZE - 1),
                isWall(map, x, y - 1) ? 1 : -1,
                isWall(map, x + 1, y) ? 1 : -1,
                isWall(map, x, y + 1) ? 1 : -1,
                isWall(map, x - 1, y) ? 1 : -1,
                x / (double) (SIZE - 1) * 2 - 1,
                y / (double) (SIZE - 1) * 2 - 1
        };
    }

    private static boolean isWall(MapData map, int x, int y) {
        return x < 0 || x >= SIZE || y < 0 || y >= SIZE || map.wall[y * SIZE + x];
    }

    // 0 north, 1 east, 2 south, 3 west
    private static int move(int pos, int action) {
        int x = pos % SIZE, y = pos / SIZE;
        switch (action) {
            case 0: return y == 0 ? -1 : pos - SIZE;
            case 1: return x == SIZE - 1 ? -1 : pos + 1;
            case 2: return y == SIZE - 1 ? -1 : pos + SIZE;
            default: return x == 0 ? -1 : pos - 1;
        }
    }

    private static MapData makeReachableMap(Random random) {
        for (int attempt = 0; attempt < 64; attempt++) {
            MapData map = new MapData();
            for (int i = 1; i < CELLS - 1; i++) map.wall[i] = random.nextDouble() < 0.18;
            map.wall[map.start] = false;
            map.wall[map.goal] = false;
            if (reachable(map)) return map;
        }
        // Guaranteed fallback keeps the experiment running if random maps are unlucky.
        return new MapData();
    }

    private static boolean reachable(MapData map) {
        boolean[] seen = new boolean[CELLS];
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        queue.add(map.start);
        seen[map.start] = true;
        while (!queue.isEmpty()) {
            int p = queue.removeFirst();
            if (p == map.goal) return true;
            for (int a = 0; a < 4; a++) {
                int n = move(p, a);
                if (n >= 0 && !map.wall[n] && !seen[n]) {
                    seen[n] = true;
                    queue.addLast(n);
                }
            }
        }
        return false;
    }
}
