package bslsjdk.ornithnpu;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 * Population-level optimization is evolutionary, not global backpropagation:
 * individuals learn locally; selection/crossover/mutation happen between runs.
 */
public final class SuperNeuronEvolutionPool {
    public interface ProgressListener {
        /** Return false to stop the generation safely. */
        boolean onStep(long completedSteps, double stepLoss);
    }

    private static final double MUTATION_RATE = 0.025;
    private static final double MUTATION_SIGMA = 0.08;
    private final List<SuperNeuronLanguageTrainer> population = new ArrayList<>();
    private final long seed;
    private long generation;
    private long lastTrainingSteps;
    private double bestValidationLoss = Double.POSITIVE_INFINITY;
    private SuperNeuronLanguageTrainer bestTrainer;
    private SuperNeuronLanguageTrainer currentTrainer;
    private boolean stoppedEarly;

    public SuperNeuronEvolutionPool(int populationSize, int vocabularySize,
                                    int embeddingSize, int hiddenSize, long seed) {
        if (populationSize < 2 || populationSize > 8)
            throw new IllegalArgumentException("populationSize must be 2..8");
        this.seed = seed;
        for (int i = 0; i < populationSize; i++) {
            population.add(new SuperNeuronLanguageTrainer(vocabularySize, embeddingSize,
                    hiddenSize, seed + 0x9E3779B97F4A7C15L * (i + 1L)));
        }
    }

    /**
     * Distributes an exact total step budget across individuals, ranks on a
     * held-out sequence, then retains elites and breeds the next generation.
     */
    public double trainAndEvolve(int[] trainingTokens, int[] validationTokens,
                                 long totalSteps, double learningRate,
                                 ProgressListener listener) {
        if (trainingTokens == null || trainingTokens.length < 2
                || validationTokens == null || validationTokens.length < 2)
            throw new IllegalArgumentException("training and validation sequences need at least two tokens");
        if (totalSteps < 1 || totalSteps > 100000L)
            throw new IllegalArgumentException("totalSteps must be 1..100000 for the mobile experiment");
        long base = totalSteps / population.size();
        long remainder = totalSteps % population.size();
        long completed = 0;
        double lossTotal = 0.0;
        long lossCount = 0;
        boolean stopped = false;
        stoppedEarly = false;
        for (int i = 0; i < population.size(); i++) {
            long budget = base + (i < remainder ? 1 : 0);
            if (budget == 0) continue;
            SuperNeuronLanguageTrainer individual = population.get(i);
            currentTrainer = individual;
            final long offset = completed;
            final long[] local = {0};
            double meanLoss = individual.trainSequenceSteps(trainingTokens, budget, learningRate,
                    (step, loss) -> {
                        local[0] = step;
                        boolean keepGoing = listener == null || listener.onStep(offset + step, loss);
                        return keepGoing;
                    });
            completed += local[0];
            if (local[0] > 0) {
                lossTotal += meanLoss * local[0];
                lossCount += local[0];
            }
            if (local[0] < budget) {
                stopped = true;
                break;
            }
        }
        lastTrainingSteps = completed;
        if (stopped || completed != totalSteps) {
            stoppedEarly = true;
            bestTrainer = currentTrainer;
            bestValidationLoss = Double.NaN;
            return lossCount == 0 ? 0.0 : lossTotal / lossCount;
        }

        population.sort(Comparator.comparingDouble(candidate -> candidate.evaluateSequence(validationTokens)));
        bestTrainer = population.get(0);
        bestValidationLoss = bestTrainer.evaluateSequence(validationTokens);
        if (!stopped && completed == totalSteps) evolvePopulation();
        return lossCount == 0 ? 0.0 : lossTotal / lossCount;
    }

    private void evolvePopulation() {
        int eliteCount = Math.max(1, population.size() / 2);
        List<SuperNeuronLanguageTrainer> next = new ArrayList<>(population.size());
        for (int i = 0; i < eliteCount; i++) next.add(population.get(i));
        Random random = new Random(seed + (++generation * 0x9E3779B97F4A7C15L));
        while (next.size() < population.size()) {
            SuperNeuronLanguageTrainer first = population.get(random.nextInt(eliteCount));
            SuperNeuronLanguageTrainer second = population.get(random.nextInt(eliteCount));
            next.add(first.reproduceWith(second, random.nextLong(), MUTATION_RATE, MUTATION_SIGMA));
        }
        population.clear();
        population.addAll(next);
    }

    public SuperNeuronLanguageTrainer getBestTrainer() {
        if (bestTrainer == null) throw new IllegalStateException("no evaluated generation yet");
        return bestTrainer;
    }
    public double getBestValidationLoss() { return bestValidationLoss; }
    public SuperNeuronLanguageTrainer getCurrentTrainer() { return currentTrainer; }
    public boolean wasStoppedEarly() { return stoppedEarly; }
    public long getGeneration() { return generation; }
    public long getLastTrainingSteps() { return lastTrainingSteps; }
    public int getPopulationSize() { return population.size(); }
}
