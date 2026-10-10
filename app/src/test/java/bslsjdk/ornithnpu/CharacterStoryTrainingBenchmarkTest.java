package bslsjdk.ornithnpu;

import org.junit.Test;
import java.util.Arrays;
import static org.junit.Assert.*;

/** Runs the requested 10,000-update character-level smoke benchmark in CI. */
public class CharacterStoryTrainingBenchmarkTest {
    @Test public void trainsTenThousandCharacterTargetsAndPrintsLossSamples() {
        String story = "小兔子住在森林边的小木屋里。一天早晨，它发现门口有一颗闪闪发光的种子。"
                + "小兔把种子种进土里，每天浇水，也和小鸟一起等它发芽。过了几天，嫩绿的小芽探出头来。"
                + "大风来时，小兔用树枝挡风，下雨时又把积水轻轻排开。后来种子长成一棵小树，结出了甜甜的果子。"
                + "小兔把果子分给小鸟、刺猬和路过的鹿。大家一起种下更多种子，让森林变得更加茂盛。"
                + "小兔明白，耐心照料和分享，会让小小的善意慢慢长大。";
        CharacterTokenizer tokenizer = CharacterTokenizer.fit(story);
        int[] all = tokenizer.encode(story);
        int split = Math.max(2, Math.min(all.length - 2, (int) (all.length * 0.8)));
        int[] train = Arrays.copyOfRange(all, 0, split);
        int[] validation = Arrays.copyOfRange(all, split - 1, all.length);
        SuperNeuronEvolutionPool pool = new SuperNeuronEvolutionPool(
                4, tokenizer.getVocabularySize(), 8, 16, 20261010L);

        final double[] windowLoss = {0.0};
        final long[] windowCount = {0};
        double meanTrainingLoss = pool.trainAndEvolve(train, validation, 10000L, 0.015,
                (step, loss) -> {
                    windowLoss[0] += loss;
                    windowCount[0]++;
                    if (step % 1000L == 0L) {
                        long usedHeap = Runtime.getRuntime().totalMemory()
                                - Runtime.getRuntime().freeMemory();
                        System.out.println("CHAR_STORY_BENCH step=" + step
                                + " windowMeanCrossEntropy=" + (windowLoss[0] / windowCount[0])
                                + " javaHeapUsedBytes=" + usedHeap);
                        windowLoss[0] = 0.0;
                        windowCount[0] = 0;
                    }
                    return true;
                });

        double before = pool.getInitialBestValidationLoss();
        double after = pool.getBestValidationLoss();
        int prediction = pool.getBestTrainer().predictNextToken(validation[validation.length - 1]);
        System.out.println("CHAR_STORY_BENCH_SUMMARY steps=" + pool.getLastTrainingSteps()
                + " vocab=" + tokenizer.getVocabularySize()
                + " meanTrainingLoss=" + meanTrainingLoss
                + " validationBefore=" + before + " validationAfter=" + after
                + " generation=" + pool.getGeneration());
        assertEquals(10000L, pool.getLastTrainingSteps());
        assertEquals(1L, pool.getGeneration());
        assertTrue(prediction >= 0 && prediction < tokenizer.getVocabularySize());
        assertTrue(Double.isFinite(meanTrainingLoss));
        assertTrue(Double.isFinite(before));
        assertTrue(Double.isFinite(after));
    }
}
