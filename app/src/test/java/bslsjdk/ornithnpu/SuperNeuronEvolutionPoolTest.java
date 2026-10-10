package bslsjdk.ornithnpu;

import org.junit.Test;
import static org.junit.Assert.*;

public class SuperNeuronEvolutionPoolTest {
    @Test public void distributesExactStepBudgetAndEvolvesWithoutPoolBackprop() {
        int[] tokens = {0, 1, 2, 1, 0, 2, 1, 0};
        SuperNeuronEvolutionPool pool = new SuperNeuronEvolutionPool(4, 3, 4, 6, 17L);
        double mean = pool.trainAndEvolve(tokens, tokens, 40, 0.01, null);
        assertEquals(40L, pool.getLastTrainingSteps());
        assertEquals(1L, pool.getGeneration());
        assertEquals(4, pool.getPopulationSize());
        assertTrue(Double.isFinite(mean));
        assertTrue(Double.isFinite(pool.getBestValidationLoss()));
        int prediction = pool.getBestTrainer().predictNextToken(1);
        assertTrue(prediction >= 0 && prediction < 3);
    }
}
