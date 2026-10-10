package bslsjdk.ornithnpu;

import org.junit.Test;

import static org.junit.Assert.*;

public class SuperNeuronLanguageTrainerTest {
    @Test
    public void outputDimensionAlwaysMatchesVocabulary() {
        SuperNeuronLanguageTrainer trainer = new SuperNeuronLanguageTrainer(11, 8, 12, 42L);
        assertEquals(11, trainer.getVocabularySize());
        int predicted = trainer.predictNextToken(3);
        assertTrue(predicted >= 0 && predicted < 11);
        assertTrue(trainer.estimatedStorageBytes() < 16L * 1024L * 1024L);
    }

    @Test
    public void supervisedSequenceTrainingReducesTrainingLoss() {
        SuperNeuronLanguageTrainer trainer = new SuperNeuronLanguageTrainer(3, 8, 12, 123L);
        int[] sequence = new int[61];
        for (int i = 0; i < sequence.length; i++) sequence[i] = (i % 2 == 0) ? 0 : 1;

        double before = trainer.evaluateSequence(sequence);
        trainer.trainSequence(sequence, 100, 0.02);
        double after = trainer.evaluateSequence(sequence);

        assertTrue("expected next-token loss to decrease; before=" + before + ", after=" + after,
                after < before);
        assertEquals(6000L, trainer.getTrainedTokenTargets());
        assertTrue(trainer.getParameterCount() > 0L);
    }

    @Test
    public void exactStepTrainingAndCheckpointRoundTrip() throws Exception {
        SuperNeuronLanguageTrainer trainer = new SuperNeuronLanguageTrainer(5, 4, 6, 19L);
        int[] tokens = {0, 1, 2, 1, 0, 3, 2, 1};
        trainer.trainSequenceSteps(tokens, 37, 0.01, null);
        assertEquals(37L, trainer.getTrainedTokenTargets());
        org.json.JSONObject checkpoint = trainer.toJson();
        SuperNeuronLanguageTrainer restored = SuperNeuronLanguageTrainer.fromJson(
                new org.json.JSONObject(checkpoint.toString()), 77L);
        assertEquals(trainer.getVocabularySize(), restored.getVocabularySize());
        assertEquals(trainer.getParameterCount(), restored.getParameterCount());
        assertEquals(trainer.getTrainedTokenTargets(), restored.getTrainedTokenTargets());
        assertEquals(trainer.evaluateSequence(tokens), restored.evaluateSequence(tokens), 1e-10);
    }

    @Test
    public void listenerCanStopTrainingAtExactStep() {
        SuperNeuronLanguageTrainer trainer = new SuperNeuronLanguageTrainer(3, 4, 6, 29L);
        int[] tokens = {0, 1, 2, 1, 0};
        trainer.trainSequenceSteps(tokens, 100, 0.01, (step, loss) -> step < 13);
        assertEquals(13L, trainer.getTrainedTokenTargets());
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsOversizedModelBeforeAllocation() {
        new SuperNeuronLanguageTrainer(65536, 64, 256, 3L);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsTokenOutsideVocabulary() {
        SuperNeuronLanguageTrainer trainer = new SuperNeuronLanguageTrainer(5, 8, 12, 2L);
        trainer.trainSequence(new int[]{0, 5}, 1, 0.01);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsTrainingCallOverWorkBudget() {
        SuperNeuronLanguageTrainer trainer = new SuperNeuronLanguageTrainer(4, 8, 12, 2L);
        trainer.trainSequence(new int[]{0, 1, 2}, 1000001, 0.01);
    }
}
