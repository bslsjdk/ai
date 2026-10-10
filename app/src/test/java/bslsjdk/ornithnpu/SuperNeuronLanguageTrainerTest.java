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
        assertTrue(trainer.estimatedStorageBytes() < 64L * 1024L * 1024L);
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
    public void softTargetDistillationAcceptsNormalizedTeacherDistribution() {
        SuperNeuronLanguageTrainer trainer = new SuperNeuronLanguageTrainer(4, 8, 12, 77L);
        double[] teacher = new double[]{0.05, 0.70, 0.20, 0.05};
        double loss = trainer.trainSoftTarget(new int[]{0, 1, 2}, teacher, 0.01);
        assertTrue(Double.isFinite(loss));
        assertTrue(loss > 0.0);
        assertEquals(1L, trainer.getTrainedTokenTargets());
        double[] predicted = trainer.predictDistribution(new int[]{0, 1, 2});
        assertEquals(4, predicted.length);
        double sum = 0.0;
        for (double p : predicted) {
            assertTrue(Double.isFinite(p));
            assertTrue(p >= 0.0 && p <= 1.0);
            sum += p;
        }
        assertEquals(1.0, sum, 1.0e-6);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsUnnormalizedTeacherDistribution() {
        SuperNeuronLanguageTrainer trainer = new SuperNeuronLanguageTrainer(4, 8, 12, 78L);
        trainer.trainSoftTarget(new int[]{0, 1}, new double[]{0.1, 0.1, 0.1, 0.1}, 0.01);
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
