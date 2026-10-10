package bslsjdk.ornithnpu;

import org.json.JSONObject;
import org.junit.Test;

import static org.junit.Assert.*;

public class SuperNeuronV2Test {
    @Test
    public void multipleInputsProduceNormalizedMultipleClassOutputs() {
        SuperNeuronV2 neuron = new SuperNeuronV2(3, 8, 5, 123L);
        double[] probabilities = neuron.forward(new double[]{0.2, -0.4, 0.8});
        assertEquals(5, probabilities.length);
        double sum = 0.0;
        for (double p : probabilities) {
            assertTrue(Double.isFinite(p));
            assertTrue(p >= 0.0 && p <= 1.0);
            sum += p;
        }
        assertEquals(1.0, sum, 1e-9);
        assertEquals(1L, neuron.getForwardSteps());
    }

    @Test
    public void supervisedTrainingReducesLossOnTinyNextTokenTask() {
        SuperNeuronV2 neuron = new SuperNeuronV2(1, 12, 2, 777L);
        double firstLoss = 0.0;
        for (int i = 0; i < 500; i++) {
            neuron.resetState();
            double[] input = new double[]{-1.0};
            neuron.forward(input);
            double loss = neuron.trainClass(0, 0.03);
            if (i == 0) firstLoss = loss;
        }
        for (int i = 0; i < 500; i++) {
            neuron.resetState();
            neuron.forward(new double[]{1.0});
            neuron.trainClass(1, 0.03);
        }

        neuron.resetState();
        double[] negative = neuron.forward(new double[]{-1.0});
        neuron.resetState();
        double[] positive = neuron.forward(new double[]{1.0});
        assertTrue("training should lower loss for class 0",
                -Math.log(Math.max(1e-12, negative[0])) < firstLoss);
        assertTrue("positive and negative contexts should produce different predictions",
                positive[1] > positive[0]);
        assertTrue(neuron.getTrainSteps() >= 1000);
    }

    @Test
    public void checkpointRoundTripPreservesPredictionsAndIdentityOfShape() throws Exception {
        SuperNeuronV2 original = new SuperNeuronV2(2, 7, 4, 45L);
        original.forward(new double[]{0.25, -0.75});
        JSONObject checkpoint = original.toJson();
        SuperNeuronV2 restored = SuperNeuronV2.fromJson(
                new JSONObject(checkpoint.toString()), 999L);

        assertEquals(2, restored.getInputCount());
        assertEquals(7, restored.getHiddenSize());
        assertEquals(4, restored.getOutputCount());
        assertEquals(original.getForwardSteps(), restored.getForwardSteps());
        assertArrayEquals(original.forward(new double[]{0.1, 0.2}),
                restored.forward(new double[]{0.1, 0.2}), 1e-12);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNonFiniteInputs() {
        SuperNeuronV2 neuron = new SuperNeuronV2(2, 4, 3, 1L);
        neuron.forward(new double[]{0.0, Double.NaN});
    }

    @Test
    public void compactPrototypeReportsBoundedStorage() {
        SuperNeuronV2 neuron = new SuperNeuronV2(8, 16, 64, 2L);
        assertTrue(neuron.estimatedStorageBytes() < 64 * 1024);
    }
}
