package bslsjdk.ornithnpu;

import org.junit.Test;
import static org.junit.Assert.*;

public class SuperNeuronNetworkV2Test {
    @Test
    public void sparseMessagesTravelAcrossSteps() {
        SuperNeuronNetworkV2 net = new SuperNeuronNetworkV2(8, 16, 3, 7L);
        net.connect(0, 1, 0, 1.0);
        assertEquals(1, net.step(new int[]{0}, new double[]{1.0}, 1, 2));
        assertTrue(Math.abs(net.getActivation(0)) > 0.0);
        assertEquals(1, net.step(new int[0], new double[0], 0, 2));
        assertTrue(Math.abs(net.getActivation(1)) > 0.0);
    }

    @Test
    public void thousandNodesUseBoundedActivationBudget() {
        SuperNeuronNetworkV2 net = new SuperNeuronNetworkV2(1000, 16000, 4, 19L);
        int[] ids = new int[1000];
        double[] values = new double[1000];
        for (int i = 0; i < ids.length; i++) { ids[i] = i; values[i] = i + 1; }
        assertEquals(12, net.step(ids, values, ids.length, 12));
        assertEquals(1000, net.getNeuronCount());
        assertTrue(net.estimatedStorageBytes() < 500000L);
    }
}
