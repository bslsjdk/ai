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
    public void sharedOutputCoreLearnsFromReward() {
        SuperNeuronNetworkV2 net = new SuperNeuronNetworkV2(4, 8, 2, 11L);
        net.connect(0, 1, 0, 1.0);
        net.step(new int[]{0}, new double[]{1.0}, 1, 1);
        double before = net.getChannelOutput(0, 0);
        net.applyReward(1.0, 0.02);
        double after = net.getChannelOutput(0, 0);
        assertTrue("shared channel parameters should update under reward", Math.abs(after - before) > 1e-8);
    }

    @Test
    public void repeatedOutgoingChannelsReuseCacheWithoutChangingSparsePropagation() {
        SuperNeuronNetworkV2 net = new SuperNeuronNetworkV2(6, 12, 3, 23L);
        net.connect(0, 1, 1, 0.7);
        net.connect(0, 2, 1, -0.4);
        net.connect(0, 3, 2, 0.9);
        assertEquals(1, net.step(new int[]{0}, new double[]{1.0}, 1, 1));
        assertEquals(3, net.step(new int[0], new double[0], 0, 4));
        for (int node = 1; node <= 3; node++) {
            assertTrue(Double.isFinite(net.getActivation(node)));
            assertTrue(Double.isFinite(net.getState(node)));
            for (int channel = 0; channel < net.getChannelCount(); channel++)
                assertTrue(Double.isFinite(net.getChannelOutput(node, channel)));
        }
    }

    @Test
    public void thousandNodesUseBoundedActivationBudget() {
        SuperNeuronNetworkV2 net = new SuperNeuronNetworkV2(1000, 16000, 4, 19L);
        int[] ids = new int[1000];
        double[] values = new double[1000];
        for (int i = 0; i < ids.length; i++) { ids[i] = i; values[i] = i + 1; }
        assertEquals(12, net.step(ids, values, ids.length, 12));
        assertEquals(1000, net.getNeuronCount());
        assertTrue(net.estimatedStorageBytes() < 600000L);
    }
}
