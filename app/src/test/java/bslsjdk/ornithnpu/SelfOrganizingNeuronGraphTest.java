package bslsjdk.ornithnpu;

import org.junit.Test;

import static org.junit.Assert.*;

public class SelfOrganizingNeuronGraphTest {
    @Test
    public void directedConnectionPropagatesInputToReadout() {
        SelfOrganizingNeuronGraph graph = new SelfOrganizingNeuronGraph(8, 16, 0.9, 7L);
        int input = graph.addNeuron();
        int output = graph.addNeuron();
        assertTrue(graph.markInputPort(input));
        assertTrue(graph.markOutputPort(output));
        assertTrue(graph.addConnection(input, output, 1.0));

        double[] result = graph.step(new double[]{1.0}, 1);
        assertEquals(1, result.length);
        assertEquals(Math.tanh(1.0), result[0], 1e-9);
        assertEquals(1L, graph.getTicks());
    }

    @Test
    public void inputsFollowPortRegistrationOrder() {
        SelfOrganizingNeuronGraph graph = new SelfOrganizingNeuronGraph(8, 16, 0.9, 8L);
        int a = graph.addNeuron();
        int b = graph.addNeuron();
        int out = graph.addNeuron();
        assertTrue(graph.markInputPort(b));
        assertTrue(graph.markInputPort(a));
        assertTrue(graph.markOutputPort(out));
        assertTrue(graph.addConnection(a, out, 1.0));

        double[] result = graph.step(new double[]{0.25, 0.75}, 1);
        assertEquals(Math.tanh(0.75), result[0], 1e-9);
    }

    @Test
    public void topologyBudgetAndReactivationAreEnforced() {
        SelfOrganizingNeuronGraph graph = new SelfOrganizingNeuronGraph(4, 1, 0.9, 9L);
        int a = graph.addNeuron(), b = graph.addNeuron(), c = graph.addNeuron();
        assertTrue(graph.addConnection(a, b, 0.5));
        assertFalse(graph.addConnection(b, c, 0.5));
        assertTrue(graph.removeConnection(a, b));
        assertTrue(graph.growConnection(a, b, -0.25));
        assertEquals(1, graph.getActiveEdgeCount());
        assertEquals(1, graph.getEdgeSlotCount());
    }

    @Test
    public void seededFactoryCreatesBoundedGraphWithPorts() {
        SelfOrganizingNeuronGraph graph =
                SelfOrganizingNeuronGraph.createRandomGraph(4, 2, 16, 40, 123L);
        assertEquals(16, graph.getNeuronCount());
        assertEquals(4, graph.getInputCount());
        assertEquals(2, graph.getOutputCount());
        assertTrue(graph.getActiveEdgeCount() > 0);
        assertTrue(graph.getActiveEdgeCount() <= 40);
        assertTrue(graph.estimatedStorageBytes() < 1024 * 1024);
    }

    @Test
    public void activeNeuronBudgetCapsPerTickActivation() {
        SelfOrganizingNeuronGraph graph = new SelfOrganizingNeuronGraph(8, 16, 0.9, 11L);
        int input = graph.addNeuron();
        int output = graph.addNeuron();
        int extra = graph.addNeuron();
        assertTrue(graph.markInputPort(input));
        assertTrue(graph.markOutputPort(output));
        assertTrue(graph.addConnection(input, output, 1.0));
        assertTrue(graph.addConnection(input, extra, 1.0));
        graph.setActiveNeuronBudget(2);

        double[] result = graph.step(new double[]{1.0}, 1);
        assertEquals(Math.tanh(1.0), result[0], 1e-9);
        assertTrue(graph.getLastActiveNeuronCount() <= 2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsWrongInputShape() {
        SelfOrganizingNeuronGraph graph = new SelfOrganizingNeuronGraph(4, 4, 0.9, 10L);
        graph.markInputPort(graph.addNeuron());
        graph.step(new double[0], 1);
    }
}
