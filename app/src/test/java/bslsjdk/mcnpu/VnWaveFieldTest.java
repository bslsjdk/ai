package bslsjdk.ornithnpu;

import org.junit.Test;

import static org.junit.Assert.*;

public class VnWaveFieldTest {
    @Test
    public void directedWavePropagatesOnlyAlongConfiguredDirection() {
        VnWaveField field = new VnWaveField(3, 1, 4, 0.1, 0.5, 0.0);
        field.addDirectedEdge(0, 1, 1.0);
        field.step(new double[]{1.0, 0.0, 0.0}, 2);
        VnWaveField.Snapshot state = field.snapshot();
        assertTrue(state.energy[1] > 0.0);
        assertEquals(0.0, state.energy[2], 1e-12);
        assertTrue(state.energy[0] >= state.energy[1]);
    }

    @Test
    public void stateRemainsBoundedDuringLongBackgroundEvolution() {
        VnWaveField field = new VnWaveField(12, 4, 48, 0.08, 0.35, 0.2);
        for (int i = 0; i < 11; i++) field.addDirectedEdge(i, i + 1, 0.8);
        field.step(new double[]{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 12);
        for (int i = 0; i < 1000; i++) field.backgroundTick();
        VnWaveField.Snapshot state = field.snapshot();
        for (double e : state.energy) assertTrue(e >= 0.0 && e <= 1.0);
        for (double p : state.phase) assertTrue(p >= 0.0 && p < Math.PI * 2.0);
        for (double a : state.activation) assertTrue(a >= -1.0 && a <= 1.0);
        assertEquals(1012L, state.ticks);
    }

    @Test
    public void activeSourceBudgetIsEnforcedAndReported() {
        VnWaveField field = new VnWaveField(20, 3, 60, 0.1, 0.4, 0.1);
        for (int i = 0; i < 19; i++) field.addDirectedEdge(i, i + 1, 0.9);
        VnWaveField.Snapshot state = field.step(new double[20], 4);
        assertTrue(state.activeSources <= 3);
        assertEquals(3, field.getActiveBudget());
    }

    @Test
    public void snapshotsAreCopiesAndSeededRunsAreDeterministic() {
        VnWaveField a = new VnWaveField(4, 2, 8, 0.1, 0.4, 0.2);
        VnWaveField b = new VnWaveField(4, 2, 8, 0.1, 0.4, 0.2);
        a.addDirectedEdge(0, 1, 0.7);
        b.addDirectedEdge(0, 1, 0.7);
        VnWaveField.Snapshot first = a.step(new double[]{1, 0, 0, 0}, 5);
        VnWaveField.Snapshot second = b.step(new double[]{1, 0, 0, 0}, 5);
        assertArrayEquals(first.energy, second.energy, 0.0);
        assertArrayEquals(first.phase, second.phase, 0.0);
        first.energy[0] = 0.0;
        assertTrue(a.snapshot().energy[0] > 0.0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsInvalidSeedShape() {
        new VnWaveField(3, 1, 2, 0.1, 0.4, 0.2).step(new double[2], 1);
    }

    @Test
    public void emptyBackgroundDoesNotWakeArbitrarySources() {
        VnWaveField field = new VnWaveField(8, 3, 16, 0.1, 0.4, 0.2);
        VnWaveField.Snapshot state = field.backgroundTick();
        assertEquals(0, state.activeSources);
        assertEquals(0.0, state.meanEnergy, 0.0);
    }

    @Test
    public void payloadEstimateStaysBoundedForMobileScale() {
        VnWaveField field = new VnWaveField(2048, 64, 65536, 0.1, 0.4, 0.2);
        for (int i = 0; i < 2048; i++) field.addDirectedEdge(i, (i + 1) % 2048, 0.5);
        assertTrue(field.estimatedPayloadBytes() < 1024L * 1024L);
        assertEquals(2048, field.getNodeCount());
        assertEquals(2048, field.getEdgeCount());
    }
}
