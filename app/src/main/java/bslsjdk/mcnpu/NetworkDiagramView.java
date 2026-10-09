package bslsjdk.ornithnpu;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.view.View;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Lightweight population-style view of the current numeric network.
 * Neurons are laid out as an independent pool rather than transformer-like stacked blocks.
 * Only the model's real input/output weights are drawn; no hidden-to-hidden edges are invented.
 */
public final class NetworkDiagramView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final ArrayList<NeuronWorkspace.Neuron> units = new ArrayList<>();
    private int inputCount = 1;
    private int outputCount = 1;

    private static final int INK = 0xFF1F2937;
    private static final int MUTED = 0xFF667085;
    private static final int INPUT = 0xFF3478B8;
    private static final int ACTIVE = 0xFF16867A;
    private static final int DISABLED = 0xFF9CA3AF;
    private static final int OUTPUT = 0xFF7257B8;
    private static final int CONNECTION = 0xFFCBD5E1;

    public NetworkDiagramView(Context context) {
        super(context);
        setBackgroundColor(0xFFFFFFFF);
        setLayerType(View.LAYER_TYPE_SOFTWARE, null);
        setContentDescription("输入节点、独立神经元群和输出节点组成的数值网络图");
    }

    public void setGraph(int inputs, int outputs, List<NeuronWorkspace.Neuron> neurons) {
        inputCount = Math.max(1, Math.min(NeuronWorkspace.MAX_INPUTS, inputs));
        outputCount = Math.max(1, Math.min(NeuronWorkspace.MAX_OUTPUTS, outputs));
        units.clear();
        if (neurons != null) for (NeuronWorkspace.Neuron n : neurons) units.add(n);
        invalidate();
    }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float density = getResources().getDisplayMetrics().density;
        float width = getWidth();
        float height = getHeight();
        if (width <= 0 || height <= 0) return;

        int shownInput = Math.min(6, inputCount);
        int shownOutput = Math.min(6, outputCount);
        int shownNeurons = Math.min(12, units.size());
        float top = 54 * density;
        float bottom = height - 38 * density;
        if (bottom < top + 40 * density) {
            top = 42 * density;
            bottom = Math.max(top + 30 * density, height - 28 * density);
        }
        float xIn = width * 0.09f;
        float xOut = width * 0.91f;
        float radius = Math.max(7.5f * density, Math.min(11.5f * density, width / 48f));

        text(canvas, "输入", xIn, 24 * density, 11 * density, INK, true, Paint.Align.CENTER);
        text(canvas, "独立神经元群", width * 0.50f, 24 * density,
                12 * density, INK, true, Paint.Align.CENTER);
        text(canvas, "输出", xOut, 24 * density, 11 * density, INK, true, Paint.Align.CENTER);

        float[] ysIn = rowPositions(shownInput, top, bottom);
        float[] ysOut = rowPositions(shownOutput, top, bottom);
        float[][] neuronXY = populationPositions(shownNeurons, width, top, bottom, density);

        // Draw only connections represented by the actual input/output weight arrays.
        for (int h = 0; h < shownNeurons; h++) {
            NeuronWorkspace.Neuron neuron = units.get(h);
            float nx = neuronXY[h][0];
            float ny = neuronXY[h][1];
            for (int i = 0; i < shownInput && i < neuron.inputWeights.length; i++) {
                float magnitude = finiteMagnitude(neuron.inputWeights[i]);
                paint.setColor(neuron.enabled
                        ? withAlpha(INPUT, Math.min(155, 28 + (int) (magnitude * 48)))
                        : 0x1F9CA3AF);
                paint.setStrokeWidth((0.45f + Math.min(1.3f, magnitude * 0.16f)) * density);
                canvas.drawLine(xIn + radius * 0.65f, ysIn[i],
                        nx - radius * 0.72f, ny, paint);
            }
            for (int o = 0; o < shownOutput && o < neuron.outputWeights.length; o++) {
                float magnitude = finiteMagnitude(neuron.outputWeights[o]);
                paint.setColor(neuron.enabled
                        ? withAlpha(ACTIVE, Math.min(175, 30 + (int) (magnitude * 52)))
                        : 0x1F9CA3AF);
                paint.setStrokeWidth((0.5f + Math.min(1.4f, magnitude * 0.17f)) * density);
                canvas.drawLine(nx + radius * 0.72f, ny,
                        xOut - radius * 0.65f, ysOut[o], paint);
            }
        }

        for (int i = 0; i < shownInput; i++)
            node(canvas, xIn, ysIn[i], radius * 0.82f, INPUT, "x" + i, density);

        for (int h = 0; h < shownNeurons; h++) {
            NeuronWorkspace.Neuron neuron = units.get(h);
            float nx = neuronXY[h][0];
            float ny = neuronXY[h][1];
            node(canvas, nx, ny, radius, neuron.enabled ? ACTIVE : DISABLED,
                    shortId(neuron.id), density);
            if (Math.abs(neuron.score) > 0.000001) {
                String score = String.format(Locale.US, "%+.2f", neuron.score);
                float labelX = nx + radius + 2 * density;
                Paint.Align align = labelX > width * 0.70f ? Paint.Align.RIGHT : Paint.Align.LEFT;
                if (align == Paint.Align.RIGHT) labelX = nx - radius - 2 * density;
                text(canvas, score, labelX, ny + 3 * density, density * 7.2f,
                        neuron.score >= 0 ? ACTIVE : 0xFFB45309, false, align);
            }
        }

        for (int o = 0; o < shownOutput; o++)
            node(canvas, xOut, ysOut[o], radius * 0.82f, OUTPUT, "y" + o, density);

        String note = "实线 = 已有输入/输出权重 · 绿 = 启用 · 灰 = 禁用";
        if (inputCount > shownInput || outputCount > shownOutput || units.size() > shownNeurons)
            note += " · 仅显示部分节点";
        text(canvas, note, width / 2, height - 9 * density,
                density * 7.2f, MUTED, false, Paint.Align.CENTER);
    }

    private float[][] populationPositions(int count, float width, float top, float bottom, float density) {
        float[][] result = new float[Math.max(0, count)][2];
        if (count == 0) return result;
        float centerX = width * 0.50f;
        float centerY = (top + bottom) * 0.50f;
        float radiusX = Math.min(width * 0.19f, 68 * density);
        float radiusY = Math.min((bottom - top) * 0.43f, 118 * density);
        // Golden-angle placement creates a compact population cloud, not stacked layers.
        double goldenAngle = Math.PI * (3.0 - Math.sqrt(5.0));
        for (int i = 0; i < count; i++) {
            double angle = i * goldenAngle - Math.PI / 2.0;
            double radial = count == 1 ? 0.0 : Math.sqrt((i + 0.55) / count);
            result[i][0] = centerX + (float) Math.cos(angle) * radiusX * (float) radial;
            result[i][1] = centerY + (float) Math.sin(angle) * radiusY * (float) radial;
        }
        return result;
    }

    private static float[] rowPositions(int count, float top, float bottom) {
        float[] ys = new float[Math.max(1, count)];
        if (ys.length == 1) ys[0] = (top + bottom) / 2;
        else for (int i = 0; i < ys.length; i++)
            ys[i] = top + (bottom - top) * i / (ys.length - 1f);
        return ys;
    }

    private void node(Canvas canvas, float x, float y, float r, int color, String label, float density) {
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(0xFFFFFFFF);
        canvas.drawCircle(x, y, r + 1.7f * density, paint);
        paint.setColor(color);
        canvas.drawCircle(x, y, r, paint);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(1.0f * density);
        paint.setColor(0xFFE5E7EB);
        canvas.drawCircle(x, y, r, paint);
        paint.setStyle(Paint.Style.FILL);
        float size = density * (label.length() > 4 ? 6.5f : 8f);
        text(canvas, label, x, y + size * 0.34f, size, 0xFFFFFFFF, true, Paint.Align.CENTER);
    }

    private void text(Canvas canvas, String value, float x, float y, float size,
                      int color, boolean bold, Paint.Align align) {
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(color);
        paint.setTextSize(size);
        paint.setTextAlign(align);
        paint.setTypeface(bold
                ? android.graphics.Typeface.create("sans-serif-medium", android.graphics.Typeface.BOLD)
                : android.graphics.Typeface.create("sans-serif", android.graphics.Typeface.NORMAL));
        canvas.drawText(value, x, y, paint);
    }

    private static String shortId(String id) {
        if (id == null) return "?";
        return id.startsWith("H-") ? id.substring(2) : id;
    }

    private static float finiteMagnitude(double value) {
        return Double.isFinite(value) ? Math.min(20f, Math.abs((float) value)) : 0f;
    }

    private static int withAlpha(int color, int alpha) {
        return (color & 0x00FFFFFF) | (Math.min(255, Math.max(0, alpha)) << 24);
    }
}
