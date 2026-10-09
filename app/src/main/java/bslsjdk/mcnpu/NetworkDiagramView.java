package bslsjdk.ornithnpu;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.view.View;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** Lightweight native canvas drawing for the input → hidden → output network. */
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
        setContentDescription("输入层连接隐藏神经元，再汇总到输出层的网络图");
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

        int shownInput = Math.min(8, inputCount);
        int shownOutput = Math.min(8, outputCount);
        int shownHidden = Math.min(16, units.size());
        float top = 58 * density;
        float bottom = height - 32 * density;
        if (bottom < top + 40 * density) bottom = top + 40 * density;
        float xIn = width * 0.13f;
        float xHidden = width * 0.50f;
        float xOut = width * 0.87f;
        float radius = Math.max(8 * density, Math.min(14 * density, width / 42f));

        text(canvas, "输入", xIn, 26 * density, density, INK, true, Paint.Align.CENTER);
        text(canvas, "隐藏层", xHidden, 26 * density, density, INK, true, Paint.Align.CENTER);
        text(canvas, "输出", xOut, 26 * density, density, INK, true, Paint.Align.CENTER);

        float[] ysIn = positions(shownInput, top, bottom);
        float[] ysHidden = positions(Math.max(1, shownHidden), top, bottom);
        float[] ysOut = positions(shownOutput, top, bottom);

        // Draw connections first so the nodes remain readable above them.
        for (int h = 0; h < shownHidden; h++) {
            NeuronWorkspace.Neuron n = units.get(h);
            int lineColor = n.enabled ? CONNECTION : 0xFFE5E7EB;
            for (int i = 0; i < shownInput; i++) {
                float magnitude = i < n.inputWeights.length ? Math.abs((float) n.inputWeights[i]) : 0;
                paint.setColor(n.enabled ? withAlpha(CONNECTION, Math.min(190, 45 + (int) (magnitude * 60))) : lineColor);
                paint.setStrokeWidth((n.enabled ? 0.7f : 0.4f) * density
                        + Math.min(1.4f * density, magnitude * 0.22f * density));
                canvas.drawLine(xIn + radius * 0.7f, ysIn[i], xHidden - radius * 0.7f, ysHidden[h], paint);
            }
            for (int o = 0; o < shownOutput; o++) {
                float magnitude = o < n.outputWeights.length ? Math.abs((float) n.outputWeights[o]) : 0;
                paint.setColor(n.enabled ? withAlpha(ACTIVE, Math.min(175, 35 + (int) (magnitude * 65))) : 0xFFE5E7EB);
                paint.setStrokeWidth((n.enabled ? 0.8f : 0.4f) * density
                        + Math.min(1.4f * density, magnitude * 0.2f * density));
                canvas.drawLine(xHidden + radius * 0.7f, ysHidden[h], xOut - radius * 0.7f, ysOut[o], paint);
            }
        }

        for (int i = 0; i < shownInput; i++) {
            node(canvas, xIn, ysIn[i], radius, INPUT, "x" + i, density);
        }
        for (int h = 0; h < shownHidden; h++) {
            NeuronWorkspace.Neuron n = units.get(h);
            node(canvas, xHidden, ysHidden[h], radius, n.enabled ? ACTIVE : DISABLED,
                    n.id.replace("H-", "H"), density);
            if (Math.abs(n.score) > 0.000001) {
                String score = String.format(Locale.US, "%+.3f", n.score);
                text(canvas, score, xHidden + radius + 3 * density, ysHidden[h] + 4 * density,
                        density * 0.72f, n.score >= 0 ? ACTIVE : 0xFFB45309, false, Paint.Align.LEFT);
            }
        }
        for (int o = 0; o < shownOutput; o++) {
            node(canvas, xOut, ysOut[o], radius, OUTPUT, "y" + o, density);
        }

        String note = "连接权重示意 · 绿色活动 / 灰色禁用";
        if (inputCount > shownInput || outputCount > shownOutput || units.size() > shownHidden)
            note += " · 图中仅绘制部分节点";
        text(canvas, note, width / 2, height - 8 * density, density * 0.76f, MUTED, false, Paint.Align.CENTER);
    }

    private void node(Canvas canvas, float x, float y, float r, int color, String label, float density) {
        paint.setColor(0xFFFFFFFF);
        paint.setStyle(Paint.Style.FILL);
        canvas.drawCircle(x, y, r + 2 * density, paint);
        paint.setColor(color);
        canvas.drawCircle(x, y, r, paint);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(1.2f * density);
        paint.setColor(0xFFE5E7EB);
        canvas.drawCircle(x, y, r, paint);
        paint.setStyle(Paint.Style.FILL);
        float size = density * (label.length() > 4 ? 7.5f : 9f);
        text(canvas, label, x, y + size * 0.36f, size, 0xFFFFFFFF, true, Paint.Align.CENTER);
    }

    private void text(Canvas canvas, String value, float x, float y, float size,
                      int color, boolean bold, Paint.Align align) {
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(color);
        paint.setTextSize(size);
        paint.setTextAlign(align);
        paint.setTypeface(bold ? android.graphics.Typeface.create("sans-serif-medium", android.graphics.Typeface.BOLD)
                : android.graphics.Typeface.create("sans-serif", android.graphics.Typeface.NORMAL));
        canvas.drawText(value, x, y, paint);
    }

    private static float[] positions(int count, float top, float bottom) {
        float[] ys = new float[Math.max(1, count)];
        if (ys.length == 1) {
            ys[0] = (top + bottom) / 2;
        } else {
            for (int i = 0; i < ys.length; i++) ys[i] = top + (bottom - top) * i / (ys.length - 1f);
        }
        return ys;
    }

    private static int withAlpha(int color, int alpha) {
        return (color & 0x00FFFFFF) | (Math.min(255, Math.max(0, alpha)) << 24);
    }
}