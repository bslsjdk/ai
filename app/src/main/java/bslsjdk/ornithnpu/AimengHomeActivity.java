package bslsjdk.ornithnpu;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public final class AimengHomeActivity extends Activity {
    private int dp(int x) { return (int)(x * getResources().getDisplayMetrics().density + 0.5f); }
    private LinearLayout.LayoutParams gap(int bottom) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, -2);
        p.bottomMargin = dp(bottom); return p;
    }
    private TextView text(String s, int size, boolean bold) {
        TextView t = new TextView(this); t.setText(s); t.setTextSize(size);
        t.setTextColor(Color.rgb(31, 46, 70));
        if (bold) t.setTypeface(null, android.graphics.Typeface.BOLD);
        t.setPadding(dp(14), dp(12), dp(14), dp(12));
        return t;
    }
    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        ScrollView scroll = new ScrollView(this); scroll.setFitsSystemWindows(true); scroll.setFillViewport(true);
        scroll.setBackgroundColor(Color.rgb(245, 247, 251));
        LinearLayout root = new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(20), dp(20), dp(20), dp(28)); scroll.addView(root);
        root.addView(text("AIMENG 神经元", 27, true), gap(4));
        root.addView(text("手机本地运行 · CPU + GPU · 不接入 NPU", 15, false), gap(18));
        root.addView(text("运行台", 18, true), gap(6));
        root.addView(text("导入 mobile_diffusion.json、生成文字、进行本地学习并查看计算后端。无需导入 Ornith 9B 或 safetensors。", 14, false), gap(8));
        Button runtime = new Button(this); runtime.setText("打开神经元运行台");
        runtime.setOnClickListener(v -> startActivity(new Intent(this, SparseDiffusionActivity.class)));
        root.addView(runtime, gap(18));
        root.addView(text("扩散监控", 18, true), gap(6));
        root.addView(text("查看扩散架构、路由机制和监控能力说明。逐步激活快照需要后续接入模型内部 trace，当前页面不会伪造实时神经元数据。", 14, false), gap(8));
        Button monitor = new Button(this); monitor.setText("打开扩散状态监控");
        monitor.setOnClickListener(v -> startActivity(new Intent(this, DiffusionMonitorActivity.class)));
        root.addView(monitor, gap(18));
        root.addView(text("安全与兼容", 16, true), gap(4));
        root.addView(text("运行内存目标上限 4 GiB。GPU 不可用时回退 CPU；是否真正使用 GPU，以运行台显示的 GPU 操作计数为准。", 14, false), gap(8));
        setContentView(scroll);
    }
}
