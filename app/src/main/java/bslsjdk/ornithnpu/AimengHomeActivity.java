package bslsjdk.ornithnpu;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public final class AimengHomeActivity extends Activity {
    private int d(int n) { return (int)(n * getResources().getDisplayMetrics().density + 0.5f); }
    private TextView card(String text) {
        TextView t = new TextView(this);
        t.setText(text); t.setTextSize(15); t.setTextColor(Color.rgb(35, 49, 72));
        t.setPadding(d(14), d(13), d(14), d(13)); t.setBackgroundColor(Color.WHITE);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, -2);
        p.bottomMargin = d(10);
        return t;
    }
    private void button(LinearLayout root, String label, Class<?> target) {
        Button b = new Button(this); b.setText(label);
        b.setOnClickListener(v -> startActivity(new Intent(this, target)));
        root.addView(b);
    }
    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        ScrollView scroll = new ScrollView(this);
        scroll.setFitsSystemWindows(true); scroll.setFillViewport(true);
        scroll.setBackgroundColor(Color.rgb(245, 247, 251));
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(d(18), d(16), d(18), d(24));
        scroll.addView(root);
        root.addView(card("AIMENG 神经元"));
        root.addView(card("手机本地推理 · CPU + 可选 OpenCL GPU · 不调用 MCNPU/NPU"));
        root.addView(card("统一手机模型包：AIMENG .aimg（二进制头、压缩权重载荷、SHA-256 校验）。旧版 mobile_diffusion.json 仍可导入；.pth 不作为手机运行格式。"));
        button(root, "打开本地聊天", AimengChatActivity.class);
        button(root, "打开神经元运行台 / 导入模型", SparseDiffusionActivity.class);
        button(root, "查看扩散步骤与神经元活动", DiffusionProbeActivity.class);
        button(root, "打开监控说明", DiffusionMonitorActivity.class);
        root.addView(card("内存策略：按需加载，禁止启动时强占 2.5GB。目标为进程峰值低于 4 GiB，必须以真机 PSS/RSS 测量为准。云端训练导出仍需与 .aimg 规格对齐。"));
        setContentView(scroll);
    }
}
