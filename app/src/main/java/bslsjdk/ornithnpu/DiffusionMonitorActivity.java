package bslsjdk.ornithnpu;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public final class DiffusionMonitorActivity extends Activity {
    private int dp(int x) { return (int)(x * getResources().getDisplayMetrics().density + 0.5f); }
    private TextView card(String text) {
        TextView v = new TextView(this);
        v.setText(text); v.setTextSize(15); v.setTextColor(Color.rgb(35, 48, 70));
        v.setPadding(dp(14), dp(13), dp(14), dp(13)); v.setBackgroundColor(Color.WHITE);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, -2); p.bottomMargin = dp(10);
        return v;
    }
    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        ScrollView scroll = new ScrollView(this); scroll.setFitsSystemWindows(true); scroll.setFillViewport(true);
        scroll.setBackgroundColor(Color.rgb(245, 247, 251));
        LinearLayout root = new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(18), dp(14), dp(18), dp(24)); scroll.addView(root);
        root.addView(card("AIMENG · 扩散状态监控"));
        root.addView(card("计算策略：GPU 优先处理适合的大型线性层，其他操作由 CPU 执行。GPU 不可用时回退 CPU。不调用 NPU。"));
        root.addView(card("扩散流程：输入编码 → 上下文投影 → 选择活跃神经元 → 沿连接传播消息 → 更新神经元状态 → 汇总状态 → 计算候选输出与停止概率。"));
        root.addView(card("观察指标：活跃神经元数量、路由能量、消息传播规模、状态变化量、停止概率、候选字符分布、推理耗时与进程 PSS。"));
        root.addView(card("当前限制：模型代码尚未导出逐扩散步的神经元状态快照，因此本页先展示真实的流程结构，不伪造实时数据。下一步需要给 forward 增加逐步 trace 接口，再把数值渲染成动态节点图。"));
        Button back = new Button(this); back.setText("关闭监控"); back.setOnClickListener(v -> finish()); root.addView(back);
        setContentView(scroll);
    }
}
