package bslsjdk.ornithnpu;
import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.widget.*;
public final class AimengHomeActivity extends Activity {
 int d(int n){return (int)(n*getResources().getDisplayMetrics().density+.5f);}
 TextView c(String s){TextView t=new TextView(this);t.setText(s);t.setTextSize(16);t.setTextColor(Color.rgb(35,49,72));t.setPadding(d(14),d(14),d(14),d(14));t.setBackgroundColor(Color.WHITE);LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(-1,-2);p.bottomMargin=d(10);return t;}
 void b(LinearLayout r,String s,Class<?> a){Button x=new Button(this);x.setText(s);x.setOnClickListener(v->startActivity(new Intent(this,a)));r.addView(x);}
 @Override protected void onCreate(Bundle b){super.onCreate(b);ScrollView sc=new ScrollView(this);sc.setFitsSystemWindows(true);sc.setFillViewport(true);sc.setBackgroundColor(Color.rgb(245,247,251));LinearLayout r=new LinearLayout(this);r.setOrientation(1);r.setPadding(d(18),d(16),d(18),d(24));sc.addView(r);r.addView(c("AIMENG 神经元"));r.addView(c("手机本地运行 · CPU + GPU · 不调用 NPU"));r.addView(c("导入 mobile_diffusion.json、生成文字并进行本地学习。不需要 Ornith 9B 或 safetensors。"));b(r,"打开神经元运行台",SparseDiffusionActivity.class);r.addView(c("扩散状态监控：查看真实前向计算的候选输出概率、后端、耗时与内存估算。逐步神经元轨迹还需接入内部 trace。"));b(r,"打开扩散状态监控",DiffusionProbeActivity.class);r.addView(c("运行时内存目标上限 4 GiB。GPU 不可用时回退 CPU，以实际 GPU 操作计数为准。"));setContentView(sc);}
}