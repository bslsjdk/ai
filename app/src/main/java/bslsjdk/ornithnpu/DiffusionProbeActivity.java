package bslsjdk.ornithnpu;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Debug;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class DiffusionProbeActivity extends Activity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private SparseDiffusionMobileModel model;
    private EditText input;
    private TextView status, output, config;
    private Button run;
    private int dp(int n) { return (int)(n * getResources().getDisplayMetrics().density + 0.5f); }
    private LinearLayout.LayoutParams gap() { LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1,-2); p.bottomMargin=dp(9); return p; }
    private TextView card(String s) { TextView t=new TextView(this); t.setText(s); t.setTextSize(14); t.setTextColor(Color.rgb(34,48,70)); t.setPadding(dp(13),dp(12),dp(13),dp(12)); t.setBackgroundColor(Color.WHITE); return t; }
    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        ScrollView sc=new ScrollView(this); sc.setFitsSystemWindows(true); sc.setFillViewport(true); sc.setBackgroundColor(Color.rgb(245,247,251));
        LinearLayout root=new LinearLayout(this); root.setOrientation(1); root.setPadding(dp(18),dp(14),dp(18),dp(24)); sc.addView(root);
        root.addView(card("扩散状态 · 实时数值探针"),gap());
        status=card("正在加载本机模型…"); root.addView(status,gap());
        config=card("模型参数待读取"); root.addView(config,gap());
        input=new EditText(this); input.setMinLines(2); input.setHint("输入提示文字，例如：你好"); root.addView(input,gap());
        run=new Button(this); run.setText("执行一次完整前向计算"); run.setEnabled(false); run.setOnClickListener(v->probe()); root.addView(run,gap());
        output=card("计算后的候选字符分布显示在这里。"); root.addView(output,gap());
        root.addView(card("这些概率来自模型真实 logits。它们能显示当前输入下模型倾向输出什么，但不是自然语言思维解释。逐扩散步的节点激活/消息能量还需要在模型 forward 中额外记录，本页不会伪造那些数据。"),gap());
        setContentView(sc); load();
    }
    private void load() {
        File f=new File(getFilesDir(),"aimeng-mobile-diffusion.json");
        if(!f.isFile()){status.setText("没有找到已导入模型。请先在神经元运行台导入 mobile_diffusion.json。");return;}
        worker.execute(()->{try{
            JSONObject r=new JSONObject(new String(Files.readAllBytes(f.toPath()),StandardCharsets.UTF_8));
            JSONObject c=r.getJSONObject("config");
            String info="神经元 "+c.optInt("neurons")+" · 宽度 "+c.optInt("width")+"\n每步激活上限 "+c.optInt("active_k")+" · 每节点出边 "+c.optInt("fanout")+"\n最大扩散步数 "+c.optInt("max_steps");
            SparseDiffusionMobileModel m=new SparseDiffusionMobileModel();m.load(f);model=m;
            long pss=Debug.getPss()/1024L;
            runOnUiThread(()->{config.setText(info);status.setText("模型就绪 · "+model.backendStatus()+" · PSS "+pss+" MiB");run.setEnabled(true);});
        }catch(Throwable e){runOnUiThread(()->status.setText("加载失败："+e.getClass().getSimpleName()+" "+String.valueOf(e.getMessage())));}});
    }
    private void probe() {
        String p=input.getText().toString();if(p.trim().isEmpty()){status.setText("请输入提示文字。");return;}
        run.setEnabled(false);status.setText("正在执行本机推理…");
        worker.execute(()->{try{
            long start=System.nanoTime();float[] logits=model.debugLogits(p);
            JSONObject r=new JSONObject(new String(Files.readAllBytes(new File(getFilesDir(),"aimeng-mobile-diffusion.json").toPath()),StandardCharsets.UTF_8));
            JSONArray tokens=r.getJSONArray("itos");float max=-Float.MAX_VALUE;for(float x:logits)max=Math.max(max,x);
            double[] probs=new double[logits.length];double sum=0;for(int i=0;i<logits.length;i++){probs[i]=Math.exp(logits[i]-max);sum+=probs[i];}
            boolean[] seen=new boolean[logits.length];StringBuilder s=new StringBuilder("下一个字符候选（前 8 名）\n");
            for(int rank=0;rank<Math.min(8,logits.length);rank++){int best=-1;for(int i=0;i<logits.length;i++)if(!seen[i]&&(best<0||probs[i]>probs[best]))best=i;if(best<0)break;seen[best]=true;
                String token=tokens.optString(best,"?").replace("\n","换行符").replace("\t","制表符");
                s.append(rank+1).append(". ").append(token).append("  ").append(String.format(Locale.ROOT,"%.2f%%",100*probs[best]/Math.max(sum,1e-30))).append("\n");}
            long ms=(System.nanoTime()-start)/1_000_000L,pss=Debug.getPss()/1024L;
            runOnUiThread(()->{output.setText(s.toString());status.setText("完成 · "+ms+" ms · "+model.backendStatus()+" · PSS "+pss+" MiB");run.setEnabled(true);});
        }catch(Throwable e){runOnUiThread(()->{status.setText("计算失败："+e.getClass().getSimpleName()+" "+String.valueOf(e.getMessage()));run.setEnabled(true);});}}
        );
    }
    @Override protected void onDestroy(){worker.shutdownNow();super.onDestroy();}
}
