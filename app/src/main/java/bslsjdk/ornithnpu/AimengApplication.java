package bslsjdk.ornithnpu;
import android.app.*;
import android.graphics.Insets;
import android.os.Bundle;
import android.view.*;
public final class AimengApplication extends Application {
 @Override public void onCreate(){super.onCreate();registerActivityLifecycleCallbacks(new ActivityLifecycleCallbacks(){
  public void onActivityCreated(Activity a,Bundle b){if(!(a instanceof SparseDiffusionActivity))return;ViewGroup g=a.findViewById(android.R.id.content);if(g==null||g.getChildCount()==0)return;View v=g.getChildAt(0);int l=v.getPaddingLeft(),t=v.getPaddingTop(),r=v.getPaddingRight(),bt=v.getPaddingBottom();v.setOnApplyWindowInsetsListener((w,i)->{Insets x=i.getInsets(WindowInsets.Type.statusBars()|WindowInsets.Type.navigationBars()|WindowInsets.Type.displayCutout());w.setPadding(l,t+x.top,r,bt+x.bottom);return i;});v.requestApplyInsets();}
  public void onActivityStarted(Activity a){} public void onActivityResumed(Activity a){} public void onActivityPaused(Activity a){} public void onActivityStopped(Activity a){} public void onActivitySaveInstanceState(Activity a,Bundle b){} public void onActivityDestroyed(Activity a){}
 });}
}