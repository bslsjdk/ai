package bslsjdk.ornithnpu;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

public final class DiffusionMonitorActivity extends Activity {
    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        TextView view = new TextView(this);
        view.setText("AIMENG 扩散过程监控");
        setContentView(view);
    }
}
