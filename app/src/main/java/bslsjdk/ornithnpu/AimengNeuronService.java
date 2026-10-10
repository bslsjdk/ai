package bslsjdk.ornithnpu;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Binder;
import android.os.Build;
import android.os.IBinder;
import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * User-controlled resident AIMENG runtime.
 *
 * It does not run a busy loop while idle. The model and its bounded residual
 * neuron state stay in this service process; checkpoints make process restarts
 * recoverable, but Android/OEM policy can still terminate the process.
 */
public final class AimengNeuronService extends Service {
    public static final String ACTION_START = "bslsjdk.ornithnpu.aimeng.START_RESIDENT";
    public static final String ACTION_STOP = "bslsjdk.ornithnpu.aimeng.STOP_RESIDENT";
    private static final String CHANNEL = "aimeng_resident_runtime";
    private static final int NOTIFICATION_ID = 43017;

    private final LocalBinder binder = new LocalBinder();
    private final ExecutorService loader = Executors.newSingleThreadExecutor();
    private SparseDiffusionMobileModel model;
    private volatile boolean residentEnabled;

    public final class LocalBinder extends Binder {
        public AimengNeuronService getService() { return AimengNeuronService.this; }
    }

    @Override public void onCreate() {
        super.onCreate();
        model = new SparseDiffusionMobileModel();
        createNotificationChannel();
    }

    public SparseDiffusionMobileModel getModel() { return model; }
    public boolean isResidentEnabled() { return residentEnabled; }

    @Override public IBinder onBind(Intent intent) { return binder; }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent == null ? ACTION_START : intent.getAction();
        if (ACTION_STOP.equals(action)) {
            checkpointQuietly();
            residentEnabled = false;
            stopForeground(STOP_FOREGROUND_REMOVE);
            stopSelf();
            return START_NOT_STICKY;
        }
        // A sticky restart can arrive with a null Intent. Restore the last
        // checkpoint instead of starting an uninitialized hot loop.
        enterForeground();
        residentEnabled = true;
        if (!model.isLoaded()) {
            loader.execute(this::restoreModelAndState);
        } else {
            checkpointQuietly();
        }
        return START_STICKY;
    }

    private void enterForeground() {
        Intent stop = new Intent(this, AimengNeuronService.class).setAction(ACTION_STOP);
        PendingIntent stopPending = PendingIntent.getService(this, 43018, stop,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Notification.Builder builder = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(android.R.drawable.stat_notify_sync_noanim)
                .setContentTitle("AIMENG 神经元驻留服务")
                .setContentText("本机状态已驻留；空闲时不持续计算，等待输入")
                .setCategory(Notification.CATEGORY_SERVICE)
                .setOngoing(true)
                .addAction(new Notification.Action.Builder(null, "停止驻留", stopPending).build());
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIFICATION_ID, builder.build(),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        } else {
            startForeground(NOTIFICATION_ID, builder.build());
        }
    }

    private void createNotificationChannel() {
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm != null && Build.VERSION.SDK_INT >= 26) {
            nm.createNotificationChannel(new NotificationChannel(CHANNEL,
                    "AIMENG 后台神经元", NotificationManager.IMPORTANCE_LOW));
        }
    }

    private File modelFile() { return new File(getFilesDir(), "aimeng-mobile-diffusion.json"); }
    private File learningFile() { return new File(getFilesDir(), "aimeng-learning-state.json"); }
    private File runtimeFile() { return new File(getFilesDir(), "aimeng-neuron-residual.bin"); }

    private void restoreModelAndState() {
        try {
            File base = modelFile();
            if (!base.isFile()) return;
            model.load(base);
            model.loadLearningState(learningFile());
            model.loadRuntimeState(runtimeFile());
        } catch (Throwable ignored) {
            // The UI can show the detailed error if the user opens the runtime.
            // Keep the service idle rather than crashing/restarting in a loop.
        }
    }

    public synchronized void checkpoint() throws Exception {
        if (model != null && model.isLoaded()) model.saveRuntimeState(runtimeFile());
    }

    private void checkpointQuietly() {
        try { checkpoint(); } catch (Throwable ignored) { }
    }

    @Override public void onTaskRemoved(Intent rootIntent) {
        checkpointQuietly();
        super.onTaskRemoved(rootIntent);
    }

    @Override public void onDestroy() {
        checkpointQuietly();
        loader.shutdownNow();
        residentEnabled = false;
        super.onDestroy();
    }
}
