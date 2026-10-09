package bslsjdk.ornithnpu;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;

import java.util.function.Consumer;

/** Receives bounded stdout/stderr and exit status from Termux RUN_COMMAND. */
public final class TermuxCommandResultReceiver extends BroadcastReceiver {
    public static volatile Consumer<String> resultHandler;
    @Override public void onReceive(Context context, Intent intent) {
        Bundle result = intent == null ? null : intent.getBundleExtra("result");
        if (result == null) {
            deliver("\n[Termux] 未收到结果包。请确认 Termux 版本支持 RUN_COMMAND 结果回传。\n");
            return;
        }
        String stdout = result.getString("stdout", "");
        String stderr = result.getString("stderr", "");
        String errmsg = result.getString("errmsg", "");
        int exit = result.getInt("exitCode", -1);
        int err = result.getInt("err", 0);
        StringBuilder b = new StringBuilder();
        if (!stdout.isEmpty()) b.append(stdout);
        if (!stderr.isEmpty()) b.append("\n[stderr]\n").append(stderr);
        if (!errmsg.isEmpty()) b.append("\n[Termux error] ").append(errmsg);
        b.append("\n[Termux 完成] 退出码=").append(exit).append("；内部错误码=").append(err).append("\n");
        deliver(b.toString());
    }
    private static void deliver(String text) {
        Consumer<String> callback = resultHandler;
        if (callback != null) callback.accept(text);
        else Log.i("AIMENG-Terminal", "Termux command completed after terminal UI closed.");
    }
}
