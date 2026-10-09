package bslsjdk.ornithnpu;

import android.os.RemoteException;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

/** Small Shizuku UserService; executes commands with the Shizuku shell identity. */
public final class TerminalShellUserService extends ITerminalShellService.Stub {
    private volatile Process current;
    private volatile boolean cancelled;

    @Override public String execute(String command, String workingDirectory, int timeoutSeconds, int maxOutputChars)
            throws RemoteException {
        cancelled = false;
        StringBuilder out = new StringBuilder();
        Process p = null;
        try {
            File dir = new File(workingDirectory == null ? "/sdcard" : workingDirectory);
            if (!dir.isDirectory()) dir = new File("/");
            String script = "ulimit -v 3145728 || { echo MEMORY_LIMIT_UNAVAILABLE; exit 125; }; cd /sdcard 2>/dev/null || cd /; eval \"$1\" 2>&1";
            p = new ProcessBuilder("/system/bin/sh", "-c", script, "aimeng", command)
                    .directory(dir).redirectErrorStream(true).start();
            current = p;
            final Process active = p;
            Thread reader = new Thread(() -> {
                try (BufferedReader br = new BufferedReader(new InputStreamReader(active.getInputStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        synchronized (out) {
                            if (out.length() < maxOutputChars) out.append(line).append('\n');
                        }
                    }
                } catch (Throwable e) {
                    synchronized (out) { if (out.length() < maxOutputChars) out.append("[output reader] ").append(e).append('\n'); }
                }
            }, "aimeng-shell-output");
            reader.setDaemon(true);
            reader.start();
            boolean ended = p.waitFor(Math.max(1, Math.min(timeoutSeconds, 120)), TimeUnit.SECONDS);
            if (!ended) {
                p.destroy();
                if (!p.waitFor(500, TimeUnit.MILLISECONDS)) p.destroyForcibly();
            }
            reader.join(1200);
            int code = ended ? p.exitValue() : 124;
            synchronized (out) {
                if (out.length() > maxOutputChars) out.setLength(maxOutputChars);
                out.append("\n[Shizuku shell 结束] exitCode=").append(code);
                if (cancelled) out.append("；已请求停止");
                if (!ended) out.append("；超过时限，已终止主进程");
                out.append('\n');
                return out.toString();
            }
        } catch (Throwable e) {
            synchronized (out) {
                out.append("\n[Shizuku shell 错误] ").append(e.getClass().getSimpleName())
                        .append(": ").append(String.valueOf(e.getMessage())).append('\n');
                return out.toString();
            }
        } finally {
            current = null;
            if (p != null && p.isAlive()) try { p.destroyForcibly(); } catch (Throwable ignored) {}
        }
    }

    @Override public void cancel() throws RemoteException {
        cancelled = true;
        Process p = current;
        if (p != null) {
            try { p.destroy(); } catch (Throwable ignored) {}
        }
    }
}
