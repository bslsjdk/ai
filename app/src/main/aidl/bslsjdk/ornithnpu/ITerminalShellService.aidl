package bslsjdk.ornithnpu;

interface ITerminalShellService {
    String execute(String command, String workingDirectory, int timeoutSeconds, int maxOutputChars);
    void cancel();
}
