# AIMENG 通用命令终端

## 两种执行环境

AIMENG 的「通用命令终端」不是只接受 Python 文件的按钮，而是一个命令提交器。它有两种环境：

1. **Android shell**：默认在应用自身 UID 下运行基础 Android shell 命令。授权并连接 Shizuku 后，可使用 Shizuku UserService 在 ADB shell 身份下运行命令。若 Shizuku 后端报告 root UID，AIMENG 会拒绝执行这一模式，避免误改系统分区。
2. **Termux 环境**：勾选“在 Termux 环境执行”后，将命令提交给 Termux 的 RUN_COMMAND 服务。这样才能使用 Termux 里实际安装的 Python、OpenJDK、Git 等工具，而不是假设 Android 系统自带它们。

命令执行结果会显示在终端输出区域。Android shell 模式默认单条命令最多 120 秒，输出最多保留 500,000 字符；Termux 模式使用 `timeout`（必须存在），同样请求 120 秒上限。两种模式都会先尝试设置 3 GiB 虚拟地址空间上限；如果 shell 不支持该限制，则拒绝执行命令。虚拟地址空间限制可能影响某些 JVM 或大型原生程序，不能将它等同于精确的 RSS 内存计量。

## 使用 Python / Java

1. 安装并首次打开 Termux。
2. 在 Termux 中执行 `pkg update`，然后安装需要的运行时，例如 `pkg install python openjdk-21 git coreutils`。
3. 在 Termux 中编辑 `~/.termux/termux.properties`，设置 `allow-external-apps=true`，然后重启 Termux。
4. 在 Android 设置里给 AIMENG 授予“在 Termux 环境中运行命令”权限。
5. 打开 AIMENG → 总览 → 通用命令终端，勾选 Termux 模式。
6. 可先运行 `python --version`、`java -version`、`javac -version`、`git --version` 检查环境。

Termux 的外部命令执行需要用户明确授权，并要求 Termux 自身允许外部应用调用。官方说明见 [Termux RUN_COMMAND Intent 文档](https://github.com/termux/termux-app/wiki/RUN_COMMAND-Intent)。

## 安装/更新 Android 应用

要使用系统包管理器执行安装或更新命令，需要先确认 Android shell 模式显示 Shizuku 已授权，并且 Shizuku 是通过无线调试/ADB 方式启动，而不是 root 身份。只对自己信任的 APK 使用安装命令；不执行来源不明的安装脚本。没有 Shizuku shell 权限时，终端会以应用自身权限运行，很多系统级命令会被 Android 拒绝。

## 限制与注意事项

- 终端不是完整的交互式 PTY：需要键盘实时输入的 TUI 程序可能无法工作。
- Termux 命令通过后台服务提交，结果在命令结束后返回；此模式不能从 AIMENG 内直接杀死 Termux 进程，命令会依赖 Termux 的 `timeout` 工具结束。
- 这不是 root 终端。AIMENG 明确拒绝 Shizuku root UID；Android shell UID 的权限仍受 Android 版本和厂商策略限制。
- 3 GiB 虚拟地址空间限制是保守保护措施，不是精确的物理内存使用量计量。训练和本地模型实验仍需观察应用 RSS/PSS 与设备温度，不能因有终端就运行无界任务。
- 终端会执行你输入的命令。不要复制来源不明的命令或脚本；某些命令可能删除文件、修改系统设置或安装不需要的软件。
