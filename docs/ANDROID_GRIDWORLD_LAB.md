# Android visual neural grid-world lab

## How to open it

1. Install the newest signed APK from the repository's `latest-apk` release after the corresponding Android build has succeeded.
2. Open **AIMENG 神经元实验**.
3. On the **总览** page, tap **打开迷宫实验 · 观看神经网络学习**.
4. Start with **训练 1,000 局**. The training runs on a background thread; the page periodically updates its progress.
5. When training finishes, use **观看策略（自动走）** to watch the blue agent attempt to reach the green goal. **单步执行** lets you inspect one decision at a time.
6. Inspect the 16 hidden-unit activation values and four action Q values. The larger Q value is the action the policy currently prefers; it is not a probability or a guarantee of success.
7. Tap **保存网络与报告** to save the checkpoint and evaluation report in the app's private `files/gridworld-lab/` directory.

## What is being tested

- A small on-device neural Q-learning model: 8 inputs, 16 ReLU hidden units, 4 action values.
- Epsilon-greedy exploration and temporal-difference updates.
- A fixed 5×5 map with walls and a goal; evaluation checks the policy from all valid starting cells.
- Visible board state, recent path, hidden activations, action values, episode count, reward and fixed-start success rate.

The game model is a **separate experiment network**. It does not silently replace or mutate the general numeric network in the main Neuron Workspace. The report is an actual Android run report, distinct from the standalone Python benchmark under `scripts/test_neural_gridworld.py`.

## Resource limits and current limitations

The model is deliberately tiny and uses only Java/Android APIs; no Python runtime or third-party numerical package is loaded. Training runs off the UI thread, and the page exposes only small arrays and a 5×5 board. As with any mobile experiment, keep an eye on device temperature and memory during longer runs.

This is the first visual benchmark, not a general-purpose Python terminal. Arbitrary shell/Python execution is not exposed because it needs an explicit runtime, cancellation, output capture, storage boundaries and resource limits. A later task runner should launch approved scripts in an isolated workspace with bounded memory/time rather than execute unrestricted commands inside the app.
