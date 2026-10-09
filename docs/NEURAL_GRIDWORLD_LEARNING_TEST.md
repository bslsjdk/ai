# AIMENG 小网络学习能力测试：网格寻路

## 目的

测试一个小型神经网络能否通过游戏反馈学会选择行动，而不是仅仅拟合人类提供的标准答案。环境为 5×5 网格，动作是上/右/下/左，碰壁扣分、每步有小惩罚、到达目标给奖励。

## 运行

需要 Python 3，不需要 NumPy 或第三方依赖：

```sh
python3 scripts/test_neural_gridworld.py --episodes 5000
```

输出到 `build/gridworld-training/`：
- `q_network.json`：训练后 8→16→4 小型 MLP 的参数检查点。
- `training_report.json`：初始/最终成功率、步数、路径效率和每 250 局的评估记录。

## 网络和训练

- 输入 8 个数：玩家坐标、目标坐标、上右下左四个方向是否被墙/边界阻挡。
- 隐藏层 16 个 ReLU 单元，输出 4 个动作的 Q 值。
- 使用 epsilon-greedy 探索和 TD(0) 神经 Q-learning 更新；没有给每个状态直接指定正确动作。
- 固定地图、随机起点。评估时关闭探索，遍历所有可通行起点。
- 训练轮数默认 5000，每局最多 45 步，模型很小，适合先做算法 smoke test。

## 如何判断

重点看 `initialEvaluation.success_rate` 和 `finalEvaluation.success_rate`，再看 `mean_steps_success` 与 `mean_optimality_ratio`。成功率提高是最基础证据；路径效率可以揭示它是否只是偶尔撞运气。

## 重要限制

这是独立、纯 Python 的小型神经网络算法基准，用于先验证学习方法。它**不是**当前 Android 工作台现有监督训练器的结果，也不能证明 Android 端已经能玩游戏。下一步需要将相同环境和训练循环移植进 Android 工作台，并让游戏策略检查点以独立格式保存，不能误当作现有 `NeuronWorkspace` 权重文件导入。
