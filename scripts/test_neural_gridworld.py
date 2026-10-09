#!/usr/bin/env python3
"""Tiny neural Q-learning benchmark for AIMENG: 5x5 grid-world.

Pure Python, no third-party dependencies. This is an algorithmic smoke test for
an actual small MLP Q-function, not a tabular Q-learning demo and not a claim
that the Android runtime has already been integrated.
"""
import argparse
import json
import math
import random
from pathlib import Path

SIZE = 5
ACTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # up, right, down, left
ACTION_NAMES = ("up", "right", "down", "left")
WALLS = {(1, 0), (1, 1), (3, 1), (3, 2), (1, 3), (2, 3)}
GOAL = (4, 4)


def relu(x):
    return x if x > 0.0 else 0.0


class TinyQNetwork:
    """8 -> 16 ReLU -> 4 MLP, trained with semi-gradient TD(0)."""
    def __init__(self, seed=20261009):
        rng = random.Random(seed)
        self.w1 = [[rng.uniform(-0.25, 0.25) for _ in range(8)] for _ in range(16)]
        self.b1 = [0.0] * 16
        self.w2 = [[rng.uniform(-0.25, 0.25) for _ in range(16)] for _ in range(4)]
        self.b2 = [0.0] * 4
        self.lr = 0.003

    def forward(self, x):
        h_pre = [sum(wi * xi for wi, xi in zip(row, x)) + b
                 for row, b in zip(self.w1, self.b1)]
        h = [relu(v) for v in h_pre]
        q = [sum(wi * hi for wi, hi in zip(row, h)) + b
             for row, b in zip(self.w2, self.b2)]
        return q, h, h_pre

    def update(self, x, action, target):
        q, h, h_pre = self.forward(x)
        error = q[action] - target
        # Clip the scalar TD gradient to keep early random rewards bounded.
        grad = max(-1.0, min(1.0, error))
        old_out = self.w2[action][:]
        for j in range(len(h)):
            self.w2[action][j] -= self.lr * grad * h[j]
        self.b2[action] -= self.lr * grad
        for j, hv in enumerate(h):
            if h_pre[j] <= 0.0:
                continue
            back = grad * old_out[j]
            for k in range(8):
                self.w1[j][k] -= self.lr * back * x[k]
            self.b1[j] -= self.lr * back
        return error * error

    def choose(self, state, epsilon, rng):
        if rng.random() < epsilon:
            return rng.randrange(4)
        q, _, _ = self.forward(state)
        best = max(q)
        choices = [i for i, value in enumerate(q) if abs(value - best) < 1e-12]
        return rng.choice(choices)

    def to_json(self):
        return {"format": "aimeng-gridworld-qnet/v1", "inputSize": 8,
                "hiddenSize": 16, "outputSize": 4, "activation": "relu",
                "w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}


def observation(pos):
    x, y = pos
    gx, gy = GOAL
    wall_flags = []
    for dx, dy in ACTIONS:
        nxt = (x + dx, y + dy)
        wall_flags.append(1.0 if not (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE)
                          or nxt in WALLS else 0.0)
    return [x / (SIZE - 1), y / (SIZE - 1), gx / (SIZE - 1),
            gy / (SIZE - 1)] + wall_flags


def step(pos, action):
    dx, dy = ACTIONS[action]
    nxt = (pos[0] + dx, pos[1] + dy)
    if not (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE) or nxt in WALLS:
        return pos, -0.12, False
    if nxt == GOAL:
        return nxt, 1.0, True
    return nxt, -0.025, False


def valid_starts():
    return [(x, y) for y in range(SIZE) for x in range(SIZE)
            if (x, y) not in WALLS and (x, y) != GOAL]


def shortest_path(start):
    queue = [(start, 0)]
    seen = {start}
    for pos, dist in queue:
        if pos == GOAL:
            return dist
        for dx, dy in ACTIONS:
            nxt = (pos[0] + dx, pos[1] + dy)
            if 0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE and nxt not in WALLS and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def evaluate(net, starts, max_steps=40):
    successes, steps_all, path_ratio = 0, [], []
    for start in starts:
        pos = start
        visited = set()
        for step_no in range(1, max_steps + 1):
            state = observation(pos)
            q, _, _ = net.forward(state)
            action = max(range(4), key=lambda i: q[i])
            pos, _, done = step(pos, action)
            if done:
                successes += 1
                steps_all.append(step_no)
                optimal = shortest_path(start)
                if optimal:
                    path_ratio.append(optimal / step_no)
                break
            # Stop a deterministic evaluation episode if the policy loops.
            marker = (pos, action)
            if marker in visited:
                break
            visited.add(marker)
    total = len(starts)
    return {"episodes": total, "successes": successes,
            "success_rate": successes / max(1, total),
            "mean_steps_success": sum(steps_all) / max(1, len(steps_all)),
            "mean_optimality_ratio": sum(path_ratio) / max(1, len(path_ratio))}


def train(episodes=5000, seed=20261009):
    rng = random.Random(seed)
    net = TinyQNetwork(seed)
    starts = valid_starts()
    initial = evaluate(net, starts)
    history = []
    gamma = 0.92
    max_steps = 45
    for episode in range(1, episodes + 1):
        start = rng.choice(starts)
        pos = start
        epsilon = max(0.04, 1.0 - 0.96 * episode / episodes)
        episode_reward = 0.0
        for _ in range(max_steps):
            state = observation(pos)
            action = net.choose(state, epsilon, rng)
            nxt, reward, done = step(pos, action)
            next_state = observation(nxt)
            next_q, _, _ = net.forward(next_state)
            target = reward if done else reward + gamma * max(next_q)
            net.update(state, action, target)
            episode_reward += reward
            pos = nxt
            if done:
                break
        if episode % 250 == 0 or episode == episodes:
            result = evaluate(net, starts)
            history.append({"episode": episode, "epsilon": epsilon,
                            "episodeReward": episode_reward, **result})
    final = evaluate(net, starts)
    return net, {"format": "aimeng-gridworld-training-report/v1",
                 "algorithm": "neural Q-learning, epsilon-greedy, TD(0)",
                 "seed": seed, "episodesTrained": episodes,
                 "mapSize": [SIZE, SIZE], "walls": [list(p) for p in sorted(WALLS)],
                 "goal": list(GOAL), "actions": list(ACTION_NAMES),
                 "initialEvaluation": initial, "finalEvaluation": final,
                 "history": history,
                 "note": "Pure-Python MLP benchmark; not an Android runtime result."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--out", type=Path, default=Path("build/gridworld-training"))
    args = parser.parse_args()
    if not 100 <= args.episodes <= 20000:
        parser.error("--episodes must be in [100, 20000]")
    args.out.mkdir(parents=True, exist_ok=True)
    net, report = train(args.episodes, args.seed)
    (args.out / "q_network.json").write_text(json.dumps(net.to_json(), indent=2) + "\n", encoding="utf-8")
    (args.out / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"initial": report["initialEvaluation"],
                      "final": report["finalEvaluation"],
                      "report": str(args.out / "training_report.json"),
                      "checkpoint": str(args.out / "q_network.json")}, indent=2))
    if report["finalEvaluation"]["success_rate"] <= report["initialEvaluation"]["success_rate"]:
        raise SystemExit("FAIL: evaluation success rate did not improve; inspect report and seed.")
    if report["finalEvaluation"]["success_rate"] < 0.50:
        raise SystemExit("FAIL: success rate is below the 50% smoke-test threshold.")
    print("PASS: held-out-by-episode deterministic grid-world evaluation improved.")


if __name__ == "__main__":
    main()
