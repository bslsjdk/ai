#!/usr/bin/env python3
"""Parameter-matched MLP vs recurrent thought-cycle Q-network on held-out maps.

Both models use the same observations, reward, map/start schedule, exploration
schedule, TD target, training budget, and held-out evaluation cases.
"""
import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import test_gridworld_generalization as env
import test_neural_gridworld as base


class RecurrentThoughtQNetwork:
    """8->9 recurrent hidden units, two shared-weight thought cycles, 4 Q outputs."""
    def __init__(self, seed):
        self.hidden = 9
        rng = random.Random(seed)
        self.w_in = [[rng.uniform(-0.5 / math.sqrt(8), 0.5 / math.sqrt(8))
                      for _ in range(8)] for _ in range(self.hidden)]
        self.w_rec = [[rng.uniform(-0.1 / math.sqrt(self.hidden), 0.1 / math.sqrt(self.hidden))
                       for _ in range(self.hidden)] for _ in range(self.hidden)]
        self.b = [0.0] * self.hidden
        self.w_out = [[rng.uniform(-0.25, 0.25) for _ in range(self.hidden)] for _ in range(4)]
        self.b_out = [0.0] * 4
        self.lr = 0.003
        self._cache = None

    def forward(self, x):
        pre1 = [sum(w * v for w, v in zip(row, x)) + b
                for row, b in zip(self.w_in, self.b)]
        h1 = [v if v > 0.0 else 0.0 for v in pre1]
        pre2 = [sum(self.w_in[j][k] * x[k] for k in range(8))
                + sum(self.w_rec[j][k] * h1[k] for k in range(self.hidden))
                + self.b[j] for j in range(self.hidden)]
        h2 = [v if v > 0.0 else 0.0 for v in pre2]
        q = [sum(w * h for w, h in zip(row, h2)) + b
             for row, b in zip(self.w_out, self.b_out)]
        self._cache = (list(x), pre1, h1, pre2, h2)
        return q, h2, pre2

    def choose(self, state, epsilon, rng):
        if rng.random() < epsilon:
            return rng.randrange(4)
        q, _, _ = self.forward(state)
        best = max(q)
        choices = [i for i, value in enumerate(q) if abs(value - best) < 1e-12]
        return rng.choice(choices)

    def update(self, x, action, target):
        q, _, _ = self.forward(x)
        x, pre1, h1, pre2, h2 = self._cache
        error = q[action] - target
        grad = max(-1.0, min(1.0, error))
        old_out = self.w_out[action][:]
        old_rec = [row[:] for row in self.w_rec]

        d2 = [grad * old_out[j] if pre2[j] > 0.0 else 0.0
              for j in range(self.hidden)]
        d1 = []
        for k in range(self.hidden):
            back = sum(d2[j] * old_rec[j][k] for j in range(self.hidden))
            d1.append(back if pre1[k] > 0.0 else 0.0)

        for j in range(self.hidden):
            self.w_out[action][j] -= self.lr * grad * h2[j]
        self.b_out[action] -= self.lr * grad

        for j in range(self.hidden):
            for k in range(8):
                self.w_in[j][k] -= self.lr * (d2[j] + d1[j]) * x[k]
            for k in range(self.hidden):
                self.w_rec[j][k] -= self.lr * d2[j] * h1[k]
            self.b[j] -= self.lr * (d2[j] + d1[j])
        return error * error

    def to_json(self):
        return {
            "format": "aimeng-recurrent-thought-qnet/v1",
            "inputSize": 8,
            "hiddenSize": self.hidden,
            "thoughtCycles": 2,
            "parameterCount": 202,
            "activation": "relu",
            "wIn": self.w_in,
            "wRec": self.w_rec,
            "b": self.b,
            "wOut": self.w_out,
            "bOut": self.b_out,
        }


def make_schedule(seed, episodes):
    rng = random.Random(800000 + seed)
    schedule = []
    for _ in range(episodes):
        world = env.make_map(rng)
        start = rng.choice([p for p in world["reachable"] if p != env.GOAL])
        schedule.append((world["walls"], start))
    return schedule


def train_model(net, seed, episodes, schedule):
    rng = random.Random(424242 + seed)
    gamma = 0.92
    started = time.time()
    for episode, (walls, start) in enumerate(schedule, start=1):
        pos = start
        epsilon = max(0.05, 1.0 - 0.95 * episode / episodes)
        for _ in range(env.MAX_STEPS):
            state = env.observation(pos, walls)
            action = net.choose(state, epsilon, rng)
            nxt, reward, done = env.step(pos, action, walls)
            next_state = env.observation(nxt, walls)
            next_q, _, _ = net.forward(next_state)
            target = reward if done else reward + gamma * max(next_q)
            net.update(state, action, target)
            pos = nxt
            if done:
                break
    return time.time() - started


def evaluate_architecture(factory, param_count, seeds, episodes, test_cases, out_dir):
    records = []
    for seed in seeds:
        schedule = make_schedule(seed, episodes)
        net = factory(seed)
        initial = env.evaluate(net, test_cases)
        seconds = train_model(net, seed, episodes, schedule)
        final = env.evaluate(net, test_cases)
        record = {
            "seed": seed,
            "episodesTrained": episodes,
            "parameterCount": param_count,
            "initialEvaluation": initial,
            "finalEvaluation": final,
            "successDelta": final["success_rate"] - initial["success_rate"],
            "trainingSeconds": round(seconds, 3),
        }
        records.append(record)
        seed_dir = out_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "training_report.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (seed_dir / "q_network.json").write_text(
            json.dumps(net.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("ARCH_SEED_RESULT " + json.dumps(record, sort_keys=True), flush=True)
    initial_rates = [r["initialEvaluation"]["success_rate"] for r in records]
    final_rates = [r["finalEvaluation"]["success_rate"] for r in records]
    return {
        "parameterCount": param_count,
        "perSeed": records,
        "aggregate": {
            "initialSuccessRateMean": statistics.mean(initial_rates),
            "finalSuccessRateMean": statistics.mean(final_rates),
            "finalSuccessRateStdev": statistics.pstdev(final_rates),
            "seedsImproved": sum(f > i for f, i in zip(final_rates, initial_rates)),
            "seedCount": len(seeds),
            "finalSuccessRateMin": min(final_rates),
            "finalSuccessRateMax": max(final_rates),
            "meanTrainingSeconds": statistics.mean(r["trainingSeconds"] for r in records),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("build/gridworld-architecture-comparison"))
    args = parser.parse_args()
    if not 100 <= args.episodes <= 20000:
        parser.error("--episodes must be in [100, 20000]")
    if not 1 <= args.seeds <= 10:
        parser.error("--seeds must be in [1, 10]")

    args.out.mkdir(parents=True, exist_ok=True)
    test_cases = env.build_test_cases(9302026, map_count=100, starts_per_map=5)
    random_result = env.random_baseline(test_cases, 9302027)
    oracle_result = env.oracle_baseline(test_cases)
    seeds = [20263001 + i for i in range(args.seeds)]
    print("AIMENG_PARAMETER_MATCHED_ARCHITECTURE_COMPARISON/v1", flush=True)
    print(f"episodes_per_seed={args.episodes} seeds={seeds} heldout_cases={len(test_cases)}", flush=True)
    print("fairness=same observation, reward, map/start schedule, epsilon schedule, TD target, test cases", flush=True)
    print("random_baseline=" + json.dumps(random_result, sort_keys=True), flush=True)
    print("bfs_oracle=" + json.dumps(oracle_result, sort_keys=True), flush=True)

    mlp = evaluate_architecture(base.TinyQNetwork, 212, seeds, args.episodes,
                                test_cases, args.out / "mlp-212")
    recurrent = evaluate_architecture(RecurrentThoughtQNetwork, 202, seeds, args.episodes,
                                      test_cases, args.out / "recurrent-thought-202")
    summary = {
        "format": "aimeng-parameter-matched-architecture-report/v1",
        "episodesPerSeed": args.episodes,
        "seeds": seeds,
        "evaluation": "100 fixed-seed held-out maps, 5 starts per map, exploration disabled",
        "trainingSchedule": "identical pre-generated map/start sequence per seed for both models",
        "randomBaseline": random_result,
        "bfsOracle": oracle_result,
        "models": {
            "mlp-212": {
                "architecture": "8-16-4 ReLU MLP",
                **mlp,
            },
            "recurrent-thought-202": {
                "architecture": "8-9 recurrent hidden units, 2 shared-weight thought cycles, 4 outputs",
                **recurrent,
            },
        },
        "limitations": [
            "This is a small navigation control benchmark, not a language-model evaluation.",
            "The recurrent model repeats hidden-state computation within one observation; it does not preserve memory across environment steps.",
            "Only a few seeds and one task distribution are tested; results do not establish broad superiority.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ARCH_COMPARISON_SUMMARY " + json.dumps({
        "mlp212": mlp["aggregate"],
        "recurrent202": recurrent["aggregate"],
    }, sort_keys=True), flush=True)
    print(f"SUMMARY_PATH={args.out / 'summary.json'}", flush=True)
    print("PASS: architecture comparison completed; compare held-out success rates and runtime", flush=True)


if __name__ == "__main__":
    main()
