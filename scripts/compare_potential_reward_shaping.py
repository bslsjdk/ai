#!/usr/bin/env python3
"""Test potential-based reward shaping without changing model parameters."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import test_gridworld_generalization as env
import test_neural_gridworld as base


def make_schedule(seed, episodes):
    rng = random.Random(950000 + seed)
    schedule = []
    for _ in range(episodes):
        world = env.make_map(rng)
        start = rng.choice([p for p in world["reachable"] if p != env.GOAL])
        schedule.append((world["walls"], start))
    return schedule


def potential(pos):
    # Observable coordinates only. Goal potential is zero.
    return -(abs(pos[0] - env.GOAL[0]) + abs(pos[1] - env.GOAL[1])) / (2.0 * (env.SIZE - 1))


def train(net, seed, episodes, schedule, shaping):
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
            if shaping:
                reward += gamma * potential(nxt) - potential(pos)
            next_state = env.observation(nxt, walls)
            next_q, _, _ = net.forward(next_state)
            target = reward if done else reward + gamma * max(next_q)
            net.update(state, action, target)
            pos = nxt
            if done:
                break
    return time.time() - started


def run_method(shaping, seeds, episodes, test_cases, out_dir):
    records = []
    for seed in seeds:
        schedule = make_schedule(seed, episodes)
        net = base.TinyQNetwork(seed)
        initial = env.evaluate(net, test_cases)
        seconds = train(net, seed, episodes, schedule, shaping)
        final = env.evaluate(net, test_cases)
        record = {
            "seed": seed,
            "episodesTrained": episodes,
            "parameterCount": 212,
            "rewardShaping": "potential-based Manhattan shaping" if shaping else "none",
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
        print("SHAPING_SEED_RESULT " + json.dumps(record, sort_keys=True), flush=True)
    initial_rates = [r["initialEvaluation"]["success_rate"] for r in records]
    final_rates = [r["finalEvaluation"]["success_rate"] for r in records]
    return {
        "parameterCount": 212,
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
    parser.add_argument("--out", type=Path, default=Path("build/gridworld-reward-shaping"))
    args = parser.parse_args()
    if not 100 <= args.episodes <= 20000:
        parser.error("--episodes must be in [100, 20000]")
    if not 1 <= args.seeds <= 10:
        parser.error("--seeds must be in [1, 10]")

    args.out.mkdir(parents=True, exist_ok=True)
    test_cases = env.build_test_cases(9302026, map_count=100, starts_per_map=5)
    random_result = env.random_baseline(test_cases, 9302027)
    oracle_result = env.oracle_baseline(test_cases)
    seeds = [20265001 + i for i in range(args.seeds)]
    print("AIMENG_POTENTIAL_BASED_REWARD_SHAPING/v1", flush=True)
    print(f"episodes_per_seed={args.episodes} seeds={seeds} heldout_cases={len(test_cases)}", flush=True)
    print("fairness=same parameter count, map/start schedule, epsilon schedule, test cases", flush=True)
    print("shaping=gamma*Phi(next)-Phi(current), Phi=-normalized Manhattan distance", flush=True)
    print("random_baseline=" + json.dumps(random_result, sort_keys=True), flush=True)
    print("bfs_oracle=" + json.dumps(oracle_result, sort_keys=True), flush=True)

    plain = run_method(False, seeds, args.episodes, test_cases, args.out / "plain-td")
    shaped = run_method(True, seeds, args.episodes, test_cases, args.out / "potential-shaped-td")
    summary = {
        "format": "aimeng-potential-shaping-report/v1",
        "episodesPerSeed": args.episodes,
        "seeds": seeds,
        "evaluation": "100 fixed-seed held-out maps, 5 starts per map, exploration disabled",
        "trainingSchedule": "identical pre-generated map/start sequence per seed for both methods",
        "randomBaseline": random_result,
        "bfsOracle": oracle_result,
        "methods": {
            "plain-td": {"description": "212-parameter MLP, original reward", **plain},
            "potential-shaped-td": {
                "description": "same MLP and TD(0), with potential-based Manhattan reward shaping",
                **shaped,
            },
        },
        "limitations": [
            "Only a small navigation control task, not a language-model benchmark.",
            "Manhattan potential is observable but imperfect around obstacles; evaluate on held-out maps.",
            "Three seeds are exploratory evidence, not statistical proof.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SHAPING_COMPARISON_SUMMARY " + json.dumps({
        "plainTD": plain["aggregate"],
        "potentialShapedTD": shaped["aggregate"],
    }, sort_keys=True), flush=True)
    print(f"SUMMARY_PATH={args.out / 'summary.json'}", flush=True)
    print("PASS: reward-shaping comparison completed", flush=True)


if __name__ == "__main__":
    main()
