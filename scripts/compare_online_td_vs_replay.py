#!/usr/bin/env python3
"""Compare online TD(0) with replay + a slowly updated target network."""
import argparse
import copy
import json
import random
import statistics
import time
from pathlib import Path

import test_gridworld_generalization as env
import test_neural_gridworld as base


def make_schedule(seed, episodes):
    rng = random.Random(910000 + seed)
    schedule = []
    for _ in range(episodes):
        world = env.make_map(rng)
        start = rng.choice([p for p in world["reachable"] if p != env.GOAL])
        schedule.append((world["walls"], start))
    return schedule


def train_online(net, seed, episodes, schedule):
    rng = random.Random(424242 + seed)
    gamma = 0.92
    start_time = time.time()
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
    return time.time() - start_time


def train_replay(net, seed, episodes, schedule):
    action_rng = random.Random(424242 + seed)
    replay_rng = random.Random(131313 + seed)
    gamma = 0.92
    capacity, warmup, batch_size, update_interval, target_sync = 2048, 64, 8, 8, 200
    replay = []
    target_net = copy.deepcopy(net)
    total_steps = 0
    start_time = time.time()

    for episode, (walls, start) in enumerate(schedule, start=1):
        pos = start
        epsilon = max(0.05, 1.0 - 0.95 * episode / episodes)
        for _ in range(env.MAX_STEPS):
            state = env.observation(pos, walls)
            action = net.choose(state, epsilon, action_rng)
            nxt, reward, done = env.step(pos, action, walls)
            next_state = env.observation(nxt, walls)
            replay.append((state, action, reward, next_state, done))
            if len(replay) > capacity:
                replay.pop(0)
            total_steps += 1

            if len(replay) >= warmup and total_steps % update_interval == 0:
                batch = replay_rng.sample(replay, batch_size)
                for s, a, r, ns, terminal in batch:
                    next_q, _, _ = target_net.forward(ns)
                    target = r if terminal else r + gamma * max(next_q)
                    net.update(s, a, target)

            if total_steps % target_sync == 0:
                target_net = copy.deepcopy(net)
            pos = nxt
            if done:
                break
    return time.time() - start_time, {
        "replayCapacity": capacity,
        "warmupTransitions": warmup,
        "batchSize": batch_size,
        "updateEveryEnvironmentSteps": update_interval,
        "targetSyncEnvironmentSteps": target_sync,
    }


def run_method(method, seeds, episodes, test_cases, out_dir):
    records = []
    for seed in seeds:
        schedule = make_schedule(seed, episodes)
        net = base.TinyQNetwork(seed)
        initial = env.evaluate(net, test_cases)
        if method == "online-td":
            seconds = train_online(net, seed, episodes, schedule)
            settings = {"algorithm": "online TD(0)"}
        else:
            seconds, replay_settings = train_replay(net, seed, episodes, schedule)
            settings = {"algorithm": "experience replay + target network", **replay_settings}
        final = env.evaluate(net, test_cases)
        record = {
            "seed": seed,
            "episodesTrained": episodes,
            "parameterCount": 212,
            "initialEvaluation": initial,
            "finalEvaluation": final,
            "successDelta": final["success_rate"] - initial["success_rate"],
            "trainingSeconds": round(seconds, 3),
            **settings,
        }
        records.append(record)
        seed_dir = out_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "training_report.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (seed_dir / "q_network.json").write_text(
            json.dumps(net.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("METHOD_SEED_RESULT " + json.dumps(record, sort_keys=True), flush=True)

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
    parser.add_argument("--out", type=Path, default=Path("build/gridworld-replay-comparison"))
    args = parser.parse_args()
    if not 100 <= args.episodes <= 20000:
        parser.error("--episodes must be in [100, 20000]")
    if not 1 <= args.seeds <= 10:
        parser.error("--seeds must be in [1, 10]")

    args.out.mkdir(parents=True, exist_ok=True)
    test_cases = env.build_test_cases(9302026, map_count=100, starts_per_map=5)
    random_result = env.random_baseline(test_cases, 9302027)
    oracle_result = env.oracle_baseline(test_cases)
    seeds = [20264001 + i for i in range(args.seeds)]
    print("AIMENG_ONLINE_TD_VS_REPLAY_TARGET/v1", flush=True)
    print(f"episodes_per_seed={args.episodes} seeds={seeds} heldout_cases={len(test_cases)}", flush=True)
    print("fairness=same parameter count, observation, reward, map/start schedule, exploration schedule, test cases", flush=True)
    print("random_baseline=" + json.dumps(random_result, sort_keys=True), flush=True)
    print("bfs_oracle=" + json.dumps(oracle_result, sort_keys=True), flush=True)

    online = run_method("online-td", seeds, args.episodes, test_cases, args.out / "online-td")
    replay = run_method("replay-target", seeds, args.episodes, test_cases, args.out / "replay-target")
    summary = {
        "format": "aimeng-online-td-vs-replay-report/v1",
        "episodesPerSeed": args.episodes,
        "seeds": seeds,
        "evaluation": "100 fixed-seed held-out maps, 5 starts per map, exploration disabled",
        "trainingSchedule": "identical pre-generated map/start sequence per seed for both methods",
        "randomBaseline": random_result,
        "bfsOracle": oracle_result,
        "methods": {
            "online-td": {"description": "212-parameter 8-16-4 MLP; online TD(0)", **online},
            "replay-target": {"description": "same 212-parameter MLP; replay buffer and periodically copied target network", **replay},
        },
        "limitations": [
            "Only a small navigation task, not a language-model benchmark.",
            "Replay changes the number and distribution of gradient updates; training episode count is matched, not exact wall-clock or gradient-update count.",
            "Three seeds are exploratory evidence, not statistical proof.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("METHOD_COMPARISON_SUMMARY " + json.dumps({
        "onlineTD": online["aggregate"],
        "replayTarget": replay["aggregate"],
    }, sort_keys=True), flush=True)
    print(f"SUMMARY_PATH={args.out / 'summary.json'}", flush=True)
    print("PASS: learning-mechanism comparison completed", flush=True)


if __name__ == "__main__":
    main()
