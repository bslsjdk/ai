#!/usr/bin/env python3
"""Repeatable multi-seed terminal benchmark for the tiny AIMENG grid-world Q-network.

This is an algorithm smoke test, not evidence of generalization: all seeds use the
same 5x5 map and the deterministic evaluation starts on that map.
"""
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "scripts" / "test_neural_gridworld.py"
SPEC = importlib.util.spec_from_file_location("aimeng_gridworld", TRAINER_PATH)
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)

DEFAULT_SEEDS = [20261009, 20261010, 20261011, 20261012, 20261013]


def random_baseline(seed, starts, max_steps=40):
    rng = random.Random(seed)
    successes = []
    for start in starts:
        pos = start
        for step_no in range(1, max_steps + 1):
            pos, _, done = TRAINER.step(pos, rng.randrange(4))
            if done:
                successes.append(step_no)
                break
    return {
        "episodes": len(starts),
        "successes": len(successes),
        "success_rate": len(successes) / max(1, len(starts)),
        "mean_steps_success": sum(successes) / max(1, len(successes)),
    }


def oracle_baseline(starts):
    distances = [TRAINER.shortest_path(start) for start in starts]
    reachable = [d for d in distances if d is not None]
    return {
        "episodes": len(starts),
        "successes": len(reachable),
        "success_rate": len(reachable) / max(1, len(starts)),
        "mean_steps_success": sum(reachable) / max(1, len(reachable)),
        "note": "BFS oracle upper bound; not a learned model",
    }


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "build" / "gridworld-suite"
    seeds = DEFAULT_SEEDS
    out_dir.mkdir(parents=True, exist_ok=True)
    starts = TRAINER.valid_starts()
    random_result = random_baseline(20261009, starts)
    oracle_result = oracle_baseline(starts)
    results = []
    started = time.time()

    print("AIMENG_GRIDWORLD_SUITE/v1", flush=True)
    print(f"episodes_per_seed={episodes} seeds={seeds} map=5x5 starts={len(starts)}", flush=True)
    print(f"parameter_count=212 network=8-16-4 activation=ReLU algorithm=epsilon-greedy+TD(0)", flush=True)
    print("evaluation=deterministic, exploration_off, same_map_all_valid_start_cells", flush=True)
    print("random_baseline=" + json.dumps(random_result, sort_keys=True), flush=True)
    print("bfs_oracle=" + json.dumps(oracle_result, sort_keys=True), flush=True)

    for seed in seeds:
        t0 = time.time()
        net, report = TRAINER.train(episodes=episodes, seed=seed)
        report["benchmarkSeed"] = seed
        report["parameterCount"] = 212
        report["evaluationScope"] = "same fixed map; all valid start cells; not held-out-map generalization"
        report["randomBaseline"] = random_result
        report["bfsOracle"] = oracle_result
        results.append(report)
        seed_dir = out_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "training_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (seed_dir / "q_network.json").write_text(
            json.dumps(net.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("SEED_RESULT " + json.dumps({
            "seed": seed,
            "seconds": round(time.time() - t0, 3),
            "initial": report["initialEvaluation"],
            "final": report["finalEvaluation"],
            "success_delta": round(report["finalEvaluation"]["success_rate"] -
                                   report["initialEvaluation"]["success_rate"], 6),
        }, sort_keys=True), flush=True)

    final_rates = [r["finalEvaluation"]["success_rate"] for r in results]
    initial_rates = [r["initialEvaluation"]["success_rate"] for r in results]
    summary = {
        "format": "aimeng-gridworld-suite-report/v1",
        "algorithm": "8-16-4 ReLU MLP neural Q-learning, epsilon-greedy, TD(0)",
        "parameterCount": 212,
        "episodesPerSeed": episodes,
        "seeds": seeds,
        "mapSize": [TRAINER.SIZE, TRAINER.SIZE],
        "validStartCount": len(starts),
        "randomBaseline": random_result,
        "bfsOracle": oracle_result,
        "perSeed": [{
            "seed": r["benchmarkSeed"],
            "initial": r["initialEvaluation"],
            "final": r["finalEvaluation"],
            "successDelta": r["finalEvaluation"]["success_rate"] -
                            r["initialEvaluation"]["success_rate"],
        } for r in results],
        "aggregate": {
            "initialSuccessRateMean": statistics.mean(initial_rates),
            "finalSuccessRateMean": statistics.mean(final_rates),
            "finalSuccessRateStdev": statistics.pstdev(final_rates),
            "seedsImproved": sum(f > i for f, i in zip(final_rates, initial_rates)),
            "seedCount": len(seeds),
            "finalSuccessRateMin": min(final_rates),
            "finalSuccessRateMax": max(final_rates),
        },
        "elapsedSeconds": round(time.time() - started, 3),
        "limitations": [
            "All evaluation uses the same fixed map used for training; this is not generalization evidence.",
            "A five-seed result is a smoke benchmark, not a statistically conclusive comparison.",
            "Compare other methods only with identical map, observation, reward, action budget, and evaluation starts.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUITE_SUMMARY " + json.dumps(summary["aggregate"], sort_keys=True), flush=True)
    print(f"SUMMARY_PATH={out_dir / 'summary.json'}", flush=True)
    print(f"TOTAL_SECONDS={summary['elapsedSeconds']}", flush=True)

    # The suite always preserves reports; CI is red if learning fails to improve on
    # at least 4/5 seeds or the mean final rate is below 50%.
    if summary["aggregate"]["seedsImproved"] < 4:
        raise SystemExit("FAIL: fewer than 4/5 seeds improved over their initial policy")
    if summary["aggregate"]["finalSuccessRateMean"] < 0.50:
        raise SystemExit("FAIL: mean final success rate below 50% smoke threshold")
    print("PASS: multi-seed learning smoke benchmark met thresholds", flush=True)


if __name__ == "__main__":
    main()
