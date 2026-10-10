#!/usr/bin/env python3
"""Held-out-map generalization test for the 212-parameter AIMENG Q-network."""
import argparse
import importlib.util
import json
import random
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "scripts" / "test_neural_gridworld.py"
SPEC = importlib.util.spec_from_file_location("aimeng_gridworld", TRAINER_PATH)
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)

SIZE = 5
GOAL = (4, 4)
ACTIONS = TRAINER.ACTIONS
ACTION_NAMES = TRAINER.ACTION_NAMES
MAX_STEPS = 40
WALL_PROBABILITY = 0.18
MIN_REACHABLE = 18


def reachable_cells(walls):
    queue = [GOAL]
    seen = {GOAL}
    for pos in queue:
        for dx, dy in ACTIONS:
            nxt = (pos[0] + dx, pos[1] + dy)
            if not (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE):
                continue
            if nxt in walls or nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return sorted(seen, key=lambda p: (p[1], p[0]))


def make_map(rng):
    cells = [(x, y) for y in range(SIZE) for x in range(SIZE) if (x, y) != GOAL]
    for _ in range(1000):
        walls = {p for p in cells if rng.random() < WALL_PROBABILITY}
        reachable = reachable_cells(walls)
        if len(reachable) >= MIN_REACHABLE:
            return {"walls": walls, "reachable": reachable}
    raise RuntimeError("could not generate a sufficiently connected map")


def observation(pos, walls):
    x, y = pos
    gx, gy = GOAL
    blocked = []
    for dx, dy in ACTIONS:
        nxt = (x + dx, y + dy)
        blocked.append(1.0 if not (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE)
                       or nxt in walls else 0.0)
    return [x / (SIZE - 1), y / (SIZE - 1), gx / (SIZE - 1), gy / (SIZE - 1)] + blocked


def step(pos, action, walls):
    dx, dy = ACTIONS[action]
    nxt = (pos[0] + dx, pos[1] + dy)
    if not (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE) or nxt in walls:
        return pos, -0.12, False
    if nxt == GOAL:
        return nxt, 1.0, True
    return nxt, -0.025, False


def shortest_path(start, walls):
    queue = [(start, 0)]
    seen = {start}
    for pos, distance in queue:
        if pos == GOAL:
            return distance
        for dx, dy in ACTIONS:
            nxt = (pos[0] + dx, pos[1] + dy)
            if not (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE):
                continue
            if nxt in walls or nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, distance + 1))
    return None


def build_test_cases(seed, map_count=100, starts_per_map=5):
    rng = random.Random(seed)
    cases = []
    for _ in range(map_count):
        world = make_map(rng)
        choices = [p for p in world["reachable"] if p != GOAL]
        selected = rng.sample(choices, min(starts_per_map, len(choices)))
        for start in selected:
            cases.append((world["walls"], start))
    return cases


def evaluate(net, cases):
    successes, successful_steps, optimality = 0, [], []
    for walls, start in cases:
        pos = start
        visited = set()
        for step_no in range(1, MAX_STEPS + 1):
            q, _, _ = net.forward(observation(pos, walls))
            action = max(range(4), key=lambda i: q[i])
            pos, _, done = step(pos, action, walls)
            if done:
                successes += 1
                successful_steps.append(step_no)
                optimal = shortest_path(start, walls)
                if optimal is not None:
                    optimality.append(optimal / step_no)
                break
            marker = (pos, action)
            if marker in visited:
                break
            visited.add(marker)
    return {
        "episodes": len(cases),
        "successes": successes,
        "success_rate": successes / max(1, len(cases)),
        "mean_steps_success": sum(successful_steps) / max(1, len(successful_steps)),
        "mean_optimality_ratio": sum(optimality) / max(1, len(optimality)),
    }


def random_baseline(cases, seed):
    rng = random.Random(seed)
    successes, steps_all = 0, []
    for walls, start in cases:
        pos = start
        for step_no in range(1, MAX_STEPS + 1):
            pos, _, done = step(pos, rng.randrange(4), walls)
            if done:
                successes += 1
                steps_all.append(step_no)
                break
    return {
        "episodes": len(cases),
        "successes": successes,
        "success_rate": successes / max(1, len(cases)),
        "mean_steps_success": sum(steps_all) / max(1, len(steps_all)),
    }


def oracle_baseline(cases):
    distances = [shortest_path(start, walls) for walls, start in cases]
    reachable = [d for d in distances if d is not None]
    return {
        "episodes": len(cases),
        "successes": len(reachable),
        "success_rate": len(reachable) / max(1, len(cases)),
        "mean_steps_success": sum(reachable) / max(1, len(reachable)),
        "note": "BFS oracle upper bound; not a learned model",
    }


def train_one(episodes, seed, test_cases):
    rng = random.Random(seed)
    net = TRAINER.TinyQNetwork(seed)
    initial = evaluate(net, test_cases)
    gamma = 0.92
    for episode in range(1, episodes + 1):
        world = make_map(rng)
        choices = [p for p in world["reachable"] if p != GOAL]
        pos = rng.choice(choices)
        walls = world["walls"]
        epsilon = max(0.05, 1.0 - 0.95 * episode / episodes)
        for _ in range(MAX_STEPS):
            state = observation(pos, walls)
            action = net.choose(state, epsilon, rng)
            nxt, reward, done = step(pos, action, walls)
            next_state = observation(nxt, walls)
            next_q, _, _ = net.forward(next_state)
            target = reward if done else reward + gamma * max(next_q)
            net.update(state, action, target)
            pos = nxt
            if done:
                break
    final = evaluate(net, test_cases)
    return net, initial, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "gridworld-generalization")
    args = parser.parse_args()
    if not 100 <= args.episodes <= 20000:
        parser.error("--episodes must be in [100, 20000]")
    if not 1 <= args.seeds <= 10:
        parser.error("--seeds must be in [1, 10]")

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    test_cases = build_test_cases(9302026, map_count=100, starts_per_map=5)
    random_result = random_baseline(test_cases, 9302027)
    oracle_result = oracle_baseline(test_cases)
    print("AIMENG_HELDOUT_MAP_GENERALIZATION/v1", flush=True)
    print(f"episodes_per_seed={args.episodes} seeds={args.seeds} heldout_maps=100 cases={len(test_cases)}", flush=True)
    print("train_maps=procedurally_generated test_maps=disjoint_fixed_seed test_starts=5_per_map", flush=True)
    print("parameter_count=212 network=8-16-4 activation=ReLU", flush=True)
    print("random_baseline=" + json.dumps(random_result, sort_keys=True), flush=True)
    print("bfs_oracle=" + json.dumps(oracle_result, sort_keys=True), flush=True)

    results = []
    for index in range(args.seeds):
        seed = 20262001 + index
        t0 = time.time()
        net, initial, final = train_one(args.episodes, seed, test_cases)
        item = {
            "seed": seed,
            "episodesTrained": args.episodes,
            "initialEvaluation": initial,
            "finalEvaluation": final,
            "successDelta": final["success_rate"] - initial["success_rate"],
            "seconds": round(time.time() - t0, 3),
        }
        results.append(item)
        seed_dir = args.out / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "training_report.json").write_text(
            json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (seed_dir / "q_network.json").write_text(
            json.dumps(net.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("SEED_RESULT " + json.dumps(item, sort_keys=True), flush=True)

    initial_rates = [r["initialEvaluation"]["success_rate"] for r in results]
    final_rates = [r["finalEvaluation"]["success_rate"] for r in results]
    summary = {
        "format": "aimeng-gridworld-heldout-map-report/v1",
        "parameterCount": 212,
        "algorithm": "8-16-4 ReLU MLP neural Q-learning, epsilon-greedy, TD(0)",
        "episodesPerSeed": args.episodes,
        "seedCount": args.seeds,
        "trainingDistribution": "random connected 5x5 maps with 18% candidate wall probability",
        "evaluation": "100 fixed-seed held-out maps, 5 starts per map, exploration disabled",
        "randomBaseline": random_result,
        "bfsOracle": oracle_result,
        "perSeed": results,
        "aggregate": {
            "initialSuccessRateMean": statistics.mean(initial_rates),
            "finalSuccessRateMean": statistics.mean(final_rates),
            "finalSuccessRateStdev": statistics.pstdev(final_rates),
            "seedsImproved": sum(f > i for f, i in zip(final_rates, initial_rates)),
            "seedCount": args.seeds,
            "finalSuccessRateMin": min(final_rates),
            "finalSuccessRateMax": max(final_rates),
        },
        "elapsedSeconds": round(time.time() - started, 3),
        "limitations": [
            "Only a 5x5 local-observation navigation task; not a language-model benchmark.",
            "The held-out maps are fixed for reproducibility; training maps are freshly generated.",
            "A few seeds are an engineering smoke test, not a statistical proof of superiority.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUITE_SUMMARY " + json.dumps(summary["aggregate"], sort_keys=True), flush=True)
    print(f"SUMMARY_PATH={args.out / 'summary.json'}", flush=True)
    print(f"TOTAL_SECONDS={summary['elapsedSeconds']}", flush=True)

    # A failure still leaves the report/checkpoints available for diagnosis.
    if summary["aggregate"]["seedsImproved"] < max(1, args.seeds - 1):
        raise SystemExit("FAIL: too few seeds improved held-out-map success rate")
    if summary["aggregate"]["finalSuccessRateMean"] <= random_result["success_rate"]:
        raise SystemExit("FAIL: mean held-out success rate did not beat random baseline")
    print("PASS: held-out-map benchmark improved over initialization and random baseline", flush=True)


if __name__ == "__main__":
    main()
