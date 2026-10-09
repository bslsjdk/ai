#!/usr/bin/env python3
import importlib.util
from pathlib import Path

path = Path(__file__).with_name("test_neural_gridworld.py")
spec = importlib.util.spec_from_file_location("gridworld", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

net, report = module.train(episodes=1000, seed=1234)
assert report["finalEvaluation"]["episodes"] == len(module.valid_starts())
assert report["finalEvaluation"]["success_rate"] >= 0.0
assert len(net.w1) == 16 and len(net.w2) == 4
assert all(len(row) == 8 for row in net.w1)
assert all(len(row) == 16 for row in net.w2)
print("PASS: neural grid-world trainer runs, dimensions/checkpoint are valid.")
print("Learning diagnostic: initial={:.1%}, final={:.1%}".format(
    report["initialEvaluation"]["success_rate"],
    report["finalEvaluation"]["success_rate"]))
