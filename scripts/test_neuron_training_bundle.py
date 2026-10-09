#!/usr/bin/env python3
"""Validate the one-file AIMENG mobile-neuron training smoke bundle.

This checks the import contract and synthetic ground-truth data only. Actual
optimizer convergence is exercised by importing the bundle into the Android app.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "neuron_training_smoke_bundle.json"


def main() -> int:
    try:
        bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"FAIL: cannot parse bundle: {exc}")
        return 1

    errors = []
    if bundle.get("format") != "aimeng-training-bundle/v1":
        errors.append("wrong format marker")
    if bundle.get("inputs") != 2 or bundle.get("outputs") != 1:
        errors.append("expected exactly 2 inputs and 1 output")
    training = bundle.get("training")
    if not isinstance(training, dict):
        errors.append("missing training configuration")
    else:
        epochs = training.get("epochs")
        rate = training.get("learningRate")
        if not isinstance(epochs, int) or not 1 <= epochs <= 5000:
            errors.append("epochs must be in [1, 5000]")
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0.00001 <= rate <= 0.1:
            errors.append("learningRate must be finite and in [0.00001, 0.1]")

    rows = bundle.get("samples")
    if not isinstance(rows, list) or not 12 <= len(rows) <= 5000:
        errors.append("samples must contain between 12 and 5000 rows")
        rows = []

    max_error = 0.0
    for index, row in enumerate(rows):
        x, y = row.get("input"), row.get("output")
        if not isinstance(x, list) or len(x) != 2 or not isinstance(y, list) or len(y) != 1:
            errors.append(f"row {index}: wrong input/output dimensions")
            continue
        values = x + y
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            errors.append(f"row {index}: non-finite or non-numeric value")
            continue
        if any(abs(v) > 1.000001 for v in x):
            errors.append(f"row {index}: input is outside normalized [-1, 1]")
        expected = 0.7 * x[0] - 0.4 * x[1] + 0.2
        max_error = max(max_error, abs(y[0] - expected))

    if max_error > 0.00001:
        errors.append(f"ground-truth formula mismatch; max absolute error={max_error}")

    if errors:
        print("FAIL: training bundle validation")
        for error in errors:
            print(f" - {error}")
        return 1

    print("PASS: training bundle format, dimensions, parameter bounds, finite values,")
    print(f"      {len(rows)} samples and target formula verified (max error {max_error:.8f}).")
    print("NOTE: this script validates the import fixture; it does not run the Android optimizer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
