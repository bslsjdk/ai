#!/usr/bin/env python3
"""Validation-only sensitivity sweep for interpolated character n-gram smoothing."""
import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
import run_real_text_baselines as base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", type=Path, required=True)
    ap.add_argument("--output-file", type=Path, required=True)
    ap.add_argument("--expected-dataset-sha256", default="e3917589f48b4e9f4b731a6b7e127cf9ddd6e1b3819c78be02a2702ee73bfdb1")
    args = ap.parse_args()

    text = args.text_file.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    train_text, val_text, test_text = base.split_text(text)
    vocab_chars = sorted(set(train_text))
    if "\ufffd" not in vocab_chars:
        vocab_chars.append("\ufffd")
    vocab = {ch: i for i, ch in enumerate(vocab_chars)}
    all_ids = base.encode(text, vocab)
    train_end, val_end = len(train_text), len(train_text) + len(val_text)
    train_ids = all_ids[:train_end]
    val_positions, val_context_starts = base.chunked_evaluation_positions(train_end, len(val_text), 64)
    test_positions, test_context_starts = base.chunked_evaluation_positions(val_end, len(test_text), 64)
    data_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if data_hash != args.expected_dataset_sha256:
        raise SystemExit(f"dataset hash mismatch: expected {args.expected_dataset_sha256}, got {data_hash}")
    if len(test_positions) != 14400:
        raise SystemExit(f"unexpected test target count: {len(test_positions)}")

    # Include wider values because the previous grid's best alpha for orders 3-5
    # sat at the upper boundary. Selection remains validation-only.
    alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    started = time.perf_counter()
    results = base.ngram_baselines(
        train_ids, all_ids, val_positions, val_context_starts,
        test_positions, test_context_starts, len(vocab),
        orders=(2, 3, 4, 5), alphas=alpha_grid
    )
    report = {
        "format": "aimeng-ngram-alpha-sensitivity/v1",
        "dataset": "Project Gutenberg Alice's Adventures in Wonderland, chronological split",
        "dataset_sha256": data_hash,
        "split": {
            "train_characters": len(train_text),
            "validation_characters": len(val_text),
            "test_characters": len(test_text),
            "validation_characters_scored": len(val_positions),
            "test_characters_scored": len(test_positions),
            "vocabulary_size": len(vocab),
            "target_alignment": "same 64-character chunk boundaries as recurrent RNN; context resets at each boundary"
        },
        "smoothing": {
            "method": "recursive Dirichlet interpolation; unigram base uses additive smoothing",
            "alpha_grid": alpha_grid,
            "selection": "minimum validation loss per order; test set used only once after selection"
        },
        "wall_seconds": round(time.perf_counter() - started, 3),
        "orders": results,
        "limitations": [
            "Single-book character prediction only.",
            "This is a smoothing-parameter sensitivity sweep, not a new neural-model experiment.",
            "Three or fewer n-gram orders may select an edge alpha; report the selected alpha and validation loss rather than assuming the grid optimum is a global optimum."
        ]
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("NGRAM_ALPHA_SENSITIVITY " + json.dumps({
        "dataset_sha256": data_hash,
        "alpha_grid": alpha_grid,
        "orders": results,
        "wall_seconds": report["wall_seconds"]
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
