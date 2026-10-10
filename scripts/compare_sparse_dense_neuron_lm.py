#!/usr/bin/env python3
"""Parameter-matched dense-vs-sparse neuron-unit language-model ablation.

Both models are trained from random initialization on the same synthetic corpus,
with the same optimizer, minibatch order, epochs, and evaluation split.
The sparse model has individually addressable hidden units with fixed sparse
input/output edges. This tests sparse connectivity, not dynamic topology.
"""
import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np


def load_baseline_module():
    path = Path(__file__).with_name("train_composed_neuron_lm.py")
    spec = importlib.util.spec_from_file_location("baseline_lm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MaskedMLP:
    def __init__(self, input_dim, vocab_size, hidden, seed, sparse=False,
                 input_edges=16, output_edges=12):
        rng = np.random.default_rng(seed)
        self.input_dim, self.vocab_size, self.hidden = input_dim, vocab_size, hidden
        self.sparse = sparse
        self.wx = (rng.standard_normal((input_dim, hidden)) *
                   math.sqrt(2.0 / (input_dim + hidden))).astype(np.float32)
        self.bh = np.zeros(hidden, dtype=np.float32)
        self.wy = (rng.standard_normal((hidden, vocab_size)) /
                   math.sqrt(hidden)).astype(np.float32)
        self.bo = np.zeros(vocab_size, dtype=np.float32)
        self.mx = np.ones_like(self.wx)
        self.my = np.ones_like(self.wy)
        if sparse:
            self.mx.fill(0)
            self.my.fill(0)
            for j in range(hidden):
                ix = rng.choice(input_dim, size=min(input_edges, input_dim), replace=False)
                oy = rng.choice(vocab_size, size=min(output_edges, vocab_size), replace=False)
                self.mx[ix, j] = 1
                self.my[j, oy] = 1
        self.wx *= self.mx
        self.wy *= self.my

    @property
    def parameter_count(self):
        return int(self.mx.sum() + self.my.sum() + self.hidden + self.vocab_size)

    def forward(self, x):
        h = np.tanh(x @ (self.wx * self.mx) + self.bh)
        logits = h @ (self.wy * self.my) + self.bo
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs, h

    def train_batch(self, x, y, lr):
        probs, h = self.forward(x)
        n = len(y)
        loss = -np.log(np.clip(probs[np.arange(n), y], 1e-9, 1)).mean()
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1
        dlogits /= n
        wy_eff = self.wy * self.my
        dwy = (h.T @ dlogits) * self.my
        dbo = dlogits.sum(axis=0)
        dh = dlogits @ wy_eff.T
        dz = dh * (1 - h * h)
        dwx = (x.T @ dz) * self.mx
        dbh = dz.sum(axis=0)
        self.wx -= lr * dwx
        self.wx *= self.mx
        self.bh -= lr * dbh
        self.wy -= lr * dwy
        self.wy *= self.my
        self.bo -= lr * dbo
        return float(loss)

    def evaluate(self, x, y):
        probs, _ = self.forward(x)
        loss = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1)).mean()
        return float(loss), float((probs.argmax(axis=1) == y).mean())


def train_one(kind, x_train, y_train, x_test, y_test, hidden, seed, args):
    sparse = kind == "sparse"
    model = MaskedMLP(x_train.shape[1], int(max(y_train.max(), y_test.max()) + 1),
                      hidden, seed, sparse, args.input_edges, args.output_edges)
    rng = np.random.default_rng(seed + 991)
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(y_train))
        for begin in range(0, len(order), args.batch_size):
            ix = order[begin:begin + args.batch_size]
            model.train_batch(x_train[ix], y_train[ix], args.learning_rate)
        loss, acc = model.evaluate(x_test, y_test)
        history.append({"epoch": epoch, "heldout_loss": loss, "heldout_accuracy": acc})
    elapsed = time.perf_counter() - started
    loss, acc = model.evaluate(x_test, y_test)
    return {
        "kind": kind, "seed": seed, "hidden_units": hidden,
        "parameter_count": model.parameter_count, "heldout_loss": loss,
        "heldout_accuracy": acc, "wall_seconds": round(elapsed, 4),
        "history": history,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[20261010, 20261011, 20261012])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.025)
    p.add_argument("--input-edges", type=int, default=16)
    p.add_argument("--output-edges", type=int, default=12)
    args = p.parse_args()
    if not 1 <= args.epochs <= 100:
        raise SystemExit("epochs must be in 1..100")
    baseline = load_baseline_module()
    train_text, test_text = baseline.build_corpus()
    vocab = {ch: i for i, ch in enumerate(sorted(set(train_text + test_text)))}
    x_train, y_train = baseline.encode_corpus(train_text, vocab, 8)
    x_test, y_test = baseline.encode_corpus(test_text, vocab, 8)
    input_dim, vocab_size = x_train.shape[1], len(vocab)
    sparse_width = 48
    sparse_params = min(args.input_edges, input_dim) * sparse_width + min(args.output_edges, vocab_size) * sparse_width + sparse_width + vocab_size
    dense_width = max(1, round((sparse_params - vocab_size) / (input_dim + vocab_size + 1)))
    results = []
    for seed in args.seeds:
        for kind, width in (("sparse", sparse_width), ("dense_parameter_matched", dense_width)):
            row = train_one(kind, x_train, y_train, x_test, y_test, width, seed, args)
            row["target_sparse_parameter_count"] = sparse_params
            row["parameter_difference_vs_sparse_pct"] = round(
                100.0 * (row["parameter_count"] - sparse_params) / sparse_params, 3)
            results.append(row)
            print("SPARSE_NEURON_ABLATION " + json.dumps({
                k: v for k, v in row.items() if k != "history"
            }, sort_keys=True), flush=True)
    by_kind = {}
    for kind in ("sparse", "dense_parameter_matched"):
        rows = [r for r in results if r["kind"] == kind]
        by_kind[kind] = {
            "runs": len(rows),
            "mean_parameters": statistics_mean(r["parameter_count"] for r in rows),
            "mean_heldout_loss": statistics_mean(r["heldout_loss"] for r in rows),
            "mean_heldout_accuracy": statistics_mean(r["heldout_accuracy"] for r in rows),
            "mean_wall_seconds": statistics_mean(r["wall_seconds"] for r in rows),
        }
    report = {
        "format": "aimeng-sparse-vs-dense-neuron-lm/v1",
        "dataset": "same synthetic arithmetic-format character-level corpus as baseline",
        "vocabulary_size": vocab_size, "context_characters": 8,
        "sparse_connections_per_hidden_unit": {
            "input_edges": min(args.input_edges, input_dim),
            "output_edges": min(args.output_edges, vocab_size),
        },
        "requested_sparse_hidden_units": sparse_width,
        "parameter_matched_dense_hidden_units": dense_width,
        "parameter_matching_note": "Dense width is rounded to nearest integer; report actual parameter difference.",
        "settings": {"epochs": args.epochs, "batch_size": args.batch_size,
                     "learning_rate": args.learning_rate, "seeds": args.seeds},
        "aggregate": by_kind, "runs": results,
        "limitations": [
            "Synthetic character-level corpus only; not a general-purpose language-model benchmark.",
            "Sparse edges are randomly initialized and fixed during training; this does not test learned routing or dynamic topology.",
            "Sparse-vs-dense differences may reflect connectivity constraints and width; parameter counts and multiple seeds are reported.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SPARSE_NEURON_ABLATION_AGGREGATE " + json.dumps(by_kind, sort_keys=True), flush=True)


def statistics_mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
