#!/usr/bin/env python3
"""Test whether gradient-guided rewiring improves a sparse neuron LM.

Compares fixed sparse edges, dynamic sparse edges (prune-and-regrow), and a
parameter-count-nearby dense network on the same synthetic corpus. This is a
small architecture experiment, not a general-purpose LLM benchmark.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_script(name, filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dynamic_train(model, x_train, y_train, x_test, y_test, args, seed):
    rng = np.random.default_rng(seed + 991)
    history = []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(y_train))
        gx = np.zeros_like(model.wx)
        gy = np.zeros_like(model.wy)
        batches = 0
        for begin in range(0, len(order), args.batch_size):
            ix = order[begin:begin + args.batch_size]
            x, y = x_train[ix], y_train[ix]
            probs, h = model.forward(x)
            n = len(y)
            dlogits = probs.copy()
            dlogits[np.arange(n), y] -= 1
            dlogits /= n
            wy_eff = model.wy * model.my
            grad_y_full = h.T @ dlogits
            dbo = dlogits.sum(axis=0)
            dh = dlogits @ wy_eff.T
            dz = dh * (1 - h * h)
            grad_x_full = x.T @ dz
            # Only active edges are optimized; inactive-edge gradients guide regrowth.
            model.wx -= args.learning_rate * grad_x_full * model.mx
            model.wx *= model.mx
            model.bh -= args.learning_rate * dz.sum(axis=0)
            model.wy -= args.learning_rate * grad_y_full * model.my
            model.wy *= model.my
            model.bo -= args.learning_rate * dbo
            gx += np.abs(grad_x_full)
            gy += np.abs(grad_y_full)
            batches += 1

        if epoch >= args.rewire_start and epoch % args.rewire_every == 0:
            rewire(model, gx / max(1, batches), gy / max(1, batches),
                   args.rewire_fraction, args.learning_rate, rng)
        loss, acc = model.evaluate(x_test, y_test)
        history.append({"epoch": epoch, "heldout_loss": loss, "heldout_accuracy": acc})
        print("DYNAMIC_SPARSE_EPOCH " + json.dumps({
            "seed": seed, "epoch": epoch, "heldout_loss": loss,
            "heldout_accuracy": acc,
            "active_input_edges": int(model.mx.sum()),
            "active_output_edges": int(model.my.sum()),
        }, sort_keys=True), flush=True)
    loss, acc = model.evaluate(x_test, y_test)
    return {"kind": "dynamic_sparse", "seed": seed, "hidden_units": model.hidden,
            "parameter_count": model.parameter_count, "heldout_loss": loss,
            "heldout_accuracy": acc, "history": history}


def rewire(model, grad_x, grad_y, fraction, learning_rate, rng):
    """Per unit, replace low-magnitude active edges with high-gradient inactive edges."""
    for j in range(model.hidden):
        active = np.flatnonzero(model.mx[:, j])
        inactive = np.flatnonzero(model.mx[:, j] == 0)
        count = min(max(1, int(round(len(active) * fraction))), len(active), len(inactive))
        if count:
            drop = active[np.argsort(np.abs(model.wx[active, j]))[:count]]
            grow = inactive[np.argsort(grad_x[inactive, j])[-count:]]
            model.mx[drop, j] = 0
            model.wx[drop, j] = 0
            model.mx[grow, j] = 1
            signs = rng.choice(np.array([-1.0, 1.0]), size=count)
            model.wx[grow, j] = signs * max(1e-3, learning_rate * float(np.mean(grad_x[grow, j])))
        active_out = np.flatnonzero(model.my[j])
        inactive_out = np.flatnonzero(model.my[j] == 0)
        count_out = min(max(1, int(round(len(active_out) * fraction))),
                        len(active_out), len(inactive_out))
        if count_out:
            drop = active_out[np.argsort(np.abs(model.wy[j, active_out]))[:count_out]]
            grow = inactive_out[np.argsort(grad_y[j, inactive_out])[-count_out:]]
            model.my[j, drop] = 0
            model.wy[j, drop] = 0
            model.my[j, grow] = 1
            signs = rng.choice(np.array([-1.0, 1.0]), size=count_out)
            model.wy[j, grow] = signs * max(1e-3, learning_rate * float(np.mean(grad_y[j, grow])))


def mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[20261010, 20261011, 20261012])
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.025)
    p.add_argument("--input-edges", type=int, default=16)
    p.add_argument("--output-edges", type=int, default=12)
    p.add_argument("--rewire-fraction", type=float, default=0.125)
    p.add_argument("--rewire-start", type=int, default=2)
    p.add_argument("--rewire-every", type=int, default=1)
    args = p.parse_args()
    if not 1 <= args.epochs <= 100 or not 0 < args.rewire_fraction <= 0.5:
        raise SystemExit("epochs must be 1..100 and rewire-fraction in (0, 0.5]")
    base = load_script("base_lm", "train_composed_neuron_lm.py")
    compare = load_script("compare_lm", "compare_sparse_dense_neuron_lm.py")
    train_text, test_text = base.build_corpus()
    vocab = {ch: i for i, ch in enumerate(sorted(set(train_text + test_text)))}
    x_train, y_train = base.encode_corpus(train_text, vocab, 8)
    x_test, y_test = base.encode_corpus(test_text, vocab, 8)
    input_dim, vocab_size = x_train.shape[1], len(vocab)
    sparse_width = 48
    sparse_params = args.input_edges * sparse_width + args.output_edges * sparse_width + sparse_width + vocab_size
    dense_width = max(1, round((sparse_params - vocab_size) / (input_dim + vocab_size + 1)))
    runs = []
    for seed in args.seeds:
        # Fixed sparse baseline from the preceding experiment.
        fixed = compare.train_one("sparse", x_train, y_train, x_test, y_test,
                                  sparse_width, seed, args)
        runs.append(fixed)
        # Dense baseline uses the nearest integer width for parameter matching.
        dense = compare.train_one("dense_parameter_matched", x_train, y_train, x_test, y_test,
                                  dense_width, seed, args)
        runs.append(dense)
        # Dynamic topology starts with the same sparse edge budget and rewires during training.
        model = compare.MaskedMLP(input_dim, vocab_size, sparse_width, seed, True,
                                  args.input_edges, args.output_edges)
        dynamic = dynamic_train(model, x_train, y_train, x_test, y_test, args, seed)
        dynamic["target_sparse_parameter_count"] = sparse_params
        dynamic["parameter_difference_vs_sparse_pct"] = round(
            100.0 * (dynamic["parameter_count"] - sparse_params) / sparse_params, 3)
        runs.append(dynamic)
        for row in (fixed, dense, dynamic):
            print("DYNAMIC_SPARSE_RESULT " + json.dumps({
                k: v for k, v in row.items() if k != "history"
            }, sort_keys=True), flush=True)

    aggregate = {}
    for kind in ("sparse", "dense_parameter_matched", "dynamic_sparse"):
        rows = [r for r in runs if r["kind"] == kind]
        aggregate[kind] = {
            "runs": len(rows),
            "mean_parameters": mean(r["parameter_count"] for r in rows),
            "mean_heldout_loss": mean(r["heldout_loss"] for r in rows),
            "mean_heldout_accuracy": mean(r["heldout_accuracy"] for r in rows),
        }
    report = {
        "format": "aimeng-dynamic-sparse-neuron-lm/v1",
        "dataset": "synthetic arithmetic-format character-level corpus",
        "context_characters": 8, "vocabulary_size": vocab_size,
        "settings": {"epochs": args.epochs, "batch_size": args.batch_size,
                     "learning_rate": args.learning_rate, "seeds": args.seeds,
                     "input_edges_per_unit": args.input_edges,
                     "output_edges_per_unit": args.output_edges,
                     "rewire_fraction": args.rewire_fraction,
                     "rewire_start": args.rewire_start, "rewire_every": args.rewire_every},
        "parameter_matching": {"sparse_parameters": sparse_params,
                               "dense_hidden_units": dense_width},
        "aggregate": aggregate, "runs": runs,
        "limitations": [
            "Synthetic character-level task only; not a general-purpose language-model benchmark.",
            "Topology is rewired by a simple gradient heuristic, not learned by a differentiable controller.",
            "Dense width is integer-rounded; compare actual parameter counts.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("DYNAMIC_SPARSE_AGGREGATE " + json.dumps(aggregate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
