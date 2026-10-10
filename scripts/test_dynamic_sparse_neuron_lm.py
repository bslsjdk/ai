#!/usr/bin/env python3
"""Test whether gradient-guided rewiring improves a sparse neuron LM.

Compares fixed sparse edges, dynamic sparse edges (prune-and-regrow), and a
parameter-count-nearby dense network on the same synthetic corpus. This is a
small architecture experiment, not a general-purpose LLM benchmark.
"""
import argparse
import importlib.util
import json
import time
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
    started = time.perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(y_train))
        gx = np.zeros_like(model.wx)
        gy = np.zeros_like(model.wy)
        signed_gx = np.zeros_like(model.wx)
        signed_gy = np.zeros_like(model.wy)
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
            signed_gx += grad_x_full
            signed_gy += grad_y_full
            batches += 1

        if epoch >= args.rewire_start and epoch % args.rewire_every == 0:
            rewire(model, gx / max(1, batches), gy / max(1, batches),
                   signed_gx / max(1, batches), signed_gy / max(1, batches),
                   args.rewire_fraction, args.learning_rate)
        loss, acc = model.evaluate(x_test, y_test)
        history.append({"epoch": epoch, "heldout_loss": loss, "heldout_accuracy": acc})
        print("DYNAMIC_SPARSE_EPOCH " + json.dumps({
            "seed": seed, "epoch": epoch, "heldout_loss": loss,
            "heldout_accuracy": acc,
            "active_input_edges": int(model.mx.sum()),
            "active_output_edges": int(model.my.sum()),
        }, sort_keys=True), flush=True)
    loss, acc = model.evaluate(x_test, y_test)
    unit_details = []
    for j in range(model.hidden):
        original = model.my[j].copy()
        model.my[j, :] = 0
        loss_without, _ = model.evaluate(x_test, y_test)
        model.my[j, :] = original
        unit_details.append({
            "id": f"neuron-{j:03d}",
            "input_feature_indices": np.flatnonzero(model.mx[:, j]).astype(int).tolist(),
            "output_token_ids": np.flatnonzero(model.my[j]).astype(int).tolist(),
            "input_weight_l2": float(np.linalg.norm(model.wx[:, j])),
            "output_weight_l2": float(np.linalg.norm(model.wy[j])),
            "ablation_loss_delta": float(loss_without - loss),
        })
    unit_details.sort(key=lambda row: row["ablation_loss_delta"], reverse=True)
    return {"kind": "dynamic_sparse", "seed": seed, "hidden_units": model.hidden,
            "parameter_count": model.parameter_count, "heldout_loss": loss,
            "heldout_accuracy": acc, "wall_seconds": round(time.perf_counter() - started, 4),
            "unit_details": unit_details, "history": history}


def rewire(model, grad_x, grad_y, signed_x, signed_y, fraction, learning_rate):
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
            model.wx[grow, j] = -learning_rate * signed_x[grow, j]
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
            model.wy[j, grow] = -learning_rate * signed_y[j, grow]


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
    p.add_argument("--rewire-fractions", type=float, nargs="+", default=[0.125, 0.25, 0.375])
    p.add_argument("--rewire-start", type=int, default=2)
    p.add_argument("--rewire-every", type=int, default=1)
    args = p.parse_args()
    if not 1 <= args.epochs <= 100 or any(not 0 < f <= 0.5 for f in args.rewire_fractions):
        raise SystemExit("epochs must be 1..100 and every rewire fraction must be in (0, 0.5]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
        rows_to_print = [fixed, dense]
        # Sweep rewiring rates from the same initial sparse topology for each seed.
        for fraction in args.rewire_fractions:
            args.rewire_fraction = fraction
            model = compare.MaskedMLP(input_dim, vocab_size, sparse_width, seed, True,
                                      args.input_edges, args.output_edges)
            dynamic = dynamic_train(model, x_train, y_train, x_test, y_test, args, seed)
            tag = str(fraction).replace(".", "_")
            dynamic["kind"] = "dynamic_sparse_f" + tag
            dynamic["rewire_fraction"] = fraction
            checkpoint_rel = f"model-seed-{seed}-rewire-{tag}.json"
            checkpoint = {
                "format": "aimeng-dynamic-sparse-neuron-checkpoint/v1",
                "seed": seed, "rewire_fraction": fraction,
                "context_characters": 8, "vocabulary": [ch for ch, _ in sorted(vocab.items(), key=lambda item: item[1])],
                "parameter_count": model.parameter_count, "hidden_units": model.hidden,
                "units": [{
                    "id": f"neuron-{j:03d}", "enabled": True, "bias": float(model.bh[j]),
                    "input_edges": [{"feature_index": int(i), "weight": float(model.wx[i, j])}
                                    for i in np.flatnonzero(model.mx[:, j])],
                    "output_edges": [{"token_id": int(k), "weight": float(model.wy[j, k])}
                                     for k in np.flatnonzero(model.my[j])],
                } for j in range(model.hidden)],
                "output_bias": model.bo.astype(float).tolist(),
                "evaluation": {"heldout_loss": dynamic["heldout_loss"],
                               "heldout_accuracy": dynamic["heldout_accuracy"]},
            }
            (args.output_dir / checkpoint_rel).write_text(
                json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            dynamic["checkpoint_file"] = checkpoint_rel
            dynamic["target_sparse_parameter_count"] = sparse_params
            dynamic["parameter_difference_vs_sparse_pct"] = round(
                100.0 * (dynamic["parameter_count"] - sparse_params) / sparse_params, 3)
            runs.append(dynamic)
            rows_to_print.append(dynamic)
        for row in rows_to_print:
            print("DYNAMIC_SPARSE_RESULT " + json.dumps({
                k: v for k, v in row.items() if k not in ("history", "unit_details")
            }, sort_keys=True), flush=True)

    aggregate = {}
    for kind in (["sparse", "dense_parameter_matched"] + [
            "dynamic_sparse_f" + str(f).replace(".", "_") for f in args.rewire_fractions]):
        rows = [r for r in runs if r["kind"] == kind]
        aggregate[kind] = {
            "runs": len(rows),
            "mean_parameters": mean(r["parameter_count"] for r in rows),
            "mean_heldout_loss": mean(r["heldout_loss"] for r in rows),
            "mean_heldout_accuracy": mean(r["heldout_accuracy"] for r in rows),
            "mean_wall_seconds": mean(r.get("wall_seconds", 0.0) for r in rows),
        }
    report = {
        "format": "aimeng-dynamic-sparse-neuron-lm/v1",
        "dataset": "synthetic arithmetic-format character-level corpus",
        "context_characters": 8, "vocabulary_size": vocab_size,
        "vocabulary": [ch for ch, _ in sorted(vocab.items(), key=lambda item: item[1])],
        "input_feature_layout": "feature_index = context_position * vocabulary_size + token_id",
        "settings": {"epochs": args.epochs, "batch_size": args.batch_size,
                     "learning_rate": args.learning_rate, "seeds": args.seeds,
                     "input_edges_per_unit": args.input_edges,
                     "output_edges_per_unit": args.output_edges,
                     "rewire_fractions": args.rewire_fractions,
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
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("DYNAMIC_SPARSE_AGGREGATE " + json.dumps(aggregate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
