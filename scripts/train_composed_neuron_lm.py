#!/usr/bin/env python3
"""Train a tiny language model whose hidden layer is represented as individually addressable neuron units.

This is a from-scratch character-level next-character model, not a wrapper around Qwen.
It is an architecture smoke test, not a claim of general-purpose language intelligence.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np


def build_corpus():
    # Small compositional corpus. Train/test use disjoint numeric combinations and
    # different sentence templates to test simple recombination rather than memorization.
    train = []
    test = []
    for a in range(1, 31):
        for b in range(1, 13):
            c = a + b
            train.append(f"add {a} and {b} equals {c}.\n")
            train.append(f"the sum of {a} plus {b} is {c}.\n")
            train.append(f"calculate {a} + {b} = {c}.\n")
            if (a * 7 + b * 11) % 13 == 0:
                train.append(f"question: {a} plus {b}. answer: {c}.\n")
    for a in range(31, 41):
        for b in range(13, 19):
            c = a + b
            test.append(f"add {a} and {b} equals {c}.\n")
            test.append(f"the sum of {a} plus {b} is {c}.\n")
            test.append(f"calculate {a} + {b} = {c}.\n")
    return "".join(train), "".join(test)


class ComposedNeuronLanguageModel:
    """One independently addressable neuron per hidden column, connected to all input features and output classes."""
    def __init__(self, vocab_size, context, hidden, seed=20261010):
        self.vocab_size, self.context, self.hidden = vocab_size, context, hidden
        rng = np.random.default_rng(seed)
        self.units = []
        scale = math.sqrt(2.0 / (context * vocab_size + hidden))
        for i in range(hidden):
            self.units.append({
                "id": f"neuron-{i:03d}",
                "input_weights": (rng.standard_normal(context * vocab_size) * scale).astype(np.float32),
                "bias": np.float32(0),
                "output_weights": (rng.standard_normal(vocab_size) / math.sqrt(hidden)).astype(np.float32),
                "enabled": True,
                "updates": 0,
            })
        self.output_bias = np.zeros(vocab_size, dtype=np.float32)

    def pack(self):
        active = [u for u in self.units if u["enabled"]]
        if not active:
            raise ValueError("at least one neuron must remain enabled")
        return (np.stack([u["input_weights"] for u in active], axis=1),
                np.array([u["bias"] for u in active], dtype=np.float32),
                np.stack([u["output_weights"] for u in active], axis=0),
                [u for u in self.units if u["enabled"]])

    def forward(self, x):
        wx, bh, wy, active = self.pack()
        h = np.tanh(x @ wx + bh)
        logits = h @ wy + self.output_bias
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs, h, active

    def train_batch(self, x, y, lr):
        probs, h, active = self.forward(x)
        n = len(y)
        loss = -np.log(np.clip(probs[np.arange(n), y], 1e-9, 1)).mean()
        dlogits = probs
        dlogits[np.arange(n), y] -= 1
        dlogits /= n
        _, _, wy, _ = self.pack()
        dwy = h.T @ dlogits
        dbo = dlogits.sum(axis=0)
        dh = dlogits @ wy.T
        dz = dh * (1 - h * h)
        dwx = x.T @ dz
        dbh = dz.sum(axis=0)
        for j, unit in enumerate(active):
            unit["input_weights"] -= lr * dwx[:, j]
            unit["bias"] -= lr * dbh[j]
            unit["output_weights"] -= lr * dwy[j]
            unit["updates"] += 1
        self.output_bias -= lr * dbo
        return float(loss)

    def evaluate(self, x, y):
        probs, _, _ = self.forward(x)
        loss = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1)).mean()
        accuracy = float((probs.argmax(axis=1) == y).mean())
        return float(loss), accuracy

    def generate(self, prefix, vocab, max_chars=32):
        """Greedy autoregressive decoding using only this trained model's own weights."""
        reverse_vocab = list(vocab.keys())
        text = prefix
        for _ in range(max_chars):
            tail = text[-self.context:]
            if len(tail) < self.context:
                tail = (" " * (self.context - len(tail))) + tail
            x = np.zeros((1, self.context * self.vocab_size), dtype=np.float32)
            for pos, ch in enumerate(tail):
                token_id = vocab.get(ch, vocab.get(" ", 0))
                x[0, pos * self.vocab_size + token_id] = 1.0
            probs, _, _ = self.forward(x)
            text += reverse_vocab[int(probs[0].argmax())]
        return text

    def save(self, path, vocab, metadata):
        units = []
        for u in self.units:
            units.append({
                "id": u["id"], "enabled": u["enabled"], "bias": float(u["bias"]),
                "input_weights": u["input_weights"].tolist(),
                "output_weights": u["output_weights"].tolist(), "updates": u["updates"],
            })
        obj = {
            "format": "aimeng-composed-neuron-language-model/v1",
            "architecture": "fixed-context one-hidden-layer character next-token predictor",
            "vocab": vocab, "context_characters": self.context,
            "parameter_count": int(sum(len(u["input_weights"]) + len(u["output_weights"]) + 1 for u in self.units) + len(self.output_bias)),
            "hidden_neuron_count": len(self.units),
            "units": units, "output_bias": self.output_bias.tolist(), "metadata": metadata,
        }
        Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def encode_corpus(text, vocab, context):
    ids = np.array([vocab[ch] for ch in text if ch in vocab], dtype=np.int64)
    if len(ids) <= context:
        raise ValueError("corpus too small for context")
    xs, ys = [], []
    for i in range(context, len(ids)):
        window = ids[i-context:i]
        x = np.zeros(context * len(vocab), dtype=np.float32)
        for pos, token in enumerate(window):
            x[pos * len(vocab) + token] = 1.0
        xs.append(x)
        ys.append(ids[i])
    return np.stack(xs), np.array(ys, dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.025)
    p.add_argument("--seed", type=int, default=20261010)
    args = p.parse_args()
    if not (2 <= args.hidden <= 128 and 2 <= args.context <= 16 and 1 <= args.epochs <= 100):
        raise SystemExit("hidden 2..128, context 2..16, epochs 1..100 required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_text, test_text = build_corpus()
    vocab = {ch: i for i, ch in enumerate(sorted(set(train_text + test_text)))}
    x_train, y_train = encode_corpus(train_text, vocab, args.context)
    x_test, y_test = encode_corpus(test_text, vocab, args.context)
    model = ComposedNeuronLanguageModel(len(vocab), args.context, args.hidden, args.seed)
    # Character-level unigram baseline, with Laplace smoothing.
    counts = np.bincount(y_train, minlength=len(vocab)).astype(np.float64) + 1
    unigram = counts / counts.sum()
    baseline_loss = float(-np.log(unigram[y_test]).mean())
    rng = np.random.default_rng(args.seed)
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(y_train))
        losses = []
        for begin in range(0, len(order), args.batch_size):
            ix = order[begin:begin + args.batch_size]
            losses.append(model.train_batch(x_train[ix], y_train[ix], args.learning_rate))
        test_loss, test_acc = model.evaluate(x_test, y_test)
        train_loss, train_acc = model.evaluate(x_train[:min(12000, len(y_train))], y_train[:min(12000, len(y_train))])
        row = {"epoch": epoch, "train_loss": train_loss, "train_next_char_accuracy": train_acc,
               "heldout_loss": test_loss, "heldout_next_char_accuracy": test_acc}
        history.append(row)
        print("COMPOSED_NEURON_LM_EPOCH " + json.dumps(row, sort_keys=True), flush=True)
    elapsed = time.perf_counter() - started
    final_loss, final_acc = model.evaluate(x_test, y_test)
    # Unit-level ablation: measure held-out loss after disabling each unit individually.
    ablation = []
    for unit in model.units:
        unit["enabled"] = False
        loss_without, _ = model.evaluate(x_test, y_test)
        unit["enabled"] = True
        ablation.append({"id": unit["id"], "loss_without": loss_without,
                         "loss_delta": float(loss_without - final_loss)})
    ablation.sort(key=lambda row: row["loss_delta"], reverse=True)
    samples = {prefix: model.generate(prefix, vocab, 28) for prefix in
               ["add 7 and 4", "the sum of 8", "calculate 12 +"]}
    metadata = {
        "seed": args.seed, "epochs": args.epochs, "learning_rate": args.learning_rate,
        "training_characters": len(train_text), "heldout_characters": len(test_text),
        "training_examples": len(y_train), "heldout_examples": len(y_test),
        "dataset_sha256": hashlib.sha256((train_text + "\0" + test_text).encode()).hexdigest(),
        "wall_seconds": round(elapsed, 4), "baseline_unigram_heldout_loss": baseline_loss,
        "final_heldout_loss": final_loss, "final_heldout_next_char_accuracy": final_acc,
        "loss_improvement_over_unigram": baseline_loss - final_loss,
        "unit_ablation_ranked": ablation,
        "greedy_generation_samples": samples,
        "history": history,
        "limits": "Synthetic arithmetic-format corpus; fixed context, one hidden layer, character-level prediction. This is not a general-purpose assistant or a fair direct score against pretrained chat LLMs."
    }
    model.save(args.output_dir / "model.json", list(vocab.keys()), metadata)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "format": "aimeng-composed-neuron-lm-report/v1", "architecture": "individually addressable hidden neuron units",
        "hidden_neurons": args.hidden, "parameter_count": int(args.hidden * (args.context * len(vocab) + len(vocab) + 1) + len(vocab)),
        "vocab_size": len(vocab), "context": args.context, "baseline_unigram_heldout_loss": baseline_loss,
        "final_heldout_loss": final_loss, "final_heldout_next_char_accuracy": final_acc,
        "loss_improvement_over_unigram": baseline_loss - final_loss, "wall_seconds": round(elapsed, 4),
        "greedy_generation_samples": samples, "top_unit_ablation": ablation[:10],
        "seed": args.seed, "dataset_sha256": metadata["dataset_sha256"],
        "history": history, "model_file": "model.json", "limitations": metadata["limits"]
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("COMPOSED_NEURON_LM_SUMMARY " + json.dumps(metadata | {
        "hidden_neurons": args.hidden, "parameter_count": int(args.hidden * (args.context * len(vocab) + len(vocab) + 1) + len(vocab))
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
