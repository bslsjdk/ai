#!/usr/bin/env python3
"""Train a from-scratch recurrent neuron-unit language model on a supplied text file.

The model has individually addressable hidden neurons with input, recurrent,
and output weights. It is a character-level vanilla RNN trained with BPTT,
not a pretrained-model wrapper and not a claim of general reasoning ability.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


class RecurrentNeuronLM:
    def __init__(self, vocab_size, hidden, seed):
        self.vocab_size, self.hidden = vocab_size, hidden
        rng = np.random.default_rng(seed)
        self.wx = (rng.standard_normal((vocab_size, hidden)) / math.sqrt(vocab_size)).astype(np.float32)
        self.wh = (rng.standard_normal((hidden, hidden)) / math.sqrt(hidden)).astype(np.float32)
        self.bh = np.zeros(hidden, dtype=np.float32)
        self.wy = (rng.standard_normal((hidden, vocab_size)) / math.sqrt(hidden)).astype(np.float32)
        self.by = np.zeros(vocab_size, dtype=np.float32)
        self.unit_enabled = np.ones(hidden, dtype=bool)

    @property
    def parameter_count(self):
        return int(self.wx.size + self.wh.size + self.bh.size + self.wy.size + self.by.size)

    def forward(self, inputs, h0=None, cache=False):
        batch, length = inputs.shape
        h = np.zeros((batch, self.hidden), dtype=np.float32) if h0 is None else h0.copy()
        h *= self.unit_enabled
        hs, logits_list = [h], []
        for t in range(length):
            x = np.eye(self.vocab_size, dtype=np.float32)[inputs[:, t]]
            z = x @ self.wx + h @ self.wh + self.bh
            h = np.tanh(z) * self.unit_enabled
            logits = h @ self.wy + self.by
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)
            hs.append(h)
            logits_list.append(probs)
        probs = np.stack(logits_list, axis=1)
        return (probs, hs) if cache else probs

    def loss_accuracy(self, inputs, targets, batch_size=32):
        losses, correct, count = 0.0, 0, 0
        for start in range(0, len(inputs), batch_size):
            x, y = inputs[start:start+batch_size], targets[start:start+batch_size]
            if not len(x):
                continue
            p = self.forward(x)
            selected = np.clip(p[np.arange(len(y))[:, None], np.arange(y.shape[1])[None, :], y], 1e-9, 1)
            losses += float(-np.log(selected).sum())
            correct += int((p.argmax(axis=2) == y).sum())
            count += y.size
        return losses / max(count, 1), correct / max(count, 1)

    def train_batch(self, inputs, targets, lr, adam, step, clip=1.0):
        probs, hs = self.forward(inputs, cache=True)
        batch, length = targets.shape
        dWx = np.zeros_like(self.wx); dWh = np.zeros_like(self.wh)
        dWy = np.zeros_like(self.wy); dBh = np.zeros_like(self.bh); dBy = np.zeros_like(self.by)
        dh_next = np.zeros((batch, self.hidden), dtype=np.float32)
        rows = np.arange(batch)
        loss = 0.0
        for t in range(length - 1, -1, -1):
            p = probs[:, t, :].copy()
            yt = targets[:, t]
            loss += float(-np.log(np.clip(p[rows, yt], 1e-9, 1)).sum())
            p[rows, yt] -= 1.0
            p /= batch * length
            dWy += hs[t + 1].T @ p
            dBy += p.sum(axis=0)
            dh = p @ self.wy.T + dh_next
            dz = dh * (1 - hs[t + 1] * hs[t + 1]) * self.unit_enabled
            dBh += dz.sum(axis=0)
            dWh += hs[t].T @ dz
            xhot = np.eye(self.vocab_size, dtype=np.float32)[inputs[:, t]]
            dWx += xhot.T @ dz
            dh_next = dz @ self.wh.T
        grads = [dWx, dWh, dBh, dWy, dBy]
        norm = math.sqrt(sum(float(np.sum(g*g)) for g in grads))
        if norm > clip:
            grads = [g * (clip / norm) for g in grads]
        params = [self.wx, self.wh, self.bh, self.wy, self.by]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (param, grad) in enumerate(zip(params, grads)):
            adam["m"][i] = b1 * adam["m"][i] + (1-b1) * grad
            adam["v"][i] = b2 * adam["v"][i] + (1-b2) * (grad * grad)
            mhat = adam["m"][i] / (1 - b1**step)
            vhat = adam["v"][i] / (1 - b2**step)
            param -= lr * mhat / (np.sqrt(vhat) + eps)
        self.wx *= self.unit_enabled[None, :]
        self.wh *= self.unit_enabled[:, None] * self.unit_enabled[None, :]
        self.wy *= self.unit_enabled[:, None]
        return loss / (batch * length)

    def generate(self, prefix, vocab, reverse_vocab, max_chars=300, temperature=0.8, seed=7):
        rng = np.random.default_rng(seed)
        h = np.zeros((1, self.hidden), dtype=np.float32)
        text = prefix
        for ch in prefix:
            token = vocab.get(ch, vocab.get(" ", 0))
            x = np.eye(self.vocab_size, dtype=np.float32)[[token]]
            h = np.tanh(x @ self.wx + h @ self.wh + self.bh) * self.unit_enabled
        current = vocab.get(prefix[-1], vocab.get(" ", 0)) if prefix else 0
        for _ in range(max_chars):
            # The current hidden state already consumed the complete prefix.
            # Predict the next character first; only then consume it into state.
            logits = (h @ self.wy + self.by)[0] / max(temperature, 0.05)
            logits -= logits.max()
            p = np.exp(logits); p /= p.sum()
            current = int(rng.choice(self.vocab_size, p=p))
            text += reverse_vocab[current]
            x = np.eye(self.vocab_size, dtype=np.float32)[[current]]
            h = np.tanh(x @ self.wx + h @ self.wh + self.bh) * self.unit_enabled
        return text

    def save(self, path, reverse_vocab, metadata, ablation):
        units = []
        for j in range(self.hidden):
            units.append({
                "id": f"recurrent-neuron-{j:03d}",
                "enabled": bool(self.unit_enabled[j]),
                "input_weights": self.wx[:, j].astype(float).tolist(),
                "recurrent_incoming_weights": self.wh[:, j].astype(float).tolist(),
                "bias": float(self.bh[j]),
                "output_weights": self.wy[j, :].astype(float).tolist(),
                "updates": metadata["optimizer_steps"],
                "ablation_loss_delta": ablation[j],
            })
        data = {
            "format": "aimeng-recurrent-neuron-lm/v1",
            "architecture": "vanilla recurrent neural network; character-level next-character prediction; BPTT",
            "vocabulary": reverse_vocab,
            "parameter_count": self.parameter_count,
            "hidden_neuron_count": self.hidden,
            "units": units,
            "output_bias": self.by.astype(float).tolist(),
            "metadata": metadata,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def batches_for(ids, context, batch_size, rng, start, end):
    max_start = end - context - 1
    starts = rng.integers(start, max_start + 1, size=batch_size)
    x = np.stack([ids[s:s+context] for s in starts])
    y = np.stack([ids[s+1:s+context+1] for s in starts])
    return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--context", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr-decay-start", type=int, default=30)
    p.add_argument("--lr-decay-every", type=int, default=15)
    p.add_argument("--lr-decay-factor", type=float, default=0.5)
    p.add_argument("--steps-per-epoch", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=20261010)
    args = p.parse_args()
    if not (8 <= args.hidden <= 128 and 8 <= args.context <= 128 and 1 <= args.epochs <= 100):
        raise SystemExit("hidden 8..128, context 8..128, epochs 1..100 required")
    if args.lr_decay_start < 1 or args.lr_decay_every < 1 or not (0 < args.lr_decay_factor <= 1):
        raise SystemExit("learning-rate decay start/every must be positive and factor must be in (0, 1]")
    text = args.text_file.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) < 10000:
        raise SystemExit(f"real-text corpus unexpectedly small: {len(text)} characters")
    # Chronological train/validation/test split. Validation selects the checkpoint;
    # the final test segment is evaluated only after training is complete.
    train_end = int(len(text) * 0.85)
    validation_end = int(len(text) * 0.90)
    train_text = text[:train_end]
    validation_text = text[train_end:validation_end]
    test_text = text[validation_end:]
    vocab_chars = sorted(set(train_text))
    if "�" not in vocab_chars:
        vocab_chars.append("�")
    vocab = {ch: i for i, ch in enumerate(vocab_chars)}
    train_ids = np.array([vocab[ch] for ch in train_text], dtype=np.int64)
    validation_ids = np.array([vocab.get(ch, vocab["�"]) for ch in validation_text], dtype=np.int64)
    test_ids = np.array([vocab.get(ch, vocab["�"]) for ch in test_text], dtype=np.int64)

    def evaluation_windows(ids, context):
        xs, ys = [], []
        for start in range(0, len(ids) - context, context):
            xs.append(ids[start:start + context])
            ys.append(ids[start + 1:start + context + 1])
        if not xs:
            raise SystemExit("evaluation segment is too small for the requested context")
        return np.asarray(xs), np.asarray(ys)

    validation_x, validation_y = evaluation_windows(validation_ids, args.context)
    test_x, test_y = evaluation_windows(test_ids, args.context)
    model = RecurrentNeuronLM(len(vocab), args.hidden, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 1)
    adam = {"m": [np.zeros_like(x) for x in [model.wx, model.wh, model.bh, model.wy, model.by]],
            "v": [np.zeros_like(x) for x in [model.wx, model.wh, model.bh, model.wy, model.by]]}
    history = []
    started = time.perf_counter()
    # Uniform character predictor is the transparent baseline.
    unigram_counts = np.bincount(train_ids, minlength=len(vocab)).astype(np.float64) + 1
    unigram = unigram_counts / unigram_counts.sum()
    baseline_loss = float(-np.log(unigram[test_ids]).mean())
    steps = 0
    best_validation_loss = float("inf")
    best_epoch = 0
    best_parameters = None
    for epoch in range(1, args.epochs + 1):
        decay_count = 0 if epoch <= args.lr_decay_start else (epoch - args.lr_decay_start + args.lr_decay_every - 1) // args.lr_decay_every
        epoch_lr = args.learning_rate * (args.lr_decay_factor ** decay_count)
        losses = []
        for _ in range(args.steps_per_epoch):
            x, y = batches_for(train_ids, args.context, args.batch_size, rng, 0, len(train_ids))
            steps += 1
            losses.append(model.train_batch(x, y, epoch_lr, adam, steps))
        validation_loss, validation_acc = model.loss_accuracy(validation_x, validation_y)
        row = {"epoch": epoch, "learning_rate": epoch_lr, "mean_train_batch_loss": float(np.mean(losses)),
               "validation_loss": validation_loss,
               "validation_bits_per_char": validation_loss / math.log(2),
               "validation_char_accuracy": validation_acc}
        history.append(row)
        print("RECURRENT_NEURON_LM_EPOCH " + json.dumps(row, sort_keys=True), flush=True)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_parameters = [x.copy() for x in (model.wx, model.wh, model.bh, model.wy, model.by)]
    if best_parameters is None:
        raise SystemExit("training did not produce a valid checkpoint")
    for param, best in zip((model.wx, model.wh, model.bh, model.wy, model.by), best_parameters):
        param[...] = best
    elapsed = time.perf_counter() - started
    final_loss, final_acc = model.loss_accuracy(test_x, test_y)
    ablation = []
    for j in range(model.hidden):
        model.unit_enabled[j] = False
        loss_without, _ = model.loss_accuracy(test_x, test_y)
        model.unit_enabled[j] = True
        ablation.append(loss_without - final_loss)
    metadata = {
        "seed": args.seed, "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "learning_rate_schedule": {"type": "step_decay", "start_after_epoch": args.lr_decay_start,
                                   "every_epochs": args.lr_decay_every, "factor": args.lr_decay_factor},
        "training_characters": len(train_text), "validation_characters": len(validation_text),
        "test_characters": len(test_text), "vocabulary_size": len(vocab),
        "context_characters": args.context, "best_validation_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "evaluation_protocol": "85% chronological train / 5% validation for checkpoint selection / 10% final test",
        "dataset_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "wall_seconds": round(elapsed, 3), "optimizer_steps": steps,
        "unigram_baseline_heldout_loss": baseline_loss,
        "final_heldout_loss": final_loss, "final_heldout_bits_per_char": final_loss / math.log(2),
        "final_heldout_char_accuracy": final_acc,
        "loss_improvement_over_unigram": baseline_loss - final_loss,
        "unit_ablation_loss_delta_mean": float(np.mean(ablation)),
        "unit_ablation_loss_delta_max": float(np.max(ablation)),
        "generation_sample": model.generate("Alice ", vocab, vocab_chars, 300, seed=args.seed),
        "limitations": "Single public-domain book, character-level prediction, vanilla RNN, fixed dense recurrent connections; not a chat model or general reasoning benchmark."
    }
    model.save(args.output_dir / "model.json", vocab_chars, metadata, ablation)
    report = {"format": "aimeng-recurrent-neuron-lm-report/v1",
              "architecture": "individually addressable recurrent neuron units",
              "parameter_count": model.parameter_count, "hidden_neurons": args.hidden,
              "context_characters": args.context, "vocabulary_size": len(vocab),
              "learning_rate_schedule": metadata["learning_rate_schedule"],
              "unigram_baseline_heldout_loss": baseline_loss,
              "final_heldout_loss": final_loss, "final_heldout_bits_per_char": final_loss / math.log(2),
              "final_heldout_char_accuracy": final_acc, "wall_seconds": round(elapsed, 3),
              "best_validation_epoch": best_epoch, "best_validation_loss": best_validation_loss,
              "evaluation_protocol": metadata["evaluation_protocol"],
              "generation_sample": metadata["generation_sample"],
              "unit_ablation_loss_delta_mean": metadata["unit_ablation_loss_delta_mean"],
              "unit_ablation_loss_delta_max": metadata["unit_ablation_loss_delta_max"],
              "dataset_sha256": metadata["dataset_sha256"], "history": history,
              "model_file": "model.json", "limitations": metadata["limitations"]}
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("RECURRENT_NEURON_LM_SUMMARY " + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
