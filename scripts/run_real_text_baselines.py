#!/usr/bin/env python3
"""Evaluate smoothed character n-grams and train a real fixed-window neural baseline.

Chronological split: 85% train, 5% validation, 10% final test. Test targets are
never used for checkpoint selection. All metrics are character-level only.
"""
import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def split_text(text):
    a, b = int(len(text) * 0.85), int(len(text) * 0.90)
    return text[:a], text[a:b], text[b:]


def encode(text, vocab):
    unk = vocab["\ufffd"]
    return np.asarray([vocab.get(ch, unk) for ch in text], dtype=np.int32)


def make_examples(all_ids, start, end, context):
    targets = np.arange(max(start, context), end, dtype=np.int32)
    return make_examples_at_positions(all_ids, targets, context)


def make_examples_at_positions(all_ids, targets, context):
    targets = np.asarray(targets, dtype=np.int32)
    x = np.stack([all_ids[t-context:t] for t in targets])
    return x, all_ids[targets]


def ngram_baselines(train_ids, all_ids, test_positions, vocab_size, orders=(2, 3, 4, 5), alpha=0.1):
    counts_by_order = {}
    totals_by_order = {}
    for order in orders:
        counts = defaultdict(Counter)
        totals = Counter()
        for i in range(order - 1, len(train_ids)):
            ctx = tuple(int(v) for v in train_ids[i-order+1:i])
            token = int(train_ids[i])
            counts[ctx][token] += 1
            totals[ctx] += 1
        counts_by_order[order] = counts
        totals_by_order[order] = totals

    unigram_counts = np.bincount(train_ids, minlength=vocab_size).astype(np.float64)
    # Additive smoothing for the base distribution.
    unigram_probs = (unigram_counts + alpha) / (unigram_counts.sum() + alpha * vocab_size)

    def prob(token, context, order):
        if order == 1:
            return float(unigram_probs[token])
        ctx = tuple(context[-(order-1):])
        c = counts_by_order[order][ctx][token]
        total = totals_by_order[order][ctx]
        lower = prob(token, ctx[1:], order - 1)
        return (c + alpha * lower) / (total + alpha)

    results = {}
    test_ids = all_ids[test_positions]
    losses, correct = {n: 0.0 for n in orders}, {n: 0 for n in orders}
    for t, token_value in zip(test_positions, test_ids):
        t = int(t)
        token = int(token_value)
        for order in orders:
            context = all_ids[max(0, t-order+1):t]
            p = max(prob(token, context, order), 1e-300)
            losses[order] += -math.log(p)
            # Exact argmax over vocabulary using the same interpolated distribution.
            best_token, best_p = 0, -1.0
            for candidate in range(vocab_size):
                candidate_p = prob(candidate, context, order)
                if candidate_p > best_p:
                    best_token, best_p = candidate, candidate_p
            correct[order] += int(best_token == token)
    n = len(test_ids)
    for order in orders:
        results[str(order)] = {
            "order": order, "context_characters": order - 1, "alpha": alpha,
            "test_characters_scored": n, "test_loss": losses[order] / n,
            "test_bits_per_char": losses[order] / n / math.log(2),
            "test_char_accuracy": correct[order] / n,
        }
    return results


class FixedWindowLM:
    def __init__(self, vocab_size, context, hidden, seed):
        self.vocab_size, self.context, self.hidden = vocab_size, context, hidden
        rng = np.random.default_rng(seed)
        self.wx = (rng.standard_normal((context, vocab_size, hidden)) /
                   math.sqrt(context * vocab_size)).astype(np.float32)
        self.bh = np.zeros(hidden, np.float32)
        self.wy = (rng.standard_normal((hidden, vocab_size)) /
                   math.sqrt(hidden)).astype(np.float32)
        self.by = np.zeros(vocab_size, np.float32)
        self.params = [self.wx, self.bh, self.wy, self.by]
        self.m = [np.zeros_like(x) for x in self.params]
        self.v = [np.zeros_like(x) for x in self.params]
        self.step = 0

    @property
    def parameter_count(self):
        return sum(x.size for x in self.params)

    def forward(self, x):
        batch = len(x)
        gathered = self.wx[np.arange(self.context)[None, :], x]
        h = np.tanh(gathered.sum(axis=1) + self.bh)
        logits = h @ self.wy + self.by
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True), h

    def train_batch(self, x, y, lr=0.001):
        p, h = self.forward(x)
        n = len(y)
        p[np.arange(n), y] -= 1.0
        p /= n
        dwy = h.T @ p
        dby = p.sum(axis=0)
        dh = (p @ self.wy.T) * (1.0 - h*h)
        dbh = dh.sum(axis=0)
        dwx = np.zeros_like(self.wx)
        for pos in range(self.context):
            np.add.at(dwx[pos], x[:, pos], dh)
        grads = [dwx, dbh, dwy, dby]
        norm = math.sqrt(sum(float(np.sum(g*g)) for g in grads))
        if norm > 1.0:
            grads = [g / norm for g in grads]
        self.step += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.m[i] = 0.9*self.m[i] + 0.1*grad
            self.v[i] = 0.999*self.v[i] + 0.001*(grad*grad)
            mh = self.m[i] / (1.0 - 0.9**self.step)
            vh = self.v[i] / (1.0 - 0.999**self.step)
            param -= lr * mh / (np.sqrt(vh) + 1e-8)

    def evaluate(self, x, y, batch_size=512):
        loss, correct, count = 0.0, 0, 0
        for s in range(0, len(y), batch_size):
            xb, yb = x[s:s+batch_size], y[s:s+batch_size]
            p, _ = self.forward(xb)
            loss += float(-np.log(np.clip(p[np.arange(len(yb)), yb], 1e-12, 1)).sum())
            correct += int((p.argmax(axis=1) == yb).sum())
            count += len(yb)
        return {"loss": loss/count, "bits_per_char": loss/count/math.log(2),
                "char_accuracy": correct/count, "characters": count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20261010, 20261011, 20261012])
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=0.001)
    ap.add_argument("--ngram-alpha", type=float, default=0.1)
    args = ap.parse_args()

    text = args.text_file.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    train_text, val_text, test_text = split_text(text)
    chars = sorted(set(train_text))
    if "\ufffd" not in chars:
        chars.append("\ufffd")
    vocab = {ch:i for i,ch in enumerate(chars)}
    all_ids = encode(text, vocab)
    train_end, val_end = len(train_text), len(train_text)+len(val_text)
    train_ids = all_ids[:train_end]
    val_x, val_y = make_examples(all_ids, train_end, val_end, args.context)
    # Match the recurrent model's test-target protocol: 64-character chunks,
    # each predicts the next 64 characters, omitting the incomplete tail.
    recurrent_context = 64
    test_positions = np.concatenate([
        np.arange(val_end + start + 1, val_end + start + recurrent_context + 1, dtype=np.int32)
        for start in range(0, len(test_text) - recurrent_context, recurrent_context)
    ])
    test_x, test_y = make_examples_at_positions(all_ids, test_positions, args.context)
    train_positions = np.arange(args.context, train_end, dtype=np.int32)
    data_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    print("NGRAM_START "+json.dumps({"orders":[2,3,4,5],"alpha":args.ngram_alpha,
                                     "train":len(train_text),"validation":len(val_text),
                                     "test":len(test_text)},sort_keys=True),flush=True)
    ngram_started = time.perf_counter()
    ngram = ngram_baselines(train_ids, all_ids, test_positions, len(vocab), alpha=args.ngram_alpha)
    ngram_seconds = time.perf_counter()-ngram_started
    for order, row in ngram.items():
        print("NGRAM_RESULT "+json.dumps(row,sort_keys=True),flush=True)

    run_rows = []
    curve_rows = []
    for seed in args.seeds:
        model = FixedWindowLM(len(vocab), args.context, args.hidden, seed)
        rng = np.random.default_rng(seed+1)
        best_val = float("inf")
        best_epoch = 0
        best_params = None
        history = []
        started = time.perf_counter()
        for epoch in range(1, args.epochs+1):
            batch_losses = []
            for _ in range(args.steps_per_epoch):
                starts = rng.choice(train_positions, size=args.batch_size, replace=True)
                x = all_ids[starts[:,None] - args.context + np.arange(args.context)[None,:]]
                y = all_ids[starts]
                p, _ = model.forward(x)
                batch_losses.append(float(-np.log(np.clip(p[np.arange(len(y)), y],1e-12,1)).mean()))
                model.train_batch(x,y,args.learning_rate)
            val = model.evaluate(val_x,val_y)
            row = {"seed":seed,"epoch":epoch,"mean_train_batch_loss":statistics.mean(batch_losses),
                   "validation_loss":val["loss"],"validation_bits_per_char":val["bits_per_char"],
                   "validation_char_accuracy":val["char_accuracy"]}
            history.append(row)
            curve_rows.append(row)
            print("FIXED_WINDOW_EPOCH "+json.dumps(row,sort_keys=True),flush=True)
            if val["loss"] < best_val:
                best_val = val["loss"]
                best_epoch = epoch
                best_params = [p.copy() for p in model.params]
        for param, saved in zip(model.params,best_params):
            param[...] = saved
        test = model.evaluate(test_x,test_y)
        elapsed = time.perf_counter()-started
        run = {"seed":seed,"parameter_count":model.parameter_count,"context_characters":args.context,
               "hidden_neurons":args.hidden,"best_validation_epoch":best_epoch,
               "best_validation_loss":best_val,"test":test,"wall_seconds":round(elapsed,3),
               "epochs":history}
        run_rows.append(run)
        print("FIXED_WINDOW_RESULT "+json.dumps({k:v for k,v in run.items() if k!="epochs"},sort_keys=True),flush=True)

    def mean_std(values):
        return {"mean":statistics.mean(values),
                "sample_stddev":statistics.stdev(values) if len(values)>1 else 0.0}
    fixed_aggregate = {}
    for key in ("loss","bits_per_char","char_accuracy"):
        vals=[r["test"][key] for r in run_rows]
        fixed_aggregate[key]=mean_std(vals)
    report = {
        "format":"aimeng-real-text-ngram-and-fixed-window-baselines/v1",
        "dataset":"Project Gutenberg Alice's Adventures in Wonderland, chronological split",
        "dataset_sha256":data_hash,
        "split":{"train_fraction":0.85,"validation_fraction":0.05,"test_fraction":0.10,
                 "test_target_alignment":"same 64-character chunk targets as the recurrent model; incomplete final chunk omitted",
                 "train_characters":len(train_text),"validation_characters":len(val_text),
                 "test_characters":len(test_text),"test_characters_scored":len(test_positions),
                 "vocabulary_size":len(vocab)},
        "ngram":{"smoothing":"recursive Dirichlet interpolation; unigram base uses additive smoothing",
                 "alpha":args.ngram_alpha,"wall_seconds":round(ngram_seconds,3),
                 "orders":ngram},
        "fixed_window":{"context_characters":args.context,"hidden_neurons":args.hidden,
                        "parameter_count":run_rows[0]["parameter_count"],
                        "settings":{"epochs":args.epochs,"steps_per_epoch":args.steps_per_epoch,
                                    "batch_size":args.batch_size,"learning_rate":args.learning_rate,
                                    "seeds":args.seeds},
                        "aggregate_test":fixed_aggregate,"runs":run_rows},
        "validation_curves":curve_rows,
        "limitations":["Single-book character prediction; no claim about chat or reasoning.",
                       "N-gram models use only preceding characters and smoothed counts; no neural training.",
                       "Fixed-window model is trained for the recorded update/target budget, which is not FLOP-identical to the recurrent model.",
                       "Metrics must only be compared when the exact dataset hash and test target positions match."]
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("REAL_TEXT_BASELINES_SUMMARY "+json.dumps({"dataset_sha256":data_hash,
          "ngram":ngram,"fixed_window_aggregate":fixed_aggregate,
          "fixed_window_parameters":run_rows[0]["parameter_count"],
          "fixed_window_mean_best_epoch":statistics.mean(r["best_validation_epoch"] for r in run_rows),
          "fixed_window_runs":[{k:v for k,v in r.items() if k!="epochs"} for r in run_rows]},
          ensure_ascii=False,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
