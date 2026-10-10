#!/usr/bin/env python3
"""Parameter-budgeted fixed-context neuron network baseline on the same real text as the RNN.

This baseline is a conventional position-specific feed-forward network. It uses
one individually addressable hidden-unit vector, but no recurrent state.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


class FixedContextNeuronLM:
    def __init__(self, vocab_size, context, hidden, seed):
        self.vocab_size, self.context, self.hidden = vocab_size, context, hidden
        rng = np.random.default_rng(seed)
        self.wx = (rng.standard_normal((context * vocab_size, hidden)) /
                   math.sqrt(context * vocab_size)).astype(np.float32)
        self.bh = np.zeros(hidden, dtype=np.float32)
        self.wy = (rng.standard_normal((hidden, vocab_size)) / math.sqrt(hidden)).astype(np.float32)
        self.by = np.zeros(vocab_size, dtype=np.float32)
        self.adam_m = [np.zeros_like(a) for a in (self.wx, self.bh, self.wy, self.by)]
        self.adam_v = [np.zeros_like(a) for a in (self.wx, self.bh, self.wy, self.by)]
        self.steps = 0

    @property
    def parameter_count(self):
        return int(self.wx.size + self.bh.size + self.wy.size + self.by.size)

    def forward(self, x):
        batch = x.shape[0]
        pos = np.arange(self.context)[None, :]
        flat_index = (pos * self.vocab_size + x).reshape(batch, self.context)
        gathered = self.wx[flat_index]
        h = np.tanh(gathered.sum(axis=1) + self.bh)
        logits = h @ self.wy + self.by
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True), h, flat_index

    def train_batch(self, x, y, lr):
        p, h, flat_index = self.forward(x)
        n = len(y)
        p[np.arange(n), y] -= 1
        p /= n
        dwy = h.T @ p
        dby = p.sum(axis=0)
        dh = p @ self.wy.T
        dz = dh * (1 - h*h)
        dbh = dz.sum(axis=0)
        dwx = np.zeros_like(self.wx)
        grad_rows = np.broadcast_to(dz[:, None, :], (n, self.context, self.hidden)).reshape(-1, self.hidden)
        np.add.at(dwx, flat_index.reshape(-1), grad_rows)
        grads = [dwx, dbh, dwy, dby]
        self.steps += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = [self.wx, self.bh, self.wy, self.by]
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.adam_m[i] = b1*self.adam_m[i] + (1-b1)*grad
            self.adam_v[i] = b2*self.adam_v[i] + (1-b2)*(grad*grad)
            mhat = self.adam_m[i] / (1-b1**self.steps)
            vhat = self.adam_v[i] / (1-b2**self.steps)
            param -= lr * mhat / (np.sqrt(vhat) + eps)

    def evaluate(self, xs, ys, batch_size=512):
        total_loss, correct, count = 0.0, 0, 0
        for start in range(0, len(ys), batch_size):
            x, y = xs[start:start+batch_size], ys[start:start+batch_size]
            p, _, _ = self.forward(x)
            total_loss += float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).sum())
            correct += int((p.argmax(axis=1) == y).sum())
            count += len(y)
        return total_loss/max(count,1), correct/max(count,1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--context", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=240)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=20261010)
    args = p.parse_args()
    text = args.text_file.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    split = int(len(text)*0.9)
    train_text, test_text = text[:split], text[split:]
    chars = sorted(set(train_text))
    if "�" not in chars: chars.append("�")
    vocab = {ch:i for i,ch in enumerate(chars)}
    train_ids = np.array([vocab[ch] for ch in train_text], dtype=np.int64)
    all_ids = np.array([vocab.get(ch,vocab["�"]) for ch in text], dtype=np.int64)
    # Each example predicts one next character from the preceding fixed context.
    starts = np.arange(split, len(all_ids))
    valid = starts >= args.context
    starts = starts[valid]
    test_x = np.stack([all_ids[s-args.context:s] for s in starts])
    test_y = all_ids[starts]
    model = FixedContextNeuronLM(len(vocab), args.context, args.hidden, args.seed)
    rng = np.random.default_rng(args.seed+1)
    history = []
    started = time.perf_counter()
    counts = np.bincount(train_ids, minlength=len(vocab)).astype(np.float64)+1
    unigram = counts/counts.sum()
    baseline = float(-np.log(unigram[test_y]).mean())
    positions = np.arange(args.context)[None,:]
    for epoch in range(1, args.epochs+1):
        train_losses = []
        for _ in range(args.steps_per_epoch):
            s = rng.integers(0, len(train_ids)-args.context, size=args.batch_size)
            x = train_ids[s[:,None]+positions]
            y = train_ids[s+args.context]
            model.train_batch(x,y,args.learning_rate)
        loss, acc = model.evaluate(test_x,test_y)
        row = {"epoch":epoch,"heldout_loss":loss,"heldout_bits_per_char":loss/math.log(2),
               "heldout_char_accuracy":acc}
        history.append(row)
        print("FIXED_CONTEXT_NEURON_LM_EPOCH "+json.dumps(row,sort_keys=True),flush=True)
    elapsed=time.perf_counter()-started
    loss,acc=model.evaluate(test_x,test_y)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={"format":"aimeng-fixed-context-neuron-lm-report/v1",
            "architecture":"position-specific fixed-context feed-forward neuron network",
            "parameter_count":model.parameter_count,"hidden_neurons":args.hidden,
            "context_characters":args.context,"vocabulary_size":len(vocab),
            "unigram_baseline_heldout_loss":baseline,"final_heldout_loss":loss,
            "final_heldout_bits_per_char":loss/math.log(2),"final_heldout_char_accuracy":acc,
            "wall_seconds":round(elapsed,3),"dataset_sha256":__import__("hashlib").sha256(text.encode()).hexdigest(),
            "settings":{"epochs":args.epochs,"steps_per_epoch":args.steps_per_epoch,
                        "batch_size":args.batch_size,"learning_rate":args.learning_rate,"seed":args.seed},
            "history":history,
            "limitations":["Same single-book held-out text as the recurrent model.",
                           "Character-level prediction, not chat or reasoning.",
                           "Parameter count is reported explicitly; training FLOPs differ from recurrent sequence training."]}
    (args.output_dir/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FIXED_CONTEXT_NEURON_LM_SUMMARY "+json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
