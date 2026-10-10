#!/usr/bin/env python3
"""Run autoregressive inference from an exported recurrent-neuron LM JSON checkpoint."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_model(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = data.get("vocabulary")
    units = data.get("units")
    if not isinstance(vocab, list) or not isinstance(units, list) or not units:
        raise ValueError("checkpoint must contain vocabulary and non-empty units")
    hidden = len(units)
    vocab_size = len(vocab)
    wx = np.stack([np.asarray(u["input_weights"], dtype=np.float32) for u in units], axis=1)
    wh = np.stack([np.asarray(u["recurrent_incoming_weights"], dtype=np.float32) for u in units], axis=1)
    bh = np.asarray([u["bias"] for u in units], dtype=np.float32)
    wy = np.stack([np.asarray(u["output_weights"], dtype=np.float32) for u in units], axis=0)
    by = np.asarray(data["output_bias"], dtype=np.float32)
    enabled = np.asarray([bool(u.get("enabled", True)) for u in units], dtype=np.float32)
    wx *= enabled[None, :]
    wh *= enabled[:, None] * enabled[None, :]
    bh *= enabled
    wy *= enabled[:, None]
    if wx.shape != (vocab_size, hidden) or wh.shape != (hidden, hidden):
        raise ValueError(f"invalid input/recurrent weight shapes: wx={wx.shape}, wh={wh.shape}")
    if wy.shape != (hidden, vocab_size) or by.shape != (vocab_size,):
        raise ValueError(f"invalid output weight shapes: wy={wy.shape}, by={by.shape}")
    if not all(np.isfinite(a).all() for a in (wx, wh, bh, wy, by)):
        raise ValueError("checkpoint contains non-finite parameters")
    return vocab, wx, wh, bh, wy, by


def generate(vocab, wx, wh, bh, wy, by, prompt, max_chars, temperature, seed):
    token_to_id = {ch: i for i, ch in enumerate(vocab)}
    fallback = token_to_id.get(" ", 0)
    rng = np.random.default_rng(seed)
    h = np.zeros((1, len(bh)), dtype=np.float32)
    for ch in prompt:
        token = token_to_id.get(ch, fallback)
        x = np.zeros((1, len(vocab)), dtype=np.float32)
        x[0, token] = 1.0
        h = np.tanh(x @ wx + h @ wh + bh)
    result = prompt
    for _ in range(max_chars):
        logits = (h @ wy + by)[0] / max(temperature, 0.05)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        if not np.isfinite(probs).all() or not np.isclose(probs.sum(), 1.0, atol=1e-5):
            raise FloatingPointError("invalid next-character probability distribution")
        token = int(rng.choice(len(vocab), p=probs))
        result += vocab[token]
        x = np.zeros((1, len(vocab)), dtype=np.float32)
        x[0, token] = 1.0
        h = np.tanh(x @ wx + h @ wh + bh)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="model.json exported by train_recurrent_neuron_lm.py")
    parser.add_argument("--prompt", default="Alice ", help="prefix text")
    parser.add_argument("--max-chars", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, help="optional path to save generated text")
    args = parser.parse_args()
    if args.max_chars < 0 or args.max_chars > 10000:
        parser.error("--max-chars must be between 0 and 10000")
    vocab, wx, wh, bh, wy, by = load_model(args.model)
    result = generate(vocab, wx, wh, bh, wy, by, args.prompt, args.max_chars, args.temperature, args.seed)
    if not result.startswith(args.prompt):
        raise RuntimeError("generation modified the prompt prefix")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
