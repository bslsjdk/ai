#!/usr/bin/env python3
"""Generate AIMENG character-level distillation examples from a small causal LM.

The script runs on a PC/Kaggle, not on the phone. It maps only teacher vocabulary
entries that decode to exactly one Unicode code point in the student's fixed
character vocabulary. Multi-character subword probabilities are deliberately NOT
split into invented per-character probabilities. mapped_mass reports how much of
the teacher's next-token probability was retained before top-k renormalization.

Output is JSONL: first a metadata record, then examples with split=train/eval.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED_TEXT = (
    "从前有一只小狐狸，名字叫团团。一天清晨，团团在森林边发现一颗发光的种子。"
    "它没有把种子带回家，而是先去问老橡树。老橡树说，种子需要阳光、清水和耐心。"
    "团团每天给种子浇一点水，还把附近的石头轻轻搬开。几天后，嫩芽钻出了泥土。"
    "小兔子和小鸟都来帮忙，大家轮流照看它。下雨时，小动物们用树叶挡住积水；"
    "刮风时，它们在旁边插上细树枝。嫩芽慢慢长成小树，春天开出许多金色的小花。"
    "团团明白，分享与耐心能让森林更美好。"
)


def student_vocab():
    seen = set()
    result = []
    for ch in SEED_TEXT:
        if ch not in seen:
            seen.add(ch)
            if len(result) < 199:
                result.append(ch)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file", help="UTF-8 Chinese corpus; omitted means a tiny built-in smoke-test story")
    parser.add_argument("--output", default="aimeng_distillation.jsonl")
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--context-limit", type=int, default=48)
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth eligible token position")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.max_examples < 20 or args.max_examples > 50000:
        raise SystemExit("--max-examples must be 20..50000")
    if args.top_k < 2 or args.top_k > 20:
        raise SystemExit("--top-k must be 2..20")
    if args.context_limit < 2 or args.context_limit > 256:
        raise SystemExit("--context-limit must be 2..256")
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else SEED_TEXT * 20
    text = "".join(text.split())
    if len(text) < 100:
        raise SystemExit("text corpus too short; provide at least 100 non-whitespace characters")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading teacher {args.model} on {device}; this stage is offline and does not run on Android.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit("A fast tokenizer with offset_mapping is required for safe character contexts.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    vocab_chars = student_vocab()
    vocab_set = set(vocab_chars)
    token_to_char = {}
    print("Mapping teacher tokens that decode to exactly one character in the student's vocabulary...")
    for token_id in range(len(tokenizer)):
        piece = tokenizer.decode([token_id], skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
        if len(piece) == 1 and piece in vocab_set:
            token_to_char[token_id] = piece
    if not token_to_char:
        raise SystemExit("No exact single-character teacher tokens overlap the student's vocabulary.")

    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if len(ids) < 2:
        raise SystemExit("Teacher tokenizer produced too few tokens.")

    # Split by source-text position to avoid putting adjacent examples in both splits.
    eval_start = int(len(text) * 0.9)
    examples = []
    skipped = 0
    for index in range(len(ids) - 1):
        start, end = offsets[index]
        if end <= start or end >= len(text):
            continue
        if index % args.stride != 0:
            continue
        context = text[:end]
        if len(context) > args.context_limit:
            context = context[-args.context_limit:]
        if len(context) < 1:
            continue
        examples.append((index, end, context))
        if len(examples) >= args.max_examples:
            break
    if len(examples) < 20:
        raise SystemExit("Too few eligible examples; provide a longer text file.")

    records = []
    with torch.inference_mode():
        for n, (index, end, context) in enumerate(examples):
            # Re-tokenize the exact context, so logits are conditioned on the same text
            # that will be shown to the character-level student.
            ctx_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            if ctx_ids.shape[1] == 0:
                skipped += 1
                continue
            logits = model(input_ids=ctx_ids).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            by_char = {}
            mapped_mass = 0.0
            for token_id, char in token_to_char.items():
                p = float(probs[token_id].item())
                if p > 0:
                    by_char[char] = by_char.get(char, 0.0) + p
                    mapped_mass += p
            if mapped_mass <= 1.0e-12:
                skipped += 1
                continue
            ranked = sorted(by_char.items(), key=lambda item: item[1], reverse=True)[:args.top_k]
            top_mass = sum(p for _, p in ranked)
            if top_mass <= 1.0e-12:
                skipped += 1
                continue
            target_probs = {char: p / top_mass for char, p in ranked}
            actual_next = text[end] if end < len(text) else ""
            split = "eval" if end >= eval_start else "train"
            records.append({
                "type": "example",
                "split": split,
                "context": context,
                "target_probs": target_probs,
                "actual_next_char": actual_next,
                "mapped_mass": mapped_mass,
                "top_k_mass": top_mass,
                "teacher": args.model,
            })
            if (n + 1) % 100 == 0:
                print(f"Prepared {n + 1}/{len(examples)} examples")
    if not records:
        raise SystemExit("No usable examples generated.")
    output = Path(args.output)
    with output.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "type": "meta",
            "format": "aimeng-character-distillation/v1",
            "teacher": args.model,
            "vocab_chars": vocab_chars,
            "examples": len(records),
            "top_k": args.top_k,
            "mapping": "exact single-codepoint tokens only; probabilities renormalized over retained top-k characters",
            "warning": "This is a restricted character-token distribution, not a full decomposition of multi-character subword probabilities."
        }, ensure_ascii=False) + "\n")
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    train_n = sum(x["split"] == "train" for x in records)
    eval_n = sum(x["split"] == "eval" for x in records)
    mean_mass = sum(x["mapped_mass"] for x in records) / len(records)
    print(f"Saved {len(records)} examples to {output.resolve()}")
    print(f"train={train_n}, eval={eval_n}, skipped={skipped}, mean mapped probability mass={mean_mass:.6f}")
    print("Next: copy the JSONL to the phone and import it in AIMENG's Knowledge Distillation page.")


if __name__ == "__main__":
    main()
