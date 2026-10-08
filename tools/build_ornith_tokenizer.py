#!/usr/bin/env python3
"""Build a compact runtime tokenizer for the single Ornith-1.5-9B MLX model.

The source of truth is the official Ornith tokenizer.json fetched by CI.  The
APK contains only the compact binary result, not the large JSON vocabulary.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path


def byte_decoder() -> dict[str, bytes]:
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    missing = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + missing)
            missing += 1
    return {chr(c): bytes((b,)) for b, c in zip(bs, cs)}


DECODER = byte_decoder()


def token_bytes(token: str) -> bytes:
    # Qwen's tokenizer is byte-level BPE.  Most entries use the GPT-2 byte
    # alphabet; fall back to UTF-8 for control / literal Unicode entries.
    try:
        return b"".join(DECODER[c] for c in token)
    except KeyError:
        return token.encode("utf-8")


def merge_pair(entry) -> tuple[str, str]:
    if isinstance(entry, list) and len(entry) == 2:
        return str(entry[0]), str(entry[1])
    parts = str(entry).split(" ", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid BPE merge: {entry!r}")
    return parts[0], parts[1]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: build_ornith_tokenizer.py tokenizer.json output.otk2", file=sys.stderr)
        return 2

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    data = json.loads(src.read_text(encoding="utf-8"))

    model = data.get("model") or {}
    vocab = model.get("vocab") or {}
    merges = model.get("merges") or []
    if not isinstance(vocab, dict):
        raise ValueError("tokenizer model.vocab is not an object")

    max_id = max((int(v) for v in vocab.values()), default=-1)
    tokens = [b""] * (max_id + 1)
    ranks = [-1] * (max_id + 1)

    for text, raw_id in vocab.items():
        idx = int(raw_id)
        if idx < 0:
            raise ValueError(f"negative vocabulary id: {idx}")
        if idx >= len(tokens):
            grow = idx + 1 - len(tokens)
            tokens.extend([b""] * grow)
            ranks.extend([-1] * grow)
        tokens[idx] = token_bytes(str(text))

    # A token produced by merge i is assigned rank i.  The runtime chooses the
    # lowest rank pair exactly like BPE.  Non-merge vocabulary entries stay -1.
    by_token = {str(text): int(raw_id) for text, raw_id in vocab.items()}
    for rank, entry in enumerate(merges):
        left, right = merge_pair(entry)
        merged = token_bytes(left) + token_bytes(right)
        idx = by_token.get(left + right)
        if idx is None:
            # Some tokenizer JSON variants use an escaped/normalized spelling
            # in vocab.  Find the byte-equivalent token when present.
            for text, raw_id in vocab.items():
                if token_bytes(str(text)) == merged:
                    idx = int(raw_id)
                    break
        if idx is not None and 0 <= idx < len(ranks):
            ranks[idx] = rank

    specials: dict[str, int] = {}
    for item in data.get("added_tokens") or []:
        if not isinstance(item, dict) or not item.get("special"):
            continue
        content = str(item.get("content", ""))
        if content:
            specials[content] = int(item["id"])

    eos = -1
    for candidate in ("<|im_end|>", "<|endoftext|>", "<|eos_token|>"):
        if candidate in specials:
            eos = specials[candidate]
            break
        if candidate in vocab:
            eos = int(vocab[candidate])
            break

    # Added/special tokens may live outside model.vocab. Extend the ID table
    # to the highest added-token ID so the runtime can embed them directly.
    if specials:
        max_special = max(specials.values())
        if max_special >= len(tokens):
            grow = max_special + 1 - len(tokens)
            tokens.extend([b""] * grow)
            ranks.extend([-1] * grow)
        for text, idx in specials.items():
            if idx < 0 or idx >= len(tokens):
                raise ValueError(f"special token id out of range: {idx}")
            if not tokens[idx]:
                tokens[idx] = text.encode("utf-8")
    # OTK2:
    # magic[4], version[u32], vocab_count[u32], bos[i32], eos[i32],
    # special_count[u32], then vocab entries (len[u32], bytes, rank[i32]),
    # then special entries (len[u32], utf8 bytes, id[u32]).
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as out:
        out.write(b"OTK2")
        out.write(struct.pack("<IIiiI", 2, len(tokens), -1, eos, len(specials)))
        for raw, rank in zip(tokens, ranks):
            out.write(struct.pack("<I", len(raw)))
            out.write(raw)
            out.write(struct.pack("<i", rank))
        for text, idx in sorted(specials.items(), key=lambda x: (len(x[0]), x[0])):
            raw = text.encode("utf-8")
            out.write(struct.pack("<I", len(raw)))
            out.write(raw)
            out.write(struct.pack("<I", idx))

    print(f"wrote {dst} vocab={len(tokens)} merges={len(merges)} specials={len(specials)} eos={eos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
