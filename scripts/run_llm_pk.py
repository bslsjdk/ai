#!/usr/bin/env python3
"""Deterministic first-phase evaluation for actual chat language models served by llama.cpp."""
import argparse
import hashlib
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def build_cases():
    cases = []
    def add(category, prompt, expected, i):
        cases.append({"id": f"{category}-{i:03d}", "category": category, "prompt": prompt, "expected": str(expected)})
    for i in range(24):
        a, b = (i * 37 + 19) % 197 + 3, (i * 23 + 11) % 89 + 2
        op = i % 4
        if op == 0: expr, ans = f"{a} + {b}", a + b
        elif op == 1: expr, ans = f"{a + b} - {a}", b
        elif op == 2:
            x, y = i % 12 + 2, i % 9 + 3
            expr, ans = f"{x} * {y}", x * y
        else:
            y = i % 7 + 2
            x = (i % 15 + 2) * y
            expr, ans = f"{x} / {y}", x // y
        add("arithmetic", f"Calculate exactly: {expr}. Return only the integer answer.", ans, i + 1)

    for i in range(16):
        premises, question, answer = [
            ("All dax are wugs. All wugs are green.", "Must every dax be green?", "yes"),
            ("No nims are tall. All zors are nims.", "Can any zor be tall under these rules?", "no"),
            ("All peks are round. Some round things are blue.", "Must some peks be blue?", "no"),
            ("Some glips are red. All red things are warm.", "Must at least one glip be warm?", "yes"),
        ][i % 4]
        add("deduction", f"Use only these premises: {premises} {question} Answer only yes or no.", answer, i + 1)

    for i in range(12):
        start, step = i + 2, i % 5 + 2
        seq = [start + step * j for j in range(5)]
        add("sequence", f"Find the next number in this arithmetic sequence: {', '.join(map(str, seq))}, ?. Return only the integer.", seq[-1] + step, i + 1)

    word_lists = [
        ["copper", "jade", "plum"], ["north", "west", "south", "east"],
        ["violet", "amber"], ["oak", "elm", "pine", "ash", "fir"],
        ["red", "cyan", "gold"], ["cat", "fox", "yak", "owl"],
        ["one", "four", "nine", "six"], ["alpha", "delta", "beta"],
        ["river", "cloud", "stone", "leaf"], ["tea", "rice"],
    ]
    for i, words in enumerate(word_lists, 1):
        add("instruction", f"Reverse the order of these items and output only the items separated by commas, with no explanation: {', '.join(words)}", ",".join(reversed(words)), i)

    names = ["Mira", "Tomas", "Nia", "Oren", "Pavel", "Suki", "Lena", "Idris", "Kira", "Bram", "Yuna", "Felix", "Zara", "Hugo", "Asha", "Noel"]
    colors = ["teal", "ochre", "indigo", "coral", "silver", "mint", "plum", "amber"]
    objects = ["compass", "lantern", "notebook", "key", "marble", "ribbon", "coin", "shell"]
    places = ["cabinet", "shelf", "drawer", "basket", "desk", "crate", "box", "pouch"]
    for i, name in enumerate(names):
        color, obj, place = colors[(i * 3 + 1) % 8], objects[(i * 5 + 2) % 8], places[(i * 3 + 4) % 8]
        decoys = [f"{names[(i+j+1)%16]} owns a {objects[(i+j+3)%8]}." for j in range(4)]
        facts = [f"{name} owns a {obj}.", f"The {obj} is stored in the {place}.", f"{name}'s assigned color is {color}."] + decoys
        if i % 2: facts.reverse()
        add("context_recall", "Read the facts carefully:\n" + "\n".join(facts) + f"\nQuestion: Where is the {obj} owned by {name} stored? Return only the place name.", place, i + 1)

    for i in range(12):
        initial, inc, dec = 17 + i * 3, i % 9 + 4, i % 5 + 1
        add("state_update", f"A counter starts at {initial}. First increase it by {inc}. Then decrease it by {dec}. What is the final value? Return only the integer.", initial + inc - dec, i + 1)
    return cases


def normalize(s):
    s = s.strip().lower()
    s = re.sub(r"<\|[^|]+\|>", " ", s)
    s = re.sub(r"^(?:answer|final answer)\s*:\s*", "", s)
    return re.sub(r"\s+", " ", s).strip().strip("'\"* ")


def matches(category, expected, output):
    out, exp = normalize(output), normalize(expected)
    if category == "instruction":
        return re.sub(r"\s+", "", out).strip(".,;") == re.sub(r"\s+", "", exp).strip(".,;")
    if category in ("arithmetic", "sequence", "state_update"):
        m = re.search(r"[-+]?\d+", out)
        return bool(m) and m.group(0) == exp and not re.search(r"[-+]?\d+\s*[/x*+]\s*[-+]?\d+", out)
    if category == "deduction":
        m = re.search(r"\b(yes|no)\b", out)
        return bool(m) and m.group(1) == exp
    return out.strip(" .,!?:;") == exp.strip(" .,!?:;")


def completion(base_url, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model or "local-model",
        "messages": [
            {"role": "system", "content": "Solve carefully. Follow the requested output format. Do not explain when asked for only an answer."},
            {"role": "user", "content": prompt}],
        "temperature": 0, "seed": 1234, "max_tokens": max_tokens, "stream": False
    }).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode())
    return data["choices"][0]["message"]["content"], time.perf_counter() - started, data.get("usage", {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="local-model")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    cases = build_cases()
    if args.limit: cases = cases[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "aimeng-llm-pk-report/v1", "model": args.model, "baseUrl": args.base_url,
        "caseCount": len(cases), "datasetSha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "decoding": {"temperature": 0, "seed": 1234, "maxTokens": args.max_tokens},
        "note": "Generated phase-one diagnostic benchmark, not a public leaderboard or proof of general intelligence."
    }
    rows = []
    for index, case in enumerate(cases, 1):
        row = dict(case)
        started = time.perf_counter()
        try:
            output, latency, usage = completion(args.base_url, args.model, case["prompt"], args.max_tokens, args.timeout)
            row.update({"output": output, "latencySeconds": round(latency, 4),
                        "correct": matches(case["category"], case["expected"], output), "usage": usage, "error": None})
        except Exception as exc:
            row.update({"output": "", "latencySeconds": round(time.perf_counter()-started, 4),
                        "correct": False, "usage": {}, "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
        print("LLM_PK_CASE " + json.dumps({"index": index, "id": row["id"], "category": row["category"],
              "correct": row["correct"], "latencySeconds": row["latencySeconds"], "error": row["error"],
              "output": row["output"][:240]}, ensure_ascii=False, sort_keys=True), flush=True)

    categories = {}
    for category in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == category]
        categories[category] = {"cases": len(subset), "correct": sum(bool(r["correct"]) for r in subset),
            "accuracy": sum(bool(r["correct"]) for r in subset)/len(subset),
            "meanLatencySeconds": statistics.mean(r["latencySeconds"] for r in subset),
            "errors": sum(r["error"] is not None for r in subset)}
    report = {**manifest,
        "overall": {"correct": sum(bool(r["correct"]) for r in rows),
            "accuracy": sum(bool(r["correct"]) for r in rows)/len(rows) if rows else 0,
            "meanLatencySeconds": statistics.mean(r["latencySeconds"] for r in rows) if rows else 0,
            "errors": sum(r["error"] is not None for r in rows)},
        "categories": categories, "results": rows}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("LLM_PK_SUMMARY " + json.dumps({"model": args.model, "caseCount": len(rows),
        "overall": report["overall"], "categories": categories, "datasetSha256": manifest["datasetSha256"],
        "output": str(args.output)}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
