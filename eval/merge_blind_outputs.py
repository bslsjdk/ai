#!/usr/bin/env python3
"""Merge separate runner outputs into one randomized, answer-only blind scoring set."""
import argparse
import json
import random
from pathlib import Path


def read_jsonl(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict) or not row.get("blind_id") or not isinstance(row.get("answer_text"), str):
            raise ValueError(f"{path}:{line_number}: expected blind_id and answer_text")
        rows.append(row)
    return rows


def read_mapping(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: mapping must be a JSON object")
    return obj


def parse_pair(value, option):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{option} must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"{option} must be LABEL=PATH")
    return label.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", action="append", required=True, type=lambda x: parse_pair(x, "--input"),
                    help="Repeat per model/run: LABEL=blind_outputs.jsonl")
    ap.add_argument("--mapping", action="append", required=True, type=lambda x: parse_pair(x, "--mapping"),
                    help="Repeat per matching model/run: LABEL=blind_mapping.json")
    ap.add_argument("--out", default="merged_blind_scoring.jsonl")
    ap.add_argument("--private-map-out", default="merged_private_mapping.json")
    ap.add_argument("--seed", type=int, default=20261011)
    args = ap.parse_args()

    inputs = dict(args.input)
    mappings = dict(args.mapping)
    if len(inputs) != len(args.input) or len(mappings) != len(args.mapping):
        ap.error("Each LABEL must be unique within --input and --mapping")
    if set(inputs) != set(mappings):
        ap.error("Each --input label must have exactly one matching --mapping label")
    if len(inputs) < 2:
        ap.error("At least two independent model/run outputs are required for meaningful blinding")

    by_task = {}
    expected_task_ids = set()
    seen = set()
    for label, blind_path in inputs.items():
        mapping = read_mapping(mappings[label])
        for entry in mapping.values():
            if not isinstance(entry, dict) or not str(entry.get("task_id", "")):
                raise ValueError(f"{label}: every mapping entry must contain a task_id")
            expected_task_ids.add(str(entry["task_id"]))
        rows = read_jsonl(blind_path)
        for row in rows:
            blind_id = row["blind_id"]
            if blind_id not in mapping:
                raise ValueError(f"{label}: blind ID {blind_id} missing from mapping file")
            task_id = str(mapping[blind_id].get("task_id", ""))
            if not task_id:
                raise ValueError(f"{label}: mapping for {blind_id} has no task_id")
            key = (label, task_id)
            if key in seen:
                raise ValueError(f"{label}: duplicate answer for task {task_id}")
            seen.add(key)
            by_task.setdefault(task_id, []).append({"label": label, "answer_text": row["answer_text"]})

    labels = set(inputs)
    # The per-run mapping includes attempted tasks even when inference failed.
    # Validate against it so a task missing from every blind output cannot vanish
    # silently from the scored corpus.
    for task_id in sorted(expected_task_ids):
        candidates = by_task.get(task_id, [])
        found = {x["label"] for x in candidates}
        if found != labels:
            raise ValueError(
                f"Task {task_id} is not present for every candidate: {sorted(found)}; "
                "check failed/incomplete runs before scoring"
            )

    rng = random.Random(args.seed)
    task_ids = list(by_task)
    rng.shuffle(task_ids)
    merged_rows = []
    private_map = {}
    counter = 1
    for task_id in task_ids:
        candidates = list(by_task[task_id])
        rng.shuffle(candidates)
        for candidate in candidates:
            candidate_id = f"CAND-{counter:06d}"
            counter += 1
            merged_rows.append({
                "task_id": task_id,
                "candidate_id": candidate_id,
                "answer_text": candidate["answer_text"],
            })
            private_map[candidate_id] = {
                "task_id": task_id,
                "source_label": candidate["label"],
            }

    Path(args.out).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged_rows),
        encoding="utf-8",
    )
    Path(args.private_map_out).write_text(
        json.dumps({"seed": args.seed, "candidate_map": private_map}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "task_count": len(task_ids),
        "candidate_count": len(merged_rows),
        "source_count": len(labels),
        "blind_scoring_file": args.out,
        "private_mapping_file": args.private_map_out,
        "warning": "Give only the scoring file to the scorer; keep the private map sealed until scores are locked.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
