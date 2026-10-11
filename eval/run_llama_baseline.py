#!/usr/bin/env python3
"""Reproducible llama.cpp mobile baseline runner. Python standard library only."""
import argparse
import hashlib
import json
import platform
import random
import re
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

PROMPT_TEMPLATE = (
    "You are completing an evaluation task. Answer the user's task directly. "
    "Do not claim to have used tools or external sources unless they were provided.\n\n"
    "MODE_INSTRUCTION:\n{thinking_instruction}\n\n"
    "TASK:\n{prompt}\n\nANSWER:"
)
SMOKE_TASK = {
    "task_id": "SMOKE-NOT-SCORED",
    "category": "smoke",
    "prompt": "Reply with exactly: LOCAL_INFERENCE_OK"
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_proc_status(pid):
    result = {}
    try:
        text = Path(f"/proc/{pid}/status").read_text(errors="replace")
        for key in ("VmRSS", "VmHWM", "VmPeak", "VmSize", "VmSwap"):
            m = re.search(rf"^{key}:\s+(\d+)\s+kB", text, re.M)
            if m:
                result[key + "_kib"] = int(m.group(1))
    except (OSError, ValueError):
        pass
    try:
        text = Path(f"/proc/{pid}/smaps_rollup").read_text(errors="replace")
        for key in ("Rss", "Pss", "Pss_Anon", "Pss_File", "Private_Clean",
                    "Private_Dirty", "SwapPss"):
            m = re.search(rf"^{key}:\s+(\d+)\s+kB", text, re.M)
            if m:
                result[key.lower() + "_kib"] = int(m.group(1))
    except (OSError, ValueError):
        result["smaps_rollup_unavailable"] = True
    return result


def read_system_memory():
    result = {}
    try:
        text = Path("/proc/meminfo").read_text(errors="replace")
        for key in ("MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached",
                    "SwapTotal", "SwapFree"):
            m = re.search(rf"^{key}:\s+(\d+)\s+kB", text, re.M)
            if m:
                result[key.lower() + "_kib"] = int(m.group(1))
    except (OSError, ValueError):
        result["available"] = False
    return result

def thermal_snapshot():
    readings = []
    roots = [Path("/sys/class/thermal"), Path("/sys/class/power_supply")]
    for root in roots:
        try:
            for p in root.glob("thermal_zone*/temp"):
                try:
                    value = int(p.read_text().strip())
                    # Android thermal zones usually use millidegrees C.
                    c = value / 1000.0 if abs(value) > 200 else float(value)
                    readings.append({"source": str(p), "celsius": round(c, 2)})
                except (OSError, ValueError):
                    continue
            for p in root.glob("*/temp"):
                if "power_supply" in str(p):
                    try:
                        value = int(p.read_text().strip())
                        if value > 100:
                            value /= 10.0
                        readings.append({"source": str(p), "celsius": round(value, 2)})
                    except (OSError, ValueError):
                        continue
        except OSError:
            pass
    return readings


def request_completion(url, prompt, n_predict, seed, timeout):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": seed,
        "cache_prompt": False,
        "stream": False
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
    obj = json.loads(body.decode("utf-8"))
    if not isinstance(obj, dict) or "content" not in obj:
        raise RuntimeError("llama.cpp /completion response missing 'content'")
    return obj


def load_tasks(path):
    raw = Path(path).read_bytes()
    tasks = []
    for lineno, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSONL at line {lineno}: {e}") from e
        if not isinstance(obj, dict) or not obj.get("task_id") or not obj.get("prompt"):
            raise ValueError(f"Line {lineno} must contain non-empty task_id and prompt")
        allowed = {"task_id", "category", "prompt"}
        extra = set(obj) - allowed
        if extra:
            raise ValueError(
                f"Line {lineno} has unsupported fields {sorted(extra)}; keep all scoring material sealed"
            )
        tasks.append(obj)
    if not tasks:
        raise ValueError("Task corpus is empty")
    ids = [str(t["task_id"]) for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("task_id values must be unique")
    return raw, tasks


def fetch_server_properties(url, timeout):
    """Best-effort capture of llama-server effective model/context properties."""
    req = urllib.request.Request(url.rstrip("/") + "/props", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=min(timeout, 15)) as response:
            return {"available": True, "http_status": response.status,
                    "body": json.loads(response.read().decode("utf-8"))}
    except Exception as exc:
        return {"available": False, "error": repr(exc)}



def split_visible_thinking(output):
    """Separate only explicit visible <think>...</think> text; hidden reasoning is not observable."""
    match = re.search(r"<think>(.*?)</think>", output, flags=re.I | re.S)
    if not match:
        return "", output.strip()
    thinking = match.group(1).strip()
    answer = (output[:match.start()] + output[match.end():]).strip()
    return thinking, answer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True, help="Loopback llama-server URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--server-pid", type=int, required=True, help="PID of llama-server for /proc memory sampling")
    ap.add_argument("--model-file", required=True, help="Exact local model file, normally GGUF")
    ap.add_argument("--tasks", help="Private JSONL live task set; omitted in smoke mode")
    ap.add_argument("--out", default="baseline_result.json")
    ap.add_argument("--blind-out", default="blind_outputs.jsonl")
    ap.add_argument("--mapping-out", default="blind_mapping.json")
    ap.add_argument("--offline-confirmed", action="store_true", help="Operator confirms airplane mode; external probes must also fail")
    ap.add_argument("--smoke", action="store_true", help="Run only a non-scoring smoke prompt")
    ap.add_argument("--thinking-mode", choices=("off", "on", "model-default"), default="off",
                    help="Fixed visible-thinking protocol for all candidates; recorded in result")
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    parsed = urlparse(args.url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        ap.error("Only an HTTP loopback URL is accepted; bind llama-server to localhost")
    if not args.offline_confirmed:
        ap.error("Refusing to run until --offline-confirmed is supplied after disabling network access")
    offline_probe = verify_offline()
    if not offline_probe["passed"]:
        ap.error("External TCP reachability detected; refusing to score. Disable all network access and retry.")
    if args.n_predict < 1 or args.timeout < 1:
        ap.error("--n-predict and --timeout must be positive")
    model_path = Path(args.model_file)
    if not model_path.is_file():
        ap.error(f"Model file not found: {model_path}")
    if not args.smoke and not args.tasks:
        ap.error("--tasks is required unless --smoke is set")

    started_utc = datetime.now(timezone.utc).isoformat()
    # Hash before timed inference; this may read several GB and warm the filesystem cache.
    model_hash = sha256_file(model_path)
    server_properties = fetch_server_properties(args.url, args.timeout)
    if args.smoke:
        tasks = [SMOKE_TASK]
        corpus_bytes = json.dumps(SMOKE_TASK, ensure_ascii=False, sort_keys=True).encode("utf-8")
        corpus_hash = sha256_bytes(corpus_bytes)
    else:
        corpus_bytes, tasks = load_tasks(args.tasks)
        corpus_hash = sha256_bytes(corpus_bytes)

    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        git_rev, dirty = None, None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    rng = random.SystemRandom()
    id_map = {}
    blind_rows = []
    measurements = []
    max_rss = 0
    max_hwm = 0
    temp_samples = []
    errors = []

    for index, task in enumerate(tasks):
        blind_id = "CAND-" + uuid.uuid4().hex[:10].upper()
        id_map[blind_id] = {"task_id": str(task["task_id"]), "run_id": run_id}
        thinking_instruction = {
            "off": "Do not emit a reasoning trace or <think> block. Return only the final answer.",
            "on": "If you emit visible reasoning, place it in <think>...</think>, then give the final answer after </think>.",
            "model-default": "Use the configured model default thinking behavior."
        }[args.thinking_mode]
        prompt = PROMPT_TEMPLATE.format(
            prompt=str(task["prompt"]), thinking_instruction=thinking_instruction
        )
        before = read_proc_status(args.server_pid)
        system_before = read_system_memory()
        temps_before = thermal_snapshot()
        started = time.perf_counter()
        try:
            response = request_completion(args.url, prompt, args.n_predict, args.seed + index, args.timeout)
            elapsed = time.perf_counter() - started
            output = str(response.get("content", ""))
            timings = response.get("timings") if isinstance(response.get("timings"), dict) else {}
            predicted_tokens = timings.get("predicted_n")
            if not isinstance(predicted_tokens, (int, float)) or predicted_tokens <= 0:
                predicted_tokens = None
            words = len(re.findall(r"\S+", output))
            chars = len(output)
            words_per_second = words / elapsed if elapsed > 0 else None
            row = {
                "blind_id": blind_id,
                "category": str(task.get("category", "unspecified")),
                "output": output,
                "output_sha256": sha256_bytes(output.encode("utf-8")),
            }
            blind_rows.append(row)
            after = read_proc_status(args.server_pid)
            system_after = read_system_memory()
            temps_after = thermal_snapshot()
            for snap in (before, after):
                max_rss = max(max_rss, snap.get("VmRSS_kib", 0))
                max_hwm = max(max_hwm, snap.get("VmHWM_kib", 0))
            temp_samples.extend(temps_before + temps_after)
            measurements.append({
                "task_id": str(task["task_id"]),
                "blind_id": blind_id,
                "status": "ok",
                "wall_time_seconds": round(elapsed, 4),
                "output_chars": chars,
                "thinking_chars": len(thinking_text),
                "answer_chars": len(answer_text),
                "llama_cpp_predicted_tokens": predicted_tokens,
                "server_generated_tokens_per_second": server_tokens_per_second,
                "llama_cpp_timings": timings,
                "process_before": before,
                "process_after": after,
                "system_memory_before": system_before,
                "system_memory_after": system_after,
                "thermal_before": temps_before,
                "thermal_after": temps_after,
                "output_sha256": row["output_sha256"],
            })
        except Exception as exc:
            elapsed = time.perf_counter() - started
            errors.append({"task_id": str(task["task_id"]), "error": repr(exc)})
            measurements.append({
                "task_id": str(task["task_id"]),
                "blind_id": blind_id,
                "status": "error",
                "wall_time_seconds": round(elapsed, 4),
                "process_before": before,
                "system_memory_before": system_before,
                "error": repr(exc),
            })

    # Shuffle the blind rows independently of task order; mapping stays in a separate private file.
    rng.shuffle(blind_rows)
    Path(args.blind_out).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind_rows),
        encoding="utf-8"
    )
    Path(args.mapping_out).write_text(
        json.dumps(id_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "schema": "mobile-llama-cpp-baseline/v1",
        "run_id": run_id,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "status": ("smoke_only" if args.smoke and not errors else
                   "smoke_failed" if args.smoke and errors else
                   "complete" if not errors else "partial_with_errors"),
        "non_scoring_smoke": bool(args.smoke),
        "smoke_expected_marker_found": (
            any("LOCAL_INFERENCE_OK" in row.get("output", "") for row in blind_rows)
            if args.smoke else None
        ),
        "offline_operator_attestation": True,
        "offline_external_probe": offline_probe,
        "offline_verification_method": "operator confirmation plus failed outbound TCP probes to 1.1.1.1:443 and 8.8.8.8:443; reachability check, not proof every network path is blocked",
        "runtime": {
            "endpoint": args.url,
            "server_pid": args.server_pid,
            "n_predict": args.n_predict,
            "temperature": 0.0,
            "thinking_mode": args.thinking_mode,
            "n_predict_output_cap": args.n_predict,
            "seed_base": args.seed,
            "seed_policy": "seed_base + zero-based task index",
            "timeout_seconds": args.timeout,
            "python": sys.version,
            "platform": platform.platform(),
            "llama_server_properties": server_properties,
        },
        "model": {
            "path": str(model_path),
            "file_size_bytes": model_path.stat().st_size,
            "sha256": model_hash,
            "format": model_path.suffix.lower().lstrip("."),
            "identity_warning": "Record model repository, conversion revision and quantization separately; filename/hash alone does not prove it is the reference Ornith model."
        },
        "corpus": {
            "path": None if args.smoke else str(Path(args.tasks)),
            "sha256_raw_bytes": corpus_hash,
            "task_count": len(tasks),
            "task_ids": [str(t["task_id"]) for t in tasks],
        },
        "code": {"git_revision": git_rev, "working_tree_dirty": dirty},
        "instrumentation": {
            "max_sampled_server_rss_kib": max_rss or None,
            "max_sampled_server_hwm_kib": max_hwm or None,
            "last_system_memory_snapshot": read_system_memory(),
            "thermal_readings_collected": len(temp_samples),
            "thermal_readings_celsius": temp_samples,
            "note": "Process RSS/PSS/HWM and system memory are sampled before/after each request, not continuously. System memory includes other processes; process PSS is proportional attribution. Thermal readings depend on Android sysfs permissions."
        },
        "summary": {
            "ok_tasks": sum(1 for x in measurements if x["status"] == "ok"),
            "failed_tasks": sum(1 for x in measurements if x["status"] != "ok"),
            "total_wall_time_seconds": round(sum(x["wall_time_seconds"] for x in measurements), 4),
        },
        "tasks": measurements,
        "errors": errors,
        "artifacts": {"blind_outputs": args.blind_out, "private_mapping": args.mapping_out},
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "run_id": run_id,
        "task_count": len(tasks),
        "ok": result["summary"]["ok_tasks"],
        "failed": result["summary"]["failed_tasks"],
        "model_sha256": model_hash,
        "corpus_sha256": corpus_hash,
        "result_file": args.out,
        "blind_outputs": args.blind_out,
        "private_mapping": args.mapping_out,
    }, ensure_ascii=False, indent=2))
    if args.smoke and not result["smoke_expected_marker_found"]:
        return 2
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
