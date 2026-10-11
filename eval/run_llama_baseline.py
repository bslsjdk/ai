#!/usr/bin/env python3
"""Offline, reproducible llama.cpp chat baseline runner. Python standard library only."""
import argparse
import hashlib
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

SMOKE_TASK = {
    "task_id": "SMOKE-NOT-SCORED",
    "category": "smoke",
    "prompt": "Reply with exactly: LOCAL_INFERENCE_OK",
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


def verify_offline(timeout=2.0):
    """Abort scoring if either direct external TCP probe succeeds."""
    attempts = []
    for host, port in (("1.1.1.1", 443), ("8.8.8.8", 443)):
        started = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                attempts.append({"host": host, "port": port, "reachable": True,
                                 "elapsed_seconds": round(time.perf_counter() - started, 4)})
        except OSError as exc:
            attempts.append({"host": host, "port": port, "reachable": False,
                             "error": repr(exc),
                             "elapsed_seconds": round(time.perf_counter() - started, 4)})
    return {"passed": all(not x["reachable"] for x in attempts), "attempts": attempts}


def process_cmdline(pid):
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    except OSError:
        return None


def battery_snapshot():
    readings = []
    root = Path("/sys/class/power_supply")
    try:
        for device in root.iterdir():
            row = {"source": str(device)}
            for field in ("capacity", "status", "voltage_now", "current_now", "temp"):
                try:
                    value = (device / field).read_text().strip()
                    row[field] = value
                except OSError:
                    continue
            if len(row) > 1:
                readings.append(row)
    except OSError:
        pass
    return readings


def thermal_snapshot():
    readings = []
    for root in (Path("/sys/class/thermal"), Path("/sys/class/power_supply")):
        try:
            for p in root.glob("thermal_zone*/temp"):
                try:
                    value = int(p.read_text().strip())
                    celsius = value / 1000.0 if abs(value) > 200 else float(value)
                    readings.append({"source": str(p), "celsius": round(celsius, 2)})
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


def sample_during_request(pid, stop_event, samples, interval=0.5):
    """Background sampler; sample failures are recorded rather than fatal."""
    while not stop_event.is_set():
        now = time.time()
        samples.append({
            "timestamp_unix": now,
            "process": read_proc_status(pid),
            "system": read_system_memory(),
            "thermal": thermal_snapshot(),
            "battery": battery_snapshot(),
        })
        stop_event.wait(interval)


def request_chat(url, prompt, n_predict, seed, timeout, thinking_mode):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict,
        "temperature": 0.0,
        "seed": seed,
        "stream": False,
    }
    # Let the model's embedded chat template render the conversation. For templates
    # supporting enable_thinking, this explicitly fixes the mode; unsupported kwargs
    # must fail visibly rather than silently reverting to another protocol.
    if thinking_mode != "model-default":
        payload["chat_template_kwargs"] = {"enable_thinking": thinking_mode == "on"}
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        obj = json.loads(response.read().decode("utf-8"))
    try:
        choice = obj["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("chat completion response missing choices[0].message") from exc
    if not isinstance(message, dict):
        raise RuntimeError("chat completion message is not an object")
    return obj, message, choice.get("finish_reason")


def load_tasks(path):
    raw = Path(path).read_bytes()
    tasks = []
    for lineno, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {lineno}: {exc}") from exc
        if not isinstance(obj, dict) or not obj.get("task_id") or not obj.get("prompt"):
            raise ValueError(f"Line {lineno} must contain non-empty task_id and prompt")
        extra = set(obj) - {"task_id", "category", "prompt"}
        if extra:
            raise ValueError(f"Line {lineno} has unsupported fields {sorted(extra)}; keep scoring material sealed")
        tasks.append(obj)
    if not tasks:
        raise ValueError("Task corpus is empty")
    ids = [str(t["task_id"]) for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("task_id values must be unique")
    return raw, tasks


def fetch_server_properties(url, timeout):
    req = urllib.request.Request(url.rstrip("/") + "/props", method="GET")
    with urllib.request.urlopen(req, timeout=min(timeout, 15)) as response:
        return {"available": True, "http_status": response.status,
                "body": json.loads(response.read().decode("utf-8"))}


def normalize_path(value):
    if not isinstance(value, str) or not value.strip():
        return None
    return os.path.realpath(os.path.expanduser(value.strip()))


def server_model_path(properties):
    body = properties.get("body", {}) if isinstance(properties, dict) else {}
    for key in ("model_path", "modelPath", "model"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value
    # Some builds nest effective properties.
    for parent in ("props", "model"):
        nested = body.get(parent)
        if isinstance(nested, dict):
            for key in ("model_path", "modelPath", "path"):
                value = nested.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    return None


def split_visible_thinking(output):
    """Separate explicit visible think tags; this does not expose hidden reasoning."""
    match = re.search(r"<think>(.*?)</think>", output, flags=re.I | re.S)
    if not match:
        return "", output.strip()
    thinking = match.group(1).strip()
    answer = (output[:match.start()] + output[match.end():]).strip()
    return thinking, answer


def is_output_cap_reached(finish_reason, completion_tokens, predicted_tokens, cap):
    """Use the server finish reason first, with token counts as fallback signals."""
    return (
        finish_reason == "length"
        or (isinstance(completion_tokens, (int, float)) and completion_tokens >= cap)
        or (isinstance(predicted_tokens, (int, float)) and predicted_tokens >= cap)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True, help="Loopback llama-server URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--server-pid", type=int, required=True, help="llama-server PID for memory sampling")
    ap.add_argument("--model-file", required=True, help="Exact local model file, normally GGUF")
    ap.add_argument("--model-source", default="ornith-ai/Ornith-1.5-9B-GGUF",
                    help="Declared model repository/source for provenance")
    ap.add_argument("--model-revision", default="9a5e9dc",
                    help="Declared immutable source revision used to obtain the file")
    ap.add_argument("--tasks", help="Private JSONL task set; omitted in smoke mode")
    ap.add_argument("--out", default="baseline_result.json")
    ap.add_argument("--blind-out", default="blind_outputs.jsonl")
    ap.add_argument("--mapping-out", default="blind_mapping.json")
    ap.add_argument("--offline-confirmed", action="store_true", help="Operator confirms airplane mode")
    ap.add_argument("--smoke", action="store_true", help="Run one non-scoring smoke prompt")
    ap.add_argument("--thinking-mode", choices=("off", "on", "model-default"), default="off")
    ap.add_argument("--n-predict", type=int, default=2048,
                    help="Maximum generated tokens; default raised to reduce reasoning truncation")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    parsed = urlparse(args.url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        ap.error("Only an HTTP loopback URL is accepted; bind llama-server to localhost")
    if not args.offline_confirmed:
        ap.error("Refusing to run until --offline-confirmed is supplied after disabling network access")
    offline_probe = verify_offline()
    if not offline_probe["passed"]:
        ap.error("External TCP reachability detected; refusing to run. Disable all network access and retry.")
    if args.n_predict < 1 or args.timeout < 1 or args.server_pid < 1:
        ap.error("--n-predict, --timeout and --server-pid must be positive")
    model_path = Path(args.model_file).expanduser().resolve()
    if not model_path.is_file():
        ap.error(f"Model file not found: {model_path}")
    if not args.smoke and not args.tasks:
        ap.error("--tasks is required unless --smoke is set")

    started_utc = datetime.now(timezone.utc).isoformat()
    model_hash = sha256_file(model_path)
    server_properties = fetch_server_properties(args.url, args.timeout)
    loaded_path_raw = server_model_path(server_properties)
    expected_path = normalize_path(str(model_path))
    loaded_path = normalize_path(loaded_path_raw)
    if not loaded_path:
        ap.error("Cannot verify loaded model path from /props; refusing to run. Save /props output and inspect the llama.cpp build.")
    if loaded_path != expected_path:
        ap.error(f"Loaded model path mismatch: --model-file resolves to {expected_path!r}, /props reports {loaded_path_raw!r} (resolved {loaded_path!r})")

    if args.smoke:
        tasks = [SMOKE_TASK]
        corpus_bytes = json.dumps(SMOKE_TASK, ensure_ascii=False, sort_keys=True).encode("utf-8")
        corpus_hash = sha256_bytes(corpus_bytes)
    else:
        corpus_bytes, tasks = load_tasks(args.tasks)
        corpus_hash = sha256_bytes(corpus_bytes)

    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        git_rev, dirty = None, None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    id_map = {}
    blind_rows = []
    measurements = []
    all_samples = []
    errors = []

    for index, task in enumerate(tasks):
        blind_id = "RUN-" + uuid.uuid4().hex[:10].upper()
        id_map[blind_id] = {"task_id": str(task["task_id"]), "run_id": run_id}
        samples = []
        stop_sampler = threading.Event()
        sampler = threading.Thread(
            target=sample_during_request,
            args=(args.server_pid, stop_sampler, samples, 0.5),
            daemon=True,
        )
        before = read_proc_status(args.server_pid)
        system_before = read_system_memory()
        thermal_before = thermal_snapshot()
        battery_before = battery_snapshot()
        started = time.perf_counter()
        sampler.start()
        try:
            response, message, finish_reason = request_chat(
                args.url, str(task["prompt"]), args.n_predict, args.seed + index,
                args.timeout, args.thinking_mode,
            )
            elapsed = time.perf_counter() - started
            stop_sampler.set()
            sampler.join(timeout=2)
            all_samples.extend(samples)
            content = message.get("content", "")
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            reasoning = message.get("reasoning_content", "")
            if not isinstance(reasoning, str):
                reasoning = ""
            tagged_thinking, answer_text = split_visible_thinking(content)
            thinking_text = "\n".join(x for x in (reasoning.strip(), tagged_thinking) if x)
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            completion_tokens = usage.get("completion_tokens")
            timings = response.get("timings") if isinstance(response.get("timings"), dict) else {}
            predicted_tokens = timings.get("predicted_n", completion_tokens)
            tokens_per_second = timings.get("predicted_per_second")
            cap_reached = is_output_cap_reached(
                finish_reason, completion_tokens, predicted_tokens, args.n_predict
            )
            # The blind file intentionally contains answer text only, with no tags,
            # token counts, lengths, timing, model name, or run metadata.
            blind_rows.append({"blind_id": blind_id, "answer_text": answer_text})
            after = read_proc_status(args.server_pid)
            system_after = read_system_memory()
            thermal_after = thermal_snapshot()
            measurements.append({
                "task_id": str(task["task_id"]), "blind_id": blind_id, "status": "ok",
                "wall_time_seconds": round(elapsed, 4),
                "finish_reason": finish_reason,
                "output_cap_reached": bool(cap_reached),
                "completion_status": "possibly_truncated_at_cap" if cap_reached else "not_cap_truncated",
                "output_chars": len(content), "thinking_chars": len(thinking_text),
                "answer_chars": len(answer_text),
                "completion_tokens": completion_tokens,
                "llama_cpp_predicted_tokens": predicted_tokens,
                "server_generated_tokens_per_second": tokens_per_second,
                "llama_cpp_timings": timings,
                "process_before": before, "process_after": after,
                "system_memory_before": system_before, "system_memory_after": system_after,
                "thermal_before": thermal_before, "thermal_after": thermal_after,
                "battery_before": battery_before, "battery_after": battery_snapshot(),
                "sample_count_during_request": len(samples),
            })
        except Exception as exc:
            elapsed = time.perf_counter() - started
            stop_sampler.set()
            sampler.join(timeout=2)
            all_samples.extend(samples)
            errors.append({"task_id": str(task["task_id"]), "error": repr(exc)})
            measurements.append({
                "task_id": str(task["task_id"]), "blind_id": blind_id, "status": "error",
                "wall_time_seconds": round(elapsed, 4), "process_before": before,
                "system_memory_before": system_before, "battery_before": battery_before,
                "sample_count_during_request": len(samples),
                "error": repr(exc),
            })

    random.SystemRandom().shuffle(blind_rows)
    Path(args.blind_out).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind_rows),
        encoding="utf-8",
    )
    Path(args.mapping_out).write_text(json.dumps(id_map, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")

    max_value = lambda key: max((s["process"].get(key, 0) for s in all_samples), default=0) or None
    max_rss = max_value("VmRSS_kib")
    max_hwm = max_value("VmHWM_kib")
    max_pss = max_value("pss_kib")
    max_system_mem_available = min(
        (s["system"].get("memavailable_kib") for s in all_samples
         if isinstance(s["system"].get("memavailable_kib"), int)),
        default=None,
    )
    cap_count = sum(1 for x in measurements if x.get("output_cap_reached"))
    result = {
        "schema": "mobile-llama-cpp-chat-baseline/v2",
        "run_id": run_id,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "status": ("smoke_only" if args.smoke and not errors else
                   "smoke_failed" if args.smoke and errors else
                   "complete" if not errors else "partial_with_errors"),
        "non_scoring_smoke": bool(args.smoke),
        "smoke_expected_marker_found": (
            any("LOCAL_INFERENCE_OK" in row["answer_text"] for row in blind_rows) if args.smoke else None
        ),
        "offline_operator_attestation": True,
        "offline_external_probe": offline_probe,
        "offline_verification_method": "operator confirmation plus failed outbound TCP probes to 1.1.1.1:443 and 8.8.8.8:443; this does not prove all network paths are blocked",
        "runtime": {
            "endpoint": args.url, "server_pid": args.server_pid,
            "server_process_cmdline": process_cmdline(args.server_pid),
            "n_predict": args.n_predict, "temperature": 0.0,
            "thinking_mode": args.thinking_mode,
            "chat_api": "/v1/chat/completions",
            "chat_template_policy": "server/model embedded chat template; enable_thinking passed as chat_template_kwargs when mode is off/on",
            "seed_base": args.seed, "seed_policy": "seed_base + zero-based task index",
            "timeout_seconds": args.timeout, "python": sys.version,
            "platform": platform.platform(), "llama_server_properties": server_properties,
            "verified_server_model_path": loaded_path_raw,
        },
        "model": {
            "path": str(model_path), "file_size_bytes": model_path.stat().st_size,
            "sha256": model_hash, "format": model_path.suffix.lower().lstrip("."),
            "declared_source": args.model_source,
            "declared_revision": args.model_revision,
            "identity_warning": "The hash identifies this local file; provenance/revision/quantization must also be recorded.",
        },
        "corpus": {
            "path": None if args.smoke else str(Path(args.tasks)),
            "sha256_raw_bytes": corpus_hash, "task_count": len(tasks),
            "task_ids": [str(t["task_id"]) for t in tasks],
        },
        "code": {"git_revision": git_rev, "working_tree_dirty": dirty},
        "instrumentation": {
            "sampling_interval_seconds": 0.5,
            "sample_count": len(all_samples),
            "max_sampled_server_rss_kib": max_rss,
            "max_sampled_server_hwm_kib": max_hwm,
            "max_sampled_server_pss_kib": max_pss,
            "min_sampled_system_memavailable_kib": max_system_mem_available,
            "last_system_memory_snapshot": read_system_memory(),
            "battery_snapshot_at_end": battery_snapshot(),
            "samples": all_samples,
            "note": "Process metrics are sampled every ~0.5s during requests plus before/after. Sampling can miss shorter peaks; process PSS is not total app memory if buffers are owned elsewhere.",
        },
        "summary": {
            "ok_tasks": sum(1 for x in measurements if x["status"] == "ok"),
            "failed_tasks": sum(1 for x in measurements if x["status"] != "ok"),
            "cap_reached_tasks": cap_count,
            "total_wall_time_seconds": round(sum(x["wall_time_seconds"] for x in measurements), 4),
            "correctness": "not scored by this runner; cap-reached/incomplete answers must be distinguished from incorrect answers during rubric scoring",
        },
        "tasks": measurements, "errors": errors,
        "artifacts": {"blind_outputs": args.blind_out, "private_mapping": args.mapping_out},
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "run_id": run_id, "task_count": len(tasks),
        "ok": result["summary"]["ok_tasks"], "failed": result["summary"]["failed_tasks"],
        "cap_reached": cap_count, "model_sha256": model_hash, "corpus_sha256": corpus_hash,
        "max_sampled_server_pss_kib": max_pss, "result_file": args.out,
        "blind_outputs": args.blind_out, "private_mapping": args.mapping_out,
    }, ensure_ascii=False, indent=2))
    if args.smoke and not result["smoke_expected_marker_found"]:
        return 2
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
