# D evaluation harness: offline mobile llama.cpp baseline

This directory provides a reproducible *measurement harness*, not a claim that the phone baseline has already run. Real on-device measurements must be generated on the target phone.

## Important model identity caveat

The project's documented reference is `ornith-ai/Ornith-1.5-9B-MLX-4bit` (MLX/Safetensors), not GGUF. llama.cpp does not directly run MLX/Safetensors as though it were GGUF. Before calling this a fair Ornith-1.5-9B baseline, record the exact GGUF source/conversion and quantization. Do not label an unrelated or differently quantized model as identical. Compare quality and performance only with that provenance disclosed.

## Pre-registered rules

1. Smoke test: run the included non-scoring smoke prompt once and verify the server responds, JSON output is valid, and instrumentation is present. Do not count it in task scores.
2. Main test: use a task set created after the protocol is frozen. Keep its prompts and expected answers out of training data. Record the SHA-256 of the exact JSONL corpus.
3. Use the same prompt template, generation cap, sampling settings, context settings, and task order for every candidate. Set temperature to 0 for deterministic baseline runs where supported.
4. Run offline: enable airplane mode / disable Wi-Fi and mobile data on the phone, then run the server bound to loopback only. The harness refuses non-loopback URLs and records the operator's offline attestation. This is a protocol check, not a technical firewall.
5. Blind scoring: this tool emits randomized candidate IDs and a separate ID mapping file. Give only the blind output file to the scorer; keep the mapping file private until scoring is locked.
6. Predeclare pass criteria in the task corpus or a separate sealed rubric. The model receives only the task prompt, never the rubric/reference.
7. Record actual runtime config, corpus/model hashes, source revision, per-task wall time, output length, approximate words/sec, process RSS/HWM, and available thermal readings.
8. One smoke run first. Do not change seed count or task set after seeing results. Failed/incomplete runs are recorded, not silently dropped.
9. The harness cannot enforce Android-wide network isolation. Offline status must be independently verified by the operator. Do not claim offline proof from loopback binding alone.

## Requirements

- Python 3.9+
- llama.cpp `llama-server` running on the phone and bound to `127.0.0.1`, with a local model loaded.
- Keep process RSS below the project hard limit of 4 GiB. This script samples the llama-server process only; inspect total app/process memory separately if other processes own model buffers.

## Smoke run

Start llama.cpp server locally, for example with a loopback-only bind. Then run:

```sh
python run_llama_baseline.py --url http://127.0.0.1:8080 --server-pid <PID> --model-file /path/to/model.gguf --offline-confirmed --smoke
```

The smoke prompt is excluded from the scored corpus.

## Main run

Create a UTF-8 JSONL task file *on the phone or via a private transfer*. Each line is an object:

```json
{"task_id":"T001","category":"math","prompt":"..."}
```

Do not put references or rubrics in the task file supplied to the model. Keep a separate sealed scoring file keyed by task_id. Then run:

```sh
python run_llama_baseline.py --url http://127.0.0.1:8080 --server-pid <PID> --model-file /path/to/model.gguf --tasks /private/live_tasks.jsonl --out /private/run.json --blind-out /private/blind_outputs.jsonl --mapping-out /private/blind_mapping.json --offline-confirmed --n-predict 512 --seed 20261011
```

The script writes a result JSON containing effective settings and per-task measurements. Protect the mapping file until blind scoring is complete. The script does not automatically grade correctness; the human/automated scorer must apply the pre-registered rubric after outputs are frozen.
