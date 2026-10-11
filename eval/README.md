# D evaluation harness: offline mobile llama.cpp baseline

This directory contains a reproducible measurement harness. It does **not** claim a phone run has occurred until a real device result JSON is saved and reviewed.

## Primary comparator: Ornith-1.5-9B

**Ornith-1.5-9B is the primary reference opponent, not an optional afterthought.** The current RNN/MGU public-dataset experiment is a separate architecture diagnostic; its loss numbers do not establish that a small model beats Ornith on tasks. Until the same phone task set has been run and scored for both sides, the PK result is **not yet measured**.

The comparison is deliberately staged:

1. **Reference feasibility:** run the non-scoring smoke test on the exact Q4_K_M artifact below. Record whether it loads and completes inference, plus model hash, server/build settings, latency, peak sampled memory, thermal and battery observations. If it cannot run, preserve the failure as a reference result; do not quietly replace the model or call a smaller model the winner.
2. **Freeze the shared task set and rubric:** include held-out tasks that test instruction following, multi-step reasoning, state tracking/memory, planning, and recovery from a wrong turn. Keep prompts identical, task order frozen, scoring key sealed, and task set SHA-256 recorded. Do not train on the held-out tasks.
3. **Run both sides:** score Ornith and each candidate small model on that exact task set. Use blind answer files and a sealed model map. Keep failed tasks, incomplete outputs, and cap-hit outputs visible in the report.
4. **Report a vector, not a single vanity score:** task success/quality, failure rate, latency (including median and p95), model/runtime memory, stability, and parameter/file size. The candidate model must stay within the project's 4 GiB runtime-RAM budget. Ornith is the reference and is measured honestly; do not assume its runtime will fit the candidate budget.
5. **Tune only after the baseline:** architecture or decoding changes must be evaluated on a development split first. Re-run the frozen held-out comparison after selecting changes; never tune against the sealed answers.

The exact model build, quantization, context length, GPU/offload settings, thread count, prompt format, and any unsupported sampling options must be recorded. Where the two models cannot use identical runtime settings because of different templates or backends, disclose the difference rather than pretending the runs are perfectly identical.

## Frozen reference artifact

Use this exact official GGUF artifact for the first smoke test:

- Repository: [ornith-ai/Ornith-1.5-9B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF)
- File: `Ornith-1.5-9B-Q4_K_M.gguf`
- Quantization: Q4_K_M
- Published file size: approximately 5.78 GB
- License shown by the repository: MIT
- File commit/revision shown on the artifact page: `9a5e9dc`
- Direct file page: [Ornith-1.5-9B-Q4_K_M.gguf](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF/blob/9a5e9dc/Ornith-1.5-9B-Q4_K_M.gguf)

The GGUF is the selected *phone-compatible artifact*, not the MLX/Safetensors artifact. The runner defaults the declared source/revision to this repository and file commit, and computes the exact local SHA-256 before inference. The local hash is the definitive check that the downloaded bytes are the same; retain it with the result.

A 5.78 GB file is larger than the project's **4 GiB runtime budget for candidate small models**. That budget is not a hardware RAM limit and is not automatically imposed on this 9B reference opponent. Report whether the reference can run stably on the phone, with its actual peak memory and settings; never claim that file size alone proves successful loading or stable inference.

## Why the runner uses chat completions

The runner calls `/v1/chat/completions` with a user message, allowing llama.cpp to apply the model's embedded chat template. It no longer injects a hand-written pseudo-chat template into `/completion`.

- `--thinking-mode off` sends `chat_template_kwargs: {"enable_thinking": false}`.
- `--thinking-mode on` sends `chat_template_kwargs: {"enable_thinking": true}`.
- `--thinking-mode model-default` omits that override.

This depends on the installed llama.cpp build and the model template supporting that keyword. If the server rejects it, do not silently fall back; record the error and resolve compatibility before comparing models. For the initial Ornith smoke run, use `off` to validate ordinary answer extraction, then run a separate non-scoring smoke with `on` if testing visible reasoning. Fix one mode for all scored candidates.

The server's `/props` must report a model path matching `--model-file` after path normalization. If the path is absent or differs, the runner aborts. The local file is SHA-256 hashed before timed inference.

## Pre-registered rules

1. Run a single non-scoring smoke test first. It checks API response, exact model-path agreement, output marker, and instrumentation.
2. Create the live task set only after the protocol is frozen. Keep prompts and expected answers out of training data; hash the exact JSONL bytes.
3. Keep generation settings, context settings, task set, and task order fixed across candidates. Temperature is set to 0.0 for the initial deterministic baseline. If the model's documented recommended sampling settings are later chosen instead, preregister the change and apply it consistently.
4. Run offline: airplane mode on, Wi-Fi/mobile data off, and server bound to loopback. The runner requires operator attestation and probes TCP reachability to `1.1.1.1:443` and `8.8.8.8:443`. A failed probe is not proof every network route is blocked.
5. The runner samples process and system memory approximately every 0.5 seconds during each request, in addition to before/after snapshots. It reports maximum sampled RSS, HWM and PSS, plus minimum sampled `MemAvailable`. Battery capacity/status/voltage and thermal readings are captured when Android exposes them. Sampling can still miss shorter peaks; process PSS may not include memory owned by another process.
6. The default output cap is 2048 generated tokens, rather than 512. `finish_reason=length` or token counts at the cap are marked `output_cap_reached`. Such answers must be reviewed as potentially truncated/incomplete, not automatically graded as incorrect. The rubric/scorer still determines correctness separately.
7. Per-run blind files contain only `blind_id` and `answer_text`. They do not expose thinking tags, output length, latency, model name, or token counts. Keep each run's mapping file private.
8. After each model/run, use `merge_blind_outputs.py` to join answers by task and shuffle candidate IDs. Give only the merged scoring JSONL to the scorer. Keep the merged private map sealed until scoring is locked. The merged file intentionally contains task IDs and answer text, not model/run labels.
9. The harness does not grade correctness. Keep the rubric/answer key separate. Report cap-reached/incomplete answers separately from wrong answers.
10. Randomize model run order across repeated rounds and document starting battery/temperature, thermal state, and time between runs. Do not compare a cold first run against a thermally throttled last run without reporting it.
11. Before each round, record starting battery percentage, charging status, battery/device temperature, and cooldown duration. The runner records battery/thermal snapshots where permissions allow. Randomize candidate model order separately for each repeat round; do not always run the same candidate first.
12. Failed/incomplete runs remain in the record. Do not silently drop them or call a winner without preregistered criteria.
12. The harness cannot enforce Android-wide network isolation. Offline status must be independently checked by the operator.

## Requirements

- Python 3.9+
- A local `llama-server` bound to `127.0.0.1`, with the selected local GGUF loaded.
- A shell/terminal environment on the phone that can run Python and read `/proc`. If Android permissions block metrics, the result must report missing fields rather than invent values.

## Smoke test

Start the server with the selected local file and bind it to loopback only. Example (adjust binary path, context and thread/GPU options to your build):

```sh
llama-server -m /absolute/path/Ornith-1.5-9B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 4096
```

Then run:

```sh
python run_llama_baseline.py --url http://127.0.0.1:8080 --server-pid <PID> --model-file /absolute/path/Ornith-1.5-9B-Q4_K_M.gguf --offline-confirmed --thinking-mode off --smoke --n-predict 256
```

The smoke run is not scored. Save the generated result JSON, blind output, and private mapping. Verify that `model.sha256`, `model.declared_revision`, `runtime.verified_server_model_path`, `smoke_expected_marker_found`, and memory fields are populated. A smoke pass does not prove task quality or stable long-run operation.

## Scored run

Each task line is an object such as:

```json
{"task_id":"T001","category":"math","prompt":"..."}
```

Do not put expected answers or rubrics in the file sent to the model. Keep a separate sealed scoring file keyed by task ID.

```sh
python run_llama_baseline.py --url http://127.0.0.1:8080 --server-pid <PID> --model-file /absolute/path/Ornith-1.5-9B-Q4_K_M.gguf --tasks /private/live_tasks.jsonl --out /private/run_ornith_q4km.json --blind-out /private/ornith_blind.jsonl --mapping-out /private/ornith_mapping.json --offline-confirmed --thinking-mode off --n-predict 2048 --seed 20261011
```

Run the other candidates with the same frozen corpus/settings, saving each blind file and mapping separately. Then combine them:

```sh
python merge_blind_outputs.py --input A=/private/a_blind.jsonl --mapping A=/private/a_mapping.json --input B=/private/b_blind.jsonl --mapping B=/private/b_mapping.json --out /private/merged_blind.jsonl --private-map-out /private/merged_private_map.json
```

Add one `--input LABEL=...` and matching `--mapping LABEL=...` pair per candidate. The labels are stored only in the private map. The merger requires each candidate to have exactly one answer for every same task ID.

## Repeat-run control

Before scored comparison, predeclare at least two rounds if device time allows. Randomize the order of candidate models separately for each round, start each round at a documented battery/temperature state, and record the run order and cooldown period in a separate operator log. Keep settings unchanged. Do not use a result gathered in a hot/throttled state as if it were a cold-start measurement.
