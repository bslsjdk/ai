# Project State

Last updated: 2026-10-11

## Mission and decision ownership

- Competition goal: build the smallest practical model/architecture that can beat the Ornith-1.5-9B reference on a preregistered set of real tasks, including on-device latency and memory.
- User is the final judge and supplies/approves live tasks and phone measurements.
- Claude: design and review.
- ChatGPT in this workflow: implement tooling, run/retrieve experiments where connected, preserve evidence, and report uncertainty. Do not announce a winner before preregistered criteria and independent review are satisfied.
- Hard deployment constraint for candidate small models: runtime RAM must remain below 4 GiB. This is a candidate-model budget, not automatically a requirement that the 9B reference itself fit.

## Repository map

### PK architecture benchmark
- Repository: https://github.com/bslsjdk/mengxi_lab
- Branch: `pk-public-dataset-benchmark`
- Frozen protocol: `pk_public_data/README.md`
- Results: `pk_public_data/results/`
- Runtime: Deepnote, CPU for the confirmed seven-seed run.
- Deepnote project: `PK Benchmark Live Progress Copy`
- Notebook ID: `71e29727fee648238bb719d35d7700ac`
- Successful run ID: `4d6f3cd3-0760-4593-aea3-91090a7f1e3e`
- Run ID: `confirm_seq128_ctx128_7seeds_20261011`
- Result CSV: `pk_public_data/results/confirm_seq128_ctx128_7seeds_20261011.csv`
- Metadata and paired analysis: `pk_public_data/results/confirm_seq128_ctx128_7seeds_20261011_metadata.json`
- Corpus SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`

### Mobile evaluation / Ornith baseline
- Repository: https://github.com/bslsjdk/ai
- Main project state: `docs/PROJECT_STATE.md`
- Evaluation harness branch: `eval/d-offline-mobile-baseline`
- Draft PR: https://github.com/bslsjdk/ai/pull/79
- Harness: `eval/run_llama_baseline.py`
- Protocol: `eval/README.md`
- Tests: `eval/test_run_llama_baseline.py`
- CI: `.github/workflows/eval-harness.yml`
- CI passed before the latest instrumentation edits; re-check the latest branch workflow before treating those edits as validated.
- No on-device smoke run or scored phone baseline has been recorded yet.

## Frozen seven-seed RNN vs MGU protocol

Read `pk_public_data/README.md` in `bslsjdk/mengxi_lab` before changing or interpreting this experiment.

- Models: RNN-128 (28,113 trainable parameters) vs MGU-94 (28,271).
- Seeds exactly: 33, 44, 55, 66, 77, 88, 99.
- `seq_len == context_len == 128`.
- Maximum 120 epochs, 60 steps/epoch, batch size 16, learning rate 0.002, weight decay 0.0001, gradient clipping 1.0.
- Validation-based checkpoint selection; score the full held-out test split.
- Dataset split is contiguous 80/10/10; dataset hash above.
- Paired difference: `d_i = RNN test loss_i - MGU test loss_i`; positive favors MGU.
- Win condition is conjunctive: mean paired difference must exceed `2.45 * SE`, and MGU must win at least 6/7 seeds.
- README flags any best checkpoint at epoch 110 or later as ceiling-adjacent; convergence is not established for such runs.

## Confirmed seven-seed result (2026-10-11)

- Deepnote run completed successfully and emitted all 14 model/seed rows with `seq_len=128` and `context_len=128`.
- Mean paired difference: 0.0158521831.
- Sample SD of paired differences: 0.0185839373.
- Standard error: 0.0070240681.
- `2.45 * SE = 0.0172089667`.
- MGU wins 6/7 seeds.
- Decision: **no robust win detected**. The win-count criterion passes, but the mean-difference threshold fails (0.0158522 is not greater than 0.0172090).
- Many best epochs are at or beyond 110, so the README's ceiling-adjacent warning applies to most rows. Do not treat this run as proof of convergence or a definitive architecture winner.
- Metadata reports `git_commit=unknown`; preserve this provenance limitation. The result CSV and metadata were committed to the benchmark branch after being extracted from the successful Deepnote run.

## Model reference and phone constraints

- Reference opponent: Ornith-1.5-9B.
- Do not conflate MLX/Safetensors with GGUF. Before a phone baseline, freeze one exact model source, conversion/release, quantization level, and file SHA-256.
- Third-party GGUF/mobile variants are candidate artifacts, not interchangeable model identities. Report the exact source and quantization.
- Available prior device diagnostics indicate about 15,665,944 kB total RAM (~14.94 GiB); exact retail phone model has not been verified from user-provided details. Candidate runtime still has the hard <4 GiB limit.
- NPU availability does not prove inference uses the NPU; record the actual execution backend.

## Evaluation protocol requirements

- Fix prompts, scoring rubrics, task set hash, model hash, decoding parameters, output cap, and thinking mode before comparing candidates.
- Keep answer keys and rubrics out of model inputs; blind candidate outputs and unblind only after scoring is locked.
- Use llama.cpp server-reported timing fields for token throughput; record per-task wall time as a separate end-to-end metric.
- Record process RSS/PSS/HWM and system-wide memory; collect thermal data when Android exposes it.
- Require operator offline confirmation and run external reachability probes. A failed probe is not proof of all network paths being blocked.
- Visible `<think>...</think>` text may be logged separately; hidden internal reasoning is not observable. Fix the thinking-mode policy and generation cap across candidates.
- Smoke test first. Failed/incomplete runs remain in the record. No result is called a win without satisfying preregistered criteria.

## Work status

### Done
- Located and reread the canonical PK README.
- Completed and archived the seven-seed context-128 run; computed the preregistered paired decision.
- Added first version of mobile evaluation harness in an isolated draft PR.
- Added PSS/system memory, server-side generation speed, visible-thinking separation, and external TCP offline probes to the harness branch.

### Next
1. Verify latest GitHub Actions result for the updated PR branch.
2. Finish review of `eval/run_llama_baseline.py`; then freeze one exact Ornith GGUF/mobile artifact.
3. Perform a phone smoke run and verify runtime memory stays below 4 GiB for the candidate model. Record separately if the 9B reference cannot fit.
4. Continue architecture follow-up only after reviewing the ceiling-adjacent warning; any expanded training budget must be predeclared before a new run.
5. Update this file whenever a result, frozen config, branch, or status changes.
