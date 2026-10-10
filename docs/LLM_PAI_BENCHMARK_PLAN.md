# Small-model reasoning-language-model PK: benchmark contract

## Objective

The target is a genuinely useful reasoning language model, not a GridWorld controller. GridWorld is only an inexpensive CI smoke test. No claim of beating larger language models is valid until this benchmark suite runs on actual text-generation models and reports reproducible results.

## Required comparison groups

1. **Size-matched control:** the smallest candidate model against a similarly sized standard causal LM.
2. **Larger reference:** the same task suite against at least one materially larger instruction model.
3. **Ablations:** candidate without recurrent/thought iterations, without external memory, without retrieval, and with the complete system.
4. **Budget-matched comparison:** report both equal-token/equal-example training budgets and equal wall-clock inference budgets. Never compare a heavily tuned candidate against an untouched baseline without labeling that asymmetry.

Model IDs, immutable revisions, tokenizer revisions, quantization, context length, decoding parameters, prompt templates, software versions, hardware, and exact dataset hashes must be recorded.

## Evaluation dimensions

| Dimension | What must be measured |
|---|---|
| Language understanding | instruction following, paraphrase invariance, ambiguity handling |
| Multi-step reasoning | arithmetic, symbolic deduction, compositional and constraint reasoning |
| Generalization | held-out templates, novel combinations, length extrapolation |
| Long-context use | retrieval from context, distractor resistance, evidence tracking |
| Memory | delayed recall, update/overwrite, conflicting memories, stale-memory rejection |
| Planning | multi-step tool/action planning in deterministic sandboxes |
| Reliability | calibration, abstention when information is insufficient, contradiction rate |
| Robustness | perturbations, irrelevant context, formatting changes, adversarial distractors |
| Efficiency | parameters, peak RSS, model bytes, tokens/s, first-token latency, energy if measurable |
| Training efficiency | data tokens, optimizer steps, wall time, peak RAM, number of seeds |
| Reproducibility | fixed manifests, per-example outputs, aggregate metrics, logs, model checksums |

A benchmark score must not collapse all dimensions into one opaque number. Publish per-task scores and a predeclared macro-average, plus worst-case performance and confidence intervals.

## Anti-contamination and fairness rules

- Split by underlying rule/template, not just by random row, for compositional generalization.
- Keep a private or generated held-out split; do not tune on the final test cases.
- Freeze prompts and decoding settings before the final run.
- Run at least 5 seeds for small stochastic experiments when practical; report every seed, not only the best.
- Compare greedy decoding first; report any sampled-decoding results separately.
- Score exact-answer tasks mechanically and use blinded rubric-based judging only where exact scoring is impossible.
- Preserve raw prompts, raw outputs, parse failures, timeouts, and per-case scores.
- Treat external benchmark scores as one evidence source, not proof of general intelligence.

## Stage gates

**Gate 0: infrastructure.** Actual text-generation inference runs end-to-end; outputs, errors, model metadata, resource use, and scores are committed as artifacts/logs.

**Gate 1: correctness.** Sanity tasks pass; scoring is unit-tested with known correct/incorrect outputs; malformed outputs and timeouts count as failures.

**Gate 2: fair baseline.** At least one size-matched and one larger open model run on identical examples and decoding settings.

**Gate 3: mechanism ablation.** Each claimed innovation is compared against a parameter-/budget-matched ablation over multiple seeds.

**Gate 4: generalization.** Candidate improves a predeclared composite without a material regression in any critical dimension; held-out data stays untouched until the final evaluation.

**Gate 5: efficiency-adjusted result.** Report quality versus model bytes, parameter count, peak RAM, latency and training compute. A tiny model may win on efficiency without winning raw capability; these are separate claims.

## Current status

- CPU-only GridWorld experiments are working and useful as a low-cost control-learning smoke test.
- No real language-model comparison has yet been completed in this repository.
- The 202-parameter recurrent GridWorld model is not a language model and has no cross-step persistent memory. It must not be represented as one.
- The next implementation step is a reproducible text-model evaluation runner and pinned candidate manifests. Only after it executes real models can the project claim language-model PK results.
