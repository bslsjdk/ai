# PK protocol: self-baseline now, Ornith reference later

Status: protocol draft for review; no phone model run or PK victory is claimed by this document.

## 1. Purpose and comparison arms

The primary end goal remains a small model that can compete with the frozen Ornith-1.5-9B-Q4_K_M reference on the same practical tasks. Download time is not a reason to stop useful work, so experiments before that file is available use **internal self-baselines**:

- **Baseline A:** the current, explicitly versioned candidate/checkpoint/configuration.
- **Candidate B:** one changed architecture, training recipe, memory mechanism, or decoding setting.
- **Optional control C:** an unchanged copy of A rerun to estimate run-to-run noise.

A and B must solve the same frozen development tasks with the same scoring rules. Record the exact commit, checkpoint hash, tokenizer, parameter count, quantization, context length, generation settings, and runtime environment for each arm. A ChatGPT conversation or its prose is not itself a model execution, benchmark result, or independent gold label. Any claims about ChatGPT's own answers must be labelled qualitative analysis unless a reproducible model endpoint and exact version are actually measured.

When Ornith is downloaded and runnable, add it as a named reference arm without replacing or rewriting earlier results. If it cannot run on the phone, preserve that as a feasibility failure; do not pretend it was benchmarked.

## 2. Dataset separation and freeze

Keep three separate sets:

1. **Development:** may be inspected during iteration; use it to diagnose and tune.
2. **Validation:** use to choose among a small number of candidate changes. Log every comparison.
3. **Sealed held-out test:** do not inspect model answers or tune prompts/configuration against its scoring key before the final comparison.

Store prompts in model-facing JSONL and expected state, acceptable answers, and scoring checklist in a separate private file. The model-facing corpus must contain only `task_id`, `category`, and `prompt`. Hash the exact bytes of both files (SHA-256) and record the hashes, versions, and freeze date in the result report. If a held-out prompt or key is accidentally exposed to training/tuning, mark that split contaminated and replace it before claiming a held-out result.

Use the same task text, task order, system instructions, output cap, and scoring rubric for every arm. Where model templates require different wrappers, preserve the same semantic instructions and document the wrapper difference.

## 3. Task coverage

The eventual held-out set should contain at least 40 tasks, with at least 8 per core category below. Include simple controls and multi-step cases; do not let a model win by verbosity.

- **Instruction following:** satisfy explicit format, language, length, and constraint requirements.
- **Multi-step reasoning:** small problems with independently checkable final answers and intermediate constraints.
- **State tracking and memory:** maintain a changing set of facts across multiple turns; test both correct recall and resistance to obsolete facts.
- **Planning:** produce a sequence of actions that satisfies all stated constraints; check preconditions, dependencies, and final state.
- **Error recovery:** detect a deliberately introduced inconsistency or failed step, revise the plan, and avoid repeating the same failure.

At least one quarter of tasks should be multi-turn or stateful. Include adversarial-but-benign cases: distracting irrelevant details, one changed constraint, conflicting old/new state where the newest valid instruction wins, and requests that are impossible under the stated constraints. Do not include tasks whose answer depends on current web access in an offline benchmark.

## 4. Task-specific checklist scoring

Each task is scored against a checklist frozen before answers are revealed. Each checklist item is binary (0 or 1) and tied to an observable requirement, not writing style. A task's score is:

`task_score = passed_checklist_items / total_checklist_items`

Record task success separately: success is 1 only when all mandatory items pass; otherwise 0. A fluent answer that violates a hard constraint fails that item. Unsupported claims do not receive credit. If a task is ambiguous or the key is wrong, flag it for adjudication and exclude it from the primary score only with a documented reason applied to every arm.

Recommended checklist fields per task:
- required final result or state;
- mandatory constraints and formatting;
- required intermediate dependencies, where relevant;
- prohibited failure modes;
- accepted alternatives and tolerance for numeric answers;
- explicit conditions under which a safe/valid refusal is the correct answer.

Do not use a single LLM judge as the sole authority. Prefer objective checks for exact values/state/format; for open-ended cases, the human operator adjudicates against the frozen checklist while blind to model identity. An LLM judge may provide a secondary score, but it must not override objective checks or the human's final adjudication.

## 5. Blinding and answer integrity

For each run, save a private mapping from random blind IDs to task IDs and model/run identity. The scorer receives shuffled answer text and task-specific checklist only. Remove model names, file names, timing, token counts, and explanatory metadata from the scoring view. Randomize answer order and, for pairwise review, randomly swap left/right positions.

Keep all attempted tasks in the result: errors, empty answers, incomplete output, timeouts, and cap-reached outputs. A failed request is a run failure, not a missing row. Distinguish:
- correct complete answer;
- incorrect answer;
- incomplete/cap-reached answer;
- runtime/request failure;
- invalid/ambiguous task awaiting adjudication.

Never silently discard a failure. If a scorer discovers model identity, record the blind break.

## 6. Run procedure

1. Freeze source commit and configuration; compute artifact and corpus hashes.
2. Run a non-scoring smoke test. It validates loading and the request/response path only.
3. Run all tasks in a documented order. Repeat at least two rounds when feasible, randomizing model order between rounds.
4. Record start/end time, device state, temperature/thermal indicators, battery/charging state, context and generation settings, per-task latency, failures, and available memory metrics.
5. Cool the device between rounds as needed. Report warm/throttled rounds rather than pretending they are cold-start measurements.
6. Score blind answers using the frozen checklist. Reveal mappings only after scores are locked.
7. Publish per-task outcomes, aggregate metrics, exclusions, raw logs, hashes, and any missing instrumentation. Keep private keys and mappings out of public commits.

## 7. Metrics and decision rules

Always report the metric vector, not just a headline score:

- task success rate and checklist completion rate;
- per-category results and per-task differences;
- request/runtime failure rate and incomplete/cap-reached rate;
- median and p95 latency;
- peak sampled process RSS/PSS/HWM and minimum system MemAvailable, with sampling limitations;
- parameter count, artifact size, quantization, context length, and measured runtime RAM;
- repeat-run variation and device thermal state.

For a self-baseline A-vs-B experiment, define the direction of improvement before running. Prefer a paired comparison over the exact same tasks. Report the paired mean difference, standard error or confidence interval, number of tasks won/lost/tied, and category regressions. A change is **promising**, not a confirmed win, if its improvement is smaller than run-to-run uncertainty, comes from a single category, or increases failures/latency/memory enough to undermine the goal. Repeat the experiment with new seeds or tasks before treating a small gain as robust.

The candidate small model must respect the project's runtime RAM limit of 4 GiB. Measure this rather than inferring it from GGUF file size. The Ornith reference is measured honestly and is not assumed to fit that candidate budget.

## 8. Reporting labels

Use only these labels:

- **Protocol-only:** test design exists; no model execution.
- **Smoke passed/failed:** request path tested; no task-quality claim.
- **Development result:** used during iteration; not held-out evidence.
- **Held-out result:** frozen tasks and key, blind scoring, complete failure accounting.
- **PK win:** only after both the candidate and Ornith have been run on the same uncontaminated held-out tasks and the predeclared decision criteria are met.

The current RNN/MGU seven-seed language-modeling experiment is an architecture diagnostic, not a mobile task result and not a PK win. The present protocol itself also does not claim that any model has passed a phone run.
