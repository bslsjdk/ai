# VN-Mobile 5.0: staged integration into the neuron workbench

This document maps the VN-Mobile proposal onto the current `feat/mobile-neuron-unit-runtime` branch. The new wave-field path is an experiment alongside the existing supervised numeric network, not a replacement for its trained weights or optimizer.

## What is implemented in this first bridge

- `VnWaveField` keeps bounded per-node energy, phase and activation arrays across calls.
- Connections are directed, have finite weights in [-1, 1], and are stored in compact primitive arrays.
- Each tick selects up to a configured top-k set of active sources. A zero-energy field with no injected disturbance remains quiescent instead of waking arbitrary nodes.
- Propagation uses a phase term `cos(phi_from - phi_to)`, while a bounded sine coupling changes destination phase.
- The Android workbench now has a dedicated “VN 波场” page. It feeds the current trained network's hidden activations into the independent wave field, derives a bounded directed graph from weight-vector similarity, and reports the highest-energy nodes, phase, activation, tick count and estimated primitive-array payload.
- The experiment never updates the trained network's weights. A topology fingerprint resets the wave state when the source network's weights, enabled flags or unit identities change.
- Wave propagation is user-triggered and bounded to 1–128 ticks per action. There is intentionally no always-running background loop yet, because continuous CPU activity would be a battery and thermal cost that must be measured before being enabled.
- The first unit tests cover directed propagation, state bounds, active-source budgets, deterministic snapshots, invalid input shapes, quiescent background behavior and a 2,048-node payload estimate.

## Deliberate differences from the proposal

- The current workbench's trained network is still a one-hidden-layer tanh numeric mapper. It is not a language model and does not provide general natural-language reasoning.
- Similarity-based edges are a prototype topology heuristic, not a learned router. The experiment has no trained direction vector, no semantic memory address space, and no answer-quality guarantee.
- Energy is an activity variable, not confidence. A high-energy state must not be interpreted as a correct answer.
- The primitive-array payload estimate excludes Java object headers, alignment, Activity/UI, dataset, optimizer, QNN and native allocations. Device RSS/PSS/HWM measurements remain authoritative.
- The proposed 2 GiB VN target is a separate future budget. The existing Android process hard ceiling remains below 4 GiB, with a conservative training-start guard and working-set limits.

## Next verification gates

1. Confirm Java compile, unit tests, Android resource processing, native CMake build and APK packaging on this exact feature branch.
2. Compare a plain baseline graph against phase-coupled propagation on deterministic associative-retrieval tasks at the same active-source/edge-work budget.
3. Record target hit rate, steps to retrieval, false stable states, edge visits, elapsed time, and energy/phase traces. Do not infer quality from visual oscillation.
4. Measure phone RSS/PSS/HWM and thermal/power behavior before considering any periodic background tick.
5. Only if the propagation experiment has repeatable quality gains, consider replacing the heuristic similarity graph with trained routing and adding an explicit confidence/abstention readout.

## Current status

Code committed to `feat/mobile-neuron-unit-runtime`. GitHub commit success proves only that the files were written; build, tests, APK installation and real-device behavior must be verified separately.
