# AIMENG Mobile Neuron Workbench

An Android workbench for small, trainable, inspectable numeric networks. The previous Ornith 1.5 9B chat path is not the current launcher or a build dependency on the neuron-lab branch.

## What is available in the workbench

- **Overview:** network/data/task status, current estimated workspace use, process RSS/PSS/HWM and the 4 GiB hard guard.
- **Neuron manager:** filter by neuron ID, select visible neurons, save one or a selected batch to a persistent local neuron library, export one neuron or a neuron pack, enable/disable units, inspect contribution and resource-adjusted scores, and delete a unit only after a confirmation.
- **Network page:** configure multiple numeric inputs, up to 128 hidden neurons, and multiple outputs; view a native input → hidden → output graph. Rebuilding explicitly warns that active network weights reset; library copies remain.
- **Training tasks:** CPU Adam training, train/validation split, best-validation checkpoint restore, early stopping, progress, cancellation, and persistent task history.
- **Run page:** enter a numeric vector and inspect output values, hidden activations, and each hidden unit's contribution to each output.
- **Data and files:** import/export the complete workspace, dataset JSON/CSV, a training task JSON, training result CSV, recent event JSONL, single neuron JSON, or a neuron-pack JSON using Android's system file picker.
- **Automatic local persistence:** model weights, output bias, samples, neuron library, task history and recent events are stored in the private app file neuron-workspace.json.

## Resource competition and safe evolution

The workbench borrows the useful part of the proposed selection idea: quality is measured on held-out numeric samples, memory and estimated compute cost contribute a small penalty, and a single generation may attempt one redundant-unit prune followed by one slightly mutated copy of an elite active unit. The parent's output contribution is split before mutation. Pruning and child acceptance are checked against the validation set with a 0.5% tolerance; rejected mutations are rolled back. The best active unit is protected, and at least one active unit is required.

The current local workspace has a conservative **64 MiB estimated data/model budget**, at most 128 hidden units, at most 16 numeric inputs/outputs and at most 5,000 samples. These are intentional early-stage caps, not claims that the system can currently run thousands or millions of neurons. Before scaling into thousands, connections need a genuinely sparse representation and per-subnet allocations, not merely a bigger limit. Estimated Java object costs are only estimates; real process RSS/PSS and native allocations remain authoritative.

**4 GiB is a hard ceiling for the whole Android app process, not a target allocation for the neuron pool.** The workbench uses a conservative 3,500 MiB process-RSS start guard for training/evolution to leave headroom for the existing native runtime, QNN staging, UI and system overhead. The process memory guard cannot guarantee a device will never be killed, so monitor real-device peak RSS/PSS before increasing limits.

## Hardware policy

- CPU owns training, score calculation, selection, task scheduling and irregular small operations.
- The existing QNN/HTP V73 INT8 path is available as an independent matrix diagnostic. It is not the generic neural network's training backend and its padded 32³ probe is not proof of a speedup.
- Small workloads stay on CPU. NPU routing needs end-to-end benchmarks including packing, padding, driver submission, synchronization, conversion and fallback.
- No GPU compute backend is claimed as implemented.

## Task/data format

A dataset JSON is aimeng-dataset/v1, with inputs, outputs, and samples, each sample containing numeric input and output arrays. Numeric CSV rows are input columns first and output columns last; a single optional nonnumeric header line is accepted. The output dimension is inferred from the currently configured network. A task JSON is a small reusable parameter preset; import the dataset separately.

Current training is supervised numeric mapping. It does **not** yet tokenize text, understand arbitrary Chinese prompts, process images, or behave as a language model.

## Build and installation identity

The app keeps applicationId = bslsjdk.ornithnpu and the existing persistent signing-key alias/cache to preserve upgrade identity. Never generate a new signing key for a release build. The feature branch must be verified with its own current Actions run and APK; a successful build on main does not prove the neuron-workbench branch compiled, and a GitHub build does not prove QNN works on a physical phone.

## Relevant files

- app/src/main/java/bslsjdk/mcnpu/NeuronLabActivity.java — workbench pages and SAF file import/export.
- app/src/main/java/bslsjdk/mcnpu/NeuronWorkspace.java — trainable multi-input/multi-output network, resource-adjusted scoring, guarded evolution, data/model/task formats and persistence.
- app/src/main/java/bslsjdk/mcnpu/NetworkDiagramView.java — lightweight input/hidden/output connection diagram.
- app/src/main/java/bslsjdk/mcnpu/AdaptiveForwardBackend.java — conservative CPU/NPU workload router used only by the separate original lab/diagnostic path.
- app/src/main/java/bslsjdk/mcnpu/NpuNeuronForward.java — QNN HTP V73 INT8 matrix forward diagnostic.
- docs/NEURON_WORKBENCH_DESIGN.md — resource competition, training and scaling safety plan.
- docs/PROJECT_MEMORY.md — project direction and verification record.
- docs/VN_MOBILE_5_IMPLEMENTATION.md — experimental VN energy/phase propagation bridge and its verification gates.
