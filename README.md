# AIMENG Mobile Neuron Lab

Android app for the current AIMENG direction: start with a few trainable computational units, measure their held-out contribution, score them for entry/exit, and verify that a real Qualcomm QNN / HTP V73 matrix multiply can execute the forward pass.

## Current app behavior

- Launcher: **AIMENG · 神经元实验**, not the retired Ornith 9B chat flow.
- Starts with 4 affine toy units; the on-device UI permits 2–16 units and 1–500 training epochs.
- Each unit learns a tiny synthetic regression target, `y = 2*x + 1`, using a CPU batch-gradient update.
- The score controller compares held-out error with a unit removed or a sleeping candidate admitted, subtracts a small compute cost, uses separate enter/exit thresholds, and keeps at least one unit active.
- A JSONL trace is saved under the app-private files directory as `neuron-lab-trace.jsonl`.
- The NPU forward test calls the existing JNI `NpuRuntime.matMulInt8Buf` path, which submits actual int8 tensors to QNN / HTP. It compares NPU-produced held-out predictions with the CPU reference and reports the absolute MSE delta. If NPU initialization or execution fails, the experiment reports the error and CPU learning remains usable.

## Build

The Android CI builds arm64-v8a, fetches the pinned QNN HTP V73 runtime stack, compiles Java and native C++, packages an APK, and signs it with the existing app signing identity so the installed app can be upgraded in place.

The retired 9B model is no longer downloaded as part of the build and is not needed to open or run the neuron lab. Legacy inference source files remain in the repository temporarily to reduce the risk of an unrelated native cleanup breaking the NPU service; they are not the launcher path and the lab does not load model weights.

## Heterogeneous compute policy

The runtime must not send each tiny neuron operation back and forth between CPU and NPU. Device transitions, tensor packing, quantization, synchronization, and padding can cost more than the computation itself.

- **CPU first:** parameter updates, score calculations, unit admission/retirement, trace writing, and control flow stay on CPU.
- **NPU when it fits:** use QNN/HTP for sufficiently large, batched matrix operations with supported shape buckets and quantization. The current 32x32x32 padded call is a correctness/availability probe for a tiny workload, not evidence of a speedup.
- **GPU is optional, not assumed:** only add a GPU backend after identifying a supported Android GPU compute API and benchmarking end-to-end latency, power, and memory. Do not duplicate every operation across CPU/GPU/NPU.
- **Dispatch by measured cost:** compare CPU time against NPU time including packing, padding, synchronization, and result conversion. Small workloads stay on CPU; NPU receives a batch only when measured total cost is lower.
- **No parallelism by slogan:** avoid concurrent CPU/NPU execution unless independent batches exist and measurement shows a benefit. Correctness and the 4 GiB whole-process budget take priority.

## Important limits

- This is a deterministic toy regression experiment, **not a language model** and not a biological brain simulation.
- Parameter learning and scoring currently run on CPU. The NPU is used for the measured forward matrix multiplication only; this is not NPU training.
- A successful native call must still be verified on the target phone. GitHub compilation cannot prove the device's HTP backend is available or that the NPU result matches CPU within tolerance.
- The whole Android app runtime must remain below 4096 MiB, with a target release gate below 3800 MiB. Increase the unit count gradually and measure actual device peak RSS/PSS before raising the default.
- The NPU backend's INT8 quantization and output scale are approximate; the UI reports the CPU/NPU error delta rather than silently treating any non-null result as correct.

## Relevant files

- `app/src/main/java/bslsjdk/mcnpu/NeuronLabActivity.java` — mobile UI and background execution.
- `app/src/main/java/bslsjdk/mcnpu/NeuronLabEngine.java` — parameter updates, contribution scores, hysteresis, safety floor and trace.
- `app/src/main/java/bslsjdk/mcnpu/NpuNeuronForward.java` — QNN HTP int8 forward path.
- `app/src/main/java/bslsjdk/mcnpu/NpuRuntime.java` and `app/src/main/cpp/mcnpu.cpp` — existing native QNN runtime and matrix-multiply backend.
- `docs/PROJECT_MEMORY.md` — current direction and verification limits.
