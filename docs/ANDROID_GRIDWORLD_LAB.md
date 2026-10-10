# Android visual neural grid-world lab

## How to open it

1. Install the newest signed APK from the repository's `latest-apk` release after the corresponding Android build has succeeded.
2. Open **AIMENG 神经元实验**.
3. On the grid-world lab page, choose the hidden-layer size and select either TD training or neuroevolution training.
4. Start with a small run (for example, 5,000 episodes/evaluations) and inspect the report before requesting a very large run. Training runs on a background worker and can be stopped.
5. Use the policy playback and single-step controls to inspect decisions, hidden activations and action Q values.
6. Copy the training report when comparing runs. Reports include throughput and held-out random-map evaluation; training wins are not the same as generalization success.
7. Checkpoints and replay memory are saved in the app's private grid-world lab directory.

## Current Android experiment

- Map: 12 × 12, with randomly generated solvable wall layouts.
- Input size: 168 = 8 local features + 144 wall-map cells + the last 8 positions represented as (x,y).
- Output: 4 action values (up, right, down, left).
- The hidden layer size is configurable. Recurrent hidden-to-hidden communication uses a bounded fan-in; two thought/communication cycles are used by the current configuration.
- TD mode uses epsilon-greedy action selection, shortest-path distance shaping, a bounded replay buffer, and periodic independent-map evaluation.
- Neuroevolution mode is separate: it mutates candidate weights, compares candidates on shared maps, retains an elite, and validates periodically. It does not use TD gradients or replay updates.
- The recurrent Q-network and its TD/BPTT updates currently run on Java CPU. The GLES 3.1 GPU path in the separate numeric Neuron Workspace is not automatically used by this grid-world Q-network.

## Training performance work

- The per-episode wall-map feature projection is cached because those 144 input features stay constant throughout one episode.
- The cache is incrementally updated whenever live TD or replay updates change input weights. Replay updates account for overlapping walls between the replay map and the current map; the projection is recomputed every 64 steps to limit floating-point drift.
- The general numeric Neuron Workspace reuses gradient, output, loss-evaluation, and GPU staging buffers to reduce per-epoch allocations and garbage collection.
- The numeric workspace's GPU hybrid path is only enabled for eligible workloads after numerical comparison and a measured speed win. It accelerates part of the hidden-layer forward pass; output projection, gradient accumulation, backpropagation and Adam remain CPU work.
- CPU-side optimizations and GPU/NPU operator probes must not be described as full hardware-accelerated training. No speedup percentage should be claimed until measured on the target phone.

## Resource and evaluation notes

The application must remain below the hard 4 GiB runtime RAM limit. Large populations, training datasets and GPU staging buffers must remain bounded. Independent random-map evaluation and the goal-adjacent action check are more useful than training reward alone; a high training win count does not prove generalization.
