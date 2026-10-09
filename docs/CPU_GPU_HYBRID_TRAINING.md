# CPU/GPU hybrid training

The neuron workspace can use the GLES 3.1 compute backend during full-batch training.
This is separate from the large neuron-pool routing benchmark.

## Execution model

- Every epoch computes hidden activations using the weights at the beginning of that epoch.
- For eligible workloads (at least 64 training samples, 32 hidden neurons, and 4 input dimensions), CPU and GPU calculate disjoint hidden-neuron ranges concurrently.
- The GPU calculates the dense input-by-weight matrix product; the CPU calculates the other hidden-neuron slice. Bias addition and `tanh`, output projection, backpropagation, and Adam updates remain on CPU.
- Adam parameters are updated only after gradients from the whole training set have been accumulated. This preserves the existing full-batch training algorithm; it does not use stale outputs from a previous epoch.
- The first eligible epoch warms the GPU, compares CPU-only and hybrid hidden-layer execution, and checks the maximum absolute activation difference against 0.001. Hybrid execution is enabled only when the check passes and the measured hybrid slice is at least 10% faster. Otherwise the workspace keeps the CPU path.
- GPU failures or malformed results fall back to CPU. Small jobs stay CPU-only because JNI, dispatch, synchronization, and readback overhead can cost more than the computation.

## Memory and correctness

- Temporary input/weight/output buffers are bounded by a 32 MiB training-side estimate; the GPU matrix runtime independently enforces its 64 MiB workset cap.
- The persistent activation scratch buffer is reused between epochs to avoid allocating thousands of arrays per epoch.
- GLES calls release their EGL context after each JNI operation so the main thread and dedicated GPU worker can safely hand off the context under the native mutex.
- Runtime RSS remains device-dependent. The application must continue enforcing its existing 4 GiB runtime RAM ceiling; the buffer estimate is not a substitute for measuring RSS.

## What is and is not accelerated

This accelerates the hidden-layer dense forward calculation during training. Output projection, gradient accumulation, backpropagation, and Adam remain CPU work. It is not a claim that the entire training loop runs on GPU. The training report records the selected backend. A device with an unavailable/slow GLES compute path will remain on CPU.
