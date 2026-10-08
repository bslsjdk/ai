# AI 项目长期工作记忆 / 防错记录

更新时间：2026-10-08

## 绝对目标仓库
当前所有 AI 项目代码、修复、性能优化、NPU/模型运行时工作，只允许操作：
https://github.com/bslsjdk/ai

除非用户明确要求，否则严禁修改、回滚或继续操作 `bslsjdk/mcnpu`。

## 当前目标模型
唯一当前目标模型是：

- **Ornith-1.5-9B-MLX-4bit**
- 官方仓库：`ornith-ai/Ornith-1.5-9B-MLX-4bit`
- 格式：MLX / Safetensors / 4-bit affine quantization
- 不是 GGUF Q4_K_M
- 当前工作名称必须写 **Ornith-1.5**，不能写成 Ornith-3.5、Ornith35 或其他不存在的目标版本。

官方模型资料已核对：Hugging Face 当前明确列出 `ornith-ai/Ornith-1.5-9B-MLX-4bit`，文件包含 `model.safetensors`、`config.json`、`tokenizer.json` 等。

## 已犯错误，必须永久避免
1. 曾把目标模型错误写成/实现成“Ornith 3.5 / ornith35”。这是错误命名，必须修正为 **Ornith-1.5**。
2. 曾误操作 `bslsjdk/mcnpu`，而当前任务目标其实是 `bslsjdk/ai`。以后不得因为文件名、包名、历史上下文相似而切错仓库。
3. 曾在错误仓库修改包名、applicationId、Manifest、签名 workflow、JNI 符号等身份相关内容。当前项目不得重复这种跨仓库误修改。
4. 任何模型架构/配置数字都必须先以当前目标模型的官方 config/权威文件和仓库实际代码为依据，不能凭相似模型或猜测填写。
5. 不得因为旧文件名里出现 `35` 就推断目标模型是 3.5。当前模型明确是 **1.5-9B**。
6. 用户要求“继续”时必须基于本仓库实际最新状态继续，不要重新假设项目背景。

## 当前硬约束
- Android runtime 内存必须控制在 **4 GiB 以下**。包括 native heap、Java heap、resident mapped pages、QNN buffer、staging、KV/recurrent state、tokenizer、临时 buffer 和 runtime 对象。
- 必须是真实文本生成，不能用假输出、探针输出冒充生成。
- NPU 目标为 Qualcomm QNN HTP V73，保持真实执行链。
- GGUF/llama.cpp 是旧基线/兼容路径，不能因为新增 MLX 路径而破坏。
- MLX 4-bit 必须按 Safetensors 中的 affine 4-bit packed weight 规则处理，不能当成 GGUF Q4_K_M。

## 当前模型事实
Ornith-1.5-9B-MLX-4bit：
- 9B dense
- hidden size 4096
- intermediate size 12288
- 32 layers
- 16 attention heads
- 4 KV heads
- attention head dim 256
- linear key heads 16
- linear value heads 32
- linear key/value head dim 128
- context length 262144
- RoPE theta 10000000
- 4-bit affine quantization，group size 64 是当前实现重点

## 当前代码命名规则
源码内部统一使用：
- Ornith15...
- ornith15_...
- ORNITH15...

禁止新增：
- Ornith35...
- ornith35_...
- ORNITH35...

如果仓库里仍存在旧的 ornith35 文件/符号，它们属于错误命名遗留，必须在修复过程中迁移到 ornith15，并确保 CMake/include/调用点一致。

## 2026-10-08 当前实现进度
- 已按官方 Qwen3.5 实现修正 full-attention 的 q_proj 8192 维交错布局：[Q_256, gate_256] per head。
- 已按官方实现修正 partial RoPE：前 32 维与后 32 维配对，频率分母为 32，而不是相邻维配对。
- 已修正 Qwen3.5 zero-centered RMSNorm：输入/后置/最终/q-k norm 使用 `(1 + weight)`；GatedDeltaNet 输出 norm 使用 `weight * SiLU(z)`。
- 已修正 DeltaNet softplus 高值截断，采用稳定 softplus。
- MLX projection 主桶按输出维动态选择，普通大投影优先 4096，32/64/128/256/512/1024 等小投影不再无意义地扩大到 4096。
- 聊天入口已收敛为单一本地 Ornith-1.5-9B-MLX-4bit 导入路径，不提供模型列表、Provider、API、Endpoint 或云端回退。
- 新增官方 tokenizer.json → 紧凑 OTK2 sidecar 的 CI 构建工具，并在 APK 构建前自动生成 tokenizer 资产。
- Native MLX 路径已从 PROBE_ONLY 接到真实 token forward/generate 入口；加载阶段仍要求真实 QNN HTP V73 tile probe 成功后才报告 loaded。
- Safetensors 权重保持 tile/streaming 读取，不整体加载约 5GB 模型文件；full-attention cache 第一阶段为 65536-token FP16 有界窗口，模型原生仍为 262144。
- MLX affine4 到 QNN int8 的动态缩放链已补回 QNN 固定 0.001 输入尺度，并加入 CPU/NPU tile 误差诊断。
- 大矩阵投影主 tile 已提升到 32 x 4096 x 4096 桶；LM head 末尾不满 4096 的行会零填充后执行。
- 已补齐 Ornith-1.5 的关键 tensor 布局校验，并修正官方 `language_model.lm_head.weight` 路径。
- 2026-10-08 的 GitHub Actions 多次在 job steps 开始前快速失败；当前没有可用的编译日志，因此不能把 CI 视为通过。

## 长上下文与记忆架构基线（2026-10-08）
- “上下文”目标不是只保留最近 N 条，而是让模型在新一轮中仍能知道会话最开始发生了什么、用户目标/约束/决定以及中间重要进展。
- 原始聊天历史必须完整保存在设备端；上下文管理器只能选择/重排进入当前 prompt 的内容，不能把“摘要”当成原始历史的替代品。
- 已借鉴 Operit 的核心思路：上下文限制、历史窗口管理、相关历史补充、长期记忆检索/总结触发的分层思想；不复制其多 Provider/云端架构。
- 当前 `AgentContext` 已改为 USER 优先：`Conversation origin` + 全历史分布式 `User history spine` + 当前问题相关旧对话 + 最近完整对话。用户消息相关性权重高于 assistant 输出，避免模型被自己的废话淹没。
- 当前 prompt 会把 `Current USER message` 强制放在最后，避免预算裁剪把最新用户指令切掉。
- 旧的固定“最近 6 条”不再决定模型能否知道前面发生了什么；`MAX_CONTEXT_MESSAGES` 只是最近窗口的一部分。
- 模型请求上下文仍配置为 65536 tokens，模型原生 context=262144；当前 native full-attention resident window 第一阶段已为 65536，长上下文依靠 FP16 KV + 4 GiB 内存预算与按需驻留，而不是一次性扩大到原生 262K 所需的多 GB 常驻 KV。
- 后续真正扩展到 64K/128K 有效上下文时，优先研究分页/分块 KV、按需重算/CPU/存储卸载和更精确的 token budget，不得一次性分配多 GB 常驻 KV。

## 参考/借鉴仓库与来源（2026-10-08）

这些项目是实现时的明确参考来源；它们的作用不同，不能混成一个“后端依赖”。

- AAswordman/Operit — Android Agent 的任务工作区、长对话上下文/记忆分层和工具执行交互思路。只借鉴架构思想，不引入其 Provider/云端模型架构。
- ggml-org/llama.cpp — GGUF 兼容路径、KV cache 管理和移动端/本地推理的工程参考。CI 当前固定到 42c787e8c191d49c01c757f4b7029d69ec0bf2c8。
- bslsjdk/npu_probe — Qualcomm QNN HTP V73 已验证的本地栈与 Android 资产来源。CI 当前固定到 8fab2ee58cd4ce17ddc18e36a6a7b4992c85c52a。
- huggingface/transformers / Qwen3.5 实现 — Qwen3.5 混合 Attention/GDN、RMSNorm、RoPE、Q/K norm、DeltaNet 等架构事实与公式参考；模型具体参数仍必须以当前 Ornith-1.5 模型资料和仓库校验器为准。
- PrismML-Eng/Bonsai-demo 与 PrismML-Eng/llama.cpp — Bonsai 2 PQ2_0、FWHT/Hadamard、GGUF 私有格式和量化 KV/后端性能取舍的参考。当前 Bonsai 路径仍与 Ornith runtime 分开。
- ornith-ai/Ornith-1.5-9B-MLX-4bit — 当前唯一目标模型的 Safetensors/MLX affine4 权重、tokenizer 与模型格式来源。

## 2026-10-08 HEAD 代码-记忆对照审计

当前 main HEAD 在本轮开始检查时为 e370e231908540228ab8bb4d89b3413f5406876f。

本轮对照发现并修复了三个工程级问题：

1. CMakeLists.txt 原本无条件把 mcnpu_ggml_backend.cpp 加进目标；没有设置 LLAMA_CPP_SRC 时仍会尝试包含 ggml 头文件，导致没有 llama.cpp 的本地 CMake 构建链不完整。现在只有设置 LLAMA_CPP_SRC 时才加入该源文件。
2. ornith_runtime.cpp 使用 std::chrono 却没有显式包含 <chrono>，依赖传递头文件是不可靠的。现在显式包含。
3. MLX decode 在达到 maxTokens 后仍执行一次完整的下一-token forward，造成无意义的额外 32-layer 计算并污染 decode 时间统计。现在最后一个请求 token 产生后立即结束。

以下仍属于待实测而非“已验证完成”的项目：

- GitHub Actions 是否真正完成 Java/C++/APK 编译；此前 runner 存在 steps 创建前快速失败，不能用它冒充编译证据。
- 真机 64K 的 VmRSS/VmHWM/VmPeak、PSS、TTFT、prefill、decode tok/s 和 QNN staging 内存。
- 64K 长上下文下 full-attention QK/AV 的实际 HTP 稳定性与图缓存内存。
- 目前 runtime 仍使用 FP16 resident KV；Q8 K/Q8 V、Q8 K/Q5 V 只是 probe/planner 研究路径，尚未成为生产 KV。
- 大矩阵 projection 仍保持按 token 的 Safetensors tile/streaming 读取，但本轮已去掉每个 tile 约 64 MiB 的 F32 权重展开，改为 packed affine4 直接转置量化到 INT8；小型静态 norm/A_log/dt/conv 权重也改为模型加载时一次读取并跨 token 复用；单 token projection 还去掉了 LM-head 的约 32 MiB F32 输出临时矩阵，只保留一行逻辑输出。这样同时降低峰值内存与 CPU/I/O 搬运，但 hot-tile 跨 token 复用与最终 decode tok/s 仍必须用真机 profiler 实测。

## 当前工作方式
用户只用手机/MT 管理器，不会直接维护复杂 C++ 工程。
因此助手应直接修改仓库、提交修复并尽量验证 CI/build，不要把实现工作重新丢给用户。

## 诊断规则
- 没有实际 CI/build 证据时，不得声称“已编译成功”。
- 每次修复先确认仓库是 `bslsjdk/ai`。
- 每次涉及模型名称、架构或配置时，先核对官方模型资料或仓库 config。
- 修复完成后优先检查编译引用、头文件引用、CMake 源文件列表和 Java/JNI 调用链。
## 2026-10-08 长上下文 Native 第一阶段落地
- full-attention resident KV 已从 FP32 改为 IEEE-754 FP16，最大驻留窗口提升到 **65536 tokens**。
- Ornith15 runtime 对第一阶段上下文硬上限为 65536；模型官方原生上限仍为 262144，后续更高上下文必须采用更省内存的方案，不能直接把 FP16 KV 线性放大。
- 64K/8 个 full-attention 层的 K+V FP16 cache 约占 2 GiB；executor 增加运行时状态预算上限 3 GiB，为 QNN、activation、tokenizer 与其他 native 状态保留余量。
- 新增 QNN FP16 MatMul bridge，full-attention 的 Q×Kᵀ 与 softmax×V 已切换到 HTP FP16 MatMul，不再用 CPU 对 64K KV 做完整点积乘法。
- attention cache 改为 K=[kv_head][head_dim][token]、V=[kv_head][token][head_dim]，可直接向 QNN 提供连续矩阵区段；ring wrap 时最多拆成两个 segment。
- 新的 attention reset 不再清零整块 KV，重置只清逻辑 token 计数；旧槽位在逻辑窗口外不会被读取，避免每次请求写扫约 2 GiB。
- Java ChatActivity 当前历史预算提升到 32 条最近窗口 / 150000 字符上限；AgentContext 使用用户优先的 origin/spine/relevant/recent，并做保守 token 估算，当前用户消息始终单独置于最后。
- 64K full-attention 的实际设备速度与 FP16 MatMul 形状稳定性仍需要真实手机验证；GitHub Actions 最近 job 仍在 steps 创建前秒退，不能替代设备验证。


## 2026-10-08 runtime completion pass

### Generation/runtime
- Ornith runtime buffers are initialized once at model load and reused across chat turns; each generation resets only logical recurrent/KV state.
- Native profiler records prompt tokens, generated tokens, tokenize/prefill/first-token/decode microseconds, RSS/HWM/Peak and affine4 weight-read bytes/time.
- ChatActivity displays the latest native runtime performance snapshot in the status line.
- Generation remains real MLX affine4 inference; no probe-only success is accepted.

### Memory
- Full-attention KV remains FP16, but storage is raw contiguous allocation without whole-range value initialization, so 64K capacity is demand-paged rather than explicitly zero-filled at load.
- Runtime memory planner still enforces the <4 GiB project constraint using a 3.5 GiB process RSS hard guard plus safety/non-state reserves.
- Long-lived runtime buffers are reused to avoid multi-gigabyte allocation churn between turns.

### Full Attention
- K is token-major [kv_head][token][head_dim].
- QK uses a QNN FP16 MatMul graph with transpose_in1, eliminating host-side K repacking for ring segments.
- Full-attention Q is scaled by 1/sqrt(256) = 1/16 after q/k normalization and RoPE, matching the current Qwen3.5 reference.
- QNN MatMul graph cache lifetime budget is 32 graphs. With N buckets 32..65536, QK and AV can create up to 24 distinct FP16 shapes during one 64K replay; production INT8 projection adds a small number of shapes.
- Wrapped-ring correctness and non-full early-window correctness must not depend on the segment starting at slot zero.

### I/O
- Safetensors reads use a thread-local persistent ifstream rather than reopening the ~5 GiB model for every tile. Full-K production tiles are read as contiguous blocks instead of row-by-row seeks.
- Affine4 tile reads are counted during generation so real decode weight traffic can be measured instead of inferred from total model file size.

### Correctness repairs
- LM-head final padded tile writes only the real vocabulary columns, preventing the final 248320-row projection tile from writing past the output buffer.
- QNN transpose-B support was added through the existing probe-verified transpose_in1 MatMul parameter path.

### Verification status
- GitHub Actions runner infrastructure has repeatedly failed before executing build steps (steps=null), including rerun attempts. This is not compile evidence.
- A previously published latest-apk predates the current runtime work and must not be treated as a current build.
- Real-device 64K VmHWM, TTFT, prefill, decode tok/s and actual QNN staging memory remain unmeasured until a current APK is built and run on hardware.
## 2026-10-08 性能/编译完整性继续修复

本轮追加修复：
- 小型非量化模型权重（layer norm、Q/K norm、A_log、dt_bias、GDN norm、conv1d、final norm）在模型加载时一次读取并驻留，generation 期间不再逐 token 从约 5 GiB Safetensors 文件重复读取。
- affine4 大矩阵 tile 不再先完整解码成约 64 MiB 的 F32 权重矩阵再量化；直接从 packed U32 affine4 + scales/biases 生成转置 INT8 B 矩阵，降低 CPU 临时内存和内存带宽。
- 单 token projection 已避免为已对齐 K 维创建多余的 32xK F32 输入填充，只保留必要的 M=32 输出桶。
- 以上改动仍需真实 Android/QNN 设备验证，尤其要测 decode tok/s、TTFT、RSS/HWM/PSS 和 HTP graph staging；代码通过静态结构检查不等于编译通过。

## 2026-10-08 LM-head 临时内存修复

单 token projection 的 M=32 是为了命中 HTP bucket，但调用方只消费第 0 行。当前实现让 projection tile 在 logical_rows=1 时只清零和写入 1×N 逻辑输出，LM-head 不再分配完整 32×248320 F32 输出缓冲；这是内存和速度优化，不改变 NPU 图的 M=32 bucket。

## 2026-10-08 CMake 对照修复补记

对提交后的实际文件再次检查时发现 CMake 源文件列表曾因一次排序修复留下两份 MCNPU_SOURCES 和两份 add_library。该重复块已删除，当前结构恢复为：先定义 QNN_INC/LLAMA_CPP_SRC，再定义一次 MCNPU_SOURCES，仅在 LLAMA_CPP_SRC 存在时追加 mcnpu_ggml_backend.cpp，最后只创建一次 mcnpu target。

## 2026-10-08 QNN graph accounting hardening

继续对照 `mcnpu.cpp` 后统一 QNN graph 计数：凡是可能长期占用 context 资源的缓存/诊断图，都在 graphCreate 成功后计数，而不是在 finalize 后或 create 前计数。这样 graph budget 与实际由 QNN context 持有的图数量保持一致，失败的 tensor/node/finalize 路径也不会让计数低估。

## 2026-10-08 声明/定义对账补记

最终静态扫描发现并修复了一处编译阻断：`ornith15_linear.h` 已加入 `logical_rows`，而 C++ 定义一度仍缺少该参数，导致函数体对未声明变量的引用。当前 header、definition 和 token caller 已统一。

## 2026-10-08 final load-path hardening

模型加载在初始化 KV/recurrent/work buffers 和静态小权重后都经过 RSS memory guard；同时移除了 `ornith_runtime.cpp` 中重复嵌套的 `#if MCNPU_HAS_LLAMA` 预处理器层，保持 llama 兼容路径条件清晰。

## 2026-10-08 continued runtime audit

- ornith15_run_projection_token 的 header/definition 参数数量曾在连续修复中短暂分叉；当前已恢复为稳定的 8 参数公开接口，内部固定以 logical_rows=1 调用 tile 路径。
- 单 token projection 的输出缓冲已压缩为 1 行后，发现 tile accumulation 仍按 32 行写的越界风险；当前已限制写入到真实 active_rows，避免 LM/head 与小 projection 的 native buffer 越界。
- QNN graph budget 现在也覆盖 Perlin diagnostic/full/hybrid graph 的实际 graphCreate 生命周期；诊断 fresh graph 创建失败时不再忽略 budget reset 结果。
- 生产 projection 仍保持 32-row bucket，以匹配已验证 HTP shape；logical_rows 只控制真实输出写入，不会把 32-row graph 错误地变成动态形状。
- 当前 decode 的主要潜在速度瓶颈仍是每个 token 对多层 affine4 projection 做 streaming weight reads；mlx_safetensors.cpp 已使用线程局部 persistent ifstream 和连续 full-K tile reads。下一阶段应优先用 affine4_io profiler 实测 read_us/bytes，再决定是否增加有界 hot-tile cache，不能直接把数 GB 权重常驻内存。
## 2026-10-08 final audit pass updates

- 当前 `main` HEAD：7301aa7e1a224352f96f5e306a6354d28393de51。
- 单 token projection 的输出写入边界已修复：`logical_rows=1` 时只写真实第 0 行，避免 32-row bucket 与 1-row 输出缓冲之间的越界。
- `ornith15_run_projection_token` 已恢复与 header 一致的公开签名，内部仍固定使用 32-row HTP bucket 和 `logical_rows=1`。
- decode forward 已复用 `step.hidden` 与 `runtime.work_c`，减少每 token/每层临时 native allocation，同时保持现有数学路径不变。
- executor 的 binary16 helper 已补齐 subnormal 转换，避免静态参数被错误 flush-to-zero。
- Perlin diagnostic/full/hybrid graph 已纳入 QNN graph budget 计数；diagnostic budget reset 失败不会再被静默忽略。
- GGUF/llama.cpp 兼容后端的 `src0 × src1` 参数顺序已修正为 activation × weight，防止旧兼容路径返回错误矩阵。
- 当前 CI 仍在 job steps 建立前失败，最新运行没有有效 compile steps/logs；因此 Java/C++/APK 仍不能标记为“已编译通过”。
- 当前下一性能重点仍是 affine4 projection streaming I/O。目标是以 profiler 数据驱动有界复用/预取，而不是把多 GB 权重整体常驻，必须同时守住 <4 GiB runtime RAM 与可接受 decode 速度。