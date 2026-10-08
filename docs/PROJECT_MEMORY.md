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
- Safetensors 权重保持 tile/streaming 读取，不整体加载约 5GB 模型文件；全注意力 cache 当前明确限制为 4096-token 有界窗口。
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
- 模型请求上下文仍配置为 65536 tokens，模型原生 context=262144；当前 native full-attention resident window 仍为 4096，因此长上下文的第一阶段依靠用户历史骨架 + 相关召回来保持连续性，而不是直接分配 64K KV 导致超过 4 GiB。
- 后续真正扩展到 64K/128K 有效上下文时，优先研究分页/分块 KV、按需重算/CPU/存储卸载和更精确的 token budget，不得一次性分配多 GB 常驻 KV。

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
