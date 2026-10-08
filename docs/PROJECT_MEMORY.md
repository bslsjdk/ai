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

## 当前工作方式
用户只用手机/MT 管理器，不会直接维护复杂 C++ 工程。
因此助手应直接修改仓库、提交修复并尽量验证 CI/build，不要把实现工作重新丢给用户。

## 诊断规则
- 没有实际 CI/build 证据时，不得声称“已编译成功”。
- 每次修复先确认仓库是 `bslsjdk/ai`。
- 每次涉及模型名称、架构或配置时，先核对官方模型资料或仓库 config。
- 修复完成后优先检查编译引用、头文件引用、CMake 源文件列表和 Java/JNI 调用链。
