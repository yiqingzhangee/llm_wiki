# Qwen 2\.5 与 Qwen 3 系列微调深度调研

## 研究背景与核心发现

Qwen（通义千问）系列大模型已迭代至 Qwen 3 世代，在开源社区和企业应用中占据重要位置。Qwen 2\.5 与 Qwen 3 在架构、训练范式和微调策略上存在显著差异：Qwen 2\.5 延续经典 Dense Transformer 架构，而 Qwen 3 引入了 **混合思维模式（Thinking/Non\-thinking）** 和 **大规模 MoE（Mixture\-of\-Experts）架构**，对微调提出了全新要求。

**核心结论**

- Qwen 2\.5 微调成熟稳定，SFT \+ LoRA/QLoRA 为主流路径，DPO/GRPO 用于偏好对齐

- Qwen 3 微调范式从"文本生成对齐"转向"推理过程建模"，四阶段后训练流水线（SFT → 推理 RL → 思维融合 → 通用 RL）成为标准

- GRPO 算法在两代模型中均表现优异，显存效率高于 PPO，已成为 RL 对齐的首选

- LLaMA\-Factory、ms\-swift、Unsloth 构成三大核心微调框架，各有侧重

- Qwen 3 的思考模式微调和 MoE 专家管理是新增的技术难点

---

## Qwen 2\.5 系列微调方法体系

### SFT（监督微调）：基础路径

Qwen 2\.5 的微调起点是 **监督微调（Supervised Fine\-Tuning）**，通过高质量指令\-回答对训练模型遵循特定任务指令。

**数据格式要求：** Qwen 2\.5 使用 **ChatML** 模板作为标准对话格式[\[1\]](https://arxiv.org/pdf/2505.09388)，典型结构如下：

```Plain Text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{用户输入}<|im_end|>
<|im_start|>assistant
{模型回答}<|im_end|>

```

**关键超参数建议：**

|**参数**|**全参数微调**|**LoRA/QLoRA**|
|---|---|---|
|学习率|1e\-5 至 5e\-6|1e\-4 至 2e\-4|
|Batch Size|128（有效批次）|16\-64|
|Epoch|2\-3（防过拟合）|3\-5|
|Warmup Ratio|0\.03\-0\.1|0\.03\-0\.1|
|上下文长度|32768（默认）|可根据数据调整至 8192|

### LoRA 与 QLoRA：参数高效微调的主力

**LoRA（Low\-Rank Adaptation）** 是 Qwen 2\.5 最广泛使用的微调方式。核心思路是冻结预训练权重，仅训练注入的低秩矩阵。

**LoRA 推荐配置：**

- **目标模块**：`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`（覆盖 Attention 和 FFN 层）

- **Rank（秩）**：8\-64，任务复杂度越高可适当增大

- **Alpha**：通常设为 Rank 的 2 倍（如 rank=16 时 alpha=32）

- **Dropout**：0\.05\-0\.1

**QLoRA** 在 LoRA 基础上引入 **4\-bit NormalFloat（NF4）量化**，大幅降低显存需求。对于 Qwen2\.5\-7B 模型，QLoRA 仅需约 **6\-12GB 显存**，可在消费级 GPU（如 RTX 3090/4090）上运行。

**显存需求对照：**

|**模型规模**|**全参数 \(FP16\)**|**LoRA \(FP16\)**|**QLoRA \(4\-bit\)**|**推荐 GPU**|
|---|---|---|---|---|
|Qwen2\.5\-1\.5B|\~8GB|\~5GB|\~3GB|RTX 3060 12GB|
|Qwen2\.5\-7B|\~32GB|\~18GB|\~8GB|RTX 4090 24GB|
|Qwen2\.5\-14B|\~60GB|\~35GB|\~14GB|A100 40GB|
|Qwen2\.5\-72B|\~300GB|\~160GB|\~48GB|4×A100 80GB|

### DPO 与 GRPO：偏好对齐方法

在 SFT 之后，**偏好对齐**是提升模型输出质量的关键步骤。

**DPO（Direct Preference Optimization）：**

- 直接通过偏好数据对优化，无需训练奖励模型

- Qwen 2\.5 通过 DPO 优化后，指令对齐胜率可提升至约 70%

- 数据格式要求提供 chosen/rejected 回答对

**GRPO（Group Relative Policy Optimization）：**

- Qwen 团队在 Qwen2\.5\-Math 上率先引入的算法[\[2\]](https://cdn.jsdelivr.net/gh/yanfeng98/paper-is-all-you-need/papers/00069-Qwen3_Technical_Report.pdf)

- **核心机制**：对同一 prompt 采样多个回答，在组内计算相对优势，无需单独的价值网络

- **显存优势**：相比 PPO 节省约 40% 显存（不需要 Value Network 和 Reward Model 的同时加载）

- 特别适合**数学推理**和**代码生成**场景，可结合外部验证器（如代码执行结果、数学答案验证）作为奖励信号

### Qwen 2\.5 特定变体的微调要点

**Qwen2\.5\-Coder（代码模型）**

- 数据配比建议：70% 代码数据 \+ 20% 通用指令 \+ 10% 数学推理[\[1\]](https://arxiv.org/pdf/2505.09388)

- 支持 Fill\-in\-the\-Middle（FIM）格式训练

- 推荐使用代码执行反馈作为 GRPO 的奖励信号

- 上下文长度建议设为 16384\-32768

**Qwen2\.5\-Math（数学模型）**

- 推荐使用 GRPO 结合外部数学验证器进行强化学习

- 链式思考（Chain\-of\-Thought）数据对性能提升显著

- 训练数据中应包含多种解题路径

- 注意处理 LaTeX 格式的特殊 token

---

## Qwen 3 系列微调方法体系

### 架构层面的根本性变化

Qwen 3 的微调逻辑变化源于两大架构创新：

**混合思维模式（Hybrid Thinking）**

Qwen 3 首次将"思考模式"与"非思考模式"整合至单一模型中[\[3\]](https://arxiv.org/abs/2505.09388)：

- **思考模式**：适用于数学、编程等复杂任务，通过 `<think>...</think>` 标签生成推理链

- **非思考模式**：适用于对话、摘要等低延迟任务

- 微调需通过特定数据标签来平衡两种模式的表现

**MoE 架构深度优化**

旗舰版 Qwen3\-235B\-A22B 拥有 128 个专家，每 Token 仅激活 22B 参数[\[4\]](https://arxiv.org/html/2505.09388v1)：

- 移除了 Qwen 2\.5\-MoE 中的共享专家（Shared Expert）

- 引入全局批次负载均衡（Global\-batch Load Balancing Loss）

- 计算成本相比同规模 Dense 模型降低约 83%，推理速度提升 40%[\[5\]](https://raw.githubusercontent.com/QwenLM/Qwen3/main/Qwen3_Technical_Report.pdf)

### 四阶段后训练流水线

Qwen 3 的后训练从传统的单阶段 SFT 演变为四阶段流程[\[6\]](https://raw.githubusercontent.com/QwenLM/Qwen3-Omni/refs/heads/main/assets/Qwen3_Omni.pdf)：

**阶段一：长链式思考冷启动（Long\-CoT Cold Start）**

- 使用多种推理模式（如 CoT、Tool\-Integrated Reasoning）数据微调

- 建立基础推理能力

**阶段二：推理强化学习（Reasoning RL）**

- 使用基于规则的奖励（Rule\-Based Reward）进行 GRPO 训练

- 奖励信号来源：数学验证器、代码沙盒执行结果

**阶段三：思维模式融合（Thinking Mode Fusion）**

- 将 Thinking（带思考过程）和 Non\-thinking（直接回答）数据混合微调

- 关键比例：约 70\-80% Thinking 数据 \+ 20\-30% Non\-thinking 数据[\[7\]](https://velog.io/@bb1702/Qwen3-Technical-Report-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)

- 使模型学会根据问题复杂度自适应选择模式

**阶段四：通用强化学习（General RL）**

- 使用通用奖励模型和人类偏好数据进行全面对齐

- 涵盖安全性、有用性、格式遵循等多维度

### GRPO 在 Qwen 3 中的核心地位

GRPO 从 Qwen 2\.5\-Math 的实验性算法，升级为 Qwen 3 全系列的标准 RL 算法[\[8\]](https://jaehun.me/en/posts/paper-review-qwen-3-technical-report/)。

**GRPO 工作原理：**

1. 对每个 prompt 生成 G 个候选回答（组大小通常为 8\-16）

2. 使用外部奖励函数（规则或模型）对每个回答评分

3. 在组内计算相对优势（advantage），高于均值的回答得到正强化

4. 直接更新策略网络，无需 Critic/Value 网络

**相比 PPO 的优势：**

- 显存占用大幅降低（不需要 Value Network）

- 训练信号更稳定（组内相对比较消除了奖励尺度问题）

- 天然契合 MoE 架构的大批量 Rollout 需求

### 思维模式微调的数据工程

**思维模式数据格式要求**

微调 Qwen 3 的思考能力时，训练数据必须包含 `<think>` 标签包裹的推理过程。对于已有高质量答案但缺失推理链的数据集，推荐使用 Qwen3\-235B 或 DeepSeek\-R1 等旗舰模型生成推理轨迹进行补充[\[9\]](https://www.reddit.com/r/LocalLLaMA/comments/1pnuy9y/how_to_you_actually_finetune_qwen3/)。

**思维模式数据示例格式：**

```JSON
{
  "messages": [
    {"role": "user", "content": "What is 25 * 37?"},
    {"role": "assistant", "content": "<think>\nLet me calculate 25 * 37.\n25 * 37 = 25 * (40 - 3) = 1000 - 75 = 925\n</think>\nThe answer is 925."}
  ]
}

```

### 小模型蒸馏策略

Qwen 3 的小规模模型（0\.6B\-8B）不再从头进行完整的 RL 训练，而是采用**强到弱蒸馏（Strong\-to\-Weak Distillation）**[\[10\]](https://www.reddit.com/r/unsloth/comments/1tnffw1/qwen_3635_multimodal_with_llamacpp_using_unsloth/)：

- 使用 235B 旗舰模型作为教师，通过教师 logits 指导小模型

- 显著缩短训练时间，同时继承大模型的推理能力

- 这一策略对于资源受限的团队尤为重要：可直接从开源 Qwen3\-235B 蒸馏定制化小模型

---

## 微调框架深度对比

### 主流框架能力矩阵

|**特性**|**LLaMA\-Factory**|**ms\-swift \(SWIFT\)**|**Unsloth**|**Axolotl**|**XTuner**|**HF TRL**|
|---|---|---|---|---|---|---|
|Qwen 2\.5 支持|完全支持|完全支持|完全支持|完全支持|完全支持|完全支持|
|Qwen 3 支持|完全支持|完全支持（一等公民）|完全支持（含 MoE 优化）|支持（含 VL 变体）|支持（含超大 MoE）|基础支持|
|SFT / LoRA / QLoRA|全部支持|全部支持|全部支持|全部支持|全部支持|全部支持|
|DPO / GRPO / PPO|全部支持|全部支持|GRPO 支持|DPO 支持|部分支持|全部支持|
|分布式训练|DeepSpeed/FSDP|Megatron\-LM 深度集成|开源版主要单卡|FSDP/DeepSpeed|DeepSpeed|Accelerate|
|可视化 GUI|有（LLaMA Board）|有|无|无|无|无|
|速度优化|标准|标准|2x 加速（Triton 内核）|标准|极致显存优化|标准|
|显存优化|中等|中等|降低 60\-80%|中等|8GB 跑 7B 模型|基础|

### LLaMA\-Factory：全能型工具箱

**定位**：功能最全面、社区最活跃的微调框架[\[11\]](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune)。

**核心优势：**

- 覆盖 SFT、DPO、PPO、ORPO、GRPO 等全部对齐算法

- 提供可视化 GUI（LLaMA Board），零代码配置微调任务

- 原生支持 Qwen 2\.5 和 Qwen 3 的 ChatML 模板

- 支持 100\+ 种模型，迁移成本极低

- 社区生态丰富，文档完善，中文支持友好

**典型使用方式：**

```YAML
# LLaMA-Factory 配置示例
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
stage: sft
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32
template: qwen
dataset: my_custom_data
output_dir: output/qwen25-7b-lora
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3

```

**局限**：在单卡极限性能优化上略逊于 Unsloth。

### ms\-swift \(SWIFT\)：Qwen 官方生态首选

**定位**：ModelScope 官方出品，对 Qwen 系列具有"一等公民"级原生支持[\[12\]](https://unsloth.ai/docs/models/qwen3.5/fine-tune)。

**核心优势：**

- 支持 600\+ LLM 和 300\+ MLLM

- 率先集成 Qwen 3 "思考模型"的 GRPO 训练支持

- 通过 **Megatron\-SWIFT** 支持超大规模全参数训练（如 Qwen3\-235B）

- 提供 `--loss_scale ignore_empty_think` 选项，专门保护思考模式的推理链路[\[13\]](https://unsloth.ai/docs/models/qwen3.6)

- 与 ModelScope Hub 无缝集成，模型下载和部署一体化

**Qwen 3 专属功能：**

- 支持 Megatron\-LM 架构下的 MoE 全参数微调

- 可在 8×A800/H100 环境下快速部署 235B 级别模型训练

- 内置思维链数据质量校验工具

### Unsloth：极致效率的单卡方案

**定位**：通过定制化 Triton 内核实现极致训练效率[\[14\]](https://unsloth.ai/docs/models/qwen3.5)。

**性能数据：**

- 微调速度提升 **2 倍**

- 显存占用降低 **60\-80%**

- Qwen3\-30B\-A3B \(MoE\) 可在 **24GB 显存**显卡上运行

- 支持 Qwen3\-235B 使用 FP8 LoRA，256GB 显存即可运行[\[15\]](https://unsloth.ai/docs/models/tutorials/qwen3-coder-how-to-run-locally)

**Qwen 3 MoE 专属优化：**

- 开发了针对 MoE 架构的专属 Triton 内核

- 支持 GRPO 训练，集成奖励函数接口

- 提供预配置的 Qwen 3 微调 Notebook

**局限**：开源版主要针对单卡优化，多卡 FSDP 或分布式训练需要 Pro/Studio 版本[\[16\]](https://qwen.readthedocs.io/en/latest/training/unsloth.html)。

### 其他框架速览

**Axolotl**

- 配置文件驱动，高度可定制

- 支持 FSDP、DeepSpeed 深度集成

- 原生支持 Qwen 3\-VL 多模态微调[\[17\]](https://github.com/QwenLM/Qwen3-VL/issues/1918)

- 配置曲线较陡，适合高级用户

**XTuner**

- 极致显存优化，支持 8GB 跑 7B 模型

- 专为超大 MoE 架构（如 Qwen3\-Coder 480B）设计训练引擎[\[18\]](https://arxiv.org/pdf/2502.16274)

- 由 InternLM 团队维护

- 适合显存严重受限的场景

---

## 微调注意事项与实战避坑

### 数据工程：微调成败的决定因素

**数据质量 \> 数据数量**

Qwen 系列微调遵循"少而精"原则。在多数任务上，1000\-5000 条高质量数据的微调效果优于 50000 条低质量数据。数据清洗和去重是微调前的必做步骤。

**数据格式规范：**

- Qwen 2\.5：严格使用 ChatML 格式，注意 `<|im_start|>` 和 `<|im_end|>` 特殊 token

- Qwen 3：在 ChatML 基础上增加 `<think>` 标签支持

- **Tokenizer 注意事项**：确认 `bos_token` 和 `eos_token` 的正确设置，错误配置会导致生成无法正常终止

**数据配比建议：**

- 通用微调：任务数据 \+ 5\-10% 通用对话数据（防止灾难性遗忘）

- 代码场景：70% 代码 \+ 20% 通用指令 \+ 10% 数学推理

- 思维模式微调：70\-80% 含 `<think>` 的推理数据 \+ 20\-30% 直接回答数据

### 灾难性遗忘的缓解策略

灾难性遗忘是微调最常见的问题之一——模型在学习新任务时，原有通用能力大幅退化[\[19\]](https://qiita.com/bostonchou/items/29e692724fb0e8418a6f)。

**解决方案矩阵：**

|**策略**|**实施方式**|**效果**|
|---|---|---|
|LoRA 微调|冻结原模型参数，仅训练低秩适配器，从根本上保留原始权重|最直接有效|
|混合数据回放|在训练集中加入 5\-10% 的通用能力数据或之前的训练数据[\[20\]](https://cs224r.stanford.edu/projects/pdfs/Final_Project_Report.pdf)|简单可靠|
|MSSR 框架|记忆感知自适应回放（Memory\-aware Self\-adaptive Rehearsal），根据样本记忆强度动态调整回放频率[\[21\]](https://blog.csdn.net/weixin_35794316/article/details/157015984)|推理保留率显著提升|
|Elastic Weight Consolidation|对重要参数施加正则化约束，限制其更新幅度|适用于多任务序贯微调|

### 采样与解码配置：Qwen 3 的敏感区

**Qwen 3 思考模式严禁使用贪婪解码**

在 Thinking 模式下使用贪婪解码会导致严重的性能退化和无限循环[\[22\]](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)。这是 Qwen 3 微调中最常被忽略的问题之一。

**推荐推理参数配置：**

|**参数**|**推荐值**|**说明**|
|---|---|---|
|temperature|0\.6|平衡创造性和一致性|
|top\_p|0\.95|控制采样范围|
|top\_k|20|限制候选 token 数量|
|min\_p|0|不设最小概率阈值|
|repetition\_penalty|1\.0|避免重复但不过度惩罚|
|presence\_penalty|1\.5|Qwen 3 专用，防止推理振荡[\[23\]](https://jarvislabs.ai/blog/vllm-sglang-trtllm-comparison)|

### MoE 模型微调的特殊挑战

微调 Qwen 3 的 MoE 变体（如 Qwen3\-235B\-A22B、Qwen3\-30B\-A3B）需要额外关注：

**专家塌陷（Expert Collapse）问题：**

- 表现：微调后仅有少数专家被更新，其余专家权重几乎不变

- 原因：负载均衡损失系数配置不当或 Router 权重被过度更新

- 解决方案：

    - 冻结 Router 层或限制其学习率[\[24\]](https://www.yottalabs.ai/post/best-llm-inference-engines-in-2026-vllm-tensorrt-llm-tgi-and-sglang-compared)

    - 合理设置负载均衡损失系数 α

    - 配置 `expert_model_parallel_size` 参数

**MoE 显存管理：**

- 激活时显存波动大，需开启 `GroupedGEMM` 优化

- 建议启用 `MoE Shared Expert Overlap` 策略[\[25\]](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)

- 对于 235B 级别模型，使用 **FP8 LoRA** 是唯一可行的单机微调方案

### 长上下文处理

|**模型**|**原生上下文**|**扩展方案**|
|---|---|---|
|Qwen 2\.5|32K\-128K（取决于具体变体）|YaRN 扩展至更长上下文|
|Qwen 3|256K \(262,144 tokens\)[\[26\]](https://www.cnblogs.com/gpustack/p/19917640)|YaRN 可扩展至 1M tokens[\[27\]](https://www.arxiv.org/pdf/2601.06288)|

**微调中的长上下文注意事项：**

- 训练数据长度应覆盖目标推理场景的典型输入长度

- 使用 Packing 策略提高长序列训练效率

- 在 vLLM 等引擎中通过设置 `rope_scaling` 因子启用 YaRN[\[28\]](https://renue.co.jp/posts/llm-inference-optimization-vllm-speculative-decoding-prompt-caching-2026)

- 长上下文微调显存需求随序列长度线性增长，建议使用 FlashAttention 2

### 量化对微调模型的影响

**Qwen 3 量化性能表现：**

- 4\-bit 量化：速度提升 9\-15%，准确率损失仅 0\-3%，基本可忽略[\[29\]](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Qwen3-Dense.html)

- FP8 量化：在 H100 等硬件上已成为微调和推理的默认平衡点

- 推荐格式：资源受限环境下使用 Q4**K\_****M 或 Q8**0 GGUF 格式[\[30\]](https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/)

**量化微调注意事项：**

- 学习率需降低：量化模型对高学习率敏感，建议从 2e\-4 降至 5e\-5[\[31\]](https://blog.csdn.net/xuebinding/article/details/150924452)

- 使用更长的 warmup 步数避免 Loss 爆炸

- 量化后模型权重的梯度分布变化，需适当调整优化器参数

---

## 常见失败模式与调试指南

### Loss 异常诊断

|**现象**|**可能原因**|**解决方案**|
|---|---|---|
|Loss 爆炸（NaN/Inf）|学习率过高；量化模型权重不稳定；数据中存在异常 token|降低学习率至 5e\-5；增加 warmup；清洗数据中的特殊字符[\[32\]](https://cs224r.stanford.edu/projects/pdfs/CS224R_Final_Report%20(8)1.pdf)|
|Loss 不下降|学习率过低；数据格式不匹配；LoRA 目标模块配置错误|检查数据模板是否匹配 ChatML；验证 LoRA 模块配置|
|Loss 下降但效果差|过拟合；数据质量问题；评估方式不当|减少 epoch；增加数据多样性；使用验证集|
|验证 Loss 上升|过拟合信号|启用 Early Stopping；降低学习率；增加 dropout|

### 思考能力丧失（Qwen 3 特有）

**现象**：微调后模型不再生成 `<think>` 推理过程，或推理质量严重下降。

**原因**：训练数据中不包含带 `<think>` 标签的思维链数据[\[33\]](https://blog.csdn.net/weixin_33816734/article/details/157288637)。

**解决方案：**

1. 确保训练数据中包含带思维过程的样本

2. 使用 ms\-swift 的 `--loss_scale ignore_empty_think` 参数保护推理链路

3. 在混合数据中维持至少 20\-30% 的思维链数据

### 推理振荡（Reasoning Oscillation）

**现象**：模型在复杂问题上陷入循环思维，反复输出相似推理步骤。

**解决方案：**

- 微调数据中增加针对性的"负面奖励"样本[\[34\]](https://github.com/sandy1990418/Finetune-Qwen2.5-VL)

- 调整 `presence_penalty` 参数（推荐 1\.5）

- 设置 `thinking_budget` 限制推理链长度

### 模式塌陷（Mode Collapse）

**现象**：过度微调思考能力后，模型在简单问题上也生成冗长推理过程。

**解决方案：**

- 使用混合比例 SFT（70\-80% 思维数据 \+ 20\-30% 非思维数据）[\[7\]](https://velog.io/@bb1702/Qwen3-Technical-Report-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)

- 在 GRPO 训练中加入简单问题样本，引导模型学会跳过不必要的推理

---

## 推理部署优化

微调完成后的推理部署同样至关重要：

|**引擎**|**核心优势**|**适用场景**|**性能参考**|
|---|---|---|---|
|vLLM|PagedAttention，吞吐量领先|高频 API 服务，中大规模部署|RTX 5090 下 7B 模型达 156 tok/s[\[35\]](https://www.f22labs.com/blogs/complete-guide-to-fine-tuning-qwen2-5-vl-model/)|
|SGLang|结构化输出优化，RadixAttention|结构化输出、动态 Prompt 场景|在结构化输出场景延迟最低[\[36\]](https://deepwiki.com/QwenLM/Qwen2.5-VL/7.2-fine-tuning)|
|TensorRT\-LLM|NVIDIA 深度优化|超大规模云端部署|高并发下比 vLLM 快 30\-50%[\[37\]](https://blog.csdn.net/gitblog_00615/article/details/144048376)|
|llama\.cpp \(GGUF\)|CPU/边缘设备推理|本地部署、嵌入式场景|支持多种量化格式|
|Ollama|一键部署，用户友好|个人开发、快速原型|基于 llama\.cpp 封装|

**LoRA 模型部署注意事项：**

- 可选择合并 LoRA 权重后部署（推荐用于生产环境）

- 也可使用 vLLM 的动态 LoRA 加载功能，同时服务多个微调版本

- 合并后的模型可以直接使用标准推理引擎，无需额外适配

---

## Qwen 2\.5 与 Qwen 3 微调策略对比

|**维度**|**Qwen 2\.5**|**Qwen 3**|
|---|---|---|
|主流架构|Dense Transformer|Dense \+ MoE 双线|
|后训练范式|SFT → DPO/GRPO|四阶段流水线（SFT → Reasoning RL → Mode Fusion → General RL）|
|核心 RL 算法|DPO 为主，GRPO 用于数学|GRPO 全面升级为标准算法|
|数据格式|ChatML|ChatML \+ `<think>` 标签|
|多语言支持|29 种语言|119\+ 种语言[\[38\]](https://www.alibabacloud.com/help/en/pai/use-cases/training-evaluation-compression-and-deployment-of-qwen2-5-coder-model)|
|上下文长度|32K\-128K|256K（可 YaRN 扩展至 1M）|
|小模型训练|独立训练|强到弱蒸馏（从 235B 蒸馏）|
|推荐首选框架|LLaMA\-Factory|LLaMA\-Factory（通用）/ ms\-swift（思维\+大规模）/ Unsloth（单卡效率）|
|特有挑战|Tokenizer 配置、灾难性遗忘|思维模式维护、专家塌陷、推理振荡、采样敏感性|

---

## 微调决策流程建议

针对不同场景的微调方案选择建议：

1. **资源充足 \+ 追求极致效果**：使用 ms\-swift 进行 Qwen 3 全参数微调或 Megatron 训练，配合四阶段后训练流程

2. **中等资源 \+ 通用任务**：使用 LLaMA\-Factory 进行 LoRA/QLoRA 微调，数据量 1000\-10000 条

3. **资源受限 \+ 快速迭代**：使用 Unsloth 在单卡上进行 QLoRA 微调，Qwen3\-30B\-A3B 可在 24GB 显存运行

4. **推理能力增强**：优先使用 GRPO \+ 规则奖励（数学/代码），确保训练数据包含思维链

5. **多语言/跨语言**：Qwen 3 天然支持 119\+ 语言，微调时仅需针对 3\-5% 关键层调整即可保持语言一致性[\[39\]](https://www.labellerr.com/blog/fine-tune-qwen-2-5-vl/)

6. **长文本场景**：充分利用 Qwen 3 的 256K 原生上下文，必要时用 YaRN 扩展，训练时使用 FlashAttention 2 和 Packing

# 参考资料

[\[](https://arxiv.org/pdf/2505.09388)[1\] 2025\-05\-15 Qwen3 Technical Report Qwen Team https://huggingface\.co/Qwen](https://arxiv.org/pdf/2505.09388)

[\[](https://siteproxy-1.j7o.com/default/https/arxiv.org/pdf/2508.10355)[2\] Making Qwen3 Think in Korean with Reinforcement Learning](https://siteproxy-1.j7o.com/default/https/arxiv.org/pdf/2508.10355)

[\[](https://blog.csdn.net/PAN_Andy/article/details/148053004)[3\] Qwen3 技术报告深度解析:双模式架构与多语言能力升级\_](https://blog.csdn.net/PAN_Andy/article/details/148053004)[自适应思考预算\-CSDN博客](https://blog.csdn.net/PAN_Andy/article/details/148053004)

[\[](https://velog.io/@bb1702/Qwen3-Technical-Report-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)[4\] \[](https://velog.io/@bb1702/Qwen3-Technical-Report-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)[쉬운 리뷰\] Qwen3 Technical Report 제대로 이해하기](https://velog.io/@bb1702/Qwen3-Technical-Report-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)

[\[](https://markaicode.com/qwen3-vs-qwen25-performance-migration-tutorial/)[5\] Qwen3 vs Qwen2\.5: Performance Improvements and Migration Tutorial](https://markaicode.com/qwen3-vs-qwen25-performance-migration-tutorial/)

[\[](https://dataguy.in/artificial-intelligence/qwen-3-vs-qwen-2-5-ai-model-upgrade-analysis/)[6\] Qwen 3 Vs Qwen 2\.5: MoE Upgrade, Benchmarks, Deployment Wins](https://dataguy.in/artificial-intelligence/qwen-3-vs-qwen-2-5-ai-model-upgrade-analysis/)

[\[](https://m.weibo.cn/detail/5280870963610330)[7\] 宝玉xp的微博](https://m.weibo.cn/detail/5280870963610330)

[\[](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/slime_qwen3_4B_GRPO.html)[8\] RL training with slime on AMD GPUs](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/slime_qwen3_4B_GRPO.html)

[\[](https://www.reddit.com/r/LocalLLaMA/comments/1pnuy9y/how_to_you_actually_finetune_qwen3/)[9\] r/LocalLLaMA on Reddit: How to you actually fine\-tune Qwen3?](https://www.reddit.com/r/LocalLLaMA/comments/1pnuy9y/how_to_you_actually_finetune_qwen3/)

[\[](https://jaehun.me/en/posts/paper-review-qwen-3-technical-report/)[10\] \[](https://jaehun.me/en/posts/paper-review-qwen-3-technical-report/)[Paper Review\] Qwen 3 Technical Report](https://jaehun.me/en/posts/paper-review-qwen-3-technical-report/)

[\[](https://aicoolies.com/comparisons/ms-swift-vs-llama-factory)[11\] ms\-swift vs LLaMA\-Factory — ModelScope Fine\-Tuning Hub vs Universal Training Orchestrator](https://aicoolies.com/comparisons/ms-swift-vs-llama-factory)

[\[](https://github.com/modelscope/ms-swift)[12\] GitHub \- modelscope/ms\-swift: Use PEFT or Full\-parameter to CPT/SFT/DPO/GRPO 600\+ LLMs \(Qwen3\.6, DeepSeek\-V4, GLM\-5\.1, InternLM3, Llama4, \.\.\.\) and 300\+ MLLMs \(Qwen3\-VL, Qwen3\-Omni, InternVL3\.5, Ovis2\.5, GLM4\.5v, Gemma4, Llava, Phi4, \.\.\.\) \(AAAI 2025\)\. · GitHub](https://github.com/modelscope/ms-swift)

[\[](https://swift.readthedocs.io/en/v3.9/BestPractices/Qwen3-Best-Practice.html)[13\] Qwen3 Best Practices](https://swift.readthedocs.io/en/v3.9/BestPractices/Qwen3-Best-Practice.html)

[\[](https://qwen.readthedocs.io/en/latest/training/unsloth.html)[14\] Unsloth](https://qwen.readthedocs.io/en/latest/training/unsloth.html)

[\[](https://unsloth.ai/docs/models/qwen3.5/fine-tune)[15\] Qwen3\.5 Fine\-tuning Guide \| Unsloth Documentation](https://unsloth.ai/docs/models/qwen3.5/fine-tune)

[\[](https://theaiengineer.substack.com/p/unsloth-vs-axolotl-vs-llama-factory)[16\] Unsloth vs Axolotl vs LLaMA\-Factory \- by Paolo Perrone](https://theaiengineer.substack.com/p/unsloth-vs-axolotl-vs-llama-factory)

[\[](https://www.spheron.network/blog/axolotl-vs-unsloth-vs-torchtune/)[17\] Axolotl vs Unsloth vs TorchTune: Best LLM Fine\-Tuning Frameworks in 2026 \| Spheron Blog](https://www.spheron.network/blog/axolotl-vs-unsloth-vs-torchtune/)

[\[](https://github.com/internlm/xtuner)[18\] GitHub \- InternLM/xtuner: An efficient, flexible and full\-featured toolkit for fine\-tuning LLM \(InternLM2, Llama3, Phi3, Qwen, Mistral, \.\.\.\)](https://github.com/internlm/xtuner)

[\[](https://lilys.ai/de/notes/openai-agent-builder-20260208/qwen-3-easy-fine-tune-reasoning)[19\] QWEN\-3: DER EINFACHSTE WEG ZUM FINE\-TUNING MIT REASONING 🙌](https://lilys.ai/de/notes/openai-agent-builder-20260208/qwen-3-easy-fine-tune-reasoning)

[\[](https://arxiv.org/html/2603.06610v1)[20\] CapTrack: Multifaceted Evaluation of Forgetting in LLM Post\-Training](https://arxiv.org/html/2603.06610v1)

[\[](https://arxiv.org/pdf/2603.09892.pdf)[21\] MSSR: Memory\-Aware Adaptive Replay for Continual LLM Fine\-Tuning](https://arxiv.org/pdf/2603.09892.pdf)

[\[](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune)[22\] Qwen3 \- How to Run \& Fine\-tune \| Unsloth Documentation](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune)

[\[](https://medium.com/@aminfadaeinejad.edu/fine-tuning-qwen3-vl-a-practical-guide-for-vision-language-model-adaptation-d66d3f61e888)[23\] Fine\-Tuning Qwen3\-VL: A Practical Guide for Vision\-Language Model Adaptation \| by Amin Fadaeinejad \| May, 2026 \| Medium](https://medium.com/@aminfadaeinejad.edu/fine-tuning-qwen3-vl-a-practical-guide-for-vision-language-model-adaptation-d66d3f61e888)

[\[](https://blog.csdn.net/gitblog_02181/article/details/149627553)[24\] 【限时免费】 释放Qwen3\-30B\-A3B\-Base的全部潜力:一份基于LoRA和MoE优化的微调指南\-CSDN博客](https://blog.csdn.net/gitblog_02181/article/details/149627553)

[\[](https://xtuner.readthedocs.io/en/latest/get_started/installation.html)[25\] Installation — XTuner 0\.2\.0 documentation](https://xtuner.readthedocs.io/en/latest/get_started/installation.html)

[\[](https://deepwiki.com/QwenLM/Qwen3-VL/4.5-context-length-extension)[26\] Context Length Extension \| QwenLM/Qwen3\-VL \| DeepWiki](https://deepwiki.com/QwenLM/Qwen3-VL/4.5-context-length-extension)

[\[](https://blog.csdn.net/gitblog_00060/article/details/150638042)[27\] Qwen3进阶技巧:超长上下文与高级功能\-CSDN博客](https://blog.csdn.net/gitblog_00060/article/details/150638042)

[\[](https://unsloth.ai/docs/models/tutorials/qwen3-coder-how-to-run-locally)[28\] Untitled](https://unsloth.ai/docs/models/tutorials/qwen3-coder-how-to-run-locally)

[\[](https://joshua8.ai/quantization-nvidia-space-heaters/)[29\] Friday Morning Space Heaters: The Real Cost of Quantization](https://joshua8.ai/quantization-nvidia-space-heaters/)

[\[](https://dasroot.net/posts/2026/05/qwen-36-quantization-bf16-gguf-q4-k-m-q8-0/)[30\] Qwen 3\.6 Quantization Deep Dive: BF16 vs GGUF, Q4](https://dasroot.net/posts/2026/05/qwen-36-quantization-bf16-gguf-q4-k-m-q8-0/)[K\_](https://dasroot.net/posts/2026/05/qwen-36-quantization-bf16-gguf-q4-k-m-q8-0/)[M vs Q8](https://dasroot.net/posts/2026/05/qwen-36-quantization-bf16-gguf-q4-k-m-q8-0/)[0](https://dasroot.net/posts/2026/05/qwen-36-quantization-bf16-gguf-q4-k-m-q8-0/)

[\[](https://www.ai-manual.ru/article/pochemu-vasha-lora-ne-uchitsya-illyuziya-poter-i-kak-eyo-slomat/)[31\] Почему ваша LoRA не учится: иллюзия потерь и как её сломать](https://www.ai-manual.ru/article/pochemu-vasha-lora-ne-uchitsya-illyuziya-poter-i-kak-eyo-slomat/)

[\[](https://blog.csdn.net/wayle123/article/details/159669568)[32\] 10\_](https://blog.csdn.net/wayle123/article/details/159669568)[故障排查与最佳实践:Qwen模型LoRA\_](https://blog.csdn.net/wayle123/article/details/159669568)[RLHF微调全流程问题诊断指南\-CSDN博客](https://blog.csdn.net/wayle123/article/details/159669568)

[\[](https://codersera.com/blog/qwen3-vl-4b-instruct-vs-qwen3-vl-4b-thinking-complete-2025-guide/)[33\] Qwen3\-VL\-4B: Instruct vs Thinking \(2026 Guide\)](https://codersera.com/blog/qwen3-vl-4b-instruct-vs-qwen3-vl-4b-thinking-complete-2025-guide/)

[\[](https://www.reddit.com/r/Qwen_AI/comments/1rjywf1/qwen_35_9b_4bit_tuning_suggestions_to_avoid/)[34\] r/Qwen\_](https://www.reddit.com/r/Qwen_AI/comments/1rjywf1/qwen_35_9b_4bit_tuning_suggestions_to_avoid/)[AI on Reddit: Qwen 3\.5 9B 4\-bit tuning suggestions to avoid \.\.\.](https://www.reddit.com/r/Qwen_AI/comments/1rjywf1/qwen_35_9b_4bit_tuning_suggestions_to_avoid/)

[\[](https://markaicode.com/benchmarks/vllm-qwen-3-rtx-5090-tokens-per-second-benchmark/)[35\] vLLM \+ Qwen 3 on RTX 5090: 156 tok/s at Q4\_](https://markaicode.com/benchmarks/vllm-qwen-3-rtx-5090-tokens-per-second-benchmark/)[K\_](https://markaicode.com/benchmarks/vllm-qwen-3-rtx-5090-tokens-per-second-benchmark/)[M](https://markaicode.com/benchmarks/vllm-qwen-3-rtx-5090-tokens-per-second-benchmark/)

[\[](https://www.yottalabs.ai/post/best-llm-inference-engines-in-2026-vllm-tensorrt-llm-tgi-and-sglang-compared)[36\] Best LLM Inference Engines \(2026\): vLLM, SGLang \& TensorRT\-LLM \| Yotta Labs](https://www.yottalabs.ai/post/best-llm-inference-engines-in-2026-vllm-tensorrt-llm-tgi-and-sglang-compared)

[\[](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)[37\] A Comprehensive Comparison of LLM Inference Engines: vLLM, TGI, TensorRT\-LLM, SGLang, llama\.cpp, and Ollama \| Enterprise Unified LLM API Gateway \(One Key for All Models\) \| n1n\.ai](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)

[\[](https://raw.githubusercontent.com/QwenLM/Qwen3/main/Qwen3_Technical_Report.pdf)[38\] Qwen3 Technical Report](https://raw.githubusercontent.com/QwenLM/Qwen3/main/Qwen3_Technical_Report.pdf)

[\[](https://openreview.net/forum?id=r00UxTl8El)[39\] LinguaMap: Which Layers of LLMs Speak Your Language and How to Tune Them?](https://openreview.net/forum?id=r00UxTl8El)

# 参考资料

[\[](https://arxiv.org/pdf/2505.09388)[1\] 2025\-05\-15 Qwen3 Technical Report Qwen Team https://huggingface\.co/Qwen](https://arxiv.org/pdf/2505.09388)

[\[](https://cdn.jsdelivr.net/gh/yanfeng98/paper-is-all-you-need/papers/00069-Qwen3_Technical_Report.pdf)[2\] 2025\-05\-13 Qwen3 Technical Report Qwen Team https://huggingface\.co/Qwen](https://cdn.jsdelivr.net/gh/yanfeng98/paper-is-all-you-need/papers/00069-Qwen3_Technical_Report.pdf)

[\[](https://arxiv.org/abs/2505.09388)[3\] \[](https://arxiv.org/abs/2505.09388)[2505\.09388\] Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

[\[](https://arxiv.org/html/2505.09388v1)[4\] Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)

[\[](https://raw.githubusercontent.com/QwenLM/Qwen3/main/Qwen3_Technical_Report.pdf)[5\] Qwen3 Technical Report](https://raw.githubusercontent.com/QwenLM/Qwen3/main/Qwen3_Technical_Report.pdf)

[\[](https://raw.githubusercontent.com/QwenLM/Qwen3-Omni/refs/heads/main/assets/Qwen3_Omni.pdf)[6\] Qwen3\-Omni Technical Report](https://raw.githubusercontent.com/QwenLM/Qwen3-Omni/refs/heads/main/assets/Qwen3_Omni.pdf)

[\[](https://velog.io/@bb1702/Qwen3-Technical-Report-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)[7\] \[](https://velog.io/@bb1702/Qwen3-Technical-Report-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)[쉬운 리뷰\] Qwen3 Technical Report 제대로 이해하기](https://velog.io/@bb1702/Qwen3-Technical-Report-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)

[\[](https://jaehun.me/en/posts/paper-review-qwen-3-technical-report/)[8\] \[](https://jaehun.me/en/posts/paper-review-qwen-3-technical-report/)[Paper Review\] Qwen 3 Technical Report](https://jaehun.me/en/posts/paper-review-qwen-3-technical-report/)

[\[](https://www.reddit.com/r/LocalLLaMA/comments/1pnuy9y/how_to_you_actually_finetune_qwen3/)[9\] r/LocalLLaMA on Reddit: How to you actually fine\-tune Qwen3?](https://www.reddit.com/r/LocalLLaMA/comments/1pnuy9y/how_to_you_actually_finetune_qwen3/)

[\[](https://www.reddit.com/r/unsloth/comments/1tnffw1/qwen_3635_multimodal_with_llamacpp_using_unsloth/)[10\] r/unsloth on Reddit: QWen 3\.6/3\.5 multimodal with llama\.cpp \(using Unsloth models\)](https://www.reddit.com/r/unsloth/comments/1tnffw1/qwen_3635_multimodal_with_llamacpp_using_unsloth/)

[\[](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune)[11\] Qwen3 \- How to Run \& Fine\-tune \| Unsloth Documentation](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune)

[\[](https://unsloth.ai/docs/models/qwen3.5/fine-tune)[12\] Qwen3\.5 Fine\-tuning Guide \| Unsloth Documentation](https://unsloth.ai/docs/models/qwen3.5/fine-tune)

[\[](https://unsloth.ai/docs/models/qwen3.6)[13\] Qwen3\.6 \- How to Run Locally \| Unsloth Documentation](https://unsloth.ai/docs/models/qwen3.6)

[\[](https://unsloth.ai/docs/models/qwen3.5)[14\] Qwen3\.5 \- How to Run Locally \| Unsloth Documentation](https://unsloth.ai/docs/models/qwen3.5)

[\[](https://unsloth.ai/docs/models/tutorials/qwen3-coder-how-to-run-locally)[15\] Untitled](https://unsloth.ai/docs/models/tutorials/qwen3-coder-how-to-run-locally)

[\[](https://qwen.readthedocs.io/en/latest/training/unsloth.html)[16\] Unsloth](https://qwen.readthedocs.io/en/latest/training/unsloth.html)

[\[](https://github.com/QwenLM/Qwen3-VL/issues/1918)[17\] Why dose fine\-tune Qwen\-2\.5\-VL\-Instruct 3B and 7B with same dataset yield different performance in task · Issue \#1918 · QwenLM/Qwen3\-VL](https://github.com/QwenLM/Qwen3-VL/issues/1918)

[\[](https://arxiv.org/pdf/2502.16274)[18\] Fine\-Tuning Qwen 2\.5 3B for Realistic Movie Dialogue Generation Kartik Gupta](https://arxiv.org/pdf/2502.16274)

[\[](https://qiita.com/bostonchou/items/29e692724fb0e8418a6f)[19\] 個人化AIアシスタント開発のためのQwen2\.5\-3Bファインチューニング実践](https://qiita.com/bostonchou/items/29e692724fb0e8418a6f)

[\[](https://cs224r.stanford.edu/projects/pdfs/Final_Project_Report.pdf)[20\] Curriculum and Augmented RL Fine\-Tuning for Aligned Language Models](https://cs224r.stanford.edu/projects/pdfs/Final_Project_Report.pdf)

[\[](https://blog.csdn.net/weixin_35794316/article/details/157015984)[21\] Qwen2\.5\-0\.5B模型微调:领域适配指南\-CSDN博客](https://blog.csdn.net/weixin_35794316/article/details/157015984)

[\[](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)[22\] vLLM vs TensorRT\-LLM vs SGLang: H100 Benchmarks \(2026\) \| Spheron Blog](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)

[\[](https://jarvislabs.ai/blog/vllm-sglang-trtllm-comparison)[23\] vLLM, SGLang, or TensorRT\-LLM? Picking an \.\.\.](https://jarvislabs.ai/blog/vllm-sglang-trtllm-comparison)

[\[](https://www.yottalabs.ai/post/best-llm-inference-engines-in-2026-vllm-tensorrt-llm-tgi-and-sglang-compared)[24\] Best LLM Inference Engines \(2026\): vLLM, SGLang \& TensorRT\-LLM \| Yotta Labs](https://www.yottalabs.ai/post/best-llm-inference-engines-in-2026-vllm-tensorrt-llm-tgi-and-sglang-compared)

[\[](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)[25\] Qwen3\.5 \& Qwen3\.6 Usage Guide \- vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)

[\[](https://www.cnblogs.com/gpustack/p/19917640)[26\] 27B 超越 397B，Qwen3\.6 模型 vLLM vs SGLang 部署指南与性能表现 \- GPUStack \- 博客园](https://www.cnblogs.com/gpustack/p/19917640)

[\[](https://www.arxiv.org/pdf/2601.06288)[27\] AIConfigurator: Lightning\-Fast Configuration Optimization for Multi\-Framework LLM Serving](https://www.arxiv.org/pdf/2601.06288)

[\[](https://renue.co.jp/posts/llm-inference-optimization-vllm-speculative-decoding-prompt-caching-2026)[28\] LLM推論最適化完全ガイド2026｜vLLM・TensorRT\-LLM・SGLang・Speculative Decoding・Prompt Caching](https://renue.co.jp/posts/llm-inference-optimization-vllm-speculative-decoding-prompt-caching-2026)

[\[](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Qwen3-Dense.html)[29\] Qwen3\-Dense\(Qwen3\-0\.6B/8B/32B\)](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Qwen3-Dense.html)

[\[](https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/)[30\] Portable vLLM Model Inference Kernels in Helion](https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/)

[\[](https://blog.csdn.net/xuebinding/article/details/150924452)[31\] 深度解析主流大模型推理部署框架\(vLLM、SGLang、TensorRT\-LLM、Ollama、XInference\)\+Qwen3全流程部署、优化、运维案例\-CSDN博客](https://blog.csdn.net/xuebinding/article/details/150924452)

[\[](https://cs224r.stanford.edu/projects/pdfs/CS224R_Final_Report%20(8)1.pdf)[32\] Instruction Following via Self\-Revision: Fine\-Tuning Qwen with Teacher Feedback and Staged Curriculum](https://cs224r.stanford.edu/projects/pdfs/CS224R_Final_Report%20(8)1.pdf)

[\[](https://blog.csdn.net/weixin_33816734/article/details/157288637)[33\] 为什么选ms\-swift?Qwen2\.5\-7B微调框架对比评测\-CSDN博客](https://blog.csdn.net/weixin_33816734/article/details/157288637)

[\[](https://github.com/sandy1990418/Finetune-Qwen2.5-VL)[34\] GitHub \- sandy1990418/Finetune\-Qwen2\.5\-VL: Fine\-tuning Qwen2\.5\-VL for vision\-language tasks \| Optimized for Vision understanding \| LoRA \& PEFT support\. · GitHub](https://github.com/sandy1990418/Finetune-Qwen2.5-VL)

[\[](https://www.f22labs.com/blogs/complete-guide-to-fine-tuning-qwen2-5-vl-model/)[35\] Complete Guide to Fine\-tuning Qwen2\.5 VL Model \- F22 Labs](https://www.f22labs.com/blogs/complete-guide-to-fine-tuning-qwen2-5-vl-model/)

[\[](https://deepwiki.com/QwenLM/Qwen2.5-VL/7.2-fine-tuning)[36\] Fine\-tuning](https://deepwiki.com/QwenLM/Qwen2.5-VL/7.2-fine-tuning)

[\[](https://blog.csdn.net/gitblog_00615/article/details/144048376)[37\] Qwen2\.5\-7B开发者完全手册:从微调到部署的完整流程\-CSDN博客](https://blog.csdn.net/gitblog_00615/article/details/144048376)

[\[](https://www.alibabacloud.com/help/en/pai/use-cases/training-evaluation-compression-and-deployment-of-qwen2-5-coder-model)[38\] Fine\-Tune and Deploy Qwen2\.5 \- Coder with SFT \& DPO on PAI \- Alibaba Cloud](https://www.alibabacloud.com/help/en/pai/use-cases/training-evaluation-compression-and-deployment-of-qwen2-5-coder-model)

[\[](https://www.labellerr.com/blog/fine-tune-qwen-2-5-vl/)[39\] Qwen 2\.5\-VL 7B Fine\-Tuning Guide for Segmentation](https://www.labellerr.com/blog/fine-tune-qwen-2-5-vl/)

