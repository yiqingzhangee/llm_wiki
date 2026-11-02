# Qwen3-VL-4B 风控文本继续预训练样例

本包演示从已有 `Qwen/Qwen3-VL-4B-Instruct` 做纯文本领域继续预训练（CPT）：保留整个模型，以 LoRA 更新语言主干，排除视觉与投影模块。若你持有的是 Thinking 或其他变体，须重新核对模板及模型配置。

核对日期：2026-11-02。已核对官方代码与文件格式；**未安装训练框架、未加载模型、未做 GPU 训练或推理验证**。这里是供首次实验验证的配置，不是官方风控配方。所有示例文本均为虚构教学材料，不代表真实案例、规则或测量结果。

## 1. 先准备哪些文档

优先选稳定、经过业务复核的知识：

| 类型 | 一份文档应保留的内容 |
|---|---|
| 产品与业务流程 | 参与角色、业务环节、事件之间的关系 |
| 风险机制 | 定义、形成机制、适用条件、证据要求、正常反例 |
| 字段说明 | 字段语义、单位、窗口、来源、缺失含义、可用时间 |
| 规则说明 | 判断逻辑、适用范围、版本、生效时间 |
| 历史复盘 | 完整过程；明确分开事前观测和事后结果 |
| 正常与边界案例 | 为什么现象类似但判断不同、还缺什么证据 |

易变的阈值、名单、实时状态由版本化外部系统提供。CPT 不能保证及时更新这些事实。

一篇知识单元可采用下面的组织方式，但不必强行把所有自然文档改成这个模板：

```text
主题：……
定义：……
机制：……
适用范围与版本：……
可观察证据：……
正常反例或不能推出的结论：……
```

保留因果链和反例比反复出现术语更有用。不要把无标签日志自动写成风险判定，也不要无限生成相似问答来充当新增独立知识。

## 2. 清洗、划分、切块的顺序

1. 建立文档清单：记录 `doc_id`、来源、版本、生效范围、更新时间与同源 `group_id`。同一原文、案件及其复盘/改写/合成变体归到同一组。
2. 抽取正文：移除导航、页脚、重复标题、乱码、废弃模板；保留表格表头、单位、否定词、上下文和版本条件。扫描文档先核对 OCR，尤其是数值和“不/未”等字词。
3. 清理敏感信息与无使用权材料；排除令牌、密钥、真实个人标识等内容。案例做一致的去标识处理，不把所有实体混成同一个名字。
4. 做精确及近似去重；冲突版本明确标记或只选目标版本，不把两个版本混写成同一事实。
5. **先按组划分，再切块或改写**。稳定知识可先按组留出约 10% 作验证，这只是起点；案件评测还需留出较新的时间段并控制实体重叠。另保留最终测试集，不用它选检查点。
6. 使用模型原有 Tokenizer 计数，按标题、段落切成能保留完整知识关系的块。此包 `cutoff_len: 2048`，应预留结束符开销；每块含结束符的总 token 数不超过该值。
7. 如长文切块必须重复定义或表头，记录重复来源并控制重叠。清洗后检查跨集合重复、文本长度、主题覆盖和正常反例覆盖。

不能把“2048 token”当成“2048 字”。当前预训练处理器在 `packing: false` 时会截断超长样本；若不提前切分，尾部反例、条件或结论可能丢失。

首次小实验可先精读并处理一个覆盖完整主题的文档批次，统计独立 token 数，再决定训练预算。没有统一的最低文档数；只有几篇材料时应同时比较检索与直接 SFT。

## 3. 最终文件格式

`data/risk_cpt_demo_train.jsonl` 每行一个 JSON 对象，只有训练正文：

```json
{"text":"主题：设备共享。\n机制：多个账户可能使用同一终端。\n正常反例：家庭成员也可能共享。\n证据要求：不能仅凭账户数量判定恶意。"}
```

这是普通文档语言建模，不需要 `instruction`、`messages` 或人工标签。不要手动塞图像占位符或自己添加 EOS；本次核对的处理器会为正文追加结束符。

`data/dataset_info.json` 将 `text` 映射到框架内部 `prompt` 列：

```json
{
  "risk_cpt_demo_train": {
    "file_name": "risk_cpt_demo_train.jsonl",
    "formatting": "alpaca",
    "columns": {"prompt": "text"}
  }
}
```

`alpaca` 在这里是注册/列映射方式，**不表示训练正文被改成指令微调样本**；训练阶段由 `stage: pt` 决定。

出处与分组信息在 `manifest_demo.jsonl`，不由该配置送入模型。若版本、适用范围是理解正文必需的条件，它们还应写进 `text`。

正式使用时替换演示训练/验证文件与清单，或新增自己的注册名称并修改 YAML。8 条演示数据只能验证读入和训练步骤；演示验证集也不能测量领域泛化。

## 4. 训练环境与首次运行

在 Linux NVIDIA GPU 训练环境中使用 Python 3.11 或以上。先选择匹配驱动/CUDA 的 PyTorch 构建；本次核对的框架要求 PyTorch 至少 2.4，且其加载器限制含 Conv3D 模型使用 PyTorch 2.9.x。不能直接复制一条 CUDA 安装命令到所有机器。

下面固定本次核对的 LLaMA-Factory 提交。命令仅供你执行，本次没有运行：

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory
git -C LLaMA-Factory checkout 9997aa078b4da35e04bb492a8a27d45141f4b80b
python -m pip install -e ./LLaMA-Factory "transformers==4.57.1"
python -m pip check
```

`transformers==4.57.1` 位于该提交的依赖范围内，其官方模型映射包含 Qwen3-VL；这不等于已经验证所有依赖组合。安装后保存依赖锁定记录，固定实际模型权重版本或使用本地不可变权重目录。

上述安装与下列训练命令都以**当前样例包根目录**为工作目录，即本 README 所在目录。YAML 中 `./data` 和 `./outputs` 均相对此工作目录。

先做一条真实训练步的检查，单独输出：

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/cpt_lora_smoke.yaml max_steps=1 warmup_steps=0 do_eval=false print_param_status=true output_dir=./outputs/qwen3vl4b-risk-cpt-probe
```

检查可训练参数名称：应为语言主干 LoRA 参数，不能出现视觉及投影层可训练参数；同时检查实际解码样本、有效标签数量、损失是否有限及 GPU 峰值显存。新数据应清除或更新旧预处理缓存。

通过后执行 20 步演示：

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/cpt_lora_smoke.yaml
```

已有同名输出时，用新的 `output_dir`；需要精确续跑时使用框架的 `resume_from_checkpoint`，不要仅靠加载适配器冒充恢复优化器状态。

如果你有本地完整权重，在 YAML 或命令覆盖 `model_name_or_path=/实际模型目录`。保留模型所需 Tokenizer、Processor 和配置文件，不要只替换一个权重文件。

## 5. 配置如何改成正式实验

样例的关键配置如下：

```yaml
stage: pt
finetuning_type: lora
lora_target: all
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
template: qwen3_vl_nothink
packing: false
```

`lora_target: all` 经本次框架版本的多模态排除逻辑筛选，但首次运行仍必须核对实际参数名称。这里保留视觉结构并冻结它，不是删除视觉模型，也不能保证视觉任务能力不受语言更新影响。

`template` 沿用官方 Instruct 示例；`stage: pt` 的正文处理路径不调用 SFT 对话编码，模板仍可能影响 Tokenizer 配置。不要把官方 SFT 示例直接当作 CPT 配方。

先用 **1 张 H20** 做 4B、LoRA、纯文本、2048 上下文的链路实验，理由是先验证数据与增益，暂不需要多卡并行。用户已有菜单对应 23 CPU / 240 GB **主机内存**；该数值不是显存。能否容纳、吞吐如何，取决于实测显存、软件、长度及激活开销，本包没有给出硬件通过保证。

正式实验建议按以下顺序调整，每项都属于候选而非最优值：

- 用真实训练、验证文件；将演示 `max_steps: 20` 删除，先设 `num_train_epochs: 1.0`，或根据 token 预算设步数，两者不要无意混用。不能把演示小数据重复 20 步理解成有效预训练。
- 学习率以 `1e-5` 为初始候选，比较更小或稍大的更新；LoRA rank 先用 8，再根据领域增益与保留能力决定是否比较 16。
- `per_device_train_batch_size: 1`、梯度累积先按显存和吞吐调整。有效样本批量为每卡批量 × 累积步数 × 卡数；不同文本长度下，样本批量不等于固定 token 批量。
- 从 2048 上下文开始，长知识关系确有需要时再比较 4096；每次改变长度重新做截断与显存检查。
- 用真实验证曲线和保留能力选检查点。训练损失下降、验证损失上涨时，减少重复或停止继续训练，不能只看训练 loss。

若启用 `packing: true`，本次处理器会在数据映射批次内拼接文档、切定长块并丢弃不足一块的尾部；EOS 不会隔离跨文档注意力。短演示数据可能因此无法生成有效样本。不要在 PT 中设置 `neat_packing: true`，该框架限制它用于 SFT。

## 6. CPT 后如何做任务适配与评估

CPT 训练的是“给定前文预测下一个 token”。要输出风险判断、证据与指定格式，应另准备有成熟标签的 SFT 数据，在 CPT 适配后继续做 SFT，或比较直接 SFT。

CPT 文本语料的回放可与领域正文同格式混合；指令回放则需单独的 SFT 阶段或显式的多目标训练实现，不能把 `messages` 数据塞进当前 PT 配置就认为保留了指令能力。

至少保留以下对照：原模型、CPT、直接 SFT、CPT 后 SFT。各组评测使用同一输入证据、规则版本与检索条件；SFT 阶段使用相同数据和配方。需要判断计算收益时，增加预算匹配对照。

检查：

1. 独立领域正文 loss / perplexity；相同 Tokenizer、长度和处理协议下比较，不能比较不同分词器的 PPL。
2. 未见案例的机制理解、正常反例、适用条件和证据不足处理。
3. 在相同误报率或精确率要求下的风险召回，以及逐类表现。
4. 指令遵循、JSON 格式、通用文本和需要保留的 OCR / 图文任务。

LoRA 输出主要是适配器，不是自动替换后的完整模型。推理必须加载同一基座和输出适配器，或在验证后合并导出；随后重新做文本与图文验证。不要仅加载原始 Qwen 权重后期待出现训练效果。

## 7. 官方依据与验证范围

源码固定在 `9997aa078b4da35e04bb492a8a27d45141f4b80b`：

- [官方数据格式](https://github.com/hiyouga/LLaMA-Factory/blob/9997aa078b4da35e04bb492a8a27d45141f4b80b/data/README.md)：预训练 `text` 列及注册方式。
- [预训练处理器](https://github.com/hiyouga/LLaMA-Factory/blob/9997aa078b4da35e04bb492a8a27d45141f4b80b/src/llamafactory/data/processor/pretrain.py)：结束符、截断、拼接与丢尾。
- [预训练工作流](https://github.com/hiyouga/LLaMA-Factory/blob/9997aa078b4da35e04bb492a8a27d45141f4b80b/src/llamafactory/train/pt/workflow.py)：非 MLM 语言建模与评估。
- [视觉模块配置](https://github.com/hiyouga/LLaMA-Factory/blob/9997aa078b4da35e04bb492a8a27d45141f4b80b/src/llamafactory/model/model_utils/visual.py)：Qwen3-VL 注册键与 LoRA 排除机制。
- [官方 Qwen3-VL LoRA SFT 示例](https://github.com/hiyouga/LLaMA-Factory/blob/9997aa078b4da35e04bb492a8a27d45141f4b80b/examples/train_lora/qwen3vl_lora_sft.yaml)：模型与模板参照，非 CPT 验证。
- [模型加载器](https://github.com/hiyouga/LLaMA-Factory/blob/9997aa078b4da35e04bb492a8a27d45141f4b80b/src/llamafactory/model/loader.py)及[配置解析器](https://github.com/hiyouga/LLaMA-Factory/blob/9997aa078b4da35e04bb492a8a27d45141f4b80b/src/llamafactory/hparams/parser.py)：多模态模型加载、预训练参数限制。
- [Transformers 4.57.1 模型映射](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/auto/modeling_auto.py)和[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)。

已做 YAML/JSONL 解析、数据登记与分组一致性检查；框架参数解析和 GPU 运行均未执行。正式开始前须完成第 4 节的一步探测。
