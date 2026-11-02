# Qwen3-VL风控继续预训练数据与操作指南

> 整理日期：2026-11-02。承接 [[Qwen3-VL风控知识继续预训练与任务适配]]，以 `Qwen/Qwen3-VL-4B-Instruct` 与固定版本 LLaMA-Factory 为操作参照。仅核对官方实现与文件格式，未执行模型加载或GPU训练。

## 数据与训练的最小闭环

领域文档 → 清洗与版本核对 → 按同源文档或案件簇划分 → 分词与语义切块 → `text` 语料 → 语言主干LoRA继续预训练 → 独立评估 → 按需要另做任务SFT。

可复制交付位于仓库 [qwen3-vl-risk-cpt/README.md](../../../qwen3-vl-risk-cpt/README.md)，包含八条虚构教学文本、数据登记、来源清单与20步链路验证配置。虚构数据只说明格式，不构成业务证据或效果测量。

## 一、收集知识单元

优先选择稳定定义、风险机制、产品流程、字段文档、成熟规则解释及正常与边界案例。保留“定义、形成机制、适用条件、观测要求、反例、版本”的关系，而非只做术语清单。

字段文档需要保留单位、统计窗口、采集来源、缺失语义与可用时间。复盘文档可以讲完整过程；用于事前判断的任务样本，不能将事后人工确认或处理结果放入决策输入，见 [[风控表征学习中的生成机制、风险定义与表征分工]]。

没有成熟结果时，不要把分析推测改成真实风险标签。经复核的合成变体仍属于原始来源的派生数据，不增加独立证据数量。频繁变化的名单、阈值和实时状态优先从外部版本化系统提供。

## 二、先分组划分，再切块

1. 建清单，记录来源、版本、生效范围与同源分组。相同文档、案件、复盘、改写和合成变体归为一组。
2. 抽取正文，清理导航、模板、乱码和重复内容；保留表头、单位、否定词与上下文。OCR需额外核验。
3. 排除敏感、无使用权或无法核实的内容；对实体做一致去标识，避免抹掉案件关系。
4. 精确与近似去重；版本冲突显式标记或选定有效版本，不能静默混合。
5. 按组划分训练与验证；案件评估还应按时间留出并控制实体重叠。最终测试集不用于选检查点。
6. 用原模型Tokenizer计数，再按标题或完整段落切块；检查条件、机制和反例没有被分散或截断。

先切块再随机划分会导致同一原文跨集合，先生成问答再划分同样存在泄漏。文档数和字符数不足以表示训练量，应记录去重后的token数、主题覆盖与实际训练token预算。

## 三、训练正文与来源清单分开

训练JSONL每行一个对象：

```json
{"text":"主题：共享设备。\n机制：多个账户可能使用同一终端。\n正常反例：合法家庭共享也可能发生。\n证据要求：不能单凭账户数确定恶意。"}
```

这是虚构格式示例。预训练不要求人工标签或聊天消息。来源、版本与分组在独立清单中保存；理解正文必需的范围和版本同时写入正文。

本次LLaMA-Factory数据登记用 `columns.prompt: text`，并通过 `stage: pt` 选择普通正文语言建模。登记的 `formatting: alpaca` 是数据列转换方式，不代表正文已变成SFT问答。

## 四、固定框架版本并验证参数范围

核对版本：LLaMA-Factory提交 `9997aa078b4da35e04bb492a8a27d45141f4b80b`。该版本包含Qwen3-VL模型加载、多模态LoRA排除与纯文本PT工作流。Transformers `4.57.1` 位于其声明依赖范围内，官方模型映射包含Qwen3-VL；未执行安装依赖组合验证。

候选关键配置：

```yaml
model_name_or_path: Qwen/Qwen3-VL-4B-Instruct
stage: pt
finetuning_type: lora
lora_target: all
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
template: qwen3_vl_nothink
cutoff_len: 2048
packing: false
```

源码将Qwen3-VL的视觉位置嵌入、patch嵌入、视觉blocks、DeepStack融合列表与投影merger列入排除范围。实际运行仍须输出可训练参数名称，确认仅语言主干LoRA参与更新。冻结视觉参数不能证明图文能力不会变化。

首次建议1张H20验证4B纯文本LoRA链路，理由是先确认数据可用与任务增益，不急于并行扩大。已知资源菜单的240GB是主机内存，不代表GPU显存；容量和吞吐须实测。完整环境准备与命令见交付README。

## 五、截断与拼接边界

本次PT处理器在正文后追加结束符。`packing: false` 会将超长文本截断到 `cutoff_len`，因此需要提前做语义切块，并为结束符预留空间。

`packing: true` 会在数据映射批次内拼接文档、切定长块并丢弃尾部不足一块的内容。EOS不隔离跨文档注意力；短演示数据可能不产生完整块。框架限制 `neat_packing: true` 用于SFT，不能在此PT配置中用它解决边界问题。参见 [[自回归掩码、解码与生成评测]]。

## 六、训练与验收

先做一步探测，确认正文解码、有效标签、可训练模块、有限损失与峰值显存；再做演示短跑。演示配置的20步只验证链路，不能据此判断知识学习成功。

换入真实数据后，删除演示 `max_steps` 并按epoch或token预算设置训练量；保守学习率、LoRA rank和上下文长度均是待验证候选。检查验证损失与保留能力，不用训练loss单独选模型。

正文回放可按PT文本格式混合；指令回放需单独SFT或明确实现多目标训练，不能把 `messages` 数据混入该PT配置后声称完成能力保留。

领域PPL只说明同处理协议下的正文预测表现。最终还需评估新案例、正常反例、规则范围、证据不足处理、固定误报率下召回、输出格式和图文能力。比较原模型、CPT、直接SFT、CPT后SFT，细节见 [[Qwen3-VL风控知识继续预训练与任务适配]]。

LoRA产物需要同一基座和适配器一起加载，或验证后合并导出；仅加载原基座不会自动包含本次更新。

## 来源与验证边界

- [官方数据格式](https://github.com/hiyouga/LLaMA-Factory/blob/9997aa078b4da35e04bb492a8a27d45141f4b80b/data/README.md)、[PT处理器](https://github.com/hiyouga/LLaMA-Factory/blob/9997aa078b4da35e04bb492a8a27d45141f4b80b/src/llamafactory/data/processor/pretrain.py)、[PT工作流](https://github.com/hiyouga/LLaMA-Factory/blob/9997aa078b4da35e04bb492a8a27d45141f4b80b/src/llamafactory/train/pt/workflow.py)：直接读取源码，单一框架实现证据。
- [多模态模块配置](https://github.com/hiyouga/LLaMA-Factory/blob/9997aa078b4da35e04bb492a8a27d45141f4b80b/src/llamafactory/model/model_utils/visual.py)、[模型加载器](https://github.com/hiyouga/LLaMA-Factory/blob/9997aa078b4da35e04bb492a8a27d45141f4b80b/src/llamafactory/model/loader.py)：参数排除与加载路径。
- [官方Qwen3-VL LoRA SFT示例](https://github.com/hiyouga/LLaMA-Factory/blob/9997aa078b4da35e04bb492a8a27d45141f4b80b/examples/train_lora/qwen3vl_lora_sft.yaml)：仅用于模型与模板参照，不当作CPT成功证据。
- [Transformers 4.57.1映射](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/auto/modeling_auto.py)、[Qwen模型卡](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)：模型类别与权重变体。

本页文件格式和配置已做静态核对；未运行框架参数解析、模型加载、GPU训练或业务效果评测。资源和超参数属于实验建议。
