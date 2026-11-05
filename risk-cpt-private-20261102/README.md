# 风控继续预训练知识单元与首批语料

基于2026-11-02候选清单中的11份材料，补读相关正文并沿标准明细引用补读2份标准。交付48个知识单元，其中40个业务知识单元、8个治理方法单元。来源改写均保留读取时点与适用范围；不是全篇、全表编译，也不是当前生效政策库。

## 文件

- [知识单元审阅稿](知识单元审阅稿.md)：方便逐条核对五字段与最小支持摘录。
- [结构化知识单元](data/knowledge_units.jsonl)：定义、机制、证据、反例、范围，加来源、状态和字段来源类型。
- [训练语料](data/risk_cpt_train.jsonl)：35条，每行仅有 `text`。
- [验证语料](data/risk_cpt_valid.jsonl)：5条，每行仅有 `text`。
- [治理方法辅助语料](data/risk_cpt_auxiliary.jsonl)：8条；默认训练配置不混入。
- [来源清单](data/sources.json)、[逐条映射](data/manifest.jsonl)、[来源覆盖与排除清单](来源覆盖与排除清单.md)。
- [数据登记](data/dataset_info.json)、[候选配置](configs/cpt_lora_candidate.yaml)。
- [结构核验](verification.json)、[Tokenizer核验](tokenizer_validation.json)、[统计](statistics.json)。

来源URL保存在独立审计文件中，训练正文不含URL、真实实体、策略编号、内部阈值或实验指标。用户于2026-11-05明确授权将语料、来源链接和摘录发布到公开GitHub仓库；本目录保留初次交付时的名称。该发布授权不等于业务签核或训练效果验证。

## 如何解释“反例”和“证据”

40个业务单元中，5个具有原文正常场景，1个具有原文误关联边界，34个原文未提供反例。缺失在JSONL中以 `null` 和 `not_provided` 标记；正文直接注明未提供，不补造政策或案例。8个方法单元均无正常反例。

证据字段是可观察信息或核对项，不代表已有具体案件或验证结果。来源快照已人工核对，业务签核、当前平台生效、图像内容及实际效果未核验。

## 划分与数据规模

先确定来源组，再渲染正文。手法表、标签导航与其引用标准整体留在训练侧；养号方案和社交意图两组整体留在验证侧。原文、引用补充与派生单元不跨集合。

本批没有独立最终测试集。5条验证仅覆盖养号汇总和社交意图，不能代表全部业务知识、规则判断或新案件泛化；不得用其损失宣称风险识别效果。

采用Qwen3-VL-4B-Instruct官方Tokenizer，固定版本 `ebb281ec70b05090aa6165b016eac8ec08e71b17`。按 `add_special_tokens=false` 分词，并为PT追加一个EOS计数：

| 集合 | 文本数 | Token数（含每条一个EOS） |
|---|---:|---:|
| 训练 | 35 | 5123 |
| 验证 | 5 | 726 |
| 方法辅助 | 8 | 1015 |

最长样本283 token，全部低于2048。仅下载Tokenizer做长度检查，未下载或加载模型权重。语料规模很小，适合来源复核与数据链路探测，不能当作充分的领域继续预训练数据量。未通过重复改写扩充独立知识。

## 使用

在本包根目录运行，框架和环境按 [已有样例包说明](../qwen3-vl-risk-cpt/README.md)准备。框架参照固定提交 `9997aa078b4da35e04bb492a8a27d45141f4b80b`；应使用与本次Tokenizer匹配的基座版本或重新做分词核验。

完成业务复核后，可先执行一步探测：

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/cpt_lora_candidate.yaml max_steps=1 warmup_ratio=0 do_eval=false print_param_status=true output_dir=./outputs/qwen3vl4b-risk-cpt-probe
```

配置默认使用训练和验证文件，冻结视觉与投影模块，仅语言主干LoRA候选更新；`packing: false`。若执行实验，先建议1张H20进行链路检查，理由是4B小批纯文本LoRA无须先扩大到多卡；容量和吞吐仍需实测。

本次未启动上述命令，也未做模型加载、训练框架参数解析、GPU训练或效果评测。配置静态核验不保证实际环境可运行。

## 复核和再生成

`build_corpus.py` 中保存人工选定的抽象单元和最小摘录，不保存原文全文；可修改后再生成。修改正文后必须重新做Tokenizer长度核验，再运行：

```bash
python3 build_corpus.py
python3 verify_corpus.py
```

结构检查不替代业务复核。所有单元都保留 `source_checked_business_review_pending` 状态；未发布、已下线、冲突和无完整范围的判定规则没有进入训练正文。
