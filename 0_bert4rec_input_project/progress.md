# BERT4Rec 输入张量更新进度

## 会话信息

- 开始日期：2026-10-28
- 当前阶段：验证与 Git 同步

## 操作记录

### 2026-10-28

- 建立研究记录。
- 确认现有 Wiki 未收录 BERT4Rec 输入张量格式。
- 审计现有推荐与事件序列页面，确定新增独立概念页，并在两页中补充关联链接。
- 下载并解析 arXiv:1904.06690v2，核验 Cloze 目标、序列截断、测试末尾 mask 和 leave-one-out 设置。
- 克隆作者参考实现 `FeiSun/BERT4Rec`，核验 TFRecord 字段、shape、padding、词表与实际模型输入。
- 核验 RecBole、Transformers4Rec 与 Hugging Face MLM collator，确认 PyTorch 生态存在稀疏位置、mask schema 和密集 `labels=-100` 三种表达。
- 新增来源页与输入张量概念页，并更新推荐、Qwen 事件序列、研究导航、来源索引、全局索引和日志。

## 验证记录

| 检查项 | 状态 | 备注 |
|---|---|---|
| 原论文与作者代码 | 已完成 | arXiv v2、作者 commit `615eaf2` |
| 实现差异 | 已完成 | RecBole、Transformers4Rec、Hugging Face |
| Wiki 链接与索引 | 已完成 | 新页面 Wikilink、新增索引路径、主来源外链通过 |
| Git 同步 | 进行中 | 待创建当天版本分支 |
