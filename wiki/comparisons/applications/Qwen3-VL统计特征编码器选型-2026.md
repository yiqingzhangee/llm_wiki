# Qwen3-VL 统计特征编码器选型（2026）

> 来源依据：[[来源-Qwen3-VL统计特征编码器调研-2026]]
>
> 工程总则：[[Qwen异构特征接入与事件序列建模]]、[[LLM统计特征编码的证据仲裁]]

## 一句话结论

不存在经公开实验确认的 Qwen3-VL-4B 统计特征“最佳编码器”。建议按以下顺序推进：

1. **快速工程验证：字段语义加法融合 + 两层 MLP 连续向量编码器**；固定英文描述先离线编码，在线只融合取值；
2. **成熟常用方案：FT-Transformer 编码器**；
3. **创新实验：PLE/周期编码 + 治理元数据 + 浅层 FT + learned-query 压缩**。

编码器先输出可独立用于分类或晚融合的连续向量；只有需要接入 Qwen 时，才通过独立 projector 生成一个 soft token。先证明连续统计通道相比确定性文本化、同预算晚融合和专用表格模型有稳定增益，再增加结构复杂度。

## 统一输入输出契约

三种编码器必须共享同一接口，避免把 token 数、Qwen 训练范围或数据处理差异误判为编码器效果。

| 项目 | 统一约束 |
|---|---|
| 数值输入 | 训练集拟合的标准化/稳健缩放值、missing mask、可选 `log1p(age)` |
| 类别输入 | 每字段独立词表，显式 `UNK`、`MISSING`；只在训练集建词表 |
| 治理信息 | 单位、窗口、来源、版本、可用时间等由明确字段表示，不靠裸值猜测 |
| 编码器主输出 | `stat_vector:[B,256]` |
| 可选 Qwen 输出 | `stat_token:[B,1,2560]`；多 token 仅作为后续消融 |
| 接入位置 | prompt 内连续的 `<|stat_pad|>` 占位，由 wrapper 替换 embedding |
| Qwen 训练范围 | 第一阶段冻结 Qwen3-VL；第二阶段才增加语言主干 LoRA |
| 视觉输入 | 保持官方 image/video placeholder 和视觉链路不变 |
| 比较预算 | 相同 `M`、batch、数据切分、训练步数和主干更新范围 |

## 三方案对比

| 维度 | A：MLP 连续向量 | B：FT-Transformer | C：PLE-FT + 查询压缩 |
|---|---|---|---|
| 定位 | 最低成本 PoC | 成熟常用代表 | 创新性候选 |
| 数值编码 | 标准化值 + missing + age | 线性数值 token 或 PLE | PLE/周期编码 + missing + age |
| 类别编码 | 字段独立 embedding | 字段独立 embedding token | 类别 + 字段 + 来源/窗口 embedding |
| 字段交互 | MLP 隐式混合 | self-attention | 浅层 self-attention |
| 连续向量读出 | 固定字段向量拼接后两层 MLP | `[CLS]` 或池化 | learned-query 压缩 |
| Qwen 适配 | 独立 MLP 输出 1 个 token | 独立 projector | Q-Former/Perceiver 式 learned queries |
| 实现复杂度 | 低 | 中 | 高 |
| 调试难度 | 低 | 中 | 高 |
| 主要风险 | 交互能力不足、扁平化丢字段结构 | 小数据过拟合、计算增加 | 组件归因困难、训练不稳定、创新不等于收益 |
| 推荐时机 | 第一周 | A 达标后 | B 出现稳定但未满足的结构瓶颈后 |

## A：最低开发成本的快速验证

### 结构

```text
每字段：描述语义 + 字段 ID + 当前取值
        ↓
固定字段顺序拼接
        ↓
两层 MLP
        ↓
stat_vector [B,256]
        ↓ 可选
zero-gated projector → stat_token [B,1,2560]
```

建议初始配置：字段内部维度 64，聚合 MLP hidden size 256，输出 `stat_vector∈R^256`。只有接入 Qwen 时才增加 `256→512→2560` projector，并输出一个 token。门控参数 `g` 初始化为 0，使用 `tanh(g) * stat_token`，使模型从“不使用统计通道”平滑开始。

### 为什么先选它

- 固定 schema 下实现和排障成本最低；
- 能快速回答“连续统计通道是否比文本化更有效”；
- 不引入字段 self-attention 和查询压缩，失败时归因更清楚。

它不是理论上最强，而是信息价值最高的首个实验。若 A 无法超过文本化或晚融合，不应直接用更复杂编码器掩盖问题。

## A+：固定英文描述的推荐实现

当每个字段都有固定英文描述时，不应把描述文本在每个样本中重复送入 Qwen，也不应先把所有描述拼成一个全局向量再与整行数值拼接。最低成本方案是：**描述离线编码，取值在线编码，在字段级直接相加，再由两层 MLP 生成一个连续向量。**

### 离线 schema 编码

为每个字段维护规范化 schema，不只保存自然语言描述：

```json
{
  "name": "payment_amount_7d",
  "description": "Total successful payment amount in the past 7 days.",
  "type": "numeric",
  "unit": "USD",
  "window": "7d",
  "missing_meaning": "No successful payment was observed.",
  "enum_values": null
}
```

将字段组织成固定模板，避免描述文风差异成为伪信号：

```text
Feature: payment_amount_7d.
Type: numeric. Unit: USD. Window: past 7 days.
Meaning: Total successful payment amount in the past 7 days.
Missing means: No successful payment was observed.
```

默认直接使用**冻结的 Qwen3-VL-4B 文本主干**做一次离线前向，取最后一个非 padding token 的最终隐藏状态作为字段语义向量 `z_j∈R^2560`。该 token 在因果注意力下已读取前面的完整描述；后续仍通过可训练投影器适配统计通道，但无需增加第二个文本模型依赖。

若 schema 预处理必须独立于 Qwen 服务，可改用 `all-MiniLM-L6-v2` 或同类小型英文 sentence encoder 产生 384 维向量，再由可训练投影器对齐；这是部署解耦选项，不是首选。

缓存至少包含：

```text
schema_version
feature_name
description_hash
text_encoder_model_id
text_encoder_revision
pooling_method
embedding_dtype
embedding
```

任一描述、单位、时间窗、枚举定义、编码器 revision 或 pooling 方法变化，都必须生成新 cache version，不能静默覆盖旧向量。

### 在线字段 token

先把缓存语义降到小维度：

```text
s_j = P_sem(z_j) ∈ R^d,  推荐 d=64
```

数值字段的动态取值编码为：

```text
u_j = MLP_num([scaled(x_j), missing_j, log1p(age_j)]) ∈ R^d
```

最小方案直接做加法融合：

```text
h_j = LayerNorm(s_j + u_j + e_field_j)
```

这里 `s_j` 保留“该字段是什么”，`u_j` 表示“当前样本中的值是什么”，`e_field_j` 是可训练字段 ID embedding。FiLM 能增强条件化能力，但不是最小工程方案；只有加法融合出现稳定容量瓶颈时才升级。

枚举字段采用两档实现：

1. **最低成本默认**：`u_j=E_j[c_j]`，即每字段独立的可训练枚举 embedding，再与字段描述和字段 ID 直接相加；
2. **枚举标签有明确英文语义时**：离线编码 `Field description + Enum label + Enum meaning`，缓存每个枚举值的语义向量，并增加一个小型可训练 residual embedding。这样能利用 `"chargeback"`、`"verified merchant"` 等标签语义，同时保留任务特定校正能力。

`UNK`、`MISSING` 必须是不同枚举项。若枚举值只是无语义内部 ID，不应把 ID 字符串交给文本编码器假装存在语义。

### 从字段表示到连续向量和可选 token

固定 schema 下，按固定字段顺序拼接所有字段表示，再用两层 MLP 得到主输出：

```text
H = Concat(h_1,...,h_F)
stat_vector = MLP_row(H) ∈ R^256
```

该向量可直接接分类头或 late-fusion 模块。只有接入 Qwen 时才执行：

```text
stat_token = tanh(g) · MLP_project(stat_vector) ∈ R^(1×2560)
```

`g` 初始化为 0。固定 schema 下，扁平 MLP 比查询池化更少组件、更容易调试；schema 经常变化或字段数非常大时，才考虑 masked pooling、FiLM 或 learned queries。

### 训练参数与默认配置

| 组件 | 默认 |
|---|---|
| 描述编码器 | 冻结 Qwen3-VL-4B 文本主干，离线运行 |
| 缓存向量 | FP16，按 schema version 保存 |
| 字段内部维度 | `d=64` |
| 数值编码器 | `3 → 64` 单层线性层 |
| 枚举编码器 | 每字段独立 embedding，维度 64 |
| 语义投影 | `2560 → 64` |
| 行级编码器 | `F×64 → 256 → 256` 两层 MLP |
| Qwen 投影 | `256 → 512 → 2560`，zero gate，默认 1 token |
| 第一阶段训练 | 冻结 Qwen 与描述 cache，只训练上述轻量模块 |

最终 gate 以零影响开始，可降低随机统计 token 对预训练主干的早期扰动。FiLM 若作为后续升级，应将其最后一层零初始化。

### 必做消融

- **无描述**：移除 `s_j`，只保留字段 ID 与取值编码；
- **描述错配**：在字段间随机置换 `z_j`；若性能不降，说明模型没有真正使用描述；
- **名称对完整描述**：只编码字段名，对比名称 + 类型 + 单位 + 窗口 + missing 含义；
- **加法对 FiLM**：只有最小加法方案达标后才比较语义条件调制；
- **枚举 ID 对枚举语义**：仅对标签有真实语义的枚举执行；
- **连续向量直投影对多 token/query pooling**：判断额外压缩器是否值得保留。

成功标准不是“加入描述后训练指标提高”，而是完整描述优于名称/无描述，描述错配显著退化，并且时间外、实体外切片的收益仍然存在。

## B：成熟常用的 FT-Transformer

### 原理

FT-Transformer 的关键不是把一整行压成一个向量，而是先把每个字段变成一个 token，使 self-attention 能显式学习字段间关系。

对第 `j` 个数值字段 `x_j`，基础 token 可写为：

```text
t_j^(num) = x_j · W_j + b_j
```

其中 `W_j,b_j∈R^d` 是字段独有参数。这个形式把一个标量映射成 `d` 维 token，同时保留“数值大小沿一个可学习方向变化”的归纳偏置。缺失值不能直接当作 0，应由插补值、missing embedding 或 missing mask 明确区分。

对第 `j` 个类别字段取值 `c_j`：

```text
t_j^(cat) = E_j[c_j] + b_j
```

`E_j` 是字段独立的 embedding 表，必须包含 `UNK` 和 `MISSING`。字段独立词表可避免两个字段中相同整数编码被错误视为同一语义。`b_j` 或额外 column embedding 提供字段身份。

将所有字段 token 与可选 `[CLS]` 拼接：

```text
T_0 = [t_CLS; t_1; t_2; ...; t_F] ∈ R^((F+1)×d)
```

每个 Transformer block 以 PreNorm 残差结构执行：

```text
T'  = T + MultiHeadAttention(LayerNorm(T))
T'' = T' + FFN(LayerNorm(T'))
```

self-attention 权重由字段 token 间的 query-key 相似度决定，因此某字段的表示可按当前样本动态吸收其他字段信息。例如，“高交易额”是否异常可以同时依赖账户年龄、国家、设备类型和缺失状态，而不是由固定交叉项决定。其字段交互复杂度约为 `O(F²d)`；表格字段数通常远小于文本序列长度，但超宽表仍需控制层数和字段数量。

原始 FT-Transformer 通常读取最终 `[CLS]` 表示并接 MLP 分类头。接入 Qwen 时不应直接照搬分类头，而应保留上下文化字段 token：

```text
H_fields = FT(T_0)[:, 1:, :]
S = Compress(H_fields)       # [B, M, d]
Z = Project(S)               # [B, M, 2560]
```

`Compress` 可先用 mean/attention pooling，只有字段数大于 token 预算且简单池化不足时才升级为 learned queries。这样 FT-Transformer 负责表格内部交互，Qwen3-VL 负责统计、文本和视觉证据的联合使用。

### 优势与局限

- 优势：字段级结构明确，类别/数值接口统一；attention 可学习样本相关的跨字段交互；输出天然是 token 序列。
- 局限：基础线性数值 token 对复杂阈值和局部非线性表达有限；小数据上可能过拟合；计算随字段数平方增长；缺失、单位和时间语义不会自动出现，仍需显式编码。
- 证据边界：FT-Transformer 的独立表格预测结果不能直接外推到 Qwen3-VL；现代 benchmark 也不支持它在所有表格数据上普适最优。

### 推荐实现

- 2 至 3 层 FT block，内部维度 128 或 192，4 至 8 heads；
- 数值先保留基础线性 token，只有其明显不足时才换 PLE；
- 字段数 `F≤M` 时可直接投影每个字段 token；`F>M` 时先做 attention pooling；
- 保留 missing、字段身份和类别 `UNK`，不要只输入插补后的值。

FT-Transformer 是本任务中“最常用/最成熟的专用编码器代表”，原因是字段 token 与 LLM token 的结构契合，而不是它在所有表格任务上都胜过 GBDT、TabM 或其他现代方法。

## C：创新性最足的组合

### 结构

```text
数值 → PLE/周期 embedding ┐
类别 → category embedding ├→ + field/source/window/missing/time embedding
治理元数据 ───────────────┘
                    ↓
             2 层 shallow FT
                    ↓
      M 个 learned queries 做交叉注意力压缩
                    ↓
       zero-gated projector → [B, M, 2560]
```

创新点不在单个组件，而在三个组合：

1. 用 PLE/周期 embedding 表达数值非线性和局部阈值；
2. 把缺失、更新时间、窗口和来源版本作为一等证据状态；
3. 用 learned queries 把变长字段集合压缩到固定 token 预算。

该方案适合研究“模型是否需要按任务动态选择字段证据”，但不应在首轮 PoC 直接启用。必须分别消融 PLE、治理元数据、FT block 和查询压缩，避免无法判断收益来源。

## Qwen3-VL wrapper 契约

训练阶段可采用以下逻辑；具体函数名需以锁定的 Transformers 版本为准：

```python
# 1. processor 仍生成原始文本/图像/视频 input_ids。
# 2. prompt 中预留 M 个 <|stat_pad|>。
position_ids = compute_qwen3vl_mrope_from_original_ids(
    input_ids=input_ids,
    attention_mask=attention_mask,
    image_grid_thw=image_grid_thw,
    video_grid_thw=video_grid_thw,
    mm_token_type_ids=mm_token_type_ids,
)

inputs_embeds = model.get_input_embeddings()(input_ids)
stat_vector = stat_encoder(numerical, categorical, metadata)  # [B, 256]
stat_tokens = stat_projector(stat_vector)                      # [B, 1, 2560]
inputs_embeds = replace_stat_placeholders(inputs_embeds, input_ids, stat_tokens)

outputs = model(
    input_ids=None,
    inputs_embeds=inputs_embeds,
    position_ids=position_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    labels=labels,
)
```

实现时最小方案只预留一个统计占位；统计占位不参与标签损失，并用原始 token 序列完成 M-RoPE 位置计算。仅验证 `forward()` 不足以证明 `generate()`、vLLM 或 SGLang 可用。

## 训练路线

### 阶段 0：强基线

- CatBoost/LightGBM；
- 统计字段的确定性文本模板；
- 统计专用模型与 Qwen 输出的 late fusion；
- 无统计通道的 Qwen3-VL。

### 阶段 1：对齐连接器

冻结 Qwen3-VL 全部参数，只训练统计编码器、投影器、零门控和任务头。目标是验证统计表示能否被现有主干读取。

### 阶段 2：有限适配

仅在阶段 1 达标后，对语言主干少数 attention/MLP 层加 LoRA；视觉塔默认继续冻结。比较“只训编码器”和“编码器 + LoRA”，确认增益不是主干参数量造成。

### 阶段 3：创新升级

仅当 FT-Transformer 已证明字段交互有稳定收益，但 token 预算或数值非线性仍是瓶颈时，进入方案 C。

## 两周 PoC

| 时间 | 工作 | 验收 |
|---|---|---|
| 第 1—2 天 | 固化时间/实体切分、预处理和四个强基线 | 泄漏审计通过，可复现实验 |
| 第 3—5 天 | 实现 A 与 wrapper，冻结 Qwen 训练 | shape、梯度、M-RoPE、图文共存单测通过 |
| 第 6—7 天 | 连续向量单独分类、late fusion 与单 token 接入消融 | 至少两个随机种子，报告均值和波动 |
| 第 8—10 天 | 实现 B，同预算比较 | 只改变编码器，其他预算一致 |
| 第 11—12 天 | 数值/类别置换、去 missing/age、时间外评估 | 证明模型使用真实统计结构 |
| 第 13—14 天 | 时延、显存、校准和最差桶复盘 | 决定停止、保留 A/B 或进入 C |

## 指标与停止条件

除主任务指标外，同时报告 PR-AUC/AUC、Brier、ECE、最差业务桶、时间外和实体外切片、p95/p99 时延、显存与吞吐。

满足任一条件即停止升级：

- A/B 未稳定超过确定性文本化与同预算 late fusion；
- 相比 CatBoost/LightGBM，收益不足以覆盖 Qwen 推理成本；
- 数值或类别随机置换后表现几乎不变，说明统计通道未被有效利用；
- 去掉 missing、时间或来源元数据后没有可解释变化；
- `generate()` 或目标服务引擎无法稳定支持自定义 embedding；
- 校准、最差桶、漂移或服务 SLO 明显退化。

## 最终推荐

- **现在就做：A，MLP 连续向量编码器。** 先验证 `stat_vector`，需要 Qwen 接入时再增加单 token projector。
- **第二个做：B，FT-Transformer。** 它是最成熟、最常用且与字段 token 接口最匹配的候选。
- **作为论文/创新方向：C，PLE-FT + learned-query。** 它的结构新意最强，但必须在 A/B 之后并做逐组件消融。

若目标只是得到最准确的表格预测，而非多模态推理，优先选择 CatBoost/LightGBM/TabM 并做 late fusion；不要为了“接入大模型”牺牲简单而强的专用基线。
