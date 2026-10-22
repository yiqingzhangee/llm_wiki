# EviState 训练代码说明

训练代码放在 `evi_state_code/src/evi_state/training/`。这套代码先解决接口和
时序问题：样本如何按 cutoff 过滤、滚动窗口如何转成增量、各模态怎样进入
六个状态槽，以及 P/C2 能否用同一批数据完成前向和反向。

它现在跑的是确定性 fixture，不会读取 Hive、线上 snapshot 或外部模型权重。

## 目录

```text
evi_state_code/
├── conf/
│   ├── data/
│   │   ├── engineering_24d.yaml
│   │   └── research_30d.yaml
│   └── trainer/
│       └── smoke_cpu.yaml
├── src/evi_state/training/
│   ├── contracts.py
│   ├── dataset.py
│   ├── batching.py
│   ├── encoders.py
│   ├── model.py
│   ├── losses.py
│   ├── trainer.py
│   ├── fixtures.py
│   └── smoke.py
└── tests/
    ├── test_training_dataset.py
    ├── test_training_components.py
    ├── test_training_model.py
    └── test_training_smoke.py
```

`contracts.py` 定义 evidence、日包、标签和序列样本。`dataset.py` 负责
cutoff、版本选择和 prefix delta。`batching.py` 在补齐张量时保留 evidence
ID、locator、available time 和 missing reason。

`encoders.py` 里有 Behavior、Statistics 编码器和视频帧集合池化。
`model.py` 包含六槽 P/C2、投影层、冻结的未来目标分支和 Detector。
损失与训练循环分别在 `losses.py` 和 `trainer.py`。

六个槽的顺序固定为：

```text
BEHAVIOR, STATISTICS, CONTENT, IMAGE, RELATION, IDENTITY
```

Profile 只进 `IDENTITY`。Policy 不参与状态更新，只传给 Detector。

## 运行

```bash
cd evi_state_code
uv sync --group dev
```

跑 24 日 P：

```bash
uv run evi-state-smoke --mode p --sequence-length 24 --seed 42
```

跑 30 日 C2：

```bash
uv run evi-state-smoke --mode c2 --sequence-length 30 --seed 42
```

检查代码：

```bash
uv run pytest -q
uv run ruff check .
```

最近一次本地结果是 33 项测试通过，P 和 C2 都完成了一次前向、反向和参数
更新。

## 已经处理的约束

- 时间统一使用 UTC Unix seconds。
- evidence 只有在 `available_time <= cutoff_time` 时可见。
- 滚动窗口按稳定 evidence ID 转成每日 delta。
- correction 和 retraction 保留版本记录，不覆盖历史。
- 缺失模态使用 mask 和 missing reason，不拿零向量冒充有效输入。
- 视频帧按无序集合池化。
- burn-in 日不计算风险损失。
- Trainer 拒绝 online snapshot。

## 还缺什么

这套代码还不能直接跑真实训练。下一步要接入真实冻结样本和 Qwen cache，并
补齐以下数据：

- Statistics、Profile、Video 的时间和 locator sidecar；
- 跨日稳定的 Behavior、Content、Image evidence ID；
- 正式 E1 产出的 TargetEncoder/TargetProjector；
- label maturity、appeal/censor 和 label mapping；
- policy registry 与 freeze manifest；
- checkpoint、评估和完整 DecisionTrace。

这些门禁没过之前，Detector 的结果只能算接口 smoke，不能当作真实风险训练
结果，更不能用来比较 P 和 C2。
