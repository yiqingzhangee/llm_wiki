# 来源：Reverse Curriculum RL（R3）

- **原始路径**：
  - [`icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.md`](../../raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.md)
- **类型**：ICML 2024 方法论文（强化学习课程设计）

## 一句话摘要

R3 通过“从接近答案的中间状态开始，再逐步回退到完整推理”的逆课程训练，把稀疏 outcome reward 近似转化为过程级监督信号。

## 关键主张（基于摘要与正文）

- 传统 outcome supervision 奖励稀疏、定位误差困难；process supervision 精细但昂贵。
- R3 利用正确示例中的中间状态构造逆课程，降低探索难度并提高正反馈密度。
- 通过混合不同难度起点训练，缓解 staged RL 的阶段切换退化问题。
- 在多个推理任务（含 CoT 与程序化推理）上，相比 SFT/RL 基线有稳定增益。

## 不确定性与边界

- 方法前提是可获得可用示例轨迹；示例质量影响课程有效性。
- 课程阶段与奖励超参数（如 KL、partial reward）对稳定性较敏感。

## 织入概念

- [[LLM推理结构与自纠错框架]]
- [[Agent任务分流：工作流与智能体边界]]