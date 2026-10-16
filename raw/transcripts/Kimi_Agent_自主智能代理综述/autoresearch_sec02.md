## 2. 实验循环Agent：AutoResearch核心范式

2026年3月7日，Andrej Karpathy在GitHub上发布了一个仅630行Python代码的MIT许可项目。48小时内，该项目获得700万次浏览、26,000个GitHub stars，一周内突破30,000，至3月底超过62,000 [^1^][^516^]。这个名为`autoresearch`的项目并非又一款AutoML工具——它标志着AI研究方法论从"人类设计实验、Agent执行操作"向"人类设定研究品味、Agent自主循环"的根本性范式转移。本章将系统解析这一范式的技术架构、学术基准、衍生生态与平台化缺口。

### 2.1 Karpathy AutoResearch的技术架构

#### 2.1.1 极简三文件设计

AutoResearch的核心架构仅包含三个文件：`prepare.py`（数据准备、tokenizer、常量定义，Agent只读）、`train.py`（模型定义、优化器、训练循环，Agent可修改）和`program.md`（自然语言形式的研究组织代码/指令）[^4^]。这种刻意极简的设计哲学旨在降低Agent的认知负担——Agent只需理解一个文件的逻辑，无需在复杂代码库中导航。

单文件修改约束并非技术限制，而是关键的legibility feature（可解释性设计）：当Agent只能触碰`train.py`时，人类review其做了什么只需要检查一个diff，而不需要reverse engineering整个代码库[^6^]。Karpathy将这一约束视为"架构安全机制"——它确保Agent的每次修改都在人类可理解的范围内，同时git版本控制提供了完整的实验溯源。

#### 2.1.2 program.md：将人类研究品味编码为自然语言规格

`program.md`是AutoResearch区别于传统AutoML/HPO框架的核心创新。传统AutoML要求人类以代码形式定义搜索空间（如Optuna的`trial.suggest_float`调用），而`program.md`允许研究者以自然语言撰写研究策略、约束条件和评估标准[^5^]。正如技术评论者Ry Walker所言，"你在programming program.md，而非编辑Python"——研究设计的抽象层级被提升到方法论文本层面。

一个典型的`program.md`包含以下要素：研究目标的自然语言描述（如"降低验证bits per byte"）、约束规则（如"不要修改prepare.py"、"每次实验固定5分钟"）、品味判断（如"偏好删除代码带来的小改进，而非增加hacky代码"），以及失败处理指令[^12^]。Karpathy本人在`program.md`中明确编码了这种"品味"判断："A tiny improvement that adds 20 lines of hacky code? Probably not worth it. A tiny improvement from deleting code? Definitely keep."[^12^] 这意味着人类研究者的隐性知识——那些难以形式化为数学约束的审美判断——首次被转化为Agent可读的规格文本。

这种机制的实际影响远超代码级优化。它将AutoResearch的定位从"模拟单个PhD学生"转变为"模拟一个研究社区"：人类扮演研究主管角色，通过编辑`program.md`设定方向；Agent扮演执行研究员角色，在规格约束内自主探索。这一劳动分工暗示了"研究设计师"（Research Designer）作为一种新兴职业的可能性——专门编写和优化`program.md`的人才，类似于SEO专家优化网页以适配搜索引擎算法。

#### 2.1.3 Ratchet Loop：无外部编排的贪婪迭代

AutoResearch的核心循环被称为"ratchet loop"（棘轮循环，源自机械工程中只允许单向运动的棘轮机构）：Agent读取`program.md`和当前`train.py` → 提出假设并修改代码 → 运行固定5分钟实验 → 比较验证指标（val_bpb）与当前最佳值 → 如果改进则`git commit`保留，否则`git revert`回滚 → 无限重复[^7^]。

这一设计的极简性体现在：没有orchestration脚本，没有`run.py`，没有外部pipeline框架。用户只需在repo中启动Claude Code或Codex并让其读取`program.md`，Agent自身就是automation layer[^8^]。典型的overnight运行（约8-12小时）可完成80-100次实验，其中约15-20次改进被保留，结果记录在`results.tsv`中，git log完整追溯每次保留的commit[^9^]。

Keep/Revert决策采用纯粹的二进制贪婪爬山（greedy hill-climbing）：如果val_bpb优于当前最佳则保留，否则完全回滚[^11^]。这不是贝叶斯优化，不是进化策略，不是CMA-ES——它是最简单的"更好就留，更差就扔"逻辑。这种设计选择反映了Karpathy的实用主义哲学：在Agent可靠性尚未成熟的阶段，简单的决策机制比复杂的优化算法更robust。

**案例：Karpathy原始运行**

Karpathy本人的首次完整运行是AutoResearch模式有效性的最直接验证。他将Agent部署在单个GPU上运行48小时，期间Agent自主完成约700次实验，在depth-12模型上将验证loss从0.862415降至0.858039，发现20个可叠加的优化[^2^]。这些优化成功transfer到depth-24大模型，最终实现11%训练速度提升（训练时间从2.02小时降至1.80小时）[^2^]。值得注意的是，Agent发现的优化中包含人类研究者遗漏的attention scaling bug修复——这表明Agent在系统性遍历代码空间时，能够发现人类因认知盲区而忽略的问题。整个运行无需人类干预，Agent在overnight期间自主完成了从假设生成到实验执行再到结果筛选的完整研究循环。

### 2.2 学术基准：LLM-based HPO vs经典方法

AutoResearch的流行引发了一个关键学术问题：LLM Agent在超参数优化（Hyperparameter Optimization, HPO）任务上能否超越经典算法？2026年4月，Ferreira等人（ELLIS Institute Tübingen、University of Freiburg、KIT、Prior Labs联合团队）在arXiv发表的"Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch"（arXiv:2603.24647）提供了该问题迄今最系统的实证回答[^13^]。

#### 2.2.1 九种方法在24小时GPU预算下的头对头比较

该研究在相同24小时GPU预算下，对比了9种HPO方法在Karpathy AutoResearch基准上的表现。方法分为三类：经典HPO（CMA-ES、TPE、SMAC、Random Search）、纯LLM-based方法（LLAMBO Paper、LLAMBO Code、Karpathy Agent Code、Karpathy Agent Prompt）以及混合方法（Centaur）[^13^]。实验设计严格控制了变量：所有方法在同一硬件上运行，使用相同的基础模型（GPT-2 style架构），以val_bpb（validation bits per byte）为统一评估指标。

**表1：九种HPO方法在24小时GPU预算下的性能对比**

| 方法 | 类型 | 最佳val_bpb | OOM率 | 核心机制 |
|------|------|-------------|-------|---------|
| Centaur [Opus 4.6] | 混合 | 0.9739 [^17^] | ≤16% [^15^] | CMA-ES内部状态共享+LLM |
| Centaur [0.8B LLM] | 混合 | 0.9763 [^14^] | ≤16% [^15^] | CMA-ES状态共享，LLM仅占30% trials [^16^] |
| TPE | 经典 | 0.9768 [^14^] | 11-16% [^15^] | Tree-structured Parzen Estimator |
| SMAC | 经典 | 0.9778 [^14^] | 11-16% [^15^] | Bayesian Optimization + RF surrogate |
| CMA-ES | 经典 | 0.9785 [^14^] | 11-16% [^15^] | 协方差矩阵自适应进化策略 |
| Karpathy Agent [Code] | LLM | 0.9814 [^14^] | 36%+ [^15^] | Qwen3.5-27B无约束代码编辑 |
| Karpathy Agent [Prompt] | LLM | >0.9814 | 48-61% [^15^] | 固定搜索空间prompt工程 |
| LLAMBO [Code] | LLM | ~0.982 | 48-61% [^15^] | LLM替代BO surrogate+代码编辑 |
| LLAMBO [Paper] | LLM | 最差 [^15^] | 48% [^15^] | LLM替代BO surrogate，搜索最多样 |
| Random Search | 基准 | ~0.985 | 56% [^49^] | 均匀随机采样 |

*注：val_bpb越低表示性能越好。数据来自Ferreira et al., arXiv:2603.24647 v4 [^14^][^15^]。*

上表揭示了两个关键规律。第一，可靠性（以OOM率衡量）是比搜索多样性更重要的性能预测因子。顶部方法的OOM率均控制在16%以下，而底部纯LLM方法的OOM率高达48-61%——即使这些方法观察到完整的trial历史，其OOM率仍与Random Search相当（56%）[^49^]。这表明LLM Agent在优化状态跟踪方面存在根本性局限：它们未能从失败实验中学习哪些搜索区域会导致内存溢出[^49^]。第二，即使允许LLM进行无约束代码编辑（Karpathy Agent Code），其表现也仅接近经典方法而无法超越，最佳结果0.9814仍劣于TPE的0.9768[^14^]。

#### 2.2.2 核心结论：经典HPO在固定搜索空间的优势不可撼动

Ferreira等人的核心结论具有高度的学术严谨性："经典HPO方法（CMA-ES、TPE、SMAC）在固定搜索空间内始终优于纯LLM-based agent，即使使用Claude Opus 4.6和Gemini 3.1 Pro Preview等前沿模型也无法改变这一结论"[^14^]。这一发现挑战了"LLM将取代所有传统优化方法"的乐观预测，揭示了LLM优化能力对搜索空间表示方式的高度依赖。

更具启示性的发现来自模型scaling实验：从0.8B到27B的模型规模扩展对固定搜索空间内的LLM方法几乎无收益，但对无约束代码编辑则至关重要[^18^]。Gemini 3.1 Flash-Lite在无约束代码编辑中的失败率高达87-94%（主要是runtime error而非OOM），但在固定搜索空间内与Qwen3.5-0.8B表现相当[^19^]。这证实了一个工程直觉：代码编辑能力对模型能力的要求远高于固定参数采样。论文的终极结论是："LLMs are most effective as a complement to classical optimizers, not as a replacement."[^54^]

#### 2.2.3 Centaur混合方法：共享内部状态的范式突破

Centaur是Ferreira等人提出的混合方法，其核心创新在于将CMA-ES的完整内部状态（均值向量μ、步长σ、协方差矩阵C）共享给LLM，使LLM能够理解经典优化器的当前信念状态并据此生成有针对性的提案[^16^]。这种方法在所有实验中取得最佳结果，且一个0.8B参数的LLM已足以超越所有纯LLM和经典方法[^16^]。

Centaur的设计选择揭示了混合智能的最优配比：LLM仅用于约30%的trials（其余由CMA-ES独立propose），0.8B模型在混合模式下足够，但纯代码编辑需要27B+规模[^17^]。当配合Opus 4.6时，Centaur进一步将最佳结果推进至0.9739，扩展了混合方法的领先优势[^17^]。这表明混合方法比纯LLM方法更能从模型扩展中获益——因为LLM在Centaur中的角色是"辅助经典优化器"而非"替代优化器"，其职责范围更窄、更聚焦，从而降低了大规模模型的能力浪费。

Centaur的深层意义在于它标志着AI优化从"替代人类/替代经典算法"的两极分化转向"深度状态共享"的新范式。未来autoresearch系统的核心竞争力，可能不再是模型规模本身，而是"如何设计LLM与经典组件的状态共享接口"[^16^]。

### 2.3 衍生生态与平台化缺口

AutoResearch的开源MIT许可和极简架构在发布后7天内催生了至少6个显著衍生变体，形成了一个快速分化的技术生态[^24^]。

#### 2.3.1 衍生变体：从ML训练到任意可测量领域

**表2：AutoResearch主要衍生变体技术对比**

| 项目 | Stars | 领域 | 核心差异化 | 关键数据 |
|------|-------|------|-----------|---------|
| karpathy/autoresearch | 62,000+ [^25^] | LLM训练优化 | 原版；program.md作为自然语言编排层 | 48h/700实验/11%提升 [^2^] |
| pi-autoresearch (Shopify) | 3,591 [^24^] | 通用代码优化 | Pi编辑器扩展；任意可测量指标 | 单元测试提速300x [^430^] |
| autokernel | 1,206 [^24^] | GPU kernel优化 | Triton/CUDA双后端；correctness harness | 比torch.compile快2.2-5.3x [^125^] |
| autoresearch-at-home | ~462 [^24^] | 分布式协作 | SETI@home风格swarm；Ensue协调 | 20+ agents/54h/1000+实验/3.2%提升 [^29^] |
| autoresearch-mlx | ~633 [^24^] | Apple Silicon移植 | MLX框架；无PyTorch依赖 | M4 Max: 2.667→1.294 val_bpb [^27^] |
| N-Autoresearch | N/A [^30^] | 多GPU并行 | Rust GPU Worker；atomic acquisition | N GPU = N并行实验；全局自适应策略 [^30^] |
| darwin-derby | 48 [^25^] | 任意领域泛化 | 论文质量、网站性能、TSP等 | "If you can measure it, you can optimize it" |

*数据来源：awesome-autoresearch curated list [^24^]、Ry Walker Research市场地图 [^25^]，截至2026年4月。*

上表展示了生态分化的三条主线。第一条是**领域泛化**：pi-autoresearch由Shopify工程师开发，将AutoResearch模式从ML训练扩展至任意软件工程指标——测试速度、bundle size、build times、Lighthouse scores均成为可优化目标[^26^]。该变体引入了persistent sessions（重启后恢复）、dashboard UI和branch-aware tracking等企业级特性[^26^]。第二条是**硬件移植**：autoresearch-mlx将框架移植到Apple MLX框架，验证了AutoResearch模式在非NVIDIA硬件上的可行性[^27^]。第三条是**分布式扩展**：autoresearch-at-home实现SETI@home风格的分布式Agent协作，20余个Agent在54小时内通过共享内存网络协调完成1000余次实验[^29^]。

**案例：Shopify的pi-autoresearch生产应用**

Shopify是AutoResearch模式最早的企业级采纳者之一，其工程师基于Karpathy原型开发了pi-autoresearch变体，将实验循环嵌入Shopify内部广泛使用的Pi编辑器。据Shopify Engineering Blog报道，该工具在两周内获得3,600+ GitHub stars和200+ forks，公司内部已建立#autoresearch-wins频道分享成果[^430^]。

Shopify CEO Tobi Lütke的个人运行更具示范意义：他将AutoResearch指向Shopify的Liquid模板引擎代码库，在overnight期间完成37次实验，获得19%整体性能提升[^3^]。更具体的成果包括模板渲染速度提升53%（来自93次自动commit）和内存分配减少61%[^3^]。Shopify工程师团队还将pi-autoresearch应用于React组件mounting优化（提速20%）和单元测试加速（提速300倍）[^430^]。这些案例表明AutoResearch模式的适用性已超越ML训练领域，进入通用软件工程优化场景。然而，这些数据主要基于博客报道和社交媒体，尚未经独立第三方验证[^3^]。

#### 2.3.2 与编码Agent融合：从项目到工作流原语

AutoResearch模式正在快速融入主流编码Agent生态，从独立项目演变为标准化的Agent工作流原语。

**表3：编码Agent AutoResearch集成状态对比**

| 编码Agent | 集成形式 | 发布时间 | 核心特性 | 成本/效率数据 |
|-----------|---------|--------|---------|--------------|
| Claude Code | Skill系统（Udit Goenka）[^33^] | 2026-03 | 通用化到任何可测量领域；608⭐ | $0.05-0.15/cycle；overnight 30 cycles约$1.5-4.5 [^34^] |
| Claude Code | MindStudio自改进skill [^35^] | 2026-03 | prompt优化专用；40-50%→75-85% pass rate | 30-50 cycles/8h |
| Codex | Autoresearch Loop Skill [^36^] | 2026-04 | modify→verify→keep/revert；单句目标输入 | 自动推断scope/metric/verification |
| Cursor | Extension + MCP [^37^] | 2026-04 | SSE实时dashboard；status/result panel | optimization loop在shared packages中 |
| 跨Agent | autoresearch-cli [^38^] | 2026-03 | 一键为Claude/Codex/Cursor/Windsurf安装 | 14项预检；autoresearch.toml配置 |

*数据来源：各集成官方文档与GitHub仓库，截至2026年4月。*

上表显示了三类集成策略。**Skill/Extension层**（Claude Code、Codex、Cursor）将AutoResearch循环封装为Agent原生功能，降低了用户的使用门槛。Claude Code的集成最为深入：Udit Goenka发布的generalized skill将AutoResearch扩展到安全审计、API性能优化等领域[^33^]；MindStudio的指南展示了如何用Claude Code + AutoResearch loop自动改进AI skill，30-50个循环可将40-50%的pass rate提升至75-85%[^35^]。**CLI工具层**（autoresearch-cli）提供了跨Agent的通用基础设施，支持`autoresearch install all`一次性为多平台安装循环指令，并包含14项预检和标准化配置[^38^]。**增强循环层**（Evo、Ouro Loop）在贪婪爬山基础上增加了更复杂的搜索结构：Evo增加了树搜索（多方向fork）、并行半自主subagents和benchmark discovery[^39^]；Ouro Loop引入了BOUND系统定义绝对约束（DANGER ZONES、NEVER DO rules），通过runtime层面的硬拦截解决通用软件工程中的安全性问题[^40^]。

基于当前集成速度（Claude Code skill发布后3周获得608 stars，Cursor Extension在VS Code Marketplace上架），行业观察者预测到2026年Q3，AutoResearch loop将成为主流编码Agent的原生功能——类似于今天的代码补全或重构建议。

#### 2.3.3 最大缺口：通用水平平台尚未出现

尽管垂直化工具在各自领域取得进展，当前生态存在一个结构性缺口：通用的"autoresearch-as-a-service"水平平台尚未出现。如Ry Walker Research的市场评估所言："垂直工具已存在，水平平台尚未出现——这才是最大的机会"[^12^]。

理想的水平平台应提供五项核心能力：统一的Agent编排层（支持任意repo + 任意可测量metric）、多Agent协调（实验认领、结果共享、全局最优同步）、基于Amdahl定律的优化调度、成本管控（API限额、自动中断），以及结果可复现性（Git-native版本控制、Docker sandbox）[^552^]。当前最接近这一愿景的是autoresearch-at-home（分布式协调）和N-Autoresearch（多GPU并行），但两者均处于早期阶段——前者仅有462 stars且依赖第三方共享内存服务[^29^]，后者缺乏生产级成熟度[^30^]。

这一缺口的技术根源在于AutoResearch的原始设计约束：它明确是research tool而非production platform，无认证、无错误恢复（`program.md`之外）、无dashboard、无可恢复性、无多GPU支持——所有生产级功能均为第三方fork添加[^53^]。对于大规模生产ML workflow，AutoResearch是"一种需要适配的模式，而非可直接部署的系统"[^53^]。这意味着2026年的autoresearch生态正处于"原型验证完成、平台层缺失"的关键转折点：垂直应用证明了模式价值，但水平基础设施的投资尚未到位。
