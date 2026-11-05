"""仅生成本地语料和审计文件；不调用训练框架或远端写入接口。"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATE = "2026-11-02"
SOURCES = [
    ("D01", "色赌诈作弊手法", "https://bytedance.larkoffice.com/wiki/MTIewGNJzipWX8kc0mjcCfhAnNd", 3125, "UX作弊手法/aRNSCM；A1:AF12、A13:T26", "risk_standard", "业务机制；逐行状态筛选，签核符号不等于已核实平台生效"),
    ("D02", "反作弊特征建设", "https://bytedance.larkoffice.com/docx/U8D4dwlwkolG8LxnYHpczw0Tnye", 7, "几种有具体物理含义的统计特征/doxcn3JZj37NlFHArSnUBFQucDc", "statistics", "统计语义；无统一窗口和正常反例"),
    ("D03", "国际电商风控特征数据建设", "https://bytedance.sg.larkoffice.com/docx/VVagdmZS1opUhpxLzJzcJ4zEnvb", 3525, "特征介绍/Aimmd4OCGoqMgWxuCh7cgX15nDh", "ecommerce_features", "电商设计框架；嵌入字段表未读取"),
    ("D04", "Account Farming 行为风险标签与解释性建设", "https://bytedance.larkoffice.com/wiki/PJUjwIyeSiMD1Ik99ajcvNScn1G", 3960, "总体方案/TR57dXeqtoia5TxF3e8lxgirgKe；账号标签/Pc5kde5BdoLj6hxG55ilnMCbgsc；团伙标签/CADOdtjxEoFZVxx0z38lg3eDgod", "account_farming", "方案和字段语义；不收模型解释模板、真实账号、阈值、处置建议"),
    ("D05", "[BRIC] 社交杀猪盘专项", "https://bytedance.sg.larkoffice.com/docx/S4zhdfrBTo6BU2xXp5JlwNCTgyc", 4797, "MVP版本/L2zGdoNawofL8bx2jYhlNyWcgLe；表中评论意图、私信意图", "social_intents", "仅意图分类；MVP完整判定因版本边界冲突排除"),
    ("D06", "BRIC标准标签（明细）", "https://bytedance.larkoffice.com/wiki/ARHTwrt4oivlN2klu2qcflsIn6e", 7048, "Social/KEVQVq；A1:T10；F7:F10富文本引用", "risk_standard", "导航仅溯源；沿引用补读D06a、D06b"),
    ("D07", "【2026】BRIC-标准四级标签和案例评审", "https://bytedance.larkoffice.com/wiki/Ur6Tw3JFBiBtpzkZNgncQKItnYd", 9866, "流程说明/Tw0XdPNjWol7oExzm3HlbQo6gxf", "standard_governance", "治理方法辅助；示例条款不当已发布标准"),
    ("D08", "BRIC 治理规则（标准）发布流程", "https://bytedance.larkoffice.com/wiki/HvTPwlCyfiLHWHkg4qRcT8TDnie", 3212, "治理规则（标准）体系说明/ITNkd4zX7o5XZfx8083mdl1eygb", "standard_governance", "标准组成辅助；图像与平台现行状态未核验"),
    ("D09", "国际化风控数据知识库（BRIC知识库）｜能力说明与使用指南", "https://bytedance.larkoffice.com/docx/Kv6hdClBKoeCVpx9GIzcMQiFn1g", 629, "知识从哪里来/doxcnu6ADijxXvsqnwpu0GILzDf", "knowledge_provenance", "来源治理辅助；不保存资产规模、库表路径"),
    ("D10", "风控基础知识", "https://bytedance.larkoffice.com/docx/ZUn8dyLBpowAWfx6wvacy6V5nWc", 8, "定义与类型/doxcn10G6VJXoWMi6JVzVFLPPcf；垃圾注册/doxcnAT7alGfF0aqUycehZkSjHd；账号被盗/doxcnVs4X8lRhokYr5nn2defxvh；业务鉴权/doxcnrtizZNch9fRoxHq3OwP6Ta", "risk_foundations", "定义节为导航；其他章节提取基础机制，不视作当前政策"),
    ("D11", "风控及ToB场景大模型替代人审实践", "https://bytedance.larkoffice.com/docx/UzScdEmaWotVYex9oYgcHHuinnf", 1436, "适配prompt的人审标准改造/I2BBdFUYioKUOqxS0VNcOTrHnsn", "annotation_methods", "标准拆解方法辅助；不收案例、模型输出、实验指标"),
    ("D06a", "投资与信贷引流行为-标准标签（D06引用）", "https://bytedance.larkoffice.com/wiki/Xx7IwY3b3iiMAhk0p3ycCQAGncP", 6912, "变更日志/QFmfdvW2coOUlmxu07VlpDqtg8c；标准细则/QfjvdZzEzo4HmcxSTt9lgI0Fggb", "risk_standard", "仅虚拟货币条款；旧荐股、旧网贷已下线；处置/申诉存在复制疑点，均排除"),
    ("D06b", "招工与刷单引流行为-标准标签（D06引用）", "https://bytedance.larkoffice.com/wiki/L4GZwkrR4iplsSkxZnucY61snZc", 5237, "标准细则/RnDDdU3T2ocs6VxhQKslEcaPgkb；兼职高收益引流行为", "risk_standard", "识别机制和正常例外；旧返利条款已下线；处罚版本不混入"),
]
SOURCE_MAP = {s[0]: s for s in SOURCES}
VALID_GROUPS = {"account_farming", "social_intents"}
UNITS = []


def add(uid, source_ids, topic, definition, mechanism, evidence, counterexamples,
        scope, anchor, excerpt, *, auxiliary=False, counterexample_kind=None):
    group_ids = {SOURCE_MAP[s][5] for s in source_ids}
    assert len(group_ids) == 1
    group = next(iter(group_ids))
    split = "auxiliary" if auxiliary else ("valid" if group in VALID_GROUPS else "train")
    unit = {
        "unit_id": uid,
        "topic": topic,
        "definition": definition,
        "mechanism": mechanism,
        "evidence": evidence,
        "counterexamples": counterexamples,
        "scope": scope,
        "source_ids": source_ids,
        "group_id": group,
        "split": split,
        "purpose": "governance_method" if auxiliary else "domain_knowledge",
        "counterexample_status": "source_provided" if counterexamples else "not_provided",
        "counterexample_kind": counterexample_kind if counterexamples else None,
        "counterexample_item_types": ["source_paraphrase"] * len(counterexamples or []),
        "provenance": {"anchor": anchor, "supporting_excerpt": excerpt},
        "review_status": "source_checked_business_review_pending",
        "limitations": ["不是现行政策生效证明", "不是已标注案例或业务效果证据"],
        "field_grounding": {
            "definition": "source_paraphrase",
            "mechanism": "source_paraphrase",
            "evidence": "source_observables_or_document_structure",
            "counterexamples": "source_paraphrase" if counterexamples else "missing",
            "scope": "source_context_plus_extraction_boundary",
        },
    }
    UNITS.append(unit)


def technique(uid, row, topic, definition, mechanism, evidence, excerpt,
              counterexamples=None, extra_sources=(), scope_extra=""):
    add(uid, ["D01", *extra_sources], topic, definition, mechanism, evidence,
        counterexamples, "手法表已读取的UX场景机制快照；不含阈值、实时策略、法律结论或自动处罚。" + scope_extra,
        f"D01:UX作弊手法/aRNSCM/第{row}行/D:F、H:O、T" +
        ("；补充来源见source_ids所列标准细则及主题" if extra_sources else ""),
        excerpt, counterexample_kind="normal_scenario" if counterexamples else None)


technique("K001", 2, "高回报投资承诺的引流机制",
    "以极高回报或可追回投资损失的承诺吸引用户参与的引流手法。",
    "收益材料强化获利预期，随后把咨询、注册或参与迁移到站外，进一步要求付费或投入资金。",
    ["收益或损失追回承诺", "盈利截图、收益战绩或提现记录", "咨询或注册引导与站外入口", "订阅费或资金投入要求"],
    "通过盈利截图、收益战绩、提现记录渲染高回报。引流至站外社群、或私人电话诱导用户缴纳订阅费或投入资金")
technique("K002", 3, "虚拟货币投资收益诱导与导流",
    "将虚拟货币、代币、NFT或区块链项目的盈利预期作为卖点，并将用户引至指定平台或项目。",
    "收益承诺、战绩材料或具体获利路径增强盈利预期；私信、外链、邀请码等承接咨询、注册和投资。所读标准要求收益诱导与流量导流同时成立。",
    ["明示或暗示盈利预期", "具体币种、策略或获利路径", "链接、邀请码、客服入口或下载入口", "咨询、注册、充值或跟投引导"],
    "判定违规需同时满足收益诱导与流量导流两大核心要素；仅单纯讨论行业概念、市场资讯的内容不纳入违规。",
    ["客观概念、政策、新闻或技术介绍，未出现收益诱导和引流。",
     "市场风险、监管动态、历史行情或公开案例的事实讨论，未推荐获利机会或导向交易。",
     "不以获利诱饵或导流开展的风险提示、反诈提醒和消费者教育。",
     "所读标准将导向已上架官方应用商店的正规虚拟货币交易平台官方APP列为该标签豁免；此项不证明所有其他风险均不存在。"],
    ("D06", "D06a"), "标准例外来自投资信贷标准的虚拟货币条款，适用范围为该标签快照，平台当前生效状态尚未核实。")
technique("K003", 4, "财富展示与付费获利诱饵",
    "将大量现金、豪车豪宅等财富展示与成功承诺、付费咨询结合的引流手法。",
    "财富画面吸引关注，成功秘诀或获利承诺促使用户私下联系，站外订阅或套餐承接资金投入。",
    ["财富展示", "成功秘诀或获利诱导话术", "私下联系或群组入口", "订阅或套餐付费要求"],
    "诱导用户咨询，缴纳订阅费或购买套餐进行诈骗")
technique("K004", 5, "赠与福利名义的信息与费用索取",
    "以无理由赠与、免费获利、福利返现等无需正常对价即可得利的承诺吸引用户。",
    "获益承诺经站外导流转入资格验证或激活流程，进一步索取身份、银行卡信息或前置费用。",
    ["无理由获益或赠与承诺", "站外领取入口", "资格验证、福利申请或激活说明", "敏感信息或手续费、押金要求"],
    "以领取资格验证、福利申请、优惠码激活、身份核验等名义要求用户提交个人/银行卡信息或支付手续费、押金、会员费等")
technique("K005", 7, "简单任务赚取现金的付费链路",
    "用极简单任务和现金奖励承诺诱导参与，再向参与者索取资金的手法。",
    "低门槛任务承诺吸引用户，联系方式把用户迁移到站外，以任务佣金或押金等名义要求缴费或投入资金。",
    ["简单任务与现金奖励承诺", "站外群组或联系入口", "任务佣金、押金或充值要求", "收费与任务兑现关系"],
    "以‘任务佣金、押金’等名义要求用户缴费，或诱导用户进一步投入资金")
technique("K006", 8, "兼职招聘的收益诱饵与前置收费",
    "以高薪、轻松、零门槛兼职或就业承诺吸引用户，并引导私域联系或参与付费流程的风险机制。",
    "求职与逐利预期被转化为报名、培训、材料或押金支出，也可能转入敏感信息收集或风险任务。所读标准将诱导联系动作与具体风险情形结合。",
    ["岗位或项目是否真实可核验", "收益与就业承诺", "私域联系或导流动作", "入职前收费、敏感信息索取、风险任务或培训贷款安排"],
    "满足以下任一情形，且伴随引导私域联系/引流等动作，可判定：",
    ["正规主体发布可核验岗位，不存在入职前收费或虚假高薪承诺，并同时满足信息真实清晰、无欺诈收费或诱导贷款、无敏感信息滥用、无违法引导的合规招聘。",
     "不能仅以居家工作或无需经验的单个词语代替所读标准要求的组合条件。"],
    ("D06", "D06b"), "例外依据招工刷单标准的兼职条款；第二项是组合条件的逻辑边界，不是新增案例。")
# K006的第二项为条款逻辑边界，单独标记，避免伪称来源提供了该正常案例。
UNITS[-1]["counterexample_kind"] = "normal_scenario_and_rule_boundary"
UNITS[-1]["counterexample_item_types"] = ["source_paraphrase", "rule_logical_boundary"]
UNITS[-1]["field_grounding"]["counterexamples"] = "source_paraphrase_and_rule_logical_boundary"
technique("K007", 10, "游戏资产交易内容中的聚集式引流",
    "多个账号围绕游戏道具交易内容和聚集联系方式引导玩家站外联系的手法描述。",
    "批量内容触达玩家，相似画风、模板或统一导流点汇集咨询，站外押金、订阅或交易承接资金。",
    ["游戏资产交易内容", "模板或联系方式的跨账号聚集", "发布频率", "站外付费要求"],
    "黑产发布批量游戏道具交易视频触达用户 -> 诱导用户联系视频内的导流点 -> 以缴纳押金、缴纳订阅费等方式诱骗用户投入资金进行诈骗")
technique("K008", 11, "毒品售卖信息与交易导流",
    "发布毒品供货或售卖信息，并将用户引至站外咨询或交易的内容链路。",
    "名称、外观、价格、库存或交易暗示传达供货信息，评论、私信和个人资料入口承接询价与下单。",
    ["供货或售卖内容", "交易暗示、库存或价格信息", "评论、私信或个人资料的联系入口", "询价与交易引导"],
    "展示毒品名称、外观、价格、库存或交易暗示，并借助评论、私信、Profile 联系方式等引导用户前往站外渠道咨询、下单或完成交易。")
technique("K009", 12, "虚假证件定制与售卖导流",
    "宣称制作、办理或售卖虚假证件，并通过联系入口承接定制和交易的手法。",
    "证件样例与定制能力展示吸引咨询，站外渠道接收资料并收取制作或交付费用。",
    ["虚假证件制作或售卖宣称", "样例、类型或定制能力", "咨询入口", "资料收集与付费安排"],
    "展示证件样例、可办理类型或定制能力 → 引导用户通过私信、Profile 或站外联系方式咨询 → 收集资料并收取定金、制作费或尾款完成非法交易")
technique("K010", 13, "作弊工具或服务的售卖导流",
    "以监听、解锁、破解、绕过验证等作弊功能为卖点，引导购买或使用工具与服务。",
    "功能演示和效果材料吸引用户，评论、私信或个人资料把需求导向站外付费、安装或使用环节。",
    ["作弊功能或服务宣称", "演示与效果材料", "联系或购买入口", "付费、安装或使用引导"],
    "宣传可提供监听、解锁、破解、绕过验证、外挂或其他作弊服务与工具")
technique("K011", 20, "文字性暗示与聚集式内容导流",
    "以性暗示文字、符号或缩写吸引互动，并以模板或聚集入口导向站外内容与服务。",
    "暗示性文案激发咨询，相似话术或统一联系方式承接转移，用户被引至外部内容或付费渠道。",
    ["性暗示文本或符号", "相似模板与话术", "导流入口", "站外内容或服务引导"],
    "使用露骨或隐晦的性暗示文字、符号、缩写及引导性文案吸引用户互动")
technique("K012", 22, "有偿性服务的招揽与联系链路",
    "以服务介绍、价格暗示、地区信息或联系方式招揽有偿性服务并承接咨询交易。",
    "服务与地区等信息吸引需求，评论、私信、个人资料或聚集式入口把咨询迁移到站外预约和付费环节。",
    ["有偿性服务介绍或招揽", "价格或地区信息", "联系方式与导流动作", "预约或付费安排"],
    "通过性暗示画面、服务介绍、价格暗示、地区信息或联系方式招揽有偿性服务")
technique("K013", 23, "色情内容的付费解锁链路",
    "展示色情内容或预告，利用试看和完整内容承诺引导打赏或付费。",
    "内容预告吸引互动，直播打赏、私信或站外渠道承接付费后获取更多内容的请求。",
    ["色情内容或预告", "试看、完整版或解锁承诺", "打赏或付费要求", "私信或站外获取路径"],
    "以试看、预告或福利为噱头吸引用户，并诱导用户通过直播打赏、私信付费或站外渠道购买更多色情内容。")
technique("K014", 24, "性暗示画面声音与站外导流",
    "用性暗示动作、姿态、声音或局部画面吸引用户，并将其引向站外内容或服务。",
    "画面或声音激发互动，相似画风、模板和聚集式导流点承接更多内容或服务的需求。",
    ["性暗示画面或声音", "相似画风与模板", "联系方式或聚集入口", "更多内容与服务引导"],
    "使用挑逗动作、暧昧姿态、性暗示声音、局部裸露或暗示性画面吸引用户")
technique("K015", 25, "赌博服务展示与参与导流",
    "展示或推广赌博服务，并以入口信息引导注册、充值和下注的内容链路。",
    "玩法、记录或平台材料传达参与机会，链接、邀请码、客服或站外联系方式承接注册与下注。",
    ["赌博服务或招揽内容", "下注或开奖记录等展示", "链接、邀请码或客服入口", "注册、充值或下注引导"],
    "宣传赌博服务、代理招募或参与入口，并通过链接、邀请码、客服或站外联系方式引导用户注册、充值和下注。")
technique("K016", 26, "赌博收益承诺与持续下注诱导",
    "以高胜率、高收益或福利承诺增强赌博获利预期并诱导参与的手法。",
    "盈利或预测战绩、内幕说法和福利材料提高可信感，再导入平台、群组或代理渠道，承接充值、跟单与下注。",
    ["收益、胜率或福利承诺", "盈利或预测材料", "平台、群组或代理入口", "充值与持续参与引导"],
    "通过稳赚、包赢、高胜率、内幕预测、跟单或充值奖励等话术，营造低风险高回报预期")

add("K017", ["D06", "D06b"], "培训服务与就业承诺的边界",
    "培训包装就业转化，是将培训收费或贷款与包上岗、保证就业等承诺绑定，但缺乏可验证对应关系的风险情形。",
    "就业承诺把求职需求转换为培训费、分期或贷款负担；所读条款同时检查服务真实性、就业对应关系和还款责任说明。",
    ["培训与就业承诺", "费用和服务范围", "分期或贷款安排", "真实可验证的就业对应关系"],
    ["机构资质、费用与服务范围清楚，不承诺包就业或包高薪，不诱导高风险贷款；还须同时满足信息真实清晰、无欺诈收费、无敏感信息滥用、无违法引导。"],
    "所读招工标准兼职条款的培训场景快照；不是所有培训收费的风险判定。",
    "D06b:标准细则/兼职高收益引流行为/培训包装就业转化、豁免场景",
    "合规培训/教育服务：机构资质与费用/服务范围说明清楚，不承诺\"包就业/包高薪\"，不诱导高风险贷款。",
    counterexample_kind="normal_scenario")
add("K018", ["D06", "D06b"], "求职工具服务的范围边界",
    "简历、面试和职业规划等求职工具服务与招聘入职服务具有不同边界。",
    "所读标准把服务边界、收费名义、敏感信息和结果承诺结合，区分工具服务与借招聘入职名义的风险行为。",
    ["服务范围", "收费是否以招聘入职为名", "敏感信息收集方式", "是否承诺具体就业或收入结果"],
    ["边界清晰，不以招聘入职为名收费或索取敏感信息，不承诺具体就业或收入结果的求职工具服务；同时满足信息真实清晰、无欺诈收费或诱导贷款、无敏感信息滥用、无违法引导。"],
    "所读招工标准兼职条款的例外快照；不构成跨标签的通用豁免。",
    "D06b:标准细则/兼职高收益引流行为/豁免场景",
    "求职工具类服务（简历/面试/职业规划等）：边界清晰，不以招聘入职为名收费或索取敏感信息，不承诺具体就业或收入结果。",
    counterexample_kind="normal_scenario")

for uid, topic, definition, mechanism, evidence, excerpt in [
    ("K019", "用户行为事件直方图", "以用户和事件类别为统计维度的计数分布。",
     "按用户及事件类别聚合计数，得到不同事件的频数，形成行为事件直方图。",
     ["用户维度", "事件类别", "各类别计数"], "C({用户ID，事件}) = 用户行为事件直方图"),
    ("K020", "用户行为时间分布", "以用户和时间桶为维度的行为计数分布。",
     "将行为归入时间桶并按用户计数，形成时间分布或活跃热力图。",
     ["用户维度", "时间桶", "各桶行为计数"], "C({用户ID，时间桶}) = 用户行为时间分布 / 活跃热力图"),
    ("K021", "用户时间集中度的条件比例", "给定用户后，其行为在不同时间桶中的条件分布。",
     "以用户为条件查看各时间桶比例，将绝对行为量转化为时间上的相对集中程度。",
     ["给定用户", "时间桶归属", "该用户的各桶条件比例"], "P({用户ID，时间桶} | {用户ID}) = 用户在时间上的集中度"),
    ("K022", "IP请求计数", "按IP维度聚合的请求数量。",
     "把请求归入对应IP并计数，得到IP下的请求量统计。",
     ["IP维度", "请求事件", "请求计数"], "C({IP}) = IP下请求数"),
    ("K023", "IP内版本分布", "给定IP后，不同客户端版本对应的条件占比。",
     "以IP为条件，统计版本与IP联合出现的比例，刻画该IP下版本构成。",
     ["IP维度", "version_code", "各版本占比"], "P({IP, version_code} | {IP}) = IP下version_code的占比"),
]:
    add(uid, ["D02"], topic, definition, mechanism, evidence, None,
        "反作弊统计特征语义；原节未规定统一窗口、桶宽、缺失语义或风险判定阈值。",
        "D02:几种有具体物理含义的统计特征/" + topic, excerpt)

for uid, topic, definition, mechanism, evidence, excerpt in [
    ("K024", "电商特征的主体与客体", "主体通常为买家或卖家；客体为与主体相对的买卖方，也可为设备、环境或身份。",
     "先确定被描述的主体，再选择其相对对象，避免把账户、设备和交易对象混成同一分析维度。",
     ["主体类型", "客体类型", "主体与客体对应关系"], "主体：主体通常是买家或者卖家；客体：通常与主体相对，可以是买家、卖家，也可能是其他客体，如设备、环境、身份等"),
    ("K025", "电商特征的时间维度", "过去一段时间窗口或某个特定时间窗口内的观测范围。",
     "时间窗口界定被纳入特征的行为范围；相同主体和函数在不同窗口下描述不同统计对象。",
     ["窗口起止或相对范围", "窗口内行为"], "时间：通常指过去一段时间窗口，或者一个特定时间窗口"),
    ("K026", "电商特征的空间变化与集中", "买家或卖家在空间上的变化与集中情况。",
     "沿空间维度观察主体位置的变化或集中，作为特征描述的一种维度。",
     ["主体", "空间位置", "变化与集中情况"], "空间：通常指买家或者卖家在空间上的变化与集中"),
    ("K027", "关系图中的社团维度", "设备、社交、支付或股权等关系图中买家或卖家所属的社团。",
     "通过不同关系建立图谱，再观察主体在图中形成的社团结构。",
     ["关系类型", "买家或卖家节点", "图谱社团"], "通过设备关系、社交关系、支付关系、股权关系等一系列关系构成的图谱数据中买家或卖家存在的社团"),
    ("K028", "行为序列特征", "主体、客体或团伙在时间窗口内的时序行为特征。",
     "保留行为先后顺序，描述指定子序列的出现次数等序列模式。",
     ["时间窗口", "时序行为", "指定子序列及出现次数"], "序列：主体、客体、团伙等过去一段时间内时序行为的特征，如某个高危子序列的出现次数"),
    ("K029", "不同时间窗口的突变特征", "通过比较不同时间窗口的行为量来描述变化的特征。",
     "用较短窗口交易金额与较长窗口交易金额的比值刻画变化。",
     ["短窗口交易金额", "长窗口交易金额", "金额比值"], "突变：主体、客体、团伙等过去一段时间行为的突变特征，如过去1天交易金额与过去7天交易金额的比值等"),
    ("K030", "短时间内行为信息冲突", "主体、客体或团伙在一段时间内出现相互冲突行为信息的特征维度。",
     "将同一主体在短窗口内的行为属性对照，例如比较登录IP城市是否不同。",
     ["同一主体", "时间窗口", "登录IP城市等行为属性"], "冲突：主体、客体、团伙等过于一段时间内行为发生的冲突，如账号过去10分钟登录IP城市不同等。"),
    ("K031", "聚合函数与统计含义", "聚合函数描述对选定主体、对象和窗口中的观测如何汇总。",
     "计数、去重计数、求和、位置与离散程度统计、比例和熵类函数提供不同统计视角。",
     ["被聚合对象", "聚合函数", "统计范围"], "SUM、COUNT、COUNT DISTINCT、MAX、MIN、平均数、中位数、众数、方差、标准差；比例、变异系数等；信息熵、互信息熵、条件熵等"),
    ("K032", "主单数量特征的五要素拆解", "账号过去30天已下单主单总数可按主体、客体、时间、条件和函数拆解。",
     "主体为账号，客体在此例中无单独指定，时间为过去30天，条件为已下单，函数为COUNT DISTINCT。",
     ["账号主体", "过去30天窗口", "已下单条件", "主单去重计数"], "特征：账号过去30天已下单主单总数；主体：账号；客体：无；时间：过去30天；条件：已下单；函数：COUNT DISTINCT"),
]:
    add(uid, ["D03"], topic, definition, mechanism, evidence, None,
        "国际电商营销风控特征设计框架的描述性知识；不将图社团、突变或冲突直接当风险标签。原节未提供正常反例、零分母或缺失处理约定。",
        "D03:特征介绍/Aimmd4OCGoqMgWxuCh7cgX15nDh/" + topic, excerpt)

add("K033", ["D04"], "单账号标签与群体聚集观察",
    "方案将单账号风险标签与某种团伙划分下的同类标签聚集结合，用于召回关联用户。",
    "先观察单账号标签，再在特定分组中同时观察同类标签数量和比例，以减少潜在误关联造成的误伤。",
    ["团伙划分方式", "单账号标签", "同类标签账号数量", "同类标签账号占比"],
    ["某种团伙划分可能存在误关联；原文因此要求同时检查标签用户数量和标签用户占比。该说明是误关联边界，不是已核验的正常案件。"],
    "Account Farming方案层范式；不保留内部阈值，不把召回等同于最终定性。",
    "D04:总体方案；团伙风险标签/CADOdtjxEoFZVxx0z38lg3eDgod/前两段",
    "团伙定性同时要求标签用户数量和标签用户占比是为了避免某一种团伙划分方式下的由于潜在误关联导致的误伤问题",
    counterexample_kind="source_boundary")
add("K034", ["D04"], "团伙标签触发人数与比例",
    "团伙行为风险特征汇总记录特定标签在分组中的触发人数。",
    "将标签触发人数除以团伙账号总数得到该标签占比，用于描述分组内的标签分布。",
    ["团伙账号总数", "标签名称", "标签触发人数", "触发人数与团伙规模的比值"], None,
    "所读方案中的汇总字段语义；统计比例不是独立风险事实，原文未定义空团伙或零分母处理。",
    "D04:团伙风险标签解释性/group_user_cnt、group_risk_labels_summary",
    "代表该团伙内有多少个账号命中了特定的单点行为风险。需要 LLM 将其转换为占比（触发人数 / 团伙规模）。")
add("K035", ["D04"], "账号关联信号的共享人数",
    "团伙网络结构汇总记录关联信号及共享该信号的人数。",
    "IP、设备或个人签名等介质可把账号串联起来；汇总同时呈现信号类型和共享规模。",
    ["关联信号类型", "共享信号的人数", "分组方式"], None,
    "所读养号方案中的字段语义；不保留真实账号或把共用IP、设备直接认定为同一人作恶。",
    "D04:团伙风险标签解释性/group_signal_type_share_summary",
    "格式为 关联信号:共享该信号的人数。代表这些账号是通过什么底层介质被串联起来的（如共用IP、共用设备、共用个人签名等）。")
add("K036", ["D05"], "社交评论的意图分类与正常互动",
    "所读专项将评论内容按意图分类，既含投资、兼职、隐私索取和导流，也含正常互动。",
    "评论意图标签描述表达内容，再由具体场景标准决定如何使用，不能把整个意图列表当诈骗账号名单。",
    ["评论内容", "索要隐私或联系动作", "投资、兼职或博彩获利表达", "问地点、拍摄或讨论内容等互动"],
    ["问地点、问拍摄、讨论内容被原文明确列为正常互动。"],
    "所读东南亚中文社交专项MVP的评论意图分类；完整MVP定性条件因摘要与明细冲突不纳入。",
    "D05:MVP版本/其他风险标签/第9项评论意图标签",
    "正常互动（问地点、问拍摄、讨论内容）",
    counterexample_kind="normal_scenario")
add("K037", ["D05"], "私信内容意图与行为观察",
    "所读专项将私信表达分为转账诱导、兼职、投资、隐私索取、导流、情感共鸣和寒暄等意图。",
    "对私信内容分类可描述表达和请求类型；此项分类本身不复原未核实的整套诈骗定性规则。",
    ["私信中的资金请求", "隐私信息请求", "站外联系引导", "情感表达或破冰寒暄"], None,
    "所读社交专项MVP的私信意图目录；原文未提供该项的完整正常例外，不据地域或人口属性直接定性。",
    "D05:MVP版本/其他风险标签/第10项私信意图标签",
    "诱导转账（借钱、急用钱、周转、充值、提现）；索要隐私（银行卡、验证码、账号、密码、身份证/护照、地址）；破冰寒暄（你好/在吗/最近怎么样/自我介绍）")

add("K038", ["D10"], "垃圾注册与账号资源获取",
    "垃圾注册是黑产批量获取账号资源并用于后续行为的入口机制。",
    "可领取权益、点赞或关注的账号具有使用价值；批量注册获得账号后可用于后续违规行为。",
    ["批量账号获取行为", "账号可使用的权益或操作", "后续账号行为"], None,
    "基础知识文档的概念机制；账号具有价值只说明可能被利用，不证明某次注册已经违规。",
    "D10:2.5垃圾注册/doxcnAT7alGfF0aqUycehZkSjHd",
    "黑产利用垃圾注册，批量获得账号，来实施前述的违法行为。")
add("K039", ["D10"], "账号价值与被盗风险分析",
    "账号被盗风险分析关注账号价值是否容易变现。",
    "余额提现、现金权益或商业广告能力提供可变现价值，使账号可能成为盗用目标。",
    ["账户余额或现金权益", "提现能力", "商业广告等变现能力"], None,
    "基础风险分析视角；是被盗可能性的分析条件，不是已发生盗号的事件证据。",
    "D10:2.6账号被盗/doxcnVs4X8lRhokYr5nn2defxvh",
    "核心看账号价值是否容易变现。例如账号的钱可以轻松提现，可以利用账号发商业广告，那么账号就存在被盗的可能性。")
add("K040", ["D10"], "业务鉴权的主体、操作与资源",
    "业务鉴权确定谁能够合法读写某个服务中的具体内容。",
    "将访问主体、读写操作、目标服务及资源内容对应，并检查授权判断机制是否有效可靠。",
    ["访问主体", "读或写操作", "目标服务与具体内容", "合法性判断机制"], None,
    "基础鉴权概念；不包含文中链接系统的具体实现或当前权限配置。",
    "D10:3.2业务鉴权/doxcnrtizZNch9fRoxHq3OwP6Ta",
    "解决谁能够合法读写什么服务的什么具体内容。合法的判断机制是否是有效的可靠的。")

add("M001", ["D07"], "标准评审的风险事实与误伤检查",
    "标准评审分别检查风险描述、细则覆盖、案例与申诉执行及体验风险。",
    "风险定性优先描述客观事实，治理侧核对覆盖与案例，体验侧检查可验证性与潜在误伤。",
    ["客观风险描述", "细则覆盖", "案例充分性", "申诉可执行性", "潜在误伤评估"], None,
    "标准评审流程方法；不将新增示例条款视作最终发布标准。",
    "D07:流程说明/评审说明",
    "标准是客观风险，尽量避免意图判断；标准在用户体验环节是否有潜在负向反馈，或处置是否容易导致潜在的体验误伤",
    auxiliary=True)
add("M002", ["D07"], "标准修订的富文本状态",
    "评审材料用颜色和删除线区分原文、变更点与删除点。",
    "黑字为原文，绿字为变更点，红色划线为删除点；抽取纯文本前需要保留这种状态信息。",
    ["变更类型", "黑字、绿字与红色划线", "新增、修改、删除或合并状态"], None,
    "所读评审页的填写约定；配色约定不自动推广到其他文档。",
    "D07:流程说明/评审内容说明",
    "黑字为原文，绿字为变更点，红色划线为删除点",
    auxiliary=True)
add("M003", ["D08"], "完整标准的组成",
    "标准内容不仅含标签路径，还包含条款、识别、处罚、申诉与案例。",
    "通过不同模块对应风险识别、处置和复核需求；目录或标签名称不能代替各模块正文。",
    ["标签路径", "标准条款", "识别细则", "处罚与申诉细则", "案例库"], None,
    "标准体系说明的方法知识；未核验管理平台现行状态和配图。",
    "D08:治理规则（标准）使用说明/核心功能/标准内容查询",
    "含标签路径、标准条款、识别细则、处罚和申诉细则、案例库",
    auxiliary=True)
add("M004", ["D09"], "业务原文与知识发布链路",
    "业务原文与Raw保留证据，结构化知识组织为可消费内容，再发布与检索。",
    "来源采集、结构化治理、质量门禁、知识发布和检索消费依次衔接，原始证据与消费层承担不同职责。",
    ["原始来源", "结构化治理记录", "质量状态", "知识发布与检索入口"], None,
    "所读知识库指南的方法；不保存资产数量、内部表位置和访问主体。",
    "D09:知识从哪里来/首段",
    "建设路径遵循“来源采集 → 结构化治理 → 质量门禁 → 知识发布 → 检索消费”。业务文档与 Raw 保留原始证据；Wiki 组织可消费的知识内容",
    auxiliary=True)
add("M005", ["D09"], "数据资产知识的来源与口径",
    "表和分析资产知识记录定义、字段、分区、口径、上下游及来源。",
    "从业务指标或消费入口回到数据来源及关联资产，帮助解释知识的计算范围和来源关系。",
    ["表定义、字段及分区", "指标口径与维度", "上下游和关联资产", "来源"], None,
    "知识库资产元数据方法；不复制具体资产清单或把元数据当业务案件。",
    "D09:知识从哪里来/业务文档、Hive表、Dataset、Dashboard行",
    "表定义、字段、分区、口径、上下游、Owner 与来源。",
    auxiliary=True)
add("M006", ["D11"], "审核识别与处置解耦",
    "将识别标签与最终处罚分别建模和执行。",
    "各场景先识别具体行为标签，标签聚合后进入处置层，由处置策略决定最终处罚。",
    ["场景内识别标签", "标签聚合", "独立处置策略"], None,
    "审核标准拆解方法；不把文中模型输出或处罚建议作为训练标签。",
    "D11:用户审核识别处置解耦",
    "人审在不同场景识别出作弊标签，标签聚合后进入处置层，由处置策略决定最终处罚。",
    auxiliary=True)
add("M007", ["D11"], "可观察元素与条件组合",
    "将专业识别标签拆成可观察元素，并用逻辑表达式明确组合关系。",
    "通过元素定义和且、或关系把复杂标准转为可执行的判断描述，再结合工作流与实际特征构造任务输入。",
    ["元素定义", "且或条件", "审核标准", "场景内容证据"], None,
    "标准拆解方法；不沿用示例人设作为诈骗事实，也不复制提示词或模型生成结论。",
    "D11:Zeroshot研判模型/首段",
    "把识别标签进一步拆解为一般常识，再结合逻辑表达式，将标准转化为逻辑语言。",
    auxiliary=True)
add("M008", ["D11"], "标准提示中的正常样本覆盖",
    "以作弊标准为主要描述内容的提示，可能对多样正常内容产生误伤。",
    "所读实践指出，提示主要描述作弊标准时，正常样本覆盖不足可能带来误判；其方案另设分流模型。",
    ["提示描述范围", "正常样本的多样性", "误伤分析", "分流与后续判定"], None,
    "原文单项实践的定性观察；不保留实验数值，不声称分流或CPT必然有效。",
    "D11:Zeroshot研判模型/对比后分析",
    "prompt主要描述了作弊标准，对内容更多样的白样本产生了较多误伤。",
    auxiliary=True)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def render(unit):
    counter = "；".join(unit["counterexamples"]) if unit["counterexamples"] else "所读来源未提供正常反例，本单元不补造案例。"
    return "\n".join([
        "主题：" + unit["topic"],
        "定义：" + unit["definition"],
        "机制：" + unit["mechanism"],
        "可观察证据或核对项：" + "；".join(unit["evidence"]) + "。",
        "反例与边界：" + counter,
        "适用范围：" + unit["scope"],
    ])


def build():
    assert len({u["unit_id"] for u in UNITS}) == len(UNITS)
    write_jsonl(DATA / "knowledge_units.jsonl", UNITS)
    manifests = []
    for split in ("train", "valid", "auxiliary"):
        subset = [u for u in UNITS if u["split"] == split]
        rows = [{"text": render(u)} for u in subset]
        write_jsonl(DATA / f"risk_cpt_{split}.jsonl", rows)
        for i, (u, row) in enumerate(zip(subset, rows), 1):
            manifests.append({
                "unit_id": u["unit_id"], "split": split, "file": f"risk_cpt_{split}.jsonl", "line": i,
                "group_id": u["group_id"], "source_ids": u["source_ids"],
                "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                "captured_at": DATE, "review_status": u["review_status"],
            })
    write_jsonl(DATA / "manifest.jsonl", manifests)
    source_rows = []
    for sid, title, url, rev, scope, group, usage in SOURCES:
        related = [u for u in UNITS if sid in u["source_ids"]]
        source_rows.append({
            "source_id": sid, "title": title, "url": url, "revision": rev,
            "read_scope": scope, "group_id": group, "usage": usage, "captured_at": DATE,
            "unit_ids": [u["unit_id"] for u in related],
            "unit_count": len(related), "full_document_read": False,
            "effective_policy_verified": False,
            "origin_candidate": sid in {f"D{i:02d}" for i in range(1, 12)},
            "via_source": "D06" if sid in {"D06a", "D06b"} else None,
        })
    (DATA / "sources.json").write_text(json.dumps(source_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    registrations = {
        f"risk_cpt_{split}": {
            "file_name": f"risk_cpt_{split}.jsonl", "formatting": "alpaca", "columns": {"prompt": "text"}
        } for split in ("train", "valid", "auxiliary")
    }
    (DATA / "dataset_info.json").write_text(json.dumps(registrations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    business = [u for u in UNITS if u["purpose"] == "domain_knowledge"]
    report = {
        "knowledge_units": len(UNITS), "business_units": len(business),
        "split_counts": dict(Counter(u["split"] for u in UNITS)),
        "source_count": len(SOURCES), "original_candidates": 11,
        "business_source_counterexamples": sum(bool(u["counterexamples"]) for u in business),
        "normal_scenario_units": sum(bool(u["counterexamples"]) and "normal_scenario" in (u["counterexample_kind"] or "") for u in business),
        "source_boundary_units": sum(u["counterexample_kind"] == "source_boundary" for u in business),
        "business_counterexamples_missing": sum(not u["counterexamples"] for u in business),
        "text_characters": {s: sum(len(render(u)) for u in UNITS if u["split"] == s) for s in ("train", "valid", "auxiliary")},
        "max_text_characters": max(len(render(u)) for u in UNITS),
        "max_text_utf8_bytes": max(len(render(u).encode()) for u in UNITS),
        "token_count": None, "tokenizer_verified": False, "training_executed": False,
        "group_policy": {"valid": sorted(VALID_GROUPS), "auxiliary_in_default_training": False},
    }
    (ROOT / "statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 知识单元审阅稿", "", "摘要均为来源改写；证据字段表示观测项或核对项，不表示已发生案件。反例缺失未补造。", ""]
    for u in UNITS:
        lines.extend([f"## {u['unit_id']} {u['topic']}", "",
                      f"集合：{u['split']}；来源：{', '.join(u['source_ids'])}；反例：{u['counterexample_status']}", "",
                      render(u), "", "来源定位：" + u["provenance"]["anchor"], "",
                      "最小支持摘录：" + u["provenance"]["supporting_excerpt"], ""])
    (ROOT / "知识单元审阅稿.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
