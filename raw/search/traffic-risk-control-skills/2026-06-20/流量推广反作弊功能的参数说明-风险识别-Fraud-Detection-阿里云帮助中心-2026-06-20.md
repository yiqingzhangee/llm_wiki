# 流量推广反作弊功能的参数说明-风险识别(Fraud Detection)-阿里云帮助中心

- 来源链接: https://help.aliyun.com/zh/fraud-detection/developer-reference/traffic-promotion-fraud-prevention-and-parameters
- 搜索日期: 2026-06-20

## 详细内容

## 流量推广反作弊功能的参数说明-风险识别(Fraud Detection)-阿里云帮助中心

**Source**: https://help.aliyun.com/zh/fraud-detection/developer-reference/traffic-promotion-fraud-prevention-and-parameters

---

[官方文档](https://help.aliyun.com/zh)

广告主在进行互联网渠道的营销推广时，经常会面临假量、刷量、转化率低等问题。阿里云流量推广反作弊产品帮助广告主、以及媒体在流量变现、渠道推广、RTA

场景识别营销风险，基于设备、网络环境、行为异常等维度实时判断风险，提升流量质量，帮助客户降本增收。目前已覆盖休闲游戏、社交、电商、零售、出行等行业头部客户。

## 版本介绍

为匹配不同行业、不同阶段企业的风控需求，流量推广反作弊分为基础版、增强版，两者的差异对比如下：

| 功能特性 | 基础版 | 增强版 |
|---|---|---|
| 实时分析 | 支持。 | 支持。 |
| 服务返回值 | 风险特征标签。 | 风险特征标签。 |
| 设备风险监测 | 支持。包括模拟器、多开、设备牧场、群控设备、云手机、HOOK等异常检测能力。 | 支持。包括模拟器、多开、设备牧场、群控设备、云手机、HOOK等异常检测能力。 |
| 设备指纹 | 不支持。 | 支持。 |
| 团伙分析 | 支持。 | 支持。 |
| 日志服务（SLS）投递 | 不支持。 | 支持。用户可授权进行日志投递，免费存储一年。 |

## 服务事件参数

服务事件参数是指公共参数中

ServiceParameters

字段的请求参数，为

JSON

格式。以下表格为**流量推广反作弊**服务（含基础版、增强版）事件需传入的请求参数。 

**

**重要 **

以下服务事件参数中，为最大化适配各类入参情况，阿里云风险识别服务不对入参的字符串进行业务格式校验，您需要自行对数据业务格式进行校验。例如，mobile

字段，需接口请求方对入参是否符合中国内地手机号业务格式（11

位纯数字、1

开头等）进行校验。

| 字段名称 | 支持版本 | 字段描述 | 数据格式 | 样例数据 | 是否必须 |
|---|---|---|---|---|---|
| deliveryMode | 基础版、增强版 | 投放模式。 | String | 例如，通用买量、RTB、RTA等。 | 可选 |
| advertisingType | 基础版、增强版 | 广告位类型。 | String | 例如，信息流、开屏等。 | 可选 |
| operateTime | 基础版、增强版 | 精确到秒的操作时间戳，格林威治时间。说明 若扫描历史数据风险，该字段需要传入历史操作时间，避免因计算时间错误造成误识别。 | Long | 例如，格林威治时间2018-04-01 12:00:00，时间戳为1522555200。 | 可选 |
| imeiMd5 | 基础版、增强版 | 设备imei的MD5编码值。 | String | IMEI设备号转成小写，再进行MD5编码，再小写。 | 可选 |
| androidIdMd5 | 基础版、增强版 | 设备Android Id的MD5编码值。 | String | Android ID保持原值，进行MD5编码，再小写。 | 可选 |
| oaidMd5 | 基础版、增强版 | 设备oaid的MD5编码值。 | String | Oaid保持原值，进行MD5编码，再小写。 | 可选 |
| idfaMd5 | 基础版、增强版 | iOS设备IDFA的MD5编码值。 | String | IDFA设备号转成大写，进行MD5编码，再小写。 | 可选 |
| macMd5 | 基础版、增强版 | MAC设备号的MD5编码值。 | String | Mac设备号保持原值，进行MD5编码，再小写。 | 可选 |
| operateSource | 基础版、增强版 | 操作来源。可选取值：PC、H5、APP。 | String | APP | 可选 |
| deviceToken | 基础版、增强版 | 通过设备风险SDK获取的设备令牌。 | String | MzQvo1d7scyZ3tl_RcJZo_QOytAjy1LWRRLoRKo5oZSoo_JGj1ZoR5JGoRo5jcdn57gV5kxVRcLER5RQoZSvRZZQRcROjcMW5csZR_RGy_55RKJ_oooqZ7dSV5gRnKxOV7eWVQQjRtlRQoAjRcM0 | 必选（集成阿里云设备SDK）。 |
| ip | 基础版、增强版 | 用户操作的真实IP地址。 | String | 42.120.XX.XX | 可选 |
| mobileMd5 | 基础版、增强版 | 用户手机号的MD5编码值。 | String | e7beea81b7a03b38508428fbeeb3**** | 可选 |

**

**重要 **

IMEI、IDFA、OAID、ANDROIDID、MAC

等设备号需要严格按照上述加密标准加密，否则将极大影响识别效果。

## 返回参数

流量推广反作弊的返回参数为风险特征标签。

返回参数`Data`中的`tags`字段值的业务含义，可参考以下表格中的操作建议（根据阿里云风控团队经验得出）。 

| 风险等级 | 标签值 | 标签含义 | 操作建议 |
|---|---|---|---|
| 高风险 | is_emulator | 设备疑似模拟器，支持Android和iOS。 | 建议进行业务过滤。 |
| 高风险 | is_rooted | 设备疑似root，iOS系统下代表疑似越狱。 | 建议进行业务过滤。 |
| 高风险 | is_hooked | 设备疑似注入式攻击。 | 建议进行业务过滤。 |
| 高风险 | …… | …… | 建议进行业务过滤。 |
| 中风险 | is_deviceCluster_m | 设备疑似群控中风险，Android系统特有。 | 建议进一步安全验证或限制高危业务使用权限。 |
| 中风险 | …… | …… | 建议进一步安全验证或限制高危业务使用权限。 |
| 低风险 | token_replay | 设备SDK重放问题。 | 建议打标观察。 |
| 低风险 | …… | …… | 建议打标观察。 |

**

**说明 **

以上风险特征标签仅为部分展示，完整的标签枚举您可以参考风险识别控制台的接入管理模块，[点击前往](https://yundunnext.console.aliyun.com/?spm=5176.2020520154.0.0.3175jUbkjUbk4z&p=safnext#/analysis/paraphrase)。

关于公共返回参数的更多信息，请参见[公共返回参数](https://help.aliyun.com/zh/fraud-detection/developer-reference/common-parameters#concept-2469521/section-kuo-ovc-bgl)。

[上一篇：登录风险识别功能及参数说明](https://help.aliyun.com/zh/fraud-detection/developer-reference/logon-fraud-detection-and-parameters)

[下一篇：其他风控产品](https://help.aliyun.com/zh/fraud-detection/developer-reference/other-risk-control-features/)

该文章对您有帮助吗？
