# Microsoft Agent Framework 概述 | Microsoft Learn

- 来源链接: https://learn.microsoft.com/zh-cn/agent-framework/overview/
- 搜索日期: 2026-06-20

## 详细内容

## Microsoft Agent Framework 概述 | Microsoft Learn

**Source**: https://learn.microsoft.com/zh-cn/agent-framework/overview/

---

使用英语阅读

---

---

注意

 访问此页面需要授权。 可以尝试登录或更改目录。 

 访问此页面需要授权。 可以尝试更改目录。 

# Microsoft代理框架

代理框架提供两个主要类别的功能：

|  | Description |
|---|---|
| 代理 | 使用 LLM 处理输入、调用 工具和MCP 服务器的单个代理，并生成响应。 支持 Microsoft Foundry、anthropic、Azure OpenAI、OpenAI、Ollama 和 more。 |
| 流程 | 基于图形的工作流，将代理和函数连接到具有类型安全的路由、检查点和人机循环支持的多步骤任务。 |

该框架还提供基础构建基块，包括模型客户端（聊天完成和响应）、状态管理的代理会话、代理内存的上下文提供程序、用于截获代理作的中间件，以及用于工具集成的 MCP 客户端。 这些组件可让你灵活地构建交互式、可靠且安全的 AI 应用程序。

## 开始

```
dotnet add package Microsoft.Agents.AI.Foundry --prerelease
```

```
using System;
using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;

AIAgent agent = new AIProjectClient(
        new Uri("https://your-foundry-service.services.ai.azure.com/api/projects/your-foundry-project"),
        new AzureCliCredential())
    .AsAIAgent(
        model: "gpt-5.4-mini",
        instructions: "You are a friendly assistant. Keep your answers brief.");

Console.WriteLine(await agent.RunAsync("What is the largest city in France?"));
```

```
pip install agent-framework
```

```
from agent_framework.foundry import FoundryChatClient
    from azure.identity import AzureCliCredential

    credential = AzureCliCredential()
    client = FoundryChatClient(
        project_endpoint="https://your-foundry-service.services.ai.azure.com/api/projects/your-foundry-project",
        model="gpt-5.4-mini",
        credential=credential,
    )

    agent = client.as_agent(
        name="HelloAgent",
        instructions="You are a friendly assistant. Keep your answers brief.",
    )
```

```
# Non-streaming: get the complete response at once
    result = await agent.run("What is the largest city in France?")
    print(f"Agent: {result}")
```

就是这样， 一个调用 LLM 并返回响应的代理。 在这里，可以 [添加工具](https://learn.microsoft.com/zh-cn/agent-framework/agents/tools/)、 [多轮对话](https://learn.microsoft.com/zh-cn/agent-framework/agents/conversations/session)、 [中间件](https://learn.microsoft.com/zh-cn/agent-framework/agents/middleware/)和 [工作流](https://learn.microsoft.com/zh-cn/agent-framework/overview/#when-to-use-agents-vs-workflows) 来生成生产应用程序。

注释

代理框架 **不会** 自动加载 `.env` 文件。 若要使用 `.env` 文件，请在应用程序开始时调用 `load_dotenv()` ，或直接在 shell 或 IDE 中设置环境变量。

[入门 - 完整教程](https://learn.microsoft.com/zh-cn/agent-framework/get-started/your-first-agent)

## 何时使用代理与工作流

| 使用代理时... | 使用工作流... |
|---|---|
| 任务是开放式或对话性质的 | 该过程具有定义完善的步骤 |
| 你需要自主工具的使用和计划 | 你需要显式控制执行顺序 |
| 单个 LLM 调用（可能配备工具）就足够了 | 多个代理或函数必须协调 |

*如果可以编写函数来处理任务，请执行该作，而不是使用 AI 代理。*

## 为什么使用代理框架？

代理框架将 AutoGen 的简单代理抽象与语义内核的企业功能（基于会话的状态管理、类型安全性、中间件、遥测）相结合，并为显式多代理业务流程添加基于图形的工作流。

[语义内核](https://github.com/microsoft/semantic-kernel)和[AutoGen](https://github.com/microsoft/autogen)开创了 AI 代理和多代理业务流程的概念。 代理框架是由同一团队创建的直接继承者。 它将 AutoGen 针对单代理和多代理模式的简单抽象与 语义内核 的企业级功能（如基于会话的状态管理、类型安全、筛选器、遥测以及广泛的模型和嵌入支持）相结合。 除了合并这两者之外，Agent Framework 还引入了工作流，使开发人员能够显式控制多代理执行路径，以及一个可靠的状态管理系统，用于长时间运行和人机循环方案。 简言之，Agent Framework 是下一代语义内核和 AutoGen。

若要了解有关从 语义内核 或 AutoGen 迁移的详细信息，请参阅 autoGen 中的 [Migration Guide from 语义内核](https://learn.microsoft.com/zh-cn/agent-framework/migration-guide/from-semantic-kernel/) 和 [Migration Guide from AutoGen](https://learn.microsoft.com/zh-cn/agent-framework/migration-guide/from-autogen/)。

语义内核和 AutoGen 都从开源社区中获益匪浅，代理框架预期也是如此。 Microsoft代理框架欢迎贡献，并将继续改进新功能。

重要

如果使用Microsoft代理框架生成使用任何第三方服务器、代理、代码或非Azure直接模型（“第三方系统”）运行的应用程序，则自行承担风险。 第三方系统是Microsoft产品条款下的非Microsoft产品，受其自己的第三方许可条款的约束。 你需对任何使用及相关成本负责。

建议查看与第三方系统共享和接收的所有数据，并识别第三方处理、共享、保留和位置数据的做法。 由你负责管理数据是否会流出组织的Azure合规性和地理边界以及任何相关影响，以及预配适当的权限、边界和审批。

你负责仔细查看和测试使用特定用例上下文中的 Microsoft Agent Framework 生成的应用程序，并做出所有适当的决策和自定义。 这包括实施自己的负责任的 AI 缓解措施，例如元系统、内容筛选器或其他安全系统，并确保应用程序满足适当的质量、可靠性、安全性和可信度标准。 另请参阅： [透明度常见问题解答](https://github.com/microsoft/agent-framework/blob/main/TRANSPARENCY_FAQS.md)

## 后续步骤

[步骤 1：第一个代理](https://learn.microsoft.com/zh-cn/agent-framework/get-started/your-first-agent)

**更深入：**

- [代理概述](https://learn.microsoft.com/zh-cn/agent-framework/agents/) - 体系结构、提供程序、工具
- [工作流概述](https://learn.microsoft.com/zh-cn/agent-framework/workflows/) - 顺序、并发、分支
- [集成](https://learn.microsoft.com/zh-cn/agent-framework/integrations/) - A2A、AG-UI、Durable Extension、M365

---

## 反馈

 此页面是否有帮助？ 

否

 需要有关本主题的帮助？ 

 想要尝试使用 Ask Learn 阐明或指导你完成本主题？ 

---

##  其他资源 

---

- 

Last updated on  2026-04-07
