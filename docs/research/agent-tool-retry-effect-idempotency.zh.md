---
date: 2026-09-14
slug: agent-tool-retry-effect-idempotency
title: "AI Agent 重试一次工具调用，怎么知道没有把事情做两遍？"
description: "工具调用可能在外部操作已成功后丢失响应；如果 Agent 把超时直接当失败重试，支付、消息、部署和云资源都可能被重复执行。"
tags:
  - Daily Report
  - AI Agent
  - Tool Use
  - Distributed Systems
  - Reliability
research_question: "AI Agent runtime 如何在超时、取消、重试、进程重启和工具服务切换之间，为同一个外部副作用保留稳定身份，同时又不把真正的新操作误判成重复？"
source_cutoff: 2026-09-14
status: daily-report
---

# AI Agent 重试一次工具调用，怎么知道没有把事情做两遍？

一个会调用工具的 Agent 请求云 API 创建任务。服务端已经把任务建好了，但响应在网络中丢失，Agent 最后只看到一个 timeout。

接下来应该怎么办？

直接重试，可能再创建一个任务；完全不重试，可能让用户的工作停在一半；把 timeout 文本交给模型，让模型猜“刚才大概成功了没有”，则是把分布式系统里的状态不确定性变成概率判断。

这个问题也不只发生在云资源上。一次模糊的重试可能重复支付、把同一封消息发两次、创建两个 ticket、启动两次部署、重复预订资源，甚至把破坏性操作再执行一次。真正难处理的不是“调用开始前就明确失败”，而是 **ambiguous completion：调用方已经不知道外部副作用到底有没有提交**。

今天的 Agent 协议已经暴露出这个边界。MCP 有 `idempotentHint`，但规范明确把 ToolAnnotations 定义为 hint：除非来自可信 server，否则客户端不能把它当成真实行为保证。2026-07-28 版 MCP 进一步把协议核心改成 stateless；Multi Round-Trip Request 在补齐用户输入时会重新发起原始 `tools/call`，而且新的 JSON-RPC request ID 必须与第一次不同。HTTP 请求断开也可以取消协议层请求，但这并不能倒流已经提交到下游系统的动作。

因此真正缺的不是更多 retry counter，而是：**给用户想做的那一个逻辑副作用一个持久身份，并且在结果未知时先做 reconciliation，再决定是否允许下一次 attempt 产生新的外部效果。**

<!-- more -->

## Request ID 不是 Effect ID

至少要把三种身份分开：

| 身份 | 例子 | 应该表示什么 |
| --- | --- | --- |
| 协议 request ID | JSON-RPC `id: 37` | 对应一次 request 和 response |
| attempt ID | `attempt-3` | 标识一次真实执行尝试，便于 tracing 与调试 |
| effect ID | `launch-report-job-2026-09-14` 或 opaque UUID | 标识工作流真正想产生的那一个外部副作用 |

当前 MCP 的语义本身就说明三者不能混为一谈。普通 `tools/call` 带 JSON-RPC request ID；如果 server 返回 `input_required`，client 会带补充输入再次调用，而规范明确要求 retry 使用不同的 JSON-RPC ID。这对 request/response correlation 是正确的，但也意味着 request ID 不适合作为逻辑操作的长期身份。

新的 stateless MCP core 让边界更明显。请求不再依赖 protocol-level session，可以落到任意 server instance。需要跨调用保持状态的工具，应显式返回 handle，再由后续调用把 handle 传回来。这个设计很适合 shopping cart、browser context、transaction 等长期对象，但它并不会自动告诉一次 one-shot mutation 的调用方：“你刚刚 timeout 的那个动作其实已经成功提交了。”

MCP `ToolAnnotations` 里还有 `idempotentHint`。它为 true 时，意思是用相同参数重复调用不应对环境产生额外效果。但同一份 schema 也明确写着：所有 annotation 都只是 hint，不能保证真实行为。即使 server 可信，这个 hint 仍然只是描述某个 tool operation 的一般性质，而不是证明某一次 provider 请求在 crash 之后还记得 dedup token，也没有说明 token 能保留多久，更没有定义两组看上去相同的 JSON 参数在业务上是否真的是“同一个动作”。

HTTP 也是如此。RFC 9110 把一部分 method 定义为 idempotent，因为重复相同请求的预期效果和执行一次相同，这使通信失败后的自动 retry 可以安全进行。但它不会让所有 `POST` 自动获得幂等性，也不会替任意业务系统生成一个逻辑 operation identity。

生产 API 通常需要再加一层。AWS 的部分 mutation API 使用 client token：相同 token 和参数再次请求，不会再次 mutation；参数改变则冲突，而且 token 有明确的有效期。Stripe 也把 idempotent replay 绑定到 idempotency key、具体 API、account/sandbox 和 retention window。它们共同说明一个事实：**idempotency 不是一个 Boolean，而是一组关于 identity、参数绑定、作用域、保留时间和 replay 的契约。**

而通用 Agent runtime 恰恰会同时跨过很多互不相同的契约。

## 最危险的状态不是 failed，而是 unknown

很多 retry middleware 只把一次工具调用建模成 success 或 failure。对于会产生外部副作用的操作，至少还要再有一个状态：

```text
not_started
in_progress
committed
failed_before_effect
unknown_after_dispatch
```

`unknown_after_dispatch` 才是这里最麻烦的状态。它可能来自：

- provider 已经提交 mutation，但 response 回来前 TCP 断开；
- client timeout，但 server 还在继续执行；
- cancellation 与不可逆步骤发生 race；
- Agent process 已经发出请求，却在持久化 result 前 crash；
- tool server 已经修改下游状态，却在返回前重启；
- gateway 在 backend 完成以后才返回 5xx。

AWS ECS 的 idempotency 文档直接描述了这类问题：资源已经被修改以后仍可能遇到 timeout 或 server issue，因此调用方无法判断 retry 会不会叠加新的 mutation。client token 的价值就在这里，它解决的不是“服务器会不会报错”，而是“报错之后是否还知道之前那次逻辑操作是什么”。

Agent 系统还多一层风险：恢复过程通常由模型参与。模型可能重新生成 tool call、调整一个 optional field、换 endpoint，或者为了“修复”而重新描述操作。这些能力在普通错误恢复中很有用，但如果第一次 mutation 可能已经存在，它们会让第二次调用绕过 provider 自己的幂等判断。

所以 runtime 不能只问“这个错误是否 transient，可以重试吗”，还必须区分：当前是在 retry **同一个 effect**，repair **同一个 effect**，还是用户真的授权了一个 **新的 effect**。

## 现有研究还缺什么

第一处缺口是 **跨 Agent layer 的 effect identity**。模型的 tool-call ID、JSON-RPC ID、trace span ID、workflow node ID、provider idempotency token、最终资源 ID 都各有用途，但大多数 Agent runtime 并没有一个从“用户想做的 mutation”开始，经过模型 retry、协议 retry、tool server restart 和下游 API 之后仍然不变的 durable identifier。

第二处缺口是 **ambiguous completion 之后的 reconciliation**。支付和云 API 往往有 idempotency token 或查询接口，但通用工具还会封装 shell、browser、email、自建 API 和多阶段 workflow。调用 timeout 后，Agent 通常缺少一个统一问题：“effect E 是否已经发生？如果发生了，它对应的真实 result 是什么？”

第三处缺口是 **idempotency contract discovery**。MCP 的 `idempotentHint` 有用，但规范故意把它设计成 advisory hint。不同 provider 的 token scope、retention period、参数等价规则和 failure behavior 都不同。一个 tool 自己可能看起来 idempotent，但内部某个 downstream call 并不是；反过来也可能成立。一个 Boolean 描述不了真实 replay boundary。

第四处缺口是 **evaluation**。很多 Agent benchmark 测 task completion、tool-call accuracy 或 latency，却很少专门在“外部系统已经 commit，但 acknowledgement 还没回来”这个点注入故障，再去数真实世界里出现了几个副作用。因此某个 runtime 可能在普通 API error 下看起来很可靠，却在真正需要 retry 时制造重复动作。

这个问题与之前的[并行 Agent effect serializability 报告](https://eunomia.dev/zh/research/parallel-agent-effect-serializability/)并不相同。之前问的是多个 worker 的副作用如何组合成一个合法结果；这里甚至可以只有一个 worker、一个用户意图。问题是多个 execution attempt 会不会把同一个 mutation 实体化多次。

## 有学术价值也有生产价值的方向

### 1. 给每一个外部 mutation 分配 durable effect identity

Agent runtime 可以在 dispatch mutation 前生成 `effect_id`，并在后续 retry 中保持不变。这个 ID 应该由 runtime 生成，而不是让语言模型临时想一个字符串。

最小的 effect record 可以是：

```text
effect_id = random stable UUID
workflow_id = parent user task
intent_hash = canonicalized mutation intent
tool = provider + tool name + contract version
authority = principal + approval/policy generation
target = normalized logical resource
attempts = [a1, a2, ...]
state = prepared | dispatched | committed | failed | unknown
provider_key = downstream idempotency token if supported
receipt = provider result/resource identity if known
retention_deadline = provider dedupe horizon if known
```

这里最重要的是把 `effect_id` 和 `attempts` 分开。一次 timeout 可以新增 attempt，但不应该自动新增 effect。如果模型修改了操作，而且修改已经足以改变 canonicalized intent，runtime 也不应偷偷复用旧 effect ID；它应该把这次变化识别成 conflicting retry，或者只有在 workflow 明确进入新动作时才生成新 effect。

adapter 再把稳定的 effect identity 映射到不同 provider 的机制上。AWS 支持的 API 可以使用 client token；Stripe 可以生成符合其 account/API scope 的 idempotency key；数据库可以使用 unique constraint 或 transaction record；文件系统可以用 staged object 加 atomic rename。完全没有 dedup 支持的工具，也至少可以声明：一旦结果进入 unknown，就必须 reconcile 或重新确认，而不是自动 replay。

评测时，应在 provider 明确 commit 后故意丢掉 response，重启 Agent runtime，再恢复同一个 workflow。第一指标不是“retry 成功率”，而是 **每一个 logical effect 实际产生了多少个 authoritative external effect**。第二指标再看误拦截合法新操作、ledger 存储开销和额外 latency。

学术贡献是一个端到端 naming model，把 transport identity、execution attempt 和外部 effect identity 分离。生产集成点则是 agent tool gateway 或 workflow runtime，因为这些位置能在调用进入任意 provider 之前统一生成 identity。

如果现有 workflow ID、MCP request ID 或 provider token 已经能在所有相关 retry/restart 里稳定存在，而且能无歧义区分 repair 与新操作，那么这个方向应该被否定。额外 effect ID 如果没有改变任何故障结果，就只是重复 metadata。

### 2. 结果不确定时进入 reconciliation，而不是自动 replay

只有 effect identity 还不够，runtime 还需要定义 ledger 进入 `unknown` 后做什么。

对于真正支持 idempotency 的 provider，可以用同一个 provider token 和同一组绑定参数 retry。对于支持 token/resource lookup 的系统，可以先查询已有 result，再决定是否需要 mutation。更弱的工具也可以提供 observation predicate，例如“是否已经存在带这个 external reference 的 ticket”“这个 release ID 是否已经产生 deployment”“provider 是否已经接受过这个 message ID”。

核心规则应该是：

> **一次 ambiguous mutation 不是普通 failure，而是一条尚未被外部事实解释清楚的 pending fact。只有 reconcile 到外部状态或明确的 dedup contract 以后，才知道能不能安全重试。**

reconciliation record 不能只有 success/failure。它应记录外部状态是证明已经 commit、证明没有 commit，还是依然无法判断。对于无法消除歧义的高风险、非幂等操作，最安全的结果可能是停止自动 retry，并给用户一个具体的 unresolved decision，而不是继续赌。

这个机制还必须知道 retention semantics。AWS client token 和 Stripe idempotency key 都不是永久存在。如果 Agent sleep 几天之后继续执行，原本的 dedup window 可能已经过期，此时“沿用同一个 token”未必仍等于“受到同一个 replay protection”。所以 runtime 应把已知 replay horizon 放进 effect record，过期后切换 policy。

评测可以覆盖 network partition、commit 后 5xx、client/server crash、cancellation race、超过 dedup TTL 的长时间 suspend，以及模型 repair 时产生 parameter drift。除了 duplicate effect，还要测 unknown state 停留时间、不必要的人类升级，以及 reconciliation 把错误外部对象绑定到 effect ID 的比例。

学术贡献是一个针对 Agent 外部副作用、在 outcome knowledge 不完整时仍然可恢复的 protocol。生产使用者包括长期运行的 cloud agent、communication agent、ticketing/billing agent 和 deployment agent。

如果盲目 retry 配合现有 provider API 在同样的 fault matrix 上已经实现零 duplicate，而且 unresolved task 更少，那么这个方向没有必要。reconciliation 不应成为每个调用都必经的重型步骤，只应覆盖那些 outcome ambiguity 会制造第二个真实副作用的边界。

### 3. 做一个测“exactly-once intent”而不是“exactly-once execution”的对抗性 benchmark

在分布式系统里，“exactly once”这个词很容易让人承诺过头。Agent runtime 很难保证整个调用链每一层只执行一次。更可操作的目标是：**一个经过用户授权的 logical intent，最终最多只能形成一个 authoritative external effect，即使系统内部执行过多个 attempt。**

benchmark 可以把这个目标直接写成 oracle。每个 workload 声明：

```text
intent I
allowed external outcome set O(I)
effect observation oracle
provider idempotency/reconciliation contract
failure injection points
```

然后在普通 unit test 很少覆盖的边界注入故障：

- dispatch 前；
- provider 收到请求后、mutation 前；
- mutation 已提交、response 前；
- response 已到 tool server、尚未到 Agent；
- cancellation 过程中；
- Agent 已 timeout 但 backend 仍在执行时；
- Agent process crash/restart 后；
- stateless tool server 被负载均衡到另一个 instance 后；
- provider dedup token 过期后；
- 模型 repair 改动某个参数时。

workload 既要包含有强 idempotency 的真实 provider adapter，也应包含故意较弱的工具，例如“append 一行”“发送消息”“没有 client token 的 create resource”。Ground truth 必须来自外部状态，而不是 tool 是否返回了 `success`。

至少把四个指标分开报告：duplicate authoritative effects、missing intended effects、把合法新动作误判成重复的 false deduplication、以及最终仍 unresolved 的 ambiguous outcomes。latency 和 token cost 放在后面。一个 task-completion 分数很高但偶尔能支付两次的 Agent runtime，不应该通过这类测试。

它的学术价值是一套针对 Agent tool reliability 的 failure model 和 correctness metric；生产价值则是为 retry middleware、MCP gateway、workflow engine 和高风险 tool adapter 做 regression testing。

如果现有 Agent reliability suite 已经能注入 post-commit acknowledgement loss，并且可以证明 restart、cancel、load balancing 和 dedup-window expiry 下的外部 effect uniqueness，那么这个 benchmark 就不该重复造轮子。第一步应该是先验证现有测试究竟覆盖到哪里。

## 现在部署 Agent runtime，可以先遵守什么规则

自动 retry 一个会 mutation 的工具之前，至少分开回答五个问题：

1. 这个 tool operation 真的是 idempotent，还是只有 annotation 或经验判断？
2. 下游 provider 是否接受稳定的 idempotency/client token，它把哪些参数、account、region 或 API scope 绑定在一起？
3. provider 会记住这个 identity 多久？
4. dispatch 后丢失 response 时，能否查询或 reconcile 已有 effect？
5. 如果以上都不成立，重复执行真的比保持 unresolved 更安全吗？

read-only 调用没有必要承担这套开销；真正有明确幂等契约的 mutation 可以在契约范围内积极 retry；高后果的 non-idempotent action 则应该有 durable effect identity，并在 ambiguous failure 后进入 reconciliation，而不是让模型猜下一次怎么调用。

当前 MCP 设计可以承载这种机制，但不会自动提供它。stateless request 和不同 JSON-RPC ID 是 transport semantics；ToolAnnotations 是有用的 hint；显式 handle 是应用状态。这些东西单独拿出来，都不是一次外部 mutation 跨 retry 的端到端身份。

## 哪些结果会改变这个判断？

如果 Agent protocol 以后标准化了一个可信的端到端 mutation identity，而且它能跨 transport retry、tool-server restart、模型 repair 和 downstream provider 继续保持同一语义，并且真实实现已经能在 fault injection 中消除 duplicate external effect，那么本文提出的 runtime-level effect ledger 就会明显变轻，甚至没有必要。

如果生产 Agent 的绝大多数工具最终被证明都是 read-only，或者业务层天然 idempotent，这个问题也会变成少数高风险 adapter 的局部工作，而不是通用 runtime 问题。

最后还应与更简单的方案竞争。如果“provider 支持 native idempotency token 就用；不支持的 mutation 一旦 outcome ambiguous 就不自动 retry”在真实 workload 上能得到相同 completion rate 和零 duplicate，那么通用 effect ledger 是过度设计。

所以最有用的边界其实很简单：**retry 是一次新的执行尝试，但不自动意味着用户产生了一个新的意图。网络不再告诉你刚才发生了什么时，Agent runtime 需要足够持久的 identity 与 reconciliation，才能守住这条线。**

## 参考资料

- Model Context Protocol，[Tools — 2026-07-28 specification](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)，访问于 2026-09-14。
- Model Context Protocol，[Schema Reference — `ToolAnnotations`](https://modelcontextprotocol.io/specification/2026-07-28/schema)，访问于 2026-09-14。
- Model Context Protocol，[SEP-2575: Make MCP Stateless](https://modelcontextprotocol.io/seps/2575-stateless-mcp)，访问于 2026-09-14。
- RFC 9110，[HTTP Semantics, Section 9.2.2: Idempotent Methods](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2)，访问于 2026-09-14。
- Amazon ECS，[Ensuring idempotency](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_Idempotency.html)，访问于 2026-09-14。
- Amazon EBS，[Ensure idempotency in StartSnapshot API requests](https://docs.aws.amazon.com/ebs/latest/userguide/ebs-direct-api-idempotency.html)，访问于 2026-09-14。
- Stripe，[API v2 overview — idempotent requests](https://docs.stripe.com/api-v2-overview)，访问于 2026-09-14。
