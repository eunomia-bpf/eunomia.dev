# 为什么 OpenTelemetry GenAI 指标目录只到引擎无关的时延与 token 计数，model-serving-signals 又怎样补齐模型服务与自动扩缩容那部分？

OpenTelemetry GenAI 约定定义的是请求级信号：操作/请求时延、首 token 时延、每输出 token 时延、以及 token 用量。它刻意保持引擎无关，所以不会覆盖任何自动扩缩容所需的服务状态信号——队列深度、请求并发、批利用率、KV 缓存压力。这些落在推理引擎自己的指标命名空间里。Kubernetes 的 `model-serving-signals` 项目（`kubernetes-sigs` 下，由 SIG Autoscaling 与 SIG Instrumentation 支持）正是在标准化这一引擎相关层：它定义引擎无关的信号契约、vLLM / SGLang / TensorRT-LLM 的引擎 profile、以及一个把引擎指标翻译成 OpenTelemetry 的 mapping exporter，让自动扩缩容能基于同一套一致的名称来做决策。

## 机制

GenAI 约定按层拆分指标目录。核心指标文档定义的是一个很小的引擎无关集合：一个客户端操作时延直方图，三个模型服务直方图（`gen_ai.server.request.duration`、`gen_ai.server.time_per_output_token`、`gen_ai.server.time_to_first_token`），再加上 workflow、agent、tool 的时延/调用指标。另一份客户端推理文档补充 token 核算——按模态细分的 `gen_ai.client.inference.usage.*` 计数器（输入、输出、cache-read、cache-write、reasoning），以及按操作的 token 直方图。这些全部是请求级量，用 `gen_ai.operation.name`、`gen_ai.provider.name`、`gen_ai.request.model` 来打键。打键的方式本身就说明了意图：约定描述的是单个模型请求或单个 agent 操作，而不是它背后的服务系统。

服务系统自身的状态——多少请求在排队、多少在途、批有多满、KV 缓存还活着多少——都不在这个目录里。每个引擎在自己的命名空间下暴露这些（vLLM、SGLang、TensorRT-LLM 各自发布不同的 gauge 集合）。约定把这些系统相关的信号明确交还给引擎，所以只看 OpenTelemetry GenAI 名称的自动扩缩容器只能看到时延和 token，对队列、批、缓存是盲的。

`model-serving-signals` 在引擎边界处补齐这块：它定义一套引擎无关的模型服务信号集合，每个引擎一个 profile 把该引擎的原生指标映射到这套集合，一个 mapping exporter 以 OpenTelemetry 形式输出，再加一个 conformance suite 把映射钉住，免得引擎升级悄悄改指标名或形状。自动扩缩容随后就基于这套通用名称，而不是各引擎各自给自己的队列深度 gauge 起的名字。

## 验证与调试路径

1. 打开 OpenTelemetry GenAI 指标文档的模型服务一节。它只列了 `gen_ai.server.request.duration`、`gen_ai.server.time_per_output_token`、`gen_ai.server.time_to_first_token` 三项。那里没有队列深度、没有并发、也没有批或缓存指标。
2. 打开客户端推理与 token 指标文档。它们补充了时延与 token 用量类仪表，但仍然没有任何服务状态信号——token 用量是核算量，不是容量信号。
3. 打开 `model-serving-signals` 仓库。确认它提供引擎 profile（vLLM、SGLang、TensorRT-LLM）以及一个 mapping exporter 加 conformance suite；这正是把引擎相关的服务指标转成自动扩缩容消费的 OpenTelemetry 名称的那一层。
4. 在真实部署里，把引擎原生 exporter 与映射后的 OpenTelemetry 序列并排抓取。你为自动扩缩容决策需要的信号——队列深度、批利用率、KV 缓存占用、SLO 达成度——应以标准化信号名的形式出现，由 mapping exporter 把引擎名称翻译成这套词表。

## 局限

- GenAI 指标约定处于 Development 状态；指标名与属性名在稳定前仍会变动。
- OpenTelemetry 目录刻意是请求级的。服务状态信号按设计是引擎相关的；只有 model-serving-signals 已映射的那部分才是可移植的。一个信号在被提进核心约定之前，就一直是引擎本地的。
- token 用量不是 SLO 的代理：低 token 工作负载照样可以卡在队列上，高 token 工作负载也可能完全没问题。只按 token 扩缩容会漏掉队列深度与 SLO 达成度，这正是服务层（而非约定本身）必须提供容量信号的原因。

## 参考

- [OpenTelemetry semantic-conventions-genai — GenAI 指标](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md) — 引擎无关的模型服务指标：`gen_ai.server.request.duration`、`gen_ai.server.time_per_output_token`、`gen_ai.server.time_to_first_token`，以及客户端操作时延直方图。
- [OpenTelemetry semantic-conventions-genai — 客户端推理指标](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md) — 推理时延、首块时延、每输出块时延，以及按操作的 token 用量直方图。
- [OpenTelemetry semantic-conventions-genai — token 指标](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-token-metrics.md) — 按模态细分的 `gen_ai.client.inference.usage.*` 计数器（输入、输出、cache-read、cache-write、reasoning）。
- [kubernetes-sigs model-serving-signals](https://github.com/kubernetes-sigs/model-serving-signals) — 面向 Kubernetes 上模型服务的引擎无关信号契约：vLLM、SGLang、TensorRT-LLM 的引擎 profile、mapping exporter、conformance suite 与自动扩缩容集成。

## 当日社区讨论

本次两个 opt-in archive 渠道（两个 CNCF OpenTelemetry 插桩渠道）产出 3 条消息。其一在问是否有计划为 Node.js 建一个 GenAI 插桩仓库以对标现有的 Python 仓库；目前没有公开答复，且与本题的服务指标边界无关。其二指向某个 agent 工具项目里的一个具体 pull request 评审讨论，本身不是一个自包含的问题。第三条，也就是本页回答的问题，描述了一个新的 Kubernetes 项目 `model-serving-signals`：把推理引擎（vLLM、SGLang、TensorRT-LLM）的指标取出来翻译成 OpenTelemetry，并询问是否已有（或计划有）比现有 GenAI 指标目录更大的目录，尤其是与模型服务和自动扩缩容相关的。上面给出的答案划定了这条边界：核心 GenAI 约定止于引擎无关的时延与 token 核算，服务状态层正是 model-serving-signals 正在标准化的部分。

本次渠道覆盖：两个 opt-in archive（eBPF 与 GenAI 插桩渠道）提供 3 条消息。可见浏览器专属渠道（eunomia-bpf 与 sched-ext 的 Discord、bpf 邮件列表、r/eBPF）本次未能复查，因为没有可见浏览器会话，所以标记为"未覆盖"，而不是"安静"。
