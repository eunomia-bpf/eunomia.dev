# 本地嵌入模型没有服务器也没有 token 计数，为什么它的 GenAI span 属性要声明期望缺失项，这些缺失项如何保持符合规范？

GenAI 语义规范把 span 的每个属性都按其取值的可得性来定级，而不是按 span 是否存在来定级。必需属性很少（操作名、提供方名），其余要么是条件必需（有值时），要么是建议级（尽力而为）。本地模型不暴露任何服务器，也不报告计费 token 用量，所以符合规范的做法是省略这些属性，并在插桩的 conformance scenario（一致性场景）中把这次省略声明为期望缺失（expected gap）。声明缺失项正是让遥测保持合规的手段。

同一条边界也让新的插桩只需几行代码就能包住一个本地库，而不必重建整套遥测管道。

## 机制

1. **定级针对的是取值可得性，而不是 span 的存在。** 嵌入 span（`gen_ai.operation.name` 为 `embeddings`，span kind 为 CLIENT）列出的必需属性很少：`gen_ai.operation.name` 与 `gen_ai.provider.name`。其余属性级别更轻：`gen_ai.request.model` 在取值可得时条件必需；`gen_ai.usage.input_tokens`、`server.address`、`gen_ai.response.model` 为建议级。建议级信号在易于获取时应当默认发出；Opt-In 级信号不得在用户未显式配置时发出。这些级别关心的是「某个值能否被诚实地产生」，而不是逼一个 span 携带目录里的每一个键。

2. **本地嵌入模型改变了每一个取值的来源。** 对托管端点而言，服务方返回的响应对象里带有模型名、服务器地址和计费 token 计数。对进程内模型而言，请求与模型共享同一地址空间：没有网络跳、没有服务器地址、没有响应模型标识、没有计费用量对象——因为没有服务方在对这次调用计费。唯一可信的取值是调用方自己控制的那几个：构造时传入的模型名，以及模型自身通过 API 暴露的嵌入维度。

3. **span 的形状由此边界得出。** 包住 encode 调用的嵌入插桩发出一个 CLIENT span（名称为 `embeddings {gen_ai.request.model}`）外加 `gen_ai.client.operation.duration` 直方图，设置 `gen_ai.request.model` 与 `gen_ai.embeddings.dimension.count`，把 `gen_ai.provider.name` 设为所属生态，并对没有取值来源的属性直接省略，同时在一致性场景中把缺失声明为期望缺失。规范把一致性场景当作插桩预先声明「本提供方家族无法产生哪些建议级属性」的地方。

4. **插桩包裹哪里由库的调用图决定。** sentence-transformers 的 `encode_query` 与 `encode_document` 是薄特化层，委托给共享的 `encode` 方法，因此只要包住 `encode` 一次就覆盖三个入口，不会双重计数。维度属性从模型实例读取，其嵌入维度访问器返回整数，取值在加载时未知则返回 null。

## 验证与排障路径

1. **读嵌入 span 的级别表。** GenAI span 规范里的 span 定义对每个属性内联标注了级别：必需、有值时条件必需、或建议级。写 span 构造器之前先读这张表，逐个属性判断本提供方家族的取值有没有真实来源。
2. **证明被省略的属性确实无源可取。** 对进程内模型，在库 API 里确认没有用量对象、没有响应模型字段、也没有可观察的网络对端。随后让一致性场景把这次缺失记录为期望缺失，而不是留给一致性测试事后发现。
3. **搭一个本地模型，检查发出的遥测。** 一个用内存权重的小模型即可跑通真实的 encode 路径而无需下载。确认 span 名称是 `embeddings {model}`，必需属性齐全，时长直方图带建议的显式桶边界，而缺失的键恰好是已声明的那几个。
4. **核对直方图桶边界。** 操作时长指标应当采用规范建议的显式桶（0.01 到 81.92 秒），因为嵌入延迟的分布横跨三个数量级。

## 限制

- `gen_ai.usage.input_tokens` 是嵌入 span 的建议级属性，但只有自带 token 计数的提供方才能填；进程内模型没有计费器，这个缺失是结构性的、永久的，不是 TODO。
- 嵌入维度访问器在加载时未知会返回 null，此时该属性被省略，基于维度的告警必须容忍其缺席。
- 记录期望缺失的一致性场景好坏取决于更新纪律：上游库一旦增加用量报告或响应标识，缺失清单必须重新推导，否则插桩会悄悄偏离目录。
- 级别是策略而非强制执行：规范不会因缺一个建议级属性而让一致性检查失败；真正的执行点是已声明的期望缺失清单。
- 本回答描述的是当前 GenAI 语义规范中 Development 状态的嵌入 span；目录稳定期间级别可能收紧。

## 参考资料

- [OpenTelemetry GenAI 语义规范：span（嵌入）](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) - 嵌入 span 定义：`gen_ai.operation.name` 应当为 `embeddings`，span kind 为 CLIENT，逐属性的需求级别，以及 `gen_ai.request.model` 的条件必需措辞。
- [OpenTelemetry GenAI 语义规范：指标](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md) - `gen_ai.client.operation.duration` 直方图定义与建议的显式桶边界。
- [OpenTelemetry GenAI 属性注册表](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md) - `gen_ai.usage.input_tokens` 的尽力而为措辞：用提供方给出的总数（若有），或从解析出的提供方输出累加。
- [OpenTelemetry Python GenAI：何时在此添加插桩](https://github.com/open-telemetry/opentelemetry-python-genai/blob/main/CONTRIBUTING.md) - 原生/一方遥测与外部插桩的边界；遥测最好由最贴近被描述库的一方持有与维护。
- [sentence-transformers SentenceTransformer API](https://github.com/UKPLab/sentence-transformers/blob/master/SentenceTransformer.py) - `encode`、`encode_query`、`encode_document` 的委托关系，以及返回整数或 null 的嵌入维度访问器。

## 今日社区讨论

本次运行监控窗口是两个 opt-in 归档源的 7 天归档窗口，共返回 1 条消息：一位新贡献者请求评审 opentelemetry-python-genai 项目中提议的 sentence-transformers 插桩（公开 issue #910 与 PR #911）。可见浏览器专属来源——eunomia-bpf 与 sched-ext 的 Discord 服务器、bpf 邮件列表、以及 r/eBPF——本次因没有可用的可见浏览器会话而无法审阅，故标记为「未覆盖」而非「安静」。

**面向本地嵌入的新插桩。** 该请求是为本地嵌入库新增一个基于共享 GenAI 工具层的新插桩包：每次 encode 调用发一个嵌入 span 与一个操作时长直方图，记录模型名与嵌入维度。它的两个开放问题正是本页回答的内容。其一，进程内模型没有服务器地址、没有响应模型标识、没有 token 用量，所以插桩在一致性场景里把这些声明为期望缺失，而不是凭空造值。其二，该包会把一个体量很大的推理依赖拉进工作区锁文件，维护者需要决定是否接受这份权重。评审尚未定稿；必需属性、条件必需属性与建议级属性之间的边界，正是让所提议遥测保持诚实的东西。

**对实践者而言边界在哪。** 给插桩本地模型的人的实际结论是：没有用量数据，span 依然有用，因为时长、模型名与维度才是驱动检索排障的属性。token 用量的缺失是结构性的，不是 bug；假设每个 GenAI span 都有 token 计数的仪表盘，会在没有用量的嵌入路径上高估检索成本。
