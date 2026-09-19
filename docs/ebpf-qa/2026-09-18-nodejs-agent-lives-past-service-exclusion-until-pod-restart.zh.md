# 为什么把服务排除出 eBPF 追踪后，eBPF 追踪器注入的 Node.js 代理仍留在进程里，直到 pod 重启才消失？

**简短回答：** 因为注入的 JS 代理是一次性的、跟随进程生命周期的附加物，OBI 的排除规则并不会把它撤掉。OBI 发现 Node.js 进程后，通过 Node 检查器协议（`SIGUSR1` + `Runtime.evaluate`）向其中注入一个 JS 代理；注入是一个排队的动作，在发现阶段接受该进程"很久之后"才执行，而代理一旦注入，就活到该进程生命结束。`exclude_instrument` 这类规则是*发现期（discovery-time）的筛选规则*——它们决定"哪些进程被插桩"，而不是"已经装进去的代理要不要被移除"。所以在测试完之后把服务排除掉，只是让 OBI 不再插桩*新的*进程化身（incarnation），而已在运行的 pod 里还带着那个代理（及其每次回调的事件循环开销），直到这些 pod 重启、产生新的进程化身。这就是为什么真正让延迟回落的是 rollout 重启。

## 两条对不齐的生命周期

混淆来自两个生命周期不同的对象：

1. **OBI 配置里的排除规则**（`discovery.exclude_instrument`，以及每条规则里的 `exclude:`）。它们是在发现期求值的筛选条件，与包含选择器用同一种定义格式。回答的是"OBI 该挑哪些进程来插桩？"；文档给它的定位正是这种用途——"避免插桩可观测性环境中常见的那些服务"，或按可执行文件路径排除端口转发器。这些规则里没有任何一条是发给进程的指令。
2. **进程内已注入的代理。** 注入器的设计注释把时机写得很直白："注入是排队的，在发现接受该进程很久之后才执行"；目标被钉在 PID + 启动时间上（"对进程化身的稳定引用"），因为"单靠数字 PID 就可能把被回收复用的进程识别、发信号并注入到原进程的位置"。注入是针对某个*进程化身*的一次性事件，不是一个需要持续维护的订阅。

代理一旦进进程，它的生命周期就是进程的生命周期。`fdextractor.js` 代理甚至在每次重新注入时*恢复*原来的 `net` 原型（这样重新注入不会重复包裹），其运行时指标的清理"放在 gate 之外：一次禁用了运行时指标的重新注入，必须拆除上一次注入装上的东西"——但已发布路径里没有任何东西会在配置变化时把代理移除。代理并不知道 OBI 排除了这个服务，它只是继续跑那些装进来的钩子。

## 为什么排除是闸，不是收回器

`exclude_instrument` 与包含选择器用同一种定义格式（namespace、标签、可执行文件路径、开放端口……），OBI 的文档把它指向"丢弃不该插桩的服务"——可观测性栈组件、与真实服务共享端口的 Docker/socat 转发器。语义上这是一个*发现过滤器*：当 OBI（重新）扫描要插桩的进程时，被排除的服务只是不被选中而已。对于一个"在被插桩之后"才被排除的服务：

- OBI 不再拾取它的新化身；
- 已经在跑的 pod 里还带着代理，`async_hooks` `before` 钩子和 `fs.existsSync` 哨兵仍在事件循环上触发；
- 重启（或 pod 自然的进程生命周期）产生一个新的化身，而这个化身现在会被发现阶段排除——只有到那时，每次回调的成本才真正消失。

所以 rollout 重启不是变通手段，而是"发现期排除"变成"对运行中进程生效"的机制。eBPF 侧行为一致：uprobe 附加在进程化身上，被排除的服务下次启动时不再被附加——你要回收的是那个正在运行的化身。

## 怎么验证你看的是哪一半

1. **先确认配置真的到位，再去测任何东西。** 确认 `exclude_instrument` / `exclude` 选择器（或 `OTEL_EBPF_TARGET_PID` 覆盖）匹配该服务，然后在一个*新启动*的 pod 上验证 OBI 没有插桩它——这是配置控制的那一半。
2. **把运行中的 pod 分开检查。** 在排除之前就被插桩过的 pod 里还带着代理。决定性信号是进程的年龄相对配置变更的时间：比它更早启动的都还在付每次回调的成本，更晚启动的则不付。把两者混在一起的延迟对比会被读成"排除没生效"。
3. **对一个"已排除但已注入"的 pod 做事件循环 profile。** 如果 p99 仍显示每次回调的哨兵成本，说明代理仍驻留；修复方式是回收该 pod（或，随着在途工作落地——一个已关闭的草稿 PR，把哨兵换成 `fs.existsSync` 并加了"关机时卸载代理"，给代理一条不依赖进程死亡的显式离场路径）。
4. **不要把"排除后仍昂贵"当成发现 bug。** 这是"排除是发现期闸"的预期后果。尚未解决的边界是：已发布流程里没有对已安装代理的活体回收；那正是那项在途卸载工作要解决的。

## 决定性的边界

`exclude_instrument`（和每条规则里的 `exclude:`）决定的是哪些*发现*事件 OBI 会响应，而不是向已经带着注入代理的进程发一条"把自己撤掉"的消息。注入是一次性、以化身作用域为界的事件——钉在 PID + 启动时间上，因为单靠 PID 可能被回收——所以代理的移除跟随进程死亡，而不是配置的措辞。决定性边界：**排除在下一个进程化身生效，而不是当场生效。** 实际后果是：测试中途排除一个服务，仍需要 rollout 重启受影响的 pod 才能让延迟改善落地；跟进动作是把排除规则保持好以覆盖新化身，并在代理卸载工作落地后用它在无需重启的情况下停止付这项成本。

## 参考资料

- [OBI 文档：Service discovery（`exclude_instrument`——"指定排除服务被插桩的筛选条件"，与包含选择器同一定义格式；每条规则的 `exclude:`）](https://opentelemetry.io/docs/zero-code/obi/configure/service-discovery/)
- [OBI 文档：Troubleshooting（按可执行文件路径把端口转发器排除出插桩——排除闸的实际用法）](https://opentelemetry.io/docs/zero-code/obi/troubleshooting/)
- [OBI 源码：`injection_target.go`（注入"是排队的，在发现很久之后才执行"；目标钉在 PID + 启动时间上，因为"单靠数字 PID 就可能把被回收复用的进程识别、发信号并注入到原进程的位置"）](https://raw.githubusercontent.com/open-telemetry/opentelemetry-ebpf-instrumentation/main/pkg/internal/nodejs/injection_target.go)
- [OBI 源码：`fdextractor.js`（代理在每次重新注入时恢复原 `net` 原型；运行时指标"Cleanup stays outside the gate: a re-injection with runtime metrics disabled must tear down what a previous injection installed"）](https://raw.githubusercontent.com/open-telemetry/opentelemetry-ebpf-instrumentation/main/pkg/internal/nodejs/fdextractor.js)
- [OBI 草稿 PR #3357：nodejs 哨兵性能 + "关机时卸载代理"（已关闭、未合并——已发布流程中尚不存在的"对运行中进程显式离场"路径）](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/pull/3357)
- [Node.js 文档：`async_hooks`（注入代理安装的 `before` 钩子；标记为实验性，性能上推荐 `AsyncLocalStorage`）](https://nodejs.org/api/async_hooks.html)

## 当日社区讨论

本监控窗口是跨两个白名单 Slack 归档（均为技术内容）的滚动一周；两个仅可见浏览器的聊天工作区与公共邮件列表/论坛表面本次未审阅，该缺口已如实记录而不视为安静。若干主题自前几日的窗口延续而来；技术上较新的角度是代理生命周期。

**Node.js 代理生命周期与排除（即上文问题）。** Node 哨兵成本归属的讨论线继续推进，维护者确认了该读法：客户端 span 父化走 fd 对 map，每次回调的哨兵让 trace 上下文 map 与活动请求对齐；另有一个消费者是外部 trace/profile 关联——所以只按"手动 span + 日志富化"来 gate 哨兵，可能会悄悄破坏那个集成。随后一位使用者报告，把某服务排除出追踪后，必须 rollout 重启所有 pod 才看到延迟回落，他由此推断"排除之后追踪进程仍挂在 pod 里"。这个报告正是本页回答的生命周期边界：代理是一次性、以化身作用域为界的附加物，发现期排除无法从运行中的进程收回它——是重启把它移除。维护者的方向（把每次回调的刷新与 fd 对关联解耦、按消费方 gate）加上那个已关闭、未合并的哨兵性能 + 关机时卸载代理的草稿 PR，意味着已发布流程里尚无活体回收路径；未解决的边界是 OBI 是否会获得一个显式的代理移除触发器。

**OBI 配置 v1→v2 迁移与 Helm chart 缺口（延续）。** 仍有一位使用者在自己的配置上撞到 "fields are outside the supported v1-to-v2 migration contract"；维护者立场未变：全有或全无的 `migrate` 是故意的（静默丢字段会产出"看起来合法但行为实质不同"的配置），建议的折中是显式的 `--allow-partial` / `--best-effort` 模式——把能迁移的迁好、报告被省略的字段，而不是做成静默默认。相关的 OBI Helm chart v2 缺口（chart 的 `_helpers.tpl` 仍往 v2 配置 schema 里塞 v1 字段）被要求到 helm-charts 仓库单独开 issue，附上 chart/OBI 版本、相关 values、渲染后的配置与校验错误。

**托管智能体 harness 与可观测性锁定（延续）。** 关于托管 Agents API 与"供应商把遥测握在手里"这一通病的讨论：当 agent 循环跑在托管 harness 里，内部 span 树只有在 harness 导出时才存在；一个不暴露 tracing 配置或外部 trace 导出器的公开 beta，意味着通用 HTTP 客户端插桩只能重建出站交叉。一位使用者提出把模型/工具调用计量成普通的 HTTP 回执头，让既有 HTTP 客户端 span 直接采到、而无需发明新的属性词汇；另一条关于 GenAI 智能体 span 形状（每个回合一个智能体 span、模型调用与工具执行作兄弟、用工具调用标识符关联）的讨论已对照当前 agent-span 约定作答。
