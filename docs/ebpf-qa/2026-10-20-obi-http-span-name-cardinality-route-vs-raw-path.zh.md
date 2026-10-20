# OBI 的 HTTP span 为什么会出现高基数的名称，OpenTelemetry 语义约定又如何限制 span 名称的基数？

span 名称是追踪系统里基数最高的单一字段，所以 OpenTelemetry 的 HTTP 语义约定把名称固定为 `{method} {target}`，并要求 `target` 必须是**低基数**的：服务端用 `http.route`，客户端用 `url.template`。该约定明确说明，一个 instrumentation **不得**把原始 URI 路径作为 target 的默认值。OpenTelemetry eBPF 探针（OBI）构造 HTTP span 名称用的正是这种形式——方法加上路由——而让名称变高基数的唯一输入，就是那个路由槽位里被填进去的东西。高基数的症状并不是默认行为，而是选择 `unmatched: path` 选项时产生的结果，它恰好是约定禁止作为默认值的那一种。

一个会随着你服务的不同 URL 数量而增长的 span 名称，是基数问题而不是正确性问题：真正决定两个请求是归并成同一个名称、还是各占一个名称的，是那个路由值。

## 机制

1. **约定的名称规则。** HTTP 语义约定规定，HTTP span 的名称在有低基数 `target` 时 SHOULD 是 `{method} {target}`，没有低基数 `target` 时 SHOULD 只是 `{method}`。其中 `{method}` 是 `http.request.method`；当方法落在已知方法集合之外时，用字面量 `HTTP`。`target` SHOULD 是服务端 span 的 `http.route` 或客户端 span 的 `url.template`，并说明 `url.template` 是路径的低基数模板（比如是 `/users/{id}` 而不是 `/users/42`）。那段话里最硬的一条护栏是最后一句：instrumentation 不得默认使用 URI 路径作为 target。

2. **OBI 把 HTTP span 命名为 `{method} {route}`。** 在 `pkg/appolly/app/request/span.go` 里，HTTP 事件类型的 `TraceName` 构造器从方法出发——方法已知就按原样输出，未知就输出 `HTTP`——并且只有在路由被解析出来时才把路由追加上去。一个没有解析出路由的 span，因此名称就只是 `{method}`。被解析出的路由，也是 OBI 导出时服务端 span 上的 `http.route`、客户端 span 上的 `url.template` 的同一个值，所以 span 名称和导出的属性保持一致。

3. **路由从哪里来。** 在 `pkg/transform/routes.go` 里，路由是从 URL 路径上匹配出来的，顺序是：用户定义的 `routes.patterns` 匹配器、再是各服务的自定义匹配器、最后是收割到的路由匹配器。当这些都没有命中时，由 `unmatched` 策略决定 target。这个开关就是基数所在：`unset` 让 target 留空，名称只剩 `{method}`；`wildcard` 把 target 收敛成一个通配 token；`heuristic` 把相似路径聚合成一个低基数模板；`low-cardinality` 同样做这种聚合，并且额外用一个路径段上限来收紧；而 `path` 把路由直接设成原始 URL 路径。

4. **为什么 `path` 是高基数情形。** 当 `unmatched` 是 `path` 时，代码把原始路径直接赋给路由，于是每个不同的 URL 路径都变成一个不同的 span 名称：`/users/42` 一个名字、`/orders/1934` 一个名字，如此类推。这是一个随流量增长的无界集合，也就是所说的基数爆炸。而随发行版下发的默认值是 `heuristic` 聚类，它把名称集合保持在有界；`low-cardinality` 选项再用每服务的路径段上限进一步收紧。约定里"不得默认使用 URI 路径"正是 `path` 只能作为可选项、而非默认值的原因。

## 验证与排查路径

1. **先读 `routes.unmatched` 的值。** 默认是 `heuristic`，所以一个新部署的 OBI 已经把路径聚合成有界名称。如果你看到的 span 名称是"每个 URL 一个"，就去找一个把 `unmatched` 设成 `path`（或者在旧版布局里空值落到了通配）的显式 `routes` 配置块——那就是要改的字段。

2. **把 span 名称和路由属性一起看。** 对一条繁忙路径，列出不同的 `http.route` / `url.template` 值，而不是完整的 `url.full`。有界命名会显示少数几个模板值；按原始路径命名则会显示每个不同 URL 一个值。span 名称和这个属性应当一致，因为它们由同一个路由值构造。

3. **单独限制 metric 序列。** 即便名称有界，按 span 名称聚合的 metric 序列还有一个"按名称聚合的上限"兜底，所以一旦出现了病态的名称，结果是被截断的 metric 集合，而不是无界集合。这个上限限住了伤害，但不修复命名本身。

## 局限

- 约定的基数规则是 SHOULD 与 MUST 混合的："不得默认使用 URI 路径"是唯一硬约束；选择 route 还是 template、以及选择低基数选项，都是 SHOULD 级的建议，所以运维者仍然可以用 `unmatched: path` 主动选进原始路径命名。
- `heuristic` 和 `low-cardinality` 是 OBI 内部的聚类策略，不是规范：它们的输出取决于 URL 聚类器和每服务路径段上限，所以对某个工作负载得到的确切模板集合是 OBI 的行为，不是语义约定能固定的东西。
- `unset` 直接把 target 去掉（名称只剩 `{method}`），是基数最低的选择，但代价是名称里不再有按路由的区分，等于用细节换基数。
- 本回答描述的是 OBI `main` 分支上 HTTP span 的行为。路由匹配管线和 `unmatched` 选项的名字都是 OBI 的具体实现；而 span 名称基数这条规则本身是任何 HTTP instrumentation 都遵循的通用语义约定。

## 参考

- [OpenTelemetry 语义约定，HTTP span 名称规则](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/http/http-spans.md) - `Name` 一节：名称是 `{method} {target}`，target SHOULD 是 `http.route` 或 `url.template`，并且 instrumentation 不得默认使用 URI 路径作为 target。
- [OBI `pkg/appolly/app/request/span.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/appolly/app/request/span.go) - `TraceName` 把 HTTP span 名称构造为方法加上解析出的路由，方法未知时退化为 `HTTP`。
- [OBI `pkg/transform/routes.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/transform/routes.go) - 路由从 URL 路径匹配的地方，以及 `unmatched` 策略（含原始路径情形）赋给 target 的地方。
- [OBI `pkg/obi/config.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/obi/config.go) - 随下发出来的默认值：`unmatched` 取启发式聚类、通配字符为 `*`、路径段基数有上限。
- [OBI span 属性目录 `spans.md`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/site/docs/spans.md) - 按 span 类型划分的属性参考，说明 OBI 发出 HTTP 客户端和服务端 span，并携带 URL 与路由属性。

## 今日的社区讨论

本次运行里，两个 opt-in 的 CNCF OpenTelemetry 探针通道贡献了一个三条消息的冻结窗口。实质性的一条、也是本草要回答的那条，是一位实践者指出 OBI 的 HTTP span 名称"按目前的方案可能基数会很高"——简言之，当一个 span 的 target 是每次请求的原始 URL 路径时，每个唯一的 path 都会产生一个不同的名字，而这正是 HTTP 语义约定里"不得默认使用 URI 路径"这条护栏所要防的基数爆炸。该线程里一个可行的后续动作，就是把这个 target 限制成路由或模板，而不是原始路径。

同一个窗口里还有一条独立、尚未解决的线——有人索要一个之前提到过的"skill 细化"条目的指针——它属于另一条线程，没有被合并进本草要。

本次运行的通道覆盖：两个 opt-in 归档共提供三条消息。仅可视浏览器才能访问的来源——eunomia-bpf 与 sched-ext 的 Discord、bpf 邮件列表、以及 r/eBPF——因本次运行没有可视浏览器会话而无法审阅，因此标记为"未覆盖"而非"安静"。
