# OpenTelemetry OBI 的 Collector 接收器为什么保持独立模块和独立版本线，为什么普通的 go get 更新不会把接收器拉到 RC 标签？

有两道相互独立的屏障。OBI 把 Collector 接收器拆成了一个嵌套 Go 模块——`collector/` 子目录有自己的 `go.mod`，import 路径保持不变——而这个嵌套模块走自己的 v0 版本线，标签打在 `collector/v0.15.2`；核心模块则打不带前缀的 `v0.15.2` 和 `v1.0.0-rc.2`。普通的 `go get -u` 到不了 RC，原因有二：Go 的 `latest` 与 `upgrade` 查询偏好正式版而非预发布版，而 `v1.0.0-rc.2` 是预发布；接收器模块在 v1 线上根本没有标签——`collector/v1.0.0-rc.2` 不存在——所以接收器侧最高可选版本停在 v0 线上。要把接收器配上 RC 测试，必须在一次更新里显式选中两个模块各自的版本。

拆分本身是 v0.15.x 发布事故的修复：一个 commit 同时挂核心标签和 Collector 标签时，`git describe` 把 Collector 版本代进了核心发布产物的路径；模块拆分（#3748）加上流水线修复（#3770）让 `v0.15.2` 得以带着这两项一起发布。

## 机制

1. **接收器是一个嵌套模块，走独立的发布线。** 自 `v0.15.2` 起，`collector/` 子目录包含一个 `go.mod`，声明接收器模块 `go.opentelemetry.io/obi/collector`；接收器包保持既有 import 路径，嵌套模块对核心模块的要求是 `v0.15.2`。Go 对子目录模块的规则是：每个版本标签都要加上模块子目录前缀——这里是 `collector/v0.15.2`——而根模块直接打 `v0.15.2` 与 `v1.0.0-rc.2`。因为 Go 在 v2 以下不要求主版本后缀，接收器可以在核心走 v1 RC 的同时独立地走 v0 线，这就是发布说明里说的「自己的模块和版本线」。

2. **为什么做拆分。** `v0.15.0` 被 BPF 校验器失败挡住；`v0.15.1` 带上了修复，但其产物构建失败，因为版本发现对同时挂有核心标签与 Collector 标签的 commit 跑 `git describe`，把 Collector 版本代进了核心产物路径。流水线修复把产物路径（二进制归档、生成源码、SBOM、校验和）钉在选定的核心标签上，`v0.15.2` 则以「替代未发布的 v0.15.0 与 v0.15.1」的名义把修复与拆分一起发布。

3. **为什么 `go get -u` 发现不了 RC。** Go 的版本查询 `latest`、`upgrade`、`patch` 都偏好正式版而非预发布版：若 `v1.2.2` 与 `v1.2.3-pre` 同时可用，`latest` 选 `v1.2.2`，尽管预发布在排序上更高。对 OBI 有两条推论。核心模块侧，`v1.0.0-rc.2` 是预发布，所以普通更新解析到最高的非预发布版本——核心是 `v0.15.2`（v0 的正式版本在查询语义上仍是「正式版」，只是没有兼容性承诺）。接收器模块侧，v1 线上没有标签：`collector/v0.15.2` 是它唯一的标签，不存在可去更新的 `collector/v1.0.0-rc.2`。RC2 发布说明原话：「仅靠 RC 标签不能通过普通的 `go get -u` 发现拆分后的接收器。」

4. **支持的配对是显式的两模块选择。** 一次 `go get` 同时给出两个模块的版本：接收器模块选 `v0.15.2`，核心模块选 `v1.0.0-rc.2`。两者随后各以一条 `require` 出现在 `go.mod` 里，接收器对核心的最低版本要求由显式选中的 RC 版本满足。

## 验证与排障路径

1. **直接读两条标签线。** 对 OBI 仓库跑 `git ls-remote --tags`，核心线（`v0.15.2`、`v1.0.0-rc.2`）与带子目录前缀的接收器线（`collector/v0.15.2`）一目了然；`collector/v1.0.0-rc.2` 的缺席就是那个任何更新查询都发现不了的事实。
2. **确认查询实际选中了什么。** `go list -m -versions` 列出含 RC 标签在内的全部标签版本，但 `go get -u` 解析到最高非预发布版——普通升级会让核心停在 v0 线、接收器停在 `collector/v0.15.2`。
3. **显式钉住 RC 配对。** 按 RC2 发布说明的配对：一次 `go get` 同时选接收器模块的 `v0.15.2` 与核心模块的 `v1.0.0-rc.2`。更新后 `go.mod` 有两条 `require`，`go.sum` 有两条模块线各一组条目。

## 限制

- RC2 里发布的遥测 schema 仍是 `0.15.2`；`1.0.0` 的 schema 切分在最终版才发生，所以 RC2 的输出还不是最终的 v1 契约。
- 遥测名称与 label 在 RC 之间发生过变化，升级时既有仪表盘与查询需要重新映射。
- 接收器的 `v0.15.2` 线是 v0 线：无兼容性承诺，且可以独立于核心演进（或落后于核心）。
- `go get -u` 的正式版偏好是 Go 模块查询的通用性质，不是 OBI 的特例：任何模块的普通更新都不会落到预发布标签上。
- 本回答描述 `v1.0.0-rc.2` 候选版本的状态；最终 `v1.0.0` 发布时可能在 v1 线上补打接收器标签，消掉这个不对称。

## 参考资料

- [OBI 发布 v1.0.0-rc.2](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/releases/tag/v1.0.0-rc.2) - 影响发布的变更：接收器成为独立模块 `go.opentelemetry.io/obi/collector`、独立的 v0 版本线、没有 `collector/v1.0.0-rc.2` 标签、显式的两版本 `go get` 配对、Config v2 作为提议中的 v1 配置契约（Config v1 冻结）、遥测 schema 保持 `0.15.2` 直至最终版。
- [OBI 发布 v0.15.2](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/releases/tag/v0.15.2) - 为什么 `v0.15.2` 替代了未发布的 v0.15.0 与 v0.15.1：挡住 v0.15.0 的 BPF 校验器失败、`git describe` 把 Collector 标签代进产物路径（#3770 修复）、校验器循环放宽的修复（#3759）、接收器拆分（#3748）。
- [Go Modules Reference](https://go.dev/ref/mod) - 「Version queries」：`latest`、`upgrade`、`patch` 偏好正式版而非预发布版；「Module directories within a repository」：子目录模块使用子目录前缀标签；「Mapping versions to commits」：根模块使用纯版本标签。
- [OBI v1.0.0 跟踪 issue](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/1133) - RC 到最终版的路径与 schema 切分。

## 今日社区讨论

本次运行监控窗口是两个 opt-in 归档源的 7 天归档窗口，共返回 5 条消息，全部在同一条线程上：OBI 从 v0.15.x 到 v1.0.0-rc.2 的发布序列。可见浏览器专属来源——eunomia-bpf 与 sched-ext 的 Discord 服务器、bpf 邮件列表、以及 r/eBPF——本次因没有可用的可见浏览器会话而无法审阅，故标记为「未覆盖」而非「安静」。

**OBI v0.15.x 发布风波。** 线程从 `v0.15.0` 被 BPF 校验器失败挡住开始，修复已在 main、v0.15.1 即将发出；随后报告 v0.15.1 的产物构建失败；最终 `v0.15.2` 宣布包含独立模块与所需修复，v0.15.2 确认可用。v0 线上操作者的实际操作：接收器使用 `collector/v0.15.2`。产物路径修复是流水线侧的变更，用户无需绕过。

**OBI v1.0.0-rc.2 社区测试。** 窗口以宣布 `v1.0.0-rc.2` 供社区测试收尾：Go Auto SDK span 捕获、扩展的 .NET 插桩与运行时指标、AWS Route53 名称解析，以及 BPF 校验、trace 上下文传播、指标方面的修复。从 RC1 升级的用户面对 Config v2 迁移与遥测兼容性变化：遥测名称与 label 已改变，Collector 接收器从此有自己的模块与版本线——正是本文回答的问题。未决边界：schema 仍是 `0.15.2`，接收器没有 v1 线标签。

**一位新贡献者的插桩提案。** 一位新贡献者请求评审 opentelemetry-python-genai 项目中提议的 sentence-transformers 插桩（公开 issue #910 与 PR #911）。该请求在评审队列中，尚无社区可总结的答案。
