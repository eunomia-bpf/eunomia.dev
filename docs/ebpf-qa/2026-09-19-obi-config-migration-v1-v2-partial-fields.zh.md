# 为什么 OBI 的 v1 到 v2 配置迁移在有些字段无法映射时拒绝写出文件，中间地带是什么？

**简短回答：** 因为 `obi config migrate` 是一个"行为保持"的转换，而不是一遍翻译。只要某个 Config v1 字段没有对应的 Config v2 等价字段，该命令就会中止并报 `migration failed: fields are outside the supported v1-to-v2 migration contract: <字段列表>`，且什么都不写（退出码 1）——它刻意不会输出一份"基本正确"的 v2 文件，因为被静默丢弃的字段会产出一个"能通过校验但行为变了"的配置。维护者所要求的那个中间地带现在以显式开关落地：`obi config migrate --allow-partial <文件>` 会为每个**能**保留的字段写出合法的 v2 配置，报告那些未映射字段，并以一个区别于 0 和 1 的独立退出码 3（`ExitPartial`）来标记缺口。部分迁移是选开行为而非默认，且它不会把原 v1 字段注释掉——它直接让这些字段不出现在 v2 输出里，并列出它们以便你逐个手工映射。

## 默认是刻意为之的全有或全无

迁移命令只读你给的那个 v1 文件，把它转成 Config v2，再把结果经 v2 校验器回灌，并检查"本应保留的每个输入字段"确实都保留了下来。在源码（`cmd/obi/internal/configcmd/configcmd.go`）里，无法映射的字段被累计进一个 `unsupported` 列表；命令计算 `partial := len(unsupported) != 0`，然后：

```
if partial && !options.allowPartial {
    return nil, "", false, fmt.Errorf(
        "fields are outside the supported v1-to-v2 migration contract: %s",
        strings.Join(unsupported, ", "),
    )
}
```

所以不加参数时，单个无法映射的字段就让整次运行失败、不写任何 v2 文件。原因在于 `migrate` 承诺的是"行为保持"的转换：静默丢字段会生成一个"看起来合法但改变了 OBI 捕获内容"的 v2 配置。文档也写明同一契约——"如果迁移命令无法保留某个设置，它会失败并在错误信息里点出那个 Config v1 字段。重新运行前先替换或淘汰那个不受支持的行为。"这正是为什么，对一个同时混有"可映射"与"不可映射"字段的配置而言，在那个开关落地之前，唯一的选择是：剥掉出问题的字段、迁移、再手工把剥掉的东西加回来。

## 中间地带现在以显式开关落地

一位使用者问的那个"中间地带"——能迁的迁好、剩下的报告出来——现在是一等公民，而不是提案。在 `configcmd.go` 里：

```
allowPartial := flags.Bool("allow-partial", false,
    "write valid v2 output when some v1 fields cannot be preserved")
...
if partial {
    return output, partialMigrationReport(replaced, unsupported), true, nil
}
```

而 `runMigrate` 在部分运行写出文件时返回 `ExitPartial`（3）：

```
ExitSuccess = 0   // 全部迁移完成，无未映射字段
ExitError   = 1   // 解析/校验/迁移失败（含全有或无命中）
ExitUsage   = 2   // 参数错误
ExitPartial = 3   // 写出了 v2 文件，但有若干 v1 字段未带过去
```

关键区别：
- `--allow-partial` 为每个可保留字段写出**合法**的 v2 配置——不是有损转储。
- 它**不会**把原 v1 字段注释掉。很多 v1 字段在 v2 结构里没有可对应的位置，无处再嵌；命令直接省略它们，并在部分报告里点名，而不是塞回原处。
- 退出码 3 是给自动化的钩子：据此分支、读取报告出的字段清单、把每个被报告的字段映射到受支持的 Config v2 或 Collector 行为。这是预期工作流，不是变通。

## 如何在不丢字段的前提下做迁移

1. 保存当前 v1 文件并记下 OBI 二进制版本（命令只读你给的文件；经环境变量、命令行参数、Helm 注入的设置不会被迁移）。
2. 先跑 `obi config migrate <v1.yaml>`。干净运行退出码 0 并打印 `migrated v1 config to OBI config v2`；撞上未映射字段则退出码 1 并给出字段清单。
3. 混合场景跑 `obi config migrate --allow-partial <v1.yaml>`。可映射子集得到一份生成的 v2 文件；退出码为 3，部分报告列出每个未带过去的字段。
4. 用 `obi config validate <生成的v2.yaml>` 确认输出合法，然后在金丝雀部署里验证行为。对每个被报告的字段，挑选受支持的 v2/Collector 等价物（迁移指南里"需手工改动的设置"表格列出了常见映射，例如 `service_name` → 顶层 `resource` 属性）并应用到 v2 文件。
5. 注意一个行为变化需保留：v1 文件若没有选择器字段则**禁用**应用捕获，而 v2 默认**包含**工作负载——所以当源文件未选择工作负载时，迁移会把 `default_action` 设为 `exclude`。迁移之后若重新启用 v2 的默认包含，就会悄悄比 v1 配置捕获更多。

## 第二个边界：Helm chart 仍会注入一个仅 v1 的字段

另一个叠加的缺口在 chart 侧，而非迁移代码侧。`opentelemetry-ebpf-instrumentation` Helm chart 渲染时会注入一个 v1 风格的字段进它生成的配置，而该字段并不属于 Config v2 schema——所以即便迁移本身干净，经未更新的 chart 渲染出的 v2 配置仍会校验失败。这是 chart 缺陷，不是 OBI 迁移代码缺陷：修法是把 chart 升级得不再注入那个仅 v1 字段（或加 v2 感知渲染），并应在 `opentelemetry-helm-charts` 仓库上报，附上 chart 与 OBI 版本、相关 values、渲染出的配置与校验错误。

## 决定性的边界

决定性边界在于：迁移**默认是"校验闸门 + 全有或全无"的转换**，而它的逃生舱是一个**显式、会出报告、且带独立退出码的部分模式**，而非静默的尽力而为。默认不输出部分文件、不把无法映射字段注释掉的原因，是"看起来合法"却丢了或挪了某字段的 v2 配置可能通过校验并运行，却改变了 OBI 的观察内容。所以实用规则是：只有当你确实会去读那份部分报告并手工映射清单里的每个字段时，才用 `--allow-partial`；对必须无损迁移的配置，保留普通 `migrate` 作为闸门。并且把 chart 缺口分开看——干净的 v2 迁移仍可能在 chart 停止注入那个仅 v1 字段之前于校验失败。

## 参考资料

- [OBI 文档：从 OBI Config v1 迁移到 Config v2（退出码表；"如果迁移命令无法保留某设置，它会失败并点出那个 Config v1 字段"；拒绝未知 v1 字段；源文件未选择工作负载时置 `default_action: exclude`）](https://opentelemetry.io/docs/zero-code/obi/configure/migrate-to-config-v2/)
- [OBI 文档：Config v2 参考（`capture.policy` / `capture.rules`、`default_action`、省略如何改变捕获）](https://opentelemetry.io/docs/zero-code/obi/configure/config-v2/)
- [OBI 源码：`cmd/obi/internal/configcmd/configcmd.go`（`obi config migrate`——`--allow-partial` 开关、`fields are outside the supported v1-to-v2 migration contract` 报错、`ExitPartial = 3`）](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/cmd/obi/internal/configcmd/configcmd.go)
- [OBI Helm chart：`opentelemetry-ebpf-instrumentation/templates/_helpers.tpl`（chart 注入的那个不属于 v2 schema 的仅 v1 字段）](https://github.com/open-telemetry/opentelemetry-helm-charts/blob/main/charts/opentelemetry-ebpf-instrumentation/templates/_helpers.tpl)
- [opentelemetry-helm-charts issues（chart 的 v2 支持缺口在此跟踪）](https://github.com/open-telemetry/opentelemetry-helm-charts/issues)

## 当日社区讨论

本监控窗口是跨两个已 opt-in 只读归档的 Slack 归档（均为 OpenTelemetry 插桩频道）的滚动一周；两个 Discord 工作区与公共邮件列表/子版表面本次未审阅，该缺口已如实记录而不视为安静。若干主题自前几日的窗口延续而来；其中有新边界、且有源码确证的，是配置迁移。

**OBI 配置 v1→v2 迁移与部分迁移边界（即上文问题）。** 一位使用者在配置撞上 `fields are outside the supported v1-to-v2 migration contract`，必须剥字段、迁移、再加回，因为没有写出 v2 文件、当时也没有部分路径；他询问中间地带。维护者立场如今在源码中落地：全有或无的默认是故意的（静默丢字段会产出"看起来合法但实质行为不同"的配置），逃生舱是显式的 `--allow-partial`——能迁的迁好、剩下的报告出来；而把原字段注释掉被排除，因为许多 v1 字段在 v2 里没有落点。未解决的边界是操作层面的而非技术层面的：`--allow-partial` 带着独立退出码落地了，但后续动作（读部分报告并手工映射每个被列出字段）仍需手工，且"我那些字段里哪些是被报告出来的"这一常见场景尚无指引。

**OBI Helm chart 的 v2 缺口（另一个边界）。** 一位使用者发现 chart 仍向渲染出的配置注入一个仅 v1 字段，因此即便干净的 v2 迁移，经未更新 chart 仍会在校验时失败；维护者要求在 `opentelemetry-helm-charts` 仓库单独开 issue，附上 chart/OBI 版本、相关 values、渲染配置与校验错误。在那落地前，实用变通是手工编辑渲染出的 configmap、在校验前去掉那个仅 v1 键。

**Node.js 哨兵成本归属（延续）。** Node.js 成本线继续，维护者确认了该读法：客户端 span 父化来自 fd 对 map，每次回调的 `async_hooks` 哨兵让 trace 上下文 map 与活动请求对齐——第三个消费者是外部 trace/profile 关联，所以只按"手动 span + 日志富化"来 gate 哨兵，可能悄悄破坏那个集成。方向是把每次回调的上下文刷新与 fd 对关联解耦、按消费方 gate；一个在途 OBI 拉取请求正针对哨兵性能与代理卸载路径。

**托管智能体 harness 与可观测性锁定（延续）。** 关于托管 agent harness：当循环跑在托管服务里，内部模型/工具 span 树只有在 harness 导出时才存在；一个不暴露 tracing 配置或外部 trace 导出器的公开 beta，意味着通用 HTTP 客户端插桩只能重建出站交叉。
