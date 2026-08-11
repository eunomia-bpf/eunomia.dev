# 为什么 `scxctl` 接受了调度器切换命令，服务却仍未按预期启动？

**简短回答：**要把调度器参数明确绑定到 `--args`，并把 CLI 退出状态与内核实际状态分开验证。如果调度器参数本身以 `-` 开头，应使用 `=`：

```console
$ scxctl switch --sched lavd --args="--autopower"
```

省略 `=` 时，选项解析器可能把 `--autopower` 当成 `scxctl` 的另一个选项，而不是 `--args` 的值。引号控制 shell 分词；等号明确以短横线开头的值归哪个选项所有。多个参数应以本机 `scxctl switch --help` 为准；已有版本使用过单个逗号分隔字符串，例如 `-a="-v,--performance"`。

请求成功也不是健康检查。还要确认 loader、查询 `scxctl` 当前状态，并检查内核 `sched_ext` 状态。`scxctl` 已迁入 `sched-ext/scx` 项目，接口仍在演进，版本差异不能忽略。

## 命令会经过三层解析器

shell 先把文本转换成 `argv`；`scxctl` 再解析自身选项并封装调度器参数；loader 启动调度器后，调度器自己的解析器才解释这些参数。shell 通常会在启动程序前去掉引号，因此下面两种形式可能不同：

```console
# 无歧义的选项和值绑定
$ scxctl switch --sched lavd --args="--autopower"

# 短横线 token 可能被解释为另一个选项
$ scxctl switch --sched lavd --args "--autopower"
```

不要把 shell 引号机械地复制到 systemd `ExecStart=`。systemd 有自己的切分规则；除非显式调用 shell，否则不会经过 shell。应先证明交互命令可用，再把相同参数边界写入 unit，并用 `systemctl cat` 检查实际定义。

## 只切换一次，然后逐层验证

从本机接口开始：

```console
$ scxctl --version
$ scxctl switch --help
$ scxctl list
$ sudo scxctl switch --sched lavd --args="--autopower"
```

再验证控制面和内核：

```console
$ systemctl status scx_loader.service --no-pager
$ scxctl get
$ cat /sys/kernel/sched_ext/state
$ cat /sys/kernel/sched_ext/root/ops
$ journalctl -u scx_loader.service -b --no-pager
```

内核文档把 `state` 定义为 `disabled`、`enabled` 或 `disabling`，`root/ops` 报告已注册调度器名称。CLI 显示成功，但 loader 失败、状态不是 `enabled`，或 `root/ops` 不匹配，都不能算部署成功。发行版 unit 名称可能不同，应查询已安装 unit；还要排查绕过 loader 直接启动的调度器，因为两个控制器可能互相替换。

按最早失败边界分类：

- unknown option 或 missing value：命令语法；
- 请求被接受但 loader 失败：检查 journal 或不受支持的调度器参数；
- 调度器启动后退出：检查内核支持、BTF、verifier 或运行时错误；
- 内核已启用但 `root/ops` 是另一个值：存在竞争控制器。

记录 CLI、调度器软件包和内核的精确版本。已弃用的独立仓库已指向官方项目，其示例可以解释参数绑定，却不能替代本机 `--help`。排障时先停掉自动重启循环，只做一次受控切换并保留首个错误，定位后再恢复重启策略。

## 参考资料

- [Linux 内核文档：`sched_ext` 状态与调度器名称](https://docs.kernel.org/scheduler/sched-ext.html)
- [官方 `sched-ext/scx` 仓库](https://github.com/sched-ext/scx)
- [独立 `scxctl` README 与参数示例（已弃用）](https://github.com/frap129/scxctl)
- [systemd 手册：`ExecStart=` 命令行](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html#Command%20lines)
- [systemd 手册：服务状态](https://www.freedesktop.org/software/systemd/man/latest/systemctl.html#status%20PATTERN%E2%80%A6)
- [BPF 邮件列表：降低 hash map 元素内存占用](https://lore.kernel.org/bpf/20260805223516.1495988-1-tjmercier@google.com/T/#t)
- [BPF 邮件列表：ARM64 arena kfunc 与 `struct_ops` 参数](https://lore.kernel.org/bpf/20260810190922.3408757-1-puranjay@kernel.org/T/#t)

## 今日社区讨论

今天通过普通可见界面检查了全部 6 个获准社区、共 15 个 allowlist 频道或公开页面；所有目标均可访问。24 小时窗口内出现了真实的调度器控制排障讨论，因此不需要 7 天回退。以下内容已删除姓名、账号、雇主、频道身份、消息链接、精确时间、私有拓扑、原始日志与可回搜措辞；没有保留原始 transcript。

### 调度器控制需要端到端检查

入选讨论涉及一次没有让服务按预期运行的调度器切换。一个以短横线开头的调度器参数没有与 CLI 选项形成无歧义绑定；修正边界后调度器能够运行。更普遍的结论是，控制 CLI 的确认信息只是一处检查点，还必须验证 loader 和内核实际启用的调度器。

一个更早、尚无回答的内核配置精简问题不在当日或回退窗口内，因此没有使用。项目通用区域主要是新成员活动、自动构建通知或安静的专项频道。

### 插桩讨论聚焦所有权

GenAI 可观测性社区讨论了特定 agent 集成应由 SDK 厂商还是 OpenTelemetry 社区负责。贡献者协调计划中的 TypeScript 插桩，同时等待另一个 SDK 是否原生支持语义约定，目标是在不制造重复 span 的前提下补齐覆盖。eBPF 插桩社区当天没有新的实质活动；近期讨论仍在区分运行时 eBPF 插桩和编译期语言插桩。

### 上游工作强调接口正确性

公开 BPF 归档涉及 hash map 布局精简、`BTF.ext` 边界加固、AF_XDP 元数据 ABI 对齐、重叠 RCU 保护、ARM64 arena 参数和聚合返回值。公开论坛是 verifier 说明和较早的性能分析文章，而不是新排障报告。通用 eBPF 聊天有项目公告，以及对前一天 socket-map 主题的后续回复，但没有更强的未解决问题。当天的共同主题是接口精度：每一层都必须对值归谁所有、如何验证成功状态达成一致。
