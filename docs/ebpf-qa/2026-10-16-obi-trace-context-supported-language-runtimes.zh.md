# OpenTelemetry eBPF 插桩如何判定它能为哪些语言运行时打上 trace 上下文，又如何验证 FUSE 守护进程线程确实继承了上下文？

FUSE 用户态守护进程只有当发起 FUSE 操作的那个线程，恰好是 OpenTelemetry 的 eBPF 插桩器（OBI）真正为其打上 trace 上下文的线程时，才能继承 OpenTelemetry 的 trace 上下文；而 OBI 只会为一组固定的、带版本门槛的运行时列表打上上下文，并不是对所有语言都适用。Go、Node.js、Java、Python、Ruby 各自有一条具名的关联机制，C++ 则没有。因此，由 C++ 客户端驱动的 FUSE 程序，正是常见 FUSE 语言集合中唯一拿不到语言级上下文交接的那种；除非 C++ 代码恰好在 OBI 看到服务请求的同一线程上发起 FUSE I/O，否则守护进程对该线程的查询就会落空。验证继承的方式是：打开 OBI 的 trace 上下文 map，读取按 FUSE 请求头里取到的调用线程 id 作为 key 的条目，并确认其中的 trace id 与 span id 和入站上下文一致；若开启填充后仍查询落空，说明问题在 OBI 一侧而非守护进程。

## 机制

边界在于 OBI 能为之关联 trace 上下文的运行时，与实际发起 FUSE 请求的线程之间：

1. **运行时支持（OBI 兼容性矩阵）。** OBI 的上下文传播是一份固定清单。网络层的 HTTP/1 与 gRPC 传播跨语言通用，但按线程的上下文关联是语言特定的：Go 库级上下文传播用于 Go 1.18+（最多六层嵌套的 goroutine）；Node.js 的 async hooks 用于 Node.js 8.0+（自定义的 SIGUSR1 处理可能干扰）；Java 线程池交接用于 JDK 8+，经由常见的任务 API；Python 的 asyncio 用于 3.9+ 且需 uvloop 事件循环；以及由 Puma 5.0+ 服务的 Ruby。C++ 未出现在矩阵中，因此纯 C++ 客户端没有任何语言特定的关联机制。
2. **线程关联。** 即便运行时受支持，OBI 也要用该运行时的交接机制把出站工作关联到入站请求；只有当这条具体匹配不可用时，OBI 才回退到与语言无关的信号：同一 OS 线程同时处理入站请求与出站工作、OBI 能关联回请求线程的子线程、或请求线程在工作由另一个工作线程完成之前先做了 socket 建立或连接存活检查。OBI 不会去重建任意的应用队列或框架上下文对象。
3. **FUSE 边界。** FUSE 守护进程读取 `fuse_in_header.pid`，而内核填入的是调用线程在挂载 pid 命名空间内的线程 id。要让守护进程继承上下文，发起 FUSE 操作的那个线程必须要么是 OBI 已将其关联到某个活跃 trace 上下文的线程，要么是经上述某条交接可触达的线程。若 C++ 客户端把 FUSE I/O 分发到与接收请求线程不同的 worker 线程，OBI 没有语言特定机制去跟随这次交接，于是 OBI 按进程与线程对作为 key 的按线程上下文字段，在发起 FUSE 的那个线程上为空，守护进程的查询便落空。

## 验证与排障路径

1. **确认客户端运行时在矩阵内。** 把客户端的语言与版本对照清单：Go 1.18+、Node.js 8.0+、JDK 8+、Python 3.9+ 且 uvloop、Ruby 由 Puma 5.0+ 服务。若客户端是纯 C++，到此为止：没有语言特定的关联，只剩与语言无关的线程信号。
2. **确认发起 FUSE 的线程是请求线程，或是可达的交接。** 对受支持的运行时，验证调用 FUSE 操作的线程是正在服务入站请求的同一线程，或是受支持交接深度内的 goroutine、async 任务或 JDK 任务。OBI 无法关联回请求线程的独立 worker 池会落空。
3. **打开 trace 上下文 map 并读取条目。** 启用 OBI 的 `populate_trace_context` 选项，然后对从 `fuse_in_header.pid` 取到的线程 id，查询 OBI trace 上下文 map 的条目。命中且 trace id 与 span id 与入站上下文一致，即确认继承；开启填充后仍落空，则指向 OBI 一侧。
4. **隔离两种失败模式。** 开启填充且运行时受支持、线程也正确却仍落空，指向关联步骤（一次 OBI 无法观察到的交接）；若填充关闭，或客户端是纯 C++，则指向支持边界（那个线程从未被打过语言特定的上下文）。

## 局限

- 纯 C++ 客户端不在 OBI 的上下文传播兼容性矩阵内，因此由 C++ 应用驱动的 FUSE 程序拿不到语言特定的 trace 上下文交接；只能依赖与语言无关的线程信号，而一旦 FUSE I/O 运行在 OBI 无法关联回请求线程的线程上，这些信号也会落空。
- 支持清单带版本门槛（Go 1.18+、Node.js 8.0+、JDK 8+、Python 3.9+ 且 uvloop、Ruby 由 Puma 5.0+ 服务）；其他版本可能也能用，但不在文档覆盖范围内。
- 即便运行时受支持，交接深度也有界（最多六层嵌套 goroutine，以及有界的 JDK 任务交接数）；落在受支持机制之外的响应式或事件循环结构仍可能落空。
- 与语言无关的回退是刻意有界的：OBI 不重建任意的应用队列、用户态调度器或框架上下文对象，因此一个把 FUSE 工作移离请求线程、且没有可观察交接的线程池会开启新 trace，而不是继承。

## References

- [OBI distributed-traces 兼容性矩阵](https://opentelemetry.io/docs/zero-code/obi/distributed-traces) - OBI 上下文传播支持的运行时与版本（Go 1.18+、Node.js 8.0+、JDK 8+、Python 3.9+ 且 uvloop、Ruby Puma 5.0+；C++ 未列出）。
- [OBI trace context association](https://opentelemetry.io/docs/zero-code/obi/context-propagation) - OBI 把出站工作关联到入站请求所用到的语言特定与语言无关机制，以及会落空的情形。
- [OBI trace 上下文 map（10-14 条目）](/zh/ebpf-qa/2026-10-14-fuse-otel-trace-propagation-via-obi-trace-map/) - FUSE 守护进程如何读取按调用线程作为 key 的 OBI 固定 trace 上下文 map。
- [kernel include/uapi/linux/fuse.h](https://github.com/torvalds/linux/blob/master/include/uapi/linux/fuse.h) - 携带调用线程 id 给 FUSE 守护进程的 `fuse_in_header.pid` 字段。
- [W3C Trace Context](https://www.w3.org/TR/trace-context/) - OBI 传播的 `traceparent` 布局。

## Community discussion today

本次巡检的两条 opt-in 归档频道（两条 CNCF OpenTelemetry 插桩频道）共产出十一条消息。本页回答的线索：一位实践者询问，借助按 pid 和 tid 读取 OBI 的 trace 上下文 map，为 FUSE 程序加上 OpenTelemetry trace 传播在多大程度上可行，特别是它对用 Go、Python、C++ 编写的常见 FUSE 守护进程是否都成立，并指出 Go 与带 asyncio 的 Python 看似受支持，但 C++ 不确定。归档里还有一位维护者对其 OpenTelemetry 语义约定统一 agent 间交互模型的 pull request 的复审与催审（是复审请求，非问题），以及一位维护者的提醒：该方案在理论上只有当设置 trace 上下文的线程与执行 FUSE 工作的线程相同时才可行。本页关闭的那个未决问题：OBI 究竟能为哪些语言运行时打上 trace 上下文，以及如何验证发起 FUSE 的线程是其中之一、并确确实实继承了同一份上下文。

频道覆盖情况：本次两条 opt-in 归档供出了十一条消息。仅可见浏览器的信源 - eunomia-bpf 与 sched-ext 两个 Discord 服务器、bpf 邮件列表、以及 r/eBPF - 本次因没有可用浏览器会话而未能审阅，故标记为未覆盖而非静默。
