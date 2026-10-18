# FUSE 守护进程对 OBI trace 上下文 map 的查询返回了一个条目时，这个条目一定是当前请求的 trace 上下文吗？它什么时候会过期？

不一定。命中不代表这个条目就是你即将处理的那个 FUSE 请求的 trace 上下文。OBI 固定的 trace 上下文 map traces_ctx_v1 按 OS 线程存的是 OBI 此刻在该线程上正在跟踪的最内层 span 的 trace 上下文。OBI 在 span 开始时写入该线程，在 span 结束时清掉或把条目回滚，因此你在调用线程上读到的值只是 OBI 最后写在那里的、仍然活跃的 span。它完全可能是复用线程上上一个请求的 span，也可能是刚结束的 client span 回滚上去的父 server span。命中的含义是"这个线程现在处于某个活跃的 OBI span 里"，而不是"这个请求就是那个 span"。守护进程必须把读到的 trace_id / span_id 当作该线程当前 OBI span 的上下文，而不能假设它就是本次 FUSE 请求的 span。

## 机制

1. **map 规格（对外接口）。** OBI 把 traces_ctx_v1 定义为 BPF_MAP_TYPE_LRU_HASH，键是原样的 u64 bpf_get_current_pid_tgid，按 (tgid << 32) | tid 打包，值是存放 trace_id 和 span_id 的 obi_ctx_info_t。它按名字 pin（LIBBPF_PIN_BY_NAME），max_entries 为 1 << 14。这个规格是 OpenTelemetry 的 OTEP 的一部分，其他组件也会读它，OBI 不会为你的线程预留槽位。

2. **span 开始时写入。** OBI 的 server_or_client_trace 在 server 或 client span 开始时用 obi_ctx__set 写入该线程的条目，存下那个 span 的 trace_id 和 span_id。这就是该线程的条目对这个 span 生效的时刻。

3. **虚拟线程载体被有意跳过。** 当请求由 Java 虚拟线程处理时，OBI 把内部的 server_traces map 按虚拟线程的逻辑 id 建键，但刻意不写一条按载体线程建键的 traces_ctx_v1 条目。源码里写明了原因：按载体建键的条目会把虚拟线程的上下文归因到接下来跑在这个载体上的任何东西。所以在虚拟线程载体上，固定的 map 要么查不到、要么是别的 span，命中不会解析到那个虚拟线程的 span。

4. **server span 结束会删除条目。** server span 结束时，OBI 调用 delete_server_trace，用 obi_ctx__del 删掉该线程的 traces_ctx_v1 条目（键是 Java 虚拟线程 TID 的除外）。此后该线程不再有自己专属的条目。

5. **client span 结束会回滚到父 span。** 当一个 client span 在 server span 内结束时，OBI 不删除条目，而是调用 obi_ctx__restore_server：从 server_traces 里查父 span，把线程条目改写回父的 trace_id 和 span_id。回滚是有守卫的：只有当条目仍指向刚结束的 client span（一次 span_id 相等检查）且父 span 存在于 server_traces 时才会执行；任一条件不满足，条目就保持原样，可能仍指向旧的 client span。

## 验证与排障路径

1. **把命中值与已知的请求对比。** 如果守护进程通过带外方式（调用方的元数据或旁路通道）得出了本次请求应有的 trace_id，把它和 map 里读到的 trace_id 对比。不一致说明这个条目是该线程上另一个仍然开着的 span，而不是本次请求。

2. **把读取时刻与 span 生命周期对齐。** 条目只在该线程的 OBI span 打开期间存在。如果你在调用方请求 span 结束之后才读，那么读到的（如果有）是回滚后的父 server span 或什么都没有，绝不会是请求自己的 span。

3. **确认线程模型。** 如果调用方是 Java 虚拟线程，那么按设计载体上的命中不会解析到虚拟线程的 span。读 map 之前先带外地把虚拟线程的逻辑 id 关联起来。

4. **把 LRU 淘汰当作额外的查不到轴。** max_entries 是 1 << 14，OBI 不会为你的线程 pin 槽位。在活跃的被 stamp 线程数很高时，LRU 列表会淘汰最久未用的条目，一个活跃线程照样会查不到。所以查不到不等于 OBI 从未为该线程写过。

## 局限

- 命中不等于"当前请求"。它是 OBI 在该 OS 线程上正在跟踪的最内层活跃 span，可能属于复用线程上上一个请求，也可能是父 server span。
- 条目是瞬时的。OBI 在 span 开始时写、在 server span 结束时删、在 client span 结束时回滚到父。必须在 span 的窗口内读，否则读到的是别的 span 或什么都没有。
- 回滚是有条件的。client span 结束时的回滚只有在条目仍指向刚结束的 client span 且父 span 存在时才触发；如果中间有过别的写入，过期的值会留下来，map 可能一直持有一个与线程当前工作不再匹配的 span。
- 虚拟线程载体按设计没有条目，LRU 压力也可能淘汰活跃线程的条目，因此命中与查不到各自都只是参考性的；只有知道 OBI 此刻在该线程上跟踪什么，它们才有意义。

## References

- [OBI bpf/shared/obi_ctx.h](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/bpf/shared/obi_ctx.h) - 固定的 traces_ctx_v1 map：BPF_MAP_TYPE_LRU_HASH、u64 键、obi_ctx_info_t 值（trace_id / span_id）、max_entries 为 1 << 14、按名字 pin；以及 obi_ctx__get / obi_ctx__set / obi_ctx__del 三个 helper。
- [OBI bpf/common/trace_lifecycle.h](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/bpf/common/trace_lifecycle.h) - server_or_client_trace 在 span 开始时写入线程条目并跳过虚拟线程载体；delete_server_trace 在 server span 结束时删除；obi_ctx__restore_server 在 client span 结束时回滚到父 server span，带 span_id 检查守卫。
- [kernel Documentation/bpf/map_hash.rst](https://github.com/torvalds/linux/blob/master/Documentation/bpf/map_hash.rst) - BPF_MAP_TYPE_LRU_HASH 在表满时自动淘汰最久未用的条目。
- [OBI trace 上下文读路径（10-14 篇）](/zh/ebpf-qa/2026-10-14-fuse-otel-trace-propagation-via-obi-trace-map/) - FUSE 守护进程如何读取固定的 map，以及前两个查不到轴（population 与 runtime 亲和性）。
- [PID 命名空间键不匹配（10-17 篇）](/zh/ebpf-qa/2026-10-17-fuse-trace-context-map-lookup-misses-in-pid-namespace/) - 本条补充的第三个、基于命名空间的查不到轴。

## 当日社区讨论

两个 opt-in 归档频道（两个 CNCF OpenTelemetry instrumentation 频道）本次运行产出了 11 条消息。24 小时窗口没有从 10-16 与 10-17 的运行前进，所以有实质内容的线程就是已经回答过的那几个：FUSE 与 OBI trace 传播线程是 10-14 已发布的条目及其 10-16、10-17 的续篇，Cilium eBPF tracepoint 线程是 10-13 已发布的条目。归档里还有一条 maintainer 对 OpenTelemetry 语义约定 pull request 的 review 状态更新（是 review 更新，不是问题），以及一条关于线程亲和性的 maintainer 备注（即 10-16 的轴）。由于窗口冻结、没有新问题进入，本篇取的是同一个 FUSE 与 OBI 线程的续篇轴：不再讨论查不到的原因，而是命中在 map 上到底意味着什么、它什么时候会过期，答案基于 OBI 源码与内核 LRU map 文档。

本次运行的频道覆盖：两个 opt-in 归档提供了 11 条消息。仅可见浏览器可访问的来源 - eunomia-bpf 与 sched-ext Discord 服务器、bpf 邮件列表、r/eBPF - 本次运行没有可用的可见浏览器会话，无法审阅，因此标记为未覆盖而非安静。
