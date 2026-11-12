# 为什么 FUSE 文件系统运行在 PID 命名空间内时，FUSE 守护进程对 OBI trace 上下文 map 的查询会查不到？

因为查询的两端落在两个不同的 PID 命名空间里。内核把 `fuse_in_header.pid` 填成调用线程的线程 id，并按挂载所在 PID 命名空间重新换算；而 OBI 给它的 `traces_ctx_v1` trace 上下文 map 打 key 用的是 `bpf_get_current_pid_tgid`，取的是该线程在 init PID 命名空间里的值。当 FUSE 守护进程（或调用方）所处的 PID 命名空间与 OBI 看到的命名空间不一致时，两个数字指向同一个线程却不相等，于是 FUSE 守护进程按请求头构造出来的 key 找不到条目，查询落空。这是第三个失败维度，与 OBI 是否开启填充、以及 OBI 把上下文关联到哪个线程都相互独立：10-14 与 10-16 两篇覆盖了前两个维度，而这一篇处理的是 PID 数字本身对不齐的情况。

## 机制

1. **FUSE 一侧（内核）。** 内核为每个请求填入 `fuse_in_header.pid`。`fuse_fill_creds` 把 `args->pid` 设为 `task_pid(current)` 经 `fc->pid_ns` 重新换算的值，而 `fc->pid_ns` 在挂载时取自挂载进程的活跃 PID 命名空间。因此请求头里的 pid 是把调用线程的线程 id 换算到挂载 PID 命名空间后的值。若挂载与调用方共享该命名空间，这个数字等于全局线程 id；若调用方在某个子 PID 命名空间里，数字会被重新换算，不再是 init 命名空间的值。
2. **OBI 一侧（trace 上下文 key）。** OBI 在 span 起点用 `bpf_get_current_pid_tgid` 给 `traces_ctx_v1` 打戳。这个 helper 读的是全局线程 id（`current->tgid` 与 `current->pid`），也就是 init PID 命名空间里的值，再把它打包进 `u64` key。OBI 自己的日志富化器以及任何外部读取方都以同样的方式给 map 打 key，所以无论各个工作负载跑在哪个命名空间里，map 的 key 都落在 init PID 命名空间。
3. **错位。** FUSE 守护进程取出 `fuse_in_header.pid`，解析调用方的线程组 id，再打包成 `(tgid << 32) | tid` 作为在 `traces_ctx_v1` 里查询的 key。但这个 `(tgid, tid)` 对是相对挂载 PID 命名空间表达的，而 OBI 写入的每个条目都是按 init PID 命名空间打的 key。只要两个命名空间不一致，打包出来的 key 就与 OBI 的 key 对不上，查询返回空——尽管 OBI 明明已经为那个线程打上了 trace 上下文。

## 验证与排障路径

1. **确认两个命名空间确实不同。** 把 `fuse_in_header.pid` 与同一线程在根 PID 命名空间里看到的 id 相对照（例如从主机读该线程的全局 pid，或把挂载 PID 命名空间与根命名空间相对照）。若两者不同，FUSE 请求头 pid 与 OBI 的 key 就落在不同命名空间，对这个 key 的查询注定落空。
2. **确认 OBI 确实有该全局线程的条目。** 用 init 命名空间的打包 key 查 `traces_ctx_v1`。命中即说明 OBI 已为该线程打了上下文；再结合按重新换算 key 查询落空，即可把原因锁定为命名空间错位，而非 map 为空。
3. **在查询前对齐命名空间。** 让 FUSE 守护进程以与 OBI 打 key 一致的 PID 命名空间（init 命名空间，或 OBI 所解析到的命名空间）运行，或在构造 key 之前解析该 FUSE 操作对应的、主机可见的线程 id。只有当 FUSE 报出的数字与 OBI 的 key 落在同一个 PID 命名空间时，查询才会命中。

## 局限

- 这种错位是静默的。命名空间不一致时两侧都不报错；查询只是返回空条目，因此守护进程看到的一次落空，与 map 为空、或 OBI 从未给该线程打过上下文这两种情形在外观上完全一致。
- 它与 OBI 填充无关。即便 OBI 已充分填充 `traces_ctx_v1`、且把上下文关联到了正确的线程，重新换算后的 FUSE 请求头 pid 依然构造出一个 OBI 从未写入的 key。
- 内核不会替 FUSE 守护进程桥接这两个命名空间。请求头只携带重新换算后的 pid；`fuse_in_header` 里没有任何字段携带 init 命名空间的线程 id，因此若守护进程需要按 OBI 的方式打 key，就必须带外解析。
- libfuse 记录了退化情形：当 FUSE 文件系统运行在某个 PID 命名空间里、却从该命名空间之外被访问时，没有可报告的 pid，字段为零，于是查询在一个退化 key 上同样落空。

## References

- [kernel fs/fuse/req.c](https://github.com/torvalds/linux/blob/master/fs/fuse/req.c) - `fuse_fill_creds` 把 `args->pid` 设为调用线程的 `task_pid(current)` 经 `fc->pid_ns` 重新换算到挂载 PID 命名空间后的值。
- [kernel fs/fuse/inode.c](https://github.com/torvalds/linux/blob/master/fs/fuse/inode.c) - `fuse_conn_init` 把 `fc->pid_ns` 取自挂载进程的活跃 PID 命名空间。
- [kernel include/uapi/linux/fuse.h](https://github.com/torvalds/linux/blob/master/include/uapi/linux/fuse.h) - FUSE 守护进程读取的 `fuse_in_header.pid` 字段。
- [kernel kernel/bpf/helpers.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/helpers.c) - `bpf_get_current_pid_tgid` 读全局线程 id；命名空间感知变体 `bpf_get_ns_current_pid_tgid` 读的是逐命名空间的值。
- [libfuse include/fuse_lowlevel.h](https://github.com/libfuse/libfuse/blob/master/include/fuse_lowlevel.h) - 注明当 FUSE 文件系统运行在 PID 命名空间内却从外部访问时，报告的客户端 pid 可能为零。
- [OBI bpf/shared/obi_ctx.h](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/bpf/shared/obi_ctx.h) - 固定的 `traces_ctx_v1` map、其 `u64` key 与 `trace_id` / `span_id` 值。
- [OBI trace 上下文 map（10-14 条目）](/zh/ebpf-qa/2026-10-14-fuse-otel-trace-propagation-via-obi-trace-map/) - FUSE 守护进程如何读取按调用线程作为 key 的 OBI 固定 trace 上下文 map，以及本篇新增的两个失败维度。

## 当日社区讨论

本次巡检的两条 opt-in 归档频道（两条 CNCF OpenTelemetry 插桩频道）共产出十一条消息。窗口相对前一天没有前移，因此其内的实质线程与已经回答过的相同：FUSE 与 OBI trace 传播线程即 10-14 已发布条目，Cilium eBPF tracepoint 线程即 10-13 已发布条目。归档里另有一位维护者对其 OpenTelemetry 语义约定（统一 agent 间交互模型）pull request 的复审与催审备注（是评审状态更新，非问题），以及一条维护者提醒：该方案理论上仅当设置 trace 上下文的线程与执行 FUSE 工作的线程相同时才可行。由于这条提醒指向的是 10-16 条目已覆盖的线程亲和维度，本篇转而处理同一线程所铺陈、却尚无任何条目回答的相邻维度：FUSE 请求头 pid 与 OBI 的 key 之间的 PID 命名空间错位，会怎样在填充与线程关联都正确的情况下仍让查询落空。

频道覆盖情况：本次两条 opt-in 归档供出了十一条消息。仅可见浏览器的信源 - eunomia-bpf 与 sched-ext 两个 Discord 服务器、bpf 邮件列表、以及 r/eBPF - 本次因没有可用浏览器会话而未能审阅，故标记为未覆盖而非静默。
