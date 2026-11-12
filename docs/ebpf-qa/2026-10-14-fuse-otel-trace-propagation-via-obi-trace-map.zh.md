# FUSE 用户态文件系统守护进程如何通过按 pid 和 tid 读取 OBI trace 上下文 map 来继承 OpenTelemetry trace propagation？

可行，但前提是两个独立的条件同时成立。内核会为每个 FUSE 请求填充 `fuse_in_header.pid`，所以 FUSE 用户态守护进程总能知道是哪个调用线程发起了请求。OpenTelemetry 的 eBPF 插桩器（OBI）维护着一个已 pin 的 map，`traces_ctx_v1`，保存当前正在服务请求的线程的 trace id 与 span id，以该线程为键。FUSE 守护进程可以读这个 pin 来继承 W3C trace 上下文。但这个 map 只在 OBI 已经为某个线程捕获到 trace 上下文、且 population 打开时才会被写入；未被追踪的协议或不支持的运行时会使其为空，FUSE 侧的查询就会 miss。而请求头里携带的是线程 id，OBI 却按进程加线程对做键，所以守护进程必须先解析出线程组才能构造出键。

## 机制

边界在"内核给守护进程的东西"与"OBI 已经存好的东西"之间：

1. **FUSE 侧（内核）。** FUSE 守护进程从 `/dev/fuse` 读到的每个请求都以 `fuse_in_header` 开头，其 `pid` 字段由内核在 `fuse_args_to_req` 里从 `fuse_args.pid` 填入，而后者在 `fs/fuse/req.c` 中被设为映射到挂载点 pid namespace 后的 `task_pid(current)`。也就是说，守护进程拿到的是发起线程的线程 id（在挂载点的 pid namespace 内），每个请求一份。它是 TID 而不是进程 id：flock 路径刻意用的是 `task_tgid`，这个对照正好说明通用请求头的 pid 是线程粒度的。
2. **OBI 侧（trace 上下文 map）。** OBI 会 pin 一个名为 `traces_ctx_v1` 的 `BPF_MAP_TYPE_LRU_HASH`，默认按名字 pin 到 `/sys/fs/bpf/`，键是 `u64`，值只有两个字段：`trace_id`（16 字节）与 `span_id`（8 字节），即 W3C Trace Context 的二元组。OBI 在 server span 开始的那一刻写入一条条目，键取自 `bpf_get_current_pid_tgid()`，把线程组与线程 id 打包成一个 64 位值。同一个 map 也喂给 OBI 自己的日志富化器，并且是外部 eBPF 程序或 FUSE 守护进程可以读取的公开表面。
3. **胶水层（守护进程）。** 要继承上下文，守护进程需要：(a) 读 `fuse_in_header.pid` 得到调用方 TID；(b) 通过读调用方 `/proc` 状态页的 `Tgid` 字段把 TID 解析成线程组 id；(c) 构造 8 字节键 `(tgid << 32) | tid`；(d) 对 `traces_ctx_v1` 的 pin 做 `bpf_map_lookup_elem`，取回 trace id 与 span id，再据此播种或延伸它自己所创建的 FUSE 操作 span。

## 验证与调试路径

1. **确认守护进程能看到调用者。** 从 `/dev/fuse` 读一个请求头，打印 `fuse_in_header.pid`，并与执行写入的那个进程的线程 id 对照；对多线程调用者，它应当跟到具体的工作线程，而不只是进程。
2. **确认 OBI 真的填充了 map。** 打开 OBI 的 `populate_trace_context` 选项，然后读 `traces_ctx_v1` 的 pin（例如打开 `/sys/fs/bpf/traces_ctx_v1`，对一个已知键执行 `bpf_map_lookup_elem`）。命中会返回 `trace_id` 与 `span_id`；miss 说明该线程没有捕获到任何上下文。
3. **按 OBI 的方式构造键。** 查询前把解析出的线程组与线程 id 打包成 `u64` 键；只带 TID 的键匹配不上 OBI 的 `bpf_get_current_pid_tgid()` 键。
4. **隔离两种失败模式。** 在 population 已打开的前提下查询 miss，指向 OBI 侧（该线程的运行时或协议没有产生 W3C 上下文，或其语言未被插桩）；命中但值陈旧，指向线程亲和性一侧（FUSE 工作跑在了别的线程上，而不是 OBI 打过标记的那个线程）。

## 局限

- `traces_ctx_v1` 是 opt-in。OBI 默认关闭它，直到有读者出现；不打开 `populate_trace_context`，外部 FUSE 读者就无东西可读。
- OBI 必须先有 trace 上下文。这个 map 里存的是 OBI 已经在该线程上捕获的上下文。未被追踪的协议，或 OBI 不插桩的运行时（维护者指出 C++ 不在覆盖之列，支持的是 Go、Java、Node.js，以及带 asyncio 的 Python），都会使条目缺失，FUSE 查询 miss，也就无东西可继承。
- 键是线程组加线程对，而请求头只带线程。需要先由 TID 解析出线程组，否则构造不出键。
- 线程亲和性很关键。OBI 把上下文打在服务请求的那个线程上。若 FUSE 工作转到别的线程，那个线程的条目就是空的；把关联放宽到整个线程组虽然可行，但当一个进程同时服务多个请求时会有歧义。
- 该 map 是 `BPF_MAP_TYPE_LRU_HASH`，高负载下忙碌线程的条目可能被逐出；应在请求到达后及时读取。

## 参考

- [kernel fs/fuse/req.c](https://github.com/torvalds/linux/blob/master/fs/fuse/req.c) —— `fuse_args.pid` 被设为调用线程的 TID，即映射到挂载点 pid namespace 后的 `task_pid(current)`。
- [kernel include/uapi/linux/fuse.h](https://github.com/torvalds/linux/blob/master/include/uapi/linux/fuse.h) —— FUSE 守护进程读取的 `fuse_in_header.pid` 字段。
- [OBI bpf/shared/obi_ctx.h](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/bpf/shared/obi_ctx.h) —— pin 的 `traces_ctx_v1` map，其 `u64` 键与 `trace_id` / `span_id` 值，以及拥有该规格的 OTEP。
- [OpenTelemetry OTEP 4855](https://github.com/open-telemetry/opentelemetry-specification/pull/4855) —— OBI 实现的外部 trace 上下文 map 规格。
- [W3C Trace Context](https://www.w3.org/TR/trace-context/) —— map 所携带的 `traceparent` 布局（`trace_id` / `span_id`）。

## 当日社区讨论

本 run 审查的两个 opt-in 存档渠道（两个 CNCF OpenTelemetry 插桩渠道）共产出十一条消息。本页回答的正是其中一条：一位实践者问，把 OpenTelemetry trace 传播加到 FUSE 程序上有多可行——他看到可以用 OBI 把 trace 信息导出到它的 trace 上下文 map，再用 pid 和 tid 在 FUSE 守护进程中收集。存档里还带了维护者的后续：FUSE 作为用户态文件系统可以读到调用线程；线程组 id 可以从调用方的状态页里通过 `/proc` 拿到；Go 与带 asyncio 的 Python 受支持而 C++ 未覆盖；且 OBI 的 map 只有在某线程上已经存在 trace 上下文时才会被填充，并且设置上下文的那个线程（例如 server 线程）还得是实际执行 FUSE 工作的那个线程。窗口里另外两个话题：一个是已发布条目回答过的讨论的交叉转发，另一个是某个评审状态的更新而非问题，因此本 run 选取了那个尚未在本站回答过、且非重复的 FUSE 与 OBI trace 传播问题。

本 run 的渠道覆盖：两个 opt-in 存档渠道共提供十一条消息。可见浏览器专属的渠道——eunomia-bpf 与 sched-ext 的 Discord 服务器、bpf 邮件列表、以及 r/eBPF——本 run 没有可用的可见浏览器会话，因此标记为"未覆盖"，而非"安静"。
