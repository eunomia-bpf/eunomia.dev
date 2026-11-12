# 为什么给 Node.js 服务启用 eBPF 追踪后，热路径的 p99 会翻三倍，而 eBPF 探针本身却很廉价？

**简短回答：** 开销几乎全部来自注入到进程内的 JavaScript 代理，而不是 eBPF 探针本身。OBI 这类基于 eBPF 的自动插桩工具，对 Node.js 的"零代码"支持并不是纯 eBPF 实现的：它通过 Node 检查器协议（`SIGUSR1` + `Runtime.evaluate`）把一个 JS 代理注入到应用里，而这个代理的 `async_hooks` `before` 钩子会在**每一个** JS 回调之前触发，用来刷新活动请求的 trace 上下文，并通过一个 `fs.accessSync` 哨兵路径把信号传给 eBPF 层。这些每次异步操作的成本落在事件循环里，而不是内核里，所以在每个请求都要走大量回调的热路径上被放大。eBPF 探针（uprobe + 读 fd 相关性的 eBPF map）本身很廉价；贵的是那个注入的 JS 代理及其钩子。

## 为什么 eBPF 探针不是主要成本

OBI 的文档把 Node.js 的支持明确描述为"使用 Node.js async hooks 在每个异步回调之前刷新活动请求上下文，并把出站 socket 与入站 socket 关联起来"。也就是说，OBI 对 Node.js 的 trace 上下文支持被分成两个截然不同的层：

1. **eBPF 层**：uprobe 打在 `uv_fs_access` 上，把注入代理发出的"哨兵"路径解码成 eBPF 事件。这部分非常轻——它只在一个特定的、受控的 uprobe 里读一段固定的 22 字节路径前缀，再写进一个小 map。
2. **进程内 JS 代理**（`fdextractor.js`）：通过检查器协议注入，它把 `net.Socket` 的原型方法包起来，并装一个 `async_hooks` 的 `before` 钩子。这个钩子每次 JS 回调前都会跑。

成本落在第 2 层，因为它在 JS 事件循环的热路径上执行。OBI 的代码把这个开销写得很直白：

```js
// fdextractor.js（节选）
if (TRACES_ENABLED) {
  // ALS store holds only incomingFd
  const als = new AsyncLocalStorage();

  net.Server.prototype.emit = function (event, ...args) { ... };

  net.Socket.prototype.write = function (data, ...rest) {
    const doWrite = () => orig.socketWrite.apply(this, [data, ...rest]);
    const store = als.getStore();
    if (store) {
      const outFd = this._handle && this._handle.fd;
      correlate(store.incomingFd, outFd, this);
    }
    return doWrite();
  };

  // 在每个异步回调之前向 BPF 层发信号，
  // 以便把当前请求的 trace 上下文刷新进 traces_ctx_v1。
  let ctxActive = false;
  orig.ctxHook = createHook({
    before() {
      const store = als.getStore();
      if (store && store.incomingFd != null && store.incomingFd >= 0) {
        ctxActive = true;
        try {
          fs.accessSync(`/dev/null/obi-ctx/${pad4(store.incomingFd)}`);
        } catch (_) {}
      } else if (ctxActive) {
        ctxActive = false;
        try {
          fs.accessSync('/dev/null/obi-noreqctx');
        } catch (_) {}
      }
    },
  });
  orig.ctxHook.enable();
}
```

`before` 钩子每次回调都触发；`fs.accessSync` 哨兵只在请求上下文内（或者在"有请求 → 无请求"的切换沿）才真正产生同步系统调用，而注释里特意强调"为了避免对每个非请求回调都做一次同步系统调用（这种回调可能非常多），只在切换时清理"。但钩子本身——它的事件循环成本、加上 socket 原型方法被包裹这件事——是每个请求都要付的固定开销，热路径会被放大。

## `fs.accessSync` 哨兵是怎么把 JS 信号送进 eBPF 的

`fs.accessSync` 在这里是安全的，因为它在 `async_hooks` 回调里跑：同步 fs 操作不会创建 `AsyncWrap` 对象，因此不会再次触发这个钩子（这正是 `fdextractor.js` 注释里强调的那条不变量）。哨兵路径被 eBPF 侧的 `uv_fs_access` uprobe 解码，OBI 的 `nodejs.c` 把它们分成几类：

```c
// bpf/generictracer/nodejs.c（节选）
SEC("uprobe/node:uv_fs_access")
int BPF_KPROBE_GUARDED(obi_uv_fs_access, void *loop, void *req, const char *path)
{
    // 注入代理（fdextractor.js, spanbridge.js）通过用一个假路径调用
    // uv_fs_access() 把信号传给 eBPF 层。有几种格式：
    //  1. fd 对关联（出站 -> 入站）: /dev/null/obi/<fd1><fd2>
    //  2. 异步上下文切换（before 钩子在每个 JS 回调前触发）:
    //     /dev/null/obi-ctx/<fd>
    //  3. 手动 span 结束（spanbridge.js）: /dev/null/obi-span/<json>
    //  4. 无请求上下文: /dev/null/obi-noreqctx
    ...
}
```

关键的 fd 对关联写进了一个独立的 eBPF map：

```c
// bpf/generictracer/nodejs.c:441
static __always_inline int handle_fd_correlation(char *buf, const u64 pid_tgid)
{
    ...
    const u64 key = (pid_tgid << 32) | fd2;
    bpf_map_update_elem(&nodejs_fd_map, &key, &fd1, BPF_ANY);
    return 0;
}
```

这个 `nodejs_fd_map`（一个以 `(pid_tgid, fd)` 为键、存入站 fd 的小 LRU 哈希）就是客户端 span 真正被父化的入口，而不是 `traces_ctx_v1`。

## `traces_ctx_v1` 到底是什么，以及为什么"关掉哨兵"不那么简单

`traces_ctx_v1` 是 OBI 维护的"当前请求 trace 上下文"map，它是**按名字 pin**（`LIBBPF_PIN_BY_NAME`）的，并且是一个 OTEP 合约的一部分（OTEP 4855，"correlating OBI traces to profiles"）：

```c
// bpf/shared/obi_ctx.h
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __type(key, u64);
    __type(value, obi_ctx_info_t);
    __uint(max_entries, 1 << 14);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} traces_ctx_v1 SEC(".maps");
```

它由 eBPF 侧在保存服务端 span 时写入，再由 `async_hooks` 的哨兵在每个 JS 回调前刷新。读它的消费者有多个：

- 手动 span 父化（`handle_node_span` 读 `obi_ctx__get`）；
- 日志富化（`logenricher.c` 读 `obi_ctx__get` 把 trace id 注进日志行）；
- Go 运行时的 goroutine 上下文交接（`go_runtime.c` 在找不到运行时任务时，从 `traces_ctx_v1` 兜底取当前请求上下文）；
- 外部 trace/profile 关联工具（因为 map 是 pin 的，外部工具可以通过 libbpf 读它）。

也就是说，这个 map 是一个**对外表面**（`LIBBPF_PIN_BY_NAME` + OTEP 合约）。这意味着如果只根据"是否启用了手动 span + 日志富化"来关掉哨兵，可能会悄悄破坏读这个 pin map 的外部关联集成。维护者的方向是：把"每个回调的上下文刷新"与"fd 对关联"解耦，对手动 span 和日志富化自动启用刷新，并给外部 profile 关联留一条显式的配置路径。

## 怎么实际定位和缓解

1. **先看事件循环，而不是先看 eBPF。** 在启用前后各 profile 一次事件循环：把 `before` 钩子 + `fs.accessSync` 哨兵的每回调成本、与 eBPF 探针的 uprobe 成本分别计时。p99 翻倍通常来自前者，而不是内核侧。
2. **确认哪些消费者真的需要 `traces_ctx_v1`。** 如果你既没开手动 span、也没开日志富化、也没有外部 trace/profile 关联，那么每次回调的上下文刷新对你就是纯成本——这是可以关掉的最直接开关。但如果外部关联工具在读那个 pin map，关掉哨兵会破坏它；这种情况下要走"解耦 + 显式配置"的路径，而不是直接关掉。
3. **按消费方来 gate，而不是全局关。** 对纯指标采集（只收 runtime metrics），OBI 注入的是同一个代理，但 trace 上下文机制（`net` 原型包裹 + `async_hooks` `before` 钩子）会被整个跳过——也就是说，指标专用的注入不会触发每次异步操作的成本。
4. **关注在途的性能工作。** 有一个改进 Node 哨兵性能的草稿 PR（#3357），以及"把每个回调的上下文刷新与 fd 对关联解耦、按消费方 gate"的方向。

## 决定性的边界

p99 翻倍的代价不在 eBPF 探针，而在注入的 JS 代理的每次异步操作成本——`async_hooks` `before` 钩子（维护 `traces_ctx_v1` 的哨兵）加上它发出的 `fs.accessSync` 信号，这些都在事件循环的热路径上。eBPF 侧的 fd 相关性 map 很廉价，真正决定 p99 的是那个在进程内、每次回调都要付的 JS 钩子成本。"关掉哨兵"不是免费的：它维护的 `traces_ctx_v1` 是一个 pin 的、按 OTEP 对外表面的 map，被多个消费者读；正确的杠杆是把刷新与 fd 对关联解耦、按消费方 gate，而不是全局关掉。

## 参考资料

- [OBI 文档：Trace context association in OBI（Node.js 用 async hooks 在每个异步回调前刷新活动请求上下文）](https://opentelemetry.io/docs/zero-code/obi/context-propagation/)
- [OBI 文档：Distributed traces with OBI（Node.js 异步钩子兼容性说明）](https://opentelemetry.io/docs/zero-code/obi/distributed-traces/)
- [OBI 文档：Trace-log correlation（日志富化读 trace 上下文）](https://opentelemetry.io/docs/zero-code/obi/trace-log-correlation/)
- [OBI 源码：`bpf/generictracer/nodejs.c`（`obi_uv_fs_access` uprobe 解码哨兵路径、`handle_fd_correlation` 写 `nodejs_fd_map`、`handle_async_switch` 刷新 `traces_ctx_v1`）](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/bpf/generictracer/nodejs.c)
- [OBI 源码：`pkg/internal/nodejs/fdextractor.js`（注入代理，`async_hooks` `before` 钩子 + `fs.accessSync` 哨兵）](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/internal/nodejs/fdextractor.js)
- [OBI 源码：`bpf/shared/obi_ctx.h`（`traces_ctx_v1`，`LIBBPF_PIN_BY_NAME`，OTEP 4855）](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/bpf/shared/obi_ctx.h)
- [OTEP 4855：correlating OBI traces to profiles](https://github.com/open-telemetry/opentelemetry-specification/pull/4855)
- [改进 Node 哨兵性能的草稿 PR #3357](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/pull/3357)
- [Node.js 文档：`async_hooks`（`createHook`、`before` 钩子；`async_hooks` 已被标记为实验性，性能上推荐 `AsyncLocalStorage`）](https://nodejs.org/api/async_hooks.html)

## 当日社区讨论

本监控窗口的技术密度很高。一个反复出现的主题是 Node.js 自动插桩的成本归属：某位使用者报告，给一个 Node 服务启用追踪后，热路径的 p99 翻了三倍，而同一批 pod 上其他路径几乎没动；profile 显示 eBPF 探针很廉价（约占进程 CPU 的 4%），真正吃掉事件循环 CPU（HTTP 服务上 25–55%、队列 worker 上 0.9%）的是注入代理的 `async_hooks` `before` 钩子加上它的 `fs.accessSync` 哨兵。维护者确认了客户端 span 父化走的是 fd 对 map，并指出 `traces_ctx_v1` 还被外部 trace/profile 关联读取，所以只按"手动 span + 日志富化"来 gate 哨兵可能会悄悄破坏那个集成；方向是把"每个回调的上下文刷新"与"fd 对关联"解耦、按消费方自动启用，并给 profile 关联留显式配置路径；另有一个改进 Node 哨兵性能的草稿 PR。

同一天还有一条关于 OBI 配置 v2 迁移的讨论：迁移命令目前是全有或全无的（`migrate` 承诺行为保持不变的转换，静默丢字段会产出"看起来合法但行为实质不同"的配置），维护者倾向于加一个显式的 `--allow-partial` / `--best-effort` 模式，把能迁移的字段迁成合法 v2、并明确报告被省略的字段，而不是把部分迁移做成默认；相关的 OBI Helm chart 对 v2 配置的支持缺陷已被单独开 issue 跟踪（注入的 `_helpers.tpl` 仍在往 v2 配置 schema 里塞 v1 字段）。还有一条关于 OBI 节点发现 bug（supervisor 包装启动的子进程有时无法被插桩）的简短跟进。
