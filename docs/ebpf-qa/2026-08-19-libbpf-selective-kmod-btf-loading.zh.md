# 为什么安装了许多内核模块的主机上，libbpf 加载 BPF 对象会变慢？

**简短回答：** libbpf 的 kernel-module BTF 路径是 lazy 的，但粒度很粗。它不会为每个 BPF object 都读取全部 module BTF；可是，一旦某个 object 需要的 type、kfunc 或 BTF attach target 无法从 `vmlinux` BTF 解析，当前 libbpf 就会遍历 kernel 中的 BTF object，并把找到的每个 module BTF 都 materialize。主机上启用 BTF 的 module 很多时，即使最终只需要一个 module，这些 userspace discovery 与 parsing 工作仍可能占据大部分启动时间。

把 tracing section 写成 `SEC("fentry/mymod:foo")` 目前不能省掉这笔开销。Module qualifier 只会缩小后续 target lookup 的范围，不会告诉 `load_module_btfs()` 应跳过哪些 BTF object。

一个 RFC patch series 提议在 `bpf_object_open_opts` 中加入显式 module-name allowlist。对于已经知道 dependency 的 caller，这是合理的接口形状；但截至本文写作时，它还不是已发布的 libbpf API。Production code 应先测量精确的 open/load boundary，保留现有 fallback，并等 selector 真正进入应用所携带的 libbpf 版本后再采用。

## 时间花在了哪里

Module BTF 并不是所有 kernel type 的第二份完整副本。Linux 可以把 module 编码成 split BTF：module 只保存自己特有的 type，共享 type 则引用 base `vmlinux` BTF。这样表示更紧凑，但 loader 仍需发现 module BTF，取得它的 file descriptor 与 metadata，用 base BTF 解析它，并保留足够 state 供 relocation 或 target lookup 使用。

当前 `load_module_btfs()` 大致执行以下步骤：

1. 通过 `bpf_btf_get_next_id()` 遍历 BTF object ID。
2. 为每个 object 取得 file descriptor 与 metadata。
3. 忽略 userspace BTF object 和 `vmlinux` 本身。
4. 以当前 object 的 `vmlinux` BTF 为 base，解析剩余的每个 kernel-module BTF。
5. 保存所有已解析 module，供后续 CO-RE、kfunc 与 attach-target resolution 搜索。

这个函数会把结果 cache 在同一个 `bpf_object` 中，所以不会为 object 内的每条 relocation 重复扫描。问题在于，一旦需要 module lookup，昂贵的 boundary 仍接近 all-or-nothing。这也解释了为什么两个很小的 object 启动时间可以差很多：完全从 `vmlinux` 解析的 object 可以避开 module path；只要一个 module-defined target，就可能把完整 module set 带进加载过程。

RFC 报告了两组有用、但未由我们独立复现的数据。在一个含 93 个 module BTF、按需加载 BPF program 的移动端 workload 中，总加载时间超过 300 ms，其中约 69% 花在 module BTF 上。另一个 scaling test 中，存在 300 个 module 时，只解析所需 BTF 把 skeleton open-and-load 的报告值从 65.1 ms 降到 38.7 ms。这些数字说明该路径值得关注，但不是对其他 kernel、object、CPU、storage stack 或 page-cache state 的性能承诺。

## 先证明瓶颈确实是 module BTF

不能只凭 loaded module 数量判断原因。Map creation、CO-RE relocation、verifier、BPF token setup 与 program loading 都可能落在同一个 startup interval 中。

首先拆开 skeleton lifecycle，不要只测 `open_and_load()`：

```c
struct example_bpf *skel;

start("open");
skel = example_bpf__open();
stop("open");
if (!skel)
        return 1;

start("load");
if (example_bpf__load(skel))
        return 1;
stop("load");
```

这里的 `start()` 和 `stop()` 代表应用中的 monotonic-clock measurement。运行足够多次，报告 distribution 而不是单个 sample；同时记录精确 kernel build、libbpf build、BPF object、module set、privilege，以及测试是 cold 还是 warm。Kernel 的 libbpf 文档把 open 与 load 定义成独立 lifecycle phase；需要更多 loader detail 时，还可以使用 `LIBBPF_LOG_LEVEL=debug`。

然后通过只读命令查看运行 kernel 暴露的 BTF file：

```bash
find /sys/kernel/btf -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | sort
find /sys/kernel/btf -mindepth 1 -maxdepth 1 -type f ! -name vmlinux | wc -l
```

这些 basename 也是未来 selector 中 module name 的最安全起点。不要从 package manifest 推导 allowlist；runtime set 与名称才是 libbpf 真正看到的对象。

要得到可信 comparison，应固定 application object 与 libbpf，使用 module set 不同但有意匹配的 test boot。不要为了 benchmark 在 production 中随意卸载 module。如果 load time 随可用 module BTF 数量变化，而 verifier 与 map cost 保持稳定，同时 debug output 证明进入了 module BTF resolution，归因才更有把握。

## Proposed selector 改变了什么

RFC 为 `bpf_object_open_opts` 增加两个 field：

```c
const char **kmod_btf_names;
size_t kmod_btf_names_cnt;
```

Pointer 缺省时，proposal 保留当前行为。Pointer 存在时，libbpf 会复制并去重 requested name；它仍需检查 BTF metadata 才能知道 object name，但会跳过 set 之外的 BTF parsing，并在所有 requested module BTF 都找到后停止迭代。因此，大部分无关 parsing 与 storage 会被消除；不过 lookup 不一定是 constant time，因为 enumeration 仍要继续到所需 name 出现为止。

按照 RFC API，generic loader 会类似这样：

```c
static const char *needed_kmods[] = {
        "mymod",
};

LIBBPF_OPTS(bpf_object_open_opts, opts,
        .kmod_btf_names = needed_kmods,
        .kmod_btf_names_cnt = 1,
);

obj = bpf_object__open_file("example.bpf.o", &opts);
```

这段代码描述的是 RFC，不是当前 released header 中已经可用的 API。当前 upstream `bpf_object_open_opts` 的最后一个 field 是 `bpf_token_path`；直接用新 initializer 编译会失败。Generated skeleton 也只能在 generator 与 linked libbpf 都支持最终 API 时传递同一组 open options。

Caller 必须提供 object 中所有将被 load 的 program 所需的完整 dependency set。遗漏提供 CO-RE target、kfunc 或 BTF attach target 的 module，会把一次性能优化变成 resolution failure。Audit 时要包括 optional autoload program，并分别测试 success path，以及故意去掉每个 required module 时的失败行为。

## 为什么 section-name qualifier 不够

对于 BTF tracing target，module-qualified section 会告诉 lookup 在解析 function 时应优先使用哪个 module。多个 module 暴露相似 name 时，这对 correctness 很有用。但在当前 libbpf 中，module BTF collection 会先被填充，lookup loop 随后才应用 qualifier。搜索范围变窄了，准备工作并没有减少。

这里最好继续区分三种 contract：

- ELF section syntax 描述 program 与 attachment target。
- CO-RE 与 extern record 描述 type 和 symbol dependency。
- Loader option 描述 userspace 如何发现并准备 runtime metadata。

如果让 section string 同时充当完整的 object-level dependency manifest，就会漏掉同一 object 其他位置的 kfunc 与 relocation dependency。显式 loader option 可以覆盖整个 object，也能独立验证。

## RFC 前后的安全部署选择

在 selective API 合入、并进入实际携带的 libbpf 之前，可移植行为仍是：只要需要 module resolution，就扫描完整 module-BTF set。因此，当前可用的选择主要属于 operational design：

- 应用模型允许时，保持 loader process 与已加载的 BPF object 长期存活，避免反复支付 on-demand startup cost。
- 真正能把 module dependency 移出 latency-sensitive path 时，再把 optional program 拆成不同 object。必须用 phase timing 证明；仅仅拆文件不构成证据。
- 如果 startup latency 是硬要求，可以维护 version-pinned vendor backport，但必须同时维护 ABI、selftest 与 fallback。不要把 draft field name 暴露成稳定 application contract。
- Upstream support 出现后，用 build 与 runtime 实际采用的精确 libbpf header/library 对 selector 做 feature gate；dependency set 未知时保留 default path。

启用 selection 前，应为 valid required module、missing required module、duplicate name、unrelated loaded module，以及没有 module BTF support 的主机增加测试，还要保留 diagnostic logging。RFC selftest 已覆盖这些 boundary；review 也指出，custom test print callback 不能吞掉其他 libbpf error。

最后要限制结论范围。Selective module BTF loading 可以减少 userspace preparation time，但不会降低 verifier complexity，不会让缺失的 BTF target 变得有效，不会改变 kernel module set，也不能保证某个固定的 end-to-end startup latency。

## 参考资料

- [RFC v3 cover letter：选择性加载 kernel-module BTF](https://lore.kernel.org/bpf/20260819090426.267-1-zhaofuyu@vivo.com/)
- [RFC v3 implementation：通过 `bpf_object_open_opts` 选择 module](https://lore.kernel.org/bpf/150459b74dba9ea3f0bd133f97039f998e25830ada87b803fcbc6c77b17bcf93@mail.kernel.org/)
- [RFC v3 selftests：valid、missing、duplicate 与 skipped module name](https://lore.kernel.org/bpf/5cec664e8218aab8a7304ab755e65410dfa07ff5ef4f19a48ee3ce1baed80aff@mail.kernel.org/)
- [当前 libbpf source：`load_module_btfs()`](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.c)
- [当前 libbpf header：`bpf_object_open_opts`](https://github.com/libbpf/libbpf/blob/master/src/libbpf.h)
- [Linux kernel 文档：split BTF 与 module BTF](https://docs.kernel.org/bpf/btf.html#btf-base-section)
- [Linux kernel 文档：libbpf lifecycle 与 logging](https://docs.kernel.org/next/bpf/libbpf/libbpf_overview.html)
- [OpenTelemetry Helm chart proposal：把默认 Prometheus exporter 与可 scrape 的 Service 关联](https://github.com/open-telemetry/opentelemetry-helm-charts/pull/2360)
- [AF_XDP patch series：metadata layout 与 zero-copy-path fix](https://lore.kernel.org/bpf/20260819160535.1472459-1-sdf@fomichev.me/)
- [Linux kernel 文档：AF_XDP](https://docs.kernel.org/networking/af_xdp.html)
- [RFC：BPF-driven proactive memory-cgroup reclaim](https://lore.kernel.org/bpf/cover.1787120833.git.zhuhui@kylinos.cn/)
- [RFC：verifier 跟踪 low 32 bits 的 scalar equality](https://lore.kernel.org/bpf/5f1b726cd88f2261f70a5aa99f94ea1434c078c1.camel@gmail.com/)

## 今日社区讨论

今天通过普通可见浏览器检查了全部 6 个获准社区、共 15 个 allowlist 频道或公开页面，所有目标均可访问。入选问题与多个公开开发讨论都落在过去 24 小时内，因此没有使用七天回退。以下内容已删除姓名、账号、雇主、社区与频道身份、closed-chat message link、精确时间、私有拓扑、原始日志与可回搜 chat phrasing；没有保留原始 transcript。

### Startup cost 正在成为 BPF loader contract 的一部分

最强讨论把 BPF loading latency 当成 application property，而不只是开发时的不便。On-demand agent、short-lived tool 与 health-critical startup path 都可能在观察到第一个 event 前支付 metadata discovery cost。因此，proposed module selector 只有在 dependency contract 可测试时才真正有价值：caller 需要 deterministic 方法声明 required module BTF，遗漏时要明确失败，同时保留 backward-compatible full-scan mode。

同一讨论还暴露了重要的 measurement habit。“BPF object 很小”不等于“loader workload 很小”。Runtime BTF inventory、CO-RE candidate、kfunc 与 attach target 决定 metadata work。大家主要关心的是如何把这些成本与 verifier time 分开，以及怎样避免 optimization 静默让 optional probe 消失。

### Metric 需要 consumer lifecycle，不能只有 enabled endpoint

一个 observability deployment 讨论把 memory growth 与 Prometheus exporter 联系起来：metric child 的过期发生在 scrape serving path 中；没有 consumer scrape endpoint 时，启用 exporter 可能保留 series，却没有预期 eviction cycle。一个 chart change 提议，只有其 managed Service enabled 时才默认启用 exporter。

这个条件有用，但不等价于“确实有人能 scrape 这个 pod”。Pod discovery、annotation 或 externally managed Service 都能提供真实 consumer，却不需要 chart-managed Service。更健壮的 configuration model 应区分三种状态：exporter enabled、discovery object 是否由本 chart 管理、scrape health 是否已观察到。Upgrade test 应覆盖三种 discovery style 与 no-consumer case；memory test 还应验证真实 series eviction，而不能只根据 YAML rendering 推断。该 change 仍在 review，当前用户应先检查 effective configuration 与实际 scrape path。

### AF_XDP fix 再次围绕 ABI layout 与 path parity

公开 networking review 包含 TX metadata 跨 ABI layout、zero-copy path 对 metadata 的处理，以及 driver 遇到 oversized frame 时的行为说明。这些改动共同说明，AF_XDP validation 不能在一条 packet path 成功后就停止。Userspace descriptor、metadata headroom、copy mode、zero-copy mode 与 driver ownership rule 共同组成一个 contract。

有效 regression matrix 应交叉覆盖 32/64-bit layout、copy/zero-copy operation、metadata enabled/disabled，以及实际部署的 driver。Generic path 的 fix 不能证明 driver parity；某个 zero-copy driver 可用，也不能证明 copy mode 按相同 offset 解释 metadata。这些还是 active patch，不是 already released kernel 的保证。

### BPF policy proposal 正在进入 memory management 与 verifier precision

一个 active proposal 允许 sleepable BPF program 根据 runtime signal 触发一次 asynchronous memory-cgroup reclaim pass，其语义以 `memory.reclaim` 为参考。关键 design question 不是 BPF 能否调用 reclaim，而是 policy feedback、rate limit、target selection 与 observability 应放在哪里。Out-of-band reclaim trigger 必须与 synchronous protection hook 分开评估，而且 one pass 不能被误解为一定满足 requested byte target。

另一个 verifier 讨论处理两个 scalar register 仅在 low 32 bits 上成立的 relation。Zero/sign extension 后丢失这种 relation，会让 compiler-generated program 被拒绝，即使后续 branch 已证明 value range。保留更多信息可能改善 compiler portability，但 state-equivalence 与 pruning rule 必须继续 sound。因此，大家关心的是提高 precision，同时不能错误合并 safety-relevant state。

其余 project-focused support 与 chat surface 较安静，只包含 onboarding、automated project notice，或在当日窗口内没有实质技术项目。公开论坛的最新 submission 已早于本次窗口。没有把任何 inaccessible target 描述成 quiet。
