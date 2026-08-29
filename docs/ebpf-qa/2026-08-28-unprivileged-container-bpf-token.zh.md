# 非特权容器能否创建自己的 BPF token？

**简短回答：**可以，但前提是 privileged manager 已先建立 token 所依赖的 authority。一个在 host initial user namespace 中没有特权的 process，只有同时满足以下条件，才能自行调用 `BPF_TOKEN_CREATE`：

- 它与一个专用 bpffs instance 的 owner 位于同一个 non-initial user namespace；
- 该 bpffs 已通过 `delegate_cmds`、`delegate_maps`、`delegate_progs` 和/或 `delegate_attachs` mask 明确配置 delegation；
- caller 在该 user namespace 中拥有 `CAP_BPF`。

返回的 file descriptor 不是能够替代 capability 的 bearer credential。后续 map、BTF 与 program operation 既要携带 token，也要在 token 对应的 user namespace 中拥有该 operation 所需的 capability。Token 改变 capability check 的求值位置，并限制可请求的功能；它不会凭空制造 privilege。

因此，orchestrator 可以让 workload 自己派生 token，但 workload 不能从零创建 delegation policy，也不能扩大它。清晰的分工应是：privileged manager 定义 least-privilege bpffs delegation，workload 在该边界内派生并使用 token。

## Delegate 与 derive 是两个不同动作

这个设计最容易混淆的地方，是“创建 token”听起来像单一 operation。实际上它是一个两阶段 protocol 的后半段。

第一阶段由 privileged manager 准备一个归 workload user namespace 所有的 bpffs instance，并在该 filesystem 上配置四组互相独立的 bit mask：

- `delegate_cmds` 限制哪些 `bpf()` command 可以使用 token；
- `delegate_maps` 限制 allowed `BPF_MAP_CREATE` 可以创建的 map type；
- `delegate_progs` 限制 allowed `BPF_PROG_LOAD` 可以加载的 program type；
- `delegate_attachs` 限制 program load 可以声明的 expected attach type。

Kernel bpffs implementation 将这些值作为 filesystem parameter 接收并保存在 superblock 中；之后创建的 token 会复制这些 mask。Workload 即使拿到了 prepared mount，也不能仅凭自己是相关 user namespace 的 owner 就重写 delegation mask。Upstream selftest 明确验证了 unprivileged child 尝试重配时会得到 `EPERM`。

第二阶段中，该 user namespace 内的 process 打开 delegated bpffs 的**根目录**并调用 `BPF_TOKEN_CREATE`。Kernel 核对 filesystem、namespace、capability 与 delegation state 后，返回一个 anonymous token file descriptor。也就是说，workload 从更高权限主体预先建立的 policy 中派生了一个具体 handle。

Manager 也可以自己派生 token，再通过 Unix-domain socket 的 `SCM_RIGHTS` 传递 file descriptor；upstream selftest 覆盖了 FD passing。两种方式的 authority boundary 相同：拿到 descriptor 不代表 receiver 可以修改 mask 或跳过本地 capability check。

## 最小的 low-level handshake

使用较新的 libbpf 时，workload 侧的 derivation 很短：

```c
int bpffs_fd = open("/run/workload-bpf", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
if (bpffs_fd < 0)
    /* report errno */;

int token_fd = bpf_token_create(bpffs_fd, NULL);
if (token_fd < 0)
    /* report errno */;
```

Direct syscall form 会设置 `attr.token_create.bpffs_fd` 并调用 `bpf(BPF_TOKEN_CREATE, ...)`。`bpffs_fd` 必须指向 filesystem root，不能是 pinned object 或 subdirectory。

后续 operation 有各自的 token field。例如 low-level map creation 要设置 `bpf_map_create_opts.token_fd`，并在 `map_flags` 中加入 `BPF_F_TOKEN_FD`；program 与 BTF load 则在各自的 option flags 中做对应设置。只提供 descriptor 却不设置 flag，或只有 flag 没有有效 token descriptor，都没有完成 handshake。

加载普通 BPF ELF object 的 application 不必在每个 call 中手工传递 descriptor。`bpf_object_open_opts.bpf_token_path` 可以指定 libbpf 应从哪个 bpffs root 派生 token。如果没有设置它，libbpf 会先检查 `LIBBPF_BPF_TOKEN_PATH`，否则尝试默认的 `/sys/fs/bpf`；空值会关闭 automatic token creation。启用后，libbpf 会自行派生 token，并把它用于该 object 支持的 map creation、BTF load 与 program load operation。

## Token 与 namespaced capability 缺一不可

Upstream selftest 对 privileged map creation 验证了四种组合：

| 是否提供 token | Token user namespace 中是否有 `CAP_BPF` | 结果 |
| --- | --- | --- |
| 否 | 否 | 失败 |
| 是 | 否 | 失败 |
| 否 | 是 | 失败 |
| 是 | 是 | 如果 command 与 map type 已被 delegate，则成功 |

Program loading 可能不只需要 `CAP_BPF`。Kernel UAPI 列出的 capability 包括 `CAP_BPF`、`CAP_PERFMON`、`CAP_NET_ADMIN` 与 `CAP_SYS_ADMIN`；token 可以让这些 check 在其 user namespace 中求值。具体需要哪些取决于 program type、helper 与 operation。Kernel selftest 特意加载了一个 XDP program，并使用会覆盖 `CAP_NET_ADMIN`、`CAP_BPF` 与 `CAP_PERFMON` check 的 helper。

所以，“unprivileged container”必须被精确定义。Workload 可以在 host initial user namespace 中没有相关 capability，同时在自己的 user namespace 中拥有范围受限的 capability；BPF token 使这些 capability 能用于选定的 BPF operation。一个在任何相关 namespace 中都没有 required capability 的 process，不会因为拿到 token FD 就获得它。

## 把 `errno` 当成未满足的前置条件

对 `BPF_TOKEN_CREATE` 本身，current kernel implementation 给出了很有用的 failure map：

- `EBADF`：提供的 bpffs descriptor 无效；
- `EINVAL`：descriptor 不是 bpffs instance 的 root；
- `EPERM`：caller 不在 bpffs owning user namespace，或在那里没有 `CAP_BPF`；
- `EOPNOTSUPP`：试图在 initial user namespace 创建 token，这种 token 无法迁移 capability check；
- `ENOENT`：bpffs instance 没有配置任何 delegation mask。

Filesystem access check 还可能返回普通 path-permission error，security module 也可以拒绝 token creation 或 use。

如果 token creation 成功，但之后的 BPF operation 以 `EPERM` 或 `EINVAL` 失败，应检查另一组事实：

1. 是否在正确的 flags field 中设置了 `BPF_F_TOKEN_FD`，并传入正确的 token FD？
2. Token 是否允许 requested `bpf()` command、map type、program type 与 expected attach type？
3. Process 是否仍在 token owning user namespace 中持有所有 required capability？
4. 是否是 verifier 拒绝 program，或者 LSM、target-specific attach check 拒绝 operation？

`BPF_OBJ_GET_INFO_BY_FD` 可以返回 token 的四组 allowed mask，`/proc/self/fdinfo/<token-fd>` 也会把它们渲染出来用于诊断。Bpffs mount option 同样能通过常规 mount information 查看。应在受控 diagnostic environment 中记录这些派生事实和 error code；不要把 token FD number 本身当作有意义的证据，因为 descriptor number 是 process-local 且可复用的。

## 这条边界没有授权什么

BPF token 的含义比“容器里的 BPF 现在不受限制”窄得多：

- Token 继承 bpffs delegation mask。当前 `BPF_TOKEN_CREATE` 不接受第二组可以让 deriving workload 扩大或缩小权限的 per-token mask。
- Verifier 仍会检查 program safety、helper availability、context access 和所有正常 program rule。
- LSM token hook 与 operation-specific security hook 仍会运行。
- Target ownership 与 attachment rule 仍然适用；允许 load program 不自动等于允许 attach 到任意 host object。
- 最后一个 reference 被 close 后 handle 即失效；传递或 duplicate FD 保留的是同一个 token，不是一个新 policy。

对 multi-tenant system，应为每个 trust boundary 使用独立 delegated bpffs instance，只 delegate workload 实际需要的 command 与 type，避免暴露 host 的 general-purpose bpffs mount。User namespace、mount configuration、capability set 与 token FD 应作为一个整体 security design。只审计其中一个，会得到误导性的结论。

## 实用的 validation sequence

在把机制接入 orchestrator 前，应使用最小、无害的 object 复现 upstream matrix：

1. 确认 normal directory、bpffs subdirectory、undelegated bpffs 与 initial user namespace 上的 token creation 都会失败。
2. 在 workload user namespace 中，确认只有拥有 local `CAP_BPF` 时才能创建 token。
3. 选一个 delegated map type，证明 token 加 capability 成功，而仅 token、仅 capability、两者皆无都失败。
4. 请求一个 mask 之外的 map 或 program type，证明它会被拒绝。
5. 对 production workload 需要的每个 program 与 attach type 重复验证，包括 additional capability 和 target permission。
6. 如果 token 或 bpffs FD 会跨越 process boundary，测试 exact `SCM_RIGHTS` 或 inherited-FD path，并关闭意外副本。

这组测试证明的是 security boundary，而不只是某个 happy-path loader 恰好能够运行。

## 参考资料

- [Linux BPF UAPI：`BPF_TOKEN_CREATE` semantics 与 user-namespace capability check](https://github.com/libbpf/libbpf/blob/master/include/uapi/linux/bpf.h#L3859-L3912)
- [Linux kernel token implementation：creation check、inherited mask、fdinfo 与 security hook](https://github.com/torvalds/linux/blob/master/kernel/bpf/token.c)
- [Linux bpffs implementation：delegation mount parameter 与 rendered mask](https://github.com/torvalds/linux/blob/master/kernel/bpf/inode.c)
- [Linux BPF selftest：delegated bpffs setup、FD passing、capability matrix 与 denied reconfiguration](https://github.com/torvalds/linux/blob/master/tools/testing/selftests/bpf/prog_tests/token.c)
- [libbpf low-level BPF API，包括 `bpf_token_create`](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/bpf.h)
- [libbpf object-open token path 与 `LIBBPF_BPF_TOKEN_PATH`](https://github.com/libbpf/libbpf/blob/master/src/libbpf.h#L2540-L2575)
- [OpenTelemetry GenAI contributor guidance：sanitized VCR cassette 与 live re-recording](https://github.com/open-telemetry/opentelemetry-python-genai/blob/main/AGENTS.md)
- [OpenTelemetry declarative configuration：semantic-convention stability opt-in](https://github.com/open-telemetry/opentelemetry-configuration/blob/main/opentelemetry_configuration.json)
- [Linux BPF 讨论：让 KASAN 覆盖 BPF JIT memory](https://lore.kernel.org/bpf/20260828-kasan-v8-0-7c1c0fdb9d7f@bootlin.com/T/#t)
- [Linux BPF 讨论：避免在 dump key-less BTF map 时发生 null dereference](https://lore.kernel.org/bpf/8b3e7f24-795d-458b-a24e-fe154b0cf03d@linux.dev/T/#t)

## 当日社区讨论

今天通过普通可见浏览器检查了全部 6 个批准社区和 15 个 allowlist 频道或公开页面，所有目标均可访问。选题来自过去 24 小时，因此没有使用七天 fallback。姓名、账号、雇主、workspace 与频道身份、message link、精确时间、私有拓扑、原始日志和可搜索回原讨论的措辞均已删除。没有保留原始 transcript，也没有进行任何社交互动。

### Delegation 需要面向失败路径的解释

最强的未解决问题询问：workload 能否自己完成完整的 BPF token handshake，以及一些看似合理的尝试为什么失败。此前缺少的是对 configuring delegation 与 deriving token 的区分。Public kernel code 与 selftest 表明，workload 可以拥有后一个步骤，但前提是 privileged manager 已建立 compatible user namespace 与 delegated bpffs。上文的 token/capability 四组合 matrix 和 `errno` map，使 trust boundary 可以被实际测试，而不是把 token creation 成功误当作充分证据。

### Provider test 必须区分 reproducibility 与 live proof

一个 instrumentation 讨论再次提出了现实的 contributor 问题：缺少 live service credential 时，如何测试 provider-specific change。Public contributor guidance 支持分层处理。Deterministic request/response behavior 应放入 sanitized VCR cassette 或 unit test；authentication header、cookie、organization identifier 与能够识别 account 的 body field 必须删除。AI-synthesized cassette 应明确标记为 temporary，并在之后使用真实 provider 重新录制。这样的 fixture 可以验证 code path，但不能证明今天的 provider endpoint 与 credential 正常。

相关 implementation work 正在把多个 provider integration 迁移到 shared GenAI utility。这提高了 common conformance suite 的价值：provider adapter 应证明相同的 span structure 与 redaction invariant，只保留一个小型、受控的 live-check layer 来覆盖 SDK serialization 与 service drift。

### Semantic-convention switch 需要 retirement plan

另一个活跃主题是 breaking telemetry-schema change 的 feature flag。每次 transition 配置一个 flag 可以让 rollout 可回滚，common stability opt-in 也支持 phased migration；但永久增长的 flag list 会变成第二套 schema。持久的设计应定义 old、duplicate-emission 与 new 三种 state，对每种 state 进行测试，记录 global 与 domain-specific configuration 共存时的 precedence，并在 downstream consumer 完成迁移后设置明确的 removal condition。

### Kernel 工作集中在让 failure state 可观察

Public kernel discussion 的活跃主题包括：JIT-generated memory 的 KASAN visibility、dump 无 key map 时的 null dereference、callback register 的 verifier precision、indirect trampoline test 与 profiler reporting。它们共同关心的是 subsystem boundary 上的 observability：先区分 JIT memory corruption、verifier-state loss 与 userspace-tool assumption，再为真正失效的 boundary 保留一个最小 selftest。

若干 project-specific chat surface 在 daily window 内较安静，其他可见活动则主要是 review、merge order 或 conflict resolution request，而不是新的 technical question。这些目标仍被记录为 accessible and checked；没有把它们转写为虚构问题，也没有把它们重复用作本篇答案的证据。
