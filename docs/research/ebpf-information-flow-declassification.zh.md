---
date: 2026-08-23
title: "eBPF 如何阻止秘密外泄，又不把整个进程都染成敏感？"
description: "进程级 eBPF 污点传播能阻止真实泄露，却会过度限制同时处理敏感与公开数据的进程。本文分析精度边界，并提出可信解密分类边界与覆盖验证。"
tags:
  - Daily Report
  - eBPF
  - Security
  - Information Flow
  - Linux
research_question: "当一个进程曾经读取敏感数据、随后又需要发送公开结果时，eBPF 信息流策略怎样阻止秘密进入未授权网络出口，同时避免把这个进程之后的全部输出都视为敏感？"
source_cutoff: 2026-08-23
status: daily-report
---

# eBPF 如何阻止秘密外泄，又不把整个进程都染成敏感？

设想一个构建服务。它会读取签名密钥，也会读取公开的 manifest 和大量源码，最后还要通过 HTTPS 上传一份公开测试报告。如果信息流策略只按进程传播标签，那么服务一旦读过密钥，整个进程就会被标记为敏感。从此以后，保守策略只有两个都不太好的选择：禁止这个进程之后的所有网络发送，连公开报告也一起拦掉；或者给某些目的地址开白名单，同时接受同一个进程也可能把密钥发到这些地址。

这里缺的不是更多 hook，而是更细的判断依据。Linux 已经给文件、进程、IPC、socket 和网络连接提供了很强的安全边界。eBPF 可以挂到 LSM 和 cgroup hook 上，whole-system provenance 研究也早已证明，内核对象之间的流转足以支持一类数据防泄漏策略。真正困难的地方出现在敏感数据和普通数据进入同一个用户态地址空间之后。内核可以知道某个进程读过秘密，也可以知道它后来向 socket 写了数据，却通常无法证明这次 write 里的哪些字节真的来自那个秘密。

<!-- more -->

因此，高保证的 eBPF 信息流控制不应该声称只靠 OS 事件就能恢复逐字节污点。更现实的设计是：继续把进程级污点当作安全默认值；当这种粗粒度状态会阻止合法输出时，只允许在一个明确、可验证的 **解密分类边界（declassification boundary）** 上降低敏感级别，同时把这条边界的覆盖情况纳入策略。如果系统看不见某条 release path，就应该保留 `unknown`，而不是把“没看到敏感数据”解释成“数据已经安全”。

这篇报告是 **eBPF Networking and Security** 系列的第一篇。它和之前的[多程序 Hook 组合报告](https://eunomia.dev/zh/research/ebpf-hook-composition-contract/)不同，后者研究多个 BPF 程序怎样共享同一个 hook；也不同于[有状态 eBPF 原子升级报告](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)，后者研究整个 BPF 应用怎样切换 generation。这里关心的是另一个 invariant：哪些信息可以被释放到外部 sink。

## 内核 provenance 在对象边界上已经很强

Linux 的 BPF LSM 接口允许特权 BPF 程序挂到 LSM hook 上，实现系统级 MAC 和 audit policy。LSM hook 覆盖文件访问、任务状态变化、IPC、socket 等安全相关操作，cgroup BPF 又提供 `connect` 等网络边界的控制点。它们的优势很直接：无论最终执行动作的是主程序、子进程、自动生成的 shell 脚本，还是闭源工具，只要动作经过同一内核边界，就会到达同一套策略。

这种对象级信息流不是新概念。[CamFlow](https://camflow.org/publications/socc-2017.pdf) 使用 Linux Security Module 和网络 hook 捕获 whole-system provenance，并展示了 data-loss prevention 等应用。[CamQuery](https://camflow.org/publications/ccs-2018.pdf) 又把 provenance query 放进运行时路径，让信息流判断可以实时执行，而不只是事后取证。它们说明：即使不修改应用，process、file、IPC 和 network object 之间的关系也能构成有用的 provenance graph。

现代 eBPF 让这类机制更容易部署，也更容易和别的内核控制组合，但它没有改变证据的基本粒度。`secret.key -> process P` 的 read edge 能说明秘密可能影响 `P`。之后看到 `P -> socket` 的 write edge，只能说明 `P` 发出了字节。中间的 `memcpy`、parser field、字符串拼接、压缩、redaction 等用户态内存变化，并不会自动变成内核对象。

因此，保守的信息流系统通常选择广泛传播标签。当前 [ActPlane rule language](https://eunomia.dev/actplane/rule-language/) 就把这个 trade-off 写得很明确：进程读取带标签文件后继承标签，之后的写操作继续传播；[support matrix](https://eunomia.dev/actplane/support-matrix/) 也说明过度 taint 是有意为之，因为宁可多保留 provenance，也不要静默漏掉派生流。这是合理的安全默认值，但也正好暴露了本文关心的生产精度问题。

## HTTPS 又增加了一层语义边界

加密流量会把这个问题表现得更明显。在 kernel socket 层，TLS application data 往往已经是密文。网络 hook 仍然可以按 cgroup、socket、IP、port 或 connection 做控制，却无法从密文判断某个字段来自哪个输入。

在用户态 TLS API 上，情况不同。OpenSSL 文档定义 `SSL_write()` 和 `SSL_write_ex()` 接收应用 buffer，再把这些字节写入 TLS connection。因此，在函数入口挂 uprobe 可以看到加密前 plaintext。[Eunomia 的 sslsniff 教程](https://eunomia.dev/zh/tutorials/30-sslsniff/) 就演示了这个机制。

但把 `SSL_write` 当成统一的 plaintext checkpoint 同样会过度承诺。OpenSSL 在启用 kernel TLS 时还提供 `SSL_sendfile()`，可以让文件数据走 zero-copy 路径。其他程序可能使用 BoringSSL、GnuTLS、rustls、语言 runtime、自定义 framing、QUIC 或静态链接。[Claude Code TLS 分析](https://eunomia.dev/blog/2026/02/13/reverse-engineering-claude-codes-ssl-traffic-with-ebpf/) 也展示了实际部署中的麻烦：即使最终确认流量走同一套 BoringSSL 路径，stripped binary、runtime call path 和 capture timing 都会影响 probe 是否真的覆盖目标请求。

所以，application boundary probe 确实能提供更细的语义，但它同时引入了一个 **coverage obligation**。安全系统必须区分两种完全不同的结果：“检查过的 TLS call 没有敏感数据”和“这次 egress 根本没走被检查的路径”。

2026 年 8 月的新预印本 [A Study of Kernel Telemetry Options for Security-Oriented Provenance](https://arxiv.org/abs/2608.11418) 从另一个角度给出了类似警告。它认为 eBPF 是很有潜力的 provenance capture 基础，同时指出被分析的许多 capture stack 无法满足安全场景对事件完整性和可用性的要求。对于 enforcement 来说，缺失证据不能被当作中性结果，coverage 的未知状态必须进入决策本身。

## 真正要决定的是：哪里可以可信地解除敏感状态

回到开头的服务。它读取签名密钥后，只想输出一句公开的 `signature verified`。合理的策略应该允许这句话离开主机，同时不允许密钥本身出去。

第一种办法是在应用内部做完整的 byte-level dynamic taint tracking。这能获得很细的精度，但需要 instruction-level 或 language-level instrumentation，部署和开销模型已经不是普通 eBPF policy plane。LSM hook 也不可能在 syscall 发生之后再反推出所有用户态数据依赖。

第二种办法是信任整个进程，让它自己解除敏感标签。实现很简单，但安全边界很弱。只要同一进程里的 bug 或攻击者能调用解密分类接口，它就可以给任意输出洗白。

第三种办法是把解密分类做成一个显式、权限更窄的边界。一个小的可信组件接收带标签输入，只执行声明过的转换，然后只产生或发送允许释放的结果。这样，kernel provenance 又重新获得一个可以观察和控制的 object/subject boundary。它放弃了透明恢复逐字节数据依赖，但换来一个更容易描述、验证和 fault test 的安全属性。

因此，核心问题不是“还能再加多少 eBPF hook”，而是“什么证据允许一个粗粒度内核标签变得更宽松，同时又不允许原来的敏感进程直接宣布自己已经干净”。

## 现有研究还缺什么

### 进程标签分不清同一地址空间里的混合 buffer

Whole-system provenance 和 lineage-level IFC 天生偏保守。进程同时消费敏感输入和公开输入后，一个进程标签只能表示这些来源的并集。对 confidentiality 来说这样安全，但长期运行的 compiler、database、agent runtime 或 web service 很快就会成为永久 tainted subject。

缺少的是一种可部署的方法：只在少数 release point 恢复精度，而不是给所有用户态指令做 instrumentation。现实后果就是典型 DLP trade-off：粗粒度 policy 要么误拦大量正常输出，要么不得不开很宽的 destination exception，最后削弱原本的保证。

一个能区分这件事的实验，应当让同一套混合 workload 分别运行在 process-level taint、byte-level taint 和 explicit declassification 下。如果真实 workload 里 coarse taint 几乎不误拦，那么新增机制没有必要。

### Application probe 缺少统一的覆盖契约

Uprobe 可以在不改内核的情况下观察 `SSL_write`、serializer、parser 和应用自己定义的 gate，但“attach 成功”并不等于“所有相关路径都经过这里”。

缺少的是机器可读的 coverage contract，把 binary identity、function/offset identity、runtime/library version 与它应该解释的 sink 绑定起来。没有这个契约，系统很容易把一次成功 attach 误认为 complete mediation。

评测应该主动在 OpenSSL `SSL_write`、OpenSSL `SSL_sendfile` + kTLS、BoringSSL、Rust TLS、direct plaintext socket 和 helper subprocess 之间切换。系统必须明确输出“covered release path”或 `unknown`，不能把没有观察到的路径静默归类为 clean。

### Declassification authority 通常没有被明确建模

信息流系统最终都需要某种 label removal 或 transform，否则长期运行后所有对象都会不断积累 taint。类似“运行 sanitizer 后清除 SECRET”的规则，只有在 sanitizer 本身是可信 release boundary 时才有意义。

缺少的是 declassifier 的 authority 和 identity model。runtime 需要知道：哪个 binary 或 service instance 能释放哪个 label，它必须满足什么 input/output relationship，这次授权属于哪个 policy generation。若面对 adversarial process，原进程自己发一个 userspace event 不能算可信证据。

Fault test 可以很直接：替换 sanitizer binary，启动同名 sibling process，重放 stale release token，并在 release 中途 kill policy controller。正确结果应该是 fail closed 或保留显式 `unknown`。

### Security benchmark 很少同时衡量漏放和误拦

把所有 network operation 都拦掉，安全指标可以很好看；把常用 endpoint 都允许，业务指标也可以很好看。两者都没有回答系统能否正确区分敏感 release 和普通 release。

真正缺少的是带 ground truth 的 mixed-flow workload。它既要知道哪些输出实际依赖 protected input，也要记录哪些公开输出应该被允许，然后同时评分 false-negative leak、false-positive block、unknown coverage、fault recovery 和 overhead。

如果一个方案通过引入未被测量的 bypass 来减少 false positive，它不应该被算成精度提升。

## 值得继续做的方向

### 用可信 release proxy 把解密分类重新变成内核可见的边界

**缺口。** Process-level taint 无法只清除一个输出，同时保留原进程里的敏感状态。

**机制。** 把 declassification 移到一个单独隔离的 release service。tainted application 通过 kernel-visible IPC 把候选输出和 release class 交给服务。服务只执行声明过的转换，例如 redaction、aggregation、signature verification 或 schema projection。BPF-LSM policy 限制什么 executable identity 可以充当 declassifier，以及它可以访问哪些 sink。服务可以自己发送释放后的结果，也可以通过带有独立 provenance 的对象交付结果，但原始 tainted process 永远拿不到“清除自己标签”的权限。

这里重要的是 separation，而不是一定要叫 proxy。只要 OS 能控制输入、输出、binary identity 与 destination authority，独立 helper process、本地服务或隔离 release worker 都可以实现这个边界。

**相对现有工作的增量。** CamFlow/CamQuery 已经证明 whole-system provenance 可以做 DLP，当前 eBPF IFC engine 也能保守传播标签。新的部分是把 declassification 变成一个带 authority 和 generation state 的跨安全域协议，而不是在原 subject 内部执行一条 label-removal rule。

**原型。** 一个基于 libbpf/BPF-LSM 的 controller 加 release-service SDK，先支持 file-to-network 和 IPC-to-network flow。policy manifest 声明 source label、允许的 declassifier identity、release class、permitted sink，以及证据缺失时的 fail-closed behavior。

**评测。** 使用会混合 secret/public input 的 build、data-processing 和 agent workload，对比 process-level taint、destination allowlist、release proxy 与 application-instrumented baseline。测量真实 secret leak、被误拦的合法 network operation、release latency、throughput、CPU overhead、policy state memory 和 helper/controller crash 后的恢复。另加一组 malicious workload，尝试 malformed request、replay 和冒充 declassifier。

**学术价值。** 核心问题是：OS-level IFC 能否通过一个很小的可信 release boundary，恢复足够的精度，而不必跟踪任意用户态内存依赖。

**生产价值。** 运维团队可以继续对复杂或闭源程序采用保守 kernel enforcement，只给 telemetry、report、redacted log 和批准的 export 开很窄、可审计的释放路径。

**失败条件。** 如果额外 IPC 和程序重构成本高于普通 application instrumentation，或者多数 workload 本来就把敏感处理隔离在独立 service 中，那么这个 proxy 没有足够价值。

### 为 opaque runtime 建立 coverage-aware egress manifest

**缺口。** Application probe 能恢复 plaintext 和业务语义，但 attach 成功无法证明全部 egress path 都被覆盖。

**机制。** 定义一个 egress manifest，把 runtime build identity 映射到预期 plaintext boundary 和下游 socket path。entry 可以包含 shared-library build ID、静态链接 function fingerprint、USDT probe 或 runtime adapter。eBPF uprobe 在这些 boundary 上关联 release event 与 socket identity，kernel network hook 则独立观察真实连接。如果 socket 发出流量却没有匹配的 covered release path，policy 把 flow 标记为 `unknown`，再按配置执行 fail closed、isolate 或 audit。

Manifest 必须显式考虑 OpenSSL `SSL_sendfile`/kTLS 和 helper subprocess 等替代路径。coverage 也会随软件升级失效，所以 controller 应保存已验证 build generation，而不是默认昨天的 offset 今天仍然正确。

**相对现有工作的增量。** 这不是普通 TLS tracing。新的要求是把“缺少对应 semantic event”本身变成 enforcement state。它也和之前 application-resource semantics 报告不同：这里的 contract 直接保护 egress sink 上的 confidentiality decision，semantic coverage 不完整会改变 release 是否允许。

**原型。** 一个 coverage validator 加 eBPF correlation runtime，支持 OpenSSL、BoringSSL、一种 Rust TLS、direct socket 与 kTLS，并在加载 policy 前输出机器可读 coverage report。

**评测。** 修改 library version、strip symbol、切换 TLS implementation、开关 kTLS、fork helper process，并重放真实 client/server workload。测量 missed path detection、false `unknown`、attach/update time 和 steady-state overhead，对比 baseline “probe attached successfully”。

**学术价值。** 问题是 heterogeneous application probes 和 kernel sink 能否提供足够可靠的 coverage contract，用来做 security decision，而不要求完整 application instrumentation。

**生产价值。** 安全团队可以明确区分“已经检查过的加密 egress”和“plaintext path 仍然不了解的 traffic”。

**失败条件。** 如果 runtime variation 让 manifest 过于脆弱，或者 kernel-side socket evidence 无法可靠识别未观察路径，这个机制应该留在 observability，而不是 hard enforcement。

### 建立同时测 leak 和 benign block 的 mixed-flow benchmark

**缺口。** 现有 provenance/policy evaluation 往往能展示一次 forbidden flow 被检测出来，却不足以回答 conservative taint 会阻止多少正常工作。

**机制。** 构造一组 workload，并让 harness 持有真实 dependency graph。case 覆盖同进程读取 secret/public file、fork/exec、pipe、Unix socket、FD passing、mmap、temporary file、compression、redaction、多种 TLS stack、kTLS/sendfile、direct socket 与 approved declassification。每个输出由 harness 根据 protected source 是否真实影响它来标注 ground truth。

**原型。** 可复现 workload suite、trace generator、attack/bypass corpus 和 scorer。scorer 同时报 leak rate、benign-block rate、unknown coverage、provenance continuity、fault 后恢复和固定 request mix 下的 overhead。

**评测。** 比较 path/destination allowlist、process-level label propagation、provenance-query enforcement、trusted release proxy 和 application-assisted 方法，并分别去掉 TLS coverage validation、declassifier identity check 和 policy-generation freshness 做 ablation。

**学术价值。** Benchmark 能把“更精确的 IFC”变成 confidentiality、availability、coverage 与 cost 之间可量化的 trade-off。

**生产价值。** 团队可以先测自己的 workload，再决定 coarse eBPF IFC 是否已经足够，还是必须先建立 release boundary 才能开启 blocking mode。

**失败条件。** 如果 dependency oracle 无法表达真实 transformation，或者攻击者可以轻易利用 benchmark 未建模的语义，它只能作为 stress suite，而不能被当成 security ground truth。

## 哪些结果会改变这个判断？

本文默认一个常见生产场景：敏感和公开数据会进入同一个长期运行的 userspace process；operator 希望保留 OS-level mediation；完整 byte-level dynamic taint 又太难或太贵，无法部署到所有程序。

有几类结果会削弱显式 declassification 的必要性。如果代表性 workload 证明 conservative process-level taint 几乎不会误拦，那么简单模型更好。如果 whole-process byte-level taint 已经能以相近 overhead 覆盖多种语言与 runtime，release proxy 也可能没有必要。如果未来 kernel 或 hardware 能给 userspace buffer 提供可信的细粒度 provenance，那么 object provenance 与 in-process flow 之间的边界也会改变。

相反，如果 mixed workload 显示 coarse taint 会大量阻止合法输出，而窄的可信 release boundary 能以较低成本保持 confidentiality，那么 declassification 就应该成为 eBPF security runtime 的一等机制，而不是每个应用自己做一个 exception。

最后的约束很简单：证据不足时可以保守；不能因为证明泄露的路径恰好不可见，就宣称敏感信息已经变得安全。

## 参考资料

- [Linux kernel：LSM BPF Programs](https://docs.kernel.org/bpf/prog_lsm.html)
- [Linux kernel：BPF program type 与 cgroup network attach point](https://docs.kernel.org/bpf/libbpf/program_types.html)
- [CamFlow: Practical Whole-System Provenance Capture, SoCC 2017](https://camflow.org/publications/socc-2017.pdf)
- [CamQuery: Runtime Analysis of Whole-System Provenance, CCS 2018](https://camflow.org/publications/ccs-2018.pdf)
- [A Study of Kernel Telemetry Options for Security-Oriented Provenance, 2026](https://arxiv.org/abs/2608.11418)
- [OpenSSL：SSL_write、SSL_write_ex 与 SSL_sendfile](https://docs.openssl.org/master/man3/SSL_write/)
- [Eunomia：使用 eBPF uprobe 捕获 TLS plaintext](https://eunomia.dev/zh/tutorials/30-sslsniff/)
- [Eunomia：ActPlane support matrix](https://eunomia.dev/actplane/support-matrix/)
- [ActPlane 源码](https://github.com/eunomia-bpf/ActPlane)
