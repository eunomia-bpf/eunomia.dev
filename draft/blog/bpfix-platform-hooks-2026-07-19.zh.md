# BPFix 平台 hook 草稿，2026-07-19

> 状态：内部草稿，未粘贴到平台编辑器，未发布。
> 证据来源：[`draft/blog/bpfix.md`](bpfix.md)、[`draft/blog/bpfix.zh.md`](bpfix.zh.md)、[bpfix 论文](https://arxiv.org/abs/2607.02748)、[bpfix GitHub 仓库](https://github.com/eunomia-bpf/bpfix)。

## X 单帖，英文

An eBPF verifier error tells you where verification stopped, not necessarily where the repair belongs.

Across 235 reproduced rejections, the same normalized terminal error mapped to as many as 9 root causes. Treat the log as a proof trace: identify the proof the rejected operation needs, then walk backward to where it disappeared.

bpfix replaces raw logs with that missing context and improves one-shot LLM repair by 11–21 points in its benchmark.

Paper: https://arxiv.org/abs/2607.02748

## LinkedIn 短帖，英文

The difficult part of an eBPF verifier rejection is often not reading the last error line. It is deciding which engineering layer actually needs to change.

In a study of 235 reproduced rejections, 44 cases were repaired outside the program source, in the compiler, environment, or verifier itself. The same normalized terminal error could also represent as many as nine distinct root causes. That makes a familiar debugging habit risky: patching the C line named by the verifier before asking what proof the verifier needed there.

bpfix treats the verifier log as a proof trace, locating the proof family and the point where the evidence disappears. In its 75-task repair benchmark, replacing raw logs with these diagnostics improved one-shot LLM repair by 11–21 percentage points.

The engineering lesson is simple: diagnose the missing proof and repair layer before changing code.

Paper: https://arxiv.org/abs/2607.02748
Code: https://github.com/eunomia-bpf/bpfix

## 知乎开头

### 候选标题

为什么 eBPF verifier 明明报在这一行，真正的修复点却常在更早的位置？

### 开头

很多 eBPF verifier 报错都会给出一条看似很具体的指令，开发者顺着那一行改了几次，程序却仍然无法加载。问题不一定在这行 C 代码本身。verifier 报告的是它停止验证的位置，而这条访问需要的包指针、范围、生命周期或来源证明，往往已经在更早的路径里丢失了。

bpfix 论文复现了 235 个 verifier 拒绝案例：47% 的 errno 是 `EINVAL`，同一个归一化终端错误最多对应 9 种根因。与其把最后一行日志当成修复指令，不如把它当作一段证明轨迹的终点，倒着寻找缺失的 verifier 可见证明。

文末延伸阅读：论文与 bpfix GitHub 仓库。正式 canonical 页面上线后，将这里的站内链接替换为 eunomia.dev 页面。

## 掘金开头

### 候选标题

别只改 verifier 报错那一行：用“证明生命周期”定位 eBPF 真正的修复点

### 开头

`R# invalid mem access 'scalar'` 这类错误很容易让人直接修改报错行，但它只说明 verifier 在这里需要一个它已经看不见的事实，例如包指针来源、map value 指针、标量范围或仍然有效的 dynptr。真正值得做的是先把访问翻译成所需证明，再沿 `log_level=2` 的抽象状态往前找：证明何时出现、在哪里消失、最后应该改源代码、编译器、环境还是 verifier。

bpfix 对 235 个可复现拒绝案例的分析显示，同一归一化终端错误最多跨 9 类根因，因此“让这条错误消失”并不等于“修对了问题”。本文会用 packet-pointer 和 map-value 两类例子，给出一套可复用的日志阅读顺序。

建议分类：`后端`；建议标签：`Linux`、`开源`、`架构`。保留论文与 GitHub 链接作为延伸阅读，正式 canonical 页面上线后再补站内链接。
