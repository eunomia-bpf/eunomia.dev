---
date: 2026-05-31
description: ActPlane 是一个基于 eBPF 的 AI Agent 策略引擎，在操作系统内核层面对 Agent 行为做观测和强制执行。本文分析 prompt、工具层、沙箱三层约束各自的系统性盲区，说明 ActPlane 如何通过标签传播和时序谓词实现确定性的 Agent harness。
---

# ActPlane: 把 Agent Harness Enforcement 下沉到内核 eBPF

你在 CLAUDE.md 里写了一条规则："不要执行 `git push`"。Agent 遵守了，它确实没有调用 git 工具。但它写了一个 Python 脚本，脚本里调了 `subprocess.run(["git", "push"])`，代码推到了远端，而 Prompt 约束从未被违反。

这个场景揭示的是一个结构性问题：**Agent 的约束和 Agent 的副作用不在同一个层面。** Prompt 约束在推理层，工具守卫在 API 层，沙箱在容器层，但 Agent 的所有副作用最终都要经过操作系统内核。每一次 exec、每一次文件 open、每一次网络 connect，不管 Agent 用什么路径到达这里，内核都在场。如果约束不在这个层面执行，Agent 总能找到一条绕过去的路，而且它甚至不需要恶意，它只是在尝试完成你交给它的任务。

[ActPlane](https://github.com/eunomia-bpf/ActPlane) 建立在这个判断之上。它通过 eBPF 在内核安装策略引擎，在系统调用层面观测和执行 Agent 约束。规则匹配了就一定执行，不依赖 Agent "记住"什么。但它做的不只是拦截，当约束触发时 Agent 收到的是人类可读的反馈，告诉它为什么被拦、该怎么做，于是 Agent 理解原因后换一条路继续完成任务。这是 harness（约束框架）和 sandbox（沙箱）的根本区别：沙箱给你一堵墙和一个 `Permission denied`，harness 给你一条规则和一个替代方案。

> 链接: https://github.com/eunomia-bpf/ActPlane

<!-- more -->

## 先看我们想要什么

在拆解问题之前，先看几个 Agent 日常工作中真正需要的约束。这些约束有一个共同点，就是 prompt 说不清楚，MCP gateway 拦不住，容器沙箱也表达不了。

禁止 Agent 执行 `git push` 听起来像一条普通的沙箱规则，但"禁止执行"的覆盖范围取决于你在哪个层面检查。Agent 可能直接调 git，也可能写一个 shell 脚本再执行，也可能生成一个 Python 程序嵌套 subprocess 调用。工具 API 层只能检查"Agent 有没有调用 git 工具"，但我们真正想问的是"有没有一个属于这个 Agent 进程树的子进程执行了 git"，而后者需要沿整个进程树追踪，两个问题的覆盖面完全不同。

再看一个时序约束：修改了 `specs/*` 之后必须同步跑 `protoc`，否则提交时应该收到提醒。我们不想阻止 Agent 编辑 spec 文件，那是它正常工作的一部分；但如果它改完 spec 就直接 commit，说明它漏了一步。这里的意图不是阻止提交，而是在提交前提醒，Agent 收到提醒后自己决定要不要补跑代码生成。

提交前必须跑测试也是类似的逻辑，但多了一个动态失效的维度：每次 Agent 修改 `src/` 下的文件，之前的测试结果就应该自动失效。追踪的不是"测试是否曾经跑过"，而是"自上次修改源文件以来测试是否跑过"。跑完测试又改了一行代码？状态重置，必须再跑一次。

还有一类强制中介约束：生产数据库 `prod.db` 只能通过 migration 工具访问，Agent 不能直接打开。不管 Agent 怎么到达文件打开调用，只要它的进程祖先链里没有执行过 `migrate` 工具，操作就应该被阻止。它关心的不是 Agent 有没有某个权限，而是 Agent 走了哪条路径进来，经过了指定的 gate 程序才放行，绕过去就拦住。

这四个约束分别涉及进程谱系追踪、操作时序、动态失效和强制中介，全都超越了静态 allow/deny 的范畴。要理解为什么解决它们需要内核级方案，先看看现有三层约束各自的盲区在哪里。

## 三层约束，三种盲区

| 方案 | 它做了什么 | 它覆盖不了什么 |
|------|----------|-------------|
| **Prompt 约束**（CLAUDE.md、AGENTS.md） | 告诉 Agent 该做什么、不该做什么 | 概率性的：长上下文中 Agent 会遗忘或非恶意地绕过 |
| **工具层守卫**（MCP gateway、AgentSpec） | 在工具 API 层面拦截和授权 | Agent shell out、链接 SDK、spawn 子进程时完全绕过 |
| **沙箱**（容器、VM、E2B、Daytona） | 隔离整个执行环境 | 全有或全无：无法表达"文件 A 只能通过脚本 A 访问"或"提交前跑测试" |

CLAUDE.md、AGENTS.md、system prompt 是目前最普遍的约束方式，本质上是自然语言指令，依赖模型在推理时"记住并遵守"。问题出在"记住"两个字上。当对话超过数万 token，早期约束在注意力分配中被后续的用户指令和工具输出稀释，模型并非忘记了规则，而是规则在决策权重中的占比随上下文膨胀而下降。我们在 [AgentCgroup 的刻画实验](agentcgroup-characterization.md)中观察到，一个任务内 Agent 平均执行几十次工具调用，对话上下文增长到非常大，约束越早设定、对话越长，被有效遵守的概率越低。比遗忘更难防的是间接违反：Agent 被告知"不要删除文件"，于是它写了一个 Makefile target 里面包含 `rm -rf`，然后调用 `make clean`。从 Agent 视角看它执行的是一个构建命令，每个决策点都是局部合理的，但全局行为链却违反了最初的策略。Prompt 约束管的是 Agent 的意图表达，实际行为不在它的管辖范围。

那就往下沉一层，在工具调用 API 拦截？MCP gateway、[AgentSpec](https://arxiv.org/abs/2503.18666)、[Progent](https://arxiv.org/abs/2504.11703) 确实比 prompt 可靠得多：如果 Agent 调用 `file_write("/etc/passwd", ...)`，工具层直接拒绝，检查的是操作本身而非意图。但几乎所有 Agent 框架都提供 shell 工具，因为 Agent 需要跑构建、执行测试、调用第三方 CLI。一旦 Agent 拿到 bash，它可以写一个 Python 脚本，脚本内部 `subprocess.run(["curl", "-X", "POST", ...])` 把数据发到外部网络。工具层看到的是 `python script.py`，对脚本内部启动了什么子进程完全不知道，调用链可能有三四层深，工具层只看到第一层。更隐蔽的是 Agent 生成的代码直接链接数据库驱动或 HTTP 客户端库，这些操作在运行时发生，根本不经过任何工具调用路径。工具层的视野止于注册工具的入口，代码执行产生的副作用全在视野之外。

那再往下沉一层，用容器把整个环境隔离起来？容器、VM、[E2B](https://github.com/e2b-dev/E2B)、[Daytona](https://github.com/daytonaio/daytona) 是目前最可靠的安全边界，对防止 Agent 逃逸到宿主机而言确实是正确的答案。但 Agent 实际需要的约束远比"能不能访问某个资源"丰富得多。"改了 proto 后必须跑 protoc 再提交"是时序约束，沙箱没有时间概念，它只知道当前瞬间哪些资源可以访问。"从数据库读出的敏感数据不能写入日志"需要追踪数据流向，沙箱粒度是进程级，不知道进程内部读了什么写了什么。同一个 `git commit`，在"刚跑完测试"和"还没跑测试"两种情况下应该有不同的策略，沙箱无法根据历史上下文做区分。

还有一个经常被低估的问题是反馈质量。沙箱拒绝操作时，Agent 收到的只是 `Permission denied` 或 `EPERM`，既不知道为什么被拒，也不知道怎么做才能满足约束。我们观察到的典型结果是 Agent 反复重试三五次然后放弃整个任务，或者更糟，它尝试另一条路径绕过，引入新的问题。沙箱回答的是"这个进程能否访问这个资源"，但 Agent 需要回答的问题远比这丰富：在什么条件下、以什么顺序、基于什么数据流历史，这个操作是否被允许？

## 所有副作用的必经之路

三层约束的盲区指向同一个方向：约束机制需要下沉到所有操作的必经之路。[AgentSight](https://arxiv.org/abs/2508.02736)（2025）用 eBPF 同时捕获 Agent 的意图层和行为层数据，提出了两者之间"semantic gap"的概念。ActPlane 在这个观测基础上加了执行力：Agent 用什么工具、写什么脚本、spawn 多少层子进程，最终所有副作用都要经过操作系统内核的系统调用。ActPlane 通过 eBPF 在内核中安装轻量级程序，hook 住进程生命周期（`sched_process_fork`、`sched_process_exec`、`sched_process_exit`）、文件操作（`sys_enter_openat`、`sys_enter_unlinkat`、`sys_enter_renameat2`）、以及网络连接（`sys_enter_connect`），然后根据标签化的信息流策略做决策。Agent 无论通过什么路径产生副作用，最终都会经过这些系统调用。

回到开头的场景就能看出差别：Agent 写了一个 Python 脚本，脚本 spawn 子进程调 git push。工具层只看到 `python script.py` 这一层，ActPlane 看到整个进程树的所有系统调用，包括三层深处那个 git。而且约束直接挂在进程树上，"Codex 的整个子进程树不能碰 git"只需要一条规则，不用在每个可能的工具入口重复设防。

但 ActPlane 做的不只是把工具层守卫下沉到内核。它还引入了两个工具层和沙箱都缺少的能力：**数据流追踪**和**时序推理**。标签可以跨越 fork/exec 和文件读写边界传播，使得"从 A 读取的数据不能流向 B"成为可表达的策略；`since` 子句让规则在事件时间线上动态更新，使得"自上次修改源文件以来是否跑过测试"成为一个会随新事件不断失效和重建的谓词。后面两节分别展开这两个机制，但在此之前先说清楚 ActPlane 的定位。

## Harness 不只是 Sandbox

沙箱画的是一条隔离边界：边界内一切被允许，边界外一切被禁止。对不可信代码来说这是正确的模型，你不信任它所以把它关在笼子里。但 Agent 不是不可信代码，它是你的协作者，你希望它完成任务，只是希望它在过程中遵守某些约束。

这些约束往往和安全权限无关，却恰恰是 Agent 在真实代码库中自主运行时最需要的规则类型。比如"提交前跑测试"属于工程流程，"用 migration 工具访问 prod.db"属于操作规范，"不要在一个 commit 里混合独立任务"属于工作习惯。沙箱无法表达这些约束，因为它们的语义超出了资源访问的范畴。但 harness 也包含沙箱的能力：当 Agent 运行不可信命令时，你可以写一条规则把整个子树限制为只读、禁网络、或只能访问特定目录。在 ActPlane 中这只是规则的一个子集，可以和工作流类规则写在同一个策略文件里。

反馈回路是 harness 设计中最核心的环节。每当规则触发时，ActPlane 会通过 Agent 框架的 hook 系统将原因反馈给 Agent：

```
🚫 KILLED: process 'git' (pid 4213, ppid 4210) — /usr/bin/git
   effect: kill
   reason: no git under the agent; use the review workflow
```

Agent 收到这个原因后就能理解约束的含义，然后走另一条路完成任务。它不需要事先"记住"不能用 git，它可以先尝试，然后被告知为什么不行以及应该怎么做。这就形成了一个有意思的架构模式：Agent 的推理仍然是概率性的（这正是 LLM 的优势），但关键约束由内核确定性地执行，违规时的反馈让 Agent 自纠而非撞墙。确定性约束与概率性决策两者通过反馈回路衔接，形成了一个兼具灵活性和可控性的架构。

## 核心机制：标签传播

ActPlane 的策略不是静态的允许/拒绝列表，而是标签化的信息流策略（labeled information-flow policies）：给进程和文件打标签，标签沿 fork/exec 边和文件读写边自动传播，规则基于标签做决策。这个模型的学术根基可以追溯到 [CamQuery](https://dl.acm.org/doi/10.1145/3243734.3243776)（CCS 2018）和 [CamFlow](https://dl.acm.org/doi/10.1145/3127479.3129249)（SoCC 2017），它们在内核溯源图上实现了跨通道的 taint 传播与执行。ActPlane 将同样的思路搬到了现代 eBPF/BPF-LSM 基础设施上，不再需要自定义内核模块，面向的也不再是远程攻击者，而是 cooperative-but-forgetful 的 AI Agent。

跟着一个例子走一遍就清楚了。假设我们要约束 Claude Code 的整个进程树，首先声明标签源：`source AGENT = exec "claude"`，意思是每当有进程执行名为 `claude` 的二进制，它就被打上 `AGENT` 标签。标签不是默认产生的，只有显式声明为 source 的行为才生成标签，系统只追踪你关心的数据流。同理，`source SCHEMA_CHANGED = file "src/protocol/**/*.proto"` 声明的是：这些路径的文件被写入时，写入者获得 `SCHEMA_CHANGED` 标签。

标签一旦产生就自动传播。沿进程谱系方向：父进程有 `AGENT` 标签，fork 出的子进程继承，子进程 exec 的新程序也继承。也就是说 Agent 启动 bash，bash 启动 Python，Python 调 subprocess 启动 git，每一层都带着 `AGENT` 标签，标签不会因为进程边界而丢失。沿文件 I/O 方向：带 `AGENT` 标签的进程写入文件 `f`，`f` 也获得 `AGENT` 标签；之后一个没有这个标签的进程读取了 `f`，它也被标记为 `AGENT`。数据通过文件中转跨越进程边界时，标签跟着数据走。

有了标签之后，规则的写法就很直接了。比如 `kill exec "git" "push" if AGENT` 的含义是：带 `AGENT` 标签的进程尝试执行 `git push` 时直接终止它。回到开头那个场景：

```
Claude Code (AGENT) 
  → bash (继承 AGENT)
    → python script.py (继承 AGENT) 
      → subprocess: git push (继承 AGENT) ← 规则在这里触发
```

不管嵌套多深、经过多少进程边界，标签沿谱系一路传播，规则在最终的 exec 调用点执行。检查点从工具入口下沉到内核的 exec 调用，无论 Agent 用什么路径到达都会被覆盖。

文件方向的标签传播则解决另一类问题。假设一个 Agent 从生产数据库导出敏感数据到 `data.csv`，此时进程获得 `DB_SENSITIVE` 标签，文件也同样获得。后续步骤读取 `data.csv` 时读取者也会获得 `DB_SENSITIVE` 标签，如果它再把内容写入日志或发到网络，`block connect if DB_SENSITIVE` 就会阻止这个网络连接。整个数据流跨越了进程和文件边界，但标签始终跟着数据走。

不过内核态追踪的粒度是系统调用级别：ActPlane 知道进程 P 写入了文件 F，但不知道写入了什么内容。这意味着会出现过标记（over-tainting）：比如一个进程读了敏感文件的一行，再写完全无关的数据到另一个文件，第二个文件也会被标记为敏感，学术上叫 label creep。实践中三个因素缓解了这个问题：Agent 的不同任务通常在不同进程树中执行，标签天然隔离；规则匹配的是操作类型加标签的组合（`block connect if DB_SENSITIVE` 只在网络连接时检查）；只有显式声明的 source 才产生标签。这是一个明确的设计权衡：宁可多标记一些，也不漏掉真正的数据流。

## 时序约束：`since` 子句

标签传播解决了"谁干的"和"数据从哪来"这两个问题。但前面提到的很多约束还涉及时间顺序，比如"提交前跑测试"和"改了 spec 后跑 protoc"，它们都需要知道"什么事情在什么事情之后发生过"。ActPlane 用 `since` 子句在事件时间线上推理，表达"在 X 发生之后、Y 发生之前，Z 不被允许"。

把标签和时序放在一起，就能写出表达力很强的策略文件。下面四条规则展示四种约束模式：

```yaml
# actplane.yaml
version: 1
policy: |
  source AGENT = exec "claude"

  # Track when protocol schema files are modified
  source SCHEMA_CHANGED = file "src/protocol/**/*.proto"

  rule no-git-branch:
    kill exec "git" "branch"   if AGENT
    kill exec "git" "worktree" if AGENT
    because "This workspace forbids creating git branches or worktrees.
             Use other git commands, or ask the user to manage branches."

  rule regenerate-after-schema:
    notify exec "git" "commit"
      if SCHEMA_CHANGED unless after exec "protoc" since write "src/protocol/**"
    because "Protocol schema changed — generated code may be stale.
             Run `make proto` to regenerate, then commit."

  rule test-before-commit:
    block exec "git" "commit"
      if AGENT unless after exec "pnpm" "test" since write "src/**"
    because "Source files changed since last test run.
             Run `pnpm test:changed`, then commit."

  rule mediate-proddb:
    block open file "**/prod.db"
      unless lineage-includes exec "**/migrate"
    because "prod.db is reachable only through the migration tool.
             Run `./migrate` to access it."
```

`no-git-branch` 是其中最简单的一条：Agent 进程树中任何尝试 `git branch` 或 `git worktree` 的进程被立即终止，不需要条件判断也不需要时序逻辑。Agent 收到 `because` 里的原因后，就知道应该用其他 git 命令或请用户管理分支。

`regenerate-after-schema` 是一条跨事件的条件规则，使用 notify 效果。它的 `unless` 子句要回答的问题是：自上次有进程写入 protocol 目录以来，是否有进程执行过 `protoc`？执行过就放行 commit，没有就提醒 Agent。关键在于 `since` 子句的动态性：每当 protocol 目录再次被写入，"已跑过 protoc"的状态被重置，必须重新跑。这是一个在事件时间线上动态更新的谓词，不是一次性的静态查询。

`test-before-commit` 的语义类似但更严格，使用 block 效果直接阻止操作：自上次写入 `src/**` 以来是否执行过 `pnpm test`？没有就不允许 commit。每次修改源文件都会重置测试状态，即使跑完测试后又改了一行代码也必须再跑一次。

`mediate-proddb` 和前三条不同，它不依赖标签，而是用 `lineage-includes` 检查进程谱系：想打开 `prod.db` 的进程，祖先链里必须包含 `migrate`。这条规则表达的是"唯一合法的访问路径"这个概念，也就是说 Agent 直接 `open("prod.db")` 会被拦住，但如果先调 `./migrate` 再由 migrate 内部打开就没问题。

传统沙箱既没有"时间"也没有"路径"的概念，只知道当前瞬间的状态。相比之下，ActPlane 既维护事件时间线让规则在时间维度上推理，也追踪进程谱系让规则检查执行路径，因此能表达工作流约束和强制中介，而不只是访问控制。

## Agent 集成

ActPlane 通过 Agent 框架的 hook 系统将规则匹配的原因反馈给 Agent。内核始终是观测和执行的唯一权威源，hook 只负责将匹配事件传递到 Agent 的决策上下文中。

Claude Code 的集成通过 `.claude/settings.local.json` 配置 `PostToolUse` 和 `PostToolUseFailure` 两个 hook：

```json
{
  "hooks": {
    "PostToolUse": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "actplane feedback-hook" }] }],
    "PostToolUseFailure": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "actplane feedback-hook" }] }]
  }
}
```

Codex 的集成通过 `.codex/hooks.json`：

```json
{
  "hooks": {
    "PostToolUse": [{ "matcher": ".*", "hooks": [{ "type": "command", "command": "actplane feedback-hook" }] }]
  }
}
```

`actplane feedback-hook` 的逻辑很简单：它检查自上次调用以来是否有新的规则匹配事件，如果有就把 `because` 内容转发到 hook 输出。Agent 每次工具调用后运行这个 hook，规则触发的原因就进入下一步决策的上下文。整个集成不需要修改 Agent 框架本身。

## 运行时架构

ActPlane 在架构上分为内核态和用户态两部分：

```
actplane.yaml ─▶ compiler (Rust) ─▶ .rodata config ─▶ eBPF kernel engine
 policy: |        parse + lower DSL    (set_global)      propagate labels,
                                                          match rules,
 matches ◀─────── ring buffer (in-process, via aya) ◀─── emit on match only
```

内核部分（`bpf/` 目录）为每个节点（进程、文件、网络端点）维护标签集，执行传播，评估编译后的规则，只在匹配时通过 ring buffer 向用户态发出事件。未匹配的操作完全不产生用户态开销，这对性能至关重要：一个活跃的 Agent 每秒可能触发数百次文件操作和进程创建，如果每个操作都通知用户态做决策，延迟不可接受。标签传播和规则匹配都在内核空间完成，用户态只在规则触发时参与。

用户态部分是 `actplane` 这个 Rust 二进制。eBPF 程序预编译为 CO-RE（Compile Once, Run Everywhere）格式嵌入其中，因此安装时不需要 clang、llvm、libbpf 或任何编译工具链。部署路径是 `cargo install actplane` → `actplane init` 生成 starter 配置 → `actplane check` 验证规则 → `sudo actplane run <command>` 在 harness 下执行 Agent。eBPF 程序经过内核验证器检查，保证不会崩溃内核或死循环。

运行时通过 [aya](https://github.com/aya-rs/aya) 在进程内加载预编译的 eBPF 对象，解析 `actplane.yaml` 并将 DSL 编译为内核配置（写入 `.rodata` 段），设定目标进程的谱系种子，然后监听 ring buffer。和 [Cilium Tetragon](https://tetragon.io/) 相比，Tetragon 的 `matchBinaries` + `followChildren` 可以沿 fork/exec 传播谱系标记，是目前最接近 ActPlane 谱系追踪的开源功能，但 Tetragon 只沿进程边传播，不跨文件和网络边，也不提供语义反馈给 Agent。

在权限方面，`actplane run` 和 `actplane watch` 需要 root 或 `CAP_BPF` + `CAP_SYS_ADMIN` 来加载 eBPF 引擎，但加载完成后目标命令会降权回当前用户运行。而 `actplane check` 完全不需要特权，只做规则的静态验证。

## 适用场景与局限

当多个 Agent 跨厂商协作时，内核级约束的优势最为明显。比如 Claude Code 调用 Codex，Codex 又调用自定义工具链，每个厂商的框架级守卫只了解自己注册的工具，Claude Code 的 hook 不知道 Codex 的权限配置，反过来也一样。框架级守卫的设计假设是"我知道 Agent 会通过哪些路径操作系统"，而跨厂商调用一出现这个假设就不成立了。OS 级规则则不同，它沿进程谱系传播，完全不关心下面跑的是谁的运行时，一条规则就能管住整个跨厂商执行树。

CI/CD 环境中 Agent 的约束要求更严格，因为构建流水线里不能推代码、不能改 CI 配置、必须测试通过才能产出构建产物，而这些时序约束正是 `since` 子句擅长表达的。在涉及敏感数据的部署场景中，Agent 还需要数据流级别的策略，比如"从 prod.db 读出来的数据不能流向网络"，传统沙箱的粒度无法追踪这种跨进程的数据流转，而标签传播恰好能够覆盖这类需求。

当然 ActPlane 也有它明确的适用边界。由于它基于 eBPF，只能运行在 Linux 5.8 以上且具备 BTF 支持（`/sys/kernel/btf/vmlinux`）的内核上，macOS 和 Windows 上的 Agent 开发场景目前覆盖不了，虽然多数生产部署确实在 Linux。加载 eBPF 程序需要 root 或 `CAP_BPF` + `CAP_SYS_ADMIN`，某些共享服务器和云容器环境拿不到这个权限。在追踪粒度上，内核态只看到系统调用层面的操作，进程内部的内存计算和加密解密不在视野内。此外 block 模式依赖 BPF-LSM，而 BPF-LSM 并非所有发行版默认开启。

## 小结

回到开头那个场景：Agent 写了一个 Python 脚本调 `subprocess.run(["git", "push"])`。在 ActPlane 下，`AGENT` 标签沿进程谱系从 Claude Code 传播到 bash、传播到 Python、传播到三层深处那个 git，规则触发后操作被拦截，Agent 收到原因和替代方案。Prompt 层面没拦住的，内核层面拦住了。

Agent 的价值在于灵活性和创造性，而部署 Agent 需要的却是可预测性和安全保证。Prompt 终究只是建议，工具层守卫一个 shell out 就能绕过，沙箱也只做 allow/deny 的资源隔离。ActPlane 在内核加了一层确定性约束，让 Agent 仍然自由推理，但关键操作由信息流规则裁决，触发约束时 Agent 拿到的是可以立即行动的反馈。它不替代前三层，而是补上它们各自的盲区。

在复杂系统中，任何单层约束都有缺口，而 Agent 天然会找到穿过去的路，所以分层约束可能是 Agent 走向生产部署的一个必要架构组件。

---

> **GitHub**: [github.com/eunomia-bpf/ActPlane](https://github.com/eunomia-bpf/ActPlane) — MIT 协议
>
> ActPlane 是 [eunomia-bpf](https://github.com/eunomia-bpf) 社区的开源项目，基于 [AgentSight](https://github.com/eunomia-bpf/agentsight/) 的 eBPF 观测基础设施构建。
