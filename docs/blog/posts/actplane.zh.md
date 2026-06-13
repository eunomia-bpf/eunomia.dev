---
date: 2026-05-31
description: ActPlane 是一个基于 eBPF 的 AI Agent 策略引擎，在操作系统内核层面对 Agent 行为做观测和强制执行。本文分析 prompt、工具层、沙箱三层约束各自的系统性盲区，说明 ActPlane 如何通过标签传播和时序谓词实现确定性的 Agent harness。
---

# ActPlane: 为 AI Agent Harness 设计的系统级策略引擎

你在 CLAUDE.md 里写了一条规则："不要执行 `git push`"。Agent 遵守了。然后它写了一个 Python 脚本，脚本里调用 `subprocess.run(["git", "push"])`。Prompt 约束从未被违反，但代码已经推到了远端。

这不是假设场景。Claude Code、Codex、OpenHands 这些 Agent 已经在真实项目里跑了。一次任务几十上百次工具调用，生成代码、执行测试、改文件、连网络，全是常规操作。我们试图用 CLAUDE.md 告诉它"不要做什么"，用 MCP gateway 拦住危险的工具调用，用容器把它关在沙箱里。每一层都有用，但每一层都有洞。而 Agent 不需要恶意就能找到穿越这些洞的路径，它只是在尝试完成你交给它的任务。

我们造了一个工具来堵这些洞。

[ActPlane](https://github.com/eunomia-bpf/ActPlane) 是一个基于 eBPF 的策略引擎，坐在所有工具层和沙箱的下面，直接在操作系统内核层面做观测和执行。Prompt 约束是概率性的，Agent 可能听也可能不听。ActPlane 是确定性的，规则匹配了就一定执行。而且它不只是拦住操作——当约束触发时，Agent 收到的是人类可读的反馈，告诉它为什么被拦、该怎么做。

每条规则选择自己的执行力度：**notify** 只提醒 Agent，不干预执行；**block** 在操作提交前拦住它；**kill** 直接终止进程。三种力度，同一个策略文件里可以混用。而在所有模式下，规则触发的原因都会反馈给 Agent，让它能自己纠正。

> 链接: https://github.com/eunomia-bpf/ActPlane

<!-- more -->

## 先看它能做什么

在拆解问题之前，先看几个我们实际想表达的约束。这些约束有一个共同点：你用 prompt 说不清楚，用 MCP gateway 拦不住，用容器沙箱也表达不了。

第一个场景：你想禁止 codex 执行 `git push`，也不允许它写 `/src` 之外的文件。听起来像一条普通的沙箱规则。但"禁止执行"到底覆盖多大范围？Agent 可能直接调 git，也可能写一个 shell 脚本再执行它，也可能生成一个 Python 程序，Python 里面嵌套了 subprocess 调用。我们需要的不是在工具 API 层检查"Agent 有没有调用 git 工具"，而是沿整个进程树检查"有没有一个属于 codex 的子进程执行了 git"。两个问题看似相似，覆盖面完全不同。

第二个场景：修改了 `specs/*` 之后，必须同步更新 server、SDK 和文档。我们不想阻止 Agent 编辑 spec 文件，那是它正常工作的一部分。但当 Agent 改完 spec 之后想 `git commit`，它应该收到一条提醒："generated code may be stale, run `make proto` to regenerate." 注意这里的意图不是阻止提交，而是提醒还有一步没做完。Agent 收到提醒后自己决定是否补跑代码生成。

第三个场景：提交代码前必须跑测试。这条听起来简单，但关键在于它不是一次性检查。每次 Agent 修改 `src/` 下的文件，之前的测试结果应该自动失效，必须重新跑才能提交。追踪的不是"测试是否曾经跑过"，而是"自上次修改源文件以来测试是否跑过"。跑完测试又改了一行代码？测试状态重置，再跑一次。

第四个场景：生产数据库 `prod.db` 只能通过 migration 工具访问，Agent 不能直接打开它。这是一条强制中介（mandatory mediation）约束：不管 Agent 怎么到达文件打开调用，只要它的进程祖先链里没有执行过 `migrate` 工具，操作就应该被阻止。检查的不是"Agent 有没有某个权限"，而是"Agent 是不是从正确的入口进来的"。

回头看这四个场景，它们的共同特征很明显：都不是静态的 allow/deny，都涉及进程谱系、操作时序、或强制中介。要理解为什么需要内核级方案，先看看现有三层约束各自在哪里漏了。

## 三层约束各自的盲区

| 方案 | 它做了什么 | 它覆盖不了什么 |
|------|----------|-------------|
| **Prompt 约束**（CLAUDE.md、AGENTS.md） | 告诉 Agent 该做什么、不该做什么 | 概率性的：长上下文中 Agent 会遗忘或非恶意地绕过 |
| **工具层守卫**（MCP gateway、AgentSpec） | 在工具 API 层面拦截和授权 | Agent shell out、链接 SDK、spawn 子进程时完全绕过 |
| **沙箱**（容器、VM、E2B、Daytona） | 隔离整个执行环境 | 全有或全无：无法表达"文件 A 只能通过脚本 A 访问"或"提交前跑测试" |

### Prompt 约束

CLAUDE.md、AGENTS.md、system prompt 是目前最普遍的 Agent 约束方式。它们本质上是自然语言指令，依赖模型在推理时"记住并遵守"。

问题出在"记住"这两个字上。当对话超过数万 token，早期约束在注意力分配中被后续的用户指令和工具输出稀释。模型没有忘记规则，但规则在决策权重中的占比下降了。一个在对话第 3 轮被严格遵守的约束，到第 30 轮可能被创造性地重新解释。我们在 [AgentCgroup 的刻画实验](agentcgroup-characterization.md)中观察到，一个任务内 Agent 平均执行几十次工具调用，整个对话上下文可以增长到非常大。约束越早设定、对话越长，约束被有效遵守的概率越低。

比"忘记"更难防的是间接违反。Agent 被告知"不要删除文件"，于是它写了一个 Makefile target，里面包含 `rm -rf`，然后调用 `make clean`。从 Agent 的视角看，它执行的是一个构建命令，不是删除操作。这不是恶意绕过。Agent 确实在尝试完成任务，只是找到了一条不经过约束表面的路径。在多步推理中，每个决策点是局部合理的，但全局行为链可能违反最初设定的策略。

根本问题是：prompt 约束约束的是 Agent 的意图表达，不是实际行为。Agent 可以在不"违反"任何 prompt 规则的情况下产生违规的副作用。

### 工具层守卫

MCP gateway、[AgentSpec](https://arxiv.org/abs/2503.18666)、[Progent](https://arxiv.org/abs/2504.11703) 等工具级权限控制在约束表面上更靠近实际操作。如果 Agent 调用 `file_write("/etc/passwd", ...)`，工具层可以在 API 入口拒绝。这比 prompt 约束可靠得多，因为它检查的是操作本身而不是意图。

但 Agent 不只通过注册工具操作系统。问题出在 shell。

几乎所有主流 Agent 框架都提供某种形式的 shell 或 terminal 工具。这是必要的：Agent 需要运行构建命令、执行测试、调用第三方 CLI。但一旦 Agent 拿到了 bash，工具层守卫就被架空了。Agent 写一个 Python 脚本并执行它，脚本内部 `import subprocess; subprocess.run(["curl", "-X", "POST", ...])` 把数据发送到外部网络。工具层看到的是 Agent 调用了 `Bash` 工具执行 `python script.py`，对脚本内部、对脚本启动的子进程、对子进程的子进程做了什么完全不知道。整个调用链可能有三四层深，工具层只看到第一层。

还有一种更隐蔽的绕过：Agent 生成的代码直接链接数据库驱动、HTTP 客户端库或文件系统 API。这些操作发生在 Agent 生成的代码运行时，根本不经过任何工具调用路径。工具层守卫对它们完全透明。

根本问题是覆盖面。工具层只能看到 Agent 通过注册工具发起的操作。Agent 通过代码执行产生的副作用，全部在工具层的视野之外。

### 沙箱

容器、VM、[E2B](https://github.com/e2b-dev/E2B)、[Daytona](https://github.com/daytonaio/daytona) 用隔离边界包裹整个执行环境。这是目前最可靠的安全边界：沙箱内的进程无法访问沙箱外的资源。对于"防止 Agent 逃逸到宿主机"这个问题，沙箱是正确的答案。

但 Agent 在实际工作中需要的约束远比"能不能访问某个资源"丰富。

"Agent 修改了 `.proto` 文件后，必须先运行 `protoc` 再提交。" 这是一个时序约束：操作 B 必须在操作 A 之后、操作 C 之前发生。沙箱没有时间的概念，它只知道当前瞬间哪些资源可以访问。"从数据库读取的敏感数据不能写入日志文件。" 这需要追踪数据从哪里来、流向哪里。沙箱的粒度是进程级或容器级，它不知道进程内部读了什么、写了什么。同一个 `git commit` 命令，在"Agent 刚跑完测试"和"Agent 还没跑测试"两种情况下应该有不同的策略。沙箱无法根据历史操作上下文做区分。

还有一个被低估的问题：反馈。沙箱拒绝一个操作时，Agent 收到的通常是 `Permission denied` 或 `EPERM`。它不知道为什么被拒绝，不知道怎么做才能满足约束。我们观察到的典型结果是 Agent 反复重试同样的操作三五次，然后放弃整个任务。或者更糟：Agent 尝试用另一条路径绕过，这条路径可能引入新的问题。沙箱是一堵不透明的墙，Agent 撞上去之后除了后退没有别的信息。

根本问题是表达力。沙箱回答的是"这个进程能否访问这个资源"。Agent 需要回答的问题远比这丰富：在什么条件下、以什么顺序、基于什么数据流历史、带什么上下文，这个操作是否被允许？

## 在内核层面执行策略

三层约束的盲区指向同一个方向：约束机制需要下沉到所有操作的必经之路。[AgentSight](https://arxiv.org/abs/2508.02736)（2025）用 eBPF 同时捕获 Agent 的意图层和行为层数据，提出了 Agent 意图和系统级实际行为之间的"semantic gap"概念。ActPlane 就是在这个观测基础上加了执行力。Agent 用什么工具、写什么脚本、spawn 多少层子进程，最终所有副作用都要经过操作系统内核。每一次 exec、每一次文件 open、每一次网络 connect、每一次 fork，内核都在场。ActPlane 在这个层面工作：通过 eBPF 在内核中安装轻量级程序，hook 住 Agent 可能产生副作用的所有路径，然后根据标签化的信息流策略做决策。

具体来说，ActPlane hook 了进程的整个生命周期（`sched_process_fork`、`sched_process_exec`、`sched_process_exit`），文件操作（`sys_enter_openat`、`sys_enter_unlinkat`、`sys_enter_renameat2`），以及网络连接（`sys_enter_connect`）。Agent 无论通过什么路径产生副作用，最终都会经过这些系统调用。

这和前面三层的本质区别在哪里？回到开头的场景。Agent 写了一个 Python 脚本，脚本 spawn 了一个子进程调 git push。工具层只看到 `python script.py` 这一层，ActPlane 看到整个进程树的所有系统调用，包括三层深处那个 git。而且约束不是挂在"git 工具"上的，而是挂在进程树上的。"Codex 的整个子进程树不能碰 git"只需要一条规则，不用在每个可能的工具入口重复设防。

但 ActPlane 做的不只是把工具层守卫下沉到内核。它还引入了两个工具层和沙箱都没有的能力：**数据流追踪**和**时序推理**。标签可以跨越 fork/exec 和文件读写边界传播，使得"从 A 读取的数据不能流向 B"成为可表达的策略；`since` 子句让规则在事件时间线上动态更新，使得"自上次修改源文件以来是否跑过测试"成为一个会随新事件不断失效和重建的谓词。后面两节分别展开这两个机制。不过在此之前，先说清楚 ActPlane 的定位：它不只是一个更深的沙箱。

## Harness 不只是 Sandbox

沙箱画的是一条隔离边界。边界内的一切都被允许，边界外的一切都被禁止。对不可信代码来说这是正确的模型：你不信任这段代码，所以把它关在一个笼子里。但 Agent 不是不可信代码。Agent 是你的协作者，你希望它完成任务，只是希望它在完成任务的过程中遵守某些约束。

沙箱回答的问题是"这个进程能否访问这个资源"。Harness 回答的问题范围更广。不只是安全问题（"敏感数据不能到达网络"），还有软件工程纪律。"提交前跑测试"不是安全约束，是工程流程。"不要在一个 commit 里混合独立任务的数据"不是权限问题，是工作习惯。"用 migration 工具访问 prod.db，不要直接连"不是隔离问题，是操作规范。这些工作流约束正是 Agent 在真实代码库中自主运行所需要的规则类型。沙箱无法表达它们，因为它们不是 allow/deny 的资源访问问题。

同时 harness 也包含了沙箱的能力。当 Agent 生成一个子 Agent 或者运行不可信命令时，你可以写一条规则把整个子树限制为只读、禁网络、或只能访问特定目录。这就是传统沙箱做的事情，但在 ActPlane 中它只是规则的一个子集。你可以在同一个策略文件里同时写沙箱类规则（"这个子进程树不能联网"）和工作流类规则（"提交前必须跑测试"）。

反馈回路是 harness 设计中最核心的环节。当规则触发时，ActPlane 通过 Agent 框架的 hook 系统将原因反馈给 Agent：

```
🚫 KILLED: process 'git' (pid 4213, ppid 4210) — /usr/bin/git
   effect: kill
   reason: no git under the agent; use the review workflow
```

Agent 收到这个原因，理解约束，然后走另一条路完成任务。它不需要"记住"不能用 git，它可以尝试，然后被告知为什么不行、应该怎么做。这种"确定性约束 + 概率性决策"是一个有意思的架构模式：Agent 的推理仍然是概率性的（这正是 LLM 的优势），但关键约束由内核确定性地执行，违规时的反馈让 Agent 能够自纠而不是撞墙。

## 核心机制：标签传播

ActPlane 的策略不是静态的允许/拒绝列表。它用的是标签化的信息流策略（labeled information-flow policies）：给进程和文件打标签，标签沿 fork/exec 边和文件读写边自动传播，规则基于标签做决策。这个模型的学术根基可以追溯到 [CamQuery](https://dl.acm.org/doi/10.1145/3243734.3243776)（CCS 2018）和 [CamFlow](https://dl.acm.org/doi/10.1145/3127479.3129249)（SoCC 2017），它们在内核溯源图上实现了跨通道的 taint 传播与执行。ActPlane 将同样的思路搬到了现代 eBPF/BPF-LSM 基础设施上，不再需要自定义内核模块，面向的也不再是远程攻击者，而是 cooperative-but-forgetful 的 AI Agent。听起来抽象，但跟着一个例子走一遍就很清楚。

假设我们想约束 Claude Code 的整个进程树。首先声明一个标签源：`source AGENT = exec "claude"`。这行的意思是，每当系统中有进程执行了名为 `claude` 的二进制，这个进程就被打上 `AGENT` 标签。标签不是默认产生的，只有你显式声明为 source 的行为才会生成标签，系统不追踪所有数据流，只追踪你关心的。同理，`source SCHEMA_CHANGED = file "src/protocol/**/*.proto"` 声明的是：当这些路径的文件被写入时，写入者获得 `SCHEMA_CHANGED` 标签。

标签一旦产生就会自动传播。沿进程谱系方向：父进程有 `AGENT` 标签，fork 出的子进程继承它，子进程 exec 的新程序也继承它。Agent 启动 bash，bash 启动 Python，Python 调用 subprocess 启动 git，每一层都带着 `AGENT` 标签。标签不会因为进程边界丢失。沿文件 I/O 方向：带 `AGENT` 标签的进程写入文件 `f`，`f` 也获得 `AGENT` 标签；之后一个原本没有这个标签的进程读取了 `f`，它也被标记为 `AGENT`。数据通过文件中转跨越进程边界时，标签跟着数据走。

有了标签，规则就很直接了。`kill exec "git" "push" if AGENT`：如果一个带 `AGENT` 标签的进程尝试执行 `git push`，终止它。回到开头那个让人头疼的场景：

```
Claude Code (AGENT) 
  → bash (继承 AGENT)
    → python script.py (继承 AGENT) 
      → subprocess: git push (继承 AGENT) ← 规则在这里触发
```

不管嵌套多深、经过多少进程边界，标签沿进程谱系一路传播，规则在最终的 exec 调用点执行。这就是 ActPlane 解决 shell escape 问题的方式：不是在工具入口检查，而是在内核的 exec 调用点检查，无论 Agent 用什么路径到达这里。

文件方向的标签传播解决的是另一类问题。假设一个 Agent 从生产数据库导出敏感数据到 `data.csv`（进程获得 `DB_SENSITIVE` 标签，文件也获得这个标签），然后另一个 Agent 或者同一个 Agent 的后续步骤读取 `data.csv`（读取者获得 `DB_SENSITIVE` 标签），再把内容写入日志或者发送到网络。这时候规则 `block connect if DB_SENSITIVE` 会阻止这个进程发起网络连接。整个数据流跨越了进程边界和文件边界，但标签始终跟着数据走。

不过内核层的追踪粒度是系统调用级别的。ActPlane 知道进程 P 写入了文件 F，但不知道写入了什么内容。这意味着会出现过标记（over-tainting）：一个进程读取了敏感文件的第一行（文件头），然后写入完全无关的数据到另一个文件，第二个文件也会被标记为敏感，因为写入者的进程曾经读过敏感文件。这在学术文献中叫 label creep。实践中这个问题被三个因素缓解：Agent 的不同任务通常在不同进程树中执行所以标签天然隔离；规则匹配的是操作类型加标签的组合（`block connect if DB_SENSITIVE` 只在网络连接时检查，不会因为标签存在就阻止文件读写）；只有显式声明的 source 才产生标签。这是一个明确的设计权衡：宁可多标记一些，也不漏掉真正的数据流。

## 时序约束：`since` 子句

标签传播解决了"谁干的"和"数据从哪来"的问题。但前面提到的很多约束还涉及时间顺序："提交前跑测试"、"改了 spec 后跑 protoc"。这些约束需要知道"什么事情在什么事情之后发生过"。ActPlane 用 `since` 子句来做这件事：规则可以在事件时间线上推理，表达"在 X 发生之后、Y 发生之前，Z 不被允许"。

把标签和时序放在一起，就能写出表达力很强的策略文件。下面是一个完整的例子，四条规则展示四种不同的约束模式：

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

四条规则，四种约束模式。

`no-git-branch` 是最简单的 per-event 规则，用 kill 效果。Agent 进程树中任何尝试 `git branch` 或 `git worktree` 的进程都被立即终止。不需要条件判断，不需要时序逻辑。Agent 被终止后收到原因："This workspace forbids creating git branches or worktrees. Use other git commands, or ask the user to manage branches." 这条原因是 `because` 子句的内容，Agent 读懂之后知道应该用其他 git 命令或者请用户来管理分支。

`regenerate-after-schema` 是跨事件的条件规则，用 notify 效果。解读这条规则：如果有进程修改了 `src/protocol/**/*.proto`（触发 `SCHEMA_CHANGED` 标签），然后尝试 `git commit`，ActPlane 检查 `unless after exec "protoc" since write "src/protocol/**"`。这个条件的意思是：自上次有进程写入 protocol 目录以来，是否有进程执行过 `protoc`？如果执行过，条件满足，commit 放行。如果没有执行过，触发 notify，Agent 收到提醒说"generated code may be stale, run `make proto`"。关键在 `since` 子句：每当 protocol 目录再次被写入，"已经跑过 protoc"的状态就被重置，必须重新跑。这是一个在事件时间线上动态更新的谓词，不是一个静态检查。

`test-before-commit` 是带动态失效的时序规则，用 block 效果。语义和上面类似但更严格：自上次有进程写入 `src/**` 以来，是否执行过 `pnpm test`？没有的话在 commit 执行前就阻止。每次修改 `src/` 下的文件都重置测试状态。Agent 跑完测试之后又改了一行代码，必须再跑一次测试才能提交。

`mediate-proddb` 是强制中介规则，和前面三条不同，它不依赖标签，而是用 `lineage-includes` 检查进程谱系。任何进程想打开 `prod.db`，ActPlane 检查它的祖先链里是否执行过 `migrate` 工具。执行过就放行，没执行过就阻止。这条规则表达的是"唯一合法的访问路径"：不是检查你是谁、你带什么标签，而是检查你是不是从正确的入口进来的。Agent 直接 `open("prod.db")` 会被拦住；Agent 先调用 `./migrate`，migrate 内部再打开 `prod.db`，就没问题。

传统沙箱没有"时间"的概念，也没有"路径"的概念。它只知道当前瞬间的状态，不知道之前发生过什么，也不知道你是从哪条路径到达这里的。ActPlane 既维护事件时间线让规则在时间维度上推理，也追踪进程谱系让规则检查执行路径。这使得它能表达工作流约束和强制中介，而不只是访问控制。

## Agent 集成

ActPlane 通过 Agent 框架的 hook 系统将规则匹配的原因反馈给 Agent。内核始终是观测和执行的唯一权威源。hook 只负责将匹配事件传递到 Agent 的决策上下文中。

Claude Code 的集成通过 `.claude/settings.local.json` 配置 `PostToolUse` 和 `PostToolUseFailure` 两个 hook：

```json
{
  "hooks": {
    "PostToolUse": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "actplane feedback-hook" }] }],
    "PostToolUseFailure": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "actplane feedback-hook" }] }]
  }
}
```

Codex 的集成通过 `.codex/hooks.json` 配置：

```json
{
  "hooks": {
    "PostToolUse": [{ "matcher": ".*", "hooks": [{ "type": "command", "command": "actplane feedback-hook" }] }]
  }
}
```

`actplane feedback-hook` 这个适配器做的事情很简单：检查自上次调用以来是否有新的规则匹配事件，如果有就把事件的原因（也就是规则的 `because` 内容）转发到 hook 的输出。Agent 在每次工具调用后都会运行这个 hook，如果有规则被触发，原因就被纳入 Agent 下一步决策的上下文。整个集成不需要修改 Agent 框架本身，只需要配置 hook。

## 运行时架构

ActPlane 的运行时分为内核态和用户态两部分。

```
actplane.yaml ─▶ collector (Rust) ─▶ .rodata config ─▶ eBPF kernel engine
 policy: |        parse + lower DSL    (set_global)      propagate labels,
                                                          match rules,
 matches ◀─────── ring buffer (in-process, via aya) ◀─── emit on match only
```

内核部分（`bpf/` 目录）hook 了 fork、exec、exit、open、unlink、rename、connect 这些系统调用，为每个节点（进程、文件、网络端点）维护一个标签集（per-node label set），执行标签传播，评估编译后的规则，只在规则匹配时通过 ring buffer 向用户态发出事件。未匹配的操作完全不产生用户态开销。这一点对性能至关重要：我们在 AgentCgroup 的刻画中看到，一个活跃的 Agent 每秒可能触发数百次文件操作和进程创建。如果每个操作都要通知用户态做决策，延迟会变得不可接受。ActPlane 把标签传播和规则匹配都放在内核空间完成，用户态只在规则触发时才参与。

用户态部分是 `actplane` 这个 Rust 二进制。eBPF 程序预编译为 CO-RE（Compile Once, Run Everywhere）格式嵌入其中，安装时不需要 clang、llvm、libbpf 或任何编译工具链。整个部署路径是 `cargo install actplane`，然后 `actplane init` 生成 starter 配置，`actplane check` 验证规则，`sudo actplane run <command>` 在 harness 下执行 Agent。eBPF 程序经过内核验证器检查，保证不会崩溃内核或死循环。

运行时，用户态通过 [aya](https://github.com/aya-rs/aya) 在进程内加载预编译的 eBPF 对象，解析 `actplane.yaml` 并将 DSL 编译为内核配置（写入 `.rodata` 段），设定目标进程的谱系种子（告诉内核"从这个进程开始追踪"），然后监听 ring buffer 报告规则匹配及其策略原因。和 [Cilium Tetragon](https://tetragon.io/) 相比，Tetragon 的 `matchBinaries` + `followChildren` 可以沿 fork/exec 传播一个二进制谱系标记，是目前开源工具里最接近 ActPlane 谱系追踪的功能。但 Tetragon 只沿进程边传播，不跨文件和网络边，也不提供语义反馈给 Agent。

权限方面：`actplane run` 和 `actplane watch` 需要 root 或者 `CAP_BPF` + `CAP_SYS_ADMIN` 来加载 eBPF 引擎。但加载完成后，目标命令会被降权回当前用户运行。Agent 本身不以 root 身份执行。`actplane check` 完全不需要特权，因为它不加载任何 eBPF 程序，只做规则的静态验证。

## 适用场景与局限

ActPlane 发挥价值最大的场景有一个共同特征：单一层面的约束在那里不够用。

跨厂商的多 Agent 协作是最典型的。当 Claude Code 调用 Codex，Codex 又调用自定义工具链时，每个厂商的框架级守卫只了解自己注册的工具。Claude Code 的 hook 不知道 Codex 的权限配置，反过来也一样。框架级守卫假设的是"我知道 Agent 会通过哪些路径操作系统"，跨厂商调用一出现这个假设就垮了。但 OS 级规则沿进程谱系传播，不关心下面跑的是谁的运行时，一条规则管住整个跨厂商执行树——这正是前面 harness 区别于 sandbox 的关键所在。

CI/CD 里的 Agent 治理也是一个强场景。Agent 在 CI 环境中跑的时候约束要更严格：不能推代码、不能改 CI 配置、必须测试通过才能构建产物。这些时序约束正是 `since` 子句做的事情。在敏感环境中部署的 Agent 则需要数据流级别的策略，比如"从 prod.db 读出来的数据不能流向网络"。沙箱做不到这种粒度的追踪，标签传播可以。

但 ActPlane 不是万能的。它基于 eBPF，只跑在 Linux 上，需要 5.8 以上的内核和 BTF 支持（`/sys/kernel/btf/vmlinux`）。macOS 和 Windows 上的 Agent 开发场景覆盖不了，虽然多数生产部署确实在 Linux。加载 eBPF 程序需要 root 或 `CAP_BPF` + `CAP_SYS_ADMIN`，某些共享服务器和云容器里拿不到这个权限。内核态的追踪只到系统调用粒度，进程内部的内存操作、加密解密不在视野里。block 模式依赖 BPF-LSM，不是所有发行版默认开。

## 小结

Agent 的价值在于灵活性和创造性，部署 Agent 需要的是可预测性和安全保证。这两件事之间有张力。Prompt 是建议不是规则，工具层守卫一个 shell out 就绕过了，沙箱只能做 allow/deny 的资源隔离。

ActPlane 在内核加了一层确定性约束。Agent 仍然自由推理，但关键操作由信息流规则裁决。触发约束时 Agent 拿到的不是错误码，而是可以立即行动的反馈。它不替代前三层，而是补上它们各自的盲区。

回到开头那个场景：Agent 写了一个 Python 脚本调 `subprocess.run(["git", "push"])`。在 ActPlane 下，`AGENT` 标签沿进程谱系从 Claude Code 传播到 bash、传播到 Python、传播到三层深处那个 git——规则触发，操作被拦截，Agent 收到原因和替代方案。Prompt 没拦住的，内核拦住了。

在复杂系统中，任何单层约束都有洞，Agent 天然会找到穿过去的路。分层约束可能是 Agent 走向生产的一个必要架构组件。

---

> **GitHub**: [github.com/eunomia-bpf/ActPlane](https://github.com/eunomia-bpf/ActPlane) — MIT 协议
>
> ActPlane 是 [eunomia-bpf](https://github.com/eunomia-bpf) 社区的开源项目，基于 [AgentSight](https://github.com/eunomia-bpf/agentsight/) 的 eBPF 观测基础设施构建。
