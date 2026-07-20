# Published Media Ledger

Last checked: 2026-07-19

This ledger records platform posts confirmed from logged-in profile pages or explicit repository references. It is intentionally conservative: if authorship or completeness is not verified, the item is marked for follow-up instead of treated as complete.

Canonical machine-readable source: one JSON file per platform under [`platforms/`](platforms/), with source rules in [`sources.json`](sources.json). Run `python .github/publisher/media/check_media_ledger.py --show-missing` to count missing source coverage per platform.

## Zhihu

Profile checked: <https://www.zhihu.com/people/yun-wei-64-11/posts>

The profile reports `文章111`. The following recent articles were visible on the article page during the check:

Follow-up platform check on 2026-07-19 used normal in-app browser scrolling on the Zhihu profile page. The page still reported `文章111`; scrolling collected 112 visible article links, which is recorded in [`platforms/zhihu.json`](platforms/zhihu.json). Do not use Zhihu API or hidden/internal endpoints for future checks.

| Status | Title | URL | Notes |
| --- | --- | --- | --- |
| Confirmed | ActPlane: 把 Agent Harness Enforcement 下沉到内核 eBPF | <https://zhuanlan.zhihu.com/p/2049849241200235065> | Recent profile article |
| Confirmed | ACRFence：防止 AI Agent 检查点恢复中的语义回滚攻击 | <https://zhuanlan.zhihu.com/p/2044683685539139618> | Matches `docs/blog/posts/agent-check-restore-safety.md` topic |
| Confirmed | 基于 eBPF 的不透明 AI Agent 运行时可观测与执行控制：超越沙箱与审批 | <https://zhuanlan.zhihu.com/p/2044679839769159474> | Matches `docs/blog/posts/runtime-security-for-opaque-ai-agents.md` topic |
| Confirmed | eBPF 教程：使用 BPF struct_ops 扩展内核子系统 | <https://zhuanlan.zhihu.com/p/1994531717411782754> | Recent profile article |
| Confirmed | eBPF 示例：使用 CUPTI 构建 GPU 火焰图分析器 | <https://zhuanlan.zhihu.com/p/1968559507836761428> | Recent profile article |
| Confirmed | GPU可观测性差距：为什么我们需要GPU上的eBPF | <https://zhuanlan.zhihu.com/p/1962141761364301211> | Recent profile article |
| Confirmed | eBPF 实例教程：使用内核跟踪点监控 GPU 驱动活动 | <https://zhuanlan.zhihu.com/p/1959268707030247127> | Recent profile article |
| Confirmed | eBPF 实例教程：跟踪 Intel NPU 内核驱动操作 | <https://zhuanlan.zhihu.com/p/1958786450960131249> | Recent profile article |
| Confirmed | eBPF 教程：结合 On-CPU 和 Off-CPU 分析的挂钟时间分析 | <https://zhuanlan.zhihu.com/p/1957046030014066935> | Recent profile article |
| Confirmed | 让 AI Agent 的一举一动都在掌控之中，基于 eBPF 的系统级可观测性工具 | <https://zhuanlan.zhihu.com/p/1943735211373363397> | Recent profile article |
| Confirmed | 系统会议中的可观测性、性能分析和调试（2015–2025） | <https://zhuanlan.zhihu.com/p/1920113411972370786> | Recent profile article |
| Confirmed | 深入GPU性能分析工具：现代加速器追踪工具的实现详解 | <https://zhuanlan.zhihu.com/p/1919343479919730801> | Recent profile article |
| Confirmed | 加速器工具箱：GPU和其他协处理器的性能分析和追踪详解 | <https://zhuanlan.zhihu.com/p/1918849547976831456> | Recent profile article |
| Confirmed | eBPF 示例教程：实现 scx_nest 内核调度器 | <https://zhuanlan.zhihu.com/p/1918839417050755847> | Recent profile article |
| Confirmed | eBPF 与机器学习可观测：追踪 CUDA GPU 操作 | <https://zhuanlan.zhihu.com/p/1918665090783224477> | Recent profile article |
| Confirmed | eBPF 教程：BPF 调度器入门 | <https://zhuanlan.zhihu.com/p/1918657795798046505> | Recent profile article |

Repo-referenced Zhihu URLs that should be kept but may need ownership/date verification:

| Status | URL | Where seen |
| --- | --- | --- |
| Referenced, verify ownership | <https://zhuanlan.zhihu.com/p/555362934> | `docs/eunomia-bpf/manual*.md` |
| Referenced, verify ownership | <https://zhuanlan.zhihu.com/p/573941739> | Wasm/eBPF posts |
| Referenced, verify ownership | <https://zhuanlan.zhihu.com/p/595257541> | Wasm posts |
| Referenced, verify ownership | <https://zhuanlan.zhihu.com/p/605542090> | Rust/Wasm posts |
| Referenced, verify ownership | <https://zhuanlan.zhihu.com/p/597705400> | Wasm community posts |

## Juejin

Profile checked: <https://juejin.cn/user/4288563097635144/posts>

The following authored articles were visible on the first article page during the check:

| Status | Title | URL | Notes |
| --- | --- | --- | --- |
| Confirmed | 多智能体系统是人工智能的未来吗？探讨OpenAI的Swarm实验 | <https://juejin.cn/post/7424407625897492514> | Latest visible authored article |
| Confirmed | eBPF 实践教程: 通过 socket 或 syscall 追踪 HTTP 等七层协议 | <https://juejin.cn/post/7280746975917228087> | Authored article |
| Confirmed | eBPF 实践教程：使用 eBPF 用户态捕获多种库的 SSL/TLS 明文数据 | <https://juejin.cn/post/7269723528961261623> | Authored article |
| Confirmed | eBPF 入门实践教程：用 bpf_send_signal 发送信号终止恶意进程 | <https://juejin.cn/post/7269763137808187453> | Authored article |
| Confirmed | OpenAI 新发布GPT 最佳实践：落地大模型应用的策略和战术 | <https://juejin.cn/post/7241495840556073021> | Authored article |
| Confirmed | eBPF 入门开发实践教程九：捕获进程调度延迟，以直方图方式记录 | <https://juejin.cn/post/7240428977062903845> | Authored article |
| Confirmed | eBPF 入门开发实践教程十：在 eBPF 中使用 hardirqs 或 softirqs 捕获中断事件 | <https://juejin.cn/post/7240427838343823415> | Authored article |
| Confirmed | eBPF 入门开发实践教程十一：在 eBPF 中使用 libbpf 开发用户态程序并跟踪 exec() 和 exit() 系统调用 | <https://juejin.cn/post/7240404579131949112> | Authored article |
| Confirmed | eBPF 入门实践教程十二：使用 eBPF 程序 profile 进行性能分析 | <https://juejin.cn/post/7240404579131916344> | Authored article |
| Confirmed | eBPF入门开发实践教程十三：统计 TCP 连接延时，并使用 libbpf 在用户态处理数据 | <https://juejin.cn/post/7240371866286997563> | Authored article |

Repo-referenced Juejin URLs that should be kept but may need ownership/date verification:

| Status | URL | Where seen |
| --- | --- | --- |
| Referenced, verify ownership | <https://juejin.cn/post/7043721713602789407> | eBPF/Wasm posts |

## X / Twitter

Profile checked: <https://x.com/yunwei37>

The profile was visible in the logged-in Chrome session as `云微` / `@yunwei37`, with `eunomia.dev` in the bio and `1,591` total posts shown at the time of the check. The repository also references `@eaborai` in SEO/social planning, but <https://x.com/eaborai> currently showed `此账号不存在`, so it is tracked as stale or unverified rather than confirmed.

The following self-authored project posts were visible through X profile/status pages or `from:yunwei37` search:

| Status | Date | Title / visible text | URL | Notes |
| --- | --- | --- | --- | --- |
| Confirmed | 2026-07-19 | "Run tests before commit" is not just a prompt rule | <https://x.com/yunwei37/status/2079002839440068969> | Links to `docs/blog/posts/ebpf-ai-agent-policy-enforcement.md`; published from `draft/media/2026-07-19/ebpf-ai-agent-policy-enforcement/x.md` |
| Confirmed | 2025-11-15 | A blog for GPU observability tools using bpftime | <https://x.com/yunwei37/status/1989812682502070525> | Links to the GPU observability blog on `eunomia.dev`; X showed `1.7万` views in the logged-in UI |
| Confirmed | 2025-07-17 | 写了一个用 ebpf 追踪 AI Agent 的小玩意 | <https://x.com/yunwei37/status/1945767621707829452> | Links to `github.com/eunomia-bpf/agentsight`; visible status page showed 5,565 views |
| Confirmed | 2024-01-21 | Bpftime now supports tracing USDT in userspace eBPF | <https://x.com/yunwei37/status/1749079236571213825> | Links to the bpftime USDT example |
| Confirmed | 2024-01-18 | A new blog to introduce our userspace eBPF runtime: bpftime: Extending eBPF from Kernel to User Space | <https://x.com/yunwei37/status/1748025809741152475> | Links to the bpftime blog on `eunomia.dev` |
| Confirmed | 2023-12-08 | bpftime can do more than just tracing; `bpf_override_return` for userspace functions or syscall tracepoints | <https://x.com/yunwei37/status/1733168796444061973> | Links to the bpftime error-inject example |
| Confirmed | 2023-11-15 | A draft preprint about how bpftime works in userspace | <https://x.com/yunwei37/status/1724929227512791143> | Links or quotes the bpftime preprint |

Third-party X mentions found during the same check, not counted as our own publishing:

| Status | URL | Notes |
| --- | --- | --- |
| Third-party mention | <https://x.com/alexei_ast> | Search snippet showed a `github.com/eunomia-bpf/bpftime` mention |
| Third-party mention | <https://x.com/VivekIntel/status/2062820166791286790/photo/1> | Search snippet referenced AgentSight and `github.com/eunomia-bpf/ag...` |
| Third-party mention | <https://x.com/TonyNashNerd/status/2059253274662895722> | Search snippet referenced `eunomia.dev` and the AgentSight paper |
| Third-party mention | X search result for `AgentSight` / `eunomia-bpf` | Logged-in X search showed posts by `@wardy_ai`, `@zeeshan_utd`, and `@cr0nym`; record exact URLs before treating them as outreach artifacts |

## LinkedIn

Profile checked in the in-app browser/sidebar: <https://www.linkedin.com/in/yunwei37/>

The visible logged-in profile is `Yusheng Zheng`, with the headline `Founder of eunomia.dev | Building Infra for AI Agents with eBPF`. Recent activity and LinkedIn content search showed authored posts for ActPlane/AgentSight, ACRFence, GPU observability, agentpprof, and the BPFix paper. Exact permalinks were visible for the BPFix recent-activity item; some search-visible posts are recorded with evidence URLs until their post permalinks are captured.

New confirmed post:

| Status | Date | Title / visible text | URL | Notes |
| --- | --- | --- | --- | --- |
| Confirmed | 2026-07-19 | A rule like "run the full test suite before committing" looks simple | <https://www.linkedin.com/feed/update/urn:li:share:7484770128912465920> | Links to `docs/blog/posts/ebpf-ai-agent-policy-enforcement.md`; published from `draft/media/2026-07-19/ebpf-ai-agent-policy-enforcement/linkedin.md` |

Machine-readable details: [`platforms/linkedin.json`](platforms/linkedin.json). Current script coverage: 5 of 118 English target source files mapped as LinkedIn-published.

## Reddit

Author checked through Reddit search and post pages: `u/yunwei123`.

The following self-authored Reddit posts were visible through `author:yunwei123` search or direct post pages:

| Status | Subreddit | Title | URL | Notes |
| --- | --- | --- | --- | --- |
| Confirmed | r/linux | eBPF Tutorial by Example: Learning eBPF Step by Step with Tools | <https://www.reddit.com/r/linux/comments/17dzoib/ebpf_tutorial_by_example_learning_ebpf_step_by/> | Authored by `yunwei123`; post links GitHub tutorial repo and `https://eunomia.dev/tutorials/` |
| Confirmed | r/eBPF | eBPF Tutorial by Example: Learning eBPF Step by Step with Tools | <https://www.reddit.com/r/eBPF/comments/17hugts/ebpf_tutorial_by_example_learning_ebpf_step_by/> | Authored by `yunwei123`; parallel tutorial announcement |
| Confirmed | r/linuxadmin | eBPF Tutorial by Example: Learning eBPF Step by Step with Tools | <https://www.reddit.com/r/linuxadmin/comments/17dzl5b/ebpf_tutorial_by_example_learning_ebpf_step_by/> | Authored by `yunwei123`; parallel tutorial announcement |
| Confirmed | r/eBPF | Using ChatGPT to Write and Trace Linux eBPF Programs with Natural Language | <https://www.reddit.com/r/eBPF/comments/116e4cu/using_chatgpt_to_write_and_trace_linux_ebpf/> | Found through `author:yunwei123` search; GPTtrace announcement |
| Confirmed | r/ChatGPTCoding | Can LLMs help understanding Large-Scale Codebases like Linux kernel? | <https://www.reddit.com/r/ChatGPTCoding/comments/1fzkrpf/can_llms_help_understanding_largescale_codebases/> | Found through `author:yunwei123` search; likely Code-Survey / LLM-codebase discussion |
| Confirmed | r/Cloud | Wasm-bpf: Build and run eBPF programs in WebAssembly | <https://www.reddit.com/r/Cloud/comments/11243ch/wasmbpf_build_and_run_ebpf_programs_in_webassembly/> | Found through `author:yunwei123` search; Wasm-bpf announcement |
| Confirmed | r/eBPF | Wasm-bpf: Build and run eBPF programs in WebAssembly | <https://www.reddit.com/r/eBPF/comments/10zwo7l/wasmbpf_build_and_run_ebpf_programs_in_webassembly/> | Found through `author:yunwei123` search; parallel Wasm-bpf announcement |

Reddit references or third-party posts found during the same check, not counted as our own publishing:

| Status | Subreddit | Title | URL | Notes |
| --- | --- | --- | --- | --- |
| Third-party post | r/ReverseEngineering | eunomia-bpf/bpftime: Userspace eBPF runtime for fast Uprobe & Syscall hook & Plugins | <https://www.reddit.com/r/ReverseEngineering/comments/17vjuvn/eunomiabpfbpftime_userspace_ebpf_runtime_for_fast/> | Authored by `tnavda`, not `yunwei123` |
| Third-party discussion | r/LocalLLaMA | I built an eBPF tracer to monitor AI agents the same way you'd monitor malware in a sandbox | <https://www.reddit.com/r/LocalLLaMA/comments/1r8yvu5/i_built_an_ebpf_tracer_to_monitor_ai_agents_the/> | About Azazel; `yunwei123` commented with an AgentSight comparison, but the post is not ours |

## Xiaohongshu / RedNote

No confirmed Xiaohongshu account or published post was found for `eunomia.dev`, `eunomia-bpf`, `bpftime`, `AgentSight`, `yunwei37`, or `云微` during this check.

Browser check: <https://www.xiaohongshu.com/search_result?keyword=eunomia.dev%20eBPF%20AgentSight> required login before showing search results in the current Chrome session. Public web search also did not return a clear project/account hit. Treat Xiaohongshu as not started until an account URL or published note URL is provided and verified.
