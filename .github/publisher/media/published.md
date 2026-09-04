# Published Media Ledger

Last checked: 2026-09-02

This ledger records platform posts confirmed from logged-in profile pages or explicit repository references. It is intentionally conservative: if authorship or completeness is not verified, the item is marked for follow-up instead of treated as complete.

Canonical machine-readable source: one JSON file per platform under [`platforms/`](platforms/), with source rules in [`sources.json`](sources.json). Run `python .github/publisher/media/check_media_ledger.py --show-missing` to count missing source coverage per platform.

## Zhihu

Profile checked: <https://www.zhihu.com/people/yun-wei-64-11/posts>

The profile reports `文章119` after the 2026-08-05 publication. The following recent articles were visible on the article page during the check:

Follow-up platform check on 2026-07-20 used normal in-app browser scrolling on the Zhihu profile page. The page reported `文章114`; scrolling collected 113 unique visible article links, which is recorded in [`platforms/zhihu.json`](platforms/zhihu.json). Exact-title matching also confirmed that the userspace-eBPF and eBPF-runtime-security tutorial sources are already covered by same-title blog posts, so they are tracked as equivalent sources instead of being republished. Do not use Zhihu API or hidden/internal endpoints for future checks.

| Status | Title | URL | Notes |
| --- | --- | --- | --- |
| Confirmed | eBPF 教程：使用 fsession 追踪慢速 vfs_read 调用 | <https://zhuanlan.zhihu.com/p/2071960422765310563> | Published 2026-08-14 from `docs/tutorials/52-fsession-latency/README.zh.md` through the normal visible Zhihu Markdown-import and editor UI; selected “在 Linux 环境下，如何有效诊断和优化高并发应用的 I/O 性能瓶颈？” and topics `ebpf`, `Linux 内核`; public article and question-page QA confirmed the exact title, complete body, 18 body headings, 20 code blocks, 1 table, source and reference links, no inline content images, and no failure placeholders |
| Confirmed | eBPF 教程：精准隔离已建立的 TCP 连接 | <https://zhuanlan.zhihu.com/p/2068597605345531007> | Published 2026-08-05 from `docs/tutorials/51-tcp-quarantine/README.zh.md` through the normal visible Zhihu Markdown-import and editor UI; selected `TCP服务器如何处理恶意连接？` and topics `TCP`, `ebpf`; source H1 was carried by the title field without changing the substantive body; public-page QA confirmed the exact title, 15 body headings, 9 code blocks, 1 table with 7 rows, 6 outbound article links, no inline images, and no failure placeholders |
| Confirmed | eBPF 入门实践教程第五十篇：使用 TCX Link 实现可组合的流量控制 | <https://zhuanlan.zhihu.com/p/2067144957194924576> | Published 2026-08-01 from `docs/tutorials/50-tcx/README.zh.md` through the normal visible Zhihu document-import and editor UI; selected `linux内核如何实践？` and topics `ebpf`, `Linux 内核`; removed the source H1 from the imported body and converted the relative lesson-20 link to its public eunomia.dev URL without changing the substantive body; public-page QA confirmed the exact title, 21 body headings, 8 code blocks, 2 tables with 12 rows, and 10 outbound article links |
| Confirmed | 为什么 eBPF verifier 报错难修：诊断鸿沟 | <https://zhuanlan.zhihu.com/p/2064970067830480995> | Published from `docs/blog/posts/bpfix.zh.md` through the normal visible Zhihu UI; selected `linux内核调试方式？`; public-page QA confirmed the exact title and complete body, 7 section headings, 3 code blocks, 4 tables, 2 loaded content images, and 10 outbound source links |
| Confirmed | eBPF 教程：检查 exec 后真正安装的可执行镜像 | <https://zhuanlan.zhihu.com/p/2063162298781398619> | Published from `docs/tutorials/54-exec-image-inspector/README.zh.md`; exact title was confirmed on the visible profile article list, and public-page DOM QA confirmed the rendered body, code blocks, table, image, and outbound links |
| Confirmed | 实证研究：AI Agent 规则需要上下文与分层强制执行 | <https://zhuanlan.zhihu.com/p/2062539029892151274> | Published from `docs/blog/posts/ebpf-ai-agent-policy-enforcement.zh.md`; selected the production-readiness AI Agent question; imported a locally prepared image-free body after unstable image imports and uploaded the ActPlane architecture cover separately |
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
| Confirmed | eBPF 运行时安全性：面临的挑战与前沿创新 | <https://zhuanlan.zhihu.com/p/667257765> | Covers both the blog source and `docs/tutorials/18-further-reading/ebpf-security.zh.md` |
| Confirmed | 用户空间 eBPF 运行时：深度解析与应用实践 | <https://zhuanlan.zhihu.com/p/662734555> | Covers both the blog source and `docs/tutorials/36-userspace-ebpf/README.zh.md` |

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

Normal visible browser pagination covered the complete authored article list. On 2026-08-28, all five visible profile pages were checked before publishing the CPU noise article; the new article then appeared as the newest item and brought the visible profile total to 47. The machine ledger remains conservative: submissions under platform review are not counted as confirmed source coverage.

On 2026-08-30, all five authored-list pages were checked again before publishing AgentCgroup. No duplicate was found; the new public article passed full-scroll QA and appeared on the author profile after review.

On 2026-08-31, the same visible duplicate and review checks preceded the HID tutorial publication. The public article passed full-scroll QA and became the newest item on the author profile.

On 2026-09-02, Agent Sandbox became publicly listed, but the mechanical update that removes a repeated body H1 remained under review. It stays pending rather than confirmed until the corrected public page can be checked.

| Status | Title | URL | Notes |
| --- | --- | --- | --- |
| Needs repair | 智能体系统架构：隔离、集成与治理的综合调研 | <https://juejin.cn/post/7680709250218426395> | Publicly listed 2026-09-02 with category `人工智能` and tags `人工智能`, `架构`, `安全`. The update has left review, but visible public-page checks on 2026-09-03 and 2026-09-04 confirmed that the complete body is rendered twice, including the opening, section sequence, and references. Repair this article in place by removing only the second copy, then repeat full public-page QA; do not republish it. |
| Confirmed | eBPF 教程：无需内核补丁修复故障的 HID 设备 | <https://juejin.cn/post/7680099381967798307> | Published and confirmed public 2026-08-31 from `docs/tutorials/49-hid/README.zh.md` through the visible Juejin editor; category `后端`; tags `Linux`, `开源`, `后端`; full-scroll public-page QA confirmed the exact title, substantively unchanged body, 14 body headings, 14 code blocks, 9 source links, no inline images, the complete conclusion and final reference, and no review or loading-error marker; the author profile lists the public URL |
| Confirmed | AgentCgroup：当 AI Agent 遇到操作系统资源 | <https://juejin.cn/post/7679373303883350059> | Published and confirmed public 2026-08-30 from `docs/blog/posts/agentcgroup-characterization.zh.md` through the visible Juejin editor; category `人工智能`; tags `Linux`, `人工智能`, `后端`; full-scroll public-page QA confirmed the exact title, complete body, 14 body headings, 7 tables, 1 Bash code block, all 8 source links, no inline images, and 0 comments; the review marker disappeared and the author profile lists the public URL |
| Confirmed | CPU 噪声会拖慢 GPU 推理吗：用 eBPF 定量测量调度器与 IRQ 影响 | <https://juejin.cn/post/7679000152964153353> | Published 2026-08-28 from `docs/blog/posts/cpu-noise-gpu-inference.zh.md` through the normal visible Juejin editor in Chrome `Yunwei`; category `人工智能`; tags `Linux`, `人工智能`, `性能优化`; full visible-page scrolling and public-page QA confirmed the exact title, 22 body headings, 12 language-labeled code blocks, 17 tables, all 6 source link targets, no inline source images, 0 comments, and no failure or review marker |
| Confirmed | eBPF 教程：用 BPF Qdisc 实现出口限速 | <https://juejin.cn/post/7677218586263912482> | Submitted 2026-08-23 and confirmed public 2026-08-24 from `docs/tutorials/53-egress-pacer/README.zh.md`; category `后端`; tags `Linux`, `开源`, `云原生`; public-page QA confirmed the exact title, 11 body headings, 9 code blocks, 1 table, source and reference links, and 0 comments; the temporarily missing Juejin-hosted diagram rendered correctly on the 2026-08-25 recheck without an article edit |
| Confirmed | eBPF 教程：使用 fsession 追踪慢速 vfs_read 调用 | <https://juejin.cn/post/7673940698534232064> | Submitted 2026-08-15 and confirmed public 2026-08-16 from `docs/tutorials/52-fsession-latency/README.zh.md` through the normal visible Juejin editor in Chrome `Yunwei`; category `后端`; tags `Linux`, `云原生`; the public page preserves the exact title and complete body with 18 body headings, 20 code blocks, 1 table, 9 article links including source and references, no loading-error marker, and 0 comments |
| Confirmed | eBPF 教程：精准隔离已建立的 TCP 连接 | <https://juejin.cn/post/7671185597861675008> | Submitted and confirmed public 2026-08-07 from `docs/tutorials/51-tcp-quarantine/README.zh.md` through the normal visible Juejin editor in Chrome `Yunwei`; category `后端`; tags `Linux`, `开源`, `安全`; the public page preserves the exact title and complete body with 15 body headings, 9 code blocks, 1 table with 7 rows, 6 source links, no source image placeholders, and 0 comments |
| Confirmed | eBPF 入门实践教程第五十篇：使用 TCX Link 实现可组合的流量控制 | <https://juejin.cn/post/7669635386159824936> | Submitted 2026-08-03 from `docs/tutorials/50-tcx/README.zh.md` through the normal visible Juejin editor in Chrome `Yunwei`; category `后端`; tags `Linux`, `开源`, `云原生`; confirmed public on 2026-08-04 with the exact title, complete rendered body, 21 body headings, 8 code blocks, 2 tables with 12 rows, 10 article links, and no comments |
| Confirmed | 为什么 eBPF verifier 报错难修：诊断鸿沟 | <https://juejin.cn/post/7667474278616547380> | Published from `docs/blog/posts/bpfix.zh.md`; category `后端`; tags `Linux`, `后端`, `开源`; front matter, the repeated H1, and `<!-- more -->` were mechanically removed and two relative image URLs were replaced without changing the substantive body; the 2026-07-29 visible public-page check confirmed the exact title, complete body, seven H2 headings, three code blocks, four tables, ten outbound links, and two loaded content images |
| Confirmed | 让 AI Agent 调优 Linux 调度器：SchedCP 与 sched-agent 的设计与评测 | <https://juejin.cn/post/7666245761520615439> | Published from `docs/blog/posts/schedcp-agentic-os.zh.md` on 2026-07-25; category `人工智能`; tags `Linux`, `人工智能`, `后端`; front matter and `<!-- more -->` were removed and the relative image URL was replaced without changing the substantive body; the 2026-07-26 visible public-page check confirmed the exact title, complete rendered body, seven article headings, and the loaded 788x600 content image |
| Confirmed | eBPF 教程：检查 exec 后实际安装的可执行镜像 | <https://juejin.cn/post/7664864449153613824> | Published from `docs/tutorials/54-exec-image-inspector/README.zh.md`; the 2026-07-22 update is now public, and the 2026-07-25 visible article check confirmed all 15 section headings and the loaded 960x418 image |
| Confirmed | 实证研究：AI Agent 规则需要上下文与分层强制执行 | <https://juejin.cn/post/7664151348536229903> | Published from `docs/blog/posts/ebpf-ai-agent-policy-enforcement.zh.md`; category `人工智能`; tags `Linux`, `AI编程`, `安全`; exact source title, images, tables, code, and outbound links were verified on the public article page |
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
| Confirmed | 2026-08-01 | A tool for backup, recovery and migration of AI agent session history | <https://x.com/yunwei37/status/2083647587132035205> | Links to `github.com/eunomia-bpf/akeep`; verified in the normal visible logged-in profile |
| Confirmed | 2026-07-22 | Introducing AgentNebula: See days or weeks of your AI Agent's work in 30 seconds | <https://x.com/yunwei37/status/2080063281637830665> | Published from `draft/media/2026-07-22/agentsight-agent-nebula/x.md` with the 10-second animation and AgentSight GitHub link |
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

New confirmed posts:

| Status | Date | Title / visible text | URL | Notes |
| --- | --- | --- | --- | --- |
| Confirmed | 2026-08-26 | eBPF Tutorial: Inspecting the Executable Image After exec | <https://www.linkedin.com/feed/update/urn:li:share:7498528283618304000/> | Published from `docs/tutorials/54-exec-image-inspector/README.md` using `draft/media/2026-08-26/54-exec-image-inspector/linkedin.md`; normal visible public-page QA confirmed the exact body, three hashtags, global visibility, the LinkedIn short-link target, and the eunomia.dev tutorial preview card with its title, image, and domain |
| Confirmed | 2026-08-25 | eBPF Tutorial: Building an Egress Pacer with BPF Qdisc | <https://www.linkedin.com/feed/update/urn:li:share:7498213534980263937/> | Published from `docs/tutorials/53-egress-pacer/README.md` using `draft/media/2026-08-24/53-egress-pacer/linkedin.md`; normal visible public-page QA confirmed the intended body, hashtags, public visibility, and eunomia.dev tutorial preview card |
| Confirmed | 2026-08-21 | eBPF Tutorial: Tracing Slow vfs_read Calls with fsession | <https://www.linkedin.com/feed/update/urn:li:share:7496484872375513088/> | Published from `docs/tutorials/52-fsession-latency/README.md` using `draft/media/2026-08-20/52-fsession-latency/linkedin.md`; normal visible public-page QA confirmed the exact body, hashtags, eunomia.dev preview card, and LinkedIn short link landing on the canonical tutorial |
| Confirmed | 2026-08-15 | AgentSight now works on macOS and Windows, too | <https://www.linkedin.com/feed/update/urn:li:share:7494569268571803649> | Published from `draft/media/2026-08-15/agentsight-cross-platform/linkedin.md`; normal visible public-page QA confirmed the exact body, public visibility, GitHub link, and AgentSight repository preview card |
| Confirmed | 2026-08-04 | eBPF Tutorial by Example 50: Composable Traffic Control with TCX Links | <https://www.linkedin.com/feed/update/urn:li:share:7490597138595229696> | Published from `docs/tutorials/50-tcx/README.md` using `draft/media/2026-08-04/50-tcx/linkedin.md`; normal visible public-page QA confirmed the exact two-sentence body, public visibility, eunomia.dev link, and tutorial preview card with its title and image |
| Confirmed | 2026-08-01 | A tool for backup, recovery and migration of AI agent session history | <https://www.linkedin.com/posts/yunwei37_github-eunomia-bpfakeep-privacy-first-activity-7489412627769548800-SV__> | Links to `github.com/eunomia-bpf/akeep`; verified in normal visible public profile activity |
| Confirmed | 2026-07-22 | Introducing AgentNebula: See days or weeks of your AI Agent's work in 30 seconds | <https://www.linkedin.com/posts/yunwei37_aiagents-observability-developertools-activity-7485819115337637888-iFAK> | Published from `draft/media/2026-07-22/agentsight-agent-nebula/linkedin.md` with the 10-second animation and AgentSight GitHub link |
| Confirmed | 2026-07-19 | A rule like "run the full test suite before committing" looks simple | <https://www.linkedin.com/feed/update/urn:li:share:7484770128912465920> | Links to `docs/blog/posts/ebpf-ai-agent-policy-enforcement.md`; published from `draft/media/2026-07-19/ebpf-ai-agent-policy-enforcement/linkedin.md` |

Machine-readable details: [`platforms/linkedin.json`](platforms/linkedin.json). Current script coverage: 11 of 124 English target source files mapped as LinkedIn-published.

## Medium

Account checked in the in-app browser: <https://medium.com/@yunwei356>

Earlier visible profile scrolling collected 62 authored story links; the newer BPFix and tutorials 51–54 public URLs were verified independently. The machine ledger maps 65 of 124 English source files; the difference includes source-equivalent mappings and remaining unmatched profile items.

New confirmed posts:

| Status | Date | Title | URL | Notes |
| --- | --- | --- | --- | --- |
| Confirmed | 2026-08-24 | eBPF Tutorial: Inspecting the Executable Image After exec | <https://medium.com/@yunwei356/ebpf-tutorial-inspecting-the-executable-image-after-exec-46a312704a2c> | Published through the documented Medium API with `ebpf`, `linux`, and `security`; the prepared Markdown kept the complete source and replaced the Requirements table with a five-item fallback; public full-scroll QA confirmed the exact visible H1, 15 source body headings, 8 code blocks, loaded diagram, source and reference links, and three tags |
| Confirmed | 2026-08-23 | eBPF Tutorial: Building an Egress Pacer with BPF Qdisc | <https://medium.com/@yunwei356/ebpf-tutorial-building-an-egress-pacer-with-bpf-qdisc-01f2a8c5fb49> | Published through the documented Medium API with `ebpf`, `linux`, and `networking`; on 2026-08-30, repaired only the flattened Environment Requirements paragraph in place as an equivalent nine-item list. Signed-out public-page QA confirmed the list, unchanged title, 9 code blocks, loaded diagram, and preserved surrounding body; no new article was created |
| Confirmed | 2026-08-17 | eBPF Tutorial: Tracing Slow vfs_read Calls with fsession | <https://medium.com/@yunwei356/ebpf-tutorial-tracing-slow-vfs-read-calls-with-fsession-defc728e227e> | Published through the documented Medium API with `ebpf`, `linux`, and `performance`; the prepared Markdown preserved the visible H1 and used a readable six-item fallback for the environment table; public full-scroll QA verified the exact title, 18 source body headings, 20 code blocks, source and reference links, three tags, and no failure markers |
| Confirmed | 2026-08-09 | eBPF Tutorial: Precisely Isolating Established TCP Connections | <https://medium.com/@yunwei356/ebpf-tutorial-precisely-isolating-established-tcp-connections-9d08644b44b4> | Published through the official Medium API with `ebpf`, `linux`, and `security`; the web editor replaced a flattened requirements table with a readable prose fallback; public full-scroll QA verified the exact title, complete body, 9 code blocks, links, tags, and no broken images |
| Confirmed | 2026-08-02 | Why eBPF Verifier Errors Are Hard to Fix: The Diagnostic Gap | <https://medium.com/@yunwei356/why-ebpf-verifier-errors-are-hard-to-fix-the-diagnostic-gap-d7904b9432f0> | Published through the Medium API using semantic HTML after Markdown parser error 2012; the visible H1 was restored in the supported web editor; public full-scroll QA verified headings, code, table fallbacks, references, links, and both images |
| Confirmed | 2026-07-19 | An Empirical Study: AI Agent Rules Need Context and Layered Enforcement | <https://medium.com/@yunwei356/an-empirical-study-ai-agent-rules-need-context-and-layered-enforcement-eunomia-423adab48a1b> | Imported from canonical eunomia.dev article; post-publish web-editor fix removed the imported `\| eunomia` title suffix; `rel=canonical` points to <https://eunomia.dev/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/>; Medium table flattening is recorded as a follow-up limitation |

Machine-readable details: [`platforms/medium.json`](platforms/medium.json).

## DEV Community

Account checked in the in-app browser: <https://dev.to/yunwei37>

The public author API listed 62 posts on 2026-08-27; the newer BPFix, AgentCgroup, and tutorials 51–54 public URLs were verified independently. The machine ledger maps 53 of 124 English source files, and scheduling uses only the resulting confirmed gaps.

New confirmed posts:

| Status | Date | Title | URL | Notes |
| --- | --- | --- | --- | --- |
| Confirmed | 2026-08-27 | AgentCgroup: What Happens When AI Agents Meet OS Resources? | <https://dev.to/yunwei37/agentcgroup-what-happens-when-ai-agents-meet-os-resources-1h2f> | Published through the documented DEV API with `ebpf`, `linux`, `ai`, and `opensource`; visible public-page QA confirmed the exact title, 14 body headings, 7 rendered tables, 1 code block, canonical notice, complete ending, and GitHub and arXiv links |
| Confirmed | 2026-08-25 | eBPF Tutorial: Inspecting the Executable Image After exec | <https://dev.to/yunwei37/ebpf-tutorial-inspecting-the-executable-image-after-exec-3el7> | Published through the documented DEV API with `ebpf`, `linux`, `security`, and `tutorial`; visible public full-scroll QA confirmed the exact title, 15 body headings, 8 code blocks, a 5-row Requirements table, a loaded diagram, and all source and reference links |
| Confirmed | 2026-08-23 | eBPF Tutorial: Building an Egress Pacer with BPF Qdisc | <https://dev.to/yunwei37/ebpf-tutorial-building-an-egress-pacer-with-bpf-qdisc-4fna> | Published through the documented DEV API with `ebpf`, `linux`, `networking`, and `tutorial`; public QA confirmed the exact title, 12 body headings, 9 code blocks, 1 table, loaded diagram, source and reference links |
| Confirmed | 2026-08-17 | eBPF Tutorial: Tracing Slow vfs_read Calls with fsession | <https://dev.to/yunwei37/ebpf-tutorial-tracing-slow-vfsread-calls-with-fsession-48p5> | Published through the documented DEV API with `ebpf`, `linux`, `performance`, and `tutorial`; public full-scroll QA verified the exact title, all 18 body headings, 20 code blocks, one rendered table, source and reference links, and no failure markers |
| Confirmed | 2026-08-10 | eBPF Tutorial: Precisely Isolating Established TCP Connections | <https://dev.to/yunwei37/ebpf-tutorial-precisely-isolating-established-tcp-connections-56ld> | Published through the DEV API with `ebpf`, `linux`, `security`, and `networking`; public full-scroll QA verified the exact title, complete body, 9 code blocks, 1 rendered table, Requirements, Summary, References, source links, and loaded article images |
| Confirmed | 2026-08-02 | Why eBPF Verifier Errors Are Hard to Fix: The Diagnostic Gap | <https://dev.to/yunwei37/why-ebpf-verifier-errors-are-hard-to-fix-the-diagnostic-gap-443o> | Published through the DEV API with canonical URL, `ebpf`, `linux`, `opensource`, and `ai`; public full-scroll QA verified title, headings, code blocks, tables, references, links, and both images |
| Confirmed | 2026-07-19 | An Empirical Study: AI Agent Rules Need Context and Layered Enforcement | <https://dev.to/yunwei37/an-empirical-study-ai-agent-rules-need-context-and-layered-enforcement-43on> | Published from DEV editor with canonical URL set to <https://eunomia.dev/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/>; post-publish web-editor fixes removed duplicate manual source note, changed tags to `opensource`, `ai`, `security`, `ebpf`, and replaced four 404 eunomia.dev image URLs with GitHub raw image URLs |

Machine-readable details: [`platforms/devto.json`](platforms/devto.json).

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
