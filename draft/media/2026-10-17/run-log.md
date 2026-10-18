# 2026-10-17 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-10-17。唯一提交时刻为 2026-10-18 07:24 +08（= 2026-10-17 16:24 PDT），落在 LA 2026-10-17。巡检开始时补发缺口为 0 条（10-09 → 10-10 → 10-11 → 10-12 → 10-13 → 10-14 → 10-15 → 10-16 → 10-17 连续巡检，无新增补发缺口），dynptr 使用正常额度，未核销任何缺口；本次结束后仍无补发缺口。知乎任务仍 `阻塞`——队列 `阻塞` 条款自 2026-09-17 起未恢复（本次未重新探测知乎，亦跳过队列内全部知乎 `排队` 行），故首个可执行的 `排队` 任务为掘金 dynptr（队列第 128 行；第 127 行知乎 dynptr 行 `阻塞`，跳过）；其后下一可执行任务为第 130 行掘金 struct_ops（LA 2026-10-18 正常额度；第 129 行为知乎 struct_ops 行，`阻塞`）。

## 运行前环境修复（浏览器恢复）

本次巡检启动时 CDP 9222 未监听，持久 Chromium 会话在宿主重启后未拉起。根因排查（新故障模式，与 patrol skill 记载的 `rc=-5` crashpad SIGTRAP 循环无关）：

- 根因：PID 1 `gem.sh` 卡在 pre-service hook 的 `until bash /run/coder-agent-config/install-agent-clis.sh; do sleep 10; done` 重试循环。该循环以 root 身份、窄 PATH（`/home/gem/.local/bin:/usr/local/bin:/usr/bin:/bin`）运行；安装脚本第 26 行 `exec runuser -u gem …` 失败 rc=127，因为 `runuser` 只存在于 `/usr/sbin`（不在窄 PATH 内）→ 无限循环 → `gem.sh` 永远走不到 `write_browser_supervisor_config` → `/var/run/gem/browser-supervisor.json` 未发布 → supervisor 饿死 → X / Chromium / CDP 全部未起。
- 修复：`sudo ln -sfn /usr/sbin/runuser /usr/local/bin/runuser`（幂等）。下一轮 hook 循环通过，`gem.sh` 继续执行，supervisor 拉起 `:99.0` + Chromium（pid 50191），CDP 开始应答 `Chrome/146.0.7680.31`。
- 决策：只修 hook 的 PATH 这一根因，让**设计内的**流程（config 发布 + X + Chromium）自己走通；不手写 infra 配置（`/var/run/gem/browser-supervisor.json` 是 `gem.sh` 的产物，手写会绕过 supervisor 的后续管理）。Juejin 登录态在持久 Chrome profile 中，重启后仍然有效，发布直接可用。

## 队列任务：掘金 dynptr（LA 10-17 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 128 行（`排队`，现翻为 `[x]`）。上一可执行掘金行第 126 行 bpf_wq 已于 10-16 巡检完成；第 127 行为知乎 dynptr 行（`阻塞`），跳过；下一可执行掘金行第 130 行为 struct_ops（LA 2026-10-18 正常额度）。
- 源文：`docs/tutorials/features/dynptr/README.zh.md`（437 行）；发布稿 `draft/media/2026-10-17/dynptr/juejin-body.md`（移除源 H1，其余正文逐字保留；14223 字符、20907 字节、436 行、7 个 H2 / 15 个 H3 / 0 个 H4、11 个代码块 [2 c：示例内核程序与示例用户态程序；7 bash：构建与运行命令；2 无语言标签：预期输出]、0 张图片、0 个表格、8 条唯一外链目标 [github.com/eunomia-bpf/bpf-developer-tutorial、github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/features/dynptr、eunomia.dev/tutorials/、docs.ebpf.io/linux/concepts/dynptrs/、docs.ebpf.io/linux/helper-function/bpf_ringbuf_reserve_dynptr/、docs.ebpf.io/linux/kfuncs/bpf_dynptr_from_skb/、docs.ebpf.io/linux/kfuncs/bpf_dynptr_slice/、docs.kernel.org/bpf/kfuncs.html]、0 条相对链接）。
- 标题逐字取源 H1「eBPF 实例教程：BPF 动态指针处理可变长度数据」（源 H1 无系列编号，发布稿不另加编号）。
- 草稿未预置：`https://juejin.cn/editor/drafts/new` 新编辑器会话（无预置草稿）；标题经 `input[placeholder*="标题"]` 原生 setter 写入并回读，正文以 base64 分块写入 `window.__b64`（3×9292 b64 字符）后按 UTF-8 解码注入 CodeMirror，写后回读 14223 字符与本地一致。
- 分类 `后端`（`.category-list .item` 真实 CDP 指针点击）；标签 `Linux`、`后端`、`性能优化`（每个 chip 经 native-setter 搜索 + 真实 CDP 指针点击可见 `.byte-select-option`；最终 chip DOM 顺序 `Linux`、`后端`、`性能优化`，沿用 17-biopattern / 19-lsm-connect / 1-helloworld / bpftrace-tutorial / cgroup / bpf_arena / bpf_iters / bpf_token / bpf_wq 教程标签组）。

## 提交

- 提交时刻 2026-10-18 07:24 +08 = 2026-10-17 16:24 PDT；提交前创作者中心基线 全部 86 / 已发布 86 / 审核中 0 / 未通过 0（bpf_wq 放行后基线）。单次弹窗「确定并发布」（完整 CDP 指针序列）返回「发布成功」并落到 `https://juejin.cn/published`。新文 id `7697186290753159202`，URL `https://juejin.cn/post/7697186290753159202`。提交后创作者中心 全部 87 / 已发布 86 / 审核中 1 / 未通过 0。

## 提交结果与判定

- 进入较长审核间隔：提交后暂存地址 `/spost/7697186290753159202`；审核中窗口内 SPA shell 对 `/post/` URL 返回 HTTP 200，curl 状态码不是公开信号，公开性以登录创作者中心为准；07:47 +08 前后轮询（约 90 秒间隔、16 次后台轮询）翻为 全部 87 / 已发布 87 / 审核中 0（审核间隔约 23 分钟，慢于 26-sudo / 20-tc / bpftrace / bpf_arena / bpf_iters / bpf_token 的 1–5 分钟快档先例，为本系列最长审核间隔；45-scx-nest 先例约 1 小时，故在已观察方差之内，非异常），随后规范 `/post/` URL 上线。
- 记 `confirmed`，带 `staged_url`（审核间隔先例：38-btf-uprobe / 17-biopattern / 1-helloworld / cgroup-tutorial / bpf_arena / bpf_iters / bpf_token / bpf_wq；区别于 bpftrace / 26-sudo / 20-tc 直接公开条目无 `staged_url`）。

## 公开页 QA（登录浏览器）

- dynptr `/post/7697186290753159202`：标题逐字「eBPF 实例教程：BPF 动态指针处理可变长度数据」（页面级 H1 仅 1 次，正文区域无重复 H1）；单份正文探测（14223 字符 artifact；渲染 DOM 对代码块空白做归一化，`.article-viewer.markdown-body` innerText 读为 13785）；7 个 H2 / 15 个 H3 / 0 个 H4；11 个代码块 [2 c / 7 bash / 2 无语言标签，渲染 DOM 自动标 `xml` 与 `ini`]；0 张图片（正文区域无内容图）；0 个表格；8 条唯一外链目标全部改写为 `link.juejin.cn?target=`、0 条裸外链；无 `审核中` / `文章有更新` / `已被删除` / `找不到页面` / `发布失败` 标记；评论 0（`暂无评论` 空态）。早期计数（07:47 +08 创作者中心放行检查点）：0 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏。

## 台账更新

- `platforms/juejin.json`：置顶新增 `juejin-7697186290753159202` 一条 `confirmed`（newest-first；带 `id`、`status`、`title`、`url`、`staged_url`、`source_path`、`checked_via` 与 notes，审核间隔故带 `staged_url`），`published[]` 74 → 75，`last_checked` → 2026-10-17。
- `sources.json`：`last_checked` → 2026-10-17（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-10-17；bpf_wq 叙述段后补记 2026-10-17 dynptr 叙述行（16:24 PDT 提交、07:47 +08 翻为已发布、约 23 分钟审核间隔、系列最长先例）；Juejin confirmed 表格首行置顶 dynptr（newest-first，含 `/post/` 正式地址与 `/spost/` 暂存地址）。
- `not-published.md`：`Last checked:` → 2026-10-17；掘金未映射 35 → 34（73/108 → 74/108）、滚动队列 5 → 4 Juejin tasks（24 Zhihu and 4 Juejin tasks，共 29 → 28）；掘金状态行前置 dynptr 子句。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-10-17；补发缺口段落前置 10-17 巡检句（正常额度 1 条、审核间隔 /spost/ 暂存、07:47 +08 翻为已发布、约 23 分钟、为本系列最长审核间隔）；Ledger 基线按检查器更新（掘金 34，映射 74/108）；剩余队列掘金 5 → 4、总计 29 → 28；第 128 行翻 `[x]` 并附提交时刻、放行时刻、分类/标签、QA 摘要、正式地址与 `confirmed`；知乎任务保持 `阻塞`。
- `community-feedback.md`：在已有 `### 2026-10-16` 块上新增 `### 2026-10-17` 块（dynptr 发布 + QA + 放行后 top-ten 复核 + 前向指针指向 10-18 正常额度掘金 struct_ops）；未在 10-16 块内再织入。
- 发布稿 `draft/media/2026-10-17/dynptr/juejin.md`：状态改「已确认公开」（post `7697186290753159202`，`confirmed`），记录正式地址、提交时刻、审核间隔放行时刻、QA 摘要，并保留正文/标题/原文/分类/标签各行。
- 验证器：`check_media_ledger.py` 通过（Juejin 74/108 映射、34 未发布、75 confirmed [2 条无 `source_path`]；exit 0）。三个 CRLF 台账（not-published.md / community-feedback.md / publishing-queue.zh.md）经字节级编辑，编辑后复查 0 条裸 LF。

## 编辑器经验

- 正文注入：body 文件 base64 化后分块写入 `window.__b64`，再 `document.querySelector('.CodeMirror').CodeMirror.setValue(new TextDecoder('utf-8').decode(Uint8Array.from(atob(window.__b64), c => c.charCodeAt(0))))`（直接 `atob` 结果为乱码，必须按 UTF-8 解码）。
- 分类 chip（`.category-list .item`）、标签选项（`.byte-select-option`）与「确定并发布」必须用 raw CDP `Input.dispatchMouseEvent`（`agent-browser mouse move/down/up`）——页内合成 `dispatchEvent` / PointerEvent 不会真正选中 chip 也不会真正提交；提交成功判据为弹窗「发布成功」并落地 `https://juejin.cn/published`。
- 驱动走 `agent-browser`（CDP 自动经 wrapper 解析到 9222）：中文经 `eval --stdin < /tmp/*.js` 走文件（inline shell 嵌中文字符串会撑破解析器）；计数读取沿用 10-13 起对 `.essays-container a.link` 行配对的读取法（本次改用按日期行 `2026-10-18 07:24` 切分 innerText 的方式逐行配对，规避卡片共祖先导致的全锚点同值问题）。

## 跟踪 top ten 复核（07:47 +08 创作者中心放行后逐行读取）

- 新文 dynptr 首读 0 展现 / 1 阅读。
- 流量累积（对比 10-16 放行后 07:18 +08 复核）：[bpf_wq](https://juejin.cn/post/7697094073967116323) 0 / 1 → 3 / 8；[bpf_token](https://juejin.cn/post/7696759034846445583) 728 / 15 → 804 / 20；[bpf_iters](https://juejin.cn/post/7696384745624141824) 624 / 13 → 694 / 16；[bpf_arena](https://juejin.cn/post/7695977967902933026) 2241 / 52 → 2322 / 53；[cgroup](https://juejin.cn/post/7695667191396040738) 6 / 12 → 8 / 12；[bpftrace](https://juejin.cn/post/7695068144415244323) 4 / 9 持平。
- 未重测窗口轮换：[1-helloworld](https://juejin.cn/post/7694934986782064649)、[17-biopattern](https://juejin.cn/post/7694534084639195174)、[19-lsm-connect](https://juejin.cn/post/7694131662616428571) 已随较新帖（dynptr / bpf_wq / bpf_token / bpf_iters / bpf_arena / cgroup / bpftrace）挤入而下移出 10-16 放行后「最新十条」复核窗口，本次读数为各自 10-17 首读（1457 / 40、1262 / 26、8 / 17；其 10-16 晨窗末次测量值分别为 1390 / 39、1245 / 21、6 / 17）；二十、二十一再次移出前十。全部被跟踪行仍读 0 评论（`暂无评论` 空态持续），无需回复或更正。
- 复核方法：登录创作者中心 `https://juejin.cn/creator/content/article/essays?status=all`，对 `.essays-container` 的 innerText 按 `2026-MM-DD HH:MM` 日期行切分逐行配对展现/阅读（行精确，避免跨行错位配对）。

## 后续

- 下一条可执行任务为队列第 130 行掘金 struct_ops（LA 2026-10-18 正常额度）；本次不越范围执行。
- 若下次巡检再遇 CDP 未起：先复查 `sudo ln -sfn /usr/sbin/runuser /usr/local/bin/runuser`（幂等），确认 `gem.sh` pre-service hook 完成、supervisor 日志 `Chromium ready on 127.0.0.1:9222`，再 `curl -s http://127.0.0.1:9222/json/version` 等待 `Chrome/…`；本次根因为 hook 窄 PATH 缺 `runuser`（新故障模式，已随本日志记录）。

## 2026-10-17 eBPF Q&A 当日运行报告（eunomia-community-radar）

### 候选选择

- 目标 slug：`fuse-trace-context-map-lookup-misses-in-pid-namespace`
- 问题：为什么 FUSE 文件系统运行在 PID 命名空间内时，FUSE 守护进程对 OBI trace 上下文 map 的查询会查不到。
- 来源：10-17 快照（两条 opt-in 归档渠道、共 11 条消息）与 10-16 快照字节相同——24 小时冻结窗口内无新增实质线程，窗口内实质线程均已在 10-13（Cilium eBPF tracepoint）、10-14（FUSE/OBI trace 传播）、10-16（OBI 运行时支持矩阵）三篇回答或为评审状态更新而非新问题。故从公开一手资料回退选源：这是 10-14 map 读取机制之外的**第三个独立失败维度**——即使 OBI 已开启填充且关联到了正确线程，只要 FUSE 守护进程（或其调用方）所处 PID 命名空间与 OBI 打 key 的命名空间不一致，按请求头构造的 key 也必然落空。
- 决定性边界：`fuse_in_header.pid` 是调用线程 TID 经挂载 PID 命名空间重新换算的值，而 OBI 的 `traces_ctx_v1` key 取 `bpf_get_current_pid_tgid`（init 命名空间值）。两者在容器化/特权命名空间的 FUSE 守护进程场景下数字不同，查询确定性落空。修复 = 让 FUSE 守护进程以与 OBI key 视角一致的 PID 命名空间运行，或在构造 key 前解析主机可见 TID。

### 对照公开一手源的验证

- 内核 `kernel/bpf/helpers.c`：`bpf_get_current_pid_tgid` 返回 `(u64)task->tgid << 32 | task->pid`（init/全局命名空间值）；旁证 `bpf_get_ns_current_pid_tgid` 返回 `task_pid_nr_ns(task, pidns)`（唯一的命名空间相对变体）。
- 内核 `fs/fuse/req.c`：`fuse_fill_creds` 把 `args->pid` 设为 `task_pid(current)` 经 `fc->pid_ns` 重新换算（调用线程 TID 于挂载 pid 命名空间内）。
- 内核 `fs/fuse/inode.c`：`fuse_conn_init` 把 `fc->pid_ns` 取自挂载进程的活跃 PID 命名空间。
- 内核 `include/uapi/linux/fuse.h`：`fuse_in_header.pid` 字段。
- libfuse `include/fuse_lowlevel.h`：注明「当 FUSE 文件系统运行在 PID 或 user 命名空间内却从命名空间外被访问时，上报的客户端 uid/gid/pid 可能为零」。
- OBI `bpf/shared/obi_ctx.h`：固定的 `traces_ctx_v1`（`BPF_MAP_TYPE_LRU_HASH`，`__type(key, u64)`，value = `trace_id[16]` + `span_id[8]`）。

### 隐私与内容

- 两页正文均无姓名 / handle / Slack-Discord 链接 / 时间戳 / 消息序号 / 雇主团队 / 内部仓库名 / IP / 主机名 / 私有日志。讨论作者与 maintainer 仅以泛称指代。公开的 kernel / libfuse / OBI 仓库链接仅出现在 `## References` / `## 参考`。
- 内容顺序符合标准：直接回答、机制、验证与调试路径、局限、参考、当日社区讨论。H1 不含下划线、反引号、撇号；EN 与 ZH H1 与索引 link text 逐字一致（`bpf_get_ns_current_pid_tgid` 等 token 拆分后最长 backticked token 为 27 字符 `bpf_get_ns_current_pid_tgid`，远低于 390px 移动端溢出门槛）。

### 覆盖披露

本 run 覆盖两个 opt-in 归档渠道（两个 CNCF OpenTelemetry 插桩渠道），10-17 快照共 11 条消息（与 10-16 字节相同，24h 冻结窗口无新增）。候选选自公开一手资料回退：窗口内实质线程均为 10-13 / 10-14 / 10-16 已发布条目或评审备注，无新问。可见浏览器专属渠道——eunomia-bpf 与 sched-ext Discord 服务器、bpf 邮件列表、r/eBPF——本次无可见浏览器会话，标记为「未覆盖」而非「安静」。

### 产物

- EN：`docs/ebpf-qa/2026-10-17-fuse-trace-context-map-lookup-misses-in-pid-namespace.md`
- ZH：`docs/ebpf-qa/2026-10-17-fuse-trace-context-map-lookup-misses-in-pid-namespace.zh.md`
- 索引链接：`docs/ebpf-qa/index.md`（`Latest Answers`）与 `docs/ebpf-qa/index.zh.md`（`最新回答`）首条。
- 已发布 QA commit（`origin/main`）：`447823c76`（完整 `447823c769f8f55c63016a5219d476b1c004d208`，`docs(ebpf-qa): fuse-trace-context-map-lookup-misses-in-pid-namespace (2026-10-17)`），恰好含四条 owned paths。
- Receipt：`/workspaces/.agent-state/eunomia-qa/receipt-2026-10-17.json`，仅由验证器写入，`status=published`，commit `447823c769f8f55c63016a5219d476b1c004d208`。

### eBPF Q&A 发布跟进（部署 + 线上验证）

- QA commit `447823c76`（4 paths）已落到 `origin/main`。首发验证 run（build→render→commit+push→public-poll）在 render 前因 Chromium headless-shell 缺失 14 个共享库（libnspr4 / libnss3 / atk / dbus / atspi / X11 extras / gbm / xkbcommon / alsa 等）无法启动而失败（commit 未生成）；`apt-get` 安装对应系统依赖（libnspr4 libnss3 libatk1.0-0 libatk-bridge2.0-0 libdbus-1-3 libatspi2.0-0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxkbcommon0 libasound2 libxfixes3）后重跑，render 通过，但 push 被并发 agent 推进的 `origin/main`（bpf-gpu-server 博客/报告 + 日报，commit `6145cd2df`，不触碰本 owned paths）拒为非 fast-forward；本地经 forward-move（`git reset --soft origin/main`，保留并发 agent 的 149 条 staged entries 与 worktree）重建 scoped commit 并 push 成功，得 `447823c76`。
- 该 QA commit 自身的 Pages `Deploy Static App` run `39080447994` 为绿色（`conclusion: success`）。首发验证 run 在 push 后的 20 分钟 public-poll 窗口内 Pages 仍在部署，public check 得 404；待该绿色 run 完成后 re-run 验证器，走 `skipped_already_published`（`content_test` / `build` / `render` / `commit_push` 均 skipped），receipt 翻为 `status=published`（commit `447823c769f8`）。
- 线上验证：四条路由（EN / ZH article + EN / ZH index）均返回 200，逐字 EN 与 ZH H1 均命中，新 slug 同时出现在两个 live 索引页，`cache-control: public, max-age=0, must-revalidate` 与 `cf-cache-status: DYNAMIC` 确认为新部署而非缓存副本。
- 操作备注：Chromium headless-shell 共享库缺失为**新故障模式**（与 10-05 的 content-test 硬计数断言、10-11/10-12 的 public-poll 404 与 forward-move 均不同），已随本日志记录；3600s 上限覆盖 build→render→commit+push→public-poll、commit 前与 push 前的 forward-move、以及对共享 checkout 上并发 agent staged entries 的 scoped `git commit --only`，均已由 10-12 / 10-13 的持久指引覆盖；本次 QA commit 自身 Pages run 即为绿色，无需取后续 commit 的 run。

This run-log commits separately as the 10-17 same-day artifact.
