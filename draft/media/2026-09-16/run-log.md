# 2026-09-16 eBPF 每日问答发布（America/Los_Angeles 自然日）

- 运行模式：eunomia-qa 定时问答巡检（eunomia-community-radar 路由）。任务日 2026-09-16，LA 时区 18:55 PDT 起执行。
- 候选来源：当日 28 条 opt-in Slack 归档消息（技术密度高）。选定问题——「为什么给 Node.js 服务启用 eBPF 追踪后，热路径 p99 会翻三倍，而 eBPF 探针本身很廉价？」，slug `nodejs-tracer-async-hooks-p99-cost`。与既往 21 篇无重复（最近的 OBI 篇为 2026-08-22 `obi-unused-prometheus-exporter-memory-growth`，主题不同：未使用的 Prometheus endpoint 内存增长 vs. 注入代理的每次异步操作成本归属）。
- 公开一手来源（全部 200 校验 + 本人逐一复核决定性引文）：
  - OBI 文档 [Trace context association](https://opentelemetry.io/docs/zero-code/obi/context-propagation/) / [Distributed traces](https://opentelemetry.io/docs/zero-code/obi/distributed-traces/) / [Trace-log correlation](https://opentelemetry.io/docs/zero-code/obi/trace-log-correlation/)：Node.js 支持明确为 "uses Node.js async hooks to refresh the active request context before async callbacks"。
  - OBI 源码（HEAD `e2fad9bb`）：`bpf/generictracer/nodejs.c`（`obi_uv_fs_access` uprobe 解码哨兵路径；`handle_fd_correlation` 写 `nodejs_fd_map`；`handle_async_switch` 刷新 `traces_ctx_v1`）；`pkg/internal/nodejs/fdextractor.js`（注入代理：`net` 原型包裹 + `async_hooks` `before` 钩子 + `fs.accessSync` 哨兵，注释明确「避免对每个非请求回调做同步系统调用」）；`bpf/shared/obi_ctx.h`（`traces_ctx_v1`，`BPF_MAP_TYPE_LRU_HASH`，`LIBBPF_PIN_BY_NAME`，OTEP 4855 合约）；`bpf/common/trace_parent.h`（`find_nodejs_parent_trace` 走 `nodejs_fd_map` → `fd_to_connection` → 服务端 trace，证实客户端 span 父化走 fd 对 map 而非 `traces_ctx_v1`）。
  - `go_runtime.c`、`logenricher.c`、`trace_lifecycle.h` 读 `traces_ctx_v1`（Go goroutine 交接 / 日志富化 / 客户端 span 结束后恢复服务端上下文），外加 OTEP 4855 pin 的外部关联表面——证实「只按手动 span + 日志富化 gate 哨兵会破坏外部 trace/profile 关联」。
  - Node.js [`async_hooks`](https://nodejs.org/api/async_hooks.html) 文档：`async_hooks` 标记为实验性，性能上推荐 `AsyncLocalStorage`；`destroy` 钩子会额外开启 Promise 实例 GC 跟踪开销。
  - 在途性能工作：OBI PR #3357（改进 Node 哨兵性能，草稿）；维护者方向 = 把「每回调上下文刷新」与「fd 对关联」解耦、按消费方 gate。
- 隐私核对：`privacy: ok`。页面仅引用公开文档、上游源码、OTEP/PR 编号，不含任何私有文本、身份、频道或链接；社区讨论部分做了匿名化（使用者/维护者角色化，去部署细节）。
- 本地校验：隔离用例 `eBPF Q&A routes resolve in both locales` 通过 1/1（~39s）。本机竞争下完整 `content.test.ts` 既有无关用例（blog/search）会卡住，属已记录前飞故障，按已发布复核路径完成。
- 提交与推送：commit `7c0f27022 docs(ebpf-qa): nodejs-tracer-async-hooks-p99-cost (2026-09-16)`。推送时远端 `origin/main` 已前进到 `e46d5f0b0`（agent-skills gitlink 同步，与本提交 4 个 docs 路径不相交）。因并发脏文件阻断 `rebase`，改用 `git reset --soft origin/main`（只移动指针、不动工作树）+ 重新 `add` 四个路径 + `commit --only` + `git push`。`7c0f27022` 已在 `origin/main`。
- 校验器：`receipt-2026-09-16.json` `status: published`（0600），全部检查 ok（`remote_contains_commit: ok`、`public: ok`、`privacy: ok`、`index_links: ok`）。`content_test`/`build`/`render`/`commit_push` 走已发布复核路径。首次校验器运行因本地 commit 尚未推送报 `not contained`；推送远端前进提交后复跑即通过。
- 线上核验（缓存穿透 curl，UA 伪装）：EN/ZH 页面 200，H1 命中；两个 `/ebpf-qa/` 与 `/zh/ebpf-qa/` 索引均链接 slug。
- 并发文件保持原样（未暂存、字节不变）：`.agents/sources/agent-skills`（gitlink）、`app/lib/site-config.generated.ts`、`app/next.config.mjs`、`.github/data-analysis/latest.md`（远端前进新增的脏文件）、`draft/media/2026-09-13/run-log.md`。
- 原始快照临时目录（chmod 700，`snapshot=ok bytes=11597 messages=28`）已删除；未提交任何快照原文。
