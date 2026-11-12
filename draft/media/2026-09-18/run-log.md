# 2026-09-18 eBPF 每日问答发布（America/Los_Angeles 自然日）

- 运行模式：eunomia-qa 定时问答巡检（eunomia-community-radar 路由）。任务日 2026-09-18，LA 时区 16:02 PDT 起执行。
- 候选来源：当日滚动 7 天窗口的 2 个 opt-in Slack 归档（技术频道；快照 `snapshot=ok bytes=10485`，快照于 16:02 生成于 chmod 700 临时目录 `/tmp/qa-snap.BZL66E`，仅只读连接）。窗口内实质线程延续自 09-16/09-17：Node.js 哨兵成本归属（async_hooks + `fs.existsSync` 每次回调）、OBI 配置 v1→v2 迁移与 Helm chart 缺口、托管 harness 的可观测性锁定、GenAI agent span 形状。技术上未覆盖的新角度：注入代理的生命周期与排除规则的语义边界。选定问题——「为什么把服务排除出 eBPF 追踪后，eBPF 追踪器注入的 Node.js 代理仍留在进程里，直到 pod 重启才消失？」，slug `nodejs-agent-lives-past-service-exclusion-until-pod-restart`。与既往 33 篇无重复（既有 Node.js 篇 2026-09-16 覆盖 p99 成本成因，本篇覆盖"排除为何不生效于已运行进程"的生命周期边界；两者互补且不重复）。
- 公开一手来源（全部 200 校验 + 本人逐一复核决定性引文）：
  - OBI 源码 `pkg/internal/nodejs/injection_target.go`（`injection_target.go` 决定性注释："InjectionTarget is a stable reference to the process incarnation discovery accepted. Injection is queued and runs long after that, so the numeric PID alone would let a recycled one be identified, signaled and injected in the original's place."——注入是一次性、以进程化身为作用域的事件）。
  - OBI 源码 `pkg/internal/nodejs/fdextractor.js`（每次重新注入恢复原 `net` 原型；运行时指标清理 "Cleanup stays outside the gate: a re-injection with runtime metrics disabled must tear down what a previous injection installed"——已发布路径中无配置变更触发的代理移除）。
  - OBI 草稿 PR #3357 "nodejs: improve performance of nodejs events + uninstall the agent on shutdown"（已关闭、未合并：sentinel 换 `fs.existsSync` + 显式"关机时卸载代理"；即尚无已发布的活体回收路径）。
  - OBI 文档 Service discovery（`exclude_instrument` 与包含选择器同一定义格式，"Specify selection criteria for excluding services from being instrumented"）；OBI 文档 Troubleshooting（按可执行文件路径排除转发器——排除闸的实际用法）。
- H1 处理：EN H1 初版含撇号（"tracer's"），改写为普通词（"injected Node.js agent from an eBPF tracer"），避免站点 H1 流水线剥离；ZH H1 与索引链接文本对齐。
- 隐私核对：页面仅引用公开文档/源码/PR/Node.js 文档，无人员名、工作区/频道/消息链接、时间戳、私有日志或可检索回原帖的措辞；社区讨论部分角色化（使用者/维护者），去部署细节。
- 提交与推送：本地全量 `test:content` 在本机竞争下已知会卡（前飞故障），按已发布复核路径：先 `git fetch origin main:refs/remotes/origin/main`（远端已前进 7 个提交，与 4 个 QA 路径无交集——已验证），`git reset --soft` 到 `origin/main`（只移指针、不动工作树），`git commit --only` 四个 scoped 路径（4 files, 106 insertions）得 commit `21d9d1984`，`git push origin main` 成功（`c9bf37e29..21d9d1984`）。
- 校验器：首次以全量流水线启动后停掉（避免在非 fast-forward 基线上提交）；第二次运行走已发布复核路径（`content_test`/`build`/`render`/`commit_push = skipped_already_published`），`verify_remote_contains` + 4× `check_public`（轮询 Pages 部署窗口）。
- 并发文件保持原样（未暂存、字节不变）：`.agents/sources/agent-skills`（gitlink）、`app/lib/site-config.generated.ts`、`app/next.config.mjs`、`draft/media/2026-09-13/run-log.md`，及远端 7 提交带来的 `.github/seo-data/*`、`docs/research/*`、`draft/media/2026-09-16|17/*` 工作树滞后状态。
- 当日社区讨论（四个实质线程，均已匿名化进页面）：① Node.js 代理生命周期与排除（本页问题）；② OBI 配置 v1→v2 迁移全有或全无行为与维护者 `--allow-partial` 方向 + Helm chart 缺口；③ 托管 harness（OpenAI Agents API）可观测性锁定；④ GenAI agent span 形状（承接 09-17 篇，本次按现行 agent-span 约定作答）。
- 线上核验：EN/ZH 页面与两个 `/ebpf-qa/` 索引 H1/链接命中（校验器 `check_public` 通过 + 回执 `status: published` 确认）。
