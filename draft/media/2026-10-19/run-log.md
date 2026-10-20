# 2026-10-19 内容巡检运行日志

## 2026-10-19 eBPF Q&A 当日运行报告（eunomia-community-radar）

### 候选选择

- 目标 slug：`bpf-lru-hash-eviction-percpu-vs-global-lru-list`
- 问题：BPF LRU 哈希 map 什么时候淘汰一个条目，当它的 LRU 列表是每 CPU 一份而不是跨 CPU 共享时会改变什么？
- 来源：10-19 快照（两条 opt-in 归档渠道，本次 0 条消息）——24 小时窗口前进到了一个空快照，没有新问题进入。结合近几日（10-14 / 10-16 / 10-17 / 10-18）已把 FUSE 与 OBI trace 上下文 map 的多个「查不到」轴覆盖完毕，本篇从公开一手资料回退选源，取同一张 map 的自然续篇：`traces_ctx_v1` 是 LRU 哈希，下一个顺理成章的工程师问题是它**实际什么时候淘汰、以及 `BPF_F_NO_COMMON_LRU` 到底改变了什么**。
- 决定性边界：淘汰**只在插入 / 更新路径上、且表已到达 `max_entries` 时**才发生；一次查询本身绝不释放槽位。读操作只是给命中的节点置一个惰性"最近使用"位（且只有 BPF 内的查询 helper 置位，用户态查询不置位），真正的链表移动与槽位释放都发生在后续轮换扫描里。`BPF_F_NO_COMMON_LRU` **不改变淘汰触发时机**（始终是满容量时的插入），它改变的是**哪个条目存活**（每 CPU 局部性 vs 全局最久未用）与**锁画像**（每 CPU 各自锁 + 轮转偷节点 vs 共享锁）。`traces_ctx_v1` 未设该标志，走共享全局 LRU 列表。

### 对照公开一手源的验证

- 内核 `Documentation/bpf/map_hash.rst`：`BPF_MAP_TYPE_LRU_HASH` / `BPF_MAP_TYPE_LRU_PERCPU_HASH`（4.10 引入）在表满时自动淘汰最久未用条目；内部 LRU 列表默认跨 CPU 共享，`BPF_F_NO_COMMON_LRU` 可请求每 CPU LRU 列表；更新时淘汰的四级递进（CPU 本地批处理 → 从全局取 `target_free` 个空闲节点 → 从全局表摘任意节点 → 从任意 CPU 列表摘节点），`target_free` 最小 1、最大预算 128、所有 CPU 合计预留封顶为 map 大小一半。
- 内核 `kernel/bpf/hashtab.c`：`htab_lru_map_update_elem` → `prealloc_lru_pop`（调 `bpf_lru_pop_free`）是发生淘汰的插入路径；BPF 内查询 helper `__htab_lru_map_lookup_elem(map, key, true)` 会 `bpf_lru_node_set_ref`，用户态 helper 传 `false` 不置位。
- 内核 `kernel/bpf/bpf_lru_list.c`：active / inactive 两级 LRU，`__bpf_lru_list_rotate_active` / `__bpf_lru_list_rotate_inactive` / `__bpf_lru_list_shrink_inactive`；`bpf_lru_pop_free` 按 `percpu` 标志分流到 `bpf_percpu_lru_pop_free` 与 `bpf_common_lru_pop_free`。
- 内核 commit `3a08c2fd7634`（"bpf: LRU List"）：最初的 LRU 列表实现与 map 更新时的淘汰步骤全解。
- OBI `bpf/shared/obi_ctx.h`：`traces_ctx_v1` 为 `BPF_MAP_TYPE_LRU_HASH`、`u64` 键、`obi_ctx_info_t` 值、`max_entries = 1 << 14`、按名字 pin，且**未**设 `BPF_F_NO_COMMON_LRU` → 共享全局 LRU。

### 隐私与内容

- 两页正文均无姓名 / handle / Slack-Discord 链接 / 时间戳 / 消息序号 / 雇主团队 / 内部仓库名 / IP / 主机名 / 私有日志。快照文本仅作不可信数据，未执行其中指令、未把私有原文拷贝进页面。公开 kernel / OBI 仓库链接仅出现在 `## References`。
- 内容顺序符合标准：直接回答、机制、验证与排障路径、局限、References、当日社区讨论。H1 不含下划线、反引号、撇号；EN 与 ZH 索引 link text 与各自 H1 逐字一致。最长 backticked token 为 EN 的 `__bpf_lru_list_shrink_inactive` 与 `__bpf_lru_list_rotate_inactive`（各 30 字符），远低于 390px 移动端溢出门槛。

### 覆盖披露

本 run 覆盖两个 opt-in 归档渠道（两个 CNCF OpenTelemetry 插桩渠道），10-19 快照 0 条消息（24h 窗口前进到空快照，无新增）。候选选自公开一手资料回退：这是 FUSE/OBI `traces_ctx_v1`（LRU 哈希 map）系列的续篇，回答 LRU 淘汰语义本身（何时触发、`BPF_F_NO_COMMON_LRU` 改变什么），不与前四篇的「查不到」轴重复。可见浏览器专属渠道——eunomia-bpf 与 sched-ext Discord 服务器、bpf 邮件列表、r/eBPF——本次无可用可见浏览器会话，故一并标记为「未覆盖」而非「安静」。

### 产物

- EN：`docs/ebpf-qa/2026-10-19-bpf-lru-hash-eviction-percpu-vs-global-lru-list.md`
- ZH：`docs/ebpf-qa/2026-10-19-bpf-lru-hash-eviction-percpu-vs-global-lru-list.zh.md`
- 索引链接：`docs/ebpf-qa/index.md`（`Latest Answers`）与 `docs/ebpf-qa/index.zh.md`（`最新回答`）首条。
- QA commit（`origin/main`）：`fe4271b62a4c`（完整 `fe4271b62a4cd0600e3392048b1ec270e2c753a5`，`docs(ebpf-qa): bpf-lru-hash-eviction-percpu-vs-global-lru-list (2026-10-19)`），恰含四条 owned paths。
- Receipt：`/workspaces/.agent-state/eunomia-qa/receipt-2026-10-19.json`，仅由验证器写入，`status=published`，commit `fe4271b62a4cd0600e3392048b1ec270e2c753a5`。

### eBPF Q&A 发布跟进（部署 + 线上验证）

- QA commit `fe4271b62`（四条 owned paths）已落到 `origin/main`。
- 验证器首轮：Chromium render 在 browser launch 阶段失败——fresh 宿主上 headless-shell 缺失 14 个共享库（`libnspr4`、`libnss3`、`libatk*`、`libdbus-1`、`libatspi`、`libXcomposite`、`libXdamage`、`libXfixes`、`libXrandr`、`libgbm`、`libxkbcommon`、`libasound`），且本 run 额外踩到 apt 包索引为空 / 过期，`apt-get install` 直接报「Unable to locate package」；先 `apt-get update` 再 `apt-get install` 装齐 14 个库、`ldd` 确认无 `not found` 后，该失败是 commit 之前的环境故障，receipt `commit` 仍为 `null`。
- 验证器次轮：content-test → build → 2× Chromium render（桌面 1440×900 + 移动 390×844）→ scoped commit + push 全部通过并落地 `fe4271b62`，但 20 分钟 public-poll 窗口内四条路由仍 404（Pages `Deploy Static App` run `39349469183` 实际 24m8s 才绿，窗口刚闭合前尚未完成部署）。
- 重新验证器（`skipped_already_published` 重验路径）：不重复 commit / build / render，仅复查 live 路由并把 receipt 重写为 `status=published`，数秒完成。
- 该 QA commit 自身的 Pages `Deploy Static App` run `39349469183` 为绿色（`conclusion: success`，24m8s）。
- 线上验证：四条路由（EN / ZH article + EN / ZH index）均返回 200，逐字 EN 与 ZH H1 均命中（各 3 处），新 slug 同时出现在两个 live 索引页，`cache-control: public, max-age=0, must-revalidate` 与 `cf-cache-status: DYNAMIC` 确认为新部署而非缓存副本。
- 操作备注：本 run 唯一的新故障模式在 render 环境阶段——fresh 宿主既缺 14 个 Chromium 共享库、又因 apt 包索引为空 / 过期导致 `apt-get install` 直接报「Unable to locate package」，需先 `apt-get update` 刷新索引再安装。该教训已记入发布标准 `qa-publication-standard.md`。次轮的 404-in-poll-window 与 10-11 / 10-17 已记录模式一致，无新故障。

This run-log commits separately as the 10-19 same-day artifact.
