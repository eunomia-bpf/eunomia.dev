# 2026-10-21 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-10-21。唯一提交时刻为公开页 JSON-LD `datePublished` 2026-10-21T16:13:55-07:00 = 16:13 PDT（= 2026-10-22 07:13:55 +08；创作者中心文章行显示 2026-10-21 16:13），落在 LA 2026-10-21。巡检开始时补发缺口为 0 条（2026-09-03 与 2026-09-16 两条缺口已于 2026-10-02 巡检核销；10-09 → 10-10 → 10-11 → 10-12 → 10-13 → 10-14 → 10-15 → 10-16 → 10-17 → 10-18 → 10-19 → 10-20 → 10-21 连续巡检，无新增补发缺口），xpu/npu-kernel-driver 使用正常额度，未核销任何缺口；本次结束后仍无补发缺口。知乎任务仍 `阻塞`——队列 `阻塞` 条款自 2026-09-17 起未恢复（本次未重新探测知乎，亦跳过队列内全部知乎 `排队` 行），故首个可执行的 `排队` 任务为掘金 xpu/npu-kernel-driver（队列第 132 行）。

## 队列任务：掘金 xpu/npu-kernel-driver（LA 10-21 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 132 行（`排队`，现翻为 `[x]`）。上一可执行掘金行第 131 行 xpu/gpu-kernel-driver 已于 10-20 巡检完成；本行翻转后队列不再剩掘金行（剩余队列共 24 个平台任务，全部为知乎侧）。
- 源文：`docs/tutorials/xpu/npu-kernel-driver/README.zh.md`；发布稿 `draft/media/2026-10-21/xpu-npu-kernel-driver/juejin-body.md`（移除源 H1，其余正文逐字保留；9180 字符、15795 字节、280 行、9 个 H2 / 2 个 H3 / 0 个 H4、9 个代码块 [9 bash，全部带语言标签]、0 张图片、0 个表格、5 条唯一外链目标 [github.com/eunomia-bpf/bpf-developer-tutorial、eunomia.dev/tutorials/、github.com/intel/linux-npu-driver、github.com/iovisor/bpftrace/blob/master/docs/reference_guide.md、github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/xpu/npu-kernel-driver]，含源尾注 blockquote [eunomia.dev/tutorials + 教程仓库链接]）。
- 标题逐字取源 H1「eBPF 实例教程：跟踪 Intel NPU 内核驱动操作」（源 H1 无系列编号，发布稿不另加编号）。
- 草稿未预置：`https://juejin.cn/editor/drafts/new` 新编辑器会话（无预置草稿）；分类 `后端`；标签 `Linux`、`后端`、`性能优化`（沿用 17-biopattern / 19-lsm-connect / 1-helloworld / bpftrace-tutorial / cgroup / bpf_arena / bpf_iters / bpf_token / bpf_wq / dynptr / struct_ops / xpu-flamegraph / xpu-gpu-kernel-driver 教程标签组）。

## 提交

- 提交时刻以公开页 JSON-LD `datePublished` 2026-10-21T16:13:55-07:00 = 16:13 PDT 为准（= 2026-10-22 07:13:55 +08；创作者中心文章行显示 2026-10-21 16:13，两处一致）。提交前基线承 10-20 xpu/gpu-kernel-driver 巡检放行后读数 全部 90 / 已发布 90 / 审核中 0 / 未通过 0。单次弹窗「确定并发布」返回「发布成功」并落到 `https://juejin.cn/published`。新文 post id `7698947426490318888`，URL `https://juejin.cn/post/7698947426490318888`。

## 提交结果与判定

- 进入审核间隔：提交后暂存地址 `/spost/7698947426490318888`；本 run 未记录窗口内中间读数，按结果判定——审核间隔结束后创作者中心读为 全部 91 / 已发布 91 / 审核中 0 / 未通过 0，随后暂存 URL 重定向至规范地址（登录态浏览器打开 `/spost/7698947426490318888` 落到 `https://juejin.cn/post/7698947426490318888`）。
- 记 `confirmed`，带 `staged_url`（审核间隔先例：38-btf-uprobe / 17-biopattern / 1-helloworld / cgroup-tutorial / bpf_arena / bpf_iters / bpf_token / bpf_wq / dynptr / struct_ops / xpu/flamegraph / xpu/gpu-kernel-driver；区别于 bpftrace / 26-sudo / 20-tc / 31-goroutine 直接公开条目无 `staged_url`）。NPU 是本系列首个观测到暂存 URL 后续重定向至 `/post/` 的放行先例。

## 公开页 QA（登录浏览器）

- xpu/npu-kernel-driver `/post/7698947426490318888`：标题逐字「eBPF 实例教程：跟踪 Intel NPU 内核驱动操作」（页面级 H1 仅 1 次，正文区域无重复 H1）；单份正文探测（9180 字符 artifact；渲染 DOM innerText 读为 8687，代码块空白归一化）；9 个 H2 / 2 个 H3 / 0 个 H4；9 个代码块 [9 bash，全部带语言标签，渲染 DOM 9 个 `<pre>`，与源 9 个围栏代码块 1:1]；0 张图片；0 个表格；5 条唯一外链目标全部改写为 `link.juejin.cn?target=`、0 条裸外链；无 `审核中` / `文章有更新` / `已被删除` / `找不到页面` / `发布失败` 标记；公开页 `暂无评论`（0 评论、0 点赞、0 收藏）。

## 台账更新

- `platforms/juejin.json`：置顶新增 `juejin-7698947426490318888` 一条 `confirmed`（newest-first；带 `id`、`status`、`title`、`url`、`staged_url`、`source_path`、`checked_via` 与 notes，审核间隔故带 `staged_url`），`published[]` 78 → 79，`last_checked` → 2026-10-21。
- `sources.json`：`last_checked` → 2026-10-21（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-10-21；xpu/gpu-kernel-driver 叙述段后补记 2026-10-21 xpu/npu-kernel-driver 叙述行（JSON-LD 16:13 PDT、无补发缺口核销、基线 90/90/0 → 91/91/0、/spost/ 现重定向 /post/）；Juejin confirmed 表格首行置顶 xpu/npu-kernel-driver（newest-first，含 `/post/` 正式地址与 QA 摘要）。
- `not-published.md`：`Last checked:` → 2026-10-21；掘金未映射 32 → 31、映射 77/109 → 78/109；滚动队列 1 → 0 个 Juejin tasks（共 25 → 24 平台任务）；Juejin 状态行前置 xpu/npu-kernel-driver 子句、机器台账计数刷新为 78/109 / 31 未映射 / 0 open rolling-queue tasks；知乎状态行不变（68/109 mapped / 41 未映射）。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-10-21；补发缺口段落前置 10-21 巡检句（正常额度 1 条、JSON-LD 16:13 PDT、/spost/ 暂存、90/90/0 基线 → 91/91/0 放行、/spost/ 现重定向 /post/）；Ledger 基线按检查器更新（知乎 41、掘金 31，映射 78/109）；剩余队列 24 个知乎任务（23 `排队` + 1 `阻塞`）、0 个掘金任务，共 24 个平台任务；第 132 行翻 `[x]` 并附 JSON-LD 提交时刻、放行结果、分类/标签、QA 摘要、正式地址与 `confirmed`；知乎任务保持 `阻塞`。
- `community-feedback.md`：在已有 `### 2026-10-20` 块上新增 `### 2026-10-21` 块（xpu/npu-kernel-driver 发布 + QA + 放行后 top-ten 复核 + 前向指针：掘金队列 0 剩余任务，下一巡检点无新发布、仅复核跟踪集）；未在 10-20 块内再织入。
- 发布稿 `draft/media/2026-10-21/xpu-npu-kernel-driver/juejin.md`：状态改「已确认公开」（post `7698947426490318888`，`confirmed`），记录正式地址、JSON-LD 提交时刻、/spost/ → /post/ 重定向与 QA 摘要，并保留正文/标题/原文/分类/标签各行。
- 验证器：`check_media_ledger.py` 通过（Juejin 78/109 映射、31 未发布、79 confirmed；Zhihu 68/109 映射、41 未发布；exit 0）。CRLF 台账经字节级编辑，编辑后复查 0 条裸 LF。

## 编辑器经验

- 标签/分类提交沿用 10-20 双通道经验（先试 CDP 指针点击 `.byte-select-option`，未命中即在选项中心改走页内合成点击）；本 run 成功提交 `后端` 分类与 `Linux`、`后端`、`性能优化` 三枚 chip，无新编辑器故障。
- 新审核间隔观察：与 10-20 xpu/gpu-kernel-driver run（审核窗口内 SPA shell 对 `/post/` URL 返回 HTTP 200、可用性须以登录创作者中心判定）不同，本 run 观测到审核间隔结束、`/spost/` 后续重定向至 `/post/`（打开暂存地址即落到规范地址）；窗口内中间读数未记录，判定以上述结果读数为准。计数读取沿用 `.essays-container` 按 `2026-MM-DD HH:MM` 日期行切分配对法（展现/阅读在标题与日期行之后，行序为 标题 → 日期 → N展现 → · → N阅读 → …）。

## 跟踪 top ten 复核（创作者中心放行后逐行读取）

- 新文 xpu/npu-kernel-driver 首读 302 展现 / 20 阅读 / 0 点赞 / 0 评论 / 0 收藏（新文首读；窗口内三次读取计数 263 / 19 → 284 / 19 → 302 / 20 递增）。
- 流量累积（对比 10-20 放行后读取）：[xpu/gpu-kernel-driver](https://juejin.cn/post/7698552541928554511) 230 / 7 / 1 点赞 / 1 收藏 → 4504 / 42 / 1 点赞 / 1 收藏（发布以来累计）；[xpu/flamegraph](https://juejin.cn/post/7698287001586040872) 393 / 19 → 527 / 20；[struct_ops](https://juejin.cn/post/7697622920013709347) 4 / 10 → 8 / 10；[dynptr](https://juejin.cn/post/7697186290753159202) 2260 / 51 / 0 点赞 → 2312 / 51 / 1 点赞；[bpf_wq](https://juejin.cn/post/7697094073967116323) 9 / 17 → 14 / 18；[bpf_token](https://juejin.cn/post/7696759034846445583) 932 / 31 → 971 / 31；[bpf_iters](https://juejin.cn/post/7696384745624141824) 810 / 21 → 824 / 28；[bpf_arena](https://juejin.cn/post/7695977967902933026) 2495 / 62 → 2508 / 64；[cgroup](https://juejin.cn/post/7695667191396040738) 8 / 19 → 12 / 23，入 #10 位；bpftrace-tutorial（10-20 第 10 位 4 / 10）旋出本窗口跟踪 top ten。
- 评论活动：新文 xpu/npu-kernel-driver 公开页 `暂无评论`；被跟踪 top ten 0 新增评论/回复（创作者中心评论卡片为懒渲染，评论以公开页为准）。
- 复核方法：登录创作者中心 `https://juejin.cn/creator/content/article/essays?status=all`，对 `.essays-container` 的 innerText 按 `2026-MM-DD HH:MM` 日期行切分逐行配对展现/阅读（行精确，避免跨行错位配对）。

## 后续

- 掘金队列已 0 剩余任务（剩余 24 个平台任务全部为知乎侧，仍 `阻塞` 于登录状态恢复条件）。下一 LA 2026-10-22 正常额度无新掘金任务可执行；下一有用检查点为创作者中心跟踪 top ten 复核（无新发布）。

This run-log commits separately as the 10-21 same-day artifact.
