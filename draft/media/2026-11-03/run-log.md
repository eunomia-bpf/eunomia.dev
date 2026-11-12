# 2026-11-03 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：LA 2026-11-03（系统日期 2026-11-04 +08）。本次巡检的实质产出是**补齐一条被平台级失败吞掉的 DEV 发布，并修掉根因**。队列里已经没有其他可执行任务：24 条知乎仍平台级 `阻塞`，掘金 / Medium / DEV / LinkedIn 均为 0 条。

## 关键发现：Dev.to 的 422 被 API 包进 200 响应，导致队列项被误删

- 定时 workflow `publish-posts`（run 41656753202，2026-11-03T14:45Z）把 `docs/blog/posts/bpf-gpu-server.md` 的 Medium + DEV 一起发了出去。Medium 成功（`medium-d8c070b5bd42`），DEV 返回 `422 Tag "io_uring" contains non-alphanumeric or prohibited unicode characters`。
- API 把这条失败放在 HTTP `200` 的 `results` 数组里（`{"platform":"devto","success":false,"error":"..."}`），而旧的 `publish.py` 只看 HTTP 状态、不看 `results`，于是打印 `✅ Publish 1/1 successful!`、把队列项 `pop` 掉并提交。DEV 副本随队列一起消失，且没有任何残留可以只重试 DEV——默认 `platforms` 是 `[medium, devto]`，直接重跑会把 Medium 重复发一遍。
- 这也解释了为什么该源在 LA 11-03 提前发布：定时 workflow 每周二 07:00 UTC 跑一次，队列项在 11-03 的窗口里被消费掉，原本排的 11-09 额度用不上了。

## 修复

- `.github/publisher/publish.py`：新增 `summarize_result()` 解析 `results`，只有请求平台全部成功才出队；任一平台失败时打印失败平台与错误、保留队列项、`exit 1`（Action 变红），并提示用 `PUBLISH_PLATFORMS` 只重试失败平台。新增 `get_platforms()` 读取 `PUBLISH_PLATFORMS`，支持 `devto` / `medium` / `devto,medium`，非法值直接报错退出。
- `.github/workflows/publish-posts.yml`：新增 `platforms` 手动输入（默认空 = 两个平台），映射到 `PUBLISH_PLATFORMS`。
- `.github/publisher/README.md`：补上平台选择、部分失败的语义和 DEV 标签规则（`io_uring` 这类含下划线的标签会被 422）。
- 回归验证（临时目录 + 假 `requests.post`，不进仓库）：两平台全成功 → 出队 rc=0；仅 devto → 出队 rc=0；devto 失败 + medium 成功 → 保留队列 rc=1 并给出重试提示；`PUBLISH_PLATFORMS=devto,twitter` → 直接报错且不动队列。

## DEV 补齐发布

- 用 `gh workflow run publish-posts.yml -f platforms=devto -f draft_only=false -f publish_count=1` 走 secret 背书的正式路径（本地 shell 无 `DEV_TO_API_KEY` / `MEDIUM_API_KEY` / `PUBLISH_PASSWORD`），run 41761210743 成功。
- 公开页 <https://dev.to/yunwei37/does-moving-a-gpu-servers-request-loop-into-the-kernel-save-cpu-d72>，`2026-11-04T02:30:51Z`，article id 4972601，tags `ebpf, gpu, kernel`，8 分钟阅读，0 反应 0 评论。
- 双路径复核一致：公开 API 的 `title` / `tag_list` / `published_at`，以及可见页 DOM 的 1 个 H1、8 个正文 H2（对应源文 8 个 `##`）、1 个内联 SVG（Mermaid）、0 个 `pre`、3 行结果表、两处论文链接为绝对地址、完整尾段。
- 准备稿：`draft/media/2026-11-03/bpf-gpu-server/devto.md`（源文去掉 frontmatter 与内部 `<!-- more -->` 标记）。

## 源链接缺陷（对后续同步有意义）

- 源文里的 `../../papers/bpf-gpu-server.pdf` 只在站点渲染器下有效。DEV 副本在 artifact 中被改成 `https://eunomia.dev/_content-assets/docs/papers/bpf-gpu-server.pdf`（实测 200、`application/pdf`、403337 字节），公开页因此可点。
- Medium 副本由 bot 直接发送源文、未做替换，两处链接落到 `https://medium.com/papers/bpf-gpu-server.pdf`（403）。`https://eunomia.dev/papers/bpf-gpu-server.pdf` 本身也是 404，说明不能简单去掉 `../../` 前缀。

## Medium 副本复核

- `d8c070b5bd42`：正文完整（约 1,976 词、8 个正文 H3、完整尾段、GitHub 链接正常），但有三个与 `5001bb9afab6` 相同的 API 导入缺陷：无可见标题（`h1: 0`，正文从 Mermaid 块开始）、Mermaid 围栏保留为 flowchart 源码、结果表压平成 `| … | — -|` 行文；另加两处死链。
- 修复路径仍被挡住：`/p/d8c070b5bd42/edit` 在可见浏览器里重定向到 `/m/signin`，medium.com 没有可用登录态。`5001bb9afab6` 的标题修复也仍 open，`59aff1cc69a5` 的 10-H2 结构漂移仍在。

## 会话与平台基线

- DEV：创作者页 `/dashboard` 落到 `/magic_links/new`（登录态失效），但公开 API 与公开页均可读；公开 author API 现列 65 篇。
- 知乎：`/creator` 仍重定向到 `/signin?next=%2Fcreator`，恢复条件未变。
- 掘金：`/creator/home` 匿名访问直接落到首页，`/creator/data/content/article` 为「找不到页面」，说明创作者中心本次不可用；改用公开文章页读阅读数，npu-kernel-driver 117、gpu-kernel-driver 89、flamegraph 55、dynptr 69、bpf_token 71（对比 11-02 的 113 / 85 / 53 / 68 / 71，仍在缓慢上升，无评论）。
- 凭据：本 shell 中 `MEDIUM_API_KEY`、`DEV_TO_API_KEY`、`PUBLISH_PASSWORD` 均不存在。

## 外部回声

- Hacker News 两条第三方提交仍在：`50138075`（bpf-gpu-server，1 分 0 评论，`thatslast`）与 `49854228`（qwen3，7 分 3 评论）。可见 item 页对匿名浏览器返回 `Sorry.`，但公开 Algolia 数据可读，本次未见新评论或分数变化。

## 台账更新

- `platforms/devto.json`：新增 `devto-d72`（`confirmed`）、`confirmed_source_urls` 补 `docs/blog/posts/bpf-gpu-server.md`、新增 11-03 observation、`last_checked` → 2026-11-03。
- `platforms/medium.json`：新增 `medium-d8c070b5bd42`（`confirmed`，含三个渲染缺陷与死链）、映射补一行、新增 11-03 observation、`last_checked` → 2026-11-03。
- `platforms/hackernews.json`：`last_checked` → 2026-11-03，新增 11-03 observation（Algolia 可读、可见页 `Sorry.`）。
- `published.md`（LF）：`Last checked:` → 2026-11-03；Medium 与 DEV 各新增一行，映射数 Medium 67 → 68/127、DEV 54 → 55/127。
- `not-published.md`（CRLF）：`Last checked:` → 2026-11-03；Medium 与 DEV 状态行改写为最新确认项与仍然 open 的修复项。
- `sources.json`：`last_checked` → 2026-11-03。
- 队列：新增 2026-11-03 小节，`bpf-gpu-server` 的 Medium 与 DEV 两项均标记 `[x]`，并注明原 11-09 额度已被消耗。
- 验证器：`python3 .github/publisher/media/check_media_ledger.py` **exit 0**（DEV 55/127、63 条 confirmed；Medium 68/127、74 条 confirmed）。

## 后续

- Medium 三条 story 的标题 / Mermaid / 表格修复（`5001bb9afab6`、`d8c070b5bd42`）都需要用户在持久可见 Chrome 重新登录 Medium；`59aff1cc69a5` 的结构漂移继续按「不重发、不静默重同步」处理。
- 下一次 API 发布英文长文时，应在 payload 里带上 `canonical_url` 并把相对链接改写为绝对地址，避免再产生自指 canonical 与 `/papers/...` 死链。
- 源文 `docs/blog/posts/bpf-gpu-server.md` 的相对 PDF 链接建议改为 `https://eunomia.dev/_content-assets/docs/papers/bpf-gpu-server.pdf`，但本次未改动源文（避免在发布提交里夹带内容变更）。
- 知乎恢复条件不变；下一次巡检仍需先读 `/creator` 是否进入创作中心。