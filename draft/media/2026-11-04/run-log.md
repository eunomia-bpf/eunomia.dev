# 2026-11-04 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：LA 2026-11-04（系统日期 2026-11-05 +08）。本次巡检的实质产出是**确认一条已由定时 workflow 发出的 DEV 发布、补齐台账与队列，并定位一个可复现的 DEV 图片缺陷**。队列里没有其他可执行任务：24 条知乎仍平台级 `阻塞`，掘金 / Medium / DEV / LinkedIn 均为 0 条；补发缺口 0。

## 发布事实

- `docs/blog/posts/gthulhu-vllm-cpu-pressure.md` 的 DEV 副本由定时 workflow `publish-posts`（run 41949983323，2026-11-05T01:39:11Z 开始，`Platforms: devto`、`🚀 LIVE MODE`）发布，`published_at` = `2026-11-05T01:39:39Z` = LA 11-04 17:39 PST。
- 公开地址 <https://dev.to/yunwei37/when-does-cpu-scheduling-help-ai-inference-5did>，article id 4978534，tags `ebpf, linux, ai, performance`，reading_time 11，0 反应 0 评论。
- 发送正文与准备稿逐行一致：`draft/media/2026-11-04/gthulhu-vllm-cpu-pressure/devto.md` 去掉首页 `# ` 标题后 359 行全部相同（publisher 的 `strip_h1` 行为）。
- 结构对账（源文 vs 公开页）：1 h1、9 个正文 H2、11 个 H3、6 个 `pre`（12 个围栏）、5 个表格（源文 5 个 table 块）、2,651 词、References 尾段完整。

## 缺陷：外部 SVG 图表在 DEV 全部显示为空

- 可见页两张图的 `<img>` 都是 `naturalWidth 0` / `naturalHeight 0`，`clientWidth` 680，`currentSrc` 为空；`Image().onload` 报错。
- 渲染 URL 是 DEV 自己的签名代理：`https://media2.dev.to/dynamic/image/width=800,height=400,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fgthulhu.org%2Fassets%2Fvllm-gthulhu%2Fchart_*.svg`。
- 直接探测该 URL：`200`、`content-type: image/webp`，但响应体是裸 SVG（3181 / 2313 字节，`<?xml ... <svg`）。Chromium 按 `image/webp` 解码 SVG 字节失败，因此 `<img>` 空。
- 同一代理的对照：把 `format=auto` 换成 `format=png`，返回 `200 image/svg+xml`、900x520，能正常解码；直接取 `https://gthulhu.org/assets/vllm-gthulhu/chart_*.svg` 也正常（900x520）。
- 因此根因是 DEV 代理对**外部 SVG** 的 `format=auto` 重编码走了「标成 webp 但原样回传 SVG」的路径，与源图本身无关；把源换成 PNG 或改用编辑器上传即可绕开，但 `https://dev.to/edit/...` 与 `/dashboard` 仍重定向到 `/magic_links/new`（未登录），本次无法修。
- 该缺陷已写入 `platforms/devto.json`（条目 `notes` + `browser_observations` + `pending_verification`）与 `published.md`，并在 `.agents/skills/devto-publisher/references/platform-preferences.md` 增加一条可复用规则。

## Medium 复核（未修复）

- `59aff1cc69a5`（gthulhu 的 Medium 副本）本次多次命中 Cloudflare `504`（`Ray ID a559ee67ef9a8acf`），可读到的一次结果是 1 个 h1、11 个 H2、11 个 H3、7 个 `pre`、0 个 `table`、2 个 `<figure>`、图片 64x64 头像 + 3 个 `naturalWidth 0` 的槽位——即 11-02 记录的 9-H2 结构漂移仍在，图片槽位也仍空。
- 重试用尽后仍返回网关页，故 `d8c070b5bd42` 本次未重读；两篇的标题 / 表格缺陷继续 open，等可见 Medium 会话。

## 台账与队列

- `platforms/devto.json`：新增 `devto-5did`（`confirmed`）、`confirmed_source_urls["docs/blog/posts/gthulhu-vllm-cpu-pressure.md"]`，`last_checked` 2026-11-04，新增 1 条 `browser_observations`，`pending_verification` 去掉已完成的 65-post 项并加上 66-post 与 SVG 两条。
- `sources.json` / `published.md` / `not-published.md` / `platforms/medium.json`（`last_checked`）：同步到 2026-11-04；`published.md` 的 DEV 段新增本行，覆盖数 55 -> 56/127，作者 API 计数 65 -> 66。
- `check_media_ledger.py`：无 `ERROR`，DEV 显示 `56/127 source files mapped, 71 not published; 65 confirmed entries`。
- 队列：新增 `### 2026-11-04 排期` 段并勾掉该 DEV 任务，剩余队列仍为知乎 24 条（全部平台级 `阻塞`）。

## 下一步

1. 登录 DEV 后修两张图表：在编辑器里替换为 PNG（或先本地栅格化再上传），然后整页滚动复查。
2. 可见 Medium 会话恢复后，修 `59aff1cc69a5` 与 `5001bb9afab6`/`d8c070b5bd42` 的标题与扁平表格。
3. 下一条可执行队列任务仍取决于知乎登录；恢复后从 `57-aggregate` 之类的首条知乎任务继续。