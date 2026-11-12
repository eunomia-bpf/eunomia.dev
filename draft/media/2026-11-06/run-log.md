# 2026-11-06 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：LA 2026-11-06（系统日期 2026-11-06 +08，UTC 11-06 10:40 时 LA 为 11-06 02:40 PST）。本次巡检完成**一条 DEV 发布**并记录两条新的 Hacker News 外部提及；知乎、Medium 编辑器仍被登录/验证墙挡住，队列里 24 条知乎任务保持平台级 `阻塞`，补发缺口 0。

## 发布事实

- `docs/blog/posts/schedcp-agentic-os.md`（2026-07-10 英文长文，此前 Medium/DEV 均无副本）按"后续新增英文长文继续交叉排入"补入队列，用 LA 11-06 正常额度发布 DEV。
- 发布路径：准备稿 `draft/media/2026-11-06/schedcp-agentic-os/devto.md`（相对图片改写为 `https://eunomia.dev/_content-assets/docs/blog/posts/imgs/schedcp-design.png`，其余逐字保留）→ `.github/publisher/posts_queue.txt` → `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。
- workflow run 42204873168 于 `2026-11-06T10:44:00Z` 公开：<https://dev.to/yunwei37/can-an-ai-agent-tune-the-linux-scheduler-inside-schedcp-and-sched-agent-4cj0>，article id 4987756，tag `ebpf, linux, ai, performance`，reading_time 8，0 反应 0 评论；workflow 自己的 `Published post and updated queue [skip ci]` 提交把队列清空。
- 正文对账：公开 API 的 `body_markdown` 与准备稿去掉 frontmatter 与首个 H1 后**逐字相同**（脚本比较为 `True`）。结构对账（源文 vs 公开页）：1 h1、6 个正文 H2（源文 6 个 `##`）、0 H3、0 代码块、0 表格、1,801 词、References 尾段完整、6 个外链均为绝对地址且可访问。
- 唯一图片（SchedCP 控制平面 PNG）经 DEV 代理正常渲染，`naturalWidth 788 × naturalHeight 600`。

## 平台状态

- DEV：公开 API 与可见页都可读；作者列表 API 在发布后数分钟仍返回缓存的 66 篇（不含 4987756），单篇端点正确，逐篇确认以单篇端点为准。
- 知乎：可见 Chrome 打开 `https://www.zhihu.com/creator` 落到 `https://www.zhihu.com/account/unhuman?type=S6E3V1&need_login=true`（"系统监测到您的网络环境存在异常…开始验证"），无 `z_c0`；比以往的 `/signin` 重定向多了一层人机验证，恢复条件仍是用户在可见 profile 完成登录+验证。
- Medium：`medium.com/me/stories/public` 返回 Cloudflare "Just a moment..."；公开文章页 `d8c070b5bd42` 可读（可见标题存在，测量表只剩扁平文本），`5001bb9afab6` 本次未读到，缺标题与 Mermaid 源码两项缺陷仍未修。
- Hacker News：3 日内窗口新增两条 eunomia.dev 链接——故事 50385391（AgentSight 项目页，rsyring，2 分 0 评论）与评论 50383008（maven29 在 Show HN "Stroq" 线程里推荐 eBPF 后端并链到 AgentSight）；qwen3（7 分 3 评论）与 bpf-gpu-server（1 分 0 评论）无变化。均非项目账号提交，也无人提问，本次不做回复。

## 遗留缺陷

1. 本轮 DEV 副本仍 canonical 自指（`publish.py` 的 `publish-multi` 请求不带 `canonical_url`），是连续第三次同因复发；`media-publisher` 的 `publish-multi` 只接受 `title`/`content`/`tags`/`is_draft`/`platforms` 五个字段（查了它的公开 payload 与表单），因此不能靠 payload 传 `canonical_url`；修法要么扩展该端点，要么直连 DEV `/api/articles` 并带 `canonical_url`（源文对应 `https://eunomia.dev/blog/2026/07/10/schedcp-agentic-linux-scheduler/`）。
2. 外部 SVG 图表在 DEV 依旧为空白（gthulhu 两张），本轮 PNG 正常，说明"PNG 化"是可用绕行方案，无需编辑器登录。
3. Medium `5001bb9afab6` 无可见标题 + Mermaid 源码裸露；编辑器与公开页都受 Cloudflare 影响，待可见会话稳定后修。

## 下一步

1. 让 canonical 生效：扩展 `media-publisher` 的 `publish-multi` 接受 `canonical_url`，或在 `publish.py` 里改直连 DEV `/api/articles`；先在下一篇文章上验证。
2. SchedCP 的 Medium 副本仍缺，可在 Medium 会话恢复后按 API 或可见编辑器补齐，并同时保留可见 H1。
3. 知乎登录恢复后从 `53-egress-pacer` 起步继续 24 条任务。