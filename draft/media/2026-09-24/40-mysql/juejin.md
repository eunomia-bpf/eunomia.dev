# 掘金发布稿：40-mysql 教程

- 状态：已发布并确认公开（2026-09-24 LA 自然日，作为核销补发缺口的当日第二条掘金任务；当日正常额度已由 41-xdp-tcpdump 使用）。正式地址 https://juejin.cn/post/7689030007914545198（`/post/` 200、`/spost/` 404，无审核等待），公开页 QA 通过，`platforms/juejin.json` 记为 `confirmed`
- 正文：`juejin-body.md`（源文 `docs/tutorials/40-mysql/README.zh.md` 移除源 H1，其余正文逐字保留；3237 字符、6071 字节、5 个 H2、4 个 H3、0 个 H4、3 个代码块 [1 bt + 1 bash + 1 console]、0 张图片、0 个表格、3 条唯一外链、无相对链接）
- 标题：使用 eBPF 跟踪 MySQL 查询（源 H1 逐字）
- 原文：https://eunomia.dev/zh/tutorials/40-mysql/
- 代码：https://github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/40-mysql
- 分类：后端
- 标签：`Linux`、`后端`、`性能优化`
- 图片：无正文图片
- 外链（3 条，保持源文）：bpftime、bpf-developer-tutorial、eunomia.dev/tutorials/
- 已核查：`platforms/juejin.json` 无 40-mysql 条目，源文未被掘金发布过，可首发布
- 提交注意：分类 chip 需真实 CDP `page.mouse` 指针序列；标签用原生 setter + 冒泡 `input` 事件后点文档级可见 `.byte-select-option`；「确定并发布」需真实指针序列，勿用合成 `click()`
- 已预置草稿：编辑器中草稿 id `7688990935071096882`（标题、3237 字符正文、分类 `后端`、标签 `Linux`/`后端`/`性能优化` 全部已服务端持久化，重载后回读一致）。到点只需打开该草稿并提交，无需重建内容。
- 可见性判定教训：`.byte-select-option` 即使在未展开的 `.byte-select-dropdown__wrap` 内也存在（宽高为 0），用 `offsetParent !== null` 过滤会误判为空；应用 `getBoundingClientRect().width > 0` 判定真正可见的选项，且必须清空标签输入框的残留值再写入，否则选项不出现。
- 提交结果判定：以 `/post/<id>` 的 HTTP 状态为准，而非创作者中心列表。返回 200 且无 `审核中`/`文章有更新` 标记即记 `confirmed`；仅 `/spost/<id>` 渲染且 `/post/<id>` 404 时记 `review_pending`（2026-09-22 43-kfuncs、2026-09-24 41-xdp-tcpdump 两种情形均出现过，后者同会话内即转公开）
- 补发核销：本次按补发缺口额度发布，核销最早缺口 2026-08-29，缺口由 4 条减为 3 条（2026-09-01、2026-09-03、2026-09-16），见 `draft/plan/publishing-queue.zh.md` 第 15 行与 `draft/media/2026-09-24/run-log.md`。
