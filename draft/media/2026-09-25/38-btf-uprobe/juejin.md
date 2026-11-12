# 掘金发布稿：38-btf-uprobe 教程

- 状态：已发布（<https://juejin.cn/post/7689065487933833262>，暂存 <https://juejin.cn/spost/7689065487933833262>；LA 2026-09-25 正常额度，提交时刻 2026-09-26 07:27 +08 = 16:27 PDT，约 08:07 +08 审核通过后正式地址生效，公开页 QA 通过；本稿由 2026-09-26 巡检准备并在同一巡检内完成发布）
- 正文：`juejin-body.md`（源文 `docs/tutorials/38-btf-uprobe/README.zh.md` 移除源 H1，其余正文逐字保留；9781 字符、14055 字节、5 个 H2、2 个 H3、0 个 H4、18 个代码块 [5 c、4 sh、9 console]、0 张图片、0 个表格、4 条唯一外链 [6 处]、无相对链接）
- 标题：借助 eBPF 和 BTF，让用户态也能一次编译、到处运行（源 H1 逐字）
- 原文：https://eunomia.dev/zh/tutorials/38-btf-uprobe/
- 代码：https://github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/38-btf-uprobe
- 分类：后端
- 标签：`Linux`、`后端`、`性能优化`
- 图片：无正文图片
- 外链（4 条唯一、6 处，保持源文）：bpf-developer-tutorial 仓库源码目录 ×2、eunomia.dev/tutorials/ ×2、bpf-developer-tutorial 仓库 ×1、bpftime 仓库 ×1
- 已核查：`platforms/juejin.json` 无 38-btf-uprobe 条目，源文未被掘金发布过，可首发布
- 提交注意：分类 chip 需真实 CDP `page.mouse` 指针序列；标签用原生 setter（先清空残留）加冒泡 `input` 事件，再按 `getBoundingClientRect().width > 0` 选出可见 `.byte-select-option` 并真实点击；「确定并发布」需真实指针序列，勿用合成 `click()`；提交前回读标题输入框并重置为源 H1
- 提交结果判定：以登录浏览器正文渲染与创作者中心 `审核中` 计数为准。掘金 SPA 壳在审核期对任意 `/post/<id>`、`/spost/<id>` 路径返回 HTTP 200，curl 状态码不是发布信号（本次 07:29 +08 curl 200/200 时仍 审核中、`/post/` 浏览器内仍「找不到页面」）；`/post/<id>` 渲染出正文且无 `审核中`/`文章有更新`/`已被删除` 标记、`/spost/<id>` 跳转 `/post/` 时记 `confirmed`；审核清除时间不定（本次约 40 分钟），持续浏览器轮询至 审核中 归零再定论。本次实际记录：07:27 提交、08:07 审核通过、`/post/7689065487933833262` 渲染公开，记 `confirmed`。
- 恢复条件：未确认公开前不得重复提交同一源文
