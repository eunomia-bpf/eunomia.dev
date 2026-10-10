# 掘金发布稿：17-biopattern 教程

- 状态：已发布（LA 2026-10-09 正常额度；post `7694534084639195174`，2026-10-10 07:17 +08 = 16:17 PDT 提交，审核间隔 07:52 +08 放行；ledger `confirmed`）
- 正文：`juejin-body.md`（源文 `docs/tutorials/17-biopattern/README.zh.md` 移除源 H1，其余正文逐字保留；10501 字符、15383 字节、4 个 H2、1 个 H3、0 个 H4、11 个代码块 [8 c：biopattern.bpf.c 完整内核态程序 ×1、全局变量 ×1、BPF map 定义 ×1、追踪点函数 ×1、两种追踪点结构定义 ×1、has_block_rq_completion 动态检测 ×1、用户态主循环 ×1、print_map 函数 ×1；2 bash：cd/make 编译 ×1、sudo ./biopattern 运行命令 ×1；1 console：sudo ./biopattern 1 10 输出 ×1]、0 张图片、0 个表格；5 条唯一外链目标 [eunomia.dev/tutorials/11-bootstrap、github eunomia-bpf/bpf-developer-tutorial（×2 实例，含 src/17-biopattern 子目录）、eunomia.dev/zh/tutorials、github iovisor/bcc biopattern.c]，0 条相对链接）
- 标题：eBPF 入门实践教程十七：编写 eBPF 程序统计随机/顺序磁盘 I/O（源 H1 逐字，含系列编号"十七"）
- 原文：https://eunomia.dev/zh/tutorials/17-biopattern/
- 代码仓库：https://github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/17-biopattern
- 分类：后端
- 标签：`Linux`、`后端`、`性能优化`（沿用 21-xdp / 22-android / 24-hide / 26-sudo / 20-tc / 19-lsm-connect 教程标签组；以实际落定 chip 为准）
- 代码块说明：源文代码围栏语言为 `c`（×8，内核态程序与用户态代码片段）、`bash`（×2，编译与运行命令）与 `console`（×1，运行输出）；掘金编辑器按语言类渲染，属平台能力差异，不改源文标签

## 发布结果（LA 2026-10-09）

- 提交：2026-10-10 07:17 +08（= 2026-10-09 16:17 PDT，LA 10-09 正常额度）于 `/editor/drafts/new` 新会话提交（草稿箱 5 条，未被本次消费），标题经原生 `HTMLInputElement` setter 写入并回读（37 字符逐字），正文 base64 分块注入 CodeMirror 写后回读 10501 字符与本地一致；分类 `后端`（`.category-list .item` 真实 CDP 指针点击 → `.active`）；标签经原生 setter 写入 + 可见 `.byte-select-option` 真实 CDP 点击逐条落定，最终 DOM 顺序 `性能优化`、`Linux`、`后端`；单次弹窗「确定并发布」（真实指针事件序列）返回「发布成功」并落到 `https://juejin.cn/published`。提交前创作者中心 全部 78 / 已发布 78 / 审核中 0 / 未通过 0；提交后 全部 79 / 已发布 78 / 审核中 1 / 未通过 0。
- 审核间隔：`/spost/7694534084639195174` 暂存，创作者中心 审核中 (1)；期间 SPA 壳对 `/post/` 一律回 200，判定以登录态创作者中心为准（与 21-xdp / 19-lsm-connect 审核期先例一致，区别于 26-sudo / 20-tc 直接公开）。托管轮询（约 87 秒一拍，驱动可见浏览器创作者中心）07:26 +08 第 1 拍 审核中 (1)，07:52 +08 第 19 拍翻为 全部 79 / 已发布 79 / 审核中 0 / 未通过 0，审核间隔约 35 分钟，规范 `/post/` URL 自放行起上线。
- 正式地址：<https://juejin.cn/post/7694534084639195174>（暂存 <https://juejin.cn/spost/7694534084639195174>，放行后由正式地址接管）
- 公开页 QA（登录浏览器，07:52 +08 放行后）：精确标题「eBPF 入门实践教程十七：编写 eBPF 程序统计随机/顺序磁盘 I/O」（页面级 H1 仅 1 次，正文区域无重复 H1）；单份正文（10501 字符）；4 H2 / 1 H3 / 0 H4；11 个代码块 [8 c、2 bash、1 console]；0 张图片（正文区域无 img）；0 个表格；5 条唯一外链目标全部改写为 `link.juejin.cn?target=`；无 `审核中` / `文章有更新` / `已被删除` 标记；评论 0（`暂无评论` 空态）。
- 早期计数器：07:52 +08 放行检查点 0 展现 / 0 阅读 / 0 点赞 / 0 评论 / 0 收藏。
- ledger：`platforms/juejin.json` 置顶 `juejin-7694534084639195174`（带 `staged_url`），`published[]` 66 → 67，`last_checked` → 2026-10-09；`sources.json` `last_checked` → 2026-10-09；验证器 `check_media_ledger.py` 通过（Juejin 66/108 映射、42 未发布、67 confirmed）。
