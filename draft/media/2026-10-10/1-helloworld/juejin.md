# 掘金发布稿：1-helloworld 教程

- 状态：已发布 / `confirmed`（LA 2026-10-10 正常额度；post `7694934986782064649`，2026-10-11 07:21 +08 = 16:21 PDT 提交，`/spost/` 暂存，创作者中心 10:14 +08 轮询翻为 已发布 (80) / 审核中 (0)，审核间隔约 173 分钟；公开页 QA 通过）
- 正文：`juejin-body.md`（源文 `docs/tutorials/1-helloworld/README.zh.md` 移除源 H1，其余正文逐字保留；7633 字符、13855 字节、7 个 H2、1 个 H3、0 个 H4、10 个代码块 [1 c：minimal.bpf.c 完整 eBPF 程序 ×1；2 shell：sudo apt install clang llvm ×1、docker run 编译 ×1；7 console：ecli -h 帮助 ×1、ecc -h 帮助 ×1、./ecc minimal.bpf.c 编译输出 ×1、docker 编译命令 ×1、ecli minimal 运行输出 ×1、trace_pipe cat 输出 ×1、tracing_on 说明 ×1]、0 张图片、0 个表格；10 条唯一外链目标 [eunomia.dev/zh/tutorials/1-helloworld/、github eunomia-bpf/bpf-developer-tutorial、github eunomia-bpf/eunomia-bpf、github eunomia-bpf/eunomia-template、github eunomia-bpf/cilium-ebpf-starter-template、github eunomia-bpf/libbpf-rs-starter-template、github eunomia-bpf/libbpf-starter-template、docs.github.com creating-a-template-repository、github eunomia-bpf/eunomia-bpf releases ecc-aarch64、ecli-aarch64]，0 条相对链接）
- 标题：eBPF 入门开发实践教程一：Hello World，基本框架和开发流程（源 H1 逐字，含系列编号"一"）
- 原文：https://eunomia.dev/zh/tutorials/1-helloworld/
- 代码仓库：https://github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/1-helloworld
- 分类：后端
- 标签：`Linux`、`后端`、`性能优化`（沿用 17-biopattern / 19-lsm-connect 教程标签组；以实际落定 chip DOM 顺序为准）
- 代码块说明：源文代码围栏语言为 `c`（×1，完整 eBPF 程序）、`shell`（×2，apt 安装与 docker 编译）与 `console`（×7，帮助、编译输出、运行输出、trace_pipe 输出）；掘金编辑器按语言类渲染，属平台能力差异，不改源文标签

## 发布结果（LA 2026-10-10）

- 提交：2026-10-11 07:21 +08（= 2026-10-10 16:21 PDT，LA 10-10 正常额度）于 `/editor/drafts/new` 新会话提交（编辑器草稿箱读为 5 条，未被本次消费），标题经原生 `HTMLInputElement` setter 写入并回读（36 字符逐字），正文 base64 分 4 块注入 CodeMirror 写后回读 7633 字符与本地一致；发布面板需先点右上「发布」展开（收起态所有 widget 宽 0），分类 `后端`（`.category-list .item` 真实 CDP 指针点击 → 回读 `.active` 为 `后端`）；标签经 index-0 原生 setter 写入 + 可见 `.byte-select-option` 真实 CDP 点击逐条落定（需先 `focus` 触发搜索），最终 DOM 顺序 `Linux`、`后端`、`性能优化`；提交前回读标题确认无污染（36 字符，无需重置）；单次弹窗「确定并发布」（真实指针事件序列）返回「发布成功」并落到 `https://juejin.cn/published`。提交前创作者中心 全部 79 / 已发布 79 / 审核中 0 / 未通过 0；提交后 全部 80 / 已发布 79 / 审核中 1 / 未通过 0。
- 审核间隔：`/spost/7694934986782064649` 暂存，创作者中心 审核中 (1)；期间 SPA 壳对 `/post/` 一律回 200，判定以登录态创作者中心为准（与 17-biopattern / 21-xdp / 19-lsm-connect 审核期先例一致，区别于 26-sudo / 20-tc 直接公开）。托管轮询（约 87 秒一拍，驱动可见浏览器创作者中心）自 07:22 +08 第 1 拍起。
- 放行：托管轮询（约 87 秒一拍，驱动可见浏览器创作者中心）07:22 +08 第 1 拍 审核中 (1)，于 10:14 +08 第 13 拍翻为 全部 80 / 已发布 80 / 审核中 0 / 未通过 0，审核间隔约 173 分钟（观测以来最长），规范 `/post/` URL 自放行起上线。
- 正式地址：<https://juejin.cn/post/7694934986782064649>（暂存 <https://juejin.cn/spost/7694934986782064649>，放行后由正式地址接管）
- 公开页 QA（登录浏览器，10:14 +08 放行后）：精确标题「eBPF 入门开发实践教程一：Hello World，基本框架和开发流程」（页面级 H1 仅 1 次，正文区域无重复 H1）；单份正文（7633 字符 artifact；渲染 innerText 7087 字符，代码块空白归一化）；7 H2 / 1 H3 / 0 H4；10 个代码块 [1 c、2 shell、7 console]；0 张图片（正文区域无 img）；0 个表格；10 条唯一外链目标全部改写为 `link.juejin.cn?target=`；无 `审核中` / `文章有更新` / `已被删除` 标记；评论 0（`暂无评论数据` 空态）。
- 早期计数器：10:14 +08 放行检查点 0 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏。
- ledger：`platforms/juejin.json` 置顶 `juejin-7694934986782064649`（带 `staged_url`），`published[]` 67 → 68，`last_checked` → 2026-10-10；`sources.json` `last_checked` → 2026-10-10；验证器 `check_media_ledger.py` 通过（Juejin 67/108 映射、41 未发布、68 confirmed）。
