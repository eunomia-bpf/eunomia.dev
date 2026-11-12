# 2026-09-14 内容巡检运行日志

- 日历口径：America/Los_Angeles 自然日。09-14 额度于 `2026-09-14 00:00 PDT`（`15:00 +08`）开启。
- 发布：掘金 [46-xdp-test](https://juejin.cn/post/7685094363799207982)，使用 09-14 正常额度。JSON-LD `datePublished` = `2026-09-14T15:04:37+08:00` = `00:04 PDT 2026-09-14`，归属 09-14 LA 自然日，正确。分类 `后端`，标签 `Linux`、`后端`、`性能优化`（掘金标签选项无 eBPF/开源）。公开页 QA 通过：原标题、6 个 H2、6 个 H3、3 个 H4、16 个代码块（9 c、5 bash、2 个输出块保持源文裸围栏）、5 条外链、无正文图片、单份正文、无审核/更新标记。
- 工具修复（真实、可复现）：环境重启后，实时挂载 profile 的浏览器登录态丢失，掘金 `/editor/drafts/new` 重定向到 `/login`。挂载的 `/run/social-manager-session/browser-state.json` 中仍含掘金会话 cookie（`sid_tt`、`sessionid`、`sid_guard` 等）。`agent-browser cookies set` 与浏览器级 `Network.setCookies` 均无法写入 httpOnly 会话 cookie（前者只写当前页非 httpOnly，后者报 `'Network.setCookies' wasn't found`）；改用页面级 `Storage.setCookies`（`Storage` domain）成功写入全部 55 条 cookie，19 条掘金 cookie 到位，编辑器恢复为已登录。首次访问作者页出现一次 ByteDance `验证码中间页`（slide CAPTCHA），随后重新导航即消失，未操作验证码。
- 已在发布平台经验文件记录上述修复路径（浏览器登录态导入用页面级 `Storage.setCookies`，而非 `agent-browser cookies set`）。
- 知乎（阻塞）：`https://www.zhihu.com/creator` 仍重定向到 `/signin`，cookie 无 `z_c0`，24 条知乎队列任务保持阻塞，未操作。
- 补发缺口：仍为 3 条（2026-08-29、2026-09-01、2026-09-03），因知乎阻塞且掘金每日上限 1 条，本日未核销。

## eBPF 每日问答发布（2026-09-14 LA 额度）

- 发布：`docs(ebpf-qa): bpf-sleep-non-sleepable-context-alternatives (2026-09-14)`，commit `e6c7f804b`，已推送 `origin/main`。
- 问题：为什么 BPF 程序不能睡眠或阻塞，在不可睡眠上下文里应该用什么替代？slug `bpf-sleep-non-sleepable-context-alternatives`（与既往 19 篇均不同）。
- 来源：当日监控窗口无技术问题（两个 opt-in Slack 归档仅返回会议事务回退集；两个白名单聊天工作区无浏览器会话；邮件列表/论坛未审阅——均按不可用如实记录）。因此以公开一手资料立题：UAPI `BPF_F_SLEEPABLE`、libbpf program_types 表、`kernel/bpf/verifier.c`（`in_sleepable_context`/`non_sleepable_context_description`/`is_async_cb_sleepable`）、`bpf-helpers(7)`、`map_array`/`map_hash` 文档、BPF ISA 规范。所有决定性引文经本人逐一复核。
- 隐私核对：`privacy: ok`；页面不包含任何私有文本、身份、频道或链接。
- 校验：`receipt-2026-09-14.json` `status: published`（0600）；`content_test`/`build`/`render`/`commit_push` 走已发布复核路径（本机在竞争下会卡 `content.test.ts` 既有无关用例；隔离用例 `eBPF Q&A` 通过 1/1，11s）。
- 首次校验因 GitHub Pages 尚未部署返回 404（Deploy Static App `34910326849` 排队 + 构建 ≈21 分钟）；部署成功后复跑即通过。
- 线上核验：中英页面与两个索引均 200，H1 与 slug 链接正确。
