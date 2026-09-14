# 2026-09-14 内容巡检运行日志

- 日历口径：America/Los_Angeles 自然日。09-14 额度于 `2026-09-14 00:00 PDT`（`15:00 +08`）开启。
- 发布：掘金 [46-xdp-test](https://juejin.cn/post/7685094363799207982)，使用 09-14 正常额度。JSON-LD `datePublished` = `2026-09-14T15:04:37+08:00` = `00:04 PDT 2026-09-14`，归属 09-14 LA 自然日，正确。分类 `后端`，标签 `Linux`、`后端`、`性能优化`（掘金标签选项无 eBPF/开源）。公开页 QA 通过：原标题、6 个 H2、6 个 H3、3 个 H4、16 个代码块（9 c、5 bash、2 个输出块保持源文裸围栏）、5 条外链、无正文图片、单份正文、无审核/更新标记。
- 工具修复（真实、可复现）：环境重启后，实时挂载 profile 的浏览器登录态丢失，掘金 `/editor/drafts/new` 重定向到 `/login`。挂载的 `/run/social-manager-session/browser-state.json` 中仍含掘金会话 cookie（`sid_tt`、`sessionid`、`sid_guard` 等）。`agent-browser cookies set` 与浏览器级 `Network.setCookies` 均无法写入 httpOnly 会话 cookie（前者只写当前页非 httpOnly，后者报 `'Network.setCookies' wasn't found`）；改用页面级 `Storage.setCookies`（`Storage` domain）成功写入全部 55 条 cookie，19 条掘金 cookie 到位，编辑器恢复为已登录。首次访问作者页出现一次 ByteDance `验证码中间页`（slide CAPTCHA），随后重新导航即消失，未操作验证码。
- 已在发布平台经验文件记录上述修复路径（浏览器登录态导入用页面级 `Storage.setCookies`，而非 `agent-browser cookies set`）。
- 知乎（阻塞）：`https://www.zhihu.com/creator` 仍重定向到 `/signin`，cookie 无 `z_c0`，24 条知乎队列任务保持阻塞，未操作。
- 补发缺口：仍为 3 条（2026-08-29、2026-09-01、2026-09-03），因知乎阻塞且掘金每日上限 1 条，本日未核销。
