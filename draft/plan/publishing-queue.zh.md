# 内容发布滚动队列

> 状态：内部执行 checklist，不发布。更新时间：2026-07-23。
> 顺序：从上到下推进，不把候选内容预先绑定到具体日期。每天最多向一个公开平台发布一篇内容；完成一项后，下一项最早在下一个自然日进入发布窗口。
> 授权：只有明确标记为 `可发布` 的任务才授权定时巡检完成真实发布。`待确认`、`待复核`、`候补` 和 `阻塞` 都不能被自动发布。

## 当前暂停与阻塞

- 全局暂停：2026-07-22 至 2026-07-23 不发布、同步或分享新内容。
- 待审核：tutorial 54 的掘金更新已提交审核；公开页通过审核后，只需核对正文版本并更新 ledger，不再创建新帖。
- 暂停：AgentPProf 暂不发布；Hacker News 暂不发；Lobsters 没有账号。
- 阻塞：小红书尚未确认账号 URL、登录状态和图片卡片工作流。

## 下一批

- [ ] `待确认` LinkedIn：发布 AgentNebula 项目帖和 10 秒演示素材。草稿：`draft/media/2026-07-22/agentsight-agent-nebula/linkedin.md`。
- [ ] `待确认` X：发布 AgentNebula 简短项目帖和 10 秒演示素材。草稿：`draft/media/2026-07-22/agentsight-agent-nebula/x.md`。
- [ ] `待复核` Medium：同步 `docs/tutorials/54-exec-image-inspector/README.md`，保留源标题和正文；先完成本地上传 artifact，再检查图片、代码和链接。
- [ ] `待复核` DEV：同步 `docs/tutorials/54-exec-image-inspector/README.md`，保留源标题和正文；在网页编辑器预览后再决定是否进入 `可发布`。
- [ ] `待复核` 知乎：同步 `docs/blog/posts/schedcp-agentic-os.zh.md`，保留源标题和正文。
- [ ] `待复核` 掘金：同步 `docs/blog/posts/schedcp-agentic-os.zh.md`，保留源标题和正文。

## 后续积压

- [ ] `候补` 知乎：`docs/tutorials/50-tcx/README.zh.md`。
- [ ] `候补` 掘金：`docs/tutorials/50-tcx/README.zh.md`。
- [ ] `候补` 掘金：`docs/tutorials/20-tc/README.zh.md`。
- [ ] `候补` 掘金：`docs/tutorials/21-xdp/README.zh.md`。
- [ ] `候补` 掘金：`docs/tutorials/22-android/README.zh.md`。
- [ ] `候补` Medium：`docs/tutorials/53-egress-pacer/README.md`。
- [ ] `候补` DEV：`docs/tutorials/53-egress-pacer/README.md`。
- [ ] `候补` 知乎：`docs/tutorials/49-hid/README.zh.md`。
- [ ] `候补` 掘金：`docs/tutorials/49-hid/README.zh.md`。
- [ ] `候补` 掘金：`docs/blog/posts/cpu-noise-gpu-inference.zh.md`。
- [ ] `候补` 掘金：`docs/blog/posts/runtime-security-for-opaque-ai-agents.zh.md`。
- [ ] `候补` 掘金：`docs/blog/posts/agent-check-restore-safety.zh.md`。
- [ ] `候补` Reddit：只有在 `r/eBPF` 当前规则和讨论仍匹配时，提交 ActPlane 实证研究的原题和文章 URL；不为凑频率发帖。

## 周期任务

- [ ] Weekly Analysis：距离上一篇公开 analysis 至少 7 天后才进入候选；执行时再选择与既有文章实质不同的 thesis。单篇至少实质参考 20 篇论文、20 个行业或开源项目，以及 10 份最近 7 天内的新闻或动态材料；达不到门槛就跳过。
- [ ] 已发内容检查：只在出现审核状态变化、渲染异常、有效评论或明确后续问题时执行，不占用发布名额，也不预先排空日期。

## 完成方式

- 每次只取最靠上的、条件已满足且明确标记为 `可发布` 的任务。
- 发布后把任务改为 `[x]`，写入公开 URL、审核状态和 ledger 结果。
- 平台文案和上传 artifact 放在 `draft/media/YYYY-MM-DD/<source-slug>/`；只有真实异常或待跟进事项才写同日 `run-log.md`。
- 已完成历史以 platform ledger、公开 URL 和 dated run log 为准，不在本文件复制日历流水账。
