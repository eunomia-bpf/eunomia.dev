# 2026-07-22 内容巡检

- 模式：`eunomia` 定时巡检；2026-07-22 至 2026-07-23 全局发布暂停生效，未创建或发布新帖。
- 子 skill：`eunomia-social-radar`。
- Tutorial 54：知乎公开页正常，3 个赞同、0 评论；掘金公开页正常，16 次阅读、0 评论，懒加载图片显示正常。
- Tutorial 54 修复：知乎原文已就地更新并通过公开页格式检查；掘金已提交同步更新，待审核页的标题、章节、代码块、表格、图片和 system-wide 实现正常，公开页在平台审核通过前仍显示旧版。
- BPFix：LinkedIn 主帖 3,138 次展示、60 个回应、5 条评论、4 次转发，没有新的未答技术问题；skill 动态 569 次展示、10 个回应、0 评论。Emergent Mind 收录正常；Hacker News 讨论仍为 flagged，仍有两条质疑且无新增回复。
- BPFix 日期修复：页脚的首次发布日期现在优先使用文章 front matter；生成的中英文页分别显示 \`2026年7月25日\` 和 \`Jul 25, 2026\`。TypeScript 和定向 ESLint 通过；Windows 构建已编译并生成全部静态页，但最后并行导出阶段遇到本机原生进程异常，等待 CI 复核。
- 问题：<https://github.com/eunomia-bpf/eunomia.dev/issues/129> 中知乎已修复、掘金待审核；<https://github.com/eunomia-bpf/eunomia.dev/issues/130> 已本地修复，待部署确认。
- 阻塞：Semantic Scholar 页面未渲染可读内容，无法核验引用数据；此前记录的 BPFix Hacker News 第三方提交缺少 permalink，当前搜索未能确认。
- 下一步：2026-07-23 继续暂停发布，只检查已发布内容的明确问题；保留 AgentNebula 的 LinkedIn/X 未发布预览。
