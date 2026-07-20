# Xiaohongshu Platform Preferences

Use this reference when adapting eunomia.dev artifacts into Xiaohongshu notes,
choosing title and cover direction, planning carousel cards, or learning from
visible platform examples before drafting.

## Source Notes

- Xiaohongshu is visual-first and mobile-first. Treat the cover, title, and
  first screen as the real hook.
- Platform taste changes quickly. When native fit matters, inspect recent
  visible notes in the browser before drafting and record only structural
  observations.
- For systems, eBPF, AI-agent safety, observability, and runtime topics, the
  strongest Xiaohongshu version usually translates a hard technical idea into a
  concrete scenario, diagram, screenshot, checklist, or "踩坑/避坑" note.

## What Xiaohongshu Rewards

- A concrete title that says what problem, result, contrast, or mistake the
  reader will understand.
- A cover that can be read at phone size: few words, one visual focus, no dense
  architecture wall.
- Carousel structure with one idea per card: problem -> failed assumption ->
  mechanism -> result -> how to try or read more.
- Personal engineering judgment: what was surprising, what changed after
  tracing/measuring/building, and what boundary the reader should remember.
- Save-worthy artifacts: mini checklist, glossary with examples, command/result
  pair, diagram, comparison table, or debugging path.

## Audience

- Chinese technical readers who may not already know eBPF, but care about AI
  agents, security, observability, performance, or developer tooling.
- Engineers and technical founders scanning for practical lessons before
  opening GitHub or a long article.
- Students and researchers who need a clear conceptual entry before the full
  paper, tutorial, or repository.

## Platform-Observation Scan

Use this scan when the user says to "多刷刷", improve platform fit, or learn
from other posts:

- Search or browse a few visible notes around adjacent terms such as AI Agent,
  eBPF, Linux, observability, security, open source, debugging, and developer
  tooling. Stop as soon as the scan yields 2-3 draft-changing patterns.
- Record only public, visible, non-personal structural observations: title
  pattern, cover style, first-card shape, body length, tag pattern, comment
  prompt, and media format.
- Do not copy distinctive wording, creator identity, private comments, or
  platform-hidden data.
- Convert observations into a short improvement checklist before drafting:
  stronger title, clearer cover, visual proof, simpler first paragraph, better
  tags, or different carousel order.
- If the scan surfaces repeated audience confusion, turn it into future
  eunomia.dev blog/tutorial ideas instead of only optimizing the post.

## Recent Browser Observations

2026-07-20 visible Xiaohongshu search results for "AI Agent 可观测性" showed
titles clustered around questions,体系拆解,面试/skill 命中,自修复,评测调优,
and "跑飞了怎么办" style failure framing.

- Stronger native angles make one problem visible immediately: can we trace
  what an agent did, prove a skill was selected correctly, or debug a failure?
- The platform rewards saveable shapes:体系拆解, checklist, "出了问题怎么查",
  trend/day-list posts, and a single concrete lesson per note.
- For our technical posts, avoid pure title bait such as "一文搞定" unless the
  note contains a real diagram, screenshot, checklist, command/result pair, or
  repo path that earns the promise.

## Rewrite Rules

- Do not translate LinkedIn copy. Rewrite from the Chinese reader's first
  question: "这和我有什么关系？我能看懂什么？我能试什么？"
- Lead with the situation, contrast, or result before naming the project.
- Replace abstract product language with concrete technical objects: kernel
  hook, eBPF trace, agent tool call, policy boundary, benchmark, command,
  screenshot, architecture layer, or GitHub example path.
- Use one main topic per note. Split a large project launch into multiple notes:
  concept entry, debugging story, diagram walkthrough, practical checklist, and
  repo/tutorial follow-up.
- Keep the 80% contribution / 20% promotion posture. The note should be useful
  even if the reader does not click the link.

## Title And Cover Patterns

- Problem: "AI Agent 出错后，怎么知道它到底调用了什么？"
- Contrast: "只看日志不够，Agent runtime 还缺这一层可观测性"
- Result: "用 eBPF 把一次 tool call 的系统行为画出来"
- Checklist: "做 Agent 安全策略前，我会先检查这 5 个边界"
- Mistake: "我们一开始把限制放在 runtime，后来发现边界还要更低"

Use these as pattern examples only. Before publishing, rewrite titles to match
the actual source evidence.

## Note Shapes

- Mini walkthrough: 5-7 carousel cards showing problem, mechanism, trace/result,
  and next step.
- Single insight: one screenshot or diagram plus 120-250 Chinese characters
  explaining the lesson.
- Saveable checklist: 5-8 checks for debugging, observability, policy, or
  reproduction.
- Paper/project explainer: one concrete tension, one mechanism diagram, one
  limitation, one source link.

## Link And Tag Treatment

- External links are secondary. Put GitHub, eunomia.dev, or paper links as
  source/extended reading after the note has delivered value.
- If links are hard to click or render poorly, prepare a concise link note such
  as "项目名/eunomia.dev 标题可搜索" and keep the exact URL in the local draft
  record for reuse elsewhere.
- Use tags sparingly and concretely: AI Agent, eBPF, Linux, 开源, 可观测性,
  安全, 性能优化, 开发者工具, depending on the actual note.

## Anti-AI Pass

Remove or rewrite:

- "重磅发布", "全网首发", "干货满满", "一文搞懂" unless the note truly earns it
- vague stacks such as "安全、可靠、高效" without saying what is constrained,
  measured, or observed
- perfect three-part slogans that do not sound like an engineer's note
- mystery hooks that hide the technical object
- promotion-first openings like "欢迎关注我们的项目"

## Browser Checks

- Inspect recent visible examples before drafting when native fit is uncertain.
- Check title and cover at phone-like size.
- Check the first card or first paragraph before any external link.
- Check that code screenshots are readable and not dominated by tiny text.
- Check tags, account, visibility, and publish settings.
- Stop before final publish unless the user explicitly confirms.
