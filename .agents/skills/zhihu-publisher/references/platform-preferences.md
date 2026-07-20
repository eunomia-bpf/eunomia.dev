# Zhihu Platform Preferences

Use this reference when choosing a Zhihu angle, title, opening, cover, project
link placement, or "想法" adaptation.

## Source Notes

- Zhihu positions itself around high-quality knowledge, experience, and opinion
  sharing. Favor material that gives readers a clearer mental model, not just a
  status update.
- Zhihu creator signals commonly emphasize creation activity, content quality,
  influence, follower interaction, community contribution, and vertical focus.
  Treat these as directionally useful but platform-changeable.
- Zhihu's own creator materials list answers, articles/columns, ideas, and
  videos. For technical topics, answers and articles should carry the full
  explanation; "想法" works better as a concise insight, artifact, or discussion
  prompt.

## What Zhihu Rewards

- Start from a concrete question or scenario: "为什么 AI Agent 需要内核态信息流控制？"
  is stronger than "我们发布了一个项目".
- Explain the mechanism before the project pitch. For eBPF/AI-agent topics,
  connect problem -> enforcement/observability mechanism -> practical scenarios
  -> limitations -> GitHub/source link.
- Make niche systems content legible with a short "适合谁读" paragraph: agent
  runtime builders, security engineers, platform/SRE teams, kernel/eBPF
  developers, or researchers.
- Use project claims as evidence. For example, "actplane 是一个内核态强制执行
  信息流控制的 AI Agent 策略引擎" needs a sentence explaining what information
  flow is constrained, where enforcement happens, and why that improves safety,
  compliance, or instruction-following reliability.
- Use one clean artifact link as extended reading. GitHub is appropriate when
  there is no hosted app or when the repo is the main application artifact.
- Preserve the maintainer's personal technical account voice: useful, precise,
  research-aware, and open to discussion.

## Style And Positioning

- Use a consulting/research/helping-solve-problems posture. Zhihu content should
  make the reader feel they learned a usable concept or judgment.
- For eunomia.dev long-form posts, default to Chinese canonical syndication.
  Preserve the source title exactly and keep the body substantively unchanged.
  Only fix rendering and set tags, column, cover, question, and other platform
  metadata. Rewrite only when the user explicitly requests it for that
  publication.
- Keep the 80% contribution / 20% promotion ratio. Project links should appear
  as source code, artifact, or extended reading after the explanation.
- Visible eunomia.dev original/canonical notes are optional. Do not add or edit
  a body link solely to satisfy a checklist; prefer GitHub, paper, or docs links
  when they are the more useful next step.
- Prefer "我们在做这个项目时发现的问题" and "这个设计能解决什么边界" over
  product-style launch language.

## Audience

- AI-agent/runtime builders who need safety, policy, and observability patterns.
- Security, compliance, SRE, and platform engineers evaluating production
  controls.
- Kernel/eBPF developers and researchers who want credible mechanism detail.
- Curious technical readers who need enough background before specialist terms.

## Syndication Rules

- Do not rewrite an already polished Chinese eunomia.dev article just to make it
  "Zhihu-native." Preserve the body and fix platform formatting.
- If a raw changelog, README abstract, English-only source, or weak opening is
  unsuitable for syndication, improve the source first or skip it. A
  platform-only rewrite requires explicit user instruction.
- For new Zhihu-native project posts, explain what the system observes,
  enforces, accelerates, or verifies. Avoid category-only descriptions.
- For Chinese copy, write naturally from the same facts instead of translating
  English sentence by sentence.
- For new project posts, include concrete use cases when they help. Do not add
  platform-only examples to a syndicated long-form article.

## Adaptation Workflow

- Extract the source facts before writing: core question, scenario, mechanism,
  evidence, artifact link, reader role, and what the reader can discuss or try.
- Pick the Zhihu surface first: answer, article, column post, "想法", or AI
  Works field. Do not compress an essay into a "想法" if the mechanism needs
  context.
- For project content, explain the boundary before the brand: what is observed,
  what is enforced, where enforcement happens, and what failure it prevents.
- Keep one clean source path per short item. GitHub is preferred when the repo
  is the artifact; eunomia.dev or papers are preferred for extended explanation.
- For long-form posts, finish a local Zhihu-specific artifact before opening
  the editor. Prefer Zhihu's visible document import path when it works; if a
  pasted/imported draft needs major structural repair, regenerate the local
  artifact and import again rather than editing fragile rich text in place.
- Run the anti-AI pass on new short or platform-native copy. Do not use it to
  rewrite syndicated long-form prose.

## What To Avoid

- Do not write a release note with a thin "we open-sourced X" opening.
- Do not overfit to slogans such as "AI Agent 策略引擎" without unpacking the
  security, reliability, and compliance consequences.
- Do not make the post a traffic funnel. Give the core explanation on Zhihu and
  use eunomia.dev/GitHub/paper links as sources or next steps.
- Do not include private strategy, customer claims, unreleased roadmap, or
  unverifiable performance numbers.

## Browser Checks

- Check the logged-in account and intended publishing surface: article, answer,
  idea, column, or AI Works form.
- Check title length, first-screen hook, cover image crop, image rendering, code
  blocks, tables, and link destinations.
- In the publish/settings step, choose a relevant question when Zhihu offers a
  question selector. This is part of final QA, not an optional afterthought.
- Check that GitHub/eunomia.dev/paper links are specific and placed as sources
  or next steps. A visible eunomia.dev canonical/source note is optional.
- Check whether column, tags, or project fields match the topic.
- Stop before final publish/submit unless the user explicitly confirms.

### Import And Image Failure Checks

- Use the visible import path `导入` -> `导入文档` ->
  `点击选择本地文档或拖动文件到窗口上传`. The first menu item and hidden file
  input may not emit a usable file chooser in browser automation.
- Normalize literal currency such as `$0.028` and `$11` to `0.028 美元` and
  `11 美元` in the local upload artifact. Zhihu document import may pair the
  dollar signs as inline math and silently delete everything between them.
- Treat imported images as failed until each figure has neither `上传失败` nor
  `加载失败` and its image has non-zero natural dimensions. Counting figure
  nodes is insufficient because failed placeholders remain in the editor.
- Remote Markdown images and DOCX-embedded images can fail partially and
  nondeterministically. After one clean re-import attempt, prefer a no-inline-
  image upload artifact plus a separately uploaded cover over repeated fragile
  rich-text repair. Keep the richer platform draft as the content record.
- Verify a cover in both places when possible: the editor should store a Zhihu
  CDN URL with non-zero dimensions, and the public page should not show a
  broken image or empty cover region.

## Zhihu "想法"

- Keep one sharp thesis, one concrete example, and one link.
- Use a visual when it clarifies the artifact: architecture diagram, terminal
  trace, benchmark chart, UI screenshot, or repo/project image.
- End with a discussion prompt when useful, such as "你会把 Agent 的安全边界放在
  runtime、sandbox，还是内核态 enforcement？"

## Short-Form Style

- Treat short Zhihu content as a compact "想法" or discussion seed, not a
  compressed press release.
- Shape: one problem sentence -> one mechanism or observation -> one concrete
  artifact/link -> one specific discussion prompt when useful.
- Keep the language precise and slightly explanatory. Avoid "重磅发布",
  "颠覆式", "赋能", "全面升级", "欢迎体验", and slogan-only openings.
- Add a specific technical object: kernel enforcement boundary, eBPF program,
  agent tool call, audit trace, repo example, paper claim, or benchmark.
- Avoid AI-tell structure: three vague benefits in a row, symmetrical slogans,
  generic "安全、合规、可靠" without saying what is enforced and where.
- If using first-person framing, make it a real engineering observation:
  "我们原来以为限制在 runtime 就够了，后来发现..." only when the source supports it.
- Short posts should still give enough context that readers understand why the
  GitHub link matters before opening it.
- Use one of these native short hooks when appropriate: concrete question,
  failed assumption, mechanism contrast, artifact screenshot, or scenario
  observation. Avoid mystery hooks.
- Comments should clarify concepts, add sources, or move reproducible issues to
  GitHub. Do not use comments as a second advertisement.

## Post-Publish Follow-Up

- Check comments, likes/collections, reposts, and private messages only when the
  user asks or follow-up was part of the task.
- Prioritize replies that clarify mechanism, answer use-case questions, correct
  misunderstandings, or point to GitHub issues/docs.
- Record repeated questions as future Zhihu answers, blog posts, or docs tasks.
- Do not like, follow, repost, send private messages, or make commitments
  without explicit user instruction.

## Quality Gate

- Before selecting a long-form source, confirm that its first screen states the
  problem and that the article provides a reusable mental model. If it does not,
  improve the source or skip syndication instead of rewriting the platform copy.
- Project links are specific GitHub, docs, paper, or demo URLs.
- The text names concrete use cases: safety, compliance, reliable instruction
  following, observability, debugging, performance, or production governance.
- The tone is explanatory and expert, not sales-heavy.
