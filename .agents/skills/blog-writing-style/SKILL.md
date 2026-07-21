---
name: blog-writing-style
description: Define the finished prose style for English and Chinese eunomia.dev blog posts. Use when writing or reviewing reader comfort, natural rhythm, technical tone, and recurring sentence-level AI tells. This skill does not define article content or workflow.
---

# Blog Writing Style

Judge the finished prose by one outcome: a technically qualified human reader
can read it comfortably once, without reconstructing missing premises or
fighting mechanical language.

This is a style reference, not a content specification. Do not use it to choose
the topic, thesis, evidence, structure, length, citations, figures, or workflow.
Examples identify patterns, not forbidden strings or sentence-shape quotas.

## Reading Experience

- Write clear, natural, technically serious prose rather than notes, an
  abstract, a benchmark ledger, or promotion copy.
- Give the intended reader enough local context to follow the next sentence
  without guessing a necessary term, premise, or change of subject. Do not turn
  this outcome into a prescribed introduction.
- Let each thought lead naturally to the next. Use transitions that express the
  real relationship instead of announcing another item.
- Prefer concrete actors, actions, comparisons, and consequences over abstract
  labels such as “the result,” “the study,” or “the advantage.”
- Keep the tone professional and restrained. Avoid hype, exaggerated urgency,
  empty drama, self-congratulation, and academic ceremony.

## Paragraphs and Sentences

- Develop one coherent thought in a paragraph. Do not place several unrelated
  fact cards next to each other.
- Let paragraph and sentence length follow the thought. A page of equally sized
  paragraphs or uniformly short sentences feels mechanical.
- Keep a condition, contrast, cause, and consequence connected when they form
  one idea. Split only at a genuine conceptual pause.
- Do not repair choppy prose by joining unrelated facts with semicolons or
  colons.
- Do not begin several adjacent paragraphs with the same abstract topic-sentence
  shape. A clear topic sentence may remain when it helps orientation.
- Do not replace one repeated pattern with another row of rhetorical questions,
  presenter cues, number-led openings, or forced transitions.
- Avoid repeated openings such as “The study…,” “The result…,” “This shows…,”
  and “Furthermore…” when a concrete subject can carry the sentence.
- Do not build several adjacent claims from repeated negative constructions such
  as “is not,” “does not,” “cannot,” `不是`、`没有`、`不意味着`、`并不`、
  `不能`. When the reader needs the positive condition, capability, relationship,
  or boundary, state it directly. Keep a negative construction when the contrast
  or limitation itself carries the meaning.
- Do not end a section by merely restating it.
- Do not use em dashes in blog prose.

## Recurring Failure Patterns

`The kernel sees a write. The harness sees a tool call. Neither can decide.`

This breaks one contrast-and-consequence chain into presentation notes. When it
is one thought, keep the relationship connected.

`研究给出了分类。结果很明显。这进一步说明了方案的优势。`

These sentences have no concrete object or progression and could be reordered
without changing their meaning.

A sequence such as `跨事件策略反复出现为四类关系。`、
`上下文依赖让强制执行更难落地。`、`两类难点会叠加。` feels mechanical
because the adjacent paragraphs repeat one abstract opening. Any individual
sentence may still be useful.

Replacing every such opening with `看看这些跨事件策略长什么样。` or
`那这套强制执行实际表现如何？` creates a different template. Variety should
follow the thought, not a rotation rule.

`代码能编译，并不意味着它选对了目标，也不意味着它不会引入开销。`

This makes the reader recover the real claim through three negatives. When the
intended point is a requirement, state it directly: `成功的调优还需要明确目标、
控制开销，并在测试中保持稳定。` The individual words are not banned; the
failure is using repeated negation as the default argumentative rhythm.

## Tone and Diction

- Avoid empty judgments such as “a major breakthrough,” “the advantage is
  clear,” “further validates the superiority,” and “truly changes everything.”
- Do not manufacture surprise, uncertainty, failure, or personal experience to
  make generated prose sound human.
- State real limitations plainly, without apology or self-attack.
- Prefer direct verbs and specific nouns. Cut filler such as “it is important to
  note that,” “in order to,” “due to the fact that,” and “with respect to.”

## English

- Use idiomatic technical English rather than paper-abstract phrasing.
- Prefer active constructions when the actor matters, without forcing every
  sentence into active voice.
- Avoid nominalized prose when a direct verb is clearer.
- Keep modifiers close to the words they qualify. Long sentences are welcome
  when their clauses form one easy-to-follow thought.
- Use straight quotation marks for English prose and code literals.

## Chinese

- Write Chinese as Chinese. The finished article should not read like an English
  paragraph translated clause by clause or preserve line-level symmetry with an
  English version. Sentence and paragraph boundaries may differ.
- Do not place a full stop after every short clause. Join a connected setup,
  contrast, condition, mechanism, and consequence with natural Chinese.
- Use Chinese for ordinary prose. English is appropriate for proper nouns and
  product names; recognized terms of art without a clear Chinese rendering;
  code, commands, paths, filenames, and identifiers; and metric acronyms.
- Put code, commands, paths, filenames, and identifiers in backticks. Expand a
  metric acronym in Chinese on first use when a verified expansion is useful;
  never guess its meaning from the letters.
- Translate ordinary concepts consistently. Add the English original on first
  use only when readers need the source terminology for recognition or search;
  do not turn prose into a parenthetical glossary.
- Use `AI Agent` or `AI 智能体` consistently for the general role, whichever
  reads more naturally in the article. Avoid `AI 编程 agent` and
  `AI 编程助手` unless that narrower distinction is essential.
- Do not mix English verbs or ordinary nouns into Chinese clauses, splice an
  English clause into a Chinese sentence, or begin a Chinese sentence with an
  English common noun when a natural Chinese subject exists.
- Render table headers in Chinese except for proper nouns, code identifiers, and
  established acronyms.
- Render English quotations in natural Chinese. Keep the original only when its
  exact wording matters.
- Use full-width punctuation in Chinese prose and half-width punctuation inside
  code, paths, commands, and quoted English.
- Use matching Chinese quotation marks for Chinese quotations and straight
  quotes for code literals. Do not turn `"git"` into `“git”`.
- Keep a half-width space between Chinese characters and Latin letters or
  digits, such as `64 个仓库` and `eBPF 程序`.
- Avoid calques such as `每系统调用开销` and repeated abstract subjects such as
  `论文`、`研究`、`这些发现`、`该评测` when the sentence can name the actual
  mechanism, workload, measurement, or consequence.

### Chinese Style Anchor

Use this kind of Chinese technical-blog rhythm as a positive anchor:

> 基于 Wasm，我们可以使用多种语言构建 eBPF 应用，并以统一、轻量级的方式管理和发布。以我们构建的示例应用 `bootstrap.wasm` 为例，大小仅为约 90K，很容易通过网络分发，并可以在不到 100ms 的时间内在另一台机器上动态部署、加载和运行，同时保留轻量级容器的隔离特性。运行时不需要内核头文件、LLVM、clang 等依赖，也不需要做任何消耗资源的重量级编译工作。

The example reads naturally because related clauses stay together, sentence
length changes with the thought, and each full stop lands after a complete
claim. Treat it as a rhythm reference, not a required content pattern.
