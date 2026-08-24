---
name: blog-writing-style
description: Define the finished prose style for English and Chinese eunomia.dev blog posts and Research pages. Use when writing or reviewing reader comfort, natural rhythm, technical tone, research readability, and recurring sentence-level AI tells. This skill does not choose the topic or evidence.
---

# Blog Writing Style

Judge the finished prose by one outcome: a technically qualified human reader
can read it comfortably once, understand the argument, and explain the main
decision without reconstructing missing premises.

This is a style reference, not a source or workflow specification. Examples
identify failure patterns, not banned words or sentence-shape quotas.

## Human Reader First

- Write for a domain-aware engineer, researcher, or maintainer who has not read
  the source corpus.
- Establish the concrete situation, actor, state change, and consequence before
  introducing a taxonomy, formal model, or coined abstraction.
- Make the title and description understandable without prior knowledge of a
  paper, project, or invented term.
- Within the opening screen, tell the reader what problem exists, why it matters,
  and what the article argues.
- A report may be deep without sounding like a paper abstract. Academic quality
  comes from evidence and reasoning, not compressed terminology.
- Organize sections around questions the reader needs answered, not around the
  order in which sources were discovered.
- A long article should remain selective. Include detail that changes the
  reader's model or decision; remove detail that only proves the author read many
  sources.

## Progressive Disclosure

Use this order when it fits the subject:

1. a concrete situation or failure;
2. the incomplete mental model that made it possible;
3. the mechanism;
4. the design or decision that changes;
5. evidence, implementation detail, and alternatives;
6. limits and falsification.

Do not front-load a dense comparison table, a list of runtime names, equations,
or a newly named property. Introduce these only after the reader knows what they
explain.

When a formal definition is useful, first explain the same idea in ordinary
language and with an example. Define every symbol locally. If removing the
notation does not reduce precision, remove it.

## Terms And Abstractions

- Use the fewest terms needed to carry the argument.
- Define a term before relying on it. A coined term should name a pattern the
  reader has already seen, not create the appearance of novelty.
- Prefer one stable term to several near-synonyms.
- Do not make adjacent headings stacks of English nouns or unexplained
  abstractions.
- Avoid paragraphs containing several new terms at once. Introduce one idea,
  show its consequence, then continue.
- After a dense technical section, give the reader a plain-language implication
  before moving on.
- Source names support the argument; they do not substitute for it. Do not write
  a paper-by-paper parade.

## Reading Experience

- Write clear, natural, technically serious prose rather than notes, an
  abstract, a benchmark ledger, or promotion copy.
- Give enough local context for the next sentence without forcing the reader to
  infer a necessary premise or a change of subject.
- Let each thought lead naturally to the next. Use transitions that express the
  actual causal, conditional, or contrasting relationship.
- Prefer concrete actors, actions, comparisons, and consequences over abstract
  labels such as "the result," "the study," or "the advantage."
- Keep the tone professional and restrained. Avoid hype, exaggerated urgency,
  empty drama, self-congratulation, and academic ceremony.
- A strong paragraph usually contains a claim, the mechanism or evidence behind
  it, and the consequence for the reader.

## Paragraphs And Sentences

- Develop one coherent thought in a paragraph. Do not place unrelated fact cards
  next to each other.
- Let paragraph and sentence length follow the thought. A page of equally sized
  paragraphs or uniformly short sentences feels mechanical.
- Keep a condition, contrast, cause, and consequence connected when they form
  one idea. Split only at a genuine conceptual pause.
- Do not repair choppy prose by joining unrelated facts with semicolons or
  colons.
- Do not begin several adjacent paragraphs with the same abstract topic-sentence
  shape.
- Do not replace one repeated pattern with another row of rhetorical questions,
  presenter cues, number-led openings, or forced transitions.
- Avoid repeated openings such as "The study," "The result," "This shows," and
  "Furthermore" when a concrete subject can carry the sentence.
- Do not build several adjacent claims from repeated negative constructions such
  as "is not," "does not," or `不是`、`没有`、`不意味着`、`并不`、`不能`.
  State the positive requirement or relationship when that is what the reader
  needs.
- Do not end a section by merely restating its first sentence.
- Do not use em dashes in public prose.

## Tables, Lists, And Diagrams

- Use a table only when every row is compared on the same dimensions.
- Do not use a table to compress explanations that readers need in prose.
- Introduce what a diagram is meant to show before the diagram appears.
- A list should make a real set or sequence visible. Do not turn every paragraph
  into bullets.
- In a research article, one recurring scenario is often more useful than many
  disconnected examples.

## Recurring Failure Patterns

`The kernel sees a write. The harness sees a tool call. Neither can decide.`

This breaks one contrast-and-consequence chain into presentation notes. Keep the
relationship connected when it is one thought.

`研究给出了分类。结果很明显。这进一步说明了方案的优势。`

These sentences have no concrete object or progression and could be reordered
without changing their meaning.

`并行 Agent 需要 contract-valid effect serializability。它还需要 semantic
resource discovery、authority revalidation 和 global outcome contract。`

This asks a Chinese reader to decode several undefined English abstractions
before seeing the problem. Start with a concrete failure, explain what must be
checked, and introduce a formal name only if it remains useful.

A sequence such as `跨事件策略反复出现为四类关系。`、`上下文依赖让强制执行更难
落地。`、`两类难点会叠加。` feels mechanical because adjacent paragraphs repeat
one abstract opening. Any individual sentence may still be useful.

`代码能编译，并不意味着它选对了目标，也不意味着它不会引入开销。`

This makes the reader recover the real claim through repeated negation. When the
point is a requirement, state it directly: `成功的调优还需要明确目标、控制开销，
并在测试中保持稳定。`

## Tone And Diction

- Avoid empty judgments such as "a major breakthrough," "the advantage is
  clear," "further validates the superiority," and "truly changes everything."
- Do not manufacture surprise, uncertainty, failure, or personal experience to
  make generated prose sound human.
- State real limitations plainly, without apology or self-attack.
- Prefer direct verbs and specific nouns. Cut filler such as "it is important to
  note that," "in order to," "due to the fact that," and "with respect to."
- Do not use the tone of a peer-review response in a public article. The reader
  needs the argument, not ceremonial claims of novelty.

## English

- Use idiomatic technical English rather than paper-abstract phrasing.
- Prefer active constructions when the actor matters, without forcing every
  sentence into active voice.
- Avoid nominalized prose when a direct verb is clearer.
- Keep modifiers close to the words they qualify.
- Spell out the practical meaning of a formal property before naming it.
- Use straight quotation marks for English prose and code literals.

## Chinese

- Write Chinese as Chinese. Do not preserve English sentence and paragraph
  boundaries or translate clause by clause.
- Use Chinese for ordinary prose. English is appropriate for proper nouns,
  identifiers, code, paths, recognized terms of art without a clear translation,
  and useful search terms on first mention.
- When a term has a clear Chinese explanation, give the Chinese explanation
  first and the English original in parentheses only when useful.
- Do not write a Chinese sentence whose grammatical load is carried mostly by
  English nouns.
- Use `AI Agent` or `AI 智能体` consistently for the general role.
- Put code, commands, paths, filenames, and identifiers in backticks.
- Render table headers in Chinese except for proper nouns, identifiers, and
  established acronyms.
- Render English quotations in natural Chinese unless exact wording matters.
- Use full-width punctuation in Chinese prose and half-width punctuation inside
  code and identifiers.
- Keep a half-width space between Chinese characters and Latin letters or
  digits, such as `64 个仓库` and `eBPF 程序`.
- Avoid calques and repeated abstract subjects such as `论文`、`研究`、`这些发现`
  when the sentence can name the actual mechanism, workload, measurement, or
  consequence.

### Chinese Style Anchor

Use this kind of Chinese technical rhythm as a positive anchor:

> 基于 Wasm，我们可以使用多种语言构建 eBPF 应用，并以统一、轻量级的方式管理和发布。以我们构建的示例应用 `bootstrap.wasm` 为例，大小仅为约 90K，很容易通过网络分发，并可以在不到 100ms 的时间内在另一台机器上动态部署、加载和运行，同时保留轻量级容器的隔离特性。运行时不需要内核头文件、LLVM、clang 等依赖，也不需要做任何消耗资源的重量级编译工作。

The example reads naturally because related clauses stay together, sentence
length follows the thought, and each full stop lands after a complete claim.
Treat it as a rhythm reference, not a required content pattern.

## Final Reader Test

Before publishing, read only the title, description, opening, headings, and
conclusion. A qualified reader should be able to answer:

- What concrete problem does this article address?
- Why do existing mechanisms leave a gap?
- What decision or design does the article recommend?
- Where does the recommendation stop applying?
- Which one or two terms are worth remembering?

If those answers require reading the references or decoding several coined
terms, rewrite the article.
