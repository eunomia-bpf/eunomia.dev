---
name: eunomia-research-report
description: Research and draft source-grounded Eunomia research briefs and weekly analysis reports for technically qualified human readers. Use when Codex needs to analyze broad AI, Agent, and infrastructure change; use platform Deep Research as an input; synthesize primary papers, official documentation, open-source artifacts, and deployment evidence; select a thesis only after research; or revise an existing Research page that is accurate but difficult to read. This skill prepares and validates drafts but does not authorize final publishing.
---

# Eunomia Research Report

Research first, choose a defensible thesis, then write a report that a human
technical reader can understand in one pass.

Academic quality comes from evidence, reasoning, scope, and falsifiability. It
does not come from paper-like tone, term density, formal notation, or a long
related-work section.

## Required Context

Read these before acting:

- `CLAUDE.md`
- `.agents/skills/blog-writing-style/SKILL.md`
- rolling publication queue: `draft/plan/publishing-queue.zh.md`, when relevant
- today's media workspace and run log, if present:
  `draft/media/YYYY-MM-DD/` and `draft/media/YYYY-MM-DD/run-log.md`
- existing Research pages under `docs/research/`
- previous reports under `draft/media/` and `docs/reports/` when checking topic
  duplication
- `references/research-method.md`
- `docs/papers/registry.yaml` only after a candidate thesis exists

This skill may be invoked directly or by `eunomia-content-patrol`. The patrol
decides when research work belongs in the operating loop; this skill owns
research, topic selection, drafting, and report QA.

## Output Modes

Choose the output mode before drafting.

### Research brief

Use `docs/research/<topic>.md` and a paired Chinese or English page when the work
is intended for the public `/research/` section.

A Research brief has a stable question, a durable URL, an explicit evidence
cutoff, and a claim that can be revised when later evidence changes. It is not a
daily news post and should not be created merely because a scheduled run needs
an output.

### Weekly analysis

Use `draft/media/YYYY-MM-DD/<topic-slug>/deep-report.zh.md` for the scheduled
weekly analysis workflow. The draft is handed back to the publishing workflow
after validation.

## Reader Contract

The intended reader is a technically qualified engineer, researcher, or
maintainer who understands the broad domain but has not read the source corpus.

The public report must satisfy all of these conditions:

- The title and description are understandable without knowing a coined term.
- Within the first 250 English words or 500 Chinese characters, the reader can
  state the concrete problem, who experiences it, why it matters, and the
  report's main claim.
- The article shows a concrete end-to-end scenario before introducing a
  taxonomy, formal property, or new abstraction.
- A named abstraction appears only after the reader has seen the failure it
  explains.
- Ordinary prose uses the reader's language. English terms appear only when
  they are established terms of art, identifiers, or useful search terms.
- The argument is organized by reader questions and engineering decisions, not
  by the order in which sources were discovered.
- Formal notation is optional. When used, it follows an intuitive explanation,
  defines every symbol locally, and adds precision that prose alone cannot.
- Tables compare one clear dimension. Do not place a dense runtime or
  related-work matrix in the opening third of the article.
- A long report remains selective. It explains the evidence that changes the
  conclusion instead of listing everything reviewed.

A technically correct article that fails this contract is not publication
ready.

## Workflow

### 1. Start With A Broad Direction

Begin with a wide editorial direction such as current AI, Agent, or systems
infrastructure. Do not decide the thesis, preferred conclusion, or relationship
to Eunomia projects before searching.

When revising an existing article for readability, preserve its evidence and
central question only if they remain defensible. Do not protect its title,
section order, coined terminology, or formal model merely because they already
exist.

### 2. Build The Research Corpus

Cover the latest seven days when the task is current analysis, with the latest
48 hours receiving first attention. Use the wider 7-30 day window for
mechanisms, contradictions, deployments, and context. Use older primary sources
for standards, prior art, and baselines.

For a scheduled broad weekly analysis, materially review at least:

- 20 distinct papers
- 20 distinct industry or open-source projects
- 10 distinct current-event sources from the latest seven days

This numeric gate applies to the broad weekly scan, not automatically to every
focused Research brief. A focused brief should use the smallest corpus that
adequately supports its question, alternatives, and boundaries. Do not pad a
brief to imitate a survey.

Count independent sources, not reposts or multiple pages repeating one
announcement.

### 3. Use Platform Deep Research Correctly

When the user asks for platform Deep Research, or when it is available and
materially useful, use it for discovery and source synthesis.

Treat the returned report as research input, not publishable prose:

- inspect the cited primary sources;
- verify dates, metrics, scope, and quoted conclusions;
- remove unsupported extrapolation;
- compare the result with existing Eunomia content;
- rebuild the argument for the intended human reader;
- do not preserve the tool's outline, phrasing, or source-by-source structure by
  default;
- never label a page as produced by Deep Research unless a complete report was
  actually returned and used.

If the platform report is unavailable or incomplete, say so in the work record
and continue only with sources that can be verified directly.

### 4. Search Efficiently And Verify Selectively

Use web search, feeds, paper indexes, official release pages, repository search,
and public project pages for discovery. Search snippets, aggregators,
newsletters, and reposts identify leads but do not support factual claims by
themselves.

Open and read primary sources for serious candidates. Use official
documentation for current product behavior and specifications. Use papers,
code, datasets, issues, and first-person engineering reports for mechanisms and
measured results.

Distinguish the date of the underlying event from the article, repost, or
indexing date.

### 5. Build The Signal Map

Capture serious candidates compactly with source type, primary URL,
publication date, event date, concrete claim, evidence strength, limitation,
and contradictory signal. Keep this in working context or the dated run log; do
not create a public source-inventory artifact.

Cluster candidates by mechanism, change, or tension rather than shared
keywords. Generate candidate questions only after clustering.

Before selecting a broad thesis, build the evidence lattice in
`references/research-method.md`. Do not fill a missing role with a weak source
merely to complete the pattern.

### 6. Form And Test The Thesis

Choose a clear, contestable thesis only after the scan. It may support,
qualify, contradict, or be unrelated to existing Eunomia projects.

Look for the strongest alternative explanation and evidence that could overturn
the thesis. Narrow or discard a thesis that depends on one marketing claim, one
repeated announcement, or social posts quoting each other.

Then compare the candidate with existing Research pages, blog posts, and public
papers. The central question, argument, or conclusion must be materially
different. A new headline, product, example, or coined term does not create a
new thesis.

Repository-owned work is one evidence node, not the destination of the
argument. Apply the same evidence and caveat standards to it as to outside work.

### 7. Design The Reader Path Before Drafting

Before writing paragraphs, be able to answer these questions in plain language:

1. What concrete situation makes this problem visible?
2. What does the reader probably believe today?
3. Where does that model fail?
4. What decision should change?
5. What evidence supports and challenges the change?
6. Where does the conclusion stop applying?

Use one recurring scenario when possible. Revisit it as the article introduces
mechanisms and design choices.

Prefer this progression unless the subject needs another natural order:

- concrete situation and stakes;
- the mistaken or incomplete mental model;
- two or three failure mechanisms;
- what existing mechanisms solve and leave open;
- the proposed design or decision;
- evidence and implementation details;
- evaluation, limitations, and falsification.

Do not start with an abstract, a source matrix, a taxonomy, a formula, or the
name of a new system property.

### 8. Draft The Public Report

For a Research brief, include frontmatter with:

- date
- reader-facing title
- description
- precise public tags
- research question
- source cutoff
- review status

For a weekly report, include its stable `report_id`, research window, cutoff,
status, thesis, and tags.

Apply `blog-writing-style` to both modes. In particular:

- write Chinese as Chinese and English as idiomatic technical English;
- define a technical term in plain language before relying on it;
- prefer concrete actors, state changes, and consequences;
- connect evidence to the claim it changes;
- summarize a dense mechanism in ordinary language before moving on;
- keep source process and browsing chronology out of public prose;
- avoid a paragraph that contains several undefined abstractions;
- do not make readers decode a sentence made mostly of English nouns inside
  Chinese grammar.

End a Chinese report with `## 参考资料`. List only sources that materially support
the argument.

### 9. Run The Human Readability Gate

Read the complete article once as a domain-aware reader who has not seen the
research notes.

The draft fails if any answer below is unclear:

- What is the problem in one sentence?
- What real user or operator decision does the article change?
- What is the main claim in ordinary language?
- What are the two or three strongest pieces of evidence?
- What would make the claim false?
- Which terms must the reader remember, and are all of them necessary?

Also check:

- no title or early heading depends on unexplained jargon;
- the opening contains a concrete situation rather than an abstract field
  overview;
- every major abstraction has a nearby example;
- literature is synthesized by mechanism instead of reviewed source by source;
- the first half is not dominated by tables, definitions, or formal notation;
- headings form a readable argument when viewed alone;
- Chinese prose is not a line-by-line translation of the English article;
- the article can lose a technical term without losing the underlying idea;
- the conclusion states the engineering judgment, not merely the article
  structure.

Rewrite when this gate fails. Do not treat grammar cleanup as sufficient.

### 10. Review Evidence And Publication Value

Check that:

- the thesis emerged from evidence and is not product advocacy by default;
- every central factual claim resolves to a primary or clearly labeled source;
- independent sources are truly independent;
- contradictions, uncertainty, and source limitations remain visible;
- analysis explains mechanisms and second-order effects rather than summarizing;
- practical implications follow from the evidence;
- repository-owned work has a clear and proportionate role;
- the public page does not expose private strategy, customer information,
  pricing, or unreleased work;
- the final Chinese source section is named `参考资料`;
- no em dash appears in public prose;
- links, frontmatter, Markdown, diagrams, and locale pairs are valid.

Length follows the evidence and reader need. Do not compress a real argument
into a news summary, and do not pad a report with unused sources, taxonomies, or
notation.

### 11. Record And Hand Back

When a separate run record is useful, update
`draft/media/YYYY-MM-DD/run-log.md` with the research window, source families
checked, selected topic or `no defensible thesis`, report path, reader test
result, and next concrete action.

When invoked by `eunomia-content-patrol`, return control to that skill for
platform adaptation, publishing authorization, and ledger work.

This skill does not authorize a final publish, repost, comment, or other social
action.
