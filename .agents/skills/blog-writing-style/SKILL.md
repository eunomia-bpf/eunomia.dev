---
name: blog-writing-style
description: Detailed prose and bilingual rule checklist for blog posts under docs/blog/posts/. Pure reference; contains no workflow. Use it when drafting or reviewing via blog-writer. Covers argument architecture, paper-to-blog transformation, paragraph load and rhythm, prose mechanics, blog antipatterns, length and richness, and Chinese-first bilingual writing. SEO/GEO rules live in the separate seo-geo skill.
---

# Blog Style Checklist (rules only)

This file is the single source of truth for blog style rules. It contains no process; the workflow lives in the `blog-writer` skill.

## Critical rules

- **High-quality insight is the blog's highest priority.** Facts, figures, and polished prose are inputs, not the finished value. Every full post must offer a source-grounded, non-obvious synthesis that changes how a practitioner understands the problem boundary, mechanism, tradeoff, or next decision. A faithful paper summary with no new reader-facing insight is still a failed blog.
- **Never change an existing published post's `slug`, filename, or URL.** Slugs are set once at creation. A missing slug on an already-published post stays missing; report it but do not add one, because adding it moves the URL.
- **Write product and project names out in full** (AgentSight, ActPlane, eBPF, bpftime). Names in prose are keywords; never abbreviate them into pronouns across paragraphs.
- **Never delete design decisions or technical content.** Compression means better prose, not less information.
- **Never change the meaning** of a sentence. If unsure, flag it.
- **Keep scope-bearing hedges** ("in our tests", "on covered hooks", "up to"): they keep claims honest. Only collapse stacked hedges down to one.
- **Facts must be faithful to their source.** Posts about a paper use the paper's current published terminology and numbers, not an older draft's.
- **Never use absolute words without structural proof.** “Cannot,” “impossible,” “always,” and “never” are Must-fix unless the mechanism or source establishes the absolute claim.
- **Never replace a complete paragraph or section during a review pass.** Fix only the necessary sentences and clauses. Splitting an overloaded paragraph is allowed only when every technical statement, qualifier, and relationship survives.
- **Never combine a data correction with opportunistic prose churn.** When updating a number, denominator, label, or figure reference, change only the factual token and the minimum grammar needed to keep the sentence correct.
- **Never silently remove a caveat, limitation, negative result, design rationale, or protected hedge.** Empty apology can be cut, but evidence-bearing boundaries are technical content.

---

## Length and richness

- A full post normally contains **1,800-2,500 English words** and roughly 180-240 lines of Markdown, including code blocks, tables, and figures. Word count measures scope; line count helps expose compressed walls of text. A full post below 1,500 words is thin unless it is deliberately scoped as a release note, erratum, or single-finding update. A post above 2,800 words needs an explicit reason and a reduction pass.
- Rich means substance, not padding. Grow a post by adding concrete examples, real measured data, code or rule snippets, figures, mechanism explanations, and FAQ entries; never by restating the same point in more words.
- Every H2 section must contain at least one thing only we can write: a first-party number, a real code/config example, a figure, or first-hand experience. A section that only paraphrases common knowledge gets cut or merged.

### Paragraph load and rhythm

- Give each paragraph one primary job: establish a scene, make a claim, explain a mechanism, present evidence, interpret evidence, state a limitation, or bridge to the next step. If a paragraph performs three or more jobs, split or restructure it.
- English body paragraphs usually land between 40 and 90 words. Inspect every paragraph above 110 words. Three consecutive paragraphs above 90 words are a **Must fix** because the reader never gets a change of pace.
- Chinese body paragraphs usually land between 120 and 260 Chinese characters. Inspect every paragraph above 320 characters. Three consecutive paragraphs above 260 characters are a **Must fix**.
- These ranges are diagnostic tripwires, not quotas. A short transition can be one sentence, and a mechanism explanation can run longer when it cannot be split without losing logic.
- Vary paragraph shape. Alternate explanation with an example, consequence, figure, quotation, or compact list. Do not make every paragraph a same-sized block of four or five declarative sentences.
- Do not compress a long post by removing blank lines. If a 3,000-word article fits into about 150 Markdown lines, inspect it for overloaded paragraphs before calling it concise.

---

## Article architecture: build an argument, not a paper digest

- Write a one-sentence thesis before the outline. Every H2 must advance that thesis, not merely cover another topic from the source material.
- Give the reader a progression they can feel. A common systems-blog progression is scenario or measured problem -> why the obvious layers fail -> mechanism -> evidence -> boundary or practical consequence. Other progressions are valid, but adjacent sections must have an explicit "therefore" relationship.
- Treat an opening scenario as the article's backbone. Return to it when explaining the mechanism or evaluation, and resolve it before the ending. A vivid hook that disappears after `<!-- more -->` is decoration, not structure.
- Do not mirror a paper's section order, RQ order, or contribution list. A sequence such as dataset -> taxonomy -> design requirements -> implementation -> evaluation is a warning that the writer expanded the paper outline instead of designing a blog argument.
- Select evidence for the thesis. A blog about one empirical finding does not need to retell every mechanism and benchmark in the paper. Preserve omitted technical depth through a direct paper or sibling-post link.
- End with a new implication, decision, boundary, or next action. Do not repeat the title, abstract, or section takeaways in a trailing summary.

### Transforming a paper into a blog post

- The paper is the source of truth, not the source of structure or voice. Preserve terminology, numbers, scope, and caveats while rebuilding the exposition for a practitioner reader.
- State the post's unique angle before drafting. Examples include an empirical finding, a mechanism walkthrough, an evaluation result, or a deployment lesson. If the angle needs an "and also" list of all paper contributions, it is still too broad.
- Use an evidence sandwich for important results: state what the result means, give the number with method and conditions, then explain the consequence or limitation. Do not publish a ledger of percentages with no interpretation between them.
- Avoid successive paragraph openings such as "The paper...", "The study...", "These findings...", "The evaluation...", and "The implementation...". Name the concrete actor, mechanism, or result, and write from the site's point of view.
- Compare the outline with existing eunomia.dev posts before drafting. If a sibling already owns a mechanism, summarize only what the new argument needs and link to the deeper treatment. One post must not compete with or silently duplicate another post's core angle.
- The author must add synthesis that the paper does not provide in the same form: a running example, operational implication, comparison across layers, deployment tradeoff, or explanation of why a number matters.

### Figures from source papers

- Build a numbered inventory of every main-body figure before outlining. Record what claim each figure supports and where the source asset comes from. The inventory is a source-fidelity aid and a selection pool, not a requirement to publish every figure.
- Select figures by argumentative value. Include a figure only when it materially advances the post's thesis, makes an important comparison easier to grasp than prose, or supplies evidence the surrounding text cannot carry as clearly. Retaining a paper section, including an empirical-study section, does not make all of that section's figures mandatory.
- Prefer a small set of high-signal figures over a paper-shaped gallery. Omit plots that are secondary to the post's topic, duplicate evidence already visible elsewhere, require disproportionate setup, or interrupt the argument. Preserve any important omitted result in prose and link to the paper for full detail.
- Introduce a figure with the claim it supports, place it directly after that discussion, and interpret the visual instead of leaving it as decoration.
- EN and ZH use the identical image payload and matching placement. Alt text and surrounding explanation are written naturally in each language.
- Do not redraw a source plot merely to change its style. Prefer an exact repository-owned copy or a stable source asset, and preserve labels, scales, legends, and uncertainty information.
- Never add a figure solely to satisfy completeness. Record the selection rationale for included figures and verify that omitting the rest does not leave a claim unsupported or misleading.

## Anti-content-farm rules

- **Titles should be as compelling as accuracy allows.** Lead with the post's strongest true insight, consequential tension, surprising measurement, or practical stake. A title must give a qualified reader a concrete reason to click, not merely label the topic or announce that a study exists.
- **Reveal the thesis without exhausting it.** No tease questions or withheld conclusions ("...告诉我们什么", "...缺什么", "what you don't know about X"), but do not flatten an insight into a generic report title. Prefer a precise tension such as a gap between what instructions demand and what one enforcement layer can observe. Numbers in titles must be real measurements with the same scope as the article.
- Reject clickbait, alarmism, vague superlatives, unsupported universals, and titles that are exciting only because they overstate the source. When attraction and fidelity conflict, fidelity wins; then find a sharper truthful angle.
- No listicle framing ("5 tips", "N 个技巧"), no hollow calls to action ("快来试试吧!", "give it a try today!"), no marketing self-praise ("powerful", "blazing fast" without numbers).
- Open with a scenario, a measurement, or a problem, never with throat clearing or product promotion.
- Every paragraph must add information the previous ones did not. Two adjacent paragraphs making the same point get merged.

---

## Punctuation rules

Adopted from the paper-writing-style rulebook; blogs follow the same discipline.

### No em dashes
Never use "——" (Chinese) or " — " (English) in blog prose. Use commas, parentheses, conjunctions, or restructure the sentence. The only exception is inside code blocks or CLI output examples.

### No semicolons joining independent clauses
Rewrite as two sentences (period), one sentence with a conjunction (", and", ", but", ", so"), or one sentence with a causal connector ("because", "since", "therefore"). Semicolons ARE acceptable inside parenthetical lists and numbered enumerations.

**Bad:** `The engine propagates labels; rules fire at each event.`
**Good:** `The engine propagates labels, and rules fire at each event.`

### Colons (avoid unless introducing a list)
Use colons only to introduce a numbered list with explicit markers like (1), (2), (3), or the Markdown structures that need them (a fenced code block, a bullet/numbered list, a table).

Never use colons for `claim: evidence`, `setup: result`, `observation: explanation`, or noun-phrase lead-ins. The Chinese noun-phrase-colon pattern is explicitly banned: "论文的全景概括：…"、"核心洞察是：…"、"X 的价值在于：…" are violations in both languages. Fold the content into the sentence ("论文在引言里概括道，……") or split into two sentences.

**Bad:** `Optimization spans three layers: source, LLVM, and JIT.` (unlabeled list)
**Bad:** `论文在 intro 中的全景概括："64% ..."` (noun-phrase lead-in)
**Good:** `The framework addresses three goals: (1) crash isolation, (2) bounded action spaces, and (3) multi-layer feedback.` (numbered list)

### Parentheticals (avoid unless necessary)
Use parentheses only for: (1) citations and figure/table references, (2) abbreviations and first-use bilingual glosses (策略（policy）, 决策合规率（DCR）), (3) short scope qualifiers. Never for lists or explanations; if a parenthetical runs longer than a few words, promote it to its own sentence.

---

## Sentence structure antipatterns

### Weak openings
Avoid starting sentences with "It is", "There is/are", "This is". Use a concrete subject.

**Bad:** `There are three hooks that the engine attaches to.`
**Good:** `The engine attaches to three hooks.`

### Subject-verb separation
Keep the grammatical subject within 7 words of its verb. If a long modifier separates them, split the sentence.

### Dangling modifiers
The modifier must attach to the grammatical subject.

**Bad:** `Using eBPF, the policy is enforced at the kernel level.`
**Good:** `Using eBPF, the engine enforces the policy at the kernel level.`

### Passive voice (when the agent matters)
Use active voice with a concrete actor. "The engine propagates labels at fork", not "Labels are propagated by the engine at fork events."

### Note-like prose / 笔记体 (hard ban, both languages)
Don't stack short sentences that read like bullet points. Merge with connective tissue into flowing prose. A pair of sentences from splitting an overlong one is fine; three or more in a row is the antipattern.

The subtler form is the **spec-sheet paragraph**: every sentence is grammatical, but the paragraph is a list of facts wearing punctuation. Example of the violation (ZH):

> 实现规模紧凑。用户态编译器和运行器约 3.2K 行 Rust 代码，eBPF 强制执行引擎约 1.8K 行 BPF C 代码。其中 BPF-LSM hook 处理操作前决策，tracepoint 处理观测。标签以 64 位掩码存储在逐对象的 BPF map 中。引擎支持最多 128 条并发规则。

Each sentence just deposits a spec. Flowing prose weaves the same facts into an argument, saying what a number buys or why a choice was made ("用户态编译器和运行器合计约 3.2K 行 Rust，内核强制执行引擎只有约 1.8K 行 BPF C。标签被压成 64 位掩码后，传播只需一次按位 OR，所以这个小规模实现仍能支持 128 条并发规则。"). Every fact keeps its number; the paragraph must read as one connected thought, not an inventory. If three consecutive sentences could be reordered without harming the paragraph, it is a spec sheet, and it must be rewritten.

### Vague referents
"This", "it", "they" must have an unambiguous antecedent. If unclear, name the referent explicitly.

### Topic position / stress position
Put known information at the sentence start (backward link) and the new, emphatic information at the end (stress position).

**Bad:** `A 40% reduction in latency results from label caching.`
**Good:** `Label caching reduces latency by 40%.`

### Evidence needs a concrete subject
Do not make a generated table, figure, or vague “result” perform the reasoning. Name the mechanism, workload, comparison, or measured object before the evidence.

**Bad:** `The generated table shows that operation stacks reduce inspection work.`
**Good:** `Operation stacks reduce inspection work, as the ranking experiment shows.`

### Choose the actor explicitly
Use “we” for an author choice or measurement, and name the concrete mechanism for system behavior. Do not hide responsibility behind passive voice or vague subjects such as “the approach,” “the result,” or “it.”

---

## Blog-specific antipatterns

| Antipattern | Example | Fix |
|---|---|---|
| Throat clearing | "In this blog post, we will explore..." | Cut; start with the hook |
| Trailing summary | Section ends with "In summary, X." | Cut the summary |
| Outline topic sentence | "The fundamental problem is X", "X 有一个共同特征：Y" | Rewrite as natural argumentative prose |
| Numbered scenario list | "First scenario: ... Second scenario: ..." | Natural transitions ("Now consider...", "Then there's...") |
| Double negative / "not X but Y" | "这不是X，而是Y", "not X but rather Y" | State what it **is**, positively |
| Feature-list structure | "Feature 1: ... Feature 2: ..." | Restructure as progressive argument |
| Paper-abstract tone | "This division explains why...", "The practical conclusion is more specific than..." | Concrete scenario or direct claim |
| Expanded paper outline | H2s reproduce the paper's RQs, design, implementation, and evaluation in order | Choose one thesis and rebuild the section progression around it |
| Abstract cascade | Paragraphs repeatedly begin "The study...", "The paper...", "The evaluation..." | Put the result, mechanism, or reader consequence in subject position |
| Data ledger | Several dense paragraphs enumerate percentages and benchmark cells | Keep the decisive evidence and interpret each result before adding another |
| Abandoned hook | Opening scenario never appears after `<!-- more -->` | Reuse it to explain the mechanism and resolve it near the end |
| Duplicate sibling | A full section repeats a mechanism already explained in another post | Summarize the dependency and link to the canonical treatment |
| Line-locked translation | EN and ZH have matching sentence and paragraph boundaries throughout | Preserve macro structure and facts, but compose each language naturally |
| Vague claims | "significantly reduces latency" | The measured number: "reduces latency by 40%" |
| Project-status prose | “We implemented X, then ran Y, then added Z” | Organize around the reader-facing claim and use implementation history only when causally relevant |
| Experiment diary | Paragraphs enumerate scripts, run IDs, or checks in execution order | Consolidate evidence around the small number of claims it answers |
| Evidence without an actor | “The table shows…”, “The results demonstrate…” | Put the measured mechanism, workload, or comparison in subject position |
| Unsupported absolute | “This can never fail”, “X is impossible” | State the proved boundary or use a source-bearing scope qualifier |
| Self-attack or apology | “This is only a preliminary result”, “Unfortunately, our design is simple” | Delete empty apology; retain the factual limitation in a scope or limitation sentence |

Bullet lists, comparison tables, question-style H2 headings, and FAQ sections are **encouraged**, not banned: they aid scanning and win featured snippets. The antipattern is prose that reads like notes, not structure that aids the reader.

---

## Word-level antipatterns

| Antipattern | Fix |
|---|---|
| "in order to" | "to" |
| "utilize" | "use" |
| "it is important to note that" | delete or rephrase |
| "due to the fact that" | "because" |
| "a number of" | "several" or the actual count |
| "is able to" / "has the ability to" | "can" |
| "prior to" / "subsequent to" | "before" / "after" |
| "in terms of" / "with respect to" | rephrase directly |
| "leverage" (as verb for "use") | "use" |

### Nominalizations
Use the verb form. "make assumption" → "assume"; "perform analysis" → "analyze".

### Unnecessary adverbs
Cut "very", "extremely", "basically", "actually", "really" unless they carry measurable meaning. Replace vague intensifiers with numbers.

---

## Claim & evidence patterns

### Claim before evidence
State what the numbers mean before giving the numbers.

**Bad:** `Overhead is 1.9% on replay and 6.5% on a kernel build.` (data dump with no claim)
**Good:** `Enforcement stays cheap enough for interactive use, adding 1.9% overhead on replay and 6.5% on a kernel build.`

### Numbers are claims
Always state what was measured, how, and under what conditions. Link to papers, repos, or prior art when making claims about related work.

### Hedging and limitations

- One hedge per claim is enough. Remove stacked wording such as “may potentially suggest,” but keep the qualifier that carries evidence scope.
- Do not hedge the existence of a measurement the source directly reports. Hedge its generalization when the workload, platform, selected subset, or confidence interval limits the claim.
- Do not write self-attacking sentences that apologize for a design choice. If the sentence names a real limitation, preserve the substance and state the exact boundary without apology.

---

## SEO / GEO

All search and AI-engine visibility rules (metadata, keyword strategy, citation-worthy writing, syndication canonical discipline, third-party framing hygiene) live in `.claude/skills/seo-geo/SKILL.md`. Read that rulebook alongside this one for every post.

---

## Chinese terminology discipline (ZH posts, hardest check)

Chinese posts are written in Chinese. The reference for what good looks like is `docs/blog/posts/actplane.zh.md`; the failure mode to prevent is glossary-style semi-translation where every other noun stays English.

- **Compose from facts, not English sentences.** Read the English paragraph, close it, then write the Chinese paragraph from its claim, evidence, and function. Reordering clauses, splitting a sentence, or merging two sentences is expected when Chinese logic reads better that way.
- **Do not preserve line-level symmetry.** EN and ZH must share the same H2/H3 progression, examples, figures, tables, claims, and caveats. Sentence count, paragraph count, and Markdown line count may differ. Near-perfect line-for-line correspondence across a full post is a review smell because it often signals translation instead of composition.
- **Default is Chinese.** English is allowed in exactly four classes:
  1. proper nouns and product names (eBPF, bpftime, Claude Code, CLAUDE.md, AGENTS.md, ActPlane, OSDI);
  2. terms of art with no accepted Chinese rendering, where translation would hurt recognition (agent, harness, prompt, hook, uprobe, verifier, checkpoint-restore);
  3. code, commands, file names, function names, always backticked;
  4. metric acronyms (DCR), which must get a Chinese expansion at first use: 决策合规率（Decision Compliance Rate, DCR）. Expansions must be verified against the source paper, never guessed from the letters.
- **Everything else translates**, one rendering per concept for the whole post, with the English original in parentheses at first occurrence when the post tracks a paper's terminology: 策略（policy）、语句（statement）、强制执行（enforcement）、跨事件（cross-event）、单事件（per-event）、自包含（self-contained）、上下文（context）、语义反馈（semantic feedback）、时序门（temporal gate）、违规轨迹（violation trace）、信息流标签（information-flow label）. Paper-terminology fidelity binds the EN post; the ZH post stays faithful through first-use glosses, not through raw English.
- **No Chinese sentence starts with an English common noun.** "Statement 的提取经过…" is a violation; write "语句的提取…" or change the subject. Check: `grep -nE '^[A-Z][A-Za-z-]+ ' file.zh.md` on prose lines.
- **Table headers in ZH posts are Chinese** (proper nouns and acronyms like DCR excepted).
- **English quotations are rendered in Chinese**; keep the original in parentheses or a footnote only when exact wording matters.
- **Ordinary words stay in Chinese.** Never mix English verbs or common nouns into Chinese sentences ("我们 measure 了", "做了一个 comparison" are violations).
- **Read-aloud test.** Every paragraph must read as natural spoken Chinese. If reading it means switching to English every few words, the paragraph fails, whatever the individual rules say. As a density reference: in a good post, the English tokens in a body paragraph are mostly proper nouns and code; if more than half the concept nouns in a paragraph are English, that is a Must fix.
- **No calque sentence structures.** "即 2.0 到 3.2 倍的改进"、"每系统调用开销" are English word order transliterated; rewrite in Chinese grammar ("提升了 2.0 到 3.2 倍"、"单次系统调用的开销").
- **No translated abstract voice.** Repeated subjects such as "论文"、"研究"、"这些发现"、"该评测" and phrases such as "优势在……展开"、"恢复率讲了同样的故事"、"开销可以放进工作流" usually preserve English rhetoric. Name the concrete result or rewrite around what the reader can now conclude.
- **Use glosses selectively.** A standard Chinese term gets its English original only on first use and only when the paper term will recur or readers may need it for search. Do not turn ordinary vocabulary into a parenthetical glossary.
- **Spacing:** half-width space between CJK and Latin/digits ("64 个仓库", "eBPF 程序", "支持 128 条规则").
- **Punctuation:** Chinese prose uses full-width punctuation (，。：；？), including around embedded English terms; half-width punctuation appears only inside code, paths, and quoted English sentences.
- **No English clause splicing.** Do not embed English clauses mid-sentence in Chinese prose.

## Bilingual consistency (EN/ZH pairs)

- Same macro structure: sections, argument flow, examples, figures, tables, claims, numbers, and caveats stay in the same order.
- Section headings correspond (e.g., "Three Layers, Three Blind Spots" ↔ "三层约束，三种盲区").
- Both files need the same `date`; `description` is localized, both within the length budget.
- Sentence boundaries, paragraph boundaries, and line counts do not need to match. Natural expression is mandatory, not optional.
- When one version changes an argument, example, fact, figure, or caveat, update the other. Purely local phrasing edits need not be mirrored mechanically.

---

## Quick self-edit pass (apply to every sentence)

1. Can I delete the first word/phrase without losing meaning?
2. Is a verb hidden inside a noun? Undo the nominalization.
3. Is the subject more than 7 words from its verb?
4. Does the sentence end on the most important new information?
5. Does "this/it/that" have a clear antecedent? If not, name the referent.
6. Is an adverb doing the work a number should do?
7. Does this sentence read like a paper abstract or meeting notes? Rewrite as blog prose.
8. Does the sentence name the actor, mechanism, workload, or comparison that produced its evidence?
9. Did an edit remove or broaden a scope-bearing hedge?
10. (ZH) Is every English fragment in this sentence a term of art, spaced and punctuated correctly?
