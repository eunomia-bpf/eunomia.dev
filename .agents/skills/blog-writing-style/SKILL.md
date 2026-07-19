---
name: blog-writing-style
description: Detailed advisory prose and bilingual checklist for blog posts under docs/blog/posts/. Pure reference; contains no workflow. Use it when drafting or reviewing via blog-writer. Covers argument architecture, paper-to-blog transformation, paragraph load and rhythm, prose mechanics, blog antipatterns, length and richness, and Chinese-first bilingual writing. SEO/GEO guidance lives in the separate seo-geo skill.
---

# Blog Style Checklist (advisory)

This file is a repertoire of writing recommendations, not an acceptance gate or a workflow. Authors and reviewers should use judgment, and they may depart from any item when the source, topic, audience, or natural expression benefits. Emphatic wording such as “must,” “never,” “ban,” and numeric tripwires indicates the strength of a preference, not a deterministic failure condition and not a reason to add another review round. Record the reason only when a deliberate departure materially affects the post.

Source accuracy, public-path stability, confidentiality, and edit-scope limits are repository or workflow constraints, not style preferences; they live in `AGENTS.md` and the `blog-writer` skill.

## Core recommendations

- **High-quality insight is the blog's highest priority.** Facts, figures, and polished prose are inputs, not the finished value. Every full post must offer a source-grounded, non-obvious synthesis that changes how a practitioner understands the problem boundary, mechanism, tradeoff, or next decision. A faithful paper summary with no new reader-facing insight is still a failed blog.
- **Attractive still means professional.** Earn attention with a concrete story, failure mode, operational question, measured tension, or reader decision, then connect it to evidence and user value. The opening should make clear what problem the article helps the reader understand, what decision or workflow it changes, and what the reader can do or judge better afterward. Do not use hype, teasing, exaggerated urgency, or empty drama.
- **Every blog needs reader background before specialized claims.** Do not jump from the title into a tool, paper result, benchmark, release note, or implementation detail. First establish the domain object, the reader's likely situation, the prerequisite mechanism or term, and why the new claim matters. Background can be compact, but a post that gives the reader no ramp into the topic is incomplete.
- **Descriptions and excerpts need background too.** The frontmatter `description` and the first paragraph before `<!-- more -->` should open with a compact background clause or sentence that names the domain problem or reader situation, then state the post's finding, tool, benchmark, or practical consequence. A description that starts cold with a project name or result often reads like a note instead of an introduction.
- **Never change an existing published post's `slug`, filename, or URL.** Slugs are set once at creation. A missing slug on an already-published post stays missing; report it but do not add one, because adding it moves the URL.
- **Write product and project names out in full** (AgentSight, ActPlane, eBPF, bpftime). Names in prose are keywords; never abbreviate them into pronouns across paragraphs.
- **Preserve evidence-bearing design decisions and technical content.** Removing repetition should not erase information, but preserving information does not require packing every fact into the same paragraph; material can move to the point where the reader is ready for it.
- **Never change the meaning** of a sentence. If unsure, flag it.
- **Keep scope-bearing hedges** ("in our tests", "on covered hooks", "up to"): they keep claims honest. Only collapse stacked hedges down to one.
- **Facts must be faithful to their source.** Posts about a paper use the paper's current published terminology and numbers, not an older draft's.
- **Never use absolute words without structural proof.** “Cannot,” “impossible,” “always,” and “never” are Must-fix unless the mechanism or source establishes the absolute claim.
- **Never replace a complete paragraph or section during a review pass.** Fix only the necessary sentences and clauses. Splitting an overloaded paragraph is allowed only when every technical statement, qualifier, and relationship survives.
- **Never combine a data correction with opportunistic prose churn.** When updating a number, denominator, label, or figure reference, change only the factual token and the minimum grammar needed to keep the sentence correct.
- **Never silently remove a caveat, limitation, negative result, design rationale, or protected hedge.** Empty apology can be cut, but evidence-bearing boundaries are technical content.

---

## Length and richness

- A full technical post should be long enough to carry its setup, evidence, mechanism, limitations, and reader-facing insight without forcing the reader back into the paper. Empirical-study and paper-based posts often benefit from substantial depth. They should not be shortened merely to hit a generic word or line target.
- Word and line counts are diagnostics, not acceptance thresholds. A short post is thin when it skips evidence or interpretation. A long post is successful when each section contributes a distinct claim, example, mechanism, boundary, or decision and the reader can navigate it without fatigue. Total length alone is never a Must-fix issue.
- The boundary between useful depth and excess is functional. Keep material when removing it would weaken the thesis, evidence chain, mechanism explanation, scope boundary, or practical decision. Merge or cut material when it only restates an adjacent point, catalogs secondary results without interpretation, duplicates a sibling post, or opens a detour the main argument never uses.
- Let topic scope determine length. A release note or single-finding correction may be brief, while a professional empirical synthesis or mechanism analysis may be substantially longer. Neither form needs an apology or a fixed word target.
- Rich means substance, not padding. Grow a post by adding concrete examples, real measured data, code or rule snippets, selected figures, mechanism explanations, limitations, and genuine reader questions; never by restating the same point in more words.
- Reduce repetition, paper-digest detours, sibling overlap, and overloaded paragraphs. Do not remove source-backed technical depth, caveats, or useful evidence simply because the article is long.
- Every H2 section must contain at least one thing only we can write: a first-party number, a real code/config example, a figure, or first-hand experience. A section that only paraphrases common knowledge gets cut or merged.

### Paragraph load and rhythm

- Give each paragraph one primary job: establish a scene, make a claim, explain a mechanism, present evidence, interpret evidence, state a limitation, or bridge to the next step. If a paragraph performs three or more jobs, split or restructure it.
- English body paragraphs usually land between 40 and 90 words. Inspect every paragraph above 110 words. Three consecutive paragraphs above 90 words are a **Must fix** because the reader never gets a change of pace.
- Chinese body paragraphs usually land between 120 and 260 Chinese characters. Inspect every paragraph above 320 characters. Three consecutive paragraphs above 260 characters are a **Must fix**.
- These ranges are diagnostic tripwires, not quotas. A short transition can be one sentence, and a mechanism explanation can run longer when it cannot be split without losing logic.
- Do not optimize for uniformly short sentences. When adjacent sentences share one subject or causal chain and the later sentence merely supplies the condition, cause, or consequence, consider joining them. A longer sentence is often clearer when it preserves one coherent thought; split where the reader benefits from a genuine conceptual pause.
- Chinese technical blog prose should not drop a period after every short clause. When a setup, contrast, condition, mechanism, and consequence belong to one thought, prefer one sentence with three or four natural clauses joined by commas or connective words, then use the period at the real turn in the argument. Split earlier only when the reader needs a conceptual pause, not because the sentence has become longer than the English version.
- Vary paragraph shape. Alternate explanation with an example, consequence, figure, quotation, or compact list. Do not make every paragraph a same-sized block of four or five declarative sentences.
- Do not compress a long post by removing blank lines. When substantial word count is packed into unusually few Markdown lines, inspect it for overloaded paragraphs before calling it concise.

### Information density and breathing room

- A paragraph can have one job and still be too dense. Watch for stretches in which every sentence introduces a new percentage, term, scope qualifier, or logical turn, leaving no sentence that tells the reader what the evidence means.
- After a dense result, consider giving the reader a plain-language interpretation, concrete example, or consequence before presenting the next result. This is useful explanation, not padding, when it reduces reconstruction work.
- An opening should establish the problem and the post's distinctive insight without previewing every supporting number. Keep the evidence needed to earn the thesis, then let later sections introduce the detailed breakdown with its denominator and interpretation.
- During a density pass, mark the points where a careful reader would need to stop and unpack the paragraph. Consider deferring secondary facts, replacing a jargon cluster with a concrete description, or separating evidence from interpretation. Do not solve density by deleting caveats or by turning one coherent causal chain into choppy sentences.

### Controlled structural variety

- A full post may use one or two editorial accents in total, such as a compact list, a source quotation, or a visually emphasized key takeaway. Technical code blocks, tables, and figures do not count toward this allowance. These accents are optional tools for rhythm, not a template or quota.
- Use a list only when several parallel items become easier to compare or scan outside prose. Keep it compact, give every item the same grammatical role, and return to the argument immediately afterward. A numbered list is appropriate only when order, rank, or sequence matters. Numbered scenario tours, feature inventories, and expanded contribution lists remain banned.
- A Markdown blockquote must contain either an exact, attributed source quotation or one complete, source-backed takeaway written in the author's voice. Never format paraphrase as somebody else's quotation, and never use a blockquote to manufacture drama.
- An emphasized takeaway must be earned by the evidence immediately around it and add a useful synthesis rather than repeat the preceding paragraph. Emphasize one decisive sentence, not a paragraph of bold prose. Do not stack two editorial accents back to back.
- EN and ZH should carry the same quoted evidence or takeaway in the corresponding place, expressed naturally in each language. Preserve exact source wording only when wording itself matters; otherwise translate the quotation and identify the source.

---

## Article architecture: build an argument, not a paper digest

- Write a one-sentence thesis before the outline. Every H2 must advance that thesis, not merely cover another topic from the source material.
- Give the reader a progression they can feel. A common systems-blog progression is scenario or measured problem -> why the obvious layers fail -> mechanism -> evidence -> boundary or practical consequence. Other progressions are valid, but adjacent sections must have an explicit "therefore" relationship.
- Vary the macro structure to fit the material. A post may unfold as an investigation, a running case, a measurement-led argument, a mechanism walkthrough, a comparison, a failure analysis, or another coherent form. Do not reuse one standard H2 skeleton across posts, and do not add an FAQ, list, takeaway box, design section, or benchmark section merely because previous posts had one.
- Treat an opening scenario as the article's backbone. Return to it when explaining the mechanism or evaluation, and resolve it before the ending. A vivid hook that disappears after `<!-- more -->` is decoration, not structure.
- Build a background ramp before the first deep claim. A reader should know what system object, workflow, failure mode, deployment setting, or paper question they are looking at before the post asks them to care about a number, figure, tool behavior, or design choice.
- Carry that ramp into metadata and excerpt text. Search snippets, social previews, and the first paragraph should not require the title to supply all context; they need their own short background phrase before the article-specific claim.
- Do not mirror a paper's section order, RQ order, or contribution list. A sequence such as dataset -> taxonomy -> design requirements -> implementation -> evaluation is a warning that the writer expanded the paper outline instead of designing a blog argument.
- Select evidence for the thesis. A blog about one empirical finding does not need to retell every mechanism and benchmark in the paper. Preserve omitted technical depth through a direct paper or sibling-post link.
- End with a new implication, decision, boundary, or next action. Do not repeat the title, abstract, or section takeaways in a trailing summary.

### Reader-perspective review

- Name the target reader before review, including what they likely know, what decision brought them to the post, and which terminology they should not be expected to know. Review the published reading experience, not the author's outline or the paper's contribution checklist.
- Test the title and opening as a promise. After the excerpt, the reader should know why the topic matters, what concrete question the post will answer, and why this article offers evidence or insight they cannot get from a generic overview.
- Test the description the same way. If the first words of the `description` do not tell a new reader what area, failure mode, workflow, or decision the post belongs to, rewrite it before tuning keywords.
- Check whether the first specialized claim has enough background. If the article names a tool, paper result, benchmark, kernel subsystem, model behavior, or implementation choice before explaining the surrounding problem and prerequisite terms, the opening needs a background pass.
- Follow the article in order and mark every point where a qualified reader would ask “why does this follow?”, “what does this term mean?”, “compared with what?”, “under which conditions?”, or “why should I care?”. Later clarification does not excuse an avoidable stumble at first use.
- At the first mention of a study, result, problem, or mechanism, name the concrete relationship the reader needs: what was measured, compared, observed, or enforced; between which objects; and with what consequence. Do not make the next sentence retroactively supply an object that the current sentence omitted.
- Trace each paragraph's information flow in reading order. Pronouns and abstractions such as “this problem,” “that layer,” “the gap,” or “enforcement technology” should have a concrete antecedent before they appear, and a causal claim should expose enough of its cause and effect to stand on first reading.
- Every important number must arrive with enough denominator, condition, comparison, and interpretation for the reader to understand its consequence. A technically correct result that forces the reader to reconstruct the argument from a paper figure still fails.
- Check reading momentum. Each section should create a reason to continue, vary the mode of explanation, and pay off the question raised before it. Flag dense stretches, repeated setup, jargon clusters, decorative figures, and detours into mechanisms already covered by sibling posts.
- At the end, the reader should be able to state the post's distinctive insight, its evidence, its boundary, and the practical decision it changes. If the reader remembers only the project name or a pile of percentages, the post needs revision.
- Review trust as part of readability. Flag titles, transitions, takeaways, and claims that feel promotional, inflated, defensive, or content-farm-like even when no individual sentence is factually false.

### References at the end

- End each blog post with a compact `## References` section in English and `## 参考文献` in Chinese. Keep it as the final section so readers can distinguish supporting sources from related-reading links in the argument.
- List 5–10 distinct primary papers, official documentation pages, upstream repositories, datasets, or other first-party sources that materially support the post. Prefer the most direct source and descriptive linked titles; do not use raw URLs, split one work into duplicate entries, or pad the list with sources the article did not rely on.
- EN and ZH should cite the same underlying sources in the same order, with link labels localized when useful. Inline links still belong near the claims they support; the final section is a compact source record, not a substitute for claim-level attribution.

### Transforming a paper into a blog post

- The paper is the source of truth, not the source of structure or voice. Preserve terminology, numbers, scope, and caveats while rebuilding the exposition for a practitioner reader.
- State the post's unique angle before drafting. Examples include an empirical finding, a mechanism walkthrough, an evaluation result, or a deployment lesson. If the angle needs an "and also" list of all paper contributions, it is still too broad.
- Use an evidence sandwich for important results: state what the result means, give the number with method and conditions, then explain the consequence or limitation. Do not publish a ledger of percentages with no interpretation between them.
- Avoid successive paragraph openings such as "The paper...", "The study...", "These findings...", "The evaluation...", and "The implementation...". Name the concrete actor, mechanism, or result, and write from the site's point of view.
- Compare the outline with existing eunomia.dev posts before drafting. If a sibling already owns a mechanism, summarize only what the new argument needs and link to the deeper treatment. One post must not compete with or silently duplicate another post's core angle.
- The author must add synthesis that the paper does not provide in the same form: a running example, operational implication, comparison across layers, deployment tradeoff, or explanation of why a number matters.

### Figures from source papers

- Build a numbered inventory of every main-body figure before outlining. Record what claim each figure supports and where the source asset comes from. The inventory is a source-fidelity aid and a selection pool, not a requirement to publish every figure.
- Treat the inventory as evidence that the figure step happened. For each figure or table, record the supported claim, source page or file, include/omit decision, and how the claim remains supported when the item is omitted.
- Select figures by argumentative value. Include a figure only when it materially advances the post's thesis, makes an important comparison easier to grasp than prose, or supplies evidence the surrounding text cannot carry as clearly. Retaining a paper section, including an empirical-study section, does not make all of that section's figures mandatory.
- Prefer a small set of high-signal figures over a paper-shaped gallery. Omit plots that are secondary to the post's topic, duplicate evidence already visible elsewhere, require disproportionate setup, or interrupt the argument. Preserve any important omitted result in prose and link to the paper for full detail.
- Introduce a figure with the claim it supports, place it directly after that discussion, and interpret the visual instead of leaving it as decoration.
- EN and ZH use the identical image payload and matching placement. Alt text and surrounding explanation are written naturally in each language.
- Do not redraw a source plot merely to change its style. Prefer an exact repository-owned copy or a stable source asset, and preserve labels, scales, legends, and uncertainty information.
- Never add a figure solely to satisfy completeness. Record the selection rationale for included figures and verify that omitting the rest does not leave a claim unsupported or misleading.

## Anti-content-farm rules

- **Titles should be as compelling as accuracy allows.** Lead with the post's strongest true insight, consequential tension, surprising measurement, or practical stake. A title earns attention through intellectual substance and gives a qualified reader a concrete reason to read, rather than merely labeling the topic or announcing that a study exists.
- Let the strongest source-backed insight determine the title's shape. An empirical-study post may foreground the evidence type, a measured contrast, a boundary, an implication, or the practical consequence, depending on what best represents the article. Vary syntax and emphasis across posts; do not prefix every empirical article the same way or turn one successful title into a house template.
- A good title preserves high-value signals instead of flattening them for brevity. Keep the evidence type when it builds trust (`An Empirical Study`, `case study`, `benchmark`), the method or subsystem when it carries the technical hook (`sched_ext`, `semantic flamegraphs`, `eBPF verifier`), and the article form when it helps the reader know what they are opening (`Inside SchedCP`, `Why ...`, `How ...`). Do not replace these signals with a generic declarative title unless the original signal is inaccurate or genuinely weaker.
- Judge title changes against nearby published titles before editing. If the current title already names the article type, core object, evidence or mechanism, and reader-facing tension, assume it is strong until a better alternative preserves those signals. Shorter is not automatically better, and homepage-card neatness is not a reason to erase distinctive technical information.
- Treat the title and first paragraph as one promise. The title should make the reader want the article, and the first paragraph should immediately prove that promise with a concrete scene, failure mode, measurement, or decision that makes the technical question feel alive. Do not let the first paragraph collapse into an abstract, project description, or paper-summary lead after a strong title.
- **Reveal the thesis without exhausting it.** No tease questions or withheld conclusions ("...告诉我们什么", "...缺什么", "what you don't know about X"), but do not flatten an insight into a generic report title. Prefer a precise tension such as a gap between what instructions demand and what one enforcement layer can observe. Numbers in titles must be real measurements with the same scope as the article.
- Keep a professional research-and-engineering voice. Reject clickbait, alarmism, exclamation marks, vague superlatives, casual hot-take language, unsupported universals, and titles that are exciting only because they overstate the source. Do not use formulas such as "X is broken/dead", "you won't believe", "the truth about X", or "everything changes" unless the literal claim is rigorously established and still appropriate for a technical publication.
- Attraction must come from specificity, credible stakes, and a non-obvious relationship between facts. When attraction and fidelity conflict, fidelity wins; then find a sharper truthful angle rather than retreating to a generic title.
- No listicle framing ("5 tips", "N 个技巧"), no hollow calls to action ("快来试试吧!", "give it a try today!"), no marketing self-praise ("powerful", "blazing fast" without numbers).
- Open with a scenario, a measurement, or a problem, never with throat clearing or product promotion.
- The first screen should answer the reader's quiet question of "why should I keep reading?" with a specific problem, user value, and evidence-backed payoff. A story opening works when it is real enough to carry the argument and resolves into the technical claim, not when it is decorative setup.
- Every paragraph should advance the argument, but reader-facing interpretation and a concrete example count as progress even when they introduce no new number. Merge adjacent paragraphs only when the second neither deepens understanding nor gives the reader useful breathing room.

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

### Missing objects and deferred explanations
An abstract noun does not tell the reader what happened. Name the measured or compared objects and their relationship when the claim first appears, especially in an opening or transition.

**Bad:** `论文量化的问题早于具体的强制执行技术。`
**Good:** `论文量化了开发者写下的规则与系统能够确定性执行的规则之间的落差。`

### Over-segmented causal chains
Sentence boundaries should mark meaningful turns in thought, not enforce an artificial preference for brevity. When two or three short sentences describe one condition, action, and consequence, consider joining them with explicit connective tissue. Keep them separate when each sentence advances an independent idea; the goal is a complete causal unit, not a long sentence for its own sake.

In Chinese posts, check whether two adjacent short sentences would be clearer as one sentence with three or four clauses. A pattern like "X 看起来很精确。Y 报告停止位置。Z 才是修复点。" often reads like notes; prefer "X 看起来很精确，但 Y 报告的只是停止位置，真正的修复经常要回到 Z。" when the three pieces form one claim.

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

Compact lists, comparison tables, question-style headings, and FAQ sections are available when they genuinely aid the reader; none is a required template. Lists, quotations, and emphasized takeaways follow the one-or-two-accent limit above. The antipattern is prose that reads like notes or a structure assembled for snippets instead of the argument.

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

All search and AI-engine visibility rules (metadata, keyword strategy, citation-worthy writing, syndication canonical discipline, third-party framing hygiene) live in `.agents/skills/seo-geo/SKILL.md`. Read that rulebook alongside this one for every post.

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

### Chinese Style Anchor

Use this kind of Chinese technical-blog rhythm as the positive anchor: "基于 Wasm，我们可以使用多种语言构建 eBPF 应用，并以统一、轻量级的方式管理和发布。以我们构建的示例应用 `bootstrap.wasm` 为例，大小仅为约 90K，很容易通过网络分发，并可以在不到 100ms 的时间内在另一台机器上动态部署、加载和运行，同时保留轻量级容器的隔离特性。运行时不需要内核头文件、LLVM、clang 等依赖，也不需要做任何消耗资源的重量级编译工作。"

This example works because it starts from a concrete capability, gives one measured artifact, explains why the number matters, and then states the operational boundary. The sentences are not uniformly short, and each period lands after a complete claim rather than after every clause. Use the same pattern for paper blogs: concrete mechanism or scenario, first-party number, practical consequence, then boundary or next step.

## Bilingual consistency (EN/ZH pairs)

- Same macro structure: sections, argument flow, examples, figures, tables, claims, numbers, and caveats stay in the same order.
- Section headings correspond (e.g., "Three Layers, Three Blind Spots" ↔ "三层约束，三种盲区").
- Both versions end with corresponding `References` / `参考文献` sections that cite the same sources in the same order.
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
11. Can a first-time reader name exactly what this sentence measures, compares, observes, or enforces without waiting for the next sentence?
12. Did punctuation split one condition-cause-consequence chain into note-like fragments that would read more naturally as one sentence?
