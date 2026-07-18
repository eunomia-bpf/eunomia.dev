---
name: blog-writing-style
description: Prose style and bilingual rule checklist for blog posts under docs/blog/posts/. Pure reference; contains no workflow. Use it as the rulebook when drafting or reviewing a post via blog-writer. Covers prose mechanics, blog antipatterns, content-farm bans, length and richness expectations, and Chinese-English mixing rules for ZH posts. SEO/GEO rules live in the separate seo-geo skill.
---

# Blog Style Checklist (rules only)

This file is the single source of truth for blog style rules. It contains no process: the review workflow lives in the `blog-review` skill, and drafting guidance lives in `tech-blog-writer`. Both reference this checklist.

## Critical rules

- **Never change an existing published post's `slug`, filename, or URL.** Slugs are set once at creation. A missing slug on an already-published post stays missing; report it but do not add one, because adding it moves the URL.
- **Write product and project names out in full** (AgentSight, ActPlane, eBPF, bpftime). Names in prose are keywords; never abbreviate them into pronouns across paragraphs.
- **Never delete design decisions or technical content.** Compression means better prose, not less information.
- **Never change the meaning** of a sentence. If unsure, flag it.
- **Keep scope-bearing hedges** ("in our tests", "on covered hooks", "up to"): they keep claims honest. Only collapse stacked hedges down to one.
- **Facts must be faithful to their source.** Posts about a paper use the paper's current published terminology and numbers, not an older draft's.

---

## Length and richness

- A full post is expected to be **around 200 lines of Markdown** (roughly 1,800-2,500 English words), including code blocks, tables, and figures. A post under ~150 lines is a short piece; that is acceptable only when deliberately scoped (release note, erratum, single-finding update). When reviewing a full post that comes in thin, report it as **Must fix: thin content**.
- Rich means substance, not padding. Grow a post by adding concrete examples, real measured data, code or rule snippets, figures, mechanism explanations, and FAQ entries; never by restating the same point in more words.
- Every H2 section must contain at least one thing only we can write: a first-party number, a real code/config example, a figure, or first-hand experience. A section that only paraphrases common knowledge gets cut or merged.

## Anti-content-farm rules

- **Titles state the finding.** No tease questions or withheld conclusions ("...告诉我们什么", "...缺什么", "what you don't know about X"). If the post has a thesis, the title says it. Numbers in titles must be real measurements.
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
Use parentheses only for: (1) citations and figure/table references, (2) abbreviations and first-use bilingual glosses (策略（policy）, 危险命令拒绝率（DCR）), (3) short scope qualifiers. Never for lists or explanations; if a parenthetical runs longer than a few words, promote it to its own sentence.

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

### Note-like prose (3+ short declaratives in a row)
Don't stack short sentences that read like bullet points. Merge with connective tissue into flowing prose. A pair of sentences from splitting an overlong one is fine; three or more in a row is the antipattern.

### Vague referents
"This", "it", "they" must have an unambiguous antecedent. If unclear, name the referent explicitly.

### Topic position / stress position
Put known information at the sentence start (backward link) and the new, emphatic information at the end (stress position).

**Bad:** `A 40% reduction in latency results from label caching.`
**Good:** `Label caching reduces latency by 40%.`

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
| Vague claims | "significantly reduces latency" | The measured number: "reduces latency by 40%" |

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
**Good:** `Enforcement stays cheap enough for interactive use: 1.9% overhead on replay and 6.5% on a kernel build.`

### Numbers are claims
Always state what was measured, how, and under what conditions. Link to papers, repos, or prior art when making claims about related work.

---

## SEO / GEO

All search and AI-engine visibility rules (metadata, keyword strategy, citation-worthy writing, syndication canonical discipline, third-party framing hygiene) live in `.claude/skills/seo-geo/SKILL.md`. Read that rulebook alongside this one for every post.

---

## Chinese terminology discipline (ZH posts, hardest check)

Chinese posts are written in Chinese. The reference for what good looks like is `docs/blog/posts/actplane.zh.md`; the failure mode to prevent is glossary-style semi-translation where every other noun stays English.

- **Default is Chinese.** English is allowed in exactly four classes:
  1. proper nouns and product names (eBPF, bpftime, Claude Code, CLAUDE.md, AGENTS.md, ActPlane, OSDI);
  2. terms of art with no accepted Chinese rendering, where translation would hurt recognition (agent, harness, prompt, hook, uprobe, verifier, checkpoint-restore);
  3. code, commands, file names, function names, always backticked;
  4. metric acronyms (DCR), which must get a Chinese expansion at first use: 危险命令拒绝率（dangerous-command refusal, DCR）.
- **Everything else translates**, one rendering per concept for the whole post, with the English original in parentheses at first occurrence when the post tracks a paper's terminology: 策略（policy）、语句（statement）、强制执行（enforcement）、跨事件（cross-event）、单事件（per-event）、自包含（self-contained）、上下文（context）、语义反馈（semantic feedback）、时序门（temporal gate）、违规轨迹（violation trace）、信息流标签（information-flow label）. Paper-terminology fidelity binds the EN post; the ZH post stays faithful through first-use glosses, not through raw English.
- **No Chinese sentence starts with an English common noun.** "Statement 的提取经过…" is a violation; write "语句的提取…" or change the subject. Check: `grep -nE '^[A-Z][A-Za-z-]+ ' file.zh.md` on prose lines.
- **Table headers in ZH posts are Chinese** (proper nouns and acronyms like DCR excepted).
- **English quotations are rendered in Chinese**; keep the original in parentheses or a footnote only when exact wording matters.
- **Ordinary words stay in Chinese.** Never mix English verbs or common nouns into Chinese sentences ("我们 measure 了", "做了一个 comparison" are violations).
- **Read-aloud test.** Every paragraph must read as natural spoken Chinese. If reading it means switching to English every few words, the paragraph fails, whatever the individual rules say. As a density reference: in a good post, the English tokens in a body paragraph are mostly proper nouns and code; if more than half the concept nouns in a paragraph are English, that is a Must fix.
- **No calque sentence structures.** "即 2.0 到 3.2 倍的改进"、"每系统调用开销" are English word order transliterated; rewrite in Chinese grammar ("提升了 2.0 到 3.2 倍"、"单次系统调用的开销").
- **Spacing:** half-width space between CJK and Latin/digits ("64 个仓库", "eBPF 程序", "支持 128 条规则").
- **Punctuation:** Chinese prose uses full-width punctuation (，。：；？), including around embedded English terms; half-width punctuation appears only inside code, paths, and quoted English sentences.
- **No English clause splicing.** Do not embed English clauses mid-sentence in Chinese prose.

## Bilingual consistency (EN/ZH pairs)

- Same structure: sections, argument flow, examples, figures, and tables in the same order.
- Section headings correspond (e.g., "Three Layers, Three Blind Spots" ↔ "三层约束，三种盲区").
- Both files need the same `date`; `description` is localized, both within the length budget.
- When one version is edited, update the other to match structure (natural expression can differ).

---

## Quick self-edit pass (apply to every sentence)

1. Can I delete the first word/phrase without losing meaning?
2. Is a verb hidden inside a noun? Undo the nominalization.
3. Is the subject more than 7 words from its verb?
4. Does the sentence end on the most important new information?
5. Does "this/it/that" have a clear antecedent? If not, name the referent.
6. Is an adverb doing the work a number should do?
7. Does this sentence read like a paper abstract or meeting notes? Rewrite as blog prose.
8. (ZH) Is every English fragment in this sentence a term of art, spaced and punctuated correctly?
