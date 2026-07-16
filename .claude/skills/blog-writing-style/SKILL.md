---
name: blog-writing-style
description: Review and fix sentence-level writing quality and SEO metadata in blog posts under docs/blog/posts/. Checks prose mechanics (punctuation, sentence structure, word choice), blog-specific antipatterns, frontmatter/SEO fields, and EN/ZH bilingual consistency. For writing a new post from scratch, use tech-blog-writer instead.
allowed-tools: Read Edit Bash(grep *) Bash(wc *)
---

# Blog Writing Review & Fix (Sentence-Level + SEO)

Review the Markdown blog post at `$ARGUMENTS` sentence by sentence for prose quality and SEO metadata. If no argument is given, ask which post to review. Posts live in `docs/blog/posts/` as `post.md` (English) and `post.zh.md` (Chinese) pairs.

Model note: the actual writing/editing pass is best run on `claude-opus-4-6[1m]` (Opus 4.6, 1M context); when the calling agent is a different model, delegate this skill's work to an Opus subagent.

Do not perform any Git operation. Return the edited post and review findings to the caller.

**Scope:** This skill reviews and fixes existing posts. For drafting a new post, structuring an argument, or writing hooks, use the `tech-blog-writer` skill; this skill is the editing pass that runs afterward.

## Review process

1. Read the entire file (and its bilingual counterpart, if one exists)
2. Check frontmatter and SEO metadata first (see SEO checklist below)
3. For each paragraph, analyze every sentence against the checklist below
4. Report issues grouped by severity: **Must fix** (clarity/logic/SEO-metadata errors), **Should fix** (antipatterns), **Consider** (style preferences)
5. For each issue, give the line number, quote the problematic text, explain the problem, and suggest a concrete rewrite
6. Apply the fixes directly with the Edit tool, Must fix first, then Should fix, then evaluate each Consider item and apply the ones that improve the text (note rejected ones with a one-line reason). Do not ask the user which fixes to apply, and do not silently discard anything below Must fix. Exception: when invoked read-only as a review subagent, report findings only and make no edits.

## Editing discipline

- **Minimal targeted edits, one sentence at a time.** Never overwrite entire sections or paragraphs at once.
- **Do not change technical content, code blocks, YAML examples, CLI output, or architecture diagrams** when doing prose edits.
- **Preserve the author's meaning.** Do not soften or strengthen claims; flag questionable claims instead of rewriting them.
- **Deep pass on first attempt.** Do a thorough review, not just mechanical surface fixes.
- **Always diff-check** after multiple edits to ensure no content was lost.
- **Verify before claiming done:** `grep -c '——' file.zh.md` and `grep -cE ' — |—' file.md` must both return 0 (code blocks excepted).

## Critical rules (for fixes)

- **Never change an existing published post's `slug`, filename, or URL.** Slugs are set once at creation. A missing slug on an already-published post stays missing; note it as a finding but do not add one, because adding it moves the URL.
- **Write product and project names out in full** (AgentSight, ActPlane, eBPF, bpftime). Names in prose are keywords; never abbreviate them into pronouns across paragraphs.
- **Never delete design decisions or technical content.** Compression means better prose, not less information.
- **Never change the meaning** of a sentence. If unsure, flag it.
- **Keep scope-bearing hedges** ("in our tests", "on covered hooks", "up to"): they keep claims honest. Only collapse stacked hedges down to one.

---

## Punctuation rules

### No em dashes
Never use "——" (Chinese) or " — " (English) in blog prose. Replace with commas, colons, semicolons, or conjunctions. The only exception is inside code blocks or CLI output examples.

### Colons, semicolons, parentheses
All allowed in blog register. Colons may introduce examples, lists, or explanations. Semicolons may join related clauses. Use parentheses sparingly; if a parenthetical runs longer than a few words, promote it to its own sentence.

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

## SEO checklist (check first, report as Must fix)

- **`description` frontmatter is required** and must be 150-160 characters: one sentence, value proposition first, primary keyword phrase included. Longer descriptions get truncated in search results; if the current one is longer, compress it so the first ~155 characters carry the full point.
- **Title (H1) at most ~60 characters** with the primary keyword phrase front-loaded. Longer titles get truncated in search results; flag but confirm before shortening a published title.
- **`date` frontmatter required.** New posts also need a `slug` (short, kebab-case, keyword-bearing); never add or change a slug on an already-published post.
- **Headings carry search phrasing.** Prefer H2s a reader would type ("What Generic eBPF Enforcement Misses") over generic labels ("Discussion", "Overview").
- **Internal links:** every post should link 2-3 related eunomia.dev posts or docs in context, and link the project GitHub repo at least once.
- **Images need descriptive `alt` text** containing the relevant term, not "image" or "figure 1".
- **First paragraph before `<!-- more -->`** must stand alone as the excerpt: hook plus the primary keyword phrase, no dangling references.

---

## Bilingual consistency (EN/ZH pairs)

- Same structure: sections, argument flow, and examples in the same order.
- Section headings correspond (e.g., "Three Layers, Three Blind Spots" ↔ "三层约束，三种盲区").
- Both files need the same `date`; `description` is localized, both within the length budget.
- When one version is edited, update the other to match structure (natural expression can differ).
- After edits: `grep -c '——' file.zh.md` returns 0.

---

## Quick self-edit pass (apply to every sentence)

1. Can I delete the first word/phrase without losing meaning?
2. Is a verb hidden inside a noun? Undo the nominalization.
3. Is the subject more than 7 words from its verb?
4. Does the sentence end on the most important new information?
5. Does "this/it/that" have a clear antecedent? If not, name the referent.
6. Is an adverb doing the work a number should do?
7. Does this sentence read like a paper abstract or meeting notes? Rewrite as blog prose.

## Fix priority

1. **SEO metadata:** missing/overlong description, missing date, truncation-length titles
2. **Clarity:** dangling modifiers, vague referents, missing motivation
3. **Structure:** note-like prose, weak openings, passive voice, blog antipatterns
4. **Word choice:** verbose phrases from the antipattern table
5. **Punctuation:** em dashes

## Output format

For each issue found:
```
L<line>: "<quoted text>"
  Problem: <what's wrong>
  Fix: "<suggested rewrite>"
```

End with a summary: total issues by severity, the top 3 most impactful changes, and the results of the em-dash grep checks. List any sentences flagged but NOT changed, with reasons.
