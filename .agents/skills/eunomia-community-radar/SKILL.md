---
name: eunomia-community-radar
description: Monitor approved eBPF, Linux, observability, and AI-infrastructure communities through ordinary visible browser UI, turn the strongest daily technical question into one source-grounded anonymous eBPF Q&A, and append an anonymized analysis of the day's wider community discussion. Use for daily Slack, Discord, mailing-list, forum, or community monitoring; daily Q&A publication; community-topic summaries; question triage; and maintaining the community watchlist. Do not use for monitoring reactions to already published Eunomia content, GitHub issue or PR patrol, or unsolicited replies.
---

# Eunomia Community Radar

Turn daily community discussion into one durable answer and one compact view of
what eBPF practitioners are trying to solve. Every successful daily run
publishes exactly one combined Q&A and community briefing. Never invent a
question or weaken the evidence standard to satisfy the cadence.

## Required Context

Read before monitoring:

- `CLAUDE.md`
- `.agents/README.md`
- `.github/publisher/media/community-watchlist.yaml`
- `.github/publisher/media/community-feedback.md`
- `docs/ebpf-qa/index.md`
- [the publication standard](references/qa-publication-standard.md)

Keep the watchlist as the allowlist. Adding a workspace, server, channel, or
forum is a durable scope change and requires an explicit repository edit.

## Workflow

### 1. Review The Full Daily Window

Use ordinary visible browser UI for Slack, Discord, Reddit, and other social
communities. Public mailing-list archives and official documentation may be
read in the normal browser. Do not use hidden APIs, internal endpoints,
exported chat datasets, network interception, or scraped message archives.

Review every accessible allowlisted channel for the previous 24 hours. Use
visible channel navigation, date dividers, and Page Up, Page Down, Home, or End
key presses. The optional
[`visible-channel-review.mjs`](scripts/visible-channel-review.mjs) helper may
collect text that is currently rendered in the visible browser and move the
same visible message pane with keyboard input. It must never launch a crawler,
call a platform endpoint, persist a transcript, or bypass access controls.

Read only approved public channels and threads. Never enter restricted,
private, direct-message, customer, or partner channels for editorial mining.
If a listed channel is inaccessible, record the coverage gap in the run result
instead of claiming it was quiet.

### 2. Select The Daily Question

Choose the most useful concrete technical question seen in the daily window.
Prefer questions that recur, expose a poorly documented boundary, or connect
several discussions. The question must be:

- relevant to eBPF, Linux observability, runtime extension, profiling,
  security, or adjacent Agent infrastructure;
- answerable from public primary sources or a reproducible local experiment;
- useful without the original participant's identity or deployment details;
  and
- materially different from an existing Q&A, tutorial, or blog post.

When no single message is strong enough, combine related same-day signals into
one practical question. If the daily window is genuinely sparse, use the most
recent unresolved recurring question found in the allowlist within seven days
and say in the internal run result that the fallback was used. Never publish a
placeholder, promotional topic, or speculation. A run that cannot access enough
sources or verify one real question is a failed run, not a fabricated report.

### 3. Write The Answer

Name the article with the anonymized practitioner question. Answer it in detail
before discussing the wider community. Write from the technical issue, not the
chat wording. Remove names, handles, employers, organization names, timestamps,
exact infrastructure, internal URLs, logs, tokens, IP addresses, and distinctive
phrasing. Never quote or link a closed-community message on the public page.

Verify the answer with official documentation, upstream repositories,
standards, kernel documentation, or papers. Search results can locate sources
but cannot replace reading them. Separate established behavior, operational
advice, and open limitations. Add a `References` section after the answer with
only the public sources used.

### 4. Summarize The Day's Discussion

After the references, add `Community discussion today`. Cover as much of the
daily window as the allowlist and browser access permit. State the number of
communities and channels reviewed, then synthesize the main technical themes,
where practitioners are getting stuck, and what questions remain unresolved.

Anonymize identity and deployment details without stripping away the technical
substance. For every substantive theme, explain the concrete problem or
symptom, the likely mechanism or boundary, the practical diagnostic or
resolution path, and what remains uncertain. Cite the public primary sources
that support these mini-answers. When the daily window contains several real
technical discussions, this section should normally be at least twice as
detailed as a terse trend-only summary; do not reduce each discussion to one
sentence.

Summarize across discussions rather than listing messages. Do not publish
participant names, handles, employers, channel names, message links, exact
timestamps, private topology, or wording that can be searched back to one
person. It is acceptable to say that a channel had no substantive technical
discussion. Never treat inaccessible channels as zero activity.

### 5. Publish The Daily Page

Publish exactly one Q&A per successful calendar-day run:

- create `docs/ebpf-qa/YYYY-MM-DD-<question-slug>.md` and its Chinese
  counterpart;
- add the question to `docs/ebpf-qa/index*.md`;
- preserve the order: detailed answer, references, community discussion;
- update `.github/publisher/media/community-feedback.md` only when a durable
  technical signal affects future work; and
- avoid a separate community report, browsing log, research memo, transcript,
  or daily artifact. The Q&A page is the public daily report.

### 6. Validate And Publish

Check both routes in a normal browser, including title, references, discussion
summary, code, mobile layout, and navigation. Run the repository content tests
and build required by `CLAUDE.md`. Preserve unrelated changes, stage explicit
paths, commit on `main`, rebase on `origin/main` if needed, and push directly.

After deployment, verify the public route before recording publication as
complete.

## Boundaries

- `eunomia-social-radar` owns reactions, citations, comments, and follow-up on
  content Eunomia has already published.
- `eunomia-community-patrol` owns GitHub issues and pull requests across the
  organization.
- This skill discovers external technical questions and publishes anonymous
  Q&A. It does not authorize replies, direct messages, follows, invitations,
  reactions, or moderation actions unless the user explicitly requests them.
- Never present a private-community observation as a public quotation or imply
  that an original participant endorsed the published answer.
