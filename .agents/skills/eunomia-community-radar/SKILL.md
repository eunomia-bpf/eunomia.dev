---
name: eunomia-community-radar
description: Monitor approved eBPF, Linux, observability, and AI-infrastructure communities through watchlist-opted-in read-only Slack archives plus ordinary visible browser UI, turn the strongest daily technical question into one source-grounded anonymous eBPF Q&A, and append an anonymized analysis of the day's wider community discussion. Use for daily Slack, Discord, mailing-list, forum, or community monitoring; daily Q&A publication; community-topic summaries; question triage; and maintaining the community watchlist. Do not use for monitoring reactions to already published Eunomia content, GitHub issue or PR patrol, or unsolicited replies.
---

# Eunomia Community Radar

Turn daily community discussion into one durable answer and one compact view of
what eBPF practitioners are trying to solve. Every daily run publishes exactly
one real, anonymized, bilingual (English and Chinese) Q&A with its community
briefing. Never invent a question or publish without a real public primary
source. Start from the previous 24 hours; when that window is sparse, fall
back to the previous 7 days; when it still yields nothing usable, keep
selecting a verifiable real question from the permitted readable historical
archives, a genuine unresolved community question, permitted public community
pages, or public primary documentation, until the run is complete. Coverage,
freshness, and access gaps are recorded honestly in the run result and never
stop the publication itself.

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

Apply the hybrid source policy declared in the watchlist:

- Slack sources marked `archive_opt_in: true` use the approved private
  read-only archive as the primary source and the visible browser as the
  fallback.
- Discord and any chat source without an archive remain visible-browser-only.
- Mailing lists and Reddit use ordinary public browser pages and never APIs.
- Official documentation may be read in the normal browser.

Do not use hidden APIs, internal endpoints, unapproved chat exports, network
interception, or scraped message archives for any source. Archive access must
satisfy every invariant in the Archive Access Invariants section.

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

When no single message in the 24-hour window is strong enough, first widen to
the previous 7 days; when that is still sparse, keep selecting and verifying a
real question from the permitted readable historical archives, a genuine
unresolved question from the monitored communities regardless of age,
permitted public community pages, or public primary documentation, until the
run is complete. Partial archive coverage, inaccessible or sparse sources, and
thin windows are recorded, never treated as a failure or a reason to stop, and
inaccessible channels are never claimed to have been quiet. Say in the
internal run result which fallback supplied the question. Never publish a
placeholder, promotional topic, or speculation, and never fabricate a
question or evidence.

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

Publish one Q&A per successful run. Before starting a new dated pair, resume
any unpublished pair left by an earlier attempt, even after the calendar date
changes. A retry must finish that candidate rather than strand it or create a
duplicate:

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
paths, commit on `main`, merge forward from `origin/main` when the remote has
advanced, and push directly. Unrelated worktree changes are normal in a shared
checkout: preserve them, stage and commit explicit owned paths, and continue.
Do not require a particular local model when the active agent can complete the
work. Keep automation direct: model allowlists, fixed roles, redundant
preflights or approvals, hidden orchestration, and gate-style validations that
turn coverage gaps, calendar rollover, or unrelated work into publication
stoppers are explicitly banned added complexity.

After deployment, verify the public route before recording publication as
complete.

## Archive Access Invariants

For opted-in Slack archive access:

- Connect only with the DSN from the `EUNOMIA_QA_ARCHIVE_DSN` environment
  variable. Never read, print, copy, or persist the DSN in files or state.
- Before reading anything, verify the connection is `transaction_read_only`
  and holds zero admin or write grants.
- The schema must uniquely match the allowlisted public workspace team name,
  and every listed channel must exist in it.
- Use Slack message `ts` for the 24h and 7d windows; never load or insert
  time.
- Query only message text and timestamps for the allowlisted workspace and
  channels. Do not query users or handles, `data` JSON columns, or files or
  attachments.
- No database writes of any kind.
- Keep raw text only in a bounded per-run temporary input file with `0600`
  permissions inside ephemeral isolated OpenCode state, and remove it on exit.
- Never store transcripts, logs, prompts, sessions, or raw text in Git or
  persistent state.
- A missing or ambiguous workspace or channel is inaccessible, not quiet.
- Public output remains anonymized and source-grounded per the publication
  standard.

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
