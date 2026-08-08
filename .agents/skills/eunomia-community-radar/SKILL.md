---
name: eunomia-community-radar
description: Monitor approved eBPF, Linux, observability, and AI-infrastructure communities for recurring technical questions, validate answers against public primary sources, and publish at most one anonymized eBPF Q&A on eunomia.dev when the question is broadly useful. Use for daily Slack, Discord, mailing-list, forum, or community monitoring; anonymous Q&A publication; community question triage; and maintaining the community watchlist. Do not use for monitoring reactions to already published Eunomia content, GitHub issue or PR patrol, or unsolicited replies.
---

# Eunomia Community Radar

Turn recurring community problems into durable, source-grounded answers without
turning private conversations into public transcripts. Daily monitoring is
expected; daily publication is not.

## Required Context

Read before monitoring:

- `CLAUDE.md`
- `.agents/README.md`
- `.github/publisher/media/community-watchlist.yaml`
- `.github/publisher/media/community-feedback.md`
- `docs/ebpf-qa/index.md`
- [the publication standard](references/qa-publication-standard.md) when a
  candidate may become a public Q&A

Keep the watchlist as the allowlist. Adding a new workspace, server, channel, or
forum is a durable scope change and requires an explicit repository edit.

## Workflow

### 1. Inspect The Allowlist

Use ordinary visible browser UI for Slack, Discord, and social communities.
Public mailing-list archives and official documentation may be read on the web.
Do not use hidden APIs, internal endpoints, exported chat datasets, or scraped
message archives.

Read only the approved channels and recent unresolved threads. Do not enter
restricted, private, direct-message, or customer channels for editorial mining.

### 2. Triage Questions

Keep a candidate only when it is:

- a concrete technical problem that recurs or is likely to recur;
- relevant to eBPF, Linux observability, runtime extension, profiling, security,
  or adjacent Agent infrastructure;
- answerable from public primary sources or a reproducible local experiment;
- useful without the original person's identity or deployment details; and
- materially different from an existing Q&A, tutorial, or blog post.

Ignore promotional threads, support requests that depend on confidential state,
questions already answered well by one canonical link, and topics selected only
to fill a schedule.

### 3. Rebuild The Question And Answer

Write from the technical issue, not from the chat wording. Remove names, handles,
employers, organization names, timestamps, exact infrastructure, internal URLs,
logs, tokens, IP addresses, and distinctive phrasing. Never quote or link a
closed-community message on the public page.

Verify the answer with official documentation, upstream repositories, standards,
kernel documentation, or papers. Search results can locate sources but cannot
replace reading them. Separate established behavior, operational advice, and
open limitations.

### 4. Apply The Publication Gate

Publish no more than one Q&A per calendar day. Skip publication when the answer
is speculative, duplicates existing content, exposes private context, or lacks
enough public evidence. A quiet day is a valid outcome.

When the gate passes:

- create `docs/ebpf-qa/YYYY-MM-DD-<slug>.md` and a Chinese counterpart when a
  faithful Chinese answer is available;
- add the answer to the matching `docs/ebpf-qa/index*.md` page;
- keep the title as a question a practitioner would actually search;
- cite only public sources near the claims they support; and
- update `.github/publisher/media/community-feedback.md` only with the durable
  technical signal and public Q&A URL, never a raw transcript.

Do not create a separate browsing log, research memo, figure inventory, or daily
artifact merely to prove the run happened.

### 5. Validate And Publish

Check the English and Chinese routes in a normal browser, including headings,
links, code, mobile layout, and navigation. Run the repository content tests and
build required by `CLAUDE.md`. Preserve unrelated changes, stage explicit paths,
commit on `main`, rebase on `origin/main` if needed, and push directly.

After deployment, verify the public route before recording publication as
complete.

## Boundaries

- `eunomia-social-radar` owns reactions, citations, comments, and follow-up on
  content Eunomia has already published.
- `eunomia-community-patrol` owns GitHub issues and pull requests across the
  organization.
- This skill discovers external technical questions and may publish anonymous
  Q&A. It does not authorize replies, direct messages, follows, invitations,
  reactions, or moderation actions unless the user explicitly requests them.
- Never present a private-community observation as a public quotation or imply
  that the original participant endorsed the published answer.
